use std::collections::{
    btree_map::{self, Entry},
    BTreeMap, HashMap, HashSet,
};

use shrinkwraprs::Shrinkwrap;

use crate::{
    parsed::{Arg, ArgDir, ArgSlot, Block, ExtOpCode, Function, InstructionRef, OpCode, OpInfo},
    raw,
    util::discard,
};

#[derive(Shrinkwrap, Debug, Eq, PartialEq, Clone, Copy, PartialOrd, Ord, Hash)]
pub struct LiftedBlockId(pub InstructionRef);

#[derive(Debug)]
pub struct LiftedFunction {
    pub name: Option<String>,
    pub blocks: BTreeMap<LiftedBlockId, LiftedBlock>,
    pub children: Vec<LiftedFunction>,
    pub constants: Vec<LiftedConstant>,
    pub count_upvals: u8,
    pub count_args: u8,
}

#[derive(Shrinkwrap, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LiftedVariableId(usize);

impl std::fmt::Display for LiftedVariableId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone)]
pub struct LiftedVariable {
    pub id: LiftedVariableId,
    pub replaces_id: Option<LiftedVariableId>,
    pub origin: (LiftedBlockId, usize),
    pub uses: HashSet<(LiftedBlockId, usize)>,
    pub mods: HashSet<(LiftedBlockId, usize)>,

    pub arg: ArgSlot,
}

impl LiftedVariable {
    pub fn new(
        id: LiftedVariableId,
        origin: (LiftedBlockId, usize),
        arg: ArgSlot,
    ) -> LiftedVariable {
        LiftedVariable {
            id,
            replaces_id: None,
            origin,
            uses: Default::default(),
            mods: Default::default(),
            arg,
        }
    }

    pub fn add_use(&mut self, block: LiftedBlockId, instr: usize) {
        self.uses.insert((block, instr));
    }

    pub fn add_mod(&mut self, block: LiftedBlockId, instr: usize) {
        self.mods.insert((block, instr));
    }

    pub fn merge(&mut self, from: LiftedVariable) {
        // Check so we're only merging blocks that make sense to merge
        //
        // More of a failsafe than an actual requirement, but you probably want to keep it since
        // if it gets hit something is most likely wrong somewhere.
        if !self.is_replacement(&from) && !self.is_equivalent(&from) {
            log::warn!("strange merge:\n{:?}\n{:?}", self, from);
        }

        // Register the merged block's creation as a modification to the existing variable
        self.mods.insert(from.origin);

        // All uses and mods the block we're merging has are now ours
        self.uses.extend(from.uses);
        self.mods.extend(from.mods);
    }

    pub fn is_replacement(&self, other: &LiftedVariable) -> bool {
        // Caused by write to register in an inner scope -> modification of existing variable
        Some(self.id) == other.replaces_id || self.replaces_id == other.replaces_id
    }

    pub fn is_equivalent(&self, other: &LiftedVariable) -> bool {
        // Multiple forks ended up at the same block and created the same variable
        self.origin == other.origin && self.arg == other.arg
    }
}

impl LiftedFunction {
    pub fn lift(p: Function<'_>) -> LiftedFunction {
        // Generate and lift all blocks
        let blocks = p
            .blocks()
            .into_iter()
            .map(|(id, block)| (LiftedBlockId(id), LiftedBlock::lift(block)))
            .collect();

        // Lift all constants
        let constants: Vec<_> = p.constants.iter().copied().map(LiftedConstant::lift).collect();

        // Lift all inner functions
        let children = p.prototypes.into_iter().map(LiftedFunction::lift).collect();

        LiftedFunction {
            name: None,
            blocks,
            children,
            constants,
            count_upvals: p.raw.up_val_count,
            count_args: p.raw.arg_count,
        }
    }

    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    pub fn with_name(mut self, name: impl Into<String>) -> LiftedFunction {
        self.set_name(name);
        self
    }

    pub fn variables(&self) -> Option<Vec<LiftedVariable>> {
        let func = self;
        let consts = &func.constants;

        let mut variable_id = 0;
        let mut new_var = || {
            variable_id += 1;
            LiftedVariableId(variable_id - 1)
        };

        let arguments = (0..self.count_args)
            .map(|i| {
                LiftedVariable::new(
                    new_var(),
                    (LiftedBlockId(InstructionRef::FunctionEntry), 0),
                    ArgSlot::Register(i as _),
                )
            })
            .map(|x| (x.arg, x));

        let mut block_stack = vec![(
            *func.blocks.first_key_value().unwrap().0,
            LiftedBlockId(InstructionRef::FunctionEntry),
            arguments.collect::<HashMap<ArgSlot, LiftedVariable>>(),
        )];

        let mut used_shut = vec![];
        let mut seen = HashSet::<(LiftedBlockId, LiftedBlockId)>::new();
        while let Some((block_id, origin, mut live_vars)) = block_stack.pop() {
            if !seen.insert((block_id, origin)) {
                // If we've already taken and analyzed the origin->block_id jump, ignore it
                // This helps preventing recursion, but we want to make sure we're not failing
                // to propagate imports/exports in extreme conditions. If we're failing some
                // very obscure edge case in loops where lifetimes aren't propagated properly,
                // start looking at whether this code is the problem or not.

                // If we hit a loop, close all variables we're holding
                used_shut.extend(live_vars.drain().map(|(_, x)| x));
                continue;
            }

            let block = func.blocks.get(&block_id).unwrap_or_else(|| {
                panic!("block_id does not exist, opt pass failure? {:?}", block_id)
            });

            let instructions = &block.instructions;
            for (idx, instr) in instructions.iter().enumerate() {
                // If we run into any of these instructions, surrender completely for now
                match instr {
                    x @ OpCode::Closure(_, _) | x @ OpCode::Close(_) => {
                        log::warn!("unsupported instruction: {}", x.text(consts));
                        return None;
                    },
                    _ => {},
                }

                for arg in instr.args_used().into_iter().filter(|x| x.slot().is_reg()) {
                    match (arg.dir(), live_vars.get_mut(&arg)) {
                        (ArgDir::Read, Some(x)) => x.add_use(block_id, idx),
                        (ArgDir::Modify, Some(x)) => x.add_mod(block_id, idx),
                        _ => {
                            log::warn!("reading unwritten {} {:?} ({})", arg, block_id, idx);
                            return None;
                        },
                    }
                }

                instr.args_out().into_iter().map(Arg::slot).filter(ArgSlot::is_reg).for_each(
                    |slot| {
                        let mut var = LiftedVariable::new(new_var(), (block_id, idx), slot);

                        // If a var using this write exists, close it
                        if let Some(var_old) = live_vars.remove(&slot) {
                            var.replaces_id = Some(var_old.id);
                            used_shut.push(var_old);
                        }

                        live_vars.insert(slot, var);
                    },
                );

                if let OpCode::Return(_, _) = instr {
                    used_shut.extend(live_vars.drain().map(|(_, x)| x));
                }
            }

            block_stack.extend(block.forks_to.iter().map(|&x| (x, block_id, live_vars.clone())));
        }

        used_shut.sort_by_cached_key(|x| x.id);

        let mut used_shut_merged = BTreeMap::<LiftedVariableId, LiftedVariable>::new();
        for used in used_shut {
            match used_shut_merged.entry(used.id) {
                Entry::Occupied(mut e) => {
                    log::debug!("merging:\n{:?}\n{:?}", e.get_mut(), used);
                    e.get_mut().merge(used)
                },
                Entry::Vacant(e) => discard(e.insert(used)),
            }
        }

        let mut used_vars = used_shut_merged;
        let mut usage_to_var: HashMap<(ArgDir, ArgSlot, (LiftedBlockId, usize)), LiftedVariableId> =
            HashMap::new();

        let mut aliases = HashMap::<LiftedVariableId, LiftedVariableId>::new();

        for var in used_vars.values() {
            let mut register = |dir, origin| {
                usage_to_var
                    .entry((dir, var.arg, origin))
                    .and_modify(|x| {
                        if var.id != *x {
                            aliases.insert(var.id, *x);
                        }
                    })
                    .or_insert(var.id);
            };

            register(ArgDir::Write, var.origin);

            for origin in var.uses.iter().copied() {
                register(ArgDir::Read, origin);
            }

            for origin in var.mods.iter().copied() {
                register(ArgDir::Modify, origin);
            }
        }

        dbg!(&aliases);
        // dbg!(&usage_to_var);

        let mut alias_flat = HashMap::new();
        for (from, mut to) in aliases {
            while let Some(&id) = alias_flat.get(&to) {
                to = id;
            }

            alias_flat.insert(from, to);
        }

        for (from, to) in alias_flat {
            log::debug!("merging {} into {}", from, to);
            let x = used_vars.remove(&from).expect("alias from nonexistent block");
            let to = used_vars.get_mut(&to).expect("alias to nonexistent block");
            to.merge(x);
        }

        Some(used_vars.into_values().collect::<Vec<_>>())
    }
}

impl LiftedFunction {
    pub fn optimize_until_complete(&mut self) -> Vec<usize> {
        (0..).map(|_| self.optimize()).take_while(|&s| s != 0).collect()
    }

    pub fn optimize(&mut self) -> usize {
        self.opt_funcs()
            + self.opt_blocks()
            + self.opt_merge_identical_blocks()
            + self.opt_merge_sequential_blocks()
            + self.opt_remove_empty_blocks()
            + self.opt_guess_names()
    }

    fn opt_funcs(&mut self) -> usize {
        self.children.iter_mut().map(LiftedFunction::optimize).sum()
    }

    fn opt_blocks(&mut self) -> usize {
        let children = &self.children[..];
        self.blocks.values_mut().map(|b| b.optimize(children)).sum()
    }

    fn opt_merge_identical_blocks(&mut self) -> usize {
        use std::collections::hash_map::Entry;

        let block_count = self.blocks.len();

        let mut rename = vec![];
        let mut blocks = HashMap::new();

        for (id, block) in std::mem::take(&mut self.blocks) {
            match blocks.entry(block) {
                Entry::Occupied(exists) => rename.push((id, *exists.get())),
                Entry::Vacant(empty) => discard(empty.insert(id)),
            }
        }

        self.blocks.extend(blocks.into_iter().map(|(b, id)| (id, b)));

        for (ref from, to) in rename {
            for block in self.blocks.values_mut() {
                for fork in block.forks_to.iter_mut() {
                    if fork == from {
                        *fork = to;
                    }
                }
            }
        }

        block_count - self.blocks.len()
    }

    fn opt_remove_empty_blocks(&mut self) -> usize {
        let block_count = self.blocks.len();

        let mut rename = vec![];

        self.blocks.retain(|&id, block| {
            // We want to avoid optimizing away specifically `while true do end`'s block
            //
            // It generates a block that consists of only `jmp -1` and since we're ignoring `jmp`
            // that means we'd normally try to optimize it away, but doing so isn't valid
            let is_while_true_do_end = matches!(*block.forks_to, [f] if f == id);

            if block.instructions.is_empty() && !is_while_true_do_end {
                rename.push((id, block.forks_to[0]));
                false
            } else {
                true
            }
        });

        for (ref from, to) in rename {
            for block in self.blocks.values_mut() {
                for fork in block.forks_to.iter_mut() {
                    if fork == from {
                        *fork = to;
                    }
                }
            }
        }

        block_count - self.blocks.len()
    }

    fn opt_merge_sequential_blocks(&mut self) -> usize {
        let mut score = 0;
        let mut block_to_origin = HashMap::<LiftedBlockId, Vec<LiftedBlockId>>::new();

        for (&origin, block) in self.blocks.iter() {
            for forks_to in block.forks_to.iter().copied() {
                block_to_origin.entry(forks_to).or_default().push(origin);
            }
        }

        let mut remapped = HashMap::new();

        for (block_id, origins) in block_to_origin {
            // If the block only has one origin
            if let [mut origin_id] = *origins {
                // Remap the origin if it has been remapped
                while let Some(&remapped_id) = remapped.get(&origin_id) {
                    origin_id = remapped_id;
                }

                let origin = &self.blocks[&origin_id];

                // And the origin only forks to one block
                if origin.forks_to == [block_id] {
                    let target_block = self.blocks.remove(&block_id).unwrap();
                    let origin_block = self.blocks.get_mut(&origin_id).unwrap();

                    remapped.insert(block_id, origin_id);
                    origin_block.merge(block_id, target_block);
                    score += 1;
                }
            }
        }

        // Update any forks to remapped blocks
        for (_, block) in self.blocks.iter_mut() {
            for fork in block.forks_to.iter_mut() {
                while let Some(&remapped_id) = remapped.get(fork) {
                    *fork = remapped_id;
                }
            }
        }

        score
    }

    fn opt_guess_names(&mut self) -> usize {
        let mut score = 0;

        for (_, block) in self.blocks.iter() {
            let an = match lifetime_analysis(&block.instructions) {
                Some(an) => an,
                None => continue,
            };

            for var in an.vars {
                let created = match var.created {
                    Some(created) => created,
                    None => continue,
                };

                let (dst, idx) = match block.instructions[created] {
                    OpCode::Closure(dst, Arg(_, ArgSlot::FnConstant(idx))) => (dst, idx),
                    OpCode::Custom(ExtOpCode::Closure(
                        dst,
                        Arg(_, ArgSlot::FnConstant(idx)),
                        _,
                    )) => (dst, idx),
                    _ => continue,
                };

                for used in var.uses {
                    match block.instructions[used] {
                        OpCode::SetGlobal(src, Arg(_, ArgSlot::Constant(g))) if src == dst => {
                            match (
                                self.children.get_mut(idx as usize),
                                self.constants.get(g.0 as usize),
                            ) {
                                (Some(func), Some(LiftedConstant::String(cnst))) => {
                                    match &func.name {
                                        Some(name) if name == cnst => continue,
                                        Some(name) if name != cnst => panic!("multiple names"),
                                        _ => {
                                            score += 1;
                                            func.set_name(cnst)
                                        },
                                    }
                                },
                                _ => continue,
                            }
                        },
                        _ => continue,
                    }
                }
            }
        }

        score
    }
}

#[derive(Debug, PartialEq)]
pub enum LiftedConstant {
    Nil,
    Bool(bool),
    Number(f64),
    String(String),
}

impl LiftedConstant {
    pub fn lift(p: &raw::LuaConstant) -> LiftedConstant {
        match p {
            raw::LuaConstant::Nil => LiftedConstant::Nil,
            &raw::LuaConstant::Bool(x) => LiftedConstant::Bool(x),
            raw::LuaConstant::Number(x) => LiftedConstant::Number(x.0),
            raw::LuaConstant::String(x) => LiftedConstant::String(x.to_string()),
        }
        //
    }
}

impl std::fmt::Display for LiftedConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiftedConstant::Nil => write!(f, "nil"),
            LiftedConstant::Bool(b) => b.fmt(f),
            LiftedConstant::Number(n) => n.fmt(f),
            LiftedConstant::String(s) => <String as std::fmt::Debug>::fmt(s, f),
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct LifetimeVariable {
    pub arg: ArgSlot,
    pub created: Option<usize>,
    pub deleted: Option<usize>,
    pub uses: Vec<usize>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct LifetimeBlock {
    pub imported: HashSet<ArgSlot>,
    pub exported: HashSet<ArgSlot>,
    pub vars: Vec<LifetimeVariable>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct LiftedBlock {
    pub instructions: Vec<OpCode>,
    pub forks_to: Vec<LiftedBlockId>,
}

impl LiftedBlock {
    pub fn lift(p: Block<'_, '_>) -> LiftedBlock {
        let instructions = p.instructions.into_iter().map(|x| x.op_code.clone()).collect();
        let forks_to = p.forks_to.into_iter().map(|i| LiftedBlockId(i.addr)).collect();

        LiftedBlock { instructions, forks_to }
    }

    pub fn merge(&mut self, id: LiftedBlockId, block: LiftedBlock) {
        if self.forks_to != [id] {
            panic!("invalid merge: {:?} != [{:?}]", self.forks_to, id);
        }

        self.instructions.extend(block.instructions);
        self.forks_to = block.forks_to;
    }

    pub fn push_front(&mut self, i: impl IntoIterator<Item = impl Into<OpCode>>) {
        self.instructions.splice(0..0, i.into_iter().map(Into::into));
    }

    pub fn push_back(&mut self, i: impl IntoIterator<Item = impl Into<OpCode>>) {
        self.instructions.extend(i.into_iter().map(Into::into));
    }
}

impl LiftedBlock {
    pub fn optimize(&mut self, protos: &[LiftedFunction]) -> usize {
        self.opt_custom_opcodes(protos)
            + self.opt_eliminate_oneshot_variables()
            + self.opt_eliminate_duplicate_forks()
            + self.opt_eliminate_nop()
    }

    fn opt_custom_opcodes(&mut self, protos: &[LiftedFunction]) -> usize {
        let mut score = 0;
        let mut instructions = Vec::with_capacity(self.instructions.len());

        let mut drain = self.instructions.drain(..);

        while let Some(instr) = drain.next() {
            match instr {
                // Rewrite OpCode::Closure and the slot allocations following it into our own equivalent
                OpCode::Closure(dest, Arg(dir, ArgSlot::FnConstant(cidx))) => {
                    let upvc = protos[cidx as usize].count_upvals.into();
                    let mut upvs = vec![];

                    for _ in 0..upvc {
                        match drain.next() {
                            Some(OpCode::Move(_, src)) => upvs.push(src),
                            Some(OpCode::GetUpVal(_, src)) => upvs.push(src),
                            None => panic!("ran out of instructions while constructing closure"),
                            _ => panic!("unexpected instruction while constructing closure"),
                        }
                    }

                    let op = ExtOpCode::Closure(dest, Arg(dir, ArgSlot::FnConstant(cidx)), upvs);
                    instructions.push(OpCode::Custom(op));

                    score += 1;
                },
                // Let any instructions we're not interested in pass
                _ => instructions.push(instr),
            }
        }

        drop(drain);
        self.instructions = instructions;

        score
    }

    fn opt_eliminate_nop(&mut self) -> usize {
        let len = self.instructions.len();

        let control_flow = match self.forks_to.len() {
            1 => None,
            _ => self.instructions.pop(),
        };

        // Only retain opcodes that aren't ONLY branching
        #[allow(clippy::match_like_matches_macro)]
        self.instructions.retain(|op_code| match op_code {
            OpCode::Jmp(_) => false,
            OpCode::LessThan(_, _, _) => false,
            OpCode::Equals(_, _, _) => false,
            OpCode::Custom(ExtOpCode::Nop) => false,
            _ => true,
        });

        if let Some(control_flow) = control_flow {
            self.instructions.push(control_flow);
        }

        len - self.instructions.len()
    }

    fn opt_eliminate_duplicate_forks(&mut self) -> usize {
        let len = self.forks_to.len();
        self.forks_to.dedup();
        len - self.forks_to.len()
    }

    fn opt_eliminate_oneshot_variables(&mut self) -> usize {
        let mut score = 0;

        let lifetime_data = match lifetime_analysis(&self.instructions) {
            Some(lifetime_data) => lifetime_data,
            _ => return 0,
        };

        for LifetimeVariable { arg, created, deleted, mut uses } in lifetime_data.vars {
            // We're only interested in variables that were both created and discarded in this block
            let created = match (created, deleted) {
                (Some(created), Some(_)) => created,
                _ => continue,
            };

            uses.sort_unstable();

            if let [used] = *uses {
                log::trace!("only used once: {} ({} -> {})", arg, created, uses[0]);

                let (solo_arg, used_arg) = match self.instructions.get(created) {
                    Some(&OpCode::GetGlobal(Arg(_, dst), Arg(_, ArgSlot::Constant(cid)))) => {
                        (dst, ArgSlot::Global(cid))
                    },
                    Some(&OpCode::LoadK(
                        Arg(_, dst @ ArgSlot::Register(_)),
                        Arg(_, cid @ ArgSlot::Constant(_)),
                    )) => (dst, cid),
                    _ => continue,
                };

                let swappable_args = match self.instructions.get_mut(used) {
                    Some(OpCode::SetGlobal(Arg(_, src), _)) => vec![src],
                    Some(OpCode::GetTable(_, Arg(_, tbl), Arg(_, idx))) => vec![tbl, idx],
                    Some(OpCode::SetTable(Arg(_, tbl), Arg(_, idx), Arg(_, val))) => {
                        vec![tbl, idx, val]
                    },
                    Some(OpCode::SetUpVal(Arg(_, val), _)) => vec![val],
                    // Comparisons
                    Some(OpCode::Equals(_, Arg(_, lhs), Arg(_, rhs))) => vec![lhs, rhs],
                    Some(OpCode::LessThan(_, Arg(_, lhs), Arg(_, rhs))) => vec![lhs, rhs],
                    Some(OpCode::LessThanOrEquals(_, Arg(_, lhs), Arg(_, rhs))) => vec![lhs, rhs],
                    // Some(LiftedOpCode(OpCode::Call()))
                    _ => continue,
                };

                for arg in swappable_args {
                    if *arg == solo_arg {
                        *arg = used_arg;
                    }
                }

                self.instructions[created] = OpCode::Custom(ExtOpCode::Nop);
                score += 1;
            }
        }

        score
    }
}

fn lifetime_analysis(instructions: &[OpCode]) -> Option<LifetimeBlock> {
    let mut imported = HashSet::new();
    let mut exported = HashSet::new();

    // Vec<(Arg, from, Vec<uses>)
    let mut used_shut = vec![];
    let mut used_open = HashMap::<ArgSlot, (Option<usize>, Vec<usize>)>::new();

    for (idx, instr) in instructions.iter().enumerate() {
        // If we run into any of these instructions, surrender completely for now
        #[allow(clippy::single_match)]
        match instr {
            OpCode::Closure(_, _) => return None,
            OpCode::Close(_) => return None,
            _ => {},
        }

        for Arg(_, slot) in instr.args_used().into_iter() {
            used_open.entry(slot).or_default().1.push(idx);
        }

        for Arg(_, slot) in instr.args_out() {
            // Register a new create
            let old_use = used_open.insert(slot, (Some(idx), vec![]));

            // If one currently exists, close it, the current instruction overwrites the value and kills it
            if let Some((start, uses)) = old_use {
                used_shut.push((slot, start, Some(idx), uses));
            };
        }

        if let OpCode::Return(_, _) = instr {
            for (arg, (from, uses)) in used_open.drain() {
                used_shut.push((arg, from, Some(idx), uses))
            }
        }
    }

    used_open.drain().for_each(|(arg, (start, uses))| {
        used_shut.push((arg, start, None, uses));
    });

    let vars = used_shut
        .into_iter()
        .map(|(arg, created, deleted, uses)| {
            if created.is_none() {
                imported.insert(arg);
            }

            if deleted.is_none() {
                exported.insert(arg);
            }

            LifetimeVariable { arg, created, deleted, uses }
        })
        .collect();

    Some(LifetimeBlock { imported, exported, vars })
}
