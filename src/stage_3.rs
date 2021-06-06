use shrinkwraprs::Shrinkwrap;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::stage_1 as s1;
use crate::stage_2::{self as s2, OpInfo};
use crate::util::discard;

#[derive(Shrinkwrap, Debug, Eq, PartialEq, Clone, Copy, PartialOrd, Ord, Hash)]
pub struct BlockId(pub s2::InstructionRef);

#[derive(Debug)]
pub struct Function {
    pub name: Option<String>,
    pub blocks: BTreeMap<BlockId, Block>,
    pub children: Vec<Function>,
    pub constants: Vec<Constant>,
    pub count_upvals: u8,
    pub count_args: u8,
    pub used_slots: HashSet<s2::ArgSlot>,
}

#[derive(Shrinkwrap, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VariableId(usize);

impl std::fmt::Display for VariableId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.0.fmt(f) }
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub id: VariableId,
    pub origin: (BlockId, usize),

    pub replaces: BTreeSet<VariableId>,
    pub uses: HashSet<(BlockId, usize)>,
    pub mods: HashSet<(BlockId, usize)>,

    pub arg: s2::ArgSlot,
}

impl Variable {
    pub fn new(id: VariableId, origin: (BlockId, usize), arg: s2::ArgSlot) -> Variable {
        Variable {
            id,
            origin,
            replaces: Default::default(),
            uses: Default::default(),
            mods: Default::default(),
            arg,
        }
    }

    pub fn new_replacement(&self, id: VariableId, origin: (BlockId, usize)) -> Variable {
        let mut replaces = self.replaces.clone();
        replaces.insert(self.id);

        let mut replacement = Variable::new(id, origin, self.arg);
        replacement.replaces = replaces;
        replacement
    }

    pub fn add_use(&mut self, block: BlockId, instr: usize) { self.uses.insert((block, instr)); }

    pub fn add_mod(&mut self, block: BlockId, instr: usize) { self.mods.insert((block, instr)); }

    pub fn merge(&mut self, from: Variable) {
        // Check so we're only merging blocks that make sense to merge
        //
        // More of a failsafe than an actual requirement, but you probably want to keep it since
        // if it gets hit something is most likely wrong somewhere.
        if !self.is_aliasing_same(&from) && !self.is_equivalent(&from) {
            log::warn!("strange merge:\n{:?}\n{:?}", self, from);
        }

        // Anything `from` replaced we also replace, except ourselves
        self.replaces.extend(from.replaces);
        self.replaces.remove(&self.id);

        // Register the merged block's creation as a modification to the existing variable
        self.mods.insert(from.origin);

        // All uses and mods the block we're merging has are now ours
        self.uses.extend(from.uses);
        self.mods.extend(from.mods);

        // If we now have uses/mods markers from our own origin, we want to get rid of those
        self.uses.remove(&self.origin);
        self.mods.remove(&self.origin);
    }

    pub fn is_aliasing_same(&self, other: &Variable) -> bool {
        // Caused by write to register in an inner scope -> modification of existing variable
        other.replaces.contains(&self.id) || !other.replaces.is_disjoint(&self.replaces)
    }

    pub fn is_equivalent(&self, other: &Variable) -> bool {
        // Multiple forks ended up at the same block and created the same variable
        self.origin == other.origin && self.arg == other.arg
    }
}

impl Function {
    pub fn lift(p: s2::Function<'_>) -> Function {
        // Generate and lift all blocks
        let mut blocks: BTreeMap<_, _> =
            p.blocks().into_iter().map(|(id, block)| (BlockId(id), Block::lift(block))).collect();

        // Lift all constants
        let constants: Vec<_> = p.constants.iter().copied().map(Constant::lift).collect();

        // Lift all inner functions
        let children = p.prototypes.into_iter().map(Function::lift).collect();

        // Find all slots used by the function, they're used mainly in the code below
        let mut slots = HashSet::new();
        for block in blocks.values() {
            for instr in block.instructions.iter() {
                slots.extend(instr.args_in().into_iter().map(|x| x.slot()));
                slots.extend(instr.args_inout().into_iter().map(|x| x.slot()));
                slots.extend(instr.args_out().into_iter().map(|x| x.slot()));
            }
        }

        // TODO: Get rid of function fallback `ExtOpCode::DefineReg`s
        // This is a ""temporary"" way to handle variables that are used in a deeper scope than
        // the one they're defined in. Think for example this:
        //
        // ```lua
        // local x
        // if false then x = 10 else x = 20 end
        // return x
        // ```
        //
        // There's two main ways to get rid of this:
        //  - At merge, determine which block the variable is most likely defined in and inject
        //    a DefineReg instruction at the start/end of it.
        //  - Keep the code below, but write an opt pass that moves all DefineReg instances to
        //    the block they're most likely supposed to be in.
        //
        // Both ways have their own upsides and downsides and there's bound to be far more
        // alternatives that I haven't thought of.
        if let Some((&first_block_id, _)) = blocks.first_key_value() {
            let mut regs =
                slots.iter().copied().filter(s2::ArgSlot::is_reg).collect::<BTreeSet<_>>();

            let mut mark_opcs: Vec<s2::OpCode> = vec![];

            if p.raw.arg_count > 0 {
                mark_opcs.push(s2::ExtOpCode::comment(format!("args: {}", p.raw.arg_count)).into());
                for reg in 0..p.raw.arg_count {
                    // We want to avoid shadowing any arguments
                    regs.remove(&s2::ArgSlot::Register(reg as _));
                }
            }

            if !regs.is_empty() {
                // TODO: Remove these comments once we start optimizing out ExtOpCode::DefineReg
                mark_opcs.push(s2::ExtOpCode::comment("fallback defines").into());
                mark_opcs.extend(regs.into_iter().map(s2::ExtOpCode::DefineReg).map(Into::into));
                mark_opcs.push(s2::ExtOpCode::comment("fallback defines end").into());
            }

            blocks.insert(
                BlockId(s2::InstructionRef::FunctionEntry),
                Block { instructions: mark_opcs, forks_to: vec![first_block_id] },
            );
        }

        Function {
            name: None,
            blocks,
            children,
            constants,
            count_upvals: p.raw.up_val_count,
            count_args: p.raw.arg_count,
            used_slots: slots,
        }
    }

    pub fn set_name(&mut self, name: impl Into<String>) { self.name = Some(name.into()); }

    pub fn with_name(mut self, name: impl Into<String>) -> Function {
        self.set_name(name);
        self
    }

    pub fn variables(&self) -> Option<Vec<Variable>> {
        let func = self;
        let consts = &func.constants;

        let mut variable_id = 0;
        let mut new_var = || {
            variable_id += 1;
            VariableId(variable_id - 1)
        };

        let arguments = (0..self.count_args)
            .map(|i| {
                Variable::new(
                    new_var(),
                    (BlockId(s2::InstructionRef::FunctionEntry), 0),
                    s2::ArgSlot::Register(i as _),
                )
            })
            .map(|x| (x.arg, x));

        let mut block_stack = vec![(
            *func.blocks.first_key_value().unwrap().0,
            BlockId(s2::InstructionRef::FunctionEntry),
            arguments.collect::<HashMap<s2::ArgSlot, Variable>>(),
        )];

        let mut used_shut = vec![];
        let mut seen = HashSet::<(BlockId, BlockId)>::new();
        while let Some((block_id, origin, mut live_vars)) = block_stack.pop() {
            if !seen.insert((block_id, origin)) {
                // If we've already taken and analyzed the origin->block_id jump, ignore it
                // This helps preventing recursion, but we want to make sure we're not failing
                // to propagate imports/exports in extreme conditions. If we're failing some
                // very obscure edge case in loops where lifetimes aren't propagated properly,
                // start looking at whether this code is the problem or not.

                // If we hit a loop, close all variables we're holding.
                // They'll eventually be merged into their originals later.
                used_shut.extend(live_vars.drain().map(|(_, x)| x));
                continue;
            }

            let block = func.blocks.get(&block_id).unwrap_or_else(|| {
                panic!("block_id does not exist, opt pass failure? {:?}", block_id)
            });

            let instructions = &block.instructions;
            for (idx, instr) in instructions.iter().enumerate() {
                // TODO: Make a decision about closure upvalue handling.
                // We might want to handle closure upvalues a bit specially, so they can't be
                // eliminated due to "not being used" when they're in reality used in the
                // closure. Or, we can just panic on them and tell people to run the opt pass that
                // replaces them with our own custom instruction instead. The latter of those two
                // options is what we're doing right now.

                // If we run into any OpCode::Closure calls, surrender completely.
                // Refer to the comment above.
                if let s2::OpCode::Closure(_, _) = instr {
                    log::warn!("unsupported instruction: {}", instr.text(consts));
                    return None;
                }

                for arg in instr.args_used().into_iter().filter(|x| x.slot().is_reg()) {
                    match (arg.dir(), live_vars.get_mut(&arg)) {
                        (s2::ArgDir::Read, Some(x)) => x.add_use(block_id, idx),
                        (s2::ArgDir::Modify, Some(x)) => x.add_mod(block_id, idx),
                        _ => {
                            log::warn!("reading unwritten {} {:?} ({})", arg, block_id, idx);
                            return None;
                        },
                    }
                }

                instr
                    .args_out()
                    .into_iter()
                    .map(s2::Arg::slot)
                    .filter(s2::ArgSlot::is_reg)
                    .for_each(|slot| {
                        let id = new_var();
                        let origin = (block_id, idx);

                        // If a var using this write exists, replace and close it
                        let var = if let Some(var_old) = live_vars.remove(&slot) {
                            let replacement = var_old.new_replacement(id, (block_id, idx));
                            used_shut.push(var_old);
                            replacement
                        } else {
                            Variable::new(id, origin, slot)
                        };

                        live_vars.insert(slot, var);
                    });

                if let s2::OpCode::Return(_, _) = instr {
                    used_shut.extend(live_vars.drain().map(|(_, x)| x));
                }
            }

            block_stack.extend(block.forks_to.iter().map(|&x| (x, block_id, live_vars.clone())));
        }

        used_shut.sort_by_cached_key(|x| x.id);

        let mut used_shut_merged = BTreeMap::<VariableId, Variable>::new();
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
        let mut usage_to_var: HashMap<(s2::ArgDir, s2::ArgSlot, (BlockId, usize)), VariableId> =
            HashMap::new();

        let mut collisions = HashMap::<VariableId, VariableId>::new();

        for var in used_vars.values() {
            let mut register = |dir, origin| {
                usage_to_var
                    .entry((dir, var.arg, origin))
                    .and_modify(|x| {
                        if var.id != *x {
                            collisions.insert(var.id, *x);
                        }
                    })
                    .or_insert(var.id);
            };

            register(s2::ArgDir::Write, var.origin);

            for origin in var.uses.iter().copied() {
                register(s2::ArgDir::Read, origin);
            }

            for origin in var.mods.iter().copied() {
                register(s2::ArgDir::Modify, origin);
            }
        }

        // dbg!(&collisions);
        // dbg!(&usage_to_var);
        let mut merged = HashMap::new();

        for (mut from, mut to) in collisions {
            // If we've already re-aliased the `from` variable, resolve it to what it's called now
            while let Some(&root) = merged.get(&from) {
                from = root;
            }

            // If we've already re-aliased the `to` variable, resolve it to what it's called now
            while let Some(&root) = merged.get(&to) {
                to = root;
            }

            if from == to {
                // If both variables have already been aliased to the same variable, there's not
                // a whole lot left for us to take care of, so we'll just ignore it completely.
                continue;
            }

            log::debug!("investigating merge (overlap: {} v {})", from, to);

            let mut n1 = used_vars.remove(&from).expect("n1 alias from nonexistent variable");
            let mut n2 = used_vars.remove(&to).expect("n2 alias to nonexistent variable");

            match () {
                () if n1.is_equivalent(&n2) => {
                    log::debug!("merging {} into {} (equivalent)", n2.id, n1.id);

                    merged.insert(n2.id, n1.id);
                    n1.merge(n2);
                    used_vars.insert(n1.id, n1);
                },
                () => {
                    let mut root_id = match n1.replaces.intersection(&n2.replaces).last() {
                        Some(&root_id) => root_id,
                        // TODO: Make sure it's sound to take these two branches. It *should* be.
                        // If one of the vars aliases the other, assume they're the same var
                        _ if n1.replaces.contains(&n2.id) => n2.id,
                        _ if n2.replaces.contains(&n1.id) => n1.id,
                        // Emergency fallback
                        _ => {
                            // If we're here, we've most likely ran into a scenario like this:
                            // ```lua
                            // local x
                            // if false then x = 10 else x = 20 end
                            // return x
                            // ```
                            // When this happens we end up encountering a variable for the first
                            // time in a scope it isn't allowed to be defined in (since the return
                            // can't access it if it's defined in one of the if's blocks.
                            //
                            // Our solution to this is to find the last common ancestor shared by
                            // all blocks the variable is used in and assume it was defined in that
                            // block instead.
                            //
                            // TODO: ExtOpCode::DefineReg
                            // We'll later probably want to inject a custom instruction that only
                            // serves to mark an explicit write of `nil` to a register when we run
                            // into cases like this. When we later emit source we'll then be able
                            // to use that marker to create but not initialize the var, but for now
                            // we'll leave that plan for the future.

                            log::error!("abandoning merge!: {:#?} v {:#?}", n1, n2);
                            used_vars.insert(n1.id, n1);
                            used_vars.insert(n2.id, n2);
                            continue;
                        },
                    };

                    while let Some(&root) = merged.get(&root_id) {
                        root_id = root;
                    }

                    log::debug!("merging {} and {} into {}", n1.id, n2.id, root_id);

                    match () {
                        () if n1.id == root_id => {
                            merged.insert(n2.id, n1.id);
                            n1.merge(n2);
                            used_vars.insert(n1.id, n1);
                        },
                        () if n2.id == root_id => {
                            merged.insert(n1.id, n2.id);
                            n2.merge(n1);
                            used_vars.insert(n2.id, n2);
                        },
                        () => {
                            let root = used_vars.get_mut(&root_id).expect("can't find root var");

                            merged.insert(n2.id, root.id);
                            merged.insert(n1.id, root.id);

                            root.merge(n1);
                            root.merge(n2);
                        },
                    }
                },
            }
        }

        Some(used_vars.into_values().collect::<Vec<_>>())
    }
}

impl Function {
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

    fn opt_funcs(&mut self) -> usize { self.children.iter_mut().map(Function::optimize).sum() }

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
        let mut block_to_origin = HashMap::<BlockId, Vec<BlockId>>::new();

        for (&origin, block) in self.blocks.iter() {
            for forks_to in block.forks_to.iter().copied() {
                block_to_origin.entry(forks_to).or_default().push(origin);
            }
        }

        let mut remapped = HashMap::new();

        for (block_id, origins) in block_to_origin {
            // If the block only has one origin
            if let [mut origin_id] = *origins {
                // If we're considering merging into FunctionEntry, don't do it
                if origin_id == BlockId(s2::InstructionRef::FunctionEntry) {
                    continue;
                }

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
                    s2::OpCode::Closure(dst, s2::Arg(_, s2::ArgSlot::FnConstant(idx))) => {
                        (dst, idx)
                    },
                    s2::OpCode::Custom(s2::ExtOpCode::Closure(
                        dst,
                        s2::Arg(_, s2::ArgSlot::FnConstant(idx)),
                        _,
                    )) => (dst, idx),
                    _ => continue,
                };

                for used in var.uses {
                    match block.instructions[used] {
                        s2::OpCode::SetGlobal(src, s2::Arg(_, s2::ArgSlot::Constant(g)))
                            if src == dst =>
                        {
                            match (
                                self.children.get_mut(idx as usize),
                                self.constants.get(g.0 as usize),
                            ) {
                                (Some(func), Some(Constant::String(cnst))) => match &func.name {
                                    Some(name) if name == cnst => continue,
                                    Some(name) if name != cnst => panic!("multiple names"),
                                    _ => {
                                        score += 1;
                                        func.set_name(cnst)
                                    },
                                },
                                _ => continue,
                            }
                        }
                        _ => continue,
                    }
                }
            }
        }

        score
    }
}

#[derive(Debug, PartialEq)]
pub enum Constant {
    Nil,
    Bool(bool),
    Number(f64),
    String(String),
}

impl Constant {
    pub fn lift(p: &s1::LuaConstant) -> Constant {
        match p {
            s1::LuaConstant::Nil => Constant::Nil,
            &s1::LuaConstant::Bool(x) => Constant::Bool(x),
            s1::LuaConstant::Number(x) => Constant::Number(x.0),
            s1::LuaConstant::String(x) => Constant::String(x.to_string()),
        }
        //
    }
}

impl std::fmt::Display for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Constant::Nil => write!(f, "nil"),
            Constant::Bool(b) => b.fmt(f),
            Constant::Number(n) => n.fmt(f),
            Constant::String(s) => <String as std::fmt::Debug>::fmt(s, f),
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct LifetimeVariable {
    pub arg: s2::ArgSlot,
    pub created: Option<usize>,
    pub deleted: Option<usize>,
    pub uses: Vec<usize>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct LifetimeBlock {
    pub imported: HashSet<s2::ArgSlot>,
    pub exported: HashSet<s2::ArgSlot>,
    pub vars: Vec<LifetimeVariable>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Block {
    pub instructions: Vec<s2::OpCode>,
    pub forks_to: Vec<BlockId>,
}

impl Block {
    pub fn lift(p: s2::Block<'_, '_>) -> Block {
        let instructions = p.instructions.into_iter().map(|x| x.op_code.clone()).collect();
        let forks_to = p.forks_to.into_iter().map(|i| BlockId(i.addr)).collect();

        Block { instructions, forks_to }
    }

    pub fn merge(&mut self, id: BlockId, block: Block) {
        if self.forks_to != [id] {
            panic!("invalid merge: {:?} != [{:?}]", self.forks_to, id);
        }

        self.instructions.extend(block.instructions);
        self.forks_to = block.forks_to;
    }

    pub fn push_front(&mut self, i: impl IntoIterator<Item = impl Into<s2::OpCode>>) {
        self.instructions.splice(0..0, i.into_iter().map(Into::into));
    }

    pub fn push_back(&mut self, i: impl IntoIterator<Item = impl Into<s2::OpCode>>) {
        self.instructions.extend(i.into_iter().map(Into::into));
    }
}

impl Block {
    pub fn optimize(&mut self, protos: &[Function]) -> usize {
        self.opt_custom_opcodes(protos)
            + self.opt_eliminate_oneshot_variables()
            + self.opt_eliminate_identical_forks()
            + self.opt_eliminate_nop()
    }

    fn opt_custom_opcodes(&mut self, protos: &[Function]) -> usize {
        let mut score = 0;
        let mut instructions = Vec::with_capacity(self.instructions.len());

        let mut drain = self.instructions.drain(..);

        while let Some(instr) = drain.next() {
            match instr {
                // Rewrite OpCode::Closure and the slot allocations following it into our own equivalent
                s2::OpCode::Closure(dest, s2::Arg(dir, s2::ArgSlot::FnConstant(cidx))) => {
                    let upvc = protos[cidx as usize].count_upvals.into();
                    let mut upvs = vec![];

                    for _ in 0..upvc {
                        match drain.next() {
                            Some(s2::OpCode::Move(_, src)) => upvs.push(src),
                            Some(s2::OpCode::GetUpVal(_, src)) => upvs.push(src),
                            None => panic!("ran out of instructions while constructing closure"),
                            _ => panic!("unexpected instruction while constructing closure"),
                        }
                    }

                    let op = s2::ExtOpCode::Closure(
                        dest,
                        s2::Arg(dir, s2::ArgSlot::FnConstant(cidx)),
                        upvs,
                    );
                    instructions.push(s2::OpCode::Custom(op));

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
            s2::OpCode::Jmp(_) => false,
            s2::OpCode::LessThan(_, _, _) => false,
            s2::OpCode::Equals(_, _, _) => false,
            s2::OpCode::Custom(s2::ExtOpCode::Nop) => false,
            _ => true,
        });

        if let Some(control_flow) = control_flow {
            self.instructions.push(control_flow);
        }

        len - self.instructions.len()
    }

    fn opt_eliminate_identical_forks(&mut self) -> usize {
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
                    Some(&s2::OpCode::GetGlobal(
                        s2::Arg(_, dst),
                        s2::Arg(_, s2::ArgSlot::Constant(cid)),
                    )) => (dst, s2::ArgSlot::Global(cid)),
                    Some(&s2::OpCode::LoadK(
                        s2::Arg(_, dst @ s2::ArgSlot::Register(_)),
                        s2::Arg(_, cid @ s2::ArgSlot::Constant(_)),
                    )) => (dst, cid),
                    _ => continue,
                };

                let swappable_args = match self.instructions.get_mut(used) {
                    Some(s2::OpCode::SetGlobal(s2::Arg(_, src), _)) => vec![src],
                    Some(s2::OpCode::GetTable(_, s2::Arg(_, tbl), s2::Arg(_, idx))) => {
                        vec![tbl, idx]
                    },
                    Some(s2::OpCode::SetTable(
                        s2::Arg(_, tbl),
                        s2::Arg(_, idx),
                        s2::Arg(_, val),
                    )) => {
                        vec![tbl, idx, val]
                    },
                    Some(s2::OpCode::SetUpVal(s2::Arg(_, val), _)) => vec![val],
                    // Comparisons
                    Some(s2::OpCode::Equals(_, s2::Arg(_, lhs), s2::Arg(_, rhs))) => {
                        vec![lhs, rhs]
                    },
                    Some(s2::OpCode::LessThan(_, s2::Arg(_, lhs), s2::Arg(_, rhs))) => {
                        vec![lhs, rhs]
                    },
                    Some(s2::OpCode::LessThanOrEquals(_, s2::Arg(_, lhs), s2::Arg(_, rhs))) => {
                        vec![lhs, rhs]
                    },
                    _ => continue,
                };

                for arg in swappable_args {
                    if *arg == solo_arg {
                        *arg = used_arg;
                    }
                }

                self.instructions[created] = s2::OpCode::Custom(s2::ExtOpCode::Nop);
                score += 1;
            }
        }

        score
    }
}

fn lifetime_analysis(instructions: &[s2::OpCode]) -> Option<LifetimeBlock> {
    let mut imported = HashSet::new();
    let mut exported = HashSet::new();

    // Vec<(Arg, from, Vec<uses>)
    let mut used_shut = vec![];
    let mut used_open = HashMap::<s2::ArgSlot, (Option<usize>, Vec<usize>)>::new();

    for (idx, instr) in instructions.iter().enumerate() {
        for crate::Arg(_, slot) in instr.args_used().into_iter() {
            used_open.entry(slot).or_default().1.push(idx);
        }

        for s2::Arg(_, slot) in instr.args_out() {
            // Register a new create
            let old_use = used_open.insert(slot, (Some(idx), vec![]));

            // If one currently exists, close it, the current instruction overwrites the value and kills it
            if let Some((start, uses)) = old_use {
                used_shut.push((slot, start, Some(idx), uses));
            };
        }

        if let s2::OpCode::Return(_, _) = instr {
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
