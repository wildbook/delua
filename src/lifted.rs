use std::collections::{BTreeMap, HashMap, HashSet};

use shrinkwraprs::Shrinkwrap;

use crate::{
    parsed::{Arg, Block, ExtOpCode, Function, InstructionRef, OpCode, OpInfo},
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
    pub up_val_count: u8,
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

        LiftedFunction { name: None, blocks, children, constants, up_val_count: p.raw.up_val_count }
    }

    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    pub fn with_name(mut self, name: impl Into<String>) -> LiftedFunction {
        self.set_name(name);
        self
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
            if block.instructions.is_empty() {
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
                    OpCode::Closure(dst, Arg::FnConstant(idx)) => (dst, idx),
                    OpCode::Custom(ExtOpCode::Closure(dst, Arg::FnConstant(idx), _)) => (dst, idx),
                    _ => continue,
                };

                for used in var.uses {
                    match block.instructions[used] {
                        OpCode::SetGlobal(src, Arg::Constant(g)) if src == dst => {
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
    pub arg: Arg,
    pub created: Option<usize>,
    pub deleted: Option<usize>,
    pub uses: Vec<usize>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct LifetimeBlock {
    pub imported: HashSet<Arg>,
    pub exported: HashSet<Arg>,
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
                OpCode::Closure(dest, Arg::FnConstant(cidx)) => {
                    let upvc = protos[cidx as usize].up_val_count.into();
                    let mut upvs = vec![];

                    for _ in 0..upvc {
                        match drain.next() {
                            Some(OpCode::Move(_, src)) => upvs.push(src),
                            Some(OpCode::GetUpVal(_, src)) => upvs.push(src),
                            None => panic!("ran out of instructions while constructing closure"),
                            _ => panic!("unexpected instruction while constructing closure"),
                        }
                    }

                    let op = ExtOpCode::Closure(dest, Arg::FnConstant(cidx), upvs);
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
                log::debug!("only used once: {} ({} -> {})", arg, created, uses[0]);

                let (solo_arg, used_arg) = match self.instructions.get(created) {
                    Some(OpCode::GetGlobal(dst, Arg::Constant(cid))) => (*dst, Arg::Global(*cid)),
                    Some(OpCode::LoadK(dst @ Arg::Register(_), cid @ Arg::Constant(_))) => {
                        (*dst, *cid)
                    },
                    _ => continue,
                };

                let swappable_args = match self.instructions.get_mut(used) {
                    Some(OpCode::SetGlobal(src, _)) => vec![src],
                    Some(OpCode::GetTable(_, tbl, idx)) => vec![tbl, idx],
                    Some(OpCode::SetTable(tbl, idx, val)) => vec![tbl, idx, val],
                    Some(OpCode::SetUpVal(val, _)) => vec![val],
                    // Comparisons
                    Some(OpCode::Equals(_, lhs, rhs)) => vec![lhs, rhs],
                    Some(OpCode::LessThan(_, lhs, rhs)) => vec![lhs, rhs],
                    Some(OpCode::LessThanOrEquals(_, lhs, rhs)) => vec![lhs, rhs],
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

impl LiftedBlock {
    pub fn lifetime_analysis(&self) -> Option<LifetimeBlock> {
        lifetime_analysis(&self.instructions)
    }
}

fn lifetime_analysis(instructions: &[OpCode]) -> Option<LifetimeBlock> {
    let mut imported = HashSet::new();
    let mut exported = HashSet::new();

    // Vec<(Arg, from, Vec<uses>)
    let mut used_shut = vec![];
    let mut used_open = HashMap::<Arg, (Option<usize>, Vec<usize>)>::new();

    for (idx, instr) in instructions.iter().enumerate() {
        // If we run into any of these instructions, surrender completely for now
        #[allow(clippy::single_match)]
        match instr {
            OpCode::Closure(_, _) => return None,
            OpCode::Close(_) => return None,
            _ => {},
        }

        for read in instr.arguments_read() {
            used_open.entry(read).or_default().1.push(idx);
        }

        for write in instr.arguments_write() {
            // Register a new write / create
            let old_use = used_open.insert(write, (Some(idx), vec![]));

            // If one currently exists, close it, the current instruction overwrites the value and kills it
            if let Some((start, uses)) = old_use {
                used_shut.push((write, start, Some(idx), uses));
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
