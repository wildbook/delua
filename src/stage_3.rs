use shrinkwraprs::Shrinkwrap;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::stage_1 as s1;
use crate::stage_2::{self as s2, OpInfo};
use crate::util::discard;

pub type Location = (BlockId, usize);

#[derive(Shrinkwrap, Debug, Eq, PartialEq, Clone, Copy, PartialOrd, Ord, Hash)]
pub struct BlockId(pub s2::InstructionRef);

#[derive(Debug, Clone)]
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
    pub origin: Location,

    pub replaces: BTreeSet<VariableId>,
    pub reads: HashSet<Location>,
    pub mods: HashSet<Location>,

    pub arg: s2::ArgSlot,
}

impl Variable {
    /// Create a new variable.
    pub fn new(id: VariableId, origin: Location, arg: s2::ArgSlot) -> Variable {
        Variable {
            id,
            origin,
            replaces: Default::default(),
            reads: Default::default(),
            mods: Default::default(),
            arg,
        }
    }

    /// Create a new replacement variable for this one.
    ///
    /// If you're replacing an existing variable and not creating one out of thin air it's
    /// important to use this function instead of `Variable::new` since otherwise aliasing
    /// resolution and other future logic will fail, most likely in quite spectacular ways.
    pub fn new_replacement(&self, id: VariableId, origin: Location) -> Variable {
        let mut replaces = self.replaces.clone();
        replaces.insert(self.id);

        let mut replacement = Variable::new(id, origin, self.arg);
        replacement.replaces = replaces;
        replacement
    }

    /// Register a location that reads this variable.
    pub fn add_read(&mut self, location: Location) { self.reads.insert(location); }

    /// Register a location that modifies this variable.
    pub fn add_mod(&mut self, location: Location) { self.mods.insert(location); }

    /// Merge another instruction into this one.
    pub fn merge(&mut self, from: Variable) {
        // Check so we're only merging blocks that make sense to merge.
        //
        // More of a failsafe than an actual requirement, but you probably want to keep it since
        // if it gets hit something is most likely wrong somewhere.
        if !self.is_aliasing_same(&from) && !self.is_equivalent(&from) {
            log::warn!("strange merge:\n{:?}\n{:?}", self, from);
        }

        // Anything `from` replaced we also replace, except ourselves.
        self.replaces.extend(from.replaces);
        self.replaces.remove(&self.id);

        // Register the merged block's creation as a modification to the existing variable.
        self.mods.insert(from.origin);

        // All uses and mods the block we're merging has are now ours.
        self.reads.extend(from.reads);
        self.mods.extend(from.mods);

        // If we now have uses/mods markers from our own origin, we want to get rid of those.
        self.reads.remove(&self.origin);
        self.mods.remove(&self.origin);
    }

    /// Check if this variable somewhere down the line replaces the same variable as the other
    /// variable does, or is a variable that the other variable replaces.
    pub fn is_aliasing_same(&self, other: &Variable) -> bool {
        // Caused by write to register in an inner scope -> modification of existing variable.
        other.replaces.contains(&self.id) || !other.replaces.is_disjoint(&self.replaces)
    }

    /// Check if this variable is equivalent to another variable.
    /// Equivalent here means "defined at the same place with the same argument", since each OpCode
    /// can only really write to an instruction once. Even if multiple writes would be allowed,
    /// they would result in one effective write, and the earlier writes would be safe to throw
    /// away.
    pub fn is_equivalent(&self, other: &Variable) -> bool {
        // Multiple forks ended up at the same block and created the same variable.
        self.origin == other.origin && self.arg == other.arg
    }
}

impl Function {
    pub fn lift(func: s2::Function<'_>) -> Function {
        // Generate and lift all blocks
        let mut blocks: BTreeMap<_, _> = func
            .blocks()
            .into_iter()
            .map(|(id, block)| (BlockId(id), Block::lift(block)))
            .collect();

        // Lift all constants
        let constants: Vec<_> = func.constants.iter().copied().map(Constant::lift).collect();

        // Lift all inner functions
        let children = func.prototypes.into_iter().map(Function::lift).collect();

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
        // the one they're defined in.
        //
        // An example that should trigger this is:
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

            // Create a vector to store our own injected opcodes in.
            let mut mark_opcs: Vec<s2::OpCode> = vec![];

            // If there's any arguments, avoid registering them here since they're not technically
            // created within the function. Pretending they are would cause issues since we'd most
            // likely end up defining uninitialized variables that would shadow the arguments.
            //
            // If / when we get rid of the explicit ExtOpCode::DefineReg instructions below, this
            // code can also be thrown out.
            let arg_count = func.raw.arg_count as _;
            if 0 < arg_count {
                mark_opcs.push(s2::ExtOpCode::comment(format!("args: {}", arg_count)).into());
                for reg in 0..arg_count {
                    // We want to avoid shadowing any arguments.
                    regs.remove(&s2::ArgSlot::Register(reg));
                }
            }

            // If this function uses any registers (that aren't passed as arguments), create a
            // temporary variable each to fall back on. See the comment on the if this is in for
            // more detailed information.
            if !regs.is_empty() {
                // TODO: Remove these comments once we start optimizing out ExtOpCode::DefineReg.
                mark_opcs.push(s2::ExtOpCode::comment("fallback defines").into());
                mark_opcs.extend(regs.into_iter().map(s2::ExtOpCode::DefineReg).map(Into::into));
                mark_opcs.push(s2::ExtOpCode::comment("fallback defines end").into());
            }

            // Insert the instructions we just generated as a pseudo-block starting at the
            // function's entrypoint. Creating such a block instead of injecting into a normal
            // block (or pretending to be a normal block ourselves) is cleaner since it allows
            // for better analysis and/or reasoning later if needed.
            blocks.insert(
                BlockId(s2::InstructionRef::FunctionEntry),
                Block { instructions: mark_opcs, forks_to: vec![first_block_id] },
            );
        }

        // Finally, construct the function.
        Function {
            name: None,
            blocks,
            children,
            constants,
            count_upvals: func.raw.up_val_count,
            count_args: func.raw.arg_count,
            used_slots: slots,
        }
    }

    /// Set the function's name.
    pub fn set_name(&mut self, name: impl Into<String>) { self.name = Some(name.into()); }

    /// Set the function's name and return it, allowing for easier chaining.
    pub fn with_name(mut self, name: impl Into<String>) -> Function {
        self.set_name(name);
        self
    }

    /// Create a vector containing all variables used in the function.
    pub fn variables(&self) -> Option<Vec<Variable>> {
        let func = self;
        let consts = &func.constants;

        // Keep an incremental id for new variable allocations, this allows us to later
        // detect what is and isn't a collision, and so on.
        let mut variable_id = 0;

        // Convenience function to generate a new variable id with.
        let mut new_var_id = || {
            variable_id += 1;
            VariableId(variable_id - 1)
        };

        // Collect all argument registers so we can mark them as being written to before the
        // function starts executing.
        let arguments = (0..self.count_args)
            .map(|i| {
                Variable::new(
                    new_var_id(),
                    (BlockId(s2::InstructionRef::FunctionEntry), 0),
                    s2::ArgSlot::Register(i as _),
                )
            })
            .map(|x| (x.arg, x));

        // Create a block to start iterating from.
        let origin_block = (
            *func.blocks.first_key_value().unwrap().0,
            BlockId(s2::InstructionRef::FunctionEntry),
            arguments.collect::<HashMap<s2::ArgSlot, Variable>>(),
        );

        // Create a stack for us to use when iterating blocks and init it with our starting point.
        let mut block_stack = vec![origin_block];

        // Track uses that are "closed", meaning anything and everything that has reached the end
        // of its life.
        //
        // This vector will eventually end up holding duplicates and so on due to
        // branching, but we'll take care of (and rely on) that after the while loop to resolve
        // variables that are not *actually* new variables, but instead writes to an existing
        // variable.
        let mut used_closed = vec![];

        // Track all jumps we've already taken, this lets us avoid reanalyzing blocks if we've
        // already arrived at them from the same source block before. The comment where this
        // set is used contains a more detailed explanation.
        let mut taken_jumps = HashSet::<(BlockId, BlockId)>::new();

        // "Recursively" iterate all blocks on the stack.
        while let Some((block_id, origin, mut live_vars)) = block_stack.pop() {
            if !taken_jumps.insert((block_id, origin)) {
                // If we've already taken and analyzed the origin->block_id jump, ignore it
                // This helps preventing recursion, but we want to make sure we're not failing
                // to propagate imports/exports in extreme conditions. If we're failing some
                // very obscure edge case in loops where lifetimes aren't propagated properly,
                // start looking at whether this code is the problem or not.

                // If we hit a loop, close all variables we're holding.
                // They'll eventually be merged into their originals later.
                used_closed.extend(live_vars.drain().map(|(_, x)| x));
                continue;
            }

            // Try to retrieve a block by its id, this should only ever fail if an opt pass has
            // messed up and deleted or re-id'd a block without updating the forks_to variable of
            // other blocks that jump(ed) to it.
            let block = func.blocks.get(&block_id).unwrap_or_else(|| {
                panic!("block_id does not exist, opt pass failure? {:?}", block_id)
            });

            // Iterate all instructions in the block we're in.
            let instructions = &block.instructions;
            for (idx, instr) in instructions.iter().enumerate() {
                // TODO: Make a decision about closure upvalue handling.
                // We might want to handle closure upvalues a bit specially, so they can't be
                // eliminated due to "not being used" when they're in reality used in the
                // closure. Or, we can just panic on them and tell people to run the opt pass that
                // replaces them with our own custom instruction instead. The latter of those two
                // options is what we're doing right now.

                // If we run into any OpCode::Closure calls, surrender completely.
                // As for why, refer to the comment above.
                if let s2::OpCode::Closure(_, _) = instr {
                    log::error!("unsupported instruction: {}", instr.text(consts));
                    return None;
                }

                // For each argument the instruction reads, register a read or modification.
                for arg in instr.args_read().into_iter().filter(|x| x.slot().is_reg()) {
                    match (arg.dir(), live_vars.get_mut(&arg)) {
                        (s2::ArgDir::Read, Some(x)) => x.add_read((block_id, idx)),
                        (s2::ArgDir::Modify, Some(x)) => x.add_mod((block_id, idx)),
                        _ => {
                            log::warn!("reading unwritten {} {:?} ({})", arg, block_id, idx);
                            return None;
                        },
                    }
                }

                // And for each *register* an instruction writes to, register a new variable.
                // If one already exists, assume we're replacing it instead of reassigning it.
                // Reassignment guesses will be taken care of later.
                instr
                    .args_out()
                    .into_iter()
                    .map(s2::Arg::slot)
                    .filter(s2::ArgSlot::is_reg)
                    .for_each(|slot| {
                        let id = new_var_id();
                        let origin = (block_id, idx);

                        let var = if let Some(var_old) = live_vars.remove(&slot) {
                            // If a var already exists in this slot, replace and close it.
                            let replacement = var_old.new_replacement(id, (block_id, idx));
                            used_closed.push(var_old);
                            replacement
                        } else {
                            // Otherwise just summon a new variable.
                            Variable::new(id, origin, slot)
                        };

                        // And finally, start tracking the variable we created above.
                        live_vars.insert(slot, var);
                    });

                // If we run into a return we want to just close all variables we're holding.
                // This is done since if a variable isn't closed it'll not be registered as having
                // ever existed once this loop is over.
                //
                // We can't manually break out here, since in theory there might be more
                // instructions following a return. In practice I have yet to see that happen.
                if let s2::OpCode::Return(_, _) = instr {
                    used_closed.extend(live_vars.drain().map(|(_, x)| x));
                }
            }

            // Put all forks on the block stack.
            // Any recursion or branching to already known blocks will be taken care of when the
            // fork is popped off of the stack, so we can treat them as any other forks here.
            block_stack.extend(block.forks_to.iter().map(|&x| (x, block_id, live_vars.clone())));
        }

        // Sort our closed variables by their key.
        // This isn't strictly required, but it makes aliasing and so on use the first occurance
        // of the variable as the "true" variable, which leads to ids that aren't jumping all over.
        used_closed.sort_by_cached_key(|x| x.id);

        // Merge all variables that were closed more than once.
        // This happens when two blocks both use the same variable, but neither modifies or creates
        // a new variable in its place before they're closed.
        //
        // An example that should trigger this is:
        // ```lua
        // local x = "x";
        // if false then
        //     print("1: " .. x)
        //     return
        // else
        //     print("2: " .. x)
        //     return
        // end
        // ```
        let mut used_vars = BTreeMap::<VariableId, Variable>::new();
        for used in used_closed {
            match used_vars.entry(used.id) {
                // If the variable wasn't seen before, just register it.
                Entry::Vacant(e) => discard(e.insert(used)),
                // If it was seen before, merge it into the existing variable.
                Entry::Occupied(mut e) => {
                    log::debug!("merging:\n{:?}\n{:?}", e.get_mut(), used);
                    e.get_mut().merge(used)
                },
            }
        }

        // A map that takes us from a use to the variable that is used there.
        // Ideally, we'd probably want to do this in a different way, but this works and so far
        // we haven't run into any issues doing it this way. Performance might be one reason.
        let mut usage_to_var: HashMap<(s2::ArgDir, s2::ArgSlot, Location), VariableId> =
            HashMap::new();

        // A set to let us track collisions.
        let mut collisions = HashSet::<(VariableId, VariableId)>::new();

        // For each variable, register all of its write/read/mod locations.
        // This allows us to later check for overlap and merge any overlapping variables.
        for var in used_vars.values() {
            // Convenience closure to save us from some code duplication.
            let mut register = |dir, origin| {
                usage_to_var
                    .entry((dir, var.arg, origin))
                    .and_modify(|x| {
                        if var.id != *x {
                            collisions.insert((var.id, *x));
                        }
                    })
                    .or_insert(var.id);
            };

            // Register the variable's origin as a write.
            register(s2::ArgDir::Write, var.origin);

            // Register all reads as reads.
            for origin in var.reads.iter().copied() {
                register(s2::ArgDir::Read, origin);
            }

            // And finally, register all mods as mods.
            for origin in var.mods.iter().copied() {
                register(s2::ArgDir::Modify, origin);
            }
        }

        let mut merged = HashMap::new();
        for (mut from, mut to) in collisions {
            // If we've already re-aliased the `from` variable, resolve it to its current id
            while let Some(&root) = merged.get(&from) {
                from = root;
            }

            // If we've already re-aliased the `to` variable, resolve it to its current id
            while let Some(&root) = merged.get(&to) {
                to = root;
            }

            if from == to {
                // If both variables have already been aliased to the same variable, there's not
                // a whole lot left for us to take care of, so we'll just ignore it completely.
                continue;
            }

            log::debug!("investigating merge (overlap: {} v {})", from, to);

            // Grab both sides of the collision, we'll re-insert them later if both aren't merged.
            let mut n1 = used_vars.remove(&from).expect("n1 alias from nonexistent variable");
            let mut n2 = used_vars.remove(&to).expect("n2 alias to nonexistent variable");

            match () {
                // If they're equivalent, merge n2 into n1.
                // It shouldn't matter which direction we do this in.
                () if n1.is_equivalent(&n2) => {
                    log::debug!("merging {} into {} (equivalent)", n2.id, n1.id);

                    merged.insert(n2.id, n1.id);
                    n1.merge(n2);
                    used_vars.insert(n1.id, n1);
                },
                // If they're not equivalent:
                () => {
                    // Find the last common variable they both replace.
                    let mut root_id = match n1.replaces.intersection(&n2.replaces).last() {
                        Some(&root_id) => root_id,
                        // TODO: Make sure it's sound to take these two branches. It *should* be.
                        // If one of the vars aliases the other, assume they're the same var.
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
                            // to use that marker to create but not initialize the var.

                            log::error!("abandoning merge: {:#?} v {:#?}", n1, n2);

                            // Re-insert the variables.
                            //
                            // If you're debugging issues related to this branch being taken you
                            // might benefit from temporarily removing these insertions so that
                            // you can easier see where something is blowing up and not just where
                            // the issue occurs. Just don't forget to re-insert them afterwards.
                            used_vars.insert(n1.id, n1);
                            used_vars.insert(n2.id, n2);

                            // Bail and hope the user sees the "abandoning merge" message.
                            // At this point we're continuing not because we want to, but because
                            // it is much easier to debug whatever made us end up here if we can
                            // look through the output instead of just a panic.
                            continue;
                        },
                    };

                    // If we've already re-aliased the variable, resolve it to its current id
                    while let Some(&root) = merged.get(&root_id) {
                        root_id = root;
                    }

                    log::debug!("merging {} and {} into {}", n1.id, n2.id, root_id);

                    // This would probably be better expressed as an if-elif-else chain. Huh.
                    match () {
                        // The two cases below should only happen if root_id has been realiased to
                        // one of the variables we're currently resolving.

                        // If n1 is the node we're looking for, merge n2 into n1.
                        () if n1.id == root_id => {
                            merged.insert(n2.id, n1.id);
                            n1.merge(n2);
                            used_vars.insert(n1.id, n1);
                        },
                        // If n2 is the node we're looking for, merge n1 into n2.
                        () if n2.id == root_id => {
                            merged.insert(n1.id, n2.id);
                            n2.merge(n1);
                            used_vars.insert(n2.id, n2);
                        },

                        // If we're merging two variables into one existing variable, go get that
                        // variable and merge both n1 and n2 into it.
                        //
                        // This happens when two blocks write to/over the same variable and it is
                        // later read from a block that both of those converge into.
                        //
                        // An example that should trigger this is:
                        // ```lua
                        // local x;
                        // if false then
                        //     x = 10
                        // else
                        //     x = 20
                        // end
                        // return x
                        // ```
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

        // Finally, collect all the variables we've "found" into a vector and return it.
        //
        // There's no real point in returning the BTreeMap, but there's also no real point in *not*
        // returning the BTreeMap, so if you're here because you want it for something or if you're
        // planning on re-building it out of the vector, just start returning it instead.
        Some(used_vars.into_values().collect::<Vec<_>>())
    }
}

impl Function {
    /// Runs `Function::optimize` until it returns zero, then returns the total score.
    pub fn optimize_until_complete(&mut self) -> Vec<usize> {
        (0..).map(|_| self.optimize()).take_while(|&s| s != 0).collect()
    }

    /// Runs all optimization passes and returns the total score.
    pub fn optimize(&mut self) -> usize {
        self.opt_funcs()
            + self.opt_blocks()
            + self.opt_merge_identical_blocks()
            + self.opt_merge_sequential_blocks()
            + self.opt_remove_empty_blocks()
            + self.opt_guess_names()
    }

    /// Runs all optimization passes for child functions.
    fn opt_funcs(&mut self) -> usize { self.children.iter_mut().map(Function::optimize).sum() }

    /// Runs all block optimization passes for each individual block.
    fn opt_blocks(&mut self) -> usize {
        let children = &self.children[..];
        self.blocks.values_mut().map(|b| b.optimize(children)).sum()
    }

    /// Merges all blocks that contain identical instructions and fork to the same block(s).
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

    /// Removes all blocks that no longer contain any instructions.
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

    /// Merges sequential blocks that only have each other as fork target / source.
    fn opt_merge_sequential_blocks(&mut self) -> usize {
        let mut score = 0;
        let mut block_to_origin = HashMap::<BlockId, Vec<BlockId>>::new();

        // Map each block to the blocks that arrive at it.
        for (&origin, block) in self.blocks.iter() {
            for forks_to in block.forks_to.iter().copied() {
                block_to_origin.entry(forks_to).or_default().push(origin);
            }
        }

        // Track any blocks we've already remapped.
        let mut remapped = HashMap::new();

        for (block_id, origins) in block_to_origin {
            // If the block only has one origin :1
            if let [mut origin_id] = *origins {
                // If we're considering merging into FunctionEntry, don't do it.
                // This is mostly to keep FunctionEntry "pure", since it's going to either not
                // exist or contain only code that we've inserted as a prologue.
                //
                // Keeping it separate makes it less confusing where that code came from,
                // and makes it more obvious that it's not technically in the original script.
                if origin_id == BlockId(s2::InstructionRef::FunctionEntry) {
                    continue;
                }

                // Remap the origin if it has been remapped.
                while let Some(&remapped_id) = remapped.get(&origin_id) {
                    origin_id = remapped_id;
                }

                // Grab a reference to the origin block.
                let origin = &self.blocks[&origin_id];

                // 1: And the origin only forks to one block.
                if origin.forks_to == [block_id] {
                    // Remove the target block and merge it into the origin block.
                    let target_block = self.blocks.remove(&block_id).unwrap();
                    let origin_block = self.blocks.get_mut(&origin_id).unwrap();

                    remapped.insert(block_id, origin_id);
                    origin_block.merge(block_id, target_block);

                    // Then give ourselves a point.
                    score += 1;
                }
            }
        }

        // Update any forks to remapped blocks.
        // This is necessary, since otherwise we'll end up with blocks that point to blocks that
        // are now gone.
        for (_, block) in self.blocks.iter_mut() {
            for fork in block.forks_to.iter_mut() {
                while let Some(&remapped_id) = remapped.get(fork) {
                    *fork = remapped_id;
                }
            }
        }

        score
    }

    // TODO: We might want to migrate this to the next stage since there we'll have "proper"
    // variables, and on top of that it'd allow us to get rid of per-block lifetime analysis.
    /// Guesses child function names by the name of the global variable they're assigned to.
    fn opt_guess_names(&mut self) -> usize {
        let mut score = 0;

        // For each block this function has:
        for (_, block) in self.blocks.iter() {
            // Run lifetime analysis, and if that fails, bail out.
            let an = match lifetime_analysis(&block.instructions) {
                Some(an) => an,
                // We might want to make this a hard error in the future; but since it'll only
                // affect guessed function names we'll just go next if we run into issues.
                None => continue,
            };

            // For each variable in the block check if it's a function constant.
            // If it is, try to track where it's being used to see if it's assigned to a global.
            for var in an.vars {
                let created = match var.created {
                    Some(created) => created,
                    None => continue,
                };

                // Check for the only two instructions that use `FnConstant` args.
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

                // Iterate through each time the variable is read after being assigned.
                for used in var.uses {
                    // This can't be an if since `if let` statements can't have guards.
                    match block.instructions[used] {
                        // If we find a use that puts a variable into a global
                        s2::OpCode::SetGlobal(src, s2::Arg(_, s2::ArgSlot::Constant(g)))
                            // and it's putting our current variable into the global.
                            //                  ^^^^^^^^^^^^^^^^^^^^ (important part)
                            if src == dst =>
                        {
                            // See if we can find both the function and the constant it's assigned
                            // to, since we'll need both.
                            match (
                                self.children.get_mut(idx as usize),
                                self.constants.get(g.0 as usize),
                            ) {
                                (Some(func), Some(Constant::String(cnst))) => match &func.name {
                                    // If we do and the name is already set to what we're guessing
                                    // there's no points for us.
                                    Some(name) if name == cnst => continue,
                                    // If we do and it's set to another name, panic.
                                    // Functions can't have multiple names and someone should go
                                    // check why we believe that they can.
                                    Some(name) if name != cnst => panic!("multiple names"),

                                    // If the function didn't already have a name, rename it and
                                    // give ourselves a point.
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

#[derive(Debug, Clone, PartialEq)]
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
    /// Lift a `stage_2::Block` to a `stage_3::Block`.
    /// This is a lossy operation, but it retains everything we're currently interested in.
    pub fn lift(p: s2::Block<'_, '_>) -> Block {
        let instructions = p.instructions.into_iter().map(|x| x.op_code.clone()).collect();
        let forks_to = p.forks_to.into_iter().map(|i| BlockId(i.addr)).collect();

        Block { instructions, forks_to }
    }

    /// Merge another block into the end of this one.
    /// This is only valid if this block (only) forks to the block that is being merged.
    pub fn merge(&mut self, id: BlockId, block: Block) {
        // Make sure we're forking to the block being merged.
        if self.forks_to != [id] {
            // TODO: We might want to have some form of error handling instead of panicking here.
            panic!("invalid merge: {:?} != [{:?}]", self.forks_to, id);
        }

        // Slap the instructions onto the end of our own.
        self.instructions.extend(block.instructions);

        // Update our forks_to to point to whatever block(s) the block we merged forked to.
        self.forks_to = block.forks_to;
    }

    /// Add an instruction to the start of this block.
    pub fn push_front(&mut self, i: impl IntoIterator<Item = impl Into<s2::OpCode>>) {
        self.instructions.splice(0..0, i.into_iter().map(Into::into));
    }

    /// Add an instruction to the end of this block.
    /// Be extremely careful to maintain control flow, or make sure to update everything relevant.
    pub fn push_back(&mut self, i: impl IntoIterator<Item = impl Into<s2::OpCode>>) {
        self.instructions.extend(i.into_iter().map(Into::into));
    }
}

impl Block {
    /// Runs all optimization passes and returns the total score.
    pub fn optimize(&mut self, protos: &[Function]) -> usize {
        self.opt_custom_opcodes(protos)
            + self.opt_eliminate_oneshot_variables()
            + self.opt_eliminate_identical_forks()
            + self.opt_eliminate_nop()
    }

    /// Rewrites instructions that are annoying to deal with into our own custom replacements:  
    ///  - `OpCode::Closure` -> `ExtOpCode::Closure`
    fn opt_custom_opcodes(&mut self, protos: &[Function]) -> usize {
        let mut score = 0;

        // We can assume we'll end up with roughly the same amount of instructions.
        let mut instructions = Vec::with_capacity(self.instructions.len());

        // Defined here so that we can manually advance it.
        let mut drain = self.instructions.drain(..);

        // Iterate through all instructions
        while let Some(instr) = drain.next() {
            match instr {
                // Rewrite OpCode::Closure and the slot allocations following it into our own equivalent
                s2::OpCode::Closure(dest, s2::Arg(dir, s2::ArgSlot::FnConstant(cidx))) => {
                    // The amount of upvalues the closure we're rewriting the call for has.
                    let upvc = protos[cidx as usize].count_upvals.into();

                    // Collect all upvalues into a vector.
                    let mut upvs = vec![];
                    for _ in 0..upvc {
                        match drain.next() {
                            // All OpCode::Closure calls are guaranteed to only have `N` Move and
                            // GetUpValue instructions following them, where `N` is the number of
                            // upvalues the closure has.
                            Some(s2::OpCode::Move(_, src)) => upvs.push(src),
                            Some(s2::OpCode::GetUpVal(_, src)) => upvs.push(src),

                            // These can only happen if we've messed up somewhere earlier.
                            None => panic!("ran out of instructions while constructing closure"),
                            _ => panic!("unexpected instruction while constructing closure"),
                        }
                    }

                    // Build and commit our replacement instruction.
                    let arg = s2::Arg(dir, s2::ArgSlot::FnConstant(cidx));
                    let op = s2::ExtOpCode::Closure(dest, arg, upvs);
                    instructions.push(s2::OpCode::Custom(op));

                    // Give ourselves a point.
                    score += 1;
                },

                // Let any instructions we're not interested in pass without interacting with them.
                _ => instructions.push(instr),
            }
        }

        // Drop the drain.
        // Required, or we'll not be able to access `instructions` below.
        drop(drain);

        // Replace the block's set of instructions with our new set.
        self.instructions = instructions;

        // And finally, return the score.
        score
    }

    /// Removes all instructions that no longer do anything.
    fn opt_eliminate_nop(&mut self) -> usize {
        let len = self.instructions.len();

        // If the block forks to more than one other block, assume that the last instruction is
        // a control flow instruction that we must preserve.
        //
        // We pop it here and push it back later so we don't have to care about it below.
        let control_flow = match self.forks_to.len() {
            1 => None,
            _ => self.instructions.pop(),
        };

        // Only retain opcodes that aren't ONLY branching.
        // It's safe to remove these as if they're in a block we can assume that they are not
        // affecting control flow anymore. That said, if this ends up causing issues, someone most
        // likely merged, rewrote, or otherwise edited a block in a way they really shouldn't have
        #[allow(clippy::match_like_matches_macro)]
        self.instructions.retain(|op_code| match op_code {
            s2::OpCode::Jmp(_) => false,
            s2::OpCode::LessThan(_, _, _) => false,
            s2::OpCode::Equals(_, _, _) => false,
            s2::OpCode::Custom(s2::ExtOpCode::Nop) => false,
            _ => true,
        });

        // Restore the backed up control flow instruction, if one was backed up.
        if let Some(control_flow) = control_flow {
            self.instructions.push(control_flow);
        }

        // Our score is the number of instructions we removed.
        len - self.instructions.len()
    }

    /// Optimize away identical forks.
    ///
    /// If we're forking to the same block twice, the two destinations probably got merged.
    /// Whatever happened, there's no point in storing the fork twice.
    fn opt_eliminate_identical_forks(&mut self) -> usize {
        // Note, if we end up using a tree instead of a vector for forks later, this entire opt
        // pass is no longer required.

        let len = self.forks_to.len();
        self.forks_to.dedup();
        len - self.forks_to.len()
    }

    /// Attempt to "inline" variables that are only written to be immediately read by another
    /// instruction that normally wouldn't be able to read it.
    fn opt_eliminate_oneshot_variables(&mut self) -> usize {
        // We might want to remove this pass once we're able to do this at a higher level instead,
        // especially if it ends up interfering when we're rebuilding source.

        let mut score = 0;

        // Run lifetime analysis, we'll need it for later.
        let lifetime_data = match lifetime_analysis(&self.instructions) {
            Some(lifetime_data) => lifetime_data,
            _ => return 0,
        };

        // For each variable we've found, if it's contained within this block, try to inline it.
        for LifetimeVariable { arg, created, deleted, mut uses } in lifetime_data.vars {
            // We're only interested in variables that were both created and discarded in this block.
            let created = match (created, deleted) {
                (Some(created), Some(_)) => created,
                _ => continue,
            };

            // Sort the uses since we want to be sure we have them in order.
            uses.sort_unstable();

            // Make sure the variable is only used once.
            if let [used] = *uses {
                log::trace!("only used once: {} ({} -> {})", arg, created, uses[0]);

                // For the instruction that created the variable, assuming it's a supported one,
                // grab which arg is written and which arg it is put into.
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

                // Get all arguments we're able to swap out.
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

                // And for the ones we can swap, swap them.
                for arg in swappable_args {
                    if *arg == solo_arg {
                        *arg = used_arg;
                    }
                }

                // Turn the creation point into a `nop`, it'll be optimized away by the nop
                // elimination pass later.
                //
                // If we were to instead just remove the instruction we'd have to either re-run
                // lifetime analysis each iteration, or track and offset all references. This was
                // deemed to be the simpler option out of the two.
                self.instructions[created] = s2::OpCode::Custom(s2::ExtOpCode::Nop);

                // Give ourselves a point for each variable we've inlined.
                score += 1;
            }
        }

        // Finally, return our score.
        score
    }
}

/// Run basic lifetime analysis on a list of instructions.
fn lifetime_analysis(instructions: &[s2::OpCode]) -> Option<LifetimeBlock> {
    // Track any variables that were defined before the block started.
    let mut imported = HashSet::new();

    // As well as any variables that were still alive when the block ended.
    let mut exported = HashSet::new();

    // Track all variables that we've already closed.
    let mut used_shut = vec![]; // Vec<(Arg, from, Vec<uses>)

    // As well as any variables that are still alive.
    let mut used_open = HashMap::<s2::ArgSlot, (Option<usize>, Vec<usize>)>::new();

    // For each instruction in the list:
    for (idx, instr) in instructions.iter().enumerate() {
        // Register any reads this instruction makes.
        for crate::Arg(_, slot) in instr.args_read().into_iter() {
            used_open.entry(slot).or_default().1.push(idx);
        }

        // Register any writes this instruction makes.
        for s2::Arg(_, slot) in instr.args_out() {
            // Register a new write.
            let old_use = used_open.insert(slot, (Some(idx), vec![]));

            // If one currently exists, close it, the current instruction overwrites the value and kills it.
            if let Some((start, uses)) = old_use {
                used_shut.push((slot, start, Some(idx), uses));
            };
        }

        // If we find an `OpCode::Return`, close any open args, since we're sure they'll not be
        // used past here.
        // This shouldn't be needed since blocks "can't" have instructions after returning, but
        // we'll leave it here anyway. Better safe than sorry.
        if let s2::OpCode::Return(_, _) = instr {
            for (arg, (from, uses)) in used_open.drain() {
                used_shut.push((arg, from, Some(idx), uses))
            }
        }
    }

    // If we exit the block without having hit a return, close any open variables and mark them
    // as not having a closing point within the block.
    used_open.drain().for_each(|(arg, (start, uses))| {
        used_shut.push((arg, start, None, uses));
    });

    // Go through all vars we'e "found", register imports and exports, and create the structs we're
    // expected to return.
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

    // Return the lifetime info we've calculated.
    Some(LifetimeBlock { imported, exported, vars })
}
