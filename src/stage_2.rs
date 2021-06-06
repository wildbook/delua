use std::collections::{hash_map, HashMap, HashSet};

use itertools::Itertools;
use shrinkwraprs::Shrinkwrap;

use crate::{stage_1, stage_3, util::IgnoreDebug};

pub trait OpInfo {
    fn text(&self, constants: &[stage_3::Constant]) -> String {
        let text = |a: &Arg| a.text(constants);
        self.text_with(text)
    }
    fn text_with(&self, arg_to_string: impl Fn(&Arg) -> String) -> String;
    fn next(&self, addr: usize) -> Vec<InstructionRef>;

    // Arguments that are strictly input
    fn args_in(&self) -> Vec<Arg>;

    // Arguments that are in-place modified
    fn args_inout(&self) -> Vec<Arg>;

    // Arguments that are strictly output
    fn args_out(&self) -> Vec<Arg>;

    // Arguments that are used (in and inout, out aren't used but created)
    fn args_used(&self) -> Vec<Arg> {
        let mut x = self.args_in();
        x.extend(self.args_inout());
        x
    }

    // Arguments that are written to (inout and out)
    fn args_written(&self) -> Vec<Arg> {
        let mut x = self.args_inout();
        x.extend(self.args_out());
        x
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ExtOpCode {
    DefineReg(ArgSlot),
    Closure(Arg, Arg, Vec<Arg>),
    Comment(String),
    Nop,
}

impl ExtOpCode {
    pub fn comment(str: impl Into<String>) -> ExtOpCode {
        ExtOpCode::Comment(str.into())
    }
}

impl From<ExtOpCode> for OpCode {
    fn from(val: ExtOpCode) -> Self {
        OpCode::Custom(val)
    }
}

impl OpInfo for ExtOpCode {
    fn text_with(&self, text: impl Fn(&Arg) -> String) -> String {
        match self {
            ExtOpCode::Nop => String::new(),
            &ExtOpCode::DefineReg(slot) => {
                format!("local {0}", text(&Arg::write(slot)))
            },
            ExtOpCode::Comment(str) => format!("--[[ {} ]]", str),
            ExtOpCode::Closure(into, cidx, args) => {
                let args = args.iter().map(|x| text(x)).join(",");
                format!("{} = CreateClosure({}, [{}])", text(into), cidx, args)
            },
        }
    }

    fn next(&self, addr: usize) -> Vec<InstructionRef> {
        let skip = |off: i32| InstructionRef::Instruction((addr as i32 + off + 1) as usize);
        vec![skip(0)]
    }

    fn args_in(&self) -> Vec<Arg> {
        match self {
            ExtOpCode::Nop => vec![],
            ExtOpCode::DefineReg(_) => vec![],
            ExtOpCode::Comment(_) => vec![],
            ExtOpCode::Closure(_, cidx, args) => {
                let mut args = args.clone();
                args.push(*cidx);
                args
            },
        }
    }

    fn args_inout(&self) -> Vec<Arg> {
        match self {
            ExtOpCode::Closure(_, _, _) => vec![],
            ExtOpCode::Comment(_) => vec![],
            ExtOpCode::Nop => vec![],
            ExtOpCode::DefineReg(_) => vec![],
        }
    }

    fn args_out(&self) -> Vec<Arg> {
        match self {
            ExtOpCode::Nop => vec![],
            ExtOpCode::Comment(_) => vec![],
            &ExtOpCode::Closure(into, _, _) => vec![into],
            &ExtOpCode::DefineReg(reg) => vec![Arg::write(reg)],
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum OpCode {
    // Arg is used when the argument is directly read / written
    // ArgSlot is used when only the argument's slot is used (for spans etc.)
    // The latter is a lot less common, and is (so far) only used for registers
    Move(Arg, Arg),
    LoadK(Arg, Arg),
    LoadBool(Arg, Arg, Arg),
    LoadNil(ArgSlot, ArgSlot),
    GetUpVal(Arg, Arg),
    GetGlobal(Arg, Arg),
    GetTable(Arg, Arg, Arg),
    SetGlobal(Arg, Arg),
    SetUpVal(Arg, Arg),
    SetTable(Arg, Arg, Arg),
    NewTable(Arg, Arg, Arg),
    This(Arg, Arg, Arg), // Self
    Add(Arg, Arg, Arg),
    Sub(Arg, Arg, Arg),
    Mul(Arg, Arg, Arg),
    Div(Arg, Arg, Arg),
    Mod(Arg, Arg, Arg),
    Pow(Arg, Arg, Arg),
    Unm(Arg, Arg),
    Not(Arg, Arg),
    Len(Arg, Arg),
    Concat(Arg, Arg, Arg),
    Jmp(Arg),
    Equals(Arg, Arg, Arg),
    LessThan(Arg, Arg, Arg),
    LessThanOrEquals(Arg, Arg, Arg),
    Test(Arg, Arg),
    TestSet(Arg, Arg, Arg),
    Call(ArgSlot, Arg, Arg),
    TailCall(ArgSlot, Arg),
    Return(ArgSlot, Arg),
    ForLoop(ArgSlot, Arg),
    ForPrep(ArgSlot, Arg),
    TForLoop(ArgSlot, Arg),
    SetList(ArgSlot, Arg, Arg),
    Close(ArgSlot),
    Closure(Arg, Arg),
    Vararg(ArgSlot, Arg),

    Custom(ExtOpCode),
}

impl OpInfo for OpCode {
    fn text_with(&self, text: impl Fn(&Arg) -> String) -> String {
        match self {
            OpCode::Move(a, b) => format!("{} = {}", text(a), text(b)),
            OpCode::LoadK(a, b) => format!("{} = {}", text(a), text(b)),
            OpCode::LoadBool(a, Arg(_, ArgSlot::Value(b)), Arg(_, ArgSlot::Value(c))) => match c {
                0 => format!("{} = {}", text(a), *b != 0),
                _ => format!("{} = {} (skip)", text(a), *b != 0),
            },
            &OpCode::LoadNil(ArgSlot::Register(a), ArgSlot::Register(b)) => {
                format!(
                    "{} = nil",
                    (a..=b).map(ArgSlot::Register).map(Arg::write).map(|x| text(&x)).join(", ")
                )
            },
            OpCode::GetUpVal(a, b) => format!("{} = {}", text(a), text(b)),
            OpCode::GetGlobal(a, b) => format!("{} = _G[{}]", text(a), text(b)),
            OpCode::GetTable(a, b, c) => format!("{} = {}[{}]", text(a), text(b), text(c)),
            OpCode::SetGlobal(a, b) => format!("_G[{}] = {}", text(b), text(a)),
            OpCode::SetUpVal(a, b) => format!("{} = {}", text(b), text(a)),
            OpCode::SetTable(a, b, c) => format!("{}[{}] = {}", text(a), text(b), text(c)),
            OpCode::NewTable(a, b, c) => {
                format!("{} = {{ }} --[[ arr: {}, hash: {} ]]", text(a), text(b), text(c))
            },
            &OpCode::This(Arg(_, ArgSlot::Register(a)), Arg(_, ArgSlot::Register(b)), c) => {
                format!(
                    "{}, {} = {b}[{c}], {b}",
                    text(&Arg(ArgDir::Write, ArgSlot::Register(a))),
                    text(&Arg(ArgDir::Write, ArgSlot::Register(a + 1))),
                    b = text(&Arg(ArgDir::Read, ArgSlot::Register(b))),
                    c = text(&c),
                )
            },
            OpCode::Add(a, b, c) => format!("{} = {} + {}", text(a), text(b), text(c)),
            OpCode::Sub(a, b, c) => format!("{} = {} - {}", text(a), text(b), text(c)),
            OpCode::Mul(a, b, c) => format!("{} = {} * {}", text(a), text(b), text(c)),
            OpCode::Div(a, b, c) => format!("{} = {} / {}", text(a), text(b), text(c)),
            OpCode::Mod(a, b, c) => format!("{} = {} % {}", text(a), text(b), text(c)),
            OpCode::Pow(a, b, c) => format!("{} = {} ^ {}", text(a), text(b), text(c)),
            OpCode::Unm(a, b) => format!("{} = -{}", text(a), text(b)),
            OpCode::Not(a, b) => format!("{} = not {}", text(a), text(b)),
            OpCode::Len(a, b) => format!("{} = #{}", text(a), text(b)),
            OpCode::Concat(a, b, c) => format!("{} = {} .. {}", text(a), text(b), text(c)),
            OpCode::Jmp(_) => "jmp".to_string(),
            OpCode::Equals(Arg(_, ArgSlot::Value(a)), b, c) => match a {
                0 => format!("{} == {}", text(b), text(c)),
                _ => format!("{} ~= {}", text(b), text(c)),
            },
            OpCode::LessThan(Arg(_, ArgSlot::Value(a)), b, c) => match a {
                0 => format!("{} < {}", text(b), text(c)),
                _ => format!("{} > {}", text(b), text(c)),
            },
            OpCode::LessThanOrEquals(Arg(_, ArgSlot::Value(a)), b, c) => match a {
                0 => format!("{} <= {}", text(b), text(c)),
                _ => format!("{} >= {}", text(b), text(c)),
            },
            OpCode::Test(a, c) => {
                format!("{} test {}", text(a), text(c))
            },
            // OpCode::TestSet(_, _, _) => {}
            &OpCode::Call(
                ArgSlot::Register(a),
                Arg(_, ArgSlot::Value(b)),
                Arg(_, ArgSlot::Value(c)),
            ) => {
                let reg_arg = match b {
                    0 => vec![Arg::read(ArgSlot::Top)],
                    _ => (a + 1..=a + b - 1)
                        .map(ArgSlot::Register)
                        .map(Arg::read)
                        .collect::<Vec<_>>(),
                };

                let reg_ret = match c {
                    0 => vec![Arg(ArgDir::Write, ArgSlot::Top)],
                    1 => vec![],
                    c => (a..=a + c - 2).map(ArgSlot::Register).map(Arg::write).collect::<Vec<_>>(),
                };

                let reg_ret = reg_ret.iter().map(&text).join(", ");
                let reg_arg = reg_arg.iter().map(&text).join(", ");

                let a = text(&Arg::read(ArgSlot::Register(a)));

                match reg_ret.len() {
                    0 => format!("{}({})", a, reg_arg),
                    _ => format!("{} = {}({})", reg_ret, a, reg_arg),
                }
            },
            &OpCode::TailCall(ArgSlot::Register(a), Arg(_, ArgSlot::Value(b))) => {
                let reg_arg = match b {
                    0 => vec![Arg::write(ArgSlot::Top)],
                    _ => (a + 1..=a + b - 1)
                        .map(ArgSlot::Register)
                        .map(Arg::read)
                        .collect::<Vec<_>>(),
                };

                let reg_arg = reg_arg.iter().map(&text).join(", ");
                format!(
                    "{} = {}({})",
                    ArgSlot::Top,
                    text(&Arg::read(ArgSlot::Register(a))),
                    reg_arg
                )
            },
            &OpCode::Return(ArgSlot::Register(a), Arg(_, ArgSlot::Value(b))) => match b {
                0 => format!("return {}", ArgSlot::Top),
                1 => "return".to_string(),
                _ => {
                    format!(
                        "return {}",
                        (a..=a + b - 2)
                            .map(ArgSlot::Register)
                            .map(Arg::read)
                            .map(|x| text(&x))
                            .join(", ")
                    )
                },
            },
            &OpCode::ForLoop(ArgSlot::Register(a), _) => {
                format!(
                    "for {} = {},{},{} do",
                    text(&Arg::write(ArgSlot::Register(a + 3))),
                    text(&Arg::modify(ArgSlot::Register(a))),
                    text(&Arg::read(ArgSlot::Register(a + 1))),
                    text(&Arg::read(ArgSlot::Register(a + 2))),
                )
            },
            &OpCode::ForPrep(ArgSlot::Register(a), _) => {
                format!(
                    "{} -= {}",
                    text(&Arg::modify(ArgSlot::Register(a))),
                    text(&Arg::read(ArgSlot::Register(a + 2)))
                )
            },
            // OpCode::TForLoop(_, _) => {}
            // OpCode::SetList(Arg::Register(a), Arg::Value(b), _) => { },
            // OpCode::Close(_) => {}
            OpCode::Closure(a, b) => format!("{} = {}", text(a), text(b)),
            &OpCode::Vararg(ArgSlot::Register(a), Arg(_, ArgSlot::Value(b))) => {
                let args = (a..=a + b - 1).map(ArgSlot::Register).collect::<Vec<_>>();
                format!("{:?} = ...", args)
            },
            OpCode::Custom(custom) => custom.text_with(text),
            _ => format!(" -- {:?}", self),
        }
    }

    fn next(&self, addr: usize) -> Vec<InstructionRef> {
        let skip = |off: i32| InstructionRef::Instruction((addr as i32 + off + 1) as usize);

        match self {
            &OpCode::LoadBool(_, _, Arg(_, ArgSlot::Value(c))) => match c {
                0 => vec![skip(0)],
                _ => vec![skip(1)],
            },
            &OpCode::Jmp(Arg(_, ArgSlot::SignedValue(a))) => vec![skip(a as _)],
            OpCode::Equals(
                Arg(_, ArgSlot::Value(0)),
                Arg(_, ArgSlot::Constant(b)),
                Arg(_, ArgSlot::Constant(c)),
            ) if b == c => {
                vec![skip(0)]
            },
            OpCode::Equals(
                Arg(_, ArgSlot::Value(_)),
                Arg(_, ArgSlot::Constant(b)),
                Arg(_, ArgSlot::Constant(c)),
            ) if b == c => {
                vec![skip(1)]
            },
            OpCode::Equals(_, _, _) => vec![skip(0), skip(1)],
            OpCode::LessThan(_, _, _) => vec![skip(0), skip(1)],
            OpCode::LessThanOrEquals(_, _, _) => vec![skip(0), skip(1)],
            OpCode::Test(_, _) => vec![skip(0), skip(1)],
            OpCode::TestSet(_, _, _) => vec![skip(0), skip(1)],
            &OpCode::ForLoop(_, Arg(_, ArgSlot::SignedValue(sbx))) => vec![skip(0), skip(sbx as _)],
            &OpCode::ForPrep(_, Arg(_, ArgSlot::SignedValue(sbx))) => vec![skip(sbx as _)],
            OpCode::TForLoop(_, _) => vec![skip(0), skip(1)],
            OpCode::Return(_, _) => vec![InstructionRef::FunctionExit],
            _ => vec![skip(0)],
        }
    }

    // All arguments that this instruction might read
    fn args_in(&self) -> Vec<Arg> {
        match self {
            &OpCode::Move(_, b) => vec![b],
            &OpCode::LoadK(_, b) => vec![b],
            &OpCode::LoadBool(_, b, c) => vec![b, c],
            &OpCode::LoadNil(_, _) => vec![],
            &OpCode::GetUpVal(_, b) => vec![b],
            &OpCode::GetGlobal(_, Arg(_, ArgSlot::Constant(b))) => {
                vec![Arg::read(ArgSlot::Global(b))]
            },
            &OpCode::GetTable(_, b, c) => vec![b, c],
            &OpCode::SetGlobal(a, _) => vec![a],
            &OpCode::SetUpVal(a, _) => vec![a],
            &OpCode::SetTable(a, b, c) => vec![a, b, c],
            &OpCode::NewTable(_, b, c) => vec![b, c],
            &OpCode::This(_, b, c) => vec![b, c],
            &OpCode::Add(_, b, c) => vec![b, c],
            &OpCode::Sub(_, b, c) => vec![b, c],
            &OpCode::Mul(_, b, c) => vec![b, c],
            &OpCode::Div(_, b, c) => vec![b, c],
            &OpCode::Mod(_, b, c) => vec![b, c],
            &OpCode::Pow(_, b, c) => vec![b, c],
            &OpCode::Unm(_, b) => vec![b],
            &OpCode::Not(_, b) => vec![b],
            &OpCode::Len(_, b) => vec![b],
            &OpCode::Concat(_, b, c) => vec![b, c],
            &OpCode::Jmp(sbx) => vec![sbx],
            &OpCode::Equals(a, b, c) => vec![a, b, c],
            &OpCode::LessThan(a, b, c) => vec![a, b, c],
            &OpCode::LessThanOrEquals(a, b, c) => vec![a, b, c],
            &OpCode::Test(a, b) => vec![a, b],
            &OpCode::TestSet(a, b, c) => vec![a, b, c],
            &OpCode::Call(ArgSlot::Register(a), Arg(_, ArgSlot::Value(b)), _) => match b {
                0 => vec![Arg::read(ArgSlot::Register(a)), Arg::read(ArgSlot::Top)],
                1 => vec![Arg::read(ArgSlot::Register(a))],
                b => (a..=a + b - 1).map(ArgSlot::Register).map(Arg::read).collect(),
            },
            &OpCode::TailCall(ArgSlot::Register(a), Arg(_, ArgSlot::Value(b))) => {
                (a..=a + b - 1).map(ArgSlot::Register).map(Arg::read).collect()
            },
            &OpCode::Return(ArgSlot::Register(a), Arg(_, ArgSlot::Value(b))) => match b < 2 {
                true => vec![],
                false => (a..=a + b - 2).map(ArgSlot::Register).map(Arg::read).collect::<Vec<_>>(),
            },
            &OpCode::ForLoop(ArgSlot::Register(a), _) => {
                (a..=a + 2).map(ArgSlot::Register).map(Arg::read).collect()
            },
            &OpCode::ForPrep(ArgSlot::Register(a), b) => {
                vec![Arg::read(ArgSlot::Register(a + 2)), b]
            },
            &OpCode::TForLoop(ArgSlot::Register(a), _) => {
                (a..=a + 3).map(ArgSlot::Register).map(Arg::read).collect()
            },
            &OpCode::SetList(ArgSlot::Register(a), Arg(_, ArgSlot::Value(b)), _) => {
                (a..=a + b).map(ArgSlot::Register).map(Arg::read).collect()
            },
            &OpCode::Vararg(_, _) => vec![],
            &OpCode::Closure(_, b) => vec![b],
            // TODO: Make sure this is correct.
            OpCode::Close(_) => vec![],
            OpCode::Custom(c) => c.args_in(),
            opc => panic!("undefined opcode: {:?}", opc),
        }
    }

    // All arguments that this instruction might write
    fn args_out(&self) -> Vec<Arg> {
        match self {
            &OpCode::Move(a, _) => vec![a],
            &OpCode::LoadK(a, _) => vec![a],
            &OpCode::LoadBool(a, _, _) => vec![a],
            &OpCode::LoadNil(ArgSlot::Register(a), ArgSlot::Register(b)) => {
                (a..=b).map(ArgSlot::Register).map(Arg::write).collect()
            },
            &OpCode::GetUpVal(a, _) => vec![a],
            &OpCode::GetGlobal(a, _) => vec![a],
            &OpCode::GetTable(a, _, _) => vec![a],
            &OpCode::SetGlobal(_, Arg(_, ArgSlot::Constant(b))) => {
                vec![Arg::write(ArgSlot::Global(b))]
            },
            &OpCode::SetUpVal(_, b) => vec![b],
            &OpCode::SetTable(_, _, _) => vec![],
            &OpCode::NewTable(a, _, _) => vec![a],
            &OpCode::This(Arg(_, ArgSlot::Register(a)), _, _) => {
                vec![Arg::write(ArgSlot::Register(a)), Arg::write(ArgSlot::Register(a + 1))]
            },
            &OpCode::Add(a, _, _) => vec![a],
            &OpCode::Sub(a, _, _) => vec![a],
            &OpCode::Mul(a, _, _) => vec![a],
            &OpCode::Div(a, _, _) => vec![a],
            &OpCode::Mod(a, _, _) => vec![a],
            &OpCode::Pow(a, _, _) => vec![a],
            &OpCode::Unm(a, _) => vec![a],
            &OpCode::Not(a, _) => vec![a],
            &OpCode::Len(a, _) => vec![a],
            &OpCode::Concat(a, _, _) => vec![a],
            &OpCode::Jmp(_) => vec![],
            &OpCode::Equals(_, _, _) => vec![],
            &OpCode::LessThan(_, _, _) => vec![],
            &OpCode::LessThanOrEquals(_, _, _) => vec![],
            &OpCode::Test(_, _) => vec![],
            &OpCode::TestSet(_, _, _) => vec![],
            &OpCode::Call(ArgSlot::Register(a), _, Arg(_, ArgSlot::Value(c))) => match c {
                0 => vec![Arg::write(ArgSlot::Top)],
                1 => vec![],
                c => (a..=a + c - 2).map(ArgSlot::Register).map(Arg::write).collect(),
            },
            &OpCode::TailCall(_, _) => vec![],
            &OpCode::Return(_, _) => vec![],
            &OpCode::ForLoop(ArgSlot::Register(a), _) => {
                vec![Arg::write(ArgSlot::Register(a + 3))]
            },
            &OpCode::ForPrep(_, _) => vec![],
            &OpCode::TForLoop(ArgSlot::Register(a), Arg(_, ArgSlot::Value(c))) => {
                (a + 3..=a + 2 + c).map(ArgSlot::Register).map(Arg::write).collect()
            },
            &OpCode::SetList(_, _, _) => vec![],
            &OpCode::Vararg(ArgSlot::Register(a), Arg(_, ArgSlot::Value(b))) => {
                (a..a + b - 1).map(ArgSlot::Register).map(Arg::write).collect()
            },
            // TODO: Make sure this is correct.
            &OpCode::Close(_) => vec![],
            &OpCode::Closure(a, _) => vec![a],
            OpCode::Custom(c) => c.args_out(),
            opc => panic!("undefined opcode: {:?}", opc),
        }
    }

    fn args_inout(&self) -> Vec<Arg> {
        match *self {
            OpCode::ForLoop(a, _) => vec![Arg::modify(a)],
            OpCode::ForPrep(a, _) => vec![Arg::modify(a)],
            _ => vec![],
        }
    }
}

impl std::fmt::Display for OpCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text(&[]))
    }
}

#[derive(Shrinkwrap, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstantId(pub u32);

impl std::fmt::Display for ConstantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConstantId({})", self.0)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ArgDir {
    Read,
    Modify,
    Write,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ArgSlot {
    Top,
    Register(u32),
    UpValue(u32),
    Value(u32),
    SignedValue(i32),
    Constant(ConstantId),
    FnConstant(u32),
    Global(ConstantId),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Shrinkwrap)]
pub struct Arg(pub ArgDir, #[shrinkwrap(main_field)] pub ArgSlot);

impl std::fmt::Display for Arg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.1.fmt(f)
    }
}

impl Arg {
    pub fn read(slot: ArgSlot) -> Arg {
        Arg(ArgDir::Read, slot)
    }

    pub fn modify(slot: ArgSlot) -> Arg {
        Arg(ArgDir::Modify, slot)
    }

    pub fn write(slot: ArgSlot) -> Arg {
        Arg(ArgDir::Write, slot)
    }

    pub fn slot(self) -> ArgSlot {
        self.1
    }

    pub fn dir(self) -> ArgDir {
        self.0
    }
}

impl ArgSlot {
    pub fn is_reg(&self) -> bool {
        matches!(self, ArgSlot::Register(_))
    }

    pub fn text(&self, consts: &[stage_3::Constant]) -> String {
        match self {
            ArgSlot::Constant(x) => match consts.get(x.0 as usize) {
                Some(c) => format!("{}", c),
                None => format!("{}", x),
            },
            ArgSlot::Global(x) => match consts.get(x.0 as usize) {
                Some(c) => format!("_G[{}]", c),
                None => format!("_G[{}]", x),
            },
            x => x.to_string(),
        }
    }
}

impl<'a> std::fmt::Display for ArgSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArgSlot::Value(x) => write!(f, "{}", x),
            ArgSlot::SignedValue(x) => write!(f, "{}", x),
            ArgSlot::Constant(x) => write!(f, "{}", x),
            ArgSlot::FnConstant(x) => write!(f, "FnConstant({})", x),
            ArgSlot::Register(x) => write!(f, "loc_{}", x),
            ArgSlot::Global(x) => write!(f, "Global[{}]", x),
            ArgSlot::UpValue(x) => write!(f, "UpValue[{}]", x),
            ArgSlot::Top => write!(f, "Arg::Top"),
        }
    }
}

impl<'a> ArgSlot {
    pub fn constant(index: u32) -> ArgSlot {
        ArgSlot::Constant(ConstantId(index))
    }

    pub fn const_or_reg(index: u32) -> ArgSlot {
        match index & (1 << 8) {
            0 => ArgSlot::Register(index & 0xFF),
            _ => ArgSlot::constant(index & 0xFF),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum InstructionRef {
    FunctionEntry,
    Instruction(usize),
    FunctionExit,
}

impl std::fmt::Display for InstructionRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstructionRef::FunctionEntry => write!(f, "FunctionEntry"),
            InstructionRef::Instruction(idx) => idx.fmt(f),
            InstructionRef::FunctionExit => write!(f, "FunctionExit"),
        }
    }
}

impl InstructionRef {
    pub fn is_instruction_idx(&self) -> bool {
        matches!(self, InstructionRef::Instruction(_))
    }
}

#[derive(Clone, Debug)]
pub struct Instruction<'a> {
    raw: &'a stage_1::LuaInstruction,
    fun: IgnoreDebug<&'a stage_1::LuaFunction>,

    pub addr: InstructionRef,
    pub op_code: OpCode,

    pub src: Vec<InstructionRef>,
    pub dst: Vec<InstructionRef>,
}

impl<'a> Instruction<'a> {
    pub fn parse(f: &'a stage_1::LuaFunction, i: &'a stage_1::LuaInstruction, addr: usize) -> Self {
        let op_code = match i.op() {
            0 => OpCode::Move(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::Register(i.b())),
            ),
            1 => OpCode::LoadK(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::constant(i.bx())),
            ),
            2 => OpCode::LoadBool(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::Value(i.b())),
                Arg::read(ArgSlot::Value(i.c())),
            ),
            3 => OpCode::LoadNil(ArgSlot::Register(i.a()), ArgSlot::Register(i.b())),
            4 => OpCode::GetUpVal(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::UpValue(i.b())),
            ),
            5 => OpCode::GetGlobal(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::constant(i.bx())),
            ),
            6 => OpCode::GetTable(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::Register(i.b())),
                Arg::read(ArgSlot::const_or_reg(i.c())),
            ),
            7 => OpCode::SetGlobal(
                Arg::read(ArgSlot::Register(i.a())),
                Arg::write(ArgSlot::constant(i.bx())),
            ),
            8 => OpCode::SetUpVal(
                Arg::read(ArgSlot::Register(i.a())),
                Arg::write(ArgSlot::UpValue(i.b())),
            ),
            9 => OpCode::SetTable(
                Arg::read(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::const_or_reg(i.b())),
                Arg::read(ArgSlot::const_or_reg(i.c())),
            ),
            10 => OpCode::NewTable(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::Value(i.b())),
                Arg::read(ArgSlot::Value(i.c())),
            ),
            11 => OpCode::This(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::Register(i.b())),
                Arg::read(ArgSlot::const_or_reg(i.c())),
            ),
            12 => OpCode::Add(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::const_or_reg(i.b())),
                Arg::read(ArgSlot::const_or_reg(i.c())),
            ),
            13 => OpCode::Sub(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::const_or_reg(i.b())),
                Arg::read(ArgSlot::const_or_reg(i.c())),
            ),
            14 => OpCode::Mul(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::const_or_reg(i.b())),
                Arg::read(ArgSlot::const_or_reg(i.c())),
            ),
            15 => OpCode::Div(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::const_or_reg(i.b())),
                Arg::read(ArgSlot::const_or_reg(i.c())),
            ),
            16 => OpCode::Mod(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::const_or_reg(i.b())),
                Arg::read(ArgSlot::const_or_reg(i.c())),
            ),
            17 => OpCode::Pow(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::const_or_reg(i.b())),
                Arg::read(ArgSlot::const_or_reg(i.c())),
            ),
            18 => OpCode::Unm(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::Register(i.b())),
            ),
            19 => OpCode::Not(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::Register(i.b())),
            ),
            20 => OpCode::Len(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::Register(i.b())),
            ),
            21 => OpCode::Concat(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::Register(i.b())),
                Arg::read(ArgSlot::Register(i.c())),
            ),
            22 => OpCode::Jmp(Arg::read(ArgSlot::SignedValue(i.bx_s()))),
            23 => OpCode::Equals(
                Arg::read(ArgSlot::Value(i.a())),
                Arg::read(ArgSlot::const_or_reg(i.b())),
                Arg::read(ArgSlot::const_or_reg(i.c())),
            ),
            24 => OpCode::LessThan(
                Arg::read(ArgSlot::Value(i.a())),
                Arg::read(ArgSlot::const_or_reg(i.b())),
                Arg::read(ArgSlot::const_or_reg(i.c())),
            ),
            25 => OpCode::LessThanOrEquals(
                Arg::read(ArgSlot::Value(i.a())),
                Arg::read(ArgSlot::const_or_reg(i.b())),
                Arg::read(ArgSlot::const_or_reg(i.c())),
            ),
            26 => {
                OpCode::Test(Arg::read(ArgSlot::Register(i.a())), Arg::read(ArgSlot::Value(i.c())))
            },
            27 => OpCode::TestSet(
                Arg::read(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::Register(i.b())),
                Arg::read(ArgSlot::Value(i.c())),
            ),
            28 => OpCode::Call(
                ArgSlot::Register(i.a()),
                Arg::read(ArgSlot::Value(i.b())),
                Arg::read(ArgSlot::Value(i.c())),
            ),
            29 => OpCode::TailCall(ArgSlot::Register(i.a()), Arg::read(ArgSlot::Value(i.b()))),
            30 => OpCode::Return(ArgSlot::Register(i.a()), Arg::read(ArgSlot::Value(i.b()))),
            31 => {
                OpCode::ForLoop(ArgSlot::Register(i.a()), Arg::read(ArgSlot::SignedValue(i.bx_s())))
            },
            32 => {
                OpCode::ForPrep(ArgSlot::Register(i.a()), Arg::read(ArgSlot::SignedValue(i.bx_s())))
            },
            33 => OpCode::TForLoop(ArgSlot::Register(i.a()), Arg::read(ArgSlot::Value(i.c()))),
            34 => OpCode::SetList(
                ArgSlot::Register(i.a()),
                Arg::read(ArgSlot::Value(i.b())),
                Arg::read(ArgSlot::Value(i.c())),
            ),
            35 => OpCode::Close(ArgSlot::Register(i.a())),
            36 => OpCode::Closure(
                Arg::write(ArgSlot::Register(i.a())),
                Arg::read(ArgSlot::FnConstant(i.bx())),
            ),
            37 => OpCode::Vararg(ArgSlot::Register(i.a()), Arg::read(ArgSlot::Value(i.b()))),
            opc => panic!("undefined opcode: {:?}", opc),
        };

        Instruction {
            raw: i,
            fun: IgnoreDebug(f),
            addr: InstructionRef::Instruction(addr),
            src: vec![],
            dst: op_code.next(addr),
            op_code,
        }
    }
}

impl<'a> std::fmt::Display for Instruction<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.addr, self.op_code)
    }
}

#[derive(Clone, Debug)]
pub struct Function<'a> {
    pub raw: &'a stage_1::LuaFunction,
    pub prototypes: Vec<Function<'a>>,
    pub instructions: Vec<Instruction<'a>>,
    pub constants: Vec<&'a stage_1::LuaConstant>,
}

impl<'a> Function<'a> {
    pub fn parse(raw: &'a stage_1::LuaFunction) -> Function {
        let mut instructions = raw
            .instructions
            .0
            .iter()
            .enumerate()
            .map(|(index, i)| Instruction::parse(&raw, i, index))
            .collect::<Vec<_>>();

        let mut jumps: HashMap<InstructionRef, Vec<InstructionRef>> = HashMap::default();

        log::trace!("|instructions");
        for i in instructions.iter() {
            log::trace!("{:#}", i);

            for &target in i.dst.iter() {
                jumps.entry(target).or_default().push(i.addr);
            }
        }

        for (addr, dst) in jumps.into_iter() {
            if let InstructionRef::Instruction(idx) = addr {
                for target in dst {
                    instructions[idx].src.push(target);
                }
            }
        }

        if let Some(instr) = instructions.get_mut(0) {
            instr.src.push(InstructionRef::FunctionEntry);
        }

        let prototypes = raw.prototypes.0.iter().map(Function::parse).collect();
        let constants = raw.constants.0.iter().collect();

        Function { raw, prototypes, instructions, constants }
    }

    pub fn instruction_at(&self, address: InstructionRef) -> Option<&Instruction<'a>> {
        self.instructions.iter().find(|x| x.addr == address)
    }

    pub fn get_src(&self, instr: &Instruction<'a>) -> Vec<&Instruction<'a>> {
        instr
            .src
            .iter()
            .copied()
            .filter(InstructionRef::is_instruction_idx)
            .map(|addr| self.instruction_at(addr).unwrap())
            .collect()
    }

    pub fn get_dst(&self, instr: &Instruction<'a>) -> Vec<&Instruction<'a>> {
        instr
            .dst
            .iter()
            .copied()
            .filter(InstructionRef::is_instruction_idx)
            .map(|addr| self.instruction_at(addr).unwrap())
            .collect()
    }

    // Whether or not a function has a "basic" control flow, meaning it only comes from one source
    // and only jumps to one destination
    pub fn is_basic_cf(&self, instr: &Instruction<'a>) -> bool {
        self.has_basic_src(instr) && self.has_basic_dst(instr)
    }

    pub fn has_basic_src(&self, instr: &Instruction<'a>) -> bool {
        self.get_src(&instr).len() == 1
    }

    pub fn has_basic_dst(&self, instr: &Instruction<'a>) -> bool {
        self.get_dst(instr).len() == 1
    }

    pub fn basic_dst(&self, instr: &Instruction<'a>) -> Option<&Instruction<'a>> {
        match *self.get_dst(&instr) {
            [dst] => Some(dst),
            _ => None,
        }
    }

    pub fn blocks(&self) -> HashMap<InstructionRef, Block<'_, 'a>> {
        let mut blocks = HashMap::default();
        let mut trails = vec![match self.instructions.get(0) {
            Some(i) => i,
            None => return HashMap::default(),
        }];

        let mut seen = HashSet::new();
        while let Some(mut instr) = trails.pop() {
            if !seen.insert(instr.addr) {
                continue;
            }

            let entry = match blocks.entry(instr.addr) {
                hash_map::Entry::Occupied(_) => continue,
                hash_map::Entry::Vacant(x) => x,
            };

            let start_addr = instr.addr;
            let mut instructions = vec![];

            while let Some(next) = self.basic_dst(instr) {
                if self.has_basic_src(next) {
                    instructions.push(instr);
                    instr = next;
                } else {
                    break;
                }
            }

            instructions.push(instr);

            let forks_to = self.get_dst(instr);

            entry.insert(Block {
                function: self,
                addr_start: start_addr,
                addr_end: instr.addr,
                instructions,
                forks_to: forks_to.clone(),
            });

            trails.extend(forks_to);
        }

        blocks
    }
}

#[derive(Clone, Debug)]
pub struct Block<'f, 'a> {
    pub addr_start: InstructionRef,
    pub addr_end: InstructionRef,
    pub function: &'f Function<'a>,
    pub instructions: Vec<&'f Instruction<'a>>,
    pub forks_to: Vec<&'f Instruction<'a>>,
}

impl<'f, 'a> std::fmt::Display for Block<'f, 'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "[{:#} -> {:#}]", self.addr_start, self.addr_end)?;

        for instr in self.instructions.iter() {
            writeln!(f, "  {}", instr)?;
        }

        for target in self.forks_to.iter() {
            writeln!(f, "  -> {:#}", target.addr)?;
        }

        Ok(())
    }
}

impl<'a> std::fmt::Display for Function<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for instr in self.instructions.iter() {
            if self.is_basic_cf(instr) {
                writeln!(f, "{}", instr)?;
            } else {
                let src = self.get_src(&instr).into_iter().map(|x| x.addr).collect::<Vec<_>>();

                let dst = self.get_dst(&instr).into_iter().map(|x| x.addr).collect::<Vec<_>>();

                writeln!(f, "{} ({:#x?} -> this -> {:#x?})", instr, src, dst)?;
            }
        }

        for child in self.prototypes.iter() {
            writeln!(f, "function UNKNOWN(...)")?;
            writeln!(f, "{}", child)?;
        }

        Ok(())
    }
}
