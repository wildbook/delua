use std::collections::{hash_map, HashMap, HashSet};

use itertools::Itertools;
use shrinkwraprs::Shrinkwrap;

use crate::{lifted::LiftedConstant, raw, util::IgnoreDebug};

pub trait OpInfo {
    fn text(&self, constants: &[LiftedConstant]) -> String;
    fn arguments_read(&self) -> Vec<Arg>;
    fn arguments_write(&self) -> Vec<Arg>;
    fn next(&self, addr: usize) -> Vec<InstructionRef>;
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ExtOpCode {
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
    fn text(&self, constants: &[LiftedConstant]) -> String {
        let _text = |a: Arg| a.text(constants);

        match self {
            ExtOpCode::Nop => String::new(),
            ExtOpCode::Comment(str) => format!("--[[ {} ]]", str),
            ExtOpCode::Closure(into, cidx, args) => {
                format!("{} = CreateClosure({}, [{}])", into, cidx, args.iter().join(","))
            },
        }
    }

    fn arguments_read(&self) -> Vec<Arg> {
        match self {
            ExtOpCode::Nop => vec![],
            ExtOpCode::Comment(_) => vec![],
            ExtOpCode::Closure(_, _, _) => vec![],
        }
    }

    fn arguments_write(&self) -> Vec<Arg> {
        match self {
            ExtOpCode::Nop => vec![],
            ExtOpCode::Comment(_) => vec![],
            ExtOpCode::Closure(into, _, _) => vec![*into],
        }
    }

    fn next(&self, addr: usize) -> Vec<InstructionRef> {
        let skip = |off: i32| InstructionRef::Instruction((addr as i32 + off + 1) as usize);
        vec![skip(0)]
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum OpCode {
    Move(Arg, Arg),
    LoadK(Arg, Arg),
    LoadBool(Arg, Arg, Arg),
    LoadNil(Arg, Arg),
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
    Call(Arg, Arg, Arg),
    TailCall(Arg, Arg),
    Return(Arg, Arg),
    ForLoop(Arg, Arg),
    ForPrep(Arg, Arg),
    TForLoop(Arg, Arg),
    SetList(Arg, Arg, Arg),
    Close(Arg),
    Closure(Arg, Arg),
    Vararg(Arg, Arg),

    Custom(ExtOpCode),
}

impl OpInfo for OpCode {
    fn text(&self, constants: &[LiftedConstant]) -> String {
        let text = |a: &Arg| a.text(constants);

        match self {
            OpCode::Move(a, b) => format!("{} = {}", text(a), text(b)),
            OpCode::LoadK(a, b) => format!("{} = {}", text(a), text(b)),
            OpCode::LoadBool(a, Arg::Value(b), Arg::Value(c)) => match c {
                0 => format!("{} = {}", text(a), *b != 0),
                _ => format!("{} = {} (skip)", text(a), *b != 0),
            },
            &OpCode::LoadNil(Arg::Register(a), Arg::Register(b)) => {
                format!("{} = nil", (a..=b).map(Arg::Register).map(|x| text(&x)).join(", "))
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
            &OpCode::This(Arg::Register(a), Arg::Register(b), c) => {
                format!(
                    "{}, {} = {b}, {b}[{c}]",
                    text(&Arg::Register(a + 1)),
                    text(&Arg::Register(a)),
                    b = text(&Arg::Register(b)),
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
            OpCode::Equals(Arg::Value(a), b, c) => match a {
                0 => format!("{} == {}", text(b), text(c)),
                _ => format!("{} ~= {}", text(b), text(c)),
            },
            OpCode::LessThan(Arg::Value(a), b, c) => match a {
                0 => format!("{} < {}", text(b), text(c)),
                _ => format!("{} > {}", text(b), text(c)),
            },
            OpCode::LessThanOrEquals(Arg::Value(a), b, c) => match a {
                0 => format!("{} <= {}", text(b), text(c)),
                _ => format!("{} >= {}", text(b), text(c)),
            },
            OpCode::Test(a, c) => {
                format!("{} test {}", text(a), text(c))
            },
            // OpCode::TestSet(_, _, _) => {}
            &OpCode::Call(Arg::Register(a), Arg::Value(b), Arg::Value(c)) => {
                let reg_ret = match c {
                    0 => vec![Arg::Top],
                    1 => vec![],
                    c => (a..=a + c - 2).map(Arg::Register).collect::<Vec<_>>(),
                };

                let reg_arg = match b {
                    0 => vec![Arg::Top],
                    _ => (a + 1..=a + b - 1).map(Arg::Register).collect::<Vec<_>>(),
                };

                let reg_ret = reg_ret.into_iter().map(|x| x.to_string()).join(", ");
                let reg_arg = reg_arg.into_iter().map(|x| x.to_string()).join(", ");

                match reg_ret.len() {
                    0 => format!("loc_{}({})", a, reg_arg),
                    _ => format!("{} = loc_{}({})", reg_ret, a, reg_arg),
                }
            },
            &OpCode::TailCall(Arg::Register(a), Arg::Value(b)) => {
                let reg_arg = match b {
                    0 => vec![Arg::Top],
                    _ => (a + 1..=a + b - 1).map(Arg::Register).collect::<Vec<_>>(),
                };

                let reg_arg = reg_arg.into_iter().map(|x| x.to_string()).join(", ");
                format!("{} = loc_{}({})", Arg::Top, a, reg_arg)
            },
            &OpCode::Return(Arg::Register(a), Arg::Value(b)) => match b {
                0 => format!("return {}", Arg::Top),
                1 => "return".to_string(),
                _ => {
                    let ret = (a..=a + b - 2).map(Arg::Register).map(|x| x.to_string()).join(", ");
                    format!("return {}", ret)
                },
            },
            &OpCode::ForLoop(Arg::Register(a), _) => {
                format!(
                    "for i = {},{},{} do",
                    text(&Arg::Register(a)),
                    text(&Arg::Register(a + 1)),
                    text(&Arg::Register(a + 2)),
                )
            },
            // OpCode::ForPrep(_, _) => {}
            // OpCode::TForLoop(_, _) => {}
            // OpCode::SetList(Arg::Register(a), Arg::Value(b), _) => { },
            // OpCode::Close(_) => {}
            OpCode::Closure(a, b) => format!("{} = {}", text(a), text(b)),
            &OpCode::Vararg(Arg::Register(a), Arg::Value(b)) => {
                let args = (a..=a + b - 1).map(Arg::Register).collect::<Vec<_>>();
                format!("{:?} = ...", args)
            },
            OpCode::Custom(custom) => custom.text(constants),
            _ => format!(" -- {:?}", self),
            // _ => format!("--"),
        }
    }

    fn next(&self, addr: usize) -> Vec<InstructionRef> {
        let skip = |off: i32| InstructionRef::Instruction((addr as i32 + off + 1) as usize);

        match self {
            &OpCode::LoadBool(_, _, Arg::Value(c)) => match c {
                0 => vec![skip(0)],
                _ => vec![skip(1)],
            },
            &OpCode::Jmp(Arg::SignedValue(a)) => vec![skip(a as _)],
            OpCode::Equals(Arg::Value(0), Arg::Constant(b), Arg::Constant(c)) if b == c => {
                vec![skip(0)]
            },
            OpCode::Equals(Arg::Value(_), Arg::Constant(b), Arg::Constant(c)) if b == c => {
                vec![skip(1)]
            },
            OpCode::Equals(_, _, _) => vec![skip(0), skip(1)],
            OpCode::LessThan(_, _, _) => vec![skip(0), skip(1)],
            OpCode::LessThanOrEquals(_, _, _) => vec![skip(0), skip(1)],
            OpCode::Test(_, _) => vec![skip(0), skip(1)],
            OpCode::TestSet(_, _, _) => vec![skip(0), skip(1)],
            &OpCode::ForLoop(_, Arg::SignedValue(sbx)) => vec![skip(0), skip(sbx as _)],
            &OpCode::ForPrep(_, Arg::SignedValue(sbx)) => vec![skip(sbx as _)],
            OpCode::TForLoop(_, _) => vec![skip(0), skip(1)],
            OpCode::Return(_, _) => vec![InstructionRef::FunctionExit],
            _ => vec![skip(0)],
        }
    }

    // All arguments that this instruction might read
    fn arguments_read(&self) -> Vec<Arg> {
        match self {
            &OpCode::Move(_, b) => vec![b],
            &OpCode::LoadK(_, b) => vec![b],
            &OpCode::LoadBool(_, b, c) => vec![b, c],
            &OpCode::LoadNil(_, b) => vec![b],
            &OpCode::GetUpVal(_, b) => vec![b],
            &OpCode::GetGlobal(_, Arg::Constant(b)) => vec![Arg::Global(b)],
            &OpCode::GetTable(_, b, c) => vec![b, c],
            &OpCode::SetGlobal(a, _) => vec![a],
            &OpCode::SetUpVal(a, _) => vec![a],
            &OpCode::SetTable(_, b, c) => vec![b, c],
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
            &OpCode::Call(Arg::Register(a), Arg::Value(b), _) => match b {
                0 => vec![Arg::Top],
                1 => vec![],
                b => (a..=a + b - 1).map(Arg::Register).collect(),
            },
            &OpCode::TailCall(Arg::Register(a), Arg::Value(b)) => {
                (a..=a + b - 1).map(Arg::Register).collect()
            },
            &OpCode::Return(Arg::Register(a), Arg::Value(b)) => match b < 2 {
                true => vec![],
                false => (a..=a + b - 2).map(Arg::Register).collect::<Vec<_>>(),
            },
            &OpCode::ForLoop(Arg::Register(a), _) => (a..=a + 2).map(Arg::Register).collect(),
            &OpCode::ForPrep(Arg::Register(a), Arg::SignedValue(b)) => {
                vec![Arg::Register(a + 2), Arg::SignedValue(b)]
            },
            &OpCode::TForLoop(Arg::Register(a), _) => (a..=a + 3).map(Arg::Register).collect(),
            &OpCode::SetList(Arg::Register(a), Arg::Value(b), _) => {
                (a..=a + b).map(Arg::Register).collect()
            },
            &OpCode::Vararg(_, _) => vec![],
            opc @ OpCode::Close(_) => panic!("OpCode::Close unsupported: {:?}", opc),
            opc @ OpCode::Closure(_, _) => panic!("OpCode::Closure unsupported: {:?}", opc),
            OpCode::Custom(c) => c.arguments_read(),
            opc => panic!("undefined opcode: {:?}", opc),
        }
    }

    // All arguments that this instruction might write
    fn arguments_write(&self) -> Vec<Arg> {
        match self {
            &OpCode::Move(a, _) => vec![a],
            &OpCode::LoadK(a, _) => vec![a],
            &OpCode::LoadBool(a, _, _) => vec![a],
            &OpCode::LoadNil(Arg::Register(a), Arg::Register(b)) => {
                (a..=b).map(Arg::Register).collect()
            },
            &OpCode::GetUpVal(a, _) => vec![a],
            &OpCode::GetGlobal(a, _) => vec![a],
            &OpCode::GetTable(a, _, _) => vec![a],
            &OpCode::SetGlobal(_, Arg::Constant(b)) => vec![Arg::Global(b)],
            &OpCode::SetUpVal(_, b) => vec![b],
            &OpCode::SetTable(a, _, _) => vec![a],
            &OpCode::NewTable(a, _, _) => vec![a],
            &OpCode::This(Arg::Register(a), _, _) => vec![Arg::Register(a), Arg::Register(a + 1)],
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
            &OpCode::Call(Arg::Register(a), _, Arg::Value(c)) => match c {
                0 => vec![Arg::Top],
                1 => vec![],
                c => (a..=a + c - 2).map(Arg::Register).collect(),
            },
            &OpCode::TailCall(_, _) => vec![],
            &OpCode::Return(_, _) => vec![],
            &OpCode::ForLoop(Arg::Register(a), _) => vec![Arg::Register(a), Arg::Register(a + 3)],
            &OpCode::ForPrep(a, _) => vec![a],
            &OpCode::TForLoop(Arg::Register(a), Arg::Value(c)) => {
                (a + 3..=a + 2 + c).map(Arg::Register).collect()
            },
            &OpCode::SetList(_, _, _) => vec![],
            &OpCode::Vararg(Arg::Register(a), Arg::Value(b)) => {
                (a..a + b - 1).map(Arg::Register).collect()
            },
            &OpCode::Closure(a, _) => vec![a],
            OpCode::Custom(c) => c.arguments_write(),
            opc @ OpCode::Close(_) => panic!("OpCode::Close unsupported: {:?}", opc),
            opc => panic!("undefined opcode: {:?}", opc),
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
pub enum Arg {
    Top,
    Register(u32),
    UpValue(u32),
    Value(u32),
    SignedValue(i32),
    Constant(ConstantId),
    FnConstant(u32),
    Global(ConstantId),
}

impl Arg {
    pub fn is_reg(&self) -> bool {
        matches!(self, Arg::Register(_))
    }

    pub fn text(&self, consts: &[LiftedConstant]) -> String {
        match self {
            Arg::Constant(x) => match consts.get(x.0 as usize) {
                Some(c) => format!("{}", c),
                None => format!("{}", x),
            },
            Arg::Global(x) => match consts.get(x.0 as usize) {
                Some(c) => format!("_G[{}]", c),
                None => format!("_G[{}]", x),
            },
            x => x.to_string(),
        }
    }
}

impl<'a> std::fmt::Display for Arg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Arg::Value(x) => write!(f, "{}", x),
            Arg::SignedValue(x) => write!(f, "{}", x),
            Arg::Constant(x) => write!(f, "{}", x),
            Arg::FnConstant(x) => write!(f, "FnConstant({})", x),
            Arg::Register(x) => write!(f, "loc_{}", x),
            Arg::Global(x) => write!(f, "Global[{}]", x),
            Arg::UpValue(x) => write!(f, "UpValue[{}]", x),
            Arg::Top => write!(f, "Arg::Top"),
        }
    }
}

impl<'a> Arg {
    pub fn constant(index: u32) -> Arg {
        Arg::Constant(ConstantId(index))
    }

    pub fn const_or_reg(index: u32) -> Arg {
        match index & (1 << 8) {
            0 => Arg::Register(index & 0xFF),
            _ => Arg::constant(index & 0xFF),
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
    raw: &'a raw::LuaInstruction,
    fun: IgnoreDebug<&'a raw::LuaFunction>,

    pub addr: InstructionRef,
    pub op_code: OpCode,

    pub src: Vec<InstructionRef>,
    pub dst: Vec<InstructionRef>,
}

impl<'a> Instruction<'a> {
    pub fn parse(f: &'a raw::LuaFunction, i: &'a raw::LuaInstruction, addr: usize) -> Self {
        let op_code = match i.op() {
            0 => OpCode::Move(Arg::Register(i.a()), Arg::Register(i.b())),
            1 => OpCode::LoadK(Arg::Register(i.a()), Arg::constant(i.bx())),
            2 => OpCode::LoadBool(Arg::Register(i.a()), Arg::Value(i.b()), Arg::Value(i.c())),
            3 => OpCode::LoadNil(Arg::Register(i.a()), Arg::Register(i.b())),
            4 => OpCode::GetUpVal(Arg::Register(i.a()), Arg::UpValue(i.b())),
            5 => OpCode::GetGlobal(Arg::Register(i.a()), Arg::constant(i.bx())),
            6 => OpCode::GetTable(
                Arg::Register(i.a()),
                Arg::Register(i.b()),
                Arg::const_or_reg(i.c()),
            ),
            7 => OpCode::SetGlobal(Arg::Register(i.a()), Arg::constant(i.bx())),
            8 => OpCode::SetUpVal(Arg::Register(i.a()), Arg::UpValue(i.b())),
            9 => OpCode::SetTable(
                Arg::Register(i.a()),
                Arg::const_or_reg(i.b()),
                Arg::const_or_reg(i.c()),
            ),
            10 => OpCode::NewTable(Arg::Register(i.a()), Arg::Value(i.b()), Arg::Value(i.c())),
            11 => {
                OpCode::This(Arg::Register(i.a()), Arg::Register(i.b()), Arg::const_or_reg(i.c()))
            },
            12 => OpCode::Add(
                Arg::Register(i.a()),
                Arg::const_or_reg(i.b()),
                Arg::const_or_reg(i.c()),
            ),
            13 => OpCode::Sub(
                Arg::Register(i.a()),
                Arg::const_or_reg(i.b()),
                Arg::const_or_reg(i.c()),
            ),
            14 => OpCode::Mul(
                Arg::Register(i.a()),
                Arg::const_or_reg(i.b()),
                Arg::const_or_reg(i.c()),
            ),
            15 => OpCode::Div(
                Arg::Register(i.a()),
                Arg::const_or_reg(i.b()),
                Arg::const_or_reg(i.c()),
            ),
            16 => OpCode::Mod(
                Arg::Register(i.a()),
                Arg::const_or_reg(i.b()),
                Arg::const_or_reg(i.c()),
            ),
            17 => OpCode::Pow(
                Arg::Register(i.a()),
                Arg::const_or_reg(i.b()),
                Arg::const_or_reg(i.c()),
            ),
            18 => OpCode::Unm(Arg::Register(i.a()), Arg::Register(i.b())),
            19 => OpCode::Not(Arg::Register(i.a()), Arg::Register(i.b())),
            20 => OpCode::Len(Arg::Register(i.a()), Arg::Register(i.b())),
            21 => OpCode::Concat(Arg::Register(i.a()), Arg::Register(i.b()), Arg::Register(i.c())),
            22 => OpCode::Jmp(Arg::SignedValue(i.bx_s())),
            23 => OpCode::Equals(
                Arg::Value(i.a()),
                Arg::const_or_reg(i.b()),
                Arg::const_or_reg(i.c()),
            ),
            24 => OpCode::LessThan(
                Arg::Value(i.a()),
                Arg::const_or_reg(i.b()),
                Arg::const_or_reg(i.c()),
            ),
            25 => OpCode::LessThanOrEquals(
                Arg::Value(i.a()),
                Arg::const_or_reg(i.b()),
                Arg::const_or_reg(i.c()),
            ),
            26 => OpCode::Test(Arg::Register(i.a()), Arg::Value(i.c())),
            27 => OpCode::TestSet(Arg::Register(i.a()), Arg::Register(i.b()), Arg::Value(i.c())),
            28 => OpCode::Call(Arg::Register(i.a()), Arg::Value(i.b()), Arg::Value(i.c())),
            29 => OpCode::TailCall(Arg::Register(i.a()), Arg::Value(i.b())),
            30 => OpCode::Return(Arg::Register(i.a()), Arg::Value(i.b())),
            31 => OpCode::ForLoop(Arg::Register(i.a()), Arg::SignedValue(i.bx_s())),
            32 => OpCode::ForPrep(Arg::Register(i.a()), Arg::SignedValue(i.bx_s())),
            33 => OpCode::TForLoop(Arg::Register(i.a()), Arg::Value(i.c())),
            34 => OpCode::SetList(Arg::Register(i.a()), Arg::Value(i.b()), Arg::Value(i.c())),
            35 => OpCode::Close(Arg::Register(i.a())),
            36 => OpCode::Closure(Arg::Register(i.a()), Arg::FnConstant(i.bx())),
            37 => OpCode::Vararg(Arg::Register(i.a()), Arg::Value(i.b())),
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
    pub raw: &'a raw::LuaFunction,
    pub prototypes: Vec<Function<'a>>,
    pub instructions: Vec<Instruction<'a>>,
    pub constants: Vec<&'a raw::LuaConstant>,
}

impl<'a> Function<'a> {
    pub fn parse(raw: &'a raw::LuaFunction) -> Function {
        let mut instructions = raw
            .instructions
            .0
            .iter()
            .enumerate()
            .map(|(index, i)| Instruction::parse(&raw, i, index))
            .collect::<Vec<_>>();

        let mut jumps: HashMap<InstructionRef, Vec<InstructionRef>> = HashMap::default();

        log::debug!("|instructions");
        for i in instructions.iter() {
            log::debug!("{:#}", i);

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
