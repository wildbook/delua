use std::collections::{hash_map, HashMap};

use crate::raw::{self, LuaConstantId};

#[derive(Copy, Clone, Debug)]
pub enum OpCode<'a> {
    Move(Arg<'a>, Arg<'a>),
    LoadK(Arg<'a>, Arg<'a>),
    LoadBool(Arg<'a>, Arg<'a>, Arg<'a>),
    LoadNil(Arg<'a>, Arg<'a>),
    GetUpVal(Arg<'a>, Arg<'a>),
    GetGlobal(Arg<'a>, Arg<'a>),
    GetTable(Arg<'a>, Arg<'a>, Arg<'a>),
    SetGlobal(Arg<'a>, Arg<'a>),
    SetUpVal(Arg<'a>, Arg<'a>),
    SetTable(Arg<'a>, Arg<'a>, Arg<'a>),
    NewTable(Arg<'a>, Arg<'a>, Arg<'a>),
    This(Arg<'a>, Arg<'a>, Arg<'a>), // Self
    Add(Arg<'a>, Arg<'a>, Arg<'a>),
    Sub(Arg<'a>, Arg<'a>, Arg<'a>),
    Mul(Arg<'a>, Arg<'a>, Arg<'a>),
    Div(Arg<'a>, Arg<'a>, Arg<'a>),
    Mod(Arg<'a>, Arg<'a>, Arg<'a>),
    Pow(Arg<'a>, Arg<'a>, Arg<'a>),
    Unm(Arg<'a>, Arg<'a>),
    Not(Arg<'a>, Arg<'a>),
    Len(Arg<'a>, Arg<'a>),
    Concat(Arg<'a>, Arg<'a>, Arg<'a>),
    Jmp(Arg<'a>),
    Equals(Arg<'a>, Arg<'a>, Arg<'a>),
    LessThan(Arg<'a>, Arg<'a>, Arg<'a>),
    LessThanOrEquals(Arg<'a>, Arg<'a>, Arg<'a>),
    Test(Arg<'a>, Arg<'a>),
    TestSet(Arg<'a>, Arg<'a>, Arg<'a>),
    Call(Arg<'a>, Arg<'a>, Arg<'a>),
    TailCall(Arg<'a>, Arg<'a>),
    Return(Arg<'a>, Arg<'a>),
    ForLoop(Arg<'a>, Arg<'a>),
    ForPrep(Arg<'a>, Arg<'a>),
    TForLoop(Arg<'a>, Arg<'a>),
    SetList(Arg<'a>, Arg<'a>, Arg<'a>),
    Close(Arg<'a>),
    Closure(Arg<'a>, Arg<'a>),
    Vararg(Arg<'a>, Arg<'a>),
}

impl<'a> OpCode<'a> {
    pub fn next(&self, addr: usize) -> Vec<usize> {
        let skip = |off: i32| (addr as i32 + off + 1) as usize;

        #[allow(clippy::collapsible_match)]
        match self {
            OpCode::LoadBool(_, _, c) => match c.as_value().unwrap() {
                0 => vec![skip(0)],
                _ => vec![skip(1)],
            },
            OpCode::Jmp(a) => vec![skip(a.as_signed_value().unwrap() as _)],
            &OpCode::Equals(Arg::Value(a), Arg::Constant(b), Arg::Constant(c)) if b == c => match a
            {
                0 => vec![skip(1)],
                _ => vec![skip(0)],
            },
            OpCode::Equals(_, _, _) => vec![skip(0), skip(1)],
            OpCode::LessThan(_, _, _) => vec![skip(0), skip(1)],
            OpCode::LessThanOrEquals(_, _, _) => vec![skip(0), skip(1)],
            OpCode::Test(_, _) => vec![skip(0), skip(1)],
            OpCode::TestSet(_, _, _) => vec![skip(0), skip(1)],
            &OpCode::ForLoop(_, Arg::SignedValue(sbx)) => vec![skip(0), skip(sbx as _)],
            &OpCode::ForPrep(_, Arg::SignedValue(sbx)) => vec![skip(sbx as _)],
            OpCode::TForLoop(_, _) => vec![skip(0), skip(1)],
            OpCode::Return(_, _) => vec![],
            _ => vec![skip(0)],
        }
    }
}

impl<'a> std::fmt::Display for OpCode<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            OpCode::Move(a, b) => write!(f, "{} = {}", a, b),
            OpCode::LoadK(a, b) => write!(f, "{} = {}", a, b),
            OpCode::LoadBool(a, b, c) => {
                write!(f, "{} = {} [skip: {}]", a, b.as_value().unwrap() != 0, c)
            }
            // OpCode::LoadNil(_, _) => {}
            // OpCode::GetUpVal(_, _) => {}
            OpCode::GetGlobal(a, b) => write!(f, "{} = _G[{}]", a, b),
            // OpCode::GetTable(_, _, _) => {}
            OpCode::SetGlobal(a, b) => write!(f, "_G[{}] = {}", b, a),
            // OpCode::SetUpVal(_, _) => {}
            OpCode::SetTable(a, b, c) => write!(f, "{}[{}] = {}", a, b, c),
            // OpCode::NewTable(_, _, _) => {}
            // OpCode::This(_, _, _) => {}
            // OpCode::Add(_, _, _) => {}
            // OpCode::Sub(_, _, _) => {}
            // OpCode::Mul(_, _, _) => {}
            // OpCode::Div(_, _, _) => {}
            // OpCode::Mod(_, _, _) => {}
            // OpCode::Pow(_, _, _) => {}
            // OpCode::Unm(_, _) => {}
            // OpCode::Not(_, _) => {}
            // OpCode::Len(_, _) => {}
            // OpCode::Concat(_, _, _) => {}
            OpCode::Jmp(_) => write!(f, "jmp"),
            OpCode::Equals(Arg::Value(a), b, c) => match a {
                0 => write!(f, "{:?} eq {:?}", b, c),
                _ => write!(f, "{:?} ne {:?}", b, c),
            },
            // OpCode::LessThan(_, _, _) => {}
            // OpCode::LessThanOrEquals(_, _, _) => {}
            OpCode::Test(a, c) => {
                write!(f, "{:?} != {:?}", a, c)
            }
            // OpCode::TestSet(_, _, _) => {}
            OpCode::Call(Arg::Register(a), Arg::Value(b), Arg::Value(c)) => {
                let reg_ret = match c {
                    0 => vec![Arg::Top],
                    1 => vec![],
                    c => (a..=a + c - 2).map(Arg::Register).collect::<Vec<_>>(),
                };

                let reg_arg = match b < 1 {
                    true => vec![],
                    false => (a + 1..=a + b - 1).map(Arg::Register).collect::<Vec<_>>(),
                };

                write!(f, "{:?} = loc_{}.call({:?})", reg_ret, a, reg_arg)
            }
            // OpCode::TailCall(_, _) => {}
            OpCode::Return(Arg::Register(a), Arg::Value(b)) => match b < 2 {
                true => write!(f, "return"),
                false => {
                    let ret = (a..=a + b - 2).map(Arg::Register).collect::<Vec<_>>();
                    write!(f, "return {:?}", ret)
                }
            },
            OpCode::ForLoop(Arg::Register(a), _) => {
                write!(
                    f,
                    "for i = {},{},{} do",
                    Arg::Register(a),
                    Arg::Register(a + 1),
                    Arg::Register(a + 2),
                )
            }
            // OpCode::ForPrep(_, _) => {}
            // OpCode::TForLoop(_, _) => {}
            // OpCode::SetList(_, _, _) => {}
            // OpCode::Close(_) => {}
            // OpCode::Closure(_, _) => {}
            OpCode::Vararg(Arg::Register(a), Arg::Value(b)) => {
                let args = (a..=a + b - 1).map(Arg::Register).collect::<Vec<_>>();
                write!(f, "{:?} = ...", args)
            }
            _ => write!(f, " -- {:?}", self),
            // _ => write!(f, "--"),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Arg<'a> {
    Value(u32),
    SignedValue(i32),
    Constant(&'a raw::LuaConstant),
    FnConstant(u32),
    Register(u32),
    Global(&'a raw::LuaConstant),
    UpValue(u32),
    Top,
}

impl<'a> Arg<'a> {
    pub fn as_value(&self) -> Option<u32> {
        match *self {
            Arg::Value(x) => Some(x),
            _ => None,
        }
    }
    pub fn as_signed_value(&self) -> Option<i32> {
        match *self {
            Arg::SignedValue(x) => Some(x),
            _ => None,
        }
    }
    pub fn as_constant(&self) -> Option<&'a raw::LuaConstant> {
        match *self {
            Arg::Constant(x) => Some(x),
            _ => None,
        }
    }
    pub fn as_fn_constant(&self) -> Option<u32> {
        match *self {
            Arg::FnConstant(x) => Some(x),
            _ => None,
        }
    }
    pub fn as_register(&self) -> Option<u32> {
        match *self {
            Arg::Register(x) => Some(x),
            _ => None,
        }
    }
    pub fn as_global(&self) -> Option<&'a raw::LuaConstant> {
        match *self {
            Arg::Global(x) => Some(x),
            _ => None,
        }
    }
    pub fn as_up_value(&self) -> Option<u32> {
        match *self {
            Arg::UpValue(x) => Some(x),
            _ => None,
        }
    }
}

impl<'a> std::fmt::Display for Arg<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Arg::Value(x) => write!(f, "{:#x}", x),
            Arg::SignedValue(x) => write!(f, "{:#x}", x),
            Arg::Constant(x) => write!(f, "{}", x),
            Arg::FnConstant(x) => write!(f, "FnConstant[{:#x}]", x),
            Arg::Register(x) => write!(f, "loc_{:x}", x),
            Arg::Global(x) => write!(f, "Global[{}]", x),
            Arg::UpValue(x) => write!(f, "UpValue[{:#x}]", x),
            Arg::Top => write!(f, "Top"),
        }
    }
}

impl<'a> Arg<'a> {
    pub fn constant(c: &[raw::LuaConstant], raw: u32) -> Arg {
        Arg::Constant(&c[raw as usize])
    }

    pub fn const_or_reg(c: &[raw::LuaConstant], raw: u32) -> Arg {
        match raw & (1 << 8) {
            0 => Arg::Register(raw & 0xFF),
            _ => Arg::constant(c, raw & 0xFF),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Instruction<'a> {
    raw: &'a raw::LuaInstruction,
    pub addr: usize,
    pub op_code: OpCode<'a>,

    pub src: Vec<usize>,
    pub dst: Vec<usize>,
}

impl<'a> Instruction<'a> {
    pub fn parse(f: &'a raw::LuaFunction, i: &'a raw::LuaInstruction, addr: usize) -> Self {
        let c = f.constants.0.as_slice();
        let op_code = match i.op() {
            0 => OpCode::Move(Arg::Register(i.a()), Arg::Register(i.b())),
            1 => OpCode::LoadK(Arg::Register(i.a()), Arg::constant(c, i.bx())),
            2 => OpCode::LoadBool(Arg::Register(i.a()), Arg::Value(i.b()), Arg::Value(i.c())),
            3 => OpCode::LoadNil(Arg::Register(i.a()), Arg::Register(i.b())),
            4 => OpCode::GetUpVal(Arg::Register(i.a()), Arg::UpValue(i.b())),
            5 => OpCode::GetGlobal(Arg::Register(i.a()), Arg::constant(c, i.bx())),
            6 => OpCode::GetTable(
                Arg::Register(i.a()),
                Arg::Register(i.b()),
                Arg::const_or_reg(c, i.c()),
            ),
            7 => OpCode::SetGlobal(Arg::Register(i.a()), Arg::constant(c, i.bx())),
            8 => OpCode::SetUpVal(Arg::Register(i.a()), Arg::UpValue(i.b())),
            9 => OpCode::SetTable(
                Arg::Register(i.a()),
                Arg::const_or_reg(c, i.b()),
                Arg::const_or_reg(c, i.c()),
            ),
            10 => OpCode::NewTable(Arg::Register(i.a()), Arg::Value(i.b()), Arg::Value(i.c())),
            11 => OpCode::This(
                Arg::Register(i.a()),
                Arg::Register(i.b()),
                Arg::const_or_reg(c, i.c()),
            ),
            12 => OpCode::Add(
                Arg::Register(i.a()),
                Arg::const_or_reg(c, i.b()),
                Arg::const_or_reg(c, i.c()),
            ),
            13 => OpCode::Sub(
                Arg::Register(i.a()),
                Arg::const_or_reg(c, i.b()),
                Arg::const_or_reg(c, i.c()),
            ),
            14 => OpCode::Mul(
                Arg::Register(i.a()),
                Arg::const_or_reg(c, i.b()),
                Arg::const_or_reg(c, i.c()),
            ),
            15 => OpCode::Div(
                Arg::Register(i.a()),
                Arg::const_or_reg(c, i.b()),
                Arg::const_or_reg(c, i.c()),
            ),
            16 => OpCode::Mod(
                Arg::Register(i.a()),
                Arg::const_or_reg(c, i.b()),
                Arg::const_or_reg(c, i.c()),
            ),
            17 => OpCode::Pow(
                Arg::Register(i.a()),
                Arg::const_or_reg(c, i.b()),
                Arg::const_or_reg(c, i.c()),
            ),
            18 => OpCode::Unm(Arg::Register(i.a()), Arg::Register(i.b())),
            19 => OpCode::Not(Arg::Register(i.a()), Arg::Register(i.b())),
            20 => OpCode::Len(Arg::Register(i.a()), Arg::Register(i.b())),
            21 => OpCode::Concat(
                Arg::Register(i.a()),
                Arg::Register(i.b()),
                Arg::Register(i.c()),
            ),
            22 => OpCode::Jmp(Arg::SignedValue(i.bx_s())),
            23 => OpCode::Equals(
                Arg::Value(i.a()),
                Arg::const_or_reg(c, i.b()),
                Arg::const_or_reg(c, i.c()),
            ),
            24 => OpCode::LessThan(
                Arg::Value(i.a()),
                Arg::const_or_reg(c, i.b()),
                Arg::const_or_reg(c, i.c()),
            ),
            25 => OpCode::LessThanOrEquals(
                Arg::Value(i.a()),
                Arg::const_or_reg(c, i.b()),
                Arg::const_or_reg(c, i.c()),
            ),
            26 => OpCode::Test(Arg::Register(i.a()), Arg::Value(i.c())),
            27 => OpCode::TestSet(
                Arg::Register(i.a()),
                Arg::Register(i.b()),
                Arg::Value(i.c()),
            ),
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
            addr,
            src: vec![],
            dst: op_code.next(addr),
            op_code,
        }
    }

    // All arguments that this instruction might read
    pub fn arguments_read(&self) -> Vec<Arg<'a>> {
        match self.op_code {
            OpCode::Move(_, b) => vec![b],
            OpCode::LoadK(_, b) => vec![b],
            OpCode::LoadBool(_, b, c) => vec![b, c],
            OpCode::LoadNil(_, b) => vec![b],
            OpCode::GetUpVal(_, b) => vec![b],
            OpCode::GetGlobal(_, Arg::Constant(b)) => vec![Arg::Global(b)],
            OpCode::GetTable(_, b, c) => vec![b, c],
            OpCode::SetGlobal(a, _) => vec![a],
            OpCode::SetUpVal(a, _) => vec![a],
            OpCode::SetTable(_, b, c) => vec![b, c],
            OpCode::NewTable(_, b, c) => vec![b, c],
            OpCode::This(_, b, c) => vec![b, c],
            OpCode::Add(_, b, c) => vec![b, c],
            OpCode::Sub(_, b, c) => vec![b, c],
            OpCode::Mul(_, b, c) => vec![b, c],
            OpCode::Div(_, b, c) => vec![b, c],
            OpCode::Mod(_, b, c) => vec![b, c],
            OpCode::Pow(_, b, c) => vec![b, c],
            OpCode::Unm(_, b) => vec![b],
            OpCode::Not(_, b) => vec![b],
            OpCode::Len(_, b) => vec![b],
            OpCode::Concat(_, b, c) => vec![b, c],
            OpCode::Jmp(sbx) => vec![sbx],
            OpCode::Equals(a, b, c) => vec![a, b, c],
            OpCode::LessThan(a, b, c) => vec![a, b, c],
            OpCode::LessThanOrEquals(a, b, c) => vec![a, b, c],
            OpCode::Test(a, b) => vec![a, b],
            OpCode::TestSet(a, b, c) => vec![a, b, c],
            OpCode::Call(Arg::Register(a), Arg::Value(b), _) => {
                (a..=a + b - 1).map(Arg::Register).collect()
            }
            OpCode::TailCall(Arg::Register(a), Arg::Value(b)) => {
                (a..=a + b - 1).map(Arg::Register).collect()
            }
            OpCode::Return(Arg::Register(a), Arg::Value(b)) => match b < 2 {
                true => vec![],
                false => (a..=a + b - 2).map(Arg::Register).collect::<Vec<_>>(),
            },
            OpCode::ForLoop(Arg::Register(a), _) => (a..=a + 2).map(Arg::Register).collect(),
            OpCode::ForPrep(Arg::Register(a), Arg::SignedValue(b)) => {
                vec![Arg::Register(a + 2), Arg::SignedValue(b)]
            }
            OpCode::TForLoop(Arg::Register(a), _) => (a..=a + 3).map(Arg::Register).collect(),
            OpCode::SetList(Arg::Register(a), Arg::Value(b), _) => {
                (a..=a + b).map(Arg::Register).collect()
            }
            OpCode::Vararg(_, _) => vec![],
            opc @ OpCode::Close(_) => panic!("OpCode::Close unsupported: {:?}", opc),
            opc @ OpCode::Closure(_, _) => panic!("OpCode::Closure unsupported: {:?}", opc),
            opc => panic!("undefined opcode: {:?}", opc),
        }
    }

    // All arguments that this instruction might write
    pub fn arguments_write(&self) -> Vec<Arg<'a>> {
        match self.op_code {
            OpCode::Move(a, _) => vec![a],
            OpCode::LoadK(a, _) => vec![a],
            OpCode::LoadBool(a, _, _) => vec![a],
            OpCode::LoadNil(Arg::Register(a), Arg::Register(b)) => {
                (a..=b).map(Arg::Register).collect()
            }
            OpCode::GetUpVal(a, _) => vec![a],
            OpCode::GetGlobal(a, _) => vec![a],
            OpCode::GetTable(a, _, _) => vec![a],
            OpCode::SetGlobal(_, Arg::Constant(b)) => vec![Arg::Global(b)],
            OpCode::SetUpVal(_, b) => vec![b],
            OpCode::SetTable(a, _, _) => vec![a],
            OpCode::NewTable(a, _, _) => vec![a],
            OpCode::This(Arg::Register(a), _, _) => vec![Arg::Register(a), Arg::Register(a + 1)],
            OpCode::Add(a, _, _) => vec![a],
            OpCode::Sub(a, _, _) => vec![a],
            OpCode::Mul(a, _, _) => vec![a],
            OpCode::Div(a, _, _) => vec![a],
            OpCode::Mod(a, _, _) => vec![a],
            OpCode::Pow(a, _, _) => vec![a],
            OpCode::Unm(a, _) => vec![a],
            OpCode::Not(a, _) => vec![a],
            OpCode::Len(a, _) => vec![a],
            OpCode::Concat(a, _, _) => vec![a],
            OpCode::Jmp(_) => vec![],
            OpCode::Equals(_, _, _) => vec![],
            OpCode::LessThan(_, _, _) => vec![],
            OpCode::LessThanOrEquals(_, _, _) => vec![],
            OpCode::Test(_, _) => vec![],
            OpCode::TestSet(_, _, _) => vec![],
            OpCode::Call(Arg::Register(a), _, Arg::Value(c)) => match c {
                0 => vec![Arg::Top],
                1 => vec![],
                c => (a..=a + c - 2).map(Arg::Register).collect(),
            },
            OpCode::TailCall(_, _) => vec![],
            OpCode::Return(_, _) => vec![],
            OpCode::ForLoop(Arg::Register(a), _) => vec![Arg::Register(a), Arg::Register(a + 3)],
            OpCode::ForPrep(a, _) => vec![a],
            OpCode::TForLoop(Arg::Register(a), Arg::Register(c)) => {
                (a + 3..=a + 2 + c).map(Arg::Register).collect()
            }
            OpCode::SetList(a, _, _) => vec![a],
            OpCode::Vararg(Arg::Register(a), Arg::Value(b)) => {
                (a..a + b - 1).map(Arg::Register).collect()
            }
            OpCode::Closure(a, _) => vec![a],
            opc @ OpCode::Close(_) => panic!("OpCode::Close unsupported: {:?}", opc),
            opc => panic!("undefined opcode: {:?}", opc),
        }
    }
}

impl<'a> std::fmt::Display for Instruction<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:#x}] {}", self.addr, self.op_code)
    }
}

#[derive(Clone, Debug)]
pub struct Function<'a> {
    raw: &'a raw::LuaFunction,
    pub prototypes: Vec<Function<'a>>,
    pub instructions: Vec<Instruction<'a>>,
    pub constants: HashMap<LuaConstantId, &'a raw::LuaConstant>,
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

        let mut jumps: HashMap<usize, Vec<usize>> = HashMap::default();

        for i in instructions.iter() {
            for &target in i.dst.iter() {
                jumps.entry(target).or_default().push(i.addr);
            }
        }

        for (addr, dst) in jumps.into_iter() {
            for target in dst {
                instructions[addr].src.push(target);
            }
        }

        let prototypes = raw.prototypes.0.iter().map(Function::parse).collect();
        let constants = raw.constants.0.iter().map(|x| (x.id(), x)).collect();

        Function {
            raw,
            prototypes,
            instructions,
            constants,
        }
    }

    pub fn instruction_at(&self, address: usize) -> Option<&Instruction<'a>> {
        self.instructions.iter().find(|x| x.addr == address)
    }

    pub fn get_src(&self, instr: &Instruction<'a>) -> Vec<&Instruction<'a>> {
        instr
            .src
            .iter()
            .map(|&addr| self.instruction_at(addr).unwrap())
            .collect()
    }

    pub fn get_dst(&self, instr: &Instruction<'a>) -> Vec<&Instruction<'a>> {
        instr
            .dst
            .iter()
            .map(|&addr| self.instruction_at(addr).unwrap())
            .collect()
    }

    pub fn get_constant(&self, id: LuaConstantId) -> Option<&'a raw::LuaConstant> {
        self.constants.get(&id).copied()
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
        self.basic_dst(instr).is_some()
    }

    pub fn basic_dst(&self, instr: &Instruction<'a>) -> Option<&Instruction<'a>> {
        let dst = self.get_dst(&instr);
        match (dst.get(0), dst.len()) {
            (Some(&i), 1) => Some(i),
            _ => None,
        }
    }

    pub fn blocks(&self) -> HashMap<usize, Block<'_, 'a>> {
        let mut blocks = HashMap::default();
        let mut trails = vec![match self.instructions.get(0) {
            Some(i) => i,
            None => return HashMap::default(),
        }];

        while let Some(mut instr) = trails.pop() {
            let entry = match blocks.entry(instr.addr) {
                hash_map::Entry::Occupied(_) => continue,
                hash_map::Entry::Vacant(x) => x,
            };

            let mut block = Block::new(instr.addr);

            while let Some(dst) = self.basic_dst(instr) {
                if !self.has_basic_src(dst) {
                    break;
                }

                block.push(instr);
                instr = dst;
            }

            block.push(instr);
            block.dst = Some(instr.addr);
            block.forks_to = self.get_dst(instr);
            entry.insert(block);
            trails.extend(self.get_dst(instr));
        }

        blocks
    }
}

#[derive(Default, Clone, Debug)]
pub struct Block<'f, 'a> {
    pub src: usize,
    pub dst: Option<usize>,
    pub instructions: Vec<&'f Instruction<'a>>,
    pub forks_to: Vec<&'f Instruction<'a>>,
}

impl<'f, 'a> Block<'f, 'a> {
    fn new(src: usize) -> Self {
        Self {
            instructions: vec![],
            src,
            dst: None,
            forks_to: vec![],
        }
    }

    fn push(&mut self, instr: &'f Instruction<'a>) {
        self.instructions.push(instr);
    }
}

impl<'f, 'a> std::fmt::Display for Block<'f, 'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.dst {
            Some(dst) => writeln!(f, "[{:#x?} -> {:#x?}]", self.src, dst)?,
            None => writeln!(f, "[{:#x?} -> void]", self.src)?,
        }

        for instr in self.instructions.iter() {
            writeln!(f, "  {}", instr)?;
        }

        for target in self.forks_to.iter() {
            writeln!(f, "  -> {:#x?}", target.addr)?;
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
                let src = self
                    .get_src(&instr)
                    .into_iter()
                    .map(|x| x.addr)
                    .collect::<Vec<_>>();

                let dst = self
                    .get_dst(&instr)
                    .into_iter()
                    .map(|x| x.addr)
                    .collect::<Vec<_>>();

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
