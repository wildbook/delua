use std::hash::Hash;
use std::io::Read;

type Result<T> = std::result::Result<T, std::io::Error>;

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct LuaHeader {
    pub magic: [u8; 4],
    pub version: u8,
    pub format_version: u8,
    pub endianness: Endianness,
    pub int_size: u8,
    pub size_t_size: u8,
    pub instruction_size: u8,
    pub lua_number_size: u8,
    pub integral_flag: IntegralType,
}

impl LuaHeader {
    const MAGIC: &'static [u8; 4] = b"\x1BLua";
}

impl LuaHeader {
    pub fn read<R: Read>(buffer: &mut R) -> Result<Self> {
        let mut header = [0; 12];
        buffer.read_exact(&mut header)?;

        let magic = [header[0], header[1], header[2], header[3]];
        assert_eq!(&magic, LuaHeader::MAGIC);

        Ok(Self {
            magic,
            version: header[4],
            format_version: header[5],
            endianness: match header[6] {
                0 => Endianness::Big,
                1 => Endianness::Little,
                x => panic!(" failed to read endianness: {}", x),
            },
            int_size: header[7],
            size_t_size: header[8],
            instruction_size: header[9],
            lua_number_size: header[10],
            integral_flag: match header[11] {
                0 => IntegralType::FloatingPoint,
                1 => IntegralType::IntegralNumber,
                x => panic!(" failed to read integral_flag: {}", x),
            },
        })
    }
}

#[repr(u8)]
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum IntegralType {
    FloatingPoint = 0,
    IntegralNumber = 1,
}

impl LuaReader for IntegralType {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self> {
        Ok(match u8::read(buffer, header)? {
            0 => IntegralType::FloatingPoint,
            1 => IntegralType::IntegralNumber,
            x => panic!("failed to parse IntegralType: {}", x),
        })
    }
}

#[repr(u8)]
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Endianness {
    Big,
    Little,
}

#[repr(u8)]
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum FuncVarArg {
    NoArg = 0,    // Indicates vararg is not used
    HasArg = 1,   // Indicates LUA_COMPAT_ARARG is defined
    IsVarArg = 2, // Indicates a function has contains a vararg
    NeedsArg = 4, // Indicates a function requires vararg to be passed, but is not used
}

impl LuaReader for FuncVarArg {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self> {
        Ok(match u8::read(buffer, header)? {
            0 => FuncVarArg::NoArg,
            1 => FuncVarArg::HasArg,
            2 => FuncVarArg::IsVarArg,
            4 => FuncVarArg::NeedsArg,
            x => panic!("failed to parse FuncVarArg: {}", x),
        })
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct LuaSizeT(pub u64);

impl LuaReader for LuaSizeT {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self> {
        Ok(Self(match header.size_t_size {
            4 => u32::read(buffer, header)? as _,
            8 => u64::read(buffer, header)?,
            x => panic!("unsupported size_t size: {}", x),
        }))
    }
}

#[derive(PartialEq, Clone, Hash)]
pub struct LuaString(pub Vec<u8>);

impl LuaReader for LuaString {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self> {
        let length = LuaSizeT::read(buffer, header)?.0;
        let mut data = vec![0; length as _];
        buffer.read_exact(&mut data)?;
        Ok(Self(data))
    }
}

impl std::fmt::Debug for LuaString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = std::str::from_utf8(&self.0[..self.0.len().max(1) - 1]).unwrap();
        write!(f, "{:?}", str)
    }
}

impl<'a> std::fmt::Display for LuaString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = std::str::from_utf8(&self.0[..self.0.len().max(1) - 1]).unwrap();
        write!(f, "{}", str)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct LuaNumber(pub f64);

impl<'a> std::fmt::Display for LuaNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
}

impl LuaReader for LuaNumber {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self> {
        Ok(LuaNumber(match (header.integral_flag, header.lua_number_size) {
            (IntegralType::FloatingPoint, 4) => f32::read(buffer, header)? as f64,
            (IntegralType::FloatingPoint, 8) => f64::read(buffer, header)? as f64,
            (IntegralType::IntegralNumber, 4) => u32::read(buffer, header)? as f64,
            (IntegralType::IntegralNumber, 8) => u64::read(buffer, header)? as f64,
            _ => panic!("failed to parse LuaNumber"),
        }))
    }
}

impl LuaReader for u8 {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self> {
        let mut bytes = [0; 1];
        buffer.read_exact(&mut bytes)?;
        Ok(match header.endianness {
            Endianness::Big => u8::from_be_bytes(bytes),
            Endianness::Little => u8::from_le_bytes(bytes),
        })
    }
}

impl LuaReader for u32 {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self> {
        let mut bytes = [0; 4];
        buffer.read_exact(&mut bytes)?;
        Ok(match header.endianness {
            Endianness::Big => u32::from_be_bytes(bytes),
            Endianness::Little => u32::from_le_bytes(bytes),
        })
    }
}

impl LuaReader for u64 {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self> {
        let mut bytes = [0; 8];
        buffer.read_exact(&mut bytes)?;
        Ok(match header.endianness {
            Endianness::Big => u64::from_be_bytes(bytes),
            Endianness::Little => u64::from_le_bytes(bytes),
        })
    }
}

impl LuaReader for f32 {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self> {
        let mut bytes = [0; 4];
        buffer.read_exact(&mut bytes)?;
        Ok(match header.endianness {
            Endianness::Big => f32::from_be_bytes(bytes),
            Endianness::Little => f32::from_le_bytes(bytes),
        })
    }
}

impl LuaReader for f64 {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self> {
        let mut bytes = [0; 8];
        buffer.read_exact(&mut bytes)?;
        Ok(match header.endianness {
            Endianness::Big => f64::from_be_bytes(bytes),
            Endianness::Little => f64::from_le_bytes(bytes),
        })
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct LuaList<T>(pub Vec<T>);

impl<T: LuaReader + std::fmt::Debug> LuaReader for LuaList<T> {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self> {
        let length = u32::read(buffer, header)?;
        let mut data = vec![];
        for _ in 0..length {
            data.push(T::read(buffer, header)?);
        }

        Ok(Self(data))
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct LuaLocal {
    pub name: LuaString,
    pub pc_start: u32,
    pub pc_end: u32,
}

impl LuaReader for LuaLocal {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self> {
        Ok(Self {
            name: LuaString::read(buffer, header)?,
            pc_start: u32::read(buffer, header)?,
            pc_end: u32::read(buffer, header)?,
        })
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct LuaDebugInfo {
    pub source_line_positions: LuaList<u32>,
    pub locals: LuaList<LuaLocal>,
    pub up_values: LuaList<LuaString>,
}

impl LuaReader for LuaDebugInfo {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self> {
        Ok(Self {
            source_line_positions: LuaList::read(buffer, header)?,
            locals: LuaList::read(buffer, header)?,
            up_values: LuaList::read(buffer, header)?,
        })
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum LuaConstant {
    Nil,
    Bool(bool),
    Number(LuaNumber),
    String(LuaString),
}

impl<'a> std::fmt::Display for LuaConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LuaConstant::Nil => write!(f, "Nil"),
            LuaConstant::Bool(x) => write!(f, "{}", x),
            LuaConstant::Number(x) => write!(f, "{}", x),
            LuaConstant::String(x) => write!(f, "\"{}\"", x),
        }
    }
}

impl LuaReader for LuaConstant {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self> {
        Ok(match u8::read(buffer, header)? {
            0 => Self::Nil,
            1 => Self::Bool(u8::read(buffer, header)? != 0),
            3 => Self::Number(LuaNumber::read(buffer, header)?),
            4 => Self::String(LuaString::read(buffer, header)?),
            x => panic!("failed to parse LuaConstant: {}", x),
        })
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct LuaInstruction(u32);

impl LuaInstruction {
    pub fn op(&self) -> u32 { self.0 & 0x3F }
    pub fn a(&self) -> u32 { self.0 >> 6 & 0xFF }
    pub fn b(&self) -> u32 { self.0 >> 23 & 0x1FF }
    pub fn c(&self) -> u32 { self.0 >> 14 & 0x1FF }
    pub fn bx(&self) -> u32 { self.0 >> 14 & 0x7FFFF }
    pub fn bx_s(&self) -> i32 { self.bx() as i32 - 0x1FFFF }
}

impl LuaReader for LuaInstruction {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self> {
        Ok(Self(u32::read(buffer, header)?))
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct LuaFunction {
    pub source_name: LuaString,
    pub starting_line_num: u32,
    pub ending_line_num: u32,
    pub up_val_count: u8,
    pub arg_count: u8,
    pub has_var_arg: FuncVarArg,
    pub max_stack_size: u8,

    pub instructions: LuaList<LuaInstruction>,
    pub constants: LuaList<LuaConstant>,
    pub prototypes: LuaList<LuaFunction>,
    pub debug: LuaDebugInfo,
}

impl LuaReader for LuaFunction {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self> {
        Ok(Self {
            source_name: LuaString::read(buffer, header)?,
            starting_line_num: u32::read(buffer, header)?,
            ending_line_num: u32::read(buffer, header)?,
            up_val_count: u8::read(buffer, header)?,
            arg_count: u8::read(buffer, header)?,
            has_var_arg: FuncVarArg::read(buffer, header)?,
            max_stack_size: u8::read(buffer, header)?,
            instructions: LuaList::read(buffer, header)?,
            constants: LuaList::read(buffer, header)?,
            prototypes: LuaList::read(buffer, header)?,
            debug: LuaDebugInfo::read(buffer, header)?,
        })
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct LuaFile {
    pub header: LuaHeader,
    pub top_level_func: LuaFunction,
}

impl LuaFile {
    pub fn read<R: Read>(buffer: &mut R) -> Result<Self> {
        let header = LuaHeader::read(buffer)?;
        let top_level_func = LuaFunction::read(buffer, &header)?;
        Ok(Self { header, top_level_func })
    }
}

pub trait LuaReader: Sized {
    fn read<R: Read>(buffer: &mut R, header: &LuaHeader) -> Result<Self>;
}
