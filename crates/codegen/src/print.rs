//! Prints a `PtxModule` as PTX text.

use crate::ast::{
    Address, Array, BinaryOp, Compare, CvtMode, Function, Instr, Label, Operand, PtxModule,
    PtxType, RegClass, RegId, Space, UnaryOp,
};
use std::fmt::{self, Write};

impl fmt::Display for PtxModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, ".version {}", self.version)?;
        writeln!(f, ".target {}", self.target)?;
        writeln!(f, ".address_size 64")?;
        for function in &self.functions {
            writeln!(f)?;
            write_function(f, function)?;
        }
        Ok(())
    }
}

fn write_function(f: &mut fmt::Formatter<'_>, function: &Function) -> fmt::Result {
    writeln!(f, ".visible .entry {}(", function.name)?;
    let params: Vec<String> =
        function.params.iter().map(|name| format!("\t.param .u64 {name}")).collect();
    writeln!(f, "{}", params.join(",\n"))?;
    writeln!(f, ")\n{{")?;
    for (class, &count) in RegClass::ALL.iter().zip(&function.registers) {
        if count > 0 {
            writeln!(f, "\t.reg {} {}<{count}>;", class.declared(), class.prefix())?;
        }
    }
    for array in &function.arrays {
        write_array(f, array)?;
    }
    for instr in &function.body {
        write_instr(f, instr)?;
    }
    writeln!(f, "}}")
}

fn write_array(f: &mut fmt::Formatter<'_>, array: &Array) -> fmt::Result {
    let space = match array.space {
        Space::Shared => ".shared",
        Space::Local | Space::Global | Space::Param => ".local",
    };
    writeln!(f, "\t{space} .align 16 .b8 {}[{}];", array.name, array.bytes.max(1))
}

fn reg(id: RegId) -> String {
    format!("{}{}", id.class.prefix(), id.index)
}

fn operand(operand: Operand) -> String {
    match operand {
        Operand::Reg(id) => reg(id),
        Operand::Int(value) => value.to_string(),
        Operand::F32(bits) => format!("0f{bits:08X}"),
    }
}

fn label(label: Label) -> String {
    format!("$L__{}", label.0)
}

fn address(address: &Address) -> String {
    let (base, offset) = match address {
        Address::Reg { base, offset } => (reg(*base), *offset),
        Address::Array { name, offset } => (name.clone(), *offset),
    };
    if offset == 0 { format!("[{base}]") } else { format!("[{base}+{offset}]") }
}

const fn space(space: Space) -> &'static str {
    match space {
        Space::Global => ".global",
        Space::Shared => ".shared",
        Space::Local => ".local",
        Space::Param => ".param",
    }
}

/// The mnemonic of a two-operand operation on `ty`.
fn binary_name(op: BinaryOp, ty: PtxType) -> &'static str {
    let float = ty == PtxType::F32;
    match op {
        BinaryOp::Add if float => "add.rn",
        BinaryOp::Sub if float => "sub.rn",
        BinaryOp::Mul if float => "mul.rn",
        BinaryOp::Div if float => "div.rn",
        BinaryOp::Min if float => "min.NaN",
        BinaryOp::Max if float => "max.NaN",
        BinaryOp::Mul => "mul.lo",
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Div => "div",
        BinaryOp::Rem => "rem",
        BinaryOp::Min => "min",
        BinaryOp::Max => "max",
        BinaryOp::And => "and",
        BinaryOp::Or => "or",
        BinaryOp::Xor => "xor",
        BinaryOp::Shl => "shl",
    }
}

const fn unary_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "neg",
        UnaryOp::Abs => "abs",
        UnaryOp::Not => "not",
        UnaryOp::Sqrt => "sqrt.rn",
        UnaryOp::Rcp => "rcp.rn",
    }
}

const fn compare_name(cmp: Compare) -> &'static str {
    match cmp {
        Compare::Eq => "eq",
        Compare::Ne => "ne",
        Compare::Lt => "lt",
        Compare::Le => "le",
        Compare::Gt => "gt",
        Compare::Ge => "ge",
        Compare::Unordered => "nan",
        Compare::NeOrUnordered => "neu",
    }
}

const fn cvt_mode(mode: CvtMode) -> &'static str {
    match mode {
        CvtMode::Plain => "",
        CvtMode::Rn => ".rn",
        CvtMode::Rz => ".rz",
        CvtMode::RnSatfinite => ".rn.satfinite",
        CvtMode::RziSat => ".rzi.sat",
        CvtMode::Rni => ".rni",
        CvtMode::Rmi => ".rmi",
        CvtMode::Rpi => ".rpi",
    }
}

fn write_instr(f: &mut fmt::Formatter<'_>, instr: &Instr) -> fmt::Result {
    let mut line = String::new();
    let _infallible = match instr {
        Instr::Label(at) => return writeln!(f, "{}:", label(*at)),
        Instr::Mov { ty, dst, src } => {
            write!(line, "mov{} {}, {}", ty.suffix(), reg(*dst), operand(*src))
        }
        Instr::Special { dst, special } => {
            write!(line, "mov.u32 {}, {}", reg(*dst), special.name())
        }
        Instr::Binary { op, ty, dst, a, b } => {
            let (a, b) = (operand(*a), operand(*b));
            write!(line, "{}{} {}, {a}, {b}", binary_name(*op, *ty), ty.suffix(), reg(*dst))
        }
        Instr::Unary { op, ty, dst, a } => {
            write!(line, "{}{} {}, {}", unary_name(*op), ty.suffix(), reg(*dst), operand(*a))
        }
        Instr::Setp { cmp, ty, dst, a, b } => {
            let (a, b) = (operand(*a), operand(*b));
            write!(line, "setp.{}{} {}, {a}, {b}", compare_name(*cmp), ty.suffix(), reg(*dst))
        }
        Instr::Selp { ty, dst, a, b, pred } => {
            let (a, b) = (operand(*a), operand(*b));
            write!(line, "selp{} {}, {a}, {b}, {}", ty.suffix(), reg(*dst), reg(*pred))
        }
        Instr::Cvt { mode, to, from, dst, src } => {
            let sources: Vec<String> = src.iter().map(|&s| operand(s)).collect();
            let (mode, to, from) = (cvt_mode(*mode), to.suffix(), from.suffix());
            write!(line, "cvt{mode}{to}{from} {}, {}", reg(*dst), sources.join(", "))
        }
        Instr::Ld { space: at, ty, dst, addr } => {
            write!(line, "ld{}{} {}, {}", space(*at), ty.suffix(), reg(*dst), address(addr))
        }
        Instr::St { space: at, ty, addr, src } => {
            write!(line, "st{}{} {}, {}", space(*at), ty.suffix(), address(addr), operand(*src))
        }
        Instr::Bra { target, guard: None } => write!(line, "bra.uni {}", label(*target)),
        Instr::Bra { target, guard: Some((pred, value)) } => {
            let negation = if *value { "" } else { "!" };
            write!(line, "@{negation}{} bra {}", reg(*pred), label(*target))
        }
        Instr::ToGlobal { dst, src } => {
            write!(line, "cvta.to.global.u64 {}, {}", reg(*dst), reg(*src))
        }
        Instr::AddressOf { dst, name } => write!(line, "mov.u64 {}, {name}", reg(*dst)),
        Instr::Increment { dst } => write!(line, "add.s64 {0}, {0}, 1", reg(*dst)),
        Instr::BarSync => write!(line, "bar.sync 0"),
        Instr::Ret => write!(line, "ret"),
    };
    writeln!(f, "\t{line};")
}
