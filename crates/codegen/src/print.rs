//! Prints a `PtxModule` as PTX text.

use crate::ast::{
    Address, Array, BinaryOp, Compare, CopySize, CvtMode, Function, Instr, Label, MatrixOrder,
    MmaOp, Operand, PtxModule, PtxType, RegClass, RegId, Space, UnaryOp,
};
use std::fmt;

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

/// `regs` as a vector operand.
fn registers(regs: &[RegId]) -> String {
    let names: Vec<String> = regs.iter().map(|&id| reg(id)).collect();
    format!("{{{}}}", names.join(", "))
}

fn vector_load(at: Space, dst: &[RegId], addr: &Address) -> String {
    format!("ld{}.v{}.b32 {}, {}", space(at), dst.len(), registers(dst), address(addr))
}

fn vector_store(at: Space, src: &[RegId], addr: &Address) -> String {
    format!("st{}.v{}.b32 {}, {}", space(at), src.len(), address(addr), registers(src))
}

/// `cp.async`, 16-byte copies cached only in L2, as CUTLASS's are, and
/// smaller ones, which `.cg` does not take, in L1 too.
fn copy_async(size: CopySize, [dst, src]: [&Address; 2], copied: Option<RegId>) -> String {
    let cache = if size == CopySize::Sixteen { "cg" } else { "ca" };
    let copied = copied.map_or_else(String::new, |count| format!(", {}", reg(count)));
    let (dst, src, bytes) = (address(dst), address(src), size.bytes());
    format!("cp.async.{cache}.shared.global {dst}, {src}, {bytes}{copied}")
}

fn ld_matrix(order: MatrixOrder, dst: &[RegId], addr: &Address) -> String {
    let trans = match order {
        MatrixOrder::Rows => "",
        MatrixOrder::Columns => ".trans",
    };
    let (count, dst, addr) = (dst.len(), registers(dst), address(addr));
    format!("ldmatrix.sync.aligned.m8n8.x{count}{trans}.shared.b16 {dst}, {addr}")
}

/// An `mma.sync` with its operands, `A` row-major and `B` column-major as
/// every variant ptxc emits takes them.
fn mma(op: MmaOp, operands: [&[RegId]; 4]) -> String {
    let MmaOp { shape, accumulator, multiplicands } = op;
    let (sum, product) = (accumulator.suffix(), multiplicands.suffix());
    let (m, n, k) = (shape.m.get(), shape.n.get(), shape.k.get());
    let operands: Vec<String> = operands.into_iter().map(registers).collect();
    let name = format!("mma.sync.aligned.m{m}n{n}k{k}.row.col{sum}{product}{product}{sum}");
    format!("{name} {}", operands.join(", "))
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
        BinaryOp::Shr => "shr",
        BinaryOp::MulHi => "mul.hi",
    }
}

const fn unary_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Clz => "clz",
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
    let text = instr_text(instr);
    if matches!(instr, Instr::Label(_)) { writeln!(f, "{text}") } else { writeln!(f, "\t{text};") }
}

/// `instr` as PTX, without the indent and semicolon an instruction takes.
fn instr_text(instr: &Instr) -> String {
    match instr {
        Instr::Label(at) => format!("{}:", label(*at)),
        Instr::Mov { ty, dst, src } => {
            format!("mov{} {}, {}", ty.suffix(), reg(*dst), operand(*src))
        }
        Instr::Special { dst, special } => {
            format!("mov.u32 {}, {}", reg(*dst), special.name())
        }
        Instr::Binary { op, ty, dst, a, b } => {
            let (a, b) = (operand(*a), operand(*b));
            format!("{}{} {}, {a}, {b}", binary_name(*op, *ty), ty.suffix(), reg(*dst))
        }
        Instr::Fma { dst, a, b, c } => {
            let (a, b, c) = (operand(*a), operand(*b), operand(*c));
            format!("fma.rn.f32 {}, {a}, {b}, {c}", reg(*dst))
        }
        Instr::Unary { op, ty, dst, a } => {
            format!("{}{} {}, {}", unary_name(*op), ty.suffix(), reg(*dst), operand(*a))
        }
        Instr::Setp { cmp, ty, dst, a, b } => {
            let (a, b) = (operand(*a), operand(*b));
            format!("setp.{}{} {}, {a}, {b}", compare_name(*cmp), ty.suffix(), reg(*dst))
        }
        Instr::Selp { ty, dst, a, b, pred } => {
            let (a, b) = (operand(*a), operand(*b));
            format!("selp{} {}, {a}, {b}, {}", ty.suffix(), reg(*dst), reg(*pred))
        }
        Instr::Cvt { mode, to, from, dst, src } => {
            let sources: Vec<String> = src.iter().map(|&s| operand(s)).collect();
            let (mode, to, from) = (cvt_mode(*mode), to.suffix(), from.suffix());
            format!("cvt{mode}{to}{from} {}, {}", reg(*dst), sources.join(", "))
        }
        Instr::Ld { space: at, ty, dst, addr } => {
            format!("ld{}{} {}, {}", space(*at), ty.suffix(), reg(*dst), address(addr))
        }
        Instr::St { space: at, ty, addr, src } => {
            format!("st{}{} {}, {}", space(*at), ty.suffix(), address(addr), operand(*src))
        }
        Instr::Bra { target, guard: None } => format!("bra.uni {}", label(*target)),
        Instr::Bra { target, guard: Some((pred, value)) } => {
            let negation = if *value { "" } else { "!" };
            format!("@{negation}{} bra {}", reg(*pred), label(*target))
        }
        Instr::ToGlobal { dst, src } => {
            format!("cvta.to.global.u64 {}, {}", reg(*dst), reg(*src))
        }
        Instr::AddressOf { dst, name } => format!("mov.u64 {}, {name}", reg(*dst)),
        Instr::Increment { dst } => format!("add.s64 {0}, {0}, 1", reg(*dst)),
        Instr::BarSync => "bar.sync 0".to_owned(),
        Instr::LdVector { space: at, dst, addr } => vector_load(*at, dst, addr),
        Instr::StVector { space: at, src, addr } => vector_store(*at, src, addr),
        Instr::CpAsync { size, dst, src, copied } => copy_async(*size, [dst, src], *copied),
        Instr::CpAsyncCommit => "cp.async.commit_group".to_owned(),
        Instr::CpAsyncWait { pending } => format!("cp.async.wait_group {pending}"),
        Instr::LdMatrix { order, dst, addr } => ld_matrix(*order, dst, addr),
        Instr::Mma { op, d, a, b, c } => mma(*op, [d, a, b, c]),
        Instr::Ret => "ret".to_owned(),
    }
}
