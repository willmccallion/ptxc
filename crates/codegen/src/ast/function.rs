//! Kernel entry points, parameters and shared-memory declarations, and the
//! builder that makes instructions through typed methods only.

use super::instr::{BinaryOp, Compare, CvtMode, Instr, Label, PtxType, Space, UnaryOp};
use super::operand::{Address, Special, Value};
use super::reg::{B16, B32, B64, Class, F32, Pred, Reg, RegClass};

/// An array a function declares in shared or local memory.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Array {
    pub(crate) space: Space,
    pub(crate) name: String,
    pub(crate) bytes: u64,
}

/// A `.entry` function: its pointer parameters, arrays, registers and body.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Function {
    pub(crate) name: String,
    pub(crate) params: Vec<String>,
    pub(crate) arrays: Vec<Array>,
    pub(crate) registers: [u32; RegClass::ALL.len()],
    pub(crate) body: Vec<Instr>,
}

/// An integer register class and the types its operations take.
pub(crate) trait Int: Class {
    const SIGNED: PtxType;
    const UNSIGNED: PtxType;
    const BITS: PtxType;
}

impl Int for B32 {
    const SIGNED: PtxType = PtxType::S32;
    const UNSIGNED: PtxType = PtxType::U32;
    const BITS: PtxType = PtxType::B32;
}

impl Int for B64 {
    const SIGNED: PtxType = PtxType::S64;
    const UNSIGNED: PtxType = PtxType::U64;
    const BITS: PtxType = PtxType::B64;
}

/// How an integer operation reads its operands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Signedness {
    Signed,
    Unsigned,
    /// As bits, for bitwise operations.
    Bits,
}

const fn int_type<C: Int>(signedness: Signedness) -> PtxType {
    match signedness {
        Signedness::Signed => C::SIGNED,
        Signedness::Unsigned => C::UNSIGNED,
        Signedness::Bits => C::BITS,
    }
}

/// Builds a function, allocating registers and labels as it goes.
pub(crate) struct FunctionBuilder {
    function: Function,
    labels: u32,
}

impl FunctionBuilder {
    pub(crate) fn new(name: &str) -> Self {
        let function = Function {
            name: name.to_owned(),
            params: Vec::new(),
            arrays: Vec::new(),
            registers: [0; RegClass::ALL.len()],
            body: Vec::new(),
        };
        Self { function, labels: 0 }
    }

    pub(crate) fn finish(mut self) -> Function {
        self.push(Instr::Ret);
        self.function
    }

    /// A new `.u64` pointer parameter, its name returned.
    pub(crate) fn param(&mut self) -> String {
        let name = format!("{}_param_{}", self.function.name, self.function.params.len());
        self.function.params.push(name.clone());
        name
    }

    /// Declares an array of `bytes` in `space`, its name returned.
    pub(crate) fn array(&mut self, space: Space, stem: &str, bytes: u64) -> String {
        let name = format!("{stem}_{}", self.function.arrays.len());
        self.function.arrays.push(Array { space, name: name.clone(), bytes });
        name
    }

    pub(crate) const fn reg<C: Class>(&mut self) -> Reg<C> {
        let count = &mut self.function.registers[C::CLASS.slot()];
        let reg = Reg::new(*count);
        *count += 1;
        reg
    }

    pub(crate) const fn label(&mut self) -> Label {
        self.labels += 1;
        Label(self.labels)
    }

    pub(crate) fn place(&mut self, label: Label) {
        self.push(Instr::Label(label));
    }

    fn push(&mut self, instr: Instr) {
        self.function.body.push(instr);
    }

    /// A thread's position along `special`.
    pub(crate) fn special(&mut self, special: Special) -> Reg<B32> {
        let dst = self.reg();
        self.push(Instr::Special { dst: dst.id(), special });
        dst
    }

    /// A register holding `value`, of the type `ty` names.
    pub(crate) fn mov<C: Class>(&mut self, ty: PtxType, value: Value<C>) -> Reg<C> {
        let dst = self.reg();
        self.push(Instr::Mov { ty, dst: dst.id(), src: value.operand() });
        dst
    }

    pub(crate) fn float(&mut self, op: BinaryOp, a: Value<F32>, b: Value<F32>) -> Reg<F32> {
        let dst = self.reg();
        self.push(Instr::Binary {
            op,
            ty: PtxType::F32,
            dst: dst.id(),
            a: a.operand(),
            b: b.operand(),
        });
        dst
    }

    pub(crate) fn float_unary(&mut self, op: UnaryOp, a: Value<F32>) -> Reg<F32> {
        let dst = self.reg();
        self.push(Instr::Unary { op, ty: PtxType::F32, dst: dst.id(), a: a.operand() });
        dst
    }

    pub(crate) fn int<C: Int>(
        &mut self,
        op: BinaryOp,
        signedness: Signedness,
        a: Value<C>,
        b: Value<C>,
    ) -> Reg<C> {
        let dst = self.reg();
        let ty = int_type::<C>(signedness);
        self.push(Instr::Binary { op, ty, dst: dst.id(), a: a.operand(), b: b.operand() });
        dst
    }

    pub(crate) fn int_unary<C: Int>(
        &mut self,
        op: UnaryOp,
        signedness: Signedness,
        a: Value<C>,
    ) -> Reg<C> {
        let dst = self.reg();
        let ty = int_type::<C>(signedness);
        self.push(Instr::Unary { op, ty, dst: dst.id(), a: a.operand() });
        dst
    }

    pub(crate) fn predicate(&mut self, op: BinaryOp, a: Reg<Pred>, b: Reg<Pred>) -> Reg<Pred> {
        let dst = self.reg();
        let (a, b) = (Value::from(a).operand(), Value::from(b).operand());
        self.push(Instr::Binary { op, ty: PtxType::Pred, dst: dst.id(), a, b });
        dst
    }

    pub(crate) fn not(&mut self, a: Reg<Pred>) -> Reg<Pred> {
        let dst = self.reg();
        self.push(Instr::Unary {
            op: UnaryOp::Not,
            ty: PtxType::Pred,
            dst: dst.id(),
            a: Value::from(a).operand(),
        });
        dst
    }

    pub(crate) fn compare_float(
        &mut self,
        cmp: Compare,
        a: Value<F32>,
        b: Value<F32>,
    ) -> Reg<Pred> {
        let dst = self.reg();
        self.push(Instr::Setp {
            cmp,
            ty: PtxType::F32,
            dst: dst.id(),
            a: a.operand(),
            b: b.operand(),
        });
        dst
    }

    pub(crate) fn compare_int<C: Int>(
        &mut self,
        cmp: Compare,
        signedness: Signedness,
        a: Value<C>,
        b: Value<C>,
    ) -> Reg<Pred> {
        let dst = self.reg();
        let ty = int_type::<C>(signedness);
        self.push(Instr::Setp { cmp, ty, dst: dst.id(), a: a.operand(), b: b.operand() });
        dst
    }

    /// `a` where `pred` holds, else `b`.
    pub(crate) fn select<C: Class>(
        &mut self,
        ty: PtxType,
        a: Value<C>,
        b: Value<C>,
        pred: Reg<Pred>,
    ) -> Reg<C> {
        let dst = self.reg();
        self.push(Instr::Selp {
            ty,
            dst: dst.id(),
            a: a.operand(),
            b: b.operand(),
            pred: pred.id(),
        });
        dst
    }

    /// `src` converted from the type `from` names to the type `to` names.
    pub(crate) fn convert<From: Class, To: Class>(
        &mut self,
        mode: CvtMode,
        (to, from): (PtxType, PtxType),
        src: Value<From>,
    ) -> Reg<To> {
        let dst = self.reg();
        self.push(Instr::Cvt { mode, to, from, dst: dst.id(), src: vec![src.operand()] });
        dst
    }

    /// Two f32 values rounded into an FP8 pair of type `to`, `high` in the
    /// high byte.
    pub(crate) fn pack_fp8(
        &mut self,
        to: PtxType,
        (high, low): (Value<F32>, Value<F32>),
    ) -> Reg<B16> {
        let dst = self.reg();
        let src = vec![high.operand(), low.operand()];
        self.push(Instr::Cvt {
            mode: CvtMode::RnSatfinite,
            to,
            from: PtxType::F32,
            dst: dst.id(),
            src,
        });
        dst
    }

    pub(crate) fn load<C: Class>(&mut self, space: Space, ty: PtxType, addr: Address) -> Reg<C> {
        let dst = self.reg();
        self.push(Instr::Ld { space, ty, dst: dst.id(), addr });
        dst
    }

    pub(crate) fn store<C: Class>(
        &mut self,
        space: Space,
        ty: PtxType,
        addr: Address,
        src: Value<C>,
    ) {
        self.push(Instr::St { space, ty, addr, src: src.operand() });
    }

    /// A branch to `target`, taken always or where `guard`'s predicate
    /// has the value given.
    pub(crate) fn branch(&mut self, target: Label, guard: Option<(Reg<Pred>, bool)>) {
        let guard = guard.map(|(pred, value)| (pred.id(), value));
        self.push(Instr::Bra { target, guard });
    }

    /// `pointer`, a kernel argument, as a global address.
    pub(crate) fn global_address(&mut self, pointer: Reg<B64>) -> Reg<B64> {
        let dst = self.reg();
        self.push(Instr::ToGlobal { dst: dst.id(), src: pointer.id() });
        dst
    }

    /// The address of array `name`.
    pub(crate) fn address_of(&mut self, name: &str) -> Reg<B64> {
        let dst = self.reg();
        self.push(Instr::AddressOf { dst: dst.id(), name: name.to_owned() });
        dst
    }

    /// Adds 1 to `counter` in place.
    pub(crate) fn increment(&mut self, counter: Reg<B64>) {
        self.push(Instr::Increment { dst: counter.id() });
    }

    pub(crate) fn barrier(&mut self) {
        self.push(Instr::BarSync);
    }

    /// The bits of `value` as another class of the same size.
    pub(crate) fn reinterpret<From: Class, To: Class>(
        &mut self,
        ty: PtxType,
        value: Reg<From>,
    ) -> Reg<To> {
        let dst = self.reg();
        self.push(Instr::Mov { ty, dst: dst.id(), src: Value::from(value).operand() });
        dst
    }
}
