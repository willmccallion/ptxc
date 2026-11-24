//! Kernel entry points, parameters and shared-memory declarations, and the
//! builder that makes instructions through typed methods only: control flow
//! and memory here, arithmetic in `arithmetic.rs`.

use super::definitions::{Definition, Var};
use super::instr::{Instr, Label, PtxType, Space};
use super::operand::{Address, Special, Value};
use super::reg::{B32, B64, Class, Pred, Reg, RegClass, RegId};
use std::collections::HashMap;

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

/// Builds a function, allocating registers and labels as it goes, and
/// recording which instruction defines each register.
pub(crate) struct FunctionBuilder {
    pub(super) function: Function,
    labels: u32,
    pub(super) definitions: HashMap<RegId, Definition>,
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
        Self { function, labels: 0, definitions: HashMap::new() }
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

    pub(super) fn push(&mut self, instr: Instr) {
        let position = self.function.body.len();
        for &dst in instr.destinations() {
            let _definition = self
                .definitions
                .entry(dst)
                .and_modify(|definition| *definition = Definition::Several)
                .or_insert(Definition::Once(position));
        }
        self.function.body.push(instr);
    }

    /// A thread's position along `special`.
    pub(crate) fn special(&mut self, special: Special) -> Reg<B32> {
        let dst = self.reg();
        self.push(Instr::Special { dst: dst.id(), special });
        dst
    }

    pub(crate) fn load<C: Class>(&mut self, space: Space, ty: PtxType, addr: Address) -> Reg<C> {
        let dst = self.reg();
        self.push(Instr::Ld { space, ty, dst: dst.id(), addr });
        dst
    }

    /// `count` adjacent 32-bit words at `addr` as one vector load.
    pub(crate) fn load_vector(&mut self, space: Space, addr: Address, count: u64) -> Vec<Reg<B32>> {
        let dst: Vec<Reg<B32>> = (0..count).map(|_| self.reg()).collect();
        let ids = dst.iter().map(|reg| reg.id()).collect();
        self.push(Instr::LdVector { space, dst: ids, addr });
        dst
    }

    /// `words` to adjacent 32-bit words at `addr` as one vector store.
    pub(crate) fn store_vector(&mut self, space: Space, addr: Address, words: &[Reg<B32>]) {
        let src = words.iter().map(|reg| reg.id()).collect();
        self.push(Instr::StVector { space, src, addr });
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
    pub(crate) fn increment(&mut self, counter: Var<B64>) {
        self.push(Instr::Increment { dst: counter.reg().id() });
    }

    pub(crate) fn barrier(&mut self) {
        self.push(Instr::BarSync);
    }
}
