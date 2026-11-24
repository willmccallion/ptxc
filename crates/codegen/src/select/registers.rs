//! Local and fragment arrays codegen keeps in registers rather than local
//! memory, as LLVM's SROA pass does before NVVM hands ptxas a kernel: those
//! of elements of at most 4 bytes whose every access is at a constant
//! index once unrolled loops are expanded. ptxas keeps such `.local`
//! arrays in registers itself, but not in a function with a `cp.async`
//! (D62). Each 4 bytes of an array is one 32-bit register; a narrower
//! element is its bytes of one.

use super::memory::Place;
use super::{CodegenError, Selector};
use crate::ast::{B32, BinaryOp, Class, CvtMode, PtxType, Reg, RegClass, Signedness, Value};
use ptxc_loop::{BufferId, Expr, ExprArena, ExprId, FragmentTile, LoweredKernel, MemScope, Stmt};
use std::collections::HashSet;

/// The bytes one register of an array holds, and of a vector access.
pub(super) const WORD_BYTES: u64 = 4;

const WORD_BITS: u32 = 32;

/// The byte of its register's word an element held in registers starts at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum WordByte {
    First,
    Second,
    Third,
    Fourth,
}

impl WordByte {
    /// The byte of its word that `offset` bytes into an array falls in.
    const fn of(offset: u64) -> Self {
        match offset % WORD_BYTES {
            0 => Self::First,
            1 => Self::Second,
            2 => Self::Third,
            _ => Self::Fourth,
        }
    }

    /// The bits of the word below the byte.
    const fn shift(self) -> u32 {
        match self {
            Self::First => 0,
            Self::Second => 8,
            Self::Third => 16,
            Self::Fourth => 24,
        }
    }
}

/// The local and fragment arrays of `kernel`, unrolled, to keep in
/// registers.
pub(super) fn register_arrays(kernel: &LoweredKernel) -> HashSet<BufferId> {
    let mut computed = Computed { exprs: kernel.exprs(), buffers: HashSet::new() };
    computed.statement(kernel.body());
    kernel
        .buffers()
        .iter_enumerated()
        .filter(|(buffer, found)| {
            matches!(found.scope, MemScope::Local | MemScope::Fragment(_))
                && found.dtype.element_size().get() <= WORD_BYTES
                && !computed.buffers.contains(buffer)
        })
        .map(|(buffer, _)| buffer)
        .collect()
}

/// The buffers some access reaches at an index that is not a constant.
struct Computed<'k> {
    exprs: &'k ExprArena,
    buffers: HashSet<BufferId>,
}

impl Computed<'_> {
    fn statement(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Seq(stmts) => stmts.iter().for_each(|stmt| self.statement(stmt)),
            Stmt::For { body, .. } | Stmt::If { then: body, .. } | Stmt::Alloc { body, .. } => {
                self.statement(body);
            }
            Stmt::LoadIndex { buffer, index, body, .. } => {
                self.access(*buffer, *index);
                self.statement(body);
            }
            Stmt::Store { buffer, index, value } => {
                self.access(*buffer, *index);
                for (read, at) in value.loads() {
                    self.access(read, at);
                }
            }
            Stmt::Barrier => {}
            Stmt::FragmentFill { fragment, .. } | Stmt::FragmentLoad { fragment, .. } => {
                self.tile(*fragment);
            }
            Stmt::Mma { a, b, accumulator, .. } => {
                for operand in [a, b, accumulator] {
                    self.tile(*operand);
                }
            }
            Stmt::FragmentEach { fragment, body, .. } => {
                self.tile(*fragment);
                self.statement(body);
            }
        }
    }

    fn access(&mut self, buffer: BufferId, index: ExprId) {
        if !matches!(self.exprs.get(index), Expr::Const(_)) {
            let _new = self.buffers.insert(buffer);
        }
    }

    fn tile(&mut self, fragment: FragmentTile) {
        self.access(fragment.buffer, fragment.tile);
    }
}

/// The register and first byte of the element `offset` bytes into an array
/// held in `words`.
pub(super) fn register_place(words: &[Reg<B32>], offset: i64) -> Result<Place, CodegenError> {
    let offset = u64::try_from(offset).map_err(|_| CodegenError::Unbound)?;
    let word = usize::try_from(offset / WORD_BYTES).map_err(|_| CodegenError::TooLarge)?;
    let word = *words.get(word).ok_or(CodegenError::Unbound)?;
    Ok(Place::Register { word, byte: WordByte::of(offset) })
}

impl Selector<'_> {
    /// The `ty` value at `place` in a register of class `C`, as a load of
    /// it would give.
    pub(super) fn read<C: Class>(
        &mut self,
        place: &Place,
        ty: PtxType,
    ) -> Result<Reg<C>, CodegenError> {
        match place {
            Place::Memory(space, addr) => Ok(self.builder.load(*space, ty, addr.clone())),
            Place::Register { word, byte } => self.read_word(*word, *byte, ty),
        }
    }

    /// Sets the `ty` value at `place` to `value`, as a store of it would.
    pub(super) fn write<C: Class>(
        &mut self,
        place: &Place,
        ty: PtxType,
        value: Reg<C>,
    ) -> Result<(), CodegenError> {
        match place {
            Place::Memory(space, addr) => {
                self.builder.store(*space, ty, addr.clone(), value.into());
                Ok(())
            }
            Place::Register { word, byte } => self.write_word(*word, *byte, ty, value),
        }
    }

    /// The `ty` field of `word` from `byte`, extended as a load extends it.
    fn read_word<C: Class>(
        &mut self,
        word: Reg<B32>,
        byte: WordByte,
        ty: PtxType,
    ) -> Result<Reg<C>, CodegenError> {
        let bits = field_bits(ty)?;
        if bits == WORD_BITS {
            return Ok(self.builder.reinterpret(PtxType::B32, word));
        }
        let shifted = self.shifted(BinaryOp::Shr, word, byte);
        let field: Reg<B32> = if ty == PtxType::S8 {
            self.builder.convert(CvtMode::Plain, (PtxType::S32, PtxType::S8), shifted.into())
        } else {
            self.builder.int(BinaryOp::And, Signedness::Bits, shifted.into(), field_mask(bits))
        };
        Ok(match C::CLASS {
            RegClass::B16 => {
                self.builder.convert(CvtMode::Plain, (PtxType::U16, PtxType::U32), field.into())
            }
            _ => self.builder.reinterpret(PtxType::B32, field),
        })
    }

    /// Sets the `ty` field of `word` from `byte` to `value`, keeping the
    /// word's other bytes.
    fn write_word<C: Class>(
        &mut self,
        word: Reg<B32>,
        byte: WordByte,
        ty: PtxType,
        value: Reg<C>,
    ) -> Result<(), CodegenError> {
        let bits = field_bits(ty)?;
        if bits == WORD_BITS {
            self.builder.assign(word, PtxType::B32, Value::from(value));
            return Ok(());
        }
        let value: Reg<B32> = match C::CLASS {
            RegClass::B16 => {
                self.builder.convert(CvtMode::Plain, (PtxType::U32, PtxType::U16), value.into())
            }
            _ => self.builder.reinterpret(PtxType::B32, value),
        };
        let bitwise = Signedness::Bits;
        let field = self.builder.int(BinaryOp::And, bitwise, value.into(), field_mask(bits));
        let placed = self.shifted(BinaryOp::Shl, field, byte);
        let others = Value::immediate(kept_bits(bits, byte));
        let kept = self.builder.int(BinaryOp::And, bitwise, word.into(), others);
        let merged = self.builder.int(BinaryOp::Or, bitwise, kept.into(), placed.into());
        self.builder.assign(word, PtxType::B32, merged.into());
        Ok(())
    }

    /// `value` shifted by `op` past the bytes before `byte`.
    fn shifted(&mut self, op: BinaryOp, value: Reg<B32>, byte: WordByte) -> Reg<B32> {
        if byte == WordByte::First {
            return value;
        }
        let by = Value::immediate(i64::from(byte.shift()));
        self.builder.int(op, Signedness::Bits, value.into(), by)
    }
}

/// The bits a value of `ty` takes in a register's word, for the types of
/// elements of at most 4 bytes.
const fn field_bits(ty: PtxType) -> Result<u32, CodegenError> {
    match ty {
        PtxType::B8 | PtxType::U8 | PtxType::S8 => Ok(8),
        PtxType::B16 => Ok(16),
        PtxType::B32 | PtxType::S32 | PtxType::F32 => Ok(WORD_BITS),
        _ => Err(CodegenError::Unsupported { op: "an element wider than a register" }),
    }
}

/// The low `bits` bits of a word.
const fn field_mask(bits: u32) -> Value<B32> {
    Value::immediate(low_bits(bits))
}

/// The bits of a word that writing a `bits`-bit field from `byte` keeps.
const fn kept_bits(bits: u32, byte: WordByte) -> i64 {
    !(low_bits(bits) << byte.shift()) & low_bits(WORD_BITS)
}

/// The integer whose low `bits` bits are set.
const fn low_bits(bits: u32) -> i64 {
    (1 << bits) - 1
}

#[cfg(test)]
mod tests {
    use super::{WordByte, kept_bits};

    #[test]
    fn each_byte_of_a_word_starts_eight_bits_past_the_one_before() {
        let shifts: Vec<u32> = (0..8).map(|offset| WordByte::of(offset).shift()).collect();

        assert_eq!(shifts, [0, 8, 16, 24, 0, 8, 16, 24]);
    }

    #[test]
    fn writing_a_field_keeps_every_bit_of_the_word_outside_it() {
        let kept = [
            kept_bits(8, WordByte::First),
            kept_bits(8, WordByte::Second),
            kept_bits(16, WordByte::Third),
        ];

        assert_eq!(kept, [0xffff_ff00, 0xffff_00ff, 0x0000_ffff]);
    }
}
