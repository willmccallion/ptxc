//! Element types, tensor types and constant data as text.

use super::attr::{TextWord, push_formatted};
use super::cursor::Cursor;
use super::element_text::{ElementText, Fnv64};
use super::error::{ParseError, ParseErrorKind};
use super::lexer::Token;
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::float::{Bf16, F8E4M3Fn, F8E5M2, F16};
use ptxc_core::{DType, Dim, HostTensor, Shape, TensorData, with_tensor_data};

/// Constants with at most this many elements print their values; larger
/// ones print a content hash, so golden files stay readable.
const INLINE_LIMIT: usize = 16;

/// Element types use their names from `ptxc-core`, shared with loop IR's
/// text.
impl TextWord for DType {
    const EXPECTED: &'static str =
        "`f32` `f16` `bf16` `f8e4m3fn` `f8e5m2` `i8` `u8` `i32` `i64` `bool` ";

    fn word(self) -> &'static str {
        self.name()
    }

    fn from_word(word: &str) -> Option<Self> {
        Self::from_name(word)
    }
}

/// Writes `f32[2, 3]`. Every layout is row-major until M4 adds others.
pub(super) fn write_type(out: &mut String, ty: &TensorType) {
    out.push_str(ty.dtype().word());
    write_dims(out, ty.shape().dims());
}

pub(super) fn write_dims(out: &mut String, dims: &[Dim]) {
    out.push('[');
    for (index, dim) in dims.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&dim.get().to_string());
    }
    out.push(']');
}

pub(super) fn parse_type(cursor: &mut Cursor) -> Result<TensorType, ParseError> {
    let position = cursor.position();
    let word = cursor.word("an element type such as `f32`")?;
    let dtype = DType::from_word(&word).ok_or_else(|| {
        ParseError::new(
            position,
            ParseErrorKind::UnknownWord { expected: DType::EXPECTED, found: word },
        )
    })?;
    let shape = parse_shape(cursor)?;
    Ok(TensorType::new(dtype, shape, Layout::RowMajor))
}

/// Reads `[2, 3]`.
pub(super) fn parse_shape(cursor: &mut Cursor) -> Result<Shape, ParseError> {
    cursor.expect_punct('[', "`[`")?;
    let mut dims = Vec::new();
    if !cursor.eat_punct(']') {
        loop {
            let (text, position) = cursor.number("a dimension")?;
            let extent = text.parse::<u64>().map_err(|_| {
                ParseError::new(position, ParseErrorKind::BadNumber { text, wanted: "dimension" })
            })?;
            dims.push(Dim::new(extent));
            if !cursor.eat_punct(',') {
                break;
            }
        }
        cursor.expect_punct(']', "`,` or `]`")?;
    }
    Shape::new(dims).map_err(|_| cursor.unexpected("a shape with at most 2^64 elements"))
}

/// Writes a constant's values, or `<elided 0x...>` with its content hash if
/// it has more than `INLINE_LIMIT` elements.
pub(super) fn write_data(out: &mut String, tensor: &HostTensor) {
    let length = with_tensor_data!(tensor.data(), values => values.len());
    if length > INLINE_LIMIT {
        push_formatted(out, format_args!("<elided 0x{:016x}>", content_hash(tensor)));
        return;
    }
    out.push('[');
    with_tensor_data!(tensor.data(), values => write_elements(out, values));
    out.push(']');
}

fn write_elements<T: ElementText>(out: &mut String, values: &[T]) {
    for (index, &value) in values.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        value.write(out);
    }
}

/// Reads a constant of type `ty`: inline values, or an elided constant found
/// among `supplied` by type and content hash.
pub(super) fn parse_data(
    cursor: &mut Cursor,
    ty: &TensorType,
    supplied: &[HostTensor],
) -> Result<HostTensor, ParseError> {
    let position = cursor.position();
    if cursor.eat_punct('<') {
        if !cursor.eat_word("elided") {
            return Err(cursor.unexpected("`elided`"));
        }
        let (hash, _) = cursor.number("a content hash")?;
        cursor.expect_punct('>', "`>`")?;
        return find_supplied(ty, &hash, supplied)
            .ok_or_else(|| ParseError::new(position, ParseErrorKind::ElidedConstant(hash)));
    }
    let texts = parse_element_texts(cursor)?;
    let data = parse_elements(ty.dtype(), &texts)
        .ok_or_else(|| cursor.unexpected("values of the constant's type"))?;
    HostTensor::new(ty.shape().clone(), data).map_err(|_| {
        let expected = usize::try_from(ty.shape().element_count().get()).unwrap_or(usize::MAX);
        ParseError::new(position, ParseErrorKind::DataLength { expected, actual: texts.len() })
    })
}

fn parse_element_texts(cursor: &mut Cursor) -> Result<Vec<String>, ParseError> {
    cursor.expect_punct('[', "`[` or `<elided ...>`")?;
    let mut texts = Vec::new();
    if !cursor.eat_punct(']') {
        loop {
            match cursor.scalar("a value")? {
                Token::Word(text) | Token::Number(text) => texts.push(text),
                Token::Value(_) | Token::QuotedValue(_) | Token::Punct(_) => {
                    return Err(cursor.unexpected("a value"));
                }
            }
            if !cursor.eat_punct(',') {
                break;
            }
        }
        cursor.expect_punct(']', "`,` or `]`")?;
    }
    Ok(texts)
}

fn parse_elements(dtype: DType, texts: &[String]) -> Option<TensorData> {
    fn each<T: ElementText>(texts: &[String]) -> Option<Vec<T>> {
        texts.iter().map(|text| T::parse(text)).collect()
    }
    Some(match dtype {
        DType::F32 => each::<f32>(texts)?.into(),
        DType::F16 => each::<F16>(texts)?.into(),
        DType::Bf16 => each::<Bf16>(texts)?.into(),
        DType::F8E4M3Fn => each::<F8E4M3Fn>(texts)?.into(),
        DType::F8E5M2 => each::<F8E5M2>(texts)?.into(),
        DType::I8 => each::<i8>(texts)?.into(),
        DType::U8 => each::<u8>(texts)?.into(),
        DType::I32 => each::<i32>(texts)?.into(),
        DType::I64 => each::<i64>(texts)?.into(),
        DType::Bool => each::<bool>(texts)?.into(),
    })
}

fn find_supplied(ty: &TensorType, hash: &str, supplied: &[HostTensor]) -> Option<HostTensor> {
    let wanted = u64::from_str_radix(hash.strip_prefix("0x")?, 16).ok()?;
    supplied
        .iter()
        .find(|tensor| {
            tensor.dtype() == ty.dtype()
                && tensor.shape() == ty.shape()
                && content_hash(tensor) == wanted
        })
        .cloned()
}

/// FNV-1a over the element type, the dims and the elements' bytes. It only
/// tells constants apart in text, so it need not resist collisions on
/// purpose.
fn content_hash(tensor: &HostTensor) -> u64 {
    let mut hash = Fnv64::new();
    hash.write(tensor.dtype().word().as_bytes());
    for dim in tensor.shape().dims() {
        hash.write(&dim.get().to_le_bytes());
    }
    with_tensor_data!(tensor.data(), values => values.iter().for_each(|value| value.hash(&mut hash)));
    hash.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_expected_words_list_every_element_type() {
        let listed: Vec<&str> =
            DType::EXPECTED.split_whitespace().map(|word| word.trim_matches('`')).collect();

        let names: Vec<&str> = DType::ALL.iter().map(|dtype| dtype.name()).collect();
        assert_eq!(listed, names);
    }
}
