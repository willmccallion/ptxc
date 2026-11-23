//! Op attributes as text: `<key=value, ...>` with words, numbers and lists.

use super::cursor::Cursor;
use super::error::{ParseError, ParseErrorKind, Position};
use super::lexer::Token;
use crate::ops::F32Attr;
use ptxc_core::{Axis, Dim, Rank};

/// An attribute value: a word, a number or a list of values.
#[derive(Clone, PartialEq, Debug)]
pub(super) enum AttrValue {
    Word(String),
    Number(String),
    List(Vec<Self>),
}

impl AttrValue {
    pub(super) fn word(word: &str) -> Self {
        Self::Word(word.to_owned())
    }

    pub(super) fn int(value: impl std::fmt::Display) -> Self {
        Self::Number(value.to_string())
    }

    /// A float, which reads back from either a word (`inf`, `nan_...`) or a
    /// number, so it is kept as a number either way.
    pub(super) fn float(value: f32) -> Self {
        Self::Number(format_f32(value))
    }

    pub(super) fn list<T>(items: impl IntoIterator<Item = T>, each: impl Fn(T) -> Self) -> Self {
        Self::List(items.into_iter().map(each).collect())
    }

    fn write(&self, out: &mut String) {
        match self {
            Self::Word(text) | Self::Number(text) => out.push_str(text),
            Self::List(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    item.write(out);
                }
                out.push(']');
            }
        }
    }
}

/// A float as text that parses back to the same bits: shortest decimal when
/// finite, `inf`/`-inf`, and `nan_<hex bits>` keeping a NaN's payload.
pub(super) fn format_f32(value: f32) -> String {
    if value.is_nan() {
        format!("nan_{:08x}", value.to_bits())
    } else if value.is_infinite() {
        (if value.is_sign_negative() { "-inf" } else { "inf" }).to_owned()
    } else {
        format!("{value:?}")
    }
}

/// Appends formatted text. Writing to a `String` cannot fail, so the
/// `fmt::Result` carries no information.
pub(super) fn push_formatted(out: &mut String, args: std::fmt::Arguments<'_>) {
    let _infallible = std::fmt::Write::write_fmt(out, args);
}

/// Writes `<key=value, ...>`, or nothing when there are no attributes.
pub(super) fn write_attrs(out: &mut String, attrs: &[(&'static str, AttrValue)]) {
    if attrs.is_empty() {
        return;
    }
    out.push('<');
    for (index, (key, value)) in attrs.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(key);
        out.push('=');
        value.write(out);
    }
    out.push('>');
}

/// Attributes read from `<...>`, each of which must be taken exactly once.
pub(super) struct Attrs {
    entries: Vec<(String, AttrValue, Position)>,
    end: Position,
}

impl Attrs {
    /// Reads `<key=value, ...>` if the cursor is at `<`.
    pub(super) fn parse(cursor: &mut Cursor) -> Result<Self, ParseError> {
        let mut entries: Vec<(String, AttrValue, Position)> = Vec::new();
        if cursor.eat_punct('<') {
            loop {
                let position = cursor.position();
                let key = cursor.word("an attribute name")?;
                if entries.iter().any(|(existing, ..)| *existing == key) {
                    return Err(ParseError::new(
                        position,
                        ParseErrorKind::UnexpectedAttribute(key),
                    ));
                }
                cursor.expect_punct('=', "`=`")?;
                entries.push((key, parse_value(cursor)?, position));
                if !cursor.eat_punct(',') {
                    break;
                }
            }
            cursor.expect_punct('>', "`,` or `>`")?;
        }
        Ok(Self { entries, end: cursor.position() })
    }

    /// The value of `key`, which must be present.
    pub(super) fn take(&mut self, key: &'static str) -> Result<Attr, ParseError> {
        self.take_optional(key)
            .ok_or_else(|| ParseError::new(self.end, ParseErrorKind::MissingAttribute(key)))
    }

    /// The value of `key`, if present.
    pub(super) fn take_optional(&mut self, key: &'static str) -> Option<Attr> {
        let index = self.entries.iter().position(|(name, ..)| name == key)?;
        let (_, value, position) = self.entries.remove(index);
        Some(Attr { key, value, position })
    }

    /// Fails if any attribute was not taken.
    pub(super) fn finish(self) -> Result<(), ParseError> {
        match self.entries.into_iter().next() {
            Some((key, _, position)) => {
                Err(ParseError::new(position, ParseErrorKind::UnexpectedAttribute(key)))
            }
            None => Ok(()),
        }
    }
}

fn parse_value(cursor: &mut Cursor) -> Result<AttrValue, ParseError> {
    if cursor.eat_punct('[') {
        let mut items = Vec::new();
        if !cursor.eat_punct(']') {
            loop {
                items.push(parse_value(cursor)?);
                if !cursor.eat_punct(',') {
                    break;
                }
            }
            cursor.expect_punct(']', "`,` or `]`")?;
        }
        return Ok(AttrValue::List(items));
    }
    match cursor.scalar("an attribute value")? {
        Token::Number(number) => Ok(AttrValue::Number(number)),
        Token::Word(word) => Ok(AttrValue::Word(word)),
        Token::Value(_) | Token::QuotedValue(_) | Token::Punct(_) => {
            Err(cursor.unexpected("an attribute value"))
        }
    }
}

/// One attribute taken from `Attrs`, read as the type an op needs.
pub(super) struct Attr {
    key: &'static str,
    value: AttrValue,
    position: Position,
}

impl Attr {
    pub(super) const fn bad(&self, problem: &'static str) -> ParseError {
        ParseError::new(self.position, ParseErrorKind::BadAttribute { key: self.key, problem })
    }

    pub(super) fn u64(&self) -> Result<u64, ParseError> {
        self.number_as("a non-negative integer")
    }

    pub(super) fn i64(&self) -> Result<i64, ParseError> {
        self.number_as("an integer")
    }

    pub(super) fn dim(&self) -> Result<Dim, ParseError> {
        self.u64().map(Dim::new)
    }

    pub(super) fn axis(&self) -> Result<Axis, ParseError> {
        self.number_as("an axis index").map(Axis::new)
    }

    pub(super) fn rank(&self) -> Result<Rank, ParseError> {
        self.number_as("a number of axes").map(Rank::new)
    }

    pub(super) fn f32(&self) -> Result<f32, ParseError> {
        let text = match &self.value {
            AttrValue::Word(text) | AttrValue::Number(text) => text,
            AttrValue::List(_) => return Err(self.bad("expected a float")),
        };
        parse_f32(text).ok_or_else(|| self.bad("expected a float"))
    }

    pub(super) fn f32_attr(&self) -> Result<F32Attr, ParseError> {
        self.f32().map(F32Attr::new)
    }

    pub(super) fn word(&self) -> Result<&str, ParseError> {
        match &self.value {
            AttrValue::Word(word) => Ok(word),
            _ => Err(self.bad("expected a word")),
        }
    }

    /// Reads one of an enum's words.
    pub(super) fn enumerated<T: TextWord>(&self) -> Result<T, ParseError> {
        let word = self.word()?;
        T::from_word(word).ok_or_else(|| {
            let found = word.to_owned();
            ParseError::new(
                self.position,
                ParseErrorKind::UnknownWord { expected: T::EXPECTED, found },
            )
        })
    }

    /// The items of a list attribute, each read as an `Attr` of the same key.
    pub(super) fn items(&self) -> Result<Vec<Self>, ParseError> {
        match &self.value {
            AttrValue::List(items) => Ok(items
                .iter()
                .map(|value| Self { key: self.key, value: value.clone(), position: self.position })
                .collect()),
            _ => Err(self.bad("expected a list")),
        }
    }

    /// The items of a list that must have exactly `N` of them.
    pub(super) fn tuple<const N: usize>(&self) -> Result<[Self; N], ParseError> {
        self.items()?.try_into().map_err(|_| self.bad("a list has the wrong length"))
    }

    fn number_as<T: std::str::FromStr>(&self, wanted: &'static str) -> Result<T, ParseError> {
        match &self.value {
            AttrValue::Number(text) => text.parse().map_err(|_| {
                ParseError::new(
                    self.position,
                    ParseErrorKind::BadNumber { text: text.clone(), wanted },
                )
            }),
            _ => Err(self.bad(wanted)),
        }
    }
}

/// Parses `format_f32`'s output back to the same bits.
pub(super) fn parse_f32(text: &str) -> Option<f32> {
    text.strip_prefix("nan_").map_or_else(
        || text.parse().ok(),
        |bits| {
            u32::from_str_radix(bits, 16).ok().map(f32::from_bits).filter(|value| value.is_nan())
        },
    )
}

/// An enum written as one word per variant.
pub(super) trait TextWord: Sized + Copy + 'static {
    /// The words allowed, for error messages.
    const EXPECTED: &'static str;

    /// The variant's word.
    fn word(self) -> &'static str;

    /// The variant `word` names.
    fn from_word(word: &str) -> Option<Self>;
}

/// Implements `TextWord` from one table, so the printer and parser cannot
/// disagree.
macro_rules! text_words {
    ($($type:ty { $($variant:path => $word:literal),+ $(,)? })+) => {$(
        impl $crate::text::attr::TextWord for $type {
            const EXPECTED: &'static str = concat!($("`", $word, "` "),+);

            fn word(self) -> &'static str {
                match self {
                    $($variant => $word,)+
                }
            }

            fn from_word(word: &str) -> Option<Self> {
                match word {
                    $($word => Some($variant),)+
                    _ => None,
                }
            }
        }
    )+};
}

pub(super) use text_words;
