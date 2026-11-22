//! Reading a token stream, with errors that say what was expected.

use super::error::{ParseError, ParseErrorKind, Position};
use super::lexer::{Spanned, Token};

/// A value as written: a real name, or the printer's number for an
/// unnamed value.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(super) enum ValueRef {
    Named(String),
    Unnamed(String),
}

/// A position in a token stream.
pub(super) struct Cursor {
    tokens: Vec<Spanned>,
    index: usize,
}

impl Cursor {
    pub(super) const fn new(tokens: Vec<Spanned>) -> Self {
        Self { tokens, index: 0 }
    }

    pub(super) fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.index).map(|spanned| &spanned.token)
    }

    /// Where the next token starts, or where the last one did at the end.
    pub(super) fn position(&self) -> Position {
        self.tokens
            .get(self.index)
            .or_else(|| self.tokens.last())
            .map_or(Position::START, |spanned| spanned.position)
    }

    pub(super) const fn at_end(&self) -> bool {
        self.index == self.tokens.len()
    }

    /// An error saying `expected` was wanted where the cursor is.
    pub(super) fn unexpected(&self, expected: &'static str) -> ParseError {
        let found = self.peek().map_or_else(|| "end of input".to_owned(), describe);
        ParseError::new(self.position(), ParseErrorKind::Unexpected { expected, found })
    }

    /// Consumes `c` if it is next.
    pub(super) fn eat_punct(&mut self, c: char) -> bool {
        let found = self.peek() == Some(&Token::Punct(c));
        if found {
            self.index += 1;
        }
        found
    }

    pub(super) fn expect_punct(
        &mut self,
        c: char,
        expected: &'static str,
    ) -> Result<(), ParseError> {
        if self.eat_punct(c) { Ok(()) } else { Err(self.unexpected(expected)) }
    }

    /// Consumes the word `word` if it is next.
    pub(super) fn eat_word(&mut self, word: &str) -> bool {
        let found = matches!(self.peek(), Some(Token::Word(next)) if next == word);
        if found {
            self.index += 1;
        }
        found
    }

    pub(super) fn word(&mut self, expected: &'static str) -> Result<String, ParseError> {
        match self.peek() {
            Some(Token::Word(word)) => {
                let word = word.clone();
                self.index += 1;
                Ok(word)
            }
            _ => Err(self.unexpected(expected)),
        }
    }

    /// A value reference and where it was written.
    pub(super) fn value(&mut self) -> Result<(ValueRef, Position), ParseError> {
        let position = self.position();
        let reference = match self.peek() {
            Some(Token::Value(name)) if name.starts_with(|c: char| c.is_ascii_digit()) => {
                ValueRef::Unnamed(name.clone())
            }
            Some(Token::Value(name) | Token::QuotedValue(name)) => ValueRef::Named(name.clone()),
            _ => return Err(self.unexpected("a value such as `%x`")),
        };
        self.index += 1;
        Ok((reference, position))
    }

    /// A number and where it was written.
    pub(super) fn number(
        &mut self,
        expected: &'static str,
    ) -> Result<(String, Position), ParseError> {
        let position = self.position();
        match self.peek() {
            Some(Token::Number(number)) => {
                let number = number.clone();
                self.index += 1;
                Ok((number, position))
            }
            _ => Err(self.unexpected(expected)),
        }
    }

    /// A word or a number, as written: a scalar in attributes and data.
    pub(super) fn scalar(&mut self, expected: &'static str) -> Result<Token, ParseError> {
        match self.peek() {
            Some(token @ (Token::Word(_) | Token::Number(_))) => {
                let token = token.clone();
                self.index += 1;
                Ok(token)
            }
            _ => Err(self.unexpected(expected)),
        }
    }
}

fn describe(token: &Token) -> String {
    match token {
        Token::Word(word) => format!("`{word}`"),
        Token::Number(number) => format!("number `{number}`"),
        Token::Value(name) => format!("value `%{name}`"),
        Token::QuotedValue(name) => format!("value `%{name:?}`"),
        Token::Punct(c) => format!("`{c}`"),
    }
}
