//! Splitting graph IR text into tokens.

use super::error::{ParseError, Position};

/// One token of graph IR text.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum Token {
    /// A word: a keyword, mnemonic, dtype or enum value (`add`, `f32`).
    Word(String),
    /// A number as written (`-3`, `0.5`, `1e-5`, `0x7fc00000`).
    Number(String),
    /// A bare value name after `%`. One starting with a digit is the
    /// printer's number for an unnamed value.
    Value(String),
    /// A quoted value name, `%"..."`, unescaped: always a real name.
    QuotedValue(String),
    /// One of `( ) [ ] < > { } , : =`.
    Punct(char),
}

/// A token and where it starts.
#[derive(Clone, Debug)]
pub(super) struct Spanned {
    pub(super) token: Token,
    pub(super) position: Position,
}

/// The tokens of `text`, skipping whitespace and `//` comments.
pub(super) fn tokenize(text: &str) -> Result<Vec<Spanned>, ParseError> {
    let mut lexer = Lexer { chars: text.chars().collect(), index: 0, position: Position::START };
    let mut tokens = Vec::new();
    while let Some(spanned) = lexer.next_token()? {
        tokens.push(spanned);
    }
    Ok(tokens)
}

struct Lexer {
    chars: Vec<char>,
    index: usize,
    position: Position,
}

impl Lexer {
    fn next_token(&mut self) -> Result<Option<Spanned>, ParseError> {
        self.skip_blank();
        let Some(first) = self.peek() else { return Ok(None) };
        let position = self.position;
        let token = match first {
            '%' => {
                self.advance();
                self.value_name(position)?
            }
            '-' | '0'..='9' => Token::Number(self.take_while(is_number_char)),
            c if is_word_start(c) => Token::Word(self.take_while(is_word_char)),
            '(' | ')' | '[' | ']' | '<' | '>' | '{' | '}' | ',' | ':' | '=' => {
                self.advance();
                Token::Punct(first)
            }
            other => return Err(ParseError::unexpected_char(position, other)),
        };
        Ok(Some(Spanned { token, position }))
    }

    /// A value name: bare word characters, or a quoted string with `\"`
    /// and `\\` escapes.
    fn value_name(&mut self, start: Position) -> Result<Token, ParseError> {
        if self.peek() != Some('"') {
            let name = self.take_while(is_word_char);
            return if name.is_empty() {
                Err(ParseError::empty_name(start))
            } else {
                Ok(Token::Value(name))
            };
        }
        self.advance();
        let mut name = String::new();
        loop {
            match self.bump() {
                Some('"') => return Ok(Token::QuotedValue(name)),
                Some('\\') => match self.bump() {
                    Some(escaped @ ('"' | '\\')) => name.push(escaped),
                    _ => return Err(ParseError::bad_escape(self.position)),
                },
                Some(c) => name.push(c),
                None => return Err(ParseError::unterminated_name(start)),
            }
        }
    }

    fn skip_blank(&mut self) {
        loop {
            match (self.peek(), self.chars.get(self.index + 1)) {
                (Some(c), _) if c.is_whitespace() => self.advance(),
                (Some('/'), Some('/')) => {
                    while self.peek().is_some_and(|c| c != '\n') {
                        self.advance();
                    }
                }
                _ => return,
            }
        }
    }

    fn take_while(&mut self, keep: impl Fn(char) -> bool) -> String {
        let mut taken = String::new();
        while let Some(c) = self.peek().filter(|&c| keep(c)) {
            taken.push(c);
            self.advance();
        }
        taken
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.index).copied()
    }

    fn advance(&mut self) {
        let _consumed: Option<char> = self.bump();
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.index += 1;
        self.position = self.position.after(c);
        Some(c)
    }
}

const fn is_word_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

/// Characters of a word or bare value name.
pub(super) const fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '.'
}

/// Characters of a number: digits, sign, point, exponent and hex digits.
const fn is_number_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '+' | '.')
}
