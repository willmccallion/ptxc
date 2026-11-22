//! Parses the text format back into a graph.

use super::attr::Attrs;
use super::cursor::{Cursor, ValueRef};
use super::error::{ParseError, ParseErrorKind, Position};
use super::lexer::{Token, tokenize};
use super::ops::decode;
use super::tensor_text::{parse_data, parse_type};
use crate::builder::GraphBuilder;
use crate::graph::Graph;
use crate::tensor_type::TensorType;
use crate::value::{ValueId, ValueName};
use ptxc_core::HostTensor;
use std::collections::HashMap;

/// Parses graph IR text. Elided constants are taken from `elided`, matched
/// by type and content hash; pass an empty slice for text with none.
///
/// Every node goes through `GraphBuilder`, so its types are inferred and
/// must equal the types the text writes after `:`.
pub fn parse(text: &str, elided: &[HostTensor]) -> Result<Graph, ParseError> {
    let mut parser = Parser {
        cursor: Cursor::new(tokenize(text)?),
        builder: GraphBuilder::new(),
        values: HashMap::new(),
        elided,
    };
    parser.header()?;
    loop {
        if parser.cursor.eat_word("return") {
            return parser.finish();
        }
        parser.statement()?;
    }
}

struct Parser<'a> {
    cursor: Cursor,
    builder: GraphBuilder,
    values: HashMap<ValueRef, ValueId>,
    elided: &'a [HostTensor],
}

impl Parser<'_> {
    /// `graph(%x: f32[2, 3], ...) {`
    fn header(&mut self) -> Result<(), ParseError> {
        if !self.cursor.eat_word("graph") {
            return Err(self.cursor.unexpected("`graph`"));
        }
        self.cursor.expect_punct('(', "`(`")?;
        if !self.cursor.eat_punct(')') {
            loop {
                let (reference, position) = self.cursor.value()?;
                self.cursor.expect_punct(':', "`:`")?;
                let input = self.builder.add_input(parse_type(&mut self.cursor)?);
                self.define(reference, position, input)?;
                if !self.cursor.eat_punct(',') {
                    break;
                }
            }
            self.cursor.expect_punct(')', "`,` or `)`")?;
        }
        self.cursor.expect_punct('{', "`{`")
    }

    /// `%a, %b = op<attrs> %x, %y : f32[2], f32[2]` or `%c = const f32[2] [1, 2]`.
    fn statement(&mut self) -> Result<(), ParseError> {
        let mut results = vec![self.cursor.value()?];
        while self.cursor.eat_punct(',') {
            results.push(self.cursor.value()?);
        }
        self.cursor.expect_punct('=', "`,` or `=`")?;
        let position = self.cursor.position();
        if self.cursor.eat_word("const") {
            return self.constant(results);
        }
        let name = self.cursor.word("an op name or `const`")?;
        let op = decode(&name, Attrs::parse(&mut self.cursor)?, position)?;
        let operands = self.operands()?;
        self.cursor.expect_punct(':', "`:` and the result types")?;
        let written = self.types()?;
        let outputs = self
            .builder
            .add_node(op, operands)
            .map_err(|error| ParseError::new(position, error.into()))?;
        self.check_results(&outputs, written, position)?;
        for ((reference, at), output) in results.into_iter().zip(outputs) {
            self.define(reference, at, output)?;
        }
        Ok(())
    }

    fn constant(&mut self, results: Vec<(ValueRef, Position)>) -> Result<(), ParseError> {
        let [(reference, at)]: [_; 1] =
            results.try_into().map_err(|_| self.cursor.unexpected("one name for a constant"))?;
        let ty = parse_type(&mut self.cursor)?;
        let tensor = parse_data(&mut self.cursor, &ty, self.elided)?;
        let value = self.builder.add_constant(tensor);
        self.define(reference, at, value)
    }

    fn operands(&mut self) -> Result<Vec<ValueId>, ParseError> {
        let mut operands = Vec::new();
        if self.at_value() {
            loop {
                operands.push(self.lookup()?);
                if !self.cursor.eat_punct(',') {
                    break;
                }
            }
        }
        Ok(operands)
    }

    fn at_value(&self) -> bool {
        matches!(self.cursor.peek(), Some(Token::Value(_) | Token::QuotedValue(_)))
    }

    fn types(&mut self) -> Result<Vec<TensorType>, ParseError> {
        let mut types = vec![parse_type(&mut self.cursor)?];
        while self.cursor.eat_punct(',') {
            types.push(parse_type(&mut self.cursor)?);
        }
        Ok(types)
    }

    fn check_results(
        &self,
        outputs: &[ValueId],
        written: Vec<TensorType>,
        position: Position,
    ) -> Result<(), ParseError> {
        let graph_types: Vec<TensorType> =
            outputs.iter().map(|&output| self.builder.value_type(output).clone()).collect();
        if graph_types == written {
            Ok(())
        } else {
            Err(ParseError::new(
                position,
                ParseErrorKind::Results { inferred: graph_types, written },
            ))
        }
    }

    fn lookup(&mut self) -> Result<ValueId, ParseError> {
        let (reference, position) = self.cursor.value()?;
        self.values.get(&reference).copied().ok_or_else(|| {
            let (ValueRef::Named(name) | ValueRef::Unnamed(name)) = reference;
            ParseError::new(position, ParseErrorKind::UndefinedValue(name))
        })
    }

    fn define(
        &mut self,
        reference: ValueRef,
        position: Position,
        value: ValueId,
    ) -> Result<(), ParseError> {
        if let ValueRef::Named(name) = &reference {
            self.builder.set_name(value, ValueName::new(name.clone()));
        }
        if self.values.insert(reference.clone(), value).is_some() {
            let (ValueRef::Named(name) | ValueRef::Unnamed(name)) = reference;
            return Err(ParseError::new(position, ParseErrorKind::Redefined(name)));
        }
        Ok(())
    }

    /// `return %y, %z }` and the end of the text.
    fn finish(mut self) -> Result<Graph, ParseError> {
        let position = self.cursor.position();
        let mut outputs = vec![self.lookup()?];
        while self.cursor.eat_punct(',') {
            outputs.push(self.lookup()?);
        }
        self.cursor.expect_punct('}', "`,` or `}`")?;
        if !self.cursor.at_end() {
            return Err(self.cursor.unexpected("the end of the text"));
        }
        self.builder.finish(outputs).map_err(|error| ParseError::new(position, error.into()))
    }
}
