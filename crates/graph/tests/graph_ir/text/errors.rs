use ptxc_graph::text::{ParseErrorKind, parse};

fn error_of(text: &str) -> (ParseErrorKind, u32, u32) {
    let error = parse(text, &[]).unwrap_err();
    (error.kind().clone(), error.position().line(), error.position().column())
}

const HEADER: &str = "graph(%x: f32[2]) {\n";

#[test]
fn an_unknown_op_is_reported_where_it_is() {
    let (kind, line, column) =
        error_of(&format!("{HEADER}  %y = frobnicate %x : f32[2]\n  return %y\n}}"));

    assert_eq!(kind, ParseErrorKind::UnknownOp("frobnicate".to_owned()));
    assert_eq!((line, column), (2, 8));
}

#[test]
fn attributes_must_be_complete_known_and_unique() {
    let missing = error_of(&format!("{HEADER}  %y = leaky_relu %x : f32[2]\n  return %y\n}}")).0;
    let unknown =
        error_of(&format!("{HEADER}  %y = neg<alpha=1.0> %x : f32[2]\n  return %y\n}}")).0;
    let repeated = error_of(&format!(
        "{HEADER}  %y = leaky_relu<alpha=1.0, alpha=2.0> %x : f32[2]\n  return %y\n}}"
    ))
    .0;

    assert_eq!(missing, ParseErrorKind::MissingAttribute("alpha"));
    assert_eq!(unknown, ParseErrorKind::UnexpectedAttribute("alpha".to_owned()));
    assert_eq!(repeated, ParseErrorKind::UnexpectedAttribute("alpha".to_owned()));
}

#[test]
fn attribute_values_must_have_the_right_form() {
    let word =
        error_of(&format!("{HEADER}  %y = gelu<approximate=fast> %x : f32[2]\n  return %y\n}}")).0;
    let float =
        error_of(&format!("{HEADER}  %y = leaky_relu<alpha=big> %x : f32[2]\n  return %y\n}}")).0;

    assert!(matches!(word, ParseErrorKind::UnknownWord { found, .. } if found == "fast"));
    assert!(matches!(float, ParseErrorKind::BadAttribute { key: "alpha", .. }));
}

#[test]
fn values_must_be_defined_once_before_use() {
    let undefined = error_of(&format!("{HEADER}  %y = neg %z : f32[2]\n  return %y\n}}")).0;
    let redefined = error_of(&format!("{HEADER}  %x = neg %x : f32[2]\n  return %x\n}}")).0;

    assert_eq!(undefined, ParseErrorKind::UndefinedValue("z".to_owned()));
    assert_eq!(redefined, ParseErrorKind::Redefined("x".to_owned()));
}

#[test]
fn written_result_types_must_match_the_type_rule() {
    let (kind, line, _) = error_of(&format!("{HEADER}  %y = neg %x : f32[3]\n  return %y\n}}"));

    assert!(matches!(kind, ParseErrorKind::Results { .. }), "{kind}");
    assert_eq!(line, 2);
    let too_many =
        error_of(&format!("{HEADER}  %y, %z = neg %x : f32[2], f32[2]\n  return %y\n}}")).0;
    assert!(matches!(too_many, ParseErrorKind::Results { .. }));
}

#[test]
fn ops_the_builder_rejects_are_reported() {
    let kind = error_of(&format!("{HEADER}  %y = not %x : bool[2]\n  return %y\n}}")).0;

    assert!(matches!(kind, ParseErrorKind::Node(_)), "{kind}");
}

#[test]
fn constants_must_fill_their_type() {
    let kind = error_of(&format!("{HEADER}  %c = const f32[3] [1.0, 2.0]\n  return %c\n}}")).0;
    let bad_value = error_of(&format!("{HEADER}  %c = const i8[1] [300]\n  return %c\n}}")).0;
    let elided =
        error_of(&format!("{HEADER}  %c = const f32[20] <elided 0x1234>\n  return %c\n}}")).0;

    assert_eq!(kind, ParseErrorKind::DataLength { expected: 3, actual: 2 });
    assert!(matches!(bad_value, ParseErrorKind::Unexpected { .. }), "{bad_value}");
    assert_eq!(elided, ParseErrorKind::ElidedConstant("0x1234".to_owned()));
}

#[test]
fn lexical_errors_carry_their_position() {
    assert_eq!(error_of("graph(%x: f32[2]) {\n  $").0, ParseErrorKind::UnexpectedChar('$'));
    assert_eq!(error_of("graph(%\"open").0, ParseErrorKind::UnterminatedName);
    assert_eq!(error_of("graph(% ").0, ParseErrorKind::EmptyName);
    assert_eq!(error_of("graph(%\"a\\n\"").0, ParseErrorKind::BadEscape);
}

#[test]
fn text_after_the_graph_or_a_missing_end_is_rejected() {
    let trailing = error_of("graph(%x: f32[2]) {\n  return %x\n} extra").0;
    let unfinished = error_of("graph(%x: f32[2]) {\n  return %x").0;

    assert!(matches!(trailing, ParseErrorKind::Unexpected { expected: "the end of the text", .. }));
    assert!(
        matches!(unfinished, ParseErrorKind::Unexpected { found, .. } if found == "end of input")
    );
}

#[test]
fn bad_numbers_and_shapes_are_rejected() {
    let dimension = error_of("graph(%x: f32[2.5]) {\n  return %x\n}").0;
    let dtype = error_of("graph(%x: f64[2]) {\n  return %x\n}").0;

    assert!(matches!(dimension, ParseErrorKind::BadNumber { wanted: "dimension", .. }));
    assert!(matches!(dtype, ParseErrorKind::UnknownWord { found, .. } if found == "f64"));
}

#[test]
fn errors_display_their_position_and_what_was_found() {
    let error =
        parse(&format!("{HEADER}  %y = frobnicate %x : f32[2]\n  return %y\n}}"), &[]).unwrap_err();
    let found = error_of("graph frob").0;

    assert_eq!(error.to_string(), "2:8: unknown op `frobnicate`");
    assert_eq!(found, ParseErrorKind::Unexpected { expected: "`(`", found: "`frob`".to_owned() });
}

#[test]
fn a_lone_slash_is_not_a_comment() {
    assert_eq!(
        error_of("graph(%x: f32[2]) { / \n return %x }").0,
        ParseErrorKind::UnexpectedChar('/')
    );
}

#[test]
fn an_op_written_without_operands_is_checked_by_its_type_rule() {
    let kind = error_of(&format!("{HEADER}  %y = neg : f32[2]\n  return %y\n}}")).0;

    assert!(matches!(kind, ParseErrorKind::Node(_)), "{kind}");
}
