//! Shape ops, on any element type; `Pad` against NumPy's `pad`.

use crate::builders::{assert_close, assert_exact, f32s, i64s, op, run, tensor};

/// `[[1, 2, 3], [4, 5, 6]]`.
fn x() -> ptxc_core::HostTensor {
    i64s(&[2, 3], &[1, 2, 3, 4, 5, 6])
}

#[test]
fn reshape_keeps_the_elements_and_expand_broadcasts_them() {
    assert_exact(&op("reshape<shape=[3, 2]>", "i64[3, 2]", &[x()]), &[3, 2], &[1, 2, 3, 4, 5, 6]);
    let column = i64s(&[2, 1], &[7, 8]);
    assert_exact(&op("expand<shape=[2, 3]>", "i64[2, 3]", &[column]), &[2, 3], &[7, 7, 7, 8, 8, 8]);
}

#[test]
fn transpose_moves_input_axis_perm_i_to_output_axis_i() {
    let cube = i64s(&[2, 3, 2], &(0..12).collect::<Vec<_>>());

    assert_exact(
        &op("transpose<permutation=[1, 0]>", "i64[3, 2]", &[x()]),
        &[3, 2],
        &[1, 4, 2, 5, 3, 6],
    );
    assert_exact(
        &op("transpose<permutation=[2, 0, 1]>", "i64[2, 2, 3]", &[cube]),
        &[2, 2, 3],
        &[0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11],
    );
}

#[test]
fn tile_repeats_and_concat_joins_any_element_type() {
    assert_exact(
        &op("tile<repeats=[1, 2]>", "i64[2, 6]", &[x()]),
        &[2, 6],
        &[1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6],
    );
    let joined = op("concat<axis=1>", "i64[2, 4]", &[x(), i64s(&[2, 1], &[7, 8])]);
    assert_exact(&joined, &[2, 4], &[1, 2, 3, 7, 4, 5, 6, 8]);
    let floats = op("concat<axis=0>", "f32[3]", &[f32s(&[1], &[1.5]), f32s(&[2], &[2.5, 3.5])]);
    assert_close(&floats, &[3], &[1.5, 2.5, 3.5]);
}

#[test]
fn slices_step_from_their_start_in_either_direction() {
    let row = i64s(&[5], &[10, 11, 12, 13, 14]);

    assert_exact(
        &op("slice<axes=[[0, 4, -2, 3]]>", "i64[3]", std::slice::from_ref(&row)),
        &[3],
        &[14, 12, 10],
    );
    assert_exact(&op("slice<axes=[[0, 1, 2, 2]]>", "i64[2]", &[row]), &[2], &[11, 13]);
    assert_exact(
        &op("slice<axes=[[1, 1, 1, 2], [0, 1, 1, 1]]>", "i64[1, 2]", &[x()]),
        &[1, 2],
        &[5, 6],
    );
}

#[test]
fn split_gives_consecutive_pieces() {
    let text = "graph(%x: i64[3, 2]) {
  %p, %q = split<axis=0, sizes=[1, 2]> %x : i64[1, 2], i64[2, 2]
  %r, %s = split<axis=1, sizes=[1, 1]> %x : i64[3, 1], i64[3, 1]
  return %p, %q, %r, %s
}
";
    let pieces = run(text, &[i64s(&[3, 2], &[1, 2, 3, 4, 5, 6])]);

    assert_exact(&pieces[0], &[1, 2], &[1, 2]);
    assert_exact(&pieces[1], &[2, 2], &[3, 4, 5, 6]);
    assert_exact(&pieces[2], &[3, 1], &[1, 3, 5]);
    assert_exact(&pieces[3], &[3, 1], &[2, 4, 6]);
}

#[test]
fn pad_modes_match_numpy_even_past_the_axis_extent() {
    let row = || i64s(&[3], &[1, 2, 3]);

    assert_exact(
        &op("pad<mode=reflect, pads=[[4, 4]]>", "i64[11]", &[row()]),
        &[11],
        &[1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3],
    );
    assert_exact(
        &op("pad<mode=edge, pads=[[4, 4]]>", "i64[11]", &[row()]),
        &[11],
        &[1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3],
    );
    assert_exact(
        &op("pad<mode=wrap, pads=[[4, 4]]>", "i64[11]", &[row()]),
        &[11],
        &[3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
    );
    assert_exact(
        &op("pad<mode=reflect, pads=[[2, 1]]>", "i64[4]", &[i64s(&[1], &[7])]),
        &[4],
        &[7, 7, 7, 7],
    );
}

#[test]
fn pad_modes_apply_per_axis() {
    let square = i64s(&[2, 2], &[1, 2, 3, 4]);
    let wide = i64s(&[2, 3], &[1, 2, 3, 4, 5, 6]);

    let constant =
        op("pad<mode=constant, pads=[[1, 0], [0, 2]]>", "i64[3, 4]", &[square, i64s(&[], &[9])]);
    let reflect = op("pad<mode=reflect, pads=[[1, 1], [2, 0]]>", "i64[4, 5]", &[wide]);

    assert_exact(&constant, &[3, 4], &[9, 9, 9, 9, 1, 2, 9, 9, 3, 4, 9, 9]);
    assert_exact(&reflect, &[4, 5], &[6, 5, 4, 5, 6, 3, 2, 1, 2, 3, 6, 5, 4, 5, 6, 3, 2, 1, 2, 3]);
}

#[test]
fn trilu_keeps_one_side_of_a_shifted_diagonal() {
    let square = || i64s(&[3, 3], &(1..=9).collect::<Vec<_>>());
    let trilu = |part: &str, diagonal: i64| {
        op(&format!("trilu<part={part}, diagonal={diagonal}>"), "i64[3, 3]", &[square()])
    };

    assert_exact(&trilu("upper", 0), &[3, 3], &[1, 2, 3, 0, 5, 6, 0, 0, 9]);
    assert_exact(&trilu("lower", 0), &[3, 3], &[1, 0, 0, 4, 5, 0, 7, 8, 9]);
    assert_exact(&trilu("upper", -1), &[3, 3], &[1, 2, 3, 4, 5, 6, 0, 8, 9]);
    assert_exact(&trilu("lower", 1), &[3, 3], &[1, 2, 0, 4, 5, 6, 7, 8, 9]);
    let batch = tensor(&[2, 1, 2], vec![true, true, true, true]);
    assert_exact(
        &op("trilu<part=lower, diagonal=0>", "bool[2, 1, 2]", &[batch]),
        &[2, 1, 2],
        &[1, 0, 1, 0],
    );
}

#[test]
fn reflect_padding_folds_a_longer_axis_back_and_forth() {
    let row = i64s(&[4], &[1, 2, 3, 4]);

    assert_exact(
        &op("pad<mode=reflect, pads=[[5, 5]]>", "i64[14]", &[row]),
        &[14],
        &[2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3],
    );
}
