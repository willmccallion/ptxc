//! The text of a cast, shared by both printers.

use ptxc_core::DType;
use ptxc_core::float::Overflow;

/// `cast<to>`, with the overflow rule only where it applies: FP8 targets.
pub(super) fn cast_name(to: DType, overflow: Overflow) -> String {
    if !matches!(to, DType::F8E4M3Fn | DType::F8E5M2) {
        return format!("cast<{}>", to.name());
    }
    let rule = match overflow {
        Overflow::Saturate => "saturate",
        Overflow::NonFinite => "nonfinite",
    };
    format!("cast<{}, {rule}>", to.name())
}

#[cfg(test)]
mod tests {
    use super::cast_name;
    use ptxc_core::DType;
    use ptxc_core::float::Overflow;

    #[test]
    fn only_fp8_casts_name_their_overflow_rule() {
        assert_eq!(cast_name(DType::F8E5M2, Overflow::NonFinite), "cast<f8e5m2, nonfinite>");
        assert_eq!(cast_name(DType::F8E4M3Fn, Overflow::Saturate), "cast<f8e4m3fn, saturate>");
        assert_eq!(cast_name(DType::I8, Overflow::Saturate), "cast<i8>");
    }
}
