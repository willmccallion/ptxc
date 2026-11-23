//! Styled terminal output on stderr (banner, sections, stats), as in PARE,
//! so stdout stays clean for piping.

use yansi::Paint;

/// The banner and a rule under it.
pub(crate) fn banner() {
    eprintln!(
        "\n  {}  {}",
        format!("ptxc v{}", env!("CARGO_PKG_VERSION")).bold(),
        "ONNX to PTX".dim()
    );
    eprintln!("  {}", "-".repeat(60).dim());
}

/// A section title.
pub(crate) fn section(title: &str) {
    eprintln!("\n  {}", title.bold());
    eprintln!("  {}", "-".repeat(40).dim());
}

/// A labelled header line: `> Label  value`.
pub(crate) fn header(label: &str, value: &str) {
    eprintln!("  {} {}  {}", ">".cyan().bold(), label.cyan().bold(), value);
}

/// One stat, its label dimmed.
pub(crate) fn stat(label: &str, value: &str) {
    eprintln!("    {:<18} {}", label.dim(), value);
}
