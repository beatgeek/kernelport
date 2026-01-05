#[test]
fn smoke_builds() {
    // Placeholder: ensures test harness runs and crate links.
    let name = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    assert!(!name.is_empty());
}
