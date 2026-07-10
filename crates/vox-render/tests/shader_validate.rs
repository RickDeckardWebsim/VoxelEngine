//! Validates that the WGSL shaders parse cleanly under naga (the same
//! frontend wgpu uses at runtime), without needing a GPU device. Catches
//! syntax/type errors in `voxel.wgsl` that `cargo build` cannot, since the
//! shader is loaded as text and only compiled at pipeline creation.

use std::path::Path;

fn parse_shader(rel: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(rel);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {rel}: {e}"));
    let _module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("naga failed to parse {rel}:\n{e}"));
}

#[test]
fn voxel_wgsl_parses() {
    parse_shader("assets/shaders/voxel.wgsl");
}
