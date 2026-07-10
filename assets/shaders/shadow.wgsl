// Shadow pass: depth-only render of chunk geometry into the shadow map.
//
// Reuses the exact same vertex + instance layout as voxel.wgsl (8-byte
// VoxelVertex with two Uint8x4 attributes, plus a per-draw mat4 instance
// matrix) so the same uploaded chunk buffers bind unchanged. Only the
// transform differs: positions go through the shadow camera's
// view-projection instead of the main camera. No color output -- the
// render pass has no color attachment, depth is written automatically.
//
// A tiny depth bias is applied on the pipeline (see ShadowPipeline) to
// fight shadow acne; the vertex shader adds nothing extra so we don't
// introduce peter-panning on the receiver side.

struct ShadowCam {
    view_proj: mat4x4f,
    // voxel_size_m in x; y/z/w unused (kept as a vec4 to satisfy the
    // 16-byte uniform alignment and avoid a separate buffer).
    params: vec4f,
};

@group(0) @binding(0) var<uniform> scam: ShadowCam;

struct Inst {
    @location(4) m0: vec4f,
    @location(5) m1: vec4f,
    @location(6) m2: vec4f,
    @location(7) m3: vec4f,
};

struct VIn {
    @location(0) pos_ao: vec4<u32>,   // x, y, z corner (voxel units), ao 0..3
    @location(1) norm_mat: vec4<u32>, // normal id, jitter, material lo/hi
};

struct VOut {
    @builtin(position) clip: vec4f,
};

@vertex
fn vs(v: VIn, inst: Inst) -> VOut {
    let model = mat4x4f(inst.m0, inst.m1, inst.m2, inst.m3);
    let local = vec3f(f32(v.pos_ao.x), f32(v.pos_ao.y), f32(v.pos_ao.z)) * scam.params.x;
    let wp = (model * vec4f(local, 1.0)).xyz;
    var out: VOut;
    out.clip = scam.view_proj * vec4f(wp, 1.0);
    return out;
}
