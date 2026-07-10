// HDR post-processing pass: tone-maps the HDR (Rgba16Float) offscreen
// render target to the swapchain's SDR format. Draws a fullscreen triangle.
//
// Tone mapping: ACES filmic approximation (Narkowicz 2015).
// Input: HDR color texture (scene rendered in linear HDR space).
// Output: tone-mapped LDR color to the swapchain.

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
};

// Fullscreen triangle: 3 vertices covering the whole screen.
// vertex 0: (-1, -1) uv (0, 1)
// vertex 1: ( 3, -1) uv (2, 1)
// vertex 2: (-1,  3) uv (0, 2)
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    var out: VOut;
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );
    var uvs = array<vec2f, 3>(
        vec2f(0.0, 1.0),
        vec2f(2.0, 1.0),
        vec2f(0.0, 2.0),
    );
    out.clip = vec4f(positions[vi], 0.0, 1.0);
    out.uv = uvs[vi];
    return out;
}

// ACES filmic tone mapping (Narkowicz 2015 approximation).
fn aces_tonemap(x: vec3f) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
    let c = textureSample(hdr_texture, hdr_sampler, in.uv);
    return vec4f(c.rgb, 1.0);
}
