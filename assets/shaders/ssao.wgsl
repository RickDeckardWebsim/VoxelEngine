// SSAO generation + blur. Two fragment entry points, one vertex shader.
// Reconstructs view-space positions and normals from the depth buffer,
// samples a hemisphere kernel, and outputs a per-pixel AO factor.

struct SsaoParams {
    inv_view_proj: mat4x4f,
    view_proj: mat4x4f,
    resolution: vec2f,
    texel_size: vec2f,
    radius: f32,
    intensity: f32,
    bias: f32,
    kernel_size: u32,
    depth_texel_size: vec2f,  // 1.0 / depth_texture_dimensions (full-res)
    _pad: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> params: SsaoParams;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read> kernel: array<vec4f>;
@group(0) @binding(4) var ao_tex: texture_2d<f32>;

// Fullscreen triangle: 3 vertices covering the screen (no vertex buffer).
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    var p = array<vec2f, 3>(
        vec2f(-1.0, -3.0),
        vec2f(-1.0, 1.0),
        vec2f(3.0, 1.0),
    );
    return vec4f(p[vi], 0.0, 1.0);
}

fn view_pos_from_ndc(uv: vec2f, depth: f32) -> vec4f {
    // glam's perspective_rh produces [0,1] clip Z (wgpu/D3D convention —
    // see camera.rs:53, frustum.rs:29-30). The depth buffer stores this
    // directly, so no remapping is needed. X and Y are still UV [0,1]
    // → NDC [-1,1].
    let clip = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let view = params.inv_view_proj * clip;
    return view / view.w;
}

// Load depth from the depth texture at UV coordinates using textureLoad
// (integer texel fetch). texture_depth_2d cannot use textureSample with a
// regular sampler — only textureSampleCompare with a comparison sampler —
// so we fetch raw depth values via textureLoad instead.
fn load_depth(uv: vec2f) -> f32 {
    let dims = textureDimensions(depth_tex);
    let c = clamp(uv, vec2f(0.0), vec2f(1.0));
    let texel = min(vec2u(c * vec2f(dims)), dims - vec2u(1));
    return textureLoad(depth_tex, texel, 0);
}

fn reconstruct_normal(uv: vec2f, depth: f32) -> vec3f {
    let ts = params.depth_texel_size;
    let l = load_depth(uv + vec2f(-ts.x, 0.0));
    let r = load_depth(uv + vec2f( ts.x, 0.0));
    let d = load_depth(uv + vec2f(0.0, -ts.y));
    let u = load_depth(uv + vec2f(0.0,  ts.y));

    let p = view_pos_from_ndc(uv, depth).xyz;
    let pl = view_pos_from_ndc(uv + vec2f(-ts.x, 0.0), l).xyz;
    let pr = view_pos_from_ndc(uv + vec2f( ts.x, 0.0), r).xyz;
    let pd = view_pos_from_ndc(uv + vec2f(0.0, -ts.y), d).xyz;
    let pu = view_pos_from_ndc(uv + vec2f(0.0,  ts.y), u).xyz;

    let dx = pr - pl;
    let dy = pu - pd;
    return normalize(cross(dy, dx));
}

@fragment
fn fs_ssao(@builtin(position) frag_pos: vec4f) -> @location(0) f32 {
    let uv = frag_pos.xy * params.texel_size;
    let depth = load_depth(uv);

    if (depth >= 1.0) {
        return 1.0;
    }

    let p = view_pos_from_ndc(uv, depth).xyz;
    let n = reconstruct_normal(uv, depth);

    // Build a TBN basis to rotate the tangent-space hemisphere kernel
    // into view space aligned with the surface normal. Without this, the
    // kernel samples along a fixed view-space direction and the AO detaches
    // from geometry ("floating shadows").
    let rvec = vec3f(0.123, 0.456, 0.789);
    let tangent = normalize(rvec - n * dot(rvec, n));
    let bitangent = cross(n, tangent);
    let tbn = mat3x3f(tangent, bitangent, n);

    var occlusion = 0.0;
    let n_samples = i32(params.kernel_size);
    for (var i = 0; i < n_samples; i = i + 1) {
        let sample_dir = (tbn * kernel[i].xyz) * kernel[i].w;
        let sample_pos = p + sample_dir * params.radius + n * params.bias;

        let proj = params.view_proj * vec4f(sample_pos, 1.0);
        let sample_ndc = proj.xy / proj.w;
        // NDC to UV: flip Y because depth buffer origin is top-left.
        let sample_uv = vec2f(sample_ndc.x * 0.5 + 0.5, 0.5 - sample_ndc.y * 0.5);

        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }

        let sample_depth = load_depth(sample_uv);
        let sample_view_z = view_pos_from_ndc(sample_uv, sample_depth).z;

        let range_check = abs(p.z - sample_view_z) < params.radius;
        if (sample_view_z > sample_pos.z && range_check) {
            occlusion += 1.0;
        }
    }
    occlusion = occlusion / f32(n_samples);

    return 1.0 - occlusion * params.intensity;
}

// --- Blur pass: 3x3 box filter over the SSAO texture. ---

@fragment
fn fs_blur(@builtin(position) frag_pos: vec4f) -> @location(0) f32 {
    let uv = frag_pos.xy * params.texel_size;
    let ts = params.texel_size;

    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            sum += textureSample(ao_tex, samp, uv + vec2f(f32(x) * ts.x, f32(y) * ts.y)).r;
        }
    }
    return sum / 9.0;
}
