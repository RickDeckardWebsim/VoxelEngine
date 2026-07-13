// SSAO generation + blur. Two fragment entry points, one vertex shader.
// Reconstructs view-space positions from the depth buffer using the
// inverse projection matrix, samples a hemisphere kernel, and outputs
// a per-pixel AO factor.
//
// Uses proj/inv_proj (NOT view_proj) — SSAO works entirely in view space,
// so the view matrix is irrelevant.

struct SsaoParams {
    proj: mat4x4f,       // projection matrix only (not view_proj)
    inv_proj: mat4x4f,   // inverse projection matrix
    texel_size: vec2f,   // 1.0 / half-res render target dims
    depth_texel_size: vec2f,  // 1.0 / full-res depth texture dims
    radius: f32,
    intensity: f32,
    bias: f32,
    kernel_size: u32,
    _pad: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<uniform> params: SsaoParams;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read> kernel: array<vec4f>;
@group(0) @binding(4) var ao_tex: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    var p = array<vec2f, 3>(
        vec2f(-1.0, -3.0),
        vec2f(-1.0, 1.0),
        vec2f(3.0, 1.0),
    );
    return vec4f(p[vi], 0.0, 1.0);
}

// Load raw depth [0,1] from the depth texture at UV.
fn load_depth(uv: vec2f) -> f32 {
    let dims = textureDimensions(depth_tex);
    let c = clamp(uv, vec2f(0.0), vec2f(1.0));
    let texel = min(vec2u(c * vec2f(dims)), dims - vec2u(1));
    return textureLoad(depth_tex, texel, 0);
}

// Reconstruct view-space position from screen UV + depth.
// UV [0,1] → NDC [-1,1] (X and Y). Depth is already [0,1] NDC Z
// (glam perspective_rh, wgpu convention — camera.rs:53).
fn view_pos_from_uv(uv: vec2f, depth: f32) -> vec3f {
    // wgpu's @builtin(position) has Y=0 at top, but glam's perspective_rh
    // expects NDC Y=+1 at top. Flip Y so reconstruction matches the
    // projection matrix's convention.
    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let view = params.inv_proj * ndc;
    return view.xyz / view.w;
}

// Reconstruct view-space normal from depth gradients.
fn reconstruct_normal(uv: vec2f, depth: f32) -> vec3f {
    let ts = params.depth_texel_size;
    let l = load_depth(uv + vec2f(-ts.x, 0.0));
    let r = load_depth(uv + vec2f( ts.x, 0.0));
    let d = load_depth(uv + vec2f(0.0, -ts.y));
    let u = load_depth(uv + vec2f(0.0,  ts.y));

    let p  = view_pos_from_uv(uv, depth);
    let pl = view_pos_from_uv(uv + vec2f(-ts.x, 0.0), l);
    let pr = view_pos_from_uv(uv + vec2f( ts.x, 0.0), r);
    let pd = view_pos_from_uv(uv + vec2f(0.0, -ts.y), d);
    let pu = view_pos_from_uv(uv + vec2f(0.0,  ts.y), u);

    let dx = pr - pl;
    let dy = pu - pd;
    return normalize(cross(dy, dx));
}

@fragment
fn fs_ssao(@builtin(position) frag_pos: vec4f) -> @location(0) f32 {
    let uv = frag_pos.xy * params.texel_size;
    let depth = load_depth(uv);

    // Sky pixels — no occlusion.
    if (depth >= 1.0) {
        return 1.0;
    }
    let p = view_pos_from_uv(uv, depth);
    let n = reconstruct_normal(uv, depth);

    // Build a TBN basis from the reconstructed normal to orient the
    // hemisphere kernel along the surface. Without this, samples go in
    // all directions (isotropic) — including through the surface —
    // producing flat, uniform darkening instead of contact shadows.
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.99);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    let tbn = mat3x3f(tangent, bitangent, n);

    // SSAO hemisphere sampling: for each kernel sample, rotate it into
    // the surface's TBN frame, project to screen space, sample depth,
    // and check if geometry is closer than the sample point.
    var occlusion = 0.0;
    for (var i = 0u; i < params.kernel_size; i = i + 1u) {
        let sample_dir = kernel[i].xyz;
        let sample_scale = kernel[i].w;
        // Orient the hemisphere sample along the surface normal.
        let oriented_dir = tbn * sample_dir;
        let sample_pos = p + oriented_dir * sample_scale * params.radius;

        // Project the sample point back to screen space. NDC Y=+1 is
        // top (projection convention), but wgpu UV Y=0 is top — flip.
        let projected = params.proj * vec4f(sample_pos, 1.0);
        let sample_uv = vec2f(
            projected.x / projected.w * 0.5 + 0.5,
            1.0 - (projected.y / projected.w * 0.5 + 0.5),
        );
        let sample_depth = load_depth(sample_uv);

        // Reconstruct the view-space Z of the actual geometry at that UV.
        let geom_z = view_pos_from_uv(sample_uv, sample_depth).z;

        // If geometry is closer to the camera than the sample point
        // (smaller Z in view space = closer in RH), the sample is
        // occluded. The bias prevents self-occlusion on flat surfaces.
        if (geom_z <= sample_pos.z + params.bias) {
            // Smooth-step the occlusion by the distance ratio so
            // samples far from the surface contribute less (prevents
            // halos at depth discontinuities).
            let range_check = smoothstep(
                0.0,
                1.0,
                params.radius / abs(p.z - geom_z),
            );
            occlusion += range_check;
        }
    }

    // Normalize and invert: 1.0 = no occlusion, 0.0 = fully occluded.
    let ao = 1.0 - (occlusion / f32(params.kernel_size)) * params.intensity;
    return clamp(ao, 0.0, 1.0);
}

// --- Blur pass: 3x3 box filter ---

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
