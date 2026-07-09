// Post-processing pass: edge detection (Sobel on depth + normal),
// material-tinted outlines, saturation boost, and dreamy color grading.
// Renders as a fullscreen triangle (3 vertices, no vertex buffer).

struct Params {
    resolution: vec2f,   // screen size in pixels
    texel_size: vec2f,   // 1.0 / resolution
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;

// Fullscreen triangle: 3 vertices covering the screen.
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    // Triangle covering [-1, 1] clip space.
    var p = array<vec2f, 3>(
        vec2f(-1.0, -3.0),
        vec2f(-1.0, 1.0),
        vec2f(3.0, 1.0),
    );
    return vec4f(p[vi], 0.0, 1.0);
}

// Sobel edge detection on a single channel.
fn sobel(tex: texture_2d<f32>, uv: vec2f, ts: vec2f) -> f32 {
    let tl = textureSample(tex, samp, uv + vec2f(-ts.x, -ts.y)).r;
    let tm = textureSample(tex, samp, uv + vec2f(0.0,  -ts.y)).r;
    let tr = textureSample(tex, samp, uv + vec2f( ts.x, -ts.y)).r;
    let ml = textureSample(tex, samp, uv + vec2f(-ts.x,  0.0)).r;
    let mr = textureSample(tex, samp, uv + vec2f( ts.x,  0.0)).r;
    let bl = textureSample(tex, samp, uv + vec2f(-ts.x,  ts.y)).r;
    let bm = textureSample(tex, samp, uv + vec2f(0.0,   ts.y)).r;
    let br = textureSample(tex, samp, uv + vec2f( ts.x,  ts.y)).r;
    let gx = abs(tr + 2.0 * mr + br - tl - 2.0 * ml - bl);
    let gy = abs(bl + 2.0 * bm + br - tl - 2.0 * tm - tr);
    return clamp(gx + gy, 0.0, 1.0);
}

// Sobel on a vec3 texture (normal).
fn sobel_vec3(tex: texture_2d<f32>, uv: vec2f, ts: vec2f) -> f32 {
    let tl = textureSample(tex, samp, uv + vec2f(-ts.x, -ts.y)).rgb;
    let tm = textureSample(tex, samp, uv + vec2f(0.0,  -ts.y)).rgb;
    let tr = textureSample(tex, samp, uv + vec2f( ts.x, -ts.y)).rgb;
    let ml = textureSample(tex, samp, uv + vec2f(-ts.x,  0.0)).rgb;
    let mr = textureSample(tex, samp, uv + vec2f( ts.x,  0.0)).rgb;
    let bl = textureSample(tex, samp, uv + vec2f(-ts.x,  ts.y)).rgb;
    let bm = textureSample(tex, samp, uv + vec2f(0.0,   ts.y)).rgb;
    let br = textureSample(tex, samp, uv + vec2f( ts.x,  ts.y)).rgb;
    let gx = length(tr + 2.0 * mr + br - tl - 2.0 * ml - bl);
    let gy = length(bl + 2.0 * bm + br - tl - 2.0 * tm - tr);
    return clamp(gx + gy, 0.0, 1.0);
}

// Boost saturation by a factor. Uses luminance-weighted mixing.
fn boost_saturation(c: vec3f, amount: f32) -> vec3f {
    let lum = dot(c, vec3f(0.299, 0.587, 0.114));
    return mix(vec3f(lum), c, amount);
}

@fragment
fn fs(@builtin(position) frag_pos: vec4f) -> @location(0) vec4f {
    let uv = frag_pos.xy * params.texel_size;
    let ts = params.texel_size;

    // Sample base color.
    let base = textureSample(color_tex, samp, uv).rgb;

    // Edge detection: depth edges (silhouette/depth discontinuities)
    // and normal edges (face/material boundaries).
    let depth_edge = sobel(depth_tex, uv, ts);
    let normal_edge = sobel_vec3(normal_tex, uv, ts);

    // Combine edges. Depth edges are stronger (silhouettes); normal
    // edges are softer (interior face boundaries). Threshold to keep
    // only meaningful edges.
    let edge = clamp(depth_edge * 2.0 + normal_edge * 1.5, 0.0, 1.0);
    let edge_mask = smoothstep(0.15, 0.4, edge);

    // Material-tinted outline: darken the base color by 65% for the
    // outline, giving a material-tinted dark edge instead of pure black.
    let outline_color = base * 0.35;
    var c = mix(base, outline_color, edge_mask);

    // Saturation boost (+30%).
    c = boost_saturation(c, 1.3);

    // Dreamy color grading: slight shadow lift, warm tint, gentle
    // contrast S-curve.
    c = c + 0.02;                           // shadow lift
    c = c * vec3f(1.02, 1.0, 0.98);         // warm tint
    c = clamp(c, vec3f(0.0), vec3f(1.0));
    // Gentle S-curve for contrast.
    c = mix(c, smoothstep(vec3f(0.0), vec3f(1.0), c), 0.15);

    return vec4f(c, 1.0);
}
