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

// Sobel edge detection on depth (rendered to Rgba16Float color texture).
fn sobel_depth(uv: vec2f, ts: vec2f) -> f32 {
    let tl = textureSample(depth_tex, samp, uv + vec2f(-ts.x, -ts.y)).r;
    let tm = textureSample(depth_tex, samp, uv + vec2f(0.0,  -ts.y)).r;
    let tr = textureSample(depth_tex, samp, uv + vec2f( ts.x, -ts.y)).r;
    let ml = textureSample(depth_tex, samp, uv + vec2f(-ts.x,  0.0)).r;
    let mr = textureSample(depth_tex, samp, uv + vec2f( ts.x,  0.0)).r;
    let bl = textureSample(depth_tex, samp, uv + vec2f(-ts.x,  ts.y)).r;
    let bm = textureSample(depth_tex, samp, uv + vec2f(0.0,   ts.y)).r;
    let br = textureSample(depth_tex, samp, uv + vec2f( ts.x,  ts.y)).r;
    let gx = abs(tr + 2.0 * mr + br - tl - 2.0 * ml - bl);
    let gy = abs(bl + 2.0 * bm + br - tl - 2.0 * tm - tr);
    return clamp(gx + gy, 0.0, 1.0);
}

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

    // Sample base color.
    var c = textureSample(color_tex, samp, uv).rgb;

    // Very subtle tone mapping — just a soft knee in highlights to
    // prevent harsh clipping, not full ACES which creates artifacts
    // with the water's flat color values.
    c = c / (c + vec3f(0.6));
    c = clamp(c, vec3f(0.0), vec3f(1.0));

    // Subtle warm tint.
    c = c * vec3f(1.01, 1.0, 0.99);

    return vec4f(c, 1.0);
}
