// Bloom: bright-pass extraction + separable Gaussian blur.
// Simplified v1: bright pass at half-res, single 13-tap blur, no mip chain.

struct BloomParams {
    resolution: vec2f,
    texel_size: vec2f,
    threshold: f32,
    knee: f32,
    intensity: f32,
    _pad: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<uniform> params: BloomParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    var p = array<vec2f, 3>(
        vec2f(-1.0, -3.0),
        vec2f(-1.0, 1.0),
        vec2f(3.0, 1.0),
    );
    return vec4f(p[vi], 0.0, 1.0);
}

@fragment
fn fs_bright(@builtin(position) frag_pos: vec4f) -> @location(0) vec4f {
    let uv = frag_pos.xy * params.texel_size;
    let color = textureSample(input_tex, samp, uv).rgb;
    let lum = dot(color, vec3f(0.299, 0.587, 0.114));
    let soft = smoothstep(params.threshold, params.threshold + params.knee, lum);
    return vec4f(color * soft, 1.0);
}

// 13-tap separable Gaussian blur (sigma ~4). Unrolled because naga
// rejects dynamic indexing of module-scope const arrays.
// Blur direction: 0 = horizontal, 1 = vertical.
override blur_direction: u32 = 0u;

@fragment
fn fs_blur(@builtin(position) frag_pos: vec4f) -> @location(0) vec4f {
    let uv = frag_pos.xy * params.texel_size;
    let ts = params.texel_size;
    let dir = select(vec2f(ts.x, 0.0), vec2f(0.0, ts.y), blur_direction == 1u);

    var sum = vec3f(0.0);
    sum += textureSample(input_tex, samp, uv + dir * -6.0).rgb * 0.002216;
    sum += textureSample(input_tex, samp, uv + dir * -5.0).rgb * 0.008764;
    sum += textureSample(input_tex, samp, uv + dir * -4.0).rgb * 0.026995;
    sum += textureSample(input_tex, samp, uv + dir * -3.0).rgb * 0.064759;
    sum += textureSample(input_tex, samp, uv + dir * -2.0).rgb * 0.120985;
    sum += textureSample(input_tex, samp, uv + dir * -1.0).rgb * 0.176037;
    sum += textureSample(input_tex, samp, uv).rgb * 0.199476;
    sum += textureSample(input_tex, samp, uv + dir *  1.0).rgb * 0.176037;
    sum += textureSample(input_tex, samp, uv + dir *  2.0).rgb * 0.120985;
    sum += textureSample(input_tex, samp, uv + dir *  3.0).rgb * 0.064759;
    sum += textureSample(input_tex, samp, uv + dir *  4.0).rgb * 0.026995;
    sum += textureSample(input_tex, samp, uv + dir *  5.0).rgb * 0.008764;
    sum += textureSample(input_tex, samp, uv + dir *  6.0).rgb * 0.002216;
    return vec4f(sum, 1.0);
}
