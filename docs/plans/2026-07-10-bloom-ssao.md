# Bloom + SSAO Post-Processing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Add bloom (bright light bleeding) and SSAO (screen-space ambient occlusion) post-processing passes to the rendering pipeline.

**Architecture:** A new `BloomSsaoPipeline` struct in `vox-render` owns SSAO generation + blur passes and bloom bright-pass + blur passes, writing to intermediate textures that the existing `PostProcessPipeline` samples in its final composite. No scene pass changes — both effects read the existing HDR color + depth textures. SSAO reconstructs normals from depth gradients (no normal G-buffer needed). Bloom uses a simplified bright-pass + 13-tap separable Gaussian blur at half-res.

**Tech Stack:** Rust, wgpu/WGSL, egui (debug sliders)

**Design doc:** `docs/plans/2026-07-10-bloom-ssao-design.md`

---

## Task 1: Add SSAO and bloom tunable parameters

**Files:**
- Modify: `crates/vox-core/src/tunables.rs` (add fields to `Tunables`)
- Modify: `crates/vox-debug/src/panels.rs` (add sliders)

**Step 1: Add tunable fields**

In `crates/vox-core/src/tunables.rs`, add to the `Tunables` struct:

```rust
/// SSAO intensity (0 = off, 1 = full effect).
pub ssao_intensity: f32,
/// SSAO sample radius in view space.
pub ssao_radius: f32,
/// Bloom intensity (0 = off, 1 = full effect).
pub bloom_intensity: f32,
/// Bloom luminance threshold for bright-pass extraction.
pub bloom_threshold: f32,
```

Add defaults in `Default::default()`:
```rust
ssao_intensity: 1.0,
ssao_radius: 0.5,
bloom_intensity: 0.8,
bloom_threshold: 0.8,
```

**Step 2: Add debug sliders**

In `crates/vox-debug/src/panels.rs`, in the `tuning_window` function, add a new section after "Movement":

```rust
ui.separator();
ui.label("Post-processing:");
ui.add(Slider::new(&mut state.tunables.ssao_intensity, 0.0..=2.0).text("SSAO intensity"));
ui.add(Slider::new(&mut state.tunables.ssao_radius, 0.1..=2.0).text("SSAO radius"));
ui.add(Slider::new(&mut state.tunables.bloom_intensity, 0.0..=2.0).text("Bloom intensity"));
ui.add(Slider::new(&mut state.tunables.bloom_threshold, 0.3..=3.0).text("Bloom threshold"));
```

**Step 3: Verify compilation**

Run: `cargo check --workspace`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/vox-core/src/tunables.rs crates/vox-debug/src/panels.rs
git commit -m "feat(tunables): add SSAO and bloom debug sliders"
```

---

## Task 2: Write the SSAO shader (`ssao.wgsl`)

**Files:**
- Create: `assets/shaders/ssao.wgsl`

**Step 1: Write the SSAO shader**

```wgsl
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
    _pad: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<uniform> params: SsaoParams;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read> kernel: array<vec4f>; // xyz = direction, w = radius scale

// Fullscreen triangle.
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    var p = array<vec2f, 3>(
        vec2f(-1.0, -3.0),
        vec2f(-1.0, 1.0),
        vec2f(3.0, 1.0),
    );
    return vec4f(p[vi], 0.0, 1.0);
}

// Reconstruct view-space position from NDC + depth.
fn view_pos_from_ndc(ndc: vec2f, depth: f32) -> vec4f {
    let clip = vec4f(ndc.x, ndc.y, depth * 2.0 - 1.0, 1.0);
    let view = params.inv_view_proj * clip;
    return view / view.w;
}

// Reconstruct normal from depth gradients (cross pattern).
fn reconstruct_normal(uv: vec2f, ts: vec2f, depth: f32) -> vec3f {
    let l = textureSample(depth_tex, samp, uv + vec2f(-ts.x, 0.0)).r;
    let r = textureSample(depth_tex, samp, uv + vec2f( ts.x, 0.0)).r;
    let d = textureSample(depth_tex, samp, uv + vec2f(0.0, -ts.y)).r;
    let u = textureSample(depth_tex, samp, uv + vec2f(0.0,  ts.y)).r;

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
    let depth = textureSample(depth_tex, samp, uv).r;

    // Skip sky pixels (depth = 1.0 = far plane).
    if (depth >= 1.0) {
        return 1.0;
    }

    let p = view_pos_from_ndc(uv, depth).xyz;
    let n = reconstruct_normal(uv, params.texel_size, depth);

    // Sample hemisphere kernel.
    var occlusion = 0.0;
    let n_samples = i32(params.kernel_size);
    for (i in 0 .. n_samples) {
        let sample_dir = kernel[i].xyz * kernel[i].w;
        let sample_pos = p + n * params.bias + sample_dir * params.radius;

        // Project sample to screen.
        let proj = params.view_proj * vec4f(sample_pos, 1.0);
        let sample_ndc = proj.xy / proj.w;
        let sample_uv = sample_ndc * 0.5 + 0.5;

        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }

        let sample_depth = textureSample(depth_tex, samp, sample_uv).r;
        let sample_view_z = view_pos_from_ndc(sample_uv, sample_depth).z;

        // If the scene geometry is closer than the sample point, it's an occluder.
        let range_check = abs(p.z - sample_view_z) < params.radius;
        if (sample_view_z > sample_pos.z && range_check) {
            occlusion += 1.0;
        }
    }
    occlusion = occlusion / f32(n_samples);

    // 1.0 = unoccluded, 0.0 = fully occluded.
    return 1.0 - occlusion * params.intensity;
}

// --- Blur pass ---

@group(0) @binding(0) var<uniform> blur_params: SsaoParams;
@group(0) @binding(1) var ao_tex: texture_2d<f32>;
@group(0) @binding(2) var blur_samp: sampler;

@fragment
fn fs_blur(@builtin(position) frag_pos: vec4f) -> @location(0) f32 {
    let uv = frag_pos.xy * blur_params.texel_size;
    let ts = blur_params.texel_size;

    // 4x4 box blur.
    var sum = 0.0;
    for (y in -1..2) {
        for (x in -1..2) {
            sum += textureSample(ao_tex, blur_samp, uv + vec2f(f32(x) * ts.x, f32(y) * ts.y)).r;
        }
    }
    return sum / 9.0;
}
```

**Step 2: Verify shader parses**

Run: `cargo test -p vox-render shader_validate -- --nocapture`
Expected: May need to add the new shader to the validation test. Check `crates/vox-render/tests/shader_validate.rs` and add `ssao.wgsl` to the list of validated shaders.

**Step 3: Commit**

```bash
git add assets/shaders/ssao.wgsl crates/vox-render/tests/shader_validate.rs
git commit -m "feat(shader): SSAO generation + blur shader"
```

---

## Task 3: Write the bloom shader (`bloom.wgsl`)

**Files:**
- Create: `assets/shaders/bloom.wgsl`

**Step 1: Write the bloom shader**

```wgsl
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

// Bright-pass: extract pixels above a luminance threshold.
@fragment
fn fs_bright(@builtin(position) frag_pos: vec4f) -> @location(0) vec4f {
    let uv = frag_pos.xy * params.texel_size;
    let color = textureSample(input_tex, samp, uv).rgb;
    let lum = dot(color, vec3f(0.299, 0.587, 0.114));
    let soft = smoothstep(params.threshold, params.threshold + params.knee, lum);
    return vec4f(color * soft, 1.0);
}

// Separable Gaussian blur — 13 taps. Direction is (1,0) for horizontal,
// (0,1) for vertical. The caller sets which via a specialization constant
// or a second bind group; here we use a uniform direction.

@group(0) @binding(0) var<uniform> blur_params: BloomParams;
@group(0) @binding(1) var blur_input: texture_2d<f32>;
@group(0) @binding(2) var blur_samp: sampler;

// 13-tap Gaussian weights (sigma ~4, precomputed).
const GAUSSIAN_WEIGHTS: array<f32, 13> = array<f32, 13>(
    0.002216, 0.008764, 0.026995, 0.064759, 0.120985, 0.176037, 0.199476,
    0.176037, 0.120985, 0.064759, 0.026995, 0.008764, 0.002216,
);

// Blur direction: 0 = horizontal, 1 = vertical.
override blur_direction: u32 = 0u;

@fragment
fn fs_blur(@builtin(position) frag_pos: vec4f) -> @location(0) vec4f {
    let uv = frag_pos.xy * blur_params.texel_size;
    let ts = blur_params.texel_size;
    let dir = select(vec2f(ts.x, 0.0), vec2f(0.0, ts.y), blur_direction == 1u);

    var sum = vec3f(0.0);
    for (i in 0..13) {
        let offset = f32(i - 6) ;
        let w = GAUSSIAN_WEIGHTS[i];
        sum += textureSample(blur_input, blur_samp, uv + dir * offset).rgb * w;
    }
    return vec4f(sum, 1.0);
}
```

**Step 2: Verify shader parses**

Run: `cargo test -p vox-render shader_validate -- --nocapture`
Expected: PASS (add `bloom.wgsl` to the validation list if needed)

**Step 3: Commit**

```bash
git add assets/shaders/bloom.wgsl crates/vox-render/tests/shader_validate.rs
git commit -m "feat(shader): bloom bright-pass + Gaussian blur shader"
```

---

## Task 4: Create `BloomSsaoPipeline` struct and SSAO textures/pipelines

**Files:**
- Create: `crates/vox-render/src/bloom_ssao.rs`
- Modify: `crates/vox-render/src/lib.rs` (add module + re-exports)

**Step 1: Write the pipeline struct**

Create `crates/vox-render/src/bloom_ssao.rs` with the struct, constructor, and SSAO textures/pipelines. The struct owns:
- SSAO pipeline + blur pipeline (two render pipelines from `ssao.wgsl`)
- AO textures (half-res, R16Float, ping-pong pair)
- SSAO kernel buffer (32 vec4 samples, generated on CPU)
- SSAO params buffer
- Bind groups for both SSAO passes

Read `crates/vox-render/src/postprocess.rs` for the pattern of creating textures, bind groups, and pipelines. The `BloomSsaoPipeline::new` takes `&Gpu`, `&str` (ssao shader source), `width`, `height`.

The half-res dimensions are `width/2` and `height/2` (minimum 1).

For the SSAO kernel: generate 32 samples on a hemisphere using a random direction + reflection approach. Upload as a storage buffer of `vec4f` (xyz = direction, w = radius scale that lerps from 0.1 to 1.0).

For the inverse view-projection: the `SsaoParams` uniform needs `inv_view_proj` and `view_proj`. These are updated per-frame via a `write_ssao_params` method that takes the camera matrices + tunable values.

Add `pub mod bloom_ssao;` to `crates/vox-render/src/lib.rs` and re-export `BloomSsaoPipeline`.

**Step 2: Verify compilation**

Run: `cargo check -p vox-render`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/vox-render/src/bloom_ssao.rs crates/vox-render/src/lib.rs
git commit -m "feat(render): BloomSsaoPipeline struct + SSAO textures and pipelines"
```

---

## Task 5: Add bloom textures and pipelines to `BloomSsaoPipeline`

**Files:**
- Modify: `crates/vox-render/src/bloom_ssao.rs`

**Step 1: Add bloom fields and pipelines**

Add to the `BloomSsaoPipeline` struct:
- `bright_pipeline` — bright-pass extraction pipeline (from `bloom.wgsl` `fs_bright`)
- `blur_h_pipeline` — horizontal Gaussian blur pipeline (`fs_blur` with `blur_direction=0`)
- `blur_v_pipeline` — vertical Gaussian blur pipeline (`fs_blur` with `blur_direction=1`)
- `bright_tex` — half-res Rgba16Float (bright extraction output)
- `bloom_tex` — half-res Rgba16Float (blurred bloom, final output)
- Bind groups for each pass

The `new` constructor now takes both shader sources: `ssao_shader: &str` and `bloom_shader: &str`.

Add a `process` method that runs all passes in sequence:
1. SSAO generation → `ao_tex`
2. SSAO blur → `ao_blur_tex`
3. Bloom bright pass → `bright_tex`
4. Bloom blur-H → `bloom_tex`
5. Bloom blur-V → `bright_tex` (ping-pong)

The method takes `&mut self`, `encoder: &mut wgpu::CommandEncoder`, and the scene's `color_view` and `depth_view` as inputs.

**Step 2: Verify compilation**

Run: `cargo check -p vox-render`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/vox-render/src/bloom_ssao.rs
git commit -m "feat(render): bloom bright-pass + blur pipelines"
```

---

## Task 6: Add resize and per-frame params update methods

**Files:**
- Modify: `crates/vox-render/src/bloom_ssao.rs`

**Step 1: Add resize method**

Add `pub fn resize(&mut self, gpu: &Gpu, width: u32, height: u32)` that recreates all textures at the new half-res dimensions and rebuilds bind groups. Follow the same pattern as `PostProcessPipeline::resize`.

**Step 2: Add per-frame params update methods**

Add `pub fn write_params(&self, queue: &wgpu::Queue, view_proj: [[f32; 4]; 4], inv_view_proj: [[f32; 4]; 4], ssao_intensity: f32, ssao_radius: f32, bloom_intensity: f32, bloom_threshold: f32)` that updates the uniform buffers for both SSAO and bloom.

**Step 3: Add accessor methods**

Add `pub fn ao_view(&self) -> &wgpu::TextureView` and `pub fn bloom_view(&self) -> &wgpu::TextureView` so the postprocess pipeline can sample them.

**Step 4: Verify compilation**

Run: `cargo check -p vox-render`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/vox-render/src/bloom_ssao.rs
git commit -m "feat(render): BloomSsaoPipeline resize + per-frame params"
```

---

## Task 7: Modify PostProcessPipeline to sample AO and bloom textures

**Files:**
- Modify: `crates/vox-render/src/postprocess.rs` (add 2 bindings to bind group layout + bind group)
- Modify: `assets/shaders/postprocess.wgsl` (add AO and bloom sampling + application)

**Step 1: Add bindings to PostProcessPipeline**

In `postprocess.rs`, add two new bind group layout entries:
- binding 5: `ao_tex` (Texture, Float, filtered)
- binding 6: `bloom_tex` (Texture, Float, filtered)

The `PostProcessPipeline::new` and `resize` methods need to accept the AO and bloom texture views. Add them as parameters to `new` and `resize`, or add a `set_aux_textures` method.

**Recommended**: Add `ao_view: &wgpu::TextureView` and `bloom_view: &wgpu::TextureView` as parameters to `PostProcessPipeline::new`. Store them (or rather, create the bind group with them). On resize, the caller provides the new views.

**Step 2: Update postprocess.wgsl**

Add bindings:
```wgsl
@group(0) @binding(5) var ao_tex: texture_2d<f32>;
@group(0) @binding(6) var bloom_tex: texture_2d<f32>;
```

In `fs`, after computing the lit color `c` (after tone mapping):
```wgsl
// SSAO: darken crevices.
let ao = textureSample(ao_tex, samp, uv).r;
c = c * ao;

// Bloom: add bright light bleeding.
let bloom = textureSample(bloom_tex, samp, uv).rgb;
c = c + bloom;
```

**Step 3: Verify compilation + shader parses**

Run: `cargo check -p vox-render && cargo test -p vox-render shader_validate -- --nocapture`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/vox-render/src/postprocess.rs assets/shaders/postprocess.wgsl
git commit -m "feat(render): postprocess samples AO + bloom textures"
```

---

## Task 8: Wire BloomSsaoPipeline into the app

**Files:**
- Modify: `crates/vox-app/src/main.rs`

**Step 1: Add BloomSsaoPipeline to VoxApp**

Add `bloom_ssao: vox_render::BloomSsaoPipeline` to the `VoxApp` struct.

In `VoxApp::new`, after creating `postprocess`:
```rust
let ssao_shader = std::fs::read_to_string(assets.join("shaders/ssao.wgsl"))?;
let bloom_shader = std::fs::read_to_string(assets.join("shaders/bloom.wgsl"))?;
let bloom_ssao = vox_render::BloomSsaoPipeline::new(
    &gpu, &ssao_shader, &bloom_shader, surf_w, surf_h,
);
```

**Step 2: Update PostProcessPipeline construction**

Pass the AO and bloom views to PostProcessPipeline::new:
```rust
let postprocess = vox_render::PostProcessPipeline::new(
    &gpu, &post_shader, surf_w, surf_h,
    bloom_ssao.ao_view(),
    bloom_ssao.bloom_view(),
);
```

**Step 3: Wire the render loop**

In the render method, between the scene pass and the postprocess composite:
```rust
// Run SSAO + bloom passes.
self.bloom_ssao.write_params(
    self.gpu.queue(),
    view_proj.to_cols_array_2d(),
    inv_view_proj.to_cols_array_2d(),
    self.tunables.ssao_intensity,
    self.tunables.ssao_radius,
    self.tunables.bloom_intensity,
    self.tunables.bloom_threshold,
);
self.bloom_ssao.process(&mut encoder, self.postprocess.color_view(), self.postprocess.depth_view());
```

The `inv_view_proj` is computed as `view_proj.inverse()` (glam Mat4).

**Step 4: Update resize handler**

```rust
self.bloom_ssao.resize(&self.gpu, width, height);
self.postprocess.resize(&self.gpu, width, height, self.bloom_ssao.ao_view(), self.bloom_ssao.bloom_view());
```

**Step 5: Verify compilation**

Run: `cargo check -p vox-app`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/vox-app/src/main.rs
git commit -m "feat(app): wire BloomSsaoPipeline into render loop"
```

---

## Task 9: Update README and verify

**Files:**
- Modify: `README.md`

**Step 1: Update README rendering section**

Add to the Rendering section:
```
- **SSAO**: Screen-space ambient occlusion reconstructs normals from the
  depth buffer and darkens crevices, under-overhangs, and contact areas
  for real depth perception. Half-resolution AO buffer with box blur.
  Intensity and radius tunable via the debug overlay.
- **Bloom**: Bright pixels (fire, ember, sun-lit surfaces) bleed light
  into surrounding areas via a bright-pass + 13-tap Gaussian blur at
  half-resolution. Intensity and threshold tunable via the debug overlay.
```

**Step 2: Run full test suite**

Run: `cargo test --workspace --lib -- --nocapture`
Expected: ALL PASS

**Step 3: Run the engine and visually verify**

Run: `cargo run -p vox-app --release`
- Confirm: crevices and under-overhangs darken (SSAO)
- Confirm: fire/ember glow and bleed light (bloom)
- Confirm: F3 debug sliders affect both effects
- Confirm: 60 FPS maintained

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document bloom + SSAO post-processing"
```

---

## Summary

| # | Task | Key change |
|---|---|---|
| 1 | Tunable parameters | SSAO + bloom fields in `Tunables`, debug sliders |
| 2 | SSAO shader | `ssao.wgsl` — generation + blur |
| 3 | Bloom shader | `bloom.wgsl` — bright pass + Gaussian blur |
| 4 | BloomSsaoPipeline struct + SSAO | New module, textures, pipelines, kernel |
| 5 | Bloom textures + pipelines | Bright pass, blur-H, blur-V |
| 6 | Resize + params update | Per-frame uniform writes, accessor methods |
| 7 | PostProcess integration | 2 new bindings, AO/bloom sampling in composite |
| 8 | App wiring | Render loop integration, resize handling |
| 9 | README + verify | Documentation, full test suite, visual check |
