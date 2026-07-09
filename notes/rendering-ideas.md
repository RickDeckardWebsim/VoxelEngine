# 5 Visual Improvements — Rendering Engineer's Brainstorm

Grounded in the actual `vox-render` pipeline: a single opaque `RenderPipeline`
writing **direct to the swapchain view** (`Frame::view`) with
`BlendState::REPLACE`, `depth_write_enabled = true`, `Depth32Float`
attachment used as `RENDER_ATTACHMENT` only, an sRGB surface, `required_features
= empty`, default limits, an 8-byte packed vertex (`Uint8x4 pos_ao` +
`Uint8x4 norm_mat`, all fields consumed), and a palette of `[rgb, jitter]`
storage vec4s. The single uniform is `CameraUniform { view_proj, cam_pos,
sun_dir, fog }` where `fog.w` is unused.

Feasibility tiers:
- **Cheap-add** — fits the existing pass/pipeline with a shader edit + a couple
  uniform floats. No new passes, no new textures, no feature gates.
- **New-pass** — needs a new render pass and/or pipeline (shadow map, HDR
  intermediate, transparent pass). Structurally additive to what exists.
- **Needs-features** — requires `required_features` / non-default limits and a
  capability gate; not free on every backend.

---

## 1. Grass Sway + Animated Foliage (Cheap-add)

**Tier: Cheap-add.** No vertex format change, no new pass, no new texture.

The vertex already carries `world_pos` (computed in `vs` from `pos_ao.xyz *
voxel_size + model`, where `pos_ao.xyz` are **integer voxel corners**, not
sub-voxel positions). Top-of-grass faces can be displaced in the vertex
shader using `world_pos.xz` as a per-voxel phase seed plus a time uniform —
exactly the pattern the codebase already uses for jitter (a hash baked at
mesh time to avoid flicker on moving debris). Sway should only move the
**top** faces of grass voxels; since `VoxelVertex` packs AO into `pos_ao.w`
and the mesh is greedy, we don't have a per-vertex "is this a top edge" bit —
but we *do* have the face normal id (`norm_mat.x`): sway only faces whose
normal is +Y **and** whose material is grass. That gives a clean,
material-and-orientation gated displacement without repacking the vertex.

Because all +Y-face vertices of a given voxel share the same integer `y`,
there is no sub-voxel height to scale amplitude by — sway keys off world XZ
and time uniformly across the whole top face. The merged-quad caveat below
is what actually produces the bending read; amplitude is a flat constant,
not a height gradient.

**What to change:**
- Plumb a time scalar. `CameraUniform.fog.w` is spare (verified: shader only
  reads `fog.xyz`). Repurpose it as `time_s`, or add a 5th `f32` to the uniform
  (the struct is already `Copy`/`Pod`, layout change is one line + shader).
- In `vs`, for +Y faces of grass material ids, displace `wp.xz` by a small
  sinusoid: `sway = sin(time * 2.0 + world_pos.x * 0.7 + world_pos.z * 0.5)
  * amplitude`, where `amplitude ~ 0.15 * voxel_size`. Keep it sub-voxel so it
  doesn't tear into neighbor geometry; clamp to grass-tops only.
- Recompute `out.clip` after displacement (already last step). Normals stay
  +Y; shading is unaffected.

**Why it's cheap:** no CPU work, no buffer rewrite, no new pipeline. A few
sin ops in VS. The existing `write_camera` is the single update point — add
`elapsed` there. This also lays the uniform plumbing the day/night and
particle features below both want.

**Risk:** greedy meshing merges coplanar grass tops into large quads, so a
naive per-vertex phase makes whole merged slabs move in lockstep. Seed the
phase off **vertex local position** — but for +Y faces, `y` is the constant
plane and the four corners vary in **X and Z**, so the seed must be
`pos_ao.xz` (not `pos_ao.xy`). Seed `pos_ao.xy` instead and two corners stay
in lockstep while the other two move — the quad shears instead of bending.
With `pos_ao.xz` each corner gets a distinct phase and the merged slab
bends. This is the one subtlety; everything else is mechanical.

---

## 2. Day/Night Sky + Sun Direction Uniform (Cheap-add → small new-pass)

**Tier: Cheap-add for the sun/light; New-pass only if you want a real sky
shader instead of a flat clear.**

`sun_dir` is currently hardcoded in `write_camera` (`Vec3::new(-0.45, -0.8,
-0.35).normalize()`, voxel_pipeline.rs:368) and mirrored into the Mario
pipeline (main.rs:825). A time-of-day scalar driving a rotating sun is the
single highest-leverage visual change: it makes the existing half-Lambert
terminator, fill light, hemisphere ambient, and fog all come alive for free.

**What to change (cheap path):**
- Add `time_of_day` (0..1, or seconds) to the app's `Tunables` (already has
  live egui sliders). Drive `sun_dir` from it in `write_camera`: rotate a
  fixed sun vector around the X axis by `time * 2π`.
- Modulate the shader's `SKY_COLOR`, `AMBIENT_SKY`, `SUN_STRENGTH`, and the
  app's `CLEAR_COLOR` (main.rs:40) by the sun's elevation — a day curve
  (warm at horizon, blue at zenith) and a night curve (dim blue + a faint
  moon fill). All of these are currently `const` in `voxel.wgsl`; convert the
  ones that vary to uniforms (pack into the spare `fog.w` + a new uniform
  vec4, or extend `CameraUniform`).
- Keep `CLEAR_COLOR` and the shader's `SKY_COLOR` in sync (they're already
  documented as needing to match, main.rs:39).

**Optional new-pass (sky shader):** replace the flat clear with a fullscreen
sky shader — a gradient + a sun disk + a procedural starfield at night. This
needs the HDR intermediate from feature 5, OR a cheap pre-world fullscreen
pass that writes the sky into the swapchain before the voxel pass loads with
`LoadOp::Load`. The latter avoids HDR entirely: clear color becomes
irrelevant, the sky pass paints the background, the voxel pass draws over it
with depth test. **Recommend the cheap path first** (uniform-driven colors +
rotating sun) — it's 90% of the perceived effect for 5% of the work.

**Why this ordering:** the sun uniform is a prerequisite for shadow mapping
(feature 4) and makes the day/night the *spine* that the other features key
off of. Do it first.

---

## 3. Transparent Water Voxels with Refraction (New-pass + meshing-split-or-double-submit)

**Tier: New-pass + a geometry split that the current meshing doesn't give
you.** The pipeline is `BlendState::REPLACE` with `depth_write_enabled =
true` and no blending — opaque-only by construction — so water needs a
second, depth-sorted, blended pass. But as verified in `greedy.rs:emit_quad`,
a chunk's geometry is one interleaved buffer of all materials, so splitting
opaque vs water is not a draw-call filter: it's either a `discard`-based
double-submit (cheap, perf tax) or a `mesh_slab` refactor to emit
per-material-bucket buffers (correct, invasive). Both paths are detailed
below.

**Two structural costs (both verified against the code):**

1. **Material-flag source — a coordinated schema change, not a shader add.**
   `MaterialDef` (material.rs:24-37) is `name/color/jitter/density/strength/
   solid` with no rendering flags, `RawMaterial` is
   `#[serde(deny_unknown_fields)]` (material.rs:61) so a new TOML key is a
   breaking parse change, and the palette buffer only uploads
   `[color.rgb, jitter]` (voxel_pipeline.rs:91-94) — the vec4 is full, there
   is no spare channel for a flag. Adding "is water / transparent" is a
   four-touch plumbing change across the stack:
   - **TOML:** add a `transparent = bool` (or `kind = "water"`) key to
     `RawMaterial`; either extend `deny_unknown_fields` deliberately or
     switch to `allow_unknown_fields` with validation.
   - **`MaterialDef`:** add a `transparent: bool` (or `render_flags: u8`)
     field, defaulting false for every existing material.
   - **Material-properties buffer:** the palette vec4 is full, so add a
     **second** `array<u32>` (or `array<vec4f>`) storage buffer of per-
     material flags bound at a new `@binding(2)`, uploaded from the registry
     alongside the palette in `VoxelPipeline::new`. One buffer, no vertex
     change, no mesh re-generation.
   - **Shader:** read the flag from the new binding; gate the transparent
     branch on `flags[mat_id] & TRANSPARENT != 0u`.
   This is the real cost of proposal #3 — the blend state below is the easy
   part. The same plumbing carries the "is grass" flag proposal #1 *could*
   use, though #1 can also key off a hardcoded grass material id and dodge
   the schema change entirely — do that for #1, take the schema hit for #3.
   Reserved material-id range is the ugly alternative (couples rendering to
   id assignment); avoid.
2. **Second render pass — but geometry is interleaved, so splitting is NOT a
   draw-call filter.** Verified in `greedy.rs:emit_quad` (lines 241-249):
   `mesh_slab` writes **all** quads — stone, dirt, grass, water — into ONE
   interleaved `MeshData.vertices`/`indices` per chunk, and `draw_chunks`
   does one `draw_indexed` over that whole buffer. There is no per-material
   sub-mesh to skip at draw time, and "separate water mesh uploads" would
   require `mesh_slab` to emit N per-material-bucket buffers — a greedy.rs
   refactor, not a render-side change. So the two honest options are:
   - **(b) Cheap first cut — `discard` in `fs` (no mesh change):** keep the
     single interleaved buffer, add a uniform `render_mode` (opaque vs
     transparent) to `CameraUniform`, run BOTH passes over ALL chunks, and
     `discard` fragments whose material flag doesn't match the pass's
     `render_mode`. Two pipelines sharing the shader, one with
     `BlendState::REPLACE` + `depth_write_enabled = true` (opaque), one with
     `BlendState::ALPHA_BLEND` + `depth_write_enabled = false` +
     `depth_compare = LessEqual` (transparent). **Real cost:** double-
     submits all geometry every frame (every chunk drawn twice) and
     `discard` disables early-z on the discarded half — fragments run the
     fragment shader to the discard point before being killed. Acceptable
     for a first cut; the perf tax scales with chunk count.
   - **(a) Do it right — per-material-bucket meshing (greedy.rs refactor):**
     extend `mesh_slab` to emit separate `MeshData` per material bucket
     (opaque vs transparent, or per-material), and `upload_chunk` to store
     N sub-meshes per chunk. Then `draw_chunks` draws opaque buckets,
     `draw_chunks_water` draws transparent buckets — no discard, no double
     submit, early-z intact. Touches `greedy.rs`, `VoxelPipeline`'s chunk
     store, and `upload_chunk`. This is the production target; (b) is the
     bridge.
   Either way the transparent pass sorts water chunks back-to-front (chunks
   are already AABB-culled; sorting by distance-to-camera is cheap). The
   tier for #3 is therefore **New-pass + meshing-split-or-double-submit** —
   not just "new-pass."

**Refraction:** true refraction needs to sample the opaque color buffer — that
requires the HDR intermediate (feature 5) and a fullscreen water pass, which
is a much bigger lift. **Cheaper fake:** offset the water fragment's `world_pos`
lookup by a small screen-space ripple (`sin(time + world_pos.xz)` perturbing
the *fog mix amount* and a subtle hue shift toward cool blue) — reads as
refractive shimmer without sampling the framebuffer. This keeps water in the
geometry pass, no fullscreen pass, no intermediate.

**Feature gates:** none for blending (core wgpu 0.20). The real costs are
the material-flag schema change (#1 above) and the geometry split
(discard-double-submit vs per-material-bucket meshing) — neither is a
feature gate, both are structural.

---

## 4. Cascaded Directional Shadow Mapping (New-pass)

**Tier: New-pass**, with one **needs-features** sub-note for point-light
shadows later.

This is the biggest single visual upgrade and the most structurally invasive.
The payoff: every cliff, tree, and debris body currently gets zero cast
shadows — the half-Lambert + AO fakes contact darkening but nothing casts.
Real sun shadows transform the look at dawn/dusk (which feature 2 now drives).

**Structural costs:**
- **Shadow-map texture.** The existing depth texture is
  `RENDER_ATTACHMENT`-only (gpu.rs:214) and is the *camera's* depth — shadows
  need a **separate** depth texture rendered from the sun's viewpoint. Create
  a dedicated `Depth32Float` (or `Depth16Unorm` to save memory) texture with
  `RENDER_ATTACHMENT` usage for rendering + a sampled view with
  `TEXTURE_BINDING` usage for the main pass. Two views of the same texture is
  fine in wgpu 0.20 via `TextureViewDescriptor` with a different aspect, or
  just two textures. **Cascades:** 3–4 textures at increasing resolution/
  extent, snapped to texel boundaries to avoid shimmer.
- **Shadow render pass.** A new `RenderPass` before the main pass, rendering
  the same chunk + body meshes from the sun's `view_proj` with a depth-only
  pipeline (no color attachment, no fragment shader output — or a minimal
  `fs` that writes nothing). This reuses `draw_chunks`/`draw_bodies` logic
  with a different camera uniform and a depth-only pipeline. Shadow acne is
  handled with `depth_bias` on the shadow pipeline's `DepthStencilState`
  (already a field, currently `Default::default()`).
- **Main-pass sampling.** Add `@binding(2) var shadow_maps: texture_depth_2d`
  (or a texture array — but `TEXTURE_BINDING_ARRAY` is a **needs-features**
  gate; for cascades, bind N separate textures or use a 2D array texture with
  `array_layer_count`, which is core). In `fs`, compute the fragment's
  shadow-clip position per cascade, pick the cascade by view-depth, sample
  with a PCF 3×3 kernel. `texture_depth_2d` + `sampler_comparison` (core in
  wgpu 0.20) gives hardware PCF.

**Feature gates:**
- **Directional (cascaded):** none required. Array textures, depth sampling,
  and `sampler_comparison` are core.
- **Point-light shadows (explosions/torches):** needs `DEPTH_CLAMPING` for
  cube-map single-pass rendering, or a 6-pass face-by-face fallback. Recommend
  **deferring point-light shadows** — explosions are brief; a fake radial
  darkening (feature: dynamic lighting below) is far cheaper. Keep shadow
  mapping for the sun only.

**Why this is #4 not #1:** it's the most invasive (new pass, new textures,
new pipeline, sun-camera culling math) and it *depends on* the sun-uniform
work in feature 2. Do 2 first, then 4.

---

## 5. HDR Post-Processing: Tone Mapping + Bloom + Volumetric Fog (New-pass)

**Tier: New-pass**, the structural keystone for several effects.

Right now `voxel_pipeline` writes directly to the sRGB swapchain view
(`Frame::view`) — there is no intermediate. This blocks *all* of: tone
mapping (must run linear → sRGB encode at the very end), bloom (needs to
sample color), SSAO (needs to sample depth), and screen-space volumetric fog
(needs to sample color + depth). The single structural change that unlocks
all of them is an **HDR intermediate**.

**Structural cost (the real work, not the shaders):**
- Create an `Rgba16Float` (non-sRGB!) offscreen texture at surface size,
  `RENDER_ATTACHMENT | TEXTURE_BINDING`. This is the new color target for the
  voxel pass (and Mario pass). The existing pass's `color_attachments` view
  changes from `frame.view()` to the HDR view.
- Add a **fullscreen-triangle post pass**: a tiny pipeline with 3 hardcoded
  vertices, sampling the HDR texture, running tone mapping (ACES or
  Reinhard) + sRGB encode, writing to `frame.view()`. This is the only pass
  that touches the swapchain.
- Depth texture for SSAO: add `TEXTURE_BINDING` usage to `create_depth_view`
  (gpu.rs:214) — one-line usage flag change, but it means the depth can now be
  *sampled*. SSAO needs a depth + normals intermediate; the voxel shader
  already outputs `world_normal` at `@location(1)` — add a `Rgba16Float`
  normals texture as a second color attachment (multiple render targets are
  core in wgpu 0.20) or reconstruct normals from depth in the SSAO pass.

**What this unlocks, in priority order:**
1. **Tone mapping** — the sRGB surface (gpu.rs:74) means current shader
   output is already gamma-encoded by writing linear values; tone mapping
   fixes the dynamic-range handling and lets the day/night curves (feature 2)
   push brights without clipping. Cheapest win once the intermediate exists.
2. **Bloom** — bright-pass (threshold ~1.0 in HDR) → mip downsample →
   gaussian blur → additive composite. Needs a couple of blur textures.
   Pairs with explosions/torches (dynamic lighting) for the "glow" read.
3. **Volumetric fog** — the current fog is a per-fragment distance lerp
   (voxel.wgsl:118-120). True volumetric = ray-march through the view frustum
   against the sun direction, sampling the shadow map (feature 4) for
   light-shaft occlusion. **This depends on shadow mapping** — do 4 first.
   Cheap fake: screen-space radial glow around the sun disk + height-based
   fog density, no ray-march.
4. **SSAO** — sample the depth+normals intermediates, generate an AO texture,
   multiply into the lit color in the post pass. The engine already has
   *vertex* AO; SSAO adds the large-scale contact darkening vertex AO can't
   catch (under overhangs, in crevices between debris).

**Feature gates:** `Rgba16Float` blending/rendering is core on Vulkan/DX12/
Metal in wgpu 0.20 but **may be unavailable on GL/low-end** — gate on
`TextureFormat::Rgba16Float.is_supported(device)` (the `Gpu` already logs the
selected format; add a capability check at init and fall back to the direct-
to-swapchain path if absent). No `required_features` needed for the common
case; only request features if you want MSAA on the HDR target.

---

## Recommended Sequencing

| Step | Feature | Tier | Why this order |
|------|---------|------|----------------|
| 1 | Grass sway (1) | Cheap-add | Establishes the `time` uniform everyone else needs. Zero risk. |
| 2 | Day/night sun (2) | Cheap-add | Prerequisite for shadows + volumetric; biggest perceived change per LOC. |
| 3 | Transparent water (3) | New-pass + meshing split | Independent of 4/5; first second-pass, but needs the discard-bridge or a greedy.rs per-material-bucket refactor. |
| 4 | HDR + tone map (5) | New-pass | Unlocks bloom + SSAO; needed before volumetric. |
| 5 | Shadow mapping (4) | New-pass | Most invasive; benefits from 2's sun + 5's intermediate. |
| 6 | Bloom / SSAO / volumetric | (5 sub) | Layered on top of 4 + 5. |

Steps 1 and 2 are pure shader/uniform edits against the existing pipeline and
can ship first. Step 3 is the first structural new pass — start with the
`discard`-double-submit bridge (no mesh change) and move to per-material-
bucket meshing only if the perf tax bites. Step 5 (HDR) is the keystone —
once it's in, bloom/SSAO/volumetric become additive fullscreen passes rather
than architectural changes.
