# 5 NEW Visual Improvements — Rendering Engineer's Round 2 Brainstorm

All five ideas are deliberately **distinct from Round 1** (grass sway, day/night
sky + sun uniform, transparent water + refraction, cascaded shadow mapping, HDR
post-processing with tone mapping / bloom / SSAO / volumetric fog). God rays are
NOT proposed here — they are light-shaft volumetrics, already named as a sub-item
of Round 1's HDR feature (#15). SSAO is likewise already claimed by #15, so this
round's GI idea is the color-bleeding / indirect-light kind, not occlusion.

Grounded in the actual `vox-render` pipeline as read from source:
- One opaque `RenderPipeline`, `BlendState::REPLACE`, `depth_write_enabled =
  true`, `Depth32Float` attachment, writes **direct to `Frame::view`** (the
  swapchain), `required_features = empty`, default limits
  (`voxel_pipeline.rs:199-233`, `main.rs:789-810`).
- 8-byte vertex: `pos_ao: vec4<u32>` (xyz corner + ao 0..3) and `norm_mat:
  vec4<u32>` (normal id, jitter, material lo, material hi) — every field
  consumed (`voxel.wgsl:23-26`, `greedy.rs:16-32`).
- Single uniform `Camera { view_proj, cam_pos, sun_dir, fog }` where `fog.w` is
  unused; palette is `array<vec4f>` storage (rgb + jitter) at `@binding(1)`
  (`voxel_pipeline.rs:21-27`, `voxel.wgsl:6-14`, `voxel_pipeline.rs:90-94`).
- Shading: half-Lambert sun + faint opposite fill + two-tone hemisphere ambient
  + **baked vertex AO** (0..3, greedy.rs:63-78) + distance fog
  (`voxel.wgsl:100-122`).
- `sun_dir` hardcoded `(-0.45, -0.8, -0.35)` in `write_camera`
  (`voxel_pipeline.rs:368`).
- Greedy meshing emits **one interleaved buffer per chunk** for all materials
  (`greedy.rs`); no per-material sub-mesh to filter at draw time.
- egui debug overlay lives in `vox-debug`, two-phase `prepare`/`paint`, paints
  inside the voxel pass after the world (`main.rs:831-833`, `vox-debug/lib.rs`).
- Destruction tools: dig, scalable dig, bomb/blast, death laser
  (`main.rs:455-480`); debris bodies are rigidbody chunks, not particles.

Feasibility tiers (same convention as Round 1):
- **Cheap-add** — shader/uniform edit against the existing pass.
- **New-pass** — new render pass and/or pipeline, structurally additive.
- **Needs-features** — gates on non-default wgpu features/limits.
- **New-system** — a new crate or major subsystem, not just a render pass.

---

## 1. Screen-Space Reflections (SSR) — Wet Stone, Water Sheen, Polished Voxels

**Tier: New-pass, depends on Round 1 #15 (HDR intermediate).**

Right now every surface is matte: the half-Lambert + ambient + AO model has no
specular term at all, and water (#13) only fakes refraction by perturbing the
fog mix. There is **no reflection** of any kind — the sky and terrain never
appear mirrored on water tops or on wet/glossy materials. SSR fills that gap
without ray-tracing the scene: it reprojects screen-space fragments along their
reflection vector and samples the HDR color buffer where the ray lands.

**What it adds over Round 1:**
- #13 (water) is **transparency + refraction** (seeing *through* the surface).
  SSR is **reflection** (seeing the world *on* the surface). They compose: a
  water fragment can both refract the fog mix (#13's fake) *and* reflect the
  sky/terrain via SSR. Neither subsumes the other.
- Requires the HDR intermediate (#15) because SSR must **sample the lit color
  buffer** — impossible while writing direct to the swapchain (the current
  path). So this is a follower of #15, not a competitor.

**What to change (after #15 lands):**
- A fullscreen SSR pass sampling the HDR color texture + the depth texture
  (both created by #15; depth just needs `TEXTURE_BINDING` usage added, which
  #15 already does for SSAO). Reuse the same fullscreen-triangle pipeline as
  the post pass.
- Per fragment with a glossy material flag: reconstruct view-space position
  from depth, reflect the view ray about the (reconstructed or MRT) normal,
  ray-march in screen space with a depth thickness test until a hit, sample
  the color buffer, apply a roughness-driven blur ( mip-chain gather for
  glossy, single tap for mirror). Fade by hit confidence and by screen-edge
  distance to avoid the classic SSR edge artifacts.
- **Material flag source:** same four-touch plumbing change Round 1 #3
  already needs (TOML `RawMaterial` + `MaterialDef` + a second per-material
  flags storage buffer at a new binding + shader read). The "is glossy /
  has reflection" flag rides the **same** buffer as "is transparent." Do them
  together — one schema change, two consumers. Key off material id + the new
  flags buffer; do **not** reserve a material-id range (couples rendering to
  id assignment).
- **Roughness source:** there is no per-vertex roughness channel and the
  palette vec4 is full (rgb + jitter). Cheapest path: a per-material
  `roughness` f32 packed into the **same** new flags buffer as a `u32` (pack
  bool flags in low bits, roughness quantized to 8 bits in the high byte). One
  buffer, one binding, no vertex change.

**Risks / honest costs:**
- SSR misses off-screen geometry (cliff behind the camera reflected in a
  forward-facing water plane) → fallback to the sky color (#12's day/night
  sky) for misses, so the reflection never shows a hole. This is why SSR pairs
  with a real sky uniform (#12), not the current flat clear.
- Screen-space ray marching is not free; cap march steps and gate the whole
  pass on `material has any reflective flag` (skip if none). Since most voxel
  materials are matte (stone, dirt, grass), the pass is usually a no-op except
  where water/glossy surfaces actually appear.
- No feature gates (depth + color sampling are core). The only structural
  prerequisite is #15.

**Why this ordering:** SSR is the highest-payoff *follower* of #15 — once the
HDR intermediate exists, SSR and bloom and SSAO are all additive fullscreen
passes. SSR specifically gives water and any future "polished/metal" material
their read, and it is the cheapest way to get specular life into a matte engine.

---

## 2. GPU Particle System — Explosion Fireballs, Impact Dust, Laser Sparks

**Tier: New-system.** A new crate at the render tier (sibling to `vox-render`),
not a shader edit. Round 1 has chain-reaction explosives (#16, gameplay) and
debris rigidbodies (existing), but **no fine particle effects**: a bomb
detonates into a binary puff of nothing, the laser carves stone silently, a dig
impact produces zero dust. Debris bodies are coarse voxel chunks — they cannot
read as fire, smoke, or sparks. This is the missing visual layer.

**What it is:**
- A GPU-instanced particle system: point sprites or camera-facing billboards
  with per-particle position, velocity, age, size, color — all simulated on the
  GPU (a compute or transform-feedback-style ping-pong buffer) so the CPU only
  emits, never steps. Emitter spawners fire on game events: blast detonation,
  laser impact, dig carve, body-ground impact.
- A new render pass (blended, additive for fire/sparks; alpha-blended for
  smoke/dust) drawing after the opaque voxel pass and before the post pass.
  Needs the HDR intermediate (#15) so additive particles can bloom (#15's
  bloom sub-feature) — a fireball that doesn't bloom reads as flat.

**Why a new crate, not a `vox-render` module:**
- Particle simulation is a self-contained subsystem with its own buffers,
  shaders, and emitter API. It belongs at the render tier alongside
  `vox-render`, depending on `vox-core` (for `MaterialRegistry`/events) and
  `vox-platform` (GPU), nothing higher. Keeping it out of `vox-render` preserves
  the current crate's "one opaque pipeline" focus.
- The engine's architecture rule: new systems are new crates at the
  gen/physics tier; a render-side effect system is the rendering analog.

**Integration points (all verified to exist):**
- **Emission triggers:** the destruction tools already return outcomes from
  `dig`/`scalable_dig`/`blast`/`death_laser` (`main.rs:455-480`) and the physics
  solver emits `ImpactEvent`s (imported in `main.rs:31`). Wire emitter spawns
  to these — no new event sources, just consumers.
- **Render pass slot:** particles draw inside the existing frame's command
  encoder, as a second pass after `draw_chunks`/`draw_bodies` and before the
  egui overlay (`main.rs:811-833`). Blended, `depth_write_enabled = false`,
  `depth_compare = LessEqual` so particles sort against terrain depth without
  writing it.
- **Time uniform:** the same `time_s` Round 1 #1 (grass sway) plumbs into
  `CameraUniform.fog.w` drives particle aging — another reason to land that
  uniform first.

**Risks / honest costs:**
- GPU particle simulation in wgpu without compute is awkward; wgpu 0.20
  compute shaders are core on Vulkan/DX12/Metal but **a capability gate on
  WebGL/GL**. Fallback: CPU-simulated particle buffers uploaded per frame
  (bounded emitter count) — fine for hundreds, not tens of thousands. Gate on
  `wgpu::Features::empty()` vs compute support at init.
- Particle count budget must be enforced (the engine already fights unbounded
  debris growth, `main.rs:126-131`); particle emitters should self-cap and
  recycle a fixed pool, never grow.
- No dependency on #15 for the *simulation*, only for bloom — a first cut can
  render particles direct to swapchain (no bloom) and upgrade when #15 lands.

**Why this matters:** destruction is the engine's core verb and it currently
has zero feedback juice. Particles are the single biggest "feel" upgrade per
effort, and they're a reusable system every future effect (weather rain, fire
from vox-sim, torches) inherits.

---

## 3. Progressive Block-Break Crack Decals — Visualizing Damage Before Fracture

**Tier: New-pass + meshing extension.** Today destruction is binary: a voxel is
either solid or carved away in one step (`dig`/`blast`/`laser` delete voxels
instantly). The material `strength` system (`tools.rs`, `main.rs:916-921`) gates
*whether* something fractures, but there is no visible damage state *before* it
goes. A crack decal that spreads across a block's face as it accumulates stress
makes every tool hit feel weighty and telegraphs imminent collapse — the visual
missing piece of the sustained-load (#17) and impact-fracture systems.

**What it is:**
- A damage value per voxel face (0..1) driving a **procedural crack texture**
  sampled in the fragment shader: a Voronoi/cellular crack pattern modulated
  by damage, darkening and widening cracks as damage → 1. At full damage the
  voxel fractures (existing carve pipeline); the cracks are the pre-fracture
  read.
- Two implementation tiers:
  - **Cheap (shader-only, no mesh change):** derive a pseudo-damage from the
    baked `jitter` channel + a per-chunk "stress" uniform, so cracks appear on
    faces the physics solver flags as stressed. Reads as weathering/cracking
    without true per-voxel state. No vertex format change, no new buffer.
  - **Correct (per-voxel damage state):** a sidecar damage array per chunk
    (the design doc's reserved sidecar pattern, same as vox-sim's per-voxel
    state) uploaded as a per-voxel `f32` or `u8` in a new storage buffer,
    sampled in `fs` by mapping the fragment's voxel coordinate → damage value.
    Requires the same "sidecar array, not packed into `Voxel`" decision vox-sim
    (#6) already commits to.

**What to change:**
- **Crack texture:** procedural in-shader (cellular noise from the fragment's
  voxel-space coordinate — reuses the `jitter_hash` pattern already in
  `greedy.rs`) to avoid a texture asset dependency. No sampler, no asset
  pipeline change.
- **Damage source:** the physics solver already computes impact impulse and
    the fracture-threshold check (`main.rs:324-333`, `main.rs:916-921`).
    Surface a per-voxel "stress ratio" (impulse / threshold, clamped 0..1) as
    the damage value. For sustained-load (#17) failures, accumulate stress
    over frames; for impact, a single hit's excess drives a flash crack.
- **Fragment shader:** after the lit color `c` is computed (`voxel.wgsl:116`),
    multiply by `(1 - crack_darkening * damage)` where `crack_darkening` comes
    from the procedural crack pattern thresholded by damage. Sub-voxel, no
    geometry change.

**Risks / honest costs:**
- The correct tier needs the sidecar damage buffer + a chunk-damage upload path
  paralleling `upload_chunk` (`voxel_pipeline.rs:246`) — a real plumbing add,
  not a shader tweak. The cheap tier dodges this but can't show *accumulating*
  damage, only a static stressed look.
- Greedy meshing merges coplanar faces (`greedy.rs`); a crack decal must be
  stable across a merged quad, so the crack pattern keys off **world-space
  voxel coordinate** (derivable from `world_pos / voxel_size`, available in
  `fs` via `in.world_pos`), not per-vertex — otherwise merged slabs show one
  crack, not a per-voxel field.
- No feature gates. The only structural cost is the damage sidecar (shared
  with vox-sim's state-array approach) if you want true accumulation.

**Why this matters:** it's the only proposed feature that makes the *process*
of destruction visible rather than just the *result*. It also gives the
sustained-load (#17) and storm-wind (#3) systems a visual language before they
were purely mechanical.

---

## 4. Voxel Cone-Traced Global Illumination — Indirect Color Bleeding & Directional Ambient

**Tier: New-system + needs-features (storage textures).** Round 1's SSAO (#15
sub-item) is **screen-space ambient occlusion** — it darkens contact/crevice
areas by sampling depth. It does **not** do color bleeding (a red wall bouncing
warm light onto a nearby white floor) or directional indirect light (sky light
diffusely illuminating north-facing slopes). Voxel cone-traced GI is a
fundamentally different, **world-space** technique: build a sparse voxel octree
(or a mipmapped 3D voxel texture) of the scene's radiance, then cone-trace it
per fragment for indirect lighting. This is the GI that SSAO cannot be.

**What it is:**
- An offline/dirty-step **voxel radiance volume**: a 3D texture (or texture
  array of mip levels) where each texel stores the direct-lit color of the
  voxel occupying that cell. Rebuilt incrementally when chunks remesh (the
  `RemeshQueue` already signals chunk changes, `main.rs:86`).
- In the fragment shader, for the indirect term: cast a few cones (wide cone
  for diffuse ambient, narrow cone for directional bounce) from the fragment's
  world position along its normal, marching through the voxel mip hierarchy,
  accumulating radiance. Replaces the current constant `AMBIENT_SKY` /
  `AMBIENT_GROUND` (`voxel.wgsl:41-43`) with scene-driven indirect light.

**Why it's distinct from SSAO and from #15:**
- SSAO is a screen-space **darkening** term; cone-traced GI is a world-space
  **color** term. They compose (SSAO modulates the cone-traced ambient).
- #15's HDR intermediate is *not* required (cone tracing reads the voxel
  volume, not the screen color buffer), though tone mapping (#15) makes the
  resulting indirect light look right. Can ship before #15.

**What to change:**
- A new crate at the render/world tier (depends on `vox-world` for voxel data,
  `vox-render`/`vox-platform` for GPU resources) owning the voxel radiance
  volume and its incremental rebuild.
- **Storage texture / 3D texture:** wgpu 3D textures are core; storage-image
  write access for the rebuild pass may need `wgpu::Features` gating on some
  backends — or rebuild via a CPU-side copy + `write_texture` (slower, no
  feature gate). Recommend the CPU-rebuild path first (chunks remesh rarely),
  GPU-rebuild later.
- **Shader:** a cone-trace function in `voxel.wgsl` sampling the 3D radiance
  texture at a new `@binding`. ~4-8 cone samples per fragment (budget-tunable
  via a `Tunables` slider, `vox-core/tunables.rs`). Replace the constant
  ambient with the trace result, fall back to the current hemisphere ambient
  if the volume is unavailable.

**Risks / honest costs:**
- Cone tracing is the most expensive proposed feature: 4-8 3D-texture samples
  with mip-anisotropy per fragment, every frame. Must be gated by a quality
  slider and a "GI on/off" toggle, with the constant ambient as the fallback.
  Memory: a 3D radiance texture at world resolution with mip chain is real
  VRAM; bound it to a fixed region around the camera and slide it (mirrors the
  streaming idea #7's load radius).
- Sparse octree is the "real" structure but a mipmapped 3D texture is the
  tractable first cut — full octree is a research project, the mip texture is
  shippable.
- The rebuild-on-remesh incremental path must handle the same chunk-edit race
  the mesh system already manages (`remesh.rs`).

**Why this matters:** it's the only proposal that makes light *bounce*. A red
cliff glowing onto sand, a torch (future dynamic light) washing a cave wall in
warm fill, a tree's shadow side picking up green bounce from grass — none of
that is achievable with SSAO or baked vertex AO. It's the ambitious capstone.

---

## 5. Live Minimap / 3D Map Overlay — Spatial Awareness Composite

**Tier: New-pass (small) + egui composite.** The engine has rich spatial
structure — chunk coordinates, player position, frustum, debris fields — but
the only HUD is the egui stats/tuning overlay (`vox-debug/panels.rs`). There is
no map. A live top-down (or isometric) minimap, rendered to a texture each
frame and composited into the HUD, gives orientation in a world that's about to
become infinite (#7 streaming) and is the natural home for tool-radius,
blast-radius, and debris-position readouts.

**What it is:**
- A small (e.g. 256×256) top-down render of the world from a camera positioned
  above the player looking down, rendered to an offscreen texture in its own
  render pass (reusing `VoxelPipeline::draw_chunks` with a top-down
  `view_proj` — no new pipeline, just a second camera uniform + a second color
  target). Cheap: only chunks near the player are drawn; frustum is tiny.
- Composited into the egui overlay as an `egui::TextureHandle` updated each
  frame, drawn in a corner window in `panels.rs`. The overlay already owns egui
  end-to-end (`vox-debug/lib.rs`), so the minimap is a new panel there, not a
  `vox-app` concern.

**What to change:**
- **Offscreen render:** a `Rgba8UnormSrgb` texture at minimap size with
  `RENDER_ATTACHMENT | COPY_SRC | TEXTURE_BINDING`. A second `RenderPass`
  before the main pass, calling `draw_chunks` with an overhead `Frustum` +
  `view_proj` centered on the player. Depth texture: reuse the main depth
  texture's format, separate allocation at minimap size (the main depth is
  screen-sized).
- **Player/tool overlays:** draw the player as a marker, the active tool's
  blast/dig radius as a circle, and (optionally) awake debris bodies as dots —
  all in the minimap pass via a tiny line/point pipeline, or as egui vector
  primitives layered over the texture in `panels.rs` (cheaper, no second
  pipeline). Recommend egui primitives for markers, texture for terrain.
- **egui integration:** `DebugOverlay` exposes a `TextureHandle` updated via
  `egui_wgpu::Renderer`'s texture management; `panels.rs` adds a `minimap()`
  window drawing `ui.image(...)`. The `OverlayState` (`vox-debug/lib.rs:30`)
  gains an optional minimap texture id + player/heading fields.

**Risks / honest costs:**
- A second `draw_chunks` pass doubles terrain draw calls for the minimap
  region — but the minimap frustum is tiny (a few chunks), so the cost is
  small. Cap the minimap radius to keep it bounded; skip the minimap pass
  entirely when the overlay is hidden (F3 off, `main.rs:761`).
- Reusing the main depth texture at a different size is not allowed; a
  dedicated minimap depth texture is a small fixed VRAM cost.
- Isometric/3D-tilted variant is a free parameter of the overhead camera (just
  tilt the view matrix) — same pass, no extra work. Recommend a shallow tilt
  for depth read over pure top-down.
- No feature gates. No dependency on #15 (can render direct to an LDR minimap
  texture; tone mapping not needed for a map).

**Why this matters:** it's the cheapest proposal and the only one that improves
*navigation* rather than fidelity. Once streaming (#7) makes the world
infinite, a map stops being a nice-to-have and becomes essential — and it
showcases the engine's spatial data (chunks, frustum, debris) in the HUD the
player actually looks at.

---

## Recommended Sequencing (Round 2, assuming Round 1 lands first)

| Step | Feature | Tier | Why this order |
|------|---------|------|----------------|
| 1 | Minimap (5) | New-pass (small) | No prerequisites; standalone; cheapest; immediate navigation value. |
| 2 | Particles (2) | New-system | No hard dep (bloom wants #15 but simulation stands alone); biggest "feel" win; reusable by every future effect. |
| 3 | Crack decals (3) | New-pass + sidecar | Cheap tier is shader-only; correct tier shares the sidecar pattern with vox-sim (#6). Pairs with sustained-load (#17). |
| 4 | SSR (1) | New-pass | Strict follower of #15 (needs HDR color+depth sampling). Highest-payoff specular upgrade once the intermediate exists. |
| 5 | Cone-traced GI (4) | New-system | Most expensive; world-space, doesn't need #15 but benefits from tone mapping. Capstone; gate behind a quality toggle. |

Steps 1-3 are independent of Round 1's HDR work and can proceed in parallel
with it. Steps 4-5 are followers (#4 strictly, #5 loosely). None of the five
overlap with Round 1's five rendering ideas.
