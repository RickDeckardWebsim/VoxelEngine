# Voxel Engine MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Build the approved from-scratch modular voxel engine MVP — "Teardown-scale Minecraft": walkable generated world with trees, place/break tools, and a blast tool that carves the world and spawns voxel-accurate rigidbody debris, at any per-world voxel scale from 0.1 m to 1.0 m.

**Architecture:** Rust Cargo workspace of nine crates with strictly downward dependencies (`vox-core → vox-world → {vox-gen, vox-mesh, vox-physics} → vox-render/vox-platform/vox-debug → vox-app`). All engine behavior (storage, noise, meshing, physics solver, destruction) is custom; third-party crates are infrastructure only. Everything below the renderer runs headless and is unit-tested.

**Tech Stack:** Rust 2024 edition (toolchain 1.92 installed), wgpu 0.20.1, winit 0.29, glam 0.28, rayon, egui 0.28 (debug overlay only), tracing, thiserror 2, toml/serde. Versions are pinned to the set proven to compile and render on this machine (see §Conventions).

**Design doc:** `docs/plans/2026-07-04-voxel-engine-design.md` (approved). Read it before starting.

---

## Conventions & Global Setup (read first)

**Working directory:** `C:\Users\dickr\Desktop\My Stuff lol\sandboxing\voxelengine` (fresh git repo, `main` branch, design doc already committed).

**Commands** (run from workspace root):
- Build: `cargo build` — Test: `cargo test` — Lint gate: `cargo clippy --all-targets -- -D warnings`
- Run app: `cargo run -p vox-app --release` (always `--release` for play; debug is too slow for physics+meshing)
- Test one crate: `cargo test -p vox-world`

**Version pins (proven on this machine — do not bump during MVP):** wgpu `0.20.1`, winit `0.29` (+`rwh_06` feature), egui/egui-wgpu/egui-winit `0.28`, glam `0.28`, bytemuck `1` (+derive), rayon `1`, serde `1` (+derive), toml `0.8`, thiserror `2`, tracing `0.1`, tracing-subscriber `0.3`, pollster `0.3`. A post-MVP task may modernize (wgpu 26+/winit 0.30) — renderer/platform crates isolate the churn.

**Windows note (from archived project):** `[profile.release] codegen-units = 4` — codegen-units=1 exhausted the paging file on this machine. Keep it.

**Engine constants** (single source of truth — define in `vox-core::consts`, reference everywhere):

| Constant | Value | Meaning |
|---|---|---|
| `CHUNK_SIZE` | `32` (usize) | voxels per chunk axis |
| `GRAVITY` | `9.81` m/s² | |
| `PHYSICS_DT` | `1.0/60.0` s | fixed step |
| `SUBSTEPS` | `2` | physics substeps per step |
| `SOLVER_ITERS` | `8` | velocity iterations per substep |
| `CONTACT_BETA` | `0.2` | Baumgarte factor |
| `CONTACT_SLOP` | `0.005` m | allowed penetration |
| `FRICTION` | `0.6` | Coulomb μ |
| `SLEEP_LIN` | `0.03` m/s | sleep threshold (linear) |
| `SLEEP_ANG` | `0.20` rad/s | sleep threshold (angular) |
| `SLEEP_FRAMES` | `45` | consecutive quiet steps before sleep |
| `PLAYER_SIZE` | `(0.6, 1.8, 0.6)` m | AABB w,h,d |
| `PLAYER_EYE` | `1.62` m | eye height |
| `STEP_HEIGHT` | `0.55` m | auto step-up |
| `JUMP_HEIGHT` | `1.25` m | jump apex |
| `REACH` | `5.0` m | tool raycast |
| `BLAST_RADIUS` | `1.5` m | default blast |
| `DEBRIS_MIN_VOXELS` | `4` | smaller detached components are discarded |
| `MAX_BODY_VOXELS` | `65_536` | detached components larger than this stay in-world (safety valve) |

**Style:** rustfmt defaults; no `unwrap()` outside tests and `main()` bootstrap; hot loops use `debug_assert!`; every public type/fn gets a one-line doc comment; SI units in all public APIs (meters, seconds, kg) — voxel-count APIs must say `_voxels` in the name.

**Commit style:** `feat(world): …`, `test(mesh): …`, `fix(physics): …` — commit after every task minimum; end commits with the Claude co-author trailer.

---

## Milestone M0 — Scaffold & Window

### Task 1: Workspace scaffold (nine crates, pinned deps)

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/vox-core/Cargo.toml`, `crates/vox-core/src/lib.rs` (and same pair for `vox-world`, `vox-gen`, `vox-mesh`, `vox-physics`, `vox-render`, `vox-platform`, `vox-debug`)
- Create: `crates/vox-app/Cargo.toml`, `crates/vox-app/src/main.rs`
- Create: `rustfmt.toml` (empty = defaults), `assets/materials/.gitkeep`, `assets/shaders/.gitkeep`

**Step 1: Root `Cargo.toml`:**

```toml
[workspace]
resolver = "2"
members = ["crates/*"]

[workspace.package]
edition = "2024"
version = "0.1.0"
license = "MIT OR Apache-2.0"

[workspace.dependencies]
vox-core = { path = "crates/vox-core" }
vox-world = { path = "crates/vox-world" }
vox-gen = { path = "crates/vox-gen" }
vox-mesh = { path = "crates/vox-mesh" }
vox-physics = { path = "crates/vox-physics" }
vox-render = { path = "crates/vox-render" }
vox-platform = { path = "crates/vox-platform" }
vox-debug = { path = "crates/vox-debug" }

wgpu = { version = "0.20.1", features = ["wgsl"] }
winit = { version = "0.29", features = ["rwh_06"] }
egui = "0.28"
egui-wgpu = "0.28"
egui-winit = "0.28"
glam = { version = "0.28", features = ["bytemuck"] }
bytemuck = { version = "1", features = ["derive"] }
rayon = "1"
serde = { version = "1", features = ["derive"] }
toml = "0.8"
thiserror = "2"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
pollster = "0.3"

[profile.dev]
opt-level = 1
[profile.dev.package."*"]
opt-level = 3

[profile.release]
lto = "thin"
codegen-units = 4   # 1 exhausts the Windows paging file on this machine
```

**Step 2:** Each crate `Cargo.toml` uses `edition.workspace = true`, `version.workspace = true`, and only workspace deps. Dependency edges (enforce by simply not listing anything else):
- `vox-core`: glam, serde, toml, thiserror, tracing
- `vox-world`: vox-core (+thiserror, tracing, glam)
- `vox-gen`: vox-core, vox-world, glam, tracing
- `vox-mesh`: vox-core, vox-world, glam, bytemuck
- `vox-physics`: vox-core, vox-world, glam, rayon, tracing
- `vox-render`: vox-core, vox-mesh, wgpu, glam, bytemuck, thiserror, tracing, pollster
- `vox-platform`: winit, glam, tracing, thiserror
- `vox-debug`: egui, egui-wgpu, egui-winit, wgpu, winit, vox-core
- `vox-app` (bin): everything above + tracing-subscriber

Each `lib.rs` starts as `//! <crate one-liner>` only. `vox-app/src/main.rs`: `fn main() { println!("voxelengine"); }`

**Step 3:** Run `cargo build` → succeeds. Run `cargo clippy --all-targets -- -D warnings` → clean.

**Step 4: Commit** `feat: scaffold nine-crate workspace with pinned infrastructure deps`

### Task 2: Window + wgpu clear + frame loop skeleton

**Files:**
- Create: `crates/vox-platform/src/lib.rs` (window+loop), `crates/vox-platform/src/input.rs`, `crates/vox-platform/src/time.rs`
- Create: `crates/vox-render/src/lib.rs`, `crates/vox-render/src/gpu.rs`
- Modify: `crates/vox-app/src/main.rs`

**Step 1 — `time.rs`:** `FrameClock { last: Instant, accumulator: f32 }` with `tick() -> FrameTiming { dt_frame: f32, physics_steps: u32, alpha: f32 }`: accumulate real dt (clamp to 0.25 s max), emit `floor(acc/PHYSICS_DT)` steps (cap 4, drop excess), `alpha = acc / PHYSICS_DT`. Unit-test the accumulator math headlessly (feed synthetic dts, assert step counts and alpha ∈ [0,1)).

**Step 2 — `input.rs`:** `InputState` with `keys_down: HashSet<KeyCode>`, `pressed_this_frame`, `mouse_delta: Vec2`, `mouse_buttons`, `wheel_delta`, fed by winit events; `end_frame()` clears per-frame sets. Actions resolved by callers (no keymap indirection yet — YAGNI).

**Step 3 — `gpu.rs`:** `Gpu::new(window: Arc<Window>) -> Result<Gpu, RenderError>`: wgpu 0.20 idioms — `Instance::new(InstanceDescriptor::default())`, `instance.create_surface(window.clone())` (Arc gives `Surface<'static>`), `request_adapter` (compatible_surface, HighPerformance), `request_device` (default limits), pick sRGB surface format, `PresentMode::Fifo`, create `Depth32Float` depth texture; `resize(w,h)` reconfigures both. `begin_frame()/present()` helpers. All fallible paths return `RenderError` (thiserror) with context.

**Step 4 — platform loop:** winit 0.29 style: `EventLoop::new()`, `WindowBuilder` (1600×900, title "voxelengine"), `event_loop.run(move |event, elwt| ...)` dispatching to an `App` trait object: `trait App { fn frame(&mut self, input: &mut InputState, timing: FrameTiming); fn resize(&mut self, w: u32, h: u32); fn window_event(&mut self, ev: &WindowEvent) -> bool; }`. `AboutToWait` → request redraw; `RedrawRequested` → clock tick + `app.frame(...)`.

**Step 5 — app:** init tracing-subscriber (env filter, default `info`), create window+Gpu, frame = render pass clearing to sky blue `(0.45, 0.66, 0.90)` + depth clear. Run: `cargo run -p vox-app --release` → window opens, sky-blue, resizes without panic, Esc closes.

**Step 6: Commit** `feat(platform,render): window, wgpu device, fixed-timestep frame loop`

---

## Milestone M1 — World, Meshing, Fly Camera

### Task 3: vox-core — coordinates, config, errors

**Files:**
- Create: `crates/vox-core/src/lib.rs`, `src/consts.rs` (table above), `src/coords.rs`, `src/config.rs`, `src/error.rs`

**Step 1 — failing tests** in `coords.rs` `#[cfg(test)]`:

```rust
#[test]
fn chunk_of_negative_voxel() {
    assert_eq!(chunk_of(IVec3::new(-1, 0, 31)), IVec3::new(-1, 0, 0));
    assert_eq!(local_of(IVec3::new(-1, 0, 33)), UVec3::new(31, 0, 1));
}
#[test]
fn world_voxel_roundtrip() {
    let cfg = WorldConfig { voxel_size_m: 0.1, ..Default::default() };
    let v = voxel_at(Vec3::new(1.05, -0.32, 0.0), cfg.voxel_size_m);
    assert_eq!(v, IVec3::new(10, -4, 0));
    let c = voxel_center_m(v, cfg.voxel_size_m);
    assert!((c - Vec3::new(1.05, -0.35, 0.05)).abs().max_element() < 1e-6);
}
```

**Step 2 — implementation** (the euclid math is the whole point — copy exactly):

```rust
pub const CHUNK: i32 = crate::consts::CHUNK_SIZE as i32;

/// Chunk position containing a world-voxel position.
pub fn chunk_of(v: IVec3) -> IVec3 { v.div_euclid(IVec3::splat(CHUNK)) }
/// Position within its chunk (0..32 on each axis).
pub fn local_of(v: IVec3) -> UVec3 { v.rem_euclid(IVec3::splat(CHUNK)).as_uvec3() }
/// World-voxel position of a chunk's minimum corner.
pub fn chunk_origin(c: IVec3) -> IVec3 { c * CHUNK }
/// Voxel containing a world-space point (meters).
pub fn voxel_at(p_m: Vec3, voxel_size_m: f32) -> IVec3 { (p_m / voxel_size_m).floor().as_ivec3() }
/// Center of a voxel in meters.
pub fn voxel_center_m(v: IVec3, voxel_size_m: f32) -> Vec3 { (v.as_vec3() + 0.5) * voxel_size_m }
```

**Step 3 — `config.rs`:** `WorldConfig { seed: u64, voxel_size_m: f32, extent_m: Vec3 }` (Default: seed 1337, 0.1, `(256.0, 64.0, 256.0)`) + `extent_chunks(&self) -> IVec3` (ceil of extent_m / (voxel_size·32)). `error.rs`: `CoreError` enum (Config/Asset variants). Tests pass → `cargo test -p vox-core`.

**Step 4: Commit** `feat(core): euclid-correct coordinate math, world config, engine constants`

### Task 4: vox-core — material registry from TOML

**Files:**
- Create: `crates/vox-core/src/material.rs`, `assets/materials/core.toml`

**Step 1 — failing tests:** parse a registry from an inline TOML string → lookup by name and id; air is always id 0 with `solid=false`; a material missing `density` produces `Err` whose `Display` contains the material name and the word `density`; duplicate names error.

**Step 2 — implementation:** `MaterialId(pub u16)`; `MaterialDef { name: String, color: [f32; 3], jitter: f32, density: f32, strength: f32, solid: bool }`; `MaterialRegistry { defs: Vec<MaterialDef>, by_name: HashMap<String, MaterialId> }` with `from_toml_str(&str) -> Result<Self, CoreError>` and `load_dir(path)` merging all `.toml` files sorted by filename. Registry index 0 is a built-in `air` (solid=false, density 0). serde structs mirror the TOML; validation converts to typed errors naming file + field.

**Step 3 — `assets/materials/core.toml`:**

```toml
[[material]]
name = "stone";   color = [0.55, 0.55, 0.57]; jitter = 0.04; density = 2600.0; strength = 8.0
[[material]]
name = "dirt";    color = [0.45, 0.32, 0.22]; jitter = 0.05; density = 1500.0; strength = 2.0
[[material]]
name = "grass";   color = [0.33, 0.55, 0.25]; jitter = 0.06; density = 1400.0; strength = 2.0
[[material]]
name = "sand";    color = [0.86, 0.79, 0.58]; jitter = 0.04; density = 1600.0; strength = 1.0
[[material]]
name = "wood";    color = [0.52, 0.37, 0.23]; jitter = 0.05; density = 700.0;  strength = 4.0
[[material]]
name = "leaves";  color = [0.28, 0.48, 0.22]; jitter = 0.10; density = 200.0;  strength = 0.5
[[material]]
name = "brick";   color = [0.68, 0.34, 0.28]; jitter = 0.05; density = 1900.0; strength = 6.0
[[material]]
name = "planks";  color = [0.62, 0.47, 0.30]; jitter = 0.04; density = 600.0;  strength = 4.0
```

(TOML note: each `[[material]]` entry on multiple lines in the real file — semicolons above are plan shorthand, expand to one key per line.)

**Step 4:** tests green; commit `feat(core): TOML material registry with typed validation errors`

### Task 5: vox-world — Chunk storage (Uniform/Dense)

**Files:** Create: `crates/vox-world/src/lib.rs`, `src/chunk.rs`

**Step 1 — failing tests:** new chunk is `Uniform(AIR)`; `set` of same value keeps it Uniform (no allocation — assert via `is_uniform()`); `set` of a different value promotes to Dense and `get` returns it everywhere correctly; setting every voxel of a Dense chunk to one value then calling `try_demote()` returns it to Uniform; out-of-bounds `get` in debug panics (`#[should_panic]` with `debug_assertions`).

**Step 2 — implementation:**

```rust
/// A voxel: material id. 0 = air.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub struct Voxel(pub u16);
pub const AIR: Voxel = Voxel(0);

pub enum ChunkStorage { Uniform(Voxel), Dense(Box<[Voxel; 32 * 32 * 32]>) }

pub struct Chunk { storage: ChunkStorage, solid_count: u32 }
```

`index(l: UVec3) = (l.y as usize) * 1024 + (l.z as usize) * 32 + l.x as usize` (y-major so horizontal slices are contiguous). `get(l) -> Voxel`, `set(l, v)` promotes Uniform→Dense on first differing write (fill array with the uniform value), maintains `solid_count` (registry-independent: `v.0 != 0`). `try_demote()` checks all-equal. Keep storage private.

**Step 3:** green; commit `feat(world): chunk with uniform/dense storage behind get-set`

### Task 6: vox-world — World map, edits, dirty tracking

**Files:** Create: `crates/vox-world/src/world.rs`

**Step 1 — failing tests:** set/get across a chunk border (voxel (31,5,5) and (32,5,5)) land in different chunks; `set_voxel` marks the containing chunk dirty; setting a voxel with `local.x == 0` also dirties the −X neighbor chunk (meshes sample the shell); `drain_dirty()` returns each chunk once and empties; `fill_box` fills an AABB of voxels across chunks; voxels outside `bounds_voxels` read as AIR and writes are ignored (logged once).

**Step 2 — implementation:** 

```rust
pub struct World {
    pub cfg: WorldConfig,
    chunks: HashMap<IVec3, Chunk>,           // streaming-ready: sparse map
    dirty: HashSet<IVec3>,                    // chunks needing remesh
    dirty_regions: Vec<(IVec3, IVec3)>,       // voxel AABBs edited this frame (physics wake)
    bounds_voxels: (IVec3, IVec3),            // finite MVP extent, min..max exclusive
}
```

`get_voxel(IVec3) -> Voxel` (absent chunk = AIR), `set_voxel(IVec3, Voxel)` (creates chunk on demand; dirty marks: own chunk + any face-neighbor chunk whose shell contains the voxel — check each axis local==0 → −axis neighbor, local==31 → +axis neighbor; corner/edge cases fall out of doing all three axes independently), `fill_box(min, max_excl, v)`, `drain_dirty() -> Vec<IVec3>`, `drain_dirty_regions()`, `solid(&self, v: IVec3) -> bool`. Also `insert_chunk(c: IVec3, chunk)` for generation (bulk, dirties chunk + all 26 neighbors... face neighbors suffice: 6).

**Step 3:** green; commit `feat(world): sparse chunk map with edit API and dirty tracking`

### Task 7: vox-world — DDA raycast

**Files:** Create: `crates/vox-world/src/raycast.rs`

**Step 1 — failing tests:** brute-force reference = step `0.01·voxel_size` along the ray sampling `solid()`; on 30 random small worlds (place ~40 random solid voxels in a 16³ area, random ray origins/dirs), DDA hit voxel == brute-force hit voxel and DDA `face` is the axis face crossed; axis-aligned ray down a voxel column hits top face (`face == IVec3::Y` when ray is −Y); ray starting inside a solid voxel returns `RayHit { voxel, face: None, dist_m: 0.0 }`; miss returns None within `max_dist_m`.

**Step 2 — implementation (Amanatides–Woo, copy carefully):**

```rust
pub struct RayHit { pub voxel: IVec3, pub face: Option<IVec3>, pub dist_m: f32 }

pub fn raycast(world: &World, origin_m: Vec3, dir: Vec3, max_dist_m: f32) -> Option<RayHit> {
    let s = world.cfg.voxel_size_m;
    let dir = dir.normalize();
    let p = origin_m / s;                       // ray in voxel space
    let mut cell = p.floor().as_ivec3();
    if world.solid(cell) {
        return Some(RayHit { voxel: cell, face: None, dist_m: 0.0 });
    }
    let step = IVec3::new(dir.x.signum() as i32, dir.y.signum() as i32, dir.z.signum() as i32);
    let inv = Vec3::new(1.0/dir.x, 1.0/dir.y, 1.0/dir.z); // inf on zero components is fine
    // distance (in voxel units along ray) to the first boundary per axis
    let mut t_max = Vec3::ZERO;
    let mut t_delta = Vec3::ZERO;
    for a in 0..3 {
        if dir[a] > 0.0      { t_max[a] = (cell[a] as f32 + 1.0 - p[a]) * inv[a]; }
        else if dir[a] < 0.0 { t_max[a] = (p[a] - cell[a] as f32) * -inv[a]; }
        else                 { t_max[a] = f32::INFINITY; }
        t_delta[a] = inv[a].abs();
    }
    let max_t = max_dist_m / s;
    loop {
        let a = if t_max.x < t_max.y {
            if t_max.x < t_max.z { 0 } else { 2 }
        } else if t_max.y < t_max.z { 1 } else { 2 };
        if t_max[a] > max_t { return None; }
        cell[a] += step[a];
        let t_enter = t_max[a];
        t_max[a] += t_delta[a];
        if world.solid(cell) {
            let mut face = IVec3::ZERO;
            face[a] = -step[a];
            return Some(RayHit { voxel: cell, face: Some(face), dist_m: t_enter * s });
        }
    }
}
```

**Step 3:** green (`cargo test -p vox-world`); commit `feat(world): DDA raycast verified against brute force`

### Task 8: vox-mesh — greedy mesher with vertex AO

The single mesher serves chunks **and** debris bodies, so it operates on an abstract sampler over an arbitrary region, not on `Chunk`.

**Files:** Create: `crates/vox-mesh/src/lib.rs`, `src/slab.rs`, `src/greedy.rs`

**Step 1 — `slab.rs`:** `VoxelSlab { min: IVec3, dims: IVec3, data: Vec<Voxel> }` — a copied region *including a 1-voxel shell* on all sides (so `dims = inner + 2`). `VoxelSlab::extract(world, inner_min, inner_dims)` copies from world (main thread, cheap memcpy-scale); `get(rel: IVec3) -> Voxel` where rel ∈ `[-1, inner+1)`. This is the thread-safety strategy: copy on main thread, mesh anywhere.

**Step 2 — failing tests (write all before implementing):**

```rust
fn slab_one(v: IVec3) -> VoxelSlab { /* 1³ inner slab with a stone voxel at v */ }

#[test] fn empty_slab_zero_quads() { ... assert_eq!(mesh.quads(), 0); }
#[test] fn single_voxel_six_quads() { ... assert_eq!(mesh.quads(), 6); }
#[test] fn two_same_material_merge() { /* 2x1x1 stone → 6 quads */ }
#[test] fn two_materials_do_not_merge() { /* stone+dirt in a row → 10 quads */ }
#[test] fn full_32_chunk_six_quads() { /* 32³ uniform stone, empty shell → exactly 6 quads */ }
#[test] fn watertight_random() {
    // 20 random slabs (16³, 30% fill): rasterize emitted quads back into a
    // face-set keyed (voxel, face); assert it equals the set of exposed faces
    // computed by brute force — each exposed face covered exactly once.
}
#[test] fn ao_corner_darkens() {
    // floor plane + one wall voxel: the two floor vertices touching the wall
    // have ao < 3, the far vertices have ao == 3.
}
```

**Step 3 — vertex format & mesh data:**

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelVertex {
    pub pos: [u8; 3],    // region-local voxel corner (0..=dims, dims ≤ 254)
    pub ao: u8,          // 0..=3
    pub normal: u8,      // 0..=5 (+X,-X,+Y,-Y,+Z,-Z)
    pub _pad: u8,
    pub material: u16,
}
pub struct MeshData { pub vertices: Vec<VoxelVertex>, pub indices: Vec<u32> }
```

**Step 4 — greedy algorithm** (per face-direction `d` of 6; axis `a = d/2`, sign `pos = d%2==0`; u,v = other two axes):

```text
for each slice s in 0..dims[a]:
  build mask[u][v]: Some(Cell{material, ao4}) if slab solid at (s) and empty at neighbor(d)
    ao4 = the four corner AO values for that face (see AO rule)
  greedy: scan mask row-major; at first unconsumed cell, grow w while mask equal,
    then grow h while entire row of width w equal; emit quad (s, u0, v0, w, h, cell);
    mark consumed. Cells are equal only if material AND all four AO corners match
    (prevents AO seams across merged quads).
emit quad → 4 vertices (corner positions in region-local voxel units), 6 indices;
  flip the diagonal when ao00 + ao11 < ao01 + ao10 (standard AO seam fix);
  wind CCW facing outward (check once on screen in Task 9; a winding bug shows
  as inside-out world with backface culling on).
```

**AO rule** for a face vertex: the three neighbors touching that corner *on the face's outer plane*: `side1`, `side2` (edge-adjacent), `corner`:

```rust
fn ao(side1: bool, side2: bool, corner: bool) -> u8 {
    if side1 && side2 { 0 } else { 3 - (side1 as u8 + side2 as u8 + corner as u8) }
}
```

**Step 5:** run tests → green. The watertight test is the load-bearing one; do not weaken it.

**Step 6: Commit** `feat(mesh): greedy mesher with AO over abstract slabs, watertight-verified`

### Task 9: vox-render — chunk pipeline, camera, culling; vox-app renders a flat world

**Files:**
- Create: `assets/shaders/voxel.wgsl`, `crates/vox-render/src/{camera.rs, pipeline.rs, meshstore.rs, frustum.rs}`
- Modify: `crates/vox-render/src/lib.rs`, `crates/vox-app/src/main.rs`

**Step 1 — `voxel.wgsl`** (one pipeline for chunks AND debris — per-draw model matrix):

```wgsl
struct Camera { view_proj: mat4x4f, cam_pos: vec4f, sun_dir: vec4f, fog: vec4f } // fog: (start, end, voxel_size, _)
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> palette: array<vec4f>; // rgb + jitter

struct Inst { @location(4) m0: vec4f, @location(5) m1: vec4f, @location(6) m2: vec4f, @location(7) m3: vec4f }
struct VIn { @location(0) pos_ao: vec4<u32>, @location(1) norm_mat: vec4<u32> }
// pos_ao = (x, y, z, ao); norm_mat = (normal_id, pad, mat_lo, mat_hi) — Uint8x4 ×2

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) color: vec3f,
    @location(1) @interpolate(flat) normal_id: u32,
    @location(2) ao: f32,
    @location(3) world_pos: vec3f,
}

const NORMALS = array<vec3f, 6>(
    vec3f(1,0,0), vec3f(-1,0,0), vec3f(0,1,0), vec3f(0,-1,0), vec3f(0,0,1), vec3f(0,0,-1));

@vertex fn vs(v: VIn, inst: Inst) -> VOut {
    let model = mat4x4f(inst.m0, inst.m1, inst.m2, inst.m3);
    let local = vec3f(f32(v.pos_ao.x), f32(v.pos_ao.y), f32(v.pos_ao.z)) * cam.fog.z;
    let wp = (model * vec4f(local, 1.0)).xyz;
    let mat_id = v.norm_mat.z | (v.norm_mat.w << 8u);
    let base = palette[mat_id];
    // deterministic per-quad jitter from position hash
    let h = fract(sin(dot(floor(wp / cam.fog.z), vec3f(12.9898, 78.233, 37.719))) * 43758.547);
    var out: VOut;
    out.clip = cam.view_proj * vec4f(wp, 1.0);
    out.color = base.rgb * (1.0 + (h - 0.5) * 2.0 * base.a);
    out.normal_id = v.norm_mat.x;
    out.ao = f32(v.pos_ao.w) / 3.0;
    out.world_pos = wp;
    return out;
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
    let n = NORMALS[in.normal_id];
    let sun = max(dot(n, -cam.sun_dir.xyz), 0.0);
    let hemi = 0.5 + 0.5 * n.y;
    let ao = 0.35 + 0.65 * in.ao;
    var c = in.color * (0.28 * hemi + 0.75 * sun) * ao;
    let dist = length(in.world_pos - cam.cam_pos.xyz);
    let f = clamp((dist - cam.fog.x) / (cam.fog.y - cam.fog.x), 0.0, 1.0);
    c = mix(c, vec3f(0.45, 0.66, 0.90), f * f);
    return vec4f(c, 1.0);
}
```

(Vertex buffer layout: 8-byte stride, two `Uint8x4` attributes at locations 0/1 — matches `VoxelVertex` exactly with material split into two bytes. Instance buffer: 64-byte mat4 as four `Float32x4`.)

**Step 2 — `camera.rs`:** fly camera (pos, yaw, pitch) → `view_proj` (perspective fovy 70°, near 0.05, far 600); WASD+Space/Shift move (speed 12 m/s, ×5 with Ctrl), mouse look while cursor grabbed. `frustum.rs`: extract 6 planes from view_proj (Gribb–Hartmann rows method), `aabb_visible(min, max) -> bool` (p-vertex test). **Headless test:** cube behind an identity-ish camera is culled, cube in front is visible; plane normals normalized.

**Step 3 — `meshstore.rs`:** `ChunkMeshStore`: per chunk key → `{vbuf, ibuf, index_count, aabb_m}`; `upload(key, MeshData, origin_m)` (create buffers via `device.create_buffer_init`), `remove(key)`, iterate visible with frustum. Palette storage buffer built once from `MaterialRegistry` (rgb + jitter as vec4).

**Step 4 — pipeline:** bind group 0 (camera uniform + palette storage), depth test LessEqual, cull back faces. Draw: for each visible chunk mesh: set instance buffer slice (one mat4 = translation of chunk origin in meters), `draw_indexed`.

**Step 5 — vox-app wiring:** build `World` (0.1 m scale, extent 64×16×64 m for now), `fill_box` a stone slab 3 m thick with grass on top; mesh every dirty chunk synchronously on startup (VoxelSlab::extract + mesh_region on rayon par_iter, collect, upload); render loop with fly camera.

**Run:** `cargo run -p vox-app --release` → flat grass world; fly around; AO visible at edges; fog at distance; no holes (watertightness is proven by test, eyeball confirms winding — if world looks inside-out, flip index order once).

**Step 6: Commit** `feat(render,app): chunk pipeline with AO/fog, fly camera, flat world on screen`

---

## Milestone M2 — Generation: Terrain & Trees

### Task 10: vox-gen — noise (hash, value, gradient, FBM)

**Files:** Create: `crates/vox-gen/src/{lib.rs, noise.rs}`

**Step 1 — failing tests:** determinism (same seed+coords → identical output across two runs and across grid re-order); range: 10k random samples of `fbm2` ∈ [-1, 1]; continuity: |f(p+1e-3) − f(p)| < 0.05 over 1k random p; seeds differ: seed 1 vs 2 fields differ at >90% of samples.

**Step 2 — implementation:** integer hash (triple32-style avalanche), lattice hashing `hash3(ix, iy, iz, seed) -> u32`:

```rust
#[inline]
fn avalanche(mut x: u32) -> u32 {
    x ^= x >> 17; x = x.wrapping_mul(0xed5ad4bb);
    x ^= x >> 11; x = x.wrapping_mul(0xac4c1b51);
    x ^= x >> 15; x = x.wrapping_mul(0x31848bab);
    x ^= x >> 14; x
}
fn hash2(ix: i32, iy: i32, seed: u32) -> u32 {
    avalanche((ix as u32).wrapping_mul(0x85297a4d)
        ^ (iy as u32).wrapping_mul(0x68e31da4) ^ seed)
}
```

`grad2(cell_hash) -> Vec2`: pick from 8 unit directions; Perlin-style gradient noise `gradient2(p: Vec2, seed) -> f32` with quintic fade `t³(t(6t−15)+10)`, output scaled ×1.4142 to ≈[-1,1]; `value3` for 3-D density (trees/caves later). `Fbm { octaves: u8, lacunarity: f32, gain: f32, seed: u32 }::sample2(p) -> f32` normalized by Σ amplitudes.

**Step 3:** green; commit `feat(gen): deterministic hash/gradient/value noise with FBM`

### Task 11: vox-gen — heightmap terrain, scale-invariance test

**Files:** Create: `crates/vox-gen/src/terrain.rs`

**Step 1 — failing tests:**
- `surface_height_in_bounds`: for 200 random columns, `height_m` ∈ [4, extent.y − 8].
- `grass_on_top`: after generating a small world, for sampled columns the top solid voxel is grass, with dirt below it and stone below ~1.5 m.
- **`scale_invariance` (THE contract test):** generate two worlds, same seed, extents 64×32×64 m, voxel 0.1 vs 1.0; for 50 sample (x,z) positions in meters, `|surface_m(w01) − surface_m(w10)| ≤ 2.0 * max_voxel` where `surface_m` finds top solid voxel and converts to meters. Trees disabled for this test (`TerrainGen::bare`).

**Step 2 — implementation:** `TerrainGen::new(cfg)` builds three FBM layers seeded from cfg.seed (`continents`: 5 octaves, wavelength 900 m, amp 22 m; `hills`: 4 oct, 160 m, 9 m; `rough`: 3 oct, 28 m, 2.2 m); `height_m(x_m, z_m) = base(=extent.y*0.45) + Σ layers` — **sampled in meters**, so scale falls out naturally. `generate(world: &mut World)`: per chunk column: compute min/max height over the chunk's (x,z) footprint (sample 4 corners + center, pad ±2 m); chunks entirely below → `insert_chunk(Uniform(stone))`; entirely above → skip; else per-column fill: stone up to `h−1.5 m`, dirt to `h−voxel`, grass top voxel. Log generation time via tracing.

**Step 3:** green including scale invariance; wire into vox-app (replace flat slab; extent from `WorldConfig::default()` — startup meshing goes through the same rayon path). **Run:** rolling terrain at 0.1 m. Commit `feat(gen): heightmap terrain with mechanical scale-invariance test`

### Task 12: vox-gen — trees (meter-parameterized, deterministic)

**Files:** Create: `crates/vox-gen/src/trees.rs`

**Step 1 — failing tests:** determinism (same seed → same wood-voxel set); meter-height contract: for one placed tree at 0.1 m and 1.0 m, wood column height in meters within 1.5 m of each other; leaves exist adjacent to branch tips; trees never overwrite stone (only air/leaves are replaceable by wood, only air by leaves).

**Step 2 — implementation:** placement — iterate 8 m grid cells over world footprint; `hash2(cell, seed^0x7ree)` → jitter (x,z) within cell + accept if `fbm_density(cell_center) > 0.15` (≈1 tree / 2–3 cells); tree origin = surface height at (x,z), skip if slope > 35° (sample 4 neighbors ±1 m) or surface isn't grass/dirt. Build (all meters, converted at stamp time):

```text
h_tree   = 6.0 + 4.0 * hash01(...)          // 6–10 m
r_base   = 0.30 + 0.15 * (h_tree/10)         // tapers to 0.10 m at top
trunk    : for y in 0..h_tree step voxel: disc(radius lerp(r_base, 0.10, y/h))
branches : n = 3 + hash % 3, at frac 0.55 + 0.12k of height:
           yaw = hash * τ, pitch 25–45° up, len 0.18*h_tree,
           voxel line (radius 0.08 m) + canopy at tip
canopies : ellipsoid rx=rz = 1.3–2.2 m, ry = 0.75*rx  (one per branch tip + crown)
           fill p where ((p-c)/r)² ≤ 1 and world voxel is AIR → leaves
```

`stamp_disc/stamp_ellipsoid/stamp_line` are small helpers over `world.set_voxel` (respect the replaceability rule). At 0.1 m a tree is ~80 voxels tall with visible branch structure; at 1.0 m it degrades gracefully to a chunky Minecraft tree (disc radius < voxel → single column; ellipsoid ~2 voxel blob).

**Step 3:** green; wire after terrain in vox-app. **Run:** forested rolling terrain at 0.1 m; also run once with `voxel_size_m = 1.0` hardcoded to eyeball the Minecraft look. Commit `feat(gen): deterministic meter-scale trees`

---

## Milestone M3 — Player & Tools

### Task 13: vox-physics — kinematic character controller

**Files:** Create: `crates/vox-physics/src/{lib.rs, character.rs}`

**Step 1 — failing tests (headless, build tiny worlds inline):**
- `rests_on_floor`: spawn 3 m above a slab, step 120 ticks → feet exactly on surface (|err| < 1e-3), `grounded == true`, vy == 0.
- `wall_blocks`: walking into a 2 m wall stops at wall minus skin; sliding along it preserves tangent motion.
- `steps_up`: a 0.4 m ledge is climbed while walking (0.55 step height); a 1.0 m ledge is not.
- `ceiling_bumps`: jumping under a low ceiling zeroes vy without sticking.
- `scale_agnostic`: `rests_on_floor` + `steps_up` pass at voxel 0.1 **and** 1.0 (0.4 m ledge = 4 voxels vs "less than one voxel" — at 1.0 m build the ledge 1 voxel = 1.0 m and assert it does NOT step; adjust the test's expectations per scale, documenting the semantic: step height is meters, world geometry is voxel-quantized).

**Step 2 — core sweep (copy exactly, it's the fiddly part):**

```rust
/// Move an AABB along one axis, clamped by solid voxels. Returns actual delta.
fn sweep_axis(world: &World, aabb: Aabb, axis: usize, delta: f32) -> f32 {
    if delta == 0.0 { return 0.0; }
    let s = world.cfg.voxel_size_m;
    const SKIN: f32 = 1e-3;
    let sign = delta.signum();
    // leading face position and its target
    let face = if sign > 0.0 { aabb.max[axis] } else { aabb.min[axis] };
    let target = face + delta;
    // voxel range swept by the leading face
    let (lo, hi) = if sign > 0.0 { (face, target) } else { (target, face) };
    let v_lo = (lo / s).floor() as i32;
    let v_hi = ((hi / s).ceil() as i32) - 1;
    // cross-section voxel range of the AABB on the other two axes
    let (u, w) = other_axes(axis);
    let u_range = axis_voxel_range(aabb, u, s); // floor(min/s) ..= ceil(max/s)-1 with SKIN inset
    let w_range = axis_voxel_range(aabb, w, s);
    let iter: Box<dyn Iterator<Item = i32>> = if sign > 0.0 {
        Box::new(v_lo..=v_hi) } else { Box::new((v_lo..=v_hi).rev()) };
    for slice in iter {
        for uu in u_range.clone() { for ww in w_range.clone() {
            let mut v = IVec3::ZERO; v[axis] = slice; v[u] = uu; v[w] = ww;
            if world.solid(v) {
                let plane = if sign > 0.0 { slice as f32 * s } else { (slice + 1) as f32 * s };
                let allowed = plane - face - sign * SKIN;
                // only clamp if the plane is actually in our path
                return if sign > 0.0 { allowed.clamp(0.0, delta) } else { allowed.clamp(delta, 0.0) };
            }
        }}
    }
    delta
}
```

`CharacterController { pos (feet center), vel, grounded, noclip }::step(world, input_move: Vec3, jump: bool, dt)`: gravity → vy; order Y then X then Z sweeps; grounded = (downward sweep clamped); jump sets `vy = (2.0 * GRAVITY * JUMP_HEIGHT).sqrt()` when grounded; step-up: if X or Z clamped and grounded → retry from `pos + (0, STEP_HEIGHT, 0)`: sweep up (may clamp), then horizontal, then sweep back down; accept iff extra horizontal distance gained and landing is grounded. Noclip skips everything. Walk speed 4.3 m/s, fly 12 m/s (app-side).

**Step 3:** green at both scales; commit `feat(physics): kinematic character controller with step-up, scale-agnostic`

### Task 14: threaded remeshing + tools (place/break) + walker wiring

**Files:**
- Create: `crates/vox-app/src/{remesh.rs, tools.rs, player.rs}`
- Modify: `crates/vox-app/src/main.rs`

**Step 1 — `remesh.rs` (the production remesh pipeline):** `RemeshQueue`: on `world.drain_dirty()` insert keys; each frame dispatch up to N(=64) jobs: sort pending by distance to camera, `VoxelSlab::extract` on main thread (cheap), `rayon::spawn` mesh_region, send `(key, MeshData)` through `std::sync::mpsc`; drain channel each frame → upload to `ChunkMeshStore` (replace or remove-if-empty). Coalescing rule: if a key is re-dirtied while in flight, mark it stale and re-dispatch on completion (guard against uploading stale meshes over fresh edits — keep a `generation: u64` per chunk key, jobs carry their generation, uploads with old generation are dropped and re-queued).

**Step 2 — `tools.rs`:** `Tool::{Place, Break, Blast}` on keys 1/2/3. Break (LMB): `raycast(eye, look, REACH)` → `set_voxel(hit.voxel, AIR)`. Place (RMB): hit.face → `set_voxel(hit.voxel + face, selected_material)` unless that voxel AABB intersects the player AABB. Material select: scroll wheel cycles registry (HUD text later; tracing::info now). Blast is Task 20 (stub logs).

**Step 3 — `player.rs`:** owns CharacterController + camera glue: camera pos = feet + `(0, PLAYER_EYE, 0)`, yaw/pitch from mouse; F toggles noclip/fly; physics-stepped in the fixed-timestep loop (`timing.physics_steps`), rendered from interpolated position (store prev/curr feet pos, lerp by alpha).

**Step 4 — wire loop order** in `main.rs`: input → (per physics step): player.step, tools.apply → remesh dispatch/drain → render. Spawn point: surface + 2 m at world center.

**Run:** walk/jump across terrain at 60 fps, break and place voxels with instant remesh (< 1 frame hitch), climb 0.4 m ledges, F to fly. Commit `feat(app): walking player, place/break tools, generation-guarded threaded remeshing`

---

## Milestone M4 — Rigidbody Physics

### Task 15: vox-physics — VoxelBody grids & mass properties

**Files:** Create: `crates/vox-physics/src/{body.rs, massprops.rs}`

**Step 1 — failing tests:**
- `box_inertia_analytic`: an 8×12×6 uniform grid (density ρ, voxel s): summed inertia == analytic solid box `I_x = M(b²+c²)/12` etc. to relative 1e-4 (the cube-decomposition is mathematically exact when each voxel contributes its own `(1/6)m s²` term — if this fails the per-voxel term is missing).
- `com_l_shape`: COM of an L of two boxes matches hand-computed value.
- `surface_points_box`: a 4³ solid grid yields 56 surface samples (all voxels except the 2³ interior).

**Step 2 — implementation:**

```rust
pub struct VoxelGrid { pub dims: IVec3, pub voxels: Vec<Voxel> }   // dense, index x + z*dx + y*dx*dz

pub struct MassProps { pub mass: f32, pub com_local: Vec3, pub inertia_com: Mat3 }

pub fn mass_props(grid: &VoxelGrid, reg: &MaterialRegistry, s: f32) -> MassProps {
    // per solid voxel: m = density * s³ at center p = (idx + 0.5) * s
    // pass 1: mass, com. pass 2 about com:
    //   I += m * ((r·r) * I3 - outer(r, r))          // parallel-axis point term
    //   I += Mat3::from_diagonal(Vec3::splat(m * s * s / 6.0))  // voxel's own cube inertia
}

pub struct Body {
    pub id: BodyId,
    pub pos: Vec3,          // world position of the COM (meters)
    pub rot: Quat,
    pub vel: Vec3, pub omega: Vec3,
    pub inv_mass: f32, pub inv_inertia_local: Mat3,
    pub grid: VoxelGrid, pub grid_offset: Vec3,   // grid min corner relative to COM, meters
    pub surface: Vec<Vec3>, // sample points, local to COM, at surface-voxel centers
    pub half_voxel: f32,    // 0.5 * s of the body's voxels (contact radius)
    pub sleep: SleepState,  pub aabb: Aabb,       // world AABB, refreshed each step
}
```

`surface_points`: voxels with ≥1 empty (or out-of-grid) face neighbor. `inv_inertia_world(&self) -> Mat3 = R * inv_I_local * Rᵀ`. `Bodies` container = `Vec<Body>` + free list (generational `BodyId(u32, u32)` — our typed arena).

**Step 3:** green; commit `feat(physics): voxel grids with exact mass properties and surface sampling`

### Task 16: vox-physics — integration, world contacts, impulse solver, sleeping

**Files:** Create: `crates/vox-physics/src/{contact.rs, solver.rs}`, modify `lib.rs` (PhysicsWorld facade)

**Step 1 — failing tests (headless; these are the make-or-break gates):**
- `cube_drop_settles`: 0.5 m wood cube dropped 3 m onto flat world: within 3 s sim (180 steps), body sleeps; rest height = half-extent ± (CONTACT_SLOP + half_voxel); no NaN anywhere (assert every step).
- `many_bodies_settle`: 20 random 2–6-voxel-extent bodies dropped from staggered heights: all asleep by 12 s, none below floor plane, total KE monotone-ish decreasing after last impact (allow 5% blips).
- `stack_five_sleeps`: five 0.4 m cubes stacked with 2 mm gaps: after 6 s all asleep, horizontal drift of top cube < 1 voxel.
- `edit_wakes`: sleeping body + `wake_region` overlapping its AABB → awake.

**Step 2 — contact generation (body vs world):** for each world-space surface sample `p_w = body.pos + body.rot * p_local`, with contact radius `r = body.half_voxel`:

```text
v = voxel_at(p_w);  s = world voxel size
if world.solid(v):
    among the 6 face neighbors of v that are EMPTY, pick the face plane nearest
    to p_w → normal n = that face's outward axis, depth = r + dist(p_w, plane)
    (all-solid fallback: n = +Y, depth = r + s/2)
else:
    for each of 3 axis face-planes of v within r of p_w: if that neighbor solid →
    contact with n pointing back into v's empty cell, depth = r − dist
Contact { p: p_w, n, depth, r_arm: p_w − body.pos, key: (body, v, face_id) }
```

**Step 3 — solver (sequential impulses; copy formulas exactly):**

```text
integrate (semi-implicit, per substep h = PHYSICS_DT / SUBSTEPS):
    vel += (0, -GRAVITY, 0) * h        (awake bodies only)
    generate contacts (rayon par_iter over awake bodies → Vec<Contact>)
    warm start: contact.acc_n/t1/t2 from persistent map by key; apply P = acc_n*n + acc_t1*t1 + acc_t2*t2
    velocity iterations (SOLVER_ITERS):
      for each contact:
        vn = dot(vel + omega × r_arm, n)
        bias = (CONTACT_BETA / h) * max(depth - CONTACT_SLOP, 0)
        kn = inv_mass + dot(n, (I_w⁻¹ (r_arm × n)) × r_arm)
        λ = (bias - vn) / kn
        new_acc = max(acc_n + λ, 0); applied = new_acc - acc_n; acc_n = new_acc
        apply P = applied * n:  vel += P * inv_mass; omega += I_w⁻¹ (r_arm × P)
        friction (t1, t2 ⟂ n):
          vt = dot(v_point, t);  kt as kn with t
          λt = -vt / kt; clamp acc_t to [-μ*acc_n, μ*acc_n]; apply delta
    integrate positions: pos += vel * h; rot = (rot + 0.5*h*Quat(omega,0)*rot).normalize()
    store contact accumulators back to persistent map (evict stale keys each full step)
    sleep bookkeeping per body: quiet = |vel| < SLEEP_LIN && |omega| < SLEEP_ANG;
      counter = quiet ? counter+1 : 0; counter > SLEEP_FRAMES → sleep (zero velocities)
    restitution: none (e = 0) — debris/rubble reads correctly without it; revisit post-MVP
```

`PhysicsWorld { bodies, contacts_cache, awake list }` facade: `step(world: &World, dt)`, `spawn(grid, transform, vel) -> BodyId`, `wake_region(aabb)` (called from world edit dirty-regions), `interpolated_transform(id, alpha)` (prev/curr snapshots).

**Step 4:** run the gate tests until genuinely green (expect to iterate on bias/slop if jitter: the documented knobs are BETA down to 0.1 or iterations up to 12; do not touch formulas). Commit `feat(physics): sequential-impulse solver with warm starting and sleeping`

### Task 17: vox-physics — body-vs-body contacts + broadphase

**Files:** Create: `crates/vox-physics/src/broadphase.rs`, extend `contact.rs`

**Step 1 — failing tests:** `two_cubes_stack`: drop one cube onto a resting cube → both sleep, top rests at combined height ± slop·2; `pile_no_explosion`: 12 cubes dropped into a 1.5 m pit (world walls) → all sleep, max |vel| ever < 25 m/s (no solver explosion).

**Step 2 — broadphase:** uniform spatial hash, cell 1.5 m: insert awake body AABBs (+ sleeping bodies touched by awake AABBs — pairs with two sleeping bodies are skipped); candidate pairs deduped via sorted `(min_id, max_id)` HashSet.

**Step 3 — narrowphase pair:** pick body with FEWER surface points as the *sampler*; transform each sample into the other body's grid space (`local = rot_b⁻¹ * (p_w − pos_b) − grid_offset_b`, then `/ s_b`); occupancy + face logic identical to world contacts but normals rotated back to world (`n_w = rot_b * n_local`). Contact solves between two dynamic bodies: `kn = inv_mass_a + inv_mass_b + dot(n, (Ia⁻¹(ra×n))×ra) + dot(n, (Ib⁻¹(rb×n))×rb)`, impulses applied equal-opposite. Waking: contact with an awake body resets both sleep counters *only if* the applied normal impulse > 0.5 · mass_smaller · SLEEP_LIN (prevents pile insomnia).

**Step 4:** green; commit `feat(physics): body-body collision via grid sampling with spatial-hash broadphase`

### Task 18: debris rendering + interpolation + spawn-box debug key

**Files:** Modify: `crates/vox-render/src/meshstore.rs` (add `BodyMeshStore`), `crates/vox-app/src/main.rs`

**Step 1:** mesh a `VoxelGrid` with the SAME `mesh_region` (a `VoxelSlab` adapter around the grid with empty shell — write `VoxelSlab::from_grid(&VoxelGrid)`); upload once at spawn; per-frame instance mat4 = `translation(pos + rot * grid_offset_render) * rotation(rot) * ...` — careful: mesh vertices are in grid units from grid min corner, so model = `Mat4::from_rotation_translation(rot, pos) * Mat4::from_translation(grid_offset)` and the shader's `* voxel_size` handles scaling (pass the BODY's voxel size — same world scale in MVP, so the shared `fog.z` works).

**Step 2:** app: fixed-step physics in the loop (after tools); render bodies from `interpolated_transform(alpha)`; **B key** spawns a 4×4×4 wood body at `eye + look*4 m` with `vel = look * 8`. HUD (tracing, once per second): awake/asleep counts, step time.

**Run:** spam B — boxes fly, tumble, collide, stack, and visibly go to sleep (stop jittering); framerate solid with 100+ bodies (most asleep). Commit `feat(render,app): debris bodies rendered with interpolation; spawn debug tool`

---

## Milestone M5 — Destruction

### Task 19: carve + connectivity + detach-to-body

**Files:** Create: `crates/vox-physics/src/destruction.rs`

**Step 1 — failing tests (headless, the signature Teardown behaviors):**
- `two_pillars_cut_one_nothing_falls`: slab on two pillars (built with `fill_box`); carve one pillar → `detached.is_empty()` (slab still anchored via the other).
- `cut_both_slab_detaches`: carve second pillar → exactly one detached component whose voxel count == slab volume ± carve overlap; spawned body mass == Σ voxel masses.
- `floating_blob_detaches`: sphere-carve that isolates a corner knob → knob detaches.
- `small_fragments_discarded`: components < DEBRIS_MIN_VOXELS are removed from world but produce no body.
- `carve_records_removed`: returned removed list matches sphere ∩ solid.

**Step 2 — carve:**

```rust
pub struct CarveResult { pub removed: Vec<(IVec3, Voxel)>, pub region: (IVec3, IVec3) }
pub fn carve_sphere(world: &mut World, center_m: Vec3, radius_m: f32) -> CarveResult
// voxel-space sphere test on voxel centers; region = AABB of removed padded by 2 voxels
```

**Step 3 — connectivity (flood from anchors):**

```text
within region R (clamped to world bounds):
  anchors = every solid voxel ON the boundary shell of R          // connects to the outside world
          ∪ every solid voxel at world floor level inside R
  BFS 6-connected over solid voxels from all anchors (bitset visited over R)
  unvisited solid voxels = unsupported → group into components (second BFS)
  per component: if len < DEBRIS_MIN_VOXELS → set AIR, drop
                 if len > MAX_BODY_VOXELS  → leave in world (safety valve), skip
                 else: min-AABB → VoxelGrid copy → set AIR in world →
                       spawn Body at COM world position (identity rotation,
                       grid_offset = grid_min_m − com_m)
```

**Step 4 — blast impulse:** for each spawned body: `vel = dir_from_center * (blast_power / max(dist_m, 0.5)) / mass.sqrt()` with `blast_power = 40.0` (tunable), `omega` = small hash-based random (±3 rad/s). Also `wake_region(R)`.

**Step 5:** green; commit `feat(physics): carve → connectivity flood → debris body pipeline`

### Task 20: blast tool + world-edit waking + the money shot

**Files:** Modify: `crates/vox-app/src/tools.rs`, `main.rs`

**Step 1:** Tool 3 (Blast, LMB): raycast → `carve_sphere(hit_point_m, blast_radius)` → destruction pipeline → debris spawns with impulse; `[`/`]` adjust radius 0.5–4 m. Break tool (single voxel) ALSO runs the connectivity pass on its small region (knocking out a lone support beam should drop things — same code path, radius = 1 voxel).

**Step 2:** wire `world.drain_dirty_regions()` → `physics.wake_region` every frame (debris resting on ground you just carved must wake). Debris bodies live until despawn key (X clears asleep debris — dev convenience).

**Step 3 — Verify the experience (the acceptance test of the whole engine):** run at 0.1 m; build a brick tower with place tool (or find trees); blast the trunk/base → upper section detaches as ONE body, falls, tumbles, settles, sleeps; repeated blasts break debris further? (No — debris re-fracture is post-MVP; blast only affects world voxels. Note this in README.) Framerate ≥ 60 during a 10-blast rampage.

**Step 4:** Commit `feat(app): blast tool — carve, detach, debris with impulse`

---

## Milestone M6 — Polish, Scale Demo, Docs

### Task 21: vox-debug — egui overlay (HUD, timings, tuning)

**Files:** Create: `crates/vox-debug/src/lib.rs`; modify vox-render (expose device/queue/format for egui-wgpu), vox-app

**Step 1:** our profiler: `ScopedTimer` → `FrameProfile { ring: [FrameSample; 240] }` per label (input/player/tools/physics/remesh/render), recorded in vox-app (vox-core hosts the tiny timer type so lower crates can be timed later).

**Step 2:** `DebugOverlay::new(device, format, window)` (egui 0.28 + egui-winit 0.29 + egui-wgpu 0.28 as in the archived project's combination — versions align); F3 toggles. Panels: FPS + frame graph (per-label ms, egui plot lines from ring), counts (chunks drawn/culled, mesh queue depth, bodies awake/asleep, contacts), sliders writing straight to a `Tunables` struct shared with app: blast radius/power, friction, CONTACT_BETA, sleep thresholds, fly speed; material picker (registry names) for place tool; crosshair dot painted center-screen.

**Step 3:** Run — toggle overlay, drag friction to 0 and watch debris skate (sliders demonstrably live). Commit `feat(debug): egui overlay with profiler, counters, live physics tuning`

### Task 22: CLI world config + full-loop scale validation

**Files:** Create: `crates/vox-app/src/args.rs`; modify `main.rs`

**Step 1:** hand-rolled arg parse (`--scale 0.1 | 1.0`, `--seed N`, `--extent X,Y,Z` meters, `--help`): loop over `std::env::args` pairs → `WorldConfig`; invalid input prints usage and exits 1 (no clap — trivial parser, tested with 4 unit tests).

**Step 2 — the scale acceptance pass:** `cargo run -p vox-app --release -- --scale 1.0 --seed 42` AND `--scale 0.1 --seed 42`: same macro terrain, same forest placement; walk, jump, step-up correct in both; trees chunky vs detailed; blast + debris behave (1 m debris = big blocky slabs, correct because mass/inertia derive from scale). Fix any surfaced scale bug AT THE SOURCE (a meters-vs-voxels confusion), never with a scale-conditional patch — then add a regression test in the offending crate.

**Step 3:** full gates: `cargo test` green, `cargo clippy --all-targets -- -D warnings` clean. Commit `feat(app): world config CLI; scale-switch validated end to end`

### Task 23: README + extension guide + tag

**Files:** Create: `README.md`

**Step 1:** README: what it is, the crate diagram (from design doc), controls table (WASD/mouse/1-2-3 tools/B spawn/F fly/F3 debug/X clear debris/[ ] radius), quickstart commands, screenshots-optional. **Extension guide** (the modularity payoff, be concrete): how to add a material (TOML only), a tool (tools.rs enum + one match arm), a new engine system (new sibling crate slotting at the gen/physics tier — cite where CA sim and ecosystem crates will mount per design doc §12), and the dependency policy. Post-MVP roadmap list (design doc §12) + explicit note: wgpu/winit/egui version modernization is a contained follow-up in vox-render/vox-platform/vox-debug.

**Step 2:** `cargo test` one last time; commit `docs: README with architecture, controls, extension guide`; `git tag v0.1.0-mvp`.

---

## Post-MVP parking lot (do NOT build now)

Streaming chunk load/unload · palette-compressed chunks + RLE saves · debris re-voxelization on settle · debris re-fracture · structural stress · CA sim crate (`vox-sim`) · ecosystem crate · shadowmaps/raytraced path · dependency modernization (wgpu 26+, winit 0.30, egui latest) · gamepad input · audio.
