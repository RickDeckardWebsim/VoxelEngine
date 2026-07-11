# Seamless World Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Replace upfront full-world terrain generation with lazy per-chunk streaming around the player, with configurable quality presets and in-memory edit retention.

**Architecture:** A `ChunkLoader` in `vox-app` generates chunks on demand within the world's finite extent, mirroring the existing `SurfaceProvider` idiom. `World` gains a clip guard for tree stamping, an edited-chunk set for eviction, and a suppress flag for generation-time writes. Quality presets control render distance + tree detail ring.

**Tech Stack:** Rust, wgpu, glam, rayon, egui. Existing crates: vox-world, vox-gen, vox-app, vox-core, vox-debug, vox-render.

**Design doc:** `docs/plans/2026-07-10-seamless-world-generation-design.md`

---

### Task 1: World — Clip Guard

**Files:**
- Modify: `crates/vox-world/src/world.rs`
- Test: `crates/vox-world/src/world.rs` (inline `#[cfg(test)] mod tests`)

**Step 1: Write the failing test**

Add to the `tests` module in `world.rs`:

```rust
#[test]
fn clip_blocks_writes_outside_region() {
    let mut w = world();
    w.set_clip(IVec3::new(0, 0, 0), IVec3::new(32, 32, 32));

    // Inside clip region — write succeeds.
    w.set_voxel(IVec3::new(10, 10, 10), STONE);
    assert_eq!(w.get_voxel(IVec3::new(10, 10, 10)), STONE);

    // Outside clip region — write silently dropped.
    w.set_voxel(IVec3::new(40, 10, 10), STONE);
    assert_eq!(w.get_voxel(IVec3::new(40, 10, 10)), AIR);
    assert!(!w.chunks().any(|(k, _)| *k == IVec3::new(1, 0, 0)),
        "clip must not allocate chunk outside region");

    // After clearing clip — writes work again.
    w.clear_clip();
    w.set_voxel(IVec3::new(40, 10, 10), STONE);
    assert_eq!(w.get_voxel(IVec3::new(40, 10, 10)), STONE);
}

#[test]
fn clip_does_not_block_reads() {
    let mut w = world();
    w.set_voxel(IVec3::new(40, 10, 10), STONE);
    w.set_clip(IVec3::new(0, 0, 0), IVec3::new(32, 32, 32));

    // Read outside clip returns the actual voxel, not air.
    assert_eq!(w.get_voxel(IVec3::new(40, 10, 10)), STONE);
    assert!(w.solid(IVec3::new(40, 10, 10)));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p vox-world -- clip_ --nocapture`
Expected: FAIL — `set_clip` / `clear_clip` methods don't exist

**Step 3: Write minimal implementation**

Add field to `World` struct (after `solid_table`):

```rust
/// When set, `set_voxel` silently drops writes outside this half-open
/// voxel box. Used during chunk generation to prevent tree stamping from
/// allocating neighbor chunks. `get_voxel` is unaffected — reads always
/// pass through. `None` = no clip (normal gameplay).
clip: Option<(IVec3, IVec3)>,
```

Initialize in `World::new`:
```rust
clip: None,
```

Add methods after `set_solid_table`:

```rust
/// Set a clip region: subsequent `set_voxel`/`edit_box` writes outside
/// `[min, max)` are silently dropped. Used during chunk generation so tree
/// stamping only writes into the target chunk. Reads are never clipped.
pub fn set_clip(&mut self, min: IVec3, max: IVec3) {
    self.clip = Some((min, max));
}

/// Remove the clip region — all in-bounds writes succeed normally.
pub fn clear_clip(&mut self) {
    self.clip = None;
}
```

Add clip guard at the top of `set_voxel`, after the `in_bounds` check:

```rust
if let Some((clip_min, clip_max)) = self.clip {
    if pos.cmplt(clip_min).any() || pos.cmpge(clip_max).any() {
        return;
    }
}
```

Add clip guard in `edit_box`, after the bounds clip and before the chunk loop:

```rust
if let Some((clip_min, clip_max)) = self.clip {
    min = min.max(clip_min);
    max = max.min(clip_max);
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p vox-world -- clip_ --nocapture`
Expected: PASS

**Step 5: Run full test suite to verify no regressions**

Run: `cargo test -p vox-world --nocapture`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add crates/vox-world/src/world.rs
git commit -m "feat(world): add clip guard for chunk generation tree stamping"
```

---

### Task 2: World — Edit Tracking + Suppress Flag

**Files:**
- Modify: `crates/vox-world/src/world.rs`
- Test: `crates/vox-world/src/world.rs` (inline tests)

**Step 1: Write the failing test**

```rust
#[test]
fn gameplay_edit_marks_chunk_edited() {
    let mut w = world();
    w.set_voxel(IVec3::new(10, 10, 10), STONE);
    assert!(w.is_edited(IVec3::new(0, 0, 0)),
        "gameplay write must mark chunk edited");
}

#[test]
fn suppressed_edit_does_not_mark_edited() {
    let mut w = world();
    w.set_suppress_edit_tracking(true);
    w.set_voxel(IVec3::new(10, 10, 10), STONE);
    assert!(!w.is_edited(IVec3::new(0, 0, 0)),
        "suppressed write must not mark chunk edited");
    // Voxel was still written.
    assert_eq!(w.get_voxel(IVec3::new(10, 10, 10)), STONE);
    w.set_suppress_edit_tracking(false);
}

#[test]
fn insert_chunk_never_marks_edited() {
    let mut w = world();
    w.insert_chunk(IVec3::new(1, 1, 1), Chunk::uniform(STONE));
    assert!(!w.is_edited(IVec3::new(1, 1, 1)),
        "insert_chunk is generation, never marks edited");
}

#[test]
fn edit_box_marks_all_touched_chunks_edited() {
    let mut w = world();
    // Box straddles chunk corner at (32, 32, 32).
    w.fill_box(IVec3::new(30, 30, 30), IVec3::new(35, 35, 35), STONE);
    for key in [
        IVec3::new(0, 0, 0),
        IVec3::new(1, 0, 0),
        IVec3::new(0, 1, 0),
        IVec3::new(0, 0, 1),
        IVec3::new(1, 1, 0),
        IVec3::new(1, 0, 1),
        IVec3::new(0, 1, 1),
        IVec3::new(1, 1, 1),
    ] {
        assert!(w.is_edited(key), "chunk {key} not marked edited");
    }
}

#[test]
fn same_value_write_does_not_mark_edited() {
    let mut w = world();
    w.set_voxel(IVec3::new(10, 10, 10), STONE);
    w.set_suppress_edit_tracking(true);
    // Re-set same value with suppression so we don't interfere.
    w.set_suppress_edit_tracking(false);
    // Now set same value again — no-op, should not newly mark.
    // Already edited from first write, so check a fresh chunk.
    w.set_voxel(IVec3::new(60, 60, 60), STONE);
    w.set_voxel(IVec3::new(60, 60, 60), STONE);
    // Chunk (1,1,1) is already edited from cross_chunk test if shared...
    // Actually, just verify the no-op doesn't create new dirty or edited.
    // Use a clean world for isolation:
    let mut w2 = world();
    w2.set_voxel(IVec3::new(10, 10, 10), STONE);
    let edited_before = w2.is_edited(IVec3::new(0, 0, 0));
    w2.set_voxel(IVec3::new(10, 10, 10), STONE); // same value
    assert!(edited_before); // was edited
    // No new chunks marked:
    assert_eq!(w2.edited_count(), 1, "same-value write should not add edited chunks");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p vox-world -- edit_marks\|suppress\|insert_chunk_never\|edit_box_marks\|same_value_write_does_not_mark --nocapture`
Expected: FAIL — `is_edited`, `set_suppress_edit_tracking`, `edited_count` don't exist

**Step 3: Write minimal implementation**

Add fields to `World` struct (after `clip`):

```rust
/// Chunks that have received gameplay writes (player tools, bombs, fire,
/// weathering). Pristine chunks (not in this set) can be evicted and
/// regenerated; edited chunks must persist in memory.
edited: FxHashSet<IVec3>,
/// When true, `set_voxel`/`edit_box` skip marking chunks as edited. Set
/// during chunk generation (terrain insert + tree stamping) so generated
/// chunks remain pristine and evictable. Gameplay writes always have this
/// false.
suppress_edit_tracking: bool,
```

Initialize in `World::new`:
```rust
edited: FxHashSet::default(),
suppress_edit_tracking: false,
```

Add methods:

```rust
/// True if chunk `key` has received gameplay writes (not just generation).
pub fn is_edited(&self, key: IVec3) -> bool {
    self.edited.contains(&key)
}

/// Number of chunks marked as edited.
pub fn edited_count(&self) -> usize {
    self.edited.len()
}

/// Set whether gameplay writes mark chunks as edited. During chunk
/// generation, set to `true` so generated terrain+trees stay pristine.
pub fn set_suppress_edit_tracking(&mut self, suppress: bool) {
    self.suppress_edit_tracking = suppress;
}
```

In `set_voxel`, after the actual write succeeds (after `chunk.set(local, v)` in the Occupied branch, and after `e.insert(...).set(local, v)` in the Vacant branch), but before `mark_dirty_with_neighbors`:

```rust
if !self.suppress_edit_tracking {
    self.edited.insert(key);
}
```

In `edit_box`, after `chunk.set(local, v)` for each change, inside the loop that applies changes:

```rust
if !self.suppress_edit_tracking {
    self.edited.insert(key);
}
```

Actually, more efficient: in `edit_box`, after the chunk loop, mark the chunk edited once if any changes were made:

```rust
if !changes.is_empty() && !self.suppress_edit_tracking {
    self.edited.insert(key);
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p vox-world -- edit_marks\|suppress\|insert_chunk_never\|edit_box_marks\|same_value_write_does_not_mark --nocapture`
Expected: PASS

**Step 5: Run full test suite**

Run: `cargo test -p vox-world --nocapture`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add crates/vox-world/src/world.rs
git commit -m "feat(world): add edit tracking with generation-time suppress flag"
```

---

### Task 3: World — Chunk Removal

**Files:**
- Modify: `crates/vox-world/src/world.rs`
- Test: `crates/vox-world/src/world.rs` (inline tests)

**Step 1: Write the failing test**

```rust
#[test]
fn remove_chunk_evicts_and_dirties_neighbors() {
    let mut w = world();
    w.insert_chunk(IVec3::new(1, 1, 1), Chunk::uniform(STONE));
    w.drain_dirty(); // clear insert dirty

    w.remove_chunk(IVec3::new(1, 1, 1));
    assert!(w.chunk_at(IVec3::new(1, 1, 1)).is_none(),
        "chunk must be removed");
    // Neighbors dirtied so their meshes update (face culling changes).
    let dirty: HashSet<_> = w.drain_dirty().into_iter().collect();
    for key in [
        IVec3::new(1, 1, 1),
        IVec3::new(2, 1, 1),
        IVec3::new(0, 1, 1),
        IVec3::new(1, 2, 1),
        IVec3::new(1, 0, 1),
        IVec3::new(1, 1, 2),
        IVec3::new(1, 1, 0),
    ] {
        assert!(dirty.contains(&key), "neighbor {key} not dirtied by removal");
    }
}

#[test]
fn remove_absent_chunk_is_noop() {
    let mut w = world();
    w.remove_chunk(IVec3::new(5, 5, 5));
    assert!(w.drain_dirty().is_empty(), "removing absent chunk dirties nothing");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p vox-world -- remove_chunk --nocapture`
Expected: FAIL — `remove_chunk` method doesn't exist

**Step 3: Write minimal implementation**

Add method to `World`, after `insert_chunk`:

```rust
/// Remove a chunk from the map, dirtying its six face neighbors (their
/// meshes sample this chunk for face culling). No-op if the chunk is
/// absent. Also removes it from the edited set — an evicted chunk is gone.
pub fn remove_chunk(&mut self, key: IVec3) {
    if self.chunks.remove(&key).is_none() {
        return;
    }
    self.edited.remove(&key);
    // Dirty neighbors so their meshes update (face culling changes when
    // an adjacent chunk disappears).
    for axis in 0..3 {
        for sign in [-1, 1] {
            let mut n = key;
            n[axis] += sign;
            self.dirty.insert(n);
        }
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p vox-world -- remove_chunk --nocapture`
Expected: PASS

**Step 5: Run full test suite**

Run: `cargo test -p vox-world --nocapture`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add crates/vox-world/src/world.rs
git commit -m "feat(world): add remove_chunk for eviction with neighbor dirtying"
```

---

### Task 4: Per-Chunk Tree Planning

**Files:**
- Modify: `crates/vox-gen/src/trees.rs`
- Test: `crates/vox-gen/src/trees.rs` (inline tests)

**Step 1: Write the failing test**

Add to `tests` module:

```rust
#[test]
fn trees_for_chunk_matches_global_plan_subset() {
    let cfg = cfg(0.1);
    let terrain = TerrainGen::new(&cfg);
    let global = plan_trees(&cfg, &terrain);

    // Pick a chunk that contains at least one tree.
    let s = cfg.voxel_size_m;
    let chunk_m = CHUNK_SIZE as f32 * s;
    let target_key = IVec3::new(2, 0, 2);
    let origin_m = target_key.as_vec3() * chunk_m;

    let per_chunk = trees_for_chunk(&cfg, &terrain, target_key);
    let global_in_chunk: Vec<&TreeInstance> = global.iter().filter(|t| {
        // Tree overlaps this chunk if its trunk position is within the
        // chunk's XY bounds (simplified — canopy reach is handled by
        // the clip mechanism at stamp time).
        let tx_vox = (t.x_m / s) as i32;
        let tz_vox = (t.z_m / s) as i32;
        let ox = target_key.x * CHUNK_SIZE as i32;
        let oz = target_key.z * CHUNK_SIZE as i32;
        tx_vox >= ox && tx_vox < ox + CHUNK_SIZE as i32 &&
        tz_vox >= oz && tz_vox < oz + CHUNK_SIZE as i32
    }).collect();

    assert_eq!(per_chunk.len(), global_in_chunk.len(),
        "per-chunk planning must match global plan for trunk-in-chunk trees");
    for (pc, g) in per_chunk.iter().zip(global_in_chunk.iter()) {
        assert_eq!(pc, g, "tree instances must match");
    }
}

#[test]
fn trees_for_chunk_is_deterministic() {
    let cfg = cfg(0.1);
    let terrain = TerrainGen::new(&cfg);
    let key = IVec3::new(3, 0, 3);
    let a = trees_for_chunk(&cfg, &terrain, key);
    let b = trees_for_chunk(&cfg, &terrain, key);
    assert_eq!(a, b, "same inputs must produce same trees");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p vox-gen -- trees_for_chunk --nocapture`
Expected: FAIL — `trees_for_chunk` doesn't exist. Need to import `CHUNK_SIZE`.

**Step 3: Write minimal implementation**

Add `CHUNK_SIZE` to the import at the top of `trees.rs`:

```rust
use vox_core::consts::CHUNK_SIZE;
```

Add function after `plan_trees`:

```rust
/// Trees whose trunk falls within chunk `key`, planned deterministically
/// from seed + terrain. This is a per-chunk slice of the same plan
/// `plan_trees` produces globally — same seed, same logic, filtered to
/// this chunk's XY bounds. The loader calls this during chunk generation
/// to stamp trees rooted in the target chunk; canopy that extends into
/// neighbor chunks is clipped by `World::set_clip` at stamp time.
///
/// **Quality rule:** Only call this for chunks within the detail ring
/// (near chunks). Far chunks skip tree rooting but still receive canopy
/// stamps from near-rooted trees.
pub fn trees_for_chunk(
    cfg: &WorldConfig,
    terrain: &TerrainGen,
    key: IVec3,
) -> Vec<TreeInstance> {
    let s = cfg.voxel_size_m;
    let chunk_m = CHUNK_SIZE as f32 * s;
    let origin_x_m = key.x as f32 * chunk_m;
    let origin_z_m = key.z as f32 * chunk_m;

    // Determine which placement cells overlap this chunk.
    let cell_min_ci = (origin_x_m / CELL_M).floor() as i32;
    let cell_max_ci = ((origin_x_m + chunk_m) / CELL_M).ceil() as i32;
    let cell_min_cj = (origin_z_m / CELL_M).floor() as i32;
    let cell_max_cj = ((origin_z_m + chunk_m) / CELL_M).ceil() as i32;

    let seed = (cfg.seed as u32) ^ ((cfg.seed >> 32) as u32) ^ 0x7EE5;
    let density = Fbm::new(3, seed ^ 0xD375);
    let mut trees = Vec::new();

    for cj in cell_min_cj..=cell_max_cj {
        for ci in cell_min_ci..=cell_max_ci {
            let h = hash2(ci, cj, seed);
            let x = ci as f32 * CELL_M + 1.0 + unit(h) * (CELL_M - 2.0);
            let z = cj as f32 * CELL_M + 1.0 + unit(h.rotate_left(11)) * (CELL_M - 2.0);

            // Trunk must fall within this chunk's XY voxel bounds.
            let tx_vox = (x / s) as i32;
            let tz_vox = (z / s) as i32;
            let ox = key.x * CHUNK_SIZE as i32;
            let oz = key.z * CHUNK_SIZE as i32;
            if tx_vox < ox || tx_vox >= ox + CHUNK_SIZE as i32 {
                continue;
            }
            if tz_vox < oz || tz_vox >= oz + CHUNK_SIZE as i32 {
                continue;
            }

            // Same filters as plan_trees.
            if x < 2.0 || z < 2.0 || x > cfg.extent_m[0] - 2.0 || z > cfg.extent_m[2] - 2.0 {
                continue;
            }
            if density.sample2(Vec2::new(x, z) / 60.0) < DENSITY_THRESHOLD {
                continue;
            }
            let base = terrain.height_m(x, z);
            let slope_x = (terrain.height_m(x + 1.0, z) - terrain.height_m(x - 1.0, z)).abs() / 2.0;
            let slope_z = (terrain.height_m(x, z + 1.0) - terrain.height_m(x, z - 1.0)).abs() / 2.0;
            if slope_x.max(slope_z) > MAX_SLOPE {
                continue;
            }
            let height = 6.0 + 4.0 * unit(h.rotate_left(19));
            if base + height + 3.0 > cfg.extent_m[1] {
                continue;
            }
            trees.push(TreeInstance {
                x_m: x,
                z_m: z,
                base_y_m: base,
                height_m: height,
                tree_seed: h,
            });
        }
    }
    trees
}
```

Export from `lib.rs`:
```rust
pub use trees::{TreeInstance, TreeMaterials, generate_trees, plan_trees, stamp_tree, trees_for_chunk};
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p vox-gen -- trees_for_chunk --nocapture`
Expected: PASS

**Step 5: Run full test suite**

Run: `cargo test -p vox-gen --nocapture`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add crates/vox-gen/src/trees.rs crates/vox-gen/src/lib.rs
git commit -m "feat(gen): add trees_for_chunk for per-chunk deterministic tree planning"
```

---

### Task 5: Quality Presets

**Files:**
- Modify: `crates/vox-app/src/args.rs`
- Modify: `crates/vox-app/src/main.rs` (to pass quality through)
- Test: `crates/vox-app/src/args.rs` (inline tests)

**Step 1: Write the failing test**

Add to `args.rs` tests module:

```rust
#[test]
fn quality_parses_low() {
    let cli = parse(["--quality", "low"].into_iter()).unwrap();
    assert_eq!(cli.quality, Quality::Low);
}

#[test]
fn quality_parses_medium_default() {
    let cli = parse([].into_iter()).unwrap();
    assert_eq!(cli.quality, Quality::Medium);
}

#[test]
fn quality_parses_ultra() {
    let cli = parse(["--quality", "ultra"].into_iter()).unwrap();
    assert_eq!(cli.quality, Quality::Ultra);
}

#[test]
fn quality_rejects_unknown() {
    assert!(parse(["--quality", "turbo"].into_iter()).is_err());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p vox-app -- quality_ --nocapture`
Expected: FAIL — `Quality` type and `quality` field don't exist

**Step 3: Write minimal implementation**

Add `Quality` enum to `args.rs`:

```rust
/// Streaming quality preset: controls render distance, tree detail ring,
/// and chunk generation budget per frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    Low,
    Medium,
    High,
    Ultra,
}

impl Quality {
    /// Render distance in chunks (radius around player).
    pub fn render_distance(self) -> i32 {
        match self {
            Quality::Low => 4,
            Quality::Medium => 8,
            Quality::High => 16,
            Quality::Ultra => 24,
        }
    }

    /// Detail ring radius in chunks. Trees root only within this ring;
    /// canopies extend beyond it into far chunks.
    pub fn detail_ring(self) -> i32 {
        match self {
            Quality::Low => 1,
            Quality::Medium => 3,
            Quality::High => 6,
            Quality::Ultra => 12,
        }
    }

    /// Maximum chunks to generate per frame.
    pub fn gen_budget(self) -> usize {
        match self {
            Quality::Low => 2,
            Quality::Medium => 4,
            Quality::High => 8,
            Quality::Ultra => 12,
        }
    }

    /// Maximum loaded chunk count (soft cap for eviction).
    pub fn chunk_cap(self) -> usize {
        let r = self.render_distance() as usize;
        // (2r+1)^2 * height_chunks estimate, plus headroom.
        let side = 2 * r + 1;
        side * side * 4 + 64
    }
}

impl Default for Quality {
    fn default() -> Self {
        Quality::Medium
    }
}

impl std::str::FromStr for Quality {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "low" => Ok(Quality::Low),
            "medium" => Ok(Quality::Medium),
            "high" => Ok(Quality::High),
            "ultra" => Ok(Quality::Ultra),
            _ => Err(format!("unknown quality '{s}', expected low|medium|high|ultra")),
        }
    }
}
```

Add `quality` field to `CliConfig`:

```rust
pub struct CliConfig {
    pub world: WorldConfig,
    pub mario_units_per_meter: f32,
    pub quality: Quality,
}
```

Add `--quality` to the arg parser `match` block:

```rust
"--quality" => {
    let v = next_value(&args, &mut i, "--quality")?;
    quality = v.parse().map_err(|e: String| format!("--quality: {e}"))?;
}
```

Add `let mut quality = Quality::default();` before the parse loop.

Update the return:
```rust
Ok(CliConfig { world: cfg, mario_units_per_meter, quality })
```

Update `usage()`:
```
--quality     streaming quality preset: low|medium|high|ultra (default medium)\n\
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p vox-app -- quality_ --nocapture`
Expected: PASS

**Step 5: Run full args test suite**

Run: `cargo test -p vox-app -- args --nocapture`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add crates/vox-app/src/args.rs
git commit -m "feat(app): add --quality CLI flag with render distance/detail/budget presets"
```

---

### Task 6: ChunkLoader — Core Streaming

**Files:**
- Create: `crates/vox-app/src/chunk_loader.rs`
- Modify: `crates/vox-app/src/main.rs` (add `mod chunk_loader;` and wire)

**Step 1: Write the ChunkLoader module**

This is a new module — no failing test first (it's an app-level integration component that needs GPU access). Test behaviorally by running the app.

```rust
//! Player-centered chunk streaming: generates chunks around the player on
//! demand, evicts them beyond render distance. Mirrors `SurfaceProvider`'s
//! center/threshold/radius idiom.
//!
//! Pristine chunks (generated, never edited) evict fully — regenerated
//! deterministically on return. Edited chunks keep their voxel data; only
//! their GPU mesh drops.

use glam::{IVec3, Vec3};
use vox_core::consts::CHUNK_SIZE;
use vox_core::{WorldConfig, chunk_origin};
use vox_gen::{TerrainGen, TerrainMaterials, TreeMaterials, stamp_tree, trees_for_chunk};
use vox_render::{Gpu, VoxelPipeline};
use vox_world::World;

use crate::args::Quality;

/// Don't re-scan unless the player moved at least this many chunks.
const RELOAD_THRESHOLD_CHUNKS: i32 = 1;

pub struct ChunkLoader {
    quality: Quality,
    last_center_chunk: IVec3,
    terrain: TerrainGen,
    terrain_mats: TerrainMaterials,
    tree_mats: TreeMaterials,
}

impl ChunkLoader {
    pub fn new(
        cfg: &WorldConfig,
        quality: Quality,
        terrain: TerrainGen,
        terrain_mats: TerrainMaterials,
        tree_mats: TreeMaterials,
    ) -> Self {
        Self {
            quality,
            last_center_chunk: IVec3::splat(i32::MAX),
            terrain,
            terrain_mats,
            tree_mats,
        }
    }

    pub fn quality(&self) -> Quality {
        self.quality
    }

    pub fn set_quality(&mut self, q: Quality) {
        self.quality = q;
        // Force reload on next update.
        self.last_center_chunk = IVec3::splat(i32::MAX);
    }

    /// Player's chunk key from world position.
    fn player_chunk(pos: Vec3, voxel_size: f32) -> IVec3 {
        let chunk_m = CHUNK_SIZE as f32 * voxel_size;
        IVec3::new(
            (pos.x / chunk_m).floor() as i32,
            (pos.y / chunk_m).floor() as i32,
            (pos.z / chunk_m).floor() as i32,
        )
    }

    /// Pre-generate chunks around a position for spawn. Synchronous —
    /// generates all chunks within the detail ring before returning.
    pub fn pregenerate_spawn(
        &mut self,
        player_pos: Vec3,
        world: &mut World,
        pipeline: &mut VoxelPipeline,
        gpu: &Gpu,
    ) {
        let center = Self::player_chunk(player_pos, world.cfg.voxel_size_m);
        let ring = self.quality.detail_ring();
        let radius = ring.max(2); // At least 2 chunks for spawn.
        self.generate_ring(world, pipeline, gpu, center, radius, ring);
        self.last_center_chunk = center;
    }

    /// Per-frame update: generate missing chunks near the player, evict
    /// chunks beyond render distance. Returns whether any changes were made.
    pub fn update(
        &mut self,
        player_pos: Vec3,
        world: &mut World,
        pipeline: &mut VoxelPipeline,
        gpu: &Gpu,
    ) -> bool {
        let s = world.cfg.voxel_size_m;
        let center = Self::player_chunk(player_pos, s);

        // Only act when the player crossed a chunk boundary.
        if (center - self.last_center_chunk).abs().max_element() < RELOAD_THRESHOLD_CHUNKS {
            return false;
        }
        self.last_center_chunk = center;

        let render_dist = self.quality.render_distance();
        let detail_ring = self.quality.detail_ring();
        let budget = self.quality.gen_budget();

        // Generate missing chunks (up to budget, nearest first).
        let generated = self.generate_missing(world, pipeline, gpu, center, render_dist, detail_ring, budget);

        // Evict chunks beyond render distance.
        let evicted = self.evict_beyond_range(world, pipeline, center, render_dist);

        generated || evicted
    }

    /// Generate missing chunks within render distance, up to `budget`,
    /// nearest to `center` first.
    fn generate_missing(
        &self,
        world: &mut World,
        pipeline: &mut VoxelPipeline,
        gpu: &Gpu,
        center: IVec3,
        render_dist: i32,
        detail_ring: i32,
        budget: usize,
    ) -> bool {
        let (bmin, bmax) = world.bounds_voxels();
        let chunk_min = chunk_of(bmin);
        let chunk_max = chunk_of(bmax - IVec3::ONE);

        // Collect missing chunks, sorted by distance from center.
        let mut missing: Vec<(i64, IVec3, bool)> = Vec::new();
        for dz in -render_dist..=render_dist {
            for dy in -render_dist..=render_dist {
                for dx in -render_dist..=render_dist {
                    let key = center + IVec3::new(dx, dy, dz);
                    if key.x < chunk_min.x || key.x > chunk_max.x { continue; }
                    if key.y < chunk_min.y || key.y > chunk_max.y { continue; }
                    if key.z < chunk_min.z || key.z > chunk_max.z { continue; }
                    if world.chunk_at(key).is_some() { continue; }
                    let dist = (dx * dx + dy * dy + dz * dz) as i64;
                    let in_detail = dx.abs() <= detail_ring
                        && dz.abs() <= detail_ring
                        && dy.abs() <= detail_ring;
                    missing.push((dist, key, in_detail));
                }
            }
        }
        missing.sort_by_key(|(d, _, _)| *d);

        let mut generated = false;
        for (_, key, in_detail) in missing.into_iter().take(budget) {
            self.generate_chunk(world, pipeline, gpu, key, in_detail);
            generated = true;
        }
        generated
    }

    /// Generate all chunks within `radius` (synchronous, for spawn).
    fn generate_ring(
        &self,
        world: &mut World,
        pipeline: &mut VoxelPipeline,
        gpu: &Gpu,
        center: IVec3,
        radius: i32,
        detail_ring: i32,
    ) {
        let (bmin, bmax) = world.bounds_voxels();
        let chunk_min = chunk_of(bmin);
        let chunk_max = chunk_of(bmax - IVec3::ONE);

        for dz in -radius..=radius {
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let key = center + IVec3::new(dx, dy, dz);
                    if key.x < chunk_min.x || key.x > chunk_max.x { continue; }
                    if key.y < chunk_min.y || key.y > chunk_max.y { continue; }
                    if key.z < chunk_min.z || key.z > chunk_max.z { continue; }
                    if world.chunk_at(key).is_some() { continue; }
                    let in_detail = dx.abs() <= detail_ring
                        && dz.abs() <= detail_ring
                        && dy.abs() <= detail_ring;
                    self.generate_chunk(world, pipeline, gpu, key, in_detail);
                }
            }
        }
    }

    /// Generate one chunk: terrain + clipped trees (if in detail ring).
    fn generate_chunk(
        &self,
        world: &mut World,
        _pipeline: &mut VoxelPipeline,
        _gpu: &Gpu,
        key: IVec3,
        in_detail: bool,
    ) {
        let s = world.cfg.voxel_size_m;

        // Suppress edit tracking during generation.
        world.set_suppress_edit_tracking(true);

        // Terrain.
        let chunk = self.terrain.fill_surface_chunk_public(key, s, self.terrain_mats);
        world.insert_chunk(key, chunk);

        // Trees: only root trees in detail-ring chunks, but stamp their
        // canopy clipped to this chunk (canopy may extend into far chunks).
        if in_detail {
            let origin = chunk_origin(key);
            let clip_min = origin;
            let clip_max = origin + IVec3::splat(CHUNK_SIZE as i32);
            world.set_clip(clip_min, clip_max);
            let trees = trees_for_chunk(&world.cfg, &self.terrain, key);
            for tree in &trees {
                stamp_tree(world, tree, self.tree_mats);
            }
            world.clear_clip();
        }

        world.set_suppress_edit_tracking(false);
    }

    /// Evict pristine chunks beyond render distance. Edited chunks keep
    /// their voxel data (only mesh drops — handled by caller via
    /// pipeline.remove_chunk).
    fn evict_beyond_range(
        &self,
        world: &mut World,
        pipeline: &mut VoxelPipeline,
        center: IVec3,
        render_dist: i32,
    ) -> bool {
        let render_dist_sq = (render_dist + 1) as i64; // +1 for hysteresis
        let render_dist_sq = render_dist_sq * render_dist_sq;
        let cap = self.quality.chunk_cap();

        let to_evict: Vec<IVec3> = world
            .chunks()
            .filter(|(key, _)| {
                let dx = key.x - center.x;
                let dy = key.y - center.y;
                let dz = key.z - center.z;
                (dx * dx + dy * dy + dz * dz) as i64 > render_dist_sq
            })
            .map(|(k, _)| k)
            .collect();

        let mut evicted = false;
        for key in to_evict {
            if world.is_edited(key) {
                // Edited: drop mesh only, keep data.
                pipeline.remove_chunk(key);
            } else {
                // Pristine: evict fully.
                world.remove_chunk(key);
                pipeline.remove_chunk(key);
            }
            evicted = true;
        }

        // Budget guard: if still over cap, evict farthest pristine first.
        let loaded = world.chunk_count();
        if loaded > cap {
            let mut pristine: Vec<(i64, IVec3)> = world
                .chunks()
                .filter(|(k, _)| !world.is_edited(*k))
                .map(|(k, _)| {
                    let dx = k.x - center.x;
                    let dy = k.y - center.y;
                    let dz = k.z - center.z;
                    (dx as i64 * dx + dy as i64 * dy + dz as i64 * dz, k)
                })
                .collect();
            pristine.sort_by_key(|(d, _)| *d);
            for (_, key) in pristine.iter().rev().take(loaded - cap) {
                world.remove_chunk(*key);
                pipeline.remove_chunk(*key);
                evicted = true;
            }
        }

        evicted
    }
}
```

**Note:** `TerrainGen::fill_surface_chunk` is currently private. We need to make it `pub` (or add a pub wrapper). This is a one-line visibility change in `vox-gen/src/terrain.rs`:

```rust
pub fn fill_surface_chunk(&self, key: IVec3, s: f32, mats: TerrainMaterials) -> Chunk {
```

Also need `chunk_of` import. Add to `vox-core/src/lib.rs` re-exports if not already:
```rust
pub use coords::{chunk_of, chunk_origin, ...};
```
(Check — `chunk_of` may already be exported. The `world.rs` already uses it from `vox_core`.)

**Step 2: Wire into main.rs**

Add `mod chunk_loader;` to main.rs module declarations (around line 16).

Add `use chunk_loader::ChunkLoader;` to imports.

Add field to `VoxApp` struct:
```rust
chunk_loader: ChunkLoader,
```

In `VoxApp::new`, after `build_terrain_world`:
- Change `build_terrain_world` to NOT call `terrain.generate()` or `generate_trees()` in streaming mode. Instead, create an empty world and let the loader generate.
- Create the `ChunkLoader` with terrain, materials.
- Call `chunk_loader.pregenerate_spawn(player_pos, &mut world, &mut pipeline, &gpu)` BEFORE `surface_height_m`.
- Then call `initial_mesh()`.
- Then `surface_height_m` + `Player::new`.

The refactored `build_terrain_world` for streaming mode:

```rust
fn build_streaming_world(
    cfg: WorldConfig,
    registry: &MaterialRegistry,
) -> Result<World, Box<dyn std::error::Error + Send + Sync>> {
    cfg.validate()?;
    let mut world = World::new(cfg);
    world.set_solid_table(solid_table(registry));
    // No upfront generation — ChunkLoader generates lazily.
    Ok(world)
}
```

In `VoxApp::new`, replace the world-building section:

```rust
let world = build_streaming_world(cfg, &registry)?;

let terrain = TerrainGen::new(&world.cfg);
let terrain_mats = TerrainMaterials::from_registry(&registry)?;
let tree_mats = TreeMaterials::from_registry(&registry)?;
let chunk_loader = ChunkLoader::new(&world.cfg, quality, terrain, terrain_mats, tree_mats);
```

Then, after the pipeline is created but before `initial_mesh()`:

```rust
// Pre-generate spawn area before surface_height_m (which needs solid voxels).
let center = Vec3::from(world.cfg.extent_m) * 0.5;
let mut chunk_loader = chunk_loader;
chunk_loader.pregenerate_spawn(center, &mut world, &mut pipeline, &gpu);
```

Then `initial_mesh()` runs as before (meshes the pre-generated chunks).

Then `surface_height_m` works because chunks exist.

Pass `quality` through `VoxApp::new` signature (add parameter).

**Step 3: Build to verify it compiles**

Run: `cargo build -p vox-app --release`
Expected: Compiles clean (may have unused warnings for now)

**Step 4: Run to verify streaming works**

Run: `cargo run -p vox-app --release`
Expected: World generates around spawn, player stands on terrain. Walking shows new chunks generating, old chunks evicting.

**Step 5: Commit**

```bash
git add crates/vox-app/src/chunk_loader.rs crates/vox-app/src/main.rs crates/vox-gen/src/terrain.rs
git commit -m "feat(app): add ChunkLoader for player-centered streaming with quality presets"
```

---

### Task 7: Main Loop — Per-Frame Update + Eviction

**Files:**
- Modify: `crates/vox-app/src/main.rs`

**Step 1: Add per-frame chunk_loader.update call**

In the `update` method, before the remesh section (around line 1940), add:

```rust
// Stream chunks around the player: generate missing, evict beyond range.
let player_pos = self.player.ctrl.pos;
let _streamed = self.chunk_loader.update(
    player_pos,
    &mut self.world,
    &mut self.pipeline,
    &self.gpu,
);
```

This goes BEFORE the remesh `absorb_dirty` call, so newly generated chunks (which `insert_chunk` marks dirty) get picked up by the remesh queue the same frame.

If chunks were evicted, invalidate grass:
```rust
if _streamed {
    self.grass_cache.invalidate();
}
```

**Step 2: Build**

Run: `cargo build -p vox-app --release`
Expected: Compiles clean

**Step 3: Run and verify streaming works while moving**

Run: `cargo run -p vox-app --release`
Expected: Walk around — chunks generate ahead, evict behind. No crashes. FPS stable.

**Step 4: Commit**

```bash
git add crates/vox-app/src/main.rs
git commit -m "feat(app): wire per-frame ChunkLoader update with grass invalidation"
```

---

### Task 8: Quality Switching Key + Debug Panel

**Files:**
- Modify: `crates/vox-app/src/main.rs`
- Modify: `crates/vox-debug/src/panels.rs`
- Modify: `crates/vox-debug/src/lib.rs`

**Step 1: Add quality cycling key**

In the key handling section (around line 1595), add:

```rust
if input.key_pressed(KeyCode::KeyQ) {
    let next = match self.chunk_loader.quality() {
        Quality::Low => Quality::Medium,
        Quality::Medium => Quality::High,
        Quality::High => Quality::Ultra,
        Quality::Ultra => Quality::Low,
    };
    self.chunk_loader.set_quality(next);
    tracing::info!(?next, "quality switched");
}
```

Add `use args::Quality;` to main.rs imports.

**Step 2: Add quality display to debug overlay**

In `OverlayState` (`vox-debug/src/lib.rs`), add:
```rust
pub quality_label: &'a str,
```

In `panels.rs` stats window, add after particles:
```rust
ui.label(format!("quality: {}", state.quality_label));
```

In main.rs where `OverlayState` is constructed (around line 2102), add:
```rust
quality_label: match self.chunk_loader.quality() {
    Quality::Low => "low",
    Quality::Medium => "medium",
    Quality::High => "high",
    Quality::Ultra => "ultra",
},
```

**Step 3: Build and run**

Run: `cargo build -p vox-app --release && cargo run -p vox-app --release`
Expected: Q key cycles quality, F3 overlay shows current quality. Render distance changes visibly.

**Step 4: Commit**

```bash
git add crates/vox-app/src/main.rs crates/vox-debug/src/panels.rs crates/vox-debug/src/lib.rs
git commit -m "feat(app): add Q key quality cycling + debug overlay quality display"
```

---

### Task 9: Tree Canopy Cross-Chunk Stamping

**Files:**
- Modify: `crates/vox-app/src/chunk_loader.rs`

**Note:** The design's root-chunk-tier rule means trees rooted in near chunks must stamp their canopy into ALL overlapping chunks — including far chunks. The current `generate_chunk` only stamps trees rooted in the target chunk. We need to also stamp canopy from near-rooted trees whose canopy REACHES INTO the target chunk, even when the target is far.

**Step 1: Add cross-chunk tree stamping to generate_chunk**

Replace the tree section of `generate_chunk` with:

```rust
// Trees: stamp all trees whose canopy overlaps this chunk.
// Tree existence is gated by the ROOT chunk's tier (root within detail
// ring → tree exists). Canopy extends into all overlapping chunks.
let origin = chunk_origin(key);
let clip_min = origin;
let clip_max = origin + IVec3::splat(CHUNK_SIZE as i32);
world.set_clip(clip_min, clip_max);

// Trees rooted in THIS chunk (only if this chunk is in the detail ring).
if in_detail {
    let trees = trees_for_chunk(&world.cfg, &self.terrain, key);
    for tree in &trees {
        stamp_tree(world, tree, self.tree_mats);
    }
}

// Trees rooted in NEIGHBORING near chunks whose canopy reaches into this
// chunk. We check all chunks within canopy-reach distance that are in the
// detail ring. Canopy reach is ~5m; at 0.1m voxels that's ~50 voxels < 2
// chunks, so checking the 3x3 neighborhood of chunks suffices.
if !in_detail {
    // This is a far chunk — check if any near chunk's trees reach into us.
    let canopy_reach_chunks = 2; // Conservative: canopy ~5m, chunk 3.2m at 0.1m.
    for dz in -canopy_reach_chunks..=canopy_reach_chunks {
        for dx in -canopy_reach_chunks..=canopy_reach_chunks {
            let neighbor = IVec3::new(key.x + dx, key.y, key.z + dz);
            // Only stamp from near (detail-ring) root chunks.
            let ndx = neighbor.x - center_chunk.x;
            let ndz = neighbor.z - center_chunk.z;
            if ndx.abs() > detail_ring || ndz.abs() > detail_ring {
                continue;
            }
            let trees = trees_for_chunk(&world.cfg, &self.terrain, neighbor);
            for tree in &trees {
                // stamp_tree with clip active — only voxels in this chunk
                // are written; the rest are dropped by the clip guard.
                stamp_tree(world, tree, self.tree_mats);
            }
        }
    }
}

world.clear_clip();
```

**Problem:** `generate_chunk` doesn't currently know `center_chunk` or `detail_ring`. Pass them as parameters.

Update `generate_chunk` signature:
```rust
fn generate_chunk(
    &self,
    world: &mut World,
    _pipeline: &mut VoxelPipeline,
    _gpu: &Gpu,
    key: IVec3,
    in_detail: bool,
    center_chunk: IVec3,
    detail_ring: i32,
) {
```

Update all call sites in `generate_missing`, `generate_ring`, and `pregenerate_spawn` to pass `center` and `detail_ring`.

**Step 2: Build and run**

Run: `cargo build -p vox-app --release && cargo run -p vox-app --release`
Expected: Trees at the detail-ring boundary are whole — no half-cut canopies. Walking toward a tree at the ring edge shows it fully formed.

**Step 3: Commit**

```bash
git add crates/vox-app/src/chunk_loader.rs
git commit -m "feat(app): stamp tree canopy across detail ring boundary (root-chunk-tier rule)"
```

---

### Task 10: Integration Smoke Test + Cleanup

**Files:**
- Modify: `crates/vox-app/src/main.rs` (cleanup)

**Step 1: Run full test suite**

Run: `cargo test --workspace --lib -- --nocapture`
Expected: All tests PASS (existing 238+ tests + new tests from Tasks 1-5)

**Step 2: Build release**

Run: `cargo build -p vox-app --release`
Expected: Compiles clean, minimal warnings

**Step 3: Manual smoke test**

Run: `cargo run -p vox-app --release`
Verify:
- [ ] Player spawns on terrain (not mid-air)
- [ ] Walking generates chunks ahead
- [ ] Walking away evicts chunks behind (memory doesn't grow unbounded)
- [ ] Digging a tunnel, walking away, walking back — tunnel persists
- [ ] Q key cycles quality (low→medium→high→ultra→low)
- [ ] F3 overlay shows current quality
- [ ] Trees at detail-ring boundary are whole (no half-cut canopies)
- [ ] `--quality low` starts with small render distance
- [ ] `--quality ultra` starts with large render distance
- [ ] `--extent 512,128,512` works with streaming (large finite world)
- [ ] No crashes after 2+ minutes of walking

**Step 4: Run at 1.0 scale**

Run: `cargo run -p vox-app --release -- --scale 1.0`
Verify:
- [ ] Streaming works at 1.0m voxels (32m chunks)
- [ ] Trees generate correctly at coarse scale

**Step 5: Remove any dead code**

- Remove `build_terrain_world` if fully replaced by `build_streaming_world`
- Remove any now-unused imports
- Verify `generate_trees` (global) is still used in tests but not in streaming path

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: seamless world generation with quality presets and edit retention"
```

---

## Task Dependencies

```
Task 1 (clip guard) ──────────┐
Task 2 (edit tracking) ───────┤
Task 3 (remove_chunk) ────────┤
Task 4 (trees_for_chunk) ─────┼──→ Task 6 (ChunkLoader) ──→ Task 7 (main loop) ──→ Task 8 (quality UI) ──→ Task 9 (canopy cross-chunk) ──→ Task 10 (smoke test)
Task 5 (quality presets) ─────┘
```

Tasks 1-5 are independent and can be parallelized. Task 6 depends on all of 1-5. Tasks 7-10 are sequential.
