# Seamless World Generation — Design

**Date:** 2026-07-10
**Status:** Approved (all 4 sections)
**Approach:** A — Stream + In-Memory Edit Retention

## Summary

Replace upfront full-world terrain generation with lazy per-chunk streaming
around the player. The world keeps a configurable finite extent (`extent_m`)
as the hard generation boundary; chunks within it generate on demand as the
player approaches and evict when the player leaves. Player edits persist
in-memory; unedited chunks regenerate deterministically from the seed.

Quality presets combine render-distance tiers with generation-detail rings.

## Architecture

### Chunk Streaming (Section 1)

A `ChunkLoader` in `vox-app` mirrors `SurfaceProvider`'s proven idiom
(`vox-sm64/src/surface.rs`): center position, move threshold, radius scan.
Each frame it checks if the player crossed a chunk boundary; if so, it
computes the desired chunk set (chunks within render distance, clipped to
`extent_m`), generates missing ones, and evicts ones beyond range.

**Generation flow (per chunk):**
1. `TerrainGen::fill_surface_chunk` builds terrain (already exists, pure per-chunk)
2. `World::insert_chunk` inserts it (existing method — marks dirty, feeds RemeshQueue)
3. Trees stamped clipped to this chunk (Section 2)
4. RemeshQueue picks up the dirty chunk and meshes it on rayon workers — no mesh pipeline changes

**Throttling:** Max N chunks generated per frame (quality-dependent, 2–12).
Generation is synchronous but cheap — one chunk is ~32² surface-band voxels.
Meshing already runs on rayon workers via RemeshQueue.

**Bounds:** `World::in_bounds` stays as-is — hard generation limit. The loader
skips chunks outside `[0, extent_voxels)`. Walking to the world edge shows the
boundary, same as current behavior, just lazier.

**No upfront generation in streaming mode.** `build_terrain_world` skips
`terrain.generate()` and `generate_trees()`. The loader generates the spawn
area before the first frame.

**Spawn sequencing (critical):** In `VoxApp::new` the sequence is
`initial_mesh()` → `TerrainGen::surface_height_m(&app.world, center.x, center.z)`
→ `Player::new(...)`. `surface_height_m` does a column scan via `world.solid()`,
which returns false for absent chunks → returns None → player spawns mid-air
and falls. **Spawn-area chunk generation must complete BEFORE
`surface_height_m`**, not merely "before the first frame." The loader wiring
must generate the spawn-area chunks first, then call `surface_height_m`, then
spawn the player.

```
Player crosses chunk boundary
  → ChunkLoader.update
  → Compute desired set: chunks within render_distance, in-bounds
  → Missing chunks? → Generate up to gen_budget, nearest first
  → Chunks beyond range? → Evict unedited: World.remove + Pipeline.remove
  → Done
```

### Per-Chunk Tree Clipping (Section 2)

**Problem:** `stamp_tree` writes trunk, branches, and canopy to absolute world
positions via `world.set_voxel()`. A tree rooted in chunk A stamps leaves into
chunk B. If B isn't loaded, `set_voxel` allocates it prematurely. If B loads
later, terrain gen overwrites the tree voxels.

**Solution:** A clip guard on `World` during tree stamping. When generating
chunk `K`, the loader:
1. Generates terrain for chunk `K` (pure, no cross-chunk writes)
2. Sets a clip region on `World` = chunk `K`'s voxel bounds
3. Calls `stamp_tree` for every tree whose canopy/trunk overlaps chunk `K`
4. Clears the clip region

**Mechanism:** `World` gains a `clip: Option<(IVec3, IVec3)>` field — a
half-open voxel box. `set_voxel` returns early (no write, no allocation) if
the position is outside the clip box. `get_voxel` is unaffected — absent
neighbor chunks read as AIR (replaceable=true), so `wood_replaceable`/leaf
checks work correctly, and the clip simply drops the write. No premature
allocation. All three stamp functions (`stamp_disc`, `stamp_line`,
`stamp_ellipsoid`) go through `set_voxel` exclusively — one guard covers every
tree write path.

**Tree-to-chunk association:** `plan_trees` produces `Vec<TreeInstance>`
deterministically from seed + terrain. The loader derives which trees touch
chunk `K` by computing the tree's bounding box (trunk + canopy radius) and
checking overlap with chunk `K`'s bounds. Pure math from
`(seed, chunk_coords)` — no global iteration needed.

**Quality interaction:** Tree existence is gated by the root chunk's tier
(Section 4). Near chunks stamp their trees (clipped to each overlapping target
chunk). Far chunks receive canopy stamps from near-rooted trees but never root
their own. Same clip mechanism, gated by root-chunk quality.

**Edit safety:** Clip is only active during generation. Player edits (tools,
bombs, carves) never set the clip — they write freely to any in-bounds chunk.

### Eviction & Edit Retention (Section 3)

**Two-tier chunk state:** Every chunk in `World.chunks` is either **edited**
(received player writes) or **pristine** (generated, never modified). `World`
gains an `edited: FxHashSet<IVec3>` tracking which chunks have received player
writes.

**Marking edits:** `World::set_voxel` and `World::edit_box` add the touched
chunk key to `edited` whenever they actually change a voxel (after the
same-value no-op guard). Player edits, tools, bombs, carves, fire, weathering
— all go through these two methods, so one insertion point covers everything.

**Suppress edit tracking during generation (critical):** Tree stamping goes
through `set_voxel` (verified: `stamp_disc`, `stamp_line`, `stamp_ellipsoid`
all call it). If `set_voxel` marks `edited` on every real change, every
tree-bearing chunk gets marked edited during generation → pristine chunks can
never be evicted → two-tier model collapses → memory grows to full world =
same as today.

**Fix:** `World` gains a `suppress_edit_tracking: bool` flag.
- During chunk generation (terrain `insert_chunk` + tree stamping): loader
  sets `suppress_edit_tracking = true`. Writes modify voxels normally, but
  `set_voxel`/`edit_box` skip the `edited.insert(key)` step.
- During gameplay (player tools, bombs, carves, fire, weathering, fluid): flag
  is `false`. All gameplay writes route through `set_voxel`/`edit_box`, so
  they mark `edited` correctly.

The flag is set/cleared in the tight scope of `ChunkLoader::generate_chunk` —
true before terrain+tree generation, false immediately after. No gameplay
code path ever touches it.

`insert_chunk` never marks `edited` regardless of the flag — it's a wholesale
replacement, only used during generation.

**Eviction rule:** When a chunk falls beyond render distance, the loader
checks `edited`:
- **Pristine** → evict fully: `World.chunks.remove(key)` +
  `VoxelPipeline::remove_chunk(key)`. On return, regenerated from terrain gen
  — identical result (deterministic seed).
- **Edited** → drop GPU mesh only: `VoxelPipeline::remove_chunk(key)`. Keep
  `World.chunks` entry. On return, re-mesh from stored data (dirty mark).
  Memory grows with total edits, bounded by finite world extent.

**Grass invalidation:** Evicted chunks' grass is dropped — call
`grass_cache.invalidate()` on any eviction.

**Budget guard:** Soft cap on total loaded chunks (quality-dependent). If
exceeded, evict farthest pristine chunks first even if within render distance
— graceful degradation rather than OOM. Edited chunks never force-evicted.

### Quality Levels (Section 4)

Quality presets combine render-distance tiers with generation-detail rings.
User picks via CLI (`--quality low|medium|high|ultra`) or in-game key.

| Preset | Render Distance (chunks) | Detail Ring (chunks) | Gen Budget (chunks/frame) |
|--------|-------------------------|----------------------|---------------------------|
| Low    | 4                       | 1                    | 2                         |
| Medium | 8                       | 3                    | 4                         |
| High   | 16                      | 6                    | 8                         |
| Ultra  | 24                      | 12                   | 12                        |

**Render distance:** How many chunks around the player exist in memory + GPU.
Beyond this, pristine chunks evict.

**Detail ring:** Tree existence is gated by the **root chunk's** tier. A tree
rooted in a near chunk (within the detail ring) exists in full — its canopy
stamps into ALL overlapping chunks regardless of the target chunk's own tier.
"Far = no trees" means "no trees rooted there," not "no tree voxels at all."
This keeps trees whole across the ring boundary — no half-cut canopies.
Terrain heightmap is identical in both rings (same `height_m` function), so
there's no height discontinuity. The quality difference is purely tree
rooting + render distance — no geometric LOD, no popping, no cracks.

**Live switching:** Changing quality at runtime adjusts render distance and
detail ring. Chunks beyond the new render distance evict (pristine only).
Chunks within the new detail ring that lack trees get re-generated (pristine
only — edited chunks keep their state). One-shot reconciliation pass, not
per-frame.

**Default:** Medium (8 chunks / 3-ring). At 0.1m voxels, 8 chunks = 25.6m
radius — enough to see across a clearing, not so much it thrashes memory.

## Components to Build

1. **`World` changes** (`vox-world/src/world.rs`):
   - `clip: Option<(IVec3, IVec3)>` field + `set_clip`/`clear_clip` methods
   - `edited: FxHashSet<IVec3>` field + `is_edited` query
   - `suppress_edit_tracking: bool` flag
   - `set_voxel`/`edit_box` respect clip + suppress flag
   - `insert_chunk` never marks edited
   - `remove_chunk(key)` method (evict from chunk map + dirty neighbors)

2. **`ChunkLoader`** (new module in `vox-app/src/chunk_loader.rs`):
   - `SurfaceProvider`-style center/threshold/radius idiom
   - `update(player_chunk, world, terrain, tree_planner, pipeline, quality)`
   - Generates missing chunks (terrain + clipped trees), evicts beyond range
   - Throttled by gen_budget per frame
   - Spawn-area pre-generation method

3. **Per-chunk tree planning** (`vox-gen/src/trees.rs`):
   - `trees_for_chunk(seed, chunk_key, cfg, terrain) -> Vec<TreeInstance>`
   - Derives which trees overlap a chunk from deterministic seed math
   - No global iteration; pure per-chunk

4. **Quality presets** (`vox-app/src/args.rs` + `vox-core/src/tunables.rs`):
   - `--quality low|medium|high|ultra` CLI flag
   - Quality struct (render_distance, detail_ring, gen_budget)
   - In-game quality switching key
   - Live reconciliation pass on switch

5. **Main loop wiring** (`vox-app/src/main.rs`):
   - `build_terrain_world` skips upfront generation in streaming mode
   - `ChunkLoader` field on `VoxApp`
   - Spawn-area generation before `surface_height_m`
   - Per-frame `chunk_loader.update(...)` call
   - Eviction triggers `pipeline.remove_chunk` + `grass_cache.invalidate`

## Non-Goals

- **Disk persistence** — can be layered on later without restructuring
- **Geometric voxel LOD** — cracks, greedy-mesh changes; not worth it
- **Infinite world** — finite extent stays; user chose "large finite world"
- **Chunk serialization format** — no disk I/O in this design
