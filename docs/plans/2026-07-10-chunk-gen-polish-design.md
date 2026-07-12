# Chunk Generation Polish — Design

**Date:** 2026-07-10
**Status:** Approved (all 4 sections)
**Approach:** A — Targeted fixes

## Summary

Four surgical fixes to the streaming chunk system: vertical canopy stamping,
re-meshing edited chunks on return, destruction/fluid-triggered generation,
and proactive pre-generation ahead of the player.

## Section 1: Vertical Canopy Stamping

**Bug:** The canopy loop in `generate_chunk` iterates `dx`/`dz` (horizontal)
but fixes `key.y`. A 10m tree's canopy reaches 3+ chunks above its root chunk
at 0.1m voxels (3.2m chunks). Those canopy voxels are silently dropped.

**Fix:** Extend the loop to iterate `dy` from `-vertical_reach` to `0` (trees
grow up, never down). Compute `vertical_reach` from max tree height:

```
max_tree_height = 6.0 + 4.0 = 10.0m (trunk) + 1.5 + 0.7 = 2.2m (crown) = 12.2m
vertical_reach = ceil(12.2 / (CHUNK_SIZE * voxel_size_m)) + 1
```

At 0.1m: `ceil(12.2 / 3.2) + 1 = 5`. At 1.0m: `ceil(12.2 / 32.0) + 1 = 2`.

The root-chunk-tier gate (detail ring) stays horizontal-only — trees root at
a specific Y level. `trees_for_chunk` ignores Y in its filter, so calling it
with `neighbor.y - 5` returns the same trees as `neighbor.y` — the clip drops
writes outside the target chunk.

## Section 2: Re-mesh Edited Chunks on Return

**Bug:** When an edited chunk is evicted (mesh dropped, data kept) and the
player returns, `generate_missing` sees `world.chunk_at(key).is_some()` →
skips it. Nothing marks it dirty → `remesh.absorb_dirty()` never re-meshes it
→ the chunk's modifications are invisible.

**Fix:** After `generate_missing` in `update()`, scan for loaded chunks
within render distance that exist in `World` but have no GPU mesh in
`VoxelPipeline`. Mark them dirty via `World::mark_dirty(key)` so the remesh
queue picks them up next frame.

**New methods:**
- `VoxelPipeline::has_chunk_mesh(key: IVec3) -> bool` — checks `self.chunks.contains_key(&key)`
- `World::mark_dirty(key: IVec3)` — inserts into the dirty set (currently private)

The reconciliation runs inside `update()` after generation, only on boundary
crossing (same trigger as the rest of `update()`).

## Section 3: Destruction/Fluid-Triggered Generation

**Gap:** When `blast()`, `carve_sphere()`, or fluid flow touches a position
in an unloaded chunk, the operation silently fails — reads as air, nothing
to carve, water flows into void.

**Fix:** Add `ChunkLoader::ensure_loaded(key)` — synchronously generates a
chunk if absent. Add `ensure_loaded_box(min_m, max_m)` for radius-based
operations. Call sites in `main.rs` BEFORE tool invocation:

1. **Bomb/ScalableDig**: Before calling `tools.blast()`/`tools.scalable_dig()`,
   compute approximate hit point from `eye + look * REACH`, call
   `ensure_loaded_box(hit_point - radius, hit_point + radius)`.

2. **DeathLaser**: Ensure chunks along the beam up to `REACH` distance.

3. **Fluid tick**: After `fluid.wake_region()` in the main loop, scan active
   fluid cells near chunk boundaries and `ensure_loaded` neighbor chunks
   before the next fluid tick.

4. **`detach_unsupported()`**: Before calling, `ensure_loaded` for chunks
   the carve touched (including neighbors) so the flood can determine real
   results instead of returning `Unknown`.

`ensure_loaded` is synchronous, cheap (one chunk = ~32² surface-band voxels),
and only generates chunks within world bounds.

## Section 4: Proactive Pre-generation

**Gap:** `update()` only fires on chunk boundary crossing. Fast-moving
players see pop-in at the render distance edge.

**Fix:** Add a separate `pregen_ahead(player_pos, player_vel)` method that
runs EVERY FRAME (outside the boundary-crossing gate) with a small budget
(1-3 chunks). It predicts the player's next position from velocity (look-
ahead ~1 second) and pre-generates the predicted chunk plus its 3×3
horizontal neighbors if they're within world bounds and not already loaded.

`update()` gains a `player_vel: Vec3` parameter. The pre-gen counts against
a small per-frame budget (separate from `gen_budget`) so it doesn't stall
frames or starve the main generation pass. Only generates chunks within
world bounds.

## Components to Build

1. **Vertical canopy loop** (`chunk_loader.rs`):
   - Compute `vertical_reach` from chunk size
   - Iterate `dy` from `-vertical_reach` to `0`
   - Keep horizontal reach at `±2`

2. **Re-mesh reconciliation** (`chunk_loader.rs`, `voxel_pipeline.rs`, `world.rs`):
   - `VoxelPipeline::has_chunk_mesh(key) -> bool`
   - `World::mark_dirty(key)`
   - Reconciliation scan in `update()` after generation

3. **`ensure_loaded`** (`chunk_loader.rs`, `main.rs`):
   - Pre-tool calls in main.rs for bomb/dig/laser
   - Post-fluid-tick boundary scan in main.rs

4. **Proactive pre-generation** (`chunk_loader.rs`, `main.rs`):
   - `pregen_ahead(player_pos, player_vel)` method, runs every frame
   - 3×3 predicted-chunk pre-gen with small per-frame budget (1-3 chunks)
   - `player_vel` parameter on `update()` (or separate call from main loop)
