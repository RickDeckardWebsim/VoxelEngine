//! Player-centered chunk streaming: generates chunks around the player on
//! demand, evicts them beyond render distance. Mirrors `SurfaceProvider`'s
//! center/threshold/radius idiom.
//!
//! Pristine chunks (generated, never edited) evict fully — regenerated
//! deterministically on return. Edited chunks keep their voxel data; only
//! their GPU mesh drops.

use glam::{IVec3, Vec3};
use vox_core::consts::CHUNK_SIZE;
use vox_core::{WorldConfig, chunk_of, chunk_origin};
use vox_gen::{ChunkBand, TerrainGen, TerrainMaterials, TreeMaterials, stamp_tree, trees_for_chunk};
use vox_render::VoxelPipeline;
use vox_world::{Chunk, World};

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
        _cfg: &WorldConfig,
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
    ) {
        let center = Self::player_chunk(player_pos, world.cfg.voxel_size_m);
        let ring = self.quality.detail_ring();
        let radius = ring.max(2); // At least 2 chunks for spawn.
        self.generate_ring(world, center, radius, ring);
        self.last_center_chunk = center;
    }

    /// Per-frame update: generate missing chunks near the player, evict
    /// chunks beyond render distance, re-mesh edited chunks whose GPU
    /// mesh was dropped, and pre-generate ahead of the player's movement.
    /// Returns whether any changes were made.
    pub fn update(
        &mut self,
        player_pos: Vec3,
        player_vel: Vec3,
        world: &mut World,
        pipeline: &mut VoxelPipeline,
    ) -> bool {
        let s = world.cfg.voxel_size_m;
        let center = Self::player_chunk(player_pos, s);

        // Pre-generate ahead of the player every frame (outside the
        // boundary-crossing gate) so chunks are ready before arrival.
        let pregen = self.pregen_ahead(player_pos, player_vel, world);

        // Only do the heavy generation + eviction + reconciliation pass
        // when the player crossed a chunk boundary.
        if (center - self.last_center_chunk).abs().max_element() < RELOAD_THRESHOLD_CHUNKS {
            return pregen;
        }
        self.last_center_chunk = center;

        let render_dist = self.quality.render_distance();
        let detail_ring = self.quality.detail_ring();
        let budget = self.quality.gen_budget();

        // Generate missing chunks (up to budget, nearest first).
        let generated = self.generate_missing(world, center, render_dist, detail_ring, budget);

        // Evict chunks beyond render distance.
        let evicted = self.evict_beyond_range(world, pipeline, center, render_dist);

        // Reconcile: re-mesh edited chunks that have data but no GPU mesh
        // (evicted earlier, player walked back). Only check edited chunks
        // — pristine chunks with no mesh are either freshly generated
        // (already dirty) or absent. Mark them dirty so the remesh queue
        // picks them up next frame.
        let mut reconciled = false;
        for dz in -render_dist..=render_dist {
            for dy in -render_dist..=render_dist {
                for dx in -render_dist..=render_dist {
                    let key = center + IVec3::new(dx, dy, dz);
                    if world.is_edited(key)
                        && world.chunk_at(key).is_some()
                        && !pipeline.has_chunk_mesh(key)
                    {
                        world.mark_dirty(key);
                        reconciled = true;
                    }
                }
            }
        }

        generated || evicted || reconciled || pregen
    }

    /// Synchronously generate a chunk at `key` if it doesn't exist and is
    /// within world bounds. Used before destructive/fluid operations that
    /// might cross into unloaded territory.
    pub fn ensure_loaded(
        &self,
        world: &mut World,
        key: IVec3,
    ) {
        if world.chunk_at(key).is_some() {
            return;
        }
        let (bmin, bmax) = world.bounds_voxels();
        let chunk_min = chunk_of(bmin);
        let chunk_max = chunk_of(bmax - IVec3::ONE);
        if key.x < chunk_min.x || key.x > chunk_max.x { return; }
        if key.y < chunk_min.y || key.y > chunk_max.y { return; }
        if key.z < chunk_min.z || key.z > chunk_max.z { return; }
        let center = self.last_center_chunk;
        let detail_ring = self.quality.detail_ring();
        self.generate_chunk(world, key, center, detail_ring);
        world.mark_dirty(key);
    }

    /// Ensure all chunks overlapping the world-space box `[min_m, max_m]`
    /// are loaded. Used before bomb/dig operations with a known radius.
    pub fn ensure_loaded_box(
        &self,
        min_m: Vec3,
        max_m: Vec3,
        world: &mut World,
    ) {
        let s = world.cfg.voxel_size_m;
        let min_vox = (min_m / s).floor().as_ivec3();
        let max_vox = (max_m / s).ceil().as_ivec3();
        let min_chunk = chunk_of(min_vox);
        let max_chunk = chunk_of(max_vox);
        for cz in min_chunk.z..=max_chunk.z {
            for cy in min_chunk.y..=max_chunk.y {
                for cx in min_chunk.x..=max_chunk.x {
                    self.ensure_loaded(world, IVec3::new(cx, cy, cz));
                }
            }
        }
    }

    /// Pre-generate chunks ahead of the player based on velocity. Runs
    /// every frame with a small budget (1-3 chunks) outside the boundary
    /// gate. Predicts the player's position ~1 second ahead and generates
    /// the 3×3 horizontal ring around the predicted chunk.
    fn pregen_ahead(
        &self,
        player_pos: Vec3,
        player_vel: Vec3,
        world: &mut World,
    ) -> bool {
        if player_vel.length_squared() < 0.01 {
            return false;
        }
        let s = world.cfg.voxel_size_m;
        let next_pos = player_pos + player_vel * 1.0; // 1-second look-ahead
        let next_chunk = Self::player_chunk(next_pos, s);
        let center = self.last_center_chunk;
        let detail_ring = self.quality.detail_ring();
        let mut generated = false;
        let mut budget = 3; // Small per-frame pre-gen budget
        for dz in -1..=1 {
            for dx in -1..=1 {
                if budget == 0 {
                    break;
                }
                let key = IVec3::new(next_chunk.x + dx, next_chunk.y, next_chunk.z + dz);
                if world.chunk_at(key).is_some() {
                    continue;
                }
                let (bmin, bmax) = world.bounds_voxels();
                let chunk_min = chunk_of(bmin);
                let chunk_max = chunk_of(bmax - IVec3::ONE);
                if key.x < chunk_min.x || key.x > chunk_max.x { continue; }
                if key.y < chunk_min.y || key.y > chunk_max.y { continue; }
                if key.z < chunk_min.z || key.z > chunk_max.z { continue; }
                self.generate_chunk(world, key, center, detail_ring);
                world.mark_dirty(key);
                budget -= 1;
                generated = true;
            }
        }
        generated
    }

    /// Generate missing chunks within render distance, up to `budget`,
    /// nearest to `center` first.
    fn generate_missing(
        &self,
        world: &mut World,
        center: IVec3,
        render_dist: i32,
        detail_ring: i32,
        budget: usize,
    ) -> bool {
        let (bmin, bmax) = world.bounds_voxels();
        let chunk_min = chunk_of(bmin);
        let chunk_max = chunk_of(bmax - IVec3::ONE);

        // Collect missing chunks, sorted by distance from center.
        let mut missing: Vec<(i64, IVec3)> = Vec::new();
        for dz in -render_dist..=render_dist {
            for dy in -render_dist..=render_dist {
                for dx in -render_dist..=render_dist {
                    let key = center + IVec3::new(dx, dy, dz);
                    if key.x < chunk_min.x || key.x > chunk_max.x { continue; }
                    if key.y < chunk_min.y || key.y > chunk_max.y { continue; }
                    if key.z < chunk_min.z || key.z > chunk_max.z { continue; }
                    if world.chunk_at(key).is_some() { continue; }
                    let dist = (dx * dx + dy * dy + dz * dz) as i64;
                    missing.push((dist, key));
                }
            }
        }
        missing.sort_by_key(|(d, _)| *d);

        let mut generated = false;
        for (_, key) in missing.into_iter().take(budget) {
            self.generate_chunk(world, key, center, detail_ring);
            generated = true;
        }
        generated
    }

    /// Generate all chunks within `radius` (synchronous, for spawn).
    fn generate_ring(
        &self,
        world: &mut World,
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
                    self.generate_chunk(world, key, center, detail_ring);
                }
            }
        }
    }

    /// Generate one chunk using the three-case height-band optimization
    /// (mirrors `TerrainGen::generate`): uniform stone below the surface
    /// band, skipped air above, per-column surface fill (with clipped trees)
    /// only in the surface band. Avoids allocating air chunks and dense
    /// 64 KB stone chunks.
    ///
    /// Tree stamping is UNCONDITIONAL across the canopy-reach neighborhood:
    /// the loop stamps all trees rooted in near (detail-ring) chunks whose
    /// canopy overlaps this chunk. When this chunk is itself near, dx=0/dz=0
    /// includes self → own-rooted trees stamp. When far, self is filtered
    /// out by the detail-ring check → only near-neighbor canopy stamps.
    /// This keeps trees whole across ALL chunk boundaries, not just the
    /// detail-ring boundary.
    fn generate_chunk(
        &self,
        world: &mut World,
        key: IVec3,
        center_chunk: IVec3,
        detail_ring: i32,
    ) {
        let s = world.cfg.voxel_size_m;

        // Suppress edit tracking during generation.
        world.set_suppress_edit_tracking(true);

        match self.terrain.chunk_band(key, s) {
            ChunkBand::Stone => {
                world.insert_chunk(key, Chunk::uniform(self.terrain_mats.stone));
            }
            ChunkBand::Air => {
                // Absent chunks read as air — nothing to insert.
            }
            ChunkBand::Surface => {
                let chunk = self.terrain.fill_surface_chunk(key, s, self.terrain_mats);
                world.insert_chunk(key, chunk);

                // Stamp all trees whose canopy overlaps this chunk. Tree
                // existence is gated by the ROOT chunk's tier (root within
                // detail ring → tree exists). The loop is unconditional —
                // it covers both own-rooted trees (dx=0, dz=0) and
                // neighbor-rooted trees. The clip guard ensures only voxels
                // within this chunk are written; the rest are dropped.
                let origin = chunk_origin(key);
                let clip_min = origin;
                let clip_max = origin + IVec3::splat(CHUNK_SIZE as i32);
                world.set_clip(clip_min, clip_max);

                // Canopy reach: trees grow up to ~12.2m (10m trunk + 2.2m
                // crown). Compute chunk-count reach from the world's voxel
                // size so it's correct at any scale.
                let chunk_m = CHUNK_SIZE as f32 * s;
                let canopy_reach_h: i32 = 2; // canopy radius ~2.2m < 1 chunk at 0.1m
                let canopy_reach_v: i32 = ((12.2 / chunk_m).ceil() as i32 + 1).max(1);
                // Trees grow up, not down: dy from -canopy_reach_v to 0.
                for dy in -canopy_reach_v..=0 {
                    for dz in -canopy_reach_h..=canopy_reach_h {
                        for dx in -canopy_reach_h..=canopy_reach_h {
                            let neighbor = IVec3::new(key.x + dx, key.y + dy, key.z + dz);
                            // Root-chunk-tier gate: only stamp from near
                            // (detail-ring) root chunks. The gate is
                            // horizontal-only — trees root at a specific Y.
                            // Self (dx=0, dz=0, dy=0) is included when this
                            // chunk is in the detail ring.
                            let ndx = neighbor.x - center_chunk.x;
                            let ndz = neighbor.z - center_chunk.z;
                            if ndx.abs() > detail_ring || ndz.abs() > detail_ring {
                                continue;
                            }
                            let trees = trees_for_chunk(&world.cfg, &self.terrain, neighbor);
                            for tree in &trees {
                                stamp_tree(world, tree, self.tree_mats);
                            }
                        }
                    }
                }

                world.clear_clip();
            }
        }

        world.set_suppress_edit_tracking(false);
    }

    /// Evict pristine chunks beyond render distance. Edited chunks keep
    /// their voxel data (only mesh drops — handled by caller via
    /// `pipeline.remove_chunk`).
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
                    (dx as i64 * dx as i64 + dy as i64 * dy as i64 + dz as i64 * dz as i64, k)
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
