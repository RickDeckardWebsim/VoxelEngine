//! The sparse voxel world: a chunk map with an edit API, dirty-chunk tracking
//! for remeshing, and dirty-region tracking for physics wake-ups.
//!
//! Chunks live in a `HashMap` — nothing assumes the world is a box — but the
//! MVP enforces a finite extent from [`WorldConfig`]: reads outside are air,
//! writes outside are ignored.

use std::collections::{HashMap, HashSet};

use glam::{IVec3, UVec3};
use vox_core::coords::{CHUNK, chunk_of, local_of};
use vox_core::{WorldConfig, chunk_origin};

use crate::chunk::{AIR, Chunk, Voxel};

/// A chunk-caching solidity lookup: repeated queries into the *same* chunk
/// skip the hash-map lookup after the first. Build once per traversal (e.g.
/// once per flood-fill), not once per query -- a 6-connected walk's
/// neighbors are almost always in the chunk just queried, so this turns a
/// hash-map lookup per step into (amortized) an array bounds check per step.
pub struct SolidLookup<'w> {
    world: &'w World,
    cached: Option<(IVec3, Option<&'w Chunk>)>,
}

impl<'w> SolidLookup<'w> {
    pub fn new(world: &'w World) -> Self {
        Self {
            world,
            cached: None,
        }
    }

    /// True when the voxel at `v` is solid (non-air). Same semantics as
    /// [`World::solid`].
    pub fn solid(&mut self, v: IVec3) -> bool {
        if !self.world.in_bounds(v) {
            return false;
        }
        let key = chunk_of(v);
        let chunk = match self.cached {
            Some((k, c)) if k == key => c,
            _ => {
                let c = self.world.chunk_at(key);
                self.cached = Some((key, c));
                c
            }
        };
        chunk.is_some_and(|c| c.get(local_of(v)) != AIR)
    }
}

/// A half-open voxel-space box `[min, max)` touched by an edit.
pub type DirtyRegion = (IVec3, IVec3);

/// The sparse voxel world.
pub struct World {
    /// Per-world configuration (voxel scale, extent, seed).
    pub cfg: WorldConfig,
    chunks: HashMap<IVec3, Chunk>,
    dirty: HashSet<IVec3>,
    dirty_regions: Vec<DirtyRegion>,
    /// Half-open world bounds in voxels, `[min, max)`.
    bounds_voxels: (IVec3, IVec3),
    warned_out_of_bounds: bool,
}

impl World {
    /// An empty world for `cfg` (all air, no chunks allocated).
    pub fn new(cfg: WorldConfig) -> Self {
        let bounds_voxels = (IVec3::ZERO, cfg.extent_voxels());
        Self {
            cfg,
            chunks: HashMap::new(),
            dirty: HashSet::new(),
            dirty_regions: Vec::new(),
            bounds_voxels,
            warned_out_of_bounds: false,
        }
    }

    /// Half-open world bounds in voxels, `[min, max)`.
    pub fn bounds_voxels(&self) -> (IVec3, IVec3) {
        self.bounds_voxels
    }

    /// True when `v` lies inside the world bounds.
    pub fn in_bounds(&self, v: IVec3) -> bool {
        let (min, max) = self.bounds_voxels;
        v.cmpge(min).all() && v.cmplt(max).all()
    }

    /// Voxel at world-voxel position `v`. Absent chunks and out-of-bounds
    /// positions read as air.
    pub fn get_voxel(&self, v: IVec3) -> Voxel {
        if !self.in_bounds(v) {
            return AIR;
        }
        match self.chunks.get(&chunk_of(v)) {
            Some(chunk) => chunk.get(local_of(v)),
            None => AIR,
        }
    }

    /// True when the voxel at `v` is solid (non-air).
    ///
    /// NOTE: this is material-id-based (id != 0). Materials with
    /// `solid = false` other than air don't exist in the MVP asset set; when
    /// they do, solidity queries must consult the registry instead.
    pub fn solid(&self, v: IVec3) -> bool {
        self.get_voxel(v) != AIR
    }

    /// Write voxel `v` at world position `pos`. Writes outside the world
    /// bounds are ignored (warned once per world). Same-value writes are
    /// no-ops and mark nothing dirty.
    pub fn set_voxel(&mut self, pos: IVec3, v: Voxel) {
        if !self.in_bounds(pos) {
            if !self.warned_out_of_bounds {
                self.warned_out_of_bounds = true;
                tracing::warn!(?pos, "ignoring out-of-bounds voxel write (warned once)");
            }
            return;
        }
        let key = chunk_of(pos);
        let local = local_of(pos);
        match self.chunks.entry(key) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let chunk = e.get_mut();
                if chunk.get(local) == v {
                    return;
                }
                chunk.set(local, v);
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                if v == AIR {
                    return; // Absent chunks are already air; don't allocate.
                }
                e.insert(Chunk::new()).set(local, v);
            }
        }
        self.mark_dirty_with_neighbors(key, local);
        self.dirty_regions.push((pos, pos + IVec3::ONE));
    }

    /// Fill the half-open voxel box `[min, max)` with `v`, clipped to bounds.
    pub fn fill_box(&mut self, min: IVec3, max: IVec3, v: Voxel) {
        self.edit_box(min, max, |_, cur| (cur != v).then_some(v));
    }

    /// Visit every voxel in the half-open box `[min, max)`, clipped to world
    /// bounds, resolving each *chunk* the box touches exactly once instead of
    /// once per voxel — [`get_voxel`](Self::get_voxel)/[`set_voxel`](Self::set_voxel)
    /// each pay a hash-map lookup per call, which dominates the cost of any
    /// edit spanning more than a handful of voxels (a blast or beam crossing
    /// real terrain touches tens of thousands). `edit(pos, current)` is
    /// called for every voxel in the box, including ones in absent (all-air)
    /// chunks; returning `Some(new)` writes it, `None` leaves it unchanged.
    /// A chunk is only allocated if `edit` actually asks to change something
    /// in it.
    pub fn edit_box(
        &mut self,
        min: IVec3,
        max: IVec3,
        mut edit: impl FnMut(IVec3, Voxel) -> Option<Voxel>,
    ) {
        let (bmin, bmax) = self.bounds_voxels;
        let min = min.max(bmin);
        let max = max.min(bmax);
        if min.cmpge(max).any() {
            return;
        }

        let ckey_min = chunk_of(min);
        let ckey_max = chunk_of(max - IVec3::ONE); // inclusive
        for cz in ckey_min.z..=ckey_max.z {
            for cy in ckey_min.y..=ckey_max.y {
                for cx in ckey_min.x..=ckey_max.x {
                    let key = IVec3::new(cx, cy, cz);
                    let origin = chunk_origin(key);
                    let local_min = (min - origin).max(IVec3::ZERO);
                    let local_max = (max - origin).min(IVec3::splat(CHUNK));
                    if local_min.cmpge(local_max).any() {
                        continue;
                    }

                    // Read pass against the chunk as it currently exists (one
                    // hash-map lookup total for this chunk, not per voxel).
                    // Absent chunks read as all-air without allocating.
                    let mut changes: Vec<(UVec3, Voxel)> = Vec::new();
                    {
                        let existing = self.chunks.get(&key);
                        for z in local_min.z..local_max.z {
                            for y in local_min.y..local_max.y {
                                for x in local_min.x..local_max.x {
                                    let local = UVec3::new(x as u32, y as u32, z as u32);
                                    let cur = existing.map_or(AIR, |c| c.get(local));
                                    let world_pos = origin + IVec3::new(x, y, z);
                                    if let Some(new_v) = edit(world_pos, cur)
                                        && new_v != cur
                                    {
                                        changes.push((local, new_v));
                                    }
                                }
                            }
                        }
                    }
                    if changes.is_empty() {
                        continue;
                    }

                    let chunk = self.chunks.entry(key).or_default();
                    for &(local, v) in &changes {
                        chunk.set(local, v);
                    }
                    for (local, _) in changes {
                        self.mark_dirty_with_neighbors(key, local);
                    }
                }
            }
        }
        self.dirty_regions.push((min, max));
    }

    /// Insert a whole generated chunk, replacing any existing one. Dirties the
    /// chunk and its six face neighbors (their meshes sample this chunk).
    pub fn insert_chunk(&mut self, key: IVec3, chunk: Chunk) {
        let origin = chunk_origin(key);
        self.dirty_regions
            .push((origin, origin + IVec3::splat(CHUNK)));
        self.chunks.insert(key, chunk);
        self.dirty.insert(key);
        for axis in 0..3 {
            for sign in [-1, 1] {
                let mut n = key;
                n[axis] += sign;
                self.dirty.insert(n);
            }
        }
    }

    /// Chunk at `key`, if it exists.
    pub fn chunk_at(&self, key: IVec3) -> Option<&Chunk> {
        self.chunks.get(&key)
    }

    /// Iterate all existing chunks.
    pub fn chunks(&self) -> impl Iterator<Item = (IVec3, &Chunk)> {
        self.chunks.iter().map(|(k, c)| (*k, c))
    }

    /// Number of chunks currently allocated.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Take the set of chunks needing a remesh. Includes neighbor keys that
    /// may not exist in the map; consumers skip those.
    pub fn drain_dirty(&mut self) -> Vec<IVec3> {
        self.dirty.drain().collect()
    }

    /// Take the voxel regions edited since the last drain (physics wake-ups).
    pub fn drain_dirty_regions(&mut self) -> Vec<DirtyRegion> {
        std::mem::take(&mut self.dirty_regions)
    }

    /// Mark `key` dirty, plus any face neighbor whose mesh can see `local`
    /// (border voxels affect the neighbor's face culling and AO).
    fn mark_dirty_with_neighbors(&mut self, key: IVec3, local: UVec3) {
        self.dirty.insert(key);
        for axis in 0..3 {
            if local[axis] == 0 {
                let mut n = key;
                n[axis] -= 1;
                self.dirty.insert(n);
            } else if local[axis] == CHUNK as u32 - 1 {
                let mut n = key;
                n[axis] += 1;
                self.dirty.insert(n);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STONE: Voxel = Voxel(1);
    const DIRT: Voxel = Voxel(2);

    fn world() -> World {
        // 1 m voxels over 128 m => 128 voxels => 4 chunks per axis.
        World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [128.0, 128.0, 128.0],
            ..WorldConfig::default()
        })
    }

    #[test]
    fn cross_chunk_set_get() {
        let mut w = world();
        w.set_voxel(IVec3::new(31, 5, 5), STONE);
        w.set_voxel(IVec3::new(32, 5, 5), DIRT);

        assert_eq!(w.get_voxel(IVec3::new(31, 5, 5)), STONE);
        assert_eq!(w.get_voxel(IVec3::new(32, 5, 5)), DIRT);
        assert_eq!(w.chunk_count(), 2, "voxels land in different chunks");
        assert!(w.chunk_at(IVec3::new(0, 0, 0)).is_some());
        assert!(w.chunk_at(IVec3::new(1, 0, 0)).is_some());
    }

    #[test]
    fn absent_chunk_reads_air() {
        let w = world();
        assert_eq!(w.get_voxel(IVec3::new(5, 5, 5)), AIR);
        assert!(!w.solid(IVec3::new(5, 5, 5)));
    }

    #[test]
    fn set_marks_chunk_dirty_and_drain_empties() {
        let mut w = world();
        w.set_voxel(IVec3::new(40, 40, 40), STONE);

        let dirty = w.drain_dirty();
        assert!(dirty.contains(&IVec3::new(1, 1, 1)));
        assert!(w.drain_dirty().is_empty(), "drain must empty the set");
    }

    #[test]
    fn interior_edit_dirties_only_its_chunk() {
        let mut w = world();
        w.set_voxel(IVec3::new(40, 40, 40), STONE); // local (8,8,8)
        let dirty = w.drain_dirty();
        assert_eq!(dirty, vec![IVec3::new(1, 1, 1)]);
    }

    #[test]
    fn border_edit_dirties_face_neighbors() {
        let mut w = world();
        // (32, 40, 63): local (0, 8, 31) in chunk (1, 1, 1).
        w.set_voxel(IVec3::new(32, 40, 63), STONE);
        let dirty: HashSet<_> = w.drain_dirty().into_iter().collect();
        let expected: HashSet<_> = [
            IVec3::new(1, 1, 1),
            IVec3::new(0, 1, 1), // local.x == 0
            IVec3::new(1, 1, 2), // local.z == 31
        ]
        .into_iter()
        .collect();
        assert_eq!(dirty, expected);
    }

    #[test]
    fn corner_edit_dirties_three_neighbors() {
        let mut w = world();
        // (32, 32, 32): local (0, 0, 0) in chunk (1, 1, 1).
        w.set_voxel(IVec3::new(32, 32, 32), STONE);
        let dirty = w.drain_dirty();
        assert_eq!(dirty.len(), 4, "chunk + three face neighbors: {dirty:?}");
        for key in [
            IVec3::new(1, 1, 1),
            IVec3::new(0, 1, 1),
            IVec3::new(1, 0, 1),
            IVec3::new(1, 1, 0),
        ] {
            assert!(dirty.contains(&key), "missing {key} in {dirty:?}");
        }
    }

    #[test]
    fn same_value_write_marks_nothing() {
        let mut w = world();
        w.set_voxel(IVec3::new(10, 10, 10), STONE);
        w.drain_dirty();
        w.drain_dirty_regions();

        w.set_voxel(IVec3::new(10, 10, 10), STONE);
        assert!(w.drain_dirty().is_empty());
        assert!(w.drain_dirty_regions().is_empty());

        // An air write into an absent chunk must not allocate one.
        let count = w.chunk_count();
        w.set_voxel(IVec3::new(100, 100, 100), AIR);
        assert!(w.drain_dirty().is_empty());
        assert_eq!(w.chunk_count(), count, "no chunk for a no-op air write");
    }

    #[test]
    fn fill_box_spans_chunks_and_reports_one_region() {
        let mut w = world();
        let min = IVec3::new(30, 30, 30);
        let max = IVec3::new(35, 35, 35);
        w.fill_box(min, max, STONE);

        for y in 30..35 {
            for z in 30..35 {
                for x in 30..35 {
                    assert_eq!(w.get_voxel(IVec3::new(x, y, z)), STONE);
                }
            }
        }
        assert_eq!(w.get_voxel(IVec3::new(35, 30, 30)), AIR, "max is exclusive");
        assert_eq!(w.chunk_count(), 8, "box straddles the chunk corner");
        assert_eq!(w.drain_dirty_regions(), vec![(min, max)]);
    }

    #[test]
    fn out_of_bounds_reads_air_and_writes_are_ignored() {
        let mut w = world();
        assert_eq!(w.get_voxel(IVec3::new(-1, 0, 0)), AIR);
        assert_eq!(w.get_voxel(IVec3::new(0, 128, 0)), AIR);

        w.set_voxel(IVec3::new(-1, 0, 0), STONE);
        w.set_voxel(IVec3::new(128, 0, 0), STONE);
        assert_eq!(w.chunk_count(), 0, "no chunk allocated for ignored writes");
        assert!(w.drain_dirty().is_empty());
        assert!(w.drain_dirty_regions().is_empty());
    }

    #[test]
    fn edits_record_dirty_regions() {
        let mut w = world();
        let pos = IVec3::new(7, 8, 9);
        w.set_voxel(pos, STONE);
        assert_eq!(w.drain_dirty_regions(), vec![(pos, pos + IVec3::ONE)]);
    }

    #[test]
    fn insert_chunk_dirties_self_and_six_neighbors() {
        let mut w = world();
        w.insert_chunk(IVec3::new(1, 1, 1), Chunk::uniform(STONE));

        assert_eq!(w.get_voxel(IVec3::new(40, 40, 40)), STONE);
        let dirty: HashSet<_> = w.drain_dirty().into_iter().collect();
        assert_eq!(dirty.len(), 7);
        assert!(dirty.contains(&IVec3::new(1, 1, 1)));
        assert!(dirty.contains(&IVec3::new(2, 1, 1)));
        assert!(dirty.contains(&IVec3::new(1, 0, 1)));
    }
}
