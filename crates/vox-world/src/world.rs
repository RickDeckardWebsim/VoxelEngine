//! The sparse voxel world: a chunk map with an edit API, dirty-chunk tracking
//! for remeshing, and dirty-region tracking for physics wake-ups.
//!
//! Chunks live in a `HashMap` — nothing assumes the world is a box — but the
//! MVP enforces a finite extent from [`WorldConfig`]: reads outside are air,
//! writes outside are ignored.

use glam::{IVec3, UVec3};
use vox_core::coords::{CHUNK, chunk_of, local_of};
// FxHashMap over the std default: the chunk map is consulted on *every*
// voxel read engine-wide (contacts, raycasts, carves, floods), where
// SipHash's collision-flood hardening buys nothing and costs plenty.
use vox_core::{FxHashMap, FxHashSet, WorldConfig, chunk_origin};

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

    /// The world's configuration (voxel size, extent, etc.) -- lets callers
    /// that already hold a [`SolidLookup`] read geometry constants without a
    /// separate `&World` borrow.
    #[inline]
    pub fn world_cfg(&self) -> &WorldConfig {
        &self.world.cfg
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
        match chunk.and_then(|c| {
            let voxel = c.get(local_of(v));
            (voxel != AIR).then_some(voxel)
        }) {
            Some(voxel) => self.world.material_is_solid(voxel.0),
            None => false,
        }
    }

    /// True when the voxel at `v` is present (non-air), regardless of
    /// solidity. Used by the connectivity flood to traverse non-solid
    /// materials (leaves, water) that are still physically present in
    /// the grid and should detach with the structure they're part of.
    pub fn present(&mut self, v: IVec3) -> bool {
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
        match chunk {
            Some(c) => c.get(local_of(v)) != AIR,
            None => false,
        }
    }
}

/// A half-open voxel-space box `[min, max)` touched by an edit.
pub type DirtyRegion = (IVec3, IVec3);

/// The sparse voxel world.
pub struct World {
    /// Per-world configuration (voxel scale, extent, seed).
    pub cfg: WorldConfig,
    chunks: FxHashMap<IVec3, Chunk>,
    dirty: FxHashSet<IVec3>,
    dirty_regions: Vec<DirtyRegion>,
    /// Half-open world bounds in voxels, `[min, max)`.
    bounds_voxels: (IVec3, IVec3),
    warned_out_of_bounds: bool,
    /// Per-material-id solidity, indexed by `Voxel.0`. `None` (the default)
    /// means "any non-air voxel is solid" -- today's behavior, preserved for
    /// every caller that never attaches a real table. `Some(table)` is set
    /// once, in production, from the actual `MaterialRegistry` (see
    /// `set_solid_table`) so a non-solid, non-air material (water) reads
    /// correctly everywhere solidity is checked: raycasts, the character
    /// controller, rigidbody-vs-world contacts, and the destruction
    /// connectivity flood all key off `World::solid`/`SolidLookup::solid`.
    solid_table: Option<Vec<bool>>,
    /// When set, `set_voxel` silently drops writes outside this half-open
    /// voxel box. Used during chunk generation to prevent tree stamping from
    /// allocating neighbor chunks. `get_voxel` is unaffected — reads always
    /// pass through. `None` = no clip (normal gameplay).
    clip: Option<(IVec3, IVec3)>,
    /// Chunks that have received gameplay writes (player tools, bombs, fire,
    /// weathering). Pristine chunks (not in this set) can be evicted and
    /// regenerated; edited chunks must persist in memory.
    edited: FxHashSet<IVec3>,
    /// When true, `set_voxel`/`edit_box` skip marking chunks as edited. Set
    /// during chunk generation (terrain insert + tree stamping) so generated
    /// chunks remain pristine and evictable. Gameplay writes always have this
    /// false.
    suppress_edit_tracking: bool,
}

impl World {
    /// An empty world for `cfg` (all air, no chunks allocated).
    pub fn new(cfg: WorldConfig) -> Self {
        let bounds_voxels = (IVec3::ZERO, cfg.extent_voxels());
        Self {
            cfg,
            chunks: FxHashMap::default(),
            dirty: FxHashSet::default(),
            dirty_regions: Vec::new(),
            bounds_voxels,
            warned_out_of_bounds: false,
            solid_table: None,
            clip: None,
            edited: FxHashSet::default(),
            suppress_edit_tracking: false,
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

    /// True when the voxel at `v` is solid: non-air, and (if a solidity
    /// table is attached) not a material marked non-solid in that table.
    pub fn solid(&self, v: IVec3) -> bool {
        let voxel = self.get_voxel(v);
        voxel != AIR && self.material_is_solid(voxel.0)
    }

    /// Attach a per-material-id solidity table (index = `Voxel.0`; build it
    /// from `MaterialRegistry` by mapping each id to its `MaterialDef.solid`).
    /// An id past the end of `table` reads as solid -- an out-of-range id
    /// should never happen in practice, and defaulting to *solid* is the
    /// safe failure direction (a stray id blocks movement/collision rather
    /// than silently letting the player walk through undefined material).
    pub fn set_solid_table(&mut self, table: Vec<bool>) {
        self.solid_table = Some(table);
    }

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

    /// True when material id `id` counts as solid, consulting the attached
    /// table if there is one. Shared by `solid()` and `SolidLookup::solid()`
    /// so the two can never disagree.
    fn material_is_solid(&self, id: u16) -> bool {
        match &self.solid_table {
            Some(table) => table.get(id as usize).copied().unwrap_or(true),
            None => true, // legacy fallback: any non-air voxel is solid
        }
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
        if let Some((clip_min, clip_max)) = self.clip {
            if pos.cmplt(clip_min).any() || pos.cmpge(clip_max).any() {
                return;
            }
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
        if !self.suppress_edit_tracking {
            self.edited.insert(key);
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
        let mut min = min.max(bmin);
        let mut max = max.min(bmax);
        if let Some((clip_min, clip_max)) = self.clip {
            min = min.max(clip_min);
            max = max.min(clip_max);
        }
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
                    if !self.suppress_edit_tracking {
                        self.edited.insert(key);
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

    /// Remove a chunk from the map, dirtying itself and its six face neighbors
    /// (neighbors' meshes sample this chunk for face culling, and the chunk's
    /// own mesh must be dropped). No-op if the chunk is absent. Also removes
    /// it from the edited set — an evicted chunk is gone.
    pub fn remove_chunk(&mut self, key: IVec3) {
        if self.chunks.remove(&key).is_none() {
            return;
        }
        self.edited.remove(&key);
        // Dirty self (so its mesh is dropped) + neighbors (their face culling
        // changes when an adjacent chunk disappears).
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
    use std::collections::HashSet;

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

    #[test]
    fn solid_defaults_to_legacy_non_air_behavior_with_no_registry_attached() {
        let mut w = World::new(WorldConfig::default());
        w.set_voxel(IVec3::new(1, 1, 1), Voxel(1));
        assert!(
            w.solid(IVec3::new(1, 1, 1)),
            "any non-air voxel is solid by default"
        );
    }

    #[test]
    fn solid_consults_the_registry_once_attached() {
        let mut w = World::new(WorldConfig::default());
        // Material id 1 = solid, id 2 = non-solid (mirrors a real registry's
        // shape without depending on vox-core's MaterialRegistry type, which
        // vox-world does not otherwise need at runtime for this).
        w.set_solid_table(vec![false, true, false]); // [air, solid_mat, water_mat]
        w.set_voxel(IVec3::new(1, 1, 1), Voxel(1));
        w.set_voxel(IVec3::new(2, 2, 2), Voxel(2));
        assert!(
            w.solid(IVec3::new(1, 1, 1)),
            "material id 1 is marked solid"
        );
        assert!(
            !w.solid(IVec3::new(2, 2, 2)),
            "material id 2 is marked non-solid"
        );
    }

    #[test]
    fn solid_table_out_of_range_material_id_falls_back_to_solid() {
        let mut w = World::new(WorldConfig::default());
        w.set_solid_table(vec![false, true]); // only ids 0..=1 known
        w.set_voxel(IVec3::new(1, 1, 1), Voxel(5)); // id 5 has no table entry
        assert!(
            w.solid(IVec3::new(1, 1, 1)),
            "unknown material ids default solid, never silently pass through"
        );
    }

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
        assert!(!w.chunks().any(|(k, _)| k == IVec3::new(1, 0, 0)),
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
    let edited_before = w.is_edited(IVec3::new(0, 0, 0));
    w.set_voxel(IVec3::new(10, 10, 10), STONE); // same value
    assert!(edited_before); // was edited
    // No new chunks marked:
    assert_eq!(w.edited_count(), 1, "same-value write should not add edited chunks");
}
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
}
