//! The active-cell fluid automaton. See the crate root docs and
//! `docs/plans/2026-07-09-fluid-sim-design.md` for the rationale.

use glam::IVec3;
use vox_core::FxHashSet;
use vox_world::{AIR, Voxel, World};

const NEIGHBORS_6: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

/// Hard ceiling on active cells processed in one `tick` call -- mirrors
/// `MAX_DEBRIS_BODIES`'s budget pattern. A tick given more active cells than
/// this processes exactly this many (in randomized order, so it's not
/// always the same subset stalling) and leaves the rest active for the next
/// tick, spreading a huge flood event (a blasted-open reservoir) over a few
/// extra ticks instead of spiking one frame.
const FLUID_TICK_BUDGET: usize = 4096;

/// How far sideways a blocked cell searches for a reachable drop (an open
/// cell with air beneath it) before giving up and settling. Larger values
/// flatten mounds over a wider area per wake, at O(8 * horizon) extra
/// lookups per stuck-but-active cell per tick. Purely a grid count -- no
/// meters involved, so scale invariance is preserved.
const FLOW_HORIZON: i32 = 8;

/// Tracks which water cells are still moving. A cell not in this set is
/// settled and costs nothing to tick -- the entire performance story of
/// this crate (mirrors `PhysicsWorld`'s sleep bookkeeping).
pub struct FluidSim {
    active: FxHashSet<IVec3>,
    /// The single voxel material this sim treats as water. Set once at
    /// construction -- never inferred from the active set (see `tick` and
    /// `wake_region` doc comments for why inference was fragile).
    water: Voxel,
    /// xorshift64* state for randomized per-tick update order (same
    /// construction as `PhysicsWorld::lifetime_rng` / `ParticleSystem`'s
    /// spawn jitter) -- avoids a visible left/right or diagonal bias in how
    /// water spreads.
    rng: u64,
}

impl FluidSim {
    pub fn new(water: Voxel) -> Self {
        Self {
            active: FxHashSet::default(),
            water,
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Number of cells currently flowing (debug-overlay stat).
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// xorshift64* -- deterministic, dependency-free.
    fn next_u64(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    /// Fill a sphere of radius `radius_vox` centered on `center` with
    /// `water_material`, skipping any cell that isn't currently air (never
    /// carves through existing terrain), and mark every filled cell active.
    /// `radius_vox` is in voxels, not meters -- the caller (the placement
    /// tool) converts from the tool's meter-radius via the world's
    /// `voxel_size_m`, same convention as `vox_physics::carve_sphere`.
    pub fn place_blob(
        &mut self,
        world: &mut World,
        center: IVec3,
        radius_vox: i32,
        water_material: Voxel,
    ) {
        let r = radius_vox.max(0);
        let min = center - IVec3::splat(r);
        let max = center + IVec3::splat(r + 1);
        let r2 = (r * r) as i64;
        let mut filled = Vec::new();
        world.edit_box(min, max, |v, cur| {
            if cur != AIR {
                return None;
            }
            let d = v - center;
            let dist2 = (d.x as i64 * d.x as i64) + (d.y as i64 * d.y as i64) + (d.z as i64 * d.z as i64);
            if dist2 > r2 {
                return None;
            }
            filled.push(v);
            Some(water_material)
        });
        self.active.extend(filled);
    }

    /// Advance every active cell by one tick: move it per `step_cell`, or
    /// drop it from the active set if it has settled. A cell that moves
    /// reactivates its destination plus water neighboring either the old or
    /// new position -- the entire wake cascade, no separate propagation pass
    /// needed.
    pub fn tick(&mut self, world: &mut World) -> usize {
        let water = self.water;

        // Snapshot exactly the positions that hold real water *before* any
        // mutation this tick, and process only those. A live re-check of
        // `world.get_voxel(pos) == water` inside the loop (as opposed to
        // this upfront filter) is unsound: a mover's own wake neighbors
        // include its future down-cell, so if that neighbor is iterated
        // after the mover in the same `for` loop, the live check would see
        // freshly-written water there and move it *again* within this same
        // tick -- a cascade whose length depends on arbitrary hash-set
        // iteration order. Filtering up front means every entry in `cells`
        // is only ever cleared by its own move below, never by another
        // mover's (a mover can never target a cell that already holds
        // water -- `is_open` only accepts AIR), so each snapshot position
        // is guaranteed to still be valid when its turn comes.
        let mut cells: Vec<IVec3> = self
            .active
            .iter()
            .copied()
            .filter(|&p| world.get_voxel(p) == water)
            .collect();

        let mut next_active = FxHashSet::default();

        // Budget: if there are more active (real-water) cells than we're
        // willing to process this call, shuffle so the overflow isn't
        // always the same tail of whatever order the hash set happened to
        // yield, then carry the overflow straight into `next_active`
        // untouched. Each carried-over position still holds real water (it
        // was filtered above and nothing this tick has mutated it yet), so
        // it is exactly as valid for next tick's snapshot filter as any
        // settled/woken cell already in `next_active` -- it's simply
        // processed on a later call instead of this one.
        if cells.len() > FLUID_TICK_BUDGET {
            for i in (1..cells.len()).rev() {
                let j = (self.next_u64() as usize) % (i + 1);
                cells.swap(i, j);
            }
            let overflow = cells.split_off(FLUID_TICK_BUDGET);
            next_active.extend(overflow);
        }
        let processed = cells.len();

        for pos in cells {
            let coin = self.next_u64() & 1 == 0;
            let coin2 = self.next_u64() & 1 == 0;
            let dest = {
                let mut is_open = |p: IVec3| world.in_bounds(p) && world.get_voxel(p) == AIR;
                let mut is_supported = |p: IVec3| {
                    world.in_bounds(p) && (world.solid(p) || world.get_voxel(p) == water)
                };
                let has_water_above = world.get_voxel(pos + IVec3::Y) == water;
                step_cell(pos, &mut is_open, &mut is_supported, has_water_above, coin, coin2)
            };
            if let Some(dest) = dest {
                world.set_voxel(pos, AIR);
                world.set_voxel(dest, water);
                next_active.insert(dest);
                // Wake only actual water, not generic non-solid cells (air
                // used to be inserted here and inflated `active_count()` by
                // one stale entry per neighboring empty voxel). Include the
                // old position's neighbors too: if an upper cell was visited
                // earlier in this snapshot, moving its support out from
                // under it must give it another turn next tick.
                for changed in [pos, dest] {
                    for n in NEIGHBORS_6 {
                        let neighbor = changed + n;
                        if world.get_voxel(neighbor) == water {
                            next_active.insert(neighbor);
                        }
                    }
                }
            } // else: settled -- not re-added
        }
        self.active = next_active;
        processed
    }

    /// Reactivate water inside or directly adjacent to `[min, max)`. World
    /// edits report the cells they changed, rather than their neighbors: a
    /// wall voxel just changed to air contains no water itself, so the
    /// one-cell halo is what wakes the settled water immediately against that
    /// new opening. Bounds are clipped before scanning.
    pub fn wake_region(&mut self, world: &World, min: IVec3, max: IVec3) {
        let (bounds_min, bounds_max) = world.bounds_voxels();
        let min = (min.max(bounds_min) - IVec3::ONE).max(bounds_min);
        let max = (max.min(bounds_max) + IVec3::ONE).min(bounds_max);
        for x in min.x..max.x {
            for y in min.y..max.y {
                for z in min.z..max.z {
                    let p = IVec3::new(x, y, z);
                    if world.get_voxel(p) == self.water {
                        self.active.insert(p);
                    }
                }
            }
        }
    }
}

/// Where a water cell at `pos` wants to move this tick, or `None` if it has
/// nowhere to go (should settle). `is_open` reports whether a cell can be
/// flowed into: empty (air) and in-bounds. Order of preference: straight
/// down, then diagonal-down (randomized left/right), then one step toward
/// the nearest drop reachable within `FLOW_HORIZON` cells sideways, then
/// pressure-gated sideways leveling onto supported terrain or water
/// (randomized left/right, then front/back) -- see the design doc §4 for
/// why this shape and not fractional pressure levels.
fn step_cell(
    pos: IVec3,
    is_open: &mut impl FnMut(IVec3) -> bool,
    is_supported: &mut impl FnMut(IVec3) -> bool,
    has_water_above: bool,
    coin: bool,
    coin2: bool,
) -> Option<IVec3> {
    let down = pos + IVec3::NEG_Y;
    if is_open(down) {
        return Some(down);
    }

    let (dx1, dx2) = if coin { (1, -1) } else { (-1, 1) };
    for dx in [dx1, dx2] {
        let diag = pos + IVec3::new(dx, -1, 0);
        if is_open(diag) {
            return Some(diag);
        }
    }
    let (dz1, dz2) = if coin2 { (1, -1) } else { (-1, 1) };
    for dz in [dz1, dz2] {
        let diag = pos + IVec3::new(0, -1, dz);
        if is_open(diag) {
            return Some(diag);
        }
    }

    // Flow: no immediate fall exists, so search all eight horizontal
    // directions -- the four axes, then the four diagonals, each group in
    // randomized order -- for a reachable drop -- an open run of same-height
    // cells ending in one with air beneath -- and take one step toward the
    // first one found. Diagonal rays deliberately check only the cells on
    // the ray, never the two orthogonal neighbors, so water can slip
    // through the seam where two solid blocks touch only at a corner --
    // accepted for a coarse voxel fluid, not an oversight.
    // This is what keeps a mound from freezing into a stable
    // stepped pyramid: its surface cells can walk over the water below them
    // until they reach the pile's edge and fall off. Unlike an
    // unconditional sideways shuffle, this only ever moves when a strictly
    // lower destination is reachable, so a flat sheet or a full basin still
    // has nowhere to go and sleeps.
    let dirs = flow_dirs(coin, coin2);
    for dir in dirs {
        for k in 1..=FLOW_HORIZON {
            let q = pos + dir * k;
            if !is_open(q) {
                break; // wall or water blocks this direction
            }
            if is_open(q + IVec3::NEG_Y) {
                return Some(pos + dir); // one step toward the drop
            }
        }
    }

    // A full/empty grid cannot represent a fractional, one-cell-deep water
    // surface. Letting an unpressurized surface cell trade places with any
    // equally-high air cell makes a partially filled puddle random-walk
    // forever. Require water directly above the source, and support below
    // the destination, so a stack can level across a shorter neighboring
    // water column while a shallow resting surface can sleep instead of
    // shuffling indefinitely.
    if !has_water_above {
        return None;
    }
    let (sx1, sx2) = if coin { (1, -1) } else { (-1, 1) };
    for dx in [sx1, sx2] {
        let side = pos + IVec3::new(dx, 0, 0);
        if is_open(side) && is_supported(side + IVec3::NEG_Y) {
            return Some(side);
        }
    }
    let (sz1, sz2) = if coin2 { (1, -1) } else { (-1, 1) };
    for dz in [sz1, sz2] {
        let side = pos + IVec3::new(0, 0, dz);
        if is_open(side) && is_supported(side + IVec3::NEG_Y) {
            return Some(side);
        }
    }

    None
}

/// The eight horizontal step directions: the four axis dirs first (shorter
/// true distance), then the four diagonals. `coin` picks the sign order
/// within each group, `coin2` picks which axis/diagonal pair leads -- same
/// de-biasing role the coins already play in the fall/spread rules.
fn flow_dirs(coin: bool, coin2: bool) -> [IVec3; 8] {
    let (x1, x2) = if coin { (IVec3::X, IVec3::NEG_X) } else { (IVec3::NEG_X, IVec3::X) };
    let (z1, z2) = if coin { (IVec3::Z, IVec3::NEG_Z) } else { (IVec3::NEG_Z, IVec3::Z) };
    let (d1, d2) = if coin {
        (IVec3::new(1, 0, 1), IVec3::new(-1, 0, -1))
    } else {
        (IVec3::new(-1, 0, -1), IVec3::new(1, 0, 1))
    };
    let (d3, d4) = if coin {
        (IVec3::new(1, 0, -1), IVec3::new(-1, 0, 1))
    } else {
        (IVec3::new(-1, 0, 1), IVec3::new(1, 0, -1))
    };
    if coin2 { [x1, x2, z1, z2, d1, d2, d3, d4] } else { [z1, z2, x1, x2, d3, d4, d1, d2] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::WorldConfig;

    const WATER: Voxel = Voxel(1);

    fn test_world() -> World {
        let mut w = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [16.0, 16.0, 16.0],
            ..WorldConfig::default()
        });
        w.set_solid_table(vec![false, false, true]); // [air, water, floor]
        let (_, max) = w.bounds_voxels();
        w.fill_box(IVec3::new(0, 0, 0), IVec3::new(max.x, 5, max.z), Voxel(2)); // floor top at y=5
        w
    }

    #[test]
    fn new_sim_has_no_active_cells() {
        let sim = FluidSim::new(WATER);
        assert_eq!(sim.active_count(), 0);
    }

    #[test]
    fn place_blob_fills_a_sphere_with_water_and_activates_every_filled_cell() {
        let mut world = test_world();
        let mut sim = FluidSim::new(WATER);
        let center = IVec3::new(8, 8, 8);
        sim.place_blob(&mut world, center, 2, WATER);

        assert_eq!(world.get_voxel(center), WATER, "center must be filled");
        assert_ne!(
            world.get_voxel(IVec3::new(8, 8, 8) + IVec3::new(10, 0, 0)),
            WATER,
            "far outside the radius must stay air"
        );
        assert!(sim.active_count() > 0, "every filled cell must be active");
    }

    #[test]
    fn place_blob_does_not_overwrite_existing_solid_terrain() {
        let mut world = test_world();
        world.set_voxel(IVec3::new(8, 8, 8), Voxel(2)); // pretend id 2 = stone
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, IVec3::new(8, 8, 8), 2, WATER);
        assert_eq!(
            world.get_voxel(IVec3::new(8, 8, 8)),
            Voxel(2),
            "a blob must not carve through existing terrain"
        );
    }

    #[test]
    fn a_single_cell_falls_under_gravity() {
        let mut world = test_world();
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, IVec3::new(8, 10, 8), 0, WATER); // 1 cell
        assert_eq!(world.get_voxel(IVec3::new(8, 10, 8)), WATER);

        for _ in 0..5 {
            sim.tick(&mut world);
        }
        assert_eq!(world.get_voxel(IVec3::new(8, 10, 8)), AIR, "must have left the start cell");
        assert_eq!(world.get_voxel(IVec3::new(8, 5, 8)), WATER, "must have fallen to the floor at y=5");
    }

    #[test]
    fn active_set_contains_only_water_after_a_move() {
        let mut world = test_world();
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, IVec3::new(8, 6, 8), 0, WATER);

        sim.tick(&mut world);

        assert_eq!(world.get_voxel(IVec3::new(8, 5, 8)), WATER);
        assert_eq!(sim.active_count(), 1, "only the moved water cell should remain active");
    }

    #[test]
    fn water_settles_on_a_flat_floor_and_leaves_the_active_set() {
        let mut world = test_world();
        let center = IVec3::new(8, 5, 8);
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, center, 0, WATER);
        sim.tick(&mut world);
        assert_eq!(sim.active_count(), 0, "a shallow cell on solid ground must sleep");
    }

    #[test]
    fn wake_region_reactivates_settled_water_inside_it() {
        let mut world = test_world();
        let center = IVec3::new(8, 5, 8);
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, center, 0, WATER);
        sim.tick(&mut world);
        assert_eq!(sim.active_count(), 0, "must have settled first");

        sim.wake_region(&world, IVec3::new(7, 4, 7), IVec3::new(10, 6, 10));
        assert!(sim.active_count() > 0, "water inside the woken region must reactivate");
    }

    #[test]
    fn wake_region_reactivates_water_adjacent_to_an_exact_dirty_cell() {
        let mut world = test_world();
        let center = IVec3::new(8, 5, 8);
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, center, 0, WATER);
        sim.tick(&mut world);
        assert_eq!(sim.active_count(), 0, "must have settled first");

        // Build a wall after the water has settled, discard that setup edit,
        // then remove exactly one wall cell. `World` reports only that one
        // changed cell; the sim must include its neighboring water itself.
        let wall = center + IVec3::X;
        world.drain_dirty_regions();
        world.set_voxel(wall, Voxel(2));
        world.drain_dirty_regions();
        world.set_voxel(wall, AIR);
        for (min, max) in world.drain_dirty_regions() {
            sim.wake_region(&world, min, max);
        }

        assert!(sim.active_count() > 0, "an adjacent water cell must wake for an exact dirty region");
    }

    #[test]
    fn wake_region_does_not_touch_settled_water_outside_it() {
        let mut world = test_world();
        let center = IVec3::new(8, 5, 8);
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, center, 0, WATER);
        sim.tick(&mut world);
        assert_eq!(sim.active_count(), 0, "must have settled first");

        sim.wake_region(&world, IVec3::new(0, 0, 0), IVec3::new(2, 2, 2)); // nowhere near the water
        assert_eq!(sim.active_count(), 0, "an unrelated edit must not wake distant settled water");
    }

    #[test]
    fn water_spreads_sideways_across_a_flat_floor() {
        // A multi-cell blob has vertical head, so its lower cells can spread
        // onto the solid floor. A single shallow cell deliberately sleeps.
        let mut world = test_world();
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, IVec3::new(8, 8, 8), 2, WATER);
        for _ in 0..40 {
            sim.tick(&mut world);
        }
        let neighbor_has_water = [IVec3::new(7, 5, 8), IVec3::new(9, 5, 8), IVec3::new(8, 5, 7), IVec3::new(8, 5, 9)]
            .iter()
            .any(|&p| world.get_voxel(p) == WATER);
        assert!(neighbor_has_water, "water must spread onto at least one flat neighbor over 40 ticks");
    }

    #[test]
    fn tall_column_levels_across_a_shorter_water_column() {
        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [16.0, 16.0, 16.0],
            ..WorldConfig::default()
        });
        world.set_solid_table(vec![false, false, true]); // [air, water, stone]
        let (_, max) = world.bounds_voxels();
        world.fill_box(IVec3::ZERO, IVec3::new(max.x, 5, max.z), Voxel(2));

        // A two-column, one-cell-wide trough. The only horizontal route is
        // between x=7 and x=8, so a 3-high column beside a 1-high column
        // must relax into two equally high columns rather than avalanche.
        world.fill_box(IVec3::new(6, 5, 7), IVec3::new(10, 9, 10), Voxel(2));
        world.fill_box(IVec3::new(7, 5, 8), IVec3::new(9, 8, 9), AIR);

        let mut sim = FluidSim::new(WATER);
        for p in [
            IVec3::new(7, 5, 8),
            IVec3::new(7, 6, 8),
            IVec3::new(7, 7, 8),
            IVec3::new(8, 5, 8),
        ] {
            sim.place_blob(&mut world, p, 0, WATER);
        }

        for _ in 0..10 {
            sim.tick(&mut world);
            if sim.active_count() == 0 {
                break;
            }
        }

        for p in [
            IVec3::new(7, 5, 8),
            IVec3::new(7, 6, 8),
            IVec3::new(8, 5, 8),
            IVec3::new(8, 6, 8),
        ] {
            assert_eq!(world.get_voxel(p), WATER, "the two columns must level to height two");
        }
        assert_eq!(sim.active_count(), 0, "the leveled surface must sleep");
    }

    #[test]
    fn a_dropped_blob_flattens_into_a_sheet_instead_of_piling() {
        // The sand-pile regression: a blob dropped onto a wide open floor
        // used to settle as a stable stepped pyramid, because surface cells
        // (no water above) could never move sideways and their diagonals
        // were already water. Real water must keep flowing outward until it
        // is one cell deep -- no cell may end up resting on top of another.
        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [40.0, 40.0, 40.0],
            ..WorldConfig::default()
        });
        world.set_solid_table(vec![false, false, true]); // [air, water, stone]
        let (_, max) = world.bounds_voxels();
        world.fill_box(IVec3::ZERO, IVec3::new(max.x, 5, max.z), Voxel(2)); // floor top at y=5

        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, IVec3::new(20, 9, 20), 2, WATER);
        let before = count_water(&world);

        for _ in 0..400 {
            sim.tick(&mut world);
            if sim.active_count() == 0 {
                break;
            }
        }

        assert_eq!(sim.active_count(), 0, "the spread sheet must sleep");
        assert_eq!(count_water(&world), before, "flattening must conserve water");
        let (min, max) = world.bounds_voxels();
        for x in min.x..max.x {
            for y in min.y..max.y {
                for z in min.z..max.z {
                    let p = IVec3::new(x, y, z);
                    if world.get_voxel(p) == WATER {
                        assert_ne!(
                            world.get_voxel(p + IVec3::Y),
                            WATER,
                            "no water may rest on other water on an open floor (pile at {p})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn partially_filled_basin_sleeps_after_spreading() {
        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [20.0, 20.0, 20.0],
            ..WorldConfig::default()
        });
        world.set_solid_table(vec![false, false, true]); // [air, water, stone]
        let (_, max) = world.bounds_voxels();
        world.fill_box(IVec3::ZERO, IVec3::new(max.x, 5, max.z), Voxel(2));

        // An 8x8 open basin with room well above the blob. Its bottom holds
        // only 64 cells, while the radius-3 blob contains more, so the final
        // state necessarily has a partially filled upper layer.
        world.fill_box(IVec3::new(3, 5, 3), IVec3::new(13, 16, 13), Voxel(2));
        world.fill_box(IVec3::new(4, 5, 4), IVec3::new(12, 16, 12), AIR);

        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, IVec3::new(8, 9, 8), 3, WATER);
        let before = count_water(&world);
        assert!(before > 64, "the test needs more water than one basin layer holds");

        for _ in 0..300 {
            sim.tick(&mut world);
            if sim.active_count() == 0 {
                break;
            }
        }

        assert_eq!(sim.active_count(), 0, "a partially filled basin must eventually sleep");
        assert_eq!(count_water(&world), before, "settling must conserve water cells");
    }

    #[test]
    fn total_water_cell_count_is_conserved_across_many_ticks() {
        let mut world = test_world();
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, IVec3::new(8, 10, 8), 2, WATER);
        let before = count_water(&world);
        for _ in 0..60 {
            sim.tick(&mut world);
        }
        let after = count_water(&world);
        assert_eq!(before, after, "moving water must never create or destroy cells");
    }

    #[test]
    fn tick_never_processes_more_than_the_budget_in_one_call() {
        // `test_world()` is only 16 voxels per axis, far too small to hold a
        // sphere wide enough to clear `FLUID_TICK_BUDGET` (4096) cells --
        // volume of a radius-r sphere is ~(4/3)*pi*r^3, so r must exceed
        // ~9.93 (since (4/3)*pi*9.93^3 ~= 4096); r=12 gives ~7238 cells,
        // comfortably over budget without being enormous. That needs a
        // world spanning at least 2*12+1 = 25 voxels per axis around the
        // blob's center, so this test builds its own larger, self-contained
        // world rather than changing the shared `test_world()` helper (used
        // by other tests that may depend on its current 16-voxel extent).
        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [40.0, 40.0, 40.0],
            ..WorldConfig::default()
        });
        world.set_solid_table(vec![false, false, true]); // [air, water, floor]
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, IVec3::new(20, 20, 20), 12, WATER);
        assert!(
            sim.active_count() > FLUID_TICK_BUDGET,
            "test needs more active cells than the budget to be meaningful: got {}",
            sim.active_count()
        );

        let processed = sim.tick(&mut world);
        assert!(processed <= FLUID_TICK_BUDGET, "must not process more than the budget in one tick: {processed}");
    }

    #[test]
    fn spread_pattern_is_identical_in_cell_counts_at_any_voxel_scale() {
        // The algorithm operates purely on grid adjacency -- voxel_size_m
        // never enters step_cell. Placing the same single cell (in voxel
        // counts) at two different scales and running the same number of
        // ticks must produce an identical relative pattern of filled cells,
        // confirming there's no hidden meters-based constant that would
        // make fine scales behave differently from coarse ones. (What this
        // does NOT make scale invariant is real-world spread *speed* -- at
        // 0.1 m voxels, 1 cell/tick covers 10x less ground per second than
        // at 1.0 m voxels for the same tick rate. That's a deliberate,
        // documented trade-off -- see design doc §6 -- not a bug this test
        // is checking for.)
        //
        // Start one voxel above the floor: the cell falls once, then sleeps.
        // The final grid-space pattern must remain independent of scale.
        fn run(scale: f32) -> Vec<IVec3> {
            let mut world = World::new(WorldConfig {
                voxel_size_m: scale,
                extent_m: [16.0 * scale, 16.0 * scale, 16.0 * scale],
                ..WorldConfig::default()
            });
            world.set_solid_table(vec![false, false, true]);
            let (_, max) = world.bounds_voxels();
            world.fill_box(IVec3::ZERO, IVec3::new(max.x, 5, max.z), Voxel(2));
            let mut sim = FluidSim::new(WATER);
            sim.place_blob(&mut world, IVec3::new(8, 6, 8), 0, WATER);
            for _ in 0..40 {
                sim.tick(&mut world);
            }
            let (min, max) = world.bounds_voxels();
            let mut filled = Vec::new();
            for x in min.x..max.x {
                for y in min.y..max.y {
                    for z in min.z..max.z {
                        if world.get_voxel(IVec3::new(x, y, z)) == WATER {
                            filled.push(IVec3::new(x, y, z));
                        }
                    }
                }
            }
            filled.sort_by_key(|p| (p.x, p.y, p.z));
            filled
        }

        // Same seed, same RNG algorithm, same starting position in voxel
        // coordinates -- the two runs must be bit-for-bit identical
        // regardless of voxel_size_m, since nothing in step_cell reads it.
        let a = run(0.1);
        let b = run(1.0);
        assert!(!a.is_empty(), "the single cell must still exist somewhere after 40 ticks");
        assert_eq!(a, b, "grid-space behavior must not depend on voxel_size_m: {a:?} vs {b:?}");
    }

    #[test]
    fn flow_finds_a_drop_reachable_only_diagonally() {
        // Water at `pos` rests on water. The four axis neighbors are walled
        // off; the only escape is the diagonal (+1, 0, +1), which has air
        // beneath it. The old 4-direction scan settles here forever.
        let pos = IVec3::new(8, 6, 8);
        let open = [IVec3::new(9, 6, 9), IVec3::new(9, 5, 9)]; // diag + its drop
        let mut is_open = |p: IVec3| open.contains(&p);
        let mut is_supported = |_: IVec3| true;
        let dest = step_cell(pos, &mut is_open, &mut is_supported, false, false, false);
        assert_eq!(
            dest,
            Some(IVec3::new(9, 6, 9)),
            "the scan must step toward a diagonal-only drop"
        );
    }

    fn count_water(world: &World) -> usize {
        let (min, max) = world.bounds_voxels();
        let mut n = 0;
        for x in min.x..max.x {
            for y in min.y..max.y {
                for z in min.z..max.z {
                    if world.get_voxel(IVec3::new(x, y, z)) == WATER {
                        n += 1;
                    }
                }
            }
        }
        n
    }
}
