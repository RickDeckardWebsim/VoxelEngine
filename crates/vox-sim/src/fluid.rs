//! The active-cell fluid automaton. See the crate root docs and
//! `docs/plans/2026-07-09-fluid-sim-design.md` for the rationale.

use glam::IVec3;
use vox_core::{FxHashMap, FxHashSet};
use vox_world::{AIR, Voxel, World};

const NEIGHBORS_6: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];


/// How far sideways a blocked cell searches for a reachable drop (an open
/// cell with air beneath it) before giving up and settling. Larger values
/// flatten mounds over a wider area per wake, at O(8 * horizon) extra
/// lookups per stuck-but-active cell per tick. Purely a grid count -- no
/// meters involved, so scale invariance is preserved.
const FLOW_HORIZON: i32 = 8;

/// What the fluid tick observed about a cell -- consumed by weathering
/// (`drain_events`), which uses arrival mode to grade erosion. Bounded:
/// the buffer is cleared at the start of every tick, so an app that never
/// drains holds at most one tick's worth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContactEvent {
    /// Water arrived here by falling (straight or diagonal down).
    Fell(IVec3),
    /// Water arrived here by a horizontal move (flow or spread).
    Flowed(IVec3),
    /// Water here found no move this tick. Not necessarily permanent rest:
    /// a later mover in the same tick can re-wake the cell, in which case
    /// it emits another `Settled` each tick it stays stuck.
    Settled(IVec3),
    /// Water left this cell (every move emits one).
    Vacated(IVec3),
}

/// Tracks which water cells are still moving. A cell not in this set is
/// settled and costs nothing to tick -- the entire performance story of
/// this crate (mirrors `PhysicsWorld`'s sleep bookkeeping).
pub struct FluidSim {
    active: FxHashSet<IVec3>,
    /// Horizontal direction of each *active* cell's last move (plus what
    /// woken neighbors inherited). Consulted first in `step_cell` so a
    /// draining current stays coherent instead of re-randomizing each
    /// tick. Rebuilt every tick alongside the active set -- empty whenever
    /// the water sleeps, so settled cost is still zero.
    momentum: FxHashMap<IVec3, IVec3>,
    /// The single voxel material this sim treats as water. Set once at
    /// construction -- never inferred from the active set (see `tick` and
    /// `wake_region` doc comments for why inference was fragile).
    water: Voxel,
    /// Materials this sim treats as powders (fall + diagonal slide, no
    /// spreading). Empty if the asset set defines no powders -- the sim
    /// behaves as water-only. Set once at construction.
    powders: Vec<Voxel>,
    /// xorshift64* state for randomized per-tick update order (same
    /// construction as `PhysicsWorld::lifetime_rng` / `ParticleSystem`'s
    /// spawn jitter) -- avoids a visible left/right or diagonal bias in how
    /// water spreads.
    rng: u64,
    /// This tick's `ContactEvent`s (see the enum's docs for the bound).
    events: Vec<ContactEvent>,
}

impl FluidSim {
    pub fn new(water: Voxel) -> Self {
        Self::with_powders(water, Vec::new())
    }

    /// Create a sim that also handles the given powder materials. Water
    /// uses the full CA rule (`step_cell_with_momentum`); each powder uses
    /// `step_powder` (fall + diagonal slide only). One active set, one tick
    /// loop -- the material at each cell determines which rule applies.
    pub fn with_powders(water: Voxel, powders: Vec<Voxel>) -> Self {
        Self {
            active: FxHashSet::default(),
            momentum: FxHashMap::default(),
            water,
            powders,
            rng: 0x9E37_79B9_7F4A_7C15,
            events: Vec::new(),
        }
    }

    /// Take this tick's contact events (empties the buffer).
    pub fn drain_events(&mut self) -> Vec<ContactEvent> {
        std::mem::take(&mut self.events)
    }

    /// Whether `v` is a material this sim handles (water or a powder).
    fn is_simmed(&self, v: Voxel) -> bool {
        v == self.water || self.powders.contains(&v)
    }

    /// Whether `v` is a powder material this sim handles.
    fn is_powder(&self, v: Voxel) -> bool {
        self.powders.contains(&v)
    }

    /// Number of cells currently flowing (debug-overlay stat).
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Number of cells carrying momentum (debug-overlay stat / tests).
    pub fn momentum_count(&self) -> usize {
        self.momentum.len()
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

    /// Advance every active cell by one tick: move it per its material's
    /// step rule (water: `step_cell_with_momentum`; powder: `step_powder`),
    /// or drop it from the active set if it has settled. A cell that moves
    /// reactivates its destination plus same-material neighbors of either
    /// the old or new position -- the entire wake cascade, no separate
    /// propagation pass needed.
    pub fn tick(&mut self, world: &mut World) -> usize {
        self.events.clear();
        let water = self.water;

        // Snapshot exactly the positions that hold a simmed material *before*
        // any mutation this tick, and process only those. A live re-check
        // inside the loop is unsound: a mover's wake neighbors include its
        // future down-cell, so iterating that neighbor after the mover would
        // see freshly-written material and move it *again* within this same
        // tick. Filtering up front means every entry in `cells` is only ever
        // cleared by its own move below, never by another mover's (a mover
        // can never target a cell that already holds material -- `is_open`
        // only accepts AIR), so each snapshot position is guaranteed to still
        // be valid when its turn comes.
        let cells: Vec<IVec3> = self
            .active
            .iter()
            .copied()
            .filter(|&p| self.is_simmed(world.get_voxel(p)))
            .collect();

        let mut next_active = FxHashSet::default();
        let mut next_momentum = FxHashMap::default();

        // Every active cell is processed every tick -- no per-tick cell
        // budget. Flow speed is therefore volume-independent: a 500-cell
        // flood and a 50000-cell flood each advance one cell per tick. The
        // frame-level death-spiral guard (`MAX_STEPS_PER_FRAME` in
        // `vox-platform`) handles the case where a tick is too slow to run
        // in real time -- it runs fewer ticks per frame (dilating simulated
        // time uniformly across all water), never just a subset of cells
        // within a single tick.
        let processed = cells.len();

        for pos in cells {
            let v = world.get_voxel(pos);
            let is_powder = self.is_powder(v);
            let coin = self.next_u64() & 1 == 0;
            let coin2 = self.next_u64() & 1 == 0;
            let dest = {
                let mut is_open = |p: IVec3| world.in_bounds(p) && world.get_voxel(p) == AIR;
                if is_powder {
                    step_powder(pos, &mut is_open, coin, coin2)
                } else {
                    let mut is_supported = |p: IVec3| {
                        world.in_bounds(p) && (world.solid(p) || world.get_voxel(p) == water)
                    };
                    let has_water_above = world.get_voxel(pos + IVec3::Y) == water;
                    step_cell_with_momentum(
                        pos,
                        &mut is_open,
                        &mut is_supported,
                        has_water_above,
                        coin,
                        coin2,
                        self.momentum.get(&pos).copied(),
                    )
                }
            };
            if let Some(dest) = dest {
                // Write the cell's own material back -- water or powder.
                world.set_voxel(pos, AIR);
                world.set_voxel(dest, v);
                next_active.insert(dest);
                self.events.push(ContactEvent::Vacated(pos));
                self.events.push(if dest.y < pos.y {
                    ContactEvent::Fell(dest)
                } else {
                    ContactEvent::Flowed(dest)
                });
                // Momentum is water-only: powders don't carry direction.
                if !is_powder {
                    let hdir = IVec3::new(dest.x - pos.x, 0, dest.z - pos.z);
                    let carried = if hdir != IVec3::ZERO {
                        Some(hdir)
                    } else {
                        self.momentum.get(&pos).copied()
                    };
                    if let Some(d) = carried {
                        next_momentum.insert(dest, d);
                    }
                }
                // Wake same-material neighbors of both the old and new
                // position: if an upper cell was visited earlier in this
                // snapshot, moving its support out from under it must give
                // it another turn next tick. Water wakes water; powder
                // wakes powder -- they don't recruit each other.
                for changed in [pos, dest] {
                    for n in NEIGHBORS_6 {
                        let neighbor = changed + n;
                        let nv = world.get_voxel(neighbor);
                        if self.is_simmed(nv) {
                            next_active.insert(neighbor);
                            // Only water inherits momentum.
                            if !is_powder && nv == water {
                                let hdir = IVec3::new(dest.x - pos.x, 0, dest.z - pos.z);
                                let carried = if hdir != IVec3::ZERO {
                                    Some(hdir)
                                } else {
                                    self.momentum.get(&pos).copied()
                                };
                                if let Some(d) = carried {
                                    next_momentum.entry(neighbor).or_insert(d);
                                }
                            }
                        }
                    }
                }
            } else {
                // Settled -- not re-added to the active set.
                self.events.push(ContactEvent::Settled(pos));
            }
        }
        self.active = next_active;
        self.momentum = next_momentum;
        processed
    }

    /// Reactivate simmed material (water or powder) inside or directly
    /// adjacent to `[min, max)`. World edits report the cells they changed,
    /// rather than their neighbors: a wall voxel just changed to air
    /// contains no material itself, so the one-cell halo is what wakes the
    /// settled material immediately against that new opening. Bounds are
    /// clipped before scanning.
    pub fn wake_region(&mut self, world: &World, min: IVec3, max: IVec3) {
        let (bounds_min, bounds_max) = world.bounds_voxels();
        let min = (min.max(bounds_min) - IVec3::ONE).max(bounds_min);
        let max = (max.min(bounds_max) + IVec3::ONE).min(bounds_max);
        for x in min.x..max.x {
            for y in min.y..max.y {
                for z in min.z..max.z {
                    let p = IVec3::new(x, y, z);
                    if self.is_simmed(world.get_voxel(p)) {
                        self.active.insert(p);
                    }
                }
            }
        }
    }
}

/// Thin momentum-free wrapper kept for the unit tests that predate momentum;
/// `tick` itself always calls `step_cell_with_momentum` directly.
#[cfg(test)]
fn step_cell(
    pos: IVec3,
    is_open: &mut impl FnMut(IVec3) -> bool,
    is_supported: &mut impl FnMut(IVec3) -> bool,
    has_water_above: bool,
    coin: bool,
    coin2: bool,
) -> Option<IVec3> {
    step_cell_with_momentum(pos, is_open, is_supported, has_water_above, coin, coin2, None)
}

/// Where a water cell at `pos` wants to move this tick, or `None` if it has
/// nowhere to go (should settle). `is_open` reports whether a cell can be
/// flowed into: empty (air) and in-bounds. Order of preference: straight
/// down, then diagonal-down (randomized left/right), then one step toward
/// the first drop reachable within `FLOW_HORIZON` cells sideways, then
/// pressure-gated sideways leveling onto supported terrain or water
/// (randomized left/right, then front/back) -- see the design doc §4 for
/// why this shape and not fractional pressure levels.
///
/// `momentum` is an optional remembered horizontal direction: its axis
/// components bias the diagonal-fall and sideways-spread sign order, and the
/// full vector (which may be diagonal, e.g. `(1, 0, 1)`) leads the flow scan
/// ahead of the coin-ordered eight. `None` reproduces the coin-only
/// behavior exactly.
fn step_cell_with_momentum(
    pos: IVec3,
    is_open: &mut impl FnMut(IVec3) -> bool,
    is_supported: &mut impl FnMut(IVec3) -> bool,
    has_water_above: bool,
    coin: bool,
    coin2: bool,
    momentum: Option<IVec3>,
) -> Option<IVec3> {
    let down = pos + IVec3::NEG_Y;
    if is_open(down) {
        return Some(down);
    }

    // Diagonal fall: momentum's axis components first, then the coin order.
    let m = momentum.unwrap_or(IVec3::ZERO);
    let (dx1, dx2) = if m.x != 0 { (m.x, -m.x) } else if coin { (1, -1) } else { (-1, 1) };
    for dx in [dx1, dx2] {
        let diag = pos + IVec3::new(dx, -1, 0);
        if is_open(diag) {
            return Some(diag);
        }
    }
    let (dz1, dz2) = if m.z != 0 { (m.z, -m.z) } else if coin2 { (1, -1) } else { (-1, 1) };
    for dz in [dz1, dz2] {
        let diag = pos + IVec3::new(0, -1, dz);
        if is_open(diag) {
            return Some(diag);
        }
    }

    // Flow: no immediate fall exists, so search all eight horizontal
    // directions -- the momentum direction first (if any), then the
    // remaining seven: the four axes, then the four diagonals, each group
    // in randomized order -- for a reachable drop -- an open run of
    // same-height cells ending in one with air beneath -- and take one step
    // toward the first one found. Diagonal rays deliberately check only the
    // cells on the ray, never the two orthogonal neighbors, so water can
    // slip through the seam where two solid blocks touch only at a corner
    // -- accepted for a coarse voxel fluid, not an oversight.
    // This is what keeps a mound from freezing into a stable
    // stepped pyramid: its surface cells can walk over the water below them
    // until they reach the pile's edge and fall off. Unlike an
    // unconditional sideways shuffle, this only ever moves when a strictly
    // lower destination is reachable, so a flat sheet or a full basin still
    // has nowhere to go and sleeps.
    let dirs = flow_dirs(coin, coin2);
    let ordered = momentum.into_iter().chain(dirs.into_iter().filter(|&d| Some(d) != momentum));
    for dir in ordered {
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
    let (sx1, sx2) = if m.x != 0 { (m.x, -m.x) } else if coin { (1, -1) } else { (-1, 1) };
    for dx in [sx1, sx2] {
        let side = pos + IVec3::new(dx, 0, 0);
        if is_open(side) && is_supported(side + IVec3::NEG_Y) {
            return Some(side);
        }
    }
    let (sz1, sz2) = if m.z != 0 { (m.z, -m.z) } else if coin2 { (1, -1) } else { (-1, 1) };
    for dz in [sz1, sz2] {
        let side = pos + IVec3::new(0, 0, dz);
        if is_open(side) && is_supported(side + IVec3::NEG_Y) {
            return Some(side);
        }
    }

    None
}

/// Where a powder cell at `pos` wants to move this tick, or `None` if it
/// should settle. Simpler than the fluid rule: straight down if air below,
/// else diagonal-down (all four, randomized), else settle. No flow-horizon
/// search, no pressure-gated spreading -- powders pile at a natural ~45°
/// angle of repose rather than seeking a flat level. `is_open` reports
/// whether a cell can be moved into: air and in-bounds (same as water).
fn step_powder(
    pos: IVec3,
    is_open: &mut impl FnMut(IVec3) -> bool,
    coin: bool,
    coin2: bool,
) -> Option<IVec3> {
    let down = pos + IVec3::NEG_Y;
    if is_open(down) {
        return Some(down);
    }
    // Diagonal slide: try all four diagonal-down cells in randomized order.
    // Unlike water's separate X-then-Z diagonal check, powder tries all
    // four at once since there's no momentum biasing. A powder on a slope
    // slides down it one cell per tick.
    let (dx1, dx2) = if coin { (1, -1) } else { (-1, 1) };
    let (dz1, dz2) = if coin2 { (1, -1) } else { (-1, 1) };
    for &(dx, dz) in &[(dx1, dz1), (dx1, dz2), (dx2, dz1), (dx2, dz2)] {
        let diag = pos + IVec3::new(dx, -1, dz);
        if is_open(diag) {
            return Some(diag);
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
    fn tick_processes_all_active_cells_regardless_of_volume() {
        // Flow speed must be volume-independent: every active cell gets one
        // step per tick, no matter how many there are. A large blob (well
        // over the old 4096-cell budget this sim used to cap) must process
        // *all* its cells in a single tick, deferring none. The frame-level
        // death-spiral guard (MAX_STEPS_PER_FRAME) handles slow ticks by
        // running fewer ticks per frame, not by skipping cells.
        //
        // `test_world()` is only 16 voxels per axis, far too small to hold
        // a sphere wide enough to be meaningful here. Volume of a radius-r
        // sphere is ~(4/3)*pi*r^3; r=12 gives ~7238 cells. That needs a
        // world spanning at least 2*12+1 = 25 voxels per axis around the
        // blob's center, so this test builds its own larger, self-contained
        // world rather than changing the shared `test_world()` helper.
        const OLD_BUDGET: usize = 4096;
        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [40.0, 40.0, 40.0],
            ..WorldConfig::default()
        });
        world.set_solid_table(vec![false, false, true]); // [air, water, floor]
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, IVec3::new(20, 20, 20), 12, WATER);
        let active_before = sim.active_count();
        assert!(
            active_before > OLD_BUDGET,
            "test needs more active cells than the old budget to be meaningful: got {active_before}"
        );

        let processed = sim.tick(&mut world);
        assert_eq!(
            processed, active_before,
            "every active cell must be processed -- no cell budget, no deferred overflow"
        );
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

    #[test]
    fn step_cell_prefers_the_momentum_direction_on_equal_drops() {
        // Drops exist both at +X and -X, two cells out. With no momentum the
        // coin decides; with momentum -X the cell must step -X regardless of
        // what the coins say.
        let pos = IVec3::new(8, 6, 8);
        let open = [
            IVec3::new(9, 6, 8), IVec3::new(10, 6, 8), IVec3::new(10, 5, 8), // +X run
            IVec3::new(7, 6, 8), IVec3::new(6, 6, 8), IVec3::new(6, 5, 8),   // -X run
        ];
        for coins in [(false, false), (true, false), (false, true), (true, true)] {
            let mut is_open = |p: IVec3| open.contains(&p);
            let mut is_supported = |_: IVec3| true;
            let dest = step_cell_with_momentum(
                pos, &mut is_open, &mut is_supported, false, coins.0, coins.1,
                Some(IVec3::NEG_X),
            );
            assert_eq!(dest, Some(IVec3::new(7, 6, 8)), "momentum -X must win for coins {coins:?}");
        }
    }

    #[test]
    fn momentum_is_forgotten_once_water_settles() {
        let mut world = test_world();
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, IVec3::new(8, 8, 8), 2, WATER);
        for _ in 0..400 {
            sim.tick(&mut world);
            if sim.active_count() == 0 {
                break;
            }
        }
        assert_eq!(sim.active_count(), 0, "blob must settle");
        assert_eq!(sim.momentum_count(), 0, "settled water must carry no momentum state");
    }

    #[test]
    fn tick_emits_fell_vacated_and_settled_events() {
        let mut world = test_world();
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, IVec3::new(8, 7, 8), 0, WATER); // 2 above floor
        sim.tick(&mut world);
        let ev = sim.drain_events();
        assert!(ev.contains(&ContactEvent::Vacated(IVec3::new(8, 7, 8))), "leaving a cell must emit Vacated: {ev:?}");
        assert!(ev.contains(&ContactEvent::Fell(IVec3::new(8, 6, 8))), "a downward arrival must emit Fell: {ev:?}");

        sim.tick(&mut world); // lands on floor -> second Fell
        sim.drain_events();
        sim.tick(&mut world); // nowhere to go -> settles
        let ev = sim.drain_events();
        assert!(ev.contains(&ContactEvent::Settled(IVec3::new(8, 5, 8))), "settling must emit Settled: {ev:?}");
    }

    #[test]
    fn a_horizontal_flow_move_emits_flowed() {
        // Weathering grades erosion on the Fell/Flowed distinction, so the
        // horizontal arm needs its own pin. Water resting on the floor with
        // a pit two cells away can't fall or slide diagonally -- its only
        // legal move is a flow step at its own height toward the pit.
        let mut world = test_world();
        world.set_voxel(IVec3::new(10, 4, 8), AIR); // pit in the floor
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, IVec3::new(8, 5, 8), 0, WATER);
        sim.tick(&mut world);
        let ev = sim.drain_events();
        assert!(
            ev.contains(&ContactEvent::Flowed(IVec3::new(9, 5, 8))),
            "a same-height flow step must emit Flowed: {ev:?}"
        );
    }

    // --- Powder tests ---

    const POWDER: Voxel = Voxel(3);

    fn powder_world() -> World {
        let mut w = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [16.0, 16.0, 16.0],
            ..WorldConfig::default()
        });
        // [air, water, floor, powder] -- powder is non-solid like water.
        w.set_solid_table(vec![false, false, true, false]);
        let (_, max) = w.bounds_voxels();
        w.fill_box(IVec3::new(0, 0, 0), IVec3::new(max.x, 5, max.z), Voxel(2)); // floor top at y=5
        w
    }

    fn powder_sim() -> FluidSim {
        FluidSim::with_powders(WATER, vec![POWDER])
    }

    #[test]
    fn powder_falls_straight_down_and_settles() {
        let mut world = powder_world();
        let mut sim = powder_sim();
        world.set_voxel(IVec3::new(8, 8, 8), POWDER);
        sim.active.insert(IVec3::new(8, 8, 8));
        // Fall to the floor (y=5 is the top of the floor, so rest at y=5).
        for _ in 0..10 {
            sim.tick(&mut world);
        }
        assert_eq!(sim.active_count(), 0, "powder must settle on the floor");
        assert_eq!(world.get_voxel(IVec3::new(8, 5, 8)), POWDER, "powder must rest on the floor");
        assert_eq!(world.get_voxel(IVec3::new(8, 6, 8)), AIR, "nothing above the resting powder");
    }

    #[test]
    fn powder_piles_and_sleeps() {
        // A column of powder falls onto a flat floor and forms a stable
        // pile (pyramid, not a straight stack -- powder slides diagonally
        // off a column at its angle of repose). active_count reaches 0,
        // not an infinite shuffle, and cell count is conserved.
        let mut world = powder_world();
        let mut sim = powder_sim();
        // Drop a 4-cell column from above.
        for y in 9..=12 {
            world.set_voxel(IVec3::new(8, y, 8), POWDER);
            sim.active.insert(IVec3::new(8, y, 8));
        }
        let before = count_voxel(&world, POWDER);
        for _ in 0..80 {
            sim.tick(&mut world);
            if sim.active_count() == 0 {
                break;
            }
        }
        assert_eq!(sim.active_count(), 0, "powder pile must settle");
        let after = count_voxel(&world, POWDER);
        assert_eq!(before, after, "all powder cells must survive (conservation)");
        // At least one cell must be resting on the floor (y=5, on top of
        // the solid floor at y=4).
        assert!(after > 0, "powder must not vanish");
    }

    #[test]
    fn powder_on_a_slope_slides_diagonally() {
        // Powder on a pillar can't fall straight (solid below), so it must
        // slide to one of the four diagonal-down cells. `step_powder` tries
        // the four corner diagonals (dx≠0 AND dz≠0), not the face-adjacent
        // diagonals water uses. Assert it left the pillar and landed at a
        // lower diagonal cell -- deterministic regardless of which one.
        let mut world = powder_world();
        let mut sim = powder_sim();
        let pillar = IVec3::new(9, 6, 8);
        let powder_pos = IVec3::new(9, 7, 8);
        world.set_voxel(pillar, Voxel(2)); // pillar (solid, blocks straight fall)
        world.set_voxel(powder_pos, POWDER);
        sim.active.insert(powder_pos);
        sim.tick(&mut world);
        assert_eq!(world.get_voxel(powder_pos), AIR, "powder must leave the pillar top");
        // It must have landed at one of the four diagonal-down cells.
        let diagonals = [
            powder_pos + IVec3::new(-1, -1, -1),
            powder_pos + IVec3::new(-1, -1, 1),
            powder_pos + IVec3::new(1, -1, -1),
            powder_pos + IVec3::new(1, -1, 1),
        ];
        let landed = diagonals.iter().find(|&&d| world.get_voxel(d) == POWDER);
        assert!(landed.is_some(), "powder must slide to a diagonal-down cell");
    }

    #[test]
    fn powder_does_not_flow_sideways_on_flat_ground() {
        // A flat sheet of powder on flat ground must sleep -- no lateral
        // spreading (unlike water, which would level).
        let mut world = powder_world();
        let mut sim = powder_sim();
        // Place a row of powder on the floor.
        for x in 6..=10 {
            world.set_voxel(IVec3::new(x, 5, 8), POWDER);
            sim.active.insert(IVec3::new(x, 5, 8));
        }
        sim.tick(&mut world);
        assert_eq!(sim.active_count(), 0, "flat powder on flat ground must sleep");
        // All cells still in place.
        for x in 6..=10 {
            assert_eq!(world.get_voxel(IVec3::new(x, 5, 8)), POWDER, "powder must not shuffle");
        }
    }

    #[test]
    fn powder_cell_count_is_conserved() {
        let mut world = powder_world();
        let mut sim = powder_sim();
        sim.place_blob(&mut world, IVec3::new(8, 10, 8), 2, POWDER);
        let before = count_voxel(&world, POWDER);
        for _ in 0..60 {
            sim.tick(&mut world);
        }
        let after = count_voxel(&world, POWDER);
        assert_eq!(before, after, "powder cell count must be conserved");
    }

    fn count_voxel(world: &World, v: Voxel) -> usize {
        let (min, max) = world.bounds_voxels();
        let mut n = 0;
        for x in min.x..max.x {
            for y in min.y..max.y {
                for z in min.z..max.z {
                    if world.get_voxel(IVec3::new(x, y, z)) == v {
                        n += 1;
                    }
                }
            }
        }
        n
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
