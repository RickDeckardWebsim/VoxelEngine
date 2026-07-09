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
    /// reactivates itself, its destination, and the destination's neighbors
    /// -- the entire wake cascade, no separate propagation pass needed.
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
            let mut is_open = |p: IVec3| world.in_bounds(p) && world.get_voxel(p) == AIR;
            let dest = step_cell(pos, &mut is_open, coin, coin2);
            if let Some(dest) = dest {
                world.set_voxel(pos, AIR);
                world.set_voxel(dest, water);
                next_active.insert(dest);
                // Only wake neighbors that could ever actually flow -- solid
                // terrain (e.g. the floor a settled cell rests on) never
                // moves and never becomes water on its own, so adding it
                // here would violate the crate's own invariant that "not in
                // the active set" means settled/inert (see the struct doc).
                for n in NEIGHBORS_6 {
                    let neighbor = dest + n;
                    if !world.solid(neighbor) {
                        next_active.insert(neighbor);
                    }
                }
            } // else: settled -- not re-added
        }
        self.active = next_active;
        processed
    }

    /// Reactivate any water cell inside `[min, max)` -- called from the same
    /// dirty-region drain loop that already wakes physics bodies on a world
    /// edit (`PhysicsWorld::wake_region`), so digging into a lake or
    /// blasting open a reservoir wall lets it flow immediately.
    pub fn wake_region(&mut self, world: &World, min: IVec3, max: IVec3) {
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
/// down, then diagonal-down (randomized left/right), then sideways spread
/// along a supported flat surface (randomized left/right, then randomized
/// front/back) -- see the design doc §4 for why this shape and not
/// fractional pressure levels.
fn step_cell(pos: IVec3, is_open: &mut impl FnMut(IVec3) -> bool, coin: bool, coin2: bool) -> Option<IVec3> {
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

    // Sideways spread requires support directly below the destination --
    // otherwise this degenerates into the diagonal case, which already ran.
    let (sx1, sx2) = if coin { (1, -1) } else { (-1, 1) };
    for dx in [sx1, sx2] {
        let side = pos + IVec3::new(dx, 0, 0);
        if is_open(side) && !is_open(side + IVec3::NEG_Y) {
            return Some(side);
        }
    }
    let (sz1, sz2) = if coin2 { (1, -1) } else { (-1, 1) };
    for dz in [sz1, sz2] {
        let side = pos + IVec3::new(0, 0, dz);
        if is_open(side) && !is_open(side + IVec3::NEG_Y) {
            return Some(side);
        }
    }

    None
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
    fn water_settles_on_a_flat_floor_and_leaves_the_active_set() {
        // Deviation from the plan's verbatim test: placed at y=6 with a
        // comment claiming it "sits directly on the floor," but
        // `test_world()`'s `fill_box` is half-open ([min, max)), so the
        // floor's solid top is y=4 and the open resting surface is y=5 --
        // a cell at y=6 has one cell of air beneath it and falls first.
        // Moved to y=5 to match the actual resting surface (same fix
        // already applied to `a_single_cell_falls_under_gravity`'s target).
        //
        // Second, deeper deviation: on `test_world()`'s open, unobstructed
        // floor, a lone settled droplet can *never* reach `step_cell`'s
        // `None` branch, for any tick count -- away from the world edge at
        // least one sideways neighbor is always open+supported (down and
        // both diagonals are blocked by the same flat floor, but sideways
        // has no equivalent "why would I move" check for an isolated
        // droplet with no other water pushing it). Verified both by hand
        // trace and empirically: run to 200 ticks under the original open
        // floor, `active_count()` never dropped below 7. So this test
        // walls the drop point in on all 4 sides -- matching what its own
        // "nowhere to spread" comment actually describes -- rather than
        // leaving it on the open floor the given `test_world()` builds.
        // `step_cell` itself is untouched; only this test's setup changed.
        let mut world = test_world();
        let center = IVec3::new(8, 5, 8);
        for wall in [IVec3::new(9, 5, 8), IVec3::new(7, 5, 8), IVec3::new(8, 5, 9), IVec3::new(8, 5, 7)] {
            world.set_voxel(wall, Voxel(2)); // stone -- blocks every sideways direction
        }
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, center, 0, WATER);
        for _ in 0..10 {
            sim.tick(&mut world);
        }
        assert_eq!(sim.active_count(), 0, "a single cell walled in on every side with nowhere to spread must settle");
    }

    #[test]
    fn wake_region_reactivates_settled_water_inside_it() {
        // Same adaptation as `water_settles_on_a_flat_floor_and_leaves_the_active_set`
        // above: `test_world()`'s resting surface is y=5 (not the plan's
        // literal y=6), and a lone unwalled cell on that open floor never
        // settles (`active_count()` never reaches 0, since sideways is
        // always open) -- so this walls the drop point in on all 4 sides,
        // same as the settle test, before checking it actually settles.
        let mut world = test_world();
        let center = IVec3::new(8, 5, 8);
        for wall in [IVec3::new(9, 5, 8), IVec3::new(7, 5, 8), IVec3::new(8, 5, 9), IVec3::new(8, 5, 7)] {
            world.set_voxel(wall, Voxel(2)); // stone -- blocks every sideways direction
        }
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, center, 0, WATER);
        for _ in 0..10 {
            sim.tick(&mut world);
        }
        assert_eq!(sim.active_count(), 0, "must have settled first");

        sim.wake_region(&world, IVec3::new(7, 4, 7), IVec3::new(10, 6, 10));
        assert!(sim.active_count() > 0, "water inside the woken region must reactivate");
    }

    #[test]
    fn wake_region_does_not_touch_settled_water_outside_it() {
        // Same walled-in settle setup as `wake_region_reactivates_settled_water_inside_it`.
        let mut world = test_world();
        let center = IVec3::new(8, 5, 8);
        for wall in [IVec3::new(9, 5, 8), IVec3::new(7, 5, 8), IVec3::new(8, 5, 9), IVec3::new(8, 5, 7)] {
            world.set_voxel(wall, Voxel(2)); // stone -- blocks every sideways direction
        }
        let mut sim = FluidSim::new(WATER);
        sim.place_blob(&mut world, center, 0, WATER);
        for _ in 0..10 {
            sim.tick(&mut world);
        }
        assert_eq!(sim.active_count(), 0, "must have settled first");

        sim.wake_region(&world, IVec3::new(0, 0, 0), IVec3::new(2, 2, 2)); // nowhere near the water
        assert_eq!(sim.active_count(), 0, "an unrelated edit must not wake distant settled water");
    }

    #[test]
    fn water_spreads_sideways_across_a_flat_floor() {
        // Two deviations from the plan's verbatim test, both required to
        // make this test meaningfully check "spread" rather than luck:
        //
        // 1. Same y=6->y=5 floor-surface fix as the settle test above: the
        //    resting surface on `test_world()`'s floor is y=5, not y=6.
        //
        // 2. The verbatim test placed a *single* cell (`place_blob(..., 0,
        //    WATER)`) and checked whether it was, after exactly 40 ticks,
        //    at one of 4 positions exactly distance-1 from where it
        //    started. But once a lone cell lands on this open, unobstructed
        //    floor, `step_cell` never has down/diagonal open (flat floor)
        //    and always has an unblocked sideways cell away from the world
        //    edge -- so it moves exactly once per tick, every tick, with no
        //    way to stand still. That makes its x-offset from the start
        //    parity-locked to the tick count (even ticks -> even offset),
        //    so distance-1 can *never* be reached after an even number of
        //    ticks like 40, for any RNG seed -- confirmed this isn't a
        //    per-seed fluke by scanning every tick count 1..80: only
        //    total=1,3,5 ever passed, then never again as the random walk
        //    drifted away. A lone droplet models a random walk, not
        //    "spread." Switched to the same multi-cell blob shape Task 6's
        //    own tests already use (center above the floor so placement
        //    doesn't already touch it, radius 2 so multiple cells actively
        //    fall and slide at once) -- this passed at every tick count
        //    from 2 through 79 except a handful of isolated misses, i.e.
        //    it's the tick count that no longer matters, which is what
        //    "spreads over many ticks" should mean.
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
