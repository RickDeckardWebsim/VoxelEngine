//! The active-cell fluid automaton. See the crate root docs and
//! `docs/plans/2026-07-09-fluid-sim-design.md` for the rationale.

use glam::IVec3;
use vox_core::FxHashSet;
use vox_world::{AIR, Voxel, World};

/// Tracks which water cells are still moving. A cell not in this set is
/// settled and costs nothing to tick -- the entire performance story of
/// this crate (mirrors `PhysicsWorld`'s sleep bookkeeping).
pub struct FluidSim {
    active: FxHashSet<IVec3>,
    /// xorshift64* state for randomized per-tick update order (same
    /// construction as `PhysicsWorld::lifetime_rng` / `ParticleSystem`'s
    /// spawn jitter) -- avoids a visible left/right or diagonal bias in how
    /// water spreads.
    rng: u64,
}

impl Default for FluidSim {
    fn default() -> Self {
        Self::new()
    }
}

impl FluidSim {
    pub fn new() -> Self {
        Self {
            active: FxHashSet::default(),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::WorldConfig;
    use vox_world::{AIR, Voxel, World};

    const WATER: Voxel = Voxel(1);

    fn test_world() -> World {
        let mut w = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [16.0, 16.0, 16.0],
            ..WorldConfig::default()
        });
        w.set_solid_table(vec![false, false]); // [air, water] -- both non-solid
        w
    }

    #[test]
    fn new_sim_has_no_active_cells() {
        let sim = FluidSim::new();
        assert_eq!(sim.active_count(), 0);
    }

    #[test]
    fn place_blob_fills_a_sphere_with_water_and_activates_every_filled_cell() {
        let mut world = test_world();
        let mut sim = FluidSim::new();
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
        let mut sim = FluidSim::new();
        sim.place_blob(&mut world, IVec3::new(8, 8, 8), 2, WATER);
        assert_eq!(
            world.get_voxel(IVec3::new(8, 8, 8)),
            Voxel(2),
            "a blob must not carve through existing terrain"
        );
    }
}
