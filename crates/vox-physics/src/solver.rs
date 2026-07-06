//! The rigidbody world: arena, sequential-impulse solver, sleeping.
//!
//! Fixed-timestep with substeps; velocity-level Baumgarte stabilization;
//! Coulomb friction on two tangents; warm starting keyed by (body, voxel,
//! face) so stacks converge across frames; settled bodies sleep at ~zero
//! cost until an impulse or a nearby world edit wakes them.

use std::collections::HashMap;

use glam::{Quat, Vec3};
use vox_core::consts::{
    CONTACT_BETA, CONTACT_SLOP, FRICTION, GRAVITY, SLEEP_ANG, SLEEP_FRAMES, SLEEP_LIN,
    SOLVER_ITERS, SUBSTEPS,
};
use vox_world::World;

use crate::body::{Body, BodyId};
use crate::contact::{Contact, ContactKey, world_contacts};

/// Hard velocity ceiling (m/s): a diverging body is clamped, not propagated.
const MAX_SPEED: f32 = 120.0;

/// The dynamic side of the engine: all rigid bodies plus solver state.
#[derive(Default)]
pub struct PhysicsWorld {
    slots: Vec<Option<Body>>,
    generations: Vec<u32>,
    free: Vec<usize>,
    warm: HashMap<ContactKey, (f32, f32, f32)>,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a body, waking it. Returns its handle.
    pub fn spawn(&mut self, body: Body) -> BodyId {
        let slot = if let Some(slot) = self.free.pop() {
            self.slots[slot] = Some(body);
            slot
        } else {
            self.slots.push(Some(body));
            self.generations.push(0);
            self.slots.len() - 1
        };
        BodyId {
            slot: slot as u32,
            generation: self.generations[slot],
        }
    }

    /// Remove a body if the handle is current.
    pub fn despawn(&mut self, id: BodyId) {
        let slot = id.slot as usize;
        if self.generations.get(slot) == Some(&id.generation) && self.slots[slot].is_some() {
            self.slots[slot] = None;
            self.generations[slot] += 1;
            self.free.push(slot);
        }
    }

    pub fn get(&self, id: BodyId) -> Option<&Body> {
        let slot = id.slot as usize;
        if self.generations.get(slot) == Some(&id.generation) {
            self.slots[slot].as_ref()
        } else {
            None
        }
    }

    /// Iterate live bodies with their ids.
    pub fn iter(&self) -> impl Iterator<Item = (BodyId, &Body)> {
        self.slots.iter().enumerate().filter_map(|(slot, b)| {
            b.as_ref().map(|body| {
                (
                    BodyId {
                        slot: slot as u32,
                        generation: self.generations[slot],
                    },
                    body,
                )
            })
        })
    }

    pub fn body_count(&self) -> usize {
        self.slots.iter().filter(|b| b.is_some()).count()
    }

    pub fn awake_count(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .filter(|b| !b.sleep.asleep)
            .count()
    }

    /// Remove all sleeping bodies (dev convenience). Returns removed count.
    pub fn clear_sleeping(&mut self) -> usize {
        let mut removed = 0;
        for slot in 0..self.slots.len() {
            if self.slots[slot].as_ref().is_some_and(|b| b.sleep.asleep) {
                self.slots[slot] = None;
                self.generations[slot] += 1;
                self.free.push(slot);
                removed += 1;
            }
        }
        removed
    }

    /// Wake every body whose AABB intersects `[min, max]` (world edits).
    pub fn wake_region(&mut self, min: Vec3, max: Vec3) {
        for body in self.slots.iter_mut().flatten() {
            let hit = body.aabb_min.cmple(max).all() && body.aabb_max.cmpge(min).all();
            if hit {
                body.sleep.asleep = false;
                body.sleep.quiet_steps = 0;
            }
        }
    }

    /// Apply a world-space impulse at the COM, waking the body.
    pub fn apply_impulse(&mut self, id: BodyId, impulse: Vec3, angular: Vec3) {
        let slot = id.slot as usize;
        if self.generations.get(slot) != Some(&id.generation) {
            return;
        }
        if let Some(body) = self.slots[slot].as_mut() {
            body.vel += impulse * body.inv_mass;
            body.omega += angular;
            body.sleep.asleep = false;
            body.sleep.quiet_steps = 0;
        }
    }

    /// Advance the simulation by `dt` seconds against the static world.
    pub fn step(&mut self, world: &World, dt: f32) {
        // Snapshot transforms once per full step for render interpolation.
        for body in self.slots.iter_mut().flatten() {
            if !body.sleep.asleep {
                body.prev_pos = body.pos;
                body.prev_rot = body.rot;
            }
        }

        let h = dt / SUBSTEPS as f32;
        for _ in 0..SUBSTEPS {
            self.substep(world, h);
        }

        // Sleep bookkeeping at full-step rate.
        for body in self.slots.iter_mut().flatten() {
            if body.sleep.asleep {
                continue;
            }
            let quiet = body.vel.length() < SLEEP_LIN && body.omega.length() < SLEEP_ANG;
            body.sleep.quiet_steps = if quiet { body.sleep.quiet_steps + 1 } else { 0 };
            if body.sleep.quiet_steps > SLEEP_FRAMES {
                body.sleep.asleep = true;
                body.vel = Vec3::ZERO;
                body.omega = Vec3::ZERO;
                // Snap the interpolation source so sleepers render exactly.
                body.prev_pos = body.pos;
                body.prev_rot = body.rot;
            }
        }
    }

    fn substep(&mut self, world: &World, h: f32) {
        // Integrate velocities and collect contacts.
        let mut contacts: Vec<Contact> = Vec::new();
        for (slot, entry) in self.slots.iter_mut().enumerate() {
            let Some(body) = entry else { continue };
            if body.sleep.asleep {
                continue;
            }
            body.vel.y -= GRAVITY * h;
            if body.vel.length() > MAX_SPEED {
                body.vel = body.vel.normalize() * MAX_SPEED;
            }
            debug_assert!(body.vel.is_finite() && body.pos.is_finite());
            world_contacts(body, slot, world, &mut contacts);
        }

        // Warm start from the previous substep's accumulated impulses.
        for c in &mut contacts {
            if let Some(&(n0, t10, t20)) = self.warm.get(&c.key) {
                c.acc_n = n0;
                c.acc_t1 = t10;
                c.acc_t2 = t20;
                let body = self.slots[c.body].as_mut().expect("contact body alive");
                let p = c.normal * n0 + c.t1 * t10 + c.t2 * t20;
                let inv_iw = body.inv_inertia_world();
                body.vel += p * body.inv_mass;
                body.omega += inv_iw * c.r_arm.cross(p);
            }
        }

        // Velocity iterations: normal impulse with Baumgarte bias, then
        // friction clamped to the Coulomb cone.
        for _ in 0..SOLVER_ITERS {
            for c in &mut contacts {
                let body = self.slots[c.body].as_mut().expect("contact body alive");
                let inv_iw = body.inv_inertia_world();

                let vn = (body.vel + body.omega.cross(c.r_arm)).dot(c.normal);
                let bias = (CONTACT_BETA / h) * (c.depth - CONTACT_SLOP).max(0.0);
                let lambda = (bias - vn) / c.kn;
                let new_acc = (c.acc_n + lambda).max(0.0);
                let applied = new_acc - c.acc_n;
                c.acc_n = new_acc;
                let p = c.normal * applied;
                body.vel += p * body.inv_mass;
                body.omega += inv_iw * c.r_arm.cross(p);

                let max_f = FRICTION * c.acc_n;
                for (t, kt, acc) in [(c.t1, c.kt1, &mut c.acc_t1), (c.t2, c.kt2, &mut c.acc_t2)] {
                    let vt = (body.vel + body.omega.cross(c.r_arm)).dot(t);
                    let lt = -vt / kt;
                    let new_t = (*acc + lt).clamp(-max_f, max_f);
                    let applied_t = new_t - *acc;
                    *acc = new_t;
                    let pt = t * applied_t;
                    body.vel += pt * body.inv_mass;
                    body.omega += inv_iw * c.r_arm.cross(pt);
                }
            }
        }

        // Integrate positions and orientations.
        for body in self.slots.iter_mut().flatten() {
            if body.sleep.asleep {
                continue;
            }
            body.pos += body.vel * h;
            let om = body.omega;
            let dq = Quat::from_xyzw(om.x, om.y, om.z, 0.0) * body.rot;
            body.rot = Quat::from_xyzw(
                body.rot.x + 0.5 * h * dq.x,
                body.rot.y + 0.5 * h * dq.y,
                body.rot.z + 0.5 * h * dq.z,
                body.rot.w + 0.5 * h * dq.w,
            )
            .normalize();
            body.refresh_aabb();
        }

        // Persist accumulated impulses for the next substep's warm start.
        self.warm.clear();
        for c in &contacts {
            self.warm.insert(c.key, (c.acc_n, c.acc_t1, c.acc_t2));
        }
    }

    /// Interpolated transform for rendering (`alpha` ∈ [0, 1)).
    pub fn interpolated_transform(&self, id: BodyId, alpha: f32) -> Option<(Vec3, Quat)> {
        self.get(id).map(|b| {
            (
                b.prev_pos.lerp(b.pos, alpha),
                b.prev_rot.slerp(b.rot, alpha),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::VoxelGrid;
    use glam::IVec3;
    use vox_core::consts::PHYSICS_DT;
    use vox_core::{MaterialRegistry, WorldConfig};
    use vox_world::Voxel;

    const STONE: Voxel = Voxel(1);

    fn registry() -> MaterialRegistry {
        MaterialRegistry::from_toml_str(
            r#"
            [[material]]
            name = "wood"
            color = [0.5, 0.4, 0.3]
            density = 700.0
            strength = 4.0
            "#,
            "test.toml",
        )
        .expect("registry")
    }

    /// Flat stone world, floor top at 4 m, 0.1 m voxels.
    fn floored_world() -> World {
        let mut world = World::new(WorldConfig {
            voxel_size_m: 0.1,
            extent_m: [32.0, 24.0, 32.0],
            ..WorldConfig::default()
        });
        let (_, max) = world.bounds_voxels();
        world.fill_box(IVec3::ZERO, IVec3::new(max.x, 40, max.z), STONE);
        world
    }

    fn cube_body(reg: &MaterialRegistry, extent_voxels: i32, com: Vec3) -> Body {
        let dims = IVec3::splat(extent_voxels);
        let grid = VoxelGrid::new(dims, vec![Voxel(1); (dims.x * dims.y * dims.z) as usize]);
        Body::from_grid(grid, reg, 0.1, com).expect("massive body")
    }

    /// splitmix64 for the multi-body test.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (self.next() >> 40) as f32 / (1u64 << 24) as f32 * (hi - lo)
        }
        fn int(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % (hi - lo) as u64) as i32
        }
    }

    #[test]
    fn cube_drop_settles_on_floor() {
        let reg = registry();
        let world = floored_world();
        let mut phys = PhysicsWorld::new();
        let id = phys.spawn(cube_body(&reg, 4, Vec3::new(16.0, 7.0, 16.0)));

        for step in 0..240 {
            phys.step(&world, PHYSICS_DT);
            let b = phys.get(id).expect("alive");
            assert!(
                b.pos.is_finite() && b.vel.is_finite() && b.rot.is_finite(),
                "NaN at step {step}"
            );
        }
        let b = phys.get(id).expect("alive");
        assert!(b.sleep.asleep, "cube must sleep within 4 s");
        assert!(
            (b.aabb_min.y - 4.0).abs() < 0.06,
            "rest height off: aabb_min.y = {}",
            b.aabb_min.y
        );
    }

    #[test]
    fn many_bodies_all_settle_above_floor() {
        let reg = registry();
        let world = floored_world();
        let mut phys = PhysicsWorld::new();
        let mut rng = Rng(0xD1CE);
        let mut ids = Vec::new();
        for _ in 0..20 {
            let e = rng.int(2, 7);
            let com = Vec3::new(
                rng.range(8.0, 24.0),
                rng.range(6.0, 12.0),
                rng.range(8.0, 24.0),
            );
            ids.push(phys.spawn(cube_body(&reg, e, com)));
        }
        for _ in 0..720 {
            phys.step(&world, PHYSICS_DT);
        }
        for id in ids {
            let b = phys.get(id).expect("alive");
            assert!(b.sleep.asleep, "all bodies must sleep by 12 s");
            assert!(
                b.aabb_min.y > 3.9,
                "body sank below the floor: {}",
                b.aabb_min.y
            );
        }
    }

    #[test]
    #[ignore = "requires body-body contacts (Task 17); un-ignored there"]
    fn stack_of_five_stays_and_sleeps() {
        let reg = registry();
        let world = floored_world();
        let mut phys = PhysicsWorld::new();
        let mut ids = Vec::new();
        for i in 0..5 {
            let y = 4.2 + i as f32 * 0.45; // 0.4 m cubes with 5 cm gaps
            ids.push(phys.spawn(cube_body(&reg, 4, Vec3::new(16.0, y, 16.0))));
        }
        for _ in 0..360 {
            phys.step(&world, PHYSICS_DT);
        }
        for (i, id) in ids.iter().enumerate() {
            let b = phys.get(*id).expect("alive");
            assert!(b.sleep.asleep, "stack body {i} must sleep");
            assert!(
                (b.pos.x - 16.0).abs() < 0.1 && (b.pos.z - 16.0).abs() < 0.1,
                "stack body {i} drifted to ({}, {})",
                b.pos.x,
                b.pos.z
            );
        }
        // Top cube must rest near 4 cubes' height above the floor.
        let top = phys.get(ids[4]).expect("alive");
        assert!(
            (top.aabb_min.y - (4.0 + 4.0 * 0.4)).abs() < 0.1,
            "top cube rests at {}",
            top.aabb_min.y
        );
    }

    #[test]
    fn world_edit_wakes_sleeping_body() {
        let reg = registry();
        let world = floored_world();
        let mut phys = PhysicsWorld::new();
        let id = phys.spawn(cube_body(&reg, 4, Vec3::new(16.0, 5.0, 16.0)));
        for _ in 0..300 {
            phys.step(&world, PHYSICS_DT);
        }
        assert!(phys.get(id).expect("alive").sleep.asleep);

        phys.wake_region(Vec3::new(15.5, 3.5, 15.5), Vec3::new(16.5, 5.0, 16.5));
        assert!(!phys.get(id).expect("alive").sleep.asleep);
    }
}
