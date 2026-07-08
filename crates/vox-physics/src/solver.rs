//! The rigidbody world: arena, sequential-impulse solver, sleeping.
//!
//! Fixed-timestep with substeps; velocity-level Baumgarte stabilization;
//! Coulomb friction on two tangents; warm starting keyed by (body, voxel,
//! face) so stacks converge across frames; settled bodies sleep at ~zero
//! cost until an impulse or a nearby world edit wakes them.

use std::collections::HashMap;

use glam::{Quat, Vec3};
use vox_core::Tunables;
use vox_core::consts::{CONTACT_SLOP, GRAVITY, SLEEP_FRAMES, SOLVER_ITERS, SUBSTEPS};
use vox_world::World;

use crate::body::{Body, BodyId};
use crate::broadphase::candidate_pairs;
use crate::contact::{Contact, ContactKey, pair_contacts, world_contacts};

/// Relative contact speed above which a sleeping body is woken by an impact,
/// as a multiple of the live `sleep_lin` tunable.
const WAKE_SPEED_FACTOR: f32 = 2.0;

/// Minimum closing speed (see [`crate::contact::Contact::approach_speed`])
/// for a contact to count as an "impact" at all, regardless of its
/// accumulated normal impulse. A body resting quietly (or mid-bounce while
/// settling) still needs a real contact impulse every single substep just
/// to hold it up against gravity/Baumgarte correction -- that impulse looks
/// identical, frame to frame, to the one from a body that just landed hard,
/// but its approach speed is near zero. Without this gate, a freshly
/// detached fragment resting on a fragile material kept re-reporting that
/// steady load as a fresh impact on *every single frame* it was settling,
/// which for a low-strength material (leaves) repeatedly re-triggered
/// impact fracture forever -- read as continuous flicker, not a one-time
/// crumble. Comfortably above gravity's own per-substep speed contribution
/// (~0.08 m/s at 60Hz/2 substeps) so ordinary settling never crosses it, but
/// well below any real collision.
const MIN_IMPACT_APPROACH_SPEED_M_S: f32 = 0.4;

/// The hardest single-contact impact a body took during one [`PhysicsWorld::step`]
/// call, in case a caller wants to fracture it (material-based impact
/// destruction). `impulse` is the contact's peak accumulated normal impulse
/// this step (kg*m/s) -- `impulse / body.mass()` is the velocity change the
/// hit imparted, a physically meaningful, mass-independent way to compare
/// against a material's strength. Only the single hardest hit per body is
/// reported per step, not every contact -- a resting stack's steady contact
/// forces are not "impacts", and a caller deciding whether to fracture a
/// body only needs to know its worst moment.
#[derive(Copy, Clone, Debug)]
pub struct ImpactEvent {
    pub body: BodyId,
    /// World-space point of the hardest contact.
    pub point_m: Vec3,
    /// Peak accumulated normal impulse at that contact this step, kg*m/s.
    pub impulse: f32,
    /// Unit direction the impact pushed `body`, i.e. away from whatever it
    /// hit and into `body`'s own volume at the contact point -- a caller
    /// fracturing the body along this can carve *into* the struck surface
    /// instead of a generic sphere straddling half in, half out of it.
    pub push_dir: Vec3,
}

/// Two distinct mutable borrows out of the slot array.
fn two_mut(slots: &mut [Option<Body>], a: usize, b: usize) -> (&mut Body, &mut Body) {
    debug_assert_ne!(a, b);
    if a < b {
        let (lo, hi) = slots.split_at_mut(b);
        (
            lo[a].as_mut().expect("body a alive"),
            hi[0].as_mut().expect("body b alive"),
        )
    } else {
        let (lo, hi) = slots.split_at_mut(a);
        (
            hi[0].as_mut().expect("body a alive"),
            lo[b].as_mut().expect("body b alive"),
        )
    }
}

/// Hard velocity ceiling (m/s): a diverging body is clamped, not propagated.
const MAX_SPEED: f32 = 120.0;

/// The dynamic side of the engine: all rigid bodies plus solver state.
#[derive(Default)]
pub struct PhysicsWorld {
    slots: Vec<Option<Body>>,
    generations: Vec<u32>,
    free: Vec<usize>,
    warm: HashMap<ContactKey, (f32, f32, f32)>,
    /// Live-tunable solver parameters (friction, contact bias, sleep
    /// thresholds); public so the debug overlay can bind sliders directly.
    pub tunables: Tunables,
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

    /// Remove all sleeping bodies (dev convenience). Returns their ids so the
    /// caller can drop any associated GPU meshes.
    pub fn clear_sleeping(&mut self) -> Vec<BodyId> {
        let mut removed = Vec::new();
        for slot in 0..self.slots.len() {
            if self.slots[slot].as_ref().is_some_and(|b| b.sleep.asleep) {
                removed.push(BodyId {
                    slot: slot as u32,
                    generation: self.generations[slot],
                });
                self.slots[slot] = None;
                self.generations[slot] += 1;
                self.free.push(slot);
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
    /// Returns each body's hardest single contact this step, if any -- see
    /// [`ImpactEvent`]. Most steps most bodies are resting quietly, so this
    /// is typically empty or small; it's the caller's job to decide what
    /// "hard enough" means (e.g. comparing against a material's strength).
    pub fn step(&mut self, world: &World, dt: f32) -> Vec<ImpactEvent> {
        // Snapshot transforms once per full step for render interpolation.
        for body in self.slots.iter_mut().flatten() {
            if !body.sleep.asleep {
                body.prev_pos = body.pos;
                body.prev_rot = body.rot;
            }
        }

        // Peak impulse (point, and push direction) per body slot, across
        // every substep.
        let mut peaks: HashMap<usize, (f32, Vec3, Vec3)> = HashMap::new();
        let h = dt / SUBSTEPS as f32;
        for _ in 0..SUBSTEPS {
            self.substep(world, h, &mut peaks);
        }
        let impacts = peaks
            .into_iter()
            .map(|(slot, (impulse, point_m, push_dir))| ImpactEvent {
                body: BodyId {
                    slot: slot as u32,
                    generation: self.generations[slot],
                },
                point_m,
                impulse,
                push_dir,
            })
            .collect();

        // Update per-body quiet counters.
        for body in self.slots.iter_mut().flatten() {
            if body.sleep.asleep {
                continue;
            }
            let quiet = body.vel.length() < self.tunables.sleep_lin
                && body.omega.length() < self.tunables.sleep_ang;
            body.sleep.quiet_steps = if quiet { body.sleep.quiet_steps + 1 } else { 0 };
        }

        // Island-consensus sleep: bodies touching each other must cross the
        // quiet threshold together and transition atomically. A pair briefly
        // holding mixed sleep state (one asleep, one awake) flips which body
        // is "sampler" vs "target" in pair_contacts, which changes the
        // warm-start contact key and re-injects a small impulse discontinuity
        // — exactly the kind of disturbance that kept resetting quiet
        // counters stack-wide before this fix.
        let islands = Self::islands(&self.slots);
        for island in islands {
            if island.len() < 2 {
                // Isolated body: the simple per-body rule already works (it
                // only ever contacts the static world, which has no sleep
                // state to desynchronize against).
                if let Some(body) = self.slots[island[0]].as_mut()
                    && !body.sleep.asleep
                    && body.sleep.quiet_steps > SLEEP_FRAMES
                {
                    body.sleep.asleep = true;
                    body.vel = Vec3::ZERO;
                    body.omega = Vec3::ZERO;
                    body.prev_pos = body.pos;
                    body.prev_rot = body.rot;
                }
                continue;
            }
            let ready = island.iter().all(|&slot| {
                self.slots[slot]
                    .as_ref()
                    .is_some_and(|b| b.sleep.asleep || b.sleep.quiet_steps > SLEEP_FRAMES)
            });
            if !ready {
                continue;
            }
            for &slot in &island {
                if let Some(body) = self.slots[slot].as_mut()
                    && !body.sleep.asleep
                {
                    body.sleep.asleep = true;
                    body.vel = Vec3::ZERO;
                    body.omega = Vec3::ZERO;
                    body.prev_pos = body.pos;
                    body.prev_rot = body.rot;
                }
            }
        }
        impacts
    }

    /// Group bodies into connected components via broadphase contact.
    /// Every live slot appears in exactly one island (singletons included).
    fn islands(slots: &[Option<Body>]) -> Vec<Vec<usize>> {
        let mut parent: Vec<usize> = (0..slots.len()).collect();
        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }
        for (a, b) in candidate_pairs(slots) {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra] = rb;
            }
        }
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (slot, entry) in slots.iter().enumerate() {
            if entry.is_some() {
                let root = find(&mut parent, slot);
                groups.entry(root).or_default().push(slot);
            }
        }
        groups.into_values().collect()
    }

    fn substep(&mut self, world: &World, h: f32, peaks: &mut HashMap<usize, (f32, Vec3, Vec3)>) {
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

        // Body-body narrowphase over broadphase candidates.
        for (a, b) in candidate_pairs(&self.slots) {
            let (asleep_a, asleep_b) = {
                let ba = self.slots[a].as_ref().expect("pair body alive");
                let bb = self.slots[b].as_ref().expect("pair body alive");
                (ba.sleep.asleep, bb.sleep.asleep)
            };
            // Sampler = fewer surface points; a sleeping target is treated as
            // static unless the impact is fast enough to wake it. If the
            // sampler would be the sleeping one, swap roles so the moving
            // body does the sampling (its points carry the velocity).
            let (mut sampler, mut target) = {
                let ba = self.slots[a].as_ref().expect("alive");
                let bb = self.slots[b].as_ref().expect("alive");
                if ba.surface.len() <= bb.surface.len() {
                    (a, b)
                } else {
                    (b, a)
                }
            };
            let mut target_asleep = if target == a { asleep_a } else { asleep_b };
            let sampler_asleep = if sampler == a { asleep_a } else { asleep_b };
            if sampler_asleep && !target_asleep {
                std::mem::swap(&mut sampler, &mut target);
                target_asleep = true;
            }

            let mut staged: Vec<Contact> = Vec::new();
            let result = {
                let sb = self.slots[sampler].as_ref().expect("alive");
                let tb = self.slots[target].as_ref().expect("alive");
                pair_contacts(sb, sampler, tb, target, target_asleep, &mut staged)
            };
            if result.contact_count == 0 {
                continue;
            }
            if target_asleep && result.max_rel_speed > WAKE_SPEED_FACTOR * self.tunables.sleep_lin {
                // Impact wakes the sleeper: regenerate as a dynamic pair.
                let tb = self.slots[target].as_mut().expect("alive");
                tb.sleep.asleep = false;
                tb.sleep.quiet_steps = 0;
                staged.clear();
                let sb_ref = self.slots[sampler].as_ref().expect("alive");
                let tb_ref = self.slots[target].as_ref().expect("alive");
                pair_contacts(sb_ref, sampler, tb_ref, target, false, &mut staged);
            }
            contacts.append(&mut staged);
        }

        // Warm start from the previous substep's accumulated impulses.
        for c in &mut contacts {
            if let Some(&(n0, t10, t20)) = self.warm.get(&c.key) {
                c.acc_n = n0;
                c.acc_t1 = t10;
                c.acc_t2 = t20;
                let p = c.normal * n0 + c.t1 * t10 + c.t2 * t20;
                Self::apply_contact_impulse(&mut self.slots, c, p);
            }
        }

        // Velocity iterations: normal impulse with Baumgarte bias, then
        // friction clamped to the Coulomb cone.
        for _ in 0..SOLVER_ITERS {
            for c in &mut contacts {
                let vn = Self::relative_velocity(&self.slots, c).dot(c.normal);
                let bias = (self.tunables.contact_beta / h) * (c.depth - CONTACT_SLOP).max(0.0);
                let lambda = (bias - vn) / c.kn;
                let new_acc = (c.acc_n + lambda).max(0.0);
                let applied = new_acc - c.acc_n;
                c.acc_n = new_acc;
                Self::apply_contact_impulse(&mut self.slots, c, c.normal * applied);

                let max_f = self.tunables.friction * c.acc_n;
                for i in 0..2 {
                    let (t, kt) = if i == 0 { (c.t1, c.kt1) } else { (c.t2, c.kt2) };
                    let vt = Self::relative_velocity(&self.slots, c).dot(t);
                    let lt = -vt / kt;
                    let acc = if i == 0 { &mut c.acc_t1 } else { &mut c.acc_t2 };
                    let new_t = (*acc + lt).clamp(-max_f, max_f);
                    let applied_t = new_t - *acc;
                    *acc = new_t;
                    Self::apply_contact_impulse(&mut self.slots, c, t * applied_t);
                }
            }
        }

        // Record each body's hardest single contact this substep, for
        // material-based impact fracture (see `ImpactEvent`). Positions
        // here are pre-integration, matching the frame `c.r_arm`/`r_arm_b`
        // were computed in when this substep's contacts were generated.
        for c in &contacts {
            if c.acc_n <= 0.0 || c.approach_speed < MIN_IMPACT_APPROACH_SPEED_M_S {
                continue;
            }
            // `c.normal` points from the target toward `c.body` (its push
            // direction); `body_b` receives the equal-and-opposite impulse,
            // so its own push direction is the reverse.
            if let Some(body) = self.slots[c.body].as_ref() {
                let point = body.pos + c.r_arm;
                let entry = peaks.entry(c.body).or_insert((0.0, point, c.normal));
                if c.acc_n > entry.0 {
                    *entry = (c.acc_n, point, c.normal);
                }
            }
            if let Some(b) = c.body_b
                && let Some(body_b) = self.slots[b].as_ref()
            {
                let point = body_b.pos + c.r_arm_b;
                let entry = peaks.entry(b).or_insert((0.0, point, -c.normal));
                if c.acc_n > entry.0 {
                    *entry = (c.acc_n, point, -c.normal);
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

    /// Apply impulse `p` (already signed for `body`) to a contact's body,
    /// and the equal-and-opposite reaction to `body_b` if present.
    fn apply_contact_impulse(slots: &mut [Option<Body>], c: &Contact, p: Vec3) {
        if let Some(b) = c.body_b {
            let (ba, bb) = two_mut(slots, c.body, b);
            let inv_iw_a = ba.inv_inertia_world();
            ba.vel += p * ba.inv_mass;
            ba.omega += inv_iw_a * c.r_arm.cross(p);
            let inv_iw_b = bb.inv_inertia_world();
            bb.vel -= p * bb.inv_mass;
            bb.omega -= inv_iw_b * c.r_arm_b.cross(p);
        } else {
            let body = slots[c.body].as_mut().expect("contact body alive");
            let inv_iw = body.inv_inertia_world();
            body.vel += p * body.inv_mass;
            body.omega += inv_iw * c.r_arm.cross(p);
        }
    }

    /// Velocity of `body`'s contact point minus `body_b`'s (zero velocity
    /// for the static world or a sleeping target).
    fn relative_velocity(slots: &[Option<Body>], c: &Contact) -> Vec3 {
        let body = slots[c.body].as_ref().expect("contact body alive");
        let va = body.vel + body.omega.cross(c.r_arm);
        if let Some(b) = c.body_b {
            let bb = slots[b].as_ref().expect("contact body_b alive");
            let vb = bb.vel + bb.omega.cross(c.r_arm_b);
            va - vb
        } else {
            va
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

    /// A body dropped from height must report a hard impact when it lands
    /// (a big peak normal impulse, at a point near the floor), but once
    /// resting quietly, later steps must report nothing of the sort --
    /// steady contact force holding a body up is not an "impact".
    #[test]
    fn a_hard_landing_reports_an_impact_event_but_resting_does_not() {
        let reg = registry();
        let world = floored_world();
        let mut phys = PhysicsWorld::new();
        let id = phys.spawn(cube_body(&reg, 4, Vec3::new(16.0, 10.0, 16.0)));

        let mut saw_impact = false;
        let mut max_impulse = 0.0f32;
        for _ in 0..300 {
            for event in phys.step(&world, PHYSICS_DT) {
                assert_eq!(event.body, id);
                saw_impact = true;
                max_impulse = max_impulse.max(event.impulse);
            }
            if phys.get(id).expect("alive").sleep.asleep {
                break;
            }
        }
        assert!(saw_impact, "a body falling 6 m onto stone must report an impact");
        assert!(max_impulse > 0.0);

        // Once settled and asleep, further steps must be quiet -- no
        // spurious impacts from steady resting contact.
        let quiet = phys.step(&world, PHYSICS_DT);
        assert!(quiet.is_empty(), "resting body must not report an impact: {quiet:?}");
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
    fn two_cubes_stack_and_both_sleep() {
        let reg = registry();
        let world = floored_world();
        let mut phys = PhysicsWorld::new();
        let bottom = phys.spawn(cube_body(&reg, 4, Vec3::new(16.0, 4.5, 16.0)));
        for _ in 0..120 {
            phys.step(&world, PHYSICS_DT);
        }
        assert!(
            phys.get(bottom).expect("alive").sleep.asleep,
            "bottom must settle first"
        );

        let top = phys.spawn(cube_body(&reg, 4, Vec3::new(16.0, 5.2, 16.0)));
        for _ in 0..240 {
            phys.step(&world, PHYSICS_DT);
        }

        let b = phys.get(bottom).expect("alive");
        let t = phys.get(top).expect("alive");
        assert!(b.sleep.asleep, "bottom cube must (re)settle asleep");
        assert!(t.sleep.asleep, "top cube must settle asleep");
        // Bottom rests on the 4 m floor (0.4 m cube), top rests on bottom.
        assert!(
            (b.aabb_min.y - 4.0).abs() < 0.02,
            "bottom rest height {}",
            b.aabb_min.y
        );
        assert!(
            (t.aabb_min.y - b.aabb_max.y).abs() < 0.02,
            "top must rest on bottom: top_min={} bottom_max={}",
            t.aabb_min.y,
            b.aabb_max.y
        );
    }

    #[test]
    fn pile_of_twelve_in_a_pit_does_not_explode() {
        let reg = registry();
        // A 1.5 m x 1.5 m walled pit on the floor (floor at 4 m, walls to 7 m).
        let mut world = floored_world();
        let (lo, hi) = (14.0_f32, 15.5_f32);
        let wall_top = 70; // 7.0 m at 0.1 m voxels
        let floor_top = 40; // 4.0 m
        let s = 0.1;
        let iv = |m: f32| (m / s) as i32;
        world.fill_box(
            IVec3::new(iv(lo), floor_top, iv(lo)),
            IVec3::new(iv(lo) + 1, wall_top, iv(hi)),
            STONE,
        );
        world.fill_box(
            IVec3::new(iv(hi), floor_top, iv(lo)),
            IVec3::new(iv(hi) + 1, wall_top, iv(hi)),
            STONE,
        );
        world.fill_box(
            IVec3::new(iv(lo), floor_top, iv(lo)),
            IVec3::new(iv(hi), wall_top, iv(lo) + 1),
            STONE,
        );
        world.fill_box(
            IVec3::new(iv(lo), floor_top, iv(hi)),
            IVec3::new(iv(hi), wall_top, iv(hi) + 1),
            STONE,
        );

        let mut phys = PhysicsWorld::new();
        let mut rng = Rng(0xBADC0DE);
        let mut ids = Vec::new();
        for i in 0..12 {
            let com = Vec3::new(
                rng.range(14.3, 15.2),
                5.0 + i as f32 * 0.35,
                rng.range(14.3, 15.2),
            );
            ids.push(phys.spawn(cube_body(&reg, 3, com)));
        }

        let mut max_speed = 0.0f32;
        for step in 0..900 {
            phys.step(&world, PHYSICS_DT);
            for id in &ids {
                let b = phys.get(*id).expect("alive");
                assert!(b.vel.is_finite() && b.pos.is_finite(), "NaN at step {step}");
                max_speed = max_speed.max(b.vel.length());
            }
        }
        assert!(
            max_speed < 25.0,
            "solver exploded: peak speed {max_speed} m/s"
        );
        for (i, id) in ids.iter().enumerate() {
            let b = phys.get(*id).expect("alive");
            assert!(
                b.pos.x > lo && b.pos.x < hi && b.pos.z > lo && b.pos.z < hi,
                "body {i} escaped the pit: {:?}",
                b.pos
            );
        }
        let asleep = ids
            .iter()
            .filter(|id| phys.get(**id).expect("alive").sleep.asleep)
            .count();
        assert!(
            asleep >= 10,
            "expected most of the pile asleep, got {asleep}/12"
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

    /// The steady contact holding a body up while it settles (not yet
    /// asleep) must never be misreported as a fresh impact -- see
    /// `MIN_IMPACT_APPROACH_SPEED_M_S`'s doc comment for the exact bug this
    /// guards against (repeated spurious impacts every settling frame,
    /// which for a fragile material meant continuous re-fracturing).
    #[test]
    fn settling_after_a_hard_landing_reports_no_further_impacts() {
        let reg = registry();
        let world = floored_world();
        let mut phys = PhysicsWorld::new();
        let id = phys.spawn(cube_body(&reg, 4, Vec3::new(16.0, 10.0, 16.0)));

        let mut saw_landing = false;
        let mut saw_impact_after_landing = false;
        for _ in 0..300 {
            let events = phys.step(&world, PHYSICS_DT);
            if saw_landing && !events.is_empty() {
                saw_impact_after_landing = true;
            }
            if !events.is_empty() {
                saw_landing = true;
            }
            if phys.get(id).expect("alive").sleep.asleep {
                break;
            }
        }
        assert!(saw_landing, "a body falling 6 m onto stone must report an impact");
        assert!(
            !saw_impact_after_landing,
            "settling under steady contact must not keep reporting fresh impacts"
        );
    }
}
