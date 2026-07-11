//! The rigidbody world: arena, sequential-impulse solver, sleeping.
//!
//! Fixed-timestep with substeps; velocity-level Baumgarte stabilization;
//! Coulomb friction on two tangents; warm starting keyed by (body, voxel,
//! face) so stacks converge across frames; settled bodies sleep at ~zero
//! cost until an impulse or a nearby world edit wakes them.

use glam::{Mat3, Quat, Vec3};
use vox_core::consts::{
    CLUTTER_LIFETIME_MAX_S, CLUTTER_LIFETIME_MIN_S, CLUTTER_MAX_VOXELS, CONTACT_SLOP, GRAVITY,
    SLEEP_FRAMES, SOLVER_ITERS, SUBSTEPS,
};
use vox_core::{FxHashMap, Tunables};
use vox_world::{SolidLookup, Voxel, World};

use crate::body::{Body, BodyId};
use crate::broadphase::Broadphase;
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

/// A distance constraint between two bodies. Maintains a fixed rest length
/// between two anchor points. Used for rope/chain segments.
#[derive(Clone, Debug)]
pub struct Joint {
    /// Slot index of body A.
    pub body_a: usize,
    /// Slot index of body B.
    pub body_b: usize,
    /// Anchor on body A, relative to COM, body-local frame (meters).
    pub anchor_a: Vec3,
    /// Anchor on body B, relative to COM, body-local frame.
    pub anchor_b: Vec3,
    /// Rest length between anchors (meters).
    pub rest_length: f32,
    /// Accumulated Lagrange multiplier (warm start).
    pub acc_lambda: f32,
    /// Compliance (inverse stiffness). 0 = rigid.
    pub compliance: f32,
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
/// Hard angular velocity ceiling (rad/s). Small bodies have a
/// disproportionately tiny moment of inertia, so an off-center contact
/// impulse (landing on one corner, say) can spin one up far harder than the
/// same impulse would a large body -- a probe against a 2x2x1 chip (the
/// same shape real debris chips use) landing corner-first measured a
/// *stable* 50-60 rad/s that never decayed, instead of settling. That's
/// bad on its own (a body spinning ~9 full rotations a second at 60Hz
/// covers ~50 degrees *per step*, so the render-side slerp between this
/// step's start/end orientation increasingly aliases into visible
/// judder -- "stutters and glitches," worse the smaller/faster the body)
/// and compounds into real lag too: `sleep_ang` (0.2 rad/s) is never met,
/// so a body stuck like this never sleeps and keeps costing full
/// broadphase/narrowphase/solver/render work forever. Comfortably above
/// every intentional spin kick this engine hands out (`BLAST_SPIN_MAX` is
/// the largest, at 3.0), comfortably below the runaway values observed.
const MAX_ANGULAR_SPEED_RAD_S: f32 = 20.0;

/// Baseline angular drag applied to every awake body, per second -- the
/// solver otherwise has *no* mechanism that removes rotation except contact
/// friction, and friction only has leverage when the contact points sit
/// off the spin axis, so debris tumbling in flight (or spinning on a
/// near-degenerate axis) held its spin literally forever. Small on
/// purpose: a blast-kicked fragment (`BLAST_SPIN_MAX` = 3 rad/s) still has
/// most of its tumble after a full second of flight.
const ANGULAR_DAMPING_AIR: f32 = 0.3;
/// Water density in kg/m³ for buoyancy calculations. Matches the water
/// material's density in core.toml.
const WATER_DENSITY: f32 = 1000.0;
/// Per-second velocity/angular damping while submerged in water — higher
/// than air drag to make water feel viscous.
const WATER_DRAG: f32 = 3.0;
/// Extra rolling-resistance drag, per second, for a *small* body that had
/// at least one contact this substep. Small debris has a disproportionately
/// tiny moment of inertia and only a couple of contact points, so contact
/// friction alone can leave it rattling/spinning right at the sleep
/// thresholds indefinitely -- never asleep (its whole contact island stays
/// awake with it, feeding "lots of debris causes lag"), and visibly
/// twitching the whole time. Real rubble stops tumbling almost immediately
/// once it's on the ground; this makes ours do the same. Deliberately not
/// applied to large bodies: a tree tipping over *is* rotation sustained
/// through a ground contact, and damping that would make felled trees fall
/// in slow motion.
const ANGULAR_DAMPING_ROLLING: f32 = 6.0;
/// A body at or below this many surface sample points counts as "small"
/// for [`ANGULAR_DAMPING_ROLLING`]. Debris chips have 3-8, a 3^3 rubble
/// cube has 26, a 4^3 fragment already has 56, a tree trunk has thousands.
const SMALL_BODY_MAX_SURFACE_POINTS: usize = 32;

/// Iterations of the positional (split-impulse) penetration-recovery pass.
const POSITION_ITERS: usize = 2;
/// Ceiling on how far one contact may move a body per position iteration,
/// in meters -- keeps a deeply-buried body (a freshly-spawned debris chip
/// overlaps the fragment it chipped off of by up to a voxel) easing out
/// over a few frames instead of teleporting.
const MAX_POSITION_CORRECTION_M: f32 = 0.05;

/// The dynamic side of the engine: all rigid bodies plus solver state.
#[derive(Default)]
pub struct PhysicsWorld {
    slots: Vec<Option<Body>>,
    generations: Vec<u32>,
    free: Vec<usize>,
    warm: FxHashMap<ContactKey, (f32, f32, f32)>,
    broadphase: Broadphase,
    contact_flags: Vec<bool>,
    pos_corr: Vec<Vec3>,
    pub tunables: Tunables,
    lifetime_rng: u64,
    joints: Vec<Joint>,
    /// Fluid materials for buoyancy (water, muddy_water, ...). Empty (default)
    /// disables buoyancy — bodies fall through fluids. Set by the app at
    /// construction from the registry's fluid materials.
    fluid_voxels: Vec<Voxel>,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            lifetime_rng: 0x2545_F491_4F6C_DD1D,
            ..Self::default()
        }
    }

    /// Set a single water material id for buoyancy (backward compat). Called
    /// once by the app at construction. Replaces any prior fluid set with a
    /// single-element vec.
    pub fn set_water_voxel(&mut self, v: Voxel) {
        self.fluid_voxels = vec![v];
    }

    /// Set the fluid materials for buoyancy (water, muddy_water, ...).
    /// Called once by the app at construction. Replaces any prior fluid set.
    pub fn set_fluid_voxels(&mut self, fluids: Vec<Voxel>) {
        self.fluid_voxels = fluids;
    }

    /// xorshift64* -- deterministic, dependency-free spawn jitter (same
    /// algorithm as `vox_app::particles::ParticleSystem::next_f32`).
    fn next_lifetime_jitter(&mut self) -> f32 {
        self.lifetime_rng ^= self.lifetime_rng << 13;
        self.lifetime_rng ^= self.lifetime_rng >> 7;
        self.lifetime_rng ^= self.lifetime_rng << 17;
        ((self.lifetime_rng >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Insert a body, waking it. Returns its handle. Small "clutter" bodies
    /// (see `vox_core::consts::CLUTTER_MAX_VOXELS`) are given a randomized
    /// 35-60s lifetime here, at the single choke point every body passes
    /// through on the way into the world -- see `tick_lifetimes` for where
    /// that countdown is spent.
    pub fn spawn(&mut self, mut body: Body) -> BodyId {
        if body.lifetime_s.is_none() && body.grid.solid_count() <= CLUTTER_MAX_VOXELS {
            let jitter = self.next_lifetime_jitter();
            body.lifetime_s = Some(
                CLUTTER_LIFETIME_MIN_S + jitter * (CLUTTER_LIFETIME_MAX_S - CLUTTER_LIFETIME_MIN_S),
            );
        }
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

    /// Tick every timed body's countdown (see `spawn`) by `dt` seconds and
    /// despawn any that reach zero, returning their ids so the caller can
    /// drop associated GPU meshes -- same contract as `clear_sleeping`. A
    /// cheap no-op scan for the overwhelming majority of bodies, which carry
    /// no lifetime at all.
    pub fn tick_lifetimes(&mut self, dt: f32) -> Vec<BodyId> {
        let mut expired = Vec::new();
        for (slot, body) in self.slots.iter_mut().enumerate() {
            let Some(body) = body else { continue };
            let Some(remaining) = body.lifetime_s.as_mut() else {
                continue;
            };
            *remaining -= dt;
            if *remaining <= 0.0 {
                expired.push(BodyId {
                    slot: slot as u32,
                    generation: self.generations[slot],
                });
            }
        }
        for &id in &expired {
            self.despawn(id);
        }
        expired
    }

    /// Remove a body if the handle is current.
    pub fn despawn(&mut self, id: BodyId) {
        let slot = id.slot as usize;
        if self.generations.get(slot) == Some(&id.generation) && self.slots[slot].is_some() {
            self.remove_joints_for_slot(slot);
            self.slots[slot] = None;
            self.generations[slot] += 1;
            self.free.push(slot);
        }
    }

    /// Add a distance joint between two bodies. Returns the joint index.
    pub fn add_joint(
        &mut self,
        a: BodyId,
        b: BodyId,
        anchor_a: Vec3,
        anchor_b: Vec3,
        rest_length: f32,
        compliance: f32,
    ) -> usize {
        let joint = Joint {
            body_a: a.slot as usize,
            body_b: b.slot as usize,
            anchor_a,
            anchor_b,
            rest_length,
            acc_lambda: 0.0,
            compliance,
        };
        self.joints.push(joint);
        self.joints.len() - 1
    }

    /// Check if two body slots are connected by a joint.
    fn are_joined(&self, a: usize, b: usize) -> bool {
        self.joints.iter().any(|j| {
            (j.body_a == a && j.body_b == b) || (j.body_a == b && j.body_b == a)
        })
    }

    /// Remove all joints referencing a given body slot (called on despawn).
    fn remove_joints_for_slot(&mut self, slot: usize) {
        self.joints.retain(|j| j.body_a != slot && j.body_b != slot);
    }

    /// Read access to joints (for debugging/rendering).
    pub fn joints(&self) -> &[Joint] {
        &self.joints
    }

    pub fn get(&self, id: BodyId) -> Option<&Body> {
        let slot = id.slot as usize;
        if self.generations.get(slot) == Some(&id.generation) {
            self.slots[slot].as_ref()
        } else {
            None
        }
    }

    /// Mutable reference to a live body, if it exists.
    pub fn get_mut(&mut self, id: BodyId) -> Option<&mut Body> {
        let slot = id.slot as usize;
        if self.generations.get(slot) == Some(&id.generation) {
            self.slots[slot].as_mut()
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
                self.remove_joints_for_slot(slot);
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
        let mut peaks: FxHashMap<usize, (f32, Vec3, Vec3)> = FxHashMap::default();
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

        // Tick damage decay on awake bodies with damage. Once per full step
        // (not per substep): 0.05/s means ~20s to heal, so substep granularity
        // is pointless. Sleeping bodies freeze their damage until woken.
        let decay = vox_core::consts::DAMAGE_DECAY_PER_S;
        for body in self.slots.iter_mut().flatten() {
            if !body.sleep.asleep && body.grid.has_damage() {
                body.grid.tick_damage_decay(dt, decay);
            }
        }

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
        let islands = self.islands();
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
    fn islands(&mut self) -> Vec<Vec<usize>> {
        let mut parent: Vec<usize> = (0..self.slots.len()).collect();
        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }
        fn union(parent: &mut [usize], a: usize, b: usize) {
            let (ra, rb) = (find(parent, a), find(parent, b));
            if ra != rb {
                parent[ra] = rb;
            }
        }
        for &(a, b) in self.broadphase.candidate_pairs(&self.slots) {
            union(&mut parent, a, b);
        }
        // Joint-connected bodies are in the same island so joined bodies
        // sleep and wake together.
        for j in &self.joints {
            if self.slots[j.body_a].is_some() && self.slots[j.body_b].is_some() {
                union(&mut parent, j.body_a, j.body_b);
            }
        }
        let mut groups: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        for (slot, entry) in self.slots.iter().enumerate() {
            if entry.is_some() {
                let root = find(&mut parent, slot);
                groups.entry(root).or_default().push(slot);
            }
        }
        groups.into_values().collect()
    }

    fn substep(&mut self, world: &World, h: f32, peaks: &mut FxHashMap<usize, (f32, Vec3, Vec3)>) {
        // Refresh every live body's cached world inverse inertia -- rotation
        // only changes at the end of a substep, so this is constant across
        // the whole contact solve below (see `Body::inv_iw`'s docs).
        // Sleeping bodies included: a sleeping pair target still contributes
        // its inertia to `pair_contacts`' effective-mass terms.
        for body in self.slots.iter_mut().flatten() {
            body.inv_iw = body.inv_inertia_world();
        }

        let mut lookup = SolidLookup::new(world);
        let mut contacts: Vec<Contact> = Vec::new();
        // Integrate velocities and collect contacts.
        for (slot, entry) in self.slots.iter_mut().enumerate() {
            let Some(body) = entry else { continue };
            if body.sleep.asleep {
                continue;
            }
            body.vel.y -= GRAVITY * h;
            // Buoyancy: if the body's bottom is submerged in a fluid voxel
            // (water or any registered fluid), apply an upward force.
            // Lightweight: samples the AABB bottom center, not per-voxel.
            let s = 2.0 * body.half_voxel;
            let bottom = Vec3::new(
                (body.aabb_min.x + body.aabb_max.x) * 0.5,
                body.aabb_min.y,
                (body.aabb_min.z + body.aabb_max.z) * 0.5,
            );
            let bottom_vox = vox_core::voxel_at(bottom, s);
            if world.in_bounds(bottom_vox) {
                if self.fluid_voxels.contains(&world.get_voxel(bottom_vox)) {
                    let body_height = (body.aabb_max.y - body.aabb_min.y).max(1e-6);
                    let submerge_depth = body_height.min(body.aabb_max.y - bottom.y + s);
                    let fraction = (submerge_depth / body_height).clamp(0.0, 1.0);
                    let dims = body.grid.dims;
                    let volume = (dims.x * dims.y * dims.z) as f32 * s * s * s;
                    let mass = body.mass();
                    let avg_density = mass / volume.max(1e-6);
                    let buoy_accel = (WATER_DENSITY / avg_density - 1.0) * GRAVITY * fraction;
                    if buoy_accel > 0.0 {
                        body.vel.y += buoy_accel * h;
                    }
                    let drag = WATER_DRAG * fraction;
                    body.vel /= 1.0 + drag * h;
                    body.omega /= 1.0 + drag * h;
                }
            }
            if body.vel.length() > MAX_SPEED {
                body.vel = body.vel.normalize() * MAX_SPEED;
            }
            if body.omega.length() > MAX_ANGULAR_SPEED_RAD_S {
                body.omega = body.omega.normalize() * MAX_ANGULAR_SPEED_RAD_S;
            }
            debug_assert!(body.vel.is_finite() && body.pos.is_finite());
            world_contacts(body, slot, &mut contacts, &mut lookup);
        }

        // Body-body narrowphase over broadphase candidates. One staging
        // buffer reused across every pair (`Vec::append` drains it but
        // keeps its capacity) instead of a fresh allocation per touching
        // pair per substep.
        let mut staged: Vec<Contact> = Vec::new();
        let pairs: Vec<(usize, usize)> = self.broadphase.candidate_pairs(&self.slots).to_vec();
        for (a, b) in pairs {
            // Skip contact between jointed bodies — the joint handles their
            // connection. Contacts between rope segments fight the joint.
            if self.are_joined(a, b) {
                continue;
            }
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

            staged.clear();
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
        // Joint warm start: disabled — was causing energy injection and
        // explosion. The velocity solve converges within 8 iterations for
        // the small joint counts in a rope chain. Just reset accumulators.
        for j in &mut self.joints {
            j.acc_lambda = 0.0;
        }

        // Velocity iterations: normal impulse stopping approach only, then
        // friction clamped to the Coulomb cone. Deliberately *no* Baumgarte
        // bias here: bias turns penetration depth into a velocity target,
        // and a contact will spend however much impulse the touching
        // bodies' masses require to reach it -- real momentum injected from
        // nothing. The measured failure: a debris chip wedged under a
        // settled 5.6 t block (chips spawn overlapping their parent
        // fragment by up to a voxel, so this is routine, not exotic) had
        // its floor contact demanding "chip up" while its block contact
        // demanded "chip down relative to block"; the only way the solver
        // could satisfy both was to lift the block, which it did --
        // block-scale impulses ramping it to ~1 m/s and shoving it
        // centimeters off its resting spot. Penetration is instead
        // recovered *positionally* (split impulse, below), which by
        // construction cannot add kinetic energy to anything.
        for _ in 0..SOLVER_ITERS {
            for c in &mut contacts {
                let vn = Self::relative_velocity(&self.slots, c).dot(c.normal);
                let lambda = -vn / c.kn;
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
            // Joint distance constraints (interleaved with contacts for
            // convergence). Read block computes all Copy-typed values so the
            // immutable borrow of self.slots ends before two_mut.
            for j in &mut self.joints {
                let (ba, bb) = match (self.slots[j.body_a].as_ref(), self.slots[j.body_b].as_ref()) {
                    (Some(a), Some(b)) => (a, b),
                    _ => continue,
                };
                if ba.sleep.asleep && bb.sleep.asleep {
                    continue;
                }
                let asleep_a = ba.sleep.asleep;
                let asleep_b = bb.sleep.asleep;
                let ra = ba.rot * j.anchor_a;
                let rb = bb.rot * j.anchor_b;
                let d = (bb.pos + rb) - (ba.pos + ra);
                let dist = d.length();
                if dist < 1e-6 {
                    continue;
                }
                let n = d / dist;
                // Velocity-based solve: cancel relative velocity along the
                // constraint axis. No position correction (disabled to
                // prevent chain feedback loops); slight sag is acceptable.
                let v_rel = (bb.vel + bb.omega.cross(rb)) - (ba.vel + ba.omega.cross(ra));
                let vn = v_rel.dot(n);
                let ima = if asleep_a { 0.0 } else { ba.inv_mass };
                let imb = if asleep_b { 0.0 } else { bb.inv_mass };
                let iwa = if asleep_a { Mat3::ZERO } else { ba.inv_iw };
                let iwb = if asleep_b { Mat3::ZERO } else { bb.inv_iw };
                let ra_cross_n = ra.cross(n);
                let rb_cross_n = rb.cross(n);
                let keff = ima + imb
                    + iwa.mul_vec3(ra_cross_n).dot(ra_cross_n)
                    + iwb.mul_vec3(rb_cross_n).dot(rb_cross_n);
                if keff <= 0.0 {
                    continue;
                }
                let lambda = -vn / (keff + j.compliance);
                // Don't accumulate — each iteration applies its own correction.
                let p = n * lambda;
                let (ba, bb) = two_mut(&mut self.slots, j.body_a, j.body_b);
                if !asleep_a {
                    ba.vel += p * ba.inv_mass;
                    ba.omega += ba.inv_iw * ra.cross(p);
                }
                if !asleep_b {
                    bb.vel -= p * bb.inv_mass;
                    bb.omega -= bb.inv_iw * rb.cross(p);
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

        // Which bodies touched anything this substep, for rolling
        // resistance below.
        self.contact_flags.clear();
        self.contact_flags.resize(self.slots.len(), false);
        for c in &contacts {
            self.contact_flags[c.body] = true;
            if let Some(b) = c.body_b {
                self.contact_flags[b] = true;
            }
        }

        // Integrate positions and orientations. Angular velocity is
        // clamped again here (not just at the top of the substep, before
        // this substep's own contact impulses were applied) so a spike from
        // *this* substep's impulse resolution never gets integrated into
        // the rotation at full strength, and never persists past this
        // substep's boundary either -- see `MAX_ANGULAR_SPEED_RAD_S`.
        // Angular damping also lives here, after the solve, so it acts on
        // the final post-impulse spin: baseline air drag for everyone, plus
        // rolling resistance for small grounded debris (see the constants'
        // docs for why each exists).
        for (slot, entry) in self.slots.iter_mut().enumerate() {
            let Some(body) = entry else { continue };
            if body.sleep.asleep {
                continue;
            }
            body.pos += body.vel * h;
            let mut damping = ANGULAR_DAMPING_AIR;
            if self.contact_flags[slot] && body.surface.len() <= SMALL_BODY_MAX_SURFACE_POINTS {
                damping += ANGULAR_DAMPING_ROLLING;
            }
            body.omega /= 1.0 + h * damping;
            if body.omega.length() > MAX_ANGULAR_SPEED_RAD_S {
                body.omega = body.omega.normalize() * MAX_ANGULAR_SPEED_RAD_S;
            }
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

        // Split-impulse penetration recovery: resolve overlap by *moving*
        // bodies apart, weighted by inverse mass, never by changing their
        // velocities (see the velocity-iterations comment above for the
        // energy-injection failure this replaces). Sequential with a
        // per-body running total so a slab resting on hundreds of similar
        // contacts converges to one correction, not hundreds stacked; the
        // inverse-mass split means a chip pinched under a heavy block eases
        // itself out while the block, thousands of times its mass, stays
        // put to within micrometers.
        self.pos_corr.clear();
        self.pos_corr.resize(self.slots.len(), Vec3::ZERO);
        for _ in 0..POSITION_ITERS {
            for c in &contacts {
                let corr_b = c.body_b.map_or(Vec3::ZERO, |b| self.pos_corr[b]);
                let already = (self.pos_corr[c.body] - corr_b).dot(c.normal);
                let remaining = (c.depth - CONTACT_SLOP) - already;
                if remaining <= 0.0 {
                    continue;
                }
                let push = (self.tunables.contact_beta * remaining).min(MAX_POSITION_CORRECTION_M);
                match c.body_b {
                    Some(b) => {
                        let ia = self.slots[c.body].as_ref().map_or(0.0, |x| x.inv_mass);
                        let ib = self.slots[b].as_ref().map_or(0.0, |x| x.inv_mass);
                        let w = ia + ib;
                        if w <= 0.0 {
                            continue;
                        }
                        self.pos_corr[c.body] += c.normal * (push * ia / w);
                        self.pos_corr[b] -= c.normal * (push * ib / w);
                    }
                    None => self.pos_corr[c.body] += c.normal * push,
                }
            }
            // Joint position correction disabled — velocity solve alone
            // prevents explosion. Position correction in a joint chain
            // creates a feedback loop (correcting one joint violates the
            // next), causing solver divergence and frame-rate death.
            // The velocity-only solve allows slight sag under gravity,
            // which is acceptable for flexible rope.
        }
        for (slot, entry) in self.slots.iter_mut().enumerate() {
            let Some(body) = entry else { continue };
            let corr = self.pos_corr[slot];
            if corr != Vec3::ZERO && !body.sleep.asleep {
                body.pos += corr;
                body.refresh_aabb();
            }
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
        // `inv_iw` is the substep-refreshed cache (see its field docs) --
        // this function runs ~25x per contact per substep, and re-deriving
        // the world inertia from the quaternion here was the hottest single
        // operation in a many-contact debris pile.
        if let Some(b) = c.body_b {
            let (ba, bb) = two_mut(slots, c.body, b);
            ba.vel += p * ba.inv_mass;
            ba.omega += ba.inv_iw * c.r_arm.cross(p);
            bb.vel -= p * bb.inv_mass;
            bb.omega -= bb.inv_iw * c.r_arm_b.cross(p);
        } else {
            let body = slots[c.body].as_mut().expect("contact body alive");
            body.vel += p * body.inv_mass;
            body.omega += body.inv_iw * c.r_arm.cross(p);
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
        assert!(
            saw_impact,
            "a body falling 6 m onto stone must report an impact"
        );
        assert!(max_impulse > 0.0);

        // Once settled and asleep, further steps must be quiet -- no
        // spurious impacts from steady resting contact.
        let quiet = phys.step(&world, PHYSICS_DT);
        assert!(
            quiet.is_empty(),
            "resting body must not report an impact: {quiet:?}"
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
        assert!(
            saw_landing,
            "a body falling 6 m onto stone must report an impact"
        );
        assert!(
            !saw_impact_after_landing,
            "settling under steady contact must not keep reporting fresh impacts"
        );
    }

    /// A small body's tiny moment of inertia means an off-center contact
    /// (landing on one corner) can spin it up far harder than the same
    /// impulse would a large body -- an earlier version of this had no
    /// ceiling on angular velocity at all (unlike linear, which already had
    /// `MAX_SPEED`), and a 2x2x1 chip landing corner-first would settle
    /// into a *stable* 50-60 rad/s that never decayed: never quiet enough
    /// to sleep (costing broadphase/narrowphase/render work forever), and
    /// covering ~50 degrees of rotation *per physics step* at 60Hz, which
    /// is well into the range where the render-side slerp between a step's
    /// start/end orientation reads as visible judder rather than a smooth
    /// spin. `MAX_ANGULAR_SPEED_RAD_S` bounds it the same way `MAX_SPEED`
    /// already bounds linear velocity.
    #[test]
    fn a_small_chip_landing_corner_first_settles_instead_of_spinning_forever() {
        let reg = registry();
        let world = floored_world();
        let mut phys = PhysicsWorld::new();
        let grid = VoxelGrid::new(IVec3::new(2, 2, 1), vec![Voxel(1); 4]);
        let mut body = Body::from_grid(grid, &reg, 0.1, Vec3::new(16.0, 10.0, 16.0)).unwrap();
        body.rot = Quat::from_euler(glam::EulerRot::XYZ, 0.4, 0.3, 0.2);
        body.prev_rot = body.rot;
        body.vel = Vec3::new(3.0, -2.0, 1.5);
        let id = phys.spawn(body);

        let mut slept = false;
        for _ in 0..600 {
            phys.step(&world, PHYSICS_DT);
            let b = phys.get(id).expect("alive");
            assert!(
                b.omega.length() <= MAX_ANGULAR_SPEED_RAD_S + 1e-3,
                "angular velocity must never exceed the hard ceiling: {}",
                b.omega.length()
            );
            if b.sleep.asleep {
                slept = true;
                break;
            }
        }
        assert!(
            slept,
            "a small chip must eventually settle and sleep, not spin forever"
        );
    }

    /// A small chip already on the ground, spun hard around the vertical
    /// axis (where friction has the least leverage on a flat-bottomed
    /// shape), must stop and sleep quickly -- rolling resistance
    /// (`ANGULAR_DAMPING_ROLLING`) exists precisely so grounded rubble
    /// can't sit there spinning/rattling at the sleep threshold forever,
    /// keeping its whole contact island awake with it.
    #[test]
    fn a_grounded_spinning_chip_stops_and_sleeps_quickly() {
        let reg = registry();
        let world = floored_world();
        let mut phys = PhysicsWorld::new();
        let grid = VoxelGrid::new(IVec3::new(2, 2, 1), vec![Voxel(1); 4]);
        let mut body = Body::from_grid(grid, &reg, 0.1, Vec3::new(16.0, 4.2, 16.0)).unwrap();
        body.omega = Vec3::new(0.0, 10.0, 0.0);
        let id = phys.spawn(body);
        let mut slept_at = None;
        for i in 0..600 {
            phys.step(&world, PHYSICS_DT);
            if phys.get(id).unwrap().sleep.asleep {
                slept_at = Some(i);
                break;
            }
        }
        let slept_at = slept_at.expect("a grounded spinning chip must sleep within 10 s");
        assert!(
            slept_at < 180,
            "rolling resistance should stop it in ~1 s, not {slept_at} steps"
        );
    }

    /// Air drag on rotation is a floor, not a brake: a body tumbling in
    /// free flight (no contacts) must keep the large majority of its spin
    /// over half a second -- blast-kicked debris visibly tumbling as it
    /// flies is a feature, and rolling resistance must not apply mid-air.
    #[test]
    fn free_flight_tumble_is_barely_damped() {
        let reg = registry();
        let world = floored_world();
        let mut phys = PhysicsWorld::new();
        let grid = VoxelGrid::new(IVec3::new(2, 2, 1), vec![Voxel(1); 4]);
        let mut body = Body::from_grid(grid, &reg, 0.1, Vec3::new(16.0, 20.0, 16.0)).unwrap();
        body.omega = Vec3::new(0.0, 3.0, 0.0);
        let id = phys.spawn(body);
        for _ in 0..30 {
            phys.step(&world, PHYSICS_DT); // 0.5 s of free fall, ~1.2 m -- nowhere near the floor
        }
        let om = phys.get(id).unwrap().omega.length();
        assert!(
            om > 2.5,
            "half a second of air drag must leave most of a 3 rad/s tumble: {om}"
        );
        assert!(om < 3.0, "but *some* drag must exist: {om}");
    }

    /// A debris chip spawned half-buried under a settled heavy block --
    /// exactly what fracture chips do routinely (they spawn at
    /// removed-voxel centers, overlapping surviving material by up to a
    /// voxel) -- must not disturb the block. Under the old Baumgarte
    /// velocity bias, the solver could only satisfy the trapped chip's two
    /// contradictory contacts (floor: "chip up"; block: "chip down,
    /// relative to the block") by *lifting the block*, and it did: ~1 m/s
    /// of ramping velocity and centimeters of drift on a 5.6 t block, from
    /// one three-voxel chip. Split-impulse (positional) penetration
    /// recovery cannot inject momentum by construction; this pins that.
    #[test]
    fn a_chip_pinched_under_a_heavy_block_does_not_move_the_block() {
        let reg = registry();
        let world = floored_world();
        let mut phys = PhysicsWorld::new();
        let big = phys.spawn(cube_body(&reg, 20, Vec3::new(16.0, 5.2, 16.0)));
        for _ in 0..300 {
            phys.step(&world, PHYSICS_DT);
        }
        assert!(
            phys.get(big).unwrap().sleep.asleep,
            "big cube must settle first"
        );
        let rest_pos = phys.get(big).unwrap().pos;

        let corner = rest_pos - Vec3::new(0.95, 0.95, 0.95);
        let grid = VoxelGrid::new(IVec3::new(2, 2, 1), vec![Voxel(1); 4]);
        let chip = phys.spawn(Body::from_grid(grid, &reg, 0.1, corner).unwrap());

        let mut max_big_vel = 0.0f32;
        let mut max_chip_vel = 0.0f32;
        for _ in 0..240 {
            phys.step(&world, PHYSICS_DT);
            max_big_vel = max_big_vel.max(phys.get(big).unwrap().vel.length());
            if let Some(c) = phys.get(chip) {
                max_chip_vel = max_chip_vel.max(c.vel.length());
            }
        }
        assert!(
            max_big_vel < 0.05,
            "the block must stay put, not get hurled: peaked at {max_big_vel} m/s"
        );
        assert!(
            max_chip_vel < 2.0,
            "the chip must ease out, not launch: peaked at {max_chip_vel} m/s"
        );
        let drift = (phys.get(big).unwrap().pos - rest_pos).length();
        assert!(drift < 0.01, "the block must not creep: drifted {drift} m");
    }

    #[test]
    fn clutter_sized_bodies_get_a_lifetime_and_larger_ones_dont() {
        let reg = registry();
        let mut phys = PhysicsWorld::new();

        let clutter = cube_body(&reg, 1, Vec3::ZERO); // 1 voxel <= CLUTTER_MAX_VOXELS
        let clutter_id = phys.spawn(clutter);
        assert!(
            phys.get(clutter_id).unwrap().lifetime_s.is_some(),
            "a 1-voxel body must be timed"
        );
        let life = phys.get(clutter_id).unwrap().lifetime_s.unwrap();
        assert!(
            (CLUTTER_LIFETIME_MIN_S..=CLUTTER_LIFETIME_MAX_S).contains(&life),
            "lifetime {life} must land in the configured 35-60s window"
        );

        let permanent = cube_body(&reg, 2, Vec3::splat(10.0)); // 8 voxels
        let permanent_id = phys.spawn(permanent);
        assert!(
            phys.get(permanent_id).unwrap().lifetime_s.is_none(),
            "an 8-voxel body must not be timed"
        );
    }

    #[test]
    fn tick_lifetimes_despawns_expired_clutter_and_leaves_others() {
        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let clutter_id = phys.spawn(cube_body(&reg, 1, Vec3::ZERO));
        let permanent_id = phys.spawn(cube_body(&reg, 2, Vec3::splat(10.0)));

        // Nowhere near expiry yet.
        let expired = phys.tick_lifetimes(1.0);
        assert!(expired.is_empty());
        assert!(phys.get(clutter_id).is_some());

        // Push it past even the longest possible lifetime.
        let expired = phys.tick_lifetimes(CLUTTER_LIFETIME_MAX_S + 1.0);
        assert_eq!(expired, vec![clutter_id]);
        assert!(
            phys.get(clutter_id).is_none(),
            "expired clutter must be despawned"
        );
        assert!(
            phys.get(permanent_id).is_some(),
            "an untimed body must survive any number of ticks"
        );
    }
    /// A body less dense than the fluid it sits in must float when that
    /// fluid is a *second* registered fluid (muddy_water), not just plain
    /// water. Before generalizing `water_voxel` to `fluid_voxels`, only the
    /// single configured water material triggered buoyancy, so a body in
    /// muddy_water sank straight through it to the floor.
    ///
    /// This test uses a dedicated registry so the fluids can be marked
    /// `solid = false` and a `set_solid_table` attached — without it,
    /// `World::solid` falls back to "any non-air voxel is solid"
    /// (world.rs legacy path), muddy_water would act as a solid platform,
    /// and the body would rest on top of it regardless of buoyancy (a
    /// false green). The body is built from "foam" (density 300) so
    /// `buoy_accel = (1000/300 - 1)*GRAVITY` exceeds gravity and the body
    /// actually rises; wood (700) is denser than water's buoyant reference
    /// (1000) net of gravity and would sink.
    #[test]
    fn body_floats_in_a_second_fluid_muddy_water() {
        // ids: 0=air, 1=stone, 2=muddy_water, 3=water, 4=foam
        let reg = MaterialRegistry::from_toml_str(
            r#"
            [[material]]
            name = "stone"
            color = [0.5, 0.5, 0.5]
            density = 2700.0
            strength = 50.0

            [[material]]
            name = "muddy_water"
            color = [0.4, 0.3, 0.1]
            density = 1100.0
            strength = 0.0
            solid = false
            fluid = true

            [[material]]
            name = "water"
            color = [0.2, 0.4, 0.8]
            density = 1000.0
            strength = 0.0
            solid = false
            fluid = true

            [[material]]
            name = "foam"
            color = [0.9, 0.9, 0.9]
            density = 300.0
            strength = 1.0
            "#,
            "buoyancy_test.toml",
        )
        .expect("registry");

        const STONE: Voxel = Voxel(1);
        const MUDDY_WATER: Voxel = Voxel(2);
        const WATER: Voxel = Voxel(3);
        const FOAM: Voxel = Voxel(4);

        // Build the solidity table from the registry so fluids (ids 2, 3)
        // read as non-solid and the body passes through them — buoyancy is
        // then the only upward force.
        let solid_table: Vec<bool> = (0..reg.len())
            .map(|i| reg.get(vox_core::MaterialId(i as u16)).is_some_and(|d| d.solid))
            .collect();

        let mut world = World::new(WorldConfig {
            voxel_size_m: 0.1,
            extent_m: [32.0, 24.0, 32.0],
            ..WorldConfig::default()
        });
        world.set_solid_table(solid_table);
        // Stone floor, top at 4.0 m (voxel y=40).
        let (_, max) = world.bounds_voxels();
        world.fill_box(IVec3::ZERO, IVec3::new(max.x, 40, max.z), STONE);
        // Muddy_water pool on top of the floor, 4.0 m → 7.0 m (voxel y 40→70).
        world.fill_box(
            IVec3::new(140, 40, 140),
            IVec3::new(180, 70, 180),
            MUDDY_WATER,
        );

        let mut phys = PhysicsWorld::new();
        // Register both water and muddy_water as buoyancy fluids. The key
        // assertion: muddy_water (the *second* fluid) triggers buoyancy.
        phys.set_fluid_voxels(vec![WATER, MUDDY_WATER]);

        // Foam (density 300) is much lighter than water (1000), so
        // buoy_accel = (1000/300 - 1)*GRAVITY ≈ 2.33*GRAVITY, far exceeding
        // gravity — the body rises and bobs near the muddy_water surface.
        let dims = IVec3::splat(4);
        let grid = VoxelGrid::new(dims, vec![FOAM; (dims.x * dims.y * dims.z) as usize]);
        let body = Body::from_grid(grid, &reg, 0.1, Vec3::new(16.0, 5.0, 16.0)).expect("massive body");
        let id = phys.spawn(body);

        for _ in 0..600 {
            phys.step(&world, PHYSICS_DT);
        }
        let b = phys.get(id).expect("alive");
        // The body must float inside the pool — well above the stone floor
        // (4.0 m) and near the muddy_water surface (7.0 m). If buoyancy
        // never fired (the pre-fix behavior), the body would sink through
        // the non-solid muddy_water and rest on the stone floor at ~4.0 m.
        assert!(
            b.aabb_min.y > 5.0,
            "foam must float in muddy_water via buoyancy, not sink to the floor: aabb_min.y = {}",
            b.aabb_min.y
        );
    }
    #[test]
    fn joint_holds_two_bodies_at_rest_length() {
        let world = floored_world();
        let reg = registry();
        let mut phys = PhysicsWorld::new();

        // Two 2x2x2 stone bodies, 1m apart, joined rigidly (compliance=0).
        let grid = VoxelGrid::new(IVec3::new(2, 2, 2), vec![Voxel(1); 8]);
        let body_a =
            Body::from_grid(grid.clone(), &reg, 0.5, Vec3::new(10.0, 20.0, 10.0)).unwrap();
        let body_b = Body::from_grid(grid, &reg, 0.5, Vec3::new(11.0, 20.0, 10.0)).unwrap();
        let id_a = phys.spawn(body_a);
        let id_b = phys.spawn(body_b);

        phys.add_joint(id_a, id_b, Vec3::ZERO, Vec3::ZERO, 1.0, 0.0);

        // Run 100 steps (~1.7s free fall from 20m — floor at 4m, still airborne).
        for step in 0..100 {
            phys.step(&world, PHYSICS_DT);
            let a = phys.get(id_a).expect("alive");
            let b = phys.get(id_b).expect("alive");
            assert!(a.pos.is_finite() && b.pos.is_finite(), "NaN at step {step}");
        }

        let a = phys.get(id_a).unwrap();
        let b = phys.get(id_b).unwrap();
        let dist = (a.pos - b.pos).length();
        assert!(
            (dist - 1.0).abs() < 0.15,
            "joint should maintain rest length ~1.0m, got {dist}"
        );
    }
}
