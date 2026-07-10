//! CPU particle simulation: dust, sparks, and smoke for destruction
//! feedback. Particles collide with the voxel world (wall deflection,
//! ceiling stop, floor settle, enclosure-aware drag) and repel each other
//! via a per-frame spatial hash -- the Teardown approach to smoke that
//! fills rooms instead of drifting through walls. Still visual-only: no
//! gameplay state reads them and nothing reads them back.
//!
//! The GPU side (`vox_render::ParticlePipeline`) only ever sees the flat
//! [`ParticleInstance`] list produced by [`ParticleSystem::instances`].

use glam::{IVec3, Vec3};
use vox_core::voxel_at;
use vox_render::{MAX_PARTICLES, ParticleInstance};
use vox_world::World;

/// Fraction of world gravity particles feel -- dust and smoke are light and
/// drag-dominated, so full gravity reads as "gravel", not "dust".
const GRAVITY_FACTOR: f32 = 0.35;
/// Per-second velocity damping (air drag) in open air.
const DRAG: f32 = 1.6;
/// Gentle upward acceleration for buoyant particles. Smoke should billow,
/// not launch like debris.
const BUOYANCY_ACCEL: f32 = 0.3;
/// Drag multiplier when a particle is enclosed (4+ solid neighbors).
const ENCLOSURE_DRAG_MULT: f32 = 2.5;
/// Peak acceleration contributed by one overlapping particle (m/s²).
const REPEL_STRENGTH: f32 = 3.0;
/// Aggregate repulsion acceleration cap. Dense smoke may have many nearby
/// particles; bounding their sum prevents a crowded plume from exploding.
const MAX_REPEL_ACCEL: f32 = 2.0;
/// Buoyant particles have a low terminal speed. This is deliberately above
/// their normal drag-limited rise speed, but catches collision/repulsion
/// spikes and large-frame integration hitches.
const MAX_BUOYANT_SPEED: f32 = 0.9;
/// Sideways acceleration while smoke is pressed against a ceiling (m/s²).
/// This is acceleration, not a per-frame velocity kick.
const CEILING_SPREAD_ACCEL: f32 = 0.25;
/// Spatial hash cell size in meters (~max smoke particle diameter).
const HASH_CELL_M: f32 = 0.3;
/// Velocity retention after hitting a floor (friction).
const FLOOR_FRICTION: f32 = 0.5;

/// One simulated particle.
#[derive(Copy, Clone, Debug)]
struct Particle {
    pos: Vec3,
    vel: Vec3,
    /// Half-size in meters (billboard extent).
    size: f32,
    /// Base color; alpha is faded by age on top of this.
    color: [f32; 4],
    age: f32,
    life: f32,
    /// Smoke rises and grows instead of falling and staying fixed-size.
    buoyant: bool,
}

/// Parameters for one burst of particles -- see the emit helpers on
/// [`ParticleSystem`] for the tuned presets destruction actually uses.
pub struct Burst {
    pub center: Vec3,
    pub count: usize,
    pub color: [f32; 3],
    /// Base outward speed, m/s; per-particle speed varies around it.
    pub speed: f32,
    /// Extra upward velocity bias, m/s -- rubble kicks up, not sideways.
    pub upward: f32,
    /// Mean lifetime, seconds; per-particle life varies around it.
    pub life: f32,
    /// Mean half-size, meters.
    pub size: f32,
    pub buoyant: bool,
}

/// All live particles plus a tiny deterministic RNG for spawn variation.
pub struct ParticleSystem {
    particles: Vec<Particle>,
    rng: u64,
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ParticleSystem {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// xorshift64* -- deterministic, dependency-free spawn jitter.
    fn next_f32(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        ((self.rng >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Uniform in [-1, 1].
    fn signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    /// Spawn one burst. If the cap would be exceeded, the *oldest* live
    /// particles are dropped first -- a fresh explosion always gets its
    /// full visual, at the cost of some old lingering smoke.
    pub fn burst(&mut self, b: Burst) {
        let count = b.count.min(MAX_PARTICLES);
        let overflow = (self.particles.len() + count).saturating_sub(MAX_PARTICLES);
        if overflow > 0 {
            self.particles.drain(..overflow);
        }
        for _ in 0..count {
            let dir = Vec3::new(self.signed(), self.signed(), self.signed());
            let dir = if dir.length_squared() > 1e-6 {
                dir.normalize()
            } else {
                Vec3::Y
            };
            let speed = b.speed * (0.4 + 0.9 * self.next_f32());
            let life = b.life * (0.6 + 0.8 * self.next_f32());
            let size = b.size * (0.6 + 0.8 * self.next_f32());
            // Slight per-particle tint variation keeps a burst from reading
            // as a single flat-colored blob.
            let tint = 0.85 + 0.3 * self.next_f32();
            self.particles.push(Particle {
                pos: b.center,
                vel: dir * speed + Vec3::Y * b.upward,
                size,
                color: [
                    (b.color[0] * tint).min(1.0),
                    (b.color[1] * tint).min(1.0),
                    (b.color[2] * tint).min(1.0),
                    1.0,
                ],
                age: 0.0,
                life: life.max(0.05),
                buoyant: b.buoyant,
            });
        }
    }

    /// Advance every particle by `dt` seconds, colliding with the voxel
    /// world and repelling nearby particles, then drop the expired.
    /// `voxel_size_m` converts world-space positions to voxel coordinates
    /// for `world.solid()` lookups.
    pub fn update(&mut self, dt: f32, world: &World, voxel_size_m: f32) {
        // --- Spatial hash for inter-particle repulsion ---
        let hash = SpatialHash::build(&self.particles);

        for i in 0..self.particles.len() {
            // Compute repulsion BEFORE the mutable borrow (it needs &self.particles).
            let repel = hash
                .repulsion(i, &self.particles)
                .clamp_length_max(MAX_REPEL_ACCEL);
            let p = &mut self.particles[i];
            p.age += dt;

            // Buoyancy / gravity.
            if p.buoyant {
                p.vel.y += BUOYANCY_ACCEL * dt;
                p.size += 0.15 * p.size * dt; // slower swell
            } else {
                p.vel.y -= vox_core::consts::GRAVITY * GRAVITY_FACTOR * dt;
            }

            // Inter-particle repulsion: push away from nearby particles.
            p.vel += repel * dt;

            // Enclosure-aware drag: sample 6 neighbors, more solids = more drag.
            let solid_count = count_solid_neighbors(world, p.pos, voxel_size_m);
            let drag = if solid_count >= 4 { DRAG * ENCLOSURE_DRAG_MULT } else { DRAG };
            p.vel /= 1.0 + drag * dt;

            // World collision: check the proposed next position component-wise
            // and deflect along walls instead of passing through.
            let next = p.pos + p.vel * dt;

            // X axis: if blocked, zero X velocity (slide along wall).
            let x_probe = voxel_at(Vec3::new(next.x, p.pos.y, p.pos.z), voxel_size_m);
            if world.in_bounds(x_probe) && world.solid(x_probe) {
                p.vel.x = 0.0;
            }
            // Z axis: same.
            let z_probe = voxel_at(Vec3::new(p.pos.x, p.pos.y, next.z), voxel_size_m);
            if world.in_bounds(z_probe) && world.solid(z_probe) {
                p.vel.z = 0.0;
            }
            // Y axis: ceiling stop (buoyant) or floor settle (non-buoyant).
            let y_probe = voxel_at(Vec3::new(p.pos.x, next.y, p.pos.z), voxel_size_m);
            if world.in_bounds(y_probe) && world.solid(y_probe) {
                if p.buoyant && p.vel.y > 0.0 {
                    // Hit a ceiling: stop rising, spread sideways.
                    p.vel.y = 0.0;
                    let seed = hash.repulsion_seed(i);
                    let sx = if seed & 1 == 0 { -1.0 } else { 1.0 };
                    let sz = if seed & 2 == 0 { -1.0 } else { 1.0 };
                    p.vel.x += sx * CEILING_SPREAD_ACCEL * dt;
                    p.vel.z += sz * CEILING_SPREAD_ACCEL * dt;
                } else if !p.buoyant && p.vel.y < 0.0 {
                    // Hit the floor: lose vertical velocity, friction on horizontal.
                    p.vel.y = 0.0;
                    p.vel.x *= FLOOR_FRICTION;
                    p.vel.z *= FLOOR_FRICTION;
                } else {
                    p.vel.y = 0.0;
                }
            }

            if p.buoyant {
                p.vel = p.vel.clamp_length_max(MAX_BUOYANT_SPEED);
            }

            // Integrate position with the corrected velocity.
            p.pos += p.vel * dt;

            // Keep the particle in bounds (don't let it escape the world).
            let (bmin, bmax) = world.bounds_voxels();
            let min_m = bmin.as_vec3() * voxel_size_m;
            let max_m = bmax.as_vec3() * voxel_size_m;
            p.pos = p.pos.clamp(min_m + voxel_size_m * 0.5, max_m - voxel_size_m * 0.5);
        }
        self.particles.retain(|p| p.age < p.life);
    }

    /// Number of live particles (debug-overlay stat).
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// Flatten to GPU instances: alpha fades quadratically with age (a slow
    /// start then a quick vanish reads better than a linear dimming).
    pub fn instances(&self) -> Vec<ParticleInstance> {
        self.particles
            .iter()
            .map(|p| {
                let t = (p.age / p.life).clamp(0.0, 1.0);
                let alpha = p.color[3] * (1.0 - t * t);
                ParticleInstance {
                    center_size: [p.pos.x, p.pos.y, p.pos.z, p.size],
                    color: [p.color[0], p.color[1], p.color[2], alpha],
                }
            })
            .collect()
    }
}

// --- Spatial hash for inter-particle repulsion ---

/// Uniform-grid spatial hash for O(1) neighbor queries. Cell size is
/// `HASH_CELL_M` in world meters. Rebuilt per frame from the live particle
/// list.
struct SpatialHash {
    /// Grid cell → particle indices in that cell.
    cells: vox_core::FxHashMap<IVec3, Vec<usize>>,
}

impl SpatialHash {
    fn build(particles: &[Particle]) -> Self {
        let mut cells: vox_core::FxHashMap<IVec3, Vec<usize>> = vox_core::FxHashMap::default();
        for (i, p) in particles.iter().enumerate() {
            let cell = (p.pos / HASH_CELL_M).floor().as_ivec3();
            cells.entry(cell).or_default().push(i);
        }
        Self { cells }
    }

    /// Repulsion force on particle `i` from its neighbors.
    fn repulsion(&self, i: usize, particles: &[Particle]) -> Vec3 {
        let p = &particles[i];
        let cell = (p.pos / HASH_CELL_M).floor().as_ivec3();
        let mut force = Vec3::ZERO;
        // Check the 3x3x3 block of cells around the particle's cell.
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = cell + IVec3::new(dx, dy, dz);
                    if let Some(indices) = self.cells.get(&key) {
                        for &j in indices {
                            if j == i {
                                continue;
                            }
                            let other = &particles[j];
                            let diff = p.pos - other.pos;
                            let dist2 = diff.length_squared();
                            // A soft overlap kernel avoids the inverse-square
                            // singularity that used to eject dense smoke at
                            // hundreds of m/s². Particle `size` is half-size,
                            // so the sum is their natural separation radius.
                            let radius = (p.size + other.size).clamp(0.02, HASH_CELL_M * 2.0);
                            if dist2 >= radius * radius {
                                continue;
                            }
                            if dist2 < 1e-8 {
                                // Co-located: deterministic push so they separate.
                                force += Vec3::new(
                                    (j as f32 * 0.137).fract() - 0.5,
                                    (i as f32 * 0.379).fract() - 0.5,
                                    ((j ^ i) as f32 * 0.617).fract() - 0.5,
                                )
                                .normalize_or_zero()
                                    * REPEL_STRENGTH;
                            } else {
                                let dist = dist2.sqrt();
                                let overlap = 1.0 - dist / radius;
                                force += diff / dist * (overlap * REPEL_STRENGTH);
                            }
                        }
                    }
                }
            }
        }
        force
    }

    /// A per-particle seed for randomized ceiling-spread direction (uses
    /// the particle index as a deterministic but varied source).
    fn repulsion_seed(&self, i: usize) -> u64 {
        (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }
}

/// Count how many of the 6 face-neighbors of the voxel at `pos_m` are solid.
/// Used for enclosure-aware drag: more solid neighbors = more drag = smoke
/// lingers in rooms instead of streaming through.
fn count_solid_neighbors(world: &World, pos_m: Vec3, voxel_size_m: f32) -> u32 {
    let v = voxel_at(pos_m, voxel_size_m);
    let mut count = 0;
    for d in [
        IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z,
    ] {
        let n = v + d;
        if world.in_bounds(n) && world.solid(n) {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::WorldConfig;
    use vox_world::{Voxel, World};

    /// An empty world with air everywhere (no solids except walls for
    /// collision tests). Large enough that particles spawned near the
    /// center never hit the world bounds during a test.
    fn empty_world() -> World {
        let mut w = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [64.0, 64.0, 64.0],
            ..WorldConfig::default()
        });
        w.set_solid_table(vec![false, true]); // [air, solid]
        w
    }

    /// A world with a solid floor at y=0 and solid ceiling at y=10.
    fn room_world() -> World {
        let mut w = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [64.0, 64.0, 64.0],
            ..WorldConfig::default()
        });
        w.set_solid_table(vec![false, true]);
        let (_, max) = w.bounds_voxels();
        // Floor: y=0, ceiling: y=10 (voxels), walls around a 20x20 room.
        w.fill_box(IVec3::new(0, 0, 0), IVec3::new(max.x, 1, max.z), Voxel(1));
        w.fill_box(IVec3::new(0, 10, 0), IVec3::new(max.x, 11, max.z), Voxel(1));
        // Walls.
        w.fill_box(IVec3::new(20, 0, 20), IVec3::new(44, 11, 21), Voxel(1));
        w.fill_box(IVec3::new(20, 0, 43), IVec3::new(44, 11, 44), Voxel(1));
        w.fill_box(IVec3::new(20, 0, 20), IVec3::new(21, 11, 44), Voxel(1));
        w.fill_box(IVec3::new(43, 0, 20), IVec3::new(44, 11, 44), Voxel(1));
        w
    }

    fn dt() -> f32 {
        1.0 / 60.0
    }
    fn dust(center: Vec3, count: usize) -> Burst {
        Burst {
            center,
            count,
            color: [0.5, 0.4, 0.3],
            speed: 2.0,
            upward: 1.0,
            life: 1.0,
            size: 0.05,
            buoyant: false,
        }
    }

    #[test]
    fn particles_age_out_and_are_removed() {
        let world = empty_world();
        let mut sys = ParticleSystem::new();
        sys.burst(dust(Vec3::new(32.0, 32.0, 32.0), 50));
        assert_eq!(sys.len(), 50);
        // Max per-particle life is 1.0 * (0.6 + 0.8) = 1.4 s.
        for _ in 0..100 {
            sys.update(dt(), &world, 1.0);
        }
        assert_eq!(sys.len(), 0, "all particles must expire: {} left", sys.len());
    }

    #[test]
    fn the_cap_drops_oldest_first_and_never_exceeds_max() {
        let mut sys = ParticleSystem::new();
        sys.burst(dust(Vec3::new(32.0, 32.0, 32.0), MAX_PARTICLES));
        assert_eq!(sys.len(), MAX_PARTICLES);
        sys.burst(dust(Vec3::new(42.0, 32.0, 32.0), 100));
        assert_eq!(sys.len(), MAX_PARTICLES, "cap must hold");
        // The newest burst must have survived (spawned at x=42).
        let inst = sys.instances();
        assert!(
            inst.iter().rev().take(100).all(|i| i.center_size[0] > 37.0),
            "the fresh burst must not be the part that was dropped"
        );
    }

    #[test]
    fn gravity_pulls_dust_down_and_fade_reaches_zero_at_end_of_life() {
        let world = empty_world();
        let mut sys = ParticleSystem::new();
        sys.burst(dust(Vec3::new(32.0, 40.0, 32.0), 1));
        for _ in 0..30 {
            sys.update(dt(), &world, 1.0);
        }
        let inst = sys.instances();
        assert_eq!(inst.len(), 1);
        let alpha = inst[0].color[3];
        assert!(alpha > 0.0 && alpha < 1.0, "mid-life alpha must be fading: {alpha}");
    }

    #[test]
    fn smoke_rises_and_grows_instead_of_falling() {
        let world = empty_world();
        let mut sys = ParticleSystem::new();
        sys.burst(Burst {
            center: Vec3::new(32.0, 32.0, 32.0),
            count: 1,
            color: [0.4, 0.4, 0.4],
            speed: 0.0,
            upward: 0.0,
            life: 5.0,
            size: 0.2,
            buoyant: true,
        });
        let size0 = sys.instances()[0].center_size[3];
        for _ in 0..60 {
            sys.update(dt(), &world, 1.0);
        }
        let i = &sys.instances()[0];
        assert!(i.center_size[1] > 32.0, "smoke must rise: y = {}", i.center_size[1]);
        assert!(i.center_size[3] > size0, "smoke must swell as it ages");
    }

    // --- World collision tests ---

    #[test]
    fn smoke_stops_at_a_ceiling() {
        // Buoyant smoke under a solid ceiling (y=10) must stop rising
        // and not pass through.
        let world = room_world();
        let mut sys = ParticleSystem::new();
        sys.burst(Burst {
            center: Vec3::new(32.0, 9.0, 32.0),
            count: 1,
            color: [0.4, 0.4, 0.4],
            speed: 0.0,
            upward: 0.0,
            life: 15.0,
            size: 0.2,
            buoyant: true,
        });
        // Run enough ticks for the smoke to reach the ceiling at y=10.
        // At 0.6 m/s² from y=9, it takes ~2-3s (~180 ticks) to reach y=10.
        for _ in 0..300 {
            sys.update(dt(), &world, 1.0);
        }
        let y = sys.instances()[0].center_size[1];
        assert!(y < 10.0, "smoke must not pass through the ceiling: y = {y}");
        // And it must actually have reached near the ceiling (not just
        // drifted harmlessly at y=9).
        assert!(y > 9.0, "smoke must have risen toward the ceiling: y = {y}");
    }

    #[test]
    fn dust_settles_on_a_floor() {
        // Non-buoyant dust falling onto a solid floor (y=0) must stop
        // falling and not pass through.
        let world = room_world();
        let mut sys = ParticleSystem::new();
        sys.burst(Burst {
            center: Vec3::new(32.0, 8.0, 32.0),
            count: 1,
            color: [0.5, 0.4, 0.3],
            speed: 0.0,
            upward: 0.0,
            life: 15.0,
            size: 0.05,
            buoyant: false,
        });
        for _ in 0..600 {
            sys.update(dt(), &world, 1.0);
        }
        let y = sys.instances()[0].center_size[1];
        assert!(y < 2.5, "dust must settle near the floor: y = {y}");
    }

    #[test]
    fn particles_repel_each_other() {
        // Two particles at the same position must push apart over time.
        let world = empty_world();
        let mut sys = ParticleSystem::new();
        let center = Vec3::new(32.0, 32.0, 32.0);
        // Spawn two particles at the same spot with zero velocity.
        sys.burst(Burst {
            center,
            count: 2,
            color: [0.4, 0.4, 0.4],
            speed: 0.0,
            upward: 0.0,
            life: 5.0,
            size: 0.2,
            buoyant: false,
        });
        // Both start at the same position.
        let inst0 = sys.instances();
        let p0 = Vec3::from_array([inst0[0].center_size[0], inst0[0].center_size[1], inst0[0].center_size[2]]);
        let p1 = Vec3::from_array([inst0[1].center_size[0], inst0[1].center_size[1], inst0[1].center_size[2]]);
        assert!((p0 - p1).length() < 0.01, "particles must start at the same position");
        // After several steps, repulsion pushes them apart.
        for _ in 0..30 {
            sys.update(dt(), &world, 1.0);
        }
        let inst1 = sys.instances();
        let q0 = Vec3::from_array([inst1[0].center_size[0], inst1[0].center_size[1], inst1[0].center_size[2]]);
        let q1 = Vec3::from_array([inst1[1].center_size[0], inst1[1].center_size[1], inst1[1].center_size[2]]);
        assert!((q0 - q1).length() > 0.05, "repulsion must push particles apart: dist = {}", (q0 - q1).length());
    }

    #[test]
    fn dense_smoke_repulsion_has_a_bounded_velocity() {
        let world = empty_world();
        let mut sys = ParticleSystem::new();
        sys.burst(Burst {
            center: Vec3::new(32.0, 32.0, 32.0),
            count: 128,
            color: [0.4, 0.4, 0.4],
            speed: 0.0,
            upward: 0.0,
            life: 5.0,
            size: 0.2,
            buoyant: true,
        });

        // A deliberately large frame exercises the aggregate repulsion
        // clamp as well as the buoyant terminal-speed guard.
        sys.update(0.25, &world, 1.0);

        assert!(
            sys.particles
                .iter()
                .all(|p| p.vel.length() <= MAX_BUOYANT_SPEED + 1e-5),
            "dense smoke must never be violently ejected"
        );
    }
}
