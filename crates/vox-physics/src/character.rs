//! Kinematic character controller: an AABB swept against the voxel grid.
//!
//! The player is deliberately not a rigidbody (industry standard): axis-
//! separated sweeps with a skin margin give precise, tunable movement.
//! All dimensions come from `vox_core::consts` in meters, so behavior is
//! identical at any voxel scale (a 0.4 m ledge steps up whether it is four
//! 0.1 m voxels or part of one 1.0 m voxel).

use glam::Vec3;
use vox_core::consts::{GRAVITY, JUMP_HEIGHT, PLAYER_SIZE, STEP_HEIGHT};
use vox_world::World;

/// Collision skin in meters: sweeps stop this far short of surfaces so
/// resting contact never re-collides.
const SKIN: f32 = 1e-3;
/// A sweep is "clamped" when it lost at least this much motion.
const CLAMP_EPS: f32 = 1e-6;

/// Axis-aligned box in meters.
#[derive(Copy, Clone, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// The voxel index range `[lo, hi]` covered by this box on `axis`,
    /// inset by the skin so boundary-flush boxes don't touch the next row.
    fn cross_range(&self, axis: usize, s: f32) -> (i32, i32) {
        let lo = ((self.min[axis] + SKIN) / s).floor() as i32;
        let hi = (((self.max[axis] - SKIN) / s).ceil() as i32) - 1;
        (lo, hi.max(lo))
    }
}

/// Move `aabb` along `axis` by `delta` meters, clamped by solid voxels.
/// Returns the achievable delta (same sign as `delta`, magnitude ≤ |delta|).
pub fn sweep_axis(world: &World, aabb: Aabb, axis: usize, delta: f32) -> f32 {
    if delta == 0.0 {
        return 0.0;
    }
    let s = world.cfg.voxel_size_m;
    let sign = delta.signum();
    let face = if sign > 0.0 {
        aabb.max[axis]
    } else {
        aabb.min[axis]
    };
    let target = face + delta;
    let (lo, hi) = if sign > 0.0 {
        (face, target)
    } else {
        (target, face)
    };
    let v_lo = (lo / s).floor() as i32;
    let v_hi = ((hi / s).ceil() as i32) - 1;

    let (u, w) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let (u_lo, u_hi) = aabb.cross_range(u, s);
    let (w_lo, w_hi) = aabb.cross_range(w, s);

    let slices: Box<dyn Iterator<Item = i32>> = if sign > 0.0 {
        Box::new(v_lo..=v_hi)
    } else {
        Box::new((v_lo..=v_hi).rev())
    };
    for slice in slices {
        let mut blocked = false;
        'scan: for uu in u_lo..=u_hi {
            for ww in w_lo..=w_hi {
                let mut v = glam::IVec3::ZERO;
                v[axis] = slice;
                v[u] = uu;
                v[w] = ww;
                if world.solid(v) {
                    blocked = true;
                    break 'scan;
                }
            }
        }
        if blocked {
            let plane = if sign > 0.0 {
                slice as f32 * s
            } else {
                (slice + 1) as f32 * s
            };
            let allowed = plane - face - sign * SKIN;
            return if sign > 0.0 {
                allowed.clamp(0.0, delta)
            } else {
                allowed.clamp(delta, 0.0)
            };
        }
    }
    delta
}

/// First-person kinematic character.
#[derive(Clone, Debug)]
pub struct CharacterController {
    /// Feet-center position in meters.
    pub pos: Vec3,
    pub vel: Vec3,
    pub grounded: bool,
    /// Fly/noclip mode: no gravity, no collision.
    pub noclip: bool,
}

impl CharacterController {
    pub fn new(pos: Vec3) -> Self {
        Self {
            pos,
            vel: Vec3::ZERO,
            grounded: false,
            noclip: false,
        }
    }

    /// The character's collision box at its current position.
    pub fn aabb(&self) -> Aabb {
        let (w, h, d) = PLAYER_SIZE;
        Aabb {
            min: self.pos - Vec3::new(w * 0.5, 0.0, d * 0.5),
            max: self.pos + Vec3::new(w * 0.5, h, d * 0.5),
        }
    }

    /// Advance one physics tick.
    ///
    /// `wish_vel`: desired horizontal velocity in m/s (y ignored unless
    /// noclip, where it is full 3-D flight).
    pub fn step(&mut self, world: &World, wish_vel: Vec3, jump: bool, dt: f32) {
        if self.noclip {
            self.pos += wish_vel * dt;
            self.vel = wish_vel;
            self.grounded = false;
            return;
        }

        self.vel.x = wish_vel.x;
        self.vel.z = wish_vel.z;
        if jump && self.grounded {
            self.vel.y = (2.0 * GRAVITY * JUMP_HEIGHT).sqrt();
        }
        self.vel.y -= GRAVITY * dt;

        // Vertical first: establishes grounding for the horizontal passes.
        let dy = self.vel.y * dt;
        let moved_y = sweep_axis(world, self.aabb(), 1, dy);
        self.pos.y += moved_y;
        let clamped_y = (moved_y - dy).abs() > CLAMP_EPS;
        self.grounded = clamped_y && dy < 0.0;
        if clamped_y {
            self.vel.y = 0.0;
        }

        // Horizontal, axis-separated (produces wall sliding for free).
        for axis in [0usize, 2] {
            let wanted = self.vel[axis] * dt;
            if wanted == 0.0 {
                continue;
            }
            let moved = sweep_axis(world, self.aabb(), axis, wanted);
            self.pos[axis] += moved;
            let clamped = (moved - wanted).abs() > CLAMP_EPS;
            if clamped && self.grounded {
                self.try_step_up(world, axis, wanted - moved);
            }
        }
    }

    /// Attempt to climb a ledge: lift by up to `STEP_HEIGHT`, retry the
    /// remaining horizontal motion, then settle back down. Applied only when
    /// it gains ground and lands on a surface.
    fn try_step_up(&mut self, world: &World, axis: usize, remaining: f32) {
        let start = self.pos;

        let up = sweep_axis(world, self.aabb(), 1, STEP_HEIGHT);
        if up <= CLAMP_EPS {
            return;
        }
        self.pos.y += up;

        let gained = sweep_axis(world, self.aabb(), axis, remaining);
        self.pos[axis] += gained;

        let down = sweep_axis(world, self.aabb(), 1, -up);
        self.pos.y += down;
        let landed = (-down - up).abs() > CLAMP_EPS; // clamped before full descent

        if gained.abs() > 1e-4 && landed {
            self.grounded = true;
        } else {
            self.pos = start;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;
    use vox_core::WorldConfig;
    use vox_core::consts::PHYSICS_DT;
    use vox_world::Voxel;

    const STONE: Voxel = Voxel(1);

    /// A world with a solid floor up to `floor_m`.
    fn floored_world(voxel_size_m: f32, floor_m: f32) -> World {
        let mut world = World::new(WorldConfig {
            voxel_size_m,
            extent_m: [32.0, 16.0, 32.0],
            ..WorldConfig::default()
        });
        let (_, max) = world.bounds_voxels();
        let floor_vox = (floor_m / voxel_size_m).round() as i32;
        world.fill_box(IVec3::ZERO, IVec3::new(max.x, floor_vox, max.z), STONE);
        world
    }

    fn settle(ctrl: &mut CharacterController, world: &World, ticks: u32) {
        for _ in 0..ticks {
            ctrl.step(world, Vec3::ZERO, false, PHYSICS_DT);
        }
    }

    #[test]
    fn rests_exactly_on_floor_at_both_scales() {
        for s in [0.1_f32, 1.0] {
            let world = floored_world(s, 4.0);
            let mut ctrl = CharacterController::new(Vec3::new(16.0, 7.0, 16.0));
            settle(&mut ctrl, &world, 240);
            assert!(ctrl.grounded, "scale {s}: must be grounded");
            assert_eq!(ctrl.vel.y, 0.0, "scale {s}: vertical velocity zeroed");
            assert!(
                (ctrl.pos.y - 4.0).abs() < 2e-3,
                "scale {s}: feet at {} m, expected ~4.0",
                ctrl.pos.y
            );
        }
    }

    #[test]
    fn wall_blocks_and_slides() {
        let s = 0.1;
        let mut world = floored_world(s, 4.0);
        // A 2 m wall across x = 18 m.
        world.fill_box(
            IVec3::new((18.0 / s) as i32, (4.0 / s) as i32, 0),
            IVec3::new((18.0 / s) as i32 + 1, (6.0 / s) as i32, (32.0 / s) as i32),
            STONE,
        );
        let mut ctrl = CharacterController::new(Vec3::new(16.0, 4.5, 16.0));
        settle(&mut ctrl, &world, 60);

        // Run at the wall diagonally for 2 seconds.
        for _ in 0..120 {
            ctrl.step(&world, Vec3::new(4.0, 0.0, 1.5), false, PHYSICS_DT);
        }
        let half_w = PLAYER_SIZE.0 * 0.5;
        assert!(
            ctrl.pos.x + half_w <= 18.0 + 1e-3,
            "wall must block: player face at {}",
            ctrl.pos.x + half_w
        );
        assert!(
            ctrl.pos.x + half_w > 18.0 - 5.0 * s,
            "player should reach the wall, at {}",
            ctrl.pos.x
        );
        assert!(
            ctrl.pos.z > 17.0,
            "tangent motion must continue (sliding), z = {}",
            ctrl.pos.z
        );
    }

    #[test]
    fn steps_up_small_ledges_only() {
        // 0.4 m ledge at fine scale: must climb.
        let s = 0.1;
        let mut world = floored_world(s, 4.0);
        world.fill_box(
            IVec3::new((18.0 / s) as i32, (4.0 / s) as i32, 0),
            IVec3::new((32.0 / s) as i32, (4.4 / s) as i32, (32.0 / s) as i32),
            STONE,
        );
        let mut ctrl = CharacterController::new(Vec3::new(16.0, 4.2, 16.0));
        settle(&mut ctrl, &world, 60);
        for _ in 0..180 {
            ctrl.step(&world, Vec3::new(3.0, 0.0, 0.0), false, PHYSICS_DT);
        }
        assert!(
            ctrl.pos.x > 19.0 && (ctrl.pos.y - 4.4).abs() < 2e-2,
            "0.4 m ledge must be climbed: pos {:?}",
            ctrl.pos
        );

        // 1.0 m ledge (one voxel at coarse scale): must NOT climb.
        let s = 1.0;
        let mut world = floored_world(s, 4.0);
        world.fill_box(IVec3::new(18, 4, 0), IVec3::new(32, 5, 32), STONE);
        let mut ctrl = CharacterController::new(Vec3::new(16.0, 4.5, 16.0));
        settle(&mut ctrl, &world, 60);
        for _ in 0..180 {
            ctrl.step(&world, Vec3::new(3.0, 0.0, 0.0), false, PHYSICS_DT);
        }
        assert!(
            ctrl.pos.y < 4.2,
            "1.0 m ledge must not be climbed: pos {:?}",
            ctrl.pos
        );
        let half_w = PLAYER_SIZE.0 * 0.5;
        assert!(ctrl.pos.x + half_w <= 18.0 + 1e-3, "blocked at the ledge");
    }

    #[test]
    fn ceiling_zeroes_upward_velocity_without_sticking() {
        let s = 0.1;
        let mut world = floored_world(s, 4.0);
        // Ceiling 2.2 m above the floor.
        world.fill_box(
            IVec3::new(0, (6.2 / s) as i32, 0),
            IVec3::new((32.0 / s) as i32, (6.4 / s) as i32, (32.0 / s) as i32),
            STONE,
        );
        let mut ctrl = CharacterController::new(Vec3::new(16.0, 4.5, 16.0));
        settle(&mut ctrl, &world, 60);

        // Jump into the ceiling.
        ctrl.step(&world, Vec3::ZERO, true, PHYSICS_DT);
        let mut peak = ctrl.pos.y;
        let mut bumped = false;
        for _ in 0..120 {
            ctrl.step(&world, Vec3::ZERO, false, PHYSICS_DT);
            peak = peak.max(ctrl.pos.y);
            if ctrl.vel.y == 0.0 && !ctrl.grounded {
                bumped = true;
            }
        }
        let head_limit = 6.2 - PLAYER_SIZE.1;
        assert!(
            peak <= head_limit + 1e-3,
            "head must stop at the ceiling: peak {peak}, limit {head_limit}"
        );
        assert!(bumped, "upward velocity must be zeroed mid-air");
        assert!(ctrl.grounded, "must fall back to the floor");
        assert!((ctrl.pos.y - 4.0).abs() < 2e-3);
    }

    #[test]
    fn jump_clears_one_meter() {
        let s = 0.1;
        let world = floored_world(s, 4.0);
        let mut ctrl = CharacterController::new(Vec3::new(16.0, 4.5, 16.0));
        settle(&mut ctrl, &world, 60);

        ctrl.step(&world, Vec3::ZERO, true, PHYSICS_DT);
        let mut peak = ctrl.pos.y;
        for _ in 0..120 {
            ctrl.step(&world, Vec3::ZERO, false, PHYSICS_DT);
            peak = peak.max(ctrl.pos.y);
        }
        let rise = peak - 4.0;
        assert!(
            rise > 1.0 && rise <= 1.35,
            "jump apex {rise} m should approach 1.25 m"
        );
        assert!(ctrl.grounded, "must land");
    }

    #[test]
    fn noclip_ignores_world() {
        let s = 0.1;
        let world = floored_world(s, 4.0);
        let mut ctrl = CharacterController::new(Vec3::new(16.0, 10.0, 16.0));
        ctrl.noclip = true;
        for _ in 0..60 {
            ctrl.step(&world, Vec3::new(0.0, -8.0, 0.0), false, PHYSICS_DT);
        }
        assert!(
            ctrl.pos.y < 3.5,
            "noclip must pass through the floor (top at 4.0 m), got {}",
            ctrl.pos.y
        );
    }
}
