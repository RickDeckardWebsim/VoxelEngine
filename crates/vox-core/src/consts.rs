//! Engine-wide tuning constants: the single source of truth for chunk size,
//! physics solver parameters, and player/tool dimensions.

/// Voxels per chunk axis.
pub const CHUNK_SIZE: usize = 32;
/// Gravitational acceleration in m/s².
pub const GRAVITY: f32 = 9.81;
/// Fixed physics timestep in seconds (60 Hz).
pub const PHYSICS_DT: f32 = 1.0 / 60.0;
/// Physics substeps per fixed step.
pub const SUBSTEPS: u32 = 2;
/// Velocity solver iterations per substep.
pub const SOLVER_ITERS: u32 = 8;
/// Baumgarte positional-correction factor for contacts.
pub const CONTACT_BETA: f32 = 0.2;
/// Allowed contact penetration in meters.
pub const CONTACT_SLOP: f32 = 0.005;
/// Coulomb friction coefficient (μ).
pub const FRICTION: f32 = 0.6;
/// Linear speed below which a body may sleep, in m/s.
pub const SLEEP_LIN: f32 = 0.03;
/// Angular speed below which a body may sleep, in rad/s.
pub const SLEEP_ANG: f32 = 0.20;
/// Consecutive quiet steps before a body is put to sleep.
pub const SLEEP_FRAMES: u32 = 45;
/// Player collision AABB (width, height, depth) in meters.
pub const PLAYER_SIZE: (f32, f32, f32) = (0.6, 1.8, 0.6);
/// Player eye height above the feet in meters.
pub const PLAYER_EYE: f32 = 1.62;
/// Maximum ledge height auto-stepped by the character controller, in meters.
pub const STEP_HEIGHT: f32 = 0.55;
/// Jump apex height in meters.
pub const JUMP_HEIGHT: f32 = 1.25;
/// Tool raycast reach in meters.
pub const REACH: f32 = 5.0;
/// Default blast radius in meters.
pub const BLAST_RADIUS: f32 = 1.5;
/// Detached components smaller than this many voxels are discarded as debris.
pub const DEBRIS_MIN_VOXELS: usize = 4;
/// Components larger than this many voxels stay in-world instead of becoming
/// rigid bodies. Must comfortably exceed a fully generated tree's
/// disconnected canopy (crown + several branch canopies, each up to a ~2.2 m
/// ellipsoid) severed near its base -- at 0.1 m voxels that can reach ~150k-
/// 200k voxels, so a cap too close to that (65_536 undershoots it) makes
/// severing a tree misfire unpredictably depending on its randomized size.
pub const MAX_BODY_VOXELS: usize = 300_000;
