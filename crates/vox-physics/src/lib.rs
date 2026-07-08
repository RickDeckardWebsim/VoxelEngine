//! Voxel-native physics: kinematic character controller, rigidbody solver,
//! and the destruction pipeline (carve → connectivity → debris).

pub mod body;
pub mod body_destruction;
pub mod broadphase;
pub mod character;
pub mod contact;
pub mod destruction;
pub mod solver;

pub use body::{Body, BodyId, GridRayHit, MassProps, VoxelGrid, mass_props, raycast_grid, surface_points};
pub use body_destruction::{
    carve_body_capsule_at, carve_body_explosion_at, carve_body_sphere_at,
    carve_body_sphere_at_impact, carve_body_voxel_at, split_components,
};
pub use character::{Aabb, CharacterController, sweep_axis};
pub use destruction::{
    CarveResult, Region, apply_blast_impulse, blast, carve_capsule, carve_explosion, carve_sphere,
    detach_unsupported, laser,
};
pub use solver::{ImpactEvent, PhysicsWorld};
