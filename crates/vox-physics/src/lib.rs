//! Voxel-native physics: kinematic character controller, rigidbody solver,
//! and the destruction pipeline (carve → connectivity → debris).

pub mod body;
pub mod broadphase;
pub mod character;
pub mod contact;
pub mod destruction;
pub mod solver;

pub use body::{Body, BodyId, MassProps, VoxelGrid, mass_props, surface_points};
pub use character::{Aabb, CharacterController, sweep_axis};
pub use destruction::{CarveResult, Region, blast, carve_sphere, detach_unsupported};
pub use solver::PhysicsWorld;
