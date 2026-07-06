//! Voxel-native physics: kinematic character controller, rigidbody solver,
//! and the destruction pipeline (carve → connectivity → debris).

pub mod body;
pub mod character;

pub use body::{Body, BodyId, MassProps, VoxelGrid, mass_props, surface_points};
pub use character::{Aabb, CharacterController, sweep_axis};
