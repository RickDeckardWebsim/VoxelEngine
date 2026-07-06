//! Voxel-native physics: kinematic character controller, rigidbody solver,
//! and the destruction pipeline (carve → connectivity → debris).

pub mod character;

pub use character::{Aabb, CharacterController, sweep_axis};
