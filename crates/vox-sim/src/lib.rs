//! Cellular-automata fluid simulation. Water lives directly in the real
//! `vox_world::World` chunk grid as an ordinary material (no separate
//! storage) -- this crate only adds *behavior*: which cells are actively
//! flowing, and the rule that moves them.
//!
//! See `docs/plans/2026-07-09-fluid-sim-design.md` for the full design
//! rationale (why full/empty cells instead of fractional levels, why
//! active-cell sleeping, why this doesn't need its own grid).

mod fluid;

pub use fluid::FluidSim;
