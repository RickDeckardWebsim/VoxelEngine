//! Cellular-automata fluid simulation. Water lives directly in the real
//! `vox_world::World` chunk grid as an ordinary material (no separate
//! storage) -- this crate only adds *behavior*: which cells are actively
//! flowing, and the rule that moves them.
//!
//! See `docs/plans/2026-07-09-fluid-sim-design.md` for the full design
//! rationale (why full/empty cells instead of fractional levels, why
//! active-cell sleeping, why this doesn't need its own grid).
//!
//! Weathering consumes the fluid tick's `ContactEvent`s to transform
//! terrain touched by water (grass -> dirt -> mud, stone -> sand).
//!
//! Fire spreads through flammable materials, consumes them to char, and is
//! extinguished by water. See `docs/plans/2026-07-09-fire-system-design.md`.

mod fluid;
mod fire;
mod weathering;

pub use fluid::{ContactEvent, FluidSim};
pub use fire::{
    EMBER_BURN_TICKS, FireEvent, FireSim, FireTable, GRASS_BURN_TICKS, LEAVES_BURN_TICKS,
    PLANKS_BURN_TICKS, SPREAD_DELAY_TICKS, WOOD_BURN_TICKS,
};
pub use weathering::{
    DIRT_SOAK_TICKS, GRASS_SOAK_TICKS, MUD_DRY_TICKS, STONE_ERODE_TICKS, STONE_FALL_BOOST,
    WeatherTable, Weathering,
};
