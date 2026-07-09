//! The active-cell fluid automaton. See the crate root docs and
//! `docs/plans/2026-07-09-fluid-sim-design.md` for the rationale.

use glam::IVec3;
use vox_core::FxHashSet;

/// Tracks which water cells are still moving. A cell not in this set is
/// settled and costs nothing to tick -- the entire performance story of
/// this crate (mirrors `PhysicsWorld`'s sleep bookkeeping).
pub struct FluidSim {
    active: FxHashSet<IVec3>,
    /// xorshift64* state for randomized per-tick update order (same
    /// construction as `PhysicsWorld::lifetime_rng` / `ParticleSystem`'s
    /// spawn jitter) -- avoids a visible left/right or diagonal bias in how
    /// water spreads.
    rng: u64,
}

impl Default for FluidSim {
    fn default() -> Self {
        Self::new()
    }
}

impl FluidSim {
    pub fn new() -> Self {
        Self {
            active: FxHashSet::default(),
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Number of cells currently flowing (debug-overlay stat).
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// xorshift64* -- deterministic, dependency-free.
    fn next_u64(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_sim_has_no_active_cells() {
        let sim = FluidSim::new();
        assert_eq!(sim.active_count(), 0);
    }
}
