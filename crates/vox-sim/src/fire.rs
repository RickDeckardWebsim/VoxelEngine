//! Fire spreading and consumption, fed by the same tick loop as the fluid
//! and weathering sims. Tracks only cells currently burning in a sparse map
//! that drains to empty at steady state (all fire burns out or is
//! extinguished → zero cost). See `docs/plans/2026-07-09-fire-system-design.md`.

use glam::IVec3;
use vox_core::FxHashMap;
use vox_world::{Voxel, World};

const NEIGHBORS_6: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

/// Burn ticks (at ~15 Hz) before a fire can spread to neighbors.
pub const SPREAD_DELAY_TICKS: u32 = 30; // ~2s
/// Burn ticks before grass is consumed → char.
pub const GRASS_BURN_TICKS: u32 = 45; // ~3s
/// Burn ticks before leaves are consumed → char.
pub const LEAVES_BURN_TICKS: u32 = 75; // ~5s
/// Burn ticks before planks are consumed → char.
pub const PLANKS_BURN_TICKS: u32 = 180; // ~12s
/// Burn ticks before wood is consumed → char.
pub const WOOD_BURN_TICKS: u32 = 225; // ~15s
/// Burn ticks before ember is consumed → char.
pub const EMBER_BURN_TICKS: u32 = 900; // ~60s

/// Material ids the fire system operates on — resolved by name in the app.
#[derive(Clone)]
pub struct FireTable {
    pub water: Voxel,
    pub ember: Voxel,
    pub char: Voxel,
    pub wood: Voxel,
    pub leaves: Voxel,
    pub planks: Voxel,
    pub grass: Voxel,
    /// All flammable material ids (wood, leaves, planks, grass, ember).
    pub flammable: Vec<Voxel>,
}

impl FireTable {
    fn is_flammable(&self, v: Voxel) -> bool {
        self.flammable.contains(&v)
    }

    /// Burn duration for a given flammable material, in ticks.
    fn burn_ticks(&self, v: Voxel) -> u32 {
        if v == self.ember {
            EMBER_BURN_TICKS
        } else if v == self.grass {
            GRASS_BURN_TICKS
        } else if v == self.leaves {
            LEAVES_BURN_TICKS
        } else if v == self.planks {
            PLANKS_BURN_TICKS
        } else {
            WOOD_BURN_TICKS // wood and any unknown flammable
        }
    }
}

#[derive(Clone, Copy)]
struct BurnState {
    ticks: u32,
}

/// Events emitted during a tick, drained by the app for particle effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FireEvent {
    /// A cell was consumed by fire and became char.
    Consumed(IVec3),
    /// A fire was extinguished by water contact.
    Extinguished(IVec3),
    /// A cell is actively burning this tick (for smoke particle emission).
    /// The intensity (0..1) scales with how far through its burn it is.
    Burning(IVec3, u32), // position, ticks burning
}

pub struct FireSim {
    table: FireTable,
    burning: FxHashMap<IVec3, BurnState>,
    events: Vec<FireEvent>,
    /// xorshift64* for randomized spread order.
    rng: u64,
}

impl FireSim {
    pub fn new(table: FireTable) -> Self {
        Self {
            table,
            burning: FxHashMap::default(),
            events: Vec::new(),
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    /// Debug/test stats.
    pub fn burning_count(&self) -> usize {
        self.burning.len()
    }

    /// Ignite a cell at `pos`. If the cell is flammable, it enters the burn
    /// map. Called by the app when placing an ember, or internally during
    /// spread.
    pub fn ignite(&mut self, world: &World, pos: IVec3) {
        let v = world.get_voxel(pos);
        if self.table.is_flammable(v) && !self.burning.contains_key(&pos) {
            self.burning.insert(pos, BurnState {
                ticks: 0,
            });
        }
    }

    /// Take this tick's events (empties the buffer).
    pub fn drain_events(&mut self) -> Vec<FireEvent> {
        std::mem::take(&mut self.events)
    }

    /// Advance the fire simulation by one tick. Checks extinguishing,
    /// spreads fire to flammable neighbors, advances burn timers, and
    /// consumes cells that have burned long enough.
    pub fn tick(&mut self, world: &mut World) {
        self.events.clear();
        let table = self.table.clone();

        // 1. Extinguish: remove burning cells adjacent to water. Ember
        //    touching water becomes char (extinguished permanently).
        let mut extinguished = Vec::new();
        self.burning.retain(|&pos, _| {
            let water_adjacent = NEIGHBORS_6.iter().any(|&n| world.get_voxel(pos + n) == table.water);
            if water_adjacent {
                extinguished.push(pos);
                false
            } else {
                true
            }
        });
        for pos in &extinguished {
            // Ember extinguished by water → char. Other materials just
            // stop burning (they survive, partially charred visually but
            // unchanged in the grid).
            if world.get_voxel(*pos) == table.ember {
                world.set_voxel(*pos, table.char);
            }
            self.events.push(FireEvent::Extinguished(*pos));
        }

        // 2. Spread: each burning cell that has burned long enough can
        //    ignite one flammable neighbor per tick (randomized order).
        //    Collect new ignitions to apply after iteration (can't mutate
        //    the map while iterating).
        // 2. Spread: each burning cell that has burned long enough can
        //    ignite one flammable neighbor per tick (randomized order).
        //    Collect spread-eligible positions first (can't call
        //    self.next_u64 while borrowing &self.burning).
        let spreaders: Vec<IVec3> = self.burning.iter()
            .filter(|(_, s)| s.ticks >= SPREAD_DELAY_TICKS)
            .map(|(&p, _)| p)
            .collect();
        let mut new_ignitions: Vec<IVec3> = Vec::new();
        for pos in &spreaders {
            // Randomized neighbor order.
            let mut dirs: [IVec3; 6] = NEIGHBORS_6;
            for i in (1..6).rev() {
                let j = (self.next_u64() as usize) % (i + 1);
                dirs.swap(i, j);
            }
            for dir in &dirs {
                let neighbor = *pos + *dir;
                if self.burning.contains_key(&neighbor) || new_ignitions.contains(&neighbor) {
                    continue;
                }
                let nv = world.get_voxel(neighbor);
                if table.is_flammable(nv) {
                    new_ignitions.push(neighbor);
                    break; // one spread per burning cell per tick
                }
            }
        }
        for pos in &new_ignitions {
            self.burning.insert(*pos, BurnState {
                ticks: 0,
            });
        }

        // 3. Advance + consume: increment tick counters, consume cells
        //    that reached their burn duration, emit events.
        let mut consumed = Vec::new();
        for (&pos, state) in &mut self.burning {
            state.ticks += 1;
            let v = world.get_voxel(pos);
            let duration = table.burn_ticks(v);
            if state.ticks >= duration {
                consumed.push(pos);
            } else {
                // Emit a burning event for smoke particle emission.
                // Every 3rd tick to avoid flooding the particle system.
                if state.ticks % 3 == 0 {
                    self.events.push(FireEvent::Burning(pos, state.ticks));
                }
            }
        }
        for pos in &consumed {
            world.set_voxel(*pos, table.char);
            self.burning.remove(pos);
            self.events.push(FireEvent::Consumed(*pos));
        }
    }

    /// Reactivate fire awareness for a region (e.g. after a world edit
    /// exposes new flammable material near a fire). The fire sim doesn't
    /// scan the world proactively — this lets the app wake it after edits.
    pub fn wake_region(&mut self, world: &World, min: IVec3, max: IVec3) {
        let (bounds_min, bounds_max) = world.bounds_voxels();
        let min = min.max(bounds_min);
        let max = max.min(bounds_max);
        for x in min.x..max.x {
            for y in min.y..max.y {
                for z in min.z..max.z {
                    let p = IVec3::new(x, y, z);
                    let v = world.get_voxel(p);
                    if v == self.table.ember && !self.burning.contains_key(&p) {
                        self.ignite(world, p);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::WorldConfig;
    use vox_world::{AIR, Voxel, World};

    const WATER: Voxel = Voxel(1);
    const WOOD: Voxel = Voxel(2);
    const LEAVES: Voxel = Voxel(3);
    const GRASS: Voxel = Voxel(4);
    const PLANKS: Voxel = Voxel(5);
    const EMBER: Voxel = Voxel(6);
    const CHAR: Voxel = Voxel(7);
    const STONE: Voxel = Voxel(8);

    fn table() -> FireTable {
        FireTable {
            water: WATER,
            ember: EMBER,
            char: CHAR,
            wood: WOOD,
            leaves: LEAVES,
            planks: PLANKS,
            grass: GRASS,
            flammable: vec![WOOD, LEAVES, GRASS, PLANKS, EMBER],
        }
    }

    fn world_with_floor(top: Voxel) -> World {
        let mut w = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [16.0, 16.0, 16.0],
            ..WorldConfig::default()
        });
        w.set_solid_table(vec![false, false, true, true, true, true, true, true, true]);
        let (_, max) = w.bounds_voxels();
        w.fill_box(IVec3::ZERO, IVec3::new(max.x, 5, max.z), STONE);
        w.fill_box(IVec3::new(0, 5, 0), IVec3::new(max.x, 6, max.z), top);
        w
    }

    #[test]
    fn ember_ignites_and_burns_neighbors() {
        let mut world = world_with_floor(WOOD);
        let mut sim = FireSim::new(table());
        let ember_pos = IVec3::new(8, 5, 8);
        world.set_voxel(ember_pos, EMBER);
        sim.ignite(&world, ember_pos);
        assert_eq!(sim.burning_count(), 1, "ember must enter the burn map");

        // Run past the spread delay; the adjacent wood must ignite.
        for _ in 0..(SPREAD_DELAY_TICKS + 10) {
            sim.tick(&mut world);
        }
        assert!(sim.burning_count() >= 2, "fire must spread to neighbors");
    }

    #[test]
    fn wood_burns_to_char_at_threshold() {
        let mut world = world_with_floor(STONE);
        let mut sim = FireSim::new(table());
        // A single wood block on stone, no neighbors to spread to.
        let pos = IVec3::new(8, 5, 8);
        world.set_voxel(pos, WOOD);
        sim.ignite(&world, pos);
        assert_eq!(sim.burning_count(), 1);

        for _ in 0..(WOOD_BURN_TICKS - 1) {
            sim.tick(&mut world);
            assert_eq!(world.get_voxel(pos), WOOD, "wood must not be consumed early");
        }
        sim.tick(&mut world);
        assert_eq!(world.get_voxel(pos), CHAR, "wood must become char at burn threshold");
        assert_eq!(sim.burning_count(), 0, "consumed cell must leave the burn map");
    }

    #[test]
    fn water_extinguishes_fire() {
        let mut world = world_with_floor(STONE);
        let mut sim = FireSim::new(table());
        let pos = IVec3::new(8, 5, 8);
        world.set_voxel(pos, WOOD);
        sim.ignite(&world, pos);

        // Place water next to the burning wood.
        world.set_voxel(pos + IVec3::X, WATER);
        sim.tick(&mut world);

        assert_eq!(sim.burning_count(), 0, "water-adjacent fire must be extinguished");
        assert_eq!(world.get_voxel(pos), WOOD, "extinguished wood survives (only ember→char)");

        let events = sim.drain_events();
        assert!(events.contains(&FireEvent::Extinguished(pos)), "must emit Extinguished event");
    }

    #[test]
    fn water_extinguishes_ember_to_char() {
        let mut world = world_with_floor(STONE);
        let mut sim = FireSim::new(table());
        let pos = IVec3::new(8, 5, 8);
        world.set_voxel(pos, EMBER);
        sim.ignite(&world, pos);
        assert_eq!(sim.burning_count(), 1);

        world.set_voxel(pos + IVec3::X, WATER);
        sim.tick(&mut world);

        assert_eq!(sim.burning_count(), 0, "ember must be extinguished");
        assert_eq!(world.get_voxel(pos), CHAR, "ember extinguished by water → char");
    }

    #[test]
    fn fire_spreads_along_a_line_of_wood() {
        let mut world = world_with_floor(STONE);
        let mut sim = FireSim::new(table());
        // Line of wood on top of stone floor.
        for x in 4..=12 {
            world.set_voxel(IVec3::new(x, 5, 8), WOOD);
        }
        // Ignite one end.
        sim.ignite(&world, IVec3::new(4, 5, 8));

        // Run for a while; fire must spread along the line.
        let mut burned = false;
        for _ in 0..(SPREAD_DELAY_TICKS * 10) {
            sim.tick(&mut world);
            // If the far end is burning or consumed, fire spread.
            if world.get_voxel(IVec3::new(12, 5, 8)) == CHAR
                || sim.burning_count() > 1
            {
                burned = true;
                break;
            }
        }
        assert!(burned, "fire must spread along the wood line");
    }

    #[test]
    fn burn_map_empties_after_all_fire_consumes() {
        let mut world = world_with_floor(STONE);
        let mut sim = FireSim::new(table());
        // A single wood block — no spread, just burns out.
        let pos = IVec3::new(8, 5, 8);
        world.set_voxel(pos, WOOD);
        sim.ignite(&world, pos);

        for _ in 0..(WOOD_BURN_TICKS + 50) {
            sim.tick(&mut world);
        }
        assert_eq!(sim.burning_count(), 0, "burn map must drain to empty at steady state");
    }

    #[test]
    fn non_flammable_material_does_not_ignite() {
        let mut world = world_with_floor(STONE);
        let mut sim = FireSim::new(table());
        let pos = IVec3::new(8, 5, 8);
        // Stone is not flammable — ignite should be a no-op.
        sim.ignite(&world, pos);
        assert_eq!(sim.burning_count(), 0, "stone must not ignite");
    }
}
