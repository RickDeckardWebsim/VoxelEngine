//! Fire spreading and consumption, fed by the same tick loop as the fluid
//! and weathering sims. Tracks only cells currently burning in a sparse map
//! that drains to empty at steady state (all fire burns out or is
//! extinguished → zero cost). See `docs/plans/2026-07-09-fire-system-design.md`.

use std::collections::VecDeque;

use glam::IVec3;
use vox_core::{FxHashMap, FxHashSet};
use vox_world::{AIR, Voxel, World};

const NEIGHBORS_6: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

/// Face preference for smoke outlets. Smoke should leave the highest open
/// face first, then a side, and only travel downward as a last resort.
const SMOKE_FACES: [IVec3; 6] = [
    IVec3::Y,
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Z,
    IVec3::NEG_Z,
    IVec3::NEG_Y,
];
const SMOKE_INTERVAL_TICKS: u32 = 10;

/// Burn ticks (at ~15 Hz) before a fire can spread to neighbors.
pub const SPREAD_DELAY_TICKS: u32 = 30; // ~2s
/// Burn ticks before grass is consumed → ash.
pub const GRASS_BURN_TICKS: u32 = 45; // ~3s
/// Burn ticks before leaves are consumed → ash.
pub const LEAVES_BURN_TICKS: u32 = 75; // ~5s
/// Burn ticks before planks are consumed → ash.
pub const PLANKS_BURN_TICKS: u32 = 180; // ~12s
/// Burn ticks before wood is consumed → ash.
pub const WOOD_BURN_TICKS: u32 = 225; // ~15s
/// Burn ticks before ember is consumed → char.
pub const EMBER_BURN_TICKS: u32 = 900; // ~60s

/// Material ids the fire system operates on — resolved by name in the app.
#[derive(Clone)]
pub struct FireTable {
    pub water: Voxel,
    pub ember: Voxel,
    pub char: Voxel,
    pub ash: Voxel,
    pub dark_ash: Voxel,
    pub wood: Voxel,
    pub leaves: Voxel,
    pub planks: Voxel,
    pub grass: Voxel,
    pub muddy_water: Voxel,
    /// All flammable material ids (wood, leaves, planks, grass, ember).
    pub flammable: Vec<Voxel>,
}

impl FireTable {
    fn is_flammable(&self, v: Voxel) -> bool {
        self.flammable.contains(&v)
    }

    /// Whether `v` is a water-like fluid (water or muddy_water).
    #[inline]
    pub fn is_wet(&self, v: Voxel) -> bool {
        v == self.water || v == self.muddy_water
    }

    /// Burn duration for a given flammable material, in ticks.
    /// Burn duration for a given flammable material, in ticks. Uses the
    /// *original* material (before it was swapped to ember on ignition),
    /// not the current voxel — otherwise a burning wood cell (now ember)
    /// would burn for ember's 900-tick duration instead of wood's 225.
    fn burn_ticks(&self, original: Voxel) -> u32 {
        if original == self.ember {
            EMBER_BURN_TICKS
        } else if original == self.grass {
            GRASS_BURN_TICKS
        } else if original == self.leaves {
            LEAVES_BURN_TICKS
        } else if original == self.planks {
            PLANKS_BURN_TICKS
        } else {
            WOOD_BURN_TICKS // wood and any unknown flammable
        }
    }
}

#[derive(Clone, Copy)]
struct BurnState {
    ticks: u32,
    /// The original material before ignition swapped it to ember. Used
    /// for burn duration and to decide the residue (ash vs char).
    original: Voxel,
}

/// Why a smoke particle is being emitted. Consumers can give brief
/// extinguish/consumption puffs different presets from ongoing fire smoke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmokeKind {
    Burning,
    Extinguished,
    Consumed,
}

/// Events emitted during a tick, drained by the app for particle effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FireEvent {
    /// A cell was consumed by fire and became ash (or char for raw ember).
    Consumed(IVec3),
    /// A fire was extinguished by water contact.
    Extinguished(IVec3),
    /// Smoke may leave only through a face-adjacent, in-bounds air voxel.
    Smoke {
        pos: IVec3,
        face: IVec3,
        kind: SmokeKind,
    },
}

pub struct FireSim {
    table: FireTable,
    burning: FxHashMap<IVec3, BurnState>,
    /// Spread-eligible cells that still have a possible fuel neighbor. Cells
    /// that find no fuel go dormant until a nearby world edit wakes them.
    spread_frontier: VecDeque<IVec3>,
    spread_queued: FxHashSet<IVec3>,
    /// Exact half-open bounds around `burning`, maintained after each tick.
    burning_bounds: Option<(IVec3, IVec3)>,
    events: Vec<FireEvent>,
    /// xorshift64* for randomized spread order.
    rng: u64,
}

impl FireSim {
    pub fn new(table: FireTable) -> Self {
        Self {
            table,
            burning: FxHashMap::default(),
            spread_frontier: VecDeque::new(),
            spread_queued: FxHashSet::default(),
            burning_bounds: None,
            events: Vec::new(),
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Debug/test stats.
    pub fn burning_count(&self) -> usize {
        self.burning.len()
    }

    /// Whether this exact world cell is tracked as actively burning.
    #[inline]
    pub fn is_burning(&self, pos: IVec3) -> bool {
        self.burning.contains_key(&pos)
    }

    /// Half-open world-voxel bounds containing every actively burning cell.
    /// Returns `None` when the simulation is idle.
    #[inline]
    pub fn burning_bounds(&self) -> Option<(IVec3, IVec3)> {
        self.burning_bounds
    }

    /// Ignite a cell at `pos`. If the cell is flammable, it enters the burn
    /// map and its voxel is swapped to ember (orange visual). The original
    /// material is stored in BurnState for correct burn duration and residue
    /// selection (ash for full burn, char for extinguished). Called by the
    /// app when placing an ember, or internally during spread.
    pub fn ignite(&mut self, world: &mut World, pos: IVec3) {
        if self.burning.contains_key(&pos) {
            return;
        }
        let v = world.get_voxel(pos);
        if !self.table.is_flammable(v) {
            return;
        }
        if v != self.table.ember {
            world.set_voxel(pos, self.table.ember);
        }
        self.burning.insert(
            pos,
            BurnState {
                ticks: 0,
                original: v,
            },
        );
        include_in_bounds(&mut self.burning_bounds, pos);
    }

    /// Drain this tick's events while retaining the allocation for reuse.
    pub fn drain_events(&mut self) -> std::vec::Drain<'_, FireEvent> {
        self.events.drain(..)
    }

    /// Advance the fire simulation by one tick. Checks extinguishing,
    /// spreads fire to flammable neighbors, advances burn timers, and
    /// consumes cells that have burned long enough.
    pub fn tick(&mut self, world: &mut World) {
        self.events.clear();
        let Self {
            table,
            burning,
            spread_frontier,
            spread_queued,
            burning_bounds,
            events,
            rng,
        } = self;

        // 1. Extinguish: remove burning cells adjacent to water. Any
        //    burning cell (now ember visually) extinguished by water
        //    becomes char (partial burn residue).
        let mut extinguished: Vec<IVec3> = Vec::new();
        let count_before_extinguish = burning.len();
        burning.retain(|&pos, _| {
            // Skip stale entries (cell extracted as debris — no longer
            // ember in the world).
            if world.get_voxel(pos) != table.ember {
                return false;
            }
            let water_adjacent = NEIGHBORS_6
                .iter()
                .any(|&n| table.is_wet(world.get_voxel(pos + n)));
            if water_adjacent {
                extinguished.push(pos);
                false
            } else {
                true
            }
        });
        let mut removed_any = burning.len() != count_before_extinguish;
        for pos in extinguished {
            world.set_voxel(pos, table.char);
            // Wet any ash neighbors of the extinguished cell.
            for n in NEIGHBORS_6 {
                let q = pos + n;
                if world.get_voxel(q) == table.ash
                    && NEIGHBORS_6
                        .iter()
                        .any(|&d| table.is_wet(world.get_voxel(q + d)))
                {
                    world.set_voxel(q, table.dark_ash);
                }
            }
            events.push(FireEvent::Extinguished(pos));
            emit_smoke(events, world, pos, SmokeKind::Extinguished);
        }

        // 2. Spread: process only cells that became eligible, successfully
        //    spread last tick, or were explicitly woken by a nearby edit.
        //    Capturing the queue length prevents a successful spreader from
        //    running twice in the same tick after it requeues itself.
        let spread_attempts = spread_frontier.len();
        for _ in 0..spread_attempts {
            let Some(pos) = spread_frontier.pop_front() else {
                break;
            };
            spread_queued.remove(&pos);
            if !burning
                .get(&pos)
                .is_some_and(|state| state.ticks >= SPREAD_DELAY_TICKS)
            {
                continue;
            }

            let mut dirs: [IVec3; 6] = NEIGHBORS_6;
            for i in (1..6).rev() {
                let j = (next_u64(rng) as usize) % (i + 1);
                dirs.swap(i, j);
            }
            let mut spread = false;
            for dir in &dirs {
                let neighbor = pos + *dir;
                if burning.contains_key(&neighbor) {
                    continue;
                }
                let original = world.get_voxel(neighbor);
                if table.is_flammable(original) {
                    if original != table.ember {
                        world.set_voxel(neighbor, table.ember);
                    }
                    burning.insert(neighbor, BurnState { ticks: 0, original });
                    include_in_bounds(burning_bounds, neighbor);
                    spread = true;
                    break;
                }
            }
            if spread && spread_queued.insert(pos) {
                spread_frontier.push_back(pos);
            }
        }

        // 3. Advance + consume: increment tick counters, consume cells
        //    that reached their burn duration. Full burn → ash; original
        //    ember → char. Emit events.
        let mut consumed: Vec<(IVec3, Voxel)> = Vec::new();
        for (&pos, state) in burning.iter_mut() {
            state.ticks += 1;
            let duration = table.burn_ticks(state.original);
            if state.ticks >= duration {
                consumed.push((pos, state.original));
            } else {
                if state.ticks == SPREAD_DELAY_TICKS && spread_queued.insert(pos) {
                    spread_frontier.push_back(pos);
                }
                if smoke_due(pos, state.ticks) {
                    emit_smoke(events, world, pos, SmokeKind::Burning);
                }
            }
        }
        for (pos, original) in consumed {
            if original == table.ember {
                world.set_voxel(pos, table.char);
            } else {
                // Phase 1 already removed every water-adjacent cell, so a
                // fully consumed non-ember cell always becomes dry ash.
                world.set_voxel(pos, table.ash);
            }
            burning.remove(&pos);
            removed_any = true;
            events.push(FireEvent::Consumed(pos));
            emit_smoke(events, world, pos, SmokeKind::Consumed);
        }

        if removed_any {
            *burning_bounds = calculate_bounds(burning);
        }
    }

    /// Reactivate fire awareness for a region (e.g. after a world edit
    /// exposes new flammable material near a fire). The fire sim doesn't
    /// scan the world proactively — this lets the app wake it after edits.
    pub fn wake_region(&mut self, world: &mut World, min: IVec3, max: IVec3) {
        let (bounds_min, bounds_max) = world.bounds_voxels();
        let min = min.max(bounds_min);
        let max = max.min(bounds_max);
        if min.cmpge(max).any() {
            return;
        }
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

        // A fuel placement just outside an exhausted fire must wake that
        // fire as well as any ember directly inside the edited region.
        let wake_min = (min - IVec3::ONE).max(bounds_min);
        let wake_max = (max + IVec3::ONE).min(bounds_max);
        for x in wake_min.x..wake_max.x {
            for y in wake_min.y..wake_max.y {
                for z in wake_min.z..wake_max.z {
                    let pos = IVec3::new(x, y, z);
                    if self
                        .burning
                        .get(&pos)
                        .is_some_and(|state| state.ticks >= SPREAD_DELAY_TICKS)
                        && self.spread_queued.insert(pos)
                    {
                        self.spread_frontier.push_back(pos);
                    }
                }
            }
        }
    }
}

#[inline]
fn next_u64(rng: &mut u64) -> u64 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    *rng
}

#[inline]
fn smoke_due(pos: IVec3, ticks: u32) -> bool {
    let phase = (pos.x as u32).wrapping_mul(0x9E37_79B1)
        ^ (pos.y as u32).wrapping_mul(0x85EB_CA77)
        ^ (pos.z as u32).wrapping_mul(0xC2B2_AE3D);
    (ticks + phase % SMOKE_INTERVAL_TICKS) % SMOKE_INTERVAL_TICKS == 0
}

fn smoke_face(world: &World, pos: IVec3) -> Option<IVec3> {
    SMOKE_FACES.into_iter().find(|&face| {
        let outlet = pos + face;
        world.in_bounds(outlet) && world.get_voxel(outlet) == AIR
    })
}

fn emit_smoke(events: &mut Vec<FireEvent>, world: &World, pos: IVec3, kind: SmokeKind) {
    if let Some(face) = smoke_face(world, pos) {
        events.push(FireEvent::Smoke { pos, face, kind });
    }
}

fn include_in_bounds(bounds: &mut Option<(IVec3, IVec3)>, pos: IVec3) {
    match bounds {
        Some((min, max)) => {
            *min = min.min(pos);
            *max = max.max(pos + IVec3::ONE);
        }
        None => *bounds = Some((pos, pos + IVec3::ONE)),
    }
}

fn calculate_bounds(burning: &FxHashMap<IVec3, BurnState>) -> Option<(IVec3, IVec3)> {
    let mut positions = burning.keys().copied();
    let first = positions.next()?;
    let mut min = first;
    let mut max = first + IVec3::ONE;
    for pos in positions {
        min = min.min(pos);
        max = max.max(pos + IVec3::ONE);
    }
    Some((min, max))
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
    const ASH: Voxel = Voxel(9);
    const DARK_ASH: Voxel = Voxel(10);
    const MUDDY_WATER: Voxel = Voxel(11);
    const STONE: Voxel = Voxel(8);

    fn table() -> FireTable {
        FireTable {
            water: WATER,
            ember: EMBER,
            char: CHAR,
            ash: ASH,
            dark_ash: DARK_ASH,
            wood: WOOD,
            leaves: LEAVES,
            planks: PLANKS,
            grass: GRASS,
            muddy_water: MUDDY_WATER,
            flammable: vec![WOOD, LEAVES, GRASS, PLANKS, EMBER],
        }
    }

    fn world_with_floor(top: Voxel) -> World {
        let mut w = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [16.0, 16.0, 16.0],
            ..WorldConfig::default()
        });
        w.set_solid_table(vec![false, false, true, true, true, true, true, true, true, true, true, false]);
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
        sim.ignite(&mut world, ember_pos);
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
        sim.ignite(&mut world, pos);
        assert_eq!(sim.burning_count(), 1);

        for _ in 0..(WOOD_BURN_TICKS - 1) {
            sim.tick(&mut world);
            assert_eq!(
                world.get_voxel(pos),
                EMBER,
                "burning wood must show as ember (orange)"
            );
        }
        sim.tick(&mut world);
        assert_eq!(
            world.get_voxel(pos),
            ASH,
            "fully burned wood must become ash"
        );
        assert_eq!(
            sim.burning_count(),
            0,
            "consumed cell must leave the burn map"
        );
    }

    #[test]
    fn water_extinguishes_fire() {
        let mut world = world_with_floor(STONE);
        let mut sim = FireSim::new(table());
        let pos = IVec3::new(8, 5, 8);
        world.set_voxel(pos, WOOD);
        sim.ignite(&mut world, pos);

        // Place water next to the burning wood.
        world.set_voxel(pos + IVec3::X, WATER);
        sim.tick(&mut world);

        assert_eq!(
            sim.burning_count(),
            0,
            "water-adjacent fire must be extinguished"
        );
        assert_eq!(
            world.get_voxel(pos),
            CHAR,
            "extinguished burning cell must become char"
        );

        let events: Vec<_> = sim.drain_events().collect();
        assert!(
            events.contains(&FireEvent::Extinguished(pos)),
            "must emit Extinguished event"
        );
    }

    #[test]
    fn water_extinguishes_ember_to_char() {
        let mut world = world_with_floor(STONE);
        let mut sim = FireSim::new(table());
        let pos = IVec3::new(8, 5, 8);
        world.set_voxel(pos, EMBER);
        sim.ignite(&mut world, pos);
        assert_eq!(sim.burning_count(), 1);

        world.set_voxel(pos + IVec3::X, WATER);
        sim.tick(&mut world);

        assert_eq!(sim.burning_count(), 0, "ember must be extinguished");
        assert_eq!(
            world.get_voxel(pos),
            CHAR,
            "ember extinguished by water → char"
        );
    }

    #[test]
    fn muddy_water_extinguishes_fire() {
        let mut world = world_with_floor(STONE);
        let mut sim = FireSim::new(table());
        let pos = IVec3::new(8, 5, 8);
        world.set_voxel(pos, WOOD);
        sim.ignite(&mut world, pos);
        // Place muddy_water next to the burning wood
        world.set_voxel(pos + IVec3::X, MUDDY_WATER);
        sim.tick(&mut world);
        assert_eq!(
            sim.burning_count(),
            0,
            "muddy_water-adjacent fire must be extinguished"
        );
        assert_eq!(
            world.get_voxel(pos),
            CHAR,
            "extinguished burning cell must become char"
        );
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
        sim.ignite(&mut world, IVec3::new(4, 5, 8));

        // Run for a while; fire must spread along the line.
        let mut burned = false;
        for _ in 0..(SPREAD_DELAY_TICKS * 10) {
            sim.tick(&mut world);
            // If the far end is burning or consumed, fire spread.
            if world.get_voxel(IVec3::new(12, 5, 8)) == ASH
                || world.get_voxel(IVec3::new(12, 5, 8)) == CHAR
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
        sim.ignite(&mut world, pos);

        for _ in 0..(WOOD_BURN_TICKS + 50) {
            sim.tick(&mut world);
        }
        assert_eq!(
            sim.burning_count(),
            0,
            "burn map must drain to empty at steady state"
        );
    }

    #[test]
    fn enclosed_burn_emits_no_smoke() {
        let mut world = world_with_floor(STONE);
        let mut sim = FireSim::new(table());
        let pos = IVec3::new(8, 5, 8);
        world.set_voxel(pos, WOOD);
        for face in NEIGHBORS_6 {
            world.set_voxel(pos + face, STONE);
        }
        sim.ignite(&mut world, pos);

        for _ in 0..WOOD_BURN_TICKS {
            sim.tick(&mut world);
            assert!(
                !sim.drain_events()
                    .any(|event| matches!(event, FireEvent::Smoke { .. })),
                "an enclosed voxel must not emit smoke"
            );
        }
    }

    #[test]
    fn top_blocked_smoke_uses_an_open_side_face() {
        let mut world = world_with_floor(STONE);
        let mut sim = FireSim::new(table());
        let pos = IVec3::new(8, 5, 8);
        world.set_voxel(pos, WOOD);
        for face in NEIGHBORS_6 {
            world.set_voxel(pos + face, STONE);
        }
        world.set_voxel(pos + IVec3::X, AIR);
        sim.ignite(&mut world, pos);

        let mut outlet = None;
        for _ in 0..SMOKE_INTERVAL_TICKS {
            sim.tick(&mut world);
            outlet = sim.drain_events().find_map(|event| match event {
                FireEvent::Smoke {
                    face,
                    kind: SmokeKind::Burning,
                    ..
                } => Some(face),
                _ => None,
            });
            if outlet.is_some() {
                break;
            }
        }
        assert_eq!(outlet, Some(IVec3::X));
    }

    #[test]
    fn exhausted_spreader_sleeps_and_wake_region_reactivates_it() {
        let mut world = world_with_floor(STONE);
        let mut sim = FireSim::new(table());
        let source = IVec3::new(8, 5, 8);
        let fuel = source + IVec3::X;
        world.set_voxel(source, WOOD);
        sim.ignite(&mut world, source);

        for _ in 0..SPREAD_DELAY_TICKS {
            sim.tick(&mut world);
        }
        assert_eq!(sim.spread_frontier.front(), Some(&source));

        sim.tick(&mut world);
        assert!(sim.spread_frontier.is_empty());
        assert!(sim.spread_queued.is_empty());
        for _ in 0..3 {
            sim.tick(&mut world);
        }
        assert!(
            sim.spread_frontier.is_empty(),
            "failed spreaders must stay asleep"
        );

        world.set_voxel(fuel, WOOD);
        sim.wake_region(&mut world, fuel, fuel + IVec3::ONE);
        assert_eq!(sim.spread_frontier.front(), Some(&source));
        sim.tick(&mut world);
        assert_eq!(world.get_voxel(fuel), EMBER);
    }

    #[test]
    fn a_spreader_ignites_at_most_one_neighbor_per_tick() {
        let mut world = world_with_floor(STONE);
        let mut sim = FireSim::new(table());
        let source = IVec3::new(8, 5, 8);
        world.set_voxel(source, WOOD);
        world.set_voxel(source + IVec3::X, WOOD);
        world.set_voxel(source + IVec3::Z, WOOD);
        sim.ignite(&mut world, source);

        for _ in 0..SPREAD_DELAY_TICKS {
            sim.tick(&mut world);
        }
        assert_eq!(sim.burning_count(), 1);
        sim.tick(&mut world);
        assert_eq!(
            sim.burning_count(),
            2,
            "only one neighbor may ignite per tick"
        );
        sim.tick(&mut world);
        assert_eq!(
            sim.burning_count(),
            3,
            "the successful source should retry next tick"
        );
    }

    #[test]
    fn non_flammable_material_does_not_ignite() {
        let mut world = world_with_floor(STONE);
        let mut sim = FireSim::new(table());
        let pos = IVec3::new(8, 5, 8);
        // Stone is not flammable — ignite should be a no-op.
        sim.ignite(&mut world, pos);
        assert_eq!(sim.burning_count(), 0, "stone must not ignite");
    }
}
