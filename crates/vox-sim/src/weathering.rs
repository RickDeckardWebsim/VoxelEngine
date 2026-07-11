//! Water-driven material transformation, fed by `ContactEvent`s from the
//! fluid tick. Never scans the world: it tracks only cells currently
//! soaking (water-adjacent grass/dirt/stone) or drying (mud that lost its
//! water). Both maps drain to empty at steady state, preserving the
//! settled-water-costs-nothing guarantee. See
//! `docs/plans/2026-07-09-water-refinement-design.md` §3.

use glam::IVec3;
use vox_core::{FxHashMap, FxHashSet};
use vox_world::{AIR, Voxel, World};

use crate::fluid::ContactEvent;

/// Soak ticks (at the fluid tick rate, ~15 Hz) before grass dies to dirt.
pub const GRASS_SOAK_TICKS: u32 = 45; // ~3 s
/// Soak ticks before dirt turns to mud.
pub const DIRT_SOAK_TICKS: u32 = 105; // ~7 s
/// Soak ticks of *flowing* contact before stone erodes to sand.
pub const STONE_ERODE_TICKS: u32 = 450; // ~30 s
/// Waterfall multiplier: stone touched by a `Fell` event *this tick* accrues
/// this many soak ticks per tick (re-derived each tick, not stored -- a
/// waterfall that pools into still water slows to 1x once the falling stops).
pub const STONE_FALL_BOOST: u32 = 5;
/// Dry ticks (no adjacent water) before mud firms back to dirt.
pub const MUD_DRY_TICKS: u32 = 300; // ~20 s
/// Dissolve ticks (at the fluid tick rate, ~15 Hz) before mud adjacent to
/// water dissolves into muddy_water.
pub const MUD_DISSOLVE_TICKS: u32 = 60; // ~4 s
/// Contact ticks before clean water adjacent to muddy_water becomes muddy.
pub const POLLUTE_SPREAD_TICKS: u32 = 90; // ~6 s
/// Settle ticks (continuously still, no moves) before muddy_water clarifies
/// to clean water and deposits sand below.
pub const MUDDY_SETTLE_TICKS: u32 = 150; // ~10 s

const NEIGHBORS_6: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

/// Material ids weathering operates on -- resolved by name in the app;
/// tests build it from raw ids.
#[derive(Clone, Copy)]
pub struct WeatherTable {
    pub water: Voxel,
    pub stone: Voxel,
    pub grass: Voxel,
    pub dirt: Voxel,
    pub mud: Voxel,
    pub sand: Voxel,
    pub muddy_water: Voxel,
}

impl WeatherTable {
    /// Whether `v` is a water-like fluid (water or muddy_water).
    #[inline]
    pub fn is_wet(&self, v: Voxel) -> bool {
        v == self.water || v == self.muddy_water
    }
}

pub struct Weathering {
    table: WeatherTable,
    soaking: FxHashMap<IVec3, u32>,
    drying: FxHashMap<IVec3, u32>,
    dissolving: FxHashMap<IVec3, u32>,
    polluting: FxHashMap<IVec3, u32>,
    settling: FxHashMap<IVec3, u32>,
}

impl Weathering {
    pub fn new(table: WeatherTable) -> Self {
        Self {
            table,
            soaking: FxHashMap::default(),
            drying: FxHashMap::default(),
            dissolving: FxHashMap::default(),
            polluting: FxHashMap::default(),
            settling: FxHashMap::default(),
        }
    }

    /// Debug/test stats.
    pub fn soaking_count(&self) -> usize {
        self.soaking.len()
    }
    pub fn drying_count(&self) -> usize {
        self.drying.len()
    }
    pub fn dissolving_count(&self) -> usize {
        self.dissolving.len()
    }
    pub fn polluting_count(&self) -> usize {
        self.polluting.len()
    }
    pub fn settling_count(&self) -> usize {
        self.settling.len()
    }

    pub fn tick(&mut self, world: &mut World, events: &[ContactEvent]) {
        let t = self.table;

        // 1. Register: water contact puts transformable neighbors on the
        // soak clock; any contact re-wets mud (cancels drying). Stone only
        // registers for *moving* water -- a settled lake never eats its
        // basin. `fell_this_tick` is the set of stone cells a `Fell` event
        // touched *this tick*; the boost is re-derived every tick from it
        // rather than stored, so a waterfall that pools into still water
        // slows to 1x the moment the falling stops (design §3.2: "~5x
        // faster" is while falling, not once fell, forever).
        let mut fell_this_tick: FxHashSet<IVec3> = FxHashSet::default();
        for &ev in events {
            let (pos, moving, fell) = match ev {
                ContactEvent::Fell(p) => (p, true, true),
                ContactEvent::Flowed(p) => (p, true, false),
                ContactEvent::Settled(p) => (p, false, false),
                ContactEvent::Vacated(p) => {
                    // The cell that moved away is no longer settling.
                    self.settling.remove(&p);
                    // Mud that just lost a water neighbor starts drying.
                    for n in NEIGHBORS_6 {
                        let q = p + n;
                        if world.get_voxel(q) == t.mud {
                            self.drying.entry(q).or_insert(0);
                        }
                    }
                    continue;
                }
            };
            for n in NEIGHBORS_6 {
                let q = pos + n;
                let v = world.get_voxel(q);
                if v == t.mud {
                    self.drying.remove(&q); // re-wetted
                    self.dissolving.entry(q).or_insert(0);
                } else if v == t.grass || v == t.dirt || (v == t.stone && moving) {
                    self.soaking.entry(q).or_insert(0);
                    if fell && v == t.stone {
                        fell_this_tick.insert(q);
                    }
                }
            }
            // When the event cell itself is muddy_water, any clean water
            // neighbor is a candidate for contact pollution.
            if world.get_voxel(pos) == t.muddy_water {
                for n in NEIGHBORS_6 {
                    let q = pos + n;
                    if world.get_voxel(q) == t.water {
                        self.polluting.entry(q).or_insert(0);
                    }
                }
                // A still muddy_water cell starts settling toward clean
                // water; a moving one (Fell/Flowed) is not still, so it
                // leaves the settling clock.
                if moving {
                    self.settling.remove(&pos);
                } else {
                    self.settling.entry(pos).or_insert(0);
                }
            }
        }

        // 2. Advance soaking. Entries whose water left, or whose material
        // changed under them (blasted, dug), simply drop out.
        let mut converted = Vec::new();
        self.soaking.retain(|&pos, ticks| {
            let v = world.get_voxel(pos);
            let threshold = if v == t.grass {
                GRASS_SOAK_TICKS
            } else if v == t.dirt {
                DIRT_SOAK_TICKS
            } else if v == t.stone {
                STONE_ERODE_TICKS
            } else {
                return false;
            };
            // Water gone -> the soak dries up without converting.
            if !NEIGHBORS_6
                .iter()
                .any(|&n| t.is_wet(world.get_voxel(pos + n)))
            {
                return false;
            }
            *ticks += if v == t.stone && fell_this_tick.contains(&pos) {
                STONE_FALL_BOOST
            } else {
                1
            };
            if *ticks >= threshold {
                converted.push((pos, v));
                return false;
            }
            true
        });
        for (pos, from) in converted {
            let to = if from == t.grass {
                t.dirt
            } else if from == t.dirt {
                t.mud
            } else {
                t.sand
            };
            world.set_voxel(pos, to);
            // Fresh dirt under standing water keeps soaking toward mud --
            // this is the grass -> dirt -> mud progression.
            if to == t.dirt {
                self.soaking.insert(pos, 0);
            }
        }

        // 2b. Advance dissolving: mud with adjacent wet cell counts toward
        // dissolving; at threshold, becomes muddy_water.
        let mut dissolved = Vec::new();
        self.dissolving.retain(|&pos, ticks| {
            if world.get_voxel(pos) != t.mud {
                return false;
            }
            if !NEIGHBORS_6
                .iter()
                .any(|&n| t.is_wet(world.get_voxel(pos + n)))
            {
                return false; // no wet neighbor
            }
            *ticks += 1;
            if *ticks >= MUD_DISSOLVE_TICKS {
                dissolved.push(pos);
                return false;
            }
            true
        });
        for pos in dissolved {
            // Conserve fluid volume: the mud cell becomes AIR (consumed),
            // and one adjacent WATER cell becomes muddy_water. This avoids
            // creating new fluid from nothing — the previous code turned
            // mud (solid) directly into muddy_water (fluid), adding +1
            // water per dissolved cell.
            if let Some(water_pos) = NEIGHBORS_6
                .iter()
                .map(|&n| pos + n)
                .find(|&q| world.get_voxel(q) == t.water)
            {
                world.set_voxel(water_pos, t.muddy_water);
            }
            world.set_voxel(pos, AIR);
        }
        // 2c. Advance polluting: clean water adjacent to muddy_water counts
        // toward pollution; at threshold, becomes muddy_water.
        let mut polluted = Vec::new();
        self.polluting.retain(|&pos, ticks| {
            if world.get_voxel(pos) != t.water {
                return false;
            }
            if !NEIGHBORS_6
                .iter()
                .any(|&n| world.get_voxel(pos + n) == t.muddy_water)
            {
                return false; // no muddy neighbor
            }
            *ticks += 1;
            if *ticks >= POLLUTE_SPREAD_TICKS {
                polluted.push(pos);
                return false;
            }
            true
        });
        for pos in polluted {
            world.set_voxel(pos, t.muddy_water);
        }

        // 3. Advance drying: mud with water back nearby stops; dry long
        // enough, it firms to dirt.
        let mut dried = Vec::new();
        self.drying.retain(|&pos, ticks| {
            if world.get_voxel(pos) != t.mud {
                return false;
            }
            if NEIGHBORS_6
                .iter()
                .any(|&n| t.is_wet(world.get_voxel(pos + n)))
            {
                return false; // wet again
            }
            *ticks += 1;
            if *ticks >= MUD_DRY_TICKS {
                dried.push(pos);
                return false;
            }
            true
        });
        for pos in dried {
            world.set_voxel(pos, self.table.dirt);
        }

        // 4. Advance settling: muddy_water that has been still for N ticks
        // clarifies to water and deposits sand below.
        let mut settled = Vec::new();
        self.settling.retain(|&pos, ticks| {
            if world.get_voxel(pos) != t.muddy_water {
                return false;
            }
            *ticks += 1;
            if *ticks >= MUDDY_SETTLE_TICKS {
                settled.push(pos);
                return false;
            }
            true
        });
        for pos in settled {
            world.set_voxel(pos, t.water);
            // Deposit sand on the solid cell below (if it's a solid material
            // that isn't already sand).
            let below = pos - IVec3::Y;
            if world.in_bounds(below) {
                let below_v = world.get_voxel(below);
                if below_v != t.sand && !t.is_wet(below_v) && world.solid(below) {
                    world.set_voxel(below, t.sand);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;
    use vox_core::WorldConfig;
    use vox_world::{AIR, Voxel, World};

    const WATER: Voxel = Voxel(1);
    const STONE: Voxel = Voxel(2);
    const GRASS: Voxel = Voxel(3);
    const DIRT: Voxel = Voxel(4);
    const MUD: Voxel = Voxel(5);
    const SAND: Voxel = Voxel(6);
    const MUDDY_WATER: Voxel = Voxel(7);

    fn table() -> WeatherTable {
        WeatherTable {
            water: WATER,
            stone: STONE,
            grass: GRASS,
            dirt: DIRT,
            mud: MUD,
            sand: SAND,
            muddy_water: MUDDY_WATER,
        }
    }

    fn world_with_floor(top: Voxel) -> World {
        let mut w = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [16.0, 16.0, 16.0],
            ..WorldConfig::default()
        });
        // air + water non-solid, everything else solid
        w.set_solid_table(vec![false, false, true, true, true, true, true]);
        let (_, max) = w.bounds_voxels();
        w.fill_box(IVec3::ZERO, IVec3::new(max.x, 5, max.z), STONE);
        w.fill_box(IVec3::new(0, 4, 0), IVec3::new(max.x, 5, max.z), top); // top layer
        w
    }

    #[test]
    fn grass_under_settled_water_dies_to_dirt_at_threshold_not_before() {
        let mut world = world_with_floor(GRASS);
        let mut weathering = Weathering::new(table());
        let cell = IVec3::new(8, 4, 8);
        world.set_voxel(cell + IVec3::Y, WATER); // still water directly on top
        let events = vec![ContactEvent::Settled(cell + IVec3::Y)];
        weathering.tick(&mut world, &events);
        for _ in 0..(GRASS_SOAK_TICKS - 2) {
            weathering.tick(&mut world, &[]);
            assert_eq!(world.get_voxel(cell), GRASS, "must not convert early");
        }
        weathering.tick(&mut world, &[]);
        assert_eq!(
            world.get_voxel(cell),
            DIRT,
            "grass must die to dirt at the soak threshold"
        );
        assert_eq!(
            weathering.soaking_count(),
            1,
            "the fresh dirt re-registers and keeps soaking"
        );
    }

    #[test]
    fn soaked_dirt_becomes_mud() {
        let mut world = world_with_floor(DIRT);
        let mut weathering = Weathering::new(table());
        let cell = IVec3::new(8, 4, 8);
        world.set_voxel(cell + IVec3::Y, WATER);
        weathering.tick(&mut world, &[ContactEvent::Settled(cell + IVec3::Y)]);
        for _ in 0..DIRT_SOAK_TICKS {
            weathering.tick(&mut world, &[]);
        }
        assert_eq!(world.get_voxel(cell), MUD, "soaked dirt must become mud");
    }

    #[test]
    fn mud_under_muddy_water_still_soaks() {
        // The is_wet generalization: dirt under muddy_water must still soak
        // toward mud. This tests that the soak water-adjacency check (line 130)
        // recognizes muddy_water as "wet".
        let mut world = world_with_floor(DIRT);
        let mut weathering = Weathering::new(table());
        let cell = IVec3::new(8, 4, 8);
        world.set_voxel(cell + IVec3::Y, MUDDY_WATER);
        // Update solid table: [air, water, stone, grass, dirt, mud, sand, muddy_water]
        world.set_solid_table(vec![false, false, true, true, true, true, true, false]);
        let events = vec![ContactEvent::Settled(cell + IVec3::Y)];
        weathering.tick(&mut world, &events);
        for _ in 0..DIRT_SOAK_TICKS {
            weathering.tick(&mut world, &[]);
        }
        assert_eq!(
            world.get_voxel(cell),
            MUD,
            "dirt under muddy_water must soak to mud — muddy_water is wet"
        );
    }

    #[test]
    fn still_water_never_erodes_stone_but_flowing_does_and_falling_is_faster() {
        // Still: Settled event over stone -> no soak entry at all.
        let mut world = world_with_floor(STONE);
        let mut weathering = Weathering::new(table());
        let cell = IVec3::new(8, 4, 8);
        world.set_voxel(cell + IVec3::Y, WATER);
        weathering.tick(&mut world, &[ContactEvent::Settled(cell + IVec3::Y)]);
        assert_eq!(
            weathering.soaking_count(),
            0,
            "still water must not register stone"
        );

        // Flowing: erodes at STONE_ERODE_TICKS.
        let mut ticks_flowing = 0;
        weathering.tick(&mut world, &[ContactEvent::Flowed(cell + IVec3::Y)]);
        while world.get_voxel(cell) == STONE {
            weathering.tick(&mut world, &[]);
            ticks_flowing += 1;
            assert!(
                ticks_flowing <= STONE_ERODE_TICKS + 2,
                "flowing erosion must finish near its threshold"
            );
        }
        assert_eq!(world.get_voxel(cell), SAND);

        // Falling: a continuous waterfall -- a Fell event *every tick* --
        // erodes ~5x sooner than the horizontal flow. (The boost is
        // re-derived per tick from this tick's Fell events, so it only
        // applies while water is actually falling onto the stone.)
        let mut world = world_with_floor(STONE);
        let mut weathering = Weathering::new(table());
        world.set_voxel(cell + IVec3::Y, WATER);
        let mut ticks_falling = 0;
        // Seed the soak entry with one Fell, then keep falling every tick.
        weathering.tick(&mut world, &[ContactEvent::Fell(cell + IVec3::Y)]);
        while world.get_voxel(cell) == STONE {
            weathering.tick(&mut world, &[ContactEvent::Fell(cell + IVec3::Y)]);
            ticks_falling += 1;
            assert!(
                ticks_falling <= STONE_ERODE_TICKS / STONE_FALL_BOOST + 2,
                "continuous waterfall erosion must be ~5x faster"
            );
        }
        assert!(
            ticks_falling < ticks_flowing / 3,
            "falling ({ticks_falling}) must be much faster than flowing ({ticks_flowing})"
        );
    }

    #[test]
    fn soak_entries_evaporate_when_the_water_leaves() {
        let mut world = world_with_floor(GRASS);
        let mut weathering = Weathering::new(table());
        let cell = IVec3::new(8, 4, 8);
        world.set_voxel(cell + IVec3::Y, WATER);
        weathering.tick(&mut world, &[ContactEvent::Settled(cell + IVec3::Y)]);
        assert_eq!(weathering.soaking_count(), 1);
        world.set_voxel(cell + IVec3::Y, AIR); // water gone before the threshold
        weathering.tick(&mut world, &[]);
        assert_eq!(
            weathering.soaking_count(),
            0,
            "no adjacent water -> entry removed"
        );
        assert_eq!(world.get_voxel(cell), GRASS, "and the grass survives");
    }

    // A waterfall that registers stone, then pools into still water, must
    // slow to 1x the moment the falling stops. The boost is re-derived each
    // tick from that tick's Fell events, not stored on the soak entry -- so
    // once only Settled events arrive, erosion continues at the normal
    // still-water-adjacency rate (design §3.2 step 2 re-verifies adjacency,
    // not motion). This guards against regressing back to a sticky flag.
    #[test]
    fn waterfall_that_settles_slows_to_normal_rate() {
        let mut world = world_with_floor(STONE);
        let mut weathering = Weathering::new(table());
        let cell = IVec3::new(8, 4, 8);
        let above = cell + IVec3::Y;
        world.set_voxel(above, WATER);

        // One Fell registers the stone; thereafter only Settled (still water).
        weathering.tick(&mut world, &[ContactEvent::Fell(above)]);
        assert_eq!(
            weathering.soaking_count(),
            1,
            "Fell must register the stone"
        );

        // Run under still water up to just past the boosted threshold. If the
        // boost were sticky, stone would erode by ~90 ticks. At the 1x rate
        // it must still be stone well past 90.
        for _ in 0..(STONE_ERODE_TICKS / STONE_FALL_BOOST + 10) {
            weathering.tick(&mut world, &[ContactEvent::Settled(above)]);
            assert_eq!(
                world.get_voxel(cell),
                STONE,
                "stone must not erode at the 5x rate once water has settled"
            );
        }
        // ...but it must still erode at the normal 1x rate eventually, since
        // the entry persists (adjacency continuation is by design).
        let mut ticks = 0;
        while world.get_voxel(cell) == STONE {
            weathering.tick(&mut world, &[ContactEvent::Settled(above)]);
            ticks += 1;
            assert!(
                ticks < STONE_ERODE_TICKS,
                "must erode near the normal threshold, not never"
            );
        }
        assert_eq!(world.get_voxel(cell), SAND);
        // Total ticks (excluding the seed) is close to STONE_ERODE_TICKS, the
        // 1x rate -- NOT STONE_ERODE_TICKS / 5.
        assert!(
            ticks > STONE_ERODE_TICKS / 2,
            "settled-water erosion must run at ~1x, not the 5x waterfall rate: {ticks}"
        );
    }

    #[test]
    fn mud_dries_back_to_dirt_only_after_water_leaves() {
        let mut world = world_with_floor(MUD);
        let mut weathering = Weathering::new(table());
        let cell = IVec3::new(8, 4, 8);
        let above = cell + IVec3::Y;
        world.set_voxel(above, WATER);

        // Wet mud is untracked and stable.
        weathering.tick(&mut world, &[ContactEvent::Settled(above)]);
        assert_eq!(
            weathering.drying_count(),
            0,
            "wet mud must not be on the drying clock"
        );

        // Water leaves -> drying starts.
        world.set_voxel(above, AIR);
        weathering.tick(&mut world, &[ContactEvent::Vacated(above)]);
        assert_eq!(weathering.drying_count(), 1);
        for _ in 0..(MUD_DRY_TICKS - 2) {
            weathering.tick(&mut world, &[]);
            assert_eq!(world.get_voxel(cell), MUD, "must not dry early");
        }
        weathering.tick(&mut world, &[]);
        assert_eq!(
            world.get_voxel(cell),
            DIRT,
            "dry mud must firm back to dirt"
        );
        assert_eq!(weathering.drying_count(), 0);
    }

    #[test]
    fn returning_water_resets_the_drying_clock() {
        let mut world = world_with_floor(MUD);
        let mut weathering = Weathering::new(table());
        let cell = IVec3::new(8, 4, 8);
        let above = cell + IVec3::Y;
        weathering.tick(&mut world, &[ContactEvent::Vacated(above)]);
        assert_eq!(weathering.drying_count(), 1);
        for _ in 0..(MUD_DRY_TICKS / 2) {
            weathering.tick(&mut world, &[]);
        }
        world.set_voxel(above, WATER); // water returns halfway
        weathering.tick(&mut world, &[ContactEvent::Fell(above)]);
        assert_eq!(
            weathering.drying_count(),
            0,
            "re-wetted mud must leave the drying clock"
        );
        assert_eq!(world.get_voxel(cell), MUD, "and stays mud");
    }

    #[test]
    fn a_fully_weathered_pool_reaches_zero_tracked_cells() {
        // The sleep guarantee, extended: settle a pool on grass, run until the
        // whole shoreline has finished transforming -- both weathering maps
        // must be empty and the fluid asleep. Steady state costs nothing.
        let mut world = world_with_floor(GRASS);
        let mut sim = crate::FluidSim::new(WATER);
        let mut weathering = Weathering::new(table());
        sim.place_blob(&mut world, IVec3::new(8, 7, 8), 1, WATER);
        for _ in 0..((GRASS_SOAK_TICKS + DIRT_SOAK_TICKS) * 3) {
            sim.tick(&mut world);
            let events = sim.drain_events();
            weathering.tick(&mut world, &events);
            for (min, max) in world.drain_dirty_regions() {
                sim.wake_region(&world, min, max);
            }
            if sim.active_count() == 0
                && weathering.soaking_count() == 0
                && weathering.drying_count() == 0
            {
                break;
            }
        }
        assert_eq!(sim.active_count(), 0, "water must sleep");
        assert_eq!(
            weathering.soaking_count(),
            0,
            "soak map must drain to empty"
        );
        assert_eq!(
            weathering.drying_count(),
            0,
            "drying map must drain to empty"
        );
        // And the ground under the pool actually transformed.
        let mut mud_count = 0;
        let (min, max) = world.bounds_voxels();
        for x in min.x..max.x {
            for y in min.y..max.y {
                for z in min.z..max.z {
                    if world.get_voxel(IVec3::new(x, y, z)) == MUD {
                        mud_count += 1;
                    }
                }
            }
        }
        assert!(mud_count > 0, "the pool's bed must have become mud");
    }
    #[test]
    fn mud_adjacent_to_water_dissolves_to_muddy_water() {
        let mut world = world_with_floor(MUD);
        let mut weathering = Weathering::new(table());
        let cell = IVec3::new(8, 4, 8);
        let above = cell + IVec3::Y;
        world.set_voxel(above, WATER);
        world.set_solid_table(vec![false, false, true, true, true, true, true, false]);
        // Seed: one Settled event registers the mud for dissolving
        weathering.tick(&mut world, &[ContactEvent::Settled(above)]);
        assert_eq!(
            weathering.dissolving_count(),
            1,
            "mud adjacent to water must enter dissolving"
        );
        // Not before the threshold
        for _ in 0..(MUD_DISSOLVE_TICKS - 2) {
            weathering.tick(&mut world, &[]);
            assert_eq!(world.get_voxel(cell), MUD, "must not dissolve early");
        }
        weathering.tick(&mut world, &[]);
        // Fluid conservation: mud becomes AIR (consumed), adjacent water
        // becomes muddy_water. No new fluid created.
        assert_eq!(
            world.get_voxel(cell),
            AIR,
            "mud must be consumed (becomes air) at the dissolve threshold"
        );
        assert_eq!(
            world.get_voxel(above),
            MUDDY_WATER,
            "adjacent water must become muddy_water at the dissolve threshold"
        );
    }
    #[test]
    fn clean_water_adjacent_to_muddy_water_becomes_muddy() {
        let mut world = world_with_floor(STONE);
        let mut weathering = Weathering::new(table());
        let water_cell = IVec3::new(7, 5, 8);
        let muddy_cell = IVec3::new(8, 5, 8);
        world.set_voxel(water_cell, WATER);
        world.set_voxel(muddy_cell, MUDDY_WATER);
        world.set_solid_table(vec![false, false, true, true, true, true, true, false]);
        // Seed: Settled on the muddy_water cell registers the clean water for polluting
        weathering.tick(&mut world, &[ContactEvent::Settled(muddy_cell)]);
        assert_eq!(
            weathering.polluting_count(),
            1,
            "clean water adjacent to muddy_water must enter polluting"
        );
        for _ in 0..(POLLUTE_SPREAD_TICKS - 1) {
            weathering.tick(&mut world, &[]);
        }
        assert_eq!(
            world.get_voxel(water_cell),
            MUDDY_WATER,
            "clean water must become muddy_water at the pollute threshold"
        );
    }
    #[test]
    fn still_muddy_water_settles_to_water_and_deposits_sand_below() {
        let mut world = world_with_floor(STONE);
        let mut weathering = Weathering::new(table());
        // test_world fills y=0..5 with Voxel(2) (floor), top at y=5.
        // Place muddy_water at y=5, floor at y=4 is STONE.
        let muddy_cell = IVec3::new(8, 5, 8);
        let floor_cell = IVec3::new(8, 4, 8); // STONE
        world.set_voxel(muddy_cell, MUDDY_WATER);
        world.set_solid_table(vec![false, false, true, true, true, true, true, false]);

        // Seed: Settled event starts the settle clock
        weathering.tick(&mut world, &[ContactEvent::Settled(muddy_cell)]);
        assert_eq!(weathering.settling_count(), 1);

        for _ in 0..(MUDDY_SETTLE_TICKS - 2) {
            weathering.tick(&mut world, &[]);
            assert_eq!(world.get_voxel(muddy_cell), MUDDY_WATER, "must not settle early");
        }
        weathering.tick(&mut world, &[]);
        assert_eq!(
            world.get_voxel(muddy_cell),
            WATER,
            "muddy_water must clarify to water after settling"
        );
        assert_eq!(
            world.get_voxel(floor_cell),
            SAND,
            "sand must be deposited on the floor below"
        );
        assert_eq!(weathering.settling_count(), 0, "settling entry cleared");
    }

    #[test]
    fn moving_muddy_water_does_not_settle() {
        let mut world = world_with_floor(STONE);
        let mut weathering = Weathering::new(table());
        let cell = IVec3::new(8, 5, 8);
        world.set_voxel(cell, MUDDY_WATER);
        world.set_solid_table(vec![false, false, true, true, true, true, true, false]);

        // Start settling
        weathering.tick(&mut world, &[ContactEvent::Settled(cell)]);
        // Then water moves -- Flowed event resets the settle timer
        for _ in 0..(MUDDY_SETTLE_TICKS / 2) {
            weathering.tick(&mut world, &[ContactEvent::Flowed(cell)]);
        }
        assert_eq!(
            world.get_voxel(cell),
            MUDDY_WATER,
            "moving muddy_water must not settle"
        );
        assert_eq!(
            weathering.settling_count(),
            0,
            "Flowed events must remove from settling"
        );
    }
    #[test]
    fn a_fully_settled_polluted_lake_reaches_zero_tracked_cells() {
        // The sleep guarantee, extended to the full pollution lifecycle:
        // mud dissolves → muddy_water, pollution diffuses, muddy_water
        // settles → water + sand, and eventually ALL weathering maps drain
        // to empty and the fluid sleeps. Steady state costs nothing.
        //
        // Geometry: a single mud cell on a stone floor with a muddy_water
        // cell above it, contained by stone walls so nothing flows sideways.
        // The muddy_water is "wet" so the mud dissolves; both muddy_water
        // cells then settle to water. The upper cell clarifies first (it
        // started settling earlier); the lower cell settles before the
        // upper cell's polluting timer reaches its threshold, so polluting
        // drains without completing — the cycle terminates.
        let mut world = world_with_floor(STONE);
        world.set_solid_table(vec![false, false, true, true, true, true, true, false]);
        let mud_cell = IVec3::new(8, 4, 8);
        world.set_voxel(mud_cell, MUD);
        // Stone walls around the fluid cell at y=5 to prevent sideways flow.
        for n in [
            IVec3::new(7, 5, 8),
            IVec3::new(9, 5, 8),
            IVec3::new(8, 5, 7),
            IVec3::new(8, 5, 9),
        ] {
            world.set_voxel(n, STONE);
        }
        let mut sim = crate::FluidSim::with_fluids_and_powders(vec![WATER, MUDDY_WATER], Vec::new());
        let mut weathering = Weathering::new(table());
        // Seed: muddy_water directly above the mud. The muddy_water is "wet"
        // so the mud starts dissolving; the muddy_water itself starts settling.
        world.set_voxel(IVec3::new(8, 5, 8), MUDDY_WATER);
        for (min, max) in world.drain_dirty_regions() {
            sim.wake_region(&world, min, max);
        }
        // Run until everything settles: mud dissolves → muddy_water,
        // muddy_water settles → water + sand, polluting starts then drains,
        // eventually steady state.
        for _ in 0..((MUD_DISSOLVE_TICKS + MUDDY_SETTLE_TICKS + POLLUTE_SPREAD_TICKS) * 3) {
            sim.tick(&mut world);
            let events = sim.drain_events();
            weathering.tick(&mut world, &events);
            for (min, max) in world.drain_dirty_regions() {
                sim.wake_region(&world, min, max);
            }
            if sim.active_count() == 0
                && weathering.soaking_count() == 0
                && weathering.drying_count() == 0
                && weathering.dissolving_count() == 0
                && weathering.polluting_count() == 0
                && weathering.settling_count() == 0
            {
                break;
            }
        }
        assert_eq!(sim.active_count(), 0, "water must sleep");
        assert_eq!(weathering.soaking_count(), 0);
        assert_eq!(weathering.drying_count(), 0);
        assert_eq!(weathering.dissolving_count(), 0);
        assert_eq!(weathering.polluting_count(), 0);
        assert_eq!(weathering.settling_count(), 0);
    }
    #[test]
    fn full_pollution_lifecycle_mud_to_muddy_water_to_water_plus_sand() {
        // Integration test: a single muddy_water cell on a stone floor
        // settles to water + sand, verified end-to-end with the fluid sim.
        // We seed muddy_water directly (bypassing dissolve) to isolate the
        // settle phase, and use a single cell with no water above to avoid
        // the limit cycle (water ↔ muddy_water re-pollution loop).
        //
        // The dissolve phase is tested by mud_adjacent_to_water_dissolves,
        // the polluting phase by clean_water_adjacent_to_muddy_water.
        // This test verifies the settle phase end-to-end with the fluid sim.
        let mut world = world_with_floor(STONE);
        let mut weathering = Weathering::new(table());
        let mut sim = crate::FluidSim::with_fluids_and_powders(
            vec![WATER, MUDDY_WATER],
            Vec::new(),
        );

        // Place a single muddy_water cell at y=5 (one above the stone floor
        // surface at y=4). The floor below (y=4) is STONE.
        let muddy_cell = IVec3::new(8, 5, 8);
        let floor_cell = IVec3::new(8, 4, 8); // STONE (floor surface)
        world.set_voxel(muddy_cell, MUDDY_WATER);
        world.set_solid_table(vec![false, false, true, true, true, true, true, false]);

        // Wake the muddy_water cell so the fluid sim processes it and
        // emits Settled events (needed for the settling timer).
        sim.wake_region(&world, muddy_cell - IVec3::ONE, muddy_cell + IVec3::ONE * 2);

        // Run: the muddy_water cell is supported (stone floor below, stone
        // walls from world_with_floor) so it can't move → emits Settled
        // each tick → settling timer advances → at MUDDY_SETTLE_TICKS,
        // clarifies to WATER and deposits SAND on the cell below.
        let mut settled = false;
        for _ in 0..(MUDDY_SETTLE_TICKS * 2) {
            sim.tick(&mut world);
            let events = sim.drain_events();
            weathering.tick(&mut world, &events);
            for (min, max) in world.drain_dirty_regions() {
                sim.wake_region(&world, min, max);
            }
            if world.get_voxel(muddy_cell) == WATER {
                settled = true;
                break;
            }
        }

        assert!(settled, "muddy_water must settle to water within 2x threshold");
        assert_eq!(
            world.get_voxel(muddy_cell),
            WATER,
            "muddy_water must clarify to water after settling"
        );
        assert_eq!(
            world.get_voxel(floor_cell),
            SAND,
            "sand must be deposited on the floor below the settled muddy_water"
        );
    }
}
