//! Water-driven material transformation, fed by `ContactEvent`s from the
//! fluid tick. Never scans the world: it tracks only cells currently
//! soaking (water-adjacent grass/dirt/stone) or drying (mud that lost its
//! water). Both maps drain to empty at steady state, preserving the
//! settled-water-costs-nothing guarantee. See
//! `docs/plans/2026-07-09-water-refinement-design.md` §3.

use glam::IVec3;
use vox_core::{FxHashMap, FxHashSet};
use vox_world::{Voxel, World};

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
}

pub struct Weathering {
    table: WeatherTable,
    soaking: FxHashMap<IVec3, u32>,
    drying: FxHashMap<IVec3, u32>,
}

impl Weathering {
    pub fn new(table: WeatherTable) -> Self {
        Self { table, soaking: FxHashMap::default(), drying: FxHashMap::default() }
    }

    /// Debug/test stats.
    pub fn soaking_count(&self) -> usize {
        self.soaking.len()
    }
    pub fn drying_count(&self) -> usize {
        self.drying.len()
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
                } else if v == t.grass || v == t.dirt || (v == t.stone && moving) {
                    self.soaking.entry(q).or_insert(0);
                    if fell && v == t.stone {
                        fell_this_tick.insert(q);
                    }
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
            if !NEIGHBORS_6.iter().any(|&n| world.get_voxel(pos + n) == t.water) {
                return false;
            }
            *ticks += if v == t.stone && fell_this_tick.contains(&pos) { STONE_FALL_BOOST } else { 1 };
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

        // 3. Advance drying: mud with water back nearby stops; dry long
        // enough, it firms to dirt.
        let mut dried = Vec::new();
        self.drying.retain(|&pos, ticks| {
            if world.get_voxel(pos) != t.mud {
                return false;
            }
            if NEIGHBORS_6.iter().any(|&n| world.get_voxel(pos + n) == t.water) {
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

    fn table() -> WeatherTable {
        WeatherTable { water: WATER, stone: STONE, grass: GRASS, dirt: DIRT, mud: MUD, sand: SAND }
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
        assert_eq!(world.get_voxel(cell), DIRT, "grass must die to dirt at the soak threshold");
        assert_eq!(weathering.soaking_count(), 1, "the fresh dirt re-registers and keeps soaking");
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
    fn still_water_never_erodes_stone_but_flowing_does_and_falling_is_faster() {
        // Still: Settled event over stone -> no soak entry at all.
        let mut world = world_with_floor(STONE);
        let mut weathering = Weathering::new(table());
        let cell = IVec3::new(8, 4, 8);
        world.set_voxel(cell + IVec3::Y, WATER);
        weathering.tick(&mut world, &[ContactEvent::Settled(cell + IVec3::Y)]);
        assert_eq!(weathering.soaking_count(), 0, "still water must not register stone");

        // Flowing: erodes at STONE_ERODE_TICKS.
        let mut ticks_flowing = 0;
        weathering.tick(&mut world, &[ContactEvent::Flowed(cell + IVec3::Y)]);
        while world.get_voxel(cell) == STONE {
            weathering.tick(&mut world, &[]);
            ticks_flowing += 1;
            assert!(ticks_flowing <= STONE_ERODE_TICKS + 2, "flowing erosion must finish near its threshold");
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
            assert!(ticks_falling <= STONE_ERODE_TICKS / STONE_FALL_BOOST + 2, "continuous waterfall erosion must be ~5x faster");
        }
        assert!(ticks_falling < ticks_flowing / 3, "falling ({ticks_falling}) must be much faster than flowing ({ticks_flowing})");
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
        assert_eq!(weathering.soaking_count(), 0, "no adjacent water -> entry removed");
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
        assert_eq!(weathering.soaking_count(), 1, "Fell must register the stone");

        // Run under still water up to just past the boosted threshold. If the
        // boost were sticky, stone would erode by ~90 ticks. At the 1x rate
        // it must still be stone well past 90.
        for _ in 0..(STONE_ERODE_TICKS / STONE_FALL_BOOST + 10) {
            weathering.tick(&mut world, &[ContactEvent::Settled(above)]);
            assert_eq!(world.get_voxel(cell), STONE,
                "stone must not erode at the 5x rate once water has settled");
        }
        // ...but it must still erode at the normal 1x rate eventually, since
        // the entry persists (adjacency continuation is by design).
        let mut ticks = 0;
        while world.get_voxel(cell) == STONE {
            weathering.tick(&mut world, &[ContactEvent::Settled(above)]);
            ticks += 1;
            assert!(ticks < STONE_ERODE_TICKS, "must erode near the normal threshold, not never");
        }
        assert_eq!(world.get_voxel(cell), SAND);
        // Total ticks (excluding the seed) is close to STONE_ERODE_TICKS, the
        // 1x rate -- NOT STONE_ERODE_TICKS / 5.
        assert!(ticks > STONE_ERODE_TICKS / 2,
            "settled-water erosion must run at ~1x, not the 5x waterfall rate: {ticks}");
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
        assert_eq!(weathering.drying_count(), 0, "wet mud must not be on the drying clock");

        // Water leaves -> drying starts.
        world.set_voxel(above, AIR);
        weathering.tick(&mut world, &[ContactEvent::Vacated(above)]);
        assert_eq!(weathering.drying_count(), 1);
        for _ in 0..(MUD_DRY_TICKS - 2) {
            weathering.tick(&mut world, &[]);
            assert_eq!(world.get_voxel(cell), MUD, "must not dry early");
        }
        weathering.tick(&mut world, &[]);
        assert_eq!(world.get_voxel(cell), DIRT, "dry mud must firm back to dirt");
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
        assert_eq!(weathering.drying_count(), 0, "re-wetted mud must leave the drying clock");
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
            if sim.active_count() == 0 && weathering.soaking_count() == 0 && weathering.drying_count() == 0 {
                break;
            }
        }
        assert_eq!(sim.active_count(), 0, "water must sleep");
        assert_eq!(weathering.soaking_count(), 0, "soak map must drain to empty");
        assert_eq!(weathering.drying_count(), 0, "drying map must drain to empty");
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
}
