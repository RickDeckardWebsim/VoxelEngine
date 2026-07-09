# Water Refinement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Cohesive water flow (8-direction drop-search + momentum memory) and water-driven weathering (grass→dirt→mud, stone→sand erosion, mud drying), per `docs/plans/2026-07-09-water-refinement-design.md`.

**Architecture:** All flow changes live in `crates/vox-sim/src/fluid.rs` (the existing CA). Weathering is a new `crates/vox-sim/src/weathering.rs` module fed by contact events the fluid tick emits — no world scanning, maps empty at steady state so settled water stays free. One new material (`mud`) in `assets/materials/core.toml`; `vox-app` wires the weathering tick after each fluid tick.

**Tech Stack:** Rust workspace; `cargo test -p vox-sim`, `cargo test --workspace`. No new dependencies.

**Invariants that must never break** (existing tests enforce them — run the full vox-sim suite after every task):
- Water-cell count conserved; flat sheets / full basins / leveled columns sleep (`active_count() == 0`); tick budget respected; behavior independent of `voxel_size_m`.

---

### Task 0: Commit the pending flow-rule fix

The working tree already contains the un-committed sand-pile fix (flow rule, `FLOW_HORIZON`, tests) plus tools/README edits.

**Step 1:** Run `cargo test --workspace` — expect all green (33 + 16 + others, 0 failures).

**Step 2:** Commit everything pending:

```bash
git add -A
git commit -m "feat(sim): flow rule -- blocked cells seek a reachable drop, fixes sand-piling"
```

---

### Task 1: 8-direction drop-search

**Files:**
- Modify: `crates/vox-sim/src/fluid.rs` (`flow_dirs`, its call site, doc comments)
- Test: same file, `mod tests`

**Step 1: Write the failing unit test** (tests live in-module and may call `step_cell` directly):

```rust
#[test]
fn flow_finds_a_drop_reachable_only_diagonally() {
    // Water at `pos` rests on water. The four axis neighbors are walled
    // off; the only escape is the diagonal (+1, 0, +1), which has air
    // beneath it. The old 4-direction scan settles here forever.
    let pos = IVec3::new(8, 6, 8);
    let open = [IVec3::new(9, 6, 9), IVec3::new(9, 5, 9)]; // diag + its drop
    let mut is_open = |p: IVec3| open.contains(&p);
    let mut is_supported = |_: IVec3| true;
    let dest = step_cell(pos, &mut is_open, &mut is_supported, false, false, false);
    assert_eq!(
        dest,
        Some(IVec3::new(9, 6, 9)),
        "the scan must step toward a diagonal-only drop"
    );
}
```

**Step 2:** Run `cargo test -p vox-sim flow_finds_a_drop` — expect FAIL (returns `None`).

**Step 3: Implement** — replace `flow_dirs` with an 8-direction version. Axis directions stay first (they are the shorter true distance); diagonals follow. Coins keep de-biasing both groups:

```rust
/// The eight horizontal step directions: the four axis dirs first (shorter
/// true distance), then the four diagonals. `coin` picks the sign order
/// within each group, `coin2` picks which axis/diagonal pair leads -- same
/// de-biasing role the coins already play in the fall/spread rules.
fn flow_dirs(coin: bool, coin2: bool) -> [IVec3; 8] {
    let (x1, x2) = if coin { (IVec3::X, IVec3::NEG_X) } else { (IVec3::NEG_X, IVec3::X) };
    let (z1, z2) = if coin { (IVec3::Z, IVec3::NEG_Z) } else { (IVec3::NEG_Z, IVec3::Z) };
    let (d1, d2) = if coin {
        (IVec3::new(1, 0, 1), IVec3::new(-1, 0, -1))
    } else {
        (IVec3::new(-1, 0, -1), IVec3::new(1, 0, 1))
    };
    let (d3, d4) = if coin {
        (IVec3::new(1, 0, -1), IVec3::new(-1, 0, 1))
    } else {
        (IVec3::new(-1, 0, 1), IVec3::new(1, 0, -1))
    };
    if coin2 { [x1, x2, z1, z2, d1, d2, d3, d4] } else { [z1, z2, x1, x2, d3, d4, d1, d2] }
}
```

No call-site change needed (`for dir in dirs` already iterates whatever the array yields). Update the flow-rule comment ("each horizontal direction" → "each of the eight horizontal directions") and the `FLOW_HORIZON` doc (`O(8 * horizon)`).

**Step 4:** Run `cargo test -p vox-sim` — all 17 tests pass (16 existing + new one). The sleep tests passing proves diagonals didn't reintroduce random-walking.

**Step 5:** Commit: `git add -A && git commit -m "feat(sim): flow scan covers all 8 horizontal directions, removes off-axis piles"`

---

### Task 2: Momentum memory

**Files:**
- Modify: `crates/vox-sim/src/fluid.rs` (`FluidSim` struct, `tick`, `step_cell` signature)
- Test: same file

**Step 1: Write the failing tests:**

```rust
#[test]
fn step_cell_prefers_the_momentum_direction_on_equal_drops() {
    // Drops exist both at +X and -X, two cells out. With no momentum the
    // coin decides; with momentum -X the cell must step -X regardless of
    // what the coins say.
    let pos = IVec3::new(8, 6, 8);
    let open = [
        IVec3::new(9, 6, 8), IVec3::new(10, 6, 8), IVec3::new(10, 5, 8), // +X run
        IVec3::new(7, 6, 8), IVec3::new(6, 6, 8), IVec3::new(6, 5, 8),   // -X run
    ];
    for coins in [(false, false), (true, false), (false, true), (true, true)] {
        let mut is_open = |p: IVec3| open.contains(&p);
        let mut is_supported = |_: IVec3| true;
        let dest = step_cell_with_momentum(
            pos, &mut is_open, &mut is_supported, false, coins.0, coins.1,
            Some(IVec3::NEG_X),
        );
        assert_eq!(dest, Some(IVec3::new(7, 6, 8)), "momentum -X must win for coins {coins:?}");
    }
}

#[test]
fn momentum_is_forgotten_once_water_settles() {
    let mut world = test_world();
    let mut sim = FluidSim::new(WATER);
    sim.place_blob(&mut world, IVec3::new(8, 8, 8), 2, WATER);
    for _ in 0..400 {
        sim.tick(&mut world);
        if sim.active_count() == 0 {
            break;
        }
    }
    assert_eq!(sim.active_count(), 0, "blob must settle");
    assert_eq!(sim.momentum_count(), 0, "settled water must carry no momentum state");
}
```

**Step 2:** `cargo test -p vox-sim momentum` — FAIL to compile (`step_cell_with_momentum`, `momentum_count` missing).

**Step 3: Implement.**

1. `FluidSim` gains a field and a stat accessor:

```rust
    /// Horizontal direction of each *active* cell's last move (plus what
    /// woken neighbors inherited). Consulted first in `step_cell` so a
    /// draining current stays coherent instead of re-randomizing each
    /// tick. Rebuilt every tick alongside the active set -- empty whenever
    /// the water sleeps, so settled cost is still zero.
    momentum: FxHashMap<IVec3, IVec3>,
```

```rust
    /// Number of cells carrying momentum (debug-overlay stat / tests).
    pub fn momentum_count(&self) -> usize {
        self.momentum.len()
    }
```

Initialize `momentum: FxHashMap::default()` in `new`.

2. Rename `step_cell` to `step_cell_with_momentum` taking one extra final parameter `momentum: Option<IVec3>`, and keep a thin `step_cell(...)` wrapper passing `None` (existing unit tests keep compiling untouched). Inside, momentum is consulted first at each stage — full replacement for the function body's decision points:

```rust
fn step_cell_with_momentum(
    pos: IVec3,
    is_open: &mut impl FnMut(IVec3) -> bool,
    is_supported: &mut impl FnMut(IVec3) -> bool,
    has_water_above: bool,
    coin: bool,
    coin2: bool,
    momentum: Option<IVec3>,
) -> Option<IVec3> {
    let down = pos + IVec3::NEG_Y;
    if is_open(down) {
        return Some(down);
    }

    // Diagonal fall: momentum's axis components first, then the coin order.
    let m = momentum.unwrap_or(IVec3::ZERO);
    let (dx1, dx2) = if m.x != 0 { (m.x, -m.x) } else if coin { (1, -1) } else { (-1, 1) };
    for dx in [dx1, dx2] {
        let diag = pos + IVec3::new(dx, -1, 0);
        if is_open(diag) {
            return Some(diag);
        }
    }
    let (dz1, dz2) = if m.z != 0 { (m.z, -m.z) } else if coin2 { (1, -1) } else { (-1, 1) };
    for dz in [dz1, dz2] {
        let diag = pos + IVec3::new(0, -1, dz);
        if is_open(diag) {
            return Some(diag);
        }
    }

    // Flow: momentum direction scanned first, then the remaining seven.
    let dirs = flow_dirs(coin, coin2);
    let ordered = momentum.into_iter().chain(dirs.into_iter().filter(|&d| Some(d) != momentum));
    for dir in ordered {
        for k in 1..=FLOW_HORIZON {
            let q = pos + dir * k;
            if !is_open(q) {
                break;
            }
            if is_open(q + IVec3::NEG_Y) {
                return Some(pos + dir);
            }
        }
    }

    if !has_water_above {
        return None;
    }
    let (sx1, sx2) = if m.x != 0 { (m.x, -m.x) } else if coin { (1, -1) } else { (-1, 1) };
    for dx in [sx1, sx2] {
        let side = pos + IVec3::new(dx, 0, 0);
        if is_open(side) && is_supported(side + IVec3::NEG_Y) {
            return Some(side);
        }
    }
    let (sz1, sz2) = if m.z != 0 { (m.z, -m.z) } else if coin2 { (1, -1) } else { (-1, 1) };
    for dz in [sz1, sz2] {
        let side = pos + IVec3::new(0, 0, dz);
        if is_open(side) && is_supported(side + IVec3::NEG_Y) {
            return Some(side);
        }
    }
    None
}

/// `step_cell_with_momentum` with no momentum -- the coin-only behavior.
fn step_cell(
    pos: IVec3,
    is_open: &mut impl FnMut(IVec3) -> bool,
    is_supported: &mut impl FnMut(IVec3) -> bool,
    has_water_above: bool,
    coin: bool,
    coin2: bool,
) -> Option<IVec3> {
    step_cell_with_momentum(pos, is_open, is_supported, has_water_above, coin, coin2, None)
}
```

Note: momentum may be a diagonal like `(1, 0, 1)`; its `.x`/`.z` components bias the axis-pair choices, and the full vector leads the flow scan.

3. In `tick`, build `next_momentum` alongside `next_active`:
   - At the top: `let mut next_momentum = FxHashMap::default();`
   - Budget overflow carry: after `next_active.extend(overflow)` also copy each overflow position's existing momentum entry into `next_momentum`.
   - Pass `self.momentum.get(&pos).copied()` as the new `step_cell_with_momentum` argument.
   - On a move:

```rust
                let hdir = IVec3::new(dest.x - pos.x, 0, dest.z - pos.z);
                // A pure vertical fall keeps whatever direction the cell
                // already had; any horizontal component overwrites it.
                let carried = if hdir != IVec3::ZERO { Some(hdir) } else { self.momentum.get(&pos).copied() };
                if let Some(d) = carried {
                    next_momentum.insert(dest, d);
                }
```

   and inside the existing wake loop, after `next_active.insert(neighbor)`:

```rust
                            if let Some(d) = carried {
                                next_momentum.entry(neighbor).or_insert(d);
                            }
```

   - At the end: `self.momentum = next_momentum;` (right before `self.active = next_active;`). Settled cells never get an entry — momentum dies with settling by construction.

**Step 4:** `cargo test -p vox-sim` — all 19 pass. Pay special attention to `spread_pattern_is_identical_in_cell_counts_at_any_voxel_scale` (momentum is coordinate-based, still scale-free — must stay green).

**Step 5:** Commit: `git commit -am "feat(sim): momentum memory -- draining water flows as a coherent current"`

---

### Task 3: Contact events from the fluid tick

**Files:**
- Modify: `crates/vox-sim/src/fluid.rs`, `crates/vox-sim/src/lib.rs`
- Test: `fluid.rs` tests

**Step 1: Failing test:**

```rust
#[test]
fn tick_emits_fell_vacated_and_settled_events() {
    let mut world = test_world();
    let mut sim = FluidSim::new(WATER);
    sim.place_blob(&mut world, IVec3::new(8, 7, 8), 0, WATER); // 2 above floor
    sim.tick(&mut world);
    let ev = sim.drain_events();
    assert!(ev.contains(&ContactEvent::Vacated(IVec3::new(8, 7, 8))), "leaving a cell must emit Vacated: {ev:?}");
    assert!(ev.contains(&ContactEvent::Fell(IVec3::new(8, 6, 8))), "a downward arrival must emit Fell: {ev:?}");

    sim.tick(&mut world); // lands on floor -> second Fell
    sim.drain_events();
    sim.tick(&mut world); // nowhere to go -> settles
    let ev = sim.drain_events();
    assert!(ev.contains(&ContactEvent::Settled(IVec3::new(8, 5, 8))), "settling must emit Settled: {ev:?}");
}
```

**Step 2:** `cargo test -p vox-sim tick_emits` — FAIL to compile.

**Step 3: Implement.** In `fluid.rs`:

```rust
/// What the fluid tick observed about a cell -- consumed by weathering
/// (`drain_events`), which uses arrival mode to grade erosion. Bounded:
/// the buffer is cleared at the start of every tick, so an app that never
/// drains holds at most one tick's worth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContactEvent {
    /// Water arrived here by falling (straight or diagonal down).
    Fell(IVec3),
    /// Water arrived here by a horizontal move (flow or spread).
    Flowed(IVec3),
    /// Water here found no move and left the active set.
    Settled(IVec3),
    /// Water left this cell (every move emits one).
    Vacated(IVec3),
}
```

`FluidSim` gains `events: Vec<ContactEvent>` (init empty) and:

```rust
    /// Take this tick's contact events (empties the buffer).
    pub fn drain_events(&mut self) -> Vec<ContactEvent> {
        std::mem::take(&mut self.events)
    }
```

In `tick`: first line `self.events.clear();`. In the move branch:

```rust
                self.events.push(ContactEvent::Vacated(pos));
                self.events.push(if dest.y < pos.y {
                    ContactEvent::Fell(dest)
                } else {
                    ContactEvent::Flowed(dest)
                });
```

In the settle branch (currently a bare comment), push `ContactEvent::Settled(pos)`.

In `lib.rs`: `pub use fluid::{ContactEvent, FluidSim};`

**Step 4:** `cargo test -p vox-sim` — all 20 pass.

**Step 5:** Commit: `git commit -am "feat(sim): fluid tick emits contact events for weathering"`

---

### Task 4: Weathering — soak transformations

**Files:**
- Create: `crates/vox-sim/src/weathering.rs`
- Modify: `crates/vox-sim/src/lib.rs`

**Step 1: Failing tests** (in `weathering.rs`'s own `mod tests`; build worlds exactly like `fluid.rs::tests::test_world`, with a solid table `[air, water, stone, grass, dirt, mud, sand]` → ids 1..=6):

```rust
use super::*;
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

    // Falling: a second stone cell erodes ~5x sooner.
    let mut world = world_with_floor(STONE);
    let mut weathering = Weathering::new(table());
    world.set_voxel(cell + IVec3::Y, WATER);
    let mut ticks_falling = 0;
    weathering.tick(&mut world, &[ContactEvent::Fell(cell + IVec3::Y)]);
    while world.get_voxel(cell) == STONE {
        weathering.tick(&mut world, &[]);
        ticks_falling += 1;
        assert!(ticks_falling <= STONE_ERODE_TICKS / STONE_FALL_BOOST + 2, "waterfall erosion must be ~5x faster");
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
```

**Step 2:** `cargo test -p vox-sim weathering` — FAIL to compile.

**Step 3: Implement `weathering.rs`:**

```rust
//! Water-driven material transformation, fed by `ContactEvent`s from the
//! fluid tick. Never scans the world: it tracks only cells currently
//! soaking (water-adjacent grass/dirt/stone) or drying (mud that lost its
//! water). Both maps drain to empty at steady state, preserving the
//! settled-water-costs-nothing guarantee. See
//! `docs/plans/2026-07-09-water-refinement-design.md` §3.

use glam::IVec3;
use vox_core::FxHashMap;
use vox_world::{Voxel, World};

use crate::fluid::ContactEvent;

/// Soak ticks (at the fluid tick rate, ~15 Hz) before grass dies to dirt.
pub const GRASS_SOAK_TICKS: u32 = 45; // ~3 s
/// Soak ticks before dirt turns to mud.
pub const DIRT_SOAK_TICKS: u32 = 105; // ~7 s
/// Soak ticks of *flowing* contact before stone erodes to sand.
pub const STONE_ERODE_TICKS: u32 = 450; // ~30 s
/// Waterfall multiplier: stone touched by *falling* water accrues this many
/// soak ticks per tick.
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

#[derive(Clone, Copy)]
struct Soak {
    ticks: u32,
    /// Stone touched by falling water accrues `STONE_FALL_BOOST` per tick.
    fall_boost: bool,
}

pub struct Weathering {
    table: WeatherTable,
    soaking: FxHashMap<IVec3, Soak>,
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

    fn water_adjacent(&self, world: &World, pos: IVec3) -> bool {
        NEIGHBORS_6.iter().any(|&n| world.get_voxel(pos + n) == self.table.water)
    }

    pub fn tick(&mut self, world: &mut World, events: &[ContactEvent]) {
        let t = self.table;

        // 1. Register: water contact puts transformable neighbors on the
        // soak clock; any contact re-wets mud (cancels drying). Stone only
        // registers for *moving* water -- a settled lake never eats its
        // basin.
        for &ev in events {
            let (pos, moving, fell) = match ev {
                ContactEvent::Fell(p) => (p, true, true),
                ContactEvent::Flowed(p) => (p, true, false),
                ContactEvent::Settled(p) => (p, false, false),
                ContactEvent::Vacated(p) => {
                    // 3. Mud that just lost a water neighbor starts drying.
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
                    let entry = self.soaking.entry(q).or_insert(Soak { ticks: 0, fall_boost: false });
                    entry.fall_boost |= fell && v == t.stone;
                }
            }
        }

        // 2. Advance soaking. Entries whose water left, or whose material
        // changed under them (blasted, dug), simply drop out.
        let mut converted = Vec::new();
        self.soaking.retain(|&pos, soak| {
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
            soak.ticks += if v == t.stone && soak.fall_boost { STONE_FALL_BOOST } else { 1 };
            if soak.ticks >= threshold {
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
                self.soaking.insert(pos, Soak { ticks: 0, fall_boost: false });
            }
        }

        // 4. Advance drying: mud with water back nearby stops; dry long
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
```

Note: `world.get_voxel` inside `retain` closures requires `world` be borrowed immutably there while `self.soaking` is borrowed mutably — this is fine (disjoint fields), but the closure captures `t` by copy to avoid borrowing `self`. If the borrow checker objects to `world` use alongside `retain`, collect keys first — but `retain` only borrows the map, `world` is a separate binding, so it compiles as written.

In `lib.rs`:

```rust
mod fluid;
mod weathering;

pub use fluid::{ContactEvent, FluidSim};
pub use weathering::{
    DIRT_SOAK_TICKS, GRASS_SOAK_TICKS, MUD_DRY_TICKS, STONE_ERODE_TICKS, STONE_FALL_BOOST,
    WeatherTable, Weathering,
};
```

(Check `vox_core::FxHashMap` exists — `FxHashSet` does, from `fluid.rs`; if the map alias is missing, add it beside the set alias in `vox-core/src/fxhash.rs`.)

**Step 4:** `cargo test -p vox-sim` — all pass (20 fluid + 4 weathering).

**Step 5:** Commit: `git commit -am "feat(sim): weathering -- grass dies, dirt soaks to mud, flowing water erodes stone"`

---

### Task 5: Weathering — drying + sleep guarantee

**Files:**
- Modify: `crates/vox-sim/src/weathering.rs` (tests only — logic landed in Task 4)

**Step 1: Failing tests:**

```rust
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
    for _ in 0..MUD_DRY_TICKS {
        weathering.tick(&mut world, &[]);
    }
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
```

**Step 2:** `cargo test -p vox-sim weathering` — the first two should PASS already (logic exists); the pool test may expose real integration bugs (conversion-under-water wake loops, events during settling). Fix whatever it exposes — that is this task's purpose. One known wrinkle: weathering's `set_voxel` calls mark dirty regions that wake the fluid; the test drains them like the app will, proving the wake→re-settle loop converges.

**Step 3:** `cargo test -p vox-sim` — all pass.

**Step 4:** Commit: `git commit -am "test(sim): mud drying cycle and the extended sleep guarantee"`

---

### Task 6: `mud` material

**Files:**
- Modify: `assets/materials/core.toml`
- Test: `crates/vox-core/src/material.rs` (`loads_shipped_core_materials` — check what it asserts; extend to include mud)

**Step 1:** Add to `core.toml` (after `sand`, order defines ids — appending after `planks`/`water` at the END is safest so existing ids don't shift... **IMPORTANT:** ids are assigned in declaration order, and worldgen/tests may bake ids. Append `mud` at the very end, after `water`):

```toml
[[material]]
name = "mud"
color = [0.30, 0.22, 0.16]
jitter = 0.05
density = 1700.0
strength = 1.0
```

**Step 2:** Extend the shipped-materials test to assert `mud` loads with the expected id/solidity, then run `cargo test -p vox-core` — expect PASS.

**Step 3:** `cargo test --workspace` — nothing else may depend on the material count; fix any test that hard-codes it.

**Step 4:** Commit: `git commit -am "feat(assets): mud material for water-soaked dirt"`

---

### Task 7: vox-app integration

**Files:**
- Modify: `crates/vox-app/src/main.rs` (field, constructor, frame loop at ~893-896)
- Test: `crates/vox-app/src/tools.rs` (integration test)

**Step 1: Failing integration test** (in `tools.rs`'s test module, alongside `digging_into_a_settled_lake_lets_it_drain`, reusing its registry/world helpers):

```rust
#[test]
fn a_pool_on_grass_turns_its_bed_to_mud() {
    // End-to-end through the real registry: place water on a grass field,
    // run the fluid + weathering loop the way the frame loop does, and the
    // grass beneath must progress grass -> dirt -> mud.
    // (build world with grass top layer via registry ids, place blob,
    // loop: fluid.tick -> drain_events -> weathering.tick -> drain dirty
    // regions -> wake; assert mud exists under the pool within
    // (GRASS_SOAK_TICKS + DIRT_SOAK_TICKS + margin) ticks.)
}
```

Write it concretely against the existing test fixtures in `tools.rs` (they build a `MaterialRegistry` from the shipped assets — grass/dirt/mud/stone/sand ids all resolve by name).

**Step 2:** Run it — FAIL (no weathering wiring exists; the test drives `Weathering` directly, so it may actually pass without touching `main.rs`. If so, it pins the contract and the remaining work is pure wiring).

**Step 3: Wire `main.rs`:**

- Helper beside `water_material`:

```rust
/// Weathering material table, or `None` (weathering disabled) if any
/// required material is missing from the asset set -- mirrors
/// `water_material`'s graceful-fallback pattern.
fn weather_table(registry: &MaterialRegistry) -> Option<vox_sim::WeatherTable> {
    let id = |name: &str| registry.id_by_name(name).map(|m| Voxel(m.0));
    Some(vox_sim::WeatherTable {
        water: id("water")?,
        stone: id("stone")?,
        grass: id("grass")?,
        dirt: id("dirt")?,
        mud: id("mud")?,
        sand: id("sand")?,
    })
}
```

- Field: `weathering: Option<vox_sim::Weathering>,` — constructor: `weathering: weather_table(&registry).map(vox_sim::Weathering::new),`
- Frame loop (replace lines ~893-896):

```rust
        let fluid_timing = self.fluid_clock.advance(timing.dt_frame);
        for _ in 0..fluid_timing.physics_steps {
            self.fluid.tick(&mut self.world);
            if let Some(w) = &mut self.weathering {
                let events = self.fluid.drain_events();
                w.tick(&mut self.world, &events);
            }
        }
```

**Step 4:** `cargo test --workspace` — everything green, including the untouched `digging_into_a_settled_lake_lets_it_drain`.

**Step 5:** Commit: `git commit -am "feat(app): wire weathering into the fluid tick loop"`

---

### Task 8: Verify end-to-end + docs

**Step 1:** `cargo test --workspace` — full green.

**Step 2:** Run the app (`cargo run --release`, or the project's run skill) and manually verify: place water on grass → bed turns dark (mud) over ~10 s; carve a slope from a pool → water streams down it coherently; a waterfall onto stone → sand appears in ~6 s; drained shoreline mud lightens back to dirt after ~20 s.

**Step 3:** Update `README.md` roadmap line for the fluid sim (mention weathering + momentum), and mark the design doc **Status: Implemented**.

**Step 4:** Commit: `git commit -am "docs: water refinement implemented -- momentum flow + weathering"`
