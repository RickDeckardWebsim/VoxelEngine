# Water Pollution (Muddy Water) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Add a `muddy_water` fluid material and a suspension lifecycle where mud dissolves into water (polluting it), pollution diffuses by contact, and still muddy water settles out — clarifying to clean water and depositing sand below.

**Architecture:** The fluid sim generalizes from a single `water: Voxel` to `fluids: Vec<Voxel>`. Every hardcoded `== water` / `Voxel(9)` / `mat_id == 9u` across sim, weathering, fire, physics, meshing, and the shader is generalized to a fluid-set check. The weathering module gains three new contact-timer maps (dissolving, polluting, settling) that drain to empty at steady state — preserving the "settled water costs nothing" sleep guarantee. All changes are behavior-preserving for single-fluid worlds (existing tests pass unmodified).

**Tech Stack:** Rust, wgpu/WGSL, rayon, glam, FxHashMap/FxHashSet

**Design doc:** `docs/plans/2026-07-10-water-pollution-design.md`

---

## Phase 1: Generalize FluidSim to multiple fluids (behavior-preserving)

This phase changes `FluidSim` from `water: Voxel` to `fluids: Vec<Voxel>` and generalizes all 5 internal `== water` checks. No new behavior — when `fluids == [water]`, everything works identically. All existing fluid tests must pass unmodified.

### Task 1: Change FluidSim struct and constructor

**Files:**
- Modify: `crates/vox-sim/src/fluid.rs:45-88` (struct + constructors)
- Modify: `crates/vox-sim/src/lib.rs:24` (re-exports if needed)

**Step 1: Write the failing test**

Add to `crates/vox-sim/src/fluid.rs` test module (after line 1255):

```rust
#[test]
fn sim_with_multiple_fluids_treats_both_as_fluid() {
    const MUDDY: Voxel = Voxel(3);
    let mut world = test_world();
    // Register muddy as solid=false in the solid table
    world.set_solid_table(vec![false, false, true, false]);
    let mut sim = FluidSim::with_fluids_and_powders(vec![WATER, MUDDY], Vec::new());
    // Place a muddy_water cell above the floor — it must fall like water
    sim.place_blob(&mut world, IVec3::new(8, 10, 8), 0, MUDDY);
    for _ in 0..5 {
        sim.tick(&mut world);
    }
    assert_eq!(
        world.get_voxel(IVec3::new(8, 10, 8)),
        AIR,
        "muddy_water must fall under gravity like water"
    );
    assert_eq!(
        world.get_voxel(IVec3::new(8, 5, 8)),
        MUDDY,
        "muddy_water must land on the floor"
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p vox-sim sim_with_multiple_fluids -- --nocapture`
Expected: FAIL — `with_fluids_and_powders` method doesn't exist

**Step 3: Write minimal implementation**

In `crates/vox-sim/src/fluid.rs`, change the `FluidSim` struct (lines 45-68):

```rust
pub struct FluidSim {
    active: FxHashSet<IVec3>,
    momentum: FxHashMap<IVec3, IVec3>,
    /// All voxel materials this sim treats as fluids (water, muddy_water, ...).
    /// Each flows with the full CA rule. Set once at construction — never
    /// inferred from the active set. A single entry reproduces the original
    /// single-water behavior exactly.
    fluids: Vec<Voxel>,
    /// Materials this sim treats as powders.
    powders: Vec<Voxel>,
    rng: u64,
    events: Vec<ContactEvent>,
}
```

Add an `is_fluid` method and update constructors:

```rust
/// Whether `v` is a fluid material this sim handles.
fn is_fluid(&self, v: Voxel) -> bool {
    self.fluids.contains(&v)
}

/// Create a sim for a single fluid (water). Equivalent to
/// `with_fluids_and_powders(vec![water], Vec::new())`.
pub fn new(water: Voxel) -> Self {
    Self::with_fluids_and_powders(vec![water], Vec::new())
}

/// Create a sim that also handles the given powder materials.
pub fn with_powders(water: Voxel, powders: Vec<Voxel>) -> Self {
    Self::with_fluids_and_powders(vec![water], powders)
}

/// Create a sim handling multiple fluid materials and powder materials.
/// Each fluid flows with the full CA rule; each powder falls and piles.
pub fn with_fluids_and_powders(fluids: Vec<Voxel>, powders: Vec<Voxel>) -> Self {
    Self {
        active: FxHashSet::default(),
        momentum: FxHashMap::default(),
        fluids,
        powders,
        rng: 0x9E37_79B9_7F4A_7C15,
        events: Vec::new(),
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p vox-sim sim_with_multiple_fluids -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/vox-sim/src/fluid.rs
git commit -m "feat(sim): generalize FluidSim to multiple fluids"
```

---

### Task 2: Generalize is_simmed and tick loop checks

**Files:**
- Modify: `crates/vox-sim/src/fluid.rs:96-98` (`is_simmed`)
- Modify: `crates/vox-sim/src/fluid.rs:163-279` (`tick` method — 4 spots: `:165`, `:208`, `:210`, `:257`)

**Step 1: Write the failing test**

Add to fluid.rs test module:

```rust
#[test]
fn muddy_water_levels_across_a_muddy_water_puddle() {
    const MUDDY: Voxel = Voxel(3);
    let mut world = test_world();
    world.set_solid_table(vec![false, false, true, false]);
    let mut sim = FluidSim::with_fluids_and_powders(vec![WATER, MUDDY], Vec::new());

    // Build a 2-deep muddy_water puddle on the floor, with a 1-deep section:
    //   y=6: M M M . .    (column 3,4 are 1-deep — only y=5 has floor)
    //   y=5: M M M M M    (floor at y=5, all cells rest on it)
    // Place a tall column at x=6 that should level sideways into the shallow
    // section — requires is_supported and has_water_above to recognize muddy.
    for x in 6..=8 {
        world.set_voxel(IVec3::new(x, 5, 8), MUDDY);
        world.set_voxel(IVec3::new(x, 6, 8), MUDDY);
    }
    // Shallow section (only y=5)
    world.set_voxel(IVec3::new(9, 5, 8), MUDDY);
    world.set_voxel(IVec3::new(10, 5, 8), MUDDY);
    // Extra column on top of the deep section
    world.set_voxel(IVec3::new(7, 7, 8), MUDDY);

    // Wake all muddy cells
    sim.wake_region(&world, IVec3::new(5, 4, 7), IVec3::new(11, 9, 9));

    for _ in 0..100 {
        sim.tick(&mut world);
        if sim.active_count() == 0 {
            break;
        }
    }

    // The extra cell should have leveled — the column at x=7 y=7 should be
    // gone (moved to a shallow section), proving muddy_water recognized
    // muddy_water below as support and muddy_water above as has_water_above.
    assert_ne!(
        world.get_voxel(IVec3::new(7, 7, 8)),
        MUDDY,
        "muddy_water must level like water, not settle like a powder"
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p vox-sim muddy_water_levels -- --nocapture`
Expected: FAIL — muddy_water can't level because `is_supported` and `has_water_above` only check `== water`

**Step 3: Write minimal implementation**

Update `is_simmed` (line 96-98):
```rust
fn is_simmed(&self, v: Voxel) -> bool {
    self.is_fluid(v) || self.powders.contains(&v)
}
```

Update `tick` method (lines 163-279). Replace `let water = self.water;` (line 165) — instead, capture the fluid set for use in closures. The key changes:

- Line 208: `is_supported` closure — change `world.get_voxel(p) == water` to `self.is_fluid(world.get_voxel(p))`
- Line 210: `has_water_above` — change `world.get_voxel(pos + IVec3::Y) == water` to `self.is_fluid(world.get_voxel(pos + IVec3::Y))`
- Line 257: momentum recruit — change `nv == water` to `self.is_fluid(nv)`

Since `is_fluid` borrows `self`, and the closures in `tick` also need `world`, you'll need to extract the fluid set as a local `Vec` or use a closure that captures a reference. The cleanest approach: since `is_fluid` just checks `self.fluids.contains(&v)`, capture the fluids vec:

```rust
pub fn tick(&mut self, world: &mut World) -> usize {
    self.events.clear();
    let fluids = self.fluids.clone(); // small vec (1-2 entries)
    let is_fluid = |v: Voxel| fluids.contains(&v);
    // ...
    // Line 208: is_supported uses is_fluid
    // Line 210: has_water_above uses is_fluid
    // Line 257: momentum recruit uses is_fluid
}
```

**Step 4: Run ALL fluid tests to verify nothing breaks**

Run: `cargo test -p vox-sim -- --nocapture`
Expected: ALL PASS (existing tests + new test)

**Step 5: Commit**

```bash
git add crates/vox-sim/src/fluid.rs
git commit -m "feat(sim): generalize tick loop to is_fluid checks"
```

---

### Task 3: Update place_blob and wake_region (already use is_simmed — verify)

**Files:**
- Verify: `crates/vox-sim/src/fluid.rs:129-155` (`place_blob`)
- Verify: `crates/vox-sim/src/fluid.rs:287-301` (`wake_region`)

`place_blob` takes a `water_material: Voxel` parameter (the caller chooses which fluid to place) and uses `is_simmed` via `self.active.extend(filled)` — no `== water` checks. `wake_region` uses `is_simmed`. Both should already work after Task 2.

**Step 1: Write a test that places muddy_water via place_blob and wakes it**

Already covered by Task 1 and Task 2 tests. Run all tests:

Run: `cargo test -p vox-sim -- --nocapture`
Expected: ALL PASS

**Step 2: Commit if any fix was needed (otherwise skip)**

---

## Phase 2: Generalize WeatherTable and existing weathering checks (behavior-preserving)

### Task 4: Add muddy_water to WeatherTable + is_wet helper

**Files:**
- Modify: `crates/vox-sim/src/weathering.rs:38-46` (`WeatherTable` struct)
- Modify: `crates/vox-sim/src/weathering.rs:187-210` (test `table()` helper)

**Step 1: Write the failing test**

Add to weathering.rs test module. First, update the test `table()` helper (line 201) to include `muddy_water`:

```rust
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
```

Then add the test:

```rust
#[test]
fn mud_under_muddy_water_still_dissolves_to_dirt() {
    // The is_wet generalization: mud under muddy_water must still soak
    // toward mud (via the grass->dirt->mud chain). This tests that the
    // soak water-adjacency check recognizes muddy_water as "wet".
    let mut world = world_with_floor(DIRT);
    let mut weathering = Weathering::new(table());
    let cell = IVec3::new(8, 4, 8);
    world.set_voxel(cell + IVec3::Y, MUDDY_WATER); // muddy water on top
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
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p vox-sim mud_under_muddy_water -- --nocapture`
Expected: FAIL — `WeatherTable` has no `muddy_water` field

**Step 3: Write minimal implementation**

Add `muddy_water` field to `WeatherTable` (line 38-46):

```rust
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
```

Add an `is_wet` helper method to `WeatherTable`:

```rust
impl WeatherTable {
    /// Whether `v` is a water-like fluid (water or muddy_water).
    #[inline]
    pub fn is_wet(&self, v: Voxel) -> bool {
        v == self.water || v == self.muddy_water
    }
}
```

Update the two existing water-adjacency checks in `tick`:
- Line 130: `world.get_voxel(pos + n) == t.water` → `t.is_wet(world.get_voxel(pos + n))`
- Line 170: `world.get_voxel(pos + n) == t.water` → `t.is_wet(world.get_voxel(pos + n))`

**Step 4: Run ALL weathering tests**

Run: `cargo test -p vox-sim weathering -- --nocapture`
Expected: ALL PASS (existing + new)

**Step 5: Commit**

```bash
git add crates/vox-sim/src/weathering.rs
git commit -m "feat(weathering): add muddy_water to WeatherTable, generalize is_wet"
```

---

### Task 5: Update main.rs weather_table to resolve muddy_water

**Files:**
- Modify: `crates/vox-app/src/main.rs:112-122` (`weather_table` function)

**Step 1: Update weather_table to include muddy_water**

```rust
fn weather_table(registry: &MaterialRegistry) -> Option<vox_sim::WeatherTable> {
    let id = |name: &str| registry.id_by_name(name).map(|m| Voxel(m.0));
    Some(vox_sim::WeatherTable {
        water: id("water")?,
        stone: id("stone")?,
        grass: id("grass")?,
        dirt: id("dirt")?,
        mud: id("mud")?,
        sand: id("sand")?,
        muddy_water: id("muddy_water")?,
    })
}
```

**Step 2: Verify compilation**

Run: `cargo check -p vox-app`
Expected: This will fail until the `muddy_water` material exists in assets (Task 9). That's expected — the `?` makes weathering disable gracefully. But the struct field must match.

Actually, this won't compile until the `WeatherTable` struct has the `muddy_water` field (added in Task 4) AND the material exists in assets. Since Task 4 added the field, this compiles. The `id("muddy_water")?` returns `None` if the material doesn't exist yet, disabling weathering. We'll add the material in Task 9.

Run: `cargo check -p vox-app`
Expected: PASS (compiles; weathering disabled at runtime since muddy_water material doesn't exist yet)

**Step 3: Commit**

```bash
git add crates/vox-app/src/main.rs
git commit -m "feat(app): resolve muddy_water in weather_table"
```

---

## Phase 3: Add muddy_water material and pollution weathering logic (TDD)

### Task 6: Add muddy_water material to assets

**Files:**
- Modify: `assets/materials/core.toml` (append after dark_ash)

**Step 1: Add the material definition**

Append to `assets/materials/core.toml`:

```toml
[[material]]
name = "muddy_water"
color = [0.28, 0.24, 0.18]
jitter = 0.04
density = 1100.0
strength = 0.0
solid = false
fluid = true
```

**Step 2: Verify it loads**

Run: `cargo test -p vox-core material -- --nocapture`
Expected: PASS (existing material loading tests should still pass)

**Step 3: Verify weather_table now resolves it**

Run: `cargo run -p vox-app --release -- --help`
Expected: Runs without "weathering disabled" log (muddy_water now found in assets)

**Step 4: Commit**

```bash
git add assets/materials/core.toml
git commit -m "feat(assets): add muddy_water material"
```

---

### Task 7: Add dissolving logic (mud → muddy_water)

**Files:**
- Modify: `crates/vox-sim/src/weathering.rs` (struct, tick, constants, lib.rs re-exports)

**Step 1: Write the failing test**

Add to weathering.rs test module:

```rust
#[test]
fn mud_adjacent_to_water_dissolves_to_muddy_water() {
    let mut world = world_with_floor(MUD);
    let mut weathering = Weathering::new(table());
    let cell = IVec3::new(8, 4, 8);
    let above = cell + IVec3::Y;
    world.set_voxel(above, WATER);
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
    assert_eq!(
        world.get_voxel(cell),
        MUDDY_WATER,
        "mud must dissolve to muddy_water at the threshold"
    );
    assert_eq!(
        weathering.settling_count(),
        1,
        "fresh muddy_water enters settling"
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p vox-sim mud_adjacent_to_water_dissolves -- --nocapture`
Expected: FAIL — `dissolving_count()` and `MUD_DISSOLVE_TICKS` don't exist

**Step 3: Write minimal implementation**

Add constants (near line 25):
```rust
/// Dissolve ticks (at the fluid tick rate, ~15 Hz) before mud adjacent to
/// water dissolves into muddy_water.
pub const MUD_DISSOLVE_TICKS: u32 = 60; // ~4 s
```

Add new maps to `Weathering` struct (line 48-52):
```rust
pub struct Weathering {
    table: WeatherTable,
    soaking: FxHashMap<IVec3, u32>,
    drying: FxHashMap<IVec3, u32>,
    dissolving: FxHashMap<IVec3, u32>,
    polluting: FxHashMap<IVec3, u32>,
    settling: FxHashMap<IVec3, u32>,
}
```

Update `new()` to initialize all maps. Add debug stat methods:
```rust
pub fn dissolving_count(&self) -> usize { self.dissolving.len() }
pub fn polluting_count(&self) -> usize { self.polluting.len() }
pub fn settling_count(&self) -> usize { self.settling.len() }
```

In `tick()`, add registration in Step 1 (after existing soak/dry registration): for each `Fell`/`Flowed`/`Settled` event on a water or muddy_water cell, examine 6 neighbors; if a neighbor is mud, enter `dissolving` at 0.

Add Step 2b (advance dissolving): each `dissolving` entry re-verifies it's still mud with an adjacent wet cell (`t.is_wet(...)`). No wet neighbor → remove. Otherwise count up; at `MUD_DISSOLVE_TICKS`, `world.set_voxel(pos, t.muddy_water)`, enter `settling` at 0, and seed `polluting` for clean-water neighbors.

**Step 4: Run test to verify it passes**

Run: `cargo test -p vox-sim mud_adjacent_to_water_dissolves -- --nocapture`
Expected: PASS

**Step 5: Run ALL weathering tests**

Run: `cargo test -p vox-sim weathering -- --nocapture`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add crates/vox-sim/src/weathering.rs crates/vox-sim/src/lib.rs
git commit -m "feat(weathering): mud dissolves to muddy_water"
```

---

### Task 8: Add polluting logic (clean water → muddy_water by contact diffusion)

**Files:**
- Modify: `crates/vox-sim/src/weathering.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn clean_water_adjacent_to_muddy_water_becomes_muddy() {
    let mut world = world_with_floor(STONE);
    let mut weathering = Weathering::new(table());
    // Line of water cells: water | muddy_water
    let y = 5; // floor top
    let water_cell = IVec3::new(7, y, 8);
    let muddy_cell = IVec3::new(8, y, 8);
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
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p vox-sim clean_water_adjacent -- --nocapture`
Expected: FAIL — `POLLUTE_SPREAD_TICKS` doesn't exist, polluting not implemented

**Step 3: Write minimal implementation**

Add constant:
```rust
/// Contact ticks before clean water adjacent to muddy_water becomes muddy.
pub const POLLUTE_SPREAD_TICKS: u32 = 90; // ~6 s
```

In `tick()` Step 1 registration: for each `Fell`/`Flowed`/`Settled` event on a `muddy_water` cell, examine 6 neighbors; if a neighbor is clean `water` (== `t.water`, NOT `t.muddy_water`), enter `polluting` at 0.

Add Step 3 (advance polluting): each `polluting` entry re-verifies it's still clean water adjacent to muddy_water. No muddy_water neighbor → remove. Otherwise count up; at `POLLUTE_SPREAD_TICKS`, `world.set_voxel(pos, t.muddy_water)`, enter `settling` at 0, and seed `polluting` for its clean-water neighbors (chain diffusion).

**Step 4: Run test to verify it passes**

Run: `cargo test -p vox-sim clean_water_adjacent -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/vox-sim/src/weathering.rs crates/vox-sim/src/lib.rs
git commit -m "feat(weathering): pollution diffuses by contact"
```

---

### Task 9: Add settling logic (muddy_water → water + sand below)

**Files:**
- Modify: `crates/vox-sim/src/weathering.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn still_muddy_water_settles_to_water_and_deposits_sand_below() {
    let mut world = world_with_floor(STONE);
    let mut weathering = Weathering::new(table());
    let cell = IVec3::new(8, 5, 8); // one above the stone floor (floor top at y=5, so floor IS y=5)
    let below = cell - IVec3::Y; // y=4 — should be stone (part of the floor fill)

    // Actually, world_with_floor fills y=0..5 with STONE, top layer at y=4.
    // So floor surface is at y=4. Place muddy_water at y=5.
    let muddy_cell = IVec3::new(8, 5, 8);
    let floor_cell = IVec3::new(8, 4, 8); // STONE
    world.set_voxel(muddy_cell, MUDDY_WATER);
    world.set_solid_table(vec![false, false, true, true, true, true, true, false]);

    // Seed: Settled event starts the settle clock
    weathering.tick(&mut world, &[ContactEvent::Settled(muddy_cell)]);
    assert_eq!(weathering.settling_count(), 1);

    for _ in 0..(MUDDY_SETTLE_TICKS - 1) {
        weathering.tick(&mut world, &[]);
        assert_eq!(
            world.get_voxel(muddy_cell),
            MUDDY_WATER,
            "must not settle early"
        );
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
    // Then water moves — Flowed event resets the settle timer
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
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p vox-sim still_muddy_water_settles -- --nocapture`
Expected: FAIL — `MUDDY_SETTLE_TICKS` doesn't exist, settling not implemented

**Step 3: Write minimal implementation**

Add constant:
```rust
/// Settle ticks (continuously still, no moves) before muddy_water clarifies
/// to clean water and deposits sand below.
pub const MUDDY_SETTLE_TICKS: u32 = 150; // ~10 s
```

In `tick()` Step 1 registration:
- For each `Settled` event on a `muddy_water` cell: enter `settling` at 0 (if absent).
- For each `Fell`/`Flowed` event on a `muddy_water` cell: remove from `settling` (moving = not settling).
- For each `Vacated(pos)` event: if `pos` was muddy_water and moved, remove from `settling`.

Add Step 4 (advance settling): each `settling` entry re-verifies it's still `muddy_water`. If material changed → remove. Otherwise count up; at `MUDDY_SETTLE_TICKS`:
- `world.set_voxel(pos, t.water)` — clarify to clean water.
- Deposit sand: check `below = pos - IVec3::Y`. If `world.get_voxel(below)` is a solid material (not air, not a fluid, not already sand), `world.set_voxel(below, t.sand)`. If below is air/fluid/sand, skip.

To check "is solid": `world.solid(below)` (the world's solid table) AND `world.get_voxel(below) != t.sand`.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p vox-sim still_muddy_water_settles moving_muddy_water -- --nocapture`
Expected: PASS

**Step 5: Run ALL weathering tests**

Run: `cargo test -p vox-sim weathering -- --nocapture`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add crates/vox-sim/src/weathering.rs crates/vox-sim/src/lib.rs
git commit -m "feat(weathering): muddy_water settles to water + sand"
```

---

### Task 10: Add sleep guarantee test for polluted lake

**Files:**
- Modify: `crates/vox-sim/src/weathering.rs` (test module)

**Step 1: Write the test**

```rust
#[test]
fn a_fully_settled_polluted_lake_reaches_zero_tracked_cells() {
    let mut world = world_with_floor(MUD);
    world.set_solid_table(vec![false, false, true, true, true, true, true, false]);
    let mut sim = crate::FluidSim::with_fluids_and_powders(vec![WATER, MUDDY_WATER], Vec::new());
    let mut weathering = Weathering::new(table());
    // Place a water blob on top of a mud floor
    sim.place_blob(&mut world, IVec3::new(8, 7, 8), 1, WATER);
    // Run until everything settles: water dissolves mud → muddy_water,
    // muddy_water settles → water + sand, eventually steady state.
    for _ in 0..((MUD_DISSOLVE_TICKS + MUDDY_SETTLE_TICKS) * 3) {
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
```

**Step 2: Run test**

Run: `cargo test -p vox-sim fully_settled_polluted_lake -- --nocapture`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/vox-sim/src/weathering.rs
git commit -m "test(weathering): polluted lake sleep guarantee"
```

---

## Phase 4: Generalize fire and physics (behavior-preserving)

### Task 11: Generalize FireTable for muddy_water extinguish

**Files:**
- Modify: `crates/vox-sim/src/fire.rs:48-60` (`FireTable` struct)
- Modify: `crates/vox-sim/src/fire.rs:221-223` (extinguish check)
- Modify: `crates/vox-sim/src/fire.rs:239-240` (dark_ash wet check)
- Modify: `crates/vox-sim/src/fire.rs:443-444` (test `table()` helper)
- Modify: `crates/vox-app/src/main.rs:126-155` (`fire_table` function)

**Step 1: Write the failing test**

Add to fire.rs test module (update test `table()` helper to include muddy_water):

```rust
#[test]
fn muddy_water_extinguishes_fire() {
    let mut world = world_with_floor(STONE);
    let mut sim = FireSim::new(table());
    let pos = IVec3::new(8, 5, 8);
    world.set_voxel(pos, WOOD);
    world.set_solid_table(vec![false, false, true, true, true, true, true, true, true, false, true, true, true, true, true]);
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
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p vox-sim muddy_water_extinguishes -- --nocapture`
Expected: FAIL — FireTable has no muddy_water field

**Step 3: Write minimal implementation**

Add `muddy_water: Voxel` to `FireTable` and an `is_wet` helper:
```rust
impl FireTable {
    /// Whether `v` is a water-like fluid (water or muddy_water).
    #[inline]
    pub fn is_wet(&self, v: Voxel) -> bool {
        v == self.water || v == self.muddy_water
    }
}
```

Update `:223`: `world.get_voxel(pos + n) == table.water` → `table.is_wet(world.get_voxel(pos + n))`
Update `:240`: `world.get_voxel(q + d) == table.water` → `table.is_wet(world.get_voxel(q + d))`

Update `fire_table()` in main.rs to add `muddy_water: id("muddy_water").unwrap_or(water)` (fall back to water if missing — fire still works, muddy_water just won't be recognized separately. But since we added the material in Task 6, it'll resolve).

Actually, to match the graceful-fallback pattern: `muddy_water: id("muddy_water")?` — but that would disable fire entirely if muddy_water is missing. Better: fall back to water:
```rust
let muddy_water = id("muddy_water").unwrap_or(id("water")?);
```
This way fire works even without muddy_water (it just won't recognize muddy_water as wet — but muddy_water won't exist anyway).

**Step 4: Run ALL fire tests**

Run: `cargo test -p vox-sim fire -- --nocapture`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add crates/vox-sim/src/fire.rs crates/vox-app/src/main.rs
git commit -m "feat(fire): muddy_water extinguishes fire"
```

---

### Task 12: Generalize PhysicsWorld buoyancy for multiple fluids

**Files:**
- Modify: `crates/vox-physics/src/solver.rs:154-156` (struct field)
- Modify: `crates/vox-physics/src/solver.rs:168-171` (setter)
- Modify: `crates/vox-physics/src/solver.rs:480-483` (buoyancy check)
- Modify: `crates/vox-app/src/main.rs:554-557` (wiring)

**Step 1: Write the failing test**

Check existing physics tests for buoyancy — find the buoyancy test pattern:

```rust
// In solver tests, add:
#[test]
fn body_floats_in_muddy_water() {
    // A wood body (density 700 < muddy_water 1100) should float in muddy_water.
    // This requires the solver to recognize muddy_water as a fluid for buoyancy.
    // ... follow existing buoyancy test pattern ...
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p vox-physics body_floats_in_muddy -- --nocapture`
Expected: FAIL — solver only checks single water_voxel

**Step 3: Write minimal implementation**

Change `water_voxel: Option<Voxel>` to `fluid_voxels: Vec<Voxel>`:
```rust
/// Fluid materials for buoyancy (water, muddy_water, ...). Empty disables
/// buoyancy.
fluid_voxels: Vec<Voxel>,
```

Update setter:
```rust
pub fn set_fluid_voxels(&mut self, fluids: Vec<Voxel>) {
    self.fluid_voxels = fluids;
}
```

Keep a compatibility method:
```rust
pub fn set_water_voxel(&mut self, v: Voxel) {
    self.fluid_voxels = vec![v];
}
```

Update buoyancy check (`:480-483`):
```rust
if world.in_bounds(bottom_vox) {
    if self.fluid_voxels.contains(&world.get_voxel(bottom_vox)) {
        // ... existing buoyancy calculation ...
    }
}
```

Update main.rs wiring:
```rust
fluid_voxels: {
    let mut fluids = Vec::new();
    if let Some(water) = registry.id_by_name("water") {
        fluids.push(Voxel(water.0));
    }
    if let Some(muddy) = registry.id_by_name("muddy_water") {
        fluids.push(Voxel(muddy.0));
    }
    let mut phys = PhysicsWorld::new();
    if !fluids.is_empty() {
        phys.set_fluid_voxels(fluids);
    }
    phys
},
```

Or more generally, build from all materials where `def.fluid == true`:
```rust
fn fluid_materials(registry: &MaterialRegistry) -> Vec<Voxel> {
    (1..registry.len())
        .filter_map(|i| {
            let def = registry.get(vox_core::MaterialId(i as u16))?;
            def.fluid.then(|| Voxel(i as u16))
        })
        .collect()
}
```

**Step 4: Run ALL physics tests**

Run: `cargo test -p vox-physics -- --nocapture`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add crates/vox-physics/src/solver.rs crates/vox-app/src/main.rs
git commit -m "feat(physics): generalize buoyancy to multiple fluids"
```

---

## Phase 5: Generalize rendering

### Task 13: Generalize meshing (slab.rs + greedy.rs)

**Files:**
- Modify: `crates/vox-mesh/src/slab.rs:113-122` (`opaque()` method)
- Modify: `crates/vox-mesh/src/greedy.rs:133` (`mesh_slab` signature)
- Modify: `crates/vox-mesh/src/greedy.rs:162` (`is_water` — fixes latent bug)
- Modify: `crates/vox-mesh/src/greedy.rs:186` (depth baking)
- Modify: `crates/vox-mesh/src/greedy.rs:246,277,297` (water_voxel param → fluids)
- Modify: `crates/vox-app/src/remesh.rs:71,92` (callers)
- Modify: `crates/vox-app/src/body_mesh.rs:49-62` (callers)
- Modify: `crates/vox-app/src/main.rs:613,683` (callers)
- Modify: `crates/vox-app/examples/stress.rs:166` (caller)

**Step 1: Change mesh_slab signature**

`mesh_slab` currently takes `water_voxel: Voxel`. Change to `fluids: &[Voxel]`:

```rust
pub fn mesh_slab(slab: &VoxelSlab, jitter_seed: IVec3, fluids: &[Voxel]) -> MeshData {
```

Update `slab.rs` `opaque()` — it currently hardcodes `Voxel(9)`. Add a `fluids` parameter or pass a closure. The cleanest approach: `VoxelSlab` already stores data; add a method that takes the fluid set:

```rust
/// Like [`solid`](Self::solid) but treats fluid materials as non-solid.
#[inline]
pub fn opaque(&self, rel: IVec3, fluids: &[Voxel]) -> bool {
    let v = self.get(rel);
    v != AIR && !fluids.contains(&v)
}
```

Update all `slab.opaque(p)` calls in greedy.rs to `slab.opaque(p, fluids)`.

Update `is_water` at greedy.rs:162:
```rust
let is_fluid = fluids.contains(&slab.get(p));
```

Update depth baking at greedy.rs:186:
```rust
if fluids.contains(&mat) {
```

Update jitter baking at greedy.rs:297:
```rust
jitter: if fluids.contains(&cell.material) {
```

**Step 2: Update all callers**

In `remesh.rs:71`: `dispatch(&mut self, world: &World, camera_pos: Vec3, fluids: &[Voxel])` — pass `&[...]` instead of `Voxel`.

In `body_mesh.rs:49-62`: `dispatch(..., fluids: &[Voxel])` — same.

In `main.rs:613,683`: pass the fluid set.

In `stress.rs:166`: pass `&[Voxel(9)]`.

In `main.rs`: build the fluid set from the registry:
```rust
fn fluid_materials(registry: &MaterialRegistry) -> Vec<Voxel> {
    (1..registry.len())
        .filter_map(|i| {
            let def = registry.get(vox_core::MaterialId(i as u16))?;
            def.fluid.then(|| Voxel(i as u16))
        })
        .collect()
}
```
(If this was already created in Task 12 for physics, reuse it.)

**Step 3: Verify compilation**

Run: `cargo check -p vox-mesh && cargo check -p vox-app`
Expected: PASS

**Step 4: Run mesh tests**

Run: `cargo test -p vox-mesh -- --nocapture`
Expected: PASS (update test callers to pass `&[Voxel(9)]`)

**Step 5: Commit**

```bash
git add crates/vox-mesh/src/slab.rs crates/vox-mesh/src/greedy.rs crates/vox-app/src/remesh.rs crates/vox-app/src/body_mesh.rs crates/vox-app/src/main.rs crates/vox-app/examples/stress.rs
git commit -m "feat(mesh): generalize fluid rendering from Voxel(9) to fluid set"
```

---

### Task 14: Generalize the WGSL shader

**Files:**
- Modify: `assets/shaders/voxel.wgsl:194-262` (fragment shader)

**Step 1: Add is_fluid helper and muddy_water_id specialization constant**

After the `water_pass` override (line 198), add:

```wgsl
// Specialization constant: the material id for muddy_water, or 0 if absent.
// 0 = air, which is_fluid will never match, so muddy_water rendering is
// disabled cleanly when the material doesn't exist.
override muddy_water_id: u32 = 0u;

fn is_fluid(id: u32) -> bool {
    return id == 9u || id == muddy_water_id;
}
```

**Step 2: Replace the 6 `mat_id == 9u` checks**

- Line 203: `if (water_pass == 0u && in.mat_id == 9u) { discard; }` → `if (water_pass == 0u && is_fluid(in.mat_id)) { discard; }`
- Line 204: `if (water_pass == 1u && in.mat_id != 9u) { discard; }` → `if (water_pass == 1u && !is_fluid(in.mat_id)) { discard; }`
- Line 223: `if (cam.sun_dir.w > 0.0 && in.mat_id != 9u)` → `if (cam.sun_dir.w > 0.0 && !is_fluid(in.mat_id))`
- Line 241: `if (cam.ambient_sky.w > 0.0 && in.mat_id != 9u)` → `if (cam.ambient_sky.w > 0.0 && !is_fluid(in.mat_id))`
- Line 251: `let alpha = select(1.0, 0.85, in.mat_id == 9u);` → `let alpha = select(1.0, select(0.85, 0.80, in.mat_id == muddy_water_id), is_fluid(in.mat_id));`
  - Clean water: 0.85 alpha. Muddy water: 0.80 alpha (slightly more opaque — murkier). Non-fluid: 1.0.
- Line 252: `if (in.mat_id == 9u) {` → keep as `if (in.mat_id == 9u) {` — the blue tint + ripple is clean-water-only. Muddy_water gets its palette color only.

**Step 3: Verify shader compiles**

Run: `cargo test -p vox-render shader_validate -- --nocapture`
Expected: PASS (the shader validation test parses the WGSL)

**Step 4: Commit**

```bash
git add assets/shaders/voxel.wgsl
git commit -m "feat(shader): generalize fluid pass to is_fluid, add muddy_water_id"
```

---

### Task 15: Generalize voxel_pipeline.rs

**Files:**
- Modify: `crates/vox-render/src/voxel_pipeline.rs:39-49` (`GpuMesh` struct — rename `has_water` to `has_fluid`)
- Modify: `crates/vox-render/src/voxel_pipeline.rs:97-103` (`VoxelPipeline::new` — add `muddy_water_id` to specialization constants)
- Modify: `crates/vox-render/src/voxel_pipeline.rs:226-229` (opaque constants)
- Modify: `crates/vox-render/src/voxel_pipeline.rs:270-273` (water constants)
- Modify: `crates/vox-render/src/voxel_pipeline.rs:347` (`has_water` → `has_fluid`)
- Modify: `crates/vox-render/src/voxel_pipeline.rs:532-547` (`draw_water` — use `has_fluid`)

**Step 1: Rename has_water to has_fluid in GpuMesh**

```rust
struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    instance: wgpu::Buffer,
    aabb_min: Vec3,
    aabb_max: Vec3,
    /// Set at upload time so the fluid pass can skip chunks without any fluid.
    has_fluid: bool,
}
```

Update `:347`:
```rust
let has_fluid = mesh.vertices.iter().any(|v| v.material == 9 || v.material == muddy_water_id as u16);
```

Wait — `mesh_slab` now produces vertices with the actual material id. The pipeline needs to know which material ids are fluids. Since `VoxelPipeline::new` takes `&MaterialRegistry`, it can look up fluid material ids at construction:

```rust
let fluid_ids: Vec<u16> = (1..registry.len())
    .filter_map(|i| {
        let def = registry.get(MaterialId(i as u16))?;
        def.fluid.then(|| i as u16)
    })
    .collect();
```

Store `fluid_ids` in `VoxelPipeline`. Use it in `upload_chunk`:
```rust
let has_fluid = mesh.vertices.iter().any(|v| self.fluid_ids.contains(&v.material));
```

And pass `muddy_water_id` to the shader specialization constants:
```rust
let muddy_water_id = registry.id_by_name("muddy_water").map(|m| m.0 as u32).unwrap_or(0);

let opaque_constants = wgpu::PipelineCompilationOptions {
    constants: &std::collections::HashMap::from([
        ("water_pass".into(), 0.0_f64),
        ("muddy_water_id".into(), muddy_water_id as f64),
    ]),
    ..Default::default()
};
// Same for water_constants
```

Update `draw_water` (rename to `draw_fluids` for clarity, or keep the name — the behavior is "draw the fluid pass"):
```rust
pub fn draw_water<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>, frustum: &Frustum) {
    pass.set_pipeline(&self.water_pipeline);
    pass.set_bind_group(0, &self.bind_group, &[]);
    for mesh in self.chunks.values() {
        if !mesh.has_fluid {
            continue;
        }
        // ... rest unchanged ...
    }
}
```

**Step 2: Verify compilation**

Run: `cargo check -p vox-render`
Expected: PASS

**Step 3: Run render tests**

Run: `cargo test -p vox-render -- --nocapture`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/vox-render/src/voxel_pipeline.rs
git commit -m "feat(render): generalize fluid pass, add muddy_water_id constant"
```

---

## Phase 6: Wire everything in main.rs

### Task 16: Build FluidSim with fluid_materials and wire all callers

**Files:**
- Modify: `crates/vox-app/src/main.rs:531` (water_voxel → fluid_materials)
- Modify: `crates/vox-app/src/main.rs:539-542` (FluidSim constructor)
- Modify: `crates/vox-app/src/main.rs:554-557` (physics fluid_voxels)
- Modify: `crates/vox-app/src/main.rs:613,683` (mesh_slab callers)
- Modify: `crates/vox-app/src/main.rs` (store fluid set in App struct)

**Step 1: Add fluid_materials helper and store in App**

Add a `fluid_materials` function (if not already added in Task 12/13):
```rust
fn fluid_materials(registry: &MaterialRegistry) -> Vec<Voxel> {
    (1..registry.len())
        .filter_map(|i| {
            let def = registry.get(vox_core::MaterialId(i as u16))?;
            def.fluid.then(|| Voxel(i as u16))
        })
        .collect()
}
```

Store `fluids: Vec<Voxel>` in the `App` struct. Update `FluidSim` construction:
```rust
fluid: vox_sim::FluidSim::with_fluids_and_powders(
    fluid_materials(&registry),
    powder_materials(&registry),
),
```

Update physics:
```rust
let mut phys = PhysicsWorld::new();
let fluids = fluid_materials(&registry);
if !fluids.is_empty() {
    phys.set_fluid_voxels(fluids);
}
phys
```

Update all `mesh_slab` callers to pass `&self.fluids` instead of a single `water_voxel`.

Update `remesh.dispatch` and `body_mesh.dispatch` calls to pass `&self.fluids`.

**Step 2: Verify compilation**

Run: `cargo check -p vox-app`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/vox-app/src/main.rs
git commit -m "feat(app): wire fluid_materials through all systems"
```

---

## Phase 7: Integration test and verification

### Task 17: Add integration test — full pollution lifecycle

**Files:**
- Modify: `crates/vox-app/src/main.rs` or `crates/vox-sim/src/weathering.rs` (wherever integration tests live)

**Step 1: Write the integration test**

A test that:
1. Creates a world with a mud floor
2. Places water on top
3. Runs the sim + weathering loop
4. Confirms: mud → muddy_water (dissolve), water becomes muddy (diffusion), then muddy_water → water + sand (settle)
5. Confirms water-cell count is conserved by the fluid sim

This may be a headless test in vox-sim if it doesn't need GPU, or a vox-app test if it uses the full pipeline.

**Step 2: Run test**

Run: `cargo test -p vox-sim pollution_lifecycle -- --nocapture`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/vox-sim/src/weathering.rs
git commit -m "test: full pollution lifecycle integration test"
```

---

### Task 18: Run full test suite and verify

**Step 1: Run all tests**

Run: `cargo test --workspace -- --nocapture`
Expected: ALL PASS

**Step 2: Run the engine and visually verify**

Run: `cargo run -p vox-app --release`
- Place water on a mud floor (hotbar slot 5)
- Observe: mud dissolves → water turns murky brown (muddy_water)
- Observe: still muddy water eventually clarifies → sand appears on the floor
- Observe: muddy water flows like water (levels, spreads)

**Step 3: Update README**

Add `muddy_water` to the Simulation section of `README.md` under fluid sim:
```
- **Water pollution**: Mud adjacent to water dissolves into muddy_water (a
  murky fluid that flows like water). Pollution diffuses by contact to
  adjacent clean water. Still muddy_water settles after ~10s, clarifying to
  clean water and depositing sand on the floor below.
```

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document water pollution system"
```

---

## Summary of all tasks

| # | Phase | Task | Key change |
|---|---|---|---|
| 1 | Fluid gen | FluidSim struct + constructor | `water: Voxel` → `fluids: Vec<Voxel>` |
| 2 | Fluid gen | Tick loop is_fluid checks | 4 spots: is_supported, has_water_above, momentum, is_simmed |
| 3 | Fluid gen | Verify place_blob/wake_region | Already use is_simmed |
| 4 | Weathering gen | WeatherTable + is_wet | Add muddy_water field, generalize :130, :170 |
| 5 | Weathering gen | main.rs weather_table | Add muddy_water resolution |
| 6 | Material | Add muddy_water to core.toml | New fluid material |
| 7 | Pollution | Dissolving logic | mud → muddy_water |
| 8 | Pollution | Polluting logic | clean water → muddy_water by contact |
| 9 | Pollution | Settling logic | muddy_water → water + sand below |
| 10 | Pollution | Sleep guarantee test | All 5 maps drain to empty |
| 11 | Fire gen | FireTable + is_wet | muddy_water extinguishes fire |
| 12 | Physics gen | Buoyancy fluid_voxels | Body floats in muddy_water |
| 13 | Render gen | Meshing (slab + greedy) | Voxel(9) → fluid set |
| 14 | Render gen | WGSL shader | is_fluid helper, muddy_water_id constant |
| 15 | Render gen | voxel_pipeline.rs | has_fluid, specialization constants |
| 16 | App wiring | Wire fluid_materials | All callers updated |
| 17 | Integration | Full lifecycle test | mud → muddy → water + sand |
| 18 | Verify | Full suite + visual | All tests pass, visual confirm |
