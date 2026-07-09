# Fire System — Ember Ignition, Fire Spread, Consumption — Design Document

**Date:** 2026-07-09
**Status:** Implemented

**User request:** "make a fire system... ember will be a single piece of
pre-lit wood that will be able to light anything nearby on fire. Plants,
wood. Water puts fire out, produces steam (smoke) for now. Smoke emission
from fire burnt particles and a proper fire spreading system."

---

## 1. Decisions of Record

| Question | Decision |
|---|---|
| Architecture | `FireSim` module in `vox-sim` (like `Weathering`), fed by the same fluid tick loop. Tracks burning cells in `FxHashMap<IVec3, BurnState>`. No "fire material" in the voxel grid — a burning cell keeps its original material until consumed. |
| Water extinguishing | Water-adjacent burning cells stop burning (removed from burn map). Ember touching water → becomes char (extinguished permanently — a wet ember is dead charcoal, not fresh wood). Steam/smoke particles emitted on extinguishing. |
| Char | A new solid material: dark gray-black, strength 0.5, density 300. The residue of consumed flammable materials. Not flammable. Walkable but fragile. |
| Fire spread | 6 face-adjacent neighbors only. A burning cell ignites flammable neighbors after a spread delay (~2s). Fire can't jump diagonal gaps. |
| Consumption | Wood ~15s, planks ~12s, leaves ~5s, grass ~3s. When consumed → char. Ember consumes after ~60s → char. |
| Water extinguishing | Water-adjacent burning cells stop burning (removed from burn map). Ember touching water → becomes wood (extinguished permanently). Steam/smoke particles emitted on extinguishing. |
| Smoke emission | Burning cells emit smoke particles (buoyant, dark gray) at a rate proportional to burn intensity. The app's frame loop spawns particles from active burn cells. |
| Flammable flag | New `flammable: bool` on `MaterialDef`, parsed from TOML. Wood, leaves, planks, grass get `flammable = true`. Ember is also flammable (it's burning wood). Char is not. |
| Sleep guarantee | Same as weathering: empty burn map at steady state = zero cost. Fire either burns out (cell consumed → char, removed from map) or is extinguished (water contact → removed from map). |

## 2. Material changes (`vox-core` + `core.toml`)

### 2.1 New `flammable` flag

`MaterialDef` gains `pub flammable: bool`, parsed from TOML as
`flammable = true`, defaulting false. Parallel to `fluid`/`powder`.

### 2.2 New materials in `core.toml`

**Ember** (id 10, after mud):
```toml
[[material]]
name = "ember"
color = [0.85, 0.35, 0.10]
jitter = 0.08
density = 600.0
strength = 2.0
flammable = true
```
Solid, warm orange-brown. It's wood that's already on fire.

**Char** (id 11, after ember):
```toml
[[material]]
name = "char"
color = [0.12, 0.10, 0.09]
jitter = 0.03
density = 300.0
strength = 0.5
```
Solid, dark, fragile. The residue of consumed flammable materials.

### 2.3 Existing materials get `flammable = true`

- `wood`: `flammable = true`
- `leaves`: `flammable = true`
- `planks`: `flammable = true`
- `grass`: `flammable = true`

## 3. FireSim (`vox-sim/src/fire.rs`)

### 3.1 BurnState

```rust
struct BurnState {
    /// Ticks since ignition (for spread timing + consumption).
    ticks: u32,
    /// True for ember cells (slow burn, longer lifetime).
    is_ember: bool,
}
```

### 3.2 FireSim struct

```rust
pub struct FireSim {
    table: FireTable,
    burning: FxHashMap<IVec3, BurnState>,
}
```

`FireTable` (resolved by name in the app, like `WeatherTable`):
```rust
pub struct FireTable {
    pub water: Voxel,
    pub ember: Voxel,
    pub char: Voxel,
    pub flammable: Vec<Voxel>,  // wood, leaves, planks, grass, ember
}
```

### 3.3 FireSim::tick

Called from the same tick loop as weathering, after `fluid.tick` and
`weathering.tick`:

1. **Register new fires:** for each newly-ember cell (placed by player)
   or newly-ignited cell (spread from a burning neighbor), add to the
   burn map. The app calls `fire.ignite(pos)` when placing an ember.
   Spread is handled in step 3.

2. **Check extinguishing:** for each burning cell, if any 6-neighbor is
   water, remove from burn map. If the cell is ember, convert it to wood
   (`world.set_voxel(pos, wood)` — but we don't have wood in the table;
   instead, ember extinguishes to char directly). Emit a `SteamEvent` so
   the app can spawn steam/smoke particles.

3. **Spread:** for each burning cell with `ticks >= SPREAD_DELAY` (30
   ticks ~2s), check 6 neighbors. If a neighbor is a flammable material
   and not already burning, add it to the burn map at 0 ticks. Each
   burning cell can spread to at most one new cell per tick (randomized
   neighbor order) to avoid exponential explosion.

4. **Advance + consume:** increment each burning cell's tick count. If
   it reaches its material's burn duration, convert the cell to char
   via `world.set_voxel` and remove from the burn map. Ember burns for
   ~900 ticks; other materials per their duration.

5. **Emit smoke events:** the sim collects `SmokeEvent { pos, intensity }`
   entries each tick for burning cells. The app drains these and spawns
   smoke particles.

### 3.4 Durations (at 15 Hz tick rate)

```rust
pub const GRASS_BURN_TICKS: u32 = 45;    // ~3s
pub const LEAVES_BURN_TICKS: u32 = 75;   // ~5s
pub const PLANKS_BURN_TICKS: u32 = 180;  // ~12s
pub const WOOD_BURN_TICKS: u32 = 225;    // ~15s
pub const EMBER_BURN_TICKS: u32 = 900;   // ~60s
pub const SPREAD_DELAY_TICKS: u32 = 30;  // ~2s before a fire can spread
```

### 3.5 Events

```rust
pub enum FireEvent {
    /// A cell was consumed by fire → became char. App may spawn extra smoke.
    Consumed(IVec3),
    /// A fire was extinguished by water. App spawns steam/smoke.
    Extinguished(IVec3),
    /// A cell is actively burning this tick. App spawns smoke particles.
    Burning(IVec3, f32), // position, intensity 0..1
}
```

The app drains these each tick to drive particle emission.

## 4. App wiring (`vox-app`)

### 4.1 Hotbar slot 6: Place Ember

`Tool` enum gains `Ember` variant at slot 6. LMB places an ember block
at the crosshair (like `place_voxel` but always places ember material).
This is simpler than the water tool — just a single voxel placement
with a fixed material.

### 4.2 FireSim field + tick

`VoxApp` gains `fire: Option<FireSim>`. The frame loop calls
`fire.tick(&mut world, &fluid)` after the weathering tick. The sim
checks water adjacency itself (it has the water voxel id in its table).

### 4.3 Smoke particle emission

After `fire.tick`, drain `fire.drain_events()` and for each
`Burning(pos, intensity)` event, spawn 1-3 smoke particles at `pos`.
For `Extinguished(pos)`, spawn a burst of white-ish steam particles.

## 5. Testing plan

All in `vox-sim` unless noted:

- **Ember ignites wood:** place ember next to wood, tick, wood enters
  burn map.
- **Fire spreads:** a line of wood blocks, ignite one end, fire spreads
  along the line.
- **Consumption:** wood burns for its duration, then becomes char.
- **Water extinguishes:** water-adjacent burning cell stops burning.
- **Ember extinguishes to char:** water touching ember converts it to
  char.
- **Sleep guarantee:** after all fire burns out, burn map is empty.
- **Integration (vox-app):** ember placed via tool, fire spreads to
  nearby wood, smoke particles emitted.

## 6. Explicitly out of scope

- Fire physics (heat, temperature fields, convection).
- Fire on debris bodies (fire only affects static world voxels).
- Fire emitting light (GPU lighting changes).
- Different fire intensities based on material (uniform spread rate).
- Wind affecting fire spread direction.
