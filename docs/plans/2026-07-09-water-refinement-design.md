# Water Refinement — Cohesive Flow + Weathering — Design Document

**Date:** 2026-07-09
**Status:** Implemented
**Builds on:** `2026-07-09-fluid-sim-design.md` (the CA fluid sim, including the
flow rule that fixed sand-piling). This document refines flow behavior and adds
water-driven material transformation. Nothing here changes the sim's core
premises: binary full/empty cells, conserved volume, active-cell sleeping,
settled water costs nothing.

---

## 1. Decisions of Record

| Question | Decision |
|---|---|
| Residual piles | Extend the drop-search from 4 axis directions to **8 horizontal directions** (adds the 4 diagonals) — off-axis drops were invisible, leaving stable off-axis piles |
| Cohesive flow | **Momentum memory**: each *active* cell remembers the horizontal direction of its last move; woken neighbors inherit the mover's direction; momentum is tried first, remaining directions stay randomized. Dropped on settle — zero idle state. |
| Rivers | Emerge from momentum + smarter search while water drains through a carved channel. **No infinite sources** — permanent rivers are a future feature, out of scope. |
| Transformations | Grass → dirt (~3 s of water contact, above or side); dirt → mud (~7 s); stone → sand from **flowing** water only (~30 s), with **falling** water (waterfalls) ~5× faster. Grass death, dirt soaking, and erosion are permanent. |
| Reversibility | Only mud reverts: **mud → dirt** after ~20 s with no adjacent water. Timer resets if water returns. |
| Architecture | New `weathering` module **inside `vox-sim`**, fed by contact events the fluid tick already generates. No world scanning, no new crate, no chunk-storage changes. |
| Materials | One new material: **mud** (solid, dark wet brown, density 1700, strength 1.0). Sand already exists. |
| Enabling | Weathering is **opt-in** via a material table built by name lookup in `vox-app`; a missing material name disables weathering gracefully. Existing `vox-sim` tests run without it, unmodified. |

## 2. Flow refinements (`fluid.rs`)

### 2.1 Eight-direction drop-search

The flow rule's horizontal scan currently walks only ±X/±Z. A drop reachable
only diagonally (e.g. around a corner of the pile) is never found, so the cell
settles and the pile survives. The scan now walks all 8 horizontal directions
(randomized order, momentum-first — see below), same `FLOW_HORIZON`, same
"move one step toward the nearest drop" result, same termination argument: a
move still happens only when a strictly lower destination is reachable, so
flat sheets and full basins still sleep.

Cost: 8 × `FLOW_HORIZON` lookups per stuck-but-active cell per tick, up from
4 ×. Only active cells pay it.

### 2.2 Momentum

`FluidSim` gains `momentum: FxHashMap<IVec3, IVec3>` — a horizontal unit-ish
direction per **active** cell:

- **Record:** when a cell moves, store the horizontal component of
  `dest - pos` (if non-zero) keyed by `dest`.
- **Recruit:** neighbors woken by a mover inherit the mover's direction if
  they don't already have one. This is what makes a draining current pull the
  water around it into the same flow — "water notices where nearby water is
  going."
- **Consult:** in `step_cell`, the momentum direction is tried first for
  diagonal falls, the drop-search, and pressure-gated spreading. All other
  directions keep today's coin-flip randomization.
- **Forget:** entries are dropped when their cell settles (and pruned for
  positions no longer active). The map is empty whenever the water sleeps.

Momentum only reorders *preference* among already-legal moves — it can never
create a move that today's rules would reject — so conservation and settling
guarantees are structurally unaffected.

## 3. Weathering (`vox-sim/src/weathering.rs`)

### 3.1 Contact events

While `FluidSim::tick` processes cells it already knows everything weathering
needs. It now appends to a per-tick event list (drained by the caller):

- `Fell(pos)` — water arrived at `pos` by moving down or diagonally down.
- `Flowed(pos)` — water arrived by a horizontal move (flow or spread rule).
- `Settled(pos)` — water at `pos` found no move and left the active set.
- `Vacated(pos)` — water left `pos` (every move emits one).

### 3.2 `Weathering` struct

```
pub struct Weathering {
    table: WeatherTable,           // grass, dirt, mud, stone, sand, water ids
    soaking: FxHashMap<IVec3, u32>, // transformable cells adjacent to water
    drying:  FxHashMap<IVec3, u32>, // mud cells that lost water contact
}
```

`Weathering::tick(&mut self, world: &mut World, events: &[ContactEvent])`:

1. **Register:** for each `Fell`/`Flowed`/`Settled` event, examine the 6
   neighbors of the water cell. Grass and dirt neighbors enter `soaking` at 0
   (if absent). Stone neighbors enter only for `Fell`/`Flowed` events
   (flowing water erodes; still water does not), with `Fell` contacts marked
   to accrue at ~5× rate (waterfall emphasis). A mud neighbor of a water
   cell has its `drying` entry removed (re-wetted).
2. **Advance soak:** each `soaking` entry re-verifies it still has an
   adjacent water cell (6-neighborhood; cheap — the map holds boundary cells
   only). No water → entry removed. Otherwise the counter advances and, at
   its material's threshold, converts via `world.set_voxel`:
   grass → dirt, dirt → mud, stone → sand. The converted cell leaves the map
   (a fresh dirt cell re-enters at 0 through future contact events or the
   next verify pass, giving the two-step grass → dirt → mud progression).
3. **Register drying:** for each `Vacated(pos)`, mud cells in its
   6-neighborhood enter `drying` at 0 (if absent).
4. **Advance drying:** each `drying` entry checks for adjacent water — if
   found, remove (wet again, handled by rule 1 next contact). Otherwise
   count up; at threshold, mud → dirt via `world.set_voxel`, entry removed.

Thresholds in fluid ticks (15 Hz): grass ~45, dirt ~105, stone ~450 flowing /
~90 falling, mud-dry ~300. Constants in `vox-core::tunables` style, tuned by
playtesting.

### 3.3 Why sleep stays free

Every map entry is either actively counting toward a conversion or gets
removed on its next verify. A fully-soaked lakebed (all mud) has no soaking
entries left; wet mud is *not* tracked — it only enters `drying` when water
actually vacates an adjacent cell, which only happens while the fluid sim is
awake. Steady state for any settled body of water: both maps empty, zero cost.

Conversions call `world.set_voxel`, so remeshing, physics wake, and fluid
wake all flow through the existing dirty-region pipeline. A conversion under
settled water briefly wakes that water; it re-settles the next tick (nothing
geometric changed). Bounded and accepted, same trade-off class as §5 of the
fluid design.

## 4. Materials and integration

- `assets/materials/core.toml` gains `mud`: color ≈ [0.30, 0.22, 0.16],
  jitter 0.05, density 1700, strength 1.0, solid (walkable).
- `vox-app` builds `WeatherTable` by registry name lookup
  (grass/dirt/mud/stone/sand/water). Any missing name → weathering disabled
  with a log line, not a crash.
- `main.rs`: after each fluid tick, drain the sim's contact events and call
  `weathering.tick(...)` — same fixed-rate loop, one added call.

## 5. Testing plan

All headless in `vox-sim` unless noted:

- **Flow:** an off-axis pile (drop reachable only diagonally) flattens; the
  flat-sheet, basin, and leveling sleep tests still pass unmodified.
- **Momentum:** with a single drain point, a majority of moves in a tick run
  toward it (directional coherence), and the momentum map is empty once
  settled.
- **Weathering:** grass with side contact dies at its threshold and not
  before; submerged dirt becomes mud; stone under a waterfall erodes ~5×
  sooner than beside a horizontal flow; still water never erodes stone; mud
  dries to dirt only after water leaves and re-wetting resets the timer;
  a fully weathered lake reaches zero active cells *and* zero weathering
  entries; water-cell count stays conserved throughout.
- **Integration (`vox-app`):** the drain-lake test still passes; a new test
  carves a slope below a settled pool and confirms water flows down it and
  the channel floor eventually erodes.

## 6. Explicitly out of scope

- Infinite/source water blocks (needed for permanent rivers).
- ~~Sand behaving as a falling granular material.~~ **Implemented** in `2026-07-09-powder-design.md` — sand and mud are now powders.
- Wetness rendering (darkened wet materials), partial-fill surfaces.
- Erosion transporting sediment (sand re-deposits downstream).
