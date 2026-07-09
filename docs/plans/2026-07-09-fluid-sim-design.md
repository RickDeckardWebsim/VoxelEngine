# Scalable CA Fluid Simulation — Design Document

**Date:** 2026-07-09
**Status:** Approved
**Vision:** Real flowing water, at any voxel scale the engine runs at (Teardown-scale 10 cm or Minecraft-scale 1 m), fully integrated with the existing destruction pipeline — dig into a lake and it drains, blast open a wall and a reservoir floods the breach. First slice: water only, discrete full/empty voxels, conserved volume, heavily optimized via active-cell sleeping so a settled body of water costs nothing regardless of how many cells it spans.

This is the `vox-sim` crate the README's roadmap already named as "planned" — a new sibling crate at the `vox-physics` tier.

---

## 1. Decisions of Record

| Question | Decision |
|---|---|
| Scope | **Water only** for this pass; core built generically enough that lava/sand are later additions, not a rewrite |
| Visual representation | **Flat full/empty voxels** — a water cell is either the opaque water material or empty air, meshed by the existing greedy mesher exactly like any other material. No sloped/partial surfaces. |
| Sim resolution | **Native voxel resolution** — one fluid cell = one world voxel, whatever `--scale` is. No second coordinate system. |
| Volume model | **Conserved** — a fixed quantity of water redistributes under gravity/spreading; no infinite source blocks |
| Destruction integration | **Fully integrated from day one** — carving/placing wakes nearby fluid, mirroring how `PhysicsWorld::wake_region` already reacts to world edits |
| How water enters the world | **New hotbar tool** that places a blob of water at the crosshair (worldgen lakes are a later fast-follow, not part of this design) |
| Tick cadence | **Fixed lower-rate ticks (~15 Hz) + active-cell sleeping**, decoupled from the 60 Hz physics/render loop |
| Algorithm | **Discrete swap-based cellular automaton** (full/empty cells moved whole), not fractional/pressure-based levels |

## 2. Foundational fix: solidity must consult the material registry

`World::solid()` / `SolidLookup::solid()` is currently `voxel != AIR` — a documented MVP shortcut. It's the single choke point used by:
- `vox_world::raycast` (tool aiming)
- `vox-physics::character` (player collision)
- `vox-physics::contact::world_contacts` (rigidbody-vs-world collision)
- `vox-physics::destruction`'s connectivity flood (is this component grounded?)

Water is non-solid and non-air, so every one of these would currently treat it as a solid wall without this fix.

**Fix:** `World` gains a solidity lookup table — `Vec<bool>` indexed by material id, built once from the `MaterialRegistry` (a new `World::new(cfg, &registry)` parameter, or an equivalent one-time setup call). `World::solid`/`SolidLookup::solid` check this table instead of `!= AIR`. Their **signatures are unchanged**, so `raycast.rs`, `character.rs`, `contact.rs`, and `destruction.rs` need no edits at all — only `World::new` and its call sites (production + tests, all of which already construct a `MaterialRegistry`) change.

The greedy mesher's `VoxelGrid`/`VoxelSlab::solid()` is a **separate, unrelated concept** (face-culling: "is this a hard visual boundary") and stays `!= AIR` unchanged — this is what makes water-against-air render a face and water-against-water cull one, with zero mesher changes.

`MaterialDef` gains one new optional field: `fluid: bool` (default `false`) — distinguishes "this material is non-solid" (could be any future decoration) from "this material is *simulated* by `vox-sim`."

## 3. Crate: `vox-sim`

New crate, `vox-core` + `vox-world` dependencies only (no `vox-physics` dependency — water doesn't need rigidbodies for this pass). Owns:

- **No separate fluid grid.** Water lives in `World`'s real chunk storage as an ordinary material. This means chunk storage, dirty-region tracking, streaming-readiness, and the greedy mesher all apply to water automatically — the "add a system = add a crate" philosophy holds because the new crate adds *behavior*, not a second copy of *storage*.
- **Active-cell set**: `FxHashSet<IVec3>` of currently-flowing water positions — the only state genuinely new to this crate, mirroring `PhysicsWorld`'s sleep bookkeeping. Settled water isn't in this set and costs nothing to tick.
- **`FluidSim::tick(&mut self, world: &mut World)`**: processes the active set once (see algorithm below), capped per call by a budget constant (same pattern as `MAX_DEBRIS_BODIES`) so a single tick can't spike frame time — overflow carries into the next tick.
- **`FluidSim::wake_region(&mut self, world: &World, min: IVec3, max: IVec3)`**: reactivates settled water inside or adjacent to an edited region. Called from the same `drain_dirty_regions()` loop in `main.rs` that already wakes physics bodies.
- **`FluidSim::place_blob(&mut self, world: &mut World, center: IVec3, radius: i32, water: Voxel)`**: fills a sphere with water and activates every cell — backs the new hotbar tool.

## 4. The algorithm

Pure, unit-testable core (mirrors `destruction.rs`'s carve functions: takes a `World`/registry and coordinates in, returns what changed, no hidden state beyond the active set). For each active cell, each tick, in randomized order (xorshift jitter, same construction as `PhysicsWorld::lifetime_rng`, to avoid axis bias):

1. **Fall**: if the cell directly below is empty and non-solid, move there.
2. **Diagonal**: else, try down-left / down-right (random order) if empty.
3. **Spread**: else, a cell with water directly above it may move sideways
   onto an open cell supported by real solid terrain.

4. **Settle**: none apply — drop out of the active set.

The pressure gate is important for a binary full/empty grid. An
unpressurized one-cell-deep surface would otherwise trade places with any
same-height air cell forever, continually remeshing a partially filled lake.
Water still spreads while a column has vertical head, then sleeps once it has
settled into a stable stepped surface.

A move reactivates only water neighboring both its old and new positions. This wakes water whose support moved earlier in the same tick without keeping air cells active.

This is deliberately full/empty, not fractional: total water-cell count is conserved by construction (moves are swaps, never creates/destroys), and rendering is exactly "is this cell water" with no threshold tuning. The cost is chunkier-looking flow than a pressure-equalized surface — accepted trade-off for the "heavily optimized, any voxel scale" goal.

## 5. Integration points

- **Ticking**: a second fixed-timestep accumulator in `vox-app::VoxApp`, independent of the 60 Hz physics loop, driving `FluidSim::tick` at a tunable `FLUID_DT` (~1/15 s to start).
- **Wake-on-edit**: one added line in the existing `main.rs` dirty-region drain loop, alongside the existing `phys.wake_region` call.
- **New tool**: mirrors `Tool::Bomb`'s shape (raycast → act) — a new `Tool` variant, a `Tools` method, a `HOTBAR` slot, one `apply_tools` match arm.
- **Rendering**: a new `water` material entry in `assets/materials/` (`solid = false`, `fluid = true`) — zero render-pipeline changes; opaque colored water for this pass. True translucency is a later follow-up reusing the particle pipeline's alpha-blend / no-depth-write pattern.
- **Accepted trade-off**: fluid-driven voxel edits flow through the same dirty-region mechanism as destructive edits, so flowing water will also call `phys.wake_region` on nearby debris every tick it's active, even without contact. Cheap (debris re-settles after a few quiet frames) and not incorrect — flagged, not solved, for v1. A separate dirty-region channel per edit source is the fix if it proves to matter.

## 6. Scale-invariance

Per the engine's existing unit contract (README: "every system is written against meters, not voxel counts"), the algorithm above is already scale-agnostic — it operates on grid adjacency alone, no physical constants in meters. The one thing worth a dedicated test (mirroring the existing terrain/tree scale-invariance tests in `vox-gen`): spread *speed*, measured in ticks-per-cell, is scale-independent by construction (always 1 cell/tick), which means water visually spreads slower in real-world terms at fine (0.1 m) scale than coarse (1 m) scale for the same tick rate. Whether that's acceptable or needs a scale-aware tick-rate/budget adjustment is worth an explicit test and a documented decision, not a silent gap.

## 7. Testing plan

All of `vox-sim` runs headless (mirrors every crate below `vox-render`):
- Algorithm unit tests: a single water cell falls under gravity; a pressurized blob spreads across a flat floor; shallow water and a partially filled basin settle; total cell count is conserved.
- Wake-on-edit: a settled lake reactivates when an adjacent cell is dug out using the exact dirty region; an edit far away leaves it settled.
- Budget: a tick given more active cells than the per-tick cap processes exactly the cap and carries the rest.
- Integration test in `vox-app` (mirrors the existing "drive the actual raycast-based blast entry point end to end" test): place a water blob, dig a channel, confirm water reaches the channel within N ticks.
- Scale-invariance test: same water placement at 0.1 m and 1.0 m scale produces the matching relative flow pattern (adjusted for the ticks-per-cell finding in §6).

## 8. Explicitly out of scope for this pass

- Lava, sand, fire (later additions per the roadmap, not designed against here beyond "the core shouldn't need a rewrite").
- Sloped/partial-fill rendering.
- Worldgen lakes/rivers.
- Buoyancy / rigidbody-vs-water interaction (debris currently passes through water like air).
- True transparency (see-through water).
- A separate dirty-region channel to avoid the debris-wake trade-off in §5.
