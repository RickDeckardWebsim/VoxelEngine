# Powder Materials — Mud & Sand as Granular Falls — Design Document

**Date:** 2026-07-09
**Status:** Implemented
**Builds on:** `2026-07-09-fluid-sim-design.md` (the CA fluid sim) and
`2026-07-09-water-refinement-design.md` (weathering, which creates mud and
sand). This document adds a new material behavior class — powders — to the
existing active-cell simulation.

**User request:** "lets have mud act as a powder, lets allow water to erode
rock to sand." The erosion (stone→sand) already exists; the new work is
making mud and sand behave as falling granular materials.

**Supersedes:** The water-refinement design doc §6 listed "Sand behaving as
a falling granular material" as out of scope. The user is explicitly
overriding that decision.

---

## 1. Decisions of Record

| Question | Decision |
|---|---|
| Scope | Both mud and sand become powders. Mud was solid (walkable); it becomes non-solid (falls, doesn't block). Sand was solid; same change. |
| Step rule | Fall + diagonal slide only: straight down if air below, else diagonal-down (randomized), else settle. No flow-horizon search, no pressure spreading. Produces stable pyramidal piles. |
| Architecture | Generalize `FluidSim` to handle powders alongside water. One active set, one tick loop, one wake. Each cell's material determines which step rule applies. |
| Material flag | New `powder: bool` on `MaterialDef`, parallel to `fluid`. A powder is `solid = false` + `powder = true`. Parsed from TOML as `powder = true`. |
| Mud solidity | Mud changes from `solid = true` to `solid = false`. Wet mud gives way under the player and physics bodies — it's a sludge, not a floor. |
| Weathering wake | Weathering's `set_voxel` conversions (dirt→mud, stone→sand) mark dirty regions. The existing `wake_region` call wakes powder cells too (not just water). The new powder cell activates and begins falling. |
| Momentum | Powders do not carry momentum. The momentum map is water-only. A powder cell that moves does not recruit neighbors into a current. |
| Contact events | Powder moves emit `ContactEvent::Fell` (downward) or `Flowed` (diagonal-down counts as fell since dest.y < pos.y — actually diagonal-down IS a fall, so all powder moves are `Fell`). `Vacated` on the old cell. Weathering can erode stone under falling sand just like under falling water. |
| Sleep guarantee | Same as water: a powder cell with no move settles and leaves the active set. A flat pile of powder sleeps. Zero cost at steady state. |

## 2. Material changes (`vox-core`)

`MaterialDef` gains:

```rust
/// Whether a `vox-sim` simulates this material as a powder (falls when
/// unsupported, piles at an angle of repose, no pressure-driven spreading).
/// Implies `solid = false`, but is distinct from `fluid`: a powder piles
/// rather than seeking a flat level.
pub powder: bool,
```

`RawMaterial` gains `powder: Option<bool>`, defaulting to `false`. Parsed
from TOML as `powder = true`.

`core.toml` changes:
- `mud`: add `solid = false, powder = true` (was solid, no powder flag).
- `sand`: add `solid = false, powder = true` (was solid, no powder flag).

## 3. Simulation changes (`vox-sim`)

### 3.1 `FluidSim` struct

```rust
pub struct FluidSim {
    active: FxHashSet<IVec3>,
    momentum: FxHashMap<IVec3, IVec3>,
    water: Voxel,
    /// Materials this sim treats as powders (fall + diagonal slide, no
    /// spreading). Empty if the asset set defines no powders.
    powders: Vec<Voxel>,
    rng: u64,
    events: Vec<ContactEvent>,
}
```

`FluidSim::new` takes `water: Voxel` as before. A new
`FluidSim::with_powders(water, powders)` constructor (or a builder/setter)
lets the app pass in the powder material ids. `new` passes an empty vec.

### 3.2 `step_powder`

```rust
fn step_powder(
    pos: IVec3,
    is_open: &mut impl FnMut(IVec3) -> bool,
    coin: bool,
    coin2: bool,
) -> Option<IVec3> {
    let down = pos + IVec3::NEG_Y;
    if is_open(down) {
        return Some(down);
    }
    // Diagonal slide: try the four diagonal-down cells in randomized order.
    // Unlike water's diagonal fall (which checks X-diagonals then Z-diagonals
    // separately), powder checks all four at once since there's no momentum
    // biasing. A powder on a slope slides down it one cell per tick.
    let (dx1, dx2) = if coin { (1, -1) } else { (-1, 1) };
    let (dz1, dz2) = if coin2 { (1, -1) } else { (-1, 1) };
    for &(dx, dz) in &[(dx1, dz1), (dx1, dz2), (dx2, dz1), (dx2, dz2)] {
        let diag = pos + IVec3::new(dx, -1, dz);
        if is_open(diag) {
            return Some(diag);
        }
    }
    None
}
```

`is_open` for powders: `world.in_bounds(p) && world.get_voxel(p) == AIR`.
Same as water — a powder can only move into air, never into water or other
powder. This means powder and water don't mix on the grid; water flows
around powder piles, and powder falls through air alongside water.

### 3.3 `tick` changes

The tick loop already snapshots active cells and filters by
`world.get_voxel(p) == water`. It now filters by "is this cell any simmed
material" (water or a powder), and dispatches to the right step function:

```rust
let cells: Vec<IVec3> = self
    .active
    .iter()
    .copied()
    .filter(|&p| {
        let v = world.get_voxel(p);
        v == water || self.powders.contains(&v)
    })
    .collect();
```

Inside the loop, per cell:

```rust
let v = world.get_voxel(pos);
let is_powder = self.powders.contains(&v);
let dest = if is_powder {
    step_powder(pos, &mut is_open, coin, coin2)
} else {
    // existing step_cell_with_momentum call
};
```

On a move, the event is always `Fell` for powders (all powder moves are
downward or diagonal-downward, so `dest.y < pos.y` always). Momentum is
not recorded for powders. The wake loop wakes water neighbors (existing)
and powder neighbors (new: `if self.powders.contains(&world.get_voxel(neighbor))`).

### 3.4 `wake_region` changes

Currently wakes only `self.water`. Now also wakes powder cells:

```rust
if world.get_voxel(p) == self.water || self.powders.contains(&world.get_voxel(p)) {
    self.active.insert(p);
}
```

### 3.5 `place_blob`

`place_blob` already takes a `water_material: Voxel` parameter. No change
needed — the caller can pass a powder material id and it'll fill and
activate cells. The tick loop handles them correctly.

## 4. App wiring (`vox-app`)

`main.rs`:
- `powder_materials()` helper: resolves all materials with `powder == true`
  from the registry, returns `Vec<Voxel>`. Empty vec if none — sim still
  works, powders just don't fall.
- Constructor: `FluidSim::with_powders(water_material(&registry), powder_materials(&registry))`.
- No frame-loop change: the existing `fluid.tick` → `weathering.tick` →
  `drain_dirty_regions` → `wake_region` loop handles powders automatically.

## 5. Testing plan

All headless in `vox-sim` unless noted:

- **Powder falls:** a single powder cell above a floor falls straight down
  and settles on the floor.
- **Powder piles:** a blob of powder falls onto a flat floor and forms a
  stable pile (active_count == 0) — not an infinite shuffle.
- **Powder slides on a slope:** powder on a one-block-step slope slides
  down diagonally.
- **Powder sleeps on flat ground:** a flat sheet of powder on flat ground
  has active_count == 0 (no lateral spreading to keep it moving).
- **Powder doesn't block water:** water beside a powder pile flows around
  it (water's `is_open` checks AIR, powder cells are not AIR).
- **Weathering activates powder:** when weathering converts dirt→mud (now
  a powder), the mud cell activates and falls if unsupported.
- **Conservation:** powder cell count is conserved across many ticks.
- **Integration (`vox-app`):** the existing `a_pool_on_grass_turns_its_bed_to_mud`
  test still passes (mud is now a powder, but it's under water so it
  doesn't fall — it piles on the bed).

## 6. Explicitly out of scope

- Powder-water displacement (water pushing powder, powder sinking in water).
- Angle of repose tuning (the diagonal-slide rule produces a natural ~45°
  pile; no configurable angle).
- Powder interacting with physics bodies (debris landing on powder, powder
  crushing under weight).
- Dust/particle effects from falling powder.
