# voxelengine

A from-scratch, modular voxel game engine — "Teardown-scale Minecraft."
Walk a procedurally generated world with trees, place and break voxels, and
blast structures into physically simulated debris that tumbles, collides,
and settles. The same engine runs at Teardown-scale (10 cm voxels) or
Minecraft-scale (1 m voxels), chosen per world at creation time.

Everything that defines engine *behavior* — voxel storage, noise, worldgen,
meshing, the rigidbody solver, destruction — is custom Rust, no game engine
or physics engine underneath. Third-party crates are infrastructure only
(GPU, windowing, math, threading, UI, data formats). See
[`docs/plans/2026-07-04-voxel-engine-design.md`](docs/plans/2026-07-04-voxel-engine-design.md)
for the full design rationale.

## Quickstart

```
cargo run -p vox-app --release
cargo run -p vox-app --release -- --scale 1.0 --seed 42
cargo run -p vox-app --release -- --help
```

Requires a DX12 or Vulkan-capable GPU (via wgpu). First launch generates
terrain and meshes it, which can take a second or two on the default 0.1 m
world.

## Controls

| Input | Action |
|---|---|
| `W A S D` | Move |
| Mouse | Look (click once to capture the cursor) |
| `Space` | Jump (walking) / fly up (noclip) |
| `Shift` | Fly down (noclip) |
| `Ctrl` (held, noclip) | 5x fly speed |
| `F` | Toggle fly / noclip |
| `1`-`9` | Hotbar: 1 Dig, 2 Scalable Dig, 3 Bomb, 4 Death Laser, 5 Place Water (6-9 reserved) |
| Mouse wheel | Adjust tool radius (Scalable Dig / Bomb / Place Water) or cycle build material (otherwise) |
| Left click | Use the active hotbar tool |
| Right click | Place selected material |
| `[` / `]` | Shrink / grow the Scalable Dig / Bomb / Place Water radius (0.5-4 m) |
| `B` | Spawn a wood debris cube in front of the player |
| `X` | Clear all sleeping (settled) debris |
| `F3` | Toggle the debug overlay (FPS, timings, tuning sliders) |
| `Esc` | Release the cursor, then exit |

### The hotbar

| Slot | Tool | Behavior |
|---|---|---|
| 1 | Dig | Breaks exactly the one voxel under the crosshair. |
| 2 | Scalable Dig | Carves a sphere of adjustable radius; severed material falls under gravity alone -- no impulse. |
| 3 | Bomb | Carves a sphere of adjustable radius and gives the debris an outward blast impulse. |
| 4 | Death Laser | An effectively infinite-range beam that tunnels straight through everything in its path in one shot -- no raycast gate, no impulse, just an instant, total cut. |
| 5 | Place Water | Fills a 0.5 m default, adjustable water source on the empty face of the targeted terrain; it starts flowing immediately. |

Slots 6-9 are reserved for future tools.

### CLI

```
voxelengine [--scale 0.1|1.0] [--seed N] [--extent X,Y,Z] [--help]
```

`--scale` is the voxel edge length in meters — this is the one setting that
switches the whole engine between Teardown-scale and Minecraft-scale.
`--extent` is the world's footprint in meters. See
[`crates/vox-app/src/args.rs`](crates/vox-app/src/args.rs).

## Architecture

Nine crates, strictly layered — nothing lower depends on anything higher:

```
vox-app        playable binary: game loop, player, tools, wiring
  |
vox-debug      egui debug overlay (HUD, timings, tuning) — quarantined
vox-render     wgpu pipelines, camera, chunk/debris draw, culling
vox-platform   winit window, input mapping, fixed-timestep loop
  |
vox-physics    rigidbody solver + destruction (carve -> connectivity -> debris)
vox-mesh       greedy meshing (pure functions, headless)
vox-gen        noise, terrain, trees (deterministic)
  |
vox-world      chunk storage, world edits, raycasting, dirty tracking
  |
vox-core       coordinates, voxel scale, material registry, config, errors
```

- `vox-world` knows nothing about rendering or physics.
- `vox-mesh` is pure data-in/data-out — no GPU types, runs headless.
- Everything below `vox-render` runs headless (unit-testable, CI-able).
- `vox-render` has no winit dependency; windows enter only as
  `wgpu::SurfaceTarget`. `vox-debug` owns egui entirely — vox-app never
  imports the `egui` crate directly.

### Unit contract

Gameplay-meaningful quantities are always in **meters/SI** in public APIs —
player height, tree height, blast radius, material density — converted to
voxel counts only at the point of use (`vox_core::coords`). This is what
makes one engine correctly run Teardown-scale or Minecraft-scale worlds:
every system is written against meters, not voxel counts, so changing
`voxel_size_m` doesn't require touching gameplay code. The scale-invariance
tests in `vox-gen` (terrain, trees) and `vox-physics` (character controller)
enforce this mechanically — the same seed produces matching terrain/tree
heights and matching player behavior at both 0.1 m and 1.0 m.

### The destruction pipeline

`vox-physics::destruction`: **carve** a shape from the world (a sphere for
Scalable Dig/Bomb, a capsule along a line for the Death Laser, recording what
was removed) -> **flood** 6-connected outward from each solid voxel exposed
by the removal, following the actual voxel shape (no artificial search box)
-> **detach** anything a flood proves is bounded into a `VoxelGrid`
rigidbody. Each flood is a proof, not a heuristic: it stops the instant it
reaches the world floor or exceeds a generous give-up cap (proof this
component connects to something far too large to be anything but ordinary
terrain) or exhausts naturally under that cap (proof it's a genuinely
isolated island, however large). Whether that proven-bounded component is
then small enough to spawn as one rigidbody is a *separate*, later decision
(`MAX_BODY_VOXELS`) — tiny fragments are discarded as dust, implausibly
large ones are left resident in the world; the two caps are deliberately
different numbers, since a real tree's canopy is routinely both "proven
disconnected" and "too big for one body" at once. Because there's no
bounding box to size, a severed tree trunk detaches its full disconnected
top even when it's standing on an enormous terrain mass, and a single voxel
broken out of ordinary terrain resolves in bounded time regardless of world
size — properties an earlier region-growing design, and then a follow-up
shared-cap bug, each got wrong in turn (see the module docs in
`destruction.rs` for what broke and why).

The rigidbody solver (`vox-physics::solver`) is a sequential-impulse solver
with warm starting, Baumgarte stabilization, Coulomb friction, and
island-consensus sleeping (touching bodies cross the sleep threshold and go
to sleep together, never individually mid-stack — see the commit history for
why that matters). Collision is voxel-grid-native: a body's surface voxels
are sampled directly against the world's or another body's voxel grid, no
convex-hull approximation.

### Destructible debris and impact fracture

Debris isn't a dead end: `vox-physics::body_destruction` runs the same
carve-then-split idea against an existing body's own grid instead of the
world. A body has no floor/anchor concept (it isn't "resting on" anything by
definition), so there's no connectivity proof to run — carving it just
splits its solid voxels into their 6-connected components, and *every*
component becomes its own fragment (subject to the same dust/oversize
policy), inheriting the parent's linear and angular velocity at its own
offset from the old center of mass. All four hotbar tools raycast against
both the static world and every live body (`Tools::raycast_scene`) and pick
whichever is closer, so debris is just as breakable as terrain.

Beyond direct tool hits, `PhysicsWorld::step` reports each body's hardest
single contact that step (`ImpactEvent`: world point + peak normal impulse).
`fracture_radius_vox` (in `vox-app/src/main.rs`, kept pure and unit-tested
apart from live GPU/registry state) compares the implied impact speed
(impulse/mass) against the actual material at that point's `strength`,
scaled by the live-tunable `fracture_sensitivity` — *higher* strength means
a *higher* threshold (harder to trigger at all), the same "higher survives
more" convention every destruction tool already uses. With the core
material set (leaves 0.5, wood 4.0, stone 8.0) that reads as: leaves give
way at the slightest bump, wood needs a real fall or throw, stone needs a
genuinely hard impact. Every material's radius starts from the same small base bite
(`FRACTURE_RADIUS_VOX`) at its own bare threshold — a tiny hit always
produces a tiny chip, never "an orb of voxels deleted from space" — and
only *grows* past that base as the impact clears the threshold by more,
scaled by how fragile the material is (stone's growth factor bottoms out
at zero, so a hard hit still only produces the base bite; leaves grows the
fastest). An earlier version scaled the *entire* radius by the per-material
factor instead of just the growth, so even leaves' gentlest fracturing hit
already carved a fixed 5x radius — "a tiny hit blows out a huge chunk."
(An even earlier version divided by strength instead of multiplying,
which inverted the whole scale the other way — stone fractured *more*
easily than wood.)

Impact fracture also scatters a sample of what it carves as small flying
debris chips (`body_destruction::carve_body_sphere_at_impact`, mirroring
the bomb's own chip idea below but launched along the actual contact push
direction and scaled by impact speed instead of blast power) rather than
letting it all simply vanish — a graze knocks a couple of chips loose, a
violent hit sends several flying, matching Teardown's "satisfying mess"
instead of a clean void. Two more things had to be reined in to actually
get that: `MAX_FRACTURE_RADIUS_VOX` puts a hard ceiling on the radius
regardless of how fragile the material or how violent the impact (the
per-material growth math could otherwise legitimately reach several
voxels on something as fragile as leaves, which read as "a chunk of the
tree's canopy vanished into a smooth spherical void," not a crumble); and
`MAX_IMPACT_CHIPS` was raised from an initial `6` to `24` so a fracture at
that radius actually scatters enough visible pieces to read as *something*
breaking apart, not a handful of specks next to an empty hole. Bodies
below `MIN_FRACTURE_BODY_VOXELS` are terminal rubble and never fracture
again — see the performance notes for the runaway cascade that rule
breaks.

A body resting or settling after a hit still needs a real contact impulse
every substep just to hold it up against gravity, and that impulse looks
identical, frame to frame, to the one from a body that just landed hard —
without a further check, a low-strength material (leaves) kept
re-reporting that steady load as a fresh impact on *every single settling
frame*, continuously re-fracturing and re-meshing itself for as long as it
took to fall asleep. This is what "flickering while breaking apart" turned
out to actually be — not a shading bug, a real repeated destruction event.
Fixed by tracking each contact's pre-solve closing speed
(`Contact::approach_speed`) alongside its accumulated impulse, and gating
impact-fracture eligibility on it (`MIN_IMPACT_APPROACH_SPEED_M_S`):
a resting/settling contact's closing speed stays near zero every step, a
genuine collision's does not.

### Bomb debris chips

A plain carve leaves nothing behind but a void, which reads as "the
material vanished" rather than "something exploded." `blast` (in
`vox-physics::destruction`) samples a capped, deterministic fraction of
whatever it just carved away and turns those voxels into small flying
L-shaped debris chips instead of clearing them to air outright, launched
outward from the blast center at a modest, hard-capped speed. Two things
had to be fixed to get here, both left as regression tests: a literal
single-voxel chip is a physics degenerate case (its only contact point
sits exactly on its own center of mass, so friction can never generate
torque, and any spin it's given never damps out); a straight two-voxel bar
fixes that for rotation *across* its length but is still degenerate for
spin *around* its own long axis. An L-shaped chip (no straight line through
all its voxel centers) has no such axis.

### Performance notes

Two hot paths were measured (`cargo run -p vox-app --release --example
stress`) and fixed at the root, not patched around:
- `World::edit_box` resolves each *chunk* a bulk edit touches once instead
  of once per voxel (`get_voxel`/`set_voxel` each cost a hash-map lookup);
  `carve_sphere`/`carve_capsule` build on it. Cut a large blast's cost by
  ~5x.
- The connectivity flood (`destruction::flood_from`) explores voxels in
  best-first order toward the world floor, not plain breadth-first, so
  proving "this connects to solid ground" doesn't require exploring a full
  sphere around the edit first when the floor is a short vertical hop away
  — the overwhelmingly common case. Cut the worst case (a beam tunneling
  through a massive terrain slab) by another ~2.8x on top of the above.

Every debris body is its own GPU buffer set and its own draw call
(`VoxelPipeline::draw_bodies`) — fine at modest counts, but unbounded over a
play session once bombs scatter debris chips: nothing despawned old debris,
and bodies were never frustum-culled at all. Two fixes address sustained
frame-time degradation specifically (as opposed to the momentary per-blast
cost above): `VoxApp` now caps live debris at `MAX_DEBRIS_BODIES`, evicting
the oldest already-*asleep* body first (never one still actively
flying/settling) via the pure, unit-tested `evict_oldest_asleep_debris`;
and `draw_bodies` now frustum-culls debris exactly like `draw_chunks`
already did for chunks. Also fixed: `blast`'s debris-chip sampler was
fully sorting *every* removed voxel (tens of thousands, for a large
terrain blast) just to keep ~40 — replaced with a partial selection
(`select_nth_unstable_by_key`), O(n) instead of O(n log n) over the whole
removed set.

A small body has a disproportionately tiny moment of inertia, so an
off-center contact (landing on one corner) can spin it up far harder than
the same impulse would a large body — and unlike linear velocity (already
capped by `MAX_SPEED`), angular velocity had no ceiling at all. A 2x2x1
debris chip landing corner-first would settle into a *stable* 50-60 rad/s
that never decayed: never quiet enough to sleep (so it cost full
broadphase/narrowphase/solver/render work forever, directly worsening "lots
of debris causes lag"), and covering ~50 degrees of rotation *per physics
step* at 60Hz, well into the range where the render-side slerp between a
step's start/end orientation reads as visible judder rather than a smooth
spin — this is what "rotation causing stutters and glitches" on small
debris specifically turned out to be. Fixed with `MAX_ANGULAR_SPEED_RAD_S`,
the same idea as `MAX_SPEED`, applied both where gravity/velocity are
integrated and again after the solver's impulse resolution each substep so
a spike never persists past the substep it occurred in.

The broadphase (`broadphase::Broadphase::candidate_pairs`, a spatial hash
over body AABBs) used to allocate a fresh hash map, hash set, and output
vector on every single call — and it was called three times per physics
step (once per substep, plus once more for the end-of-step sleep-island
grouping), so with `MAX_DEBRIS_BODIES` debris around that was real,
repeated allocation/rehashing overhead for no behavioral benefit. Fixed by
making `Broadphase` a persistent, reusable piece of `PhysicsWorld` state
(`.clear()`-and-reuse instead of allocate-fresh) and having the
sleep-island grouping share the same scratch state instead of rebuilding
its own from scratch. The narrowphase's per-pair contact staging buffer is
hoisted and reused the same way.

Three deeper fixes came out of the "small debris rotation looks glitchy,
and lots of debris still lags" report, each verified headlessly:
- **The fracture cascade** (`MIN_FRACTURE_BODY_VOXELS` in `vox-app`): a
  fracture scatters 3-voxel chips; a chip's next bounce trivially clears a
  fragile material's fracture threshold (leaves need only 0.5 m/s of
  delta-v — every bounce qualifies); and every fragment of a 3-voxel body
  is below `DEBRIS_MIN_VOXELS`, so the chip vanished as dust while fresh
  chips spawned from what was removed — grid clones, component labeling,
  remeshing, and GPU buffer churn on *every bounce of every chip*,
  compounding forever. Bodies below the new floor are terminal rubble:
  they bounce and settle, never re-fracture. This was both the visible
  popping/vanishing glitches and a large share of the sustained lag.
- **Angular damping** (`ANGULAR_DAMPING_AIR` + `ANGULAR_DAMPING_ROLLING`):
  the solver had *no* mechanism that removes rotation except contact
  friction, which has no leverage when contact points sit near the spin
  axis — so debris could spin at the sleep threshold indefinitely, keeping
  its whole contact island awake (lag) and visibly twitching (glitches).
  Everyone now gets a whisper of air drag; *small* bodies (by surface
  point count, so a tipping tree is exempt and still falls at full speed)
  get strong rolling resistance while in contact, so grounded rubble stops
  tumbling almost immediately, sleeps, and frees its island to sleep too.
- **World-inertia caching** (`Body::inv_iw`): the impulse loop re-derived
  the world-space inverse inertia tensor (quaternion→matrix + two mat3
  multiplies) ~25 times per contact per substep; rotation only changes at
  substep boundaries, so it's now computed once per body per substep.
  Measured ~16% off the settling-pile average (A/B, 300 bodies), and the
  worst single step dropped from ~47ms to ~21ms across this round's
  changes combined.

For lower-end machines, two engine-wide constant-factor passes (no content
or behavior change):
- **`vox_core::fxhash`** (a dependency-free FxHash): Rust's default map
  hasher is SipHash, which pays for collision-flood resistance a game
  hashing its own voxel coordinates doesn't need. Every hot collection now
  uses the fast hasher — above all the world's chunk map, consulted on
  *every voxel read engine-wide* (contacts, raycasts, carves, floods), plus
  the broadphase cells/dedup, the solver's warm-start and impact-peak maps,
  the connectivity flood's visited set, and the render/remesh bookkeeping.
  Steady-state physics step cost measured roughly **2x faster** across pile
  sizes (e.g. 100-body settling p50 0.175ms → 0.072ms), and it's also more
  deterministic (FxHash has no per-process random seed).
- **Chunk-caching in `world_contacts`**: each surface point costs up to 7
  solidity queries, each formerly a fresh chunk-map lookup; consecutive
  points are spatially coherent, so a `SolidLookup` (the same cache the
  destruction flood already used) amortizes nearly all of them away. A
  falling tree — thousands of surface points, every substep until it
  sleeps — is exactly this case.

"Small debris caught under large debris makes the large debris react
oddly" led to the solver's single deepest fix: **split-impulse penetration
recovery**. The old Baumgarte term turned penetration depth into a
*velocity* target inside the impulse solve, and a contact will spend
however much impulse the touching bodies' masses require to reach a
velocity target — real momentum injected from nothing. Debris chips spawn
overlapping the fragment they chipped off of (by up to a voxel — routine,
not exotic), and a probe reproduced the report exactly: one three-voxel
chip wedged under a settled 5.6-tonne block gave the solver two
contradictory demands (floor: "chip up"; block: "chip down relative to the
block") that it could only satisfy by lifting the block — ramping it to
~1 m/s and shoving it centimeters. Penetration is now recovered
*positionally* (bodies moved apart directly, weighted by inverse mass,
sequentially so hundreds of same-face contacts converge to one correction
instead of stacking), which by construction cannot add kinetic energy: the
same probe now shows the block peaking at 0.000 m/s with 0.1 mm of drift,
and every rest-height/stacking/settling test passes unchanged. Relatedly,
warm-start keys changed from (body, target *cell*, face) to (body, target,
*surface-point index*, face): several points can alias into one cell, and
colliding keys made the warm-start map replay one contact's accumulated
impulse into every contact sharing the key — plus point identity survives
sliding across cell boundaries, so warm starting persists where it used to
reset.

Debris body meshing is threaded (`BodyMeshQueue`, mirroring the chunk
`RemeshQueue`) for large one-off spawns, but a body has no uploaded mesh
until its meshing job is collected, so routing *every* spawn through the
queue meant a body was invisible for the frame or two that took — and
since splitting a body during destruction always produces several small
fragments in the very same frame the original is despawned, that read as
the whole cluster flickering/vanishing on every hit. Fixed by meshing
small bodies (`INLINE_MESH_VOXEL_BUDGET`, 200,000 voxels — raised from an
initial 64,000 once it turned out a felled tree's trunk-plus-canopy is one
connected mass that easily clears that: a single canopy ellipsoid alone is
tens of thousands of voxels) synchronously in `upload_debris_mesh`
instead: the stress example measures even a 40³ cube at ~1.7ms average,
cheap enough to eat inline, and extrapolates to only ~5ms at the new
budget — one rare tree-felling hitch, not a per-hit cost.

For whatever still clears even that (raised) budget, `VoxApp::replace_body`
closes the gap a different way instead of just accepting it: the old
mesh is kept exactly where it is (frozen, at its last known transform)
until *every* one of its replacement fragments' async meshes has arrived,
rather than being removed the instant the old body despawns. This matters
specifically because a large mass like a tree trunk stays that large
across *many* subsequent hits, not just the first one — the earlier,
budget-only fix made the initial felling instant but every later hit on
the same trunk would still pop it out of existence for a frame, which is
exactly what "invisible for a solid frame every time damage is applied"
turned out to mean.

## Extending the engine

The whole point of the crate layering is that new systems are *additions*,
not edits to existing ones.

**Add a material** — pure data, no code: drop a `.toml` file into
`assets/materials/`. Every `*.toml` in that directory loads in
case-insensitive filename order (see the header comment in
[`assets/materials/core.toml`](assets/materials/core.toml) for the schema).
Duplicate names across files are a load error, not a silent override.

**Add a tool** — add a variant to `Tool` in
[`crates/vox-app/src/tools.rs`](crates/vox-app/src/tools.rs), a method on
`Tools` implementing it, a slot in the `HOTBAR` table, and one match arm in
`VoxApp::apply_tools` (in `main.rs`) to wire it to input. `Tool::Bomb` is the
fullest example: raycast -> `vox_physics::blast` -> done.

**Add a whole new engine system** (e.g. ecosystem/creature life) — add it as
a **new sibling crate** at the `vox-gen`/`vox-physics` tier, the way
`vox-sim` (the cellular-automata fluid sim) already does: it can depend on
`vox-core` and `vox-world` (and `vox-physics` if it needs bodies) without any
existing crate changing. This is deliberate: the layering was chosen so that
"add a concept" means "add a crate," not "thread a new dependency through
six existing files."

**Dependency policy**: third-party crates are infrastructure only (GPU,
windowing, math, threading, UI, data formats, error derives). Everything
that defines engine behavior is ours. Before adding a new crate dependency,
ask whether it *behaves* like part of the game (noise, ECS, physics) — if
so, it probably shouldn't be a dependency.

## Testing

```
cargo test              # ~242 tests, everything below vox-render runs headless
cargo clippy --all-targets -- -D warnings
cargo run -p vox-app --release --example stress   # headless perf probe, not a test
```

Everything below `vox-render` is unit-tested, including the physics solver
(single/multi-body settling, stacking, a confined pile stress test),
destruction (bridge/pillar severing, floating-fragment detection, a severed
structure atop a huge terrain mass, and a bounded-time check on breaking a
single voxel deep in terrain), procedural generation
(deterministic noise, terrain and tree scale-invariance), and the greedy
mesher (watertightness verified against a brute-force reference on random
inputs). `vox-app`'s tools/CLI have their own tests, including one that
drives the *actual* raycast-based blast entry point end to end: blast a
pillar's base, confirm the upper section detaches, tumbles, and sleeps.

## Roadmap (post-MVP, not built)

- Streaming chunk load/unload beyond the current finite-but-sparse world map
  (the `HashMap<ChunkPos, Chunk>` storage is already streaming-ready).
- Palette-compressed chunks and an RLE binary save format (the chunk
  storage's `Uniform`/`Dense` enum already hides this behind `get`/`set`).
- ~~A cellular-automata simulation crate (`vox-sim`): falling sand, fire,
  water — a sibling crate at the physics tier.~~ **Implemented:** a
  cellular-automata fluid sim (`vox-sim`) with active-cell sleeping, 8-
  direction drop-search, momentum memory for cohesive flow, water-driven
  weathering (grass→dirt→mud, stone→sand erosion with waterfall boost,
  mud drying), and powder materials (mud and sand fall and pile at an
  angle of repose). See `docs/plans/2026-07-09-fluid-sim-design.md`,
  `docs/plans/2026-07-09-water-refinement-design.md`, and
  `docs/plans/2026-07-09-powder-design.md`.
- An ecosystem/life crate: creatures, growth, populations.
- Structural stress (load propagation -> creaking collapses) layered on top
  of the existing connectivity pass.
- Debris re-freezing into the world once fully settled, and debris
  re-fracturing under a second hit.
- A raytraced renderer path behind the existing renderer interface;
  shadow maps; transparency.
- **Dependency modernization**: wgpu 0.20 / winit 0.29 / egui 0.28 are
  pinned to the exact combination proven to compile and render on this
  machine at MVP time. Upgrading (e.g. to wgpu 26+, winit 0.30) is a
  contained follow-up isolated to `vox-render`, `vox-platform`, and
  `vox-debug` — no other crate touches these APIs directly.

## License

MIT OR Apache-2.0.
