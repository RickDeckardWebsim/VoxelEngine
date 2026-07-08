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
| `1`-`9` | Hotbar: 1 Dig, 2 Scalable Dig, 3 Bomb, 4 Death Laser (5-9 reserved) |
| Mouse wheel | Adjust tool radius (Scalable Dig / Bomb) or cycle build material (otherwise) |
| Left click | Use the active hotbar tool |
| Right click | Place selected material |
| `[` / `]` | Shrink / grow the Scalable Dig / Bomb radius (0.5-4 m) |
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

Slots 5-9 are reserved for future tools.

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
instead of a clean void.

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

Debris body meshing is threaded (`BodyMeshQueue`, mirroring the chunk
`RemeshQueue`) for large one-off spawns, but a body has no uploaded mesh
until its meshing job is collected, so routing *every* spawn through the
queue meant a body was invisible for the frame or two that took — and
since splitting a body during destruction always produces several small
fragments in the very same frame the original is despawned, that read as
the whole cluster flickering/vanishing on every hit. Fixed by meshing
small bodies (`INLINE_MESH_VOXEL_BUDGET`, 64,000 voxels) synchronously in
`upload_debris_mesh` instead: the stress example measures even a 40³ cube
at ~1.7ms average, cheap enough to eat inline. Only genuinely large spawns
still defer to the background queue.

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

**Add a whole new engine system** (e.g. the planned cellular-automata
fluid/fire sim, or ecosystem/creature life) — add it as a **new sibling
crate** at the `vox-gen`/`vox-physics` tier: it can depend on `vox-core` and
`vox-world` (and `vox-physics` if it needs bodies) without any existing
crate changing. This is deliberate: the layering was chosen so that
"add a concept" means "add a crate," not "thread a new dependency through
six existing files."

**Dependency policy**: third-party crates are infrastructure only (GPU,
windowing, math, threading, UI, data formats, error derives). Everything
that defines engine behavior is ours. Before adding a new crate dependency,
ask whether it *behaves* like part of the game (noise, ECS, physics) — if
so, it probably shouldn't be a dependency.

## Testing

```
cargo test              # ~190 tests, everything below vox-render runs headless
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
- A cellular-automata simulation crate (`vox-sim`): falling sand, fire,
  water — a sibling crate at the physics tier.
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
