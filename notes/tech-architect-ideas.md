# 5 Technical Platform Systems — Tech Architect's Round 2 Brainstorm

Grounded in the actual source. Every claim cites a file and line range.
These are NEW — none overlap with round 1 (CA sim, streaming, save/load,
ecosystem, weather). Each is a new crate at the gen/physics/app tier per
the architecture rules (README:274-280: "add a concept" means "add a
crate"), and each makes the engine more powerful as a *platform*, not
just a demo.

The five are ordered by how foundational they are. Cross-system tensions
are named explicitly where they exist — that's what makes this
architecture, not a feature list.

---

## 26. vox-cmd — Command Queue & Tiered Undo/Redo

**Tier: New crate at world/app tier.** Foundational — the editor (#27)
and the modding API (#28) both depend on reversible edits.

### The gap

Every edit today is a one-way fire-and-forget. `World::set_voxel`
(world.rs:117-144) writes a voxel, marks the chunk dirty, pushes a dirty
region, and returns nothing. `World::edit_box` (world.rs:161-223) does
the same for box fills — and the closure it calls (world.rs:196-203)
computes `cur` vs `new_v` per voxel, but **discards `cur`**. The diff is
right there, currently thrown away. The tool dispatch in
`VoxApp::apply_tools` (main.rs:451-504) consumes a `CarveOutcome`
(tools.rs:61-65) that tracks only *spawned* and *removed* body ids —
not the voxel diffs that caused them. There is no way to reverse any
edit. A misclicked bomb is permanent.

### What it is — and the honest scoping

Not "undo everything." A **two-tier** system, because the destruction
pipeline is chaotic and lossy in ways that make full undo a different
problem than world-edit undo.

#### Tier 1: World-edit undo (cheap, high-value, ships first)

Wrap `set_voxel` and `fill_box`/`edit_box` in reversible commands. Each
command captures the **voxel diff** — the `(IVec3, Voxel_old, Voxel_new)`
set — on `apply`, and swaps it back on `undo`. The `edit_box` closure
already has `cur` in scope (world.rs:197: `let cur = existing.map_or(AIR,
|c| c.get(local))`) — a command-aware variant collects `(local, cur,
new_v)` triples instead of discarding `cur`. `dig` and `place_voxel`
(tools.rs:238-262, 386-403) are single-voxel versions of the same
pattern — trivially reversible.

This covers: place, dig (single voxel), fill-box, paint, replace, erase,
prefab stamp. All of the editor (#27) operations. All creative building.
**This is 80% of the value at 20% of the cost.**

#### Tier 2: Destruction undo (hard, scoped honestly)

A bomb doesn't just edit the world — it carves the world (voxel diff),
**detaches** unsupported components into rigidbodies
(`detach_unsupported`, tools.rs:253), scatters them with blast impulses
(tools.rs:296-298), and those bodies then **fracture on impact**
(`body_destruction`, README:141-152), **settle via the solver**
(README:133-139), and eventually get **evicted** by
`evict_oldest_asleep_debris` (main.rs:960-982) when the count exceeds
`MAX_DEBRIS_BODIES = 200` (main.rs:143).

Undoing a bomb means reversing all of that. The destruction pipeline is
deliberately lossy: fragments below a cap are "discarded as dust" and
implausibly large ones are "left resident in the world" (README:122-124)
— two different caps, by design. You cannot reconstruct dust. The
eviction policy (main.rs:960) despawns the *oldest asleep* body to make
room — by the time you undo, the body that flew off may have been
evicted and its slot reused. **Full destruction undo is not a diff
problem; it's a state-snapshot problem.**

The honest scope: Tier 2 undo uses **physics snapshots** — before a
destructive command, snapshot the `PhysicsWorld` (body arena, positions,
velocities, sleep state). On undo, restore the snapshot + the voxel
diff. The physics world is small (≤200 bodies), so a rolling ring
buffer of the last N destructive commands is feasible. But this means
undo of a bomb rolls back *everything that happened since the bomb* —
not just the bomb. That's the honest tradeoff: you can't undo one bomb
in the middle of a chain without rolling back the chain. **Document
this as "rewind to checkpoint," not "undo one action," for destructive
commands.**

The alternative — inverse-operation undo (despawn spawned bodies,
restore their voxels) — **only works within the same frame**, before
physics has stepped. Once a body has moved, it's no longer where it was
spawned; once it's fractured, the fragments don't map back to the
original. I considered this and rejected it as too fragile to be
useful. Snapshots or nothing for destruction.

### Where it slots in

The command queue lives between input and tools. `apply_tools`
(main.rs:451) currently calls tool methods directly. The refactor: each
tool use constructs a `Command`, pushes it onto the queue, and calls
`command.apply()`. `Ctrl+Z` pops and calls `undo()`. The `CarveOutcome`
flows out the same way — the upload/remesh pipeline (main.rs:485-490)
doesn't change, it just reacts to the outcome that `apply` or `undo`
returns.

Tier 1 commands need no changes to vox-world beyond the `edit_box`
diff-collection variant. Tier 2 needs `PhysicsWorld` (or its
constituent `Body` arena + broadphase) to be `Clone` — not the case
today, but the structures are simple typed arenas (body.rs, broadphase.rs)
that can be made `Clone` without architectural change.

### Cross-system tension

Tier 2 undo and #29 (replay) **both need physics snapshots** and
**both care about determinism** — but for different reasons. Undo
needs snapshots because destruction is lossy; replay needs
determinism because it reconstructs state from inputs. They share
infrastructure (snapshot format, `PhysicsWorld::Clone`) but are not
the same system. Keep them separate; reuse the snapshot primitives.

### What it unlocks

- The editor (#27) is dead on arrival without undo.
- Creative building (round 1 #4) needs it.
- The modding API (#28) can expose `queue.push(command)`.
- Debugging: step back through edits to find what broke.

### Cost estimate

Tier 1: ~400 LOC (Command trait, SetVoxel/FillBox commands, ring
buffer, `Ctrl+Z` wiring, `edit_box` diff variant). Tier 2: ~600
additional LOC (PhysicsWorld Clone, snapshot ring buffer, destructive-
command checkpointing). **Ship Tier 1 first; Tier 2 is opt-in per
destructive command.**

---

## 27. vox-edit — In-Engine Editor with Brush Tools & Prefab Stamping

**Tier: New crate at app tier.** Depends on vox-cmd (#26) for undo.

### The gap

The tool system is a fixed 4-variant `Tool` enum (tools.rs:14-28):
`Dig`, `ScalableDig`, `Bomb`, `DeathLaser`. Adding a tool, per the
README's own documentation (README:268-272), requires: a new enum
variant, a method on `Tools`, a slot in the `HOTBAR` const
(tools.rs:36), and **a match arm in `VoxApp::apply_tools`**
(main.rs:455-480). Four touch points for one tool. There is no brush
shape abstraction — `scalable_dig` is hardcoded to a sphere
(tools.rs:335), `blast` to an `ExplosionShape` (tools.rs:264-270).
There is no way to stamp a saved shape at runtime. The closest thing
is `generate_trees` (trees.rs), which stamps tree structures — but
it's compiled into vox-gen, not a runtime tool.

### What it is

#### Brush tools: operation × shape

A `Brush` abstraction that separates *what* from *where*:

- **Operations** (the "what"): `Paint` (set material), `Replace` (swap
  one material for another), `Erase` (set AIR), `Smooth` (3D box blur
  over material ids — cheap, greedy meshing only cares about face
  adjacency), `Noise` (jitter material ids using vox-gen's existing
  noise, noise.rs). Each is a pure function
  `(center: IVec3, shape: &Shape) -> Vec<(IVec3, Voxel)>`.
- **Shapes** (the "where"): `Sphere` (existing blast shape),
  `Box` (maps to `fill_box`, world.rs:147), `Line` (voxel-traverse
  algorithm already exists in raycast.rs for the death laser),
  `Cylinder` (new — extrude a disc along an axis).

The existing `edit_box` (world.rs:161) is the perfect sink — it
resolves chunks once, collects diffs, marks dirty with neighbors. A
`BrushOp` produces a `(IVec3, Voxel)` list; the editor feeds it
through a new `World::apply_diff` that reuses `edit_box`'s chunk-
resolved loop but takes a pre-computed diff. **Every brush operation
flows through vox-cmd** (#26) — that's why the editor depends on it.

#### Prefab stamping

A prefab is a serialized voxel sub-volume. Save: `VoxelSlab::extract`
(vox-mesh/slab.rs, already used for meshing in main.rs:236) captures
a region into a compact `Vec<Voxel>` + dimensions + origin. Store it
as a binary asset. Stamp: translate to a target origin and write via
`apply_diff`.

This is exactly what `generate_trees` (trees.rs) does — it builds a
tree shape and stamps it into the world. The architecture is already
proven; vox-edit makes it a first-class runtime tool instead of
compile-time Rust. Rotation: 90° increments are a simple axis-swap on
the voxel array. Mirroring: swap and negate. Scaling: nearest-neighbor
(integer scale only — sub-voxel scaling doesn't make sense for voxel
data).

#### Selection box

A visual wireframe AABB. The debug overlay (vox-debug) already draws
line primitives — reuse that. The `raycast` against the world
(raycast.rs) gives the hit voxel; dragging extends the box. Used for
prefab save, box-fill, copy/paste, and area-clear.

### Where it slots in

vox-edit is a new crate at the app tier. It depends on vox-world, vox-
core, vox-mesh (for `VoxelSlab`), and vox-cmd. The `Tool` enum stays
for the gameplay tools — the editor is a separate mode, toggled like
Mario mode is today (main.rs:642-644), with its own input dispatch.
This keeps the fast-path FPS gameplay untouched.

### What it unlocks

- The engine becomes a level editor, not just a sandbox.
- Prefab libraries: share structures as files.
- The modding API (#28) can register custom brushes.
- Terrain gen could load prefabs — `generate_trees` →
  `generate_structures` (a village = N house prefabs along a road).
- Creative building (round 1 #4) gets real tools.

### Cost estimate

~1200 LOC. Brush trait + 5 ops + 4 shapes, prefab save/load, selection
box rendering (reuses vox-debug), egui palette panel. No render
pipeline changes — the editor draws into the same voxel world with the
same pipeline.

---

## 28. vox-mod — Data-Driven Content & Tool API

**Tier: New crate at app tier.** Depends on vox-core (extends the
registry), vox-edit (registers brushes), vox-gen (registers gen hooks).

### The gap — and the real platform play

Content is mostly hardcoded. Materials load from TOML (material.rs:50-
68) — the one data-driven surface — but `RawMaterial` is
`#[serde(deny_unknown_fields)]` (material.rs:60), so **every new
material property is a breaking parse change**. Tools are the least
extensible part: the `Tool` enum + `HOTBAR` const + 4 match arms
(README:268-272). Terrain gen is Rust (`TerrainGen` in terrain.rs), tree
placement is Rust (`generate_trees` in trees.rs). Adding "explosive
material" (round 1 #16) or "flammable material" (vox-sim) means editing
`MaterialDef`, `RawMaterial`, the TOML schema, the shader palette
buffer, and every consumer.

### The policy tension — and its reconciliation

The dependency policy (README:282-286) says: *"Everything that defines
engine behavior is ours. Before adding a new crate dependency, ask
whether it behaves like part of the game — if so, it probably shouldn't
be a dependency."*

A scripting runtime (Lua, Rhai) that defines tool behavior, gen rules,
or destruction logic **is engine behavior living in user scripts** —
it tensions the policy directly. A mod that implements "siege warfare"
in Lua has game logic in a third-party runtime. That's not
infrastructure; that's the game.

**The reconciliation: data-described tools with native Rust effectors,
not a scripting runtime.** The engine owns the *effectors* (the
primitive operations: carve, fill, spawn body, raycast, detach). Mods
*describe* how to compose them — in data, not in a scripting language.
This keeps all behavior-defining code in Rust (ours) while making the
*composition* data-driven (theirs). The line: "you describe what
primitives to call and in what order; we provide and execute the
primitives." No mod ever writes a loop, a branch, or a function — it
declares a recipe.

This is the same split the destruction pipeline already embodies:
`vox_physics::blast` (the effector) is ours; the *decision* to call it
(the tool match arm) is composition. vox-mod moves the composition into
data.

#### Tier 1: Open-ended material properties (no scripting, no behavior)

Extend `MaterialDef` with a `properties: HashMap<String, PropertyValue>`
field (`Float`/`Int`/`Bool`/`String`). The TOML schema switches from
`deny_unknown_fields` to a typed `properties` sub-table:

```toml
[[material]]
name = "gunpowder"
color = [0.2, 0.2, 0.2]
density = 1200
strength = 0.3
[material.properties]
explosive = true
blast_radius = 3.0
```

Systems opt-in: destruction checks `properties.get("explosive")`,
vox-sim checks `properties.get("flammable")`, the shader checks
`properties.get("transparent")`. **This is the schema change that
round 1 #3 (water), #16 (explosives), and vox-sim (flammable) all
need** — and it's backward-compatible: unknown properties are ignored
by systems that don't look for them. The `deny_unknown_fields` removal
is the one breaking change; everything else is additive. No behavior
is defined by data — data only *tags* materials, and Rust code (ours)
reads the tags.

#### Tier 2: Data-described tools (recipes, not scripts)

A tool definition is a TOML recipe that composes native effectors:

```toml
[[tool]]
name = "megabomb"
slot = 5
shape = "sphere"
radius = 5.0
[[tool.steps]]
effector = "carve"
shape = "explosion"
radius_m = 5.0
[[tool.steps]]
effector = "detach_unsupported"
[[tool.steps]]
effector = "apply_blast_impulse"
power = 50.0
```

The engine has a fixed set of native effectors (carve, fill, detach,
impulse, raycast, spawn_body, place). A tool recipe is an ordered list
of effector invocations with parameters. The `Tool` enum becomes a
thin wrapper: gameplay tools are compiled-in recipes (the current 4
tools become default recipes), mod tools are loaded recipes. The match
arm in `apply_tools` (main.rs:455-480) becomes a recipe executor — one
code path, not N arms. **This is the real fix for the "4 touch points
per tool" problem** (README:268-272): a new tool is a TOML file, not
four code edits.

What a recipe **cannot** do: branch on runtime state, loop, call
arbitrary functions. If a mod needs "if the hit material is stone, do
X, else do Y" — that's a new effector (ours, in Rust), not a script.
The effector API grows as mods need more primitives, but every
primitive is engine code under our control. **This is the
reconciliation: mods compose our primitives; they don't define new
ones.**

#### Tier 3: Data-driven generation (recipe stamps, not scripts)

A `ContentPack` is a directory: `materials/` (TOML), `prefabs/` (voxel
data from #27), `gen_rules/` (TOML: "stamp prefab X at density Y in
biome Z"). vox-gen exposes a `GenHook` registration point;
content packs register hooks that stamp prefabs per their rules.
`generate_trees` (trees.rs) becomes the built-in default hook. New
biomes, structures, features — all from data files. The hook signature
is `fn(chunk_key, rng) -> Vec<(IVec3, Voxel)>` — a pure function
returning a stamp diff. No scripting.

### Where it slots in

vox-mod is a new crate at the app tier. It depends on vox-core, vox-
world, vox-physics, vox-gen, and vox-edit. It does NOT depend on vox-
render — mods describe behavior, not rendering. The content pack
loader runs at startup alongside the existing `MaterialRegistry::load_
dir` (main.rs:156).

### Cross-system tension

Tier 2 (data-described tools) and #27 (editor brushes) **both compose
effectors** — a brush is a (shape × operation) recipe, a tool is a
(steps) recipe. They should share the effector registry. Design them
with a common `Effector` trait from the start, or you'll have two
parallel composition systems that can't interop. The editor's "paint
brush" and a mod's "paint tool" should be the same effector call.

### What it unlocks

- Tier 1 alone unblocks round 1 #3, #16, and vox-sim — no code changes
  to those systems, just property tags.
- Tier 2 fixes the "4 touch points per tool" problem: a new tool is a
  TOML file.
- Tier 3 lets modders ship new biomes and structures as data.
- The engine becomes a platform without surrendering the "behavior is
  ours" policy.

### Cost estimate

Tier 1: ~400 LOC. Tier 2: ~800 LOC (effector trait, 6-8 native
effectors factored out of existing tools, recipe parser, recipe
executor, default recipes for the current 4 tools). Tier 3: ~500 LOC
(content pack loader, GenHook registration, stamp pass). **No
scripting runtime. No new behavioral dependency.**

---

## 29. vox-determinism — Replay & Lockstep Networking Foundation

**Tier: New crate at app tier.** Depends on vox-core, vox-world, vox-
physics. **One system, not two** — replay and networking share a single
determinism seam, and designing them together is cheaper than designing
either alone.

### The gap — and why this is one system

The engine is *almost* deterministic. World generation is seed-drive
(`WorldConfig` seed → `TerrainGen::new`, main.rs:68). Physics is fixed-
step (`PHYSICS_DT`, main.rs:696-698). Blast and impact use explicit
seeds (`blast_seed`/`impact_seed`, main.rs:200-201, incremented per
use, tools.rs:283). The scale-invariance tests (README:104-107) already
enforce that "the same seed produces matching terrain/tree heights and
matching player behavior at both 0.1 m and 1.0 m" — determinism is a
tested invariant for gen and the character controller.

**But replay and lockstep networking need MORE than gen determinism —
they need full simulation determinism, including the destruction
pipeline and debris lifecycle.** That's where it breaks. Designing
replay alone would surface the determinism gaps; designing networking
alone would surface the *same* gaps. Designing them together means you
fix the determinism once and get both.

### The determinism gaps (the real work)

#### Gap 1: Variable physics step count

`timing.physics_steps` (main.rs:696) is computed from real elapsed time
in the platform layer — it varies per frame. Two runs with the same
seed and inputs will diverge if frame timing differs, which it always
does. **Fix:** the recording stores the step count per frame; playback
uses it directly. For live play, keep adaptive stepping; for replay/
lockstep, override to a fixed step count. This is a one-field change in
the frame loop, but it's the load-bearing one.

#### Gap 2: Nondeterministic debris eviction

`evict_oldest_asleep_debris` (main.rs:960-982) despawns the **oldest
asleep** body when count exceeds `MAX_DEBRIS_BODIES = 200`
(main.rs:143). "Oldest asleep" depends on **which bodies are asleep**,
which depends on **how many physics steps have run**, which depends on
**frame timing** (Gap 1). Even with Gap 1 fixed, the eviction *order*
depends on the `VecDeque<BodyId>` (`debris_order`, main.rs:132), which
is insertion-ordered — deterministic *given* the same spawn sequence.
But "which body is oldest asleep" changes if a body that was awake in
run A is asleep in run B (because B ran more steps before the
eviction check). **This is the single nondeterminism source that
breaks lockstep.**

Two fixes, with different costs:

1. **Make eviction deterministic (harder, better for lockstep):**
   replace "oldest asleep" with "oldest, period — but never evict a
   awake body" (skip awake bodies deterministically by *body id
   ordering*, not by sleep state-at-eviction-time). This makes the
   eviction decision depend only on spawn order, not on simulation
   state. The tradeoff: you may sit over budget longer (can't evict a
   sleeping body if an older awake body is ahead of it in the queue),
   but the existing code already accepts this (main.rs:956-959:
   "it's fine to briefly sit over budget rather than yank debris out
   from under the player mid-flight"). The bounded one-pass loop
   (main.rs:966) stays.
2. **Snapshot-based replay (easier, sufficient for replay, not
   lockstep):** don't try to make the simulation deterministic —
   instead, periodically snapshot full `PhysicsWorld` + `World` state
   and replay by restoring snapshots + interpolating. This is how most
   game replay systems actually work (deterministic replay is a
   research project; snapshot replay is engineering). **Recommend this
   for replay; recommend fix #1 only if/when you build lockstep
   networking.**

#### Gap 3: Float determinism across platforms

Rust floats are IEEE-754 and deterministic on the same platform/arch.
Cross-platform replay (x86 vs ARM, different compiler modes) is not
guaranteed. **Document the limitation: replays are same-machine-same-
arch.** Lockstep networking across heterogeneous hardware would need
fixed-point math or a validated-soft-float path — out of scope for v1.

#### Gap 4: HashMap iteration order

`World::chunks` is a `HashMap<IVec3, Chunk>` (world.rs:60).
`drain_dirty` (world.rs:259-261) and `drain_dirty_regions`
(world.rs:264-266) return unordered collections. If any sim-relevant
logic depends on iteration order of these, it's nondeterministic.
**Audit finding:** the debris eviction uses `debris_order` (VecDeque,
insertion-ordered — safe). Body stepping is per-body independent. Chunk
meshing is per-chunk independent. The dirty *set* feeds the remesh
queue, which sorts by distance-to-camera (remesh.rs:71-98) — **camera
position is input, which is recorded, so this is deterministic given
recorded input.** No sim-relevant path depends on HashMap order as far
as I can trace. Flag this as "audit passed for current code; re-audit
when adding new systems."

### What it is — the two consumers of one foundation

#### Replay (snapshot-based, ships first)

At the start of each frame, serialize the `InputState` subset that
affects simulation (key presses, mouse clicks, mouse delta, wheel
delta — NOT mouse position or window state) plus `blast_seed`/
`impact_seed` and the physics step count. Write to a ring buffer or
file. ~50 bytes/frame of input + 8 bytes of seeds/steps. A 60-second
clip at 60fps is ~180 KB.

Playback: reconstruct the world from the seed (`build_terrain_world`,
main.rs:61-74), then feed recorded input into `VoxApp::frame`
(main.rs:580) with the recorded step count override (Gap 1 fix).
Periodic `PhysicsWorld` snapshots (every ~60 frames) serve as
resync checkpoints in case of drift. `KeyR` toggles record, `KeyP`
toggles playback — same pattern as Mario mode toggle (main.rs:642).

#### Lockstep networking (deterministic, ships second, builds on the audit)

Once the determinism audit (Gaps 1-4) is done for replay, lockstep
networking is **plumbing on top**: a transport layer (raw UDP or a
thin reliable-ordered layer) that broadcasts each player's `InputState`
+ seeds + step count to all peers. Each peer runs the same `VoxApp::
frame` with the union of all players' inputs. The determinism work is
the hard part — the transport is ~500 LOC of socket code. **This is
why they're one system:** the 2000 LOC of determinism audit + snapshot
infrastructure is shared; only the transport is networking-specific.

**The eviction fix (Gap 2, fix #1) is only needed for lockstep, not
for snapshot-replay.** This is the key sequencing insight: ship replay
with snapshot-based resync (no eviction fix needed), then do the
eviction-determinism fix when you build lockstep. Don't pay the
eviction-fix cost until you need it.

### Where it slots in

vox-determinism is a new crate at the app tier. It wraps the
`InputState` that `VoxApp::frame` consumes. Recording: a `Recorder`
that `VoxApp` holds. Playback: a `Replay` that swaps the input source.
The `App` trait's `frame` signature doesn't change. The transport
(for networking) is a separate sub-module gated behind a feature flag
so headless/CI builds don't link socket code.

### Cross-system tension

This system **shares physics snapshots with #26 (undo)** — both need
`PhysicsWorld::Clone` and a snapshot ring buffer. Build the snapshot
primitives once in a shared location (vox-core or a small vox-snapshot
crate) and have both #26 and #29 consume them. Don't duplicate.

This system **is a prerequisite for round 1 #5 (Destruction Derby)**
which needs "networking or AI opponents." The determinism work makes
the networking path viable; without it, Destruction Derby is local-
only.

### What it unlocks

- **Bug reproduction:** record the 10 seconds before a crash, replay
  under the debugger.
- **Clip sharing:** a replay file (seed + inputs + snapshots) is tiny;
  anyone with the same engine version can watch it.
- **Networking foundation:** the determinism audit + transport =
  lockstep multiplayer.
- **Automated testing:** replay a session in CI, assert world state
  matches a golden snapshot. Catches regressions in gen/physics/
  meshing.
- **Time scrubbing:** with a ring buffer, scrub backwards. Pairs with
  #26's undo.

### Cost estimate

~1200 LOC. Recorder/Replay structs, input serialization (trivial —
`InputState` is plain data), the Gap 1 step-count override, periodic
snapshot serialization (~400 LOC shared with #26), the determinism
audit (the real work — tracing every sim-relevant path), and `KeyR`/
`KeyP` wiring. Lockstep transport: ~500 additional LOC, gated behind
a feature flag. The Gap 2 eviction-determinism fix: ~100 LOC, only
needed for lockstep.

---

## 30. vox-gpu — GPU Compute Meshing

**Tier: New crate at mesh/render tier.** Most architecturally
ambitious; biggest performance play; names a real tradeoff.

### The gap

Greedy meshing runs on CPU via rayon. The pipeline (remesh.rs):
`RemeshQueue::dispatch` (remesh.rs:71-98) extracts a `VoxelSlab` (a
copy of the chunk + 1-voxel neighbor border) per dirty chunk, sends
it to a rayon worker, which calls `mesh_slab` (vox-mesh/greedy.rs,
21.5KB), and returns a `MeshData` (vertices + indices). `collect`
(remesh.rs:100-113) uploads finished meshes, dropping stale results
via generation tracking (remesh.rs:6-8). Budget: `MAX_DISPATCH_PER_
FRAME = 64` (remesh.rs:21).

This works but is CPU-bound. A large edit (a bomb carving 10 chunks)
generates 10+ neighbor dirty chunks. At 64 dispatches/frame and ~1ms
per mesh on rayon, that's a ~10-frame backlog. The generation/
staleness tracking exists (remesh.rs:27-35) precisely because CPU
meshing can't keep up with rapid edits.

### The headless-testability tradeoff — named honestly

**vox-mesh is pure data-in/data-out, runs headless, serves both world
chunks and debris-body grids** (vox-mesh/src/lib.rs:3-4). The README
makes this a first-class guarantee: *"vox-mesh is pure data-in/data-
out — no GPU types, runs headless"* (README:91). The scale-invariance
tests (README:104-107) and ~190-test suite (README:291) depend on it.

**GPU compute meshing breaks this guarantee.** A compute shader cannot
run headless; it needs a GPU device. If vox-gpu replaces vox-mesh in
the live remesh path, the meshing logic is no longer unit-testable
without a GPU — and CI (README:291: "everything below vox-render runs
headless") can't test it.

**The reconciliation: keep vox-mesh as the headless reference path;
vox-gpu is an alternative backend, not a replacement.** vox-mesh
stays — it remains the testable, CI-able, pure-Rust meshing path.
vox-gpu is an *opt-in* GPU backend that the live `RemeshQueue` can
dispatch to instead of rayon. Both produce `MeshData`; the render
pipeline doesn't know which produced it. Tests run against vox-mesh
(always); the live app can use vox-gpu (when a GPU is present). This
costs duplication (the meshing logic exists in both Rust and WGSL)
but preserves the headless guarantee. **The alternative — making vox-
gpu the only path — would mean meshing bugs are only reproducible on
a GPU, in a window, with a frame loop. That's a regression in
debuggability that the current architecture was deliberately designed
to prevent (README:92: "Everything below vox-render runs headless").**

### What it is

#### The data flow

1. **Voxel data upload.** Each chunk's voxels (a `Chunk` is 32³ = 32K
   u16 = 64 KB, chunk.rs) live in a storage buffer, packed 2 voxels
   per u32. A dirty chunk is uploaded as a 64 KB write — replacing
   `VoxelSlab::extract` (CPU copy) with GPU border-fetch: the compute
   shader reads the chunk + 6 neighbors directly from the voxel
   buffer.
2. **Compute pass.** A workgroup processes one chunk's face culling.
   Greedy meshing on GPU is the hard part — the CPU algorithm
   (greedy.rs) is a sequential merge of coplanar quads per axis.
   Parallelization strategies (see "hard parts" below).
3. **Vertex buffer.** The compute shader writes `VoxelVertex` (the
   8-byte packed `Uint8x4 pos_ao + Uint8x4 norm_mat`,
   voxel_pipeline.rs) into a storage buffer with `atomicAdd` for the
   vertex count. The render pass binds this as a vertex buffer.
4. **Staleness.** The generation tracking (remesh.rs:27-35) moves to
   GPU: each chunk's voxel buffer carries a generation; the compute
   pass checks it before writing; a mismatch means the chunk was
   edited mid-mesh and the result is discarded.

#### The hard parts

1. **Greedy meshing on GPU is a research problem.** The CPU algorithm
   is inherently sequential (merge runs along an axis). Strategies:
   - **Per-face-plane workgroup:** one workgroup per (chunk, axis,
     slice). Each thread handles one row's merge. Needs workgroup
     shared memory + barriers. Complex but parallelizes the inner
     loop.
   - **Per-voxel emit + separate merge pass:** pass 1 emits one quad
     per exposed face (no merge) with atomic append. Pass 2 merges
     coplanar adjacent quads. Simpler pass 1, complex pass 2.
   - **Skip greedy, emit per-face (v1):** one quad per exposed face,
     no merging. Loses the quad-count reduction (a flat wall goes
     from 1 quad to 1024). But trivially parallel. **This is the v1
     — validate the data flow, measure whether the vertex increase
     matters** (modern GPUs handle millions of vertices; the current
     scene is thousands of chunks × hundreds of quads).
2. **Buffer management.** Today each chunk has its own vertex buffer
   (`upload_chunk`, voxel_pipeline.rs). GPU meshing needs either
   per-chunk storage buffers (same model, computed on GPU) or a
   single mega-buffer with per-chunk offsets (more efficient, needs
   atomic offset management + compaction).
3. **wgpu compute.** wgpu 0.20 supports compute pipelines + storage
   buffers (core, no features needed). A compute pipeline is a new
   `ComputePipeline` + bind group, parallel to the existing
   `RenderPipeline`.

### Where it slots in

vox-gpu is a new crate at the mesh/render tier. It depends on vox-
core and vox-render. It does **not** replace vox-mesh — it's an
alternative backend. The live `RemeshQueue` (remesh.rs) gains a
backend selector: rayon+vox-mesh (default, headless-testable) or
vox-gpu (when present). vox-mesh stays for the initial world mesh
(main.rs:226-252, a one-time parallel batch — rayon is fine for
that) and for all headless tests.

### Cross-system tension

vox-gpu and the shader hot-reload concern (below) **both need the
pipeline-mesh-store separation**. Today `VoxelPipeline` owns the
shader, the bind group, AND the chunk mesh store (`chunks:
HashMap<IVec3, GpuMesh>`, voxel_pipeline.rs:70) — all in one struct.
GPU compute meshing needs to write into the mesh store; hot-reload
needs to rebuild the shader without touching the mesh store. **Both
needs are blocked by the same coupling.** Fix it once: split
`VoxelPipeline` into a `ShaderPipeline` (shader + bind group +
camera uniform) and a `ChunkMeshStore` (the `HashMap<IVec3, GpuMesh>`
+ upload/remove/draw). Both vox-gpu and hot-reload become possible
without fighting each other.

### What it unlocks

- **Frame-perfect meshing:** edits appear next frame, not 10 frames
  later.
- **Scalability:** bounded by voxel memory, not CPU mesh throughput.
- **Streaming foundation:** round 1 #7 (vox-stream) needs fast
  meshing as chunks load.
- **Debris meshing:** the `body_mesh` queue (main.rs:89) has the same
  bottleneck. The `INLINE_MESH_VOXEL_BUDGET = 64_000` synchronous
  threshold (main.rs:148) becomes irrelevant when meshing is free.

### Cost estimate

~2000 LOC, high risk. Compute shader (WGSL), buffer management,
dispatch/collect logic, the v1 per-face emit. The greedy-on-GPU
optimization is a follow-up research task. **Recommend: (1) split
`VoxelPipeline` first (shared prerequisite, ~300 LOC, unblocks hot-
reload too); (2) prototype vox-gpu on one chunk to measure the per-
face vertex increase; (3) commit to the full refactor only if the
prototype validates.**

---

## Bonus: Hot-Reload Shaders — and why it's blocked

The shader is read once as a `String` (main.rs:157) and baked into a
`RenderPipeline` (voxel_pipeline.rs:85-88: `create_shader_module` →
pipeline). **But `VoxelPipeline` owns more than the shader** — it owns
the chunk mesh store (`chunks: HashMap<IVec3, GpuMesh>`,
voxel_pipeline.rs:70), the body mesh store (`bodies`, line 71), the
camera buffer, the bind group, and the voxel size (line 72). You can't
rebuild the pipeline to hot-reload a shader without destroying the mesh
store — which means re-uploading every chunk's geometry. That's a full
scene reload, not a hot reload.

**The structural cost is separating shader/pipeline from mesh-store.**
Split `VoxelPipeline` into:
- `VoxelShaderPipeline`: the `RenderPipeline` + camera buffer + bind
  group + material palette. Rebuild this on shader change — cheap
  (recompile shader, recreate pipeline + bind group, ~ms).
- `ChunkMeshStore`: the `HashMap<IVec3, GpuMesh>` + `HashMap<BodyMeshKey,
  GpuBodyMesh>` + upload/remove/draw methods. Persists across shader
  reloads — untouched.

Then hot-reload is: file-watch `assets/shaders/voxel.wgsl`, on change
rebuild `VoxelShaderPipeline`, swap it in. The mesh store and all
uploaded geometry survive. ~400 LOC total (the split + a file watcher
+ the swap). **This is too small to be a standalone "system" — but
the pipeline/mesh-store split it requires is a prerequisite for #30
(GPU compute meshing) too, so do the split as part of #30's
groundwork and hot-reload comes along for near-free.**

---

## Recommended Sequencing

| Step | System | Why this order |
|------|--------|----------------|
| 1 | vox-cmd Tier 1 (#26) | Bedrock. World-edit undo. No deps. ~400 LOC. |
| 2 | vox-mod Tier 1 (#28) | Unblocks round 1 #3/#16/vox-sim. ~400 LOC. Independent of #26. |
| 3 | vox-edit (#27) | Depends on #26. Turns engine into an editor. |
| 4 | vox-determinism replay (#29) | Snapshot-based. Shares snapshots with #26. The audit de-risks networking. |
| 5 | Pipeline/mesh-store split | Prerequisite for #30 AND hot-reload. ~300 LOC. |
| 6 | vox-gpu prototype (#30) | One-chunk prototype to de-risk. Keep vox-mesh as headless path. |
| 7 | vox-mod Tier 2+3 (#28) | Data-described tools + gen recipes. Depends on effector refactor. |
| 8 | vox-determinism lockstep (#29) | Needs eviction-determinism fix. Builds on replay's audit. |

#26-Tier-1 and #28-Tier-1 can run in parallel (no dependency). #27
depends on #26. #29's replay is independent of all; #29's lockstep
depends on the audit. #30 depends on the pipeline split. **The
pipeline split (#5) is the highest-leverage small task** — it
unblocks both GPU meshing and shader hot-reload.

---

## What I deliberately did NOT propose (and why)

- **A standalone networking crate (vox-net):** replay and lockstep
  share one determinism foundation (#29). Proposing networking
  separately would duplicate the determinism audit. Once #29's replay
  lands, lockstep networking is a transport module on top — a
  follow-up within the same crate, not a separate system.
- **A scripting runtime (Lua/Rhai) for mods:** tensions the dependency
  policy (README:282-286: "everything that defines engine behavior is
  ours"). Reconciled via data-described tools with native Rust
  effectors (#28 Tier 2) — mods compose our primitives, they don't
  define new behavior in a third-party runtime.
- **GPU physics:** the solver (solver.rs, 33KB) is a sequential-
  impulse constraint solver (README:133-139). GPU physics is a bigger
  research problem than GPU meshing and the body count (≤200) doesn't
  justify it. The bottleneck is meshing, not physics.
