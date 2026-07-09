# 5 New Physics & Simulation Ideas — Physics Engineer's Round 2 Brainstorm

Grounded in the actual `vox-physics` crate: a `PhysicsWorld` (solver.rs:83-93)
holding a slot-arena of voxel `Body`s (body.rs:252-275), advanced by a
fixed-timestep sequential-impulse solver with substeps (solver.rs:323-477).
Each substep: integrate gravity into awake bodies (solver.rs:331), generate
world contacts as surface-sample-points-vs-voxel-grid (contact.rs:114-161) and
body-body pairs via `pair_contacts` (contact.rs:215-324), warm-start from the
previous substep's accumulated impulses keyed by `(body, voxel, face)`
(solver.rs:388-397), run `SOLVER_ITERS=8` velocity iterations with Baumgarte
positional correction + Coulomb friction on two tangents (solver.rs:401-423),
then integrate positions (solver.rs:454-470). The destruction pipeline
(destruction.rs) is carve → 6-connected seed-flood → detach-unsupported-as-
rigidbody; body_destruction.rs does the same against a body's own grid. The
broadphase is a spatial-hash over body AABBs (broadphase.rs:14-54). Materials
are `MaterialDef { name, color, jitter, density, strength, solid }` loaded from
`#[serde(deny_unknown_fields)]` TOML (material.rs:23-37, 59-68), so any new
material key is a coordinated schema change, not a casual add.

**Feasibility tiers:**
- **Solver-extension** — extends the existing sequential-impulse loop or
  substep with a new constraint/force type. No new pipeline, reuses the
  impulse machinery.
- **New-sibling-crate** — new crate at the gen/physics tier (per design doc
  §12), depends on vox-core/vox-world only, mounts alongside vox-physics.
- **Schema-change** — requires touching `MaterialDef` + `RawMaterial`
  (`deny_unknown_fields`), the TOML assets, and every consumer. The real
  structural cost, not just "add a field."

Round 1 already covers: chain-reaction explosives (#16), sustained-load
structural failure (#17), fluid sim (#18), terrain erosion (#19), buoyancy
(#20), and fire/water/falling-sand CA (#6). None of the five below overlap.

---

## 1. vox-rope — Rope, Chain & Cable Constraint Networks (Solver-extension + new sibling crate)

**Tier: Solver-extension + new sibling crate.** A new `vox-rope` crate at the
physics tier (depends on vox-core + vox-physics) that adds **distance
constraints** between rigidbodies as a first-class pass inside the existing
sequential-impulse solver — not a parallel solver.

A rope is a chain of small voxel-bodies (1-3 voxels each, spawned via
`Body::from_grid`, body.rs:278-312) connected by distance constraints with a
rest length and a max-stretch break threshold. Chains hang under gravity,
swing, go taut under load, and snap when tension exceeds the rope material's
strength — reusing `MaterialDef::strength` (material.rs:34) the exact same way
impact fracture already does (the `ImpactEvent.impulse / body.mass()` vs
strength comparison in vox-app).

**The structural cost is small because the solver is already the right shape:**
- The substep loop (solver.rs:401-423) iterates `SOLVER_ITERS` times over a
  `Vec<Contact>`. A distance constraint is the *same math* as a contact with a
  different rest condition: `effective_mass` (contact.rs:96-112) computes
  `k = 1/m + n·((I⁻¹(r×n))×r)` for arbitrary attach points — a rope attach
  point is just `r_arm`/`r_arm_b` on two bodies with a normal along the
  constraint axis. The lambda accumulation and `apply_contact_impulse`
  (solver.rs:481-496) are reused verbatim; the only new logic is the bias term
  (distance error instead of penetration depth) and the break check
  (`acc_n > break_threshold` → remove constraint + spawn the freed end as a
  detached body via the existing `split_components` path).
- Warm starting (solver.rs:388-397) keys on `(body, voxel, face)`; rope
  constraints key on `(body_a, body_b, constraint_id)` — a parallel
  `HashMap<ConstraintKey, f32>` in the same `self.warm` pattern. No second
  solver, no second integration pass.
- Gravity is already applied per-substep (solver.rs:331); rope bodies get it
  for free. The `MAX_SPEED` clamp (solver.rs:81, 332) already bounds a
  whiplashing rope end.

**What to change:**
- New `vox-rope` crate: `DistanceConstraint { body_a, body_b, r_a, r_b,
  rest_length, break_impulse }` + a `RopeBuilder` that spawns N small
  `VoxelGrid` bodies and wires N-1 constraints between consecutive COM-offset
  attach points.
- `PhysicsWorld` gains a `constraints: Vec<DistanceConstraint>` and the
  substep gains a constraint-solve loop interleaved with the contact loop
  (same `SOLVER_ITERS`). This is the one touch into vox-physics itself — a
  `pub` method or an extension trait, since vox-rope sits above vox-physics.
- Tension for break detection = the accumulated normal impulse `acc_n`
  (already computed for contacts, solver.rs:406-408). When it exceeds
  `break_impulse` (derived from `MaterialDef::strength`), the constraint is
  removed; the now-freed bodies are already in the solver and just fly apart.
- Visual: a rope body's mesh is its `VoxelGrid` via the existing
  `body_mesh.rs` pipeline — zero new rendering.

**Gameplay:** swing across chasms on chains, wrecking balls (heavy body on a
rope — constrained to an arc), chain-link bridges that sag under load and
snap, grappling-hook tool (a one-body distance constraint between the
kinematic player and an anchor — the player isn't a `Body`, so this is a
kinematic-to-dynamic constraint where the player's "attachment point" is a
fixed world position that the player updates each frame). Mario: grab a chain
and swing, yank debris by a tether.

**Why it's not round 1:** round 1 #17 (sustained-load failure) is a
material-failure model for beams under static stress — it has no constraint
solver, no body-to-body linkage, no ropes. Distance constraints are a
fundamentally new solver capability that round 1 never touches.

---

## 2. vox-cloth — Cloth Simulation for Flags, Capes & Tarps (New sibling crate + small render path)

**Tier: New sibling crate + a small isolated render path.** A `vox-cloth` crate
at the physics tier. A cloth sheet is **not** a rigidbody — it's a mass-spring
grid: a 2D array of point masses connected by structural (axial), shear
(diagonal), and bend (skip-one) springs, integrated with the same fixed
`PHYSICS_DT = 1/60` timestep (consts.rs:9).

**The cloth reuses the existing collision primitives directly:**
- Each cloth node is treated as a tiny sphere that gets pushed out of solid
  voxels through the nearest empty face — this is *exactly* what
  `world_contacts` does for rigidbody surface points (contact.rs:114-161):
  the "point inside a solid voxel → push out via nearest empty face" branch
  (contact.rs:125-136) and the "point near a solid face → contact before
  penetration" branch (contact.rs:148-158). A cloth node is a degenerate
  body with one surface point, zero inertia, and `half_voxel` replaced by a
  smaller cloth-node radius. `face_dist` (contact.rs:77-94) and `FACE_DIRS`
  (contact.rs:58-66) are reused as-is.
- Cloth-vs-rigidbody uses `pair_contacts` (contact.rs:215-324) with the cloth
  node as the "sampler" (fewer points → it samples, per solver.rs:353). The
  body is the target. The contact assembly math is identical.
- Wind force per node = the same wind impulse the weather system (#10) applies
  to debris bodies, but per-point instead of per-body. No new force model.

**Materials & tearing:** cloth stiffness and tear threshold come from
`MaterialDef` — a new cloth material with low `density`, high stretch, low
`strength` (a flag rips before a steel mesh does). When a spring's strain
exceeds the material's strength, the spring is removed and a tear propagates
across the grid. This is a per-spring break check, the cloth analog of the
rope's per-constraint break (idea #1) and the impact fracture's per-voxel
break — same `MaterialDef::strength` convention.

**The one piece that needs render-side work** (and it's small): cloth is a
thin triangle mesh, **not** the voxel greedy mesher. A new lightweight render
path in vox-render that takes the per-node positions (a `Vec<Vec3>` updated
each frame on the CPU) and uploads them as a vertex buffer + a fixed index
buffer of `2*(N-1)` triangles per row. This is an isolated pipeline (one
vertex buffer, one index buffer, one draw call per cloth sheet), not a
restructure of the voxel pipeline — the voxel `RenderPipeline` and its
interleaved chunk buffers are untouched.

**Gameplay:** flags on castles that flap in storm wind (#10), capes for the
player/Mario, tarps covering structures that get shredded by explosions, nets
that catch falling debris. Mario: a cape that flaps (visual, plus a wing-glide
if a power-up is active).

**Why it's not round 1:** round 1 #11 (grass sway) is a vertex-shader
displacement with no physics, no collision, no tearing. Round 1 #18 (fluid
sim) is a pressure/velocity field. Cloth is a 2D deformable mass-spring
surface with world/body collision — a distinct physical model from both.

---

## 3. vox-field — Gravity Wells, Anti-Gravity Zones & Magnetic Fields (Solver-extension + new sibling crate)

**Tier: Solver-extension + new sibling crate.** A `vox-field` crate at the
physics tier that introduces **volumetric force fields** — regions where a
custom acceleration is applied to every awake rigidbody each substep, before
the contact solve.

**The hook point is precise:** `PhysicsWorld::substep` applies gravity in one
line per awake body (solver.rs:331: `body.vel.y -= GRAVITY * h`). A field pass
is an additional acceleration accumulated into `body.vel` in the same loop,
before `world_contacts` runs (solver.rs:336). Three field types:

1. **Gravity well** — a radial attractor at a point: `a = strength / (r² +
   ε)` toward the center. Creates orbital debris, stuff falling sideways.
   `MAX_SPEED` (solver.rs:81, 332) already clamps diverging bodies, so a
   strong well can't blow up the solver.
2. **Anti-gravity zone** — a box or sphere region where the gravity line
   (solver.rs:331) is negated or reversed. Debris floats upward or hovers.
   Implemented as a per-body acceleration override: if the body's AABB
   (body.rs:270-271) intersects the zone, replace `GRAVITY * h` with the
   zone's acceleration.
3. **Magnetic field** — a field that only affects bodies made of "magnetic"
   materials (a new `MaterialDef` flag). Magnetic bodies attract/repel each
   other within a radius and can be attracted to anchor points. The pairwise
   interaction uses `candidate_pairs` from the broadphase spatial hash
   (broadphase.rs:14-54) to find nearby magnetic bodies cheaply — it's O(n²)
   over the *magnetic subset only*, not all bodies.

**Fields are authored as voxel-placed entities:** a special "gravity well"
voxel material that, when present in the world, emits a field centered on
itself. This reuses the existing voxel-as-material-authoring pattern (like the
explosive block in round 1 #16) — place a gravity-well block, get a field.
Multiple wells compose (sum of accelerations). The field source scan is a
world-grid traversal, the same `SolidLookup` pattern the flood fill uses
(world.rs:21-51, destruction.rs:384-413).

**What to change:**
- New `vox-field` crate: `Field { kind, center, radius, strength, falloff }`
  + a `FieldWorld` that collects fields from placed source voxels each step.
- vox-physics: the substep's gravity loop (solver.rs:331) gains a hook for
  per-body field acceleration. Cleanest as a `pub` method on `PhysicsWorld`
  that vox-field calls, or a trait extension — vox-field sits above
  vox-physics, same tier relationship as vox-rope.
- The magnetic flag requires the schema change (see below).

**Schema change (the real cost for magnetism only):** `RawMaterial` is
`#[serde(deny_unknown_fields)]` (material.rs:60), so adding `magnetic: bool`
is a breaking TOML parse change: add the key to `RawMaterial` (material.rs:61-
68), add `magnetic: bool` to `MaterialDef` (material.rs:24-37, defaulting
`false`), update the validation in `impl MaterialRegistry` (material.rs:78-
296), and add the key to every TOML asset that wants magnetic materials. The
gravity-well and anti-grav fields need no schema change (they're identified by
material *name* or a dedicated source-voxel type, not a flag). This is the
same four-touch plumbing the rendering engineer documented for the water
transparency flag (rendering-ideas.md:124-149).

**Gameplay:** build a gravity well to collect debris into orbit, anti-grav
shafts to float up, magnetic rails that pull metal debris along a path. Mario:
anti-grav zones let him hover-jump; magnetic boots (if a magnetic material
flag is wired).

**Why it's not round 1:** round 1 #20 (buoyancy) is a single force type in a
single medium (density vs water). vox-field is a general volumetric force-field
*system* with multiple composing field types (wells, anti-grav, magnetism)
that apply to all bodies. No round 1 idea covers non-uniform gravity,
attractors, or material-selective forces.

---

## 4. vox-thermal — Temperature Diffusion & Heat Propagation Through Solids (New sibling crate + schema change)

**Tier: New sibling crate + schema change.** A `vox-thermal` crate at the
gen/physics tier (depends on vox-core + vox-world). A per-voxel **temperature
field** stored as sidecar `f32` arrays — the exact pattern design doc §4
reserves: *"Voxel = u16 material id … Per-voxel state (damage, temperature)
added later as sidecar arrays"* (design doc, 2026-07-04-voxel-engine-design.md
line 59). This is a continuous scalar field with diffusion physics, **not**
another cellular-automaton fire-spread — round 1 #6 (vox-sim) already owns CA
fire/water/falling-sand.

**The physics is the heat equation, discretized on the voxel grid:**
- Each tick, a voxel's temperature moves toward the weighted average of its 6
  face-neighbors, weighted by the harmonic mean of thermal conductivity at
  each interface (the standard explicit finite-difference form of ∂T/∂t =
  α∇²T). The 6-connected neighbor traversal already exists: `SolidLookup`
  (world.rs:21-51) and the flood fill's `DIRS` (destruction.rs:328-336) walk
  exactly these neighbors. vox-thermal reuses the same traversal pattern.
- Conductivity is per-material: a new `MaterialDef` field,
  `thermal_conductivity` in W/(m·K), with sensible defaults (stone conducts
  more than wood, wood more than leaves). This is a schema change (below).

**Heat sources & sinks (the integration points):**
- **Fire (from vox-sim #6):** burning voxels emit constant heat into
  neighbors. vox-thermal consumes the fire state; it doesn't duplicate it.
- **Heat tool:** a new tool (tools.rs match arm) that adds thermal energy to
  voxels — a debug-panel/Tunables-driven scalar.
- **Lava material:** emits constant heat (a high baseline temperature).
- **Death laser:** converts a fraction of its destruction energy into heat —
  the crater left by `carve_capsule` (destruction.rs:281-326) is warm.
- **Ignition (the feedback to vox-sim):** when a voxel's temperature exceeds
  its material's ignition point, it catches fire and feeds back into #6's
  fire CA. This is the bridge: vox-sim owns *fire state*, vox-thermal owns
  *temperature state*; ignition is the handoff.

**Structural consequences (the gameplay payoff):**
1. **Heat weakens materials:** a hot voxel's effective `strength` drops as
   temperature rises: `eff_strength = strength * (1 - k * norm_temp)`. A
   heated beam collapses under load it would normally survive — heat becomes a
   new input to the sustained-load failure model (#17). The `strength` field
   (material.rs:34) is already read at fracture time; vox-thermal exposes a
   per-voxel multiplier that the fracture check multiplies in.
2. **Thermal expansion (visual):** hot voxels expand subtly — a vertex offset
   in the mesher driven by temperature, the same pattern as grass sway
   (rendering-ideas.md:22-68) but keyed off the temperature sidecar instead of
   a time uniform. Cheap, optional, visual-only.
3. **Heat transfer to debris bodies:** a hot body transfers heat to its
   contacts via the existing contact graph — the `ContactKey` (contact.rs:20)
  already identifies exactly which body-face pairs are touching. A hot debris
   body warms and eventually ignites adjacent flammable bodies.

**Schema change (the real cost):** add `thermal_conductivity: f32` and
`ignition_temp: f32` to `MaterialDef` (material.rs:24-37) and the
corresponding optional fields to `RawMaterial` (material.rs:61-68, which is
`deny_unknown_fields` — a breaking TOML parse change). Add defaults to every
existing material in the TOML assets, and validation in `impl MaterialRegistry`
(material.rs:78-296). This is the same four-touch plumbing as the magnetic
flag (#3) and the rendering water flag (rendering-ideas.md:124-149). The
sidecar array storage itself is additive — design doc §4 line 59 already
reserves it, so no `Voxel`/`Chunk` struct change (the `ChunkStorage` enum,
chunk.rs:20-25, stays as-is; the temperature field is a parallel `HashMap` or
chunk-bounded `Box<[f32; CHUNK_VOLUME]>`).

**Rate:** updated at a tunable sub-physics rate (e.g. 10 Hz, not 60 Hz) — heat
diffuses slower than rigidbody motion. Headlessly testable (no GPU, no
window).

**Why it's not round 1:** round 1 #6 (vox-sim) is cellular automata — discrete
state transitions (burning → charred, water → steam). Temperature diffusion is
a continuous scalar PDE (the heat equation) with material-dependent
conductivity and feedback into structural strength and ignition. It's a
different physical model occupying a different state representation
(continuous `f32` sidecar vs discrete CA state). No round 1 idea covers heat,
temperature, or thermal material weakening — and design doc §4 explicitly
reserves "temperature" as the canonical example of a sidecar field.

---

## 5. vox-quake — Earthquake & Tremor Events With Seismic Wave Propagation (New sibling crate, event-driven)

**Tier: New sibling crate, event-driven (not a persistent system).** A
`vox-quake` crate at the physics tier. An earthquake is a triggered event that
applies a time-varying ground acceleration to every rigidbody in a radius,
with a **propagating wave front** — not an instantaneous global impulse.

**Two components:**

1. **Seismic wave.** A displacement wave travels outward from an epicenter at
   a tunable speed (e.g. 300 m/s for P-waves). At each rigidbody, the applied
   acceleration = `peak_accel * sin(k*r - ω*t) * envelope(r)`, where `r` is
   distance from epicenter and `envelope` is a radial falloff. The wave is
   evaluated per-body per-substep, injected as an additional acceleration in
   the same gravity loop (solver.rs:331) — same hook point as vox-field (#3).
   **The key visual/physical property:** resting bodies are woken *in
   sequence* as the front reaches them. The solver's `wake_region`
   (solver.rs:181-189) is called with a moving AABB that tracks the wave
   front radius each step — debris on the ground starts bouncing as the shock
   arrives, not all at once. This reuses the exact waking mechanism that
   `blast` and `laser` already call (destruction.rs:684, 709).

2. **Structural resonance.** The wave's frequency can match a structure's
   natural frequency, amplifying shaking. The resonance analysis reuses the
   connectivity graph from the destruction pipeline: `detach_unsupported`
   (destruction.rs:428-491) runs a 6-connected seed-flood that proves whether
   a component is anchored to the world floor or bounded/disconnected. A tall
   tower whose flood reaches the floor only through a thin base is
   "resonance-vulnerable" — the quake applies a base-shake impulse to the
   ground-floor voxels, and if the shake exceeds a material's
   `MaterialDef::strength` (material.rs:34), voxels crack. This feeds into the
   sustained-load failure model (#17) and the impact fracture pipeline
   (`ImpactEvent.impulse / body.mass()` vs `Tunables::fracture_sensitivity`,
   tunables.rs:34) — the shake impulse is compared against the same
   per-material threshold, just with a seismic source instead of a collision.

**What to change:**
- New `vox-quake` crate: `Earthquake { epicenter, magnitude, duration,
  wave_speed, frequency }` + a `tick(dt)` that returns per-body accelerations
  and a `wake_region` AABB for the current front radius.
- vox-physics: the substep gravity loop (solver.rs:331) gains the same
  per-body acceleration hook as vox-field (#3) — a quake is just a
  time-varying field. If #3 lands first, vox-quake is a field type; if not,
  vox-quake brings its own hook. Either way it's the same one-line injection
  point.
- `Tunables` (tunables.rs:13-35) gains `quake_magnitude`, `quake_duration`,
  `quake_wave_speed` for live-tunable triggering via the debug panel.

**Event lifecycle:** an earthquake runs for a duration and fades (envelope
amplitude → 0). It is not a persistent system — `vox-quake` holds active
events in a `Vec`, ticks them, and removes them when they expire. Multiple
earthquakes can overlap (amplitudes sum). Triggered via debug panel, a tool,
or a gameplay script.

**Gameplay:** a natural-disaster tool (hotbar slot), a survival mode where
earthquakes periodically ravage your build, a siege weapon that triggers a
localized tremor under a wall. Mario: a quake event launches him if he's
grounded when the shock arrives (the ground acceleration applies to the
character controller's ground check).

**Why it's not round 1:** round 1 #16 (chain explosives) is a discrete
instantaneous blast event. Round 1 #17 (structural failure) is sustained
*static* load. An earthquake is neither — it's a **time-distributed,
spatially-delayed propagating wave** that applies impulses across a region
over seconds, with structural resonance as an emergent consequence. Round 1
#3 (storm survivor) has wind/flood/lightning but no ground-shaking seismic
mechanic. No round 1 idea covers seismic events or propagating disturbance
fields.

---

## Recommended Sequencing

| Step | Idea | Tier | Why this order |
|------|------|------|----------------|
| 1 | vox-rope (1) | Solver-extension | Extends the impulse loop with distance constraints — the most natural solver addition, and the break-on-tension pattern establishes the per-constraint failure convention #2 and #5 reuse. |
| 2 | vox-field (3) | Solver-extension + schema | The per-body acceleration hook in the gravity loop (solver.rs:331) is the injection point for both #3 and #5. Land it once; #5 becomes a field type. The magnetic flag schema change is the first `MaterialDef` extension — establish the pattern. |
| 3 | vox-thermal (4) | New crate + schema | The second schema change (`thermal_conductivity`, `ignition_temp`) follows the pattern #2 established. The temperature sidecar is design-doc-reserved (§4 line 59) and pairs with vox-sim #6 (fire). Needs #6 to exist for the ignition handoff. |
| 4 | vox-cloth (2) | New crate + render path | Isolated from the solver work (own integrator, own render path). Can land any time, but the mass-spring break-on-strain pattern is clearer once #1's rope breaks are familiar. |
| 5 | vox-quake (5) | New crate, event-driven | Builds on #2's acceleration hook and reuses #17's stress model. Cheapest to add last because it's a consumer of the solver hook + connectivity flood, not a new mechanism. |

Steps 1 and 2 touch the solver and establish the two reusable extension points
(distance constraints, per-body field acceleration) plus the `MaterialDef`
schema-change pattern. Steps 3-5 are consumers that build on those hooks. The
schema changes (#2 magnetic, #3 thermal) are the real structural costs — both
are the same four-touch `deny_unknown_fields` plumbing the rendering engineer
already documented (rendering-ideas.md:124-149), so doing them back-to-back
amortizes the pattern.
