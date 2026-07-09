# 5 Gameplay Ideas — Gameplay Designer's Brainstorm (Round 2)

Grounded in the actual destruction pipeline: **carve** a shape (sphere for
Scalable Dig/Bomb, capsule along a line for the Death Laser, recording what was
removed) → **flood** 6-connected outward from each newly-exposed solid voxel,
following the real voxel shape with no search box → **detach** anything a flood
proves bounded into a `VoxelGrid` rigidbody. Every carve returns
`CarveResult { removed: Vec<(IVec3, Voxel)>, region, spawned }` — a complete
position+material log of what was removed — and blasts are deterministic from a
per-shot `seed` (`ExplosionShape::new(center, radius, seed)`). The solver
reports `ImpactEvent { body, point_m, impulse, push_dir }` each step, and
material **strength** (`core.toml`: stone 8, brick 6, wood/planks 4, dirt/grass
2, sand 1, leaves 0.5) gates both blast survival and impact fracture
(`impulse / body.mass()` vs strength, routed through the same `carve_body_*`
path as the tools). The four hotbar tools: 1 Dig (one voxel, zero splash), 2
Scalable Dig (tunable sphere, debris falls under gravity only), 3 Bomb (sphere +
outward blast impulse), 4 Death Laser (infinite-range instant cut). Slots 5–9
reserved. Adding a new material is near-free — drop a `[[material]]` block in a
`*.toml` under `assets/materials/` (the README is explicit that every file there
auto-loads); tagged below as **[+material, near-free]** where an idea wants one.

Feasibility tiers (mirroring the rendering engineer's round-2 vocabulary):
- **Works-today** — playable with what ships in `vox-app`, no new crates.
- **Needs-new-crate** — a new sibling at the gen/physics tier, per the
  architecture rules (nothing lower depends on anything higher).
- **Needs-engine-change** — touches an existing core crate's behavior, not
  just additive logic.

All five are distinct from Round 1 (siege warfare, rally racing, storm
survival, creative building, destruction derby) and from each other.

---

## 1. Controlled Demolition Engineering

**Tier: Works-today.** The inverse of Siege Warfare — don't breach, *bring it
down safely.* A tower must collapse into its own footprint, or fall toward a
chosen compass direction, scored against a collateral radius (zero bonus; a
shattered neighboring wall fails the contract).

The puzzle is pure carve+connectivity+strength, which is exactly what runs on
every tool click today. Sequence single-voxel **Dig** cuts (slot 1) to sever
the load-bearing pillars on *one side*; `detach_unsupported`'s 6-connected
flood proves the remaining mass is no longer anchored and converts it to a
rigidbody that tumbles under gravity — the same mechanism that drops a severed
floating tree. Material `strength` gates how aggressively you can undercut
before an unintended cut triggers premature collapse, so each level's material
mix *is* its difficulty (a stone tower tolerates wide cuts; a sand one is
nervous). The **Death Laser** (slot 4, instant infinite cut) is the high-stakes
"one clean slice" tool: perfect for a drop-in-place demolition, instant fail
for a directional one.

No new systems. The only addition is a scoring layer in `vox-app`: measure the
debris pile's bounding footprint vs. the target zone after all bodies sleep
(the solver already reports settle state, and `clear_sleeping` enumerates
settled bodies). Levels are pre-built structures placed by hand or a tiny
deterministic `vox-gen` variant — no crate boundary crossed.

**Why it earns the slot over Round 1's Siege:** Siege is *offensive* carve —
any breach wins. Demolition is *surgical* carve — the constraint is what you
*don't* break, and the connectivity flood is the judge, not an HP bar.

---

## 2. Voxel Archaeology — Excavate Without Destroying

**Tier: Needs-new-crate** (a `vox-gen` ruin-layer module; **[+material
"ceramic" strength 0.3, near-free]**). Buried fossils / ruined structures are
placed by a new ruin-layer pass — a sibling module to terrain/trees in
`vox-gen`, same deterministic-per-chunk pattern. The player excavates the
overburden to expose the artifact — **but the artifact is a fragile material**
(leaves at 0.5, or a new ceramic at 0.3), and any impact above its fracture
threshold shatters a piece irrecoverably.

Two existing mechanisms carry it. (1) The **material-strength-gated impact
fracture** in `solver.rs::ImpactEvent` already compares
`impulse / body.mass()` against strength and routes a hard-enough hit through
the *same* `carve_body_*` path the tools use — so a bomb blast near a ceramic
fossil, or a debris chunk landing on it, shatters exactly the voxels the impact
reached, deterministically from the blast seed. (2) The **carve recording**
(`CarveResult.removed`) tells the game exactly what material each removed voxel
was, so "did you just destroy a fossil voxel?" is a one-line check in the carve
callback, not a new scan. The scalpel-vs-sledgehammer tradeoff is the whole
loop: **Dig** (slot 1, one voxel, zero splash) is safe; **Bomb** (slot 3) and
**Death Laser** (slot 4) are fast but risk the artifact. **Scalable Dig** (slot
2) is the tension point — its tunable radius means *you* choose how brave you
are.

The new thing is the ruin generator (a `vox-gen` module) + a "fragile artifact"
material flag — the material itself is a near-free `.toml` drop. No physics
changes.

**Why it earns the slot:** uniquely uses *strength-gated fracture as a
constraint you work around* rather than as a goal — no other idea on either
list treats destruction as the thing you must *avoid*.

---

## 3. Rube-Goldberg Demolition Machine

**Tier: Needs-engine-change** for the full static-structure domino; a
narrower **debris-on-debris** variant is **Works-today** (see below). Build a
chain-reaction machine that topples a target across a level using
domino-severing and impact cascade. The player sets up the *trigger*; gravity
and the solver do the rest.

**What works today — the debris-on-debris cascade.** Three links already
exist and compose without any engine change. (a) **Sever → detach → fall:**
cut a support with Dig and `detach_unsupported`'s 6-connected flood converts
the freed mass to a debris rigidbody that tumbles under the solver — same
path that drops a severed floating tree. (b) **Hard-landing self-chip:** the
falling body's own `ImpactEvent` is compared against *its own* material
strength and routed through `carve_body_sphere_at_impact(event.body, ...)`,
so a fragile body shatters where it strikes. (c) **Debris-on-debris
fracture:** a falling body striking a *resting debris body* triggers an
`ImpactEvent` on the target body too, so a heavy falling chunk can crack a
lighter resting one. These three compose into a real cascade *through the
debris body graph* — knock one loose, it falls and cracks another, which
shifts and topples a third. **Bomb** (slot 3, outward blast impulse) is the
reliable first-domino starter; the **Death Laser** is the surgical
single-cut initiator. Fully headless-testable today.

**What does NOT work today — the static-structure domino.** The tempting
version — a falling block fractures the *static chunked World tower* it
lands on, and that tower then detaches and falls into the next — does **not**
exist. I verified `apply_impact_fracture` (main.rs:337–379) only ever calls
`carve_body_sphere_at_impact(..., event.body, ...)` — it carves the *falling
body itself* (material looked up in `body.grid`), never the static `World`.
And `detach_unsupported` is invoked only by `blast` / explicit tool carves,
never by a debris body landing on terrain. So a falling block will *not*
fracture static terrain it hits, and static terrain will *not* then detach as
a domino. Making that work is a clean, additive engine change: a
**world-side impact-fracture path** alongside the existing body-side one —
when an `ImpactEvent`'s contact point lies in static `World`, route it
through `carve_sphere` (world) + `detach_unsupported` instead of
`carve_body_sphere_at_impact`. That's Needs-engine-change (new logic in
`vox-physics`, reusing the existing carve+flood, no solver rewrite) and it
unlocks the full domino-through-buildings fantasy.

Material `strength` is the tuning knob for either variant: stone (8) takes a
real impact to topple, leaves (0.5) cascade from a touch.

**Why it earns the slot:** the only idea where the *player authors a
multi-stage physical chain reaction* whose correctness is proven by the
connectivity flood at each link. Round 1's Chain-Reaction Explosives (#16) is
about *explosive material* cascading; this is about *structural* cascade —
sever, fall, shatter, sever again — and needs no new `explosive` flag. The
honest split: a debris-on-debris machine is playable now; the
static-structure domino is one clean additive engine change away.

---

## 4. Time-Rewind Demolition Sandbox

**Tier: Needs-new-crate** (`vox-rewind` at physics tier for the full scrub-
through-collapse version; a Works-today single-step-undo prototype fits in
`vox-app`). A "Braid-for-destruction" mode: carve freely, then scrub a timeline
to rewind the world to any prior state, fork from there, and try a different
cut. The keystone enabler is that **every destructive action is already fully
invertible in data**: every carve returns `removed: Vec<(IVec3, Voxel)>` — a
complete position+material log of what was removed — and blasts are
deterministic from `seed` (`ExplosionShape::new(center, radius, seed)`).
Undoing a carve is literally writing those `(IVec3, Voxel)` pairs back; undoing
a blast is replaying its seeded shape to know which spawned debris to despawn
and which carved voxels to restore.

Works-today prototype: a ring buffer of the last N carves in `vox-app` + a
restore key (`Z`) that replays the inverse. This covers Dig / Scalable Dig /
Laser (pure carves) and *most* of Bomb (carve + spawned bodies; despawn the
bodies, restore the voxels). Full version: `vox-rewind` snapshots the
`PhysicsWorld` body graph + world delta per step, enabling scrub *through* the
destruction as it plays — watch the tower fall, rewind to mid-collapse, fork.
That's genuinely new engine machinery (a deterministic-time model), but it
rides the same seed-and-record rails the blast system already built, so it's an
additive sibling crate, not a rewrite of the solver.

**Why it earns the slot:** the only idea that turns the *destruction recording
the engine already produces* into a first-class gameplay verb. A generic
photography/screenshot mode could exist in any engine; rewind is uniquely
meaningful *here* because every carve is a reversible, seeded, logged event —
it would be far weaker in a voxel engine that didn't already record `removed`.

---

## 5. Voxel Seismograph — Mining With Cave-In Risk

**Tier: Needs-engine-change** (a read-only flood-risk probe in `vox-physics`,
surfaced through the existing `vox-debug` egui overlay). A mining game where
the hazard isn't monsters, it's the roof. Carve a tunnel into a mountain; the
**connectivity flood** that already runs on every carve becomes your cave-in
warning — except now it's exposed to the player as a *seismograph* readout:
bounded-component volumes near your tunnel tell you how much unsupported mass
is overhead, and if you sever the last anchor, `detach_unsupported` converts
that mass to a rigidbody and it *falls on you* via the solver.

The engine already computes exactly the quantity this needs: `detach_unsupported`
floods from every newly-exposed voxel and either reaches the world floor
(anchored) or exhausts under a cap (bounded → candidate body). Exposing the
*bounded-but-not-yet-spawned* volume — components under `MAX_BODY_VOXELS` that
are one cut from detaching — as a UI warning is a read-only query against the
same flood. The engine-change is that this query doesn't exist yet: the flood
currently runs inside `detach_unsupported` and only reports what it *detached*,
not the at-risk volumes it proved bounded but left resident. A
`flood_risk_probe` variant that reports bounded-component volumes without
spawning bodies is a new public function in `vox-physics`, not just additive
`vox-app` logic — hence the tier. Material `strength` sets the margin: a stone
ceiling (8) tolerates a wider cut before the flood goes bounded; sand (1)
cascades almost immediately, so the same tunnel in sand is a death trap. The
**Bomb** is the reckless miner's tool (fast, but the blast's wake-region widens
the unsupported zone beyond what you can see); **Dig** is the careful one.

**Why it earns the slot:** the only idea that makes the *connectivity flood
itself* the core gameplay loop — not a consequence of your actions, but the
information you play against. Round 1's Sustained-Load Failure (#17) is about
*weight* collapsing towers; this is about *severance* collapsing mines — same
flood, opposite direction of stress, and it works because the flood is already
a proven proof, not a heuristic.
