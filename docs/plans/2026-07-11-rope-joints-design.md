# Rope + Joints — Design Document

**Date:** 2026-07-11
**Status:** Approved
**Builds on:** The existing sequential-impulse physics solver (`solver.rs`),
voxel rigidbody system (`body.rs`, `body_destruction.rs`), and material
registry (`vox-core/material.rs`). This document adds distance-constraint
joints, a rope material, and rope spawning.

---

## 1. Decisions of Record

| Question | Decision |
|---|---|
| Joint type | **Distance constraint** — maintains rest length between two anchor points on two bodies. Allows free rotation, resists stretching. Simplest joint for rope/chain. |
| Joint storage | `joints: Vec<Joint>` on `PhysicsWorld` — persistent across steps (unlike contacts). Stores slot indices, body-local anchors, rest length, accumulated Lagrange multiplier for warm starting. |
| Joint solving | Inside the `SOLVER_ITERS` loop (solver.rs:602), interleaved with contacts. Standard sequential-impulse: compute distance violation, apply equal-and-opposite impulse. Position correction in split-impulse pass. |
| Joint-sleep | Feed joint pairs into island union-find (solver.rs:450). Sleeping body = static anchor. Wake on relative motion > threshold. |
| Rope segments | Each segment is a `Body` with a `VoxelGrid` of `rope` voxels (4×4×5). Connected by joints at end-face centers. Participates in all existing physics: collision, fracture, buoyancy, fire, particles. |
| Rope material | `solid = true`, density 400, strength 2.0, flammable = true. Color: tan/brown [0.65, 0.50, 0.30]. |
| Joint breaking | Joints removed when either body is despawned. Rope can be cut by carving through a segment (body splits, joint references stale body). |
| Rope spawning | 5 segments spawned near player at start, hanging from a point above. Also spawnable via a rope tool (hotbar slot 7) that creates segments between two clicked points. |
| Anchor points | Body-local offsets from COM, stored on the Joint struct. Rotated by body orientation each substep to get world-space anchor positions. |

## 2. Joint struct and solver integration

### 2.1 Joint struct

```rust
/// A distance constraint between two bodies. Maintains a fixed rest length
/// between two anchor points. Used for rope/chain segments.
pub struct Joint {
    /// Slot index of body A (must be valid during solve).
    pub body_a: usize,
    /// Slot index of body B.
    pub body_b: usize,
    /// Anchor point on body A, relative to COM, body-local frame (meters).
    pub anchor_a: Vec3,
    /// Anchor point on body B, relative to COM, body-local frame.
    pub anchor_b: Vec3,
    /// Rest length between anchors (meters).
    pub rest_length: f32,
    /// Accumulated Lagrange multiplier (warm start).
    pub acc_lambda: f32,
    /// Compliance (inverse stiffness). 0 = rigid, higher = softer.
    pub compliance: f32,
}
```

### 2.2 PhysicsWorld additions

```rust
pub struct PhysicsWorld {
    // ... existing fields ...
    /// Distance constraints between bodies. Persistent across steps.
    joints: Vec<Joint>,
}
```

Methods:
- `add_joint(a: BodyId, b: BodyId, anchor_a: Vec3, anchor_b: Vec3, rest_length: f32) -> usize` — returns joint index
- `remove_joint(idx: usize)` — swap-remove
- `remove_joints_for_body(slot: usize)` — called when a body is despawned
- `joints(&self) -> &[Joint]` — read access for rendering/debugging

### 2.3 Joint solve (inside SOLVER_ITERS loop)

For each joint, each solver iteration:

1. Get both bodies via `two_mut(slots, joint.body_a, joint.body_b)`.
2. Compute world-space anchor positions: `pa = body_a.pos + body_a.rot * anchor_a`, `pb = body_b.pos + body_b.rot * anchor_b`.
3. Compute distance vector: `d = pb - pa`, `dist = d.length()`, `n = d / dist` (if dist > epsilon, else arbitrary).
4. Compute constraint violation: `C = dist - rest_length`.
5. Compute effective mass: `keff = inv_mass_a + inv_mass_b + (ra × n)·inv_iw_a·(ra × n) + (rb × n)·inv_iw_b·(rb × n)` where `ra = body_a.rot * anchor_a`, `rb = body_b.rot * anchor_b`.
6. Compute impulse: `lambda = -C / (keff + compliance)`. Add to `acc_lambda` for warm start.
7. Apply equal-and-opposite impulse: `body_a.vel += n * lambda * inv_mass_a`, `body_a.omega += inv_iw_a * (ra × n * lambda)`, `body_b -= ...`.

If one body is sleeping: treat as static (inv_mass = 0, inv_iw = 0). If relative speed > wake threshold: wake the sleeper.

### 2.4 Warm start

At the start of each substep (before velocity iterations), apply `acc_lambda` from the previous substep:
```
body_a.vel += n * acc_lambda * inv_mass_a
body_a.omega += inv_iw_a * (ra × n * acc_lambda)
body_b -= ...
```
Reset `acc_lambda` to 0 before the velocity iterations (standard warm-start pattern).

### 2.5 Position correction

In the split-impulse pass (solver.rs:712-735), add joint distance drift correction:
```
C = dist - rest_length
correction = n * C / keff
body_a.pos_corr += correction * inv_mass_a
body_b.pos_corr -= correction * inv_mass_b
```
This runs within `POSITION_ITERS` (2 iterations), alongside contact penetration correction.

### 2.6 Island union-find

In `islands()` (solver.rs:442-464), after the broadphase pair unions, add joint pairs:
```rust
for j in &self.joints {
    if both slots valid and both bodies exist {
        islands.union(j.body_a, j.body_b);
    }
}
```

### 2.7 Joint cleanup on despawn

In `despawn()` (solver.rs:240-247), after incrementing generation and pushing to free list:
```rust
self.joints.retain(|j| j.body_a != slot && j.body_b != slot);
```

## 3. Rope material

`assets/materials/core.toml` gains:

```toml
[[material]]
name = "rope"
color = [0.65, 0.50, 0.30]
jitter = 0.06
density = 400.0
strength = 2.0
solid = true
flammable = true
```

## 4. Rope spawning

### 4.1 Auto-spawn at start

In `VoxApp::new`, after world build and physics init, spawn 5 rope segments hanging from a point near the player:

1. Find a point ~5m above the player's spawn position.
2. Create 5 bodies, each a 4×4×5 voxel grid of `rope` material, oriented vertically.
3. Connect consecutive segments with joints at their end-face centers (rest length = 0, or a small gap).
4. The top segment could be anchored to the world (a static anchor) or just left to fall.

**Simpler approach**: just spawn the 5 segments stacked vertically and let them fall — the joints hold them together as a rope. No world anchor needed for v1.

### 4.2 Rope tool (hotbar slot 7)

When the player selects slot 7 and clicks two points:
1. Compute the distance between the two clicked points.
2. Determine the number of segments: `n = (distance / segment_length).ceil()`.
3. Spawn `n` rope segment bodies along the line between the two points.
4. Connect consecutive segments with joints.
5. The first and last segments could be anchored to the world at the clicked points (future enhancement — v1 just spawns them as free bodies).

**v1 simplification**: Just spawn 5 segments near the player on key press (like the existing B key for wood debris). The tool-based placement between two points is a v2 enhancement.

## 5. Rope segment construction

Each segment is a `VoxelGrid` of dimensions `(4, 5, 4)` filled with `rope` material voxels. At 0.1m voxel scale, that's 0.4m × 0.5m × 0.4m per segment. 5 segments = 2.5m of rope.

The anchor points for joints are at the centers of the top and bottom faces of each segment, in body-local frame:
- `anchor_top = (0, +half_height, 0)` relative to COM
- `anchor_bottom = (0, -half_height, 0)` relative to COM

The rest length between segments is 0 (they touch end-to-end) or a small gap (0.05m for visual separation).

## 6. Joint breaking

- **Body despawn**: `despawn()` removes all joints referencing that body's slot.
- **Fracture**: when a rope segment is carved through, `split_components` despawns the original body and spawns fragments. The original body's joints are removed. The fragments don't inherit joints (they're new bodies). This means cutting a rope segment severs the chain — the rope falls apart at that point.
- **Future**: max-tension breaking (joint breaks if constraint force exceeds a threshold). Out of scope for v1.

## 7. Rendering

Rope segments render as regular debris bodies — the existing `VoxelPipeline::draw_bodies` handles them. The `rope` material gets a palette entry (tan/brown). No special rendering needed.

Joints themselves are not rendered (invisible constraints). A future debug visualization could draw lines between joint anchors.

## 8. Testing plan

- **Joint struct**: distance constraint maintains rest length between two bodies
- **Warm start**: accumulated lambda persists across substeps, improves convergence
- **Sleep**: joined bodies sleep/wake together via island union-find
- **Despawn cleanup**: despawning a body removes its joints
- **Rope spawn**: 5 segments created near player, connected by 4 joints
- **Rope fracture**: carving through a segment severs the chain
- **Rope fire**: rope is flammable, burns (existing fire system applies once body fire consumption is implemented)

## 9. Explicitly out of scope

- World-anchored joints (joint to static world point) — future
- Rope tool with two-point placement — v2
- Max-tension joint breaking — future
- Joint debug visualization — future
- Hinge/cone-twist/6-DOF joints — future (distance constraint is sufficient for rope)
- Rope winding/coiling simulation — future
