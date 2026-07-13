//! Destruction against an existing body's own grid, not the static world.
//!
//! A rigidbody has no floor/anchor concept -- it isn't "resting on" anything
//! by definition, it's already a free object in space. So carving material
//! out of one is simpler than [`crate::destruction`]: there's no connectivity
//! proof needed, just plain 6-connected component labeling of whatever solid
//! material remains. Every component becomes its own fragment (subject to
//! the same dust/oversize policy as world destruction), inheriting the
//! parent's linear and angular velocity -- a fragment picks up the parent's
//! spin at its own offset from the old center of mass, matching ordinary
//! rigid-body kinematics (`v_point = v_com + omega x r`).

use std::collections::VecDeque;

use glam::{IVec3, Mat3, Quat, Vec3};
use vox_core::{FxHashMap, FxHashSet, MaterialRegistry};
use vox_core::consts::{DEBRIS_MIN_VOXELS, MAX_BODY_VOXELS};
use vox_world::{AIR, Voxel};

use vox_core::voxel_center_m;

use crate::BodyId;
use crate::body::{Body, VoxelGrid, mass_props, MassProps};
use crate::destruction::{ExplosionShape, small_hash};
use crate::solver::{PhysicsWorld, Joint};

const DIRS: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
];

#[inline]
fn grid_index(dims: IVec3, p: IVec3) -> usize {
    (p.x + p.z * dims.x + p.y * dims.x * dims.z) as usize
}

/// Visit every voxel in `[min, max)` (clipped to the grid's own bounds),
/// calling `edit(local_pos, current)`; `Some(new)` writes it. Returns the
/// positions where a solid voxel became air (what was "removed").
fn edit_grid_box(
    grid: &mut VoxelGrid,
    min: IVec3,
    max: IVec3,
    mut edit: impl FnMut(IVec3, Voxel) -> Option<Voxel>,
) -> Vec<(IVec3, Voxel)> {
    let min = min.max(IVec3::ZERO);
    let max = max.min(grid.dims);
    let mut removed = Vec::new();
    if min.cmpge(max).any() {
        return removed;
    }
    for z in min.z..max.z {
        for y in min.y..max.y {
            for x in min.x..max.x {
                let p = IVec3::new(x, y, z);
                let cur = grid.get(p);
                let Some(new_v) = edit(p, cur) else { continue };
                if new_v == cur {
                    continue;
                }
                if cur != AIR && new_v == AIR {
                    removed.push((p, cur));
                }
                let idx = grid_index(grid.dims, p);
                grid.voxels[idx] = new_v;
            }
        }
    }
    removed
}

/// Remove exactly the one voxel at `local_voxel` (grid-local voxel
/// coordinates), if solid. A tiny sphere can't reliably do this: a raycast
/// hit point sits on the voxel's *face*, roughly half a voxel's diagonal
/// away from its center, so a distance-to-center sphere check needs a
/// radius comfortably above that to avoid missing the very voxel it's
/// supposed to hit -- at which point it may as well not be a distance
/// check at all. This is what backs a single-voxel dig tool hitting debris:
/// an exact single-voxel break, matching how it already works against the
/// static world.
pub fn carve_body_voxel(grid: &mut VoxelGrid, local_voxel: IVec3) -> Vec<(IVec3, Voxel)> {
    edit_grid_box(grid, local_voxel, local_voxel + IVec3::ONE, |_, cur| {
        (cur != AIR).then_some(AIR)
    })
}

/// Remove every solid voxel within `radius_m` of `center_local_m` (in the
/// grid's own local-meter frame, origin at its minimum corner). Same shape
/// as [`crate::destruction::carve_sphere`], just against a body's dense grid
/// instead of the chunked world.
pub fn carve_body_sphere(
    grid: &mut VoxelGrid,
    center_local_m: Vec3,
    radius_m: f32,
    voxel_size_m: f32,
) -> Vec<(IVec3, Voxel)> {
    let s = voxel_size_m;
    let r_vox = (radius_m / s).ceil() as i32;
    let center_vox = (center_local_m / s).floor().as_ivec3();
    let min = center_vox - IVec3::splat(r_vox);
    let max = center_vox + IVec3::splat(r_vox + 1);
    edit_grid_box(grid, min, max, |p, cur| {
        if cur == AIR {
            return None;
        }
        let c = (p.as_vec3() + 0.5) * s;
        ((c - center_local_m).length() <= radius_m).then_some(AIR)
    })
}

/// Remove every solid voxel within a jagged [`ExplosionShape`] centered at
/// `center_local_m` (in the grid's own local-meter frame). Same shape as
/// [`crate::destruction::carve_explosion`], against a body's dense grid --
/// so a Bomb hit on debris looks as chaotic as one on the world.
pub fn carve_body_explosion(
    grid: &mut VoxelGrid,
    center_local_m: Vec3,
    radius_m: f32,
    voxel_size_m: f32,
    seed: u32,
) -> Vec<(IVec3, Voxel)> {
    let shape = ExplosionShape::new(center_local_m, radius_m, seed);
    let s = voxel_size_m;
    let (bmin, bmax) = shape.bounds_m();
    let min = (bmin / s).floor().as_ivec3();
    let max = (bmax / s).ceil().as_ivec3();
    edit_grid_box(grid, min, max, |p, cur| {
        if cur == AIR {
            return None;
        }
        let c = (p.as_vec3() + 0.5) * s;
        shape.contains(c).then_some(AIR)
    })
}

/// Remove every solid voxel within `radius_m` of the line segment from
/// `start_local_m` to `end_local_m`. Same shape as
/// [`crate::destruction::carve_capsule`], against a body's dense grid.
pub fn carve_body_capsule(
    grid: &mut VoxelGrid,
    start_local_m: Vec3,
    end_local_m: Vec3,
    radius_m: f32,
    voxel_size_m: f32,
) -> Vec<(IVec3, Voxel)> {
    let s = voxel_size_m;
    let seg = end_local_m - start_local_m;
    let seg_len_sq = seg.length_squared();
    let pad = Vec3::splat(radius_m);
    let min = ((start_local_m.min(end_local_m) - pad) / s)
        .floor()
        .as_ivec3();
    let max = ((start_local_m.max(end_local_m) + pad) / s)
        .ceil()
        .as_ivec3();
    edit_grid_box(grid, min, max, |p, cur| {
        if cur == AIR {
            return None;
        }
        let c = (p.as_vec3() + 0.5) * s;
        let t = if seg_len_sq > 1e-9 {
            ((c - start_local_m).dot(seg) / seg_len_sq).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let closest = start_local_m + seg * t;
        ((c - closest).length() <= radius_m).then_some(AIR)
    })
}

/// Split `grid`'s solid voxels into their 6-connected components. Unlike
/// world destruction there's no anchor to prove: a body isn't attached to
/// anything, so *every* component is reported, however small -- callers
/// apply the dust/oversize policy themselves. Each is cropped to its own
/// minimal bounding box; the paired `IVec3` is that box's minimum corner in
/// the *original* grid's local voxel coordinates, for reconstructing world
/// placement.
pub fn split_components(grid: &VoxelGrid) -> Vec<(VoxelGrid, IVec3)> {
    let mut visited = vec![false; grid.voxels.len()];
    let mut out = Vec::new();
    for z in 0..grid.dims.z {
        for y in 0..grid.dims.y {
            for x in 0..grid.dims.x {
                let start = IVec3::new(x, y, z);
                let start_idx = grid_index(grid.dims, start);
                if visited[start_idx] || !grid.solid(start) {
                    continue;
                }
                let mut comp = Vec::new();
                let mut queue = VecDeque::new();
                queue.push_back(start);
                visited[start_idx] = true;
                while let Some(v) = queue.pop_front() {
                    comp.push(v);
                    for d in DIRS {
                        let n = v + d;
                        if n.cmplt(IVec3::ZERO).any() || n.cmpge(grid.dims).any() {
                            continue;
                        }
                        let nidx = grid_index(grid.dims, n);
                        if !visited[nidx] && grid.solid(n) {
                            visited[nidx] = true;
                            queue.push_back(n);
                        }
                    }
                }

                let mut min = IVec3::splat(i32::MAX);
                let mut max = IVec3::splat(i32::MIN);
                for &v in &comp {
                    min = min.min(v);
                    max = max.max(v);
                }
                let dims = max - min + IVec3::ONE;
                let total = (dims.x * dims.y * dims.z) as usize;
                let mut voxels = vec![AIR; total];
                let mut damage = vec![0.0; total];
                for &v in &comp {
                    let mat = grid.get(v);
                    let dmg = grid.damage_at(v);
                    let l = v - min;
                    let idx = grid_index(dims, l);
                    voxels[idx] = mat;
                    damage[idx] = dmg;
                }
                out.push((VoxelGrid::new_with_damage(dims, voxels, damage), min));
            }
        }
    }
    out
}

/// The parent body's state needed to place and set the motion of each
/// fragment [`finish_carve`] produces. `Copy`: both `finish_carve` and
/// [`spawn_impact_chips`] need their own snapshot of it after the same
/// carve, taken before the parent body is despawned.
#[derive(Clone, Copy)]
struct ParentState {
    pos: Vec3,
    rot: Quat,
    vel: Vec3,
    omega: Vec3,
    grid_offset: Vec3,
    /// Parent's topology revision at carve time. Each fragment inherits
    /// `parent_revision + 1` so the solver can tell a freshly-split body
    /// apart from its predecessor's stale warm-start state.
    topology_revision: u32,
}

impl ParentState {
    /// Map a point in the body's own local-meter frame (origin at its grid's
    /// minimum corner, matching [`carve_body_sphere`] et al.) to world space.
    fn local_to_world_m(&self, local_m: Vec3) -> Vec3 {
        self.pos + self.rot * (local_m + self.grid_offset)
    }
}

/// Given the `split_components` output (each `(sub_grid, sub_min)` where
/// `sub_min` is the fragment's min corner in *original* grid voxel coords),
/// an `anchor_voxel` (the anchor point's body-local voxel position in the
/// original grid), and a kernel of *offsets* from that anchor, find the
/// index of the fragment that owns the majority of the anchor's kernel
/// voxels. Each kernel entry is `anchor_voxel + offset` — an absolute
/// original-grid voxel coordinate. A voxel is "owned" by a fragment if it
/// falls within `[sub_min, sub_min + dims)` and is solid in that fragment's
/// grid. Returns `None` if no fragment owns a strict majority (> half of
/// the kernel voxels that are solid in *any* fragment), in which case the
/// joint is detached.
fn majority_fragment(
    components: &[(VoxelGrid, IVec3)],
    anchor_voxel: IVec3,
    kernel: &[IVec3],
) -> Option<usize> {
    if kernel.is_empty() {
        return None;
    }
    let mut counts = vec![0usize; components.len()];
    let mut total_solid = 0usize;
    for &offset in kernel {
        let kv = anchor_voxel + offset;
        for (i, (sub_grid, sub_min)) in components.iter().enumerate() {
            let local = kv - *sub_min;
            if local.cmplt(IVec3::ZERO).any() || local.cmpge(sub_grid.dims).any() {
                continue;
            }
            if sub_grid.solid(local) {
                counts[i] += 1;
                total_solid += 1;
                break; // a voxel belongs to at most one fragment
            }
        }
    }
    if total_solid == 0 {
        return None;
    }
    let threshold = total_solid / 2;
    counts
        .iter()
        .enumerate()
        .find(|&(_, &c)| c > threshold)
        .map(|(i, _)| i)
}

/// Turn one carved grid into 0+ replacement bodies, positioned and given
/// velocity to match the parent they came from, and spawn them. Shared by
/// [`carve_body_sphere_at`]/[`carve_body_capsule_at`] once the grid itself
/// has already been mutated.
///
/// `parent_slot` is the raw slot index of the parent body (before despawn).
/// `taken_joints` is the set of joints that were detached from the parent
/// via [`PhysicsWorld::take_joints_for_slot`] *before* despawn — they are
/// transferred to whichever fragment owns the majority of their kernel
/// voxels, or detached if no majority. Empty kernels (the backward-compatible
/// default) always detach: no fragment wins, so the joint is dropped. Pass an
/// empty `Vec` when the caller has no joints to transfer.
fn finish_carve(
    phys: &mut PhysicsWorld,
    registry: &MaterialRegistry,
    grid: VoxelGrid,
    voxel_size_m: f32,
    parent: ParentState,
    parent_slot: usize,
    taken_joints: Vec<Joint>,
) -> Vec<BodyId> {
    let components = split_components(&grid);
    let n = components.len();

    // Pre-compute spawn decisions for each component: (sub_min, props,
    // spawnable). All Copy, no borrows into `components` — so the later
    // `components.into_iter()` in the spawn loop is borrow-free.
    let mut comp_meta: Vec<(IVec3, MassProps, bool)> = Vec::with_capacity(n);
    for (sub_grid, sub_min) in &components {
        let count = sub_grid.solid_count();
        let spawnable = (DEBRIS_MIN_VOXELS..=MAX_BODY_VOXELS).contains(&count);
        let props = if spawnable {
            mass_props(sub_grid, registry, voxel_size_m)
        } else {
            MassProps { mass: 0.0, com_local: Vec3::ZERO, inertia_com: Mat3::IDENTITY }
        };
        let spawnable = spawnable && props.mass > 0.0;
        comp_meta.push((*sub_min, props, spawnable));
    }

    // Joint transfer — kernel vote: for each taken joint, determine which
    // fragment owns the majority of kernel voxels on the parent's side.
    // The parent is body_a (use kernel_a + anchor_a) or body_b (use
    // kernel_b + anchor_b); the other endpoint belongs to a different,
    // still-live body and is untouched. Vote *before* spawning, while
    // `components` (and its sub-grids) are still borrowed. Empty kernel or
    // no majority → None (joint will be detached, the backward-compatible
    // default). The anchor voxel is derived from the COM-relative anchor:
    // grid-min corner sits at `grid_offset` from COM, so
    // `anchor_voxel = floor((anchor - grid_offset) / voxel_size_m)`.
    let joint_winners: Vec<Option<usize>> = taken_joints
        .iter()
        .map(|j| {
            let (kernel, anchor) = if j.body_a == parent_slot {
                (&j.kernel_a, j.anchor_a)
            } else if j.body_b == parent_slot {
                (&j.kernel_b, j.anchor_b)
            } else {
                // Joint doesn't reference the parent slot — shouldn't happen
                // (take_joints_for_slot only returns joints touching the
                // slot), but guard against it: no transfer.
                return None;
            };
            if kernel.is_empty() {
                return None;
            }
            let anchor_voxel = ((anchor - parent.grid_offset) / voxel_size_m)
                .floor()
                .as_ivec3();
            majority_fragment(&components, anchor_voxel, kernel)
        })
        .collect();
    // `components` borrow ends here — all vote results are plain `Option<usize>`.

    // Spawn all fragments, recording the new slot per component index.
    let mut ids = Vec::new();
    let mut new_slots: Vec<Option<usize>> = vec![None; n];
    for (i, (sub_grid, sub_min)) in components.into_iter().enumerate() {
        let (_, props, spawnable) = comp_meta[i];
        if !spawnable {
            continue;
        }
        let sub_com_world = parent.pos
            + parent.rot
                * (sub_min.as_vec3() * voxel_size_m + parent.grid_offset + props.com_local);
        if let Some(mut body) = Body::from_grid(sub_grid, registry, voxel_size_m, sub_com_world) {
            body.rot = parent.rot;
            body.prev_rot = parent.rot;
            body.vel = parent.vel + parent.omega.cross(sub_com_world - parent.pos);
            body.omega = parent.omega;
            // Each fragment inherits parent_revision + 1: the grid was
            // carved and split, a topology change. last_step_revision stays
            // 0 (from from_grid's default) so the warm-start guard sees
            // topology_revision != last_step_revision on the first step and
            // skips the stale impulses left in `self.warm` by the parent
            // (which may have occupied this same slot before despawn).
            body.topology_revision = parent.topology_revision.wrapping_add(1);
            body.refresh_aabb();
            let id = phys.spawn(body);
            new_slots[i] = Some(id.slot as usize);
            ids.push(id);
        }
    }

    // Apply joint transfer: re-point each joint from the parent slot to the
    // winning fragment's new slot, adjusting the anchor from the parent's
    // COM frame to the fragment's COM frame. Joints with no winner (empty
    // kernel, no majority, or winning fragment wasn't spawned) are simply
    // dropped — detached, the backward-compatible default.
    //
    // The anchor adjustment: the parent's anchor is COM-relative. The
    // fragment's COM differs by `sub_min * voxel_size + grid_offset +
    // props.com_local` (in the parent's body frame, since fragment.rot ==
    // parent.rot). So `new_anchor = anchor - sub_min * voxel_size -
    // grid_offset - props.com_local`.
    for (mut joint, winner) in taken_joints.into_iter().zip(joint_winners) {
        let Some(wi) = winner else { continue };
        let Some(new_slot) = new_slots[wi] else { continue };
        let (sub_min, props, _) = comp_meta[wi];
        let anchor_delta = sub_min.as_vec3() * voxel_size_m + parent.grid_offset + props.com_local;
        if joint.body_a == parent_slot {
            joint.anchor_a -= anchor_delta;
        }
        if joint.body_b == parent_slot {
            joint.anchor_b -= anchor_delta;
        }
        phys.rejoint(joint, parent_slot, new_slot);
    }

    ids
}
/// Apply damage to a debris body's voxels. Sub-threshold impacts call this
/// instead of carving. Adds damage to the specified voxels; any voxel reaching
/// 1.0 crumbles (becomes AIR). If any voxels crumble, the body is despawned
/// and [`finish_carve`] splits + respawns fragments. If none crumble, the body
/// is mutated in-place with `damage_dirty` set.
///
/// Returns `Some(Vec<BodyId>)` if the body was despawned (crumble case --
/// caller should `replace_body` with the returned IDs), or `None` if the body
/// was mutated in-place (no crumble -- body still exists at the same ID).
pub fn apply_body_damage(
    phys: &mut PhysicsWorld,
    registry: &MaterialRegistry,
    id: BodyId,
    damage_voxels: &[(IVec3, f32)],
    voxel_size_m: f32,
) -> Option<Vec<BodyId>> {
    // Clone the grid to apply damage, because we need to read the parent
    // state before a possible despawn -- can't borrow the body mutably while
    // also reading it for position/rotation.
    let body = phys.get(id)?;
    let mut grid = body.grid.clone();
    let parent = ParentState {
        pos: body.pos,
        rot: body.rot,
        vel: body.vel,
        omega: body.omega,
        grid_offset: body.grid_offset,
        topology_revision: body.topology_revision,
    };
    let mut crumbled = false;
    for &(voxel_pos, amount) in damage_voxels {
        if grid.add_damage(voxel_pos, amount) {
            if grid.damage_at(voxel_pos) >= 1.0 && grid.solid(voxel_pos) {
                grid.set(voxel_pos, AIR);
                crumbled = true;
            }
        }
    }

    if crumbled {
        // Crumble is a topology change (voxels became air). The crumble
        // path goes through finish_carve, which sets each fragment to
        // parent.topology_revision + 1 -- that satisfies "increment when
        // voxels crumble."
        let parent_slot = id.slot as usize;
        let taken_joints = phys.take_joints_for_slot(parent_slot);
        phys.despawn(id);
        Some(finish_carve(phys, registry, grid, voxel_size_m, parent, parent_slot, taken_joints))
    } else {
        // No crumble -- damage-only is not a topology change (no voxels
        // became air, surface/contact geometry is unchanged). Mutate the
        // grid in-place, set damage_dirty for the render system, but leave
        // topology_revision alone.
        if let Some(body) = phys.get_mut(id) {
            body.grid = grid;
            body.damage_dirty = true;
        }
        None
    }
}


/// Remove exactly one voxel (grid-local coordinates) from an existing
/// body's own grid, splitting it into however many disconnected fragments
/// result. Same despawn/replace/return semantics as [`carve_body_sphere_at`].
/// Use this over a tiny-radius sphere for single-voxel breaks -- see
/// [`carve_body_voxel`] for why a sphere can't do this reliably.
pub fn carve_body_voxel_at(
    phys: &mut PhysicsWorld,
    registry: &MaterialRegistry,
    id: BodyId,
    local_voxel: IVec3,
) -> Vec<BodyId> {
    let Some(body) = phys.get(id) else {
        return Vec::new();
    };
    let voxel_size_m = body.half_voxel * 2.0;
    let mut grid = body.grid.clone();
    let removed = carve_body_voxel(&mut grid, local_voxel);
    if removed.is_empty() {
        return Vec::new();
    }
    let parent = ParentState {
        pos: body.pos,
        rot: body.rot,
        vel: body.vel,
        omega: body.omega,
        grid_offset: body.grid_offset,
        // Pass the parent's current revision; finish_carve adds +1.
        topology_revision: body.topology_revision,
    };
    let parent_slot = id.slot as usize;
    let taken_joints = phys.take_joints_for_slot(parent_slot);
    phys.despawn(id);
    finish_carve(phys, registry, grid, voxel_size_m, parent, parent_slot, taken_joints)
}

/// Carve a sphere out of an existing body's own grid (world-space center),
/// splitting it into however many disconnected fragments result. Despawns
/// the original body and spawns 0+ replacements (see the module docs for how
/// each fragment's placement/velocity is derived). Returns the ids of every
/// body spawned; empty (and the original body left untouched) if nothing
/// was removed.
pub fn carve_body_sphere_at(
    phys: &mut PhysicsWorld,
    registry: &MaterialRegistry,
    id: BodyId,
    center_world_m: Vec3,
    radius_m: f32,
) -> Vec<BodyId> {
    let Some(body) = phys.get(id) else {
        return Vec::new();
    };
    let voxel_size_m = body.half_voxel * 2.0;
    let center_local = body.rot.inverse() * (center_world_m - body.pos) - body.grid_offset;
    let mut grid = body.grid.clone();
    let removed = carve_body_sphere(&mut grid, center_local, radius_m, voxel_size_m);
    if removed.is_empty() {
        return Vec::new();
    }
    let parent = ParentState {
        pos: body.pos,
        rot: body.rot,
        vel: body.vel,
        omega: body.omega,
        grid_offset: body.grid_offset,
        // Pass the parent's current revision; finish_carve adds +1.
        topology_revision: body.topology_revision,
    };
    let parent_slot = id.slot as usize;
    let taken_joints = phys.take_joints_for_slot(parent_slot);
    phys.despawn(id);
    finish_carve(phys, registry, grid, voxel_size_m, parent, parent_slot, taken_joints)
}

/// Fraction of an impact fracture's removed voxels that become tiny flying
/// debris chips instead of simply vanishing -- see
/// [`crate::destruction::spawn_debris_chips`] for the same idea against
/// world terrain. An impact fracture typically removes only a handful of
/// voxels at a time (unlike a bomb's crater), so the fraction is higher and
/// the cap much lower.
const IMPACT_CHIP_FRACTION: f32 = 0.5;
/// Absolute cap on chips per fracture event, regardless of how much was
/// removed. Paired with `vox-app`'s `MAX_FRACTURE_RADIUS_VOX` (which bounds
/// how much a single fracture can ever remove), so this is no longer the
/// binding constraint on total debris load the way it was when radius was
/// unbounded -- raised from an earlier `6` because that read as "a handful
/// of specks next to a big empty hole" rather than a chunk actually
/// breaking apart into pieces.
const MAX_IMPACT_CHIPS: usize = 24;
/// Chip launch speed as a fraction of the impact speed that caused the
/// fracture -- a graze barely nudges its chips loose, a violent hit sends
/// them flying: the same "proportional to what actually happened" idea as
/// the fracture radius itself ([`crate::destruction`]'s `fracture_radius_vox`
/// analog lives in `vox-app`, but the chips scale with the same input).
const IMPACT_CHIP_SPEED_SCALE: f32 = 0.5;
/// Hard ceiling on a chip's launch speed -- same reasoning as
/// `spawn_debris_chips`'s own cap: a chip's mass barely varies, so scaling
/// unboundedly with impact speed could fling light rubble across the map
/// instead of scattering it visibly around the impact.
const IMPACT_CHIP_MAX_SPEED_M_S: f32 = 4.0;
/// Max angular speed (rad/s) randomly added to a chip's inherited spin.
const IMPACT_CHIP_SPIN_MAX: f32 = 0.3;

/// Turn a deterministic sample of an impact fracture's removed voxels into
/// small flying debris chips launched along `impact_dir`, scaled by
/// `impact_speed`, instead of letting all of it simply vanish into empty
/// space -- "tiny chunks fly off, a satisfying mess," not "an orb of
/// voxels deleted from space." Chips are a small L-shaped triomino, not a
/// single voxel: see `spawn_debris_chips`'s own doc comment for why a lone
/// voxel is a physics-degenerate rotation case. `parent` must be a snapshot
/// of the carved body's transform/motion taken *before* it was despawned
/// (its own position/rotation no longer exist afterward).
#[expect(clippy::too_many_arguments, reason = "internal chip assembly")]
fn spawn_impact_chips(
    phys: &mut PhysicsWorld,
    registry: &MaterialRegistry,
    removed: &[(IVec3, Voxel)],
    voxel_size_m: f32,
    parent: &ParentState,
    impact_dir: Vec3,
    impact_speed: f32,
    seed: u32,
) -> Vec<BodyId> {
    if removed.is_empty() {
        return Vec::new();
    }
    let target =
        ((removed.len() as f32 * IMPACT_CHIP_FRACTION) as usize).clamp(1, MAX_IMPACT_CHIPS);

    // Same deterministic partial-selection sampling as `spawn_debris_chips`.
    let mut ranked: Vec<(u32, usize)> = removed
        .iter()
        .enumerate()
        .map(|(i, _)| (small_hash(seed, i as u32), i))
        .collect();
    if target < ranked.len() {
        ranked.select_nth_unstable_by_key(target - 1, |&(h, _)| h);
        ranked.truncate(target);
    }

    let dir = if impact_dir.length_squared() > 1e-9 {
        impact_dir.normalize()
    } else {
        Vec3::Y
    };
    let speed = (impact_speed * IMPACT_CHIP_SPEED_SCALE).min(IMPACT_CHIP_MAX_SPEED_M_S);

    // Build a map of removed voxels for O(1) adjacency + material lookup.
    let removed_map: FxHashMap<IVec3, Voxel> = removed.iter().copied().collect();
    let removed_set: FxHashSet<IVec3> = removed_map.keys().copied().collect();

    let mut ids = Vec::with_capacity(target);
    for &(h, i) in ranked.iter().take(target) {
        let (seed_vox, _mat) = removed[i];

        // Build the chip from the actual removed voxels: flood-fill from
        // the seed through the removed set (6-connected), up to a small
        // cap. This produces a chunk shaped like the real carved material,
        // not a generic template shape.
        let cluster = crate::destruction::cluster_removed(&removed_set, seed_vox, 8);
        if cluster.is_empty() {
            continue;
        }
        let mut min = IVec3::splat(i32::MAX);
        let mut max = IVec3::splat(i32::MIN);
        for &v in &cluster {
            min = min.min(v);
            max = max.max(v);
        }
        let dims = max - min + IVec3::ONE;
        let mut voxels = vec![AIR; (dims.x * dims.y * dims.z) as usize];
        for &v in &cluster {
            let mat = removed_map.get(&v).copied().unwrap_or(AIR);
            let l = v - min;
            voxels[grid_index(dims, l)] = mat;
        }
        let grid = VoxelGrid::new(dims, voxels);
        let cluster_center_local = voxel_center_m(min, voxel_size_m)
            + dims.as_vec3() * voxel_size_m * 0.5;
        let cluster_center_world = parent.local_to_world_m(cluster_center_local);
        let Some(mut body) = Body::from_grid(grid, registry, voxel_size_m, cluster_center_world)
        else {
            continue;
        };
        // Chips are spawned into potentially reused freed slots. A fresh
        // body defaults to revision 0 == last_step 0, which would make the
        // warm-start guard think the body is unchanged and re-inject the
        // stale impulses left by whatever body previously occupied this
        // slot. Set parent+1 so the guard fires on the first step.
        body.topology_revision = parent.topology_revision.wrapping_add(1);

        // Radial scatter from the cluster's actual position, not the seed.
        let radial_dir = (cluster_center_world - parent.pos).normalize_or(dir);
        let blend = 0.4 + (small_hash(h, 42) as f32 / u32::MAX as f32) * 0.4; // 0.4..0.8
        let chip_dir = (dir * (1.0 - blend) + radial_dir * blend).normalize_or(dir);
        body.vel = parent.vel + chip_dir * speed;
        let h2 = small_hash(h, i as u32);
        body.omega = parent.omega
            + Vec3::new(
                ((h2 & 0xFF) as f32 / 255.0 - 0.5) * 2.0,
                (((h2 >> 8) & 0xFF) as f32 / 255.0 - 0.5) * 2.0,
                (((h2 >> 16) & 0xFF) as f32 / 255.0 - 0.5) * 2.0,
            ) * IMPACT_CHIP_SPIN_MAX;

        ids.push(phys.spawn(body));
    }
    ids
}

/// Same as [`carve_body_sphere_at`], but for the material-based
/// impact-fracture path: also scatters a sample of the carved material as
/// small flying debris chips (see [`spawn_impact_chips`]) instead of
/// letting it all simply vanish. `impact_dir` should be the direction the
/// impact pushed into the body (`ImpactEvent::push_dir`); `impact_speed`
/// scales chip launch speed; `seed` makes the chip sample and their
/// spin/velocity jitter deterministic.
#[expect(clippy::too_many_arguments, reason = "internal chip assembly")]
pub fn carve_body_sphere_at_impact(
    phys: &mut PhysicsWorld,
    registry: &MaterialRegistry,
    id: BodyId,
    center_world_m: Vec3,
    radius_m: f32,
    impact_dir: Vec3,
    impact_speed: f32,
    seed: u32,
) -> Vec<BodyId> {
    let Some(body) = phys.get(id) else {
        return Vec::new();
    };
    let voxel_size_m = body.half_voxel * 2.0;
    let center_local = body.rot.inverse() * (center_world_m - body.pos) - body.grid_offset;
    let mut grid = body.grid.clone();
    let removed = carve_body_sphere(&mut grid, center_local, radius_m, voxel_size_m);
    if removed.is_empty() {
        return Vec::new();
    }
    let parent = ParentState {
        pos: body.pos,
        rot: body.rot,
        vel: body.vel,
        omega: body.omega,
        grid_offset: body.grid_offset,
        // Pass the parent's current revision; finish_carve adds +1.
        topology_revision: body.topology_revision,
    };
    let parent_slot = id.slot as usize;
    let taken_joints = phys.take_joints_for_slot(parent_slot);
    phys.despawn(id);
    let mut ids = finish_carve(phys, registry, grid, voxel_size_m, parent, parent_slot, taken_joints);
    ids.extend(spawn_impact_chips(
        phys,
        registry,
        &removed,
        voxel_size_m,
        &parent,
        impact_dir,
        impact_speed,
        seed,
    ));
    ids
}

/// Carve a jagged [`ExplosionShape`] out of an existing body's own grid
/// (world-space center) -- the Bomb tool hitting debris. Same
/// despawn/replace/return semantics as [`carve_body_sphere_at`]; `seed`
/// should be the blast's own per-shot seed (also driving the impulse's
/// spin), so the fragment shape and its motion are both deterministic from
/// one call.
pub fn carve_body_explosion_at(
    phys: &mut PhysicsWorld,
    registry: &MaterialRegistry,
    id: BodyId,
    center_world_m: Vec3,
    radius_m: f32,
    seed: u32,
) -> Vec<BodyId> {
    let Some(body) = phys.get(id) else {
        return Vec::new();
    };
    let voxel_size_m = body.half_voxel * 2.0;
    let center_local = body.rot.inverse() * (center_world_m - body.pos) - body.grid_offset;
    let mut grid = body.grid.clone();
    let removed = carve_body_explosion(&mut grid, center_local, radius_m, voxel_size_m, seed);
    if removed.is_empty() {
        return Vec::new();
    }
    let parent = ParentState {
        pos: body.pos,
        rot: body.rot,
        vel: body.vel,
        omega: body.omega,
        grid_offset: body.grid_offset,
        // Pass the parent's current revision; finish_carve adds +1.
        topology_revision: body.topology_revision,
    };
    let parent_slot = id.slot as usize;
    let taken_joints = phys.take_joints_for_slot(parent_slot);
    phys.despawn(id);
    finish_carve(phys, registry, grid, voxel_size_m, parent, parent_slot, taken_joints)
}

/// Carve a tunnel through an existing body's own grid (world-space
/// endpoints), splitting it into however many disconnected fragments
/// result. Same placement/velocity/return semantics as
/// [`carve_body_sphere_at`].
pub fn carve_body_capsule_at(
    phys: &mut PhysicsWorld,
    registry: &MaterialRegistry,
    id: BodyId,
    start_world_m: Vec3,
    end_world_m: Vec3,
    radius_m: f32,
) -> Vec<BodyId> {
    let Some(body) = phys.get(id) else {
        return Vec::new();
    };
    let voxel_size_m = body.half_voxel * 2.0;
    let inv_rot = body.rot.inverse();
    let start_local = inv_rot * (start_world_m - body.pos) - body.grid_offset;
    let end_local = inv_rot * (end_world_m - body.pos) - body.grid_offset;
    let mut grid = body.grid.clone();
    let removed = carve_body_capsule(&mut grid, start_local, end_local, radius_m, voxel_size_m);
    if removed.is_empty() {
        return Vec::new();
    }
    let parent = ParentState {
        pos: body.pos,
        rot: body.rot,
        vel: body.vel,
        omega: body.omega,
        grid_offset: body.grid_offset,
        // Pass the parent's current revision; finish_carve adds +1.
        topology_revision: body.topology_revision,
    };
    let parent_slot = id.slot as usize;
    let taken_joints = phys.take_joints_for_slot(parent_slot);
    phys.despawn(id);
    finish_carve(phys, registry, grid, voxel_size_m, parent, parent_slot, taken_joints)
}

#[cfg(test)]
mod tests {
    use super::*;

    const STONE: Voxel = Voxel(1);

    fn registry() -> MaterialRegistry {
        MaterialRegistry::from_toml_str(
            r#"
            [[material]]
            name = "stone"
            color = [0.5, 0.5, 0.5]
            density = 2600.0
            strength = 8.0
            "#,
            "test.toml",
        )
        .expect("registry")
    }

    fn solid_grid(dims: IVec3) -> VoxelGrid {
        VoxelGrid::new(dims, vec![STONE; (dims.x * dims.y * dims.z) as usize])
    }

    #[test]
    fn carving_the_middle_of_a_bar_splits_it_into_two_bodies() {
        // A 1x1x20 bar (voxel size 1m): carving out its middle must split it
        // into two separate fragments, each inheriting the parent's motion.
        let reg = registry();
        let grid = solid_grid(IVec3::new(1, 1, 20));
        let mut phys = PhysicsWorld::new();
        let mut body = Body::from_grid(grid, &reg, 1.0, Vec3::new(0.0, 10.0, 0.0)).unwrap();
        body.vel = Vec3::new(1.0, 0.0, 0.0);
        body.omega = Vec3::new(0.0, 2.0, 0.0);
        let id = phys.spawn(body);

        // Middle of the bar in world space: bar spans local y in [0, 20),
        // COM at local y=10 == world y=10 (grid_offset centers it), so the
        // middle voxel (local y=9..10) sits at world y ~ 0.0 (grid_offset.y
        // = -10). Just carve at the body's own reported position (its COM),
        // which is exactly the middle of a uniform bar.
        let center_world = phys.get(id).unwrap().pos;
        let spawned = carve_body_sphere_at(&mut phys, &reg, id, center_world, 0.6);

        assert!(phys.get(id).is_none(), "the original body must be gone");
        assert_eq!(spawned.len(), 2, "must split into exactly two fragments");
        for &fid in &spawned {
            let f = phys.get(fid).expect("alive");
            assert!(
                f.omega == Vec3::new(0.0, 2.0, 0.0),
                "fragments must inherit the parent's angular velocity"
            );
        }
    }

    #[test]
    fn carving_a_corner_off_a_cube_leaves_one_smaller_body() {
        let reg = registry();
        let grid = solid_grid(IVec3::splat(6));
        let mut phys = PhysicsWorld::new();
        let body = Body::from_grid(grid, &reg, 0.5, Vec3::ZERO).unwrap();
        let id = phys.spawn(body);

        let corner_world = phys.get(id).unwrap().pos + Vec3::splat(1.4); // near a corner
        let spawned = carve_body_sphere_at(&mut phys, &reg, id, corner_world, 0.6);

        assert!(phys.get(id).is_none());
        assert_eq!(
            spawned.len(),
            1,
            "chipping a corner must not split the cube"
        );
        let f = phys.get(spawned[0]).unwrap();
        assert!(
            f.grid.solid_count() < 6 * 6 * 6,
            "must have lost some voxels"
        );
    }

    #[test]
    fn missing_a_body_entirely_is_a_harmless_no_op() {
        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let grid = solid_grid(IVec3::splat(3));
        let id = phys.spawn(Body::from_grid(grid, &reg, 0.2, Vec3::ZERO).unwrap());
        phys.despawn(id);

        let spawned = carve_body_sphere_at(&mut phys, &reg, id, Vec3::ZERO, 1.0);
        assert!(spawned.is_empty());
    }

    #[test]
    fn missing_the_body_with_the_carve_is_a_harmless_no_op() {
        // Carving somewhere far outside the body's own grid removes nothing.
        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let grid = solid_grid(IVec3::splat(3));
        let body = Body::from_grid(grid, &reg, 0.2, Vec3::ZERO).unwrap();
        let id = phys.spawn(body);

        let spawned = carve_body_sphere_at(&mut phys, &reg, id, Vec3::splat(100.0), 0.5);
        assert!(spawned.is_empty());
        assert!(phys.get(id).is_some(), "untouched body must survive");
    }

    #[test]
    fn capsule_tunnels_through_the_core_of_a_body_leaving_a_hollow_tube() {
        // A 20x3x3 bar; a thin beam straight through its central axis only
        // hollows out the center column, leaving the surrounding ring of 8
        // columns intact and still connected (face-adjacent all the way
        // around, at every length position) -- one fragment, not zero, not
        // split into pieces.
        let reg = registry();
        let grid = solid_grid(IVec3::new(20, 3, 3));
        let original_count = grid.solid_count();
        let mut phys = PhysicsWorld::new();
        let body = Body::from_grid(grid, &reg, 0.2, Vec3::ZERO).unwrap();
        let id = phys.spawn(body);

        let pos = phys.get(id).unwrap().pos;
        let start = pos + Vec3::new(-10.0, 0.0, 0.0);
        let end = pos + Vec3::new(10.0, 0.0, 0.0);
        let spawned = carve_body_capsule_at(&mut phys, &reg, id, start, end, 0.15);

        assert!(phys.get(id).is_none(), "the original body must be gone");
        assert_eq!(
            spawned.len(),
            1,
            "hollowing the core must not split the tube"
        );
        let remaining = phys.get(spawned[0]).unwrap().grid.solid_count();
        assert!(
            remaining < original_count,
            "must have removed the center column: {remaining} vs {original_count}"
        );
    }

    #[test]
    fn split_components_carries_damage() {
        // Build a 4x1x1 grid, damage the left half, disconnect by removing middle.
        let mut grid = VoxelGrid::new(IVec3::new(4, 1, 1), vec![STONE; 4]);
        grid.add_damage(IVec3::new(0, 0, 0), 0.7);
        grid.set(IVec3::new(2, 0, 0), AIR); // disconnect left from right
        let components = split_components(&grid);
        assert_eq!(components.len(), 2, "must split into 2 components");
        // Left component (voxel 0) should carry damage 0.7.
        let left = components.iter().find(|(_, min)| min.x == 0).unwrap();
        assert_eq!(left.0.damage_at(IVec3::ZERO), 0.7, "left fragment must carry damage");
        // Right component (voxel 3) should have 0 damage.
        let right = components.iter().find(|(_, min)| min.x == 3).unwrap();
        assert_eq!(right.0.damage_at(IVec3::ZERO), 0.0, "right fragment must be pristine");
    }

    #[test]
    fn fresh_body_starts_at_revision_zero() {
        let reg = registry();
        let grid = solid_grid(IVec3::splat(3));
        let body = Body::from_grid(grid, &reg, 0.2, Vec3::ZERO).unwrap();
        assert_eq!(body.topology_revision, 0, "fresh body starts at revision 0");
        assert_eq!(body.last_step_revision, 0);
    }

    #[test]
    fn carved_fragments_inherit_parent_revision_plus_one() {
        // carve_*_at passes the parent's current revision (no bump), and
        // finish_carve adds +1. So fragments of a revision-0 body land at 1.
        let reg = registry();
        let grid = solid_grid(IVec3::new(1, 1, 20));
        let mut phys = PhysicsWorld::new();
        let body = Body::from_grid(grid, &reg, 1.0, Vec3::new(0.0, 10.0, 0.0)).unwrap();
        let id = phys.spawn(body);

        let center_world = phys.get(id).unwrap().pos;
        let spawned = carve_body_sphere_at(&mut phys, &reg, id, center_world, 0.6);

        assert_eq!(spawned.len(), 2, "must split into two");
        for &fid in &spawned {
            let f = phys.get(fid).expect("alive");
            assert_eq!(
                f.topology_revision, 1,
                "fragment of a revision-0 body must be at revision 1 (finish_carve +1)"
            );
        }
    }

    #[test]
    fn in_place_damage_does_not_bump_revision() {
        // apply_body_damage without a crumble is damage-only: no voxels
        // became air, surface/contact geometry is unchanged. The topology
        // revision must NOT change (it tracks geometry, not damage values).
        let reg = registry();
        let grid = solid_grid(IVec3::splat(3));
        let mut phys = PhysicsWorld::new();
        let id = phys.spawn(Body::from_grid(grid, &reg, 0.2, Vec3::ZERO).unwrap());

        let before = phys.get(id).unwrap().topology_revision;
        // Sub-threshold damage: 0.3 won't crumble a pristine voxel.
        let result = apply_body_damage(
            &mut phys,
            &reg,
            id,
            &[(IVec3::ZERO, 0.3)],
            0.2,
        );
        assert!(result.is_none(), "no crumble -- body stays in-place");
        let after = phys.get(id).unwrap().topology_revision;
        assert_eq!(after, before, "in-place damage must not bump revision");
    }

    #[test]
    fn crumble_fragments_inherit_parent_revision_plus_one() {
        // Crumble goes through finish_carve, which gives fragments
        // parent.topology_revision + 1. A revision-0 body that crumbles
        // directly (single hit >= 1.0) produces fragments at revision 1.
        let reg = registry();
        let grid = solid_grid(IVec3::splat(3));
        let mut phys = PhysicsWorld::new();
        let id = phys.spawn(Body::from_grid(grid, &reg, 0.2, Vec3::ZERO).unwrap());

        // A single hit of 1.0 crumbles the voxel immediately.
        let result = apply_body_damage(&mut phys, &reg, id, &[(IVec3::ZERO, 1.0)], 0.2);
        assert!(result.is_some(), "must crumble and despawn");
        let spawned = result.unwrap();
        for &fid in &spawned {
            let f = phys.get(fid).expect("alive");
            assert_eq!(
                f.topology_revision, 1,
                "crumble fragment of a revision-0 body must be at revision 1"
            );
        }
    }

    #[test]
    fn joint_with_kernel_transfers_to_majority_fragment() {
        // A 1x1x20 bar (voxel size 1m) joined to a static anchor body.
        // The bar runs along z (dims 1x1x20, grid_offset = (-0.5, -0.5, -10)).
        // The joint's kernel_a is a 3-voxel neighborhood around the anchor at
        // z=18 (near the far end). Carving the middle splits the bar; the far
        // fragment owns all 3 kernel voxels, so the joint transfers to it.
        let reg = registry();
        let grid = solid_grid(IVec3::new(1, 1, 20));
        let mut phys = PhysicsWorld::new();

        // Bar at COM (0,10,0); grid spans world z [0, 20).
        let bar = Body::from_grid(grid, &reg, 1.0, Vec3::new(0.0, 10.0, 0.0)).unwrap();
        let bar_id = phys.spawn(bar);

        // A separate anchor body to join to.
        let anchor_grid = solid_grid(IVec3::splat(2));
        let anchor_body =
            Body::from_grid(anchor_grid, &reg, 1.0, Vec3::new(5.0, 15.0, 0.0)).unwrap();
        let anchor_id = phys.spawn(anchor_body);

        // Anchor on bar at COM + (0, 0, 8) → grid voxel (0, 0, 18).
        // Kernel: 3 voxels centered on (0, 0, 18) along z.
        let anchor_bar = Vec3::new(0.0, 0.0, 8.0);
        let kernel = vec![
            IVec3::new(0, 0, -1),
            IVec3::new(0, 0, 0),
            IVec3::new(0, 0, 1),
        ];
        phys.add_joint_with_kernel(
            bar_id,
            anchor_id,
            anchor_bar,
            Vec3::ZERO,
            5.0,
            0.0,
            kernel,
            Vec::new(),
            Vec3::ZERO,
            Vec3::ZERO,
        );
        assert_eq!(phys.joints().len(), 1, "joint must exist before carve");

        // Carve at the bar's COM (world z=10): removes voxel z=10, splitting
        // into [0..10) = 10 voxels and [11..20) = 9 voxels. Same as the
        // existing bar-split test.
        let center_world = phys.get(bar_id).unwrap().pos;
        let spawned = carve_body_sphere_at(&mut phys, &reg, bar_id, center_world, 0.6);
        assert_eq!(spawned.len(), 2, "must split into two fragments");
        assert!(phys.get(bar_id).is_none(), "original bar must be gone");

        // The joint must still exist, now pointing from a fragment to anchor_id.
        assert_eq!(phys.joints().len(), 1, "joint must transfer, not detach");
        let j = &phys.joints()[0];
        assert!(
            j.body_a == anchor_id.slot as usize || j.body_b == anchor_id.slot as usize,
            "joint must still reference the anchor body"
        );
        // The other side must be one of the spawned fragments, not the old bar slot.
        let frag_slot = if j.body_a == anchor_id.slot as usize {
            j.body_b
        } else {
            j.body_a
        };
        assert!(
            spawned.iter().any(|id| id.slot as usize == frag_slot),
            "joint must point to a spawned fragment, not the old bar slot"
        );
    }

    #[test]
    fn joint_with_empty_kernel_detaches_on_fracture() {
        // Same 1x1x20 bar as the transfer test, but with an empty kernel
        // (backward-compatible default). The joint must detach when the bar
        // splits — no transfer.
        let reg = registry();
        let grid = solid_grid(IVec3::new(1, 1, 20));
        let mut phys = PhysicsWorld::new();

        let bar = Body::from_grid(grid, &reg, 1.0, Vec3::new(0.0, 10.0, 0.0)).unwrap();
        let bar_id = phys.spawn(bar);

        let anchor_grid = solid_grid(IVec3::splat(2));
        let anchor_body =
            Body::from_grid(anchor_grid, &reg, 1.0, Vec3::new(5.0, 15.0, 0.0)).unwrap();
        let anchor_id = phys.spawn(anchor_body);

        // add_joint (no kernel) = backward-compatible.
        phys.add_joint(bar_id, anchor_id, Vec3::new(0.0, 0.0, 8.0), Vec3::ZERO, 5.0, 0.0);
        assert_eq!(phys.joints().len(), 1);

        let center_world = phys.get(bar_id).unwrap().pos;
        let _spawned = carve_body_sphere_at(&mut phys, &reg, bar_id, center_world, 0.6);

        assert_eq!(
            phys.joints().len(),
            0,
            "empty-kernel joint must detach on fracture (backward compat)"
        );
    }

    #[test]
    fn joint_with_no_majority_detaches() {
        // Kernel straddles the carve boundary so no fragment owns a majority.
        // 1x1x20 bar at COM (0,10,0); grid_offset = (-0.5, -0.5, -10).
        // Anchor at COM + (0,0,0) → grid voxel (0,0,10). Kernel offsets
        // [-2, 0, +2] along z → voxels 8, 10, 12. Carving at the COM removes
        // voxels 9 and 10 (both within 0.6m of world z=0). Voxel 8 survives
        // in the bottom fragment [0..9), voxel 12 survives in the top
        // fragment [11..20) — 1 vs 1, no majority → detach.
        let reg = registry();
        let grid = solid_grid(IVec3::new(1, 1, 20));
        let mut phys = PhysicsWorld::new();

        let bar = Body::from_grid(grid, &reg, 1.0, Vec3::new(0.0, 10.0, 0.0)).unwrap();
        let bar_id = phys.spawn(bar);

        let anchor_grid = solid_grid(IVec3::splat(2));
        let anchor_body =
            Body::from_grid(anchor_grid, &reg, 1.0, Vec3::new(5.0, 15.0, 0.0)).unwrap();
        let anchor_id = phys.spawn(anchor_body);

        let anchor_bar = Vec3::new(0.0, 0.0, 0.0);
        let kernel = vec![
            IVec3::new(0, 0, -2),
            IVec3::new(0, 0, 0),
            IVec3::new(0, 0, 2),
        ];
        phys.add_joint_with_kernel(
            bar_id,
            anchor_id,
            anchor_bar,
            Vec3::ZERO,
            5.0,
            0.0,
            kernel,
            Vec::new(),
            Vec3::ZERO,
            Vec3::ZERO,
        );

        let center_world = phys.get(bar_id).unwrap().pos;
        let _spawned = carve_body_sphere_at(&mut phys, &reg, bar_id, center_world, 0.6);
        assert_eq!(
            phys.joints().len(),
            0,
            "no-majority kernel must detach the joint"
        );
    }
}
