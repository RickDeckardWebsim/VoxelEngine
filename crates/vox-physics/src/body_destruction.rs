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

use glam::{IVec3, Quat, Vec3};
use vox_core::{FxHashMap, FxHashSet, MaterialRegistry};
use vox_core::consts::{DEBRIS_MIN_VOXELS, MAX_BODY_VOXELS};
use vox_world::{AIR, Voxel};

use vox_core::voxel_center_m;

use crate::BodyId;
use crate::body::{Body, VoxelGrid, mass_props};
use crate::destruction::{ExplosionShape, small_hash};
use crate::solver::PhysicsWorld;

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
}

impl ParentState {
    /// Map a point in the body's own local-meter frame (origin at its grid's
    /// minimum corner, matching [`carve_body_sphere`] et al.) to world space.
    fn local_to_world_m(&self, local_m: Vec3) -> Vec3 {
        self.pos + self.rot * (local_m + self.grid_offset)
    }
}

/// Turn one carved grid into 0+ replacement bodies, positioned and given
/// velocity to match the parent they came from, and spawn them. Shared by
/// [`carve_body_sphere_at`]/[`carve_body_capsule_at`] once the grid itself
/// has already been mutated.
fn finish_carve(
    phys: &mut PhysicsWorld,
    registry: &MaterialRegistry,
    grid: VoxelGrid,
    voxel_size_m: f32,
    parent: ParentState,
) -> Vec<BodyId> {
    let mut ids = Vec::new();
    for (sub_grid, sub_min) in split_components(&grid) {
        let count = sub_grid.solid_count();
        if !(DEBRIS_MIN_VOXELS..=MAX_BODY_VOXELS).contains(&count) {
            continue; // dust, or an oversize safety valve -- either way, gone.
        }
        let props = mass_props(&sub_grid, registry, voxel_size_m);
        if props.mass <= 0.0 {
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
            body.refresh_aabb();
            ids.push(phys.spawn(body));
        }
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
        // Despawn + split + respawn, same as the fracture path.
        phys.despawn(id);
        Some(finish_carve(phys, registry, grid, voxel_size_m, parent))
    } else {
        // No crumble -- mutate the body's grid in-place.
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
    };
    phys.despawn(id);
    finish_carve(phys, registry, grid, voxel_size_m, parent)
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
    };
    phys.despawn(id);
    finish_carve(phys, registry, grid, voxel_size_m, parent)
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
    };
    phys.despawn(id);
    let mut ids = finish_carve(phys, registry, grid, voxel_size_m, parent);
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
    };
    phys.despawn(id);
    finish_carve(phys, registry, grid, voxel_size_m, parent)
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
    };
    phys.despawn(id);
    finish_carve(phys, registry, grid, voxel_size_m, parent)
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
}
