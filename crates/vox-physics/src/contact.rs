//! Contact generation: body surface points versus the world grid and versus
//! other bodies' grids.
//!
//! Every surface sample point behaves as a small sphere of radius
//! `half_voxel`. Points inside solid voxels push out through their nearest
//! empty face; points within the radius of a neighboring solid face contact
//! before penetrating. Normals are voxel-face-aligned in the *owning grid's
//! frame* — world contacts get world-axis normals, body-body contacts get the
//! target body's rotated face normals. This is the voxel-native narrowphase:
//! no convex hulls, just grids sampling grids.

use glam::{IVec3, Mat3, Quat, Vec3};
use vox_core::voxel_at;
use vox_world::World;

use crate::body::Body;

/// Stable identity of a contact across steps: (body a, body b or MAX for the
/// static world, cell in the target grid, face id).
pub type ContactKey = (u32, u32, IVec3, u8);

/// Slot value in keys standing for the static world.
pub const WORLD_SLOT: u32 = u32::MAX;

/// One contact. `body` receives `+impulse`, `body_b` (if any) receives the
/// opposite; the normal points from the target toward `body`.
pub struct Contact {
    pub body: usize,
    /// The other dynamic body, or `None` when contacting the static world or
    /// a sleeping body treated as static.
    pub body_b: Option<usize>,
    pub normal: Vec3,
    pub depth: f32,
    /// Contact point relative to each COM (world orientation).
    pub r_arm: Vec3,
    pub r_arm_b: Vec3,
    pub key: ContactKey,
    pub t1: Vec3,
    pub t2: Vec3,
    pub kn: f32,
    pub kt1: f32,
    pub kt2: f32,
    pub acc_n: f32,
    pub acc_t1: f32,
    pub acc_t2: f32,
    /// How fast `body` (relative to `body_b`, or the static world) was
    /// closing on this contact *before* this substep resolved it -- i.e.
    /// gravity/prior velocity only, no constraint impulse yet. A body
    /// resting quietly has an accumulated normal impulse (`acc_n`) that
    /// looks identical frame to frame to one that just landed hard (both
    /// simply "hold the body up against gravity"), but its approach speed
    /// is near zero every single step, where a genuine impact's is not.
    /// This is what separates a real collision from steady load-bearing
    /// contact for impact-fracture purposes (see `PhysicsWorld::substep`).
    pub approach_speed: f32,
}

/// Face directions with ids matching the mesher convention.
pub const FACE_DIRS: [(u8, IVec3); 6] = [
    (0, IVec3::X),
    (1, IVec3::NEG_X),
    (2, IVec3::Y),
    (3, IVec3::NEG_Y),
    (4, IVec3::Z),
    (5, IVec3::NEG_Z),
];

/// Any orthonormal tangent basis for a unit normal.
#[inline]
fn tangent_basis(n: Vec3) -> (Vec3, Vec3) {
    let helper = if n.x.abs() > 0.9 { Vec3::Y } else { Vec3::X };
    let t1 = n.cross(helper).normalize();
    let t2 = n.cross(t1);
    (t1, t2)
}

/// Signed distance from point `p` to the face plane of cell `v` in direction
/// `dir`, in a grid whose cells have edge `s` and origin at 0.
#[inline]
fn face_dist(p: Vec3, v: IVec3, dir: IVec3, s: f32) -> f32 {
    if dir.x == 1 {
        (v.x + 1) as f32 * s - p.x
    } else if dir.x == -1 {
        p.x - v.x as f32 * s
    } else if dir.y == 1 {
        (v.y + 1) as f32 * s - p.y
    } else if dir.y == -1 {
        p.y - v.y as f32 * s
    } else if dir.z == 1 {
        (v.z + 1) as f32 * s - p.z
    } else {
        p.z - v.z as f32 * s
    }
}

/// `k = 1/m + n · ((I⁻¹ (r × n)) × r)` summed over both bodies of a contact.
#[inline]
fn effective_mass(
    n: Vec3,
    inv_mass_a: f32,
    inv_iw_a: &Mat3,
    r_a: Vec3,
    b_terms: Option<(f32, &Mat3, Vec3)>,
) -> f32 {
    let ra_n = r_a.cross(n);
    let mut k = inv_mass_a + n.dot((*inv_iw_a * ra_n).cross(r_a));
    if let Some((inv_mass_b, inv_iw_b, r_b)) = b_terms {
        let rb_n = r_b.cross(n);
        k += inv_mass_b + n.dot((*inv_iw_b * rb_n).cross(r_b));
    }
    k
}

/// Generate world contacts for one awake body into `out`.
pub fn world_contacts(body: &Body, slot: usize, world: &World, out: &mut Vec<Contact>) {
    let s = world.cfg.voxel_size_m;
    let r_point = body.half_voxel;
    let inv_iw = body.inv_inertia_world();

    for &p_local in &body.surface {
        let r_arm = body.rot * p_local;
        let p_w = body.pos + r_arm;
        let v = voxel_at(p_w, s);

        if world.solid(v) {
            let mut best: Option<(Vec3, f32, u8)> = None;
            for (face_id, dir) in FACE_DIRS {
                if world.solid(v + dir) {
                    continue;
                }
                let d = face_dist(p_w, v, dir, s);
                if best.is_none_or(|(_, bd, _)| d < bd) {
                    best = Some((dir.as_vec3(), d, face_id));
                }
            }
            let (n, dist, face_id) = best.unwrap_or((Vec3::Y, s * 0.5, 2));
            push_world_contact(
                out,
                body,
                slot,
                r_arm,
                n,
                r_point + dist,
                v,
                face_id,
                &inv_iw,
            );
        } else {
            for (face_id, dir) in FACE_DIRS {
                if !world.solid(v + dir) {
                    continue;
                }
                let d = face_dist(p_w, v, dir, s);
                if d < r_point {
                    let n = -dir.as_vec3();
                    push_world_contact(out, body, slot, r_arm, n, r_point - d, v, face_id, &inv_iw);
                }
            }
        }
    }
}

#[expect(clippy::too_many_arguments, reason = "internal contact assembly")]
fn push_world_contact(
    out: &mut Vec<Contact>,
    body: &Body,
    slot: usize,
    r_arm: Vec3,
    n: Vec3,
    depth: f32,
    voxel: IVec3,
    face_id: u8,
    inv_iw: &Mat3,
) {
    let (t1, t2) = tangent_basis(n);
    // Pre-solve velocity at this point (gravity for this substep already
    // integrated, no contact impulse yet): its component into the surface
    // (opposite `n`, which points away from it) is how fast the body was
    // actually closing on the world at the moment this contact was found.
    let point_vel = body.vel + body.omega.cross(r_arm);
    let approach_speed = (-point_vel).dot(n);
    out.push(Contact {
        body: slot,
        body_b: None,
        normal: n,
        depth,
        r_arm,
        r_arm_b: Vec3::ZERO,
        key: (slot as u32, WORLD_SLOT, voxel, face_id),
        t1,
        t2,
        kn: effective_mass(n, body.inv_mass, inv_iw, r_arm, None),
        kt1: effective_mass(t1, body.inv_mass, inv_iw, r_arm, None),
        kt2: effective_mass(t2, body.inv_mass, inv_iw, r_arm, None),
        acc_n: 0.0,
        acc_t1: 0.0,
        acc_t2: 0.0,
        approach_speed,
    })
}

/// Result of pair narrowphase.
pub struct PairResult {
    /// Peak |relative normal velocity| across penetrating points, used by the
    /// caller to decide whether to wake a sleeping target.
    pub max_rel_speed: f32,
    pub contact_count: usize,
}

/// Generate contacts between two bodies. `sampler` samples the `target`'s
/// grid; roles are chosen by the caller (fewer surface points samples).
///
/// If `target_static` is true the target contributes no mass terms and takes
/// no impulses (a sleeping body treated as scenery).
pub fn pair_contacts(
    sampler: &Body,
    sampler_slot: usize,
    target: &Body,
    target_slot: usize,
    target_static: bool,
    out: &mut Vec<Contact>,
) -> PairResult {
    let s_t = target.half_voxel * 2.0;
    let r_point = sampler.half_voxel;
    let inv_iw_a = sampler.inv_inertia_world();
    let inv_iw_b = target.inv_inertia_world();
    let inv_rot_t: Quat = target.rot.inverse();

    let mut result = PairResult {
        max_rel_speed: 0.0,
        contact_count: 0,
    };

    for &p_local in &sampler.surface {
        let r_arm = sampler.rot * p_local;
        let p_w = sampler.pos + r_arm;
        // Into the target's grid frame (origin at grid min corner).
        let in_target = inv_rot_t * (p_w - target.pos) - target.grid_offset;
        let cell = (in_target / s_t).floor().as_ivec3();

        // Quick reject: outside the grid entirely (with one-cell margin).
        if cell.cmplt(IVec3::splat(-1)).any() || cell.cmpgt(target.grid.dims).any() {
            continue;
        }

        let mut found: Option<(Vec3, f32, IVec3, u8)> = None;
        if target.grid.solid(cell) {
            let mut best: Option<(IVec3, f32, u8)> = None;
            for (face_id, dir) in FACE_DIRS {
                if target.grid.solid(cell + dir) {
                    continue;
                }
                let d = face_dist(in_target, cell, dir, s_t);
                if best.is_none_or(|(_, bd, _)| d < bd) {
                    best = Some((dir, d, face_id));
                }
            }
            if let Some((dir, dist, face_id)) = best {
                found = Some((dir.as_vec3(), r_point + dist, cell, face_id));
            }
        } else {
            // Nearest penetrating face within the point radius.
            let mut best: Option<(IVec3, f32, u8)> = None;
            for (face_id, dir) in FACE_DIRS {
                if !target.grid.solid(cell + dir) {
                    continue;
                }
                let d = face_dist(in_target, cell, dir, s_t);
                if d < r_point && best.is_none_or(|(_, bd, _)| r_point - d > bd) {
                    best = Some((-dir, r_point - d, face_id));
                }
            }
            if let Some((n_local, depth, face_id)) = best {
                found = Some((n_local.as_vec3(), depth, cell, face_id));
            }
        }

        let Some((n_local, depth, cell, face_id)) = found else {
            continue;
        };
        // Rotate the face normal into world space; it pushes the sampler out.
        let n = target.rot * n_local;
        let r_arm_b = p_w - target.pos;

        let va = sampler.vel + sampler.omega.cross(r_arm);
        let vb = target.vel + target.omega.cross(r_arm_b);
        let rel = va - vb;
        result.max_rel_speed = result.max_rel_speed.max(rel.dot(n).abs());
        result.contact_count += 1;
        // Same sign convention as `push_world_contact`: positive means
        // `sampler` is closing on `target` along this contact's normal.
        let approach_speed = (-rel).dot(n);

        let b_terms = if target_static {
            None
        } else {
            Some((target.inv_mass, &inv_iw_b, r_arm_b))
        };
        let (t1, t2) = tangent_basis(n);
        out.push(Contact {
            body: sampler_slot,
            body_b: if target_static {
                None
            } else {
                Some(target_slot)
            },
            normal: n,
            depth,
            r_arm,
            r_arm_b,
            key: (sampler_slot as u32, target_slot as u32, cell, face_id),
            t1,
            t2,
            kn: effective_mass(n, sampler.inv_mass, &inv_iw_a, r_arm, b_terms),
            kt1: effective_mass(t1, sampler.inv_mass, &inv_iw_a, r_arm, b_terms),
            kt2: effective_mass(t2, sampler.inv_mass, &inv_iw_a, r_arm, b_terms),
            acc_n: 0.0,
            acc_t1: 0.0,
            acc_t2: 0.0,
            approach_speed,
        });
    }
    result
}
