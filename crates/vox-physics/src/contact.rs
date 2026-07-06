//! Contact generation: body surface points versus the world voxel grid.
//!
//! Every surface sample point behaves as a small sphere of radius
//! `half_voxel`. Points inside solid voxels push out through their nearest
//! empty face; points within the radius of a neighboring solid face contact
//! before penetrating. Normals are always voxel-face-aligned, which keeps the
//! math simple and debris "feeling" voxel-accurate.

use glam::{IVec3, Mat3, Vec3};
use vox_core::voxel_at;
use vox_world::World;

use crate::body::Body;

/// Stable identity of a contact across steps (warm starting).
pub type ContactKey = (u32, IVec3, u8);

/// One contact between a body and the static world.
pub struct Contact {
    /// Arena slot of the body.
    pub body: usize,
    pub normal: Vec3,
    pub depth: f32,
    /// Contact point relative to the body's COM (world orientation).
    pub r_arm: Vec3,
    pub key: ContactKey,
    // Tangent basis (axis-aligned; normals are face-aligned).
    pub t1: Vec3,
    pub t2: Vec3,
    // Effective masses, precomputed at generation.
    pub kn: f32,
    pub kt1: f32,
    pub kt2: f32,
    // Accumulated impulses (warm-started).
    pub acc_n: f32,
    pub acc_t1: f32,
    pub acc_t2: f32,
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

/// Axis-aligned tangent basis for a face normal.
#[inline]
fn tangent_basis(n: Vec3) -> (Vec3, Vec3) {
    if n.x.abs() > 0.5 {
        (Vec3::Y, Vec3::Z)
    } else if n.y.abs() > 0.5 {
        (Vec3::X, Vec3::Z)
    } else {
        (Vec3::X, Vec3::Y)
    }
}

/// Signed distance from point `p` to the face plane of voxel `v` in
/// direction `dir` (positive = inside the voxel, toward the far side).
#[inline]
fn face_dist(p: Vec3, v: IVec3, dir: IVec3, s: f32) -> f32 {
    // For +axis: plane at (v[a]+1)*s, distance = plane - p[a].
    // For -axis: plane at v[a]*s, distance = p[a] - plane.
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

/// Effective-mass helper: `k = 1/m + n · ((I⁻¹ (r × n)) × r)`.
#[inline]
fn effective_mass(inv_mass: f32, inv_inertia_w: &Mat3, r: Vec3, n: Vec3) -> f32 {
    let rn = r.cross(n);
    inv_mass + n.dot((*inv_inertia_w * rn).cross(r))
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
            // Inside a solid voxel: push out through the nearest empty face.
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
            push_contact(
                out,
                body,
                slot,
                r_arm,
                n,
                r_point + dist,
                v,
                face_id,
                inv_iw,
            );
        } else {
            // Near-face pre-contact: any adjacent solid within the radius.
            for (face_id, dir) in FACE_DIRS {
                if !world.solid(v + dir) {
                    continue;
                }
                let d = face_dist(p_w, v, dir, s);
                if d < r_point {
                    // Push away from the solid neighbor.
                    let n = -dir.as_vec3();
                    push_contact(out, body, slot, r_arm, n, r_point - d, v, face_id, inv_iw);
                }
            }
        }
    }
}

#[expect(clippy::too_many_arguments, reason = "internal contact assembly")]
fn push_contact(
    out: &mut Vec<Contact>,
    body: &Body,
    slot: usize,
    r_arm: Vec3,
    n: Vec3,
    depth: f32,
    voxel: IVec3,
    face_id: u8,
    inv_iw: Mat3,
) {
    let (t1, t2) = tangent_basis(n);
    out.push(Contact {
        body: slot,
        normal: n,
        depth,
        r_arm,
        key: (slot as u32, voxel, face_id),
        t1,
        t2,
        kn: effective_mass(body.inv_mass, &inv_iw, r_arm, n),
        kt1: effective_mass(body.inv_mass, &inv_iw, r_arm, t1),
        kt2: effective_mass(body.inv_mass, &inv_iw, r_arm, t2),
        acc_n: 0.0,
        acc_t1: 0.0,
        acc_t2: 0.0,
    });
}
