//! Greedy meshing with baked vertex ambient occlusion.
//!
//! For each of the six face directions, exposed faces are collected into a
//! per-slice 2-D mask and merged into maximal rectangles. Cells merge only
//! when material AND all four corner AO values match, so merged quads never
//! smear AO across a seam.

use glam::IVec3;
use vox_world::Voxel;

use crate::slab::VoxelSlab;

/// One mesh vertex, 8 bytes. Positions are voxel-corner coordinates relative
/// to the slab's inner minimum (`0..=dims`), scaled by `voxel_size` and
/// transformed in the shader.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelVertex {
    /// Corner position in voxel units, relative to the region minimum.
    pub pos: [u8; 3],
    /// Ambient-occlusion level, `0` (fully occluded) ..= `3` (open).
    pub ao: u8,
    /// Face normal id: 0..6 = +X, -X, +Y, -Y, +Z, -Z.
    pub normal: u8,
    /// Deterministic per-vertex jitter (0..=255), baked in once at mesh-build
    /// time from this vertex's position -- see `mesh_slab`'s `jitter_seed`
    /// parameter for why this has to be baked rather than computed from
    /// world position in the shader every frame.
    pub jitter: u8,
    /// Material id of the face.
    pub material: u16,
}

/// Mesh geometry for one region: quads as an indexed triangle list.
#[derive(Default)]
pub struct MeshData {
    pub vertices: Vec<VoxelVertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    /// Number of quads (each quad is 4 vertices / 6 indices).
    pub fn quads(&self) -> usize {
        self.vertices.len() / 4
    }

    /// True when the mesh has no geometry.
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
}

/// The six face directions: (normal id, axis, sign).
const FACE_DIRS: [(u8, usize, i32); 6] = [
    (0, 0, 1),
    (1, 0, -1),
    (2, 1, 1),
    (3, 1, -1),
    (4, 2, 1),
    (5, 2, -1),
];

/// Ambient occlusion for a face corner given its three outer-plane neighbors.
/// Classic rule: two occluded sides fully darken the corner regardless of the
/// diagonal.
#[inline]
fn ao(side1: bool, side2: bool, corner: bool) -> u8 {
    if side1 && side2 {
        0
    } else {
        3 - (u8::from(side1) + u8::from(side2) + u8::from(corner))
    }
}

/// A meshable face cell: merged only with cells equal in both fields.
#[derive(Copy, Clone, PartialEq, Eq)]
struct Cell {
    material: Voxel,
    /// Corner AO in (du, dv) order: `[ao00, ao10, ao01, ao11]`.
    ao4: [u8; 4],
}

/// Deterministic per-vertex jitter hash, baked into the mesh once here
/// rather than recomputed from world position in the shader every frame.
/// An earlier version hashed the vertex's *world* position dynamically in
/// WGSL so per-voxel color variation stayed put in world space instead of
/// tiling identically with the mesh's own local coordinates -- fine for a
/// chunk (which never moves), but for a tumbling debris body, world
/// position changes continuously as it rotates/translates, so the "fixed"
/// per-voxel jitter recomputed each frame actually shifted constantly,
/// reading as flicker on every moving body's surface. Baking it in at mesh
/// time instead makes it a fixed property of the geometry, stable
/// regardless of how the object subsequently moves.
///
/// `seed` anchors the pattern to roughly where in a larger space this mesh
/// sits (a chunk's origin, so neighboring chunks don't all tile the
/// identical repeating pattern chunk meshes' own 0..32 local coordinates
/// would otherwise produce); bodies pass `IVec3::ZERO` since their own
/// local grid is already small and irregular enough per shape.
#[inline]
fn jitter_hash(seed: IVec3, local: [u8; 3]) -> u8 {
    let p = seed + IVec3::new(local[0] as i32, local[1] as i32, local[2] as i32);
    let mut x = (p.x as u32)
        .wrapping_mul(0x8529_7a4d)
        ^ (p.y as u32).wrapping_mul(0x68e3_1da4)
        ^ (p.z as u32).wrapping_mul(0x1b56_c4e9);
    x ^= x >> 15;
    x = x.wrapping_mul(0x2c1b_3c6d);
    x ^= x >> 12;
    x = x.wrapping_mul(0x297a_2d39);
    x ^= x >> 15;
    (x & 0xFF) as u8
}

/// Greedy-mesh a slab into quads. `jitter_seed` anchors the baked per-vertex
/// jitter pattern (see `jitter_hash`) -- pass a chunk's world origin for
/// chunks, `IVec3::ZERO` for a body's own local mesh.
pub fn mesh_slab(slab: &VoxelSlab, jitter_seed: IVec3) -> MeshData {
    let mut mesh = MeshData::default();
    let dims = slab.inner_dims;

    for (normal_id, axis, sign) in FACE_DIRS {
        // Tangent axes: u, v are the other two axes in ascending order.
        let (u_axis, v_axis) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let (du, dv) = (dims[u_axis], dims[v_axis]);
        let mut normal = IVec3::ZERO;
        normal[axis] = sign;
        let mut u_dir = IVec3::ZERO;
        u_dir[u_axis] = 1;
        let mut v_dir = IVec3::ZERO;
        v_dir[v_axis] = 1;

        let mut mask: Vec<Option<Cell>> = vec![None; (du * dv) as usize];

        for slice in 0..dims[axis] {
            // Build the mask of exposed faces in this slice.
            for v in 0..dv {
                for u in 0..du {
                    let mut p = IVec3::ZERO;
                    p[axis] = slice;
                    p[u_axis] = u;
                    p[v_axis] = v;
                    let cell = if slab.solid(p) && !slab.solid(p + normal) {
                        let outer = p + normal;
                        let mut ao4 = [0u8; 4];
                        for (i, (cu, cv)) in
                            [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate()
                        {
                            let u_off = if cu == 0 { -u_dir } else { u_dir };
                            let v_off = if cv == 0 { -v_dir } else { v_dir };
                            ao4[i] = ao(
                                slab.solid(outer + u_off),
                                slab.solid(outer + v_off),
                                slab.solid(outer + u_off + v_off),
                            );
                        }
                        Some(Cell {
                            material: slab.get(p),
                            ao4,
                        })
                    } else {
                        None
                    };
                    mask[(u + v * du) as usize] = cell;
                }
            }

            // Greedy rectangle merge over the mask.
            for v0 in 0..dv {
                let mut u0 = 0;
                while u0 < du {
                    let Some(cell) = mask[(u0 + v0 * du) as usize] else {
                        u0 += 1;
                        continue;
                    };
                    // Grow width while cells match.
                    let mut w = 1;
                    while u0 + w < du && mask[(u0 + w + v0 * du) as usize] == Some(cell) {
                        w += 1;
                    }
                    // Grow height while the whole row of width `w` matches.
                    let mut h = 1;
                    'grow: while v0 + h < dv {
                        for uu in u0..u0 + w {
                            if mask[(uu + (v0 + h) * du) as usize] != Some(cell) {
                                break 'grow;
                            }
                        }
                        h += 1;
                    }
                    emit_quad(
                        &mut mesh, cell, normal_id, axis, sign, slice, u_axis, v_axis, u0, v0, w, h,
                        jitter_seed,
                    );
                    for vv in v0..v0 + h {
                        for uu in u0..u0 + w {
                            mask[(uu + vv * du) as usize] = None;
                        }
                    }
                    u0 += w;
                }
            }
        }
    }
    mesh
}

/// Append one merged quad as 4 vertices and 6 indices.
#[expect(clippy::too_many_arguments, reason = "internal plumbing of mesh_slab")]
fn emit_quad(
    mesh: &mut MeshData,
    cell: Cell,
    normal_id: u8,
    axis: usize,
    sign: i32,
    slice: i32,
    u_axis: usize,
    v_axis: usize,
    u0: i32,
    v0: i32,
    w: i32,
    h: i32,
    jitter_seed: IVec3,
) {
    // Corner positions on the face plane, in (du, dv) order 00, 10, 01, 11.
    let plane = if sign > 0 { slice + 1 } else { slice };
    let corner = |cu: i32, cv: i32| -> [u8; 3] {
        let mut p = IVec3::ZERO;
        p[axis] = plane;
        p[u_axis] = u0 + cu * w;
        p[v_axis] = v0 + cv * h;
        debug_assert!(p.cmpge(IVec3::ZERO).all() && p.cmple(IVec3::splat(255)).all());
        [p.x as u8, p.y as u8, p.z as u8]
    };
    let positions = [corner(0, 0), corner(1, 0), corner(0, 1), corner(1, 1)];

    let base = mesh.vertices.len() as u32;
    for (i, pos) in positions.into_iter().enumerate() {
        mesh.vertices.push(VoxelVertex {
            pos,
            ao: cell.ao4[i],
            normal: normal_id,
            jitter: jitter_hash(jitter_seed, pos),
            material: cell.material.0,
        });
    }

    // Vertex order is 00, 10, 01, 11. Triangulate along the diagonal that
    // matches the AO gradient (standard anisotropy fix), then orient the
    // winding so the face normal points outward: for +axis faces
    // cross(u_dir, v_dir) already equals +normal when (axis, u, v) is an even
    // permutation of XYZ — which (x,yz), (y,xz) flipped, (z,xy) are not all,
    // so derive orientation from the axis directly.
    let [a00, a10, a01, a11] = cell.ao4;
    let flipped = u32::from(a00) + u32::from(a11) < u32::from(a10) + u32::from(a01);
    // Winding for a face whose cross(u, v) points toward +axis:
    //   axis 0 (u=y, v=z): cross(y, z) = +x
    //   axis 1 (u=x, v=z): cross(x, z) = -y  (odd permutation)
    //   axis 2 (u=x, v=y): cross(x, y) = +z
    let uv_cross_matches_positive = axis != 1;
    let ccw_for_positive = sign > 0;
    let forward = uv_cross_matches_positive == ccw_for_positive;

    let quad: [u32; 6] = match (flipped, forward) {
        // Diagonal 00-11.
        (false, true) => [0, 1, 3, 0, 3, 2],
        (false, false) => [0, 3, 1, 0, 2, 3],
        // Diagonal 10-01.
        (true, true) => [1, 3, 2, 1, 2, 0],
        (true, false) => [1, 2, 3, 1, 0, 2],
    };
    mesh.indices.extend(quad.into_iter().map(|i| base + i));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use vox_world::AIR;

    const STONE: Voxel = Voxel(1);
    const DIRT: Voxel = Voxel(2);

    /// Deterministic splitmix64 (dependency-free test randomness).
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    /// Build a slab from a set of solid voxels inside `dims`.
    fn slab_of(dims: IVec3, solids: &[(IVec3, Voxel)]) -> VoxelSlab {
        let mut data = vec![AIR; (dims.x * dims.y * dims.z) as usize];
        for &(p, v) in solids {
            let idx = (p.x + p.z * dims.x + p.y * dims.x * dims.z) as usize;
            data[idx] = v;
        }
        VoxelSlab::from_grid(dims, &data)
    }

    #[test]
    fn empty_slab_zero_quads() {
        let slab = slab_of(IVec3::splat(4), &[]);
        let mesh = mesh_slab(&slab, IVec3::ZERO);
        assert_eq!(mesh.quads(), 0);
        assert!(mesh.is_empty());
    }

    #[test]
    fn single_voxel_six_quads() {
        let slab = slab_of(IVec3::splat(3), &[(IVec3::splat(1), STONE)]);
        let mesh = mesh_slab(&slab, IVec3::ZERO);
        assert_eq!(mesh.quads(), 6);
        assert_eq!(mesh.indices.len(), 36);
    }

    /// Regression test for a rendering bug: jitter used to be recomputed in
    /// the shader from each vertex's *world* position every frame. That's
    /// stable for a chunk (which never moves) but not for a tumbling debris
    /// body, whose world position changes continuously -- the "fixed"
    /// per-voxel jitter recomputed from a moving position actually shifted
    /// every frame, which read as flicker on every rotating fragment. Baking
    /// the jitter into the mesh at build time (this test's whole point)
    /// means it's a pure function of local geometry and the caller's seed:
    /// re-meshing identical geometry with the same seed must produce
    /// byte-for-byte identical jitter, with no hidden dependency on anything
    /// that could vary as an object moves.
    #[test]
    fn jitter_is_deterministic_from_local_geometry_and_seed_alone() {
        let slab = slab_of(
            IVec3::splat(5),
            &[
                (IVec3::new(1, 1, 1), STONE),
                (IVec3::new(2, 1, 1), STONE),
                (IVec3::new(1, 2, 1), STONE),
            ],
        );
        let mesh_a = mesh_slab(&slab, IVec3::new(7, -3, 42));
        let mesh_b = mesh_slab(&slab, IVec3::new(7, -3, 42));
        let jitter_a: Vec<u8> = mesh_a.vertices.iter().map(|v| v.jitter).collect();
        let jitter_b: Vec<u8> = mesh_b.vertices.iter().map(|v| v.jitter).collect();
        assert_eq!(jitter_a, jitter_b, "same geometry + same seed must match exactly");

        // A body (seed always zero) meshed twice must also match, and a
        // *different* seed (a different chunk's origin) must generally
        // produce a different pattern -- confirming the seed actually
        // participates, not just the local position.
        let mesh_c = mesh_slab(&slab, IVec3::ZERO);
        let jitter_c: Vec<u8> = mesh_c.vertices.iter().map(|v| v.jitter).collect();
        assert_ne!(jitter_a, jitter_c, "different seeds should not collide onto the same pattern");
    }

    #[test]
    fn two_same_material_merge_to_six_quads() {
        let slab = slab_of(
            IVec3::new(2, 1, 1),
            &[(IVec3::new(0, 0, 0), STONE), (IVec3::new(1, 0, 0), STONE)],
        );
        let mesh = mesh_slab(&slab, IVec3::ZERO);
        assert_eq!(mesh.quads(), 6, "coplanar same-material faces must merge");
    }

    #[test]
    fn two_materials_do_not_merge() {
        let slab = slab_of(
            IVec3::new(2, 1, 1),
            &[(IVec3::new(0, 0, 0), STONE), (IVec3::new(1, 0, 0), DIRT)],
        );
        let mesh = mesh_slab(&slab, IVec3::ZERO);
        // 2 end caps + 4 long sides split in two each = 2 + 8 = 10.
        assert_eq!(mesh.quads(), 10);
    }

    #[test]
    fn full_uniform_region_meshes_to_six_quads() {
        let dims = IVec3::splat(32);
        let mut solids = Vec::new();
        for y in 0..32 {
            for z in 0..32 {
                for x in 0..32 {
                    solids.push((IVec3::new(x, y, z), STONE));
                }
            }
        }
        let slab = slab_of(dims, &solids);
        let mesh = mesh_slab(&slab, IVec3::ZERO);
        assert_eq!(mesh.quads(), 6, "each full face merges into one quad");
        // Corner coordinates must span the whole region.
        let max = mesh.vertices.iter().map(|v| v.pos[0]).max().unwrap();
        assert_eq!(max, 32);
    }

    /// Every exposed face must be covered by exactly one emitted quad cell.
    #[test]
    fn watertight_on_random_slabs() {
        let mut rng = Rng(0xFACADE);
        for round in 0..20 {
            let dims = IVec3::splat(12);
            let mut solids = Vec::new();
            for y in 0..dims.y {
                for z in 0..dims.z {
                    for x in 0..dims.x {
                        if rng.next_u64() % 10 < 3 {
                            let mat = Voxel((rng.next_u64() % 3 + 1) as u16);
                            solids.push((IVec3::new(x, y, z), mat));
                        }
                    }
                }
            }
            let slab = slab_of(dims, &solids);
            let mesh = mesh_slab(&slab, IVec3::ZERO);

            // Brute-force expected exposed faces.
            let mut expected: HashSet<(IVec3, u8)> = HashSet::new();
            for y in 0..dims.y {
                for z in 0..dims.z {
                    for x in 0..dims.x {
                        let p = IVec3::new(x, y, z);
                        if !slab.solid(p) {
                            continue;
                        }
                        for (normal_id, axis, sign) in FACE_DIRS {
                            let mut n = IVec3::ZERO;
                            n[axis] = sign;
                            if !slab.solid(p + n) {
                                expected.insert((p, normal_id));
                            }
                        }
                    }
                }
            }

            // Rasterize emitted quads back into face cells.
            let mut actual: HashSet<(IVec3, u8)> = HashSet::new();
            for quad in mesh.vertices.chunks_exact(4) {
                let normal_id = quad[0].normal;
                let (_, axis, sign) = FACE_DIRS[normal_id as usize];
                let (u_axis, v_axis) = match axis {
                    0 => (1, 2),
                    1 => (0, 2),
                    _ => (0, 1),
                };
                let corner =
                    |v: &VoxelVertex| IVec3::new(v.pos[0] as i32, v.pos[1] as i32, v.pos[2] as i32);
                let (c00, c11) = (corner(&quad[0]), corner(&quad[3]));
                let plane = c00[axis];
                let cell_slice = if sign > 0 { plane - 1 } else { plane };
                for u in c00[u_axis]..c11[u_axis] {
                    for v in c00[v_axis]..c11[v_axis] {
                        let mut cell = IVec3::ZERO;
                        cell[axis] = cell_slice;
                        cell[u_axis] = u;
                        cell[v_axis] = v;
                        assert!(
                            actual.insert((cell, normal_id)),
                            "round {round}: face covered twice: {cell} dir {normal_id}"
                        );
                    }
                }
            }
            assert_eq!(actual, expected, "round {round}: coverage mismatch");
        }
    }

    /// Triangle winding: geometric normals must point along the face normal.
    #[test]
    fn winding_is_outward_ccw() {
        let mut rng = Rng(0xBEEF);
        let dims = IVec3::splat(8);
        let mut solids = Vec::new();
        for y in 0..dims.y {
            for z in 0..dims.z {
                for x in 0..dims.x {
                    if rng.next_u64() % 10 < 4 {
                        solids.push((IVec3::new(x, y, z), STONE));
                    }
                }
            }
        }
        let slab = slab_of(dims, &solids);
        let mesh = mesh_slab(&slab, IVec3::ZERO);
        assert!(!mesh.is_empty());

        for tri in mesh.indices.chunks_exact(3) {
            let p = |i: u32| {
                let v = &mesh.vertices[i as usize];
                glam::Vec3::new(v.pos[0] as f32, v.pos[1] as f32, v.pos[2] as f32)
            };
            let (a, b, c) = (p(tri[0]), p(tri[1]), p(tri[2]));
            let geometric = (b - a).cross(c - a);
            let (_, axis, sign) = FACE_DIRS[mesh.vertices[tri[0] as usize].normal as usize];
            let mut n = glam::Vec3::ZERO;
            n[axis] = sign as f32;
            assert!(
                geometric.dot(n) > 0.0,
                "triangle winding not CCW toward face normal: {a} {b} {c} vs {n}"
            );
        }
    }

    #[test]
    fn ao_darkens_corners_next_to_walls() {
        // A 2x1x2 floor with a wall voxel standing on one corner.
        let slab = slab_of(
            IVec3::new(2, 2, 2),
            &[
                (IVec3::new(0, 0, 0), STONE),
                (IVec3::new(1, 0, 0), STONE),
                (IVec3::new(0, 0, 1), STONE),
                (IVec3::new(1, 0, 1), STONE),
                (IVec3::new(0, 1, 0), STONE), // wall on top of (0,0,0)
            ],
        );
        let mesh = mesh_slab(&slab, IVec3::ZERO);

        // Top faces (+Y, normal id 2) of the floor at y=1 (excluding the wall
        // voxel's own top at y=2).
        let top_floor: Vec<&VoxelVertex> = mesh
            .vertices
            .iter()
            .filter(|v| v.normal == 2 && v.pos[1] == 1)
            .collect();
        assert!(!top_floor.is_empty(), "floor top faces exist");
        let occluded = top_floor.iter().filter(|v| v.ao < 3).count();
        let open = top_floor.iter().filter(|v| v.ao == 3).count();
        assert!(occluded > 0, "vertices near the wall must darken");
        assert!(open > 0, "vertices away from the wall must stay open");

        // And the differing AO must have split the floor into >1 top quad.
        let top_quads = mesh
            .vertices
            .chunks_exact(4)
            .filter(|q| q[0].normal == 2 && q[0].pos[1] == 1)
            .count();
        assert!(
            top_quads > 1,
            "AO seam must prevent merging into a single quad"
        );
    }

    #[test]
    fn vertex_is_pod_and_8_bytes() {
        assert_eq!(std::mem::size_of::<VoxelVertex>(), 8);
        let v = VoxelVertex {
            pos: [1, 2, 3],
            ao: 3,
            normal: 0,
            jitter: 0,
            material: 7,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&v);
        assert_eq!(bytes.len(), 8);
    }
}
