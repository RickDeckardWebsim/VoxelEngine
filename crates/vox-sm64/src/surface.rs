//! Voxel terrain → SM64 collision surface bridge.
//!
//! libsm64 uses triangle-surface collision (SM64's original system).
//! The voxel engine uses a grid. This module converts nearby voxel
//! terrain into `SM64Surface` triangles that Mario can collide with.
//!
//! ## Strategy
//!
//! `sm64_static_surfaces_load` replaces the *entire* static surface set
//! on every call (it frees + reallocates internally), so we can't call
//! it every frame. Instead, we rebuild the surface list only when Mario
//! has moved more than [`RESURFACE_THRESHOLD_M`] from the last
//! generation point — typically every 1-2 seconds, not every tick.
//!
//! ### Greedy face merging
//!
//! Instead of emitting 2 triangles per exposed voxel face (which at
//! 0.1 m voxels produces hundreds of thousands of triangles), we merge
//! coplanar adjacent faces into larger rectangles — the same principle
//! as `vox-mesh`'s greedy mesher, but outputting collision triangles.
//! A flat 32×32 chunk top goes from 2048 triangles to 2.
//!
//! ### Chunk-based scanning
//!
//! We iterate `World::chunks()` and filter by chunk distance from
//! Mario first, then scan voxels within nearby chunks. Uniform all-air
//! chunks (the bulk of any world) are skipped entirely.

use crate::ffi::SM64Surface;
use crate::meters_to_sm64;
use glam::{IVec3, UVec3, Vec3};
use vox_core::consts::CHUNK_SIZE;
use vox_core::chunk_origin;
use vox_world::{AIR, World};

/// SM64 surface type: default solid surface.
const SURFACE_DEFAULT: i16 = 0x0000;
/// Terrain type: grass (affects footstep sounds, not collision).
const TERRAIN_GRASS: u16 = 0x0000;

/// How far around Mario (in meters) to generate collision surfaces.
pub const SURFACE_RADIUS_M: f32 = 15.0;

/// Don't regenerate surfaces if Mario moved less than this many meters
/// since the last generation. Higher = less frequent rebuilds = less lag.
const RESURFACE_THRESHOLD_M: f32 = 4.0;

/// Manages streaming collision surfaces around Mario's position.
pub struct SurfaceProvider {
    last_center: Vec3,
    surfaces: Vec<SM64Surface>,
}

impl SurfaceProvider {
    pub fn new() -> Self {
        Self {
            last_center: Vec3::splat(f32::MAX),
            surfaces: Vec::new(),
        }
    }

    pub fn update(&mut self, mario_pos_m: Vec3, world: &World) -> bool {
        if (mario_pos_m - self.last_center).length() < RESURFACE_THRESHOLD_M
            && !self.surfaces.is_empty()
        {
            return false;
        }
        self.last_center = mario_pos_m;
        self.surfaces = voxel_surfaces_near(world, mario_pos_m, SURFACE_RADIUS_M);
        tracing::debug!(
            surfaces = self.surfaces.len(),
            pos = ?mario_pos_m,
            "regenerated SM64 collision surfaces"
        );
        true
    }

    pub fn surfaces(&self) -> &[SM64Surface] {
        &self.surfaces
    }
}

impl Default for SurfaceProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate SM64 collision surfaces for all exposed voxel faces within
/// `radius_m` meters of `center_m`, using greedy face merging.
///
/// Vertices are in SM64 integer units (meters × [`crate::SM64_UNITS_PER_METER`]).
pub fn voxel_surfaces_near(world: &World, center_m: Vec3, radius_m: f32) -> Vec<SM64Surface> {
    let voxel_size = world.cfg.voxel_size_m;
    let chunk_size_m = CHUNK_SIZE as f32 * voxel_size;


    let mut surfaces = Vec::new();

    for (chunk_key, chunk) in world.chunks() {
        // Skip chunks outside the radius: check if the chunk's AABB
        // overlaps the sphere around center_m. Use a simple expanded
        // bounding-box test (chunk_min - radius to chunk_max + radius).
        let chunk_origin_m = chunk_origin(chunk_key).as_vec3() * voxel_size;
        let chunk_max_m = chunk_origin_m + chunk_size_m;
        let overlaps = chunk_origin_m.x <= center_m.x + radius_m
            && chunk_max_m.x >= center_m.x - radius_m
            && chunk_origin_m.y <= center_m.y + radius_m
            && chunk_max_m.y >= center_m.y - radius_m
            && chunk_origin_m.z <= center_m.z + radius_m
            && chunk_max_m.z >= center_m.z - radius_m;
        if !overlaps {
            continue;
        }
        if chunk.solid_count() == 0 {
            continue;
        }
        greedy_faces_for_chunk(world, chunk_key, chunk, voxel_size, &mut surfaces);
    }

    surfaces
}

/// The 6 face directions: (normal axis, normal sign).
/// Axis: 0=X, 1=Y, 2=Z. Sign: +1 or -1.
// Floor + ceiling only for now. Wall faces dominate the surface count
// at 0.1m voxels (~80% of exposed faces). Mario can walk on floors
// without wall collision — we'll add walls back once the surface count
// is manageable with better merging or a spatial grid.
const FACES: [(u32, i32); 2] = [
    (1, 1),  // +Y (top/floor)
    (1, -1), // -Y (bottom/ceiling)
];

/// Generate greedily-merged collision surfaces for all 6 face directions
/// within a single chunk.
fn greedy_faces_for_chunk(
    world: &World,
    chunk_key: IVec3,
    chunk: &vox_world::Chunk,
    voxel_size: f32,
    surfaces: &mut Vec<SM64Surface>,
) {
    let cs = CHUNK_SIZE as i32;

    for &(axis, sign) in &FACES {

        // For each slice along the normal axis
        for slice in 0..cs {
            // Build a 2D grid: is the face at this cell exposed?
            // grid[u][v] = true if the voxel at (slice along axis, u, v)
            // is solid AND its neighbor in the normal direction is air.
            let mut grid = vec![[false; 32]; 32];

            for u in 0..cs {
                for v in 0..cs {
                    // Convert (axis, slice, u, v) to chunk-local coords
                    let (lx, ly, lz) = local_coords(axis, slice, u, v);
                    let local = UVec3::new(lx as u32, ly as u32, lz as u32);
                    if chunk.get(local) == AIR {
                        continue;
                    }

                    // Check if neighbor in the normal direction is air
                    let (nx, ny, nz) = neighbor_offset(axis, sign);
                    let is_exposed = is_neighbor_air(world, chunk_key, chunk, lx, ly, lz, nx, ny, nz);
                    grid[u as usize][v as usize] = is_exposed;
                }
            }

            // Greedy merge the 2D grid into rectangles
            merge_grid(grid, axis, sign, slice, chunk_key, voxel_size, surfaces);
        }
    }
}

/// Convert (face axis, slice index, grid u, grid v) to chunk-local (x, y, z).
fn local_coords(axis: u32, slice: i32, u: i32, v: i32) -> (i32, i32, i32) {
    match axis {
        0 => (slice, u, v), // X faces: slice=X, grid=Y×Z
        1 => (u, slice, v), // Y faces: slice=Y, grid=X×Z
        _ => (u, v, slice), // Z faces: slice=Z, grid=X×Y
    }
}

/// Get the neighbor offset for a face direction.
fn neighbor_offset(axis: u32, sign: i32) -> (i32, i32, i32) {
    match axis {
        0 => (sign, 0, 0),
        1 => (0, sign, 0),
        _ => (0, 0, sign),
    }
}

/// Check if a neighbor voxel is air, using chunk-local fast path when possible.
fn is_neighbor_air(
    world: &World,
    chunk_key: IVec3,
    chunk: &vox_world::Chunk,
    lx: i32,
    ly: i32,
    lz: i32,
    dx: i32,
    dy: i32,
    dz: i32,
) -> bool {
    let nlx = lx + dx;
    let nly = ly + dy;
    let nlz = lz + dz;
    let cs = CHUNK_SIZE as i32;

    if nlx >= 0 && nlx < cs && nly >= 0 && nly < cs && nlz >= 0 && nlz < cs {
        chunk.get(UVec3::new(nlx as u32, nly as u32, nlz as u32)) == AIR
    } else {
        // Cross-chunk: compute world voxel coordinate
        let vx = chunk_key.x * cs + lx + dx;
        let vy = chunk_key.y * cs + ly + dy;
        let vz = chunk_key.z * cs + lz + dz;
        world.get_voxel(IVec3::new(vx, vy, vz)) == AIR
    }
}

/// Greedily merge a 2D boolean grid into rectangles, emitting 2 triangles
/// per rectangle. Each cell in the grid is one voxel face; merged
/// rectangles span multiple contiguous cells.
fn merge_grid(
    grid: Vec<[bool; 32]>,
    axis: u32,
    sign: i32,
    slice: i32,
    chunk_key: IVec3,
    voxel_size: f32,
    surfaces: &mut Vec<SM64Surface>,
) {
    let cs = CHUNK_SIZE as usize;
    // Visited marks cells already consumed by a merged rectangle
    let mut visited = vec![[false; 32]; 32];

    for u in 0..cs {
        for v in 0..cs {
            if visited[u][v] || !grid[u][v] {
                continue;
            }

            // Find the widest span in the v direction at this u
            let mut max_v = v;
            while max_v + 1 < cs && grid[u][max_v + 1] && !visited[u][max_v + 1] {
                max_v += 1;
            }

            // Extend in the u direction as far as all rows match the span
            let mut max_u = u;
            'outer: loop {
                if max_u + 1 >= cs {
                    break;
                }
                for vv in v..=max_v {
                    if !grid[max_u + 1][vv] || visited[max_u + 1][vv] {
                        break 'outer;
                    }
                }
                max_u += 1;
            }

            // Mark all cells in the rectangle as visited
            for uu in u..=max_u {
                for vv in v..=max_v {
                    visited[uu][vv] = true;
                }
            }

            // Emit 2 triangles for this merged rectangle
            emit_merged_face(
                axis, sign, slice, u, v, max_u, max_v, chunk_key, voxel_size, surfaces,
            );
        }
    }
}

/// Emit 2 collision triangles for a merged face rectangle.
///
/// The rectangle spans grid cells [u0..u1, v0..v1] at the given slice.
/// We compute the 4 corner positions in world space, then emit 2
/// triangles with correct winding (CCW when viewed from outside).
fn emit_merged_face(
    axis: u32,
    sign: i32,
    slice: i32,
    u0: usize,
    v0: usize,
    u1: usize,
    v1: usize,
    chunk_key: IVec3,
    voxel_size: f32,
    surfaces: &mut Vec<SM64Surface>,
) {

    // The face is at the boundary between slice and slice+sign.
    // For +sign faces: the face is at the top of the voxel (slice + 1 in world units)
    // For -sign faces: the face is at the bottom of the voxel (slice in world units)
    let face_slice = if sign > 0 { slice + 1 } else { slice };

    // World voxel coordinates of the rectangle corners
    // (axis determines which coord is the slice, a1/a2 are the grid axes)
    let origin = chunk_origin(chunk_key);
    let (s_x, s_y, s_z) = (origin.x as f32 * voxel_size, origin.y as f32 * voxel_size, origin.z as f32 * voxel_size);

    // Compute corner positions in meters, then convert to SM64 units
    // For each axis, the position is: chunk_origin + voxel_coord * voxel_size
    let (p0, p1, p2, p3) = compute_rect_corners(
        axis, s_x, s_y, s_z, face_slice, u0, v0, u1, v1, voxel_size,
    );

    // Convert to SM64 integer units
    let c = |p: Vec3| -> (i32, i32, i32) {
        (meters_to_sm64(p.x), meters_to_sm64(p.y), meters_to_sm64(p.z))
    };
    let (ax, ay, az) = c(p0);
    let (bx, by, bz) = c(p1);
    let (cx, cy, cz) = c(p2);
    let (dx, dy, dz) = c(p3);

    // Emit 2 triangles with winding CCW when viewed from outside
    // (the direction the normal points, i.e. +sign on axis)
    match (axis, sign) {
        // +Y face (top): CCW when viewed from above (normal points up).
        // Corners are p0=(x0,z0) p1=(x1,z0) p2=(x1,z1) p3=(x0,z1) CW.
        // For CCW: p0→p3→p2 and p0→p2→p1.
        (1, 1) => {
            surfaces.push(make_surface(ax, ay, az, dx, dy, dz, cx, cy, cz));
            surfaces.push(make_surface(ax, ay, az, cx, cy, cz, bx, by, bz));
        }
        // -Y face (bottom): CW from above (normal points down)
        (1, -1) => {
            surfaces.push(make_surface(ax, ay, az, bx, by, bz, cx, cy, cz));
            surfaces.push(make_surface(ax, ay, az, cx, cy, cz, dx, dy, dz));
        }
        // +X face: reversed winding so normal points +X
        (0, 1) => {
            surfaces.push(make_surface(ax, ay, az, cx, cy, cz, bx, by, bz));
            surfaces.push(make_surface(ax, ay, az, dx, dy, dz, cx, cy, cz));
        }
        // -X face: normal winding so normal points -X
        (0, -1) => {
            surfaces.push(make_surface(ax, ay, az, bx, by, bz, cx, cy, cz));
            surfaces.push(make_surface(ax, ay, az, cx, cy, cz, dx, dy, dz));
        }
        // +Z face
        (2, 1) => {
            surfaces.push(make_surface(ax, ay, az, bx, by, bz, cx, cy, cz));
            surfaces.push(make_surface(ax, ay, az, cx, cy, cz, dx, dy, dz));
        }
        // -Z face
        (2, -1) => {
            surfaces.push(make_surface(ax, ay, az, cx, cy, cz, bx, by, bz));
            surfaces.push(make_surface(ax, ay, az, dx, dy, dz, cx, cy, cz));
        }
        _ => unreachable!(),
    }
}

/// Compute the 4 corner positions (in meters) of a merged face rectangle.
/// Returns (p0, p1, p2, p3) in clockwise order when viewed from outside.
fn compute_rect_corners(
    axis: u32,
    s_x: f32,
    s_y: f32,
    s_z: f32,
    face_slice: i32,
    u0: usize,
    v0: usize,
    u1: usize,
    v1: usize,
    voxel_size: f32,
) -> (Vec3, Vec3, Vec3, Vec3) {
    // Position along the face axis (fixed)
    let s_pos = face_slice as f32 * voxel_size;
    // Grid axis positions: u and v spans
    let u0_pos = u0 as f32 * voxel_size;
    let u1_pos = (u1 + 1) as f32 * voxel_size;
    let v0_pos = v0 as f32 * voxel_size;
    let v1_pos = (v1 + 1) as f32 * voxel_size;

    match axis {
        0 => {
            // X faces: slice=X, grid=Y×Z
            let x = s_x + s_pos;
            let y0 = s_y + u0_pos;
            let y1 = s_y + u1_pos;
            let z0 = s_z + v0_pos;
            let z1 = s_z + v1_pos;
            (Vec3::new(x, y0, z0), Vec3::new(x, y0, z1), Vec3::new(x, y1, z1), Vec3::new(x, y1, z0))
        }
        1 => {
            // Y faces: slice=Y, grid=X×Z
            let y = s_y + s_pos;
            let x0 = s_x + u0_pos;
            let x1 = s_x + u1_pos;
            let z0 = s_z + v0_pos;
            let z1 = s_z + v1_pos;
            (Vec3::new(x0, y, z0), Vec3::new(x1, y, z0), Vec3::new(x1, y, z1), Vec3::new(x0, y, z1))
        }
        _ => {
            // Z faces: slice=Z, grid=X×Y
            let z = s_z + s_pos;
            let x0 = s_x + u0_pos;
            let x1 = s_x + u1_pos;
            let y0 = s_y + v0_pos;
            let y1 = s_y + v1_pos;
            (Vec3::new(x0, y0, z), Vec3::new(x1, y0, z), Vec3::new(x1, y1, z), Vec3::new(x0, y1, z))
        }
    }
}

/// Build an `SM64Surface` from 3 integer vertices (SM64 units).
fn make_surface(
    ax: i32, ay: i32, az: i32,
    bx: i32, by: i32, bz: i32,
    cx: i32, cy: i32, cz: i32,
) -> SM64Surface {
    SM64Surface {
        type_: SURFACE_DEFAULT,
        force: 0,
        terrain: TERRAIN_GRASS,
        vertices: [[ax, ay, az], [bx, by, bz], [cx, cy, cz]],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::WorldConfig;
    use vox_world::Voxel;

    fn test_world() -> World {
        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [32.0, 32.0, 32.0],
            seed: 42,
        });
        world.set_voxel(IVec3::new(5, 5, 5), Voxel(1));
        world
    }

    #[test]
    fn single_voxel_produces_2_faces_4_triangles() {
        let world = test_world();
        let surfaces = voxel_surfaces_near(&world, Vec3::new(5.5, 5.5, 5.5), 10.0);
        // Floor-only: top + bottom = 2 faces × 2 tris = 4
        assert_eq!(surfaces.len(), 4);
    }

    #[test]
    fn flat_3x3_merges_to_4_triangles() {
        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [32.0, 32.0, 32.0],
            seed: 42,
        });
        for x in 4..=6 {
            for z in 4..=6 {
                world.set_voxel(IVec3::new(x, 5, z), Voxel(1));
            }
        }
        let surfaces = voxel_surfaces_near(&world, Vec3::new(5.0, 5.5, 5.0), 10.0);
        // Floor-only: top 3×3 merged = 2 tris, bottom 3×3 merged = 2 tris = 4
        assert_eq!(surfaces.len(), 4);
    }

    #[test]
    fn buried_voxel_produces_only_outer_floor() {
        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [32.0, 32.0, 32.0],
            seed: 42,
        });
        for x in 4..=6 {
            for y in 4..=6 {
                for z in 4..=6 {
                    world.set_voxel(IVec3::new(x, y, z), Voxel(1));
                }
            }
        }
        let surfaces = voxel_surfaces_near(&world, Vec3::new(5.0, 5.0, 5.0), 10.0);
        // Floor-only: top 3×3 = 2 tris, bottom 3×3 = 2 tris = 4 total
        assert_eq!(surfaces.len(), 4);
    }

    #[test]
    fn empty_world_produces_no_surfaces() {
        let world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [32.0, 32.0, 32.0],
            seed: 1,
        });
        let surfaces = voxel_surfaces_near(&world, Vec3::new(16.0, 16.0, 16.0), 10.0);
        assert_eq!(surfaces.len(), 0);
    }

    #[test]
    fn surface_provider_skips_until_threshold() {
        let world = test_world();
        let mut provider = SurfaceProvider::new();
        let pos = Vec3::new(5.5, 5.5, 5.5);

        assert!(provider.update(pos, &world));
        assert!(!provider.surfaces().is_empty());
        assert!(!provider.update(pos, &world));
        assert!(provider.update(pos + Vec3::new(5.0, 0.0, 0.0), &world));
    }

    #[test]
    fn coordinates_scale_to_sm64_units() {
        assert_eq!(meters_to_sm64(0.0), 0);
        assert_eq!(meters_to_sm64(1.0), 30);
        assert_eq!(meters_to_sm64(1.5), 45);
    }
}
