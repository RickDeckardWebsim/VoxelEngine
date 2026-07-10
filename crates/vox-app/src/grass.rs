//! Grass blade generation: scans nearby chunks for grass-top voxels and
//! generates 3D blade geometry (thin standing quads) with wind sway.
//!
//! Called each frame from the app. Generates blades for grass voxels
//! within a radius of the camera, skipping chunks without grass.

use glam::{IVec3, Vec3};
use vox_core::chunk_origin;
use vox_core::consts::CHUNK_SIZE;
use vox_render::GrassVertex;
use vox_world::{AIR, World};

/// How far around the camera to generate grass blades (meters).
const GRASS_RADIUS_M: f32 = 30.0; // Reduced from 60 for performance

/// Cached grass blade vertices. Regenerated at most every N frames.
/// Wind sway is applied in the vertex shader via game_time, so the blade
/// positions only need refreshing when the camera moves significantly.
pub struct GrassCache {
    vertices: Vec<GrassVertex>,
    last_cam_pos: Vec3,
    frame_counter: u32,
}

impl GrassCache {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            last_cam_pos: Vec3::splat(f32::MAX),
            frame_counter: 0,
        }
    }

    /// Get grass vertices, regenerating at most every 30 frames or when
    /// the camera moved more than 5m.
    pub fn get_or_regen(
        &mut self,
        world: &World,
        cam_pos: Vec3,
        voxel_size: f32,
        game_time: f32,
    ) -> &[GrassVertex] {
        self.frame_counter += 1;
        let moved = (cam_pos - self.last_cam_pos).length_squared() > 25.0; // 5m
        if moved || self.frame_counter >= 30 {
            self.vertices = generate_grass(world, cam_pos, voxel_size, game_time);
            self.last_cam_pos = cam_pos;
            self.frame_counter = 0;
        }
        &self.vertices
    }
}

const BLADES_PER_VOXEL: usize = 4;

/// Generate grass blade vertices for all grass-top voxels near the camera.
/// Returns a flat Vec of GrassVertex — 4 per blade, forming a quad.
pub fn generate_grass(
    world: &World,
    cam_pos: Vec3,
    voxel_size: f32,
    game_time: f32,
) -> Vec<GrassVertex> {
    let mut vertices = Vec::new();
    let vs = voxel_size;

    // Convert camera position to voxel coordinates.
    let cam_voxel = IVec3::new(
        (cam_pos.x / vs) as i32,
        (cam_pos.y / vs) as i32,
        (cam_pos.z / vs) as i32,
    );
    let radius_voxels = (GRASS_RADIUS_M / vs) as i32;

    // Determine chunk range to scan.
    let min_chunk = chunk_origin(
        IVec3::new(cam_voxel.x - radius_voxels, 0, cam_voxel.z - radius_voxels) / CHUNK_SIZE as i32,
    ) / CHUNK_SIZE as i32;
    let max_chunk = chunk_origin(
        IVec3::new(cam_voxel.x + radius_voxels, 0, cam_voxel.z + radius_voxels) / CHUNK_SIZE as i32,
    ) / CHUNK_SIZE as i32;


    for cx in min_chunk.x..=max_chunk.x {
        for cz in min_chunk.z..=max_chunk.z {
            // Scan Y chunks from top down — grass is on the surface.
            let max_y_chunk = (world.cfg.extent_m[1] / (CHUNK_SIZE as f32 * vs)) as i32 + 1;
            for cy in (0..=max_y_chunk).rev() {
                let chunk_key = IVec3::new(cx, cy, cz);
                let origin = chunk_origin(chunk_key);
                let mut found_grass = false;
                for lx in 0..CHUNK_SIZE as i32 {
                    for lz in 0..CHUNK_SIZE as i32 {
                        for ly in (0..CHUNK_SIZE as i32).rev() {
                            let pos = IVec3::new(origin.x + lx, origin.y + ly, origin.z + lz);
                            let dx = pos.x as f32 * vs - cam_pos.x;
                            let dz = pos.z as f32 * vs - cam_pos.z;
                            if dx * dx + dz * dz > GRASS_RADIUS_M * GRASS_RADIUS_M {
                                continue;
                            }
                            let voxel = world.get_voxel(pos);
                            if voxel != vox_world::Voxel(3) {
                                continue;
                            }
                            let above = world.get_voxel(pos + IVec3::Y);
                            if above != AIR {
                                continue;
                            }
                            found_grass = true;
                            let center = Vec3::new(
                                pos.x as f32 * vs + vs * 0.5,
                                pos.y as f32 * vs + vs,
                                pos.z as f32 * vs + vs * 0.5,
                            );
                            for b in 0..BLADES_PER_VOXEL {
                                let bi = b as i32;
                                let h = hash01(pos.x * 17 + bi, pos.y * 31 + bi, pos.z * 13 + bi);
                                let h2 = hash01(pos.x * 7 + bi * 3, pos.y * 11 + bi * 5, pos.z * 19 + bi * 7);
                                let h3 = hash01(pos.x * 23 + bi * 11, pos.y * 5 + bi * 17, pos.z * 29 + bi * 2);
                                let offset_x = (h - 0.5) * vs * 0.7;
                                let offset_z = (h2 - 0.5) * vs * 0.7;
                                let base = Vec3::new(center.x + offset_x, center.y, center.z + offset_z);
                                let height = vs * (0.4 + h3 * 0.8);
                                let width = vs * (0.04 + h * 0.04);
                                let wind = (game_time * 1.5 + pos.x as f32 * 0.7 + pos.z as f32 * 0.5 + b as f32 * 1.3).sin();
                                let tip_offset_x = wind * height * 0.15;
                                let wind2 = (game_time * 1.1 + pos.z as f32 * 0.9 + b as f32 * 2.1).cos();
                                let tip_offset_z = wind2 * height * 0.10;
                                let facing = h2 * std::f32::consts::TAU;
                                let fx = facing.cos();
                                let fz = facing.sin();
                                let half_w = width * 0.5;
                                let tip = Vec3::new(base.x + tip_offset_x, base.y + height, base.z + tip_offset_z);
                                let bl = Vec3::new(base.x - fx * half_w, base.y, base.z - fz * half_w);
                                let br = Vec3::new(base.x + fx * half_w, base.y, base.z + fz * half_w);
                                let tl = Vec3::new(tip.x - fx * half_w * 0.3, tip.y, tip.z - fz * half_w * 0.3);
                                let tr = Vec3::new(tip.x + fx * half_w * 0.3, tip.y, tip.z + fz * half_w * 0.3);
                                vertices.push(GrassVertex { position: [bl.x, bl.y, bl.z], height_factor: 0.0 });
                                vertices.push(GrassVertex { position: [br.x, br.y, br.z], height_factor: 0.0 });
                                vertices.push(GrassVertex { position: [tl.x, tl.y, tl.z], height_factor: 1.0 });
                                vertices.push(GrassVertex { position: [tr.x, tr.y, tr.z], height_factor: 1.0 });
                                vertices.push(GrassVertex { position: [br.x, br.y, br.z], height_factor: 0.0 });
                                vertices.push(GrassVertex { position: [tr.x, tr.y, tr.z], height_factor: 1.0 });
                            }
                        }
                    }
                }
                if found_grass {
                    break; // Found grass in this XZ column, skip lower Y chunks.
                }
            }
        }
    }

    // Cap to max blade count.
    let max_verts = vox_render::MAX_GRASS_BLADES * 6;
    if vertices.len() > max_verts {
        vertices.truncate(max_verts);
    }

    vertices
}

fn hash01(x: i32, y: i32, z: i32) -> f32 {
    let n = (x.wrapping_mul(374761393) ^ y.wrapping_mul(668265263) ^ z.wrapping_mul(2147483647)) as u32;
    let n = n.wrapping_mul(2246822519);
    (n >> 8) as f32 / 16777216.0
}
