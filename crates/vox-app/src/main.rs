//! Voxel engine application entry point: world, meshing, camera, frame loop.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use vox_core::consts::CHUNK_SIZE;
use vox_core::{MaterialRegistry, WorldConfig, chunk_origin, voxel_at};
use vox_gen::{TerrainGen, TerrainMaterials, TreeMaterials, generate_trees};
use vox_mesh::{VoxelSlab, mesh_slab};
use vox_platform::{App, FrameControl, FrameTiming, InputState, run_app};
use vox_render::{Camera, Frustum, Gpu, VoxelPipeline};
use vox_world::World;
use winit::event::MouseButton;
use winit::keyboard::KeyCode;
use winit::window::{CursorGrabMode, Window};

/// Sky-blue clear color (linear-space RGBA); must match the shader's fog sky.
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.45,
    g: 0.66,
    b: 0.90,
    a: 1.0,
};

/// Fly-camera speed in m/s (`Ctrl` multiplies by 5).
const FLY_SPEED: f32 = 12.0;
/// Mouse look sensitivity in radians per pixel of raw mouse delta.
const LOOK_SENSITIVITY: f32 = 0.0025;
/// Fog end distance in meters.
const FOG_END_M: f32 = 220.0;

/// Locate the `assets/` directory: the workspace copy during development,
/// else `assets/` beside the executable.
fn assets_dir() -> PathBuf {
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets");
    if dev.is_dir() {
        return dev;
    }
    PathBuf::from("assets")
}

/// Build the world: noise terrain from the world config.
fn build_terrain_world(
    registry: &MaterialRegistry,
) -> Result<World, Box<dyn std::error::Error + Send + Sync>> {
    let cfg = WorldConfig {
        voxel_size_m: 0.1,
        extent_m: [128.0, 48.0, 128.0],
        ..WorldConfig::default()
    };
    cfg.validate()?;
    let mut world = World::new(cfg);
    let mats = TerrainMaterials::from_registry(registry)?;
    let terrain = TerrainGen::new(&world.cfg);
    terrain.generate(&mut world, mats);
    let tree_mats = TreeMaterials::from_registry(registry)?;
    let planted = generate_trees(&mut world, &terrain, tree_mats);
    tracing::info!(trees = planted, "forest planted");
    Ok(world)
}

/// The engine application.
struct VoxApp {
    window: Arc<Window>,
    gpu: Gpu,
    pipeline: VoxelPipeline,
    world: World,
    camera: Camera,
    grabbed: bool,
    // FPS/stat reporting.
    frames: u32,
    last_report: Instant,
}

impl VoxApp {
    fn new(window: Arc<Window>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let assets = assets_dir();
        let registry = MaterialRegistry::load_dir(&assets.join("materials"))?;
        let shader = std::fs::read_to_string(assets.join("shaders/voxel.wgsl"))?;

        let build_start = Instant::now();
        let world = build_terrain_world(&registry)?;
        tracing::info!(
            chunks = world.chunk_count(),
            elapsed_ms = build_start.elapsed().as_millis() as u64,
            "world built"
        );

        let size = window.inner_size();
        let gpu = Gpu::new(window.clone(), size.width, size.height)?;
        let pipeline = VoxelPipeline::new(&gpu, &shader, &registry, world.cfg.voxel_size_m);

        let mut app = Self {
            window,
            gpu,
            pipeline,
            world,
            camera: Camera::new(Vec3::ZERO),
            grabbed: false,
            frames: 0,
            last_report: Instant::now(),
        };
        app.mesh_dirty_chunks();

        // Spawn the camera above the terrain surface at the world center.
        let center = Vec3::from(app.world.cfg.extent_m) * 0.5;
        let surface = TerrainGen::surface_height_m(&app.world, center.x, center.z)
            .unwrap_or(app.world.cfg.extent_m[1] * 0.5);
        app.camera.pos = Vec3::new(center.x, surface + 4.0, center.z);
        Ok(app)
    }

    /// Mesh and upload every dirty chunk (parallel mesh, sequential upload).
    fn mesh_dirty_chunks(&mut self) {
        let keys = self.world.drain_dirty();
        if keys.is_empty() {
            return;
        }
        let start = Instant::now();
        let world = &self.world;
        let meshes: Vec<(IVec3, vox_mesh::MeshData)> = keys
            .par_iter()
            .filter(|key| world.chunk_at(**key).is_some())
            .map(|key| {
                let slab =
                    VoxelSlab::extract(world, chunk_origin(*key), IVec3::splat(CHUNK_SIZE as i32));
                (*key, mesh_slab(&slab))
            })
            .collect();
        let meshed = meshes.len();
        let mut quads = 0usize;
        for (key, mesh) in meshes {
            quads += mesh.quads();
            self.pipeline.upload_chunk(&self.gpu, key, &mesh);
        }
        tracing::info!(
            chunks = meshed,
            quads,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "meshed dirty chunks"
        );
    }

    fn set_grab(&mut self, grab: bool) {
        let mode = if grab {
            CursorGrabMode::Locked
        } else {
            CursorGrabMode::None
        };
        let result = self.window.set_cursor_grab(mode).or_else(|_| {
            self.window.set_cursor_grab(if grab {
                CursorGrabMode::Confined
            } else {
                CursorGrabMode::None
            })
        });
        match result {
            Ok(()) => {
                self.window.set_cursor_visible(!grab);
                self.grabbed = grab;
            }
            Err(err) => tracing::warn!(%err, "cursor grab change failed"),
        }
    }

    fn update_camera(&mut self, input: &InputState, dt: f32) {
        if self.grabbed {
            let d = input.mouse_delta;
            self.camera
                .look(-d.x * LOOK_SENSITIVITY, -d.y * LOOK_SENSITIVITY);
        }
        let mut wish = Vec3::ZERO;
        let forward = self.camera.forward();
        let right = self.camera.right();
        if input.key_down(KeyCode::KeyW) {
            wish += forward;
        }
        if input.key_down(KeyCode::KeyS) {
            wish -= forward;
        }
        if input.key_down(KeyCode::KeyD) {
            wish += right;
        }
        if input.key_down(KeyCode::KeyA) {
            wish -= right;
        }
        if input.key_down(KeyCode::Space) {
            wish += Vec3::Y;
        }
        if input.key_down(KeyCode::ShiftLeft) {
            wish -= Vec3::Y;
        }
        if wish != Vec3::ZERO {
            let boost = if input.key_down(KeyCode::ControlLeft) {
                5.0
            } else {
                1.0
            };
            self.camera.pos += wish.normalize() * FLY_SPEED * boost * dt;
        }
    }

    fn report_stats(&mut self, stats: vox_render::DrawStats) {
        self.frames += 1;
        if self.last_report.elapsed().as_secs_f32() >= 1.0 {
            tracing::info!(
                fps = self.frames,
                drawn = stats.drawn,
                culled = stats.culled,
                cam = ?voxel_at(self.camera.pos, self.world.cfg.voxel_size_m),
                "frame stats"
            );
            self.frames = 0;
            self.last_report = Instant::now();
        }
    }
}

impl App for VoxApp {
    fn frame(&mut self, input: &mut InputState, timing: FrameTiming) -> FrameControl {
        if input.key_pressed(KeyCode::Escape) {
            if self.grabbed {
                self.set_grab(false);
            } else {
                return FrameControl::Exit;
            }
        }
        if input.mouse_clicked(MouseButton::Left) && !self.grabbed {
            self.set_grab(true);
        }
        self.update_camera(input, timing.dt_frame);

        let (w, h) = self.gpu.surface_size();
        let aspect = w as f32 / h.max(1) as f32;
        let view_proj = self.camera.view_proj(aspect);
        self.pipeline
            .write_camera(&self.gpu, view_proj, self.camera.pos, FOG_END_M);
        let frustum = Frustum::from_view_proj(view_proj);

        let frame = match self.gpu.begin_frame() {
            Ok(frame) => frame,
            Err(err) if err.is_transient() => {
                tracing::warn!(error = %err, "transient surface error; skipping frame");
                return FrameControl::Continue;
            }
            Err(err) => {
                tracing::error!(error = %err, "fatal render error; shutting down");
                return FrameControl::Exit;
            }
        };

        let mut encoder =
            self.gpu
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("frame-encoder"),
                });
        let stats;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("voxel-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame.view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.gpu.depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            stats = self.pipeline.draw_chunks(&mut pass, &frustum);
        }
        self.gpu.queue().submit([encoder.finish()]);
        frame.present();

        self.report_stats(stats);
        FrameControl::Continue
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.gpu.resize(width, height);
    }

    fn window_event(&mut self, _event: &winit::event::WindowEvent) -> bool {
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    run_app(vox_core::consts::PHYSICS_DT, |window| {
        Ok(Box::new(VoxApp::new(window)?))
    })?;
    Ok(())
}
