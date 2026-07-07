//! Voxel engine application: world, player, tools, threaded remeshing, render.

mod args;
mod player;
mod remesh;
mod tools;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use glam::{IVec3, Mat4, Vec3};
use rayon::prelude::*;

use player::Player;
use remesh::RemeshQueue;
use tools::{Tool, Tools};

use vox_core::consts::CHUNK_SIZE;
use vox_core::{
    FrameProfile, MaterialRegistry, ScopedTimer, Tunables, WorldConfig, chunk_origin, voxel_at,
};
use vox_debug::{DebugOverlay, OverlayState};
use vox_gen::{TerrainGen, TerrainMaterials, TreeMaterials, generate_trees};
use vox_mesh::{VoxelSlab, mesh_slab};
use vox_physics::{Body, PhysicsWorld, VoxelGrid};
use vox_platform::{App, FrameControl, FrameTiming, InputState, run_app};
use vox_render::{Camera, Frustum, Gpu, VoxelPipeline};
use vox_world::{Voxel, World};
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

/// Build the world: noise terrain + forest from the world config.
fn build_terrain_world(
    cfg: WorldConfig,
    registry: &MaterialRegistry,
) -> Result<World, Box<dyn std::error::Error + Send + Sync>> {
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
    registry: MaterialRegistry,
    player: Player,
    camera: Camera,
    tools: Tools,
    remesh: RemeshQueue,
    phys: PhysicsWorld,
    /// Incrementing seed so repeated blasts get varied debris spin.
    blast_seed: u32,
    grabbed: bool,
    frames: u32,
    last_report: Instant,
    /// Live-tunable parameters (friction, blast power, fly speed, ...),
    /// edited by the debug overlay's sliders and synced into the systems
    /// that actually consume them once per frame.
    tunables: Tunables,
    /// Rolling per-phase frame timings, shown in the debug overlay.
    profile: FrameProfile,
    debug_overlay: DebugOverlay,
    debug_visible: bool,
    /// Registry material names (excluding air), cached once for the debug
    /// overlay's material picker.
    material_names: Vec<String>,
    /// Index into `material_names` (not into the registry — the offset from
    /// `Tools`' own 1-based, air-inclusive indexing is handled where it's
    /// synced back after the overlay runs).
    selected_material: usize,
    /// Chunk draw/cull counts from the previous frame — the overlay is built
    /// before this frame's `draw_chunks` runs, so it necessarily shows the
    /// prior frame's numbers (one frame of latency, imperceptible in a HUD).
    last_draw_stats: vox_render::DrawStats,
}

impl VoxApp {
    fn new(
        window: Arc<Window>,
        cfg: WorldConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let assets = assets_dir();
        let registry = MaterialRegistry::load_dir(&assets.join("materials"))?;
        let shader = std::fs::read_to_string(assets.join("shaders/voxel.wgsl"))?;

        let build_start = Instant::now();
        let world = build_terrain_world(cfg, &registry)?;
        tracing::info!(
            chunks = world.chunk_count(),
            elapsed_ms = build_start.elapsed().as_millis() as u64,
            "world built"
        );

        let size = window.inner_size();
        let gpu = Gpu::new(window.clone(), size.width, size.height)?;
        let pipeline = VoxelPipeline::new(&gpu, &shader, &registry, world.cfg.voxel_size_m);
        let tools = Tools::new(&registry);
        let debug_overlay = DebugOverlay::new(gpu.device(), gpu.surface_format(), &window);
        // Air (id 0) is never player-selectable; the picker mirrors that.
        let material_names: Vec<String> = registry
            .iter()
            .skip(1)
            .map(|(_, def)| def.name.clone())
            .collect();
        // Tools' material_index is 1-based (air-inclusive); the picker's
        // index is 0-based into material_names — mirrors set_material_index.
        let selected_material = tools.material_index() - 1;

        let mut app = Self {
            window,
            gpu,
            pipeline,
            world,
            registry,
            player: Player::new(Vec3::ZERO),
            camera: Camera::new(Vec3::ZERO),
            tools,
            remesh: RemeshQueue::new(),
            phys: PhysicsWorld::new(),
            blast_seed: 0,
            grabbed: false,
            frames: 0,
            last_report: Instant::now(),
            tunables: Tunables::default(),
            profile: FrameProfile::new(),
            debug_overlay,
            debug_visible: false,
            material_names,
            selected_material,
            last_draw_stats: vox_render::DrawStats::default(),
        };
        app.initial_mesh();

        // Spawn on the terrain surface at the world center.
        let center = Vec3::from(app.world.cfg.extent_m) * 0.5;
        let surface = TerrainGen::surface_height_m(&app.world, center.x, center.z)
            .unwrap_or(app.world.cfg.extent_m[1] * 0.5);
        app.player = Player::new(Vec3::new(center.x, surface + 0.2, center.z));
        Ok(app)
    }

    /// Synchronous parallel meshing of the freshly generated world.
    fn initial_mesh(&mut self) {
        let keys = self.world.drain_dirty();
        let _ = self.world.drain_dirty_regions();
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
            "initial world mesh"
        );
    }

    /// Spawn a debris body at `origin_m` (a solid `extent`^3 wood cube),
    /// meshing and uploading it immediately, with `vel_m_s` initial velocity.
    fn spawn_debris(&mut self, origin_m: Vec3, extent: i32, vel_m_s: Vec3) {
        let wood = self
            .registry
            .id_by_name("wood")
            .map(|m| Voxel(m.0))
            .unwrap_or(Voxel(1));
        let dims = IVec3::splat(extent);
        let voxels = vec![wood; (dims.x * dims.y * dims.z) as usize];
        let grid = VoxelGrid::new(dims, voxels.clone());
        let Some(mut body) =
            Body::from_grid(grid, &self.registry, self.world.cfg.voxel_size_m, origin_m)
        else {
            return; // Massless grid (shouldn't happen for a solid cube).
        };
        body.vel = vel_m_s;
        let id = self.phys.spawn(body);

        let slab = VoxelSlab::from_grid(dims, &voxels);
        let mesh = mesh_slab(&slab);
        self.pipeline
            .upload_body(&self.gpu, (id.slot, id.generation), &mesh);
        tracing::info!(?id, ?origin_m, "spawned debris body");
    }

    /// Rewrite every awake debris body's GPU transform from the interpolated
    /// physics state. Chunk mesh vertices are in grid-voxel corner units
    /// scaled by `voxel_size_m` in the shader; the same scaling applies to
    /// debris, so the model matrix carries only translation and rotation.
    fn sync_debris_render(&mut self, alpha: f32) {
        for (id, body) in self.phys.iter() {
            let (pos, rot) = self
                .phys
                .interpolated_transform(id, alpha)
                .expect("id came from iter()");
            // grid_offset is already in meters (mass_props computes com_local
            // in meters); the shader's `local` is also meters after scaling
            // grid-corner units by voxel_size_m, so no unit conversion here.
            let model = Mat4::from_rotation_translation(rot, pos)
                * Mat4::from_translation(body.grid_offset);
            self.pipeline
                .update_body_transform(&self.gpu, (id.slot, id.generation), model);
        }
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

    /// Tool input: LMB uses the active tool (break/blast), RMB places.
    fn apply_tools(&mut self, input: &InputState) {
        let eye = self.player.eye(1.0);
        let look = self.player.look_dir();
        if input.mouse_clicked(MouseButton::Left) {
            match self.tools.tool {
                Tool::Blast => {
                    let seed = self.blast_seed;
                    self.blast_seed = self.blast_seed.wrapping_add(1);
                    self.tools.blast(
                        &mut self.world,
                        &mut self.phys,
                        &self.registry,
                        eye,
                        look,
                        self.tunables.blast_power,
                        seed,
                    );
                }
                _ => {
                    if let Some(v) = self.tools.break_voxel(&mut self.world, eye, look) {
                        Tools::check_broken_support(
                            &mut self.world,
                            &mut self.phys,
                            &self.registry,
                            v,
                        );
                    }
                }
            }
        }
        if input.mouse_clicked(MouseButton::Right) {
            self.tools
                .place_voxel(&mut self.world, eye, look, self.player.ctrl.aabb());
        }
        if input.wheel_delta.abs() >= 1.0 {
            let steps = input.wheel_delta as i32;
            self.tools.cycle_material(steps, &self.registry);
        }
    }

    fn report_stats(&mut self, stats: vox_render::DrawStats) {
        self.frames += 1;
        if self.last_report.elapsed().as_secs_f32() >= 1.0 {
            tracing::info!(
                fps = self.frames,
                drawn = stats.drawn,
                culled = stats.culled,
                queue = self.remesh.pending_len(),
                in_flight = self.remesh.in_flight,
                bodies = self.phys.body_count(),
                bodies_awake = self.phys.awake_count(),
                pos = ?voxel_at(self.player.ctrl.pos, self.world.cfg.voxel_size_m),
                grounded = self.player.ctrl.grounded,
                "frame stats"
            );
            self.frames = 0;
            self.last_report = Instant::now();
        }
    }
}

impl App for VoxApp {
    fn frame(&mut self, input: &mut InputState, timing: FrameTiming) -> FrameControl {
        // Measured manually (not via ScopedTimer's RAII guard): this block
        // calls several &mut self methods (set_grab, spawn_debris, ...),
        // which would conflict with a live &mut self.profile.input borrow.
        let input_start = Instant::now();
        if input.key_pressed(KeyCode::Escape) {
            if self.grabbed {
                self.set_grab(false);
            } else {
                return FrameControl::Exit;
            }
        }
        let mut grabbed_this_frame = false;
        if input.mouse_clicked(MouseButton::Left) && !self.grabbed {
            self.set_grab(true);
            grabbed_this_frame = true;
        }
        if input.key_pressed(KeyCode::KeyF) {
            self.player.toggle_fly();
        }
        if input.key_pressed(KeyCode::KeyB) {
            let origin = self.player.eye(1.0) + self.player.look_dir() * 4.0;
            self.spawn_debris(origin, 4, self.player.look_dir() * 8.0);
        }
        if input.key_pressed(KeyCode::KeyX) {
            let removed = self.phys.clear_sleeping();
            for id in &removed {
                self.pipeline.remove_body((id.slot, id.generation));
            }
            if !removed.is_empty() {
                tracing::info!(count = removed.len(), "cleared sleeping debris");
            }
        }
        for (key, tool) in [
            (KeyCode::Digit1, Tool::Place),
            (KeyCode::Digit2, Tool::Break),
            (KeyCode::Digit3, Tool::Blast),
        ] {
            if input.key_pressed(key) {
                self.tools.tool = tool;
                tracing::info!(tool = ?tool, "tool selected");
            }
        }
        if input.key_pressed(KeyCode::BracketLeft) {
            self.tools.shrink_blast_radius();
            tracing::info!(radius_m = self.tools.blast_radius, "blast radius");
        }
        if input.key_pressed(KeyCode::BracketRight) {
            self.tools.grow_blast_radius();
            tracing::info!(radius_m = self.tools.blast_radius, "blast radius");
        }
        if input.key_pressed(KeyCode::F3) {
            self.debug_visible = !self.debug_visible;
        }
        self.profile
            .input
            .push(input_start.elapsed().as_secs_f32() * 1000.0);

        // Sync the debug overlay's live tunables into the systems that
        // actually consume them (both fields are pub; this is a cheap copy,
        // not a real coupling).
        self.player.fly_speed = self.tunables.fly_speed;
        self.phys.tunables = self.tunables;

        if self.grabbed {
            self.player.look(input.mouse_delta);
        }
        {
            let _t = ScopedTimer::new(&mut self.profile.player);
            self.player
                .fixed_steps(&self.world, input, timing.physics_steps);
        }
        if self.grabbed && !grabbed_this_frame {
            // Manual timing: apply_tools takes &mut self as a whole, which
            // would conflict with a live &mut self.profile.tools borrow.
            let tools_start = Instant::now();
            self.apply_tools(input);
            self.profile
                .tools
                .push(tools_start.elapsed().as_secs_f32() * 1000.0);
        }
        {
            let _t = ScopedTimer::new(&mut self.profile.physics);
            for _ in 0..timing.physics_steps {
                self.phys.step(&self.world, vox_core::consts::PHYSICS_DT);
            }
        }

        // Wake any resting debris whose ground was just carved/edited from
        // under it, then remesh: absorb edits, dispatch to workers, upload.
        let eye = self.player.eye(timing.alpha);
        {
            let _t = ScopedTimer::new(&mut self.profile.remesh);
            let s = self.world.cfg.voxel_size_m;
            for (min, max) in self.world.drain_dirty_regions() {
                self.phys.wake_region(min.as_vec3() * s, max.as_vec3() * s);
            }
            self.remesh.absorb_dirty(&mut self.world);
            self.remesh.dispatch(&self.world, eye);
            self.remesh.collect(&self.gpu, &mut self.pipeline);
        }
        self.sync_debris_render(timing.alpha);

        // Camera from the interpolated player eye.
        self.camera.pos = eye;
        self.camera.yaw = self.player.yaw;
        self.camera.pitch = self.player.pitch;
        let (w, h) = self.gpu.surface_size();
        let aspect = w as f32 / h.max(1) as f32;
        let view_proj = self.camera.view_proj(aspect);
        self.pipeline
            .write_camera(&self.gpu, view_proj, self.camera.pos, FOG_END_M);
        let frustum = Frustum::from_view_proj(view_proj);

        let render_start = Instant::now();
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

        // Debug overlay UI must be built and its buffers uploaded before the
        // render pass opens (buffer uploads can't happen mid-pass); painting
        // itself happens inside the pass, right after the world.
        let prepared_overlay = self.debug_visible.then(|| {
            let (w, h) = self.gpu.surface_size();
            let state = OverlayState {
                profile: &self.profile,
                tunables: &mut self.tunables,
                fps: self.frames,
                chunks_drawn: self.last_draw_stats.drawn,
                chunks_culled: self.last_draw_stats.culled,
                mesh_queue: self.remesh.pending_len(),
                bodies_awake: self.phys.awake_count(),
                bodies_total: self.phys.body_count(),
                blast_radius: &mut self.tools.blast_radius,
                material_names: &self.material_names,
                selected_material: &mut self.selected_material,
            };
            self.debug_overlay.prepare(
                &self.window,
                self.gpu.device(),
                self.gpu.queue(),
                &mut encoder,
                (w, h),
                state,
            )
        });
        self.tools.set_material_index(self.selected_material + 1);

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
            self.pipeline.draw_bodies(&mut pass);
            if let Some(prepared) = &prepared_overlay {
                self.debug_overlay.paint(&mut pass, prepared);
            }
        }
        self.gpu.queue().submit([encoder.finish()]);
        frame.present();
        self.profile
            .render
            .push(render_start.elapsed().as_secs_f32() * 1000.0);
        self.last_draw_stats = stats;

        self.report_stats(stats);
        FrameControl::Continue
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.gpu.resize(width, height);
    }

    fn window_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        self.debug_overlay.on_window_event(&self.window, event)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli_args: Vec<String> = std::env::args().skip(1).collect();
    if args::wants_help(cli_args.iter().map(String::as_str)) {
        println!("{}", args::usage());
        return Ok(());
    }
    let cfg = match args::parse(cli_args.iter().map(String::as_str)) {
        Ok(cfg) => cfg,
        Err(msg) => {
            eprintln!("error: {msg}\n\n{}", args::usage());
            std::process::exit(1);
        }
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    tracing::info!(
        voxel_size_m = cfg.voxel_size_m,
        seed = cfg.seed,
        extent_m = ?cfg.extent_m,
        "world config"
    );

    run_app(vox_core::consts::PHYSICS_DT, |window| {
        Ok(Box::new(VoxApp::new(window, cfg)?))
    })?;
    Ok(())
}
