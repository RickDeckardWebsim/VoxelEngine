//! Voxel engine application: world, player, tools, threaded remeshing, render.

mod args;
mod body_mesh;
mod player;
mod remesh;
mod tools;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use glam::{IVec3, Mat4, Vec3};
use rayon::prelude::*;

use player::Player;
use body_mesh::BodyMeshQueue;
use remesh::RemeshQueue;
use tools::{Tool, Tools};

use vox_core::consts::CHUNK_SIZE;
use vox_core::{
    FrameProfile, MaterialId, MaterialRegistry, ScopedTimer, Tunables, WorldConfig, chunk_origin,
    voxel_at,
};
use vox_debug::{DebugOverlay, OverlayState};
use vox_gen::{TerrainGen, TerrainMaterials, TreeMaterials, generate_trees};
use vox_mesh::{VoxelSlab, mesh_slab};
use vox_physics::{Body, BodyId, ImpactEvent, PhysicsWorld, VoxelGrid};
use vox_platform::{App, FrameControl, FrameTiming, InputState, run_app};
use vox_render::{Camera, Frustum, Gpu, VoxelPipeline};
use vox_world::{AIR, Voxel, World};
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
    /// Threaded debris mesh generation (see `body_mesh`'s module docs for
    /// why bodies don't need the generation/staleness tracking chunks do).
    body_mesh: BodyMeshQueue,
    phys: PhysicsWorld,
    /// Incrementing seed so repeated blasts get varied debris spin.
    blast_seed: u32,
    /// Incrementing seed so repeated impact-fracture chips get varied
    /// spin/velocity jitter (same idea as `blast_seed`).
    impact_seed: u32,
    grabbed: bool,
    frames: u32,
    /// Frame count from the most recently *completed* one-second window —
    /// the stable value the overlay displays. `frames` itself is a live
    /// accumulator that counts 0 up to roughly the frame rate and resets
    /// every second, so reading it directly (as the overlay used to) shows
    /// a sawtooth, not an FPS.
    last_fps: u32,
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
    /// Every debris body ever spawned, oldest first -- every debris body is
    /// its own GPU buffer set and its own draw call (see
    /// `VoxelPipeline::draw_bodies`'s doc comment on why), so letting this
    /// grow without bound over a play session (repeated bombing especially,
    /// now that a bomb also scatters small chip debris) is the single
    /// biggest driver of the engine bogging down over time. Used by
    /// `enforce_debris_budget` to evict the oldest *settled* debris once the
    /// total exceeds `MAX_DEBRIS_BODIES`.
    debris_order: VecDeque<BodyId>,
}

/// Hard cap on total debris bodies alive at once. Past this, the oldest
/// already-asleep (settled) debris is evicted to make room for new debris,
/// never anything still actively flying/settling -- see
/// `VoxApp::enforce_debris_budget`.
const MAX_DEBRIS_BODIES: usize = 200;

/// Bodies at or below this voxel count mesh synchronously in
/// `VoxApp::upload_debris_mesh` instead of going through the threaded
/// `body_mesh` queue -- see that function's doc comment for why.
const INLINE_MESH_VOXEL_BUDGET: usize = 64_000;

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
        let debug_overlay = DebugOverlay::new(
            gpu.device(),
            gpu.surface_format(),
            Some(vox_render::DEPTH_FORMAT),
            &window,
        );
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
            body_mesh: BodyMeshQueue::new(),
            phys: PhysicsWorld::new(),
            blast_seed: 0,
            impact_seed: 0,
            grabbed: false,
            frames: 0,
            last_fps: 0,
            last_report: Instant::now(),
            tunables: Tunables::default(),
            profile: FrameProfile::new(),
            debug_overlay,
            debug_visible: false,
            material_names,
            selected_material,
            last_draw_stats: vox_render::DrawStats::default(),
            debris_order: VecDeque::new(),
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
                let origin = chunk_origin(*key);
                let slab = VoxelSlab::extract(world, origin, IVec3::splat(CHUNK_SIZE as i32));
                (*key, mesh_slab(&slab, origin))
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
        let grid = VoxelGrid::new(dims, voxels);
        let Some(mut body) =
            Body::from_grid(grid, &self.registry, self.world.cfg.voxel_size_m, origin_m)
        else {
            return; // Massless grid (shouldn't happen for a solid cube).
        };
        body.vel = vel_m_s;
        let id = self.phys.spawn(body);
        self.upload_debris_mesh(id);
        tracing::info!(?id, ?origin_m, "spawned debris body");
    }

    /// Mesh and upload an already-spawned body's GPU representation. Every
    /// path that calls `phys.spawn` must follow up with this — a body with
    /// no uploaded mesh is simulated (falls, collides, sleeps) but never
    /// drawn, which looks indistinguishable from the material having simply
    /// vanished. Does nothing if `id` is no longer alive.
    ///
    /// Small bodies mesh right here, synchronously: the stress example
    /// measures even a 40^3 cube (64,000 voxels) at ~1.7ms average, cheap
    /// enough not to be worth deferring. This matters because splitting a
    /// body during destruction is the overwhelmingly common case, and it
    /// always produces several small fragments in the same frame the old
    /// body is despawned — sending every one of those through the
    /// background queue meant the whole cluster popped out of existence for
    /// the frame or two its meshes took to come back, which is exactly the
    /// "flicker"/"goes invisible as it destroys" symptom: not a rendering
    /// glitch, but a real gap where nothing was uploaded yet. Only bodies
    /// above the budget (a large one-off spawn, e.g. a whole tree trunk
    /// detached at once) still go through `body_mesh` to keep that meshing
    /// cost off the main thread.
    fn upload_debris_mesh(&mut self, id: BodyId) {
        let Some(body) = self.phys.get(id) else {
            return;
        };
        let voxel_count = (body.grid.dims.x * body.grid.dims.y * body.grid.dims.z) as usize;
        if voxel_count <= INLINE_MESH_VOXEL_BUDGET {
            let slab = VoxelSlab::from_grid(body.grid.dims, &body.grid.voxels);
            let mesh = mesh_slab(&slab, IVec3::ZERO);
            self.pipeline
                .upload_body(&self.gpu, (id.slot, id.generation), &mesh);
        } else {
            self.body_mesh.dispatch(
                (id.slot, id.generation),
                body.grid.dims,
                body.grid.voxels.clone(),
            );
        }
        self.debris_order.push_back(id);
    }

    /// Keep total debris body count under `MAX_DEBRIS_BODIES`: evict the
    /// oldest already-settled debris (see `evict_oldest_asleep_debris`) and
    /// drop each evicted body's GPU mesh too.
    fn enforce_debris_budget(&mut self) {
        for id in evict_oldest_asleep_debris(&mut self.phys, &mut self.debris_order, MAX_DEBRIS_BODIES) {
            self.pipeline.remove_body((id.slot, id.generation));
        }
    }

    /// Material-based impact destruction: check each impact this frame's
    /// physics step(s) produced against the material actually at that
    /// point. A hit whose speed (impulse/mass -- the velocity change the
    /// contact imparted, independent of the body's own mass) exceeds that
    /// material's own fracture threshold chips debris apart right where it
    /// was struck, reusing the same carve+split mechanism the tools use.
    /// Tougher materials (higher `strength`, same convention as the
    /// destruction tools: "higher values survive bigger blasts") need a
    /// harder hit to trigger at all, *and* lose proportionally less volume
    /// when they do -- stone barely chips, leaves come apart at the
    /// slightest touch, wood sits in between but sheds a bigger bite once
    /// it does go (see [`fracture_radius_vox`]). See
    /// `Tunables::fracture_sensitivity` for the overall threshold dial.
    fn apply_impact_fracture(&mut self, impacts: Vec<ImpactEvent>) {
        for event in impacts {
            let Some(body) = self.phys.get(event.body) else {
                continue;
            };
            let impact_speed = event.impulse / body.mass();
            let voxel_size_m = body.half_voxel * 2.0;
            let local = body.rot.inverse() * (event.point_m - body.pos) - body.grid_offset;
            let local_voxel = (local / voxel_size_m).floor().as_ivec3();
            let material = body.grid.get(local_voxel);
            if material == AIR {
                continue;
            }
            let Some(def) = self.registry.get(MaterialId(material.0)) else {
                continue;
            };
            let Some(radius_vox) =
                fracture_radius_vox(def.strength, impact_speed, self.tunables.fracture_sensitivity)
            else {
                continue;
            };
            let radius_m = voxel_size_m * radius_vox;

            // Carve centered a little *into* the body along the actual
            // contact push direction, not straddling half in / half out of
            // the surface the way a sphere planted exactly on the contact
            // point would -- "directly remove what impacts what" instead of
            // a generic point-in-space blob.
            let push = event.push_dir.normalize_or_zero();
            let center = event.point_m + push * (radius_m * 0.5);

            let seed = self.impact_seed;
            self.impact_seed = self.impact_seed.wrapping_add(1);
            let spawned = vox_physics::carve_body_sphere_at_impact(
                &mut self.phys,
                &self.registry,
                event.body,
                center,
                radius_m,
                push,
                impact_speed,
                seed,
            );
            if self.phys.get(event.body).is_none() {
                self.pipeline
                    .remove_body((event.body.slot, event.body.generation));
                for id in spawned {
                    self.upload_debris_mesh(id);
                }
            }
        }
    }

    /// Rewrite every awake debris body's GPU transform from the interpolated
    /// physics state. Chunk mesh vertices are in grid-voxel corner units
    /// scaled by `voxel_size_m` in the shader; the same scaling applies to
    /// debris, so the model matrix carries only translation and rotation.
    ///
    /// Skips asleep bodies entirely: the solver's integration step already
    /// skips them (a sleeping body's `pos`/`rot` are frozen -- that's the
    /// whole point of sleeping), so their last-written GPU transform is
    /// still exactly correct. Without this, every debris body pays a
    /// `write_buffer` call *every frame forever*, even sitting perfectly
    /// still -- with up to `MAX_DEBRIS_BODIES` debris around and most of it
    /// asleep in steady state (normal play, well after a blast), that's
    /// dozens to hundreds of pointless GPU writes a frame, scaling with
    /// exactly "the longer you play, the worse it gets."
    fn sync_debris_render(&mut self, alpha: f32) {
        for (id, body) in self.phys.iter() {
            if body.sleep.asleep {
                continue;
            }
            let (pos, rot) = self
                .phys
                .interpolated_transform(id, alpha)
                .expect("id came from iter()");
            // grid_offset is already in meters (mass_props computes com_local
            // in meters); the shader's `local` is also meters after scaling
            // grid-corner units by voxel_size_m, so no unit conversion here.
            let model = Mat4::from_rotation_translation(rot, pos)
                * Mat4::from_translation(body.grid_offset);
            self.pipeline.update_body_transform(
                &self.gpu,
                (id.slot, id.generation),
                model,
                body.aabb_min,
                body.aabb_max,
            );
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

    /// Tool input: LMB uses the active hotbar tool, RMB places.
    fn apply_tools(&mut self, input: &InputState) {
        let eye = self.player.eye(1.0);
        let look = self.player.look_dir();
        if input.mouse_clicked(MouseButton::Left) {
            let outcome = match self.tools.tool {
                Tool::Dig => self
                    .tools
                    .dig(&mut self.world, &mut self.phys, &self.registry, eye, look),
                Tool::ScalableDig => {
                    self.tools
                        .scalable_dig(&mut self.world, &mut self.phys, &self.registry, eye, look)
                }
                Tool::Bomb => {
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
                    )
                }
                Tool::DeathLaser => {
                    self.tools
                        .death_laser(&mut self.world, &mut self.phys, &self.registry, eye, look)
                }
            };
            // A carved body is despawned and replaced, not updated in
            // place -- drop its old mesh before uploading the new
            // fragments', or a frozen ghost of the original stays on
            // screen forever at its last known transform.
            for id in outcome.removed {
                self.pipeline.remove_body((id.slot, id.generation));
            }
            for id in outcome.spawned {
                self.upload_debris_mesh(id);
            }
        }
        if input.mouse_clicked(MouseButton::Right) {
            self.tools
                .place_voxel(&mut self.world, eye, look, self.player.ctrl.aabb());
        }
        if input.wheel_delta.abs() >= 1.0 {
            let steps = input.wheel_delta as i32;
            if self.tools.has_adjustable_radius() {
                self.tools.adjust_radius(steps);
            } else {
                self.tools.cycle_material(steps, &self.registry);
            }
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
            self.last_fps = self.frames;
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
        const HOTBAR_KEYS: [KeyCode; 9] = [
            KeyCode::Digit1,
            KeyCode::Digit2,
            KeyCode::Digit3,
            KeyCode::Digit4,
            KeyCode::Digit5,
            KeyCode::Digit6,
            KeyCode::Digit7,
            KeyCode::Digit8,
            KeyCode::Digit9,
        ];
        for (i, key) in HOTBAR_KEYS.into_iter().enumerate() {
            if input.key_pressed(key)
                && let Some(tool) = self.tools.select_hotbar_slot(i as u8 + 1)
            {
                tracing::info!(tool = ?tool, "tool selected");
            }
        }
        if input.key_pressed(KeyCode::BracketLeft) {
            self.tools.shrink_radius();
            tracing::info!(radius_m = self.tools.radius_m, "tool radius");
        }
        if input.key_pressed(KeyCode::BracketRight) {
            self.tools.grow_radius();
            tracing::info!(radius_m = self.tools.radius_m, "tool radius");
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
        let impacts = {
            let _t = ScopedTimer::new(&mut self.profile.physics);
            let mut impacts = Vec::new();
            for _ in 0..timing.physics_steps {
                impacts.extend(self.phys.step(&self.world, vox_core::consts::PHYSICS_DT));
            }
            impacts
        };
        self.apply_impact_fracture(impacts);
        self.enforce_debris_budget();

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
            self.body_mesh.collect(&self.gpu, &mut self.pipeline);
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
                fps: self.last_fps,
                chunks_drawn: self.last_draw_stats.drawn,
                chunks_culled: self.last_draw_stats.culled,
                mesh_queue: self.remesh.pending_len(),
                body_mesh_in_flight: self.body_mesh.in_flight,
                bodies_awake: self.phys.awake_count(),
                bodies_total: self.phys.body_count(),
                tool_radius: &mut self.tools.radius_m,
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
            let chunk_stats = self.pipeline.draw_chunks(&mut pass, &frustum);
            let body_stats = self.pipeline.draw_bodies(&mut pass, &frustum);
            stats = vox_render::DrawStats {
                drawn: chunk_stats.drawn + body_stats.drawn,
                culled: chunk_stats.culled + body_stats.culled,
            };
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

/// Radius, in voxels, of the smallest possible fracture -- what *every*
/// material chips loose the instant an impact just barely clears its own
/// threshold, regardless of how fragile or tough that material is. A tiny
/// hit must always produce a tiny bite, never "an orb of voxels deleted
/// from space": see [`fracture_radius_vox`]'s doc comment for how growth
/// past this baseline is what actually differs per material.
const FRACTURE_RADIUS_VOX: f32 = 1.25;
/// Strength a material's crumble *growth rate* is calibrated against
/// (stone's own strength: the toughest common building material, so its
/// growth factor bottoms out at 0 -- stone barely chips even under a
/// genuinely hard hit, it just keeps producing the base radius above).
const FRACTURE_REFERENCE_STRENGTH: f32 = 8.0;
/// Clamp on the per-material growth factor derived from
/// `FRACTURE_REFERENCE_STRENGTH / strength` -- keeps an extremely fragile
/// material (leaves, strength 0.5) from growing arbitrarily fast as
/// `strength` approaches zero, while still growing much faster than a
/// tougher one for the same excess impact speed.
const CRUMBLE_SCALE_RANGE: (f32, f32) = (1.0, 5.0);
/// Cap on how far the square-root growth term (see
/// [`fracture_radius_vox`]) is allowed to run, so a truly enormous single
/// impact (e.g. a falling boulder) still can't carve out most of a small
/// body in one hit.
const ENERGY_GROWTH_CAP: f32 = 1.5;

/// Pure impact-fracture decision, kept free of live GPU/registry state so
/// it's unit-testable without a whole `VoxApp`: given a material's
/// `strength`, the impact speed it just took, and the live
/// `fracture_sensitivity` dial, should it fracture, and if so, what carve
/// radius (in voxels, before scaling by the body's own voxel size)?
/// `None` if the impact doesn't clear the material's threshold.
///
/// Direct proportionality on the threshold: a tougher material needs *more*
/// impact speed to fracture at all, matching the same `strength` convention
/// every other destruction tool already uses (higher = harder to destroy).
/// An earlier version of this divided instead of multiplied, which made
/// stone fracture *more* easily than wood.
///
/// The radius is `FRACTURE_RADIUS_VOX` plus a growth term that is zero at
/// the bare threshold and increases with *how far past* it the impact was,
/// scaled by how fragile the material is. This matters more than it might
/// look: an earlier version scaled the *entire* radius by a per-material
/// factor (leaves' factor pinned at its max range regardless of impact
/// strength), so even the gentlest leaf-fracturing hit already carved a 5x
/// radius -- "a tiny hit blows out a huge chunk." Splitting growth from the
/// base fixes that: every material's bare-threshold hit is the same small
/// bite, and *only* harder impacts open up the gap between a tough
/// material (stone's factor bottoms out at 0 growth -- it just keeps
/// producing the base bite) and a fragile one (leaves grows fastest).
fn fracture_radius_vox(strength: f32, impact_speed: f32, fracture_sensitivity: f32) -> Option<f32> {
    if strength <= 0.0 {
        return None; // Already as fragile as it gets; nothing to compare against.
    }
    let threshold = fracture_sensitivity * strength;
    if impact_speed < threshold {
        return None;
    }
    let fragility = (FRACTURE_REFERENCE_STRENGTH / strength).clamp(CRUMBLE_SCALE_RANGE.0, CRUMBLE_SCALE_RANGE.1);
    let excess = (impact_speed / threshold - 1.0).max(0.0);
    let growth = excess.sqrt().min(ENERGY_GROWTH_CAP);
    Some(FRACTURE_RADIUS_VOX * (1.0 + growth * (fragility - 1.0)))
}

/// Pure eviction decision behind `VoxApp::enforce_debris_budget`, kept free
/// of GPU/pipeline state so it's unit-testable directly against a
/// `PhysicsWorld`: while `phys` has more than `max_bodies` live bodies, pop
/// the oldest entry from `order`; if it's still alive and asleep, despawn
/// it (recorded in the returned list, so the caller can also drop its GPU
/// mesh); if it's still awake, requeue it at the back and keep looking
/// further back in time instead. Bounded to one pass over `order`'s
/// current length, so a run of nothing-but-awake debris can't turn this
/// into an unbounded loop -- it's fine to briefly sit over budget rather
/// than yank debris out from under the player mid-flight.
fn evict_oldest_asleep_debris(
    phys: &mut PhysicsWorld,
    order: &mut VecDeque<BodyId>,
    max_bodies: usize,
) -> Vec<BodyId> {
    let mut evicted = Vec::new();
    let mut attempts = order.len();
    while phys.body_count() > max_bodies && attempts > 0 {
        attempts -= 1;
        let Some(id) = order.pop_front() else {
            break;
        };
        match phys.get(id) {
            None => {} // Already gone (despawned elsewhere); drop the stale entry.
            Some(body) if body.sleep.asleep => {
                phys.despawn(id);
                evicted.push(id);
            }
            Some(_) => order.push_back(id), // Still active; retry later.
        }
    }
    evicted
}

#[cfg(test)]
mod debris_budget_tests {
    use super::*;

    fn asleep_body(pos: Vec3) -> Body {
        let grid = VoxelGrid::new(IVec3::splat(2), vec![Voxel(1); 8]);
        let reg = MaterialRegistry::from_toml_str(
            "[[material]]\nname = \"stone\"\ncolor = [0.5,0.5,0.5]\ndensity = 2600.0\nstrength = 8.0\n",
            "test.toml",
        )
        .expect("registry");
        let mut body = Body::from_grid(grid, &reg, 0.2, pos).expect("massive");
        body.sleep.asleep = true;
        body
    }

    #[test]
    fn evicts_oldest_first_once_over_budget() {
        let mut phys = PhysicsWorld::new();
        let mut order = VecDeque::new();
        let mut ids = Vec::new();
        for i in 0..5 {
            let id = phys.spawn(asleep_body(Vec3::splat(i as f32)));
            order.push_back(id);
            ids.push(id);
        }

        let evicted = evict_oldest_asleep_debris(&mut phys, &mut order, 3);

        assert_eq!(evicted, vec![ids[0], ids[1]], "must evict the two oldest, in order");
        assert_eq!(phys.body_count(), 3);
        assert!(phys.get(ids[2]).is_some(), "the three newest must survive");
    }

    #[test]
    fn under_budget_evicts_nothing() {
        let mut phys = PhysicsWorld::new();
        let mut order = VecDeque::new();
        for i in 0..3 {
            order.push_back(phys.spawn(asleep_body(Vec3::splat(i as f32))));
        }

        let evicted = evict_oldest_asleep_debris(&mut phys, &mut order, 10);

        assert!(evicted.is_empty());
        assert_eq!(phys.body_count(), 3);
    }

    #[test]
    fn never_evicts_an_awake_body_even_over_budget() {
        let mut phys = PhysicsWorld::new();
        let mut order = VecDeque::new();
        let mut awake = asleep_body(Vec3::ZERO);
        awake.sleep.asleep = false;
        let awake_id = phys.spawn(awake);
        order.push_back(awake_id);
        for i in 1..4 {
            order.push_back(phys.spawn(asleep_body(Vec3::splat(i as f32))));
        }

        let evicted = evict_oldest_asleep_debris(&mut phys, &mut order, 1);

        assert!(
            !evicted.contains(&awake_id),
            "an awake body must never be evicted, even to satisfy the budget"
        );
        assert!(phys.get(awake_id).is_some());
    }

    #[test]
    fn a_stale_entry_for_an_already_despawned_body_is_dropped_harmlessly() {
        let mut phys = PhysicsWorld::new();
        let mut order = VecDeque::new();
        let stale_id = phys.spawn(asleep_body(Vec3::ZERO));
        order.push_back(stale_id);
        phys.despawn(stale_id); // despawned by something else entirely

        // A second, still-alive body keeps `phys.body_count()` above the
        // budget, so the eviction loop actually has a reason to process
        // the stale front-of-queue entry instead of exiting immediately.
        let real_id = phys.spawn(asleep_body(Vec3::ONE));
        order.push_back(real_id);

        let evicted = evict_oldest_asleep_debris(&mut phys, &mut order, 0);

        assert!(
            !evicted.contains(&stale_id),
            "a stale id must not be reported as evicted -- it was already gone"
        );
        assert_eq!(evicted, vec![real_id], "the real, still-alive body must still be evicted");
        assert!(order.is_empty(), "the stale entry must still be dropped from the queue");
    }
}

#[cfg(test)]
mod fracture_tests {
    use super::*;

    /// The core material set's real strengths (see
    /// `assets/materials/core.toml`): leaves 0.5, wood 4.0, stone 8.0.
    const LEAVES: f32 = 0.5;
    const WOOD: f32 = 4.0;
    const STONE: f32 = 8.0;

    #[test]
    fn tougher_materials_need_more_speed_to_fracture_at_all() {
        // Same impact speed, everywhere between wood's and stone's
        // thresholds: wood must fracture, stone must not.
        let sensitivity = 1.0;
        let speed = WOOD * sensitivity + 0.1;
        assert!(fracture_radius_vox(WOOD, speed, sensitivity).is_some());
        assert!(fracture_radius_vox(STONE, speed, sensitivity).is_none());
    }

    #[test]
    fn leaves_fracture_on_a_much_gentler_impact_than_wood_or_stone() {
        let sensitivity = 1.0;
        let gentle = LEAVES * sensitivity + 0.05;
        assert!(
            fracture_radius_vox(LEAVES, gentle, sensitivity).is_some(),
            "leaves must give way at a gentle speed"
        );
        assert!(
            fracture_radius_vox(WOOD, gentle, sensitivity).is_none(),
            "the same gentle speed must not touch wood"
        );
        assert!(
            fracture_radius_vox(STONE, gentle, sensitivity).is_none(),
            "the same gentle speed must not touch stone"
        );
    }

    #[test]
    fn weaker_materials_crumble_a_bigger_radius_once_they_do_fracture() {
        // Exactly stone's own threshold: clears every material's threshold
        // (stone's is the highest) while keeping stone's `excess` at 1.0, so
        // its energy scale-up is a no-op and it lands exactly on the
        // reference radius below.
        let sensitivity = 1.0;
        let hard_hit = STONE * sensitivity;
        let leaves_r = fracture_radius_vox(LEAVES, hard_hit, sensitivity).expect("must fracture");
        let wood_r = fracture_radius_vox(WOOD, hard_hit, sensitivity).expect("must fracture");
        let stone_r = fracture_radius_vox(STONE, hard_hit, sensitivity).expect("must fracture");
        assert!(
            leaves_r > wood_r,
            "leaves ({leaves_r}) must crumble more than wood ({wood_r})"
        );
        assert!(
            wood_r > stone_r,
            "wood ({wood_r}) must crumble more than stone ({stone_r})"
        );
        assert_eq!(stone_r, FRACTURE_RADIUS_VOX, "stone is the reference material");
    }

    #[test]
    fn massless_material_never_fractures() {
        assert!(fracture_radius_vox(0.0, 1000.0, 1.0).is_none());
    }
}
