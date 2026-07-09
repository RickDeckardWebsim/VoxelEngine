//! Mario mode: toggleable SM64 character integration.
//!
//! Press `M` to toggle. On first activation, loads the SM64 ROM and
//! builds the render pipeline. Mario spawns at the player's position
//! with a third-person camera. Press `M` again to return to the
//! first-person walker.
//!
//! Mario's movement (running, jumping, wall-kicks, dives, etc.) is
//! simulated by libsm64. Voxel terrain is converted to collision
//! surfaces each time Mario moves far enough to need new geometry.

use std::collections::HashMap;
use std::path::Path;

use glam::{Quat, Vec3};
use vox_render::{MarioCameraUniform, MarioPipeline};
use vox_sm64::{
    aabb_box_surfaces, quat_to_sm64_euler, MarioInputs, Sm64, SurfaceObject, SurfaceProvider,
    SURFACE_RADIUS_M,
};
use vox_world::World;
use winit::keyboard::KeyCode;

/// Base third-person camera distance behind Mario, in meters.
/// Scaled by Mario's world-space height relative to the original 30 units/m.
const CAM_DISTANCE_BASE: f32 = 10.0;
/// Base third-person camera height above Mario, in meters.
const CAM_HEIGHT_BASE: f32 = 5.0;

/// State for Mario mode. Created on first toggle, reused on subsequent
/// toggles (so we don't reload the ROM every time).
pub struct MarioMode {
    sm64: Sm64,
    audio: Option<crate::audio::Sm64Audio>,
    mario: Option<vox_sm64::Mario>,
    pipeline: MarioPipeline,
    surfaces: SurfaceProvider,
    /// Camera yaw (radians), controlled by mouse while in Mario mode.
    pub cam_yaw: f32,
    /// Camera pitch (radians), clamped.
    pub cam_pitch: f32,
    /// 30 Hz accumulator for SM64 tick timing. SM64's simulation runs
    /// at 30 FPS internally; calling sm64_mario_tick every render frame
    /// makes Mario move 2x+ too fast. This accumulates real time and
    /// only ticks when enough has elapsed.
    tick_accumulator: f32,
    /// SM64's fixed tick rate (30 Hz).
    tick_rate: f32,
    /// Mario's last known position in SM64 units (for rendering between
    /// ticks and for model scaling).
    last_pos_sm64: [f32; 3],
    /// Model scale: shrinks Mario's mesh around his center.
    model_scale: f32,
    /// SM64 units per meter — controls Mario's size relative to the world.
    /// Higher = smaller Mario (160 SM64 units / units_per_meter = height in meters).
    /// Default 60 → Mario ~2.67m. Animations and physics are unaffected;
    /// only the world-space scale changes.
    units_per_meter: f32,
    /// Previous tick's Mario position (SM64 units) for position
    /// interpolation — translate the whole mesh by the delta,
    /// smooth 120 FPS movement without per-vertex interpolation.
    prev_tick_pos: [f32; 3],
    /// Previous tick's vertex positions (for interpolation between
    /// 30 Hz ticks). Same layout as MarioGeometry::positions.
    prev_positions: Vec<[f32; 3]>,
    /// Vertex count at the previous tick (for safe interpolation).
    prev_vertex_count: usize,
    /// Previous tick's Mario action — used to detect action transitions
    /// (e.g. ground pound → ground pound land) that should trigger
    /// engine-side effects like terrain carving.
    prev_action: u32,
    /// A ground-pound landing detected during the last tick, holding the
    /// impact position in meters. Drained by the caller via
    /// [`pending_ground_pound`].
    pending_ground_pound: Option<Vec3>,
    /// Fractional tick progress (0..1) for interpolation between 30 Hz
    /// ticks. At 120 FPS render, this cycles 0→1 four times per tick.
    pub tick_alpha: f32,
    /// libsm64 surface objects for nearby debris bodies, keyed by a
    /// stable hash of the body id. Created when debris enters
    /// [`vox_sm64::SURFACE_RADIUS_M`] of Mario, moved each tick to
    /// track the body, and dropped (→ deleted from libsm64) when the
    /// body leaves range or despawns. Lets Mario stand on / wall-slide
    /// / ride falling debris.
    debris_objects: HashMap<u64, vox_sm64::SurfaceObject>,
}

impl MarioMode {
    /// Initialize Mario mode: load the ROM, build the render pipeline.
    /// Call once (lazily, on first `M` press). The ROM path is looked
    /// up relative to the assets directory, then the working directory.
    pub fn init(
        gpu: &vox_render::Gpu,
        rom_path: &Path,
        mario_shader: &str,
        units_per_meter: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        tracing::info!(rom = %rom_path.display(), "loading SM64 ROM");

        let rom = std::fs::read(rom_path)
            .map_err(|e| format!("failed to read SM64 ROM at {}: {e}", rom_path.display()))?;

        let sm64 = Sm64::init(&rom)?;

        // Initialize libsm64's audio banks from the same ROM, then open the
        // cpal output stream + feeder thread. Audio is best-effort: if no
        // output device is available Mario mode still works silently.
        crate::audio::Sm64Audio::init(&rom);
        let audio = crate::audio::Sm64Audio::start();
        if audio.is_none() {
            tracing::warn!("SM64 audio disabled (no output device / stream)");
        }

        let (tex_w, tex_h) = sm64.texture_dimensions();
        let pipeline = MarioPipeline::new(
            gpu,
            mario_shader,
            sm64.texture_rgba(),
            tex_w,
            tex_h,
        );

        tracing::info!("Mario mode initialized (ROM loaded, pipeline built)");
        Ok(Self {
        sm64,
        audio,
        mario: None,
        pipeline,
        surfaces: SurfaceProvider::new(units_per_meter),
        cam_yaw: 0.0,
        cam_pitch: 0.2,
        tick_accumulator: 0.0,
        model_scale: 1.0,
        units_per_meter,
        prev_tick_pos: [0.0; 3],
        prev_positions: vec![[0.0; 3]; vox_sm64::ffi::SM64_GEO_MAX_TRIANGLES as usize * 3],
        prev_vertex_count: 0,
        tick_rate: 30.0,
        last_pos_sm64: [0.0; 3],
        tick_alpha: 0.0,
        prev_action: 0,
        pending_ground_pound: None,
        debris_objects: HashMap::new(),
        })
    }

    /// Spawn Mario at the given position (in meters). Returns an error
    /// if the position isn't above a surface — but since we load
    /// surfaces around the spawn point first, this should succeed.
    pub fn spawn(&mut self, pos_m: Vec3, world: &World) -> Result<(), vox_sm64::Sm64Error> {
        let pos_sm64 = pos_m * self.units_per_meter;
        let surfaces = vox_sm64::voxel_surfaces_near(world, pos_m, vox_sm64::SURFACE_RADIUS_M, self.units_per_meter);
        tracing::info!(
            surfaces = surfaces.len(),
            pos_m = ?pos_m,
            pos_sm64 = ?pos_sm64,
            "initial surface load for Mario spawn"
        );
        self.sm64.load_surfaces(&surfaces);

        // Spawn Mario at the given position. The caller already adds
        // a small offset above the player's feet.
        tracing::info!(
            spawn_x = pos_sm64.x,
            spawn_y = pos_sm64.y,
            spawn_z = pos_sm64.z,
            "attempting sm64_mario_create"
        );
        let mario = self.sm64.create_mario(pos_sm64.x, pos_sm64.y, pos_sm64.z)?;
        self.last_pos_sm64 = [pos_sm64.x, pos_sm64.y, pos_sm64.z];
        self.prev_tick_pos = [pos_sm64.x, pos_sm64.y, pos_sm64.z];
        self.tick_accumulator = 0.0;
        self.mario = Some(mario);
        Ok(())
    }

    /// Despawn Mario (return to FPS mode). The SM64 state and pipeline
    /// are kept alive for the next toggle. Debris surface objects are
    /// dropped (→ deleted from libsm64) so they don't linger while
    /// Mario is inactive or reference stale body ids on respawn.
    pub fn despawn(&mut self) {
        self.mario = None;
        let n = self.debris_objects.len();
        self.debris_objects.clear();
        if n > 0 {
            tracing::debug!(cleared = n, "cleared debris surface objects on despawn");
        }
        tracing::info!("Mario despawned");
    }

    /// True if Mario is currently active (spawned).
    pub fn is_active(&self) -> bool {
        self.mario.is_some()
    }

    /// Tick Mario's simulation and update surfaces. Call once per frame
    /// while Mario mode is active. Returns Mario's world-space position
    /// in meters (for the camera).
    ///
    /// SM64 runs at 30 Hz internally. We accumulate real frame time
    /// and only call sm64_mario_tick when enough has elapsed, so
    /// Mario's speed is correct regardless of render FPS.
    pub fn tick(&mut self, world: &World, input: &vox_platform::InputState, dt: f32) -> Vec3 {
        if self.mario.is_none() {
            return Vec3::ZERO;
        }
        // Accumulate time and tick SM64 at 30 Hz. Clamp dt to prevent
        // a huge frame delta (e.g. after world gen) from causing
        // hundreds of ticks in one frame.
        self.tick_accumulator += dt.min(0.1);
        let tick_dt = 1.0 / self.tick_rate;

        // Compute inputs once (same for all ticks this frame)
        let stick = self.movement_stick(input);
        let inputs = MarioInputs {
            cam_look_x: self.cam_look_x(),
            cam_look_z: self.cam_look_z(),
            stick_x: stick.x,
            stick_y: stick.y,
            button_a: input.key_down(KeyCode::Space),
            button_b: input.key_down(KeyCode::KeyJ) || input.key_down(KeyCode::KeyB),
            button_z: input.key_down(KeyCode::ShiftLeft) || input.key_down(KeyCode::KeyK),
        };

        let mut ticks_this_frame = 0;
        while self.tick_accumulator >= tick_dt && ticks_this_frame < 3 {
            self.tick_accumulator -= tick_dt;
            ticks_this_frame += 1;
            // Save previous state for position interpolation
            self.prev_tick_pos = self.last_pos_sm64;
            let geo = self.mario.as_ref().unwrap().geometry();
            let n = geo.num_vertices();
            self.prev_vertex_count = n;
            self.prev_positions[..n].copy_from_slice(&geo.positions[..n]);
            // Now tick
            let state = self.mario.as_mut().unwrap().tick(inputs);
            self.last_pos_sm64 = [state.position.x, state.position.y, state.position.z];
            // Detect a ground-pound landing: the action transitions from
            // the airborne pound into its landing action on the tick Mario
            // impacts the ground. Record the impact point (in meters) so
            // the frame loop can carve a crater there.
            if self.prev_action == vox_sm64::ACT_GROUND_POUND
                && state.action == vox_sm64::ACT_GROUND_POUND_LAND
            {
                let pos_m = state.position / self.units_per_meter;
                self.pending_ground_pound = Some(pos_m);
            }
            self.prev_action = state.action;
        }
        // Compute tick_alpha BEFORE render_interpolated so it uses the
        // correct fractional progress for this frame.
        self.tick_alpha = (self.tick_accumulator / tick_dt).clamp(0.0, 1.0);

        // On frames where no tick happened AND we've ticked at least once,
        // re-evaluate Mario's geometry at the interpolated animation state.
        // Can't do this before the first tick — the graph node system
        // needs initialization from sm64_mario_tick first.
        if false && ticks_this_frame == 0 && self.mario.is_some() && self.tick_alpha < 1.0 {
            self.mario.as_mut().unwrap().render_interpolated(self.tick_alpha);
        }

        // Interpolate Mario's position between previous and current tick
        // by tick_alpha. The mesh renders at the current tick position
        // (30 Hz, no ghosting), but the camera tracks this smooth
        // interpolated position (120 FPS) — no shake.
        let interp_sm64 = [
            self.prev_tick_pos[0] + (self.last_pos_sm64[0] - self.prev_tick_pos[0]) * self.tick_alpha,
            self.prev_tick_pos[1] + (self.last_pos_sm64[1] - self.prev_tick_pos[1]) * self.tick_alpha,
            self.prev_tick_pos[2] + (self.last_pos_sm64[2] - self.prev_tick_pos[2]) * self.tick_alpha,
        ];
        let pos_m = Vec3::from(interp_sm64) / self.units_per_meter;

        // Stream collision surfaces if Mario moved enough
        if self.surfaces.update(pos_m, world) {
            self.sm64.load_surfaces(self.surfaces.surfaces());
        }

        pos_m
    }

    /// If a ground pound landed during the most recent tick, return the
    /// impact position (in meters) and clear the pending flag. The frame
    /// loop calls this each frame to decide whether to carve a crater.
    pub fn pending_ground_pound(&mut self) -> Option<Vec3> {
        self.pending_ground_pound.take()
    }
    /// Mario's current world-space position in meters (derived from the
 /// last tick's SM64 position). Used by the frame loop to decide which
 /// debris bodies are near enough to feed to [`update_debris`].
 pub fn mario_pos_m(&self) -> Vec3 {
     Vec3::from(self.last_pos_sm64) / self.units_per_meter
 }

    /// Synchronize libsm64 surface objects for nearby debris bodies.
    ///
    /// Each tuple is `(body_key, pos_m, rot, grid_min_m, grid_max_m)`
    /// where `grid_min_m`/`grid_max_m` are the body's **local** box
    /// extents in meters (relative to the COM), i.e. `grid_offset` and
    /// `grid_offset + grid.dims * voxel_size`. We build a 12-triangle
    /// AABB box in local SM64 units and register it as a surface object
    /// whose transform is `(pos_m, rot)` — libsm64 rotates + translates
    /// the local box into world space each update, so the collision box
    /// tracks the body exactly.
    ///
    /// Bodies already tracked get [`SurfaceObject::move_to`]; bodies no
    /// longer in `bodies` (out of range or despawned) are dropped, which
    /// deletes them from libsm64. Call once per frame after [`tick`].
    pub fn update_debris(
        &mut self,
        bodies: impl Iterator<Item = (u64, Vec3, Quat, Vec3, Vec3)>,
    ) {
        let upm = self.units_per_meter;
        let mario_pos = self.mario_pos_m();
        // Margin so a body entering the radius doesn't pop its surface
        // object in and out at the boundary each frame.
        let radius = SURFACE_RADIUS_M + 2.0;

        let mut seen: Vec<u64> = Vec::new();
        for (key, pos_m, rot, grid_min_m, grid_max_m) in bodies {
            // Quick AABB-vs-point distance reject: if the body's world
            // AABB (pos + local box) is farther than `radius` from
            // Mario on any axis, skip. The caller already pre-filters,
            // but this keeps us correct if it doesn't.
            let wmin = pos_m + grid_min_m.min(grid_max_m);
            let wmax = pos_m + grid_max_m.max(grid_min_m);
            let d = Vec3::new(
                (mario_pos.x - wmin.x).max(0.0).max(wmax.x - mario_pos.x),
                (mario_pos.y - wmin.y).max(0.0).max(wmax.y - mario_pos.y),
                (mario_pos.z - wmin.z).max(0.0).max(wmax.z - mario_pos.z),
            );
            if d.length_squared() > radius * radius {
                continue;
            }
            seen.push(key);

            // Local box in SM64 integer units.
            let lo = grid_min_m.min(grid_max_m) * upm;
            let hi = grid_max_m.max(grid_min_m) * upm;
            let min_i = [lo.x as i32, lo.y as i32, lo.z as i32];
            let max_i = [hi.x as i32, hi.y as i32, hi.z as i32];

            let transform = vox_sm64::ffi::SM64ObjectTransform {
                position: [pos_m.x * upm, pos_m.y * upm, pos_m.z * upm],
                eulerRotation: quat_to_sm64_euler(rot),
            };

            if let Some(obj) = self.debris_objects.get(&key) {
                obj.move_to(&transform);
            } else {
                let surfaces = aabb_box_surfaces(min_i, max_i);
                match SurfaceObject::create(&surfaces, transform) {
                    Ok(obj) => {
                        tracing::debug!(key, surfaces = surfaces.len(), "debris surface object created");
                        self.debris_objects.insert(key, obj);
                    }
                    Err(e) => {
                        tracing::warn!(key, error = %e, "failed to create debris surface object");
                    }
                }
            }
        }

        // Drop surface objects for bodies no longer present.
        let stale: Vec<u64> = self
            .debris_objects
            .keys()
            .filter(|k| !seen.contains(k))
            .copied()
            .collect();
        for k in stale {
            self.debris_objects.remove(&k);
        }
    }
    /// Apply mouse look to the third-person camera.
    pub fn look(&mut self, delta: glam::Vec2) {
        let sensitivity = 0.0025;
        self.cam_yaw = (self.cam_yaw - delta.x * sensitivity) % std::f32::consts::TAU;
        let limit = std::f32::consts::FRAC_PI_2 - 0.1;
        self.cam_pitch = (self.cam_pitch - delta.y * sensitivity).clamp(-limit, limit);
    }

    /// Camera position (third-person, behind and above Mario).
    /// The camera sits at Mario's position minus the look direction
    /// (scaled by distance), plus a small fixed height offset for a
    /// comfortable over-the-shoulder view.
    pub fn camera_pos(&self, mario_pos_m: Vec3) -> Vec3 {
        let dir = self.camera_look_dir();
        // Scale camera distance/height by Mario's world-space size.
        // At 30 units/m (original), Mario is 5.3m and cam is 10m/5m.
        // At 60 units/m, Mario is 2.67m and cam scales to 5m/2.5m.
        let scale = 30.0 / self.units_per_meter;
        mario_pos_m - dir * (CAM_DISTANCE_BASE * scale) + Vec3::new(0.0, CAM_HEIGHT_BASE * scale * 0.5, 0.0)
    }

    /// Camera look direction (from camera toward Mario). Matches the
    /// engine's `Camera::forward()` convention: `(-sy*cp, sp, -cy*cp)`.
    pub fn camera_look_dir(&self) -> Vec3 {
        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        Vec3::new(-sy * cp, sp, -cy * cp)
    }

    /// SM64 camera look inputs. SM64 computes `yaw = atan2s(camLookZ, camLookX)`,
    /// so we pass the direction the camera *faces* in SM64's convention:
    /// X = sin(yaw), Z = cos(yaw). This is the opposite of the engine's
    /// `Camera::forward()` which negates both.
    fn cam_look_x(&self) -> f32 {
        self.cam_yaw.sin()
    }
    fn cam_look_z(&self) -> f32 {
        self.cam_yaw.cos()
    }

    /// Map WASD to an analog stick vector (camera-relative).
    /// Full stick deflection (1.0) — Mario runs at native SM64 speed.
    /// At 1m voxel scale, Mario's ~5m height and ~5 m/s run speed
    /// feel natural relative to the terrain.
    fn movement_stick(&self, input: &vox_platform::InputState) -> glam::Vec2 {
        const STICK_SCALE: f32 = 1.0;
        let mut x: f32 = 0.0;
        let mut y: f32 = 0.0;
        if input.key_down(KeyCode::KeyW) {
            y += 1.0;
        }
        if input.key_down(KeyCode::KeyS) {
            y -= 1.0;
        }
        if input.key_down(KeyCode::KeyA) {
            x -= 1.0;
        }
        if input.key_down(KeyCode::KeyD) {
            x += 1.0;
        }
        let mag = (x * x + y * y).sqrt();
        if mag > 1.0 {
            x /= mag;
            y /= mag;
        }
        glam::Vec2::new(-x * STICK_SCALE, y * STICK_SCALE)
    }

    /// Update the Mario pipeline's camera uniform and draw Mario's mesh.
    pub fn render<'p>(
        &'p self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'p>,
        view_proj: [[f32; 4]; 4],
        cam_pos: Vec3,
        sun_dir: Vec3,
        fog_start: f32,
        fog_end: f32,
    ) {
        // Interpolated Mario position to match the camera target. The
        // mesh vertices are authored at the current tick position
        // (last_pos_sm64); the shader translates them by
        // (interp_pos - tick_pos) so we can reuse the vertex buffer
        // across interpolated render frames without re-uploading.
        let a = self.tick_alpha;
        let interp_pos = [
            self.prev_tick_pos[0] + (self.last_pos_sm64[0] - self.prev_tick_pos[0]) * a,
            self.prev_tick_pos[1] + (self.last_pos_sm64[1] - self.prev_tick_pos[1]) * a,
            self.prev_tick_pos[2] + (self.last_pos_sm64[2] - self.prev_tick_pos[2]) * a,
        ];
        let cam = MarioCameraUniform {
            view_proj,
            cam_pos: [cam_pos.x, cam_pos.y, cam_pos.z, 1.0],
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
            fog: [fog_start, fog_end, self.units_per_meter, self.model_scale],
            interp_pos: [interp_pos[0], interp_pos[1], interp_pos[2], 0.0],
            tick_pos: [
                self.last_pos_sm64[0],
                self.last_pos_sm64[1],
                self.last_pos_sm64[2],
                0.0,
            ],
        };
        self.pipeline.update_camera(queue, &cam);

        if let Some(mario) = &self.mario {
            self.pipeline.draw(queue, pass, mario.geometry());
        }
    }

    /// Locate the SM64 ROM file. Looks in:
    /// 1. A `Super Mario 64 (USA)/` subdirectory (relative to assets)
    /// 2. The working directory
    pub fn find_rom(assets_dir: &Path) -> Option<std::path::PathBuf> {
        // Check common locations
        let candidates = [
            assets_dir.join("Super Mario 64 (USA)/Super Mario 64 (USA).z64"),
            assets_dir.join("baserom.us.z64"),
            Path::new("Super Mario 64 (USA)/Super Mario 64 (USA).z64").to_path_buf(),
            Path::new("baserom.us.z64").to_path_buf(),
        ];
        for candidate in &candidates {
            if candidate.exists() {
                return Some(candidate.clone());
            }
        }
        None
    }
}

impl Drop for MarioMode {
    fn drop(&mut self) {
        // CRITICAL ordering: stop the audio feeder before `sm64` (and the
        // other fields) drop. Sm64::drop calls sm64_global_terminate(),
        // which frees the C audio state the feeder thread is still ticking
        // against — a use-after-free if the feeder outlives it. Taking
        // `audio` here joins the feeder, so the subsequent field-drop of
        // Sm64Audio (now None) is a no-op.
        if let Some(audio) = self.audio.take() {
            drop(audio);
        }
    }
}
