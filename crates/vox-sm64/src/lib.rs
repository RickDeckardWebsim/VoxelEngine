//! Safe wrapper around libsm64 — Super Mario 64 as a library.
//!
//! This crate provides a Rust-friendly interface to Mario's movement,
//! physics, and animation as extracted from the SM64 decompilation.
//! It bridges Mario's triangle-based collision system to the voxel
//! engine's grid-based world by converting voxel terrain top-faces
//! into `SM64Surface` triangles.
//!
//! ## Usage
//! ```no_run
//! use vox_sm64::{Sm64, MarioInputs};
//!
//! // Initialize with a SM64 US ROM (SHA1: 9bef1128717f958171a4afac3ed78ee2bb4e86ce)
//! let rom = std::fs::read("baserom.us.z64").unwrap();
//! let sm64 = Sm64::init(&rom).unwrap();
//!
//! // Load collision surfaces from your world
//! let surfaces: Vec<vox_sm64::ffi::SM64Surface> = vec![];
//! sm64.load_surfaces(&surfaces);
//!
//! // Spawn Mario
//! let mut mario = sm64.create_mario(0.0, 100.0, 0.0).unwrap();
//!
//! // Tick each frame
//! let state = mario.tick(MarioInputs {
//!     stick_x: 0.5,
//!     button_a: true,
//!     ..Default::default()
//! });
//! ```

pub mod ffi;
mod surface;

pub use surface::{voxel_surfaces_near, SurfaceProvider, SURFACE_RADIUS_M};

use ffi::*;

/// Error from libsm64 operations.
#[derive(Debug)]
pub enum Sm64Error {
    /// The ROM file could not be read.
    RomRead(std::io::Error),
    /// The ROM hash doesn't match the expected SM64 US ROM
    /// (SHA1: 9bef1128717f958171a4afac3ed78ee2bb4e86ce).
    InvalidRom,
    /// Mario could not be created at the given position (must be above
    /// a loaded surface).
    InvalidMarioPosition,
}

impl std::fmt::Display for Sm64Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RomRead(e) => write!(f, "failed to read ROM file: {e}"),
            Self::InvalidRom => write!(
                f,
                "invalid ROM: expected SM64 US (SHA1 9bef1128717f958171a4afac3ed78ee2bb4e86ce)"
            ),
            Self::InvalidMarioPosition => {
                write!(f, "Mario spawn position is not above a loaded surface")
            }
        }
    }
}

impl std::error::Error for Sm64Error {}

/// Expected SHA1 hash of the valid SM64 US ROM.
pub const ROM_SHA1: &str = "9bef1128717f958171a4afac3ed78ee2bb4e86ce";

/// SM64's internal coordinate scale: 1 meter = this many SM64 units.
/// SM64 uses a fixed-point system where the level geometry is in
/// integer coordinates. We scale voxel meters to SM64's integer space
/// so Mario's movement feels correct relative to the terrain.
pub const SM64_UNITS_PER_METER: f32 = 30.0;

/// Convert meters to SM64 integer units.
pub fn meters_to_sm64(m: f32) -> i32 {
    (m * SM64_UNITS_PER_METER) as i32
}

/// Convert SM64 units back to meters.
pub fn sm64_to_meters(u: f32) -> f32 {
    u / SM64_UNITS_PER_METER
}

/// Handle to the initialized libsm64 global state.
///
/// libsm64 uses a single global instance internally (it was designed
/// for embedding into an existing engine, not for multiple instances).
/// This wrapper ensures `sm64_global_init` is called once and
/// `sm64_global_terminate` is called on drop.
pub struct Sm64 {
    /// Mario's texture atlas (RGBA8, 704×64). Upload to GPU once.
    texture: Vec<u8>,
}

impl Sm64 {
    /// Initialize libsm64 with a SM64 US ROM. Extracts Mario's texture
    /// and animation data. Must be called once before any other API.
    pub fn init(rom: &[u8]) -> Result<Self, Sm64Error> {
        let texture_size = (SM64_TEXTURE_WIDTH * SM64_TEXTURE_HEIGHT * 4) as usize;
        let mut texture = vec![0u8; texture_size];

        // SAFETY: rom is a valid byte slice; texture is pre-allocated to
        // the exact size libsm64 expects (W*H*4 RGBA bytes).
        unsafe {
            sm64_global_init(rom.as_ptr(), texture.as_mut_ptr());
        }


        tracing::info!(
            texture_w = SM64_TEXTURE_WIDTH,
            texture_h = SM64_TEXTURE_HEIGHT,
            "libsm64 initialized"
        );

        Ok(Self { texture })
    }

    /// Mario's texture atlas as RGBA8 bytes (704 wide × 64 tall).
    /// Upload this to a GPU texture once; it doesn't change.
    pub fn texture_rgba(&self) -> &[u8] {
        &self.texture
    }

    /// Texture atlas dimensions.
    pub fn texture_dimensions(&self) -> (u32, u32) {
        (SM64_TEXTURE_WIDTH, SM64_TEXTURE_HEIGHT)
    }

    /// Load static collision surfaces into libsm64. These are the
    /// triangles Mario collides with — typically generated from your
    /// world's terrain. Call this whenever the surface set changes
    /// (e.g. Mario moved to a new area, or terrain was edited).
    pub fn load_surfaces(&self, surfaces: &[SM64Surface]) {
        // SAFETY: surfaces is a valid slice; libsm64 copies the data
        // internally so the slice doesn't need to outlive this call.
        unsafe {
            sm64_static_surfaces_load(surfaces.as_ptr(), surfaces.len() as u32);
        }
    }

    /// Create Mario at the given position (in SM64 units, not meters).
    /// The position must be above a loaded surface. Returns a Mario
    /// handle that can be ticked each frame.
    pub fn create_mario(&self, x: f32, y: f32, z: f32) -> Result<Mario, Sm64Error> {
        // SAFETY: libsm64 is initialized (we hold Sm64); coordinates
        // are plain floats.
        let id = unsafe { sm64_mario_create(x, y, z) };
        tracing::debug!(raw_id = id, x, y, z, "sm64_mario_create returned");
        if id < 0 {
            Err(Sm64Error::InvalidMarioPosition)
        } else {
            tracing::info!(mario_id = id, x, y, z, "Mario created");
            Ok(Mario::new(id))
        }
    }
}

impl Drop for Sm64 {
    fn drop(&mut self) {
        // SAFETY: we initialized libsm64 in Sm64::init; terminate on drop.
        unsafe {
            sm64_global_terminate();
        }
    }
}

/// A Mario instance. Tick once per frame to advance his simulation
/// and get back his animated state + mesh geometry.
pub struct Mario {
    id: i32,
    geometry: MarioGeometry,
}

impl Mario {
    fn new(id: i32) -> Self {
        Self {
            id,
            geometry: MarioGeometry::new(),
        }
    }

    /// Advance Mario's simulation by one frame. Returns his new state
    /// (position, velocity, action, animation). His geometry buffers
    /// are updated in-place and can be read via [`Mario::geometry`].
    ///
    /// Should be called at ~30 Hz (SM64's original tick rate) or
    /// higher. The caller is responsible for interpolating between
    /// ticks for smooth rendering.
    pub fn tick(&mut self, inputs: MarioInputs) -> MarioTickResult {
        let c_inputs = SM64MarioInputs::from(inputs);

        let mut state = SM64MarioState::default();
        let mut geo_buffers: SM64MarioGeometryBuffers = (&mut self.geometry).into();

        // SAFETY: mario_id is valid (we created it); all pointers point
        // to owned, correctly-sized buffers.
        // Set alpha=1.0 for the full tick — this tells gfx_adapter to
        // save the bone matrices for interpolation on subsequent frames.
        unsafe {
            gfx_adapter_set_interp_alpha(1.0);
            sm64_mario_tick(
                self.id,
                &c_inputs,
                &mut state,
                &mut geo_buffers,
            );
        }

        self.geometry.num_triangles = geo_buffers.numTrianglesUsed as usize;

        MarioTickResult {
            position: glam::Vec3::from_array(state.position),
            velocity: glam::Vec3::from_array(state.velocity),
            face_angle: state.faceAngle,
            forward_velocity: state.forwardVelocity,
            health: state.health,
            action: state.action,
            anim_id: state.animID,
            anim_frame: state.animFrame,
            flags: state.flags,
        }
    }

    /// Read-only access to Mario's current geometry (vertices for
    /// rendering). Updated by [`Mario::tick`] or [`Mario::render_interpolated`].
    pub fn geometry(&self) -> &MarioGeometry {
        &self.geometry
    }

    /// Re-evaluate Mario's geometry at an interpolated animation state,
    /// WITHOUT advancing the simulation. Call this on render frames
    /// between ticks with `alpha` < 1.0 to get smooth 60/120fps
    /// animation. The gfx_adapter blends bone matrices between the
    /// previous tick and current tick by `alpha`.
    ///
    /// Must be called after at least one [`Mario::tick`] (which saves
    /// the bone matrices). Use `alpha = 0.0` for the previous tick's
    /// pose, `alpha = 1.0` for the current tick's pose.
    pub fn render_interpolated(&mut self, alpha: f32) {
        let mut geo_buffers: SM64MarioGeometryBuffers = (&mut self.geometry).into();
        unsafe {
            gfx_adapter_set_interp_alpha(alpha);
            sm64_mario_render_geometry(self.id, &mut geo_buffers);
        }
        self.geometry.num_triangles = geo_buffers.numTrianglesUsed as usize;
    }

    /// Teleport Mario to a position (in SM64 units).
    pub fn set_position(&mut self, x: f32, y: f32, z: f32) {
        unsafe { sm64_set_mario_position(self.id, x, y, z) };
    }
}

impl Drop for Mario {
    fn drop(&mut self) {
        // SAFETY: mario_id is valid; libsm64 is still initialized.
        unsafe { sm64_mario_delete(self.id) };
    }
}

/// Inputs for one Mario tick frame.
#[derive(Copy, Clone, Debug, Default)]
pub struct MarioInputs {
    /// Camera look direction X (world-space, for camera-relative movement).
    pub cam_look_x: f32,
    /// Camera look direction Z (world-space).
    pub cam_look_z: f32,
    /// Analog stick X (-1.0..1.0).
    pub stick_x: f32,
    /// Analog stick Y (-1.0..1.0).
    pub stick_y: f32,
    /// A button (jump).
    pub button_a: bool,
    /// B button (punch/kick/dive).
    pub button_b: bool,
    /// Z button (crouch/ground pound/long jump).
    pub button_z: bool,
}

impl From<MarioInputs> for SM64MarioInputs {
    fn from(i: MarioInputs) -> Self {
        Self {
            camLookX: i.cam_look_x,
            camLookZ: i.cam_look_z,
            stickX: i.stick_x,
            stickY: i.stick_y,
            buttonA: i.button_a as u8,
            buttonB: i.button_b as u8,
            buttonZ: i.button_z as u8,
        }
    }
}

/// Mario's state after a tick.
#[derive(Copy, Clone, Debug)]
pub struct MarioTickResult {
    /// World-space position (in SM64 units — divide by
    /// [`SM64_UNITS_PER_METER`] for meters).
    pub position: glam::Vec3,
    /// World-space velocity (SM64 units/sec).
    pub velocity: glam::Vec3,
    /// Facing angle in degrees (0 = +Z, clockwise).
    pub face_angle: f32,
    /// Forward movement speed (SM64 units/sec).
    pub forward_velocity: f32,
    /// Health (0-8, 8 = full).
    pub health: i16,
    /// Current action bitmask (SM64 action enum).
    pub action: u32,
    /// Current animation ID.
    pub anim_id: i32,
    /// Current animation frame.
    pub anim_frame: i16,
    /// Status flags.
    pub flags: u32,
}

/// Mario's animated mesh geometry. Up to 1024 triangles (3072 vertices).
/// Each vertex has position, normal, color, and UV. Updated by
/// [`Mario::tick`]; read these to fill GPU vertex buffers for rendering.
pub struct MarioGeometry {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub num_triangles: usize,
}

impl MarioGeometry {
    fn new() -> Self {
        let max_verts = SM64_GEO_MAX_TRIANGLES as usize * 3;
        Self {
            positions: vec![[0.0; 3]; max_verts],
            normals: vec![[0.0; 3]; max_verts],
            colors: vec![[0.0; 3]; max_verts],
            uvs: vec![[0.0; 2]; max_verts],
            num_triangles: 0,
        }
    }

    /// Number of vertices currently in use (num_triangles * 3).
    pub fn num_vertices(&self) -> usize {
        self.num_triangles * 3
    }
}

impl From<&mut MarioGeometry> for SM64MarioGeometryBuffers {
    fn from(geo: &mut MarioGeometry) -> Self {
        Self {
            position: geo.positions.as_mut_ptr() as *mut f32,
            normal: geo.normals.as_mut_ptr() as *mut f32,
            color: geo.colors.as_mut_ptr() as *mut f32,
            uv: geo.uvs.as_mut_ptr() as *mut f32,
            numTrianglesUsed: (geo.positions.len() / 3) as u16,
        }
    }
}
