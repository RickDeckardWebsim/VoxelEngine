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
pub use ffi::{
    ACT_FLAG_AIR, ACT_FLAG_ATTACKING, ACT_GROUND_POUND, ACT_GROUND_POUND_LAND,
    ACT_TRIPLE_JUMP,
};

use ffi::*;
use sha1::{Sha1, Digest};

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
        // Validate the ROM hash before passing it to libsm64 — a wrong
        // ROM would crash the C code or produce garbage. This lets us
        // fail gracefully with a clear error message instead.
        let hash = Sha1::digest(rom);
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        if hex != ROM_SHA1 {
            tracing::error!(
                actual = %hex,
                expected = %ROM_SHA1,
                "ROM SHA1 mismatch"
            );
            return Err(Sm64Error::InvalidRom);
        }

        let texture_size = (SM64_TEXTURE_WIDTH * SM64_TEXTURE_HEIGHT * 4) as usize;
        let mut texture = vec![0u8; texture_size];

        // SAFETY: rom is a validated SM64 US ROM byte slice; texture is
        // pre-allocated to the exact size libsm64 expects (W*H*4 RGBA bytes).
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
        self.geometry.version = self.geometry.version.wrapping_add(1);

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
    }

    /// Teleport Mario to a position (in SM64 units).
    pub fn set_position(&mut self, x: f32, y: f32, z: f32) {
        unsafe { sm64_set_mario_position(self.id, x, y, z) };
    }

    /// Directly set Mario's velocity (in SM64 units/sec). Useful for
    /// applying external impulses (e.g. a ground-pound launch) without
    /// going through the normal input/action system.
    pub fn set_velocity(&mut self, x: f32, y: f32, z: f32) {
        unsafe { sm64_set_mario_velocity(self.id, x, y, z) };
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
    /// Monotonic version stamp, incremented every time the geometry
    /// buffers are refreshed by [`Mario::tick`] or
    /// [`Mario::render_interpolated`]. Renderers compare it against
    /// the last uploaded version to skip redundant `write_buffer`
    /// calls when Mario is idle and the mesh is unchanged frame-to-frame.
    pub version: u64,
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
            version: 0,
        }
    }

    /// Number of vertices currently in use (num_triangles * 3).
    pub fn num_vertices(&self) -> usize {
        self.num_triangles * 3
    }

    /// Current geometry version stamp (see [`MarioGeometry::version`]).
    pub fn version(&self) -> u64 {
        self.version
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

// ── Surface objects (dynamic collision) ───────────────────────────────

/// SM64 surface type for a default solid surface (matches surface.rs).
const SURFACE_DEFAULT: i16 = 0x0000;
/// Terrain type: grass (affects footsteps, not collision).
const SURFACE_TERRAIN_GRASS: u16 = 0x0000;

/// Owned handle to a libsm64 *surface object*: a set of triangles that
/// can move and rotate each frame. Mario collides with these exactly
/// like static terrain — he can stand on them, wall-slide, and ride
/// moving platforms (libsm64 tracks the platform under Mario and
/// carries him with it via `surfaces_object_get_transform_ptr`).
///
/// The triangles are stored in **local** coordinates and baked into
/// world space each call to [`SurfaceObject::move_to`] using the
/// supplied transform (libsm64's `mtxf_rotate_zxy_and_translate`).
/// Drop deletes the object from libsm64 — no leaks.
pub struct SurfaceObject {
    id: u32,
    /// Owned local surfaces. libsm64 copies these on create, but we
    /// keep them so the `*mut SM64Surface` we hand it stays valid in
    /// case a future libsm64 build stops copying. The pointer is only
    /// read inside `sm64_surface_object_create`.
    _surfaces: Vec<SM64Surface>,
}

impl SurfaceObject {
    /// Create a surface object from local-space triangles + an initial
    /// transform. The surfaces are copied by libsm64 internally
    /// (`surfaces_load_object` does a `memcpy`), so the slice may be
    /// dropped after this call — but this handle keeps its own copy.
    ///
    /// `transform.position` is in **SM64 units** (meters ×
    /// [`SM64_UNITS_PER_METER`]); `transform.eulerRotation` is in
    /// **degrees** (pitch, yaw, roll).
    pub fn create(
        surfaces: &[SM64Surface],
        transform: SM64ObjectTransform,
    ) -> Result<Self, Sm64Error> {
        let owned = surfaces.to_vec();
        let obj = SM64SurfaceObject {
            transform,
            surfaceCount: owned.len() as u32,
            surfaces: owned.as_ptr() as *mut SM64Surface,
        };
        // SAFETY: libsm64 is initialized (caller holds Sm64); `obj`
        // points at a valid SM64SurfaceObject whose `surfaces` pointer
        // references `owned`, alive for this entire call.
        let id = unsafe { sm64_surface_object_create(&obj) };
        tracing::debug!(surface_object_id = id, count = owned.len(), "surface object created");
        Ok(Self { id, _surfaces: owned })
    }

    /// Update the surface object's transform. libsm64 rebakes the local
    /// triangles into world space using the new position + rotation.
    pub fn move_to(&self, transform: &SM64ObjectTransform) {
        // SAFETY: `id` is a valid, live surface object (we created it
        // and haven't dropped it).
        unsafe { sm64_surface_object_move(self.id, transform) };
    }

    /// Raw libsm64 object id (for diagnostics).
    pub fn id(&self) -> u32 {
        self.id
    }
}

impl Drop for SurfaceObject {
    fn drop(&mut self) {
        // SAFETY: `id` is valid and live until this drop. libsm64 also
        // clears Mario's platform reference if he was riding this
        // object (see sm64_surface_object_delete in libsm64.c).
        unsafe { sm64_surface_object_delete(self.id) };
        tracing::debug!(surface_object_id = self.id, "surface object deleted");
    }
}

/// Convert a glam quaternion to SM64 euler rotation in **degrees** as
/// `[pitch, yaw, roll]` — the layout `SM64ObjectTransform.eulerRotation`
/// expects.
///
/// **Order:** `mtxf_rotate_zxy_and_translate` builds the world matrix
/// `R = Ry(yaw) · Rx(pitch) · Rz(roll)` (verified by expanding the
/// matrix entries it writes — despite the function's name, the *acting*
/// rotation is Y·X·Z). glam's `EulerRot::YXZ` decomposes a quaternion
/// into exactly that product and returns `(yaw, pitch, roll)`, so we
/// use it directly. (Using `ZXY` would give `Rz·Rx·Ry` — wrong order;
/// single-axis tests can't tell them apart since a one-axis rotation
/// is identical under any ordering — only a compound yaw+pitch case
/// distinguishes them, which is why `compound_rotation_round_trips`
/// exists below.)
///
/// **Sign:** libsm64 converts each input degree value through
/// `CONVERT_ANGLE(x) = -x/180*32768` before feeding it to `sins`/`coss`
/// (plain `sin`/`cos` table lookups, math_util.h:28-29). So
/// `sins(CONVERT_ANGLE(θ°)) = sin(-θ°) = -sin(θ°)`. Tracing the matrix
/// for a pure yaw `y` shows it computes `Ry(-y)`, not `Ry(+y)`. A glam
/// `Quat::from_rotation_y(+y)` encodes `Ry(+y)`, so to make libsm64
/// reproduce the glam rotation we **negate** every angle: feeding `-θ`
/// makes `CONVERT_ANGLE(-θ) = +θ` internally. Identity → `[0, 0, 0]`
/// (negation of zero is zero, which is why the identity test alone
/// cannot catch the sign bug).
pub fn quat_to_sm64_euler(rot: glam::Quat) -> [f32; 3] {
    let (yaw, pitch, roll) = rot.to_euler(glam::EulerRot::YXZ);
    let rad_to_deg = 180.0 / std::f32::consts::PI;
    [-pitch * rad_to_deg, -yaw * rad_to_deg, -roll * rad_to_deg]
}

/// Build 12 triangles (6 faces × 2) for an axis-aligned box in **local**
/// coordinates, given min/max corners already in SM64 units. The box is
/// centered at the transform origin; the transform's rotation orients
/// it in world space.
///
/// Winding is counter-clockwise when viewed from outside each face, so
/// the surface normals point outward — matching how `surface.rs` emits
/// terrain faces and what libsm64's collision code expects.
pub fn aabb_box_surfaces(min: [i32; 3], max: [i32; 3]) -> Vec<SM64Surface> {
    let [x0, y0, z0] = min;
    let [x1, y1, z1] = max;
    let mut s = Vec::with_capacity(12);
    // Helper using the crate's surface fields directly.
    let mut tri = |a: [i32; 3], b: [i32; 3], c: [i32; 3]| {
        s.push(SM64Surface {
            type_: SURFACE_DEFAULT,
            force: 0,
            terrain: SURFACE_TERRAIN_GRASS,
            vertices: [a, b, c],
        });
    };
    // +X face (normal +X): viewed from +X looking toward -X,
    // CCW is (y0,z0)->(y0,z1)->(y1,z1) etc. Use the standard box
    // winding: for each +normal face, list corners CCW from outside.
    // -X
    tri([x0, y0, z1], [x0, y1, z1], [x0, y1, z0]);
    tri([x0, y0, z1], [x0, y1, z0], [x0, y0, z0]);
    // +X
    tri([x1, y0, z0], [x1, y1, z0], [x1, y1, z1]);
    tri([x1, y0, z0], [x1, y1, z1], [x1, y0, z1]);
    // -Y (floor below; normal -Y)
    tri([x0, y0, z0], [x1, y0, z0], [x1, y0, z1]);
    tri([x0, y0, z0], [x1, y0, z1], [x0, y0, z1]);
    // +Y (floor on top; normal +Y) — Mario stands here.
    tri([x0, y1, z1], [x1, y1, z1], [x1, y1, z0]);
    tri([x0, y1, z1], [x1, y1, z0], [x0, y1, z0]);
    // -Z
    tri([x0, y0, z0], [x0, y1, z0], [x1, y1, z0]);
    tri([x0, y0, z0], [x1, y1, z0], [x1, y0, z0]);
    // +Z
    tri([x0, y0, z1], [x1, y0, z1], [x1, y1, z1]);
    tri([x0, y0, z1], [x1, y1, z1], [x0, y1, z1]);
    s
}

#[cfg(test)]
mod surface_object_tests {
    use super::*;

    #[test]
    fn identity_quat_is_zero_euler() {
        let e = quat_to_sm64_euler(glam::Quat::IDENTITY);
        assert!(e[0].abs() < 1e-4, "pitch = {}", e[0]);
        assert!(e[1].abs() < 1e-4, "yaw = {}", e[1]);
        assert!(e[2].abs() < 1e-4, "roll = {}", e[2]);
    }

    #[test]
    fn aabb_box_has_twelve_surfaces() {
        let s = aabb_box_surfaces([0, 0, 0], [10, 20, 30]);
        assert_eq!(s.len(), 12);
        // Every vertex lies within the box.
        for surf in &s {
            for v in &surf.vertices {
                assert!(v[0] >= 0 && v[0] <= 10);
                assert!(v[1] >= 0 && v[1] <= 20);
                assert!(v[2] >= 0 && v[2] <= 30);
            }
        }
    }

    #[test]
    fn yaw_quat_decomposes_to_negated_yaw() {
        // glam::Quat::from_rotation_y(+90°) encodes Ry(+90°). libsm64's
        // CONVERT_ANGLE negates, so to reproduce Ry(+90°) we must feed
        // it -90° → eulerRotation[1] = -90. pitch/roll stay 0.
        // This is the test the identity case can't catch: negation of
        // zero is zero.
        let q = glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let e = quat_to_sm64_euler(q);
        assert!(e[0].abs() < 1e-3, "pitch = {} (want ~0)", e[0]);
        assert!((e[1] + 90.0).abs() < 1e-3, "yaw = {} (want ~-90)", e[1]);
        assert!(e[2].abs() < 1e-3, "roll = {} (want ~0)", e[2]);
    }

    #[test]
    fn pitch_quat_decomposes_to_negated_pitch() {
        // 90° about X → pitch must be -90 (negated). At exactly 90° the
        // ZXY decomposition hits the asin gimbal edge, so allow a small
        // absolute tolerance rather than the 1e-3 used elsewhere.
        let q = glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
        let e = quat_to_sm64_euler(q);
        assert!((e[0] + 90.0).abs() < 0.05, "pitch = {} (want ~-90)", e[0]);
        assert!(e[1].abs() < 1e-3, "yaw = {} (want ~0)", e[1]);
        assert!(e[2].abs() < 1e-3, "roll = {} (want ~0)", e[2]);
    }

    /// The decisive test for axis **ordering**: a compound yaw+pitch
    /// rotation decomposes differently under `ZXY` vs `YXZ`, so only
    /// the correct order round-trips. We de-negate the output (undo the
    /// sign fix), reconstruct `Ry(yaw)·Rx(pitch)·Rz(roll)` (the product
    /// `mtxf_rotate_zxy_and_translate` builds), and assert it equals the
    /// original quaternion's matrix. Fails with `EulerRot::ZXY`, passes
    /// with `EulerRot::YXZ`.
    #[test]
    fn compound_rotation_round_trips() {
        let q = glam::Quat::from_rotation_y(0.5) * glam::Quat::from_rotation_x(0.3);
        let e = quat_to_sm64_euler(q);
        let deg_to_rad = std::f32::consts::PI / 180.0;
        // De-negate: the output is [-pitch, -yaw, -roll] degrees.
        let pitch = -e[0] * deg_to_rad;
        let yaw = -e[1] * deg_to_rad;
        let roll = -e[2] * deg_to_rad;
        let reconstructed = glam::Mat3::from_rotation_y(yaw)
            * glam::Mat3::from_rotation_x(pitch)
            * glam::Mat3::from_rotation_z(roll);
        let expected = glam::Mat3::from_quat(q);
        // Element-wise compare with tolerance.
        for i in 0..3 {
            for j in 0..3 {
                let diff = (reconstructed.col(i)[j] - expected.col(i)[j]).abs();
                assert!(diff < 1e-4, "mat[{j}][{i}]: recon={} expected={} diff={}", reconstructed.col(i)[j], expected.col(i)[j], diff);
            }
        }
    }
}
