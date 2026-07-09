//! FFI bindings to libsm64 — Super Mario 64's movement/physics/rendering
//! extracted from the decompilation project as a C library.
//!
//! All structs are `#[repr(C)]` matching `libsm64.h` exactly. The safe
//! wrapper layer in [`crate::Sm64`] and [`crate::Mario`] handles lifetime
//! and ownership semantics.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::os::raw::c_char;

// ── Constants ──────────────────────────────────────────────────────────

pub const SM64_TEXTURE_WIDTH: u32 = 64 * 11;
pub const SM64_TEXTURE_HEIGHT: u32 = 64;
pub const SM64_GEO_MAX_TRIANGLES: u32 = 1024;

// ── Mario action constants (from decomp/include/sm64.h) ───────────────
// Action flags (masks): the high bits of an action encode its category.
/// Set while Mario is airborne (jump/fall/etc.).
pub const ACT_FLAG_AIR: u32 = 0x00000800;
/// Set for attacking actions (ground pound, dive, punch, etc.).
pub const ACT_FLAG_ATTACKING: u32 = 0x00800000;

// Specific actions we react to from the engine.
/// Ground pound (mid-air, after crouch+jump).
pub const ACT_GROUND_POUND: u32 = 0x008008A9;
/// Ground pound landing — the single tick where Mario impacts the ground.
pub const ACT_GROUND_POUND_LAND: u32 = 0x0080023C;
/// Triple jump (the highest of the three jump tiers).
pub const ACT_TRIPLE_JUMP: u32 = 0x01000882;

// ── Structs (match libsm64.h layout exactly) ───────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SM64Surface {
    pub type_: i16,
    pub force: i16,
    pub terrain: u16,
    pub vertices: [[i32; 3]; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct SM64MarioInputs {
    pub camLookX: f32,
    pub camLookZ: f32,
    pub stickX: f32,
    pub stickY: f32,
    pub buttonA: u8,
    pub buttonB: u8,
    pub buttonZ: u8,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct SM64ObjectTransform {
    pub position: [f32; 3],
    pub eulerRotation: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SM64SurfaceObject {
    pub transform: SM64ObjectTransform,
    pub surfaceCount: u32,
    pub surfaces: *mut SM64Surface,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct SM64MarioState {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub faceAngle: f32,
    pub forwardVelocity: f32,
    pub health: i16,
    pub action: u32,
    pub animID: i32,
    pub animFrame: i16,
    pub flags: u32,
    pub particleFlags: u32,
    pub invincTimer: i16,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SM64MarioGeometryBuffers {
    pub position: *mut f32,
    pub normal: *mut f32,
    pub color: *mut f32,
    pub uv: *mut f32,
    pub numTrianglesUsed: u16,
}

// ── Function pointers for callbacks ────────────────────────────────────

pub type SM64DebugPrintFunctionPtr = Option<extern "C" fn(*const c_char)>;
pub type SM64PlaySoundFunctionPtr = Option<extern "C" fn(u32, *mut f32)>;

// ── extern "C" function declarations ───────────────────────────────────

unsafe extern "C" {
    pub fn sm64_register_debug_print_function(
        debugPrintFunction: SM64DebugPrintFunctionPtr,
    );
    pub fn sm64_register_play_sound_function(
        playSoundFunction: SM64PlaySoundFunctionPtr,
    );

    pub fn sm64_global_init(rom: *const u8, outTexture: *mut u8);
    pub fn sm64_global_terminate();

    pub fn sm64_audio_init(rom: *const u8);
    pub fn sm64_audio_tick(
        numQueuedSamples: u32,
        numDesiredSamples: u32,
        audio_buffer: *mut i16,
    ) -> u32;

    pub fn sm64_static_surfaces_load(
        surfaceArray: *const SM64Surface,
        numSurfaces: u32,
    );

    pub fn sm64_mario_create(x: f32, y: f32, z: f32) -> i32;
    pub fn sm64_mario_tick(
        marioId: i32,
        inputs: *const SM64MarioInputs,
        outState: *mut SM64MarioState,
        outBuffers: *mut SM64MarioGeometryBuffers,
    );
    pub fn sm64_mario_delete(marioId: i32);

    pub fn sm64_surface_object_create(
        surfaceObject: *const SM64SurfaceObject,
    ) -> u32;
    pub fn sm64_surface_object_move(
        objectId: u32,
        transform: *const SM64ObjectTransform,
    );
    pub fn sm64_surface_object_delete(objectId: u32);

    pub fn sm64_surface_find_wall_collision(
        xPtr: *mut f32,
        yPtr: *mut f32,
        zPtr: *mut f32,
        offsetY: f32,
        radius: f32,
    ) -> i32;
    pub fn sm64_surface_find_floor_height(x: f32, y: f32, z: f32) -> f32;

    pub fn sm64_set_mario_action(marioId: i32, action: u32);
    pub fn sm64_set_mario_position(marioId: i32, x: f32, y: f32, z: f32);
    pub fn sm64_set_mario_velocity(marioId: i32, x: f32, y: f32, z: f32);
    pub fn sm64_set_mario_faceangle(marioId: i32, y: f32);
    pub fn gfx_adapter_set_interp_alpha(alpha: f32);
    pub fn sm64_mario_render_geometry(marioId: i32, outBuffers: *mut SM64MarioGeometryBuffers);
}
