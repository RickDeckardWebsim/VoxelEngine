//! System trait and engine context for the ECS frame loop.
//!
//! Each system is a struct that owns its state and implements `System`.
//! The frame loop constructs `EngineCtx` per-phase and calls each system's
//! `update` in order. Built-in systems run first; modder systems append.

use glam::Vec3;
use vox_core::MaterialRegistry;
use vox_ecs as ecs;
use vox_physics::{ImpactEvent, PhysicsWorld};
use vox_render::{Gpu, VoxelPipeline};
use vox_sim::FluidSim;
use vox_world::World;

use crate::chunk_loader::ChunkLoader;

/// Cross-system data carried between phases. Each system reads/writes
/// what it needs; the frame loop resets per-frame fields at the start.
pub struct FrameData {
    pub dt: f32,
    pub physics_steps: u32,
    pub impacts: Vec<ImpactEvent>,
    pub player_pos: Vec3,
    pub player_vel: Vec3,
    pub grabbed: bool,
}

impl Default for FrameData {
    fn default() -> Self {
        Self {
            dt: 0.0,
            physics_steps: 0,
            impacts: Vec::new(),
            player_pos: Vec3::ZERO,
            player_vel: Vec3::ZERO,
            grabbed: false,
        }
    }
}

/// Per-phase engine context. Constructed fresh for each frame phase with
/// only the references that phase needs. Systems receive `&mut EngineCtx`
/// and can call any engine method.
pub struct EngineCtx<'a> {
    pub world: &'a mut World,
    pub phys: &'a mut PhysicsWorld,
    pub fluid: &'a mut FluidSim,
    pub registry: &'a MaterialRegistry,
    pub chunk_loader: &'a mut ChunkLoader,
    pub pipeline: &'a mut VoxelPipeline,
    pub gpu: &'a Gpu,
    pub frame: &'a mut FrameData,
}

/// Render-phase context. Separate from `EngineCtx` because render systems
/// need GPU encoder/pipeline access but not world/physics mutation.
pub struct RenderCtx<'a> {
    pub gpu: &'a Gpu,
    pub pipeline: &'a mut VoxelPipeline,
    pub frame: &'a FrameData,
}

/// A system: owns its state, runs once per frame phase.
///
/// Built-in systems (Input, Player, Physics, Fluid, Streaming, Remesh,
/// Render, etc.) implement this trait. Modder systems append after
/// built-ins and run in registration order.
pub trait System: Send {
    /// Human-readable name for debugging.
    fn name(&self) -> &str;

    /// Update phase: called once per frame in system order. Receives
    fn update(&mut self, ctx: &mut EngineCtx, ecs: &mut ecs::World);

    /// Render phase: called after all update phases. Optional — most
    /// systems don't render directly (the RenderSystem handles that).
    fn render(&mut self, _ctx: &mut RenderCtx, _ecs: &ecs::World) {}

    /// Serialize this system's state for scene save. Return `None` for
    /// stateless systems or systems that don't participate in scenes.
    fn save(&self) -> Option<serde_json::Value> {
        None
    }

    /// Deserialize and restore this system's state from a scene load.
    fn load(&mut self, _data: serde_json::Value) {}
}
