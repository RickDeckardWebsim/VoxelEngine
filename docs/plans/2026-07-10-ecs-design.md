# Entity-Component System — Design

**Date:** 2026-07-10
**Status:** Approved (all 4 sections)
**Approach:** C — Incremental (system extraction first, then ECS, then migration)

## Summary

Add a full ECS framework to the voxel engine via three incremental phases:
1. Extract the 835-line `frame()` into ordered system calls with clear boundaries
2. Add ECS entity registry + component storage alongside extracted systems
3. Migrate existing systems into ECS components one at a time

Each phase is independently shippable. The 249-test suite gates every step.

## Phase 1: System Extraction

Split `VoxApp::frame()` into ~11 ordered systems matching the natural
boundaries in the current code:

1. InputSystem — key/mouse, tool selection, quality toggle
2. PlayerSystem — player/mario/replay update + tool application
3. PhysicsSystem — phys.step, impact fracture, debris budget, clutter expiry
4. FluidSystem — fluid tick, weathering, fire, fire→particles
5. StreamingSystem — chunk_loader.update
6. RemeshSystem — dirty regions, wake physics/fluid/fire, remesh dispatch/collect
7. ParticleSystem — particle update + body colliders
8. DayNightSystem — game time, sun direction
9. CameraSystem — camera update, write uniforms, frustum
10. ShadowSystem — shadow camera update
11. RenderSystem — frame begin, HUD, debug overlay, render, postprocess, present

### EngineCtx (per-phase bridge)

Constructed fresh for each frame phase, dropped between phases. Mirrors
the current scoped-block borrow patterns exactly.

```rust
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
```

### FrameData (cross-system communication)

```rust
pub struct FrameData {
    pub dt: f32,
    pub impacts: Vec<ImpactEvent>,
    pub dirty_regions: Vec<DirtyRegion>,
    pub player_pos: Vec3,
    pub camera: Camera,
}
```

### Systems own their state

Each system is a struct holding its own state:
- PhysicsSystem: stateless (reads/writes via ctx.phys)
- DebrisBudgetSystem: owns debris_order
- FireSystem: owns burning_bodies
- ReplaySystem: owns replay state
- DayNightSystem: owns game_time, always_day
- etc.

## Phase 2: ECS Entity Registry

New `vox-ecs` crate:

```rust
pub struct EntityId(u32, u32); // index + generation

pub struct World {
    entities: Vec<Option<u32>>,
    free: Vec<usize>,
    components: FxHashMap<TypeId, Box<dyn Any>>,
}
```

- spawn() → EntityId
- despawn(id) → frees slot, bumps generation
- insert<T: Component>(id, T)
- get<T: Component>(id) → Option<&T>
- query<T>() → impl Iterator<(EntityId, &T)>

Component trait: blanket impl for any 'static + Send + Sync type.

## Phase 3: System Trait + Scheduler

```rust
pub trait System: Send + Sync {
    fn name(&self) -> &str;
    fn update(&mut self, ctx: &mut EngineCtx, ecs: &mut ecs::World);
    fn render(&mut self, ctx: &mut RenderCtx, ecs: &ecs::World) {}
    fn save(&self) -> Option<serde_json::Value> { None }
    fn load(&mut self, _data: serde_json::Value) {}
}
```

Frame loop: ordered list of systems. Built-ins registered first, modder
systems appended.

## Phase 4: Scene Editor + Serialization

Scene file (JSON):
```json
{
  "entities": [
    { "id": 0, "components": {
      "Transform": { "pos": [1,2,3], "rot": [0,0,0,1] },
      "Door": { "open": false, "speed": 1.0 }
    }}
  ]
}
```

F3 overlay integration:
- Entity list panel (spawn/delete)
- Component inspector (inline egui editing)
- Save/Load scene buttons
- Click-to-select in 3D

Each component type registers serialize/deserialize fns. Components
without serde are skipped with a warning.

## Components to Build

1. `vox-ecs` crate — EntityId, World, Component trait, storage, queries
2. `EngineCtx` + `FrameData` — per-phase bridge struct
3. `System` trait — update/render/save/load hooks
4. System extraction — split frame() into 11 system structs
5. Scene serialization — JSON save/load
6. Debug overlay editor — entity list, component inspector, save/load
