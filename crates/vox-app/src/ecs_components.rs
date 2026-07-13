//! Built-in ECS components and the ECS tick system.
//!
//! These are the first components and the simplest possible system —
//! proving the ECS works end-to-end. Modder-defined components follow
//! the same pattern: any `'static + Send + Sync` type is a component.

use glam::Vec3;
use serde::{Deserialize, Serialize};
use vox_ecs as ecs;

/// World-space position + facing. The most common component.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transform {
    pub pos: Vec3,
    pub rot: glam::Quat,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            pos: Vec3::ZERO,
            rot: glam::Quat::IDENTITY,
        }
    }
}

/// Linear velocity in m/s. Entities with Transform + Velocity are
/// moved by the ECS tick each frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Velocity {
    pub vel: Vec3,
}

/// A named entity — shows in the editor's entity list.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Name(pub String);

/// A projectile: despawns after `lifetime` seconds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Projectile {
    pub lifetime: f32,
}

/// Tick all entities with Transform + Velocity: move them by `vel * dt`.
/// Projectiles have their lifetime decremented; expired ones are despawned.
pub fn tick_ecs(world: &mut ecs::World, dt: f32) {
    // Collect entities to update (can't mutate while iterating).
    let updates: Vec<(ecs::EntityId, Vec3)> = world
        .query::<Velocity>()
        .filter(|(id, _)| world.get::<Transform>(*id).is_some())
        .map(|(id, v)| (id, v.vel * dt))
        .collect();

    for (id, delta) in updates {
        if let Some(t) = world.get_mut::<Transform>(id) {
            t.pos += delta;
        }
    }

    // Tick projectile lifetimes and despawn expired ones.
    let expired: Vec<ecs::EntityId> = {
        let mut expired = Vec::new();
        for (id, p) in world.query_mut::<Projectile>() {
            p.lifetime -= dt;
            if p.lifetime <= 0.0 {
                expired.push(id);
            }
        }
        expired
    };
    for id in expired {
        world.despawn(id);
    }
}

/// Spawn a projectile at `pos` with `vel` m/s, living for `lifetime` seconds.
pub fn spawn_projectile(world: &mut ecs::World, pos: Vec3, vel: Vec3, lifetime: f32) -> ecs::EntityId {
    let id = world.spawn();
    world.insert(id, Transform { pos, rot: glam::Quat::IDENTITY });
    world.insert(id, Velocity { vel });
    world.insert(id, Projectile { lifetime });
    world.insert(id, Name("Projectile".to_string()));
    id
}
