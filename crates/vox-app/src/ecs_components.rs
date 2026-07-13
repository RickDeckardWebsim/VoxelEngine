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

/// Serialize all alive entities and their known components to JSON.
/// Unknown component types (not registered here) are skipped.
pub fn save_scene(world: &ecs::World) -> serde_json::Value {
    let entities: Vec<serde_json::Value> = world.entities().map(|id| {
        let mut components = serde_json::Map::new();
        if let Some(t) = world.get::<Transform>(id) {
            components.insert("Transform".to_string(), serde_json::to_value(t).unwrap());
        }
        if let Some(v) = world.get::<Velocity>(id) {
            components.insert("Velocity".to_string(), serde_json::to_value(v).unwrap());
        }
        if let Some(n) = world.get::<Name>(id) {
            components.insert("Name".to_string(), serde_json::to_value(n).unwrap());
        }
        if let Some(p) = world.get::<Projectile>(id) {
            components.insert("Projectile".to_string(), serde_json::to_value(p).unwrap());
        }
        serde_json::json!({
            "slot": id.slot(),
            "components": components,
        })
    }).collect();
    serde_json::json!({ "entities": entities })
}

/// Deserialize entities from JSON and spawn them in the world.
/// Clears all existing entities first. Returns the number loaded.
pub fn load_scene(world: &mut ecs::World, data: &serde_json::Value) -> usize {
    // Clear existing entities.
    let existing: Vec<ecs::EntityId> = world.entities().collect();
    for id in existing {
        world.despawn(id);
    }
    let empty = Vec::new();
    let entities = data.get("entities").and_then(|e| e.as_array()).unwrap_or(&empty);
    let mut count = 0;
    for entity in entities {
        let id = world.spawn();
        if let Some(c) = entity.get("components") {
            if let Some(t) = c.get("Transform") {
                if let Ok(t) = serde_json::from_value::<Transform>(t.clone()) {
                    world.insert(id, t);
                }
            }
            if let Some(v) = c.get("Velocity") {
                if let Ok(v) = serde_json::from_value::<Velocity>(v.clone()) {
                    world.insert(id, v);
                }
            }
            if let Some(n) = c.get("Name") {
                if let Ok(n) = serde_json::from_value::<Name>(n.clone()) {
                    world.insert(id, n);
                }
            }
            if let Some(p) = c.get("Projectile") {
                if let Ok(p) = serde_json::from_value::<Projectile>(p.clone()) {
                    world.insert(id, p);
                }
            }
        }
        count += 1;
    }
    count
}

/// Save scene to a file. Returns Ok(()) on success.
pub fn save_scene_to_file(world: &ecs::World, path: &str) -> std::io::Result<()> {
    let data = save_scene(world);
    let json = serde_json::to_string_pretty(&data).unwrap();
    std::fs::write(path, json)
}

/// Load scene from a file. Returns the number of entities loaded.
pub fn load_scene_from_file(world: &mut ecs::World, path: &str) -> std::io::Result<usize> {
    let json = std::fs::read_to_string(path)?;
    let data: serde_json::Value = serde_json::from_str(&json).unwrap_or(serde_json::Value::Null);
    Ok(load_scene(world, &data))
}
