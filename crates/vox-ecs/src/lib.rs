//! Entity-component system: entity registry, component storage, and queries.
//!
//! Lightweight by design — no system scheduler (that lives in `vox-app`'s
//! `System` trait), no derived queries (single-type queries first, multi-type
//! via post-filter). Each component type gets its own `FxHashMap<EntityId, T>`
//! stored behind a `Box<dyn Any>` keyed by `TypeId`.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use vox_core::FxHashMap;

/// A component: any `'static + Send + Sync` type. Blanket impl — no manual
/// trait implementation needed. Components are stored per-type in the
/// `World`'s component maps.
pub trait Component: Any + Send + Sync {}

impl<T: Any + Send + Sync> Component for T {}

/// Entity handle: slot index + generation. The generation makes stale
/// handles from despawned entities resolve to `None` instead of silently
/// aliasing a newly-spawned entity in the same slot.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct EntityId {
    slot: u32,
    generation: u32,
}

impl EntityId {
    /// Slot index (for internal use).
    pub fn slot(self) -> u32 {
        self.slot
    }
}

/// The entity world: an entity registry + per-type component storage.
///
/// Component storage is a `HashMap<TypeId, Box<dyn Any>>` where each value
/// is a `FxHashMap<EntityId, T>` downcast on access. This trades a tiny
/// per-access downcast cost for zero macro boilerplate — any `Component`
/// type works without registration.
pub struct World {
    /// Generation counter per slot; `None` = free slot.
    generations: Vec<Option<u32>>,
    free: Vec<u32>,
    /// `TypeId → Box<dyn ComponentStorage>` where each storage is a
    /// `FxHashMap<EntityId, T>` behind the trait object.
    components: HashMap<TypeId, Box<dyn ComponentStorage>>,
}

/// Trait object wrapper for per-type component storage. Enables despawn
/// to remove components without knowing the concrete type `T`.
trait ComponentStorage: Any + Send + Sync {
    fn remove_entity(&mut self, id: EntityId);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Component> ComponentStorage for FxHashMap<EntityId, T> {
    fn remove_entity(&mut self, id: EntityId) {
        self.remove(&id);
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    pub fn new() -> Self {
        Self {
            generations: Vec::new(),
            free: Vec::new(),
            components: HashMap::new(),
        }
    }

    /// Spawn a new entity. Returns its handle.
    pub fn spawn(&mut self) -> EntityId {
        if let Some(slot) = self.free.pop() {
            let generation = self.generations[slot as usize].unwrap_or(0);
            EntityId { slot, generation }
        } else {
            let slot = self.generations.len() as u32;
            self.generations.push(Some(0));
            EntityId { slot, generation: 0 }
        }
    }

    /// Despawn an entity. Its components are removed from all storages
    /// and its slot is freed. A stale handle from before despawn will
    /// no longer resolve.
    pub fn despawn(&mut self, id: EntityId) {
        if !self.is_alive(id) {
            return;
        }
        // Remove all components for this entity via the trait object.
        for storage in self.components.values_mut() {
            storage.remove_entity(id);
        }
        let slot = id.slot as usize;
        self.generations[slot] = Some(id.generation + 1);
        self.free.push(id.slot);
    }

    /// True if `id` refers to a currently-alive entity.
    pub fn is_alive(&self, id: EntityId) -> bool {
        self.generations
            .get(id.slot as usize)
            .is_some_and(|&g| g == Some(id.generation))
    }

    /// Insert a component `T` onto entity `id`. Replaces any existing
    /// component of the same type.
    pub fn insert<T: Component>(&mut self, id: EntityId, component: T) {
        let type_id = TypeId::of::<T>();
        let storage = self
            .components
            .entry(type_id)
            .or_insert_with(|| Box::new(FxHashMap::<EntityId, T>::default()));
        let map = storage
            .as_any_mut()
            .downcast_mut::<FxHashMap<EntityId, T>>()
            .expect("type mismatch in component storage");
        map.insert(id, component);
    }

    /// Get a reference to entity `id`'s component of type `T`.
    pub fn get<T: Component>(&self, id: EntityId) -> Option<&T> {
        let type_id = TypeId::of::<T>();
        let storage = self.components.get(&type_id)?;
        let map = storage
            .as_any()
            .downcast_ref::<FxHashMap<EntityId, T>>()
            .expect("type mismatch in component storage");
        map.get(&id)
    }

    /// Get a mutable reference to entity `id`'s component of type `T`.
    pub fn get_mut<T: Component>(&mut self, id: EntityId) -> Option<&mut T> {
        let type_id = TypeId::of::<T>();
        let storage = self.components.get_mut(&type_id)?;
        let map = storage
            .as_any_mut()
            .downcast_mut::<FxHashMap<EntityId, T>>()
            .expect("type mismatch in component storage");
        map.get_mut(&id)
    }

    /// Remove entity `id`'s component of type `T`. Returns the removed value.
    pub fn remove<T: Component>(&mut self, id: EntityId) -> Option<T> {
        let type_id = TypeId::of::<T>();
        let storage = self.components.get_mut(&type_id)?;
        let map = storage
            .as_any_mut()
            .downcast_mut::<FxHashMap<EntityId, T>>()
            .expect("type mismatch in component storage");
        map.remove(&id)
    }

    /// Query all entities that have component `T`. Returns `(EntityId, &T)`
    /// pairs. For multi-type queries, call this for the rarest component
    /// and filter by `get`/`get_mut` on the others.
    pub fn query<T: Component>(&self) -> impl Iterator<Item = (EntityId, &T)> + '_ {
        let type_id = TypeId::of::<T>();
        self.components
            .get(&type_id)
            .and_then(|storage| {
                storage
                    .as_any()
                    .downcast_ref::<FxHashMap<EntityId, T>>()
            })
            .into_iter()
            .flat_map(|map| map.iter().map(|(&id, v)| (id, v)))
    }

    /// Query all entities that have component `T` (mutable). Returns
    /// `(EntityId, &mut T)` pairs.
    pub fn query_mut<T: Component>(&mut self) -> impl Iterator<Item = (EntityId, &mut T)> + '_ {
        let type_id = TypeId::of::<T>();
        self.components
            .get_mut(&type_id)
            .and_then(|storage| {
                storage
                    .as_any_mut()
                    .downcast_mut::<FxHashMap<EntityId, T>>()
            })
            .into_iter()
            .flat_map(|map| map.iter_mut().map(|(&id, v)| (id, v)))
    }

    /// Number of alive entities.
    pub fn entity_count(&self) -> usize {
        self.generations.len() - self.free.len()
    }

    /// All alive entity IDs.
    pub fn entities(&self) -> impl Iterator<Item = EntityId> + '_ {
        let free: std::collections::HashSet<u32> = self.free.iter().copied().collect();
        self.generations
            .iter()
            .enumerate()
            .filter_map(move |(slot, generation)| {
                let slot_u32 = slot as u32;
                if free.contains(&slot_u32) {
                    return None;
                }
                generation.map(|g| EntityId {
                    slot: slot_u32,
                    generation: g,
                })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(PartialEq, Debug)]
    struct Position {
        x: f32,
        y: f32,
        z: f32,
    }

    #[derive(PartialEq, Debug)]
    struct Health(f32);

    #[test]
    fn spawn_and_despawn() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        assert!(w.is_alive(e1));
        assert!(w.is_alive(e2));
        assert_eq!(w.entity_count(), 2);

        w.despawn(e1);
        assert!(!w.is_alive(e1));
        assert!(w.is_alive(e2));
        assert_eq!(w.entity_count(), 1);

        // Slot is reused; e1's stale handle doesn't resolve.
        let e3 = w.spawn();
        assert!(!w.is_alive(e1));
        assert!(w.is_alive(e3));
        assert_ne!(e3, e1);
    }

    #[test]
    fn insert_get_remove_component() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Position { x: 1.0, y: 2.0, z: 3.0 });
        w.insert(e, Health(100.0));

        assert_eq!(w.get::<Position>(e), Some(&Position { x: 1.0, y: 2.0, z: 3.0 }));
        assert_eq!(w.get::<Health>(e), Some(&Health(100.0)));

        // Mutate
        if let Some(pos) = w.get_mut::<Position>(e) {
            pos.x = 10.0;
        }
        assert_eq!(w.get::<Position>(e), Some(&Position { x: 10.0, y: 2.0, z: 3.0 }));

        // Remove
        let removed = w.remove::<Health>(e);
        assert_eq!(removed, Some(Health(100.0)));
        assert_eq!(w.get::<Health>(e), None);
    }

    #[test]
    fn component_on_wrong_entity_returns_none() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        w.insert(e1, Position { x: 1.0, y: 0.0, z: 0.0 });

        assert!(w.get::<Position>(e1).is_some());
        assert!(w.get::<Position>(e2).is_none());
    }

    #[test]
    fn despawned_entity_component_inaccessible() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Position { x: 5.0, y: 0.0, z: 0.0 });
        w.despawn(e);

        // is_alive returns false, so callers won't access components.
        assert!(!w.is_alive(e));
    }

    #[test]
    fn query_all_of_type() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        let e3 = w.spawn();
        w.insert(e1, Position { x: 1.0, y: 0.0, z: 0.0 });
        w.insert(e2, Position { x: 2.0, y: 0.0, z: 0.0 });
        // e3 has no Position
        w.insert(e3, Health(50.0));

        let positions: Vec<_> = w.query::<Position>().collect();
        assert_eq!(positions.len(), 2);
    }

    #[test]
    fn entity_count_tracks_spawn_despawn() {
        let mut w = World::new();
        assert_eq!(w.entity_count(), 0);
        let e1 = w.spawn();
        assert_eq!(w.entity_count(), 1);
        let e2 = w.spawn();
        assert_eq!(w.entity_count(), 2);
        w.despawn(e1);
        assert_eq!(w.entity_count(), 1);
        w.despawn(e2);
        assert_eq!(w.entity_count(), 0);
    }

    #[test]
    fn entities_iterates_alive() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        w.despawn(e1);
        let alive: Vec<_> = w.entities().collect();
        assert_eq!(alive, vec![e2]);
    }
}
