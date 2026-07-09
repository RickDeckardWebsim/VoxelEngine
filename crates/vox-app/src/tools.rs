//! Player tools, selected from a 1-9 hotbar: normal single-voxel dig,
//! scalable-radius dig, an explosive bomb, a long-range death laser, and
//! placing water. Placing (right-click) is independent of the active tool.

use glam::{IVec3, Vec3};
use vox_core::consts::REACH;
use vox_core::{MaterialRegistry, voxel_center_m};
use vox_physics::{Aabb, BodyId, PhysicsWorld};
use vox_sim::FluidSim;
use vox_world::{AIR, Voxel, World, raycast};

/// The selectable hotbar tools. Slots 6-9 are reserved (not yet assigned to
/// a tool); selecting one of them leaves the previously active tool in
/// place.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Tool {
    /// Slot 1: break exactly the one voxel under the crosshair.
    Dig,
    /// Slot 2: carve a sphere of adjustable radius (see [`Tools::radius_m`]).
    /// No blast impulse -- severed material just falls.
    ScalableDig,
    /// Slot 3: carve a sphere of adjustable radius and give the debris an
    /// outward impulse from the blast center.
    Bomb,
    /// Slot 4: an effectively infinite-range beam that tunnels straight
    /// through everything in its path in one shot -- no impulse, just an
    /// instant, precise cut.
    DeathLaser,
    /// Slot 5: place a sphere of water at the crosshair target using its own
    /// adjustable source radius.
    PlaceWater,
}

/// Hotbar key (1-9) to tool mapping. Slots past the array select nothing.
pub const HOTBAR: [(u8, Tool); 5] = [
    (1, Tool::Dig),
    (2, Tool::ScalableDig),
    (3, Tool::Bomb),
    (4, Tool::DeathLaser),
    (5, Tool::PlaceWater),
];

/// Radius bounds for [`Tool::ScalableDig`] and [`Tool::Bomb`], adjustable
/// via `[`/`]` or the mouse wheel while one of those tools is active, in
/// meters.
const TOOL_RADIUS_MIN: f32 = 0.5;
const TOOL_RADIUS_MAX: f32 = 4.0;
/// Per-keypress / per-wheel-notch radius step, in meters.
const TOOL_RADIUS_STEP: f32 = 0.25;
/// Water starts as a modest source rather than inheriting Bomb's 1.5 m
/// radius. At the default 0.1 m voxel scale this is a radius-five blob,
/// small enough to flow as a pool without immediately becoming a huge
/// budget-limited avalanche.
const WATER_RADIUS_DEFAULT: f32 = 0.5;
/// Beam range for [`Tool::DeathLaser`], in meters -- deliberately far beyond
/// any reasonable world size ("infinite reach"); [`vox_physics::laser`]
/// clamps its own search box to the world's actual bounds, so this never
/// costs more than the world it's cutting through.
const DEATH_LASER_RANGE_M: f32 = 10_000.0;
/// Beam radius for [`Tool::DeathLaser`], in meters. Not adjustable -- the
/// laser's whole point is an instant, maximal cut, not a tunable one.
const DEATH_LASER_RADIUS_M: f32 = 1.5;

/// Result of a tool use that might carve an existing body: ids newly
/// spawned (need a mesh uploaded -- an id with no uploaded mesh is
/// simulated but never drawn) and ids removed (need their old mesh
/// dropped). A carved body is always despawned and replaced, even when it
/// splits into exactly one still-whole-looking fragment -- there's no
/// partial update, only despawn-and-respawn. Empty on both sides when the
/// tool hit nothing, or hit a body but removed nothing from it.
#[derive(Default)]
pub struct CarveOutcome {
    pub spawned: Vec<BodyId>,
    pub removed: Vec<BodyId>,
    /// World-space point the tool actually struck, if it struck anything --
    /// where destruction feedback (dust/spark particles) should originate.
    pub impact_m: Option<Vec3>,
    /// The material at the struck voxel *before* it was carved (AIR when
    /// nothing was hit) -- lets feedback particles take on the color of the
    /// material actually being destroyed.
    pub impact_material: Voxel,
}

/// What a scene raycast hit: a specific static-world voxel, or an existing
/// body plus the exact grid-local voxel hit on it (needed for an exact
/// single-voxel dig; see [`carve_body_voxel_at`](vox_physics::carve_body_voxel_at)).
enum SceneHit {
    World(IVec3),
    Body(BodyId, IVec3),
}

/// Raycast against both the static world and every live body, returning
/// whichever is closer along with the world-space hit point. Debris bodies
/// are typically few and small, so a linear scan per click is cheap --
/// this only runs once per tool use, not per frame.
fn raycast_scene(
    world: &World,
    phys: &PhysicsWorld,
    eye_m: Vec3,
    look: Vec3,
    max_dist_m: f32,
) -> Option<(SceneHit, Vec3)> {
    let dir = look.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }

    let mut best_dist = max_dist_m;
    let mut best: Option<SceneHit> = None;
    if let Some(hit) = raycast(world, eye_m, dir, max_dist_m) {
        best_dist = hit.dist_m;
        best = Some(SceneHit::World(hit.voxel));
    }
    for (id, body) in phys.iter() {
        let voxel_size_m = body.half_voxel * 2.0;
        let inv_rot = body.rot.inverse();
        let local_origin = inv_rot * (eye_m - body.pos) - body.grid_offset;
        let local_dir = inv_rot * dir;
        let Some(hit) =
            vox_physics::raycast_grid(&body.grid, local_origin, local_dir, best_dist, voxel_size_m)
        else {
            continue;
        };
        if hit.dist_m < best_dist {
            best_dist = hit.dist_m;
            best = Some(SceneHit::Body(id, hit.voxel));
        }
    }
    // Nudge slightly past the entry point: `best_dist` lands exactly on the
    // hit surface, which is fine for any real carve radius but not for a
    // radius as tiny as Dig's single-voxel carve -- floating-point rounding
    // can put a boundary-exact point a hair on the empty side of the face,
    // missing the voxel entirely. A tenth of a millimeter is far below any
    // real voxel scale in this engine (0.1 m at the finest) but comfortably
    // clears float imprecision.
    const SURFACE_NUDGE_M: f32 = 1e-4;
    best.map(|t| (t, eye_m + dir * (best_dist + SURFACE_NUDGE_M)))
}

/// The material a scene hit is about to destroy -- read *before* carving
/// (afterwards the voxel is air, or the body is despawned entirely).
fn hit_material(world: &World, phys: &PhysicsWorld, hit: &SceneHit) -> Voxel {
    match *hit {
        SceneHit::World(voxel) => world.get_voxel(voxel),
        SceneHit::Body(id, local_voxel) => phys
            .get(id)
            .map(|b| b.grid.get(local_voxel))
            .unwrap_or(AIR),
    }
}

/// Build a [`CarveOutcome`] from one of `vox_physics::carve_body_*_at`'s
/// results: if `id` still exists afterward, nothing was actually removed
/// and the (untouched) body needs no mesh update at all -- the default,
/// empty outcome. Otherwise it was despawned (replaced by 0+ fragments).
fn body_outcome(phys: &PhysicsWorld, id: BodyId, spawned: Vec<BodyId>) -> CarveOutcome {
    if phys.get(id).is_some() {
        return CarveOutcome::default();
    }
    CarveOutcome {
        spawned,
        removed: vec![id],
        ..CarveOutcome::default()
    }
}

/// Carve a sphere out of body `id` and report the outcome (see
/// [`body_outcome`]).
fn carve_body_sphere(
    phys: &mut PhysicsWorld,
    registry: &MaterialRegistry,
    id: BodyId,
    center_world_m: Vec3,
    radius_m: f32,
) -> CarveOutcome {
    let spawned = vox_physics::carve_body_sphere_at(phys, registry, id, center_world_m, radius_m);
    body_outcome(phys, id, spawned)
}

/// Tool state: active tool, selected build material, and independently
/// adjustable destructive-tool and water-source radii.
pub struct Tools {
    pub tool: Tool,
    pub radius_m: f32,
    water_radius_m: f32,
    /// Index into the registry (skips air).
    material_index: usize,
    material_count: usize,
}

impl Tools {
    pub fn new(registry: &MaterialRegistry) -> Self {
        Self {
            tool: Tool::Dig,
            radius_m: vox_core::consts::BLAST_RADIUS,
            water_radius_m: WATER_RADIUS_DEFAULT,
            material_index: 1,
            material_count: registry.len(),
        }
    }

    /// True for tools whose affected-area radius is adjustable ([`Tool::ScalableDig`],
    /// [`Tool::Bomb`]) -- used to decide whether the mouse wheel resizes the
    /// tool or cycles build material.
    pub fn has_adjustable_radius(&self) -> bool {
        matches!(self.tool, Tool::ScalableDig | Tool::Bomb | Tool::PlaceWater)
    }

    /// Radius currently controlled by the wheel, bracket keys, HUD, and
    /// debug overlay. Water intentionally keeps its own source radius so a
    /// prior large bomb cannot turn it into an enormous voxel avalanche.
    pub fn active_radius_m(&self) -> f32 {
        match self.tool {
            Tool::PlaceWater => self.water_radius_m,
            _ => self.radius_m,
        }
    }

    /// Mutable counterpart to [`Tools::active_radius_m`].
    pub fn active_radius_mut(&mut self) -> &mut f32 {
        match self.tool {
            Tool::PlaceWater => &mut self.water_radius_m,
            _ => &mut self.radius_m,
        }
    }

    /// Select a hotbar slot (1-9) by key number. Slots not in [`HOTBAR`] do
    /// nothing. Returns the newly active tool if the slot changed it.
    pub fn select_hotbar_slot(&mut self, slot: u8) -> Option<Tool> {
        let tool = HOTBAR.iter().find(|(s, _)| *s == slot)?.1;
        self.tool = tool;
        Some(tool)
    }

    /// Shrink the tool radius by one step, clamped to [`TOOL_RADIUS_MIN`].
    pub fn shrink_radius(&mut self) {
        let radius = self.active_radius_mut();
        *radius = (*radius - TOOL_RADIUS_STEP).max(TOOL_RADIUS_MIN);
    }

    /// Grow the tool radius by one step, clamped to [`TOOL_RADIUS_MAX`].
    pub fn grow_radius(&mut self) {
        let radius = self.active_radius_mut();
        *radius = (*radius + TOOL_RADIUS_STEP).min(TOOL_RADIUS_MAX);
    }

    /// Adjust the tool radius by `steps` notches (e.g. mouse wheel delta),
    /// clamped to [`TOOL_RADIUS_MIN`]/[`TOOL_RADIUS_MAX`].
    pub fn adjust_radius(&mut self, steps: i32) {
        let radius = self.active_radius_mut();
        *radius = (*radius + steps as f32 * TOOL_RADIUS_STEP).clamp(TOOL_RADIUS_MIN, TOOL_RADIUS_MAX);
    }

    /// Currently selected build material.
    pub fn material(&self) -> Voxel {
        Voxel(self.material_index as u16)
    }

    /// Set the selected build material directly by registry id (1..N,
    /// skipping air). Out-of-range ids are ignored. Used by the debug
    /// overlay's material picker.
    pub fn set_material_index(&mut self, index: usize) {
        if index >= 1 && index < self.material_count {
            self.material_index = index;
        }
    }

    /// Current material's registry id (1..N, skipping air).
    pub fn material_index(&self) -> usize {
        self.material_index
    }

    /// Cycle the build material by `steps` (mouse wheel), skipping air.
    pub fn cycle_material(&mut self, steps: i32, registry: &MaterialRegistry) {
        let n = self.material_count as i32 - 1; // excluding air
        if n <= 0 {
            return;
        }
        let cur = self.material_index as i32 - 1;
        let next = (cur + steps).rem_euclid(n) + 1;
        self.material_index = next as usize;
        if let Some(def) = registry.get(vox_core::MaterialId(next as u16)) {
            tracing::info!(material = %def.name, "selected build material");
        }
    }

    /// Dig: break exactly the one voxel or body-voxel under the crosshair,
    /// then detach whatever that leaves unsupported (world target) or
    /// whatever the body splits into (body target). See [`CarveOutcome`].
    pub fn dig(
        &self,
        world: &mut World,
        phys: &mut PhysicsWorld,
        registry: &MaterialRegistry,
        eye_m: Vec3,
        look: Vec3,
    ) -> CarveOutcome {
        let Some((hit, hit_point_m)) = raycast_scene(world, phys, eye_m, look, REACH) else {
            return CarveOutcome::default();
        };
        let material = hit_material(world, phys, &hit);
        let mut outcome = match hit {
            SceneHit::World(voxel) => {
                world.set_voxel(voxel, AIR);
                CarveOutcome {
                    spawned: vox_physics::detach_unsupported(world, phys, registry, &[voxel]),
                    ..CarveOutcome::default()
                }
            }
            SceneHit::Body(id, local_voxel) => {
                let spawned = vox_physics::carve_body_voxel_at(phys, registry, id, local_voxel);
                body_outcome(phys, id, spawned)
            }
        };
        outcome.impact_m = Some(hit_point_m);
        outcome.impact_material = material;
        outcome
    }

    /// Blast the crosshair target: carve a jagged explosion shape (a base
    /// crater plus outward shrapnel spikes -- see
    /// `vox_physics::destruction::ExplosionShape` -- not a plain sphere),
    /// detach whatever becomes unsupported, and give the debris a blast
    /// impulse. `power` is the live-tunable blast strength; `seed` drives
    /// both the crater's shape and per-body spin variation — pass a
    /// different value each call. See [`CarveOutcome`].
    #[allow(
        clippy::too_many_arguments,
        reason = "plain parameter list is clearer here than a params struct"
    )]
    pub fn blast(
        &self,
        world: &mut World,
        phys: &mut PhysicsWorld,
        registry: &MaterialRegistry,
        eye_m: Vec3,
        look: Vec3,
        power: f32,
        seed: u32,
    ) -> CarveOutcome {
        let Some((hit, hit_point_m)) = raycast_scene(world, phys, eye_m, look, REACH) else {
            return CarveOutcome::default();
        };
        let material = hit_material(world, phys, &hit);
        let mut outcome = match hit {
            SceneHit::World(_) => CarveOutcome {
                spawned: vox_physics::blast(
                    world,
                    phys,
                    registry,
                    hit_point_m,
                    self.radius_m,
                    power,
                    seed,
                )
                .spawned,
                ..CarveOutcome::default()
            },
            SceneHit::Body(id, _) => {
                let spawned = vox_physics::carve_body_explosion_at(
                    phys,
                    registry,
                    id,
                    hit_point_m,
                    self.radius_m,
                    seed,
                );
                let outcome = body_outcome(phys, id, spawned);
                vox_physics::apply_blast_impulse(phys, &outcome.spawned, hit_point_m, power, seed);
                outcome
            }
        };
        outcome.impact_m = Some(hit_point_m);
        outcome.impact_material = material;
        outcome
    }

    /// Scalable dig: carve a sphere of [`Tools::radius_m`] at the crosshair
    /// target and detach whatever becomes unsupported, with no blast
    /// impulse -- severed material just falls, unlike [`Tools::blast`]. See
    /// [`CarveOutcome`].
    pub fn scalable_dig(
        &self,
        world: &mut World,
        phys: &mut PhysicsWorld,
        registry: &MaterialRegistry,
        eye_m: Vec3,
        look: Vec3,
    ) -> CarveOutcome {
        let Some((hit, hit_point_m)) = raycast_scene(world, phys, eye_m, look, REACH) else {
            return CarveOutcome::default();
        };
        let material = hit_material(world, phys, &hit);
        let mut outcome = match hit {
            SceneHit::World(_) => {
                let mut carve = vox_physics::carve_sphere(world, hit_point_m, self.radius_m);
                let removed: Vec<IVec3> = carve.removed.iter().map(|&(v, _)| v).collect();
                let ids = vox_physics::detach_unsupported(world, phys, registry, &removed);
                let s = world.cfg.voxel_size_m;
                phys.wake_region(carve.region.0.as_vec3() * s, carve.region.1.as_vec3() * s);
                carve.spawned = ids;
                CarveOutcome {
                    spawned: carve.spawned,
                    ..CarveOutcome::default()
                }
            }
            SceneHit::Body(id, _) => carve_body_sphere(phys, registry, id, hit_point_m, self.radius_m),
        };
        outcome.impact_m = Some(hit_point_m);
        outcome.impact_material = material;
        outcome
    }

    /// Death laser: fire an effectively infinite-range beam from the eye
    /// along `look`, instantly tunneling through everything in its path --
    /// the static world *and* every body in the beam's way -- and detaching
    /// whatever becomes unsupported. No raycast gate, no impulse, just an
    /// immediate, total cut. See [`CarveOutcome`].
    pub fn death_laser(
        &self,
        world: &mut World,
        phys: &mut PhysicsWorld,
        registry: &MaterialRegistry,
        eye_m: Vec3,
        look: Vec3,
    ) -> CarveOutcome {
        let end_m = eye_m + look * DEATH_LASER_RANGE_M;
        // The beam itself needs no raycast gate, but destruction feedback
        // wants the *entry* point -- where the beam first strikes something.
        let entry = raycast_scene(world, phys, eye_m, look, DEATH_LASER_RANGE_M);
        let (impact_m, impact_material) = match &entry {
            Some((hit, point)) => (Some(*point), hit_material(world, phys, hit)),
            None => (None, AIR),
        };
        let mut outcome = CarveOutcome {
            spawned: vox_physics::laser(world, phys, registry, eye_m, end_m, DEATH_LASER_RADIUS_M)
                .spawned,
            impact_m,
            impact_material,
            ..CarveOutcome::default()
        };

        // The beam also tunnels through every body already in its path --
        // debris is small, so attempting every live body is cheap even
        // though most will report "nothing removed".
        let ids: Vec<BodyId> = phys.iter().map(|(id, _)| id).collect();
        for id in ids {
            let sub =
                vox_physics::carve_body_capsule_at(phys, registry, id, eye_m, end_m, DEATH_LASER_RADIUS_M);
            let sub_outcome = body_outcome(phys, id, sub);
            outcome.removed.extend(sub_outcome.removed);
            outcome.spawned.extend(sub_outcome.spawned);
        }
        outcome
    }

    /// Place the selected material against the hit face, unless it would
    /// intersect the player.
    pub fn place_voxel(&self, world: &mut World, eye_m: Vec3, look: Vec3, player: Aabb) {
        let Some(hit) = raycast(world, eye_m, look, REACH) else {
            return;
        };
        let Some(face) = hit.face else {
            return; // Eye inside a solid voxel; nowhere to place.
        };
        let target = hit.voxel + face;
        let s = world.cfg.voxel_size_m;
        let c = voxel_center_m(target, s);
        let half = s * 0.5;
        let overlaps = (c.x + half > player.min.x && c.x - half < player.max.x)
            && (c.y + half > player.min.y && c.y - half < player.max.y)
            && (c.z + half > player.min.z && c.z - half < player.max.z);
        if !overlaps {
            world.set_voxel(target, self.material());
        }
    }

    /// Place Water: fill a sphere of the water-source radius with water at the
    /// crosshair target and hand the filled cells to `fluid` so they start
    /// flowing immediately. Unlike every other tool, this never touches
    /// `PhysicsWorld` -- it can't hit or carve a body, only the static
    /// world, and only into existing air.
    pub fn place_water(
        &self,
        world: &mut World,
        fluid: &mut FluidSim,
        registry: &MaterialRegistry,
        eye_m: Vec3,
        look: Vec3,
    ) {
        let dir = look.normalize_or_zero();
        if dir == Vec3::ZERO {
            return;
        }
        let Some(hit) = raycast(world, eye_m, dir, REACH) else {
            return;
        };
        let Some(water_id) = registry.id_by_name("water") else {
            return; // asset set doesn't define water; nothing to place
        };
        let Some(face) = hit.face else {
            return; // Eye started inside terrain, so there is no empty hit face.
        };
        let s = world.cfg.voxel_size_m;
        let center_vox = hit.voxel + face;
        let radius_vox = (self.water_radius_m / s).round() as i32;
        fluid.place_blob(world, center_vox, radius_vox, Voxel(water_id.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::WorldConfig;
    use vox_core::consts::PHYSICS_DT;

    fn registry() -> MaterialRegistry {
        MaterialRegistry::from_toml_str(
            r#"
            [[material]]
            name = "stone"
            color = [0.5, 0.5, 0.5]
            density = 2600.0
            strength = 8.0
            [[material]]
            name = "wood"
            color = [0.5, 0.4, 0.3]
            density = 700.0
            strength = 4.0
            "#,
            "test.toml",
        )
        .expect("registry")
    }

    #[test]
    fn hotbar_slot_five_selects_place_water() {
        let mut tools = Tools::new(&registry());

        assert_eq!(tools.select_hotbar_slot(5), Some(Tool::PlaceWater));
        assert_eq!(tools.tool, Tool::PlaceWater);
    }

    #[test]
    fn water_keeps_a_smaller_radius_than_bomb_tools() {
        let mut tools = Tools::new(&registry());
        let destructive_radius = tools.radius_m;

        tools.select_hotbar_slot(5);
        assert_eq!(tools.active_radius_m(), WATER_RADIUS_DEFAULT);
        tools.grow_radius();
        assert!(tools.active_radius_m() > WATER_RADIUS_DEFAULT);

        tools.select_hotbar_slot(3);
        assert_eq!(tools.active_radius_m(), destructive_radius);
    }

    /// A 1-voxel-wide wood pillar at a fixed, known footprint (x,z = 5,5),
    /// resting on a thick stone floor, rising to `height_vox` voxels above
    /// it. Single-voxel cross-section: no corners, so a centered sphere
    /// carve clears it uniformly at every height (matching the proven
    /// geometry in `vox_physics::destruction`'s own pillar tests). The floor
    /// is thick enough (12 m) that a generous blast severing the pillar's
    /// base can't also blow all the way through it -- otherwise the debris
    /// would fall through the blast's own crater, which is realistic but
    /// not what this test is checking. Bomb's crater is a jagged
    /// `ExplosionShape`, not a plain sphere: its shrapnel spikes reach up to
    /// ~2.6x the nominal radius (e.g. ~7.9 m for this test's 3 m blasts), so
    /// the floor must clear that reach with real margin, not just the
    /// nominal radius.
    const FLOOR_THICKNESS_VOX: i32 = 60;
    /// Pillar footprint, centered in a generously large floor so debris
    /// picking up lateral velocity from the blast (plus several rampage
    /// blasts) has real room to drift without exiting the world's bounds
    /// entirely — once outside, there's no floor anywhere and it free-falls
    /// forever, which is a test-world-sizing issue, not a solver bug.
    const PILLAR_XZ_VOX: i32 = 160;

    fn wood_tower(voxel_size_m: f32, height_vox: i32) -> World {
        let mut world = World::new(WorldConfig {
            voxel_size_m,
            extent_m: [64.0, 24.0, 64.0],
            ..WorldConfig::default()
        });
        let (_, max) = world.bounds_voxels();
        world.fill_box(
            IVec3::ZERO,
            IVec3::new(max.x, FLOOR_THICKNESS_VOX, max.z),
            Voxel(1),
        );
        let base = IVec3::new(PILLAR_XZ_VOX, FLOOR_THICKNESS_VOX, PILLAR_XZ_VOX);
        world.fill_box(base, base + IVec3::new(1, height_vox, 1), Voxel(2));
        world
    }

    /// The full player-facing entry point (raycast → carve → detach →
    /// impulse), exercised end to end: blasting a wood tower's base
    /// detaches the upper section as tumbling debris that eventually
    /// settles, with no NaNs or solver blow-up across repeated blasts.
    #[test]
    fn blasting_a_tower_base_detaches_the_top_which_settles() {
        let s = 0.2;
        let mut world = wood_tower(s, 40); // an 8m tall tower
        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let mut tools = Tools::new(&reg);
        tools.tool = Tool::Bomb;
        tools.radius_m = 3.0;

        // Footprint is voxel x,z = PILLAR_XZ_VOX -> center at (+0.5)*s
        // meters. Aim a level, axis-aligned ray straight down +X at the
        // pillar's western face, low enough to be within the base stub.
        let floor_top_m = FLOOR_THICKNESS_VOX as f32 * s;
        let tower_top_m = floor_top_m + 40.0 * s;
        let px = PILLAR_XZ_VOX as f32 * s;
        let cz = px + 0.5 * s;
        let eye = Vec3::new(px - 2.0, floor_top_m + 1.0, cz);
        let look = Vec3::X;
        let spawned = tools.blast(&mut world, &mut phys, &reg, eye, look, 40.0, 1);

        assert!(phys.body_count() > 0, "the upper tower section must detach");
        assert_eq!(
            spawned.spawned.len(),
            phys.body_count(),
            "blast must report every spawned body's id back to the caller \
             (the caller needs these to upload debris meshes -- an id that \
             gets lost here means real, physically-simulated debris that \
             never appears on screen)"
        );
        for (_, body) in phys.iter() {
            assert!(body.vel.is_finite() && body.pos.is_finite());
            assert!(
                body.pos.y < tower_top_m + 1.0,
                "detached body should be part of the tower, not the whole extent"
            );
        }

        // "Blast rampage": a few more blasts well up in open air near the
        // remaining structure (not the ground the first blast already
        // exposed — repeatedly blasting the exact same low spot would just
        // carve a hole straight through the floor, which is a floor design
        // problem, not a solver one), watching for divergence.
        for i in 0..5 {
            let y = floor_top_m + 3.0 + i as f32 * 0.5;
            let rampage_eye = Vec3::new(px - 2.0, y, cz);
            tools.blast(&mut world, &mut phys, &reg, rampage_eye, look, 40.0, 2 + i);
            for _ in 0..30 {
                phys.step(&world, PHYSICS_DT);
                for (_, body) in phys.iter() {
                    assert!(
                        body.vel.is_finite() && body.pos.is_finite() && body.vel.length() < 200.0,
                        "solver diverged after blast {i}"
                    );
                }
            }
        }

        // Let everything finish settling. Longer than it used to be: the
        // bomb now also scatters small flying debris chips (on top of the
        // detached structural fragments), which means more bodies bouncing
        // off each other during the cascade -- more collisions to damp out
        // before the scene as a whole goes quiet.
        for _ in 0..900 {
            phys.step(&world, PHYSICS_DT);
        }
        let awake = phys.awake_count();
        assert!(
            awake * 4 <= phys.body_count().max(1),
            "most debris should be asleep by now: {awake}/{} awake",
            phys.body_count()
        );
        for (_, body) in phys.iter() {
            assert!(body.pos.y > -1.0, "nothing should fall through the floor");
        }
    }

    /// Scalable dig (hotbar slot 2) through the raycast entry point: unlike
    /// `blast`, the severed section must detach with *no* impulse -- it
    /// starts at rest and only gravity/impacts move it from there.
    #[test]
    fn scalable_dig_detaches_the_tower_top_with_no_impulse() {
        let s = 0.2;
        let mut world = wood_tower(s, 40);
        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let mut tools = Tools::new(&reg);
        tools.tool = Tool::ScalableDig;
        tools.radius_m = 3.0;

        let floor_top_m = FLOOR_THICKNESS_VOX as f32 * s;
        let px = PILLAR_XZ_VOX as f32 * s;
        let cz = px + 0.5 * s;
        let eye = Vec3::new(px - 2.0, floor_top_m + 1.0, cz);
        let spawned = tools.scalable_dig(&mut world, &mut phys, &reg, eye, Vec3::X);

        assert!(phys.body_count() > 0, "the upper tower section must detach");
        assert_eq!(spawned.spawned.len(), phys.body_count());
        for (_, body) in phys.iter() {
            assert_eq!(
                body.vel,
                Vec3::ZERO,
                "scalable dig must not impart any impulse, unlike a bomb"
            );
        }
    }

    /// Death laser (hotbar slot 4) through the raycast-free entry point: an
    /// "infinite range" beam fired at a tower far beyond normal tool reach
    /// must still tunnel through it and detach the severed top.
    #[test]
    fn death_laser_reaches_far_beyond_normal_tool_range_and_detaches_debris() {
        let s = 0.2;
        let mut world = wood_tower(s, 40);
        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let tools = Tools::new(&reg);

        let floor_top_m = FLOOR_THICKNESS_VOX as f32 * s;
        let px = PILLAR_XZ_VOX as f32 * s;
        let cz = px + 0.5 * s;
        // Stand far outside REACH (5 m) -- a normal tool couldn't touch the
        // tower from here at all.
        let eye = Vec3::new(px - 50.0, floor_top_m + 1.0, cz);
        let spawned = tools.death_laser(&mut world, &mut phys, &reg, eye, Vec3::X);

        assert!(
            phys.body_count() > 0,
            "the laser must reach and detach the tower's upper section from far beyond REACH"
        );
        assert_eq!(spawned.spawned.len(), phys.body_count());
    }

    /// A free-floating debris body (as if already spawned by an earlier
    /// blast), sitting in open air with nothing else nearby, so any tool's
    /// world raycast can't possibly hit anything first.
    fn floating_debris(reg: &MaterialRegistry, phys: &mut PhysicsWorld, dims: IVec3) -> BodyId {
        let voxels = vec![Voxel(1); (dims.x * dims.y * dims.z) as usize];
        let grid = vox_physics::VoxelGrid::new(dims, voxels);
        let body = vox_physics::Body::from_grid(grid, reg, 0.2, Vec3::new(50.0, 50.0, 50.0))
            .expect("solid grid must be massive");
        phys.spawn(body)
    }

    /// The actual point of this whole feature: debris that already exists
    /// (e.g. from an earlier blast) must itself be breakable by every tool,
    /// not just the static world. Dig should chip exactly one voxel off,
    /// reporting the original body as removed and a smaller replacement as
    /// spawned.
    #[test]
    fn dig_hits_and_chips_an_existing_debris_body() {
        let mut world = World::new(WorldConfig {
            voxel_size_m: 0.2,
            extent_m: [128.0, 128.0, 128.0],
            ..WorldConfig::default()
        });
        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let tools = Tools::new(&reg);
        let id = floating_debris(&reg, &mut phys, IVec3::splat(6));
        let original_count = phys.get(id).unwrap().grid.solid_count();

        let pos = phys.get(id).unwrap().pos;
        let eye = pos + Vec3::new(-3.0, 0.0, 0.0);
        let outcome = tools.dig(&mut world, &mut phys, &reg, eye, Vec3::X);

        assert_eq!(outcome.removed, vec![id], "the original body must be reported removed");
        assert_eq!(outcome.spawned.len(), 1, "chipping one voxel must not split a 6^3 cube");
        let remaining = phys.get(outcome.spawned[0]).unwrap().grid.solid_count();
        assert_eq!(remaining, original_count - 1, "must have removed exactly one voxel");
    }

    /// Bomb hitting debris must despawn the original and give the resulting
    /// fragment(s) an outward blast impulse, just like hitting the world.
    #[test]
    fn bomb_hits_debris_and_gives_it_an_impulse() {
        let mut world = World::new(WorldConfig {
            voxel_size_m: 0.2,
            extent_m: [128.0, 128.0, 128.0],
            ..WorldConfig::default()
        });
        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let mut tools = Tools::new(&reg);
        tools.tool = Tool::Bomb;
        tools.radius_m = 1.0;
        let id = floating_debris(&reg, &mut phys, IVec3::splat(10));

        let pos = phys.get(id).unwrap().pos;
        let eye = pos + Vec3::new(-3.0, 0.0, 0.0);
        let outcome = tools.blast(&mut world, &mut phys, &reg, eye, Vec3::X, 40.0, 1);

        assert_eq!(outcome.removed, vec![id]);
        assert!(!outcome.spawned.is_empty(), "must produce at least one fragment");
        for &fid in &outcome.spawned {
            let f = phys.get(fid).unwrap();
            assert!(f.vel.length() > 0.1, "bomb must impart velocity to debris fragments too");
        }
    }

    /// Death laser must tunnel through debris in its path even when the
    /// beam's raycast-free design means it never "aims" at anything --
    /// every live body in range gets attempted.
    #[test]
    fn death_laser_tunnels_through_debris_in_its_path() {
        let mut world = World::new(WorldConfig {
            voxel_size_m: 0.2,
            extent_m: [128.0, 128.0, 128.0],
            ..WorldConfig::default()
        });
        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let tools = Tools::new(&reg);
        let id = floating_debris(&reg, &mut phys, IVec3::splat(6));

        let pos = phys.get(id).unwrap().pos;
        let eye = pos + Vec3::new(-20.0, 0.0, 0.0);
        let outcome = tools.death_laser(&mut world, &mut phys, &reg, eye, Vec3::X);

        assert!(outcome.removed.contains(&id), "the beam must reach the debris body");
    }

    fn registry_with_tree_materials() -> MaterialRegistry {
        MaterialRegistry::from_toml_str(
            r#"
            [[material]]
            name = "stone"
            color = [0.5, 0.5, 0.5]
            density = 2600.0
            strength = 8.0
            [[material]]
            name = "wood"
            color = [0.5, 0.4, 0.3]
            density = 700.0
            strength = 4.0
            [[material]]
            name = "leaves"
            color = [0.2, 0.5, 0.2]
            density = 300.0
            strength = 1.0
            "#,
            "test.toml",
        )
        .expect("registry")
    }

    /// A real, generator-produced tree (multi-voxel-wide tapered trunk plus
    /// branches and canopy leaves), severed across its full base
    /// cross-section. Earlier destruction tests only covered hand-built,
    /// single-voxel-wide columns; this is the case the user actually hit:
    /// does a real, wide trunk detach its whole disconnected upper section
    /// (trunk, branches, and canopy) once fully severed, given the canopy's
    /// true size (tens of thousands of voxels at 0.1 m scale)?
    #[test]
    fn severing_a_real_tree_trunk_detaches_the_whole_tree() {
        let s = 0.1;
        let mut world = World::new(WorldConfig {
            voxel_size_m: s,
            extent_m: [64.0, 32.0, 64.0],
            ..WorldConfig::default()
        });
        let reg = registry_with_tree_materials();
        let mats = vox_gen::TreeMaterials::from_registry(&reg).expect("tree materials");

        let ground_top_vox = (10.0 / s) as i32;
        let (_, max) = world.bounds_voxels();
        world.fill_box(
            IVec3::ZERO,
            IVec3::new(max.x, ground_top_vox, max.z),
            Voxel(reg.id_by_name("stone").unwrap().0),
        );

        let tree = vox_gen::TreeInstance {
            x_m: 32.0,
            z_m: 32.0,
            base_y_m: 10.0,
            height_m: 8.0,
            tree_seed: 0xC0FFEE,
        };
        vox_gen::stamp_tree(&mut world, &tree, mats);

        // A cut layer 0.3 m above the ground: comfortably inside the trunk's
        // tapered base radius (~0.3-0.45 m), well below any branch (branches
        // start at 55%+ of the tree's height).
        let cut_y = ((10.3) / s) as i32;
        let mut cut_voxels = Vec::new();
        for z in 0..max.z {
            for x in 0..max.x {
                let v = IVec3::new(x, cut_y, z);
                if world.get_voxel(v) == mats.wood {
                    cut_voxels.push(v);
                }
            }
        }
        assert!(
            cut_voxels.len() > 1,
            "expected a multi-voxel-wide trunk cross-section at the cut \
             layer, found {}",
            cut_voxels.len()
        );

        // Sever the whole cross-section at once (one `check_broken_support`
        // call per voxel would be a faithful frame-by-frame replay of a
        // player clicking through the ring, but is redundant here: every
        // intermediate call re-floods the still-fully-connected tree from
        // scratch only to conclude "still anchored", which is correct but
        // needlessly expensive to repeat dozens of times in one test -- the
        // per-click cost itself is exercised by the single-voxel break tests
        // elsewhere. What this test needs to prove is that severing a real,
        // multi-voxel-wide, generator-produced tree detaches correctly at
        // all, given its true (large) canopy size.
        for &v in &cut_voxels {
            world.set_voxel(v, AIR);
        }
        let mut phys = PhysicsWorld::new();
        vox_physics::detach_unsupported(&mut world, &mut phys, &reg, &cut_voxels);

        assert!(
            phys.body_count() > 0,
            "the fully-severed tree above the cut must detach as debris, \
             not vanish or stay floating in place"
        );
        let total_debris_voxels: usize = phys.iter().map(|(_, b)| b.grid.solid_count()).sum();
        assert!(
            total_debris_voxels > 50,
            "detached debris ({total_debris_voxels} voxels) looks far too \
             small to be the whole severed tree (trunk + branches + canopy)"
        );
    }

    fn registry_with_water() -> MaterialRegistry {
        MaterialRegistry::from_toml_str(
            r#"
            [[material]]
            name = "stone"
            color = [0.5, 0.5, 0.5]
            density = 2000.0
            strength = 5.0

            [[material]]
            name = "water"
            color = [0.1, 0.3, 0.8]
            density = 1000.0
            strength = 0.0
            solid = false
            fluid = true
            "#,
            "test.toml",
        )
        .expect("registry")
    }

    fn solid_table_for(reg: &MaterialRegistry) -> Vec<bool> {
        (0..reg.len())
            .map(|i| reg.get(vox_core::MaterialId(i as u16)).is_some_and(|d| d.solid))
            .collect()
    }

    fn stone_id(reg: &MaterialRegistry) -> Voxel {
        Voxel(reg.id_by_name("stone").unwrap().0)
    }

    fn water_voxel(reg: &MaterialRegistry) -> Voxel {
        Voxel(reg.id_by_name("water").unwrap().0)
    }

    #[test]
    fn place_water_uses_the_empty_voxel_on_the_hit_face() {
        // With 2 m voxels, Water's 0.5 m default radius rounds to zero
        // voxels. That makes the test distinguish the face-adjacent target
        // from the struck solid voxel exactly.
        let reg = registry_with_water();
        let water = water_voxel(&reg);
        let stone = stone_id(&reg);
        let mut world = World::new(WorldConfig {
            voxel_size_m: 2.0,
            extent_m: [32.0, 32.0, 32.0],
            ..WorldConfig::default()
        });
        world.set_solid_table(solid_table_for(&reg));
        let hit_voxel = IVec3::new(4, 4, 4);
        let target = hit_voxel + IVec3::NEG_X;
        world.set_voxel(hit_voxel, stone);

        let mut sim = FluidSim::new(water);
        let tools = Tools::new(&reg);
        tools.place_water(
            &mut world,
            &mut sim,
            &reg,
            Vec3::new(5.0, 9.0, 9.0),
            Vec3::X,
        );

        assert_eq!(world.get_voxel(hit_voxel), stone, "water must not overwrite the struck terrain voxel");
        assert_eq!(world.get_voxel(target), water, "water must begin on the hit face's adjacent air voxel");
    }

    #[test]
    fn place_water_fills_a_sphere_at_the_crosshair_and_activates_it() {
        let reg = registry_with_water();
        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [32.0, 32.0, 32.0],
            ..WorldConfig::default()
        });
        world.set_solid_table(solid_table_for(&reg));
        let (_, max) = world.bounds_voxels();
        world.fill_box(IVec3::ZERO, IVec3::new(max.x, 10, max.z), stone_id(&reg));

        let mut sim = vox_sim::FluidSim::new(water_voxel(&reg));
        let tools = Tools::new(&reg);
        tools.place_water(&mut world, &mut sim, &reg, Vec3::new(16.0, 15.0, 16.0), Vec3::new(0.0, -1.0, 0.0));

        assert!(sim.active_count() > 0, "placing water must activate at least one cell");
    }

    /// True if any voxel in the half-open box `[min, max)` holds `water`.
    fn any_water_in(world: &World, water: Voxel, min: IVec3, max: IVec3) -> bool {
        for x in min.x..max.x {
            for y in min.y..max.y {
                for z in min.z..max.z {
                    if world.get_voxel(IVec3::new(x, y, z)) == water {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// End-to-end simulation integration: a water blob settles in a basin,
    /// a terrain edit is drained through World's real dirty-region API, and
    /// the resulting wake drives water through the new downhill spillway.
    /// `Tools::place_water` is covered separately above; this test keeps the
    /// setup grid-exact while exercising the same headless simulation path
    /// used by the application frame loop.
    #[test]
    fn digging_into_a_settled_lake_lets_it_drain() {
        let reg = registry_with_water();
        let water = water_voxel(&reg);
        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [40.0, 20.0, 40.0],
            ..WorldConfig::default()
        });
        world.set_solid_table(solid_table_for(&reg));
        let (_, max) = world.bounds_voxels();

        // Global floor under the whole world footprint: `fill_box` is
        // half-open, so this leaves y=4 as the top solid layer and y=5 as
        // the open resting surface everywhere, both inside and outside the
        // basin -- water that escapes through the breach lands on this same
        // floor instead of free-falling into a void once it clears the
        // wall.
        world.fill_box(IVec3::ZERO, IVec3::new(max.x, 5, max.z), stone_id(&reg));

        // Basin: a 12x12 stone box (walls spanning y=5..15, i.e. resting on
        // the floor) with its interior (x/z 10..20) hollowed back out to
        // air. The hollow-out must happen *after* the solid walls are
        // built: `place_blob` (and `fill_box` in general) writes whatever
        // it's told regardless of what's already there, so building the
        // walls first and clearing the interior second is what leaves a
        // 1-voxel-thick wall standing on every side rather than either no
        // walls or no interior.
        world.fill_box(IVec3::new(9, 5, 9), IVec3::new(21, 15, 21), stone_id(&reg));
        world.fill_box(IVec3::new(10, 5, 10), IVec3::new(20, 15, 20), AIR);

        let mut sim = FluidSim::new(water);
        // A blob comfortably inside the 9-voxel-wide interior (radius 3, so
        // diameter 7), centered well above the floor so it genuinely falls
        // and spreads rather than starting already at rest -- and clear of
        // every wall, so it starts out fully confined by construction, not
        // by luck.
        sim.place_blob(&mut world, IVec3::new(15, 10, 15), 3, water);
        assert!(sim.active_count() > 0, "placing the blob must activate cells");

        let basin_interior = (IVec3::new(10, 5, 10), IVec3::new(20, 15, 20));
        assert!(
            any_water_in(&world, water, basin_interior.0, basin_interior.1),
            "basin must actually contain water right after placement"
        );

        // Let it settle. Determined empirically (not guessed): this basin's
        // radius-3 blob (~123 cells) reaches `active_count() == 0` well
        // before 150 ticks once it has finished falling and spreading
        // across the 9x9 interior floor; 300 leaves real margin without
        // masking a genuine convergence problem (a basin that never settles
        // at all would still fail this at any reasonable budget).
        const SETTLE_TICK_BUDGET: usize = 300;
        let mut settled = false;
        for _ in 0..SETTLE_TICK_BUDGET {
            sim.tick(&mut world);
            if sim.active_count() == 0 {
                settled = true;
                break;
            }
        }
        assert!(
            settled,
            "lake must settle to 0 active cells within {SETTLE_TICK_BUDGET} ticks -- if it \
             doesn't, either the basin isn't actually confining the water (an unintended \
             escape route) or convergence is genuinely this slow at this basin size"
        );
        assert!(
            any_water_in(&world, water, basin_interior.0, basin_interior.1),
            "basin must still hold water once settled (didn't all leak out before the wall \
             was ever breached)"
        );

        // No manufactured vertical head needed: the flow rule lets a
        // blocked cell walk toward any drop within its horizon, so opening
        // the breach below is enough for the water beside it to find the
        // way out. Just confirm the settled lake actually reaches the wall
        // that's about to be breached.
        let spillway = IVec3::new(10, 5, 15);
        assert_eq!(world.get_voxel(spillway), water, "the settled lake must cover the spillway floor cell");

        // Model the real frame-loop handoff precisely: discard all old
        // regions, make the two terrain edits, then wake from exactly the
        // regions reported by World. The opening is at floor height; the
        // lowered exterior cell gives the water a diagonal gravity path out
        // of the basin rather than relying on an unphysical flat random walk.
        world.drain_dirty_regions();
        let breach = IVec3::new(9, 5, 15);
        assert_eq!(world.get_voxel(breach), stone_id(&reg), "sanity: the breach point must start as a real wall");
        world.set_voxel(breach, AIR);
        let downhill = IVec3::new(8, 4, 15);
        assert_eq!(world.get_voxel(downhill), stone_id(&reg), "sanity: the spillway must begin with solid terrain below it");
        world.set_voxel(downhill, AIR);
        for (min, max) in world.drain_dirty_regions() {
            sim.wake_region(&world, min, max);
        }
        assert!(
            sim.active_count() > 0,
            "digging into the settled lake must reactivate water near the breach"
        );

        // Tick forward and confirm water reaches *outside* the basin
        // (x < 9, the wall's outer face) through the breach specifically,
        // not just "some water somewhere moved".
        const DRAIN_TICK_BUDGET: usize = 300;
        let outside = (IVec3::new(0, 0, 0), IVec3::new(9, 15, max.z));
        let mut escaped = false;
        for _ in 0..DRAIN_TICK_BUDGET {
            sim.tick(&mut world);
            // This is the same post-tick dirty drain that the app uses.
            // It keeps the test on the real wake-on-edit path instead of
            // relying on a manually padded wake box.
            for (min, max) in world.drain_dirty_regions() {
                sim.wake_region(&world, min, max);
            }
            if any_water_in(&world, water, outside.0, outside.1) {
                escaped = true;
                break;
            }
        }
        assert!(
            escaped,
            "water must flow out through the breach and reach outside the basin within \
             {DRAIN_TICK_BUDGET} ticks"
        );
    }

    /// Registry with the full weathering material set (grass, dirt, mud,
    /// stone, sand, water) -- mirrors the shipped `core.toml` ids by name
    /// resolution, the same path `weather_table` in `main.rs` takes.
    fn registry_with_weathering() -> MaterialRegistry {
        MaterialRegistry::from_toml_str(
            r#"
            [[material]]
            name = "stone"
            color = [0.55, 0.55, 0.57]
            density = 2600.0
            strength = 8.0

            [[material]]
            name = "dirt"
            color = [0.45, 0.32, 0.22]
            density = 1500.0
            strength = 2.0

            [[material]]
            name = "grass"
            color = [0.33, 0.55, 0.25]
            density = 1400.0
            strength = 2.0

            [[material]]
            name = "sand"
            color = [0.86, 0.79, 0.58]
            density = 1600.0
            strength = 1.0
            solid = false
            powder = true

            [[material]]
            name = "mud"
            color = [0.30, 0.22, 0.16]
            density = 1700.0
            strength = 1.0
            solid = false
            powder = true

            [[material]]
            name = "water"
            color = [0.16, 0.35, 0.62]
            density = 1000.0
            strength = 0.0
            solid = false
            fluid = true
            "#,
            "test_weathering.toml",
        )
        .expect("registry")
    }

    fn voxel_by_name(reg: &MaterialRegistry, name: &str) -> Voxel {
        Voxel(reg.id_by_name(name).unwrap().0)
    }

    /// End-to-end through the real registry: place water on a grass field,
    /// run the fluid + weathering loop the way the frame loop does, and the
    /// grass beneath must progress grass -> dirt -> mud. This is the
    /// integration contract Task 7 pins -- if `main.rs` ever drops the
    /// `drain_events -> weathering.tick` call, the sim still works but this
    /// test proves the material transformation is actually wired.
    #[test]
    fn a_pool_on_grass_turns_its_bed_to_mud() {
        let reg = registry_with_weathering();
        let water = voxel_by_name(&reg, "water");
        let grass = voxel_by_name(&reg, "grass");
        let mud = voxel_by_name(&reg, "mud");

        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [24.0, 24.0, 24.0],
            ..WorldConfig::default()
        });
        world.set_solid_table(solid_table_for(&reg));
        let (_, max) = world.bounds_voxels();

        // Solid stone floor, grass top layer.
        world.fill_box(IVec3::ZERO, IVec3::new(max.x, 5, max.z), voxel_by_name(&reg, "stone"));
        world.fill_box(IVec3::new(0, 4, 0), IVec3::new(max.x, 5, max.z), grass);

        // Build the weathering table from the registry exactly like main.rs.
        let table = vox_sim::WeatherTable {
            water,
            stone: voxel_by_name(&reg, "stone"),
            grass,
            dirt: voxel_by_name(&reg, "dirt"),
            mud,
            sand: voxel_by_name(&reg, "sand"),
        };
        let mut sim = FluidSim::with_powders(water, vec![voxel_by_name(&reg, "mud"), voxel_by_name(&reg, "sand")]);
        let mut weathering = vox_sim::Weathering::new(table);

        // Place a small pool and run the loop the way the frame loop does:
        // fluid.tick -> drain_events -> weathering.tick -> drain dirty
        // regions -> wake. The grass beneath must progress to mud.
        sim.place_blob(&mut world, IVec3::new(12, 8, 12), 2, water);

        let budget = (vox_sim::GRASS_SOAK_TICKS + vox_sim::DIRT_SOAK_TICKS) * 3;
        let mut found_mud = false;
        for _ in 0..budget {
            sim.tick(&mut world);
            let events = sim.drain_events();
            weathering.tick(&mut world, &events);
            for (min, max) in world.drain_dirty_regions() {
                sim.wake_region(&world, min, max);
            }
            // Check for mud under the pool area.
            for x in 9..16 {
                for z in 9..16 {
                    if world.get_voxel(IVec3::new(x, 4, z)) == mud {
                        found_mud = true;
                    }
                }
            }
            if found_mud {
                break;
            }
        }
        assert!(found_mud, "the pool's grass bed must turn to mud within the soak budget");
    }

    /// Sand placed in midair must fall, pile on the floor, and settle --
    /// exercising the powder path end-to-end through the real registry and
    /// `FluidSim::with_powders`, the same wiring `main.rs` uses.
    #[test]
    fn sand_falls_and_piles_as_a_powder() {
        let reg = registry_with_weathering();
        let water = voxel_by_name(&reg, "water");
        let sand = voxel_by_name(&reg, "sand");
        let stone = voxel_by_name(&reg, "stone");

        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [24.0, 24.0, 24.0],
            ..WorldConfig::default()
        });
        world.set_solid_table(solid_table_for(&reg));
        let (_, max) = world.bounds_voxels();
        // Stone floor (top at y=4, powder rests at y=5).
        world.fill_box(IVec3::ZERO, IVec3::new(max.x, 5, max.z), stone);

        let mut sim = FluidSim::with_powders(water, vec![sand, voxel_by_name(&reg, "mud")]);

        // Place a blob of sand in midair and let it fall.
        sim.place_blob(&mut world, IVec3::new(12, 12, 12), 2, sand);
        assert!(sim.active_count() > 0, "placing sand must activate cells");

        let before = count_material(&world, sand);
        for _ in 0..80 {
            sim.tick(&mut world);
            for (min, max) in world.drain_dirty_regions() {
                sim.wake_region(&world, min, max);
            }
            if sim.active_count() == 0 {
                break;
            }
        }
        assert_eq!(sim.active_count(), 0, "sand pile must settle");
        let after = count_material(&world, sand);
        assert_eq!(before, after, "sand cell count must be conserved");
        // Sand must have fallen to near the floor (y=5 is the resting surface).
        let mut near_floor = false;
        for x in 8..16 {
            for z in 8..16 {
                if world.get_voxel(IVec3::new(x, 5, z)) == sand {
                    near_floor = true;
                }
            }
        }
        assert!(near_floor, "sand must pile on the floor, not stay suspended");
    }

    fn count_material(world: &World, v: Voxel) -> usize {
        let (min, max) = world.bounds_voxels();
        let mut n = 0;
        for x in min.x..max.x {
            for y in min.y..max.y {
                for z in min.z..max.z {
                    if world.get_voxel(IVec3::new(x, y, z)) == v {
                        n += 1;
                    }
                }
            }
        }
        n
    }
}
