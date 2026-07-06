//! Destruction: carve voxels from the world, then determine which surviving
//! material is still structurally supported and which has become debris.
//!
//! Pipeline: **carve** (remove a shape, recording what was removed) →
//! **flood** (6-connected BFS from "anchors" — solid voxels on the edge of
//! the affected region, or resting on the world floor — outward through the
//! remaining solid material) → **detach** (anything the flood never reaches
//! is unsupported: extracted into a [`VoxelGrid`] and spawned as a
//! [`Body`]). Tiny fragments are discarded as dust; implausibly large
//! components are left in the world as a safety valve against an unbounded
//! physics budget.
//!
//! The anchor rule is a bounded local approximation, not a whole-world
//! reachability search: material touching the padded region's outer shell is
//! *assumed* connected to the unexamined rest of the world. This holds for
//! the common case (freshly generated terrain, localized destruction) and
//! keeps the flood's cost proportional to the blast, not the world size.

use std::collections::{HashSet, VecDeque};

use glam::{IVec3, Vec3};
use vox_core::consts::{DEBRIS_MIN_VOXELS, MAX_BODY_VOXELS};
use vox_core::{MaterialRegistry, voxel_at, voxel_center_m};
use vox_world::{AIR, Voxel, World};

use crate::BodyId;
use crate::body::{Body, VoxelGrid, mass_props};
use crate::solver::PhysicsWorld;

/// Extra voxels searched beyond the carved region on every side. Must see at
/// least one still-solid layer to seed anchors correctly.
const REGION_PAD: i32 = 2;

/// Blast impulse tuning: base strength (arbitrary units tuned by feel) and
/// the maximum per-axis angular kick in rad/s.
const BLAST_POWER: f32 = 40.0;
const BLAST_SPIN_MAX: f32 = 3.0;
/// Floor on distance-from-center used in the impulse falloff, so a blast
/// centered inside debris doesn't produce an infinite/huge speed.
const BLAST_MIN_DIST_M: f32 = 0.5;

/// A half-open voxel-space box `[min, max)`.
pub type Region = (IVec3, IVec3);

/// What a carve removed, and the region subsequently searched for
/// connectivity (the removal's bounding box, padded).
pub struct CarveResult {
    pub removed: Vec<(IVec3, Voxel)>,
    pub region: Region,
}

/// Remove every solid voxel whose center lies within `radius_m` of
/// `center_m`. Returns what was removed and the padded region to search for
/// newly-unsupported material.
pub fn carve_sphere(world: &mut World, center_m: Vec3, radius_m: f32) -> CarveResult {
    let s = world.cfg.voxel_size_m;
    let r_vox = (radius_m / s).ceil() as i32;
    let center_vox = voxel_at(center_m, s);

    let mut removed = Vec::new();
    let mut min = IVec3::splat(i32::MAX);
    let mut max = IVec3::splat(i32::MIN);
    for dz in -r_vox..=r_vox {
        for dy in -r_vox..=r_vox {
            for dx in -r_vox..=r_vox {
                let v = center_vox + IVec3::new(dx, dy, dz);
                if (voxel_center_m(v, s) - center_m).length() > radius_m {
                    continue;
                }
                let existing = world.get_voxel(v);
                if existing != AIR {
                    removed.push((v, existing));
                    world.set_voxel(v, AIR);
                    min = min.min(v);
                    max = max.max(v);
                }
            }
        }
    }
    let region = if removed.is_empty() {
        (center_vox, center_vox + IVec3::ONE)
    } else {
        (
            min - IVec3::splat(REGION_PAD),
            max + IVec3::splat(REGION_PAD + 1),
        )
    };
    CarveResult { removed, region }
}

/// The six face-adjacent directions.
const DIRS: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
];

/// Find material in `region` that is no longer structurally supported,
/// extract it from the world, and spawn each surviving component as a
/// sleeping-eligible (initially awake, zero-velocity) rigid body. Components
/// under [`DEBRIS_MIN_VOXELS`] are discarded as dust; components over
/// [`MAX_BODY_VOXELS`] are left in the world untouched. Returns the spawned
/// body ids.
pub fn detach_unsupported(
    world: &mut World,
    phys: &mut PhysicsWorld,
    registry: &MaterialRegistry,
    region: Region,
) -> Vec<BodyId> {
    let (region_min, region_max) = region;
    let floor_y = world.bounds_voxels().0.y;

    // Pass 1: collect solid voxels in the region and seed anchors — solid
    // voxels touching the region's outer shell (assumed connected to
    // whatever lies beyond it) or resting on the world floor.
    let mut solid: HashSet<IVec3> = HashSet::new();
    let mut anchors: Vec<IVec3> = Vec::new();
    for z in region_min.z..region_max.z {
        for y in region_min.y..region_max.y {
            for x in region_min.x..region_max.x {
                let v = IVec3::new(x, y, z);
                if !world.solid(v) {
                    continue;
                }
                solid.insert(v);
                let on_shell = x == region_min.x
                    || x == region_max.x - 1
                    || y == region_min.y
                    || y == region_max.y - 1
                    || z == region_min.z
                    || z == region_max.z - 1;
                if on_shell || y == floor_y {
                    anchors.push(v);
                }
            }
        }
    }

    // Pass 2: BFS from anchors marks everything still supported.
    let mut supported: HashSet<IVec3> = HashSet::new();
    let mut queue: VecDeque<IVec3> = VecDeque::new();
    for a in anchors {
        if supported.insert(a) {
            queue.push_back(a);
        }
    }
    while let Some(v) = queue.pop_front() {
        for d in DIRS {
            let n = v + d;
            if solid.contains(&n) && supported.insert(n) {
                queue.push_back(n);
            }
        }
    }

    // Pass 3: group every unsupported voxel into connected components.
    let mut visited: HashSet<IVec3> = HashSet::new();
    let mut components: Vec<Vec<IVec3>> = Vec::new();
    for &start in &solid {
        if supported.contains(&start) || visited.contains(&start) {
            continue;
        }
        let mut comp = Vec::new();
        let mut q = VecDeque::new();
        q.push_back(start);
        visited.insert(start);
        while let Some(v) = q.pop_front() {
            comp.push(v);
            for d in DIRS {
                let n = v + d;
                if solid.contains(&n) && !supported.contains(&n) && visited.insert(n) {
                    q.push_back(n);
                }
            }
        }
        components.push(comp);
    }

    // Pass 4: extract each component per the size policy.
    let voxel_size_m = world.cfg.voxel_size_m;
    let mut ids = Vec::new();
    for comp in components {
        if comp.len() < DEBRIS_MIN_VOXELS {
            for &v in &comp {
                world.set_voxel(v, AIR);
            }
            continue;
        }
        if comp.len() > MAX_BODY_VOXELS {
            continue; // Safety valve: leave the material in the world.
        }

        let mut min = IVec3::splat(i32::MAX);
        let mut max = IVec3::splat(i32::MIN);
        for &v in &comp {
            min = min.min(v);
            max = max.max(v);
        }
        let dims = max - min + IVec3::ONE;
        let mut voxels = vec![AIR; (dims.x * dims.y * dims.z) as usize];
        for &v in &comp {
            let mat = world.get_voxel(v);
            let l = v - min;
            let idx = (l.x + l.z * dims.x + l.y * dims.x * dims.z) as usize;
            voxels[idx] = mat;
            world.set_voxel(v, AIR);
        }

        let grid = VoxelGrid::new(dims, voxels);
        let props = mass_props(&grid, registry, voxel_size_m);
        debug_assert!(props.mass > 0.0, "extracted component must have mass");
        let com_world_m = min.as_vec3() * voxel_size_m + props.com_local;
        if let Some(body) = Body::from_grid(grid, registry, voxel_size_m, com_world_m) {
            ids.push(phys.spawn(body));
        }
    }
    ids
}

/// Deterministic small hash for per-body blast variation.
#[inline]
fn small_hash(a: u32, b: u32) -> u32 {
    let mut x = a.wrapping_mul(0x8529_7a4d) ^ b.wrapping_mul(0x68e3_1da4);
    x ^= x >> 17;
    x = x.wrapping_mul(0xed5a_d4bb);
    x ^= x >> 11;
    x = x.wrapping_mul(0xac4c_1b51);
    x ^= x >> 15;
    x
}

/// Give each listed body a blast impulse radiating from `center_m`, falling
/// off with distance and scaled down for heavier bodies, plus a small
/// deterministic angular kick.
fn apply_blast_impulse(phys: &mut PhysicsWorld, ids: &[BodyId], center_m: Vec3, seed: u32) {
    for (i, &id) in ids.iter().enumerate() {
        let Some(body) = phys.get(id) else { continue };
        let offset = body.pos - center_m;
        let dist = offset.length();
        let dir = if dist > 1e-6 { offset / dist } else { Vec3::Y };
        let mass = body.mass();
        let speed = BLAST_POWER / dist.max(BLAST_MIN_DIST_M) / mass.sqrt();

        let h = small_hash(seed, i as u32);
        let spin = Vec3::new(
            ((h & 0xFF) as f32 / 255.0 - 0.5) * 2.0,
            (((h >> 8) & 0xFF) as f32 / 255.0 - 0.5) * 2.0,
            (((h >> 16) & 0xFF) as f32 / 255.0 - 0.5) * 2.0,
        ) * BLAST_SPIN_MAX;

        // `apply_impulse` scales its first argument by inverse mass, so
        // passing `dir * speed * mass` yields exactly `dir * speed` velocity.
        phys.apply_impulse(id, dir * speed * mass, spin);
    }
}

/// Carve a sphere, detach anything left unsupported, give the new debris a
/// blast impulse, and wake any resting bodies the carve disturbed. `seed`
/// drives the (deterministic) per-body angular kick.
pub fn blast(
    world: &mut World,
    phys: &mut PhysicsWorld,
    registry: &MaterialRegistry,
    center_m: Vec3,
    radius_m: f32,
    seed: u32,
) -> CarveResult {
    let carve = carve_sphere(world, center_m, radius_m);
    let ids = detach_unsupported(world, phys, registry, carve.region);
    apply_blast_impulse(phys, &ids, center_m, seed);

    let s = world.cfg.voxel_size_m;
    phys.wake_region(carve.region.0.as_vec3() * s, carve.region.1.as_vec3() * s);
    carve
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::WorldConfig;

    const STONE: Voxel = Voxel(1);

    fn registry() -> MaterialRegistry {
        MaterialRegistry::from_toml_str(
            r#"
            [[material]]
            name = "stone"
            color = [0.5, 0.5, 0.5]
            density = 2600.0
            strength = 8.0
            "#,
            "test.toml",
        )
        .expect("registry")
    }

    fn test_world() -> World {
        World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [32.0, 32.0, 32.0],
            ..WorldConfig::default()
        })
    }

    /// Two 1-voxel pillars (x=5 and x=20, z=5) rising from the floor (y=0)
    /// to y=10, bridged by a slab at y=10 spanning both.
    fn two_pillar_bridge() -> World {
        let mut world = test_world();
        world.fill_box(IVec3::new(5, 0, 5), IVec3::new(6, 10, 6), STONE);
        world.fill_box(IVec3::new(20, 0, 5), IVec3::new(21, 10, 6), STONE);
        world.fill_box(IVec3::new(5, 10, 5), IVec3::new(21, 11, 6), STONE);
        world
    }

    const WHOLE_WORLD: Region = (IVec3::new(0, 0, 0), IVec3::new(32, 32, 32));

    #[test]
    fn two_pillars_cut_one_nothing_falls() {
        let mut world = two_pillar_bridge();
        world.fill_box(IVec3::new(5, 0, 5), IVec3::new(6, 10, 6), AIR);

        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let ids = detach_unsupported(&mut world, &mut phys, &reg, WHOLE_WORLD);

        assert!(ids.is_empty(), "bridge still anchored via the other pillar");
        assert_eq!(world.get_voxel(IVec3::new(12, 10, 5)), STONE);
    }

    #[test]
    fn cut_both_slab_detaches() {
        let mut world = two_pillar_bridge();
        world.fill_box(IVec3::new(5, 0, 5), IVec3::new(6, 10, 6), AIR);
        world.fill_box(IVec3::new(20, 0, 5), IVec3::new(21, 10, 6), AIR);

        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let ids = detach_unsupported(&mut world, &mut phys, &reg, WHOLE_WORLD);
        assert_eq!(ids.len(), 1, "exactly one detached component");

        let body = phys.get(ids[0]).expect("alive");
        let expected_mass = 16.0 * 2600.0; // 16x1x1 slab, 1 m^3 voxels
        assert!(
            (body.mass() - expected_mass).abs() / expected_mass < 1e-4,
            "mass {} vs expected {expected_mass}",
            body.mass()
        );
        assert_eq!(world.get_voxel(IVec3::new(12, 10, 5)), AIR);
    }

    #[test]
    fn floating_blob_detaches() {
        let mut world = test_world();
        // A floor-anchored block, a thin 1-voxel bridge, and a knob just
        // past it — all interior to the region a carve of the bridge will
        // produce, so nothing here touches the region's shell by accident.
        world.fill_box(IVec3::new(0, 0, 0), IVec3::new(10, 10, 10), STONE);
        world.fill_box(IVec3::new(10, 4, 4), IVec3::new(11, 6, 6), STONE); // bridge, 4 voxels
        world.fill_box(IVec3::new(11, 4, 4), IVec3::new(12, 6, 6), STONE); // knob, 4 voxels

        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let carve = carve_sphere(&mut world, Vec3::new(10.5, 5.0, 5.0), 0.8);
        assert_eq!(carve.removed.len(), 4, "must remove exactly the bridge");

        let ids = detach_unsupported(&mut world, &mut phys, &reg, carve.region);
        assert_eq!(ids.len(), 1, "knob must detach as its own body");
        let body = phys.get(ids[0]).expect("alive");
        assert_eq!(body.grid.solid_count(), 4);
        // The anchored block must survive untouched.
        assert_eq!(world.get_voxel(IVec3::new(5, 5, 5)), STONE);
    }

    #[test]
    fn small_fragments_discarded() {
        let mut world = test_world();
        world.fill_box(IVec3::new(0, 0, 0), IVec3::new(2, 2, 2), STONE);
        world.fill_box(IVec3::new(5, 5, 5), IVec3::new(6, 6, 6), STONE); // isolated single voxel

        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let region = (IVec3::new(0, 0, 0), IVec3::new(10, 10, 10));
        let ids = detach_unsupported(&mut world, &mut phys, &reg, region);

        assert!(ids.is_empty(), "1 voxel is below DEBRIS_MIN_VOXELS");
        assert_eq!(world.get_voxel(IVec3::new(5, 5, 5)), AIR, "must be removed");
        assert_eq!(phys.body_count(), 0);
    }

    #[test]
    fn carve_records_removed() {
        let mut world = test_world();
        world.fill_box(IVec3::new(0, 0, 0), IVec3::new(10, 10, 10), STONE);

        let center = Vec3::new(5.0, 5.0, 5.0);
        let radius = 3.0;
        let carve = carve_sphere(&mut world, center, radius);

        let mut expected = 0usize;
        for z in 0..10 {
            for y in 0..10 {
                for x in 0..10 {
                    let v = IVec3::new(x, y, z);
                    if (voxel_center_m(v, 1.0) - center).length() <= radius {
                        expected += 1;
                        assert_eq!(world.get_voxel(v), AIR, "must be carved: {v}");
                    }
                }
            }
        }
        assert_eq!(carve.removed.len(), expected);
    }

    #[test]
    fn blast_wakes_and_moves_spawned_debris() {
        let mut world = two_pillar_bridge();
        let reg = registry();
        let mut phys = PhysicsWorld::new();

        // Blast pillar A's base away entirely (radius covers the 1x10x1
        // column); pillar B remains, so nothing should detach yet — verify
        // the plain carve+detach compose correctly through `blast`.
        let carve = blast(
            &mut world,
            &mut phys,
            &reg,
            Vec3::new(5.5, 15.0, 5.5),
            12.0,
            7,
        );
        assert!(carve.removed.len() > 10, "must remove a large chunk");
        // Both pillars and the slab are gone or detached; whatever spawned
        // must be awake with a nonzero blast velocity.
        for (_, body) in phys.iter() {
            assert!(!body.sleep.asleep);
            assert!(body.vel.length() > 0.1, "blast must impart velocity");
        }
    }
}
