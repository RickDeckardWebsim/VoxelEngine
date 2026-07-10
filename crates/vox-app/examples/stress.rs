//! Headless performance probe: not a correctness test, a measurement tool.
//! Run with `cargo run -p vox-app --release --example stress` for real
//! numbers -- debug builds are 10-50x slower and not representative.
//!
//! Measures the three things "scaling" actually depends on: physics step
//! time as debris piles up, connectivity-check cost for large destructive
//! edits, and per-fragment meshing cost. Prints a plain-text report; no
//! assertions, since this is for humans deciding what to optimize next.

use std::time::Instant;

use glam::{IVec3, Vec3};
use vox_core::consts::PHYSICS_DT;
use vox_core::{MaterialRegistry, WorldConfig};
use vox_mesh::{VoxelSlab, mesh_slab};
use vox_physics::{Body, PhysicsWorld, VoxelGrid};
use vox_world::{Voxel, World};

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
        "stress.toml",
    )
    .expect("registry")
}

fn percentile(sorted_ms: &[f32], p: f32) -> f32 {
    let idx = ((sorted_ms.len() as f32 - 1.0) * p).round() as usize;
    sorted_ms[idx]
}

fn report(label: &str, mut samples_ms: Vec<f32>) {
    samples_ms.sort_by(|a, b| a.total_cmp(b));
    let sum: f32 = samples_ms.iter().sum();
    let avg = sum / samples_ms.len() as f32;
    println!(
        "{label:<28} avg={:>7.3}ms  p50={:>7.3}ms  p95={:>7.3}ms  max={:>7.3}ms  n={}",
        avg,
        percentile(&samples_ms, 0.50),
        percentile(&samples_ms, 0.95),
        samples_ms.last().copied().unwrap_or(0.0),
        samples_ms.len()
    );
}

/// A world with a large contiguous terrain slab -- the "real Teardown-scale
/// map" case, not an isolated test structure.
fn terrain_world(voxel_size_m: f32, extent_xz: f32, floor_thickness_m: f32) -> World {
    let mut world = World::new(WorldConfig {
        voxel_size_m,
        extent_m: [extent_xz, 40.0, extent_xz],
        ..WorldConfig::default()
    });
    let (_, max) = world.bounds_voxels();
    let floor_top = (floor_thickness_m / voxel_size_m) as i32;
    world.fill_box(IVec3::ZERO, IVec3::new(max.x, floor_top, max.z), STONE);
    world
}

fn stress_physics_pile(reg: &MaterialRegistry, body_count: usize, cube_vox: i32) {
    let s = 0.2;
    let world = terrain_world(s, 64.0, 4.0);
    let mut phys = PhysicsWorld::new();

    let voxels = vec![STONE; (cube_vox * cube_vox * cube_vox) as usize];
    let dims = IVec3::splat(cube_vox);
    let footprint = (body_count as f32).sqrt().ceil() as i32;
    for i in 0..body_count {
        let gx = i as i32 % footprint;
        let gz = i as i32 / footprint;
        let origin = Vec3::new(
            10.0 + gx as f32 * (cube_vox as f32 * s + 0.3),
            10.0 + (i as f32 * 0.01), // slight height stagger to avoid a perfectly synced drop
            10.0 + gz as f32 * (cube_vox as f32 * s + 0.3),
        );
        let grid = VoxelGrid::new(dims, voxels.clone());
        if let Some(body) = Body::from_grid(grid, reg, s, origin) {
            phys.spawn(body);
        }
    }

    let mut settling = Vec::new();
    let mut settled = Vec::new();
    let total_steps = 600; // 10s at 60Hz
    for step in 0..total_steps {
        let t0 = Instant::now();
        phys.step(&world, PHYSICS_DT);
        let dt = t0.elapsed().as_secs_f32() * 1000.0;
        // Front half: bodies landing and grinding against each other while
        // awake -- the many-contact solver worst case, and the phase the
        // player actually feels right after a blast. Back half: steady
        // state, mostly asleep -- the "long after the blast" cost.
        if step < total_steps / 2 {
            settling.push(dt);
        } else {
            settled.push(dt);
        }
    }
    report(
        &format!("physics/{body_count}-bodies-settling"),
        settling,
    );
    report(
        &format!("physics/{body_count}-bodies-settled"),
        settled,
    );
    println!(
        "  (awake={}, total={})",
        phys.awake_count(),
        phys.body_count()
    );
}

fn stress_blast_in_huge_terrain(reg: &MaterialRegistry) {
    let s = 0.2;
    let mut world = terrain_world(s, 128.0, 6.0);
    let mut phys = PhysicsWorld::new();

    let mut times = Vec::new();
    for i in 0..10 {
        let center = Vec3::new(20.0 + i as f32 * 8.0, 5.0, 20.0);
        let t0 = Instant::now();
        vox_physics::blast(&mut world, &mut phys, reg, center, 3.0, 40.0, i as u32);
        times.push(t0.elapsed().as_secs_f32() * 1000.0);
    }
    report("blast-in-huge-terrain", times);
}

fn stress_laser_through_terrain(reg: &MaterialRegistry) {
    let s = 0.2;
    let mut world = terrain_world(s, 128.0, 6.0);
    let mut phys = PhysicsWorld::new();

    let mut carve_times = Vec::new();
    let mut detach_times = Vec::new();
    let mut removed_counts = Vec::new();
    for i in 0..10 {
        let start = Vec3::new(0.0, 5.0, 10.0 + i as f32 * 8.0);
        let end = start + Vec3::new(10_000.0, 0.0, 0.0);

        let t0 = Instant::now();
        let carve = vox_physics::carve_capsule(&mut world, start, end, 1.5);
        carve_times.push(t0.elapsed().as_secs_f32() * 1000.0);
        removed_counts.push(carve.removed.len());

        let removed: Vec<_> = carve.removed.iter().map(|&(v, _)| v).collect();
        let t1 = Instant::now();
        vox_physics::detach_unsupported(&mut world, &mut phys, reg, &removed);
        detach_times.push(t1.elapsed().as_secs_f32() * 1000.0);
    }
    println!("  removed voxel counts: {removed_counts:?}");
    report("laser/carve_capsule", carve_times);
    report("laser/detach_unsupported", detach_times);
}

fn stress_meshing(sizes: &[i32]) {
    for &n in sizes {
        let dims = IVec3::splat(n);
        let voxels = vec![STONE; (n * n * n) as usize];
        let mut times = Vec::new();
        for _ in 0..20 {
            let t0 = Instant::now();
            let slab = VoxelSlab::from_grid(dims, &voxels);
            let _mesh = mesh_slab(&slab, IVec3::ZERO);
            times.push(t0.elapsed().as_secs_f32() * 1000.0);
        }
        report(&format!("mesh-slab/{n}^3-cube"), times);
    }
}

fn main() {
    let reg = registry();

    println!("== physics: settling pile, varying body counts ==");
    for &count in &[25usize, 100, 300] {
        stress_physics_pile(&reg, count, 4);
    }

    println!("\n== destruction against a large contiguous terrain mass ==");
    stress_blast_in_huge_terrain(&reg);
    stress_laser_through_terrain(&reg);

    println!("\n== meshing cost per debris fragment ==");
    stress_meshing(&[4, 10, 20, 40]);
}
