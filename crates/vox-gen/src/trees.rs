//! Procedural trees, parameterized entirely in meters.
//!
//! Generation is split in two: [`plan_trees`] decides *where and what* purely
//! from seed + terrain (scale-free, trivially testable), and
//! [`stamp_tree`] writes one planned tree into a world at that world's voxel
//! scale. At 0.1 m voxels a tree is ~80 voxels tall with visible branches; at
//! 1.0 m it degrades gracefully into a chunky Minecraft-style tree.

use glam::{IVec3, Vec2, Vec3};
use vox_core::{MaterialRegistry, WorldConfig, voxel_at, voxel_center_m};
use vox_world::{AIR, Voxel, World};

use crate::noise::{Fbm, hash2};
use crate::terrain::TerrainGen;

/// Tree material ids resolved once from the registry.
#[derive(Copy, Clone, Debug)]
pub struct TreeMaterials {
    pub wood: Voxel,
    pub leaves: Voxel,
}

impl TreeMaterials {
    pub fn from_registry(reg: &MaterialRegistry) -> Result<Self, vox_core::CoreError> {
        let id = |name: &str| -> Result<Voxel, vox_core::CoreError> {
            reg.id_by_name(name)
                .map(|m| Voxel(m.0))
                .ok_or_else(|| vox_core::CoreError::Asset {
                    path: "assets/materials".into(),
                    reason: format!("trees require core material `{name}`"),
                })
        };
        Ok(Self {
            wood: id("wood")?,
            leaves: id("leaves")?,
        })
    }
}

/// One planned tree, all quantities in meters (scale-free).
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct TreeInstance {
    /// Trunk axis position.
    pub x_m: f32,
    pub z_m: f32,
    /// Ground height at the trunk (terrain surface).
    pub base_y_m: f32,
    /// Total tree height above ground.
    pub height_m: f32,
    /// Per-tree hash driving branches and canopy shapes.
    pub tree_seed: u32,
}

/// Placement grid cell size in meters.
const CELL_M: f32 = 8.0;
/// Forest density threshold: higher = sparser forests.
const DENSITY_THRESHOLD: f32 = 0.15;
/// Maximum terrain slope (rise over 1 m run) a tree tolerates.
const MAX_SLOPE: f32 = 0.7; // ≈ 35°

#[inline]
fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// Decide tree positions for a world, purely from config + terrain.
pub fn plan_trees(cfg: &WorldConfig, terrain: &TerrainGen) -> Vec<TreeInstance> {
    let seed = (cfg.seed as u32) ^ ((cfg.seed >> 32) as u32) ^ 0x7EE5;
    let density = Fbm::new(3, seed ^ 0xD375);
    let cells_x = (cfg.extent_m[0] / CELL_M) as i32;
    let cells_z = (cfg.extent_m[2] / CELL_M) as i32;
    let mut trees = Vec::new();

    for cj in 0..cells_z {
        for ci in 0..cells_x {
            let h = hash2(ci, cj, seed);
            // Jittered position inside the cell, 1 m margin.
            let x = ci as f32 * CELL_M + 1.0 + unit(h) * (CELL_M - 2.0);
            let z = cj as f32 * CELL_M + 1.0 + unit(h.rotate_left(11)) * (CELL_M - 2.0);
            // World-border margin.
            if x < 2.0 || z < 2.0 || x > cfg.extent_m[0] - 2.0 || z > cfg.extent_m[2] - 2.0 {
                continue;
            }
            // Forest density field.
            if density.sample2(Vec2::new(x, z) / 60.0) < DENSITY_THRESHOLD {
                continue;
            }
            // Slope check: ±1 m samples.
            let base = terrain.height_m(x, z);
            let slope_x = (terrain.height_m(x + 1.0, z) - terrain.height_m(x - 1.0, z)).abs() / 2.0;
            let slope_z = (terrain.height_m(x, z + 1.0) - terrain.height_m(x, z - 1.0)).abs() / 2.0;
            if slope_x.max(slope_z) > MAX_SLOPE {
                continue;
            }
            // Headroom: don't plant into the world ceiling.
            let height = 6.0 + 4.0 * unit(h.rotate_left(19));
            if base + height + 3.0 > cfg.extent_m[1] {
                continue;
            }
            trees.push(TreeInstance {
                x_m: x,
                z_m: z,
                base_y_m: base,
                height_m: height,
                tree_seed: h,
            });
        }
    }
    trees
}

/// True if `existing` may be replaced by wood.
#[inline]
fn wood_replaceable(existing: Voxel, mats: TreeMaterials) -> bool {
    existing == AIR || existing == mats.leaves
}

/// Stamp one planned tree into the world at the world's voxel scale.
pub fn stamp_tree(world: &mut World, tree: &TreeInstance, mats: TreeMaterials) {
    let s = world.cfg.voxel_size_m;
    let h = tree.height_m;
    let axis = Vec2::new(tree.x_m, tree.z_m);

    // Trunk: tapered discs from ground to height.
    let r_base = 0.30 + 0.15 * (h / 10.0);
    let mut y = tree.base_y_m;
    while y < tree.base_y_m + h {
        let frac = (y - tree.base_y_m) / h;
        let radius = r_base + (0.10 - r_base) * frac;
        stamp_disc(world, axis, y, radius, mats.wood, |v| {
            wood_replaceable(v, mats)
        });
        y += s;
    }

    // Branches with canopies at their tips.
    let n_branches = 3 + (tree.tree_seed % 3) as usize;
    for k in 0..n_branches {
        let hb = hash2(tree.tree_seed as i32, k as i32, 0xB4A2C);
        let frac = 0.55 + 0.12 * k as f32 / n_branches as f32 + 0.08 * unit(hb);
        let yaw = unit(hb.rotate_left(7)) * std::f32::consts::TAU;
        let pitch = (25.0 + 20.0 * unit(hb.rotate_left(13))).to_radians();
        let len = 0.18 * h + 0.8 * unit(hb.rotate_left(23));
        let dir = Vec3::new(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        );
        let start = Vec3::new(axis.x, tree.base_y_m + frac * h, axis.y);
        let tip = stamp_line(world, start, dir, len, mats, s);
        let rc = 1.3 + 0.9 * unit(hb.rotate_left(29));
        stamp_ellipsoid(world, tip, Vec3::new(rc, 0.75 * rc, rc), mats.leaves);
    }

    // Crown canopy on top.
    let crown_r = 1.5 + 0.7 * unit(tree.tree_seed.rotate_left(27));
    let crown = Vec3::new(axis.x, tree.base_y_m + h, axis.y);
    stamp_ellipsoid(
        world,
        crown,
        Vec3::new(crown_r, 0.75 * crown_r, crown_r),
        mats.leaves,
    );
}

/// Stamp a horizontal disc of `material` at height `y_m`. The voxel containing
/// the axis is always stamped; larger radii include voxels whose centers fall
/// inside `radius_m`.
fn stamp_disc(
    world: &mut World,
    axis: Vec2,
    y_m: f32,
    radius_m: f32,
    material: Voxel,
    replaceable: impl Fn(Voxel) -> bool,
) {
    let s = world.cfg.voxel_size_m;
    let center = voxel_at(Vec3::new(axis.x, y_m, axis.y), s);
    let r_vox = (radius_m / s).ceil() as i32;
    for dz in -r_vox..=r_vox {
        for dx in -r_vox..=r_vox {
            let v = center + IVec3::new(dx, 0, dz);
            let c = voxel_center_m(v, s);
            let in_disc = (Vec2::new(c.x, c.z) - axis).length() <= radius_m;
            if (in_disc || (dx == 0 && dz == 0)) && replaceable(world.get_voxel(v)) {
                world.set_voxel(v, material);
            }
        }
    }
}

/// Stamp a wood line from `start_m` along `dir` for `len_m`; returns the tip.
fn stamp_line(
    world: &mut World,
    start_m: Vec3,
    dir: Vec3,
    len_m: f32,
    mats: TreeMaterials,
    s: f32,
) -> Vec3 {
    let steps = (len_m / (s * 0.5)).ceil().max(1.0) as i32;
    let mut p = start_m;
    for i in 0..=steps {
        p = start_m + dir * (len_m * i as f32 / steps as f32);
        let v = voxel_at(p, s);
        if wood_replaceable(world.get_voxel(v), mats) {
            world.set_voxel(v, mats.wood);
        }
    }
    p
}

/// Fill an axis-aligned ellipsoid with leaves, only into air.
fn stamp_ellipsoid(world: &mut World, center_m: Vec3, radii_m: Vec3, leaves: Voxel) {
    let s = world.cfg.voxel_size_m;
    let min = voxel_at(center_m - radii_m, s);
    let max = voxel_at(center_m + radii_m, s);
    for y in min.y..=max.y {
        for z in min.z..=max.z {
            for x in min.x..=max.x {
                let v = IVec3::new(x, y, z);
                let d = (voxel_center_m(v, s) - center_m) / radii_m;
                if d.length_squared() <= 1.0 && world.get_voxel(v) == AIR {
                    world.set_voxel(v, leaves);
                }
            }
        }
    }
    // Guarantee at least the center voxel so coarse scales get a canopy.
    let c = voxel_at(center_m, s);
    if world.get_voxel(c) == AIR {
        world.set_voxel(c, leaves);
    }
}

/// Plan and stamp all trees for a world. Returns the number planted.
pub fn generate_trees(world: &mut World, terrain: &TerrainGen, mats: TreeMaterials) -> usize {
    let trees = plan_trees(&world.cfg, terrain);
    for tree in &trees {
        stamp_tree(world, tree, mats);
    }
    trees.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::TerrainMaterials;

    fn mats() -> TreeMaterials {
        TreeMaterials {
            wood: Voxel(5),
            leaves: Voxel(6),
        }
    }

    fn cfg(voxel_size_m: f32) -> WorldConfig {
        WorldConfig {
            seed: 777,
            voxel_size_m,
            extent_m: [64.0, 40.0, 64.0],
        }
    }

    #[test]
    fn planning_is_deterministic_and_scale_free() {
        let fine = cfg(0.1);
        let coarse = cfg(1.0);
        let terrain = TerrainGen::new(&fine);
        let a = plan_trees(&fine, &terrain);
        let b = plan_trees(&fine, &terrain);
        assert_eq!(a, b, "same inputs must plan identical trees");
        assert!(!a.is_empty(), "seed 777 must plant at least one tree");

        // The plan is meters + seed only: voxel scale must not affect it.
        let c = plan_trees(&coarse, &TerrainGen::new(&coarse));
        assert_eq!(a, c, "voxel scale must not change the plan");
    }

    #[test]
    fn heights_are_in_the_specified_band() {
        let cfg = cfg(0.1);
        let terrain = TerrainGen::new(&cfg);
        for tree in plan_trees(&cfg, &terrain) {
            assert!(
                (6.0..=10.0).contains(&tree.height_m),
                "height {} outside 6-10 m",
                tree.height_m
            );
        }
    }

    /// Stamp the same instance at both scales on flat ground: the wood column
    /// must reach the same height in meters within 1.5 m.
    #[test]
    fn stamped_height_matches_across_scales() {
        let tree = TreeInstance {
            x_m: 20.0,
            z_m: 20.0,
            base_y_m: 10.0,
            height_m: 8.0,
            tree_seed: 0xC0FFEE,
        };
        let mut tops = Vec::new();
        for s in [0.1_f32, 1.0] {
            let mut world = World::new(cfg(s));
            // Flat stone ground up to 10 m.
            let ground_top = (10.0 / s) as i32;
            let (_, max) = world.bounds_voxels();
            world.fill_box(IVec3::ZERO, IVec3::new(max.x, ground_top, max.z), Voxel(1));
            stamp_tree(&mut world, &tree, mats());

            // Scan the trunk column for the topmost wood voxel.
            let wx = (tree.x_m / s) as i32;
            let wz = (tree.z_m / s) as i32;
            let mut top_m = 0.0;
            for wy in (0..max.y).rev() {
                if world.get_voxel(IVec3::new(wx, wy, wz)) == mats().wood {
                    top_m = (wy + 1) as f32 * s;
                    break;
                }
            }
            assert!(top_m > 0.0, "no wood found at scale {s}");
            tops.push(top_m);
        }
        let expected = 10.0 + 8.0;
        for (i, top) in tops.iter().enumerate() {
            assert!(
                (top - expected).abs() <= 1.5,
                "scale {i}: trunk top {top} m vs expected {expected} m"
            );
        }
        assert!(
            (tops[0] - tops[1]).abs() <= 1.5,
            "scales disagree: {tops:?}"
        );
    }

    #[test]
    fn leaves_exist_near_the_crown_and_never_replace_solids() {
        let s = 0.1;
        let mut world = World::new(cfg(s));
        let (_, max) = world.bounds_voxels();
        world.fill_box(
            IVec3::ZERO,
            IVec3::new(max.x, (10.0 / s) as i32, max.z),
            Voxel(1),
        );
        // A stone pillar right where the crown will be.
        let pillar_x = (20.0 / s) as i32;
        let pillar_z = (20.0 / s) as i32 + 5;
        world.fill_box(
            IVec3::new(pillar_x, 0, pillar_z),
            IVec3::new(pillar_x + 1, (20.0 / s) as i32, pillar_z + 1),
            Voxel(1),
        );

        let tree = TreeInstance {
            x_m: 20.0,
            z_m: 20.0,
            base_y_m: 10.0,
            height_m: 8.0,
            tree_seed: 0xC0FFEE,
        };
        stamp_tree(&mut world, &tree, mats());

        // Leaves near the crown center.
        let crown = voxel_at(Vec3::new(20.0, 18.0, 20.0), s);
        let mut leaf_count = 0;
        for dy in -10i32..=10 {
            for dz in -20i32..=20 {
                for dx in -20i32..=20 {
                    if world.get_voxel(crown + IVec3::new(dx, dy, dz)) == mats().leaves {
                        leaf_count += 1;
                    }
                }
            }
        }
        assert!(leaf_count > 50, "crown region has only {leaf_count} leaves");

        // The stone pillar survived untouched through the canopy region.
        for wy in 0..(20.0 / s) as i32 {
            assert_eq!(
                world.get_voxel(IVec3::new(pillar_x, wy, pillar_z)),
                Voxel(1),
                "pillar overwritten at y={wy}"
            );
        }
        // Ground under the trunk survived (wood never replaces stone).
        let under = voxel_at(Vec3::new(20.0, 9.95, 20.0), s);
        assert_eq!(world.get_voxel(under), Voxel(1));
    }

    #[test]
    fn full_generation_smoke_at_coarse_scale() {
        let cfg1 = cfg(1.0);
        let terrain = TerrainGen::new(&cfg1);
        let mut world = World::new(cfg1);
        terrain.generate(
            &mut world,
            TerrainMaterials {
                stone: Voxel(1),
                dirt: Voxel(2),
                grass: Voxel(3),
            },
        );
        let planted = generate_trees(&mut world, &terrain, mats());
        assert!(planted > 0, "coarse world must still get trees");
    }
}
