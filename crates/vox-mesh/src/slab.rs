//! [`VoxelSlab`]: a dense copy of a voxel region plus a 1-voxel shell.
//!
//! Slabs are extracted on the main thread (a bounded memcpy-scale copy) and
//! meshed on worker threads — copying is the thread-safety strategy, so the
//! mesher never touches live world data. The same type wraps debris-body
//! grids, giving chunks and debris one meshing code path.

use glam::IVec3;
use vox_world::{AIR, Voxel, World};

/// Maximum inner dimension of a slab per axis. Mesh vertex positions are
/// quantized to `u8` corner coordinates (`0..=dims`), so `dims <= 254`.
pub const MAX_SLAB_DIM: i32 = 254;

/// A dense copy of a voxel region including a 1-voxel shell on every side.
pub struct VoxelSlab {
    /// World-voxel position of the inner region's minimum corner.
    pub inner_min: IVec3,
    /// Inner region size in voxels (shell excluded).
    pub inner_dims: IVec3,
    /// Padded contents, `(inner_dims + 2)` per axis, x-major rows.
    data: Vec<Voxel>,
}

impl VoxelSlab {
    /// Copy the world region `[inner_min, inner_min + inner_dims)` plus its
    /// 1-voxel shell. Out-of-world samples read as air.
    pub fn extract(world: &World, inner_min: IVec3, inner_dims: IVec3) -> Self {
        let mut slab = Self::filled(inner_min, inner_dims, AIR);
        let padded = slab.padded_dims();
        for y in 0..padded.y {
            for z in 0..padded.z {
                for x in 0..padded.x {
                    let rel = IVec3::new(x, y, z) - IVec3::ONE;
                    let v = world.get_voxel(inner_min + rel);
                    if v != AIR {
                        let idx = slab.index(rel);
                        slab.data[idx] = v;
                    }
                }
            }
        }
        slab
    }

    /// Wrap a dense body grid (x-major rows, `dims` sized) with an air shell.
    /// `data` is indexed `x + z*dx + y*dx*dz`, matching `VoxelGrid`.
    pub fn from_grid(dims: IVec3, data: &[Voxel]) -> Self {
        debug_assert_eq!(
            data.len() as i64,
            dims.x as i64 * dims.y as i64 * dims.z as i64
        );
        let mut slab = Self::filled(IVec3::ZERO, dims, AIR);
        for y in 0..dims.y {
            for z in 0..dims.z {
                for x in 0..dims.x {
                    let src = (x + z * dims.x + y * dims.x * dims.z) as usize;
                    if data[src] != AIR {
                        let idx = slab.index(IVec3::new(x, y, z));
                        slab.data[idx] = data[src];
                    }
                }
            }
        }
        slab
    }

    /// An all-`fill` slab (shell included).
    fn filled(inner_min: IVec3, inner_dims: IVec3, fill: Voxel) -> Self {
        assert!(
            inner_dims.cmpgt(IVec3::ZERO).all()
                && inner_dims.cmple(IVec3::splat(MAX_SLAB_DIM)).all(),
            "slab inner dims out of range: {inner_dims}"
        );
        let padded = inner_dims + 2;
        let len = (padded.x * padded.y * padded.z) as usize;
        Self {
            inner_min,
            inner_dims,
            data: vec![fill; len],
        }
    }

    /// Padded size (`inner_dims + 2`).
    fn padded_dims(&self) -> IVec3 {
        self.inner_dims + 2
    }

    /// Dense index of slab-relative position `rel` (each axis `-1..=inner`).
    #[inline]
    fn index(&self, rel: IVec3) -> usize {
        let p = rel + IVec3::ONE;
        let d = self.padded_dims();
        debug_assert!(
            p.cmpge(IVec3::ZERO).all() && p.cmplt(d).all(),
            "slab-relative position out of range: {rel}"
        );
        (p.x + p.z * d.x + p.y * d.x * d.z) as usize
    }

    /// Voxel at slab-relative position `rel` (each axis `-1..=inner_dims`).
    #[inline]
    pub fn get(&self, rel: IVec3) -> Voxel {
        self.data[self.index(rel)]
    }

    /// True when the voxel at `rel` is solid (non-air).
    #[inline]
    pub fn solid(&self, rel: IVec3) -> bool {
        self.get(rel) != AIR
    }

    /// Like [`solid`](Self::solid) but treats water (material 9) as
    /// non-solid. Used for face culling so that faces between solid
    /// terrain and water ARE generated (visible through translucent water),
    /// while faces between water and water, or between two solid voxels,
    /// are still culled.
    #[inline]
    pub fn opaque(&self, rel: IVec3) -> bool {
        let v = self.get(rel);
        v != AIR && v != vox_world::Voxel(9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::WorldConfig;

    const STONE: Voxel = Voxel(1);

    #[test]
    fn extract_copies_region_and_shell() {
        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [64.0; 3],
            ..WorldConfig::default()
        });
        world.set_voxel(IVec3::new(10, 10, 10), STONE);
        world.set_voxel(IVec3::new(9, 10, 10), STONE); // will land in the shell

        let slab = VoxelSlab::extract(&world, IVec3::new(10, 10, 10), IVec3::splat(4));
        assert_eq!(slab.get(IVec3::new(0, 0, 0)), STONE);
        assert_eq!(slab.get(IVec3::new(-1, 0, 0)), STONE, "shell sees neighbor");
        assert_eq!(slab.get(IVec3::new(1, 0, 0)), AIR);
        assert_eq!(slab.get(IVec3::new(4, 4, 4)), AIR, "far shell corner");
    }

    #[test]
    fn extract_outside_world_reads_air() {
        let world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [32.0; 3],
            ..WorldConfig::default()
        });
        // Region flush against the world origin: shell samples negative coords.
        let slab = VoxelSlab::extract(&world, IVec3::ZERO, IVec3::splat(2));
        assert_eq!(slab.get(IVec3::splat(-1)), AIR);
    }

    #[test]
    fn from_grid_wraps_with_air_shell() {
        // 2x1x1 grid of stone.
        let dims = IVec3::new(2, 1, 1);
        let data = vec![STONE, STONE];
        let slab = VoxelSlab::from_grid(dims, &data);
        assert_eq!(slab.get(IVec3::new(0, 0, 0)), STONE);
        assert_eq!(slab.get(IVec3::new(1, 0, 0)), STONE);
        assert_eq!(slab.get(IVec3::new(2, 0, 0)), AIR, "shell");
        assert_eq!(slab.get(IVec3::new(-1, 0, 0)), AIR, "shell");
    }
}
