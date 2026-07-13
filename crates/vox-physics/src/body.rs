//! Rigid bodies made of voxels: local grids, mass properties, and the body
//! arena.

use glam::{IVec3, Mat3, Quat, Vec3};
use vox_core::MaterialRegistry;
use vox_world::{AIR, Voxel};

/// A dense little voxel grid owned by a body. Indexed `x + z*dx + y*dx*dz`.
#[derive(Clone, Debug)]
pub struct VoxelGrid {
    pub dims: IVec3,
    pub voxels: Vec<Voxel>,
    /// Per-voxel damage, 0.0 (pristine) to 1.0 (crumbled). Parallel to
    /// `voxels`, same length. Only meaningful on debris bodies.
    pub damage: Vec<f32>,
}

impl VoxelGrid {
    pub fn new(dims: IVec3, voxels: Vec<Voxel>) -> Self {
        debug_assert_eq!(
            voxels.len() as i64,
            dims.x as i64 * dims.y as i64 * dims.z as i64
        );
        let damage = vec![0.0; voxels.len()];
        Self { dims, voxels, damage }
    }

    /// Construct a grid with an existing damage field (e.g. carrying damage
    /// through `split_components`). `damage.len()` must equal `voxels.len()`.
    pub fn new_with_damage(dims: IVec3, voxels: Vec<Voxel>, damage: Vec<f32>) -> Self {
        debug_assert_eq!(
            voxels.len() as i64,
            dims.x as i64 * dims.y as i64 * dims.z as i64
        );
        debug_assert_eq!(voxels.len(), damage.len());
        Self { dims, voxels, damage }
    }

    #[inline]
    fn index(&self, p: IVec3) -> usize {
        (p.x + p.z * self.dims.x + p.y * self.dims.x * self.dims.z) as usize
    }

    /// Voxel at `p`; out-of-bounds reads as air.
    #[inline]
    pub fn get(&self, p: IVec3) -> Voxel {
        if p.cmpge(IVec3::ZERO).all() && p.cmplt(self.dims).all() {
            self.voxels[self.index(p)]
        } else {
            AIR
        }
    }

    /// Set voxel at `p`; out-of-bounds writes are silently dropped.
    #[inline]
    pub fn set(&mut self, p: IVec3, v: Voxel) {
        if p.cmpge(IVec3::ZERO).all() && p.cmplt(self.dims).all() {
            let idx = self.index(p);
            self.voxels[idx] = v;
        }
    }

    #[inline]
    pub fn solid(&self, p: IVec3) -> bool {
        self.get(p) != AIR
    }

    /// Number of solid voxels.
    pub fn solid_count(&self) -> usize {
        self.voxels.iter().filter(|v| **v != AIR).count()
    }

    /// Damage at `p`; out-of-bounds reads as 0.0 (pristine).
    #[inline]
    pub fn damage_at(&self, p: IVec3) -> f32 {
        if p.cmpge(IVec3::ZERO).all() && p.cmplt(self.dims).all() {
            self.damage[self.index(p)]
        } else {
            0.0
        }
    }

    /// Add damage to a solid voxel. Returns true if the voxel accepted damage.
    #[inline]
    pub fn add_damage(&mut self, p: IVec3, amount: f32) -> bool {
        if p.cmpge(IVec3::ZERO).all() && p.cmplt(self.dims).all() {
            let idx = self.index(p);
            if self.voxels[idx] != AIR {
                self.damage[idx] = (self.damage[idx] + amount).min(1.0);
                return true;
            }
        }
        false
    }

    /// Decay all damage by `decay_rate * dt`. Returns true if any damage changed.
    pub fn tick_damage_decay(&mut self, dt: f32, decay_rate: f32) -> bool {
        let mut changed = false;
        for d in &mut self.damage {
            if *d > 0.0 {
                *d = (*d - decay_rate * dt).max(0.0);
                changed = true;
            }
        }
        changed
    }

    /// Whether any voxel has nonzero damage.
    pub fn has_damage(&self) -> bool {
        self.damage.iter().any(|&d| d > 0.0)
    }
}

/// Result of a body-grid raycast, in the grid's own local voxel coordinates.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct GridRayHit {
    pub voxel: IVec3,
    pub dist_m: f32,
}

/// Cast a ray from `origin_m` (in the grid's own local-meter frame, origin
/// at its minimum corner) along `dir` for at most `max_dist_m` meters,
/// returning the first solid voxel hit. Same Amanatides-Woo DDA algorithm as
/// [`vox_world::raycast`], just against a body's dense in-memory grid
/// instead of the chunked world -- no chunk indirection needed, so this is
/// simpler, and there's no need for an early-exit bounds check: bodies are
/// small (debris-scale), so the loop running out via `max_dist_m` costs
/// little even without one.
pub fn raycast_grid(
    grid: &VoxelGrid,
    origin_m: Vec3,
    dir: Vec3,
    max_dist_m: f32,
    voxel_size_m: f32,
) -> Option<GridRayHit> {
    let s = voxel_size_m;
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO || !origin_m.is_finite() || max_dist_m <= 0.0 {
        return None;
    }

    let p = origin_m / s;
    let mut cell = p.floor().as_ivec3();
    if grid.solid(cell) {
        return Some(GridRayHit {
            voxel: cell,
            dist_m: 0.0,
        });
    }

    let step = IVec3::new(
        dir.x.signum() as i32,
        dir.y.signum() as i32,
        dir.z.signum() as i32,
    );
    let mut t_max = Vec3::ZERO;
    let mut t_delta = Vec3::ZERO;
    for a in 0..3 {
        if dir[a] > 0.0 {
            t_max[a] = (cell[a] as f32 + 1.0 - p[a]) / dir[a];
            t_delta[a] = 1.0 / dir[a];
        } else if dir[a] < 0.0 {
            t_max[a] = (p[a] - cell[a] as f32) / -dir[a];
            t_delta[a] = -1.0 / dir[a];
        } else {
            t_max[a] = f32::INFINITY;
            t_delta[a] = f32::INFINITY;
        }
    }

    let max_t = max_dist_m / s;
    loop {
        let a = if t_max.x < t_max.y {
            if t_max.x < t_max.z { 0 } else { 2 }
        } else if t_max.y < t_max.z {
            1
        } else {
            2
        };
        if t_max[a] > max_t {
            return None;
        }
        cell[a] += step[a];
        let t_enter = t_max[a];
        t_max[a] += t_delta[a];
        if grid.solid(cell) {
            return Some(GridRayHit {
                voxel: cell,
                dist_m: t_enter * s,
            });
        }
    }
}

/// Mass, center of mass, and inertia tensor about the COM (body frame).
#[derive(Copy, Clone, Debug)]
pub struct MassProps {
    pub mass: f32,
    /// COM relative to the grid's minimum corner, meters.
    pub com_local: Vec3,
    pub inertia_com: Mat3,
}

/// Sum per-voxel point masses plus each voxel's own cube inertia — an exact
/// decomposition of the uniform-density solid.
pub fn mass_props(grid: &VoxelGrid, reg: &MaterialRegistry, voxel_size_m: f32) -> MassProps {
    let s = voxel_size_m;
    let v_cell = s * s * s;

    let mut mass = 0.0f32;
    let mut weighted = Vec3::ZERO;
    for y in 0..grid.dims.y {
        for z in 0..grid.dims.z {
            for x in 0..grid.dims.x {
                let v = grid.get(IVec3::new(x, y, z));
                if v == AIR {
                    continue;
                }
                let density = reg
                    .get(vox_core::MaterialId(v.0))
                    .map(|d| d.density)
                    .unwrap_or(1000.0);
                let m = density * v_cell;
                let p = (Vec3::new(x as f32, y as f32, z as f32) + 0.5) * s;
                mass += m;
                weighted += p * m;
            }
        }
    }
    if mass <= 0.0 {
        return MassProps {
            mass: 0.0,
            com_local: Vec3::ZERO,
            inertia_com: Mat3::IDENTITY,
        };
    }
    let com = weighted / mass;

    let mut inertia = Mat3::ZERO;
    let cube_term = s * s / 6.0;
    for y in 0..grid.dims.y {
        for z in 0..grid.dims.z {
            for x in 0..grid.dims.x {
                let v = grid.get(IVec3::new(x, y, z));
                if v == AIR {
                    continue;
                }
                let density = reg
                    .get(vox_core::MaterialId(v.0))
                    .map(|d| d.density)
                    .unwrap_or(1000.0);
                let m = density * v_cell;
                let r = (Vec3::new(x as f32, y as f32, z as f32) + 0.5) * s - com;
                // Point-mass parallel-axis term.
                let rr = r.length_squared();
                let point = Mat3::from_cols(
                    Vec3::new(rr - r.x * r.x, -r.y * r.x, -r.z * r.x),
                    Vec3::new(-r.x * r.y, rr - r.y * r.y, -r.z * r.y),
                    Vec3::new(-r.x * r.z, -r.y * r.z, rr - r.z * r.z),
                );
                inertia += point.mul_scalar(m);
                // The voxel's own inertia about its center.
                inertia += Mat3::from_diagonal(Vec3::splat(m * cube_term));
            }
        }
    }
    MassProps {
        mass,
        com_local: com,
        inertia_com: inertia,
    }
}

/// Surface sample points: centers of solid voxels with at least one empty
/// face neighbor, in meters relative to the grid's minimum corner.
pub fn surface_points(grid: &VoxelGrid, voxel_size_m: f32) -> Vec<Vec3> {
    const DIRS: [IVec3; 6] = [
        IVec3::X,
        IVec3::NEG_X,
        IVec3::Y,
        IVec3::NEG_Y,
        IVec3::Z,
        IVec3::NEG_Z,
    ];
    let mut points = Vec::new();
    for y in 0..grid.dims.y {
        for z in 0..grid.dims.z {
            for x in 0..grid.dims.x {
                let p = IVec3::new(x, y, z);
                if !grid.solid(p) {
                    continue;
                }
                if DIRS.iter().any(|d| !grid.solid(p + *d)) {
                    points.push((p.as_vec3() + 0.5) * voxel_size_m);
                }
            }
        }
    }
    points
}

/// Sleep bookkeeping.
#[derive(Copy, Clone, Default, Debug)]
pub struct SleepState {
    pub quiet_steps: u32,
    pub asleep: bool,
}

/// Handle to a body in the arena.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct BodyId {
    pub slot: u32,
    pub generation: u32,
}

/// A voxel rigid body. `pos` is the world position of the center of mass.
#[derive(Clone, Debug)]
pub struct Body {
    pub pos: Vec3,
    pub rot: Quat,
    pub vel: Vec3,
    pub omega: Vec3,
    pub inv_mass: f32,
    pub inv_inertia_local: Mat3,
    pub grid: VoxelGrid,
    /// Grid minimum corner relative to the COM, body frame, meters.
    pub grid_offset: Vec3,
    /// Surface sample points relative to the COM, body frame, meters.
    pub surface: Vec<Vec3>,
    /// Half of this body's voxel edge length (contact radius).
    pub half_voxel: f32,
    pub sleep: SleepState,
    /// World-space AABB, refreshed each step.
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    /// Transform snapshot from the previous full step (render interpolation).
    pub prev_pos: Vec3,
    pub prev_rot: Quat,
    /// Cached world-space inverse inertia (`R * I_local^-1 * R^T`), refreshed
    /// by the solver once per substep (rotation only changes at substep
    /// boundaries, so it's constant across a substep's whole contact solve).
    /// Reading this instead of calling [`Self::inv_inertia_world`] matters
    /// because the solver's impulse loop otherwise re-derives it -- a
    /// quaternion-to-matrix conversion plus two mat3 multiplies -- roughly
    /// 25 times per contact per substep (warm start + iterations x three
    /// impulse applications), which made it the hottest single operation in
    /// a many-contact debris pile. May be stale between a spawn that then
    /// hand-edits `rot` (e.g. fragment placement) and the next substep, but
    /// nothing reads it in that window.
    pub inv_iw: Mat3,
    /// Seconds remaining before this body is auto-despawned, or `None` for
    /// bodies that persist until something else removes them (the debris
    /// budget, a carve, going to sleep and getting cleared). Set once at
    /// spawn time for small "clutter" chips -- see
    /// `vox_core::consts::CLUTTER_MAX_VOXELS` -- and ticked down by
    /// `PhysicsWorld::tick_lifetimes`.
    pub lifetime_s: Option<f32>,
    /// True when damage values changed since last mesh. Cleared after re-mesh.
    pub damage_dirty: bool,
    /// Monotonically incremented whenever the body's voxel grid is modified
    /// (carve, split, crumble). Infrastructure for invalidating stale data
    /// that depends on geometry: the solver's warm-start impulses, the
    /// render system's re-mesh decision, and future parallel fracture jobs.
    /// A brand-new body starts at 0; each fragment from a split inherits
    /// `parent_revision + 1`.
    pub topology_revision: u32,
    /// Snapshot of [`topology_revision`](Self::topology_revision) taken at the
    /// end of the previous full [`PhysicsWorld::step`]. The warm-start code
    /// compares the two: if they differ, the body's geometry changed between
    /// steps and the old accumulated contact impulses are stale, so warm
    /// starting is skipped for that body's contacts.
    pub last_step_revision: u32,
}

impl Body {
    /// Build a body from a voxel grid. Returns `None` for massless grids.
    pub fn from_grid(
        grid: VoxelGrid,
        reg: &MaterialRegistry,
        voxel_size_m: f32,
        com_world: Vec3,
    ) -> Option<Self> {
        let props = mass_props(&grid, reg, voxel_size_m);
        if props.mass <= 0.0 {
            return None;
        }
        let surface = surface_points(&grid, voxel_size_m)
            .into_iter()
            .map(|p| p - props.com_local)
            .collect();
        let mut body = Self {
            pos: com_world,
            rot: Quat::IDENTITY,
            vel: Vec3::ZERO,
            omega: Vec3::ZERO,
            inv_mass: 1.0 / props.mass,
            inv_inertia_local: props.inertia_com.inverse(),
            grid_offset: -props.com_local,
            grid,
            surface,
            half_voxel: voxel_size_m * 0.5,
            sleep: SleepState::default(),
            aabb_min: Vec3::ZERO,
            aabb_max: Vec3::ZERO,
            prev_pos: com_world,
            prev_rot: Quat::IDENTITY,
            lifetime_s: None,
            damage_dirty: false,
            topology_revision: 0,
            last_step_revision: 0,
            inv_iw: Mat3::IDENTITY,
        };
        body.refresh_aabb();
        body.inv_iw = body.inv_inertia_world();
        Some(body)
    }

    /// Inverse inertia tensor in world space.
    #[inline]
    pub fn inv_inertia_world(&self) -> Mat3 {
        let r = Mat3::from_quat(self.rot);
        r * self.inv_inertia_local * r.transpose()
    }

    /// Recompute the world AABB from the rotated grid bounds.
    pub fn refresh_aabb(&mut self) {
        let s = 2.0 * self.half_voxel;
        let ext = self.grid.dims.as_vec3() * s;
        let corners = [
            Vec3::ZERO,
            Vec3::new(ext.x, 0.0, 0.0),
            Vec3::new(0.0, ext.y, 0.0),
            Vec3::new(0.0, 0.0, ext.z),
            Vec3::new(ext.x, ext.y, 0.0),
            Vec3::new(ext.x, 0.0, ext.z),
            Vec3::new(0.0, ext.y, ext.z),
            ext,
        ];
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for c in corners {
            let w = self.pos + self.rot * (self.grid_offset + c);
            min = min.min(w);
            max = max.max(w);
        }
        self.aabb_min = min;
        self.aabb_max = max;
    }

    /// Total mass in kg.
    pub fn mass(&self) -> f32 {
        1.0 / self.inv_mass
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> MaterialRegistry {
        MaterialRegistry::from_toml_str(
            r#"
            [[material]]
            name = "test"
            color = [0.5, 0.5, 0.5]
            density = 1000.0
            strength = 1.0
            "#,
            "test.toml",
        )
        .expect("test registry")
    }

    fn solid_grid(dims: IVec3) -> VoxelGrid {
        VoxelGrid::new(dims, vec![Voxel(1); (dims.x * dims.y * dims.z) as usize])
    }

    #[test]
    fn box_inertia_matches_analytic() {
        // 8 x 12 x 6 voxels at 0.1 m, density 1000 kg/m³.
        let reg = registry();
        let s = 0.1;
        let grid = solid_grid(IVec3::new(8, 12, 6));
        let props = mass_props(&grid, &reg, s);

        let (a, b, c) = (0.8f32, 1.2, 0.6);
        let mass = 1000.0 * a * b * c;
        assert!((props.mass - mass).abs() / mass < 1e-5);
        assert!((props.com_local - Vec3::new(0.4, 0.6, 0.3)).length() < 1e-6);

        // Solid box analytic inertia; the voxel decomposition is exact.
        let ix = mass * (b * b + c * c) / 12.0;
        let iy = mass * (a * a + c * c) / 12.0;
        let iz = mass * (a * a + b * b) / 12.0;
        let d = props.inertia_com;
        assert!(
            (d.col(0).x - ix).abs() / ix < 1e-4,
            "Ix {} vs {}",
            d.col(0).x,
            ix
        );
        assert!(
            (d.col(1).y - iy).abs() / iy < 1e-4,
            "Iy {} vs {}",
            d.col(1).y,
            iy
        );
        assert!(
            (d.col(2).z - iz).abs() / iz < 1e-4,
            "Iz {} vs {}",
            d.col(2).z,
            iz
        );
        // Off-diagonals vanish for a symmetric box.
        assert!(d.col(0).y.abs() < 1e-3 && d.col(0).z.abs() < 1e-3);
    }

    #[test]
    fn l_shape_com_is_correct() {
        // An L: 2x1x1 base plus 1x1x1 on top of the first cell (voxel size 1).
        let reg = registry();
        let mut voxels = vec![AIR; 2 * 2];
        // dims (2, 2, 1): index = x + z*2 + y*2*1 = x + y*2 (z=0)
        voxels[0] = Voxel(1); // (0,0,0)
        voxels[1] = Voxel(1); // (1,0,0)
        voxels[2] = Voxel(1); // (0,1,0)
        let grid = VoxelGrid::new(IVec3::new(2, 2, 1), voxels);
        let props = mass_props(&grid, &reg, 1.0);

        // Three unit cubes at centers (0.5,0.5), (1.5,0.5), (0.5,1.5).
        let expected = Vec3::new((0.5 + 1.5 + 0.5) / 3.0, (0.5 + 0.5 + 1.5) / 3.0, 0.5);
        assert!((props.com_local - expected).length() < 1e-6);
        assert!((props.mass - 3000.0).abs() < 1e-2);
    }

    #[test]
    fn surface_points_of_solid_box() {
        // 4³ solid: all but the 2³ interior are surface voxels.
        let grid = solid_grid(IVec3::splat(4));
        let pts = surface_points(&grid, 0.1);
        assert_eq!(pts.len(), 64 - 8);
    }

    #[test]
    fn body_from_grid_centers_on_com() {
        let reg = registry();
        let grid = solid_grid(IVec3::splat(4));
        let body = Body::from_grid(grid, &reg, 0.1, Vec3::new(5.0, 5.0, 5.0)).expect("massive");
        // grid_offset must put the grid's center at the COM.
        assert!((body.grid_offset + Vec3::splat(0.2)).length() < 1e-6);
        assert!((body.aabb_min - Vec3::new(4.8, 4.8, 4.8)).length() < 1e-5);
        assert!((body.aabb_max - Vec3::new(5.2, 5.2, 5.2)).length() < 1e-5);
        assert!((body.mass() - 1000.0 * 0.4f32.powi(3)).abs() < 1e-3);
    }

    #[test]
    fn raycast_grid_hits_the_near_face() {
        let grid = solid_grid(IVec3::splat(4));
        let hit = raycast_grid(&grid, Vec3::new(-1.0, 0.2, 0.2), Vec3::X, 5.0, 0.1)
            .expect("must hit the near face");
        assert_eq!(hit.voxel, IVec3::new(0, 2, 2));
        assert!((hit.dist_m - 1.0).abs() < 1e-4, "got {}", hit.dist_m);
    }

    #[test]
    fn raycast_grid_misses_when_aimed_away() {
        let grid = solid_grid(IVec3::splat(4));
        assert!(raycast_grid(&grid, Vec3::new(-1.0, 0.2, 0.2), Vec3::NEG_X, 5.0, 0.1).is_none());
    }

    #[test]
    fn damage_accumulates_and_caps_at_1() {
        let mut grid = VoxelGrid::new(IVec3::new(2, 2, 2), vec![Voxel(1); 8]);
        assert_eq!(grid.damage_at(IVec3::new(0, 0, 0)), 0.0);
        grid.add_damage(IVec3::new(0, 0, 0), 0.5);
        assert_eq!(grid.damage_at(IVec3::new(0, 0, 0)), 0.5);
        grid.add_damage(IVec3::new(0, 0, 0), 0.7);
        assert_eq!(grid.damage_at(IVec3::new(0, 0, 0)), 1.0, "damage caps at 1.0");
    }

    #[test]
    fn damage_does_not_accumulate_on_air() {
        let mut grid = VoxelGrid::new(IVec3::new(2, 1, 1), vec![AIR, Voxel(1)]);
        assert!(!grid.add_damage(IVec3::new(0, 0, 0), 0.5), "air rejects damage");
        assert!(grid.add_damage(IVec3::new(1, 0, 0), 0.5), "solid accepts damage");
    }

    #[test]
    fn damage_decays_to_zero() {
        let mut grid = VoxelGrid::new(IVec3::new(1, 1, 1), vec![Voxel(1)]);
        grid.add_damage(IVec3::ZERO, 0.5);
        assert!(grid.has_damage());
        grid.tick_damage_decay(1.0, 0.05);
        assert_eq!(grid.damage_at(IVec3::ZERO), 0.45);
        for _ in 0..20 {
            grid.tick_damage_decay(1.0, 0.05);
        }
        assert_eq!(grid.damage_at(IVec3::ZERO), 0.0);
        assert!(!grid.has_damage());
    }
}
