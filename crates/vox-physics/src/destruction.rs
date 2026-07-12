//! Destruction: carve voxels from the world, then determine which surviving
//! material is still structurally supported and which has become debris.
//!
//! Pipeline: **carve** (remove a shape, recording what was removed) →
//! **flood** (6-connected BFS, one flood per solid voxel newly exposed by the
//! removal, following the *actual* voxel shape — not an artificial search
//! box) → **detach** (anything a flood proves is bounded and small is
//! unsupported: extracted into a [`VoxelGrid`] and spawned as a [`Body`]).
//! Tiny fragments are discarded as dust; implausibly large components are
//! left in the world as a safety valve against an unbounded physics budget.
//!
//! Each flood is a *proof*, not a heuristic: it terminates the instant it
//! either (a) reaches the world floor or exceeds a generous give-up cap —
//! proof this component connects to (or is) something far too large to be
//! anything but ordinary terrain, so it's anchored — or (b) exhausts
//! naturally under that cap, meaning it really is a bounded, disconnected
//! island (however large). There is no artificial search region to size
//! correctly: the flood follows whatever shape the material actually has, so
//! a thin severed tree trunk terminates in a few hundred steps regardless of
//! how much solid terrain happens to sit nearby, and an edit deep inside a
//! huge contiguous mass (ordinary terrain) is recognized as anchored the
//! moment the flood proves it's huge, not after exhaustively rescanning an
//! ever-growing bounding box.
//!
//! The give-up cap and [`MAX_BODY_VOXELS`] are deliberately *different*
//! numbers answering *different* questions — see the comment on
//! [`FLOOD_GIVE_UP_VOXELS`]. A disconnected component can legitimately be
//! both "proven bounded" and "too big for one rigidbody" at once (a fully
//! generated tree's canopy routinely is); that's a size-policy decision made
//! after the flood, not a reason to misreport it as anchored.
//!
//! (An earlier version of this pipeline searched a padded bounding box around
//! the edit and grew it until its outer shell was clean of solid material.
//! That worked for isolated test structures floating in empty space, but on
//! real terrain — an effectively unbounded contiguous solid mass — the shell
//! never goes clean, so every single-voxel break maxed out the growth budget
//! rescanning an ever-larger box (the reported dig-lag), and the safe-default
//! fallback then credited *everything* touching that huge, still-dirty
//! region as supported — including a genuinely severed tree-top sitting near
//! the same boundary (the reported floating-tree bug). Both symptoms were the
//! same root cause; the seed-flood design above has no bounding box to grow,
//! so neither failure mode exists. A follow-up bug used one cap for both the
//! flood's give-up threshold and the body-size policy; a real tree's canopy
//! is large enough to exceed that shared cap, so it was misreported as
//! anchored — left resident and unsimulated in the world, indistinguishable
//! from debris floating forever. See [`FLOOD_GIVE_UP_VOXELS`].)

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use glam::{IVec3, Vec3, Vec3Swizzles};
use vox_core::consts::{DEBRIS_MIN_VOXELS, MAX_BODY_VOXELS};
use vox_core::{FxHashMap, FxHashSet, MaterialRegistry, voxel_at, voxel_center_m};
use vox_world::{AIR, SolidLookup, Voxel, World};

use crate::BodyId;
use crate::body::{Body, VoxelGrid, mass_props};
use crate::solver::PhysicsWorld;

/// Extra voxels searched beyond the carved region on every side when
/// computing [`CarveResult::region`] — used only to size the `wake_region`
/// call after a blast, not for connectivity analysis.
const REGION_PAD: i32 = 2;

/// Maximum per-axis angular kick applied to blast debris, in rad/s. Blast
/// strength itself is a caller-supplied, live-tunable parameter (see
/// [`blast`]), not a fixed constant.
const BLAST_SPIN_MAX: f32 = 3.0;
/// Floor on distance-from-center used in the impulse falloff, so a blast
/// centered inside debris doesn't produce an infinite/huge speed.
const BLAST_MIN_DIST_M: f32 = 0.5;

/// A half-open voxel-space box `[min, max)`.
pub type Region = (IVec3, IVec3);

/// What a carve removed, a padded bounding box around the removal (used
/// only to size the post-blast wake-up query, not connectivity), and — for
/// [`blast`] specifically — the bodies it spawned from newly-unsupported
/// material. `carve_sphere` alone never spawns bodies, so `spawned` is empty
/// there; callers that need the connectivity pass to run must call
/// [`detach_unsupported`] (or [`blast`], which does it for them) themselves.
pub struct CarveResult {
    pub removed: Vec<(IVec3, Voxel)>,
    pub region: Region,
    pub spawned: Vec<BodyId>,
}

/// Remove every solid voxel whose center lies within `radius_m` of
/// `center_m`. Returns what was removed and a padded bounding box around the
/// removal (for [`PhysicsWorld::wake_region`]; connectivity analysis no
/// longer needs a search region at all — see [`detach_unsupported`]).
pub fn carve_sphere(world: &mut World, center_m: Vec3, radius_m: f32) -> CarveResult {
    let s = world.cfg.voxel_size_m;
    let r_vox = (radius_m / s).ceil() as i32;
    let center_vox = voxel_at(center_m, s);
    let box_min = center_vox - IVec3::splat(r_vox);
    let box_max = center_vox + IVec3::splat(r_vox + 1);

    let mut removed = Vec::new();
    let mut min = IVec3::splat(i32::MAX);
    let mut max = IVec3::splat(i32::MIN);
    // `edit_box` resolves each touched chunk once instead of once per voxel
    // (see its doc comment) -- the dominant cost at real destruction scale.
    world.edit_box(box_min, box_max, |v, cur| {
        if cur == AIR || (voxel_center_m(v, s) - center_m).length() > radius_m {
            return None;
        }
        removed.push((v, cur));
        min = min.min(v);
        max = max.max(v);
        Some(AIR)
    });
    let region = if removed.is_empty() {
        (center_vox, center_vox + IVec3::ONE)
    } else {
        (
            min - IVec3::splat(REGION_PAD),
            max + IVec3::splat(REGION_PAD + 1),
        )
    };
    CarveResult {
        removed,
        region,
        spawned: Vec::new(),
    }
}

/// Number of outward "shrapnel" spikes an [`ExplosionShape`] casts. Chosen
/// to look chaotic without costing too much: each voxel in the shape's
/// bounding box is tested against every spike, so this multiplies carve
/// cost directly.
const EXPLOSION_SPIKE_COUNT: usize = 18;
/// Spike length, as a multiple of the base radius: how far past the core
/// crater the shrapnel reaches, before per-spike jitter.
const EXPLOSION_SPIKE_LENGTH_RANGE: (f32, f32) = (1.2, 2.4);
/// Spike thickness, as a multiple of the base radius, before per-spike
/// jitter -- thin fingers, not fat tubes, so the base sphere still reads as
/// the "core" of the blast.
const EXPLOSION_SPIKE_RADIUS_RANGE: (f32, f32) = (0.08, 0.22);
/// Direction jitter applied to each spike's otherwise-even distribution
/// (radians of solid angle, roughly) -- without this, evenly-spaced spikes
/// look mechanical/procedural rather than chaotic.
const EXPLOSION_DIR_JITTER: f32 = 0.35;

/// A jagged, deterministic "explosion" shape: a base sphere (the crater
/// core) plus a starburst of thin outward spikes of varying length,
/// thickness, and direction. A plain sphere reads as an obviously clean,
/// artificial hole; real explosions (and Teardown's own destruction, per
/// its developers) never look that tidy. The spikes also do double duty
/// for gameplay: material that survives *between* two spikes but loses its
/// connection to the rest of the structure becomes its own disconnected
/// debris fragment for free, via the same connectivity pass every other
/// carve already runs -- more spikes naturally means more, and more
/// irregular, debris, without any bespoke fragment-generation code.
///
/// Deterministic from `seed` (reuses the blast's own per-shot seed, so a
/// replay or a test produces the identical shape) via the same small-hash
/// scheme already used for per-body blast spin variation.
pub(crate) struct ExplosionShape {
    center: Vec3,
    radius_m: f32,
    /// (direction, length_m, radius_m) per spike.
    spikes: Vec<(Vec3, f32, f32)>,
    max_reach_m: f32,
}

impl ExplosionShape {
    pub(crate) fn new(center: Vec3, radius_m: f32, seed: u32) -> Self {
        let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
        let n = EXPLOSION_SPIKE_COUNT;
        let mut spikes = Vec::with_capacity(n);
        let mut max_reach_m: f32 = radius_m;
        for i in 0..n {
            // Even coverage of the sphere (fibonacci-sphere sampling)...
            let y = 1.0 - (i as f32 / (n - 1).max(1) as f32) * 2.0;
            let ring_r = (1.0 - y * y).max(0.0).sqrt();
            let theta = golden_angle * i as f32;
            let even_dir = Vec3::new(theta.cos() * ring_r, y, theta.sin() * ring_r);

            // ...then jittered per-spike so it doesn't look mechanically
            // even, using the same deterministic small-hash approach as
            // `apply_blast_impulse`'s spin variation.
            let h = small_hash(seed, i as u32);
            let jitter = Vec3::new(
                (h & 0xFF) as f32 / 255.0 - 0.5,
                ((h >> 8) & 0xFF) as f32 / 255.0 - 0.5,
                ((h >> 16) & 0xFF) as f32 / 255.0 - 0.5,
            ) * EXPLOSION_DIR_JITTER;
            let dir = (even_dir + jitter).normalize_or(even_dir);

            // A second, independent hash for length/radius so they don't
            // correlate with the direction jitter above (which would reuse
            // the same bytes).
            let h2 = small_hash(seed, i as u32 ^ 0x5EED_0000);
            let unit = |bits: u32, range: (f32, f32)| {
                range.0 + (bits as f32 / 255.0) * (range.1 - range.0)
            };
            let length = radius_m * unit(h2 & 0xFF, EXPLOSION_SPIKE_LENGTH_RANGE);
            let spike_radius = radius_m * unit((h2 >> 8) & 0xFF, EXPLOSION_SPIKE_RADIUS_RANGE);

            max_reach_m = max_reach_m.max(length + spike_radius);
            spikes.push((dir, length, spike_radius));
        }
        Self {
            center,
            radius_m,
            spikes,
            max_reach_m,
        }
    }

    /// The bounding box (world-space meters) this shape can possibly touch.
    pub(crate) fn bounds_m(&self) -> (Vec3, Vec3) {
        let pad = Vec3::splat(self.max_reach_m);
        (self.center - pad, self.center + pad)
    }

    pub(crate) fn contains(&self, p: Vec3) -> bool {
        if (p - self.center).length() <= self.radius_m {
            return true;
        }
        for &(dir, length, spike_radius) in &self.spikes {
            let seg = dir * length;
            let t = ((p - self.center).dot(seg) / seg.length_squared()).clamp(0.0, 1.0);
            let closest = self.center + seg * t;
            if (p - closest).length() <= spike_radius {
                return true;
            }
        }
        false
    }
}

/// Remove every solid voxel within a jagged [`ExplosionShape`] centered at
/// `center_m` -- the Bomb tool's "cool destruction", not a clean sphere.
/// Same return shape as [`carve_sphere`]. `seed` should be the blast's own
/// per-shot seed (also driving debris spin), so the crater shape and the
/// debris that flies out of it are both deterministic from the same call.
pub fn carve_explosion(world: &mut World, center_m: Vec3, radius_m: f32, seed: u32) -> CarveResult {
    let shape = ExplosionShape::new(center_m, radius_m, seed);
    let s = world.cfg.voxel_size_m;
    let (bmin, bmax) = shape.bounds_m();
    let box_min = voxel_at(bmin, s);
    let box_max = voxel_at(bmax, s) + IVec3::ONE;

    let mut removed = Vec::new();
    let mut min = IVec3::splat(i32::MAX);
    let mut max = IVec3::splat(i32::MIN);
    world.edit_box(box_min, box_max, |v, cur| {
        if cur == AIR || !shape.contains(voxel_center_m(v, s)) {
            return None;
        }
        removed.push((v, cur));
        min = min.min(v);
        max = max.max(v);
        Some(AIR)
    });
    let region = if removed.is_empty() {
        (voxel_at(center_m, s), voxel_at(center_m, s) + IVec3::ONE)
    } else {
        (
            min - IVec3::splat(REGION_PAD),
            max + IVec3::splat(REGION_PAD + 1),
        )
    };
    CarveResult {
        removed,
        region,
        spawned: Vec::new(),
    }
}

/// Remove every solid voxel within `radius_m` of the line segment from
/// `start_m` to `end_m` (inclusive) — a carved tunnel, for very long-range
/// destruction (a beam weapon) where sampling many individual spheres along
/// the path would multiply the cost and repeatedly revisit the same
/// voxels. The search box is clamped to the world's own voxel bounds, so a
/// nominally huge `end_m` (an "infinite range" beam) never costs more than
/// the world it's actually cutting through. Returns what was removed and a
/// padded bounding box around it, exactly like [`carve_sphere`].
pub fn carve_capsule(world: &mut World, start_m: Vec3, end_m: Vec3, radius_m: f32) -> CarveResult {
    let s = world.cfg.voxel_size_m;
    let seg = end_m - start_m;
    let seg_len_sq = seg.length_squared();

    let (world_min, world_max) = world.bounds_voxels();
    let pad = Vec3::splat(radius_m);
    let box_min = voxel_at(start_m.min(end_m) - pad, s).max(world_min);
    let box_max = (voxel_at(start_m.max(end_m) + pad, s) + IVec3::ONE).min(world_max);

    let mut removed = Vec::new();
    let mut min = IVec3::splat(i32::MAX);
    let mut max = IVec3::splat(i32::MIN);
    world.edit_box(box_min, box_max, |v, cur| {
        if cur == AIR {
            return None;
        }
        let c = voxel_center_m(v, s);
        let t = if seg_len_sq > 1e-9 {
            ((c - start_m).dot(seg) / seg_len_sq).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let closest = start_m + seg * t;
        if (c - closest).length() > radius_m {
            return None;
        }
        removed.push((v, cur));
        min = min.min(v);
        max = max.max(v);
        Some(AIR)
    });
    let region = if removed.is_empty() {
        (box_min, box_min + IVec3::ONE)
    } else {
        (
            min - IVec3::splat(REGION_PAD),
            max + IVec3::splat(REGION_PAD + 1),
        )
    };
    CarveResult {
        removed,
        region,
        spawned: Vec::new(),
    }
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

/// Early-bailout cap for [`flood_from`]'s "is this obviously anchored"
/// check — deliberately much larger than [`MAX_BODY_VOXELS`]. The two caps
/// answer different questions and must not share a value: this one only
/// bounds worst-case flood cost against a pathological component that never
/// reaches the world floor (a floating island with no connection down, or a
/// contiguous mass so vast the floor is many steps away); `MAX_BODY_VOXELS`
/// decides whether an already-proven-disconnected component is small enough
/// to spawn as one rigidbody. Conflating them (an earlier version of this
/// function did) misclassifies any large-but-genuinely-disconnected object
/// — a severed tree's full canopy can reach ~150k-200k voxels at 0.1 m
/// scale — as "anchored", silently leaving it resident and unsimulated in
/// the world: visually indistinguishable from debris that floats forever.
const FLOOD_GIVE_UP_VOXELS: usize = 3_000_000;

/// Result of flooding outward from one seed voxel.
enum FloodResult {
    /// The flood reached the world floor, or exceeded [`FLOOD_GIVE_UP_VOXELS`]
    /// voxels before running out of solid material to explore: proof this
    /// component connects to (or is) something far too large to be debris.
    Anchored,
    /// The flood exhausted every reachable solid voxel while staying under
    /// the cap: a genuinely bounded, disconnected island. Its member voxels.
    Bounded(Vec<IVec3>),
    /// The flood reached an unloaded chunk whose contents are unknown
    /// (terrain exists but hasn't been generated in the streamed world).
    /// Treated like `Anchored` — don't extract — because we can't prove
    /// the structure is disconnected. The flood must not traverse through
    /// phantom-solid voxels (every voxel in an unloaded chunk would read
    /// as present, flooding through 32k+ phantom voxels and their unloaded
    /// neighbors, almost always hitting the floor or give-up cap for the
    /// wrong reason).
    Unknown,
}

/// 6-connected flood from `start` through solid voxels, following the
/// actual voxel shape (no artificial search box). Terminates the moment it
/// proves the component is anchored (world floor, or over
/// [`FLOOD_GIVE_UP_VOXELS`]), so cost is proportional to the *smaller* of
/// "how big this component is" and "how big the give-up cap is" — never to
/// how large the surrounding world happens to be. Every visited voxel is
/// recorded in `visited`, shared across every seed in one
/// [`detach_unsupported`] call, so seeds that turn out to be part of the
/// same component (or the same already-anchored mass) are never re-explored.
///
/// Expansion order is best-first toward the floor (a small-root-heap keyed
/// by remaining vertical distance), not plain breadth-first. Plain BFS
/// explores an actual sphere around `start` before it happens to reach the
/// floor even when the floor is only a short vertical hop away — the
/// overwhelmingly common case for ordinary terrain, where a seed sitting a
/// few voxels above solid ground pays for exploring sideways and upward
/// first regardless. Prioritizing "closest to the floor" finds that short
/// path almost immediately instead. This changes nothing about correctness:
/// a `Bounded` result still explores every reachable voxel regardless of
/// order (there's nowhere left to go before the queue/heap empties either
/// way) — only how fast an `Anchored` result is reached changes.
fn flood_from(
    world: &World,
    lookup: &mut SolidLookup<'_>,
    start: IVec3,
    visited: &mut FxHashSet<IVec3>,
) -> FloodResult {
    let floor_y = world.bounds_voxels().0.y;
    // Keyed by (distance-to-floor, x, y, z) -- IVec3 isn't Ord, so the voxel
    // itself is encoded directly into the (fully Ord) key tuple; the
    // trailing coordinates are just an arbitrary but stable tiebreaker.
    let key = |v: IVec3| ((v.y - floor_y).abs(), v.x, v.y, v.z);
    let mut heap: BinaryHeap<Reverse<(i32, i32, i32, i32)>> = BinaryHeap::new();
    let mut component = Vec::new();
    heap.push(Reverse(key(start)));
    visited.insert(start);
    while let Some(Reverse((_, x, y, z))) = heap.pop() {
        let v = IVec3::new(x, y, z);
        if y == floor_y || component.len() >= FLOOD_GIVE_UP_VOXELS {
            return FloodResult::Anchored;
        }
        component.push(v);
        for d in DIRS {
            let n = v + d;
            if lookup.is_unloaded(n) {
                // Unloaded chunk: contents unknown. Can't prove
                // disconnection — assume anchored, don't extract.
                return FloodResult::Unknown;
            }
            if lookup.present(n) && visited.insert(n) {
                heap.push(Reverse(key(n)));
            }
        }
    }
    FloodResult::Bounded(component)
}

/// Find material that became newly unsupported when `removed` was carved
/// away, extract it from the world, and spawn each surviving component as a
/// sleeping-eligible (initially awake, zero-velocity) rigid body. Components
/// under [`DEBRIS_MIN_VOXELS`] are discarded as dust; components over
/// [`MAX_BODY_VOXELS`] are left in the world untouched. Returns the spawned
/// body ids.
///
/// `removed` is just the positions that were carved away (their former
/// contents don't matter here) — pass a [`CarveResult::removed`]'s positions
/// for a blast, or a single position for a single-voxel break. Every solid
/// voxel still adjacent to one of them is a seed: the natural starting point
/// for asking "did removing my neighbor just cut me off from support?". See
/// the module docs for why this replaced a bounding-box-based search.
pub fn detach_unsupported(
    world: &mut World,
    phys: &mut PhysicsWorld,
    registry: &MaterialRegistry,
    removed: &[IVec3],
) -> Vec<BodyId> {
    let mut visited: FxHashSet<IVec3> = FxHashSet::default();
    let mut components: Vec<Vec<IVec3>> = Vec::new();
    // Shared across every seed in this call, not rebuilt per flood: most
    // seeds from the same edit land in the same handful of chunks, so the
    // chunk-lookup cache carries over between them too.
    let mut lookup = SolidLookup::new(world);
    for &r in removed {
        for d in DIRS {
            let seed = r + d;
            if visited.contains(&seed) || !lookup.present(seed) {
                continue;
            }
            if let FloodResult::Bounded(component) =
                flood_from(world, &mut lookup, seed, &mut visited)
            {
                components.push(component);
            }
        }
    }

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

/// Deterministic small hash for per-body blast variation. `pub(crate)`:
/// `body_destruction`'s impact-fracture chips reuse it for the same
/// deterministic-sampling trick against a body's own local grid.
#[inline]
pub(crate) fn small_hash(a: u32, b: u32) -> u32 {
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
/// deterministic angular kick. `power` is the tunable blast strength
/// ([`vox_core::Tunables::blast_power`], typically). Public so callers can
/// apply the same "explosion" feel to bodies carved directly (e.g. the Bomb
/// tool hitting an existing debris body, not just the static world).
pub fn apply_blast_impulse(
    phys: &mut PhysicsWorld,
    ids: &[BodyId],
    center_m: Vec3,
    power: f32,
    seed: u32,
) {
    for (i, &id) in ids.iter().enumerate() {
        let Some(body) = phys.get(id) else { continue };
        let offset = body.pos - center_m;
        let dist = offset.length();
        let dir = if dist > 1e-6 { offset / dist } else { Vec3::Y };
        let mass = body.mass();
        let speed = power / dist.max(BLAST_MIN_DIST_M) / mass.sqrt();

        // Clamp the upward velocity component. Without this, debris whose
        // COM is above the blast center (common for ground-level blasts
        // whose downward ExplosionShape spikes carve vertical channels into
        // terrain) gets launched straight up at unrealistic speeds, producing
        // "tall pillars rising from the ground." Horizontal outburst is the
        // dominant visual of a real explosion; the vertical component is
        // secondary. Cap it at 40% of the horizontal speed.
        let mut vel = dir * speed;
        let h_speed = vel.xz().length();
        if vel.y > h_speed * 0.4 {
            vel.y = h_speed * 0.4;
        }

        let h = small_hash(seed, i as u32);
        let spin = Vec3::new(
            ((h & 0xFF) as f32 / 255.0 - 0.5) * 2.0,
            (((h >> 8) & 0xFF) as f32 / 255.0 - 0.5) * 2.0,
            (((h >> 16) & 0xFF) as f32 / 255.0 - 0.5) * 2.0,
        ) * BLAST_SPIN_MAX;

        // `apply_impulse` scales its first argument by inverse mass, so
        // passing `vel * mass` yields exactly `vel` velocity.
        phys.apply_impulse(id, vel * mass, spin);
    }
}

/// Roughly this fraction of an explosion's removed voxels become small
/// flying debris chips instead of simply vanishing into air -- capped in
/// absolute count (below) so a huge terrain crater doesn't spawn hundreds
/// of one-voxel bodies.
const DEBRIS_CHIP_FRACTION: f32 = 0.12;
/// Absolute cap on chips per blast, regardless of how much was removed.
const MAX_DEBRIS_CHIPS: usize = 40;
/// Chip outward speed falloff, as a fraction of the main blast impulse's
/// own falloff shape (see [`apply_blast_impulse`]).
const DEBRIS_CHIP_SPEED_SCALE: f32 = 0.6;
/// Hard ceiling on a debris chip's launch speed, in m/s. Unlike
/// `apply_blast_impulse` (which divides by `sqrt(mass)` for real structural
/// fragments, appropriately slowing down heavy ones), a chip's mass is
/// always tiny (one voxel) and barely varies, so that same division would
/// blow up to an unrealistic speed for anything near the blast center --
/// launching light rubble clear across the map instead of scattering it
/// visibly around the crater. Chips ignore mass entirely and are simply
/// capped: "a chunk gets knocked a few meters," not "a bullet."
const DEBRIS_CHIP_MAX_SPEED_M_S: f32 = 6.0;
/// Max angular speed (rad/s) randomly given to a debris chip. Deliberately
/// tiny (not zero -- a little tumble as it flies out still reads well): a
/// small two-voxel chip only ever rests on one or two contact points, so
/// friction has very little leverage to damp rotation once it's on the
/// ground. Unlike a normal structural fragment, any real spin here can
/// take a very long time to settle out even after position and linear
/// velocity are already stable -- and a hard collision in flight can add
/// much more on top of this regardless, so this only bounds our own
/// contribution, not the total a chip might end up spinning at.
const DEBRIS_CHIP_SPIN_MAX: f32 = 0.3;

/// Turn a deterministic sample of `removed` into small (single-voxel)
/// flying debris chips with outward-radial velocity from `center_m`, scaled
/// by `power` — a bomb should leave visible rubble scattered around the
/// crater it just carved, not a clean void with nothing to show for it.
/// The voxels themselves are assumed already cleared to air by the
/// caller's carve; this only adds new bodies representing some of what was
/// there. Deterministic from `seed` (same seed, same chips, same throw).
fn spawn_debris_chips(
    phys: &mut PhysicsWorld,
    registry: &MaterialRegistry,
    removed: &[(IVec3, Voxel)],
    voxel_size_m: f32,
    center_m: Vec3,
    power: f32,
    seed: u32,
) -> Vec<BodyId> {
    if removed.is_empty() {
        return Vec::new();
    }
    let target =
        ((removed.len() as f32 * DEBRIS_CHIP_FRACTION) as usize).clamp(1, MAX_DEBRIS_CHIPS);

    // Deterministic sample: rank every removed voxel by a seeded hash and
    // take the lowest `target`. A full sort here would cost O(n log n) over
    // *every* removed voxel just to keep a couple dozen -- for a large
    // terrain blast (tens of thousands of removed voxels) that dwarfs
    // everything else `blast` does. A partial selection only needs to
    // partition the smallest `target` to the front, without fully
    // ordering the rest: O(n) average instead.
    let mut ranked: Vec<(u32, usize)> = removed
        .iter()
        .enumerate()
        .map(|(i, _)| (small_hash(seed, i as u32), i))
        .collect();
    if target < ranked.len() {
        ranked.select_nth_unstable_by_key(target - 1, |&(h, _)| h);
        ranked.truncate(target);
    }

    // Build a map of removed voxels for O(1) adjacency + material lookup.
    let removed_map: FxHashMap<IVec3, Voxel> = removed.iter().copied().collect();
    let removed_set: FxHashSet<IVec3> = removed_map.keys().copied().collect();

    let mut ids = Vec::with_capacity(target);
    for &(h, i) in ranked.iter().take(target) {
        let (seed_vox, _mat) = removed[i];

        // Build the chip from the actual removed voxels: flood-fill from
        // the seed through the removed set (6-connected), up to a small
        // cap. This produces a chunk shaped like the real carved material
        // — a piece of tree trunk, a fragment of stone wall — not a
        // generic template shape.
        let cluster = cluster_removed(&removed_set, seed_vox, 8);
        if cluster.is_empty() {
            continue;
        }
        let mut min = IVec3::splat(i32::MAX);
        let mut max = IVec3::splat(i32::MIN);
        for &v in &cluster {
            min = min.min(v);
            max = max.max(v);
        }
        let dims = max - min + IVec3::ONE;
        let mut voxels = vec![AIR; (dims.x * dims.y * dims.z) as usize];
        for &v in &cluster {
            let mat = removed_map.get(&v).copied().unwrap_or(AIR);
            let l = v - min;
            voxels[(l.x + l.z * dims.x + l.y * dims.x * dims.z) as usize] = mat;
        }
        let grid = VoxelGrid::new(dims, voxels);
        // Position the body so the cluster's world-space voxels land where
        // they were carved from. Body::from_grid places COM; we pass the
        // cluster's bounding-box center as the initial position, which
        // from_grid corrects to the actual COM.
        let cluster_center_m = voxel_center_m(min, voxel_size_m)
            + dims.as_vec3() * voxel_size_m * 0.5;
        let Some(mut body) = Body::from_grid(grid, registry, voxel_size_m, cluster_center_m)
        else {
            continue;
        };

        let offset = cluster_center_m - center_m;
        let dist = offset.length();
        let dir = if dist > 1e-6 { offset / dist } else { Vec3::Y };
        let speed = (power * DEBRIS_CHIP_SPEED_SCALE / dist.max(BLAST_MIN_DIST_M))
            .min(DEBRIS_CHIP_MAX_SPEED_M_S);
        body.vel = dir * speed;

        let h2 = small_hash(h, i as u32);
        body.omega = Vec3::new(
            ((h2 & 0xFF) as f32 / 255.0 - 0.5) * 2.0,
            (((h2 >> 8) & 0xFF) as f32 / 255.0 - 0.5) * 2.0,
            (((h2 >> 16) & 0xFF) as f32 / 255.0 - 0.5) * 2.0,
        ) * DEBRIS_CHIP_SPIN_MAX;

        ids.push(phys.spawn(body));
    }
    ids
}

/// Flood-fill through `removed` from `seed`, up to `cap` voxels. Returns
/// the cluster of removed voxels connected to the seed (6-connected), up
/// to the cap. Used to build chips shaped like the actual carved material
/// instead of generic templates.
///
/// Returns an empty vec if the cluster is degenerate: fewer than 3 voxels
/// (useless — no torque-free axis of rotation, bad inertia tensor) or
/// all voxels collinear (a straight bar — torque-free along its axis,
/// physically degenerate spin).
pub(crate) fn cluster_removed(removed: &FxHashSet<IVec3>, seed: IVec3, cap: usize) -> Vec<IVec3> {
    let mut cluster = Vec::with_capacity(cap);
    let mut visited = FxHashSet::default();
    let mut queue = VecDeque::new();
    queue.push_back(seed);
    visited.insert(seed);
    while let Some(v) = queue.pop_front() {
        cluster.push(v);
        if cluster.len() >= cap {
            break;
        }
        for d in DIRS {
            let n = v + d;
            if removed.contains(&n) && visited.insert(n) {
                queue.push_back(n);
            }
        }
    }
    // Reject degenerate clusters: too small or all-collinear.
    if cluster.len() < 3 {
        return Vec::new();
    }
    if is_collinear(&cluster) {
        return Vec::new();
    }
    cluster
}

/// True if all voxels in `cluster` lie on a single axis-aligned line.
fn is_collinear(cluster: &[IVec3]) -> bool {
    let min = cluster.iter().copied().fold(IVec3::splat(i32::MAX), IVec3::min);
    let max = cluster.iter().copied().fold(IVec3::splat(i32::MIN), IVec3::max);
    let span = max - min;
    // Collinear if at most one axis has nonzero span.
    span.x.min(1) + span.y.min(1) + span.z.min(1) <= 1
}

/// Carve a jagged [`ExplosionShape`] (not a plain sphere -- see its docs),
/// detach anything left unsupported, give the new debris a blast impulse,
/// scatter a sample of the carved-away material as small flying debris
/// chips instead of letting it simply vanish (see [`spawn_debris_chips`] --
/// a bomb should leave visible rubble around the crater, not a clean void),
/// and wake any resting bodies the carve disturbed. `power` is the blast
/// strength (pass [`vox_core::Tunables::blast_power`] for the live-tunable
/// default); `seed` drives the crater's shape, the debris chip sample and
/// their spin, and the detached-fragment impulse, so the whole blast is
/// reproducible from one seed.
pub fn blast(
    world: &mut World,
    phys: &mut PhysicsWorld,
    registry: &MaterialRegistry,
    center_m: Vec3,
    radius_m: f32,
    power: f32,
    seed: u32,
) -> CarveResult {
    let mut carve = carve_explosion(world, center_m, radius_m, seed);
    let removed_positions: Vec<IVec3> = carve.removed.iter().map(|&(v, _)| v).collect();
    let mut ids = detach_unsupported(world, phys, registry, &removed_positions);
    apply_blast_impulse(phys, &ids, center_m, power, seed);

    let s = world.cfg.voxel_size_m;
    ids.extend(spawn_debris_chips(
        phys,
        registry,
        &carve.removed,
        s,
        center_m,
        power,
        seed,
    ));

    phys.wake_region(carve.region.0.as_vec3() * s, carve.region.1.as_vec3() * s);
    carve.spawned = ids;
    carve
}

/// Carve a tunnel from `start_m` to `end_m` and detach anything left
/// unsupported, waking any resting bodies the carve disturbed. Unlike
/// [`blast`], spawned debris gets no impulse — a beam is a precise,
/// instantaneous cut, not an explosion; severed material simply falls.
/// `end_m` may be far beyond any reasonable world size ("infinite range");
/// [`carve_capsule`] clamps its own search box to the world's actual
/// bounds, so this stays cheap regardless.
pub fn laser(
    world: &mut World,
    phys: &mut PhysicsWorld,
    registry: &MaterialRegistry,
    start_m: Vec3,
    end_m: Vec3,
    radius_m: f32,
) -> CarveResult {
    let mut carve = carve_capsule(world, start_m, end_m, radius_m);
    let removed_positions: Vec<IVec3> = carve.removed.iter().map(|&(v, _)| v).collect();
    let ids = detach_unsupported(world, phys, registry, &removed_positions);

    let s = world.cfg.voxel_size_m;
    phys.wake_region(carve.region.0.as_vec3() * s, carve.region.1.as_vec3() * s);
    carve.spawned = ids;
    carve
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::WorldConfig;
    use vox_core::consts::PHYSICS_DT;

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

    /// The positions cleared by `world.fill_box(min, max, AIR)` over the
    /// given box — the `removed` argument `detach_unsupported` expects.
    fn box_positions(min: IVec3, max: IVec3) -> Vec<IVec3> {
        let mut v = Vec::new();
        for z in min.z..max.z {
            for y in min.y..max.y {
                for x in min.x..max.x {
                    v.push(IVec3::new(x, y, z));
                }
            }
        }
        v
    }

    #[test]
    fn two_pillars_cut_one_nothing_falls() {
        let mut world = two_pillar_bridge();
        let cut_min = IVec3::new(5, 0, 5);
        let cut_max = IVec3::new(6, 10, 6);
        world.fill_box(cut_min, cut_max, AIR);
        let removed = box_positions(cut_min, cut_max);

        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let ids = detach_unsupported(&mut world, &mut phys, &reg, &removed);

        assert!(ids.is_empty(), "bridge still anchored via the other pillar");
        assert_eq!(world.get_voxel(IVec3::new(12, 10, 5)), STONE);
    }

    #[test]
    fn severed_tall_column_detaches_its_full_upper_section() {
        // A 60-voxel-tall, 1-wide pillar resting on the floor, severed near
        // its base. The flood must follow the pillar all the way to its top
        // (57 voxels away) and correctly find no floor and no cap-exceeding
        // mass along the way, despite there being no bounding box telling it
        // how far to look.
        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [32.0, 96.0, 32.0],
            ..WorldConfig::default()
        });
        world.fill_box(IVec3::new(5, 0, 5), IVec3::new(6, 5, 6), STONE);
        world.fill_box(IVec3::new(5, 5, 5), IVec3::new(6, 65, 6), STONE); // 60 voxels tall
        let cut_min = IVec3::new(5, 5, 5);
        let cut_max = IVec3::new(6, 8, 6);
        world.fill_box(cut_min, cut_max, AIR); // sever at y=5..8
        let removed = box_positions(cut_min, cut_max);

        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let ids = detach_unsupported(&mut world, &mut phys, &reg, &removed);

        assert_eq!(ids.len(), 1, "the 57-voxel upper section must detach");
        let body = phys.get(ids[0]).expect("alive");
        assert_eq!(
            body.grid.solid_count(),
            65 - 8,
            "must capture the FULL upper section"
        );
    }

    #[test]
    fn a_large_but_bounded_component_still_detaches_not_misreported_as_anchored() {
        // A 100,000-voxel floating block (bigger than the *old* shared cap
        // of 65_536 that used to answer both "is this obviously anchored"
        // and "is this too big for one rigidbody") connected to an anchored
        // floor only by a thin bridge. This is the exact failure mode a real
        // severed tree hits: a large, genuinely disconnected, genuinely
        // *bounded* component must never be misclassified as anchored just
        // because it's big -- "big" and "anchored" are different questions.
        // Regression test for a bug where `flood_from`'s early-bailout and
        // the body-size policy shared one constant, so any component over
        // 65_536 voxels was wrongly reported as anchored and silently left
        // resident (and unsimulated) in the world.
        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [128.0, 64.0, 128.0],
            ..WorldConfig::default()
        });
        world.fill_box(IVec3::new(0, 0, 0), IVec3::new(10, 10, 10), STONE); // anchored floor block
        world.fill_box(IVec3::new(10, 4, 4), IVec3::new(11, 6, 6), STONE); // bridge
        // 50x40x50 = 100,000 voxels, floating just past the bridge.
        world.fill_box(IVec3::new(11, 4, 4), IVec3::new(61, 44, 54), STONE);

        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let carve = carve_sphere(&mut world, Vec3::new(10.5, 5.0, 5.0), 0.8);
        assert_eq!(carve.removed.len(), 4, "must remove exactly the bridge");
        let removed: Vec<IVec3> = carve.removed.iter().map(|&(v, _)| v).collect();

        let ids = detach_unsupported(&mut world, &mut phys, &reg, &removed);
        assert_eq!(ids.len(), 1, "the 100,000-voxel block must detach");
        let body = phys.get(ids[0]).expect("alive");
        assert_eq!(body.grid.solid_count(), 100_000);
        assert_eq!(
            world.get_voxel(IVec3::new(5, 5, 5)),
            STONE,
            "floor untouched"
        );
    }

    #[test]
    fn thin_structure_atop_a_huge_terrain_mass_still_detaches() {
        // The scenario that broke the old bounding-box-growth design: a
        // thin, tree-trunk-like column standing on top of a large contiguous
        // slab of "terrain" (100x100 voxels, much bigger than any reasonable
        // search box). Severing the column partway up must still detach its
        // top, and must do so without the huge terrain mass anywhere nearby
        // fooling the analysis into treating the severed top as anchored.
        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [128.0, 64.0, 128.0],
            ..WorldConfig::default()
        });
        world.fill_box(IVec3::new(0, 0, 0), IVec3::new(100, 3, 100), STONE); // huge terrain slab
        world.fill_box(IVec3::new(50, 3, 50), IVec3::new(51, 33, 51), STONE); // 30-voxel trunk
        let cut_min = IVec3::new(50, 10, 50);
        let cut_max = IVec3::new(51, 13, 51);
        world.fill_box(cut_min, cut_max, AIR); // sever partway up the trunk
        let removed = box_positions(cut_min, cut_max);

        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let ids = detach_unsupported(&mut world, &mut phys, &reg, &removed);

        assert_eq!(ids.len(), 1, "the severed trunk top must detach");
        let body = phys.get(ids[0]).expect("alive");
        assert_eq!(
            body.grid.solid_count(),
            33 - 13,
            "must capture the full top"
        );
        // The huge terrain slab must be completely untouched.
        assert_eq!(world.get_voxel(IVec3::new(10, 1, 10)), STONE);
        assert_eq!(world.get_voxel(IVec3::new(90, 1, 90)), STONE);
    }

    #[test]
    fn breaking_a_voxel_deep_in_huge_terrain_detaches_nothing_and_is_bounded() {
        // Breaking a single voxel out of the middle of a large contiguous
        // terrain mass must (a) detach nothing -- the surrounding mass is
        // obviously still there and anchored -- and (b) do so in bounded
        // work, not by rescanning an ever-larger region. This is the
        // performance-sensitive case: it must stay fast regardless of how
        // large the world is, since real terrain is effectively unbounded.
        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [200.0, 40.0, 200.0],
            ..WorldConfig::default()
        });
        world.fill_box(IVec3::new(0, 0, 0), IVec3::new(200, 20, 200), STONE);
        let broken = IVec3::new(100, 10, 100);
        world.set_voxel(broken, AIR);

        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let start = std::time::Instant::now();
        let ids = detach_unsupported(&mut world, &mut phys, &reg, &[broken]);
        let elapsed = start.elapsed();

        assert!(ids.is_empty(), "surrounding terrain is obviously anchored");
        assert_eq!(phys.body_count(), 0);
        assert!(
            elapsed < std::time::Duration::from_millis(200),
            "a single-voxel break in terrain must resolve quickly, took {elapsed:?}"
        );
    }

    #[test]
    fn laser_tunnels_through_a_wall_and_detaches_the_severed_top() {
        // A thick wall with a pillar-supported overhang above head height;
        // firing the beam straight through the wall at head height must
        // punch a tunnel clear through the wall (not just crater one face)
        // while leaving the overhang's own support (elsewhere) untouched.
        let mut world = World::new(WorldConfig {
            voxel_size_m: 0.5,
            extent_m: [40.0, 20.0, 40.0],
            ..WorldConfig::default()
        });
        world.fill_box(IVec3::new(0, 0, 0), IVec3::new(80, 4, 80), STONE); // floor
        world.fill_box(IVec3::new(30, 4, 0), IVec3::new(34, 40, 80), STONE); // thick wall

        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let start_m = Vec3::new(0.0, 6.0, 20.0);
        let end_m = Vec3::new(1_000.0, 6.0, 20.0); // "infinite" range
        let carve = laser(&mut world, &mut phys, &reg, start_m, end_m, 1.0);

        assert!(carve.removed.len() > 20, "must carve a real tunnel");
        // The tunnel must reach clean through both faces of the wall.
        assert_eq!(
            world.get_voxel(IVec3::new(31, 12, 40)),
            AIR,
            "near face open"
        );
        assert_eq!(
            world.get_voxel(IVec3::new(66, 12, 40)),
            AIR,
            "far face open"
        );
        // Material well above and below the tunnel survives.
        assert_eq!(world.get_voxel(IVec3::new(31, 35, 40)), STONE);
        assert_eq!(world.get_voxel(IVec3::new(31, 4, 40)), STONE);
    }

    #[test]
    fn laser_range_is_clamped_to_world_bounds_and_stays_fast() {
        // An "infinite" beam fired across a modest world must not cost more
        // than the world it actually passes through.
        let mut world = World::new(WorldConfig {
            voxel_size_m: 1.0,
            extent_m: [64.0, 32.0, 64.0],
            ..WorldConfig::default()
        });
        world.fill_box(IVec3::new(0, 0, 0), IVec3::new(64, 10, 64), STONE);

        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let start = std::time::Instant::now();
        laser(
            &mut world,
            &mut phys,
            &reg,
            Vec3::new(-500.0, 5.0, 32.0),
            Vec3::new(5_000.0, 5.0, 32.0),
            1.5,
        );
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "an out-of-range beam must clamp to world bounds, took {elapsed:?}"
        );
    }

    #[test]
    fn cut_both_slab_detaches() {
        let mut world = two_pillar_bridge();
        let cut_a = (IVec3::new(5, 0, 5), IVec3::new(6, 10, 6));
        let cut_b = (IVec3::new(20, 0, 5), IVec3::new(21, 10, 6));
        world.fill_box(cut_a.0, cut_a.1, AIR);
        world.fill_box(cut_b.0, cut_b.1, AIR);
        let mut removed = box_positions(cut_a.0, cut_a.1);
        removed.extend(box_positions(cut_b.0, cut_b.1));

        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let ids = detach_unsupported(&mut world, &mut phys, &reg, &removed);
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
        // past it.
        world.fill_box(IVec3::new(0, 0, 0), IVec3::new(10, 10, 10), STONE);
        world.fill_box(IVec3::new(10, 4, 4), IVec3::new(11, 6, 6), STONE); // bridge, 4 voxels
        world.fill_box(IVec3::new(11, 4, 4), IVec3::new(12, 6, 6), STONE); // knob, 4 voxels

        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let carve = carve_sphere(&mut world, Vec3::new(10.5, 5.0, 5.0), 0.8);
        assert_eq!(carve.removed.len(), 4, "must remove exactly the bridge");
        let removed: Vec<IVec3> = carve.removed.iter().map(|&(v, _)| v).collect();

        let ids = detach_unsupported(&mut world, &mut phys, &reg, &removed);
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
        let isolated = IVec3::new(5, 5, 5);
        world.set_voxel(isolated, STONE); // isolated single voxel

        // Removing one of the anchored block's own voxels exposes a
        // neighbor of the isolated voxel too, so route seeds through it: cut
        // a face voxel adjacent to `isolated` instead, matching how a real
        // break would expose it. Simplest: treat `isolated` itself as if it
        // were just exposed by carving away everything around it.
        let removed = vec![
            IVec3::new(4, 5, 5),
            IVec3::new(6, 5, 5),
            IVec3::new(5, 4, 5),
            IVec3::new(5, 6, 5),
            IVec3::new(5, 5, 4),
            IVec3::new(5, 5, 6),
        ];

        let reg = registry();
        let mut phys = PhysicsWorld::new();
        let ids = detach_unsupported(&mut world, &mut phys, &reg, &removed);

        assert!(ids.is_empty(), "1 voxel is below DEBRIS_MIN_VOXELS");
        assert_eq!(world.get_voxel(isolated), AIR, "must be removed as dust");
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

    /// The whole point of `ExplosionShape`: it must remove strictly more
    /// material than a plain sphere of the same radius (the shrapnel spikes
    /// reach past the base crater), so a Bomb reads as more than "a clean
    /// hole".
    #[test]
    fn explosion_removes_more_than_a_plain_sphere_of_the_same_radius() {
        let mut world_sphere = test_world();
        world_sphere.fill_box(IVec3::new(0, 0, 0), IVec3::new(20, 20, 20), STONE);
        let mut world_explosion = test_world();
        world_explosion.fill_box(IVec3::new(0, 0, 0), IVec3::new(20, 20, 20), STONE);

        let center = Vec3::new(10.0, 10.0, 10.0);
        let radius = 3.0;
        let plain = carve_sphere(&mut world_sphere, center, radius);
        let jagged = carve_explosion(&mut world_explosion, center, radius, 42);

        assert!(
            jagged.removed.len() > plain.removed.len(),
            "explosion ({}) must remove more than a plain sphere ({})",
            jagged.removed.len(),
            plain.removed.len()
        );
        // The shrapnel must actually reach past the base radius somewhere.
        let reaches_past_radius = jagged
            .removed
            .iter()
            .any(|&(v, _)| (voxel_center_m(v, 1.0) - center).length() > radius + 0.5);
        assert!(
            reaches_past_radius,
            "spikes must extend past the base crater"
        );
    }

    /// Same center, radius, and seed must carve an identical shape every
    /// time -- a replay, or a test, must be able to rely on that.
    #[test]
    fn explosion_shape_is_deterministic_given_the_same_seed() {
        let mut world_a = test_world();
        world_a.fill_box(IVec3::new(0, 0, 0), IVec3::new(20, 20, 20), STONE);
        let mut world_b = test_world();
        world_b.fill_box(IVec3::new(0, 0, 0), IVec3::new(20, 20, 20), STONE);

        let center = Vec3::new(10.0, 10.0, 10.0);
        let a = carve_explosion(&mut world_a, center, 3.0, 1234);
        let b = carve_explosion(&mut world_b, center, 3.0, 1234);

        let mut a_sorted: Vec<IVec3> = a.removed.iter().map(|&(v, _)| v).collect();
        let mut b_sorted: Vec<IVec3> = b.removed.iter().map(|&(v, _)| v).collect();
        a_sorted.sort_by_key(|v| (v.x, v.y, v.z));
        b_sorted.sort_by_key(|v| (v.x, v.y, v.z));
        assert_eq!(a_sorted, b_sorted, "same seed must carve the same shape");
    }

    /// The actual gameplay payoff, isolated from chance: a hub with four
    /// thin bridges reaching out along +X/-X/+Z/-Z, each starting just
    /// *beyond* the base crater's radius -- a plain sphere can't touch them
    /// at all, but the longer-reaching shrapnel spikes (1.2x-2.4x the base
    /// radius) can sever one, detaching that arm's outer block as its own
    /// fragment. A plain `carve_sphere` on the identical layout must detach
    /// nothing (nothing it removes was ever load-bearing for the arms).
    #[test]
    fn shrapnel_spikes_reach_past_the_base_radius_and_sever_a_bridge_a_sphere_would_miss() {
        const RADIUS: f32 = 2.0;
        const S: f32 = 0.2;
        let hub = IVec3::new(40, 40, 40);

        // A central hub with four thin bridges (each starting just *past*
        // the base crater radius) reaching out along +X/-X/+Z/-Z, each
        // ending in its own small outer block.
        fn build_world(hub: IVec3) -> World {
            let mut world = World::new(WorldConfig {
                voxel_size_m: S,
                extent_m: [32.0, 32.0, 32.0],
                ..WorldConfig::default()
            });
            world.fill_box(hub - IVec3::splat(2), hub + IVec3::splat(3), STONE); // hub
            let bridge_start_vox = ((RADIUS + 0.3) / S) as i32;
            let bridge_end_vox = bridge_start_vox + 6;
            for axis in [0usize, 2] {
                for sign in [-1, 1] {
                    let mut lo = hub - IVec3::splat(1);
                    let mut hi = hub + IVec3::splat(2);
                    lo[axis] = hub[axis] + sign * bridge_end_vox;
                    hi[axis] = hub[axis] + sign * bridge_start_vox;
                    let (min, max) = (lo.min(hi), lo.max(hi) + IVec3::ONE);
                    world.fill_box(min, max, STONE); // one bridge + its outer block
                }
            }
            world
        }

        let center_m = voxel_center_m(hub, S);
        let reg = registry();

        let mut world_sphere = build_world(hub);
        let sphere = carve_sphere(&mut world_sphere, center_m, RADIUS);
        let sphere_removed: Vec<IVec3> = sphere.removed.iter().map(|&(v, _)| v).collect();
        let mut phys_sphere = PhysicsWorld::new();
        let sphere_ids =
            detach_unsupported(&mut world_sphere, &mut phys_sphere, &reg, &sphere_removed);
        assert!(
            sphere_ids.is_empty(),
            "a plain sphere of this radius must not reach any bridge"
        );

        let found_a_severing_seed = (0..40u32).any(|seed| {
            let mut probe = build_world(hub);
            let carve = carve_explosion(&mut probe, center_m, RADIUS, seed);
            let removed: Vec<IVec3> = carve.removed.iter().map(|&(v, _)| v).collect();
            let mut phys = PhysicsWorld::new();
            !detach_unsupported(&mut probe, &mut phys, &reg, &removed).is_empty()
        });
        assert!(
            found_a_severing_seed,
            "at least one seed's shrapnel spikes must reach far enough to \
             sever a bridge a plain sphere of the same radius cannot touch"
        );
    }

    #[test]
    fn blast_wakes_and_moves_spawned_debris() {
        let mut world = two_pillar_bridge();
        let reg = registry();
        let mut phys = PhysicsWorld::new();

        // Blast pillar A's base away entirely (radius covers the 1x10x1
        // column); pillar B remains, so nothing should detach yet -- verify
        // the plain carve+detach compose correctly through `blast`.
        let carve = blast(
            &mut world,
            &mut phys,
            &reg,
            Vec3::new(5.5, 15.0, 5.5),
            12.0,
            40.0,
            7,
        );
        assert!(carve.removed.len() > 10, "must remove a large chunk");
        // Both pillars and the slab are gone or detached; whatever spawned
        // (structural fragments and small flying debris chips alike) must
        // be awake with a nonzero blast velocity. Chips are light and can
        // land far from the center, so their speed can be much smaller than
        // a big structural fragment's -- check for "moving at all", not a
        // fixed minimum.
        for (_, body) in phys.iter() {
            assert!(!body.sleep.asleep);
            assert!(body.vel.length() > 0.0, "blast must impart velocity");
        }
    }

    #[test]
    fn blast_scatters_debris_chips_instead_of_letting_everything_vanish() {
        let mut world = test_world();
        world.fill_box(IVec3::new(0, 0, 0), IVec3::new(20, 20, 20), STONE);
        let reg = registry();
        let mut phys = PhysicsWorld::new();

        let carve = blast(
            &mut world,
            &mut phys,
            &reg,
            Vec3::new(10.0, 10.0, 10.0),
            3.0,
            40.0,
            5,
        );

        assert!(!carve.removed.is_empty(), "must carve something");
        assert!(
            phys.body_count() > 0,
            "some of the carved-away material must survive as flying debris \
             chips, not just vanish into air"
        );
        for (_, body) in phys.iter() {
            let sc = body.grid.solid_count();
            assert!(
                sc >= 3 && sc <= 8,
                "each chip is a small debris fragment (3-8 voxels from real \
                 voxel clustering), got {sc}"
            );
            assert!(!body.sleep.asleep);
        }
    }

    /// Regression test for a bug caught while building debris chips: a
    /// straight two-voxel bar has an axis of rotation (along its own
    /// length) where every contact point sits exactly on that axis, giving
    /// friction zero torque arm to damp it -- a chip given spin around that
    /// axis keeps that *exact* angular velocity forever, never crosses the
    /// sleep threshold, and never sleeps, no matter how long it rests. An
    /// L-shaped chip (no straight line through all its voxel centers) has
    /// no such axis. Landing on a floor and given a substantial spin, a
    /// chip must actually settle and sleep within a generous but bounded
    /// number of steps.
    #[test]
    fn a_spinning_debris_chip_actually_settles_and_sleeps() {
        // An isolated chip (built exactly the way `spawn_debris_chips` does:
        // a 2x2x1 grid with one corner missing) dropped onto a large, plain
        // floor with a hard spin already going -- deliberately minimal, so
        // a solver instability from some *other* interaction (a blast's
        // full mess of overlapping fragments, say) can't be confused with
        // the specific thing this test checks: does this shape's spin
        // actually damp out and let it sleep, or does it spin forever?
        let mut world = test_world();
        world.fill_box(IVec3::new(0, 0, 0), IVec3::new(32, 10, 32), STONE);

        let reg = registry();
        let mut voxels = vec![STONE; 4];
        voxels[1] = AIR; // an L, not a straight bar or a full square
        let grid = VoxelGrid::new(IVec3::new(2, 2, 1), voxels);
        let mut body =
            Body::from_grid(grid, &reg, 1.0, Vec3::new(16.0, 12.0, 16.0)).expect("massive");
        body.omega = Vec3::new(4.0, -5.0, 6.0);

        let mut phys = PhysicsWorld::new();
        phys.spawn(body);

        for _ in 0..1200 {
            phys.step(&world, PHYSICS_DT);
        }
        assert_eq!(
            phys.awake_count(),
            0,
            "an L-shaped chip's spin must damp out and let it sleep"
        );
    }
}
