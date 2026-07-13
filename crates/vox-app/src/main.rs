//! Voxel engine application: world, player, tools, threaded remeshing, render.

mod args;
#[cfg(feature = "mario")]
mod audio;
mod body_mesh;
mod chunk_loader;
mod day_night;
mod grass;
#[cfg(feature = "mario")]
mod mario;
#[cfg(not(feature = "mario"))]
#[path = "mario_disabled.rs"]
mod mario;
mod particles;
mod player;
mod remesh;
mod replay;
mod ecs_components;
mod systems;
mod tools;

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use glam::{IVec3, Mat4, Vec3};
use rayon::prelude::*;
use args::Quality;
use body_mesh::BodyMeshQueue;
use chunk_loader::ChunkLoader;
use particles::{Burst, ParticleSystem};
use player::Player;
use remesh::RemeshQueue;
use tools::{CarveOutcome, HOTBAR, Tool, Tools};

use vox_core::consts::{CHUNK_SIZE, REACH, SLEEP_FRAMES};
use vox_core::{
    FrameProfile, FxHashMap, FxHashSet, MaterialId, MaterialRegistry, ScopedTimer, Tunables,
    WorldConfig, chunk_origin, voxel_at,
};
use vox_debug::hud::HudState;
use vox_debug::{DebugOverlay, OverlayState};
use vox_gen::{TerrainGen, TerrainMaterials, TreeMaterials};
use vox_mesh::{VoxelSlab, mesh_slab};
use vox_physics::{Body, BodyId, ImpactEvent, PhysicsWorld, VoxelGrid};
use vox_platform::{App, FrameControl, FrameTiming, InputState, run_app};
use vox_render::{
    BodyMeshKey, Camera, Frustum, Gpu, ParticlePipeline, ShadowPipeline, VoxelPipeline,
};
use vox_world::{AIR, Voxel, World};
use winit::event::MouseButton;
use winit::keyboard::KeyCode;
use winit::window::{CursorGrabMode, Window};

/// Fog end distance in meters.
const FOG_END_M: f32 = 220.0;

/// Locate the `assets/` directory: the workspace copy during development,
/// else `assets/` beside the executable.
fn assets_dir() -> PathBuf {
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets");
    if dev.is_dir() {
        return dev;
    }
    PathBuf::from("assets")
}

/// Build an empty streaming world: validates the config, creates the
/// `World`, attaches the solidity table — but performs NO upfront terrain
/// or tree generation. The `ChunkLoader` generates chunks lazily around
/// the player instead.
fn build_streaming_world(
    cfg: WorldConfig,
    registry: &MaterialRegistry,
) -> Result<World, Box<dyn std::error::Error + Send + Sync>> {
    cfg.validate()?;
    let mut world = World::new(cfg);
    world.set_solid_table(solid_table(registry));
    // No upfront generation — ChunkLoader generates lazily.
    Ok(world)
}

/// Build `World`'s per-material-id solidity table from the registry (index
/// = material id, air included). See `World::set_solid_table`'s doc comment
/// for why this must be attached before any gameplay system (raycasts,
/// character collision, rigidbody contacts, the destruction flood) runs.
fn solid_table(registry: &MaterialRegistry) -> Vec<bool> {
    (0..registry.len())
        .map(|i| {
            registry
                .get(vox_core::MaterialId(i as u16))
                .is_some_and(|d| d.solid)
        })
        .collect()
}

/// Weathering material table, or `None` (weathering disabled) if any
/// required material is missing from the asset set. A missing material name
/// disables weathering with a log line, not a crash.
fn weather_table(registry: &MaterialRegistry) -> Option<vox_sim::WeatherTable> {
    let id = |name: &str| registry.id_by_name(name).map(|m| Voxel(m.0));
    Some(vox_sim::WeatherTable {
        water: id("water")?,
        stone: id("stone")?,
        grass: id("grass")?,
        dirt: id("dirt")?,
        mud: id("mud")?,
        sand: id("sand")?,
        muddy_water: id("muddy_water")?,
    })
}

/// Fire material table, or `None` (fire disabled) if any required material
/// is missing from the asset set.
fn fire_table(registry: &MaterialRegistry) -> Option<vox_sim::FireTable> {
    let id = |name: &str| registry.id_by_name(name).map(|m| Voxel(m.0));
    let water = id("water")?;
    let wood = id("wood")?;
    let leaves = id("leaves")?;
    let planks = id("planks")?;
    let grass = id("grass")?;
    let ember = id("ember")?;
    let char = id("char")?;
    let ash = id("ash")?;
    let muddy_water = id("muddy_water").unwrap_or(id("water")?);
    let dark_ash = id("dark_ash")?;
    let flammable = (1..registry.len())
        .filter_map(|i| {
            let def = registry.get(MaterialId(i as u16))?;
            def.flammable.then(|| Voxel(i as u16))
        })
        .collect();
    Some(vox_sim::FireTable {
        water,
        ember,
        char,
        ash,
        dark_ash,
        wood,
        leaves,
        planks,
        grass,
        flammable,
        muddy_water,
    })
}

/// All materials the registry marks as powders, as `Voxel` ids. Empty if
/// the asset set defines no powders -- the sim runs water-only. The sim
/// handles each powder with `step_powder` (fall + diagonal slide, no
/// spreading); see `FluidSim::with_powders`.
fn powder_materials(registry: &MaterialRegistry) -> Vec<Voxel> {
    (1..registry.len())
        .filter_map(|i| {
            let def = registry.get(vox_core::MaterialId(i as u16))?;
            def.powder.then(|| Voxel(i as u16))
        })
        .collect()
}

/// All materials the registry marks as fluids, as `Voxel` ids. Empty if
/// the asset set defines no fluids -- the sim runs powder-only. The sim
/// handles each fluid with the full CA rule (fall, spread, level).
fn fluid_materials(registry: &MaterialRegistry) -> Vec<Voxel> {
    (1..registry.len())
        .filter_map(|i| {
            let def = registry.get(vox_core::MaterialId(i as u16))?;
            def.fluid.then(|| Voxel(i as u16))
        })
        .collect()
}

/// All fluid materials with their densities, as (Voxel, density) pairs.
/// Used for per-fluid buoyancy — muddy water (1100) provides more buoyancy
/// than clean water (1000).
fn fluid_densities(registry: &MaterialRegistry) -> Vec<(Voxel, f32)> {
    (1..registry.len())
        .filter_map(|i| {
            let def = registry.get(vox_core::MaterialId(i as u16))?;
            def.fluid.then(|| (Voxel(i as u16), def.density))
        })
        .collect()
}

/// Place smoke just inside the face-adjacent air voxel selected by FireSim,
/// with tangent-only jitter so the origin can never be pushed back through
/// the burning surface. `salt` varies repeat emissions from the same cell.
fn fire_smoke_origin(pos: IVec3, face: IVec3, voxel_size_m: f32, salt: u32) -> Vec3 {
    debug_assert!(
        [
            IVec3::X,
            IVec3::NEG_X,
            IVec3::Y,
            IVec3::NEG_Y,
            IVec3::Z,
            IVec3::NEG_Z,
        ]
        .contains(&face)
    );

    fn mix(mut x: u32) -> u32 {
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^ (x >> 16)
    }

    let seed = mix((pos.x as u32).wrapping_mul(0x9e37_79b9)
        ^ (pos.y as u32).wrapping_mul(0x85eb_ca6b)
        ^ (pos.z as u32).wrapping_mul(0xc2b2_ae35)
        ^ salt.wrapping_mul(0x27d4_eb2d));
    let jitter_a = (seed & 0xffff) as f32 / 65_535.0 - 0.5;
    let jitter_b = (mix(seed ^ 0xa511_e9b3) & 0xffff) as f32 / 65_535.0 - 0.5;
    let normal = face.as_vec3();
    let tangent_a = if face.y != 0 { Vec3::X } else { Vec3::Y };
    let tangent_b = normal.cross(tangent_a);

    vox_core::voxel_center_m(pos, voxel_size_m)
        + normal * (voxel_size_m * 0.55)
        + tangent_a * (jitter_a * voxel_size_m * 0.4)
        + tangent_b * (jitter_b * voxel_size_m * 0.4)
}

/// One exposed ember outlet on a detached body. The local center is a voxel
/// center in body coordinates; only faces whose adjacent body voxel is air
/// are retained, so interior embers never become smoke sources.
#[derive(Clone, Debug)]
struct BodySmokeOutlet {
    local_center: Vec3,
    open_faces: Vec<IVec3>,
}

#[derive(Clone, Debug)]
struct BodyFireVisual {
    outlets: Vec<BodySmokeOutlet>,
    cursor: usize,
    cooldown: u8,
}

fn body_smoke_outlets(body: &Body, ember: Voxel, voxel_size_m: f32) -> Vec<BodySmokeOutlet> {
    let faces = [
        IVec3::X,
        IVec3::NEG_X,
        IVec3::Y,
        IVec3::NEG_Y,
        IVec3::Z,
        IVec3::NEG_Z,
    ];
    body.surface
        .iter()
        .filter_map(|sample| {
            let local_voxel = ((*sample - body.grid_offset) / voxel_size_m)
                .floor()
                .as_ivec3();
            (body.grid.get(local_voxel) == ember).then(|| BodySmokeOutlet {
                local_center: *sample,
                open_faces: faces
                    .iter()
                    .copied()
                    .filter(|face| body.grid.get(local_voxel + *face) == AIR)
                    .collect(),
            })
        })
        .filter(|outlet| !outlet.open_faces.is_empty())
        .collect()
}

fn body_smoke_origin(
    body: &Body,
    outlet: &BodySmokeOutlet,
    face: IVec3,
    voxel_size_m: f32,
    salt: u32,
) -> Vec3 {
    fn mix(mut x: u32) -> u32 {
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^ (x >> 16)
    }
    let seed = mix(salt.wrapping_mul(0x27d4_eb2d));
    let jitter_a = (seed & 0xffff) as f32 / 65_535.0 - 0.5;
    let jitter_b = (mix(seed ^ 0xa511_e9b3) & 0xffff) as f32 / 65_535.0 - 0.5;
    let normal = face.as_vec3();
    let tangent_a = if face.y != 0 { Vec3::X } else { Vec3::Y };
    let tangent_b = normal.cross(tangent_a);
    let local = outlet.local_center
        + normal * (voxel_size_m * 0.55)
        + tangent_a * (jitter_a * voxel_size_m * 0.4)
        + tangent_b * (jitter_b * voxel_size_m * 0.4);
    body.pos + body.rot * local
}

fn next_body_smoke_origin(
    body: &Body,
    visual: &mut BodyFireVisual,
    world: &World,
    voxel_size_m: f32,
) -> Option<Vec3> {
    if visual.outlets.is_empty() {
        return None;
    }
    let start = visual.cursor % visual.outlets.len();
    for outlet_offset in 0..visual.outlets.len() {
        let outlet_index = (start + outlet_offset) % visual.outlets.len();
        let outlet = &visual.outlets[outlet_index];
        let mut faces = outlet.open_faces.clone();
        faces.sort_by(|a, b| {
            (body.rot * b.as_vec3())
                .y
                .total_cmp(&(body.rot * a.as_vec3()).y)
        });
        for (face_index, face) in faces.into_iter().enumerate() {
            let origin = body_smoke_origin(
                body,
                outlet,
                face,
                voxel_size_m,
                (start + outlet_offset * 7 + face_index) as u32,
            );
            let voxel = voxel_at(origin, voxel_size_m);
            if world.in_bounds(voxel) && world.get_voxel(voxel) == AIR {
                visual.cursor = (outlet_index + 1) % visual.outlets.len();
                return Some(origin);
            }
        }
    }
    visual.cursor = (start + 1) % visual.outlets.len();
    None
}

/// The engine application.
struct VoxApp {
    window: Arc<Window>,
    gpu: Gpu,
    pipeline: VoxelPipeline,
    /// Directional shadow pipeline (#14): renders chunks to a 2048x2048
    /// depth-only shadow map each frame; the main voxel pass samples it to
    /// attenuate sunlight on occluded terrain. Follows the player position
    /// and re-orients with the sun each frame.
    shadow_pipeline: ShadowPipeline,
    world: World,
    /// Player-centered chunk streaming: generates chunks on demand within
    /// render distance, evicts pristine chunks beyond it.
    chunk_loader: ChunkLoader,
    registry: MaterialRegistry,
    player: Player,
    camera: Camera,
    tools: Tools,
    remesh: RemeshQueue,
    /// Threaded debris mesh generation (see `body_mesh`'s module docs for
    /// why bodies don't need the generation/staleness tracking chunks do).
    body_mesh: BodyMeshQueue,
    phys: PhysicsWorld,
    fluid: vox_sim::FluidSim,
    /// Full set of fluid material voxels (water, muddy_water, ...) -- the
    /// materials `mesh_slab` treats as translucent fluids when meshing
    /// chunks and debris. Stored once at construction (see
    /// `fluid_materials`) and reused at every mesh call site so a newly
    /// added fluid renders without each caller being touched.
    fluids: Vec<Voxel>,
    /// Fluid ticks run at their own fixed rate (`FLUID_DT`), independent of
    /// the 60 Hz physics loop -- see the design doc §5.
    fluid_clock: vox_platform::FrameClock,
    /// Water-driven material transformation (grass→dirt→mud, stone→sand),
    /// fed by the fluid tick's contact events. `None` when the asset set
    /// is missing a material weathering needs (see `weather_table`).
    weathering: Option<vox_sim::Weathering>,
    /// Fire simulation: ember ignition, fire spread, consumption to char.
    /// `None` when the asset set is missing a material fire needs.
    fire: Option<vox_sim::FireSim>,
    /// Debris bodies known to carry an ember. Keeping this sparse avoids a
    /// dense scan of every body on each fire tick.
    burning_bodies: FxHashMap<BodyId, BodyFireVisual>,
    /// Incrementing seed so repeated blasts get varied debris spin.
    blast_seed: u32,
    /// Incrementing seed so repeated impact-fracture chips get varied
    /// spin/velocity jitter (same idea as `blast_seed`).
    impact_seed: u32,
    grabbed: bool,
    frames: u32,
    /// Frame count from the most recently *completed* one-second window —
    /// the stable value the overlay displays. `frames` itself is a live
    /// accumulator that counts 0 up to roughly the frame rate and resets
    /// every second, so reading it directly (as the overlay used to) shows
    /// a sawtooth, not an FPS.
    last_fps: u32,
    last_report: Instant,
    /// Live-tunable parameters (friction, blast power, fly speed, ...),
    /// edited by the debug overlay's sliders and synced into the systems
    /// that actually consume them once per frame.
    tunables: Tunables,
    /// Rolling per-phase frame timings, shown in the debug overlay.
    profile: FrameProfile,
    debug_overlay: DebugOverlay,
    debug_visible: bool,
    /// Registry material names (excluding air), cached once for the debug
    /// overlay's material picker.
    material_names: Vec<String>,
    /// Index into `material_names` (not into the registry — the offset from
    /// `Tools`' own 1-based, air-inclusive indexing is handled where it's
    /// synced back after the overlay runs).
    selected_material: usize,
    /// Chunk draw/cull counts from the previous frame — the overlay is built
    /// before this frame's `draw_chunks` runs, so it necessarily shows the
    /// prior frame's numbers (one frame of latency, imperceptible in a HUD).
    last_draw_stats: vox_render::DrawStats,
    /// Every debris body ever spawned, oldest first -- every debris body is
    /// its own GPU buffer set and its own draw call (see
    /// `VoxelPipeline::draw_bodies`'s doc comment on why), so letting this
    /// grow without bound over a play session (repeated bombing especially,
    /// now that a bomb also scatters small chip debris) is the single
    /// biggest driver of the engine bogging down over time. Used by
    /// `enforce_debris_budget` to evict the oldest *settled* debris once the
    /// total exceeds `MAX_DEBRIS_BODIES`.
    debris_order: VecDeque<BodyId>,
    /// Mario mode state. `None` = not initialized yet (ROM not loaded).
    /// `Some` with `is_active() == false` = initialized but Mario not
    /// spawned. `Some` with `is_active() == true` = Mario is running.
    mario_mode: Option<mario::MarioMode>,
    /// SM64 units per meter for Mario mode. Higher = smaller Mario.
    /// Set once from CLI, read when initializing MarioMode.
    #[cfg(feature = "mario")]
    mario_units_per_meter: f32,
    /// Game time accumulator for day/night cycle (seconds).
    game_time: f32,
    /// When true, the day/night cycle is frozen at noon (full daylight).
    /// Toggled via the F3 debug overlay.
    always_day: bool,
    /// Old debris meshes kept alive past their body's despawn, each waiting
    /// on the set of its replacement fragments' async mesh jobs still in
    /// flight -- see `replace_body`'s doc comment for why this exists (a
    /// carved body that's too big to mesh inline would otherwise vanish for
    /// the frame or more its fragments' meshes take).
    pending_body_removal: HashMap<BodyMeshKey, HashSet<BodyMeshKey>>,
    /// CPU-simulated destruction feedback particles (see `particles`).
    particles: ParticleSystem,
    particle_pipeline: ParticlePipeline,
    /// Cel-shading + post-process pipeline: offscreen HDR color + depth
    /// textures, fullscreen edge-detection/saturation/grading pass that
    /// composites the scene to the swapchain.
    postprocess: vox_render::PostProcessPipeline,
    /// SSAO + bloom post-processing pipeline: generates ambient occlusion
    /// and bloom from the scene's HDR color + depth, feeding the results
    /// into the cel-shading composite pass.
    bloom_ssao: vox_render::BloomSsaoPipeline,
    /// Grass blade render pipeline.
    grass_pipeline: vox_render::GrassPipeline,
    /// Cached grass blade vertices (throttled regeneration).
    grass_cache: grass::GrassCache,
    /// In-engine editor mode: when active, LMB paints a sphere of the
    /// selected material at the crosshair target and RMB erases a sphere of
    /// AIR, instead of the normal hotbar tools. Toggled with E. The player
    /// still flies/looks normally — only mouse-button meaning changes.
    editor_active: bool,
    /// Brush radius in meters for editor paint/erase, adjustable with the
    /// mouse wheel while editor mode is active.
    editor_radius: f32,
    /// Undo stack: each entry is a voxel diff (pos, old_voxel) from a world edit.
    undo_stack: Vec<Vec<(IVec3, Voxel)>>,
    /// Redo stack: entries popped from undo_stack by Ctrl+Z, re-applied by Ctrl+Y.
    redo_stack: Vec<Vec<(IVec3, Voxel)>>,
    /// Snapshot-based replay state (record with R, play back with P).
    /// Only captures player + camera + debris body transforms, not the
    /// voxel world -- see `replay` module docs.
    replay: replay::ReplayState,
    /// Procedural crack-decal intensity (0 = off, >0 = visible cracks).
    /// Foundation for damage visualization; not yet wired to actual voxel
    /// damage state — set to 0 by default so cracks are invisible until a
    /// future change drives this from per-voxel damage.
    crack_intensity: f32,
    /// ECS entity world for custom gameplay entities (doors, NPCs, projectiles).
    ecs: vox_ecs::World,
    /// Cross-system data carried between frame phases.
    frame_data: systems::FrameData,
    /// Cached render values from system_render_prep, consumed by the
    /// inline GPU render passes in frame().
    last_view_proj: glam::Mat4,
    last_frustum: vox_render::Frustum,
    last_shadow_frustum: vox_render::Frustum,
}

/// Hard cap on total debris bodies alive at once. Past this, the oldest
/// already-asleep (settled) debris is evicted to make room for new debris,
/// never anything still actively flying/settling -- see
/// `VoxApp::enforce_debris_budget`.
const MAX_DEBRIS_BODIES: usize = 200;

/// Bodies at or below this voxel count mesh synchronously in
/// `VoxApp::upload_debris_mesh` instead of going through the threaded
/// `body_mesh` queue -- see that function's doc comment for why. Raised
/// from an initial 64,000: a felled tree's trunk-plus-canopy is one
/// connected mass that can easily clear 100,000+ voxels (a single canopy
/// ellipsoid alone is tens of thousands), and that's a rare, one-off event
/// worth a several-millisecond synchronous hitch rather than an invisible
/// frame -- the stress example extrapolates ~5ms at this size, against a
/// 16.6ms frame budget at 60Hz.
const INLINE_MESH_VOXEL_BUDGET: usize = 200_000;

/// Whether `upload_debris_mesh` meshed a body right there on the spot or
/// handed it to the background queue -- see `replace_body`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MeshDispatch {
    Sync,
    Async,
}

impl VoxApp {
    fn new(
        window: Arc<Window>,
        cfg: WorldConfig,
        mario_units_per_meter: f32,
        quality: Quality,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let assets = assets_dir();
        let registry = MaterialRegistry::load_dir(&assets.join("materials"))?;
        let shader = std::fs::read_to_string(assets.join("shaders/voxel.wgsl"))?;
        let shadow_shader = std::fs::read_to_string(assets.join("shaders/shadow.wgsl"))?;
        let particle_shader = std::fs::read_to_string(assets.join("shaders/particle.wgsl"))?;
        let post_shader = std::fs::read_to_string(assets.join("shaders/postprocess.wgsl"))?;
        let grass_shader = std::fs::read_to_string(assets.join("shaders/grass.wgsl"))?;
        let build_start = Instant::now();
        let world = build_streaming_world(cfg, &registry)?;
        // Terrain generator + materials for lazy chunk generation.
        let terrain = TerrainGen::new(&world.cfg);
        let terrain_mats = TerrainMaterials::from_registry(&registry)?;
        let tree_mats = TreeMaterials::from_registry(&registry)?;
        let chunk_loader = ChunkLoader::new(&world.cfg, quality, terrain, terrain_mats, tree_mats);
        tracing::info!(
            chunks = world.chunk_count(),
            elapsed_ms = build_start.elapsed().as_millis() as u64,
            "streaming world initialized"
        );

        let size = window.inner_size();
        let gpu = Gpu::new(window.clone(), size.width, size.height)?;
        // Shadow pipeline must be created before the voxel pipeline so its
        // shadow-sample bind group layout can be wired in as group 1.
        let shadow_pipeline = ShadowPipeline::new(&gpu, &shadow_shader, world.cfg.voxel_size_m);
        let pipeline = VoxelPipeline::new(
            &gpu,
            &shader,
            &registry,
            world.cfg.voxel_size_m,
            Some(shadow_pipeline.sample_bind_group_layout()),
        );
        let particle_pipeline = ParticlePipeline::new(&gpu, &particle_shader);
        let tools = Tools::new(&registry);
        let (surf_w, surf_h) = gpu.surface_size();
        let ssao_shader = std::fs::read_to_string(assets.join("shaders/ssao.wgsl"))?;
        let bloom_shader = std::fs::read_to_string(assets.join("shaders/bloom.wgsl"))?;
        let bloom_ssao = vox_render::BloomSsaoPipeline::new(
            &gpu, &ssao_shader, &bloom_shader, surf_w, surf_h,
        );
        let postprocess = vox_render::PostProcessPipeline::new(
            &gpu, &post_shader, surf_w, surf_h,
            bloom_ssao.ao_view(),
            bloom_ssao.bloom_view(),
        );
        let grass_pipeline = vox_render::GrassPipeline::new(&gpu, &grass_shader);
        let debug_overlay = DebugOverlay::new(gpu.device(), gpu.surface_format(), None, &window);
        // Air (id 0) is never player-selectable; the picker mirrors that.
        let material_names: Vec<String> = registry
            .iter()
            .skip(1)
            .map(|(_, def)| def.name.clone())
            .collect();
        // Tools' material_index is 1-based (air-inclusive); the picker's
        // index is 0-based into material_names — mirrors set_material_index.
        let selected_material = tools.material_index() - 1;
        let weathering = weather_table(&registry).map(vox_sim::Weathering::new);
        if weathering.is_none() {
            tracing::info!(
                "weathering disabled -- a required material is missing from the asset set"
            );
        }
        let fire = fire_table(&registry).map(vox_sim::FireSim::new);
        if fire.is_none() {
            tracing::info!("fire disabled -- a required material is missing from the asset set");
        }
        #[cfg(not(feature = "mario"))]
        let _ = mario_units_per_meter;
        let fluids = fluid_materials(&registry);
        let mut app = Self {
            window,
            gpu,
            pipeline,
            shadow_pipeline,
            world,
            chunk_loader,
            fluid: vox_sim::FluidSim::with_fluids_and_powders(
                fluids.clone(),
                powder_materials(&registry),
            ),
            phys: {
                let mut phys = PhysicsWorld::new();
                let fluid_buoyancy = fluid_densities(&registry);
                if !fluid_buoyancy.is_empty() {
                    phys.set_fluid_voxels(fluid_buoyancy);
                }
                phys
            },
            fluid_clock: vox_platform::FrameClock::new(vox_core::consts::FLUID_DT),
            weathering,
            fire,
            burning_bodies: FxHashMap::default(),
            registry,
            player: Player::new(Vec3::ZERO),
            camera: Camera::new(Vec3::ZERO),
            tools,
            remesh: RemeshQueue::new(),
            body_mesh: BodyMeshQueue::new(),
            fluids,
            blast_seed: 0,
            impact_seed: 0,
            grabbed: false,
            frames: 0,
            last_fps: 0,
            last_report: Instant::now(),
            tunables: Tunables::default(),
            profile: FrameProfile::new(),
            debug_overlay,
            debug_visible: false,
            material_names,
            selected_material,
            last_draw_stats: vox_render::DrawStats::default(),
            debris_order: VecDeque::new(),
            mario_mode: None,
            #[cfg(feature = "mario")]
            mario_units_per_meter,
            game_time: 60.0, // Start at noon (halfway through 120s cycle)
            always_day: false,
            pending_body_removal: HashMap::new(),
            editor_active: false,
            editor_radius: 2.0,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            frame_data: systems::FrameData::default(),
            last_view_proj: glam::Mat4::IDENTITY,
            last_frustum: vox_render::Frustum::from_view_proj(glam::Mat4::IDENTITY),
            last_shadow_frustum: vox_render::Frustum::from_view_proj(glam::Mat4::IDENTITY),
            crack_intensity: 0.0,
            replay: replay::ReplayState::default(),
            ecs: vox_ecs::World::new(),
            particles: ParticleSystem::new(),
            particle_pipeline,
            postprocess,
            bloom_ssao,
            grass_pipeline,
            grass_cache: grass::GrassCache::new(),
        };
        // Pre-generate spawn chunks before meshing and surface height
        // scan — both need solid voxels to exist. Disjoint field borrows.
        let center = Vec3::from(app.world.cfg.extent_m) * 0.5;
        app.chunk_loader.pregenerate_spawn(center, &mut app.world);

        app.initial_mesh();

        let surface = TerrainGen::surface_height_m(&app.world, center.x, center.z)
            .unwrap_or(app.world.cfg.extent_m[1] * 0.5);
        app.player = Player::new(Vec3::new(center.x, surface + 0.2, center.z));
        Ok(app)
    }

    /// Synchronous parallel meshing of the freshly generated world.
    fn initial_mesh(&mut self) {
        let keys = self.world.drain_dirty();
        let _ = self.world.drain_dirty_regions();
        let start = Instant::now();
        let world = &self.world;
        let fluids = &self.fluids;
        let meshes: Vec<(IVec3, vox_mesh::MeshData)> = keys
            .par_iter()
            .filter(|key| world.chunk_at(**key).is_some())
            .map(|key| {
                let origin = chunk_origin(*key);
                let slab = VoxelSlab::extract(world, origin, IVec3::splat(CHUNK_SIZE as i32));
                (*key, mesh_slab(&slab, origin, fluids))
            })
            .collect();
        let meshed = meshes.len();
        let mut quads = 0usize;
        for (key, mesh) in meshes {
            quads += mesh.quads();
            self.pipeline.upload_chunk(&self.gpu, key, &mesh);
        }
        tracing::info!(
            chunks = meshed,
            quads,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "initial world mesh"
        );
    }

    /// Spawn a debris body at `origin_m` (a solid `extent`^3 wood cube),
    /// meshing and uploading it immediately, with `vel_m_s` initial velocity.
    fn spawn_debris(&mut self, origin_m: Vec3, extent: i32, vel_m_s: Vec3) {
        let wood = self
            .registry
            .id_by_name("wood")
            .map(|m| Voxel(m.0))
            .unwrap_or(Voxel(1));
        let dims = IVec3::splat(extent);
        let voxels = vec![wood; (dims.x * dims.y * dims.z) as usize];
        let grid = VoxelGrid::new(dims, voxels);
        let Some(mut body) =
            Body::from_grid(grid, &self.registry, self.world.cfg.voxel_size_m, origin_m)
        else {
            return; // Massless grid (shouldn't happen for a solid cube).
        };
        body.vel = vel_m_s;
        let id = self.phys.spawn(body);
        self.upload_debris_mesh(id);
        tracing::info!(?id, ?origin_m, "spawned debris body");
    }

    /// Spawn a rope: 5 segments of 2×2×5 rope voxels (20 voxels each),
    /// connected by joints, hanging from a point above the player.
    fn spawn_rope(&mut self) {
        let rope_voxel = self.registry.id_by_name("rope").map(|m| Voxel(m.0));
        let Some(rope_voxel) = rope_voxel else {
            tracing::warn!("rope material not found, cannot spawn rope");
            return;
        };

        let voxel_size = self.world.cfg.voxel_size_m;
        let seg_dims = IVec3::new(2, 5, 2);
        let seg_voxels = vec![rope_voxel; (2 * 5 * 2) as usize]; // 20 voxels
        let seg_height_m = 5.0 * voxel_size; // 0.5m at 0.1m scale
        let half_height = seg_height_m * 0.5; // 0.25m

        // Spawn point: 3m above eye level, 2m forward along look direction.
        let base_pos = self.player.eye(1.0) + Vec3::new(0.0, 3.0, 0.0) + self.player.look_dir() * 2.0;

        let mut prev_id: Option<BodyId> = None;

        for i in 0..5 {
            // Stack segments vertically, top segment first.
            let seg_center = base_pos + Vec3::new(0.0, -i as f32 * seg_height_m, 0.0);
            let grid = VoxelGrid::new(seg_dims, seg_voxels.clone());
            let Some(body) = Body::from_grid(grid, &self.registry, voxel_size, seg_center)
            else {
                continue;
            };
            let id = self.phys.spawn(body);
            self.upload_debris_mesh(id);
            // Pin the top segment (i==0) to the world so the rope hangs
            // from a fixed point instead of free-falling. Without this, the
            // entire rope accelerates under gravity and all KE must be
            // absorbed by 4 joints in 2 iterations — the root cause of the
            // rope freakout on collision.
            if i == 0 {
                self.phys.pin(id);
            }

            if let Some(prev) = prev_id {
                // Connect bottom of previous segment to top of this segment.
                // rest_length = seg_height_m (end-to-end, not overlapping).
                // compliance = 0.0 (rigid — rope should be firm, not soft).
                let anchor_prev = Vec3::new(0.0, -half_height, 0.0);
                let anchor_this = Vec3::new(0.0, half_height, 0.0);
                self.phys.add_joint(prev, id, anchor_prev, anchor_this, seg_height_m, 0.0);
            }

            prev_id = Some(id);
        }

        tracing::info!(?prev_id, "spawned 5-segment rope");
    }

    /// Mesh and upload an already-spawned body's GPU representation. Every
    /// path that calls `phys.spawn` must follow up with this — a body with
    /// no uploaded mesh is simulated (falls, collides, sleeps) but never
    /// drawn, which looks indistinguishable from the material having simply
    /// vanished. Does nothing if `id` is no longer alive. Returns whether
    /// meshing was dispatched to the background queue (`Async`) or done
    /// right here (`Sync`) — callers replacing an existing body use this to
    /// decide whether the old mesh needs to be kept alive a little longer
    /// (see [`Self::replace_body`]).
    ///
    /// Small bodies mesh right here, synchronously: the stress example
    /// measures even a 40^3 cube (64,000 voxels) at ~1.7ms average, cheap
    /// enough not to be worth deferring. This matters because splitting a
    /// body during destruction is the overwhelmingly common case, and it
    /// always produces several small fragments in the same frame the old
    /// body is despawned. Only bodies above the budget (a large one-off
    /// mass -- a whole tree's trunk-plus-canopy is one connected component
    /// easily past 100,000 voxels) still go through `body_mesh` to keep
    /// that meshing cost off the main thread.
    fn upload_debris_mesh(&mut self, id: BodyId) -> MeshDispatch {
        let Some(body) = self.phys.get(id) else {
            return MeshDispatch::Sync;
        };
        let body_outlets = self
            .registry
            .id_by_name("ember")
            .map(|m| body_smoke_outlets(body, Voxel(m.0), self.world.cfg.voxel_size_m));
        let voxel_count = (body.grid.dims.x * body.grid.dims.y * body.grid.dims.z) as usize;
        let dispatch = if voxel_count <= INLINE_MESH_VOXEL_BUDGET {
            let slab = VoxelSlab::from_grid_with_damage(
                body.grid.dims,
                &body.grid.voxels,
                &body.grid.damage,
            );
            let mesh = mesh_slab(&slab, IVec3::ZERO, &self.fluids);
            tracing::info!(slot = id.slot, verts = mesh.vertices.len(), "upload_debris_mesh sync");
            self.pipeline
                .upload_body(&self.gpu, (id.slot, id.generation), &mesh);
            MeshDispatch::Sync
        } else {
            self.body_mesh.dispatch(
                (id.slot, id.generation),
                body.grid.dims,
                body.grid.voxels.clone(),
                body.grid.damage.clone(),
                &self.fluids,
            );
            MeshDispatch::Async
        };
        if let Some(outlets) = body_outlets.filter(|outlets| !outlets.is_empty()) {
            self.burning_bodies.insert(
                id,
                BodyFireVisual {
                    outlets,
                    cursor: 0,
                    cooldown: 0,
                },
            );
        } else {
            self.burning_bodies.remove(&id);
        }
        if !self.debris_order.contains(&id) {
            self.debris_order.push_back(id);
        }
        dispatch
    }

    /// Despawn-and-replace bookkeeping shared by every carve path (tool
    /// hits and impact fracture): a carved body is always despawned and 0+
    /// fragments spawned in its place, never updated in place. If every
    /// fragment meshes synchronously (the common case for anything but a
    /// genuinely huge mass), the old mesh is dropped immediately -- its
    /// replacements are already on screen this same frame, so there's no
    /// gap to cover. If any fragment is large enough to need threaded
    /// meshing, the old mesh is instead kept exactly where it is (frozen,
    /// at its last known transform) until *every* one of that group's async
    /// meshes has arrived, instead of vanishing for the frame or more that
    /// takes: a felled tree's trunk-plus-canopy is one connected mass well
    /// past `INLINE_MESH_VOXEL_BUDGET`, and it stays that large across many
    /// subsequent hits, not just the first -- which is exactly what made
    /// this "invisible for a solid frame every time damage is applied"
    /// rather than a one-off pop on the initial break.
    fn replace_body(&mut self, old_id: BodyId, spawned: Vec<BodyId>) {
        self.burning_bodies.remove(&old_id);
        let old_key = (old_id.slot, old_id.generation);
        if spawned.is_empty() {
            self.pipeline.remove_body(old_key);
            return;
        }
        let mut pending = HashSet::new();
        for id in spawned {
            if self.upload_debris_mesh(id) == MeshDispatch::Async {
                pending.insert((id.slot, id.generation));
            }
        }
        if pending.is_empty() {
            self.pipeline.remove_body(old_key);
        } else {
            self.pending_body_removal.insert(old_key, pending);
        }
    }

    /// Drop any old ghost mesh whose replacement fragments have all finished
    /// meshing this frame -- see [`Self::replace_body`]. Must run after
    /// `body_mesh.collect` so `uploaded` reflects this frame's arrivals.
    fn resolve_pending_removals(&mut self, uploaded: &[BodyMeshKey]) {
        if uploaded.is_empty() || self.pending_body_removal.is_empty() {
            return;
        }
        let mut finished = Vec::new();
        for (&old_key, waiting) in &mut self.pending_body_removal {
            for key in uploaded {
                waiting.remove(key);
            }
            if waiting.is_empty() {
                finished.push(old_key);
            }
        }
        for old_key in finished {
            self.pending_body_removal.remove(&old_key);
            self.pipeline.remove_body(old_key);
        }
    }

    /// Keep total debris body count under `MAX_DEBRIS_BODIES`: evict the
    /// oldest already-settled debris (see `evict_oldest_asleep_debris`) and
    /// drop each evicted body's GPU mesh too.
    fn enforce_debris_budget(&mut self) {
        for id in
            evict_oldest_asleep_debris(&mut self.phys, &mut self.debris_order, MAX_DEBRIS_BODIES)
        {
            tracing::info!(slot = id.slot, "budget evicted");
            self.burning_bodies.remove(&id);
            self.pipeline.remove_body((id.slot, id.generation));
        }
    }

    /// Despawn any clutter-sized debris (see
    /// `vox_core::consts::CLUTTER_MAX_VOXELS`) whose 35-60s lifetime just
    /// ran out, and drop its GPU mesh. This is what actually keeps a busy
    /// destruction site cheap: a felled tree at small voxel scales sheds
    /// far more gravel-sized chips than `MAX_DEBRIS_BODIES` eviction alone
    /// would ever clear in a reasonable time, since eviction only fires
    /// once the global cap is hit.
    fn expire_clutter(&mut self, dt: f32) {
        for id in self.phys.tick_lifetimes(dt) {
            tracing::info!(slot = id.slot, "clutter expired");
            self.burning_bodies.remove(&id);
            self.pipeline.remove_body((id.slot, id.generation));
        }
    }

    /// Streaming system: generate chunks around the player, evict beyond
    /// render distance. Self-contained — no cross-system data needed.
    fn system_streaming(&mut self) {
        let player_pos = self.player.ctrl.pos;
        let streamed = self.chunk_loader.update(
            player_pos,
            self.player.ctrl.vel,
            &mut self.world,
            &mut self.pipeline,
        );
        if streamed {
            self.grass_cache.invalidate();
        }
    }

    /// Day/night system: advance game time (unless frozen at noon).
    /// The `dn` computation stays inline in frame() since the render
    /// section uses it extensively.
    fn system_day_night(&mut self, dt: f32) {
        if !self.always_day {
            self.game_time += dt;
        } else {
            self.game_time = 60.0; // Noon (halfway through 120s cycle)
        }
    }

    /// Remesh system: wake physics/fluid/fire around dirty regions,
    /// dispatch chunk meshing to workers, upload finished meshes.
    /// Returns the count of uploaded meshes for pending-removal resolution.
    fn system_remesh(&mut self, eye: Vec3) -> Vec<(u32, u32)> {
        let _t = ScopedTimer::new(&mut self.profile.remesh);
        let s = self.world.cfg.voxel_size_m;
        let dirty = self.world.drain_dirty_regions();
        if !dirty.is_empty() {
            self.grass_cache.invalidate();
        }
        for (min, max) in &dirty {
            self.phys.wake_region(min.as_vec3() * s, max.as_vec3() * s);
            self.fluid.wake_region(&self.world, *min, *max);
        }
        for (min, max) in dirty {
            if let Some(fire) = &mut self.fire {
                fire.wake_region(&mut self.world, min, max);
            }
        }
        self.remesh.absorb_dirty(&mut self.world);
        self.remesh
            .dispatch(&self.world, eye, &self.fluids);
        self.remesh.collect(&self.gpu, &mut self.pipeline);
        self.body_mesh.collect(&self.gpu, &mut self.pipeline)
    }

    /// Input system: handle all key/mouse input. Returns true if the
 /// app should exit (Escape when not grabbed).
    fn system_input(&mut self, input: &mut InputState) -> bool {
        self.frame_data.grabbed_this_frame = false;
        if input.key_pressed(KeyCode::Escape) {
            if self.grabbed {
                self.set_grab(false);
            } else {
                return true;
            }
        }
        if input.mouse_clicked(MouseButton::Left) && !self.grabbed {
            self.set_grab(true);
            self.frame_data.grabbed_this_frame = true;
        }
        if input.key_pressed(KeyCode::KeyF) {
            self.player.toggle_fly();
        }
        if input.key_pressed(KeyCode::KeyB) {
            let origin = self.player.eye(1.0) + self.player.look_dir() * 4.0;
            self.spawn_debris(origin, 4, self.player.look_dir() * 8.0);
        }
        if input.key_pressed(KeyCode::KeyG) {
            let pos = self.player.eye(1.0) + self.player.look_dir() * 2.0;
            let vel = self.player.look_dir() * 20.0;
            let id = ecs_components::spawn_projectile(&mut self.ecs, pos, vel, 5.0);
            // Spawn a small debris body so the projectile is visible —
            // the ECS entity tracks it logically, the debris body is the visual.
            self.spawn_debris(pos, 2, vel);
            // Particle burst at spawn for feedback.
            self.particles.burst(Burst {
                center: pos,
                count: 8,
                color: [1.0, 0.8, 0.3],
                speed: 2.0,
                upward: 0.3,
                life: 0.5,
                size: 0.08,
                buoyant: false,
            });
            tracing::info!(?id, "spawned ECS projectile + debris body");
        }
        if input.key_pressed(KeyCode::KeyT) {
            let t0 = std::time::Instant::now();
            self.spawn_rope();
            tracing::info!(elapsed_ms = t0.elapsed().as_millis(), "spawn_rope took");
        }
        if input.key_pressed(KeyCode::KeyX) {
            let removed = self.phys.clear_sleeping();
            for id in &removed {
                self.burning_bodies.remove(id);
                self.pipeline.remove_body((id.slot, id.generation));
            }
            if !removed.is_empty() {
                tracing::info!(count = removed.len(), "cleared sleeping debris");
            }
        }
        const HOTBAR_KEYS: [KeyCode; 9] = [
            KeyCode::Digit1, KeyCode::Digit2, KeyCode::Digit3,
            KeyCode::Digit4, KeyCode::Digit5, KeyCode::Digit6,
            KeyCode::Digit7, KeyCode::Digit8, KeyCode::Digit9,
        ];
        for (i, key) in HOTBAR_KEYS.into_iter().enumerate() {
            if input.key_pressed(key)
                && let Some(tool) = self.tools.select_hotbar_slot(i as u8 + 1)
            {
                tracing::info!(tool = ?tool, "tool selected");
            }
        }
        if input.key_pressed(KeyCode::BracketLeft) {
            self.tools.shrink_radius();
            tracing::info!(radius_m = self.tools.active_radius_m(), "tool radius");
        }
        if input.key_pressed(KeyCode::BracketRight) {
            self.tools.grow_radius();
            tracing::info!(radius_m = self.tools.active_radius_m(), "tool radius");
        }
        if input.key_pressed(KeyCode::F3) {
            self.debug_visible = !self.debug_visible;
        }
        if input.key_pressed(KeyCode::KeyM) {
            self.toggle_mario_mode();
        }
        if input.key_pressed(KeyCode::KeyE) {
            self.editor_active = !self.editor_active;
            tracing::info!("editor mode toggled active={}", self.editor_active);
        }
        if input.key_pressed(KeyCode::KeyR) {
            self.replay.toggle_recording();
        }
        if input.key_pressed(KeyCode::F5) {
            match ecs_components::save_scene_to_file(&self.ecs, "scene.json") {
                Ok(()) => tracing::info!("scene saved to scene.json"),
                Err(e) => tracing::warn!("scene save failed: {e}"),
            }
        }
        if input.key_pressed(KeyCode::F6) {
            match ecs_components::load_scene_from_file(&mut self.ecs, "scene.json") {
                Ok(n) => tracing::info!(entities = n, "scene loaded from scene.json"),
                Err(e) => tracing::warn!("scene load failed: {e}"),
            }
        }
        if input.key_pressed(KeyCode::KeyP) {
            self.replay.start_playback();
        }
        if input.key_pressed(KeyCode::KeyQ) {
            let next = match self.chunk_loader.quality() {
                Quality::Low => Quality::Medium,
                Quality::Medium => Quality::High,
                Quality::High => Quality::Ultra,
                Quality::Ultra => Quality::Low,
            };
            self.chunk_loader.set_quality(next);
            tracing::info!(?next, "quality switched");
        }
        if input.key_down(KeyCode::ControlLeft) && input.key_pressed(KeyCode::KeyZ) {
            self.undo();
        }
        if input.key_down(KeyCode::ControlLeft) && input.key_pressed(KeyCode::KeyY) {
            self.redo();
        }
        // Sync tunables into consuming systems.
        self.player.fly_speed = self.tunables.fly_speed;
        self.phys.tunables = self.tunables;
        false
    }

    /// Player system: player/mario/replay update + tool application.
    /// Stores mario_active and mario_pos in frame_data for later systems.
    fn system_player(&mut self, input: &mut InputState, timing: FrameTiming) {
        let mario_active = self.mario_mode.as_ref().is_some_and(|m| m.is_active());
        self.frame_data.mario_active = mario_active;
        let mut mario_pos = Vec3::ZERO;

        if self.frame_data.mario_active {
            if self.grabbed {
                self.mario_mode.as_mut().unwrap().look(input.mouse_delta);
            }
            mario_pos = self
                .mario_mode
                .as_mut()
                .map(|m| m.tick(&self.world, input, timing.dt_frame))
                .unwrap_or(Vec3::ZERO);
            if let Some(impact_m) = self.mario_mode.as_mut().unwrap().pending_ground_pound() {
                self.apply_ground_pound(impact_m);
            }
        } else if self.replay.is_playing() {
            // Replay playback: snapshot drives player + camera.
        } else {
            if self.grabbed {
                self.player.look(input.mouse_delta);
            }
            {
                let _t = ScopedTimer::new(&mut self.profile.player);
                self.player
                    .fixed_steps(&self.world, input, timing.physics_steps);
            }
        }
        if self.grabbed && !self.frame_data.grabbed_this_frame && !mario_active && !self.replay.is_playing() {
            let tools_start = Instant::now();
            self.apply_tools(input);
            self.profile
                .tools
                .push(tools_start.elapsed().as_secs_f32() * 1000.0);
        }
        self.frame_data.mario_pos = mario_pos;
    }

    /// Physics system: step physics, apply impact fracture, manage debris
    /// budget and clutter lifetime.
    fn system_physics(&mut self, timing: FrameTiming) {
        let impacts = {
            let _t = ScopedTimer::new(&mut self.profile.physics);
            let step_start = std::time::Instant::now();
            let mut impacts = Vec::new();
            for _ in 0..timing.physics_steps {
                impacts.extend(self.phys.step(&self.world, vox_core::consts::PHYSICS_DT));
            }
            let elapsed = step_start.elapsed().as_millis();
            if elapsed > 50 {
                tracing::warn!(elapsed_ms = elapsed, bodies = self.phys.body_count(), "slow physics step");
            }
            impacts
        };
        self.apply_impact_fracture(impacts);
        // Debug: check rope segment health for NaN/divergence.
        if !self.phys.joints().is_empty() {
            if let Some(j) = self.phys.joints().first() {
                if let Some(body) = self.phys.iter().find(|(id, _)| id.slot as usize == j.body_a) {
                    let (_, b) = body;
                    if !b.pos.is_finite() || !b.vel.is_finite() || b.vel.length() > 50.0 {
                        tracing::warn!(pos=?b.pos, vel=?b.vel, vel_len=b.vel.length(), "rope segment 0 DIVERGED");
                    }
                }
            }
        }
        let damage_dirty_ids: Vec<BodyId> = self
            .phys
            .iter()
            .filter(|(_, b)| b.damage_dirty)
            .map(|(id, _)| id)
            .collect();
        for id in damage_dirty_ids {
            self.upload_debris_mesh(id);
            if let Some(body) = self.phys.get_mut(id) {
                body.damage_dirty = false;
            }
        }
        self.enforce_debris_budget();
        self.expire_clutter(timing.physics_steps as f32 * vox_core::consts::PHYSICS_DT);
    }

    /// Fluid system: fluid tick, weathering, fire, fire events → particles,
    /// burning body management, body smoke. Runs at fluid tick rate.
    fn system_fluid(&mut self, timing: FrameTiming) {
        let fluid_timing = self.fluid_clock.advance(timing.dt_frame);
        for _ in 0..fluid_timing.physics_steps {
            {
                const DIRS6: [IVec3; 6] = [
                    IVec3::X, IVec3::NEG_X, IVec3::Y,
                    IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z,
                ];
                let neighbor_chunks: FxHashSet<IVec3> = self
                    .fluid
                    .active_chunk_keys()
                    .flat_map(|ck| DIRS6.into_iter().map(move |d| ck + d))
                    .collect();
                for ck in neighbor_chunks {
                    if self.world.chunk_at(ck).is_none() {
                        self.chunk_loader.ensure_loaded(&mut self.world, ck);
                    }
                }
            }
            self.fluid.tick(&mut self.world);
            if let Some(w) = &mut self.weathering {
                let events = self.fluid.drain_events();
                w.tick(&mut self.world, &events);
            } else {
                self.fluid.drain_events();
            }
            let mut spawned_ids = Vec::new();
            let mut remesh_ids: FxHashSet<BodyId> = FxHashSet::default();
            if let Some(f) = &mut self.fire {
                f.tick(&mut self.world);
                let s = self.world.cfg.voxel_size_m;
                let mut consumed_positions = Vec::new();
                for event in f.drain_events() {
                    match event {
                        vox_sim::FireEvent::Smoke { pos, face, kind } => {
                            let (count, color, speed, upward, life, size, salt) = match kind {
                                vox_sim::SmokeKind::Burning => {
                                    (1, [0.35, 0.34, 0.33], 0.04, 0.08, 4.5, s * 0.35, 0)
                                }
                                vox_sim::SmokeKind::Extinguished => {
                                    (3, [0.7, 0.7, 0.75], 0.12, 0.15, 2.5, s * 0.3, 1)
                                }
                                vox_sim::SmokeKind::Consumed => {
                                    (2, [0.25, 0.22, 0.20], 0.08, 0.1, 3.5, s * 0.35, 2)
                                }
                            };
                            let center = fire_smoke_origin(
                                pos, face, s,
                                (self.particles.len() as u32).wrapping_add(salt),
                            );
                            self.particles.burst(Burst {
                                center, count, color, speed, upward, life, size,
                                buoyant: true,
                            });
                        }
                        vox_sim::FireEvent::Consumed(pos) => consumed_positions.push(pos),
                        vox_sim::FireEvent::Extinguished(_) => {}
                    }
                }
                if !consumed_positions.is_empty() {
                    spawned_ids = vox_physics::detach_unsupported(
                        &mut self.world, &mut self.phys, &self.registry,
                        &consumed_positions,
                    );
                }
                let ember = self.registry.id_by_name("ember").map(|m| Voxel(m.0));
                let mut ignite_world: FxHashSet<IVec3> = FxHashSet::default();
                let mut ignite_body: FxHashSet<(BodyId, IVec3)> = FxHashSet::default();
                if let Some(ember) = ember {
                    let phys = &self.phys;
                    self.burning_bodies.retain(|id, _| phys.get(*id).is_some());
                    let world_fire_bounds_m = f.burning_bounds().map(|(min, max)| {
                        ((min - IVec3::ONE).as_vec3() * s, (max + IVec3::ONE).as_vec3() * s)
                    });
                    for (body_id, body) in self.phys.iter() {
                        if body.sleep.asleep { continue; }
                        let carries_fire = self.burning_bodies.contains_key(&body_id);
                        let near_world_fire = world_fire_bounds_m.is_some_and(|(min, max)| {
                            body.aabb_max.cmpge(min).all() && body.aabb_min.cmple(max).all()
                        });
                        if !carries_fire && !near_world_fire { continue; }
                        for sample in &body.surface {
                            let local_voxel = ((*sample - body.grid_offset) / s).floor().as_ivec3();
                            let body_voxel = body.grid.get(local_voxel);
                            let world_voxel = voxel_at(body.pos + body.rot * *sample, s);
                            if body_voxel == ember && carries_fire {
                                for face in [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z] {
                                    let neighbor = world_voxel + face;
                                    let neighbor_voxel = self.world.get_voxel(neighbor);
                                    if neighbor_voxel != ember
                                        && self.registry.get(MaterialId(neighbor_voxel.0))
                                            .is_some_and(|def| def.flammable)
                                    {
                                        ignite_world.insert(neighbor);
                                    }
                                }
                            } else if near_world_fire
                                && body_voxel != AIR && body_voxel != ember
                                && self.registry.get(MaterialId(body_voxel.0))
                                    .is_some_and(|def| def.flammable)
                                && [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z]
                                    .iter().any(|&face| f.is_burning(world_voxel + face))
                            {
                                ignite_body.insert((body_id, local_voxel));
                            }
                        }
                    }
                    for (body_id, local_voxel) in ignite_body {
                        if let Some(body) = self.phys.get_mut(body_id) {
                            body.grid.set(local_voxel, ember);
                            remesh_ids.insert(body_id);
                        }
                    }
                }
                for pos in ignite_world {
                    f.ignite(&mut self.world, pos);
                }
            }
            for id in spawned_ids {
                self.upload_debris_mesh(id);
            }
            for id in remesh_ids {
                self.upload_debris_mesh(id);
            }
            let body_smoke_ids: Vec<BodyId> = self.burning_bodies.keys().copied().collect();
            for body_id in body_smoke_ids {
                let Some(body) = self.phys.get(body_id) else {
                    self.burning_bodies.remove(&body_id);
                    continue;
                };
                let Some(visual) = self.burning_bodies.get_mut(&body_id) else { continue; };
                if visual.cooldown > 0 {
                    visual.cooldown -= 1;
                    continue;
                }
                if let Some(origin) =
                    next_body_smoke_origin(body, visual, &self.world, self.world.cfg.voxel_size_m)
                {
                    visual.cooldown = 10;
                    self.particles.burst(Burst {
                        center: origin, count: 1, color: [0.35, 0.34, 0.33],
                        speed: 0.03, upward: 0.07, life: 4.5,
                        size: self.world.cfg.voxel_size_m * 0.35, buoyant: true,
                    });
                } else {
                    visual.cooldown = 2;
                }
            }
        }
    }

    /// Render prep: particle update, day/night, camera setup, frustum,
    /// shadow camera. Everything before gpu.begin_frame() — the actual
    /// GPU render passes stay inline in frame() due to frame lifetime.
    fn system_render_prep(&mut self, timing: FrameTiming) {
        let body_colliders: Vec<particles::BodyCollisionRef> = self
            .phys
            .iter()
            .map(|(_, b)| particles::BodyCollisionRef {
                aabb_min: b.aabb_min,
                aabb_max: b.aabb_max,
                pos: b.pos,
                inv_rot: b.rot.inverse(),
                grid_offset: b.grid_offset,
                dims: b.grid.dims,
                voxels: &b.grid.voxels,
            })
            .collect();
        self.particles.update(
            timing.dt_frame,
            &self.world,
            self.world.cfg.voxel_size_m,
            &body_colliders,
        );
        self.system_day_night(timing.dt_frame);
        let dn = day_night::compute(self.game_time);
        let eye = self.player.eye(timing.alpha);
        if self.frame_data.mario_active {
            let mode = self.mario_mode.as_ref().unwrap();
            self.camera.pos = mode.camera_pos(self.frame_data.mario_pos);
            self.camera.yaw = mode.cam_yaw;
            self.camera.pitch = mode.cam_pitch;
        } else {
            self.camera.pos = eye;
            self.camera.yaw = self.player.yaw;
            self.camera.pitch = self.player.pitch;
        }
        let (w, h) = self.gpu.surface_size();
        let aspect = w as f32 / h.max(1) as f32;
        let view_proj = self.camera.view_proj(aspect);
        self.pipeline.write_camera(
            &self.gpu,
            view_proj,
            self.camera.pos,
            FOG_END_M,
            dn.sun_dir,
            dn.sun_strength,
            dn.sky_color,
            dn.fill_strength,
            dn.ambient_strength,
            dn.sun_color,
            dn.ambient_sky,
            dn.ambient_ground,
            self.crack_intensity,
            self.game_time,
        );
        let cam_right = self.camera.right();
        let cam_up = cam_right.cross(self.camera.forward()).normalize();
        self.particle_pipeline
            .write_camera(&self.gpu, view_proj, cam_right, cam_up);
        let particle_instances = self.particles.instances();
        self.particle_pipeline
            .upload(&self.gpu, &particle_instances);
        self.last_view_proj = view_proj;
        self.last_frustum = Frustum::from_view_proj(view_proj);
        let shadow_focus = if self.frame_data.mario_active {
            self.frame_data.mario_pos
        } else {
            self.camera.pos
        };
        self.last_shadow_frustum = self.shadow_pipeline.write_camera(
            &self.gpu,
            dn.sun_dir,
            shadow_focus,
            self.world.cfg.voxel_size_m,
        );
    }

    /// Material-based impact destruction: check each impact this frame's
    /// physics step(s) produced against the material actually at that
    /// point. A hit whose speed (impulse/mass -- the velocity change the
    /// contact imparted, independent of the body's own mass) exceeds that
    /// material's own fracture threshold chips debris apart right where it
    /// was struck, reusing the same carve+split mechanism the tools use.
    /// Tougher materials (higher `strength`, same convention as the
    /// destruction tools: "higher values survive bigger blasts") need a
    /// harder hit to trigger at all, *and* lose proportionally less volume
    /// when they do -- stone barely chips, leaves come apart at the
    /// slightest touch, wood sits in between but sheds a bigger bite once
    /// it does go (see [`fracture_radius_vox`]). See
    /// `Tunables::fracture_sensitivity` for the overall threshold dial.
    fn apply_impact_fracture(&mut self, impacts: Vec<ImpactEvent>) {
        for event in impacts {
            tracing::info!(slot = event.body.slot, impulse = event.impulse, "impact event");
            let Some(body) = self.phys.get(event.body) else {
                continue;
            };
            // Terminal rubble never re-fractures -- see
            // `MIN_FRACTURE_BODY_VOXELS` for the cascade this gate breaks.
            if body.grid.solid_count() < MIN_FRACTURE_BODY_VOXELS {
                continue;
            }
            // Jointed bodies (rope segments) skip impact fracture: the
            // interleaved contact+joint solver produces large phantom
            // impulses when a segment touches terrain while constrained
            // by joints. These are solver artifacts, not real impacts —
            // the rope's strength (50.0) should survive any real fall,
            // but the feedback impulses blow past it. Rope is still
            // cuttable by tools (dig/bomb/laser carve directly, bypassing
            // fracture entirely).
            if self.phys.joints().iter().any(|j| {
                j.body_a == event.body.slot as usize || j.body_b == event.body.slot as usize
            }) {
                continue;
            }
            let impact_speed = event.impulse / body.mass();
            let voxel_size_m = body.half_voxel * 2.0;
            let local = body.rot.inverse() * (event.point_m - body.pos) - body.grid_offset;
            let local_voxel = (local / voxel_size_m).floor().as_ivec3();
            let material = body.grid.get(local_voxel);
            if material == AIR {
                continue;
            }
            let Some(def) = self.registry.get(MaterialId(material.0)) else {
                continue;
            };
            let Some(radius_vox) = fracture_radius_vox(
                def.strength,
                impact_speed,
                self.tunables.fracture_sensitivity,
            ) else {
                // Below fracture threshold -- try the damage path. Impacts
                // between 30% and 100% of the threshold accumulate damage
                // instead of fracturing outright.
                let fracture_threshold = self.tunables.fracture_sensitivity * def.strength;
                if impact_speed >= fracture_threshold * DAMAGE_THRESHOLD_FACTOR {
                    let ratio = impact_speed / fracture_threshold;
                    let damage_amount = ratio * ratio * DAMAGE_RATE;

                    const DIRS6: [IVec3; 6] = [
                        IVec3::X,
                        IVec3::NEG_X,
                        IVec3::Y,
                        IVec3::NEG_Y,
                        IVec3::Z,
                        IVec3::NEG_Z,
                    ];
                    let mut damage_voxels = vec![(local_voxel, damage_amount)];
                    for &d in &DIRS6 {
                        damage_voxels.push((local_voxel + d, damage_amount * 0.5));
                    }

                    let result = vox_physics::apply_body_damage(
                        &mut self.phys,
                        &self.registry,
                        event.body,
                        &damage_voxels,
                        voxel_size_m,
                    );
                    if let Some(spawned) = result {
                        // Body was despawned (voxels crumbled). Dust + replace.
                        self.particles.burst(Burst {
                            center: event.point_m,
                            count: 4,
                            color: def.color,
                            speed: 1.0,
                            upward: 0.5,
                            life: 0.5,
                            size: 0.03,
                            buoyant: false,
                        });
                        self.replace_body(event.body, spawned);
                    }
                    // If result is None, the body was mutated in-place with
                    // damage_dirty set. Re-meshing will be handled by the
                    // damage_dirty check in the render loop.
                }
                continue;
            };
            let radius_m = voxel_size_m * radius_vox;

            // Carve centered a little *into* the body along the actual
            // contact push direction, not straddling half in / half out of
            // the surface the way a sphere planted exactly on the contact
            // point would -- "directly remove what impacts what" instead of
            // a generic point-in-space blob.
            let push = event.push_dir.normalize_or_zero();
            let center = event.point_m + push * (radius_m * 0.5);

            let seed = self.impact_seed;
            self.impact_seed = self.impact_seed.wrapping_add(1);
            let spawned = vox_physics::carve_body_sphere_at_impact(
                &mut self.phys,
                &self.registry,
                event.body,
                center,
                radius_m,
                push,
                impact_speed,
                seed,
            );
            if self.phys.get(event.body).is_none() {
                // Dust in the fractured material's color, scaled by both
                // the bite size and how hard the hit actually was.
                self.particles.burst(Burst {
                    center: event.point_m,
                    count: ((radius_vox * 5.0) as usize).clamp(4, 18),
                    color: def.color,
                    speed: (impact_speed * 0.4).clamp(0.8, 3.0),
                    upward: 0.6,
                    life: 0.8,
                    size: 0.045,
                    buoyant: false,
                });
                self.replace_body(event.body, spawned);
            }
        }
    }

    /// Rewrite every awake debris body's GPU transform from the interpolated
    /// physics state. Chunk mesh vertices are in grid-voxel corner units
    /// scaled by `voxel_size_m` in the shader; the same scaling applies to
    /// debris, so the model matrix carries only translation and rotation.
    ///
    /// Skips asleep bodies entirely: the solver's integration step already
    /// skips them (a sleeping body's `pos`/`rot` are frozen -- that's the
    /// whole point of sleeping), so their last-written GPU transform is
    /// still exactly correct. Without this, every debris body pays a
    /// `write_buffer` call *every frame forever*, even sitting perfectly
    /// still -- with up to `MAX_DEBRIS_BODIES` debris around and most of it
    /// asleep in steady state (normal play, well after a blast), that's
    /// dozens to hundreds of pointless GPU writes a frame, scaling with
    /// exactly "the longer you play, the worse it gets."
    fn sync_debris_render(&mut self, alpha: f32) {
        for (id, body) in self.phys.iter() {
            if body.sleep.asleep {
                continue;
            }
            let (pos, rot) = self
                .phys
                .interpolated_transform(id, alpha)
                .expect("id came from iter()");
            // grid_offset is already in meters (mass_props computes com_local
            // in meters); the shader's `local` is also meters after scaling
            // grid-corner units by voxel_size_m, so no unit conversion here.
            let model = Mat4::from_rotation_translation(rot, pos)
                * Mat4::from_translation(body.grid_offset);
            self.pipeline.update_body_transform(
                &self.gpu,
                (id.slot, id.generation),
                model,
                body.aabb_min,
                body.aabb_max,
            );
        }
    }

    fn set_grab(&mut self, grab: bool) {
        let mode = if grab {
            CursorGrabMode::Locked
        } else {
            CursorGrabMode::None
        };
        let result = self.window.set_cursor_grab(mode).or_else(|_| {
            self.window.set_cursor_grab(if grab {
                CursorGrabMode::Confined
            } else {
                CursorGrabMode::None
            })
        });
        match result {
            Ok(()) => {
                self.window.set_cursor_visible(!grab);
                self.grabbed = grab;
            }
            Err(err) => tracing::warn!(%err, "cursor grab change failed"),
        }
    }

    /// Editor brush input (only called when `editor_active`): LMB paints a
    /// sphere of the selected material at the crosshair target, RMB erases a
    /// sphere of AIR. The wheel adjusts `editor_radius` directly (bypassing
    /// the tool hotbar's radius, which routes through `active_radius_mut`
    /// and would resize water or cycle material depending on the active
    /// tool). Every changed voxel is recorded as a (pos, old_voxel) diff and
    /// pushed onto the undo stack — fully undoable with Ctrl+Z, since the
    /// brush only touches the static world (no debris is ever spawned).
    fn apply_editor_brush(&mut self, input: &InputState, eye: Vec3, look: Vec3) {
        let radius = self.editor_radius;
        if input.mouse_clicked(MouseButton::Left) {
            // selected_material is 0-based into material_names (air
            // excluded); +1 converts to the registry id the tools use.
            let material = Voxel((self.selected_material + 1) as u16);
            let diff = self
                .tools
                .editor_brush(&mut self.world, eye, look, radius, material);
            if !diff.is_empty() {
                self.undo_stack.push(diff);
                self.redo_stack.clear();
                if self.undo_stack.len() > 50 {
                    self.undo_stack.remove(0);
                }
            }
        }
        if input.mouse_clicked(MouseButton::Right) {
            let diff = self
                .tools
                .editor_brush(&mut self.world, eye, look, radius, AIR);
            if !diff.is_empty() {
                self.undo_stack.push(diff);
                self.redo_stack.clear();
                if self.undo_stack.len() > 50 {
                    self.undo_stack.remove(0);
                }
            }
        }
        if input.wheel_delta.abs() >= 1.0 {
            let steps = input.wheel_delta as i32;
            self.editor_radius = (self.editor_radius + steps as f32 * 0.25).clamp(0.5, 8.0);
        }
    }

    /// Tool input: LMB uses the active hotbar tool, RMB places.
    fn apply_tools(&mut self, input: &InputState) {
        let eye = self.player.eye(1.0);
        let look = self.player.look_dir();
        if self.editor_active {
            self.apply_editor_brush(input, eye, look);
            return;
        }
        if input.mouse_clicked(MouseButton::Left) {
            // Ensure chunks at the tool's impact point are loaded before
            // the carve — otherwise the tool hits air at the streaming
            // boundary and the effect is lost.
            let hit_point = eye + look * REACH;
            let radius = self.tools.active_radius_m();
            match self.tools.tool {
                Tool::Bomb | Tool::ScalableDig => {
                    self.chunk_loader.ensure_loaded_box(
                        hit_point - Vec3::splat(radius),
                        hit_point + Vec3::splat(radius),
                        &mut self.world,
                    );
                }
                Tool::DeathLaser => {
                    // Ensure chunks along the beam up to REACH.
                    self.chunk_loader.ensure_loaded_box(
                        eye - Vec3::splat(REACH),
                        eye + look * REACH + Vec3::splat(REACH),
                        &mut self.world,
                    );
                }
                _ => {}
            }
            let outcome = match self.tools.tool {
                Tool::Dig => {
                    self.tools
                        .dig(&mut self.world, &mut self.phys, &self.registry, eye, look)
                }
                Tool::ScalableDig => self.tools.scalable_dig(
                    &mut self.world,
                    &mut self.phys,
                    &self.registry,
                    eye,
                    look,
                ),
                Tool::Bomb => {
                    let seed = self.blast_seed;
                    self.blast_seed = self.blast_seed.wrapping_add(1);
                    self.tools.blast(
                        &mut self.world,
                        &mut self.phys,
                        &self.registry,
                        eye,
                        look,
                        self.tunables.blast_power,
                        seed,
                    )
                }
                Tool::DeathLaser => self.tools.death_laser(
                    &mut self.world,
                    &mut self.phys,
                    &self.registry,
                    eye,
                    look,
                ),
                Tool::PlaceWater => {
                    self.tools.place_water(
                        &mut self.world,
                        &mut self.fluid,
                        &self.registry,
                        eye,
                        look,
                    );
                    CarveOutcome::default()
                }
                Tool::Ember => {
                    if let Some(pos) = self.tools.place_ember(
                        &mut self.world,
                        &self.registry,
                        eye,
                        look,
                        self.player.ctrl.aabb(),
                    ) {
                        if let Some(f) = &mut self.fire {
                            f.ignite(&mut self.world, pos);
                        }
                    }
                    CarveOutcome::default()
                }
            };
            // A carved body is despawned and replaced, not updated in
            // place. `removed` is empty for a carve that only ever touched
            // fresh material (world terrain, or no existing body in range)
            // -- those fragments have no old ghost to protect, just upload
            // them. Otherwise it's exactly the one body the carve hit (every
            // `Tools` method that populates `removed` puts at most one id in
            // it -- see `body_outcome`); `replace_body` keeps its mesh alive
            // until every replacement fragment is ready (see its own doc
            // comment).
            self.emit_tool_particles(&outcome);
            // Push voxel diff onto undo stack — but only for operations that
            // didn't spawn debris bodies. Destruction tools (blast, scalable_dig,
            // laser) call detach_unsupported which deletes voxels NOT in the
            // diff and spawns physics bodies we can't despawn on undo. So we
            // only undo safe operations: dig (single voxel, no debris) and
            // place_voxel. This is an honest limitation — destruction undo
            // needs body despawning + full diff capture, deferred.
            if !outcome.voxel_diff.is_empty() && outcome.spawned.is_empty() {
                self.undo_stack.push(outcome.voxel_diff);
                self.redo_stack.clear();
                if self.undo_stack.len() > 50 {
                    self.undo_stack.remove(0);
                }
            }
            match outcome.removed.first() {
                Some(&old_id) => self.replace_body(old_id, outcome.spawned),
                None => {
                    for id in outcome.spawned {
                        self.upload_debris_mesh(id);
                    }
                }
            }
        }
        if input.mouse_clicked(MouseButton::Right) {
            if let Some(diff) =
                self.tools
                    .place_voxel(&mut self.world, eye, look, self.player.ctrl.aabb())
            {
                self.undo_stack.push(vec![diff]);
                self.redo_stack.clear();
                if self.undo_stack.len() > 50 {
                    self.undo_stack.remove(0);
                }
            }
        }
        if input.wheel_delta.abs() >= 1.0 {
            let steps = input.wheel_delta as i32;
            if self.tools.has_adjustable_radius() {
                self.tools.adjust_radius(steps);
            } else {
                self.tools.cycle_material(steps, &self.registry);
            }
        }
    }

    /// Undo the last world edit: restore old voxel values, clear redo stack
    /// entry for this undo (push current values for redo).
    fn undo(&mut self) {
        let Some(diff) = self.undo_stack.pop() else {
            return;
        };
        // Capture current values for redo (voxels may have been re-edited).
        let redo_diff: Vec<(IVec3, Voxel)> = diff
            .iter()
            .map(|&(pos, _)| (pos, self.world.get_voxel(pos)))
            .collect();
        // Restore old values (set_voxel marks dirty + dirty_regions).
        for &(pos, old_voxel) in &diff {
            self.world.set_voxel(pos, old_voxel);
        }
        self.redo_stack.push(redo_diff);
        // Wake physics for the edited regions.
        let s = self.world.cfg.voxel_size_m;
        for (min, max) in self.world.drain_dirty_regions() {
            self.phys.wake_region(min.as_vec3() * s, max.as_vec3() * s);
        }
        tracing::info!(voxels = diff.len(), "undo");
    }

    /// Redo the last undone edit: re-apply the saved current values.
    fn redo(&mut self) {
        let Some(diff) = self.redo_stack.pop() else {
            return;
        };
        // Capture current values for undo (so undo can undo the redo).
        let undo_diff: Vec<(IVec3, Voxel)> = diff
            .iter()
            .map(|&(pos, _)| (pos, self.world.get_voxel(pos)))
            .collect();
        // Re-apply the redo values.
        for &(pos, voxel) in &diff {
            self.world.set_voxel(pos, voxel);
        }
        self.undo_stack.push(undo_diff);
        // Wake physics for the edited regions (set_voxel marks dirty).
        let s = self.world.cfg.voxel_size_m;
        for (min, max) in self.world.drain_dirty_regions() {
            self.phys.wake_region(min.as_vec3() * s, max.as_vec3() * s);
        }
        tracing::info!(voxels = diff.len(), "redo");
    }

    /// Carve a crater where Mario's ground pound landed, detach any
    /// material left unsupported, spawn it as debris, and give the
    /// fragments an outward blast impulse — the same pipeline a bomb uses,
    /// but centered on Mario's impact point with a fixed crater radius.
    /// The wake + remesh pass later in the frame picks up the dirty region
    /// automatically (same as `Tools::blast`).
    fn apply_ground_pound(&mut self, impact_m: Vec3) {
        const CRATER_RADIUS_M: f32 = 1.5;
        const POUND_POWER: f32 = 12.0;
        let seed = self.blast_seed;
        self.blast_seed = self.blast_seed.wrapping_add(1);
        let result = vox_physics::blast(
            &mut self.world,
            &mut self.phys,
            &self.registry,
            impact_m,
            CRATER_RADIUS_M,
            POUND_POWER,
            seed,
        );
        // Upload meshes for every debris fragment the blast spawned, so
        // they're visible immediately (mirrors `apply_tools`'s post-blast
        // handling for a world-only hit, where `removed` is empty).
        for id in result.spawned {
            self.upload_debris_mesh(id);
        }
        tracing::info!(
            ?impact_m,
            radius = CRATER_RADIUS_M,
            "ground pound crater carved"
        );
    }

    /// The palette color of `v`, for destruction-feedback particles; a
    /// neutral gray when the material is unknown or air.
    fn material_color(&self, v: Voxel) -> [f32; 3] {
        self.registry
            .get(MaterialId(v.0))
            .map(|d| d.color)
            .unwrap_or([0.6, 0.58, 0.55])
    }

    /// Destruction feedback for one tool use: dust in the destroyed
    /// material's color, plus tool-specific flavor (sparks and smoke for
    /// the bomb, hot sparks for the laser). Purely visual -- no gameplay
    /// state reads these.
    fn emit_tool_particles(&mut self, outcome: &CarveOutcome) {
        let Some(center) = outcome.impact_m else {
            return;
        };
        let dust = self.material_color(outcome.impact_material);
        match self.tools.tool {
            Tool::Dig => self.particles.burst(Burst {
                center,
                count: 10,
                color: dust,
                speed: 1.6,
                upward: 0.8,
                life: 0.7,
                size: 0.035,
                buoyant: false,
            }),
            Tool::ScalableDig => {
                let r = self.tools.radius_m;
                self.particles.burst(Burst {
                    center,
                    count: (r * 14.0) as usize + 8,
                    color: dust,
                    speed: 1.2 + r,
                    upward: 1.0,
                    life: 0.9,
                    size: 0.05,
                    buoyant: false,
                });
            }
            Tool::Bomb => {
                let r = self.tools.radius_m;
                // Hot sparks first, then a wave of material dust, then slow
                // rising smoke -- three layers with different speeds/lives
                // read as one explosion instead of one flat puff.
                self.particles.burst(Burst {
                    center,
                    count: 30,
                    color: [1.0, 0.65, 0.2],
                    speed: 7.0 + r * 2.0,
                    upward: 2.0,
                    life: 0.5,
                    size: 0.05,
                    buoyant: false,
                });
                self.particles.burst(Burst {
                    center,
                    count: (r * 16.0) as usize + 16,
                    color: dust,
                    speed: 4.0 + r,
                    upward: 1.5,
                    life: 1.1,
                    size: 0.06,
                    buoyant: false,
                });
                self.particles.burst(Burst {
                    center,
                    count: 14,
                    color: [0.35, 0.34, 0.33],
                    speed: 1.2,
                    upward: 0.8,
                    life: 2.4,
                    size: 0.30,
                    buoyant: true,
                });
            }
            Tool::DeathLaser => {
                self.particles.burst(Burst {
                    center,
                    count: 24,
                    color: [1.0, 0.35, 0.25],
                    speed: 6.0,
                    upward: 0.5,
                    life: 0.4,
                    size: 0.04,
                    buoyant: false,
                });
                self.particles.burst(Burst {
                    center,
                    count: 10,
                    color: dust,
                    speed: 2.5,
                    upward: 1.0,
                    life: 0.8,
                    size: 0.05,
                    buoyant: false,
                });
            }
            // Placing water never carves anything -- `outcome.impact_m` is
            // always `None` for it, so the early return above already
            // exits before this match runs; kept here only to stay
            // exhaustive.
            Tool::PlaceWater => {}
            Tool::Ember => {}
        }
    }

    fn report_stats(&mut self, stats: vox_render::DrawStats) {
        self.frames += 1;
        if self.last_report.elapsed().as_secs_f32() >= 1.0 {
            tracing::info!(
                fps = self.frames,
                drawn = stats.drawn,
                culled = stats.culled,
                queue = self.remesh.pending_len(),
                in_flight = self.remesh.in_flight,
                bodies = self.phys.body_count(),
                bodies_awake = self.phys.awake_count(),
                pos = ?voxel_at(self.player.ctrl.pos, self.world.cfg.voxel_size_m),
                grounded = self.player.ctrl.grounded,
            );
            self.last_fps = self.frames;
            self.frames = 0;
            self.last_report = Instant::now();
        }
    }

    /// Toggle Mario mode on/off. On first activation, lazily loads the
    /// SM64 ROM and builds the Mario render pipeline. Subsequent
    /// toggles spawn/despawn Mario at the player's position.
    #[cfg(feature = "mario")]
    fn toggle_mario_mode(&mut self) {
        // Initialize Mario mode if not yet done
        if self.mario_mode.is_none() {
            let assets = assets_dir();
            let rom_path = match mario::MarioMode::find_rom(&assets) {
                Some(p) => p,
                None => {
                    tracing::warn!(
                        "SM64 ROM not found — run enable_mario.bat with your SM64 US ROM, or place baserom.us.z64 in the roms/ directory"
                    );
                    return;
                }
            };
            let mario_shader = match std::fs::read_to_string(assets.join("shaders/mario.wgsl")) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("mario.wgsl not found: {e}");
                    return;
                }
            };
            match mario::MarioMode::init(
                &self.gpu,
                &rom_path,
                &mario_shader,
                self.mario_units_per_meter,
            ) {
                Ok(mode) => self.mario_mode = Some(mode),
                Err(e) => {
                    tracing::error!("Mario mode init failed: {e}");
                    return;
                }
            }
        }

        // Toggle spawn/despawn
        if let Some(mode) = self.mario_mode.as_mut() {
            if mode.is_active() {
                mode.despawn();
            } else {
                // Spawn Mario just above the player's current position.
                // The player is already grounded, so this is right above
                // the terrain surface.
                let pos = self.player.ctrl.pos;
                let spawn_pos = Vec3::new(pos.x, pos.y + 1.0, pos.z);
                mode.cam_yaw = self.player.yaw;
                mode.cam_pitch = self.player.pitch;

                if let Err(e) = mode.spawn(spawn_pos, &self.world) {
                    tracing::warn!("Mario spawn failed: {e}");
                }
            }
        }
    }

    #[cfg(not(feature = "mario"))]
    fn toggle_mario_mode(&mut self) {
        tracing::warn!(
            "Mario support is disabled; run with `cargo run -p vox-app --features mario`"
        );
    }
}

impl App for VoxApp {
    fn frame(&mut self, input: &mut InputState, timing: FrameTiming) -> FrameControl {
        let input_start = Instant::now();
        let exit = self.system_input(input);
        if exit {
            return FrameControl::Exit;
        }
        self.profile
            .input
            .push(input_start.elapsed().as_secs_f32() * 1000.0);

        self.system_player(input, timing);
        self.system_physics(timing);
        ecs_components::tick_ecs(&mut self.ecs, timing.dt_frame);
        // Replay: record (throttled internally to 1/sec) or apply the next
        // playback snapshot to the player + debris bodies. Runs after the
        // physics step so recorded body transforms are this frame's final
        // state, and after debris eviction so evicted bodies aren't captured.
        if self.replay.recording {
            self.replay.record(&self.player, self.game_time, &self.phys);
        } else if self.replay.is_playing() {
            if self
                .replay
                .playback_step(&mut self.player, &mut self.phys, &mut self.game_time)
            {
                // Keep the render-interpolated eye exactly on the snapshot
                // (fixed_steps is skipped during playback, so prev_pos would
                // otherwise lag the directly-written ctrl.pos).
                self.player.sync_prev_pos();
            }
        }
        // Feed nearby debris bodies to Mario as moving collision surfaces.
        // Must run after the physics step (fresh transforms) and after
        // debris eviction/lifetime expiry (so we don't register surfaces
        // for bodies that are already gone this frame).
        if self.frame_data.mario_active {
            let mario_pos = self.frame_data.mario_pos;
            let radius = mario::SURFACE_RADIUS_M + 2.0;
            let mut nearby: Vec<(u64, Vec3, glam::Quat, Vec3, Vec3)> = Vec::new();
            for (_id, body) in self.phys.iter() {
                let voxel_size = body.half_voxel * 2.0;
                let grid_min = body.grid_offset;
                let grid_max = body.grid_offset + body.grid.dims.as_vec3() * voxel_size;
                // Cheap world-AABB vs Mario distance pre-filter.
                let wmin = body.pos + grid_min.min(grid_max);
                let wmax = body.pos + grid_max.max(grid_min);
                let d = Vec3::new(
                    (mario_pos.x - wmin.x).max(0.0).max(wmax.x - mario_pos.x),
                    (mario_pos.y - wmin.y).max(0.0).max(wmax.y - mario_pos.y),
                    (mario_pos.z - wmin.z).max(0.0).max(wmax.z - mario_pos.z),
                );
                if d.length_squared() > radius * radius {
                    continue;
                }
                // Stable key: hash the BodyId (slot + generation).
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut h = DefaultHasher::new();
                _id.hash(&mut h);
                let key = h.finish();
                nearby.push((key, body.pos, body.rot, grid_min, grid_max));
            }
            self.mario_mode
                .as_mut()
                .unwrap()
                .update_debris(nearby.into_iter());
        }

        self.system_fluid(timing);

        // Stream chunks around the player: generate missing, evict beyond range.
        self.system_streaming();
        // Wake any resting debris whose ground was just carved/edited from
        // under it, then remesh: absorb edits, dispatch to workers, upload.
        let eye = self.player.eye(timing.alpha);
        let uploaded = self.system_remesh(eye);
        self.resolve_pending_removals(&uploaded);
        self.sync_debris_render(timing.alpha);
        self.system_render_prep(timing);

        let dn = day_night::compute(self.game_time);
        let view_proj = self.last_view_proj;
        let frustum = self.last_frustum.clone();
        let shadow_frustum = self.last_shadow_frustum.clone();
        let (w, h) = self.gpu.surface_size();
        let aspect = w as f32 / h.max(1) as f32;
        let render_start = Instant::now();
        let frame = match self.gpu.begin_frame() {
            Ok(frame) => frame,
            Err(err) if err.is_transient() => {
                tracing::warn!(error = %err, "transient surface error; skipping frame");
                return FrameControl::Continue;
            }
            Err(err) => {
                tracing::error!(error = %err, "fatal render error; shutting down");
                return FrameControl::Exit;
            }
        };

        let mut encoder =
            self.gpu
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("frame-encoder"),
                });

        // HUD + debug overlay UI must be built and their buffers uploaded
        // before the render pass opens (buffer uploads can't happen
        // mid-pass); painting itself happens inside the pass, after the
        // world. The HUD (crosshair, hotbar) draws every frame; the debug
        // windows only when F3-visible.
        let mut hud_slots: [Option<&str>; 9] = [None; 9];
        for (i, (_, tool)) in HOTBAR.iter().enumerate() {
            hud_slots[i] = Some(tool_label(*tool));
        }
        let hud_state = HudState {
            slots: hud_slots,
            active: HOTBAR
                .iter()
                .position(|(_, t)| *t == self.tools.tool)
                .unwrap_or(0),
            radius_m: self
                .tools
                .has_adjustable_radius()
                .then_some(self.tools.active_radius_m()),
            material_name: self
                .material_names
                .get(self.selected_material)
                .map(String::as_str)
                .unwrap_or("(none)"),
            material_color: self
                .registry
                .get(MaterialId((self.selected_material + 1) as u16))
                .map(|d| d.color)
                .unwrap_or([0.8, 0.8, 0.8]),
        };
        let ecs_entity_list: Vec<(u32, String, [f32; 3])> = self
            .ecs
            .entities()
            .filter_map(|id| {
                let name = self.ecs.get::<ecs_components::Name>(id).map(|n| n.0.clone()).unwrap_or("Unnamed".to_string());
                let pos = self.ecs.get::<ecs_components::Transform>(id).map(|t| [t.pos.x, t.pos.y, t.pos.z]).unwrap_or([0.0; 3]);
                Some((id.slot(), name, pos))
            })
            .collect();
        let debug_state = self.debug_visible.then(|| OverlayState {
            profile: &self.profile,
            tunables: &mut self.tunables,
            fps: self.last_fps,
            chunks_drawn: self.last_draw_stats.drawn,
            chunks_culled: self.last_draw_stats.culled,
            mesh_queue: self.remesh.pending_len(),
            body_mesh_in_flight: self.body_mesh.in_flight,
            bodies_awake: self.phys.awake_count(),
            bodies_total: self.phys.body_count(),
            particles: self.particles.len(),
            tool_radius: self.tools.active_radius_mut(),
            material_names: &self.material_names,
            selected_material: &mut self.selected_material,
            always_day: &mut self.always_day,
            quality_label: match self.chunk_loader.quality() {
                Quality::Low => "low",
                Quality::Medium => "medium",
                Quality::High => "high",
                Quality::Ultra => "ultra",
            },
            ecs_entity_count: self.ecs.entity_count(),
            ecs_entities: &ecs_entity_list,
        });
        let prepared_overlay = self.debug_overlay.prepare(
            &self.window,
            self.gpu.device(),
            self.gpu.queue(),
            &mut encoder,
            (w, h),
            &hud_state,
            debug_state,
        );
        self.tools.set_material_index(self.selected_material + 1);

        // Shadow pass (#14): render visible chunks into the shadow map
        // (depth-only, no color attachment) before the main pass samples it.
        // The pass must close (drop the shadow pass encoder guard) before
        // the main pass opens on the same encoder. Frustum culling reuses
        // the main camera frustum -- chunks outside the view are also
        // outside the shadow receiver region.
        if dn.sun_strength > 0.001 {
            let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow-pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.shadow_pipeline.shadow_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.pipeline.draw_chunks_shadow(
                &self.shadow_pipeline,
                &mut shadow_pass,
                &shadow_frustum,
            );
        }

        let stats;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("voxel-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.postprocess.color_view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(day_night::clear_color(&dn)),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.postprocess.depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(1, self.shadow_pipeline.sample_bind_group(), &[]);
            let chunk_stats = self.pipeline.draw_chunks_opaque(&mut pass, &frustum);
            let body_stats =
                self.pipeline
                    .draw_bodies(&mut pass, &frustum, self.camera.pos, FOG_END_M);
            stats = vox_render::DrawStats {
                drawn: chunk_stats.drawn + body_stats.drawn,
                culled: chunk_stats.culled + body_stats.culled,
            };
            if let Some(mode) = &self.mario_mode {
                if mode.is_active() {
                    mode.render(
                        self.gpu.queue(),
                        &mut pass,
                        view_proj.to_cols_array_2d(),
                        self.camera.pos,
                        dn.sun_dir,
                        dn.sun_strength,
                        dn.sky_color,
                        dn.fill_strength,
                        dn.ambient_strength,
                        dn.sun_color,
                        dn.ambient_sky,
                        dn.ambient_ground,
                        FOG_END_M * 0.55,
                        FOG_END_M,
                    );
                }
            }
            // Grass and particles before water so water blends on top.
            self.particle_pipeline.draw(&mut pass);
            let grass_voxel = self
                .registry
                .id_by_name("grass")
                .map(|id| Voxel(id.0))
                .unwrap_or(Voxel(3));
            let grass_verts = self.grass_cache.get_or_regen(
                &self.world,
                self.camera.pos,
                self.world.cfg.voxel_size_m,
                grass_voxel,
            );
            self.grass_pipeline.write_camera(
                self.gpu.queue(),
                view_proj.to_cols_array_2d(),
                self.camera.pos,
                dn.sun_dir,
                dn.sun_strength,
                dn.sky_color,
                dn.fill_strength,
                dn.ambient_strength,
                dn.sun_color,
                dn.ambient_sky,
                dn.ambient_ground,
                self.game_time,
                self.world.cfg.voxel_size_m,
                FOG_END_M,
            );
            self.grass_pipeline
                .draw(self.gpu.queue(), &mut pass, grass_verts);
            // Water last — alpha-blends over terrain, grass, and particles.
            self.pipeline.draw_water(&mut pass, &frustum);
        }

        // SSAO + bloom passes: generate AO and bloom from the scene's HDR color + depth.
        let proj = self.camera.projection(aspect);
        let inv_proj = proj.inverse();
        self.bloom_ssao.write_params(
            self.gpu.queue(),
            proj.to_cols_array_2d(),
            inv_proj.to_cols_array_2d(),
            self.tunables.ssao_intensity,
            self.tunables.ssao_radius,
            self.tunables.bloom_intensity,
            self.tunables.bloom_threshold,
        );
        self.bloom_ssao.process(
            self.gpu.device(),
            &mut encoder,
            self.postprocess.color_view(),
            self.postprocess.depth_view(),
        );

        // Post-process pass: composite the offscreen HDR color + depth
        // through the fullscreen edge-detection/saturation/grading shader
        // onto the swapchain frame.
        self.postprocess.process(&mut encoder, frame.view());

        // UI pass: debug overlay drawn directly on the swapchain, on top
        // of the post-processed scene. No depth attachment — egui's
        // pipeline is built against None in its constructor.
        {
            let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame.view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.debug_overlay.paint(&mut ui_pass, &prepared_overlay);
        }
        self.gpu.queue().submit([encoder.finish()]);
        frame.present();
        self.profile
            .render
            .push(render_start.elapsed().as_secs_f32() * 1000.0);
        self.last_draw_stats = stats;

        self.report_stats(stats);
        FrameControl::Continue
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.gpu.resize(width, height);
        self.bloom_ssao.resize(&self.gpu, width, height);
        self.postprocess.resize(
            &self.gpu,
            width,
            height,
            self.bloom_ssao.ao_view(),
            self.bloom_ssao.bloom_view(),
        );
    }

    fn window_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        self.debug_overlay.on_window_event(&self.window, event)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli_args: Vec<String> = std::env::args().skip(1).collect();
    if args::wants_help(cli_args.iter().map(String::as_str)) {
        println!("{}", args::usage());
        return Ok(());
    }
    let cli = match args::parse(cli_args.iter().map(String::as_str)) {
        Ok(cli) => cli,
        Err(msg) => {
            eprintln!("error: {msg}\n\n{}", args::usage());
            std::process::exit(1);
        }
    };
    let cfg = cli.world;
    let mario_units_per_meter = cli.mario_units_per_meter;
    let quality = cli.quality;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    tracing::info!(
        voxel_size_m = cfg.voxel_size_m,
        seed = cfg.seed,
        extent_m = ?cfg.extent_m,
        mario_units_per_meter,
        "world config"
    );

    run_app(vox_core::consts::PHYSICS_DT, |window| {
        Ok(Box::new(VoxApp::new(window, cfg, mario_units_per_meter, quality)?))
    })?;
    Ok(())
}

/// Short human-facing label for a hotbar slot.
fn tool_label(tool: Tool) -> &'static str {
    match tool {
        Tool::Dig => "Dig",
        Tool::ScalableDig => "Carve",
        Tool::Bomb => "Bomb",
        Tool::DeathLaser => "Laser",
        Tool::PlaceWater => "Water",
        Tool::Ember => "Ember",
    }
}

/// Radius, in voxels, of the smallest possible fracture -- what *every*
/// material chips loose the instant an impact just barely clears its own
/// threshold, regardless of how fragile or tough that material is. A tiny
/// hit must always produce a tiny bite, never "an orb of voxels deleted
/// from space": see [`fracture_radius_vox`]'s doc comment for how growth
/// past this baseline is what actually differs per material.
const FRACTURE_RADIUS_VOX: f32 = 1.25;
/// Strength a material's crumble *growth rate* is calibrated against
/// (stone's own strength: the toughest common building material, so its
/// growth factor bottoms out at 0 -- stone barely chips even under a
/// genuinely hard hit, it just keeps producing the base radius above).
const FRACTURE_REFERENCE_STRENGTH: f32 = 8.0;
/// Clamp on the per-material growth factor derived from
/// `FRACTURE_REFERENCE_STRENGTH / strength` -- keeps an extremely fragile
/// material (leaves, strength 0.5) from growing arbitrarily fast as
/// `strength` approaches zero, while still growing much faster than a
/// tougher one for the same excess impact speed.
const CRUMBLE_SCALE_RANGE: (f32, f32) = (1.0, 5.0);
/// Cap on how far the square-root growth term (see
/// [`fracture_radius_vox`]) is allowed to run, so a truly enormous single
/// impact (e.g. a falling boulder) still can't carve out most of a small
/// body in one hit.
const ENERGY_GROWTH_CAP: f32 = 1.5;
/// Absolute ceiling on the fracture radius, in voxels, regardless of how
/// fragile the material or how violent the impact -- a hard backstop on top
/// of `ENERGY_GROWTH_CAP`/`CRUMBLE_SCALE_RANGE`'s already-clamped growth.
/// Impact fracture is meant to read as "a chunk breaks free and crumbles,"
/// not "a chunk of the object vanishes into a smooth spherical void": a
/// single-hit carve this large on something like a tree's leaf canopy (a
/// low-strength material with the fastest growth) looked exactly like the
/// latter before this cap existed, however many small flying chips were
/// scattered around its edge.
const MAX_FRACTURE_RADIUS_VOX: f32 = 3.0;
/// Bodies with fewer solid voxels than this are terminal rubble: they
/// bounce and settle, but never impact-fracture again. Without this floor,
/// destruction *cascaded*: a fracture scatters 3-voxel chips, a chip's next
/// bounce trivially clears a fragile material's threshold (leaves fracture
/// at 0.5 m/s of delta-v -- every bounce), and since every fragment of a
/// 3-voxel body is below `DEBRIS_MIN_VOXELS` (4), the chip vanished as dust
/// while `spawn_impact_chips` emitted fresh chips from what was removed --
/// each generation re-cloning grids, re-running component labeling,
/// re-meshing, and churning GPU buffers, every single bounce, until the
/// debris budget filled with popping, vanishing, respawning specks. That
/// churn was both the "debris glitches around" report and a large share of
/// the "lots of debris causes lag" one.
const MIN_FRACTURE_BODY_VOXELS: usize = 16;

/// Impacts below this fraction of the fracture threshold do nothing.
const DAMAGE_THRESHOLD_FACTOR: f32 = 0.3;
/// Damage gained per hit at exactly the fracture threshold.
const DAMAGE_RATE: f32 = 0.3;
/// Damage at which a voxel crumbles (becomes air). The actual crumble check
/// lives in `vox_physics::apply_body_damage` (`>= 1.0`); this constant
/// documents the threshold and is reserved for future app-side gating.
#[expect(dead_code, reason = "documents the crumble threshold; enforced in vox-physics")]
const DAMAGE_CRUMBLE: f32 = 1.0;

/// Pure impact-fracture decision, kept free of live GPU/registry state so
/// it's unit-testable without a whole `VoxApp`: given a material's
/// `strength`, the impact speed it just took, and the live
/// `fracture_sensitivity` dial, should it fracture, and if so, what carve
/// radius (in voxels, before scaling by the body's own voxel size)?
/// `None` if the impact doesn't clear the material's threshold.
///
/// Direct proportionality on the threshold: a tougher material needs *more*
/// impact speed to fracture at all, matching the same `strength` convention
/// every other destruction tool already uses (higher = harder to destroy).
/// An earlier version of this divided instead of multiplied, which made
/// stone fracture *more* easily than wood.
///
/// The radius is `FRACTURE_RADIUS_VOX` plus a growth term that is zero at
/// the bare threshold and increases with *how far past* it the impact was,
/// scaled by how fragile the material is. This matters more than it might
/// look: an earlier version scaled the *entire* radius by a per-material
/// factor (leaves' factor pinned at its max range regardless of impact
/// strength), so even the gentlest leaf-fracturing hit already carved a 5x
/// radius -- "a tiny hit blows out a huge chunk." Splitting growth from the
/// base fixes that: every material's bare-threshold hit is the same small
/// bite, and *only* harder impacts open up the gap between a tough
/// material (stone's factor bottoms out at 0 growth -- it just keeps
/// producing the base bite) and a fragile one (leaves grows fastest).
fn fracture_radius_vox(strength: f32, impact_speed: f32, fracture_sensitivity: f32) -> Option<f32> {
    if strength <= 0.0 {
        return None; // Already as fragile as it gets; nothing to compare against.
    }
    let threshold = fracture_sensitivity * strength;
    if impact_speed < threshold {
        return None;
    }
    let fragility = (FRACTURE_REFERENCE_STRENGTH / strength)
        .clamp(CRUMBLE_SCALE_RANGE.0, CRUMBLE_SCALE_RANGE.1);
    let excess = (impact_speed / threshold - 1.0).max(0.0);
    let growth = excess.sqrt().min(ENERGY_GROWTH_CAP);
    let radius = FRACTURE_RADIUS_VOX * (1.0 + growth * (fragility - 1.0));
    Some(radius.min(MAX_FRACTURE_RADIUS_VOX))
}

/// Pure eviction decision behind `VoxApp::enforce_debris_budget`, kept free
/// of GPU/pipeline state so it's unit-testable directly against a
/// `PhysicsWorld`: while `phys` has more than `max_bodies` live bodies, pop
/// the oldest entry from `order`; if it's still alive and asleep, despawn
/// it (recorded in the returned list, so the caller can also drop its GPU
/// mesh); if it's still awake, requeue it at the back and keep looking
/// further back in time instead. Bounded to one pass over `order`'s
/// current length, so a run of nothing-but-awake debris can't turn this
/// into an unbounded loop -- it's fine to briefly sit over budget rather
/// than yank debris out from under the player mid-flight.
fn evict_oldest_asleep_debris(
    phys: &mut PhysicsWorld,
    order: &mut VecDeque<BodyId>,
    max_bodies: usize,
) -> Vec<BodyId> {
    let mut evicted = Vec::new();
    let mut attempts = order.len();
    while phys.body_count() > max_bodies && attempts > 0 {
        attempts -= 1;
        let Some(id) = order.pop_front() else {
            break;
        };
        match phys.get(id) {
            None => {} // Already gone (despawned elsewhere); drop the stale entry.
            Some(body) if body.sleep.asleep => {
                phys.despawn(id);
                evicted.push(id);
            }
            Some(_) => order.push_back(id), // Still active; retry later.
        }
    }
    evicted
}

#[cfg(test)]
mod fire_smoke_tests {
    use super::*;

    #[test]
    fn smoke_origin_is_inside_the_selected_air_neighbor() {
        let pos = IVec3::new(8, 7, 6);
        let voxel_size = 0.1;
        for face in [
            IVec3::X,
            IVec3::NEG_X,
            IVec3::Y,
            IVec3::NEG_Y,
            IVec3::Z,
            IVec3::NEG_Z,
        ] {
            let origin = fire_smoke_origin(pos, face, voxel_size, 42);
            assert_eq!(
                voxel_at(origin, voxel_size),
                pos + face,
                "smoke must start in the exposed neighbor for face {face}"
            );
        }
    }

    fn body_with_ember(voxel: IVec3) -> Body {
        let dims = IVec3::splat(3);
        let mut voxels = vec![Voxel(1); 27];
        let index = (voxel.x + voxel.z * dims.x + voxel.y * dims.x * dims.z) as usize;
        voxels[index] = Voxel(2);
        let grid = VoxelGrid::new(dims, voxels);
        let registry = MaterialRegistry::from_toml_str(
            "[[material]]\nname=\"stone\"\ncolor=[0.5,0.5,0.5]\ndensity=2600.0\nstrength=8.0\n\n[[material]]\nname=\"ember\"\ncolor=[0.8,0.3,0.1]\ndensity=600.0\nstrength=2.0\nflammable=true\n",
            "fire-smoke-test.toml",
        )
        .expect("test registry");
        Body::from_grid(grid, &registry, 0.1, Vec3::ZERO).expect("body mass")
    }

    #[test]
    fn interior_ember_has_no_body_smoke_outlet() {
        let body = body_with_ember(IVec3::ONE);
        assert!(body_smoke_outlets(&body, Voxel(2), 0.1).is_empty());
    }

    #[test]
    fn exposed_ember_only_emits_through_air_faces() {
        let body = body_with_ember(IVec3::new(0, 1, 1));
        let outlets = body_smoke_outlets(&body, Voxel(2), 0.1);
        assert_eq!(outlets.len(), 1);
        assert_eq!(outlets[0].open_faces, vec![IVec3::NEG_X]);
        let origin = body_smoke_origin(&body, &outlets[0], IVec3::NEG_X, 0.1, 7);
        let body_local = (body.rot.conjugate() * (origin - body.pos) - body.grid_offset) / 0.1;
        assert_eq!(body_local.floor().as_ivec3(), IVec3::new(-1, 1, 1));
    }
}

#[cfg(test)]
mod debris_budget_tests {
    use super::*;

    fn asleep_body(pos: Vec3) -> Body {
        let grid = VoxelGrid::new(IVec3::splat(2), vec![Voxel(1); 8]);
        let reg = MaterialRegistry::from_toml_str(
            "[[material]]\nname = \"stone\"\ncolor = [0.5,0.5,0.5]\ndensity = 2600.0\nstrength = 8.0\n",
            "test.toml",
        )
        .expect("registry");
        let mut body = Body::from_grid(grid, &reg, 0.2, pos).expect("massive");
        body.sleep.asleep = true;
        body
    }

    #[test]
    fn evicts_oldest_first_once_over_budget() {
        let mut phys = PhysicsWorld::new();
        let mut order = VecDeque::new();
        let mut ids = Vec::new();
        for i in 0..5 {
            let id = phys.spawn(asleep_body(Vec3::splat(i as f32)));
            order.push_back(id);
            ids.push(id);
        }

        let evicted = evict_oldest_asleep_debris(&mut phys, &mut order, 3);

        assert_eq!(
            evicted,
            vec![ids[0], ids[1]],
            "must evict the two oldest, in order"
        );
        assert_eq!(phys.body_count(), 3);
        assert!(phys.get(ids[2]).is_some(), "the three newest must survive");
    }

    #[test]
    fn under_budget_evicts_nothing() {
        let mut phys = PhysicsWorld::new();
        let mut order = VecDeque::new();
        for i in 0..3 {
            order.push_back(phys.spawn(asleep_body(Vec3::splat(i as f32))));
        }

        let evicted = evict_oldest_asleep_debris(&mut phys, &mut order, 10);

        assert!(evicted.is_empty());
        assert_eq!(phys.body_count(), 3);
    }

    #[test]
    fn never_evicts_an_awake_body_even_over_budget() {
        let mut phys = PhysicsWorld::new();
        let mut order = VecDeque::new();
        let mut awake = asleep_body(Vec3::ZERO);
        awake.sleep.asleep = false;
        let awake_id = phys.spawn(awake);
        order.push_back(awake_id);
        for i in 1..4 {
            order.push_back(phys.spawn(asleep_body(Vec3::splat(i as f32))));
        }

        let evicted = evict_oldest_asleep_debris(&mut phys, &mut order, 1);

        assert!(
            !evicted.contains(&awake_id),
            "an awake body must never be evicted, even to satisfy the budget"
        );
        assert!(phys.get(awake_id).is_some());
    }

    #[test]
    fn a_stale_entry_for_an_already_despawned_body_is_dropped_harmlessly() {
        let mut phys = PhysicsWorld::new();
        let mut order = VecDeque::new();
        let stale_id = phys.spawn(asleep_body(Vec3::ZERO));
        order.push_back(stale_id);
        phys.despawn(stale_id); // despawned by something else entirely

        // A second, still-alive body keeps `phys.body_count()` above the
        // budget, so the eviction loop actually has a reason to process
        // the stale front-of-queue entry instead of exiting immediately.
        let real_id = phys.spawn(asleep_body(Vec3::ONE));
        order.push_back(real_id);

        let evicted = evict_oldest_asleep_debris(&mut phys, &mut order, 0);

        assert!(
            !evicted.contains(&stale_id),
            "a stale id must not be reported as evicted -- it was already gone"
        );
        assert_eq!(
            evicted,
            vec![real_id],
            "the real, still-alive body must still be evicted"
        );
        assert!(
            order.is_empty(),
            "the stale entry must still be dropped from the queue"
        );
    }
}

#[cfg(test)]
mod fracture_tests {
    use super::*;

    /// The core material set's real strengths (see
    /// `assets/materials/core.toml`): leaves 0.5, wood 4.0, stone 8.0.
    const LEAVES: f32 = 0.5;
    const WOOD: f32 = 4.0;
    const STONE: f32 = 8.0;

    #[test]
    fn tougher_materials_need_more_speed_to_fracture_at_all() {
        // Same impact speed, everywhere between wood's and stone's
        // thresholds: wood must fracture, stone must not.
        let sensitivity = 1.0;
        let speed = WOOD * sensitivity + 0.1;
        assert!(fracture_radius_vox(WOOD, speed, sensitivity).is_some());
        assert!(fracture_radius_vox(STONE, speed, sensitivity).is_none());
    }

    #[test]
    fn leaves_fracture_on_a_much_gentler_impact_than_wood_or_stone() {
        let sensitivity = 1.0;
        let gentle = LEAVES * sensitivity + 0.05;
        assert!(
            fracture_radius_vox(LEAVES, gentle, sensitivity).is_some(),
            "leaves must give way at a gentle speed"
        );
        assert!(
            fracture_radius_vox(WOOD, gentle, sensitivity).is_none(),
            "the same gentle speed must not touch wood"
        );
        assert!(
            fracture_radius_vox(STONE, gentle, sensitivity).is_none(),
            "the same gentle speed must not touch stone"
        );
    }

    #[test]
    fn weaker_materials_crumble_a_bigger_radius_once_they_do_fracture() {
        // Exactly stone's own threshold: clears every material's threshold
        // (stone's is the highest) while keeping stone's `excess` at 1.0, so
        // its energy scale-up is a no-op and it lands exactly on the
        // reference radius below.
        let sensitivity = 1.0;
        let hard_hit = STONE * sensitivity;
        let leaves_r = fracture_radius_vox(LEAVES, hard_hit, sensitivity).expect("must fracture");
        let wood_r = fracture_radius_vox(WOOD, hard_hit, sensitivity).expect("must fracture");
        let stone_r = fracture_radius_vox(STONE, hard_hit, sensitivity).expect("must fracture");
        assert!(
            leaves_r > wood_r,
            "leaves ({leaves_r}) must crumble more than wood ({wood_r})"
        );
        assert!(
            wood_r > stone_r,
            "wood ({wood_r}) must crumble more than stone ({stone_r})"
        );
        assert_eq!(
            stone_r, FRACTURE_RADIUS_VOX,
            "stone is the reference material"
        );
    }

    #[test]
    fn massless_material_never_fractures() {
        assert!(fracture_radius_vox(0.0, 1000.0, 1.0).is_none());
    }
}
