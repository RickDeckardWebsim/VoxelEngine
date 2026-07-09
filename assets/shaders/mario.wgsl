// Mario pipeline: renders libsm64's dynamic per-frame geometry (up to
// 1024 triangles). Vertices come from sm64_mario_tick's geometry
// buffers: position (float3), normal (float3), color (float3), uv
// (float2). Textured with Mario's 704×64 RGBA atlas extracted from
// the ROM. Simple sun + ambient lighting matching the voxel pipeline's
// look, plus distance fog for consistency.

struct Camera {
    view_proj: mat4x4f,
    cam_pos: vec4f,
    sun_dir: vec4f,          // xyz = direction the sun shines toward (unit)
    fog: vec4f,              // x = start (m), y = end (m), z = model_scale, w = unused
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var mario_sampler: sampler;
@group(0) @binding(2) var mario_texture: texture_2d<f32>;

struct VIn {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec3f,
    @location(3) uv: vec2f,
};

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) color: vec3f,
    @location(1) world_normal: vec3f,
    @location(2) world_pos: vec3f,
    @location(3) uv: vec2f,
};

const SKY_COLOR = vec3f(0.45, 0.66, 0.90);
const AMBIENT_SKY = vec3f(0.40, 0.46, 0.56);
const AMBIENT_GROUND = vec3f(0.24, 0.22, 0.19);
const AMBIENT_STRENGTH = 0.35;
const SUN_COLOR = vec3f(1.0, 0.95, 0.85);
const SUN_STRENGTH = 0.45;
const FOG_COLOR = vec3f(0.45, 0.66, 0.90);

// SM64 units per meter (must match vox_sm64::SM64_UNITS_PER_METER)
const SM64_SCALE = 30.0;

@vertex
fn vs_main(in: VIn) -> VOut {
    var out: VOut;
    // libsm64 outputs absolute positions in SM64 integer units.
    // We need to: (1) convert world position to meters, (2) scale the
    // model so Mario is a reasonable size relative to the voxel terrain.
    //
    // Mario's native model is ~160 SM64 units tall. At 30 units/meter
    // that's 5.3m — too big. We use cam.fog.z as a model_scale uniform
    // to shrink the model while keeping the world position correct.
    //
    // The trick: the vertex position is absolute (world + model offset).
    // We extract the world position by rounding to the surface grid,
    // then scale only the sub-voxel model offset. Simpler approach:
    // just scale the entire position by model_scale/SM64_SCALE, which
    // makes Mario smaller but also moves him closer to origin. Instead,
    // we scale around Mario's position: world_pos = mario_world_pos +
    // (vertex_pos - mario_world_pos) * model_scale.
    //
    // But we don't have mario_world_pos in the shader. Alternative:
    // the CPU pre-scales the vertices before upload. The caller
    // extracts Mario's center position and scales the offsets.
    // For now, just divide by SM64_SCALE — Mario will be big but
    // positioned correctly. The CPU-side fix in MarioPipeline::draw
    // handles the model scaling.
    let world_pos = in.position / SM64_SCALE;
    out.clip = cam.view_proj * vec4f(world_pos, 1.0);
    out.color = in.color;
    out.world_normal = normalize(in.normal);
    out.world_pos = world_pos;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
    let normal = normalize(in.world_normal);
    let sun_dir = normalize(cam.sun_dir.xyz);

    // Lighting: simple half-Lambert (matches libsm64's reference renderer)
    let light = 0.5 + 0.5 * clamp(dot(normal, sun_dir), 0.0, 1.0);

    // Alpha-masked overlay: vertex color is the base body color
    // (skin, hat, overalls), texture overrides only where alpha=1
    // (eyes, buttons, sideburns, emblem). This is exactly how
    // libsm64's reference GL renderer does it.
    let tex_color = textureSample(mario_texture, mario_sampler, in.uv);
    let main_color = mix(in.color, tex_color.rgb, tex_color.a);
    let lit = main_color * light;

    // Distance fog — matches the voxel pipeline
    let dist = length(cam.cam_pos.xyz - in.world_pos);
    let fog_factor = clamp((cam.fog.y - dist) / (cam.fog.y - cam.fog.x), 0.0, 1.0);
    let final_color = mix(FOG_COLOR, lit, fog_factor);

    return vec4f(final_color, 1.0);
}
