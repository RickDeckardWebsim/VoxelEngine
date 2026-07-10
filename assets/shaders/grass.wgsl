// Grass blade pipeline: renders thin 3D blades standing up from grass
// voxels. Each blade is a 2-triangle quad with wind sway applied to the
// tip in the vertex shader. Alpha-blended, depth-tested.
// Lighting matches the voxel pipeline for day/night consistency.

struct Camera {
    view_proj: mat4x4f,
    cam_pos: vec4f,
    sun_dir: vec4f,          // xyz = sun direction, w = sun strength
    fog: vec4f,              // x = start, y = end, z = voxel size, w = ambient strength
    sky_color: vec4f,        // xyz = sky/fog color, w = fill strength
    sun_color: vec4f,        // xyz = sun color, w = game time
    ambient_sky: vec4f,      // xyz = ambient sky tint, w = unused
    ambient_ground: vec4f,   // xyz = ambient ground tint, w = unused
};

@group(0) @binding(0) var<uniform> cam: Camera;

struct VIn {
    @location(0) position: vec3f,    // world position of this vertex (meters)
    @location(1) height_factor: f32, // 0=base, 1=tip — drives wind + color
};

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) height_factor: f32,
    @location(1) world_pos: vec3f,
};

@vertex
fn vs_main(in: VIn) -> VOut {
    var out: VOut;
    var world_pos = in.position;

    // Wind sway: offset blade tips (height_factor > 0) with time-based sin.
    if (in.height_factor > 0.5) {
        let t = cam.sun_color.w; // game time
        let wind_x = sin(t * 1.5 + world_pos.x * 0.7 + world_pos.z * 0.5) * 0.15 * in.height_factor;
        let wind_z = cos(t * 1.1 + world_pos.z * 0.9 + world_pos.x * 0.3) * 0.10 * in.height_factor;
        world_pos.x += wind_x;
        world_pos.z += wind_z;
    }

    out.clip = cam.view_proj * vec4f(world_pos, 1.0);
    out.height_factor = in.height_factor;
    out.world_pos = world_pos;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
    // Color gradient: match the voxel grass palette (material 3).
    let base_green = vec3f(0.20, 0.35, 0.15);
    let tip_green = vec3f(0.33, 0.55, 0.25);
    let color = mix(base_green, tip_green, in.height_factor);

    // Lighting: hemisphere ambient + sun, matching the voxel pipeline.
    // Grass faces up (+Y), so hemi_t ≈ 1.0 (full sky ambient).
    let ambient = cam.ambient_sky.xyz * cam.fog.w;
    let ndotl = dot(vec3f(0.0, 1.0, 0.0), -cam.sun_dir.xyz);
    let sun = pow(clamp(ndotl * 0.5 + 0.5, 0.0, 1.0), 1.5) * cam.sun_dir.w * cam.sun_color.xyz;
    let lit = color * (ambient + sun);

    // Distance fog — matches the voxel pipeline.
    let dist = length(cam.cam_pos.xyz - in.world_pos);
    let f = clamp((dist - cam.fog.x) / (cam.fog.y - cam.fog.x), 0.0, 1.0);
    let final_color = mix(lit, cam.sky_color.xyz, f * f);

    return vec4f(final_color, 0.95);
}
