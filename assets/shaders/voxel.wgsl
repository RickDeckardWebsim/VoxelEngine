// Cel-shaded voxel pipeline: chunks and debris bodies share this shader.
// Lighting is quantized into 4 bands (painterly cel-shading). Outputs both
// color and world-space normals for the post-process edge detection pass.

struct Camera {
    view_proj: mat4x4f,
    cam_pos: vec4f,
    sun_dir: vec4f,          // xyz = direction the sun shines toward (unit)
    fog: vec4f,              // x = start (m), y = end (m), z = voxel size (m)
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> palette: array<vec4f>; // rgb + jitter

struct Inst {
    @location(4) m0: vec4f,
    @location(5) m1: vec4f,
    @location(6) m2: vec4f,
    @location(7) m3: vec4f,
};

struct VIn {
    @location(0) pos_ao: vec4<u32>,   // x, y, z corner (voxel units), ao 0..3
    @location(1) norm_mat: vec4<u32>, // normal id, jitter 0..255, material lo, material hi
};

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) color: vec3f,
    @location(1) world_normal: vec3f,
    @location(2) ao: f32,
    @location(3) world_pos: vec3f,
    @location(4) mat_id: f32,
    @location(5) jitter_raw: f32,
};

const SKY_COLOR = vec3f(0.45, 0.66, 0.90);
// Ambient is two-tone (sky-facing vs. ground-facing) rather than a flat
// scalar: a cool tint from above, a warmer/darker one from below, mixed by
// how much the surface faces up vs. down. Reads far less "flat gray" in
// shadow than a single ambient number ever can.
const AMBIENT_SKY = vec3f(0.50, 0.58, 0.70);
const AMBIENT_GROUND = vec3f(0.30, 0.27, 0.24);
const AMBIENT_STRENGTH = 0.55;
const SUN_STRENGTH = 0.85;
// Faint light from the direction opposite the sun -- like real bounce
// light off the sky and surroundings -- so a face angled away from the
// sun still reads as lit, not flat black. Small on purpose: it's a floor,
// not a second sun.
const FILL_STRENGTH = 0.12;

// Face normal from id (0..6 = +X, -X, +Y, -Y, +Z, -Z). Arithmetic instead of
// a const-array lookup: naga rejects dynamic indexing of module constants.
fn face_normal(id: u32) -> vec3f {
    let s = 1.0 - 2.0 * f32(id & 1u);
    let axis = id >> 1u;
    var n = vec3f(0.0, 0.0, 0.0);
    if axis == 0u {
        n.x = s;
    } else if axis == 1u {
        n.y = s;
    } else {
        n.z = s;
    }
    return n;
}

@vertex
fn vs(v: VIn, inst: Inst) -> VOut {
    let model = mat4x4f(inst.m0, inst.m1, inst.m2, inst.m3);
    let local = vec3f(f32(v.pos_ao.x), f32(v.pos_ao.y), f32(v.pos_ao.z)) * cam.fog.z;
    let wp = (model * vec4f(local, 1.0)).xyz;

    let mat_id = v.norm_mat.z | (v.norm_mat.w << 8u);
    let base = palette[mat_id];
    // Jitter is baked into the mesh once at build time (see vox-mesh's
    // `jitter_hash`), not recomputed here from world position: hashing a
    // *moving* vertex's world position dynamically made the jitter shift
    // continuously as a debris body translated/rotated, which read as
    // flicker on its surface -- chunks never move, so they never showed it,
    // matching the exact "only on detached bodies" symptom this fixed.
    let h = f32(v.norm_mat.y) / 255.0;
    let mat_id_v = v.norm_mat.z | (v.norm_mat.w << 8u);
    let water_id_v = u32(cam.fog.w);
    // For water faces, the jitter field holds water depth, not color
    // jitter — don't apply it as color variation.
    let color_jitter = select(h, 0.5, mat_id_v == water_id_v);

    var out: VOut;
    out.clip = cam.view_proj * vec4f(wp, 1.0);
    out.color = base.rgb * (1.0 + (color_jitter - 0.5) * 2.0 * base.a);
    // Chunks never rotate, so their local and world axes coincide -- but a
    // debris body's instance matrix carries real rotation (it tumbles), and
    // lighting a tumbling body against its *local* (un-rotated) face normal
    // makes the lit/shadowed faces stay fixed to the body instead of the
    // world's actual sun direction: it looks like the light is glued to the
    // object and spinning with it. Rotating the normal by the model matrix
    // here (translation-free, via w=0) fixes that for both cases uniformly.
    let local_n = face_normal(v.norm_mat.x);
    out.world_normal = normalize((model * vec4f(local_n, 0.0)).xyz);
    out.ao = f32(v.pos_ao.w) / 3.0;
    out.world_pos = wp;
    out.mat_id = f32(mat_id);
    out.jitter_raw = f32(v.norm_mat.y);
    return out;
}

struct FOut {
    @location(0) color: vec4f,
    @location(1) normal: vec4f,  // world normal encoded to 0..1 for edge detection
    @location(2) depth_out: vec4f, // linear depth for edge detection (Rgba16Float)
};

@fragment
fn fs(in: VOut) -> FOut {
    let n = normalize(in.world_normal);
    let ndotl = dot(n, -cam.sun_dir.xyz);
    let water_id = u32(cam.fog.w);
    let is_water = u32(in.mat_id) == water_id;

    var c: vec3f;

    if is_water {
        // Water: flat, uniform color — no per-face lighting, no AO, no
        // specular, no Fresnel. Just the base water color darkened by
        // depth. This looks clean and smooth, not glitchy.
        c = in.color * 0.8;

        // Gentle depth-based darkening.
        let water_depth = in.jitter_raw / 255.0;
        let absorption = clamp(1.0 - exp(-water_depth * 15.0), 0.0, 0.6);
        let deep_color = vec3f(0.02, 0.06, 0.12);
        c = mix(c, deep_color, absorption);
    } else {
        // Standard lighting for non-water materials.
        let sun = pow(clamp(ndotl * 0.5 + 0.5, 0.0, 1.0), 1.5) * SUN_STRENGTH;
        let fill = max(-ndotl, 0.0) * FILL_STRENGTH;
        let hemi_t = clamp(0.5 + 0.5 * n.y, 0.0, 1.0);
        let ambient = mix(AMBIENT_GROUND, AMBIENT_SKY, hemi_t) * AMBIENT_STRENGTH;
        let ao = 0.45 + 0.55 * in.ao;
        c = in.color * (ambient + vec3f(sun + fill)) * ao;
    }

    let dist = length(in.world_pos - cam.cam_pos.xyz);
    let f = clamp((dist - cam.fog.x) / (cam.fog.y - cam.fog.x), 0.0, 1.0);
    c = mix(c, SKY_COLOR, f * f);

    let enc_n = n * 0.5 + 0.5;
    let fog_n = mix(enc_n, vec3f(0.5, 0.5, 0.5), f * f);

    var out: FOut;
    out.color = vec4f(c, 1.0);
    out.normal = vec4f(fog_n, 1.0);
    let ndc_depth = in.clip.z / in.clip.w;
    out.depth_out = vec4f(ndc_depth, 0.0, 0.0, 1.0);
    return out;
}
