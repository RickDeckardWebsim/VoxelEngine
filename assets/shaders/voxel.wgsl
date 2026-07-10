// Opaque voxel pipeline: chunks and debris bodies share this shader.
// Shading: material palette color + per-vertex jitter hash, directional sun
// with a soft (half-Lambert) terminator, a faint opposite-direction fill
// light, a two-tone sky/ground ambient, baked vertex AO, distance fog.

struct Camera {
    view_proj: mat4x4f,
    cam_pos: vec4f,
    sun_dir: vec4f,          // xyz = sun direction (unit), w = sun strength
    fog: vec4f,              // x = start (m), y = end (m), z = voxel size (m), w = ambient strength
    sky_color: vec4f,        // xyz = sky/fog color, w = fill light strength
    sun_color: vec4f,        // xyz = sun color (linear RGB), w = game time (seconds)
    ambient_sky: vec4f,      // xyz = ambient sky tint, w = crack decal intensity (0 = off)
    ambient_ground: vec4f,   // xyz = ambient ground tint, w = unused
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> palette: array<vec4f>; // rgb + jitter
// Shadow camera uniform, must match `shadow.wgsl`'s `ShadowCam`.
struct ShadowCam {
    view_proj: mat4x4f,
    params: vec4f,  // x = voxel_size_m; y/z/w unused
};

@group(1) @binding(0) var<uniform> shadow_cam: ShadowCam;
@group(1) @binding(1) var shadow_map: texture_depth_2d;
@group(1) @binding(2) var shadow_sampler: sampler_comparison;

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
    @location(4) @interpolate(flat) mat_id: u32,
};
// All lighting constants are now uniforms (cam.*) for day/night cycle.

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

// --- Procedural crack decals (#43) ---
// Pure-visual crack overlay driven by cam.ambient_sky.w (crack_intensity).
// Not tied to real damage state yet; intensity is 0 by default so the
// pattern is invisible until a future change drives it from per-voxel damage.
// Branching dark lines are built from a hash of world position: several
// ridges at different frequencies + offsets are combined, and only the
// sharpest crests read as cracks.

// Integer hash -> [0,1). Used to seed per-cell crack jitter so the pattern
// isn't a perfectly regular grid of lines.
fn hash13(x: i32, y: i32, z: i32) -> f32 {
    var h = bitcast<u32>(x) * 374761393u + bitcast<u32>(y) * 668265263u + bitcast<u32>(z) * 2147483647u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h) / 4294967296.0;
}

// One crack ridge: a high-frequency sin field whose near-zero crossings form
// a thin line. Returns ~1 near a line, ~0 elsewhere. `freq` sets line
// spacing, `width` sets line thickness, `off` shifts the grid per cell so
// ridges at different frequencies don't align into a single pattern.
fn crack_ridge(p: vec3f, freq: f32, width: f32, off: f32) -> f32 {
    let v = sin(p.x * freq + off) * 1.7 + sin(p.y * freq * 1.3 + off * 2.1) * 1.3 + sin(p.z * freq * 0.9 + off * 0.7) * 1.1;
    // Ridge: peak at v≈0, decays smoothly. width controls falloff.
    return 1.0 - smoothstep(0.0, width, abs(v));
}

// Combined crack intensity in [0,1]. `p` must be in VOXEL-space (world_pos
// divided by voxel size) so the pattern is scale-invariant. Three ridges of
// decreasing frequency (~1-2 crossings per voxel face) layered for a
// branching feel, plus a per-voxel-cell hash gate so not every voxel cracks.
fn crack_factor(p: vec3f) -> f32 {
    let cell = vec3<i32>(floor(p));
    let gate = hash13(cell.x, cell.y, cell.z);
    // Only ~60% of voxels get cracks; the rest stay clean.
    if (gate < 0.4) {
        return 0.0;
    }
    let r1 = crack_ridge(p, 2.2, 0.20, gate * 6.28);
    let r2 = crack_ridge(p, 3.7, 0.14, gate * 12.4);
    let r3 = crack_ridge(p, 5.5, 0.10, gate * 3.7);
    // Sharpen: cracks are thin, so take a max-ish combination but keep the
    // strongest ridge dominant to avoid a flat noisy overlay.
    let m = max(r1, max(r2, r3));
    return clamp(m, 0.0, 1.0);
}

// --- Shadow mapping (#14) ---
// PCF 3x3 sampling of the directional shadow map. Returns a visibility
// factor in [0,1]: 1.0 = fully lit, 0.0 = fully shadowed. The comparison
// sampler (sampler_comparison with LessEqual) returns 1.0 when the
// fragment's depth is <= the stored depth, i.e. the fragment is closer to
// the light and thus lit.
//
// A constant receiver bias (in clip-space depth units) is subtracted from
// the fragment depth before the comparison to fight shadow acne on
// surfaces that face the sun nearly head-on. The shadow pipeline also
// applies a constant + slope-scaled depth bias on the *writer* side; the
// receiver bias here is the second line of defense.
fn shadow_visibility(world_pos: vec3f) -> f32 {
    let clip = shadow_cam.view_proj * vec4f(world_pos, 1.0);
    // Outside the shadow camera's near/far or clip box: treat as lit so we
    // don't black out terrain beyond the 100 m shadow extent.
    if (clip.w <= 0.0) {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    // NDC outside [-1,1]: beyond the orthographic box -- lit.
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z > 1.0 || ndc.z < -1.0) {
        return 1.0;
    }
    // Convert to shadow-map UV (flip Y: WGSL texture coords have origin at
    // the top-left, NDC y=+1 is the top of the viewport).
    let uv = vec2f(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
    // Depth the fragment *would* write to the shadow map, biased to avoid
    // self-shadowing (acne). 0.002 was tuned for a 100 m ortho box at
    // 2048x2048 with the writer-side bias of constant 2 / slope 1.5.
    let ref_depth = ndc.z * 0.5 + 0.5 - 0.002;

    // PCF 3x3: sample the comparison sampler at 9 offsets, average the
    // result. texel size is 1/2048.
    let texel = 1.0 / 2048.0;
    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2f(f32(x), f32(y)) * texel;
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + offset, ref_depth);
        }
    }
    return sum / 9.0;
}

@vertex
fn vs(v: VIn, inst: Inst) -> VOut {
    let model = mat4x4f(inst.m0, inst.m1, inst.m2, inst.m3);
    let local = vec3f(f32(v.pos_ao.x), f32(v.pos_ao.y), f32(v.pos_ao.z)) * cam.fog.z;
    var wp = (model * vec4f(local, 1.0)).xyz;

    let mat_id = v.norm_mat.z | (v.norm_mat.w << 8u);

    let base = palette[mat_id];
    // Jitter is baked into the mesh once at build time (see vox-mesh's
    // `jitter_hash`), not recomputed here from world position: hashing a
    // *moving* vertex's world position dynamically made the jitter shift
    // continuously as a debris body translated/rotated, which read as
    // flicker on its surface -- chunks never move, so they never showed it,
    // matching the exact "only on detached bodies" symptom this fixed.
    let h = f32(v.norm_mat.y) / 255.0;

    var out: VOut;
    out.clip = cam.view_proj * vec4f(wp, 1.0);
    out.color = base.rgb * (1.0 + (h - 0.5) * 2.0 * base.a);
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
    out.mat_id = mat_id;
    return out;
}

// Specialization constant: 0 = opaque pass (skip water), 1 = water pass
// (skip non-water). Two pipelines share this shader; the opaque pipeline
// has depth_write_enabled=true, the water pipeline has it false so terrain
// behind water is not depth-culled.
override water_pass: u32 = 0u;

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    // Pass selection: each pipeline variant only draws its own materials.
    if (water_pass == 0u && in.mat_id == 9u) { discard; }
    if (water_pass == 1u && in.mat_id != 9u) { discard; }

    let n = normalize(in.world_normal);
    let ndotl = dot(n, -cam.sun_dir.xyz);
    // Half-Lambert wrap: softens the sun terminator into a gradient.
    var sun = pow(clamp(ndotl * 0.5 + 0.5, 0.0, 1.0), 1.5) * cam.sun_dir.w * cam.sun_color.xyz;
    let fill = max(-ndotl, 0.0) * cam.sky_color.w;

    // Shadow mapping (#14): sample the directional shadow map and attenuate
    // direct sunlight by ~50% on occluded fragments. At night (sun_strength
    // ~0) the sun term is already zero, so skip the texture fetch entirely.
    // Water (mat 9) is excluded from receiving shadows -- a transparent
    // surface darkening under shadow reads wrong.
    if (cam.sun_dir.w > 0.0 && in.mat_id != 9u) {
        let vis = shadow_visibility(in.world_pos);
        // vis=1 fully lit (sun unchanged); vis=0 fully shadowed (sun halved).
        // Ambient and fill light are untouched, so shadowed terrain dims
        // toward ambient rather than going black.
        sun = sun * mix(0.5, 1.0, vis);
    }

    let hemi_t = clamp(0.5 + 0.5 * n.y, 0.0, 1.0);
    let ambient = mix(cam.ambient_ground.xyz, cam.ambient_sky.xyz, hemi_t) * cam.fog.w;

    let ao = 0.45 + 0.55 * in.ao;
    var c = in.color * (ambient + sun + vec3f(fill)) * ao;

    // Procedural crack decals (#43): dark branching lines on solid voxels,
    // driven by cam.ambient_sky.w. Intensity 0 => no cracks (clean multiply
    // by 0). Water (mat 9) is skipped — cracks belong on solid terrain.
    // Applied to lit color before fog so distant cracked voxels still fog.
    if (cam.ambient_sky.w > 0.0 && in.mat_id != 9u) {
        let k = crack_factor(in.world_pos / cam.fog.z) * cam.ambient_sky.w;
        c = mix(c, vec3f(0.05, 0.04, 0.03), clamp(k, 0.0, 1.0) * 0.55);
    }

    let dist = length(in.world_pos - cam.cam_pos.xyz);
    let f = clamp((dist - cam.fog.x) / (cam.fog.y - cam.fog.x), 0.0, 1.0);
    c = mix(c, cam.sky_color.xyz, f * f);
    // Water (material ID 9): semi-transparent with a subtle refraction ripple
    // that perturbs the fog mix slightly via a sin wave on world XZ + time.
    let alpha = select(1.0, 0.85, in.mat_id == 9u);
    if (in.mat_id == 9u) {
        let t = cam.sun_color.w;
        let ripple = sin(in.world_pos.x * 3.0 + t * 2.0) * 0.5 + sin(in.world_pos.z * 2.3 + t * 1.7) * 0.5;
        c = mix(c, cam.sky_color.xyz, clamp(f * f + ripple * 0.04, 0.0, 1.0));
        // Slight blue tint for water
        c = mix(c, vec3f(0.10, 0.25, 0.45), 0.25);
    }

    return vec4f(c, alpha);
}
