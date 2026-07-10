// GPU compute meshing: reads chunk voxel data from a storage buffer,
// performs face culling + AO computation, and emits vertices for each
// exposed face. One workgroup of 4³ processes a 32³ chunk in 8³ groups.
//
// Each thread processes one voxel and emits up to 6 quads (36 vertices).
// Vertices are written to a per-thread region of the output buffer.

const CHUNK_SIZE: u32 = 32u;
const SHELL: u32 = 1u;
const PADDED: u32 = CHUNK_SIZE + 2u * SHELL; // 34
const VERTS_PER_VOXEL: u32 = 36u; // 6 faces × 6 verts (2 tris)
const MAX_VERTICES: u32 = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * VERTS_PER_VOXEL;

struct VoxelData {
    data: array<u32>,
};

struct VertexBuffer {
    vertices: array<u32>, // packed pairs of u32 (pos_ao, norm_mat)
};

struct DrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<storage, read> voxel_data: VoxelData;
@group(0) @binding(1) var<storage, read_write> vertex_buffer: VertexBuffer;
@group(0) @binding(2) var<storage, read_write> draw_args: DrawIndirect;

fn voxel_at(x: u32, y: u32, z: u32) -> u32 {
    return voxel_data.data[x + y * PADDED + z * PADDED * PADDED];
}

fn is_solid(x: u32, y: u32, z: u32) -> bool {
    return voxel_at(x, y, z) != 0u;
}

fn ao(side1: bool, side2: bool, corner: bool) -> u32 {
    if (side1 && side2) { return 0u; }
    return 3u - (u32(side1) + u32(side2) + u32(corner));
}

fn pack_vertex(px: u32, py: u32, pz: u32, ao_val: u32, normal_id: u32, jitter_val: u32, material: u32) -> vec2<u32> {
    let pos_ao = px | (py << 8u) | (pz << 16u) | (ao_val << 24u);
    let norm_mat = normal_id | (jitter_val << 8u) | ((material & 0xFFu) << 16u) | ((material >> 8u) << 24u);
    return vec2<u32>(pos_ao, norm_mat);
}

fn hash(x: u32, y: u32, z: u32) -> u32 {
    var n = x ^ (y * 374761393u) ^ (z * 668265263u);
    n = n * 2246822519u;
    return (n >> 16u) & 255u;
}

// Write a vertex to the thread's region in the output buffer.
fn write_vertex(thread_base: u32, local_idx: u32, v: vec2<u32>) {
    let global_idx = thread_base + local_idx;
    vertex_buffer.vertices[global_idx * 2u + 0u] = v.x;
    vertex_buffer.vertices[global_idx * 2u + 1u] = v.y;
}

// Emit a quad (6 vertices = 2 triangles) for one face.
fn emit_quad(thread_base: u32, counter: ptr<function, u32>,
    v0: vec2<u32>, v1: vec2<u32>, v2: vec2<u32>, v3: vec2<u32>) {
    let c = *counter;
    write_vertex(thread_base, c + 0u, v0);
    write_vertex(thread_base, c + 1u, v1);
    write_vertex(thread_base, c + 2u, v2);
    write_vertex(thread_base, c + 3u, v0);
    write_vertex(thread_base, c + 4u, v2);
    write_vertex(thread_base, c + 5u, v3);
    *counter = c + 6u;
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x + SHELL;
    let y = gid.y + SHELL;
    let z = gid.z + SHELL;

    if (gid.x >= CHUNK_SIZE || gid.y >= CHUNK_SIZE || gid.z >= CHUNK_SIZE) {
        return;
    }

    let mat = voxel_at(x, y, z);
    if (mat == 0u) {
        return;
    }

    let jitter_val = hash(gid.x, gid.y, gid.z);
    let ox = gid.x;
    let oy = gid.y;
    let oz = gid.z;

    // Each thread gets its own region: voxel_index * 36 vertices.
    let thread_base = (gid.x + gid.y * CHUNK_SIZE + gid.z * CHUNK_SIZE * CHUNK_SIZE) * VERTS_PER_VOXEL;
    var vert_count: u32 = 0u;

    // +X face (normal_id = 0)
    if (!is_solid(x + 1u, y, z)) {
        let v0 = pack_vertex(ox+1u, oy,   oz,   ao(is_solid(x+1u,y-1u,z), is_solid(x+1u,y,z-1u), is_solid(x+1u,y-1u,z-1u)), 0u, jitter_val, mat);
        let v1 = pack_vertex(ox+1u, oy+1u,oz,   ao(is_solid(x+1u,y+1u,z), is_solid(x+1u,y,z-1u), is_solid(x+1u,y+1u,z-1u)), 0u, jitter_val, mat);
        let v2 = pack_vertex(ox+1u, oy+1u,oz+1u,ao(is_solid(x+1u,y+1u,z), is_solid(x+1u,y,z+1u), is_solid(x+1u,y+1u,z+1u)), 0u, jitter_val, mat);
        let v3 = pack_vertex(ox+1u, oy,   oz+1u,ao(is_solid(x+1u,y-1u,z), is_solid(x+1u,y,z+1u), is_solid(x+1u,y-1u,z+1u)), 0u, jitter_val, mat);
        emit_quad(thread_base, &vert_count, v0, v1, v2, v3);
    }
    // -X face (normal_id = 1)
    if (!is_solid(x - 1u, y, z)) {
        let v0 = pack_vertex(ox, oy,   oz+1u,ao(is_solid(x-1u,y-1u,z), is_solid(x-1u,y,z+1u), is_solid(x-1u,y-1u,z+1u)), 1u, jitter_val, mat);
        let v1 = pack_vertex(ox, oy+1u,oz+1u,ao(is_solid(x-1u,y+1u,z), is_solid(x-1u,y,z+1u), is_solid(x-1u,y+1u,z+1u)), 1u, jitter_val, mat);
        let v2 = pack_vertex(ox, oy+1u,oz,   ao(is_solid(x-1u,y+1u,z), is_solid(x-1u,y,z-1u), is_solid(x-1u,y+1u,z-1u)), 1u, jitter_val, mat);
        let v3 = pack_vertex(ox, oy,   oz,   ao(is_solid(x-1u,y-1u,z), is_solid(x-1u,y,z-1u), is_solid(x-1u,y-1u,z-1u)), 1u, jitter_val, mat);
        emit_quad(thread_base, &vert_count, v0, v1, v2, v3);
    }
    // +Y face (normal_id = 2)
    if (!is_solid(x, y + 1u, z)) {
        let v0 = pack_vertex(ox,   oy+1u,oz,   ao(is_solid(x-1u,y+1u,z), is_solid(x,y+1u,z-1u), is_solid(x-1u,y+1u,z-1u)), 2u, jitter_val, mat);
        let v1 = pack_vertex(ox+1u,oy+1u,oz,   ao(is_solid(x+1u,y+1u,z), is_solid(x,y+1u,z-1u), is_solid(x+1u,y+1u,z-1u)), 2u, jitter_val, mat);
        let v2 = pack_vertex(ox+1u,oy+1u,oz+1u,ao(is_solid(x+1u,y+1u,z), is_solid(x,y+1u,z+1u), is_solid(x+1u,y+1u,z+1u)), 2u, jitter_val, mat);
        let v3 = pack_vertex(ox,   oy+1u,oz+1u,ao(is_solid(x-1u,y+1u,z), is_solid(x,y+1u,z+1u), is_solid(x-1u,y+1u,z+1u)), 2u, jitter_val, mat);
        emit_quad(thread_base, &vert_count, v0, v1, v2, v3);
    }
    // -Y face (normal_id = 3)
    if (!is_solid(x, y - 1u, z)) {
        let v0 = pack_vertex(ox,   oy, oz+1u,ao(is_solid(x-1u,y-1u,z), is_solid(x,y-1u,z+1u), is_solid(x-1u,y-1u,z+1u)), 3u, jitter_val, mat);
        let v1 = pack_vertex(ox+1u,oy, oz+1u,ao(is_solid(x+1u,y-1u,z), is_solid(x,y-1u,z+1u), is_solid(x+1u,y-1u,z+1u)), 3u, jitter_val, mat);
        let v2 = pack_vertex(ox+1u,oy, oz,   ao(is_solid(x+1u,y-1u,z), is_solid(x,y-1u,z-1u), is_solid(x+1u,y-1u,z-1u)), 3u, jitter_val, mat);
        let v3 = pack_vertex(ox,   oy, oz,   ao(is_solid(x-1u,y-1u,z), is_solid(x,y-1u,z-1u), is_solid(x-1u,y-1u,z-1u)), 3u, jitter_val, mat);
        emit_quad(thread_base, &vert_count, v0, v1, v2, v3);
    }
    // +Z face (normal_id = 4)
    if (!is_solid(x, y, z + 1u)) {
        let v0 = pack_vertex(ox,   oy,   oz+1u,ao(is_solid(x,y-1u,z+1u), is_solid(x+1u,y,z+1u), is_solid(x+1u,y-1u,z+1u)), 4u, jitter_val, mat);
        let v1 = pack_vertex(ox,   oy+1u,oz+1u,ao(is_solid(x,y+1u,z+1u), is_solid(x+1u,y,z+1u), is_solid(x+1u,y+1u,z+1u)), 4u, jitter_val, mat);
        let v2 = pack_vertex(ox+1u,oy+1u,oz+1u,ao(is_solid(x,y+1u,z+1u), is_solid(x-1u,y,z+1u), is_solid(x-1u,y+1u,z+1u)), 4u, jitter_val, mat);
        let v3 = pack_vertex(ox+1u,oy,   oz+1u,ao(is_solid(x,y-1u,z+1u), is_solid(x-1u,y,z+1u), is_solid(x-1u,y-1u,z+1u)), 4u, jitter_val, mat);
        emit_quad(thread_base, &vert_count, v0, v1, v2, v3);
    }
    // -Z face (normal_id = 5)
    if (!is_solid(x, y, z - 1u)) {
        let v0 = pack_vertex(ox+1u,oy,   oz,ao(is_solid(x,y-1u,z-1u), is_solid(x-1u,y,z-1u), is_solid(x-1u,y-1u,z-1u)), 5u, jitter_val, mat);
        let v1 = pack_vertex(ox+1u,oy+1u,oz,ao(is_solid(x,y+1u,z-1u), is_solid(x-1u,y,z-1u), is_solid(x-1u,y+1u,z-1u)), 5u, jitter_val, mat);
        let v2 = pack_vertex(ox,   oy+1u,oz,ao(is_solid(x,y+1u,z-1u), is_solid(x+1u,y,z-1u), is_solid(x+1u,y+1u,z-1u)), 5u, jitter_val, mat);
        let v3 = pack_vertex(ox,   oy,   oz,ao(is_solid(x,y-1u,z-1u), is_solid(x+1u,y,z-1u), is_solid(x+1u,y-1u,z-1u)), 5u, jitter_val, mat);
        emit_quad(thread_base, &vert_count, v0, v1, v2, v3);
    }

    // Fill remaining slots with degenerate vertices (material 0 = air, discarded by shader).
    for (var i = vert_count; i < VERTS_PER_VOXEL; i = i + 1u) {
        write_vertex(thread_base, i, vec2<u32>(0u, 0u));
    }

    // First thread initializes the draw args.
    if (gid.x == 0u && gid.y == 0u && gid.z == 0u) {
        draw_args.vertex_count = MAX_VERTICES;
        draw_args.instance_count = 1u;
        draw_args.first_vertex = 0u;
        draw_args.first_instance = 0u;
    }
}
