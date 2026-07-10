//! The opaque voxel render pipeline and per-region GPU mesh store.
//!
//! One pipeline draws both world chunks and (later) debris bodies: every draw
//! is an 8-byte-vertex mesh plus a per-draw model matrix supplied through a
//! one-instance vertex buffer.


use glam::{IVec3, Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

use vox_core::{MaterialRegistry, consts::CHUNK_SIZE};
use vox_mesh::{MeshData, VoxelVertex};

use crate::frustum::Frustum;
use crate::gpu::{DEPTH_FORMAT, Gpu};

/// Camera + environment uniform, must match `voxel.wgsl`.
/// All lighting is driven by uniforms for day/night cycle support.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub cam_pos: [f32; 4],
    /// xyz = sun direction (unit), w = sun strength.
    pub sun_dir: [f32; 4],
    /// x = fog start (m), y = fog end (m), z = voxel size (m), w = ambient strength.
    pub fog: [f32; 4],
    /// xyz = sky/fog color (linear RGB), w = fill light strength.
    pub sky_color: [f32; 4],
    /// xyz = sun color (linear RGB), w = unused.
    pub sun_color: [f32; 4],
    /// xyz = ambient sky tint, w = crack decal intensity (0 = off, >0 = procedural cracks visible).
    pub ambient_sky: [f32; 4],
    /// xyz = ambient ground tint, w = unused.
    pub ambient_ground: [f32; 4],
}

/// One uploaded mesh: vertex/index buffers plus its model transform and
/// world-space bounds for culling.
struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    instance: wgpu::Buffer,
    aabb_min: Vec3,
    aabb_max: Vec3,
    /// Whether this chunk's mesh contains any water (material ID 9) voxels.
    /// Set at upload time so the water pass can skip chunks without water.
    has_water: bool,
}

/// Stats for the debug HUD.
#[derive(Copy, Clone, Default, Debug)]
pub struct DrawStats {
    pub drawn: u32,
    pub culled: u32,
}

/// A debris body's mesh: geometry is static (meshed once at spawn), the
/// instance transform (and world-space bounds, for culling) is rewritten
/// every frame via `update_body_transform`.
struct GpuBodyMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    instance: wgpu::Buffer,
    aabb_min: Vec3,
    aabb_max: Vec3,
}

/// Identifies a debris body's GPU mesh; callers use their physics body
/// handle's (slot, generation) pair so a despawned-and-reused slot can never
/// collide with a stale mesh.
pub type BodyMeshKey = (u32, u32);

/// The voxel pipeline: shader, bind group (camera + palette), and the chunk
/// mesh store.
pub struct VoxelPipeline {
    pipeline: wgpu::RenderPipeline,
    /// Water-only pipeline variant: alpha blending, depth_write enabled,
    /// specialization constant water_pass=1 so only mat_id==9 fragments
    /// survive. Draws after the opaque pass so terrain behind water is
    /// already in the depth buffer and shows through the alpha blend.
    water_pipeline: wgpu::RenderPipeline,
    camera_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    chunks: vox_core::FxHashMap<IVec3, GpuMesh>,
    bodies: vox_core::FxHashMap<BodyMeshKey, GpuBodyMesh>,
    voxel_size_m: f32,
}

impl VoxelPipeline {
    /// Build the pipeline. `shader_source` is the WGSL text (the app owns
    /// asset loading); the palette is baked from the material registry.
    /// `shadow_sample_bgl` is the bind group layout (group 1) the fragment
    /// shader uses to sample the shadow map; pass `None` only in tests that
    /// never run a render pass.
    pub fn new(
        gpu: &Gpu,
        shader_source: &str,
        registry: &MaterialRegistry,
        voxel_size_m: f32,
        shadow_sample_bgl: Option<&wgpu::BindGroupLayout>,
    ) -> Self {
        let device = gpu.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Palette: rgb + jitter per material id.
        let palette: Vec<[f32; 4]> = registry
            .iter()
            .map(|(_, def)| [def.color[0], def.color[1], def.color[2], def.jitter])
            .collect();
        let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material palette"),
            contents: bytemuck::cast_slice(&palette),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel camera uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel bind group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: palette_buf.as_entire_binding(),
                },
            ],
        });

        // Group 1: shadow map sampling (uniform view-proj + texture +
        // comparison sampler). Optional so headless test fixtures can build
        // a pipeline without a shadow map.
        let bgl_refs: Vec<&wgpu::BindGroupLayout> = match shadow_sample_bgl {
            Some(shadow) => vec![&bgl, shadow],
            None => vec![&bgl],
        };
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel pipeline layout"),
            bind_group_layouts: &bgl_refs,
            push_constant_ranges: &[],
        });

        // Vertex: two Uint8x4 attributes over the 8-byte VoxelVertex.
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VoxelVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint8x4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint8x4,
                    offset: 4,
                    shader_location: 1,
                },
            ],
        };
        // Instance: one mat4 per draw.
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: 64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 4,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 5,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 6,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 48,
                    shader_location: 7,
                },
            ],
        };

        let opaque_constants = wgpu::PipelineCompilationOptions {
            constants: &std::collections::HashMap::from([("water_pass".into(), 0.0_f64)]),
            ..Default::default()
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("voxel pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[vertex_layout.clone(), instance_layout.clone()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs",
                compilation_options: opaque_constants,
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.surface_format(),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
        });

        // Water pipeline: same shader/layout, but specialization constant
        // water_pass=1 (only mat_id==9 fragments survive), alpha blending.
        // Depth writes enabled so void behind water is properly occluded —
        // terrain (drawn in opaque pass) still shows through the alpha blend.
        let water_constants = wgpu::PipelineCompilationOptions {
            constants: &std::collections::HashMap::from([("water_pass".into(), 1.0_f64)]),
            ..Default::default()
        };
        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("voxel water pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[vertex_layout, instance_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs",
                compilation_options: water_constants,
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.surface_format(),
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: Default::default(),
            multiview: None,
        });

        Self {
            pipeline,
            water_pipeline,
            camera_buf,
            bind_group,
            chunks: Default::default(),
            bodies: Default::default(),
            voxel_size_m,
        }
    }

    /// Upload (or replace) the mesh for a chunk. An empty mesh removes it.
    pub fn upload_chunk(&mut self, gpu: &Gpu, key: IVec3, mesh: &MeshData) {
        if mesh.is_empty() {
            self.chunks.remove(&key);
            return;
        }
        let origin_m = key.as_vec3() * (CHUNK_SIZE as f32) * self.voxel_size_m;
        let chunk_extent_m = CHUNK_SIZE as f32 * self.voxel_size_m;
        let model = Mat4::from_translation(origin_m);

        let device = gpu.device();
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let instance = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk instance"),
            contents: bytemuck::cast_slice(&model.to_cols_array()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let has_water = mesh.vertices.iter().any(|v| v.material == 9);
        self.chunks.insert(
            key,
            GpuMesh {
                vertices,
                indices,
                index_count: mesh.indices.len() as u32,
                instance,
                aabb_min: origin_m,
                aabb_max: origin_m + Vec3::splat(chunk_extent_m),
                has_water,
            },
        );
    }

    /// Remove a chunk's mesh (e.g. the chunk became empty).
    pub fn remove_chunk(&mut self, key: IVec3) {
        self.chunks.remove(&key);
    }

    /// Number of resident chunk meshes.
    pub fn chunk_mesh_count(&self) -> usize {
        self.chunks.len()
    }

    /// Render all visible chunks into the shadow map via `shadow`. Called
    /// inside the shadow render pass (opened by the app with
    /// `shadow.shadow_view()` as its depth attachment). Frustum culling
    /// reuses the main camera frustum -- a chunk outside the view frustum is
    /// also outside the shadow receiver region, so skipping it saves a draw
    /// without losing visible shadows.
    pub fn draw_chunks_shadow<'p>(
        &'p self,
        shadow: &'p ShadowPipeline,
        pass: &mut wgpu::RenderPass<'p>,
        frustum: &Frustum,
    ) {
        shadow.draw_chunks(pass, &self.chunks, frustum);
    }

    /// Upload a debris body's mesh once at spawn. Geometry is in grid-voxel
    /// units (as produced by `mesh_slab` over a `VoxelSlab::from_grid`); the
    /// per-frame transform is written separately via `update_body_transform`.
    pub fn upload_body(&mut self, gpu: &Gpu, key: BodyMeshKey, mesh: &MeshData) {
        if mesh.is_empty() {
            self.bodies.remove(&key);
            return;
        }
        let device = gpu.device();
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("body vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("body indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        // Identity placeholder; the first `update_body_transform` call before
        // any draw overwrites it with the body's real transform.
        let instance = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("body instance"),
            contents: bytemuck::cast_slice(&Mat4::IDENTITY.to_cols_array()),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        self.bodies.insert(
            key,
            GpuBodyMesh {
                vertices,
                indices,
                index_count: mesh.indices.len() as u32,
                instance,
                aabb_min: Vec3::ZERO,
                aabb_max: Vec3::ZERO,
            },
        );
    }

    /// Rewrite a resident body's per-frame model matrix and world-space
    /// bounds (the latter used for frustum culling in `draw_bodies`, the
    /// same way chunks already are). No-op if the body's mesh isn't
    /// uploaded (e.g. this frame's despawn already raced ahead).
    pub fn update_body_transform(
        &mut self,
        gpu: &Gpu,
        key: BodyMeshKey,
        model: Mat4,
        aabb_min: Vec3,
        aabb_max: Vec3,
    ) {
        if let Some(mesh) = self.bodies.get_mut(&key) {
            gpu.queue().write_buffer(
                &mesh.instance,
                0,
                bytemuck::cast_slice(&model.to_cols_array()),
            );
            mesh.aabb_min = aabb_min;
            mesh.aabb_max = aabb_max;
        }
    }

    /// Drop a debris body's mesh (despawned or cleared).
    pub fn remove_body(&mut self, key: BodyMeshKey) {
        self.bodies.remove(&key);
    }

    /// Number of resident debris body meshes.
    pub fn body_mesh_count(&self) -> usize {
        self.bodies.len()
    }
    /// Update the camera/environment uniform for this frame.
    /// All lighting parameters are passed in for day/night cycle support.
    pub fn write_camera(
        &self,
        gpu: &Gpu,
        view_proj: Mat4,
        cam_pos: Vec3,
        fog_end_m: f32,
        sun_dir: Vec3,
        sun_strength: f32,
        sky_color: Vec3,
        fill_strength: f32,
        ambient_strength: f32,
        sun_color: Vec3,
        ambient_sky: Vec3,
        ambient_ground: Vec3,
        crack_intensity: f32,
        game_time: f32,
    ) {
        let uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            cam_pos: Vec4::from((cam_pos, 1.0)).to_array(),
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, sun_strength],
            fog: [fog_end_m * 0.55, fog_end_m, self.voxel_size_m, ambient_strength],
            sky_color: [sky_color.x, sky_color.y, sky_color.z, fill_strength],
            sun_color: [sun_color.x, sun_color.y, sun_color.z, game_time],
            ambient_sky: [ambient_sky.x, ambient_sky.y, ambient_sky.z, crack_intensity],
            ambient_ground: [ambient_ground.x, ambient_ground.y, ambient_ground.z, 0.0],
        };
        gpu.queue()
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&uniform));
    }

    /// Record all visible chunk draws into `pass`. Returns draw statistics.
    ///
    /// Two sub-passes within the same render pass:
    /// 1. Opaque pass — all chunks with the opaque pipeline (water fragments
    ///    are discarded by the specialization constant). Depth writes on.
    /// 2. Water pass — only chunks containing water (material ID 9), drawn
    ///    with the water pipeline (alpha blending, depth writes off) so
    ///    terrain behind water is visible through the semi-transparent
    ///    surface.
    /// Draw opaque chunk geometry (water fragments discarded by shader).
    /// Returns draw/cull stats. Call `draw_water` after grass/particles
    /// for correct translucency layering.
    pub fn draw_chunks_opaque<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        frustum: &Frustum,
    ) -> DrawStats {
        let mut stats = DrawStats::default();
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        for mesh in self.chunks.values() {
            if !frustum.aabb_visible(mesh.aabb_min, mesh.aabb_max) {
                stats.culled += 1;
                continue;
            }
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_vertex_buffer(1, mesh.instance.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            stats.drawn += 1;
        }
        stats
    }

    /// Draw water-only chunk geometry (non-water fragments discarded).
    /// Alpha-blended, depth_write disabled. Call after grass/particles
    /// so they show through the translucent water.
    pub fn draw_water<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        frustum: &Frustum,
    ) {
        pass.set_pipeline(&self.water_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        for mesh in self.chunks.values() {
            if !mesh.has_water {
                continue;
            }
            if !frustum.aabb_visible(mesh.aabb_min, mesh.aabb_max) {
                continue;
            }
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_vertex_buffer(1, mesh.instance.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
    }

    /// Draw all chunk geometry (opaque + water). Convenience method for
    /// callers that don't need grass between the passes.
    pub fn draw_chunks<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        frustum: &Frustum,
    ) -> DrawStats {
        let stats = self.draw_chunks_opaque(pass, frustum);
        self.draw_water(pass, frustum);
        stats
    }

    /// Draw every resident debris body whose world-space bounds (as of the
    /// last `update_body_transform` call) are inside `frustum` -- same
    /// culling `draw_chunks` already does, now applied to debris too now
    /// that a single bomb can scatter dozens of small bodies at once.
    ///
    /// Bodies beyond `max_draw_distance` meters from `cam_pos` are also
    /// culled: small debris chips 200m away contribute nothing to the image
    /// (well past fog end) but still cost a draw call and vertex fetch.
    /// Distance is measured to the closest point of the body's world-space
    /// AABB, so a large body straddling the threshold is never split — it
    /// stays visible while any part of it is within range.
    /// Assumes the pipeline/bind group are already bound (call after
    /// `draw_chunks` in the same pass, or call `bind` first).
    pub fn draw_bodies<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        frustum: &Frustum,
        cam_pos: Vec3,
        max_draw_distance: f32,
    ) -> DrawStats {
        // draw_chunks leaves the water pipeline bound (pass 2); rebind the
        // opaque pipeline so debris bodies render with depth writes on.
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let mut stats = DrawStats::default();
        let max_dist_sq = max_draw_distance * max_draw_distance;
        for mesh in self.bodies.values() {
            if !frustum.aabb_visible(mesh.aabb_min, mesh.aabb_max) {
                stats.culled += 1;
                continue;
            }
            // Distance-based cull: closest point on the world-space AABB to
            // the camera. Using the squared distance avoids a sqrt per body.
            let closest = cam_pos.clamp(mesh.aabb_min, mesh.aabb_max);
            if (closest - cam_pos).length_squared() > max_dist_sq {
                stats.culled += 1;
                continue;
            }
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_vertex_buffer(1, mesh.instance.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            stats.drawn += 1;
        }
        stats
    }

    /// Bind the pipeline and its bind group (needed if drawing bodies without
    /// having first called `draw_chunks` in this pass).
    pub fn bind<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
    }
}

// ---------------------------------------------------------------------------
// Shadow mapping (#14): a single-cascade directional shadow map.
//
// The shadow pass renders all visible chunk geometry into a depth-only
// texture (2048x2048) from an orthographic "sun camera" that follows the
// player. The main voxel pass then binds the resulting depth texture and
// samples it (PCF 3x3) to attenuate direct sunlight by ~50% on fragments
// that fail the depth test -- i.e. are occluded from the sun.
//
// Only chunks are rendered into the shadow map (debris bodies are too many
// draw calls for a first pass and their motion would make shadows shimmer).
// The shadow camera is an orthographic box centered on the player, looking
// back along the sun direction, with a 100 m half-extent so it covers the
// nearby visible terrain. A constant depth bias on the shadow pipeline plus
// a receiver-side bias in the fragment shader keep acne and peter-panning
// in check.
// ---------------------------------------------------------------------------

/// Shadow-map dimensions (texels per side).
const SHADOW_MAP_SIZE: u32 = 2048;
/// Half-extent of the orthographic shadow camera box, in meters.
const SHADOW_HALF_EXTENT: f32 = 100.0;

/// Shadow camera uniform, must match `shadow.wgsl`'s `ShadowCam`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowUniform {
    view_proj: [[f32; 4]; 4],
    /// x = voxel_size_m; y/z/w unused (pad to vec4 alignment).
    params: [f32; 4],
}

/// The directional shadow pipeline: owns the shadow map texture + view, a
/// depth-only render pipeline reusing the chunk vertex data, and a bind
/// group (uniform + sampler + texture) handed to the main pass for sampling.
pub struct ShadowPipeline {
    pipeline: wgpu::RenderPipeline,
    // Retention fields: the texture owns the GPU memory the view references,
    // and the sampler is bound into `sample_bind_group`. Neither is read
    // directly after construction, but both must outlive the views/bind
    // groups that reference them, so they stay in the struct.
    #[allow(dead_code)]
    shadow_map: wgpu::Texture,
    shadow_view: wgpu::TextureView,
    #[allow(dead_code)]
    sampler: wgpu::Sampler,
    uniform_buf: wgpu::Buffer,
    /// Bind group handed to the *main* voxel pass (group 1) so its fragment
    /// shader can sample the shadow map. Bound fresh each frame after
    /// `write_camera` so the uniform always matches this frame's shadow
    /// view-proj.
    sample_bind_group: wgpu::BindGroup,
    /// Bind group layout for `sample_bind_group` -- stored so the main
    /// pipeline layout can reference it.
    sample_bgl: wgpu::BindGroupLayout,
    /// Bind group used by the shadow *render* pass (group 0): just the
    /// shadow camera uniform.
    render_bind_group: wgpu::BindGroup,
}

impl ShadowPipeline {
    /// Build the shadow map texture, sampler, depth-only pipeline, and bind
    /// groups. `shadow_shader_source` is the WGSL text of `shadow.wgsl`.
    pub fn new(gpu: &Gpu, shadow_shader_source: &str, voxel_size_m: f32) -> Self {
        let device = gpu.device();

        let shadow_map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow map"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_map.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow camera uniform"),
            size: std::mem::size_of::<ShadowUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout for the *shadow render* pass (group 0): just the
        // shadow camera uniform.
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow render bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow render bind group"),
            layout: &render_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // Bind group layout for the *main* pass's shadow sampling (group 1):
// the shadow view-proj uniform, the shadow map texture, and a sampler.
        let sample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow sample bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let sample_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow sample bind group"),
            layout: &sample_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow shader"),
            source: wgpu::ShaderSource::Wgsl(shadow_shader_source.into()),
        });

        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow pipeline layout"),
            bind_group_layouts: &[&render_bgl],
            push_constant_ranges: &[],
        });

        // Vertex + instance layouts must match voxel.wgsl exactly so the
        // same chunk buffers bind without modification.
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VoxelVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint8x4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint8x4,
                    offset: 4,
                    shader_location: 1,
                },
            ],
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: 64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 4,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 5,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 6,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 48,
                    shader_location: 7,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow pipeline"),
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[vertex_layout, instance_layout],
            },
            // No fragment: depth-only pass, no color targets.
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                // Keep back-face culling consistent with the main pipeline so
                // the shadow geometry matches what the eye sees.
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                // Constant bias fights acne on near-perpendicular surfaces;
                // slope-scaled bias handles grazing angles. Values tuned for
                // a 100 m ortho box at 2048x2048.
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 1.5,
                    clamp: 0.05,
                },
            }),
            multisample: Default::default(),
            multiview: None,
        });

        // Seed the uniform with a sensible default so the first frame's
        // sample bind group is valid even before write_camera is called.
        let initial = ShadowUniform {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            params: [voxel_size_m, 0.0, 0.0, 0.0],
        };
        gpu.queue()
            .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&initial));

        Self {
            pipeline,
            shadow_map,
            shadow_view,
            sampler,
            uniform_buf,
            sample_bind_group,
            sample_bgl,
            render_bind_group,
        }
    }

    /// The bind group layout the *main* voxel pipeline must include as
    /// group 1 so its fragment shader can sample the shadow map.
    pub fn sample_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.sample_bgl
    }

    /// The bind group the main pass binds at group 1 to sample the shadow
    /// map. Re-bound each frame (the uniform is updated in place, so the
    /// bind group object is stable, but callers bind it every pass).
    pub fn sample_bind_group(&self) -> &wgpu::BindGroup {
        &self.sample_bind_group
    }

    /// Depth attachment view for the shadow render pass.
    pub fn shadow_view(&self) -> &wgpu::TextureView {
        &self.shadow_view
    }

    /// Recompute the shadow camera and upload its view-projection.
    ///
    /// The shadow camera is an orthographic box centered on `focus`
    /// (typically the player position), looking back along the sun
    /// direction (from the sun toward the scene). The box spans
    /// `SHADOW_HALF_EXTENT` meters in every axis so nearby terrain is
    /// covered; far plane is placed well beyond the box depth to avoid
    /// clipping.
    pub fn write_camera(&self, gpu: &Gpu, sun_dir: Vec3, focus: Vec3, voxel_size_m: f32) {
        // `sun_dir` points TOWARD the sun (as day_night defines it; the
        // voxel shader uses `-sun_dir` as the light-travel vector). The
        // shadow camera must look along the light-travel direction, i.e.
        // from the sun toward the scene, so `light_dir = -sun_dir`.
        let dir = sun_dir.normalize_or_zero();
        let light_dir = if dir.length_squared() < 1e-4 {
            // Degenerate fallback: a steep downward angle.
            Vec3::new(-0.4, 0.8, -0.3)
        } else {
            -dir
        };

        // Place the eye "at the sun" (along -light_dir from focus, i.e.
        // opposite the light-travel direction, which is where the sun is)
        // and look toward `focus` along the light-travel direction.
        let eye = focus - light_dir * SHADOW_HALF_EXTENT;
        let up = if light_dir.y.abs() > 0.99 {
            // Near-vertical light: world-up would be parallel to the view
            // direction; use +X as the fallback up axis.
            Vec3::X
        } else {
            Vec3::Y
        };

        let view = Mat4::look_to_rh(eye, light_dir, up);
        // Orthographic box: symmetric extents in x/y, generous z range so
        // everything in the box is captured. Near/far are in view space
        // (positive z is behind the camera in RH look_to, so near=0 and
        // far = 2*extent covers eye +/- extent).
        let proj = Mat4::orthographic_rh(
            -SHADOW_HALF_EXTENT,
            SHADOW_HALF_EXTENT,
            -SHADOW_HALF_EXTENT,
            SHADOW_HALF_EXTENT,
            0.0,
            SHADOW_HALF_EXTENT * 4.0,
        );
        let view_proj = proj * view;

        let uniform = ShadowUniform {
            view_proj: view_proj.to_cols_array_2d(),
            params: [voxel_size_m, 0.0, 0.0, 0.0],
        };
        gpu.queue()
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniform));
    }

    /// Render all visible chunks into the shadow map. The pass must already
    /// be open with `shadow_view` as its depth attachment and no color
    /// attachment; the caller clears depth to 1.0.
    fn draw_chunks<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        chunks: &'p vox_core::FxHashMap<IVec3, GpuMesh>,
        frustum: &Frustum,
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.render_bind_group, &[]);
        for mesh in chunks.values() {
            if !frustum.aabb_visible(mesh.aabb_min, mesh.aabb_max) {
                continue;
            }
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_vertex_buffer(1, mesh.instance.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
    }
}
