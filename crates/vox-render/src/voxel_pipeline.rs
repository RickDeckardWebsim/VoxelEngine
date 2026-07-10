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
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub cam_pos: [f32; 4],
    pub sun_dir: [f32; 4],
    /// x = fog start (m), y = fog end (m), z = voxel size (m), w = unused.
    pub fog: [f32; 4],
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
    camera_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    chunks: vox_core::FxHashMap<IVec3, GpuMesh>,
    bodies: vox_core::FxHashMap<BodyMeshKey, GpuBodyMesh>,
    voxel_size_m: f32,
}

impl VoxelPipeline {
    /// Build the pipeline. `shader_source` is the WGSL text (the app owns
    /// asset loading); the palette is baked from the material registry.
    pub fn new(
        gpu: &Gpu,
        shader_source: &str,
        registry: &MaterialRegistry,
        voxel_size_m: f32,
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
            label: Some("camera uniform"),
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

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel pipeline layout"),
            bind_group_layouts: &[&bgl],
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

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("voxel pipeline"),
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
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: crate::postprocess::COLOR_FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: crate::postprocess::NORMAL_FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: crate::postprocess::DEPTH_COPY_FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
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

        Self {
            pipeline,
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
        self.chunks.insert(
            key,
            GpuMesh {
                vertices,
                indices,
                index_count: mesh.indices.len() as u32,
                instance,
                aabb_min: origin_m,
                aabb_max: origin_m + Vec3::splat(chunk_extent_m),
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
    pub fn write_camera(&self, gpu: &Gpu, view_proj: Mat4, cam_pos: Vec3, fog_end_m: f32) {
        let sun = Vec3::new(-0.45, -0.8, -0.35).normalize();
        let uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            cam_pos: Vec4::from((cam_pos, 1.0)).to_array(),
            sun_dir: Vec4::from((sun, 0.0)).to_array(),
            fog: [fog_end_m * 0.55, fog_end_m, self.voxel_size_m, 0.0],
        };
        gpu.queue()
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&uniform));
    }

    /// Record all visible chunk draws into `pass`. Returns draw statistics.
    pub fn draw_chunks<'p>(
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

    /// Draw every resident debris body whose world-space bounds (as of the
    /// last `update_body_transform` call) are inside `frustum` -- same
    /// culling `draw_chunks` already does, now applied to debris too now
    /// that a single bomb can scatter dozens of small bodies at once.
    /// Assumes the pipeline/bind group are already bound (call after
    /// `draw_chunks` in the same pass, or call `bind` first).
    pub fn draw_bodies<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>, frustum: &Frustum) -> DrawStats {
        let mut stats = DrawStats::default();
        for mesh in self.bodies.values() {
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

    /// Bind the pipeline and its bind group (needed if drawing bodies without
    /// having first called `draw_chunks` in this pass).
    pub fn bind<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
    }
}
