//! Grass blade pipeline: renders thin 3D blades standing up from grass
//! voxels. Each blade is a 2-triangle quad with wind sway in the vertex
//! shader. Blades are regenerated each frame from nearby grass-top voxels.
//!
//! The app scans chunks near the camera for grass-top faces and generates
//! blade vertices (4 per blade, 3-5 blades per voxel). The vertex buffer
//! is rewritten each frame — like the particle system, not the chunk mesher.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::gpu::{DEPTH_FORMAT, Gpu};

/// Max grass blades. 100k blades = 400k vertices = ~5MB. Plenty for
/// nearby terrain; blades beyond ~60m aren't generated.
pub const MAX_GRASS_BLADES: usize = 100_000;

/// One grass blade vertex: position + color gradient factor.
/// 4 vertices per blade (base-left, base-right, tip-left, tip-right).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct GrassVertex {
    /// World-space position of this vertex (meters).
    pub position: [f32; 3],
    /// 0.0 = base (dark green), 1.0 = tip (bright green, sways with wind).
    pub height_factor: f32,
}
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GrassCameraUniform {
    view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    /// xyz = sun direction, w = sun strength
    sun_dir: [f32; 4],
    /// x = fog start, y = fog end, z = voxel size, w = ambient strength
    fog: [f32; 4],
    /// xyz = sky/fog color, w = fill strength
    sky_color: [f32; 4],
    /// xyz = sun color, w = game time
    sun_color: [f32; 4],
    /// xyz = ambient sky tint, w = unused
    ambient_sky: [f32; 4],
    /// xyz = ambient ground tint, w = unused
    ambient_ground: [f32; 4],
}

/// Pipeline + persistent vertex buffer for grass blade rendering.
pub struct GrassPipeline {
    pipeline: wgpu::RenderPipeline,
    camera_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_buf: wgpu::Buffer,
}

impl GrassPipeline {
    pub fn new(gpu: &Gpu, shader_source: &str) -> Self {
        let device = gpu.device();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grass-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grass camera uniform"),
            size: std::mem::size_of::<GrassCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grass bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grass bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(camera_buf.as_entire_buffer_binding()),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grass layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GrassVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    shader_location: 0,
                    offset: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    shader_location: 1,
                    offset: 12,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("grass pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[vertex_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.surface_format(),
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // blades are thin, culling would hide them
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false, // don't write depth — blades are thin
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grass vertex buffer"),
            size: (MAX_GRASS_BLADES * 6 * std::mem::size_of::<GrassVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            camera_buf,
            bind_group,
            vertex_buf,
        }
    }

    /// Update the camera uniform with day/night lighting params.
    pub fn write_camera(
        &self,
        queue: &wgpu::Queue,
        view_proj: [[f32; 4]; 4],
        cam_pos: glam::Vec3,
        sun_dir: glam::Vec3,
        sun_strength: f32,
        sky_color: glam::Vec3,
        fill_strength: f32,
        ambient_strength: f32,
        sun_color: glam::Vec3,
        ambient_sky: glam::Vec3,
        ambient_ground: glam::Vec3,
        game_time: f32,
        voxel_size: f32,
        fog_end: f32,
    ) {
        let uniform = GrassCameraUniform {
            view_proj,
            cam_pos: [cam_pos.x, cam_pos.y, cam_pos.z, 1.0],
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, sun_strength],
            fog: [fog_end * 0.55, fog_end, voxel_size, ambient_strength],
            sky_color: [sky_color.x, sky_color.y, sky_color.z, fill_strength],
            sun_color: [sun_color.x, sun_color.y, sun_color.z, game_time],
            ambient_sky: [ambient_sky.x, ambient_sky.y, ambient_sky.z, 0.0],
            ambient_ground: [ambient_ground.x, ambient_ground.y, ambient_ground.z, 0.0],
        };
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&uniform));
    }

    /// Upload grass blade vertices and draw them. `vertices` is a flat slice
    /// of GrassVertex — 4 per blade, forming a quad. The caller generates
    /// the blade geometry from nearby grass voxels.
    pub fn draw<'p>(
        &'p self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'p>,
        vertices: &[GrassVertex],
    ) {
        if vertices.is_empty() {
            return;
        }
        let byte_len = (vertices.len() * std::mem::size_of::<GrassVertex>()).min(
            MAX_GRASS_BLADES * 6 * std::mem::size_of::<GrassVertex>(),
        );
        let count = (byte_len / std::mem::size_of::<GrassVertex>()) as u32;
        queue.write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(&vertices[..count as usize]));

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.draw(0..count, 0..1);
    }
}
