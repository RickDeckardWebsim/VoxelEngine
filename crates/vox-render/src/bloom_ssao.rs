//! Bloom + SSAO post-processing pipeline.
//!
//! Generates a screen-space ambient-occlusion buffer from the scene depth
//! (SSAO generation + 3×3 box blur) and a bloom buffer from the scene HDR
//! color (bright-pass extraction + separable Gaussian blur). Both run at
//! half resolution. The outputs (`ao_view`, `bloom_view`) are consumed by
//! `PostProcessPipeline`'s final composite pass.
//!
//! The scene depth and color textures are owned by `PostProcessPipeline`;
//! `process()` receives them per-frame and creates bind groups on the fly
//! (cheap — the heavy resources are pre-created in `new`).

use wgpu::util::DeviceExt;

use crate::gpu::Gpu;

/// Half-res AO buffer format (single channel, high precision for the
/// 0..1 occlusion factor).
const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
/// Half-res bloom buffer format (HDR RGB + alpha).
const BLOOM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Number of SSAO hemisphere samples (must match the kernel buffer).
const SSAO_KERNEL_SIZE: u32 = 32;

/// Rust mirror of the `SsaoParams` WGSL uniform struct.
///
/// WGSL uniform layout (16-byte struct alignment):
/// ```text
/// inv_view_proj  mat4x4f  64
/// view_proj      mat4x4f  64
/// resolution     vec2f     8
/// texel_size     vec2f     8
/// radius         f32       4
/// intensity      f32       4
/// bias           f32       4
/// kernel_size    u32       4
/// _pad           f32       4
/// _pad2          f32       4
/// _pad3          f32       4
/// _pad4          f32       4   (implicit WGSL struct-size padding → 176)
/// ```
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoParamsUniform {
    inv_view_proj: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    resolution: [f32; 2],
    texel_size: [f32; 2],
    radius: f32,
    intensity: f32,
    bias: f32,
    kernel_size: u32,
    _pad: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
}

/// Rust mirror of the `BloomParams` WGSL uniform struct.
///
/// WGSL uniform layout:
/// ```text
/// resolution   vec2f   8
/// texel_size   vec2f   8
/// threshold    f32     4
/// knee         f32     4
/// intensity    f32     4
/// _pad         f32     4
/// _pad2        f32     4
/// _pad3        f32     4
/// _pad4        f32     4   (implicit WGSL struct-size padding → 48)
/// _pad5        f32     4
/// ```
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomParamsUniform {
    resolution: [f32; 2],
    texel_size: [f32; 2],
    threshold: f32,
    knee: f32,
    intensity: f32,
    _pad: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

/// Owns the SSAO + bloom render pipelines, intermediate textures, and
/// parameter/kernel buffers. Call [`write_params`](Self::write_params)
/// once per frame to upload camera matrices and tunables, then
/// [`process`](Self::process) to run the five fullscreen passes.
pub struct BloomSsaoPipeline {
    // SSAO
    ssao_pipeline: wgpu::RenderPipeline,
    ssao_blur_pipeline: wgpu::RenderPipeline,
    ao_tex: wgpu::TextureView,
    ao_blur_tex: wgpu::TextureView,
    ssao_params_buf: wgpu::Buffer,
    ssao_kernel_buf: wgpu::Buffer,
    // Bind group layouts (kept so process() can create per-frame bind groups).
    ssao_bind_layout: wgpu::BindGroupLayout,
    ssao_blur_bind_layout: wgpu::BindGroupLayout,

    // Bloom
    bright_pipeline: wgpu::RenderPipeline,
    blur_h_pipeline: wgpu::RenderPipeline,
    blur_v_pipeline: wgpu::RenderPipeline,
    bright_tex: wgpu::TextureView,
    bloom_tex: wgpu::TextureView,
    bloom_params_buf: wgpu::Buffer,
    bright_bind_layout: wgpu::BindGroupLayout,
    blur_bind_layout: wgpu::BindGroupLayout,

    // Shared
    sampler: wgpu::Sampler,
    width: u32,
    height: u32,
    half_w: u32,
    half_h: u32,
}

impl BloomSsaoPipeline {
    /// Create the pipeline and all intermediate resources at the given
    /// **full** resolution; internal textures are half-res.
    pub fn new(gpu: &Gpu, ssao_shader: &str, bloom_shader: &str, width: u32, height: u32) -> Self {
        let device = gpu.device();
        let half_w = (width / 2).max(1);
        let half_h = (height / 2).max(1);

        // --- Intermediate textures (half-res) ---
        let ao_tex = create_color_texture(device, half_w, half_h, AO_FORMAT, "ssao-ao");
        let ao_blur_tex = create_color_texture(device, half_w, half_h, AO_FORMAT, "ssao-ao-blur");
        let bright_tex = create_color_texture(device, half_w, half_h, BLOOM_FORMAT, "bloom-bright");
        let bloom_tex = create_color_texture(device, half_w, half_h, BLOOM_FORMAT, "bloom-blur");

        // --- Sampler (linear, clamp-to-edge) ---
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom-ssao-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // --- SSAO kernel (32 hemisphere samples, CPU-generated) ---
        let kernel = generate_ssao_kernel();
        let ssao_kernel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssao-kernel"),
            contents: bytemuck::cast_slice(&kernel),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // --- Params uniform buffers (identity matrices to start) ---
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let ssao_params = SsaoParamsUniform {
            inv_view_proj: identity,
            view_proj: identity,
            resolution: [half_w as f32, half_h as f32],
            texel_size: [1.0 / half_w as f32, 1.0 / half_h as f32],
            radius: 0.5,
            intensity: 1.0,
            bias: 0.025,
            kernel_size: SSAO_KERNEL_SIZE,
            _pad: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,
        };
        let ssao_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssao-params"),
            contents: bytemuck::bytes_of(&ssao_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bloom_params = BloomParamsUniform {
            resolution: [half_w as f32, half_h as f32],
            texel_size: [1.0 / half_w as f32, 1.0 / half_h as f32],
            threshold: 1.0,
            knee: 0.2,
            intensity: 0.8,
            _pad: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
        };
        let bloom_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bloom-params"),
            contents: bytemuck::bytes_of(&bloom_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Shaders ---
        let ssao_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao-shader"),
            source: wgpu::ShaderSource::Wgsl(ssao_shader.into()),
        });
        let bloom_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom-shader"),
            source: wgpu::ShaderSource::Wgsl(bloom_shader.into()),
        });

        // --- Bind group layouts ---
        // SSAO generation: params(0) + depth_tex(1) + sampler(2) + kernel(3)
        let ssao_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao-gen-layout"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // SSAO blur: params(0) + sampler(2) + ao_tex(4)
        let ssao_blur_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao-blur-layout"),
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
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Bloom bright-pass and blur share the same binding shape:
        // params(0) + input_tex(1) + sampler(2). We create two identical
        // layouts (wgpu BindGroupLayout is not Clone) so the struct can own
        // one per pipeline.
        let bright_bind_layout = create_bloom_bind_layout(device, "bloom-bright-layout");
        let blur_bind_layout = create_bloom_bind_layout(device, "bloom-blur-layout");

        // --- Pipelines ---
        let ssao_pipeline = create_fullscreen_pipeline(
            device,
            "ssao-pipeline",
            &ssao_module,
            "vs",
            "fs_ssao",
            &[&ssao_bind_layout],
            AO_FORMAT,
            Default::default(),
        );
        let ssao_blur_pipeline = create_fullscreen_pipeline(
            device,
            "ssao-blur-pipeline",
            &ssao_module,
            "vs",
            "fs_blur",
            &[&ssao_blur_bind_layout],
            AO_FORMAT,
            Default::default(),
        );
        let bright_pipeline = create_fullscreen_pipeline(
            device,
            "bloom-bright-pipeline",
            &bloom_module,
            "vs",
            "fs_bright",
            &[&bright_bind_layout],
            BLOOM_FORMAT,
            Default::default(),
        );
        let blur_h_pipeline = create_fullscreen_pipeline(
            device,
            "bloom-blur-h-pipeline",
            &bloom_module,
            "vs",
            "fs_blur",
            &[&blur_bind_layout],
            BLOOM_FORMAT,
            wgpu::PipelineCompilationOptions {
                constants: &std::collections::HashMap::from([("blur_direction".into(), 0.0_f64)]),
                ..Default::default()
            },
        );
        let blur_v_pipeline = create_fullscreen_pipeline(
            device,
            "bloom-blur-v-pipeline",
            &bloom_module,
            "vs",
            "fs_blur",
            &[&blur_bind_layout],
            BLOOM_FORMAT,
            wgpu::PipelineCompilationOptions {
                constants: &std::collections::HashMap::from([("blur_direction".into(), 1.0_f64)]),
                ..Default::default()
            },
        );

        Self {
            ssao_pipeline,
            ssao_blur_pipeline,
            ao_tex,
            ao_blur_tex,
            ssao_params_buf,
            ssao_kernel_buf,
            ssao_bind_layout,
            ssao_blur_bind_layout,

            bright_pipeline,
            blur_h_pipeline,
            blur_v_pipeline,
            bright_tex,
            bloom_tex,
            bloom_params_buf,
            bright_bind_layout,
            blur_bind_layout,

            sampler,
            width,
            height,
            half_w,
            half_h,
        }
    }

    /// Run all five passes: SSAO generation, SSAO blur, bloom bright-pass,
    /// bloom blur-H, bloom blur-V (ping-pong).
    ///
    /// `color_view` is the scene HDR color (full-res, owned by
    /// `PostProcessPipeline`); `depth_view` is the scene depth (full-res).
    /// Bind groups are created per-frame — cheap because the heavy
    /// resources (buffers, textures) are pre-created.
    pub fn process(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        // 1. SSAO generation: read depth → ao_tex
        let ssao_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao-gen-bg"),
            layout: &self.ssao_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.ssao_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.ssao_kernel_buf.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ao_tex,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.ssao_pipeline);
            pass.set_bind_group(0, &ssao_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 2. SSAO blur: read ao_tex → ao_blur_tex
        let blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao-blur-bg"),
            layout: &self.ssao_blur_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.ssao_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.ao_tex),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao-blur-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ao_blur_tex,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.ssao_blur_pipeline);
            pass.set_bind_group(0, &blur_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 3. Bloom bright-pass: read color → bright_tex
        let bright_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom-bright-bg"),
            layout: &self.bright_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.bloom_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom-bright-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bright_tex,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.bright_pipeline);
            pass.set_bind_group(0, &bright_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 4. Bloom blur-H: read bright_tex → bloom_tex
        let blur_h_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom-blur-h-bg"),
            layout: &self.blur_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.bloom_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.bright_tex),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom-blur-h-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_tex,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.blur_h_pipeline);
            pass.set_bind_group(0, &blur_h_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 5. Bloom blur-V: read bloom_tex → bright_tex (ping-pong)
        let blur_v_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom-blur-v-bg"),
            layout: &self.blur_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.bloom_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_tex),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom-blur-v-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bright_tex,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.blur_v_pipeline);
            pass.set_bind_group(0, &blur_v_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Upload camera matrices and tunables for this frame.
    pub fn write_params(
        &self,
        queue: &wgpu::Queue,
        view_proj: [[f32; 4]; 4],
        inv_view_proj: [[f32; 4]; 4],
        ssao_intensity: f32,
        ssao_radius: f32,
        bloom_intensity: f32,
        bloom_threshold: f32,
    ) {
        let ssao_params = SsaoParamsUniform {
            inv_view_proj,
            view_proj,
            resolution: [self.half_w as f32, self.half_h as f32],
            texel_size: [1.0 / self.half_w.max(1) as f32, 1.0 / self.half_h.max(1) as f32],
            radius: ssao_radius,
            intensity: ssao_intensity,
            bias: 0.025,
            kernel_size: SSAO_KERNEL_SIZE,
            _pad: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,
        };
        queue.write_buffer(&self.ssao_params_buf, 0, bytemuck::bytes_of(&ssao_params));

        let bloom_params = BloomParamsUniform {
            resolution: [self.half_w as f32, self.half_h as f32],
            texel_size: [1.0 / self.half_w.max(1) as f32, 1.0 / self.half_h.max(1) as f32],
            threshold: bloom_threshold,
            knee: 0.2,
            intensity: bloom_intensity,
            _pad: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
        };
        queue.write_buffer(&self.bloom_params_buf, 0, bytemuck::bytes_of(&bloom_params));
    }

    /// Recreate intermediate textures at the new size. Call when the
    /// window resizes.
    pub fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        self.half_w = (width / 2).max(1);
        self.half_h = (height / 2).max(1);
        let device = gpu.device();

        self.ao_tex = create_color_texture(device, self.half_w, self.half_h, AO_FORMAT, "ssao-ao");
        self.ao_blur_tex =
            create_color_texture(device, self.half_w, self.half_h, AO_FORMAT, "ssao-ao-blur");
        self.bright_tex =
            create_color_texture(device, self.half_w, self.half_h, BLOOM_FORMAT, "bloom-bright");
        self.bloom_tex =
            create_color_texture(device, self.half_w, self.half_h, BLOOM_FORMAT, "bloom-blur");

        // Update params with new resolution/texel_size (matrices stay as-is;
        // the next write_params call refreshes them).
        self.write_params(
            gpu.queue(),
            identity_matrix(),
            identity_matrix(),
            1.0,
            0.5,
            0.8,
            1.0,
        );
    }

    /// Final blurred AO factor (half-res R16Float). 1.0 = no occlusion,
    /// 0.0 = fully occluded.
    pub fn ao_view(&self) -> &wgpu::TextureView {
        &self.ao_blur_tex
    }

    /// Final blurred bloom buffer (half-res Rgba16Float). After
    /// `process()`, `bright_tex` holds the blur-V output (ping-pong end).
    pub fn bloom_view(&self) -> &wgpu::TextureView {
        &self.bright_tex
    }
}

// --- Helpers ---

/// Create a half-res render-target texture view.
fn create_color_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Create the bloom bind group layout: params(uniform, 0) +
/// input_tex(texture, 1) + sampler(2). Used by both the bright-pass and
/// blur pipelines.
fn create_bloom_bind_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
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
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Build a fullscreen-triangle render pipeline (no vertex buffer, 3 verts).
fn create_fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    vs_entry: &str,
    fs_entry: &str,
    bg_layouts: &[&wgpu::BindGroupLayout],
    target_format: wgpu::TextureFormat,
    fragment_compilation_options: wgpu::PipelineCompilationOptions,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bloom-ssao-pipeline-layout"),
        bind_group_layouts: bg_layouts,
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: vs_entry,
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: fs_entry,
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: fragment_compilation_options,
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

/// Generate 32 hemisphere samples for SSAO. Each sample is a vec4f:
/// `xyz` = unit direction, `w` = scale (0.1 → 1.0, squared for closer-weight).
fn generate_ssao_kernel() -> [[f32; 4]; SSAO_KERNEL_SIZE as usize] {
    // Deterministic pseudo-random generator (same kernel every run —
    // SSAO is stable; per-pixel rotation comes from the depth-derived
    // normal in the shader).
    let mut seed: u32 = 0x9E3779B9; // golden ratio fractional part
    let mut rand_f32 = || {
        // xorshift32 → [0,1)
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32) / (u32::MAX as f32)
    };

    let mut kernel = [[0.0_f32; 4]; SSAO_KERNEL_SIZE as usize];
    for i in 0..SSAO_KERNEL_SIZE as usize {
        // Random direction in tangent-space hemisphere.
        let mut dir = [
            rand_f32() * 2.0 - 1.0,
            rand_f32() * 2.0 - 1.0,
            rand_f32(), // hemisphere: z in [0,1]
        ];
        // Normalize.
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if len > 0.0 {
            dir[0] /= len;
            dir[1] /= len;
            dir[2] /= len;
        }
        // Scale: lerp 0.1 → 1.0, squared (closer samples weigh more).
        let scale = 0.1 + (rand_f32() * 0.9);
        let scale = scale * scale;
        kernel[i] = [dir[0], dir[1], dir[2], scale];
    }
    kernel
}

fn identity_matrix() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
