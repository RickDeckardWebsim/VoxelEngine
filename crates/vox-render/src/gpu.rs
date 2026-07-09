//! GPU bootstrap: instance, surface, device/queue, surface configuration,
//! depth buffer, and per-frame swapchain texture acquisition.

use crate::RenderError;

/// Depth buffer format used by the main render pass.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Owns the wgpu device, queue, window surface, and depth buffer.
pub struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    adapter_info: wgpu::AdapterInfo,
}

impl Gpu {
    /// Initialize wgpu against a window-like surface target.
    ///
    /// `target` is anything convertible into a `'static` surface target — in
    /// practice `Arc<winit::window::Window>`, passed in by the application
    /// crate (this crate deliberately does not name winit). `width` and
    /// `height` are the window's current inner size in physical pixels.
    ///
    /// Blocks on adapter/device acquisition; call once at startup, before
    /// the event loop runs.
    pub fn new(
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<Self, RenderError> {
        let backends = parse_backend_env();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::default(),
        });
        let surface = instance.create_surface(target)?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or(RenderError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        tracing::info!(
            adapter = %adapter_info.name,
            backend = ?adapter_info.backend,
            driver = %adapter_info.driver,
            "selected GPU adapter"
        );

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("vox-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))?;

        let caps = surface.get_capabilities(&adapter);
        // Prefer an sRGB format so shader outputs in linear space are encoded
        // correctly on present; fall back to whatever the surface offers.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|format| format.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .ok_or(RenderError::NoSurfaceFormat)?;
        tracing::info!(
            format = ?format,
            srgb = format.is_srgb(),
            "surface format selected"
        );

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps
                .alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto),
            view_formats: Vec::new(),
        };
        surface.configure(&device, &config);
        let depth_view = create_depth_view(&device, config.width, config.height);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            depth_view,
            adapter_info,
        })
    }

    /// Resize the surface and depth buffer to the new inner size in physical
    /// pixels. Zero-sized dimensions (minimized window) and no-op size
    /// changes are ignored.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if width == self.config.width && height == self.config.height {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth_view(&self.device, width, height);
    }

    /// Acquire the next swapchain frame.
    ///
    /// On `Lost`/`Outdated` the surface is reconfigured here and an error is
    /// still returned: callers should skip rendering this frame and try
    /// again next frame.
    pub fn begin_frame(&self) -> Result<Frame, RenderError> {
        match self.surface.get_current_texture() {
            Ok(texture) => {
                let view = texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                Ok(Frame { texture, view })
            }
            Err(err @ (wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated)) => {
                // debug, not warn: this fires per-frame during transient
                // storms (resize bursts); the caller decides what to surface.
                tracing::debug!(error = %err, "surface lost/outdated; reconfigured, skipping frame");
                self.surface.configure(&self.device, &self.config);
                Err(RenderError::AcquireFrame(err))
            }
            Err(err) => Err(RenderError::AcquireFrame(err)),
        }
    }

    /// The wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Texture format the surface was configured with.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Depth attachment view matching the current surface size.
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// Info about the adapter selected at startup (name, backend, driver).
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Current surface size in physical pixels as `(width, height)`.
    pub fn surface_size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}

/// One acquired swapchain frame: render into [`Frame::view`], then call
/// [`Frame::present`].
pub struct Frame {
    texture: wgpu::SurfaceTexture,
    view: wgpu::TextureView,
}

impl Frame {
    /// Color attachment view for this frame.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Schedule the frame for presentation to the window.
    pub fn present(self) {
        self.texture.present();
    }
}

/// Create a `Depth32Float` attachment view. The view keeps the underlying
/// texture alive, so the texture handle itself is not stored.
fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vox-depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Select the wgpu backend from the `WGPU_BACKEND` environment variable,
/// falling back to `Backends::PRIMARY` (Vulkan + DX12 + Metal) so the
/// engine stays portable across machines. Set `WGPU_BACKEND=vulkan` to
/// force Vulkan — useful on AMD Windows drivers where wgpu 0.20's D3D12
/// backend hits a swapchain resource-state bug (`INVALID_SUBRESOURCE_STATE`
/// every frame, eventually crashing with `STATUS_STACK_BUFFER_OVERRUN`).
/// Accepted values (case-insensitive): `vulkan`, `dx12`, `metal`, `gl`,
/// `primary`, `all`.
fn parse_backend_env() -> wgpu::Backends {
    match std::env::var("WGPU_BACKEND")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("vulkan") => wgpu::Backends::VULKAN,
        Some("dx12") => wgpu::Backends::DX12,
        Some("metal") => wgpu::Backends::METAL,
        Some("gl") => wgpu::Backends::GL,
        Some("primary") => wgpu::Backends::PRIMARY,
        Some("all") => wgpu::Backends::all(),
        _ => wgpu::Backends::PRIMARY,
    }
}
