//! wgpu-based renderer: GPU bootstrap, the opaque voxel pipeline (drawing
//! both world chunks and debris bodies), frustum culling, and a fly camera.
//! This crate deliberately has no winit dependency: window types enter only
//! as [`wgpu::SurfaceTarget`].

pub mod bloom_ssao;
pub mod camera;
pub mod frustum;
pub mod gpu;
pub mod grass_pipeline;
#[cfg(feature = "mario")]
pub mod mario_pipeline;
pub mod particles;
pub mod postprocess;
pub mod voxel_pipeline;

pub use bloom_ssao::BloomSsaoPipeline;
pub use camera::Camera;
pub use frustum::Frustum;
pub use gpu::{DEPTH_FORMAT, Frame, Gpu};
pub use grass_pipeline::{GrassPipeline, GrassVertex, MAX_GRASS_BLADES};
#[cfg(feature = "mario")]
pub use mario_pipeline::{MarioCameraUniform, MarioPipeline, MarioVertex};
pub use particles::{MAX_PARTICLES, ParticleInstance, ParticlePipeline};
pub use postprocess::{COLOR_FORMAT, PostProcessPipeline};
pub use voxel_pipeline::{BodyMeshKey, DrawStats, ShadowPipeline, VoxelPipeline};

/// Errors from GPU initialization and per-frame surface operations.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// No surface-compatible adapter was found on any backend. Usually
    /// means missing/broken GPU drivers.
    #[error(
        "no compatible GPU adapter found (searched all backends for a surface-compatible adapter)"
    )]
    NoAdapter,
    /// The window could not be turned into a rendering surface.
    #[error("failed to create rendering surface from window: {0}")]
    CreateSurface(#[from] wgpu::CreateSurfaceError),
    /// The adapter refused the device request (features/limits mismatch or
    /// driver loss during init).
    #[error("failed to acquire GPU device from adapter: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    /// The surface reports no usable texture formats for this adapter.
    #[error("surface reports no supported texture formats for the selected adapter")]
    NoSurfaceFormat,
    /// Acquiring the next swapchain frame failed; the frame should be
    /// skipped (the surface is reconfigured internally when recoverable).
    #[error("failed to acquire frame from surface: {0}")]
    AcquireFrame(#[from] wgpu::SurfaceError),
}

impl RenderError {
    /// True if the error is expected to clear on a following frame, so the
    /// caller should skip this frame and keep running.
    ///
    /// Transient: frame acquisition hiccups — `Lost`/`Outdated` (surface was
    /// reconfigured internally) and `Timeout` (compositor stall; retrying is
    /// the standard response). Everything else — initialization failures and
    /// `OutOfMemory` — is fatal and the caller should shut down.
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            Self::AcquireFrame(
                wgpu::SurfaceError::Lost
                    | wgpu::SurfaceError::Outdated
                    | wgpu::SurfaceError::Timeout
            )
        )
    }
}
