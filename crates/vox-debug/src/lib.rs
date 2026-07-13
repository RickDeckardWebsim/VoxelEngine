//! egui-based debug overlay: FPS/frame-timing graph, world/physics counters,
//! live physics tuning sliders, a material picker, and a crosshair.
//!
//! This crate owns egui entirely — plumbing *and* panel layout. The caller
//! (vox-app) never touches the `egui` crate directly: it fills in an
//! [`OverlayState`] each frame from its own counters and calls
//! [`DebugOverlay::prepare`]; every widget shown is defined in [`panels`].
//!
//! Two-phase per frame, mirroring `vox_render::VoxelPipeline`'s
//! write-then-draw split: [`DebugOverlay::prepare`] runs before the render
//! pass exists (egui input, layout, tessellation, buffer upload) and returns
//! an owned [`PreparedFrame`]; [`DebugOverlay::paint`] records the draw
//! calls once the caller has opened its render pass. Splitting them is
//! required, not stylistic: the tessellated primitives must outlive the
//! pass, and a pass's borrow can't be satisfied by data created after it.

pub mod hud;
mod panels;

use std::sync::Arc;

use egui::{ClippedPrimitive, Context, ViewportId};
use egui_wgpu::{Renderer, ScreenDescriptor};
use egui_winit::State;
use vox_core::{FrameProfile, Tunables};
use winit::event::WindowEvent;
use winit::window::Window;

/// Everything the overlay's UI closure can read or edit. Owned by the app;
/// borrowed mutably for one frame's `prepare` call.
pub struct OverlayState<'a> {
    pub profile: &'a FrameProfile,
    pub tunables: &'a mut Tunables,
    pub fps: u32,
    pub chunks_drawn: u32,
    pub chunks_culled: u32,
    pub mesh_queue: usize,
    pub body_mesh_in_flight: usize,
    pub bodies_awake: usize,
    pub bodies_total: usize,
    pub particles: usize,
    pub tool_radius: &'a mut f32,
    pub material_names: &'a [String],
    pub selected_material: &'a mut usize,
    pub always_day: &'a mut bool,
    pub quality_label: &'a str,
    pub ecs_entity_count: usize,
    pub ecs_entities: &'a [(u32, String, [f32; 3])],
}
/// Output of [`DebugOverlay::prepare`]: owned, tessellated draw data ready
/// to record into a render pass via [`DebugOverlay::paint`].
pub struct PreparedFrame {
    clipped_primitives: Vec<ClippedPrimitive>,
    screen: ScreenDescriptor,
}

/// The egui overlay: context, winit integration state, and the wgpu
/// renderer. Visibility (F3) is the caller's concern — `prepare`/`paint`
/// always draw when called, so gate the call site instead.
pub struct DebugOverlay {
    ctx: Context,
    winit_state: State,
    renderer: Renderer,
}

impl DebugOverlay {
    /// Build the overlay against an existing GPU device and surface format.
    /// `depth_format` must match whatever depth-stencil attachment (if any)
    /// the render pass this overlay paints into actually uses -- egui's
    /// pipeline is validated against the pass it's recorded into, and a
    /// mismatch (`None` here vs. a real depth attachment on the pass) is a
    /// wgpu validation panic, not a silent fallback.
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        window: &Arc<Window>,
    ) -> Self {
        let ctx = Context::default();
        let winit_state = State::new(
            ctx.clone(),
            ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
        );
        let renderer = Renderer::new(device, surface_format, depth_format, 1);
        Self {
            ctx,
            winit_state,
            renderer,
        }
    }

    /// Feed a window event to egui. Returns `true` if egui consumed it (the
    /// caller should not also treat it as game input).
    pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.winit_state.on_window_event(window, event).consumed
    }

    /// Build this frame's UI from `hud` (always drawn) and `debug` (the F3
    /// windows -- pass `None` when hidden) and upload its draw data. Call
    /// before opening the render pass; hand the result to [`Self::paint`]
    /// inside it.
    #[expect(
        clippy::too_many_arguments,
        reason = "one-per-frame plumbing call; a params struct would only relabel it"
    )]
    pub fn prepare(
        &mut self,
        window: &Window,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        size: (u32, u32),
        hud_state: &hud::HudState<'_>,
        debug: Option<OverlayState<'_>>,
    ) -> PreparedFrame {
        let raw_input = self.winit_state.take_egui_input(window);
        let full_output = self.ctx.run(raw_input, |ctx| {
            hud::build(ctx, hud_state);
            if let Some(state) = debug {
                panels::build(ctx, state);
            }
        });
        self.winit_state
            .handle_platform_output(window, full_output.platform_output);

        let clipped_primitives = self
            .ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen = ScreenDescriptor {
            size_in_pixels: [size.0, size.1],
            pixels_per_point: full_output.pixels_per_point,
        };
        for (id, delta) in &full_output.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, delta);
        }
        self.renderer
            .update_buffers(device, queue, encoder, &clipped_primitives, &screen);
        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
        PreparedFrame {
            clipped_primitives,
            screen,
        }
    }

    /// Record the prepared frame's draw calls into an open render pass.
    pub fn paint<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>, prepared: &'p PreparedFrame) {
        self.renderer
            .render(pass, &prepared.clipped_primitives, &prepared.screen);
    }
}
