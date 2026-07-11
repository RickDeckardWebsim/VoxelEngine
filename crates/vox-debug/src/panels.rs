//! The overlay's actual widgets: FPS/frame-timing graph, counts, physics
//! tuning sliders, and a material picker. (The crosshair and hotbar are
//! player-facing HUD, not debug -- see [`crate::hud`].)

use egui::{Color32, Context, RichText, Slider, Stroke, Window};

use crate::OverlayState;

/// Build every panel for one frame.
pub fn build(ctx: &Context, mut state: OverlayState<'_>) {
    stats_window(ctx, &state);
    tuning_window(ctx, &mut state);
}

fn stats_window(ctx: &Context, state: &OverlayState<'_>) {
    Window::new("Stats")
        .default_pos([8.0, 8.0])
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(RichText::new(format!("FPS: {}", state.fps)).strong());
            ui.separator();

            ui.label("Frame timings (ms, last / avg over ~4s):");
            for (label, ring) in state.profile.labeled() {
                ui.horizontal(|ui| {
                    ui.monospace(format!("{label:>8}"));
                    ui.monospace(format!("{:>6.2} / {:>6.2}", ring.last(), ring.average()));
                });
                frame_graph_line(ui, ring);
            }
            ui.separator();

            ui.label(format!(
                "chunks drawn/culled: {}/{}",
                state.chunks_drawn, state.chunks_culled
            ));
            ui.label(format!("remesh queue: {}", state.mesh_queue));
            ui.label(format!(
                "body mesh in-flight: {}",
                state.body_mesh_in_flight
            ));
            ui.label(format!(
                "bodies awake/total: {}/{}",
                state.bodies_awake, state.bodies_total
            ));
            ui.label(format!("particles: {}", state.particles));
        });
}

/// A thin sparkline for one timing ring: a row of vertical bars scaled to
/// the ring's own peak, so slow systems don't visually swamp fast ones.
fn frame_graph_line(ui: &mut egui::Ui, ring: &vox_core::TimingRing) {
    let samples: Vec<f32> = ring.oldest_to_newest().collect();
    if samples.is_empty() {
        return;
    }
    let peak = samples.iter().cloned().fold(0.0_f32, f32::max).max(0.001);
    let height = 18.0;
    let (rect, _resp) = ui.allocate_exact_size(
        egui::vec2(samples.len() as f32, height),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    for (i, &ms) in samples.iter().enumerate() {
        let x = rect.left() + i as f32;
        let bar_h = (ms / peak).clamp(0.0, 1.0) * height;
        painter.line_segment(
            [
                egui::pos2(x, rect.bottom()),
                egui::pos2(x, rect.bottom() - bar_h),
            ],
            Stroke::new(1.0, Color32::from_rgb(120, 200, 255)),
        );
    }
}

fn tuning_window(ctx: &Context, state: &mut OverlayState<'_>) {
    Window::new("Tuning")
        .default_pos([8.0, 420.0])
        .resizable(false)
        .show(ctx, |ui| {
            ui.label("Physics feel:");
            ui.add(Slider::new(&mut state.tunables.friction, 0.0..=1.5).text("friction"));
            ui.add(Slider::new(&mut state.tunables.contact_beta, 0.0..=1.0).text("contact_beta"));
            ui.add(
                Slider::new(&mut state.tunables.sleep_lin, 0.0..=0.5)
                    .text("sleep_lin (m/s)")
                    .logarithmic(true),
            );
            ui.add(
                Slider::new(&mut state.tunables.sleep_ang, 0.0..=2.0)
                    .text("sleep_ang (rad/s)")
                    .logarithmic(true),
            );
            ui.add(
                Slider::new(&mut state.tunables.fracture_sensitivity, 0.2..=10.0)
                    .text("fracture_sensitivity")
                    .logarithmic(true),
            );
            ui.separator();

            ui.label("Bomb power / dig-and-bomb radius:");
            ui.add(Slider::new(&mut state.tunables.blast_power, 1.0..=200.0).text("power"));
            ui.add(Slider::new(state.tool_radius, 0.5..=4.0).text("radius (m)"));
            ui.separator();

            ui.label("Movement:");
            ui.add(Slider::new(&mut state.tunables.fly_speed, 1.0..=40.0).text("fly speed (m/s)"));
            ui.separator();
            ui.separator();
            ui.label("Post-processing:");
            ui.add(Slider::new(&mut state.tunables.ssao_intensity, 0.0..=2.0).text("SSAO intensity"));
            ui.add(Slider::new(&mut state.tunables.ssao_radius, 0.1..=2.0).text("SSAO radius"));
            ui.add(Slider::new(&mut state.tunables.bloom_intensity, 0.0..=2.0).text("Bloom intensity"));
            ui.add(Slider::new(&mut state.tunables.bloom_threshold, 0.3..=3.0).text("Bloom threshold"));

            ui.label("Build material:");
            egui::ComboBox::from_label("")
                .selected_text(
                    state
                        .material_names
                        .get(*state.selected_material)
                        .map(String::as_str)
                        .unwrap_or("(none)"),
                )
                .show_ui(ui, |ui| {
                    for (i, name) in state.material_names.iter().enumerate() {
                        ui.selectable_value(state.selected_material, i, name);
                    }
                });
        });
}
