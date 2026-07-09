use std::path::PathBuf;

const C_FILES: &[&str] = &[
    "src/debug_print.c",
    "src/decomp/audio/copt/seq_channel_layer_process_script_copt.inc.c",
    "src/decomp/audio/data.c",
    "src/decomp/audio/effects.c",
    "src/decomp/audio/external.c",
    "src/decomp/audio/globals_start.c",
    "src/decomp/audio/heap.c",
    "src/decomp/audio/load.c",
    "src/decomp/audio/load_dat.c",
    "src/decomp/audio/load_sh.c",
    "src/decomp/audio/playback.c",
    "src/decomp/audio/port_eu.c",
    "src/decomp/audio/port_sh.c",
    "src/decomp/audio/seqplayer.c",
    "src/decomp/audio/shindou_debug_prints.c",
    "src/decomp/audio/synthesis.c",
    "src/decomp/audio/synthesis_sh.c",
    "src/decomp/engine/geo_layout.c",
    "src/decomp/engine/graph_node.c",
    "src/decomp/engine/graph_node_manager.c",
    "src/decomp/engine/guMtxF2L.c",
    "src/decomp/engine/math_util.c",
    "src/decomp/engine/surface_collision.c",
    "src/decomp/game/behavior_actions.c",
    "src/decomp/game/interaction.c",
    "src/decomp/game/mario_actions_airborne.c",
    "src/decomp/game/mario_actions_automatic.c",
    "src/decomp/game/mario_actions_cutscene.c",
    "src/decomp/game/mario_actions_moving.c",
    "src/decomp/game/mario_actions_object.c",
    "src/decomp/game/mario_actions_stationary.c",
    "src/decomp/game/mario_actions_submerged.c",
    "src/decomp/game/mario.c",
    "src/decomp/game/mario_misc.c",
    "src/decomp/game/mario_step.c",
    "src/decomp/game/object_stuff.c",
    "src/decomp/game/platform_displacement.c",
    "src/decomp/game/rendering_graph_node.c",
    "src/decomp/game/sound_init.c",
    "src/decomp/global_state.c",
    "src/decomp/mario/geo.inc.c",
    "src/decomp/mario/model.inc.c",
    "src/decomp/memory.c",
    "src/decomp/pc/alBnkfNew.c",
    "src/decomp/pc/mixer.c",
    "src/decomp/pc/ultra_reimplementation.c",
    "src/decomp/tools/convUtils.c",
    "src/decomp/tools/libmio0.c",
    "src/decomp/tools/n64graphics.c",
    "src/decomp/tools/utils.c",
    "src/fake_interaction.c",
    "src/gfx_adapter.c",
    "src/libsm64.c",
    "src/load_anim_data.c",
    "src/load_audio_data.c",
    "src/load_surfaces.c",
    "src/load_tex_data.c",
    "src/obj_pool.c",
    "src/play_sound.c",
];

fn main() {
    let libsm64_dir = PathBuf::from("../../libsm64");

    // Verify the mario geometry was imported (python script must have run)
    let mario_geo = libsm64_dir.join("src/decomp/mario/geo.inc.c");
    if !mario_geo.exists() {
        panic!(
            "Mario geometry not found at {}. Run `python3 import-mario-geo.py` in the libsm64 directory first.",
            mario_geo.display()
        );
    }

    cc::Build::new()
        .flags(["-fno-strict-aliasing", "-fPIC", "-fvisibility=hidden"])
        .warnings(false)
        .define("SM64_LIB_EXPORT", None)
        .define("GBI_FLOATS", None)
        .define("VERSION_US", None)
        .define("NO_SEGMENTED_MEMORY", None)
        .files(C_FILES.iter().map(|f| libsm64_dir.join(f)))
        .include(libsm64_dir.join("src/decomp/include"))
        .compile("sm64");

    // Tell cargo to rerun if any C source or header changes.
    // Without this, editing .c files doesn't trigger recompilation
    // and stale .o files get linked, causing crashes.
    for f in C_FILES {
        println!("cargo:rerun-if-changed={}", libsm64_dir.join(f).display());
    }
    println!("cargo:rerun-if-changed={}", libsm64_dir.join("src/libsm64.h").display());
    println!("cargo:rerun-if-changed={}", libsm64_dir.join("src/gfx_adapter.h").display());
}
