//! Minimal hand-rolled CLI argument parsing for world configuration.
//! No external arg-parsing crate: the surface is tiny (three flags) and a
//! dependency isn't worth it for this.

use vox_core::WorldConfig;

/// Streaming quality preset: controls render distance, tree detail ring,
/// and chunk generation budget per frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    Low,
    Medium,
    High,
    Ultra,
}

impl Quality {
    /// Render distance in chunks (radius around player).
    pub fn render_distance(self) -> i32 {
        match self {
            Quality::Low => 4,
            Quality::Medium => 8,
            Quality::High => 16,
            Quality::Ultra => 24,
        }
    }

    /// Detail ring radius in chunks. Trees root only within this ring;
    /// canopies extend beyond it into far chunks.
    pub fn detail_ring(self) -> i32 {
        match self {
            Quality::Low => 1,
            Quality::Medium => 3,
            Quality::High => 6,
            Quality::Ultra => 12,
        }
    }

    /// Maximum chunks to generate per frame.
    pub fn gen_budget(self) -> usize {
        match self {
            Quality::Low => 2,
            Quality::Medium => 4,
            Quality::High => 8,
            Quality::Ultra => 12,
        }
    }

    /// Maximum loaded chunk count (soft cap for eviction).
    pub fn chunk_cap(self) -> usize {
        let r = self.render_distance() as usize;
        // (2r+1)^2 * height_chunks estimate, plus headroom.
        let side = 2 * r + 1;
        side * side * 4 + 64
    }
}

impl Default for Quality {
    fn default() -> Self {
        Quality::Medium
    }
}

impl std::str::FromStr for Quality {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "low" => Ok(Quality::Low),
            "medium" => Ok(Quality::Medium),
            "high" => Ok(Quality::High),
            "ultra" => Ok(Quality::Ultra),
            _ => Err(format!("unknown quality '{s}', expected low|medium|high|ultra")),
        }
    }
}

/// `voxelengine [--scale 0.1|1.0] [--mario-scale N] [--seed N] [--extent X,Y,Z] [--quality low|medium|high|ultra] [--help]`
pub fn usage() -> String {
    "voxelengine [--scale 0.1|1.0] [--mario-scale N] [--seed N] [--extent X,Y,Z] [--quality low|medium|high|ultra] [--help]\n\n\
     --scale       voxel edge length in meters (default 0.1)\n\
     --mario-scale SM64 units per meter — higher = smaller Mario (default 125, Mario ~1.3m)\n\
     --seed        world generation seed (default 1337)\n\
     --extent      world size in meters, comma-separated X,Y,Z (default 128,48,128)\n\
     --quality     streaming quality preset: low|medium|high|ultra (default medium)\n\
     --help        show this message"
        .to_string()
}

/// True if `args` requests help (checked before [`parse`] so the caller can
/// print usage and exit 0, distinct from a parse error exiting 1).
pub fn wants_help<'a>(args: impl Iterator<Item = &'a str>) -> bool {
    args.into_iter().any(|a| a == "--help" || a == "-h")
}
/// Parsed CLI configuration: world config plus Mario-specific options
/// that don't belong in `WorldConfig`.
#[derive(Debug)]
pub struct CliConfig {
    pub world: WorldConfig,
    /// SM64 units per meter for Mario mode. Higher = smaller Mario.
    /// Default 60 → Mario is ~2.67m tall (160 SM64 units / 60).
    pub mario_units_per_meter: f32,
    /// Streaming quality preset (render distance, tree detail, gen budget).
    pub quality: Quality,
}

/// Parse CLI overrides on top of [`WorldConfig::default`]. Returns a
/// human-readable message (not including usage text) on any failure —
/// unknown flag, missing value, unparseable number, or a value that fails
/// [`WorldConfig::validate`].
pub fn parse<'a>(args: impl Iterator<Item = &'a str>) -> Result<CliConfig, String> {
    let mut cfg = WorldConfig::default();
    let mut mario_units_per_meter: f32 = 125.0;
    let mut quality = Quality::default();
    let args: Vec<&str> = args.collect();
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--scale" => {
                let v = next_value(&args, &mut i, "--scale")?;
                cfg.voxel_size_m = v
                    .parse()
                    .map_err(|_| format!("--scale: invalid number '{v}'"))?;
            }
            "--mario-scale" => {
                let v = next_value(&args, &mut i, "--mario-scale")?;
                mario_units_per_meter = v
                    .parse()
                    .map_err(|_| format!("--mario-scale: invalid number '{v}'"))?;
                if mario_units_per_meter <= 0.0 || !mario_units_per_meter.is_finite() {
                    return Err(format!(
                        "--mario-scale: must be positive and finite, got {mario_units_per_meter}"
                    ));
                }
            }
            "--seed" => {
                let v = next_value(&args, &mut i, "--seed")?;
                cfg.seed = v
                    .parse()
                    .map_err(|_| format!("--seed: invalid number '{v}'"))?;
            }
            "--extent" => {
                let v = next_value(&args, &mut i, "--extent")?;
                let parts: Vec<&str> = v.split(',').collect();
                if parts.len() != 3 {
                    return Err(format!("--extent: expected X,Y,Z got '{v}'"));
                }
                let mut extent = [0.0f32; 3];
                for (slot, p) in extent.iter_mut().zip(parts) {
                    *slot = p
                        .trim()
                        .parse()
                        .map_err(|_| format!("--extent: invalid number '{p}'"))?;
                }
                cfg.extent_m = extent;
            }
            "--quality" => {
                let v = next_value(&args, &mut i, "--quality")?;
                quality = v.parse().map_err(|e: String| format!("--quality: {e}"))?;
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
        i += 1;
    }
    cfg.validate().map_err(|e| e.to_string())?;
    Ok(CliConfig {
        world: cfg,
        mario_units_per_meter,
        quality,
    })
}

fn next_value(args: &[&str], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    args.get(*i)
        .map(|s| s.to_string())
        .ok_or_else(|| format!("{flag}: missing value"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_str(s: &str) -> Result<CliConfig, String> {
        parse(s.split_whitespace())
    }

    #[test]
    fn no_args_yields_default() {
        let cfg = parse_str("").expect("empty args must parse");
        assert_eq!(cfg.world.seed, WorldConfig::default().seed);
        assert_eq!(cfg.world.voxel_size_m, WorldConfig::default().voxel_size_m);
        assert_eq!(cfg.mario_units_per_meter, 125.0);
    }

    #[test]
    fn scale_and_seed_are_parsed() {
        let cfg = parse_str("--scale 1.0 --seed 42").expect("must parse");
        assert_eq!(cfg.world.voxel_size_m, 1.0);
        assert_eq!(cfg.world.seed, 42);
    }

    #[test]
    fn mario_scale_is_parsed() {
        let cfg = parse_str("--mario-scale 90").expect("must parse");
        assert_eq!(cfg.mario_units_per_meter, 90.0);
    }

    #[test]
    fn mario_scale_rejects_zero() {
        let err = parse_str("--mario-scale 0").expect_err("must reject zero");
        assert!(err.contains("--mario-scale"));
    }

    #[test]
    fn extent_is_parsed() {
        let cfg = parse_str("--extent 64,32,64").expect("must parse");
        assert_eq!(cfg.world.extent_m, [64.0, 32.0, 64.0]);
    }

    #[test]
    fn unknown_flag_errors() {
        let err = parse_str("--bogus").expect_err("must reject unknown flag");
        assert!(err.contains("--bogus"), "error must name the flag: {err}");
    }

    #[test]
    fn missing_value_errors() {
        let err = parse_str("--scale").expect_err("must reject a dangling flag");
        assert!(err.contains("--scale"), "error must name the flag: {err}");
    }

    #[test]
    fn invalid_number_errors() {
        let err = parse_str("--seed not-a-number").expect_err("must reject bad number");
        assert!(err.contains("--seed"), "error must name the flag: {err}");
    }

    #[test]
    fn malformed_extent_errors() {
        let err = parse_str("--extent 1,2").expect_err("must reject wrong component count");
        assert!(err.contains("--extent"));
    }

    #[test]
    fn out_of_range_scale_is_rejected_by_validate() {
        let err = parse_str("--scale 0").expect_err("scale 0 must fail WorldConfig::validate");
        assert!(
            err.contains("voxel_size_m"),
            "error must name the field: {err}"
        );
    }

    #[test]
    fn wants_help_detects_both_spellings() {
        assert!(wants_help("--help".split_whitespace()));
        assert!(wants_help("-h".split_whitespace()));
        assert!(!wants_help("--scale 1.0".split_whitespace()));
    }
    #[test]
    fn quality_parses_low() {
        let cli = parse(["--quality", "low"].into_iter()).unwrap();
        assert_eq!(cli.quality, Quality::Low);
    }

    #[test]
    fn quality_parses_medium_default() {
        let cli = parse([].into_iter()).unwrap();
        assert_eq!(cli.quality, Quality::Medium);
    }

    #[test]
    fn quality_parses_ultra() {
        let cli = parse(["--quality", "ultra"].into_iter()).unwrap();
        assert_eq!(cli.quality, Quality::Ultra);
    }

    #[test]
    fn quality_rejects_unknown() {
        assert!(parse(["--quality", "turbo"].into_iter()).is_err());
    }
}
