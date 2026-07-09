//! Material definitions and a registry that loads them from TOML assets.
//!
//! Materials are declared as `[[material]]` tables. Index 0 is always the
//! built-in `air` material and asset files must not declare it. Ids are
//! assigned in declaration order — air first, then file content; when loading
//! a directory, files merge sorted by filename.

use std::collections::HashMap;
use std::path::Path;

use crate::error::CoreError;

/// Identifier of a material in the registry. 0 is always air.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct MaterialId(pub u16);

impl MaterialId {
    /// The built-in air material, always at index 0.
    pub const AIR: MaterialId = MaterialId(0);
}

/// One material definition. SI units: density in kg/m³.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialDef {
    /// Unique registry name (e.g. `"stone"`).
    pub name: String,
    /// Linear RGB base color, each component in `0..=1`.
    pub color: [f32; 3],
    /// Per-voxel color variation amplitude, `0..=1`.
    pub jitter: f32,
    /// Density in kg/m³.
    pub density: f32,
    /// Destruction resistance (relative, dimensionless).
    pub strength: f32,
    /// Whether the material occupies space; air is the canonical non-solid.
    pub solid: bool,
    /// Whether a `vox-sim`-family crate simulates this material as a fluid
    /// (flows, spreads, settles). Implies `solid = false`, but is a distinct
    /// flag: a future decorative non-solid material (tall grass, say) should
    /// not be picked up by the fluid sim just because it's non-solid.
    pub fluid: bool,
    /// Whether `vox-sim` simulates this material as a powder (falls when
    /// unsupported, piles at an angle of repose, no pressure-driven
    /// spreading). Implies `solid = false`, but is distinct from `fluid`:
    /// a powder piles rather than seeking a flat level. Mud and sand are
    /// powders; water is a fluid.
    pub powder: bool,
}

/// Registry of material definitions with stable `u16` ids.
///
/// Index 0 is always the built-in `air`. Declared materials follow in source
/// order; [`load_dir`](Self::load_dir) merges files sorted by filename.
#[derive(Clone, Debug)]
pub struct MaterialRegistry {
    defs: Vec<MaterialDef>,
    by_name: HashMap<String, MaterialId>,
}

/// One parsed TOML document: a list of `[[material]]` tables.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDoc {
    #[serde(default)]
    material: Vec<RawMaterial>,
}

/// One `[[material]]` table with every key optional, so presence checks can
/// produce validation errors that name the material.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMaterial {
    name: Option<String>,
    color: Option<[f32; 3]>,
    jitter: Option<f32>,
    density: Option<f32>,
    strength: Option<f32>,
    solid: Option<bool>,
    fluid: Option<bool>,
    powder: Option<bool>,
}

/// Validation error for a named material: `material '{name}': {detail}`.
fn material_error(origin: &str, name: &str, detail: impl std::fmt::Display) -> CoreError {
    CoreError::Asset {
        path: origin.to_string(),
        reason: format!("material '{name}': {detail}"),
    }
}

impl MaterialRegistry {
    /// Registry holding only the built-in air material at id 0.
    fn with_air() -> Self {
        let air = MaterialDef {
            name: "air".to_string(),
            color: [0.0, 0.0, 0.0],
            jitter: 0.0,
            density: 0.0,
            strength: 0.0,
            solid: false,
            fluid: false,
            powder: false,
        };
        let mut by_name = HashMap::new();
        by_name.insert(air.name.clone(), MaterialId::AIR);
        Self {
            defs: vec![air],
            by_name,
        }
    }

    /// Parse one TOML document. `origin` is the filename used in error
    /// messages.
    pub fn from_toml_str(source: &str, origin: &str) -> Result<Self, CoreError> {
        let mut registry = Self::with_air();
        registry.add_source(source, origin)?;
        Ok(registry)
    }

    /// Read all `*.toml` files in `dir` sorted by filename (ASCII
    /// case-insensitive, so order matches on case-insensitive filesystems) and
    /// merge them in order. A duplicate material name across files is an
    /// error; a missing or unreadable directory is a [`CoreError::Asset`].
    pub fn load_dir(dir: &Path) -> Result<Self, CoreError> {
        let entries = std::fs::read_dir(dir).map_err(|e| CoreError::Asset {
            path: dir.display().to_string(),
            reason: format!("cannot read material directory: {e}"),
        })?;
        let mut files = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| CoreError::Asset {
                path: dir.display().to_string(),
                reason: format!("cannot read directory entry: {e}"),
            })?;
            let path = entry.path();
            let is_toml = path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"));
            if is_toml && path.is_file() {
                files.push(path);
            }
        }
        // Merge order = filename order, case-folded so `MyMod.toml` does not
        // jump ahead of `core.toml` on case-insensitive filesystems.
        files.sort_by_key(|p| {
            p.file_name()
                .map(|n| n.to_string_lossy().to_ascii_lowercase())
                .unwrap_or_default()
        });

        let mut registry = Self::with_air();
        for path in files {
            let origin = path.display().to_string();
            let source = std::fs::read_to_string(&path).map_err(|e| CoreError::Asset {
                path: origin.clone(),
                reason: format!("cannot read file: {e}"),
            })?;
            registry.add_source(&source, &origin)?;
        }
        Ok(registry)
    }

    /// Parse `source` and append its materials to the registry.
    fn add_source(&mut self, source: &str, origin: &str) -> Result<(), CoreError> {
        let doc: RawDoc = toml::from_str(source).map_err(|e| CoreError::Asset {
            path: origin.to_string(),
            // toml's Display is self-describing ("TOML parse error at ...").
            reason: e.to_string(),
        })?;
        for (index, raw) in doc.material.into_iter().enumerate() {
            self.add_material(raw, index, origin)?;
        }
        Ok(())
    }

    /// Validate one raw material and append it. `index` is its zero-based
    /// position within the source, used when it has no usable name.
    fn add_material(
        &mut self,
        raw: RawMaterial,
        index: usize,
        origin: &str,
    ) -> Result<(), CoreError> {
        let Some(name) = raw.name else {
            return Err(CoreError::Asset {
                path: origin.to_string(),
                reason: format!("material #{}: missing required key 'name'", index + 1),
            });
        };
        if name.is_empty() {
            return Err(CoreError::Asset {
                path: origin.to_string(),
                reason: format!("material #{}: name must be non-empty", index + 1),
            });
        }
        if name == "air" {
            return Err(material_error(
                origin,
                &name,
                "name 'air' is reserved for the built-in air material",
            ));
        }
        if self.by_name.contains_key(&name) {
            return Err(material_error(origin, &name, "duplicate material name"));
        }

        let Some(color) = raw.color else {
            return Err(material_error(
                origin,
                &name,
                "missing required key 'color'",
            ));
        };
        for (i, &c) in color.iter().enumerate() {
            if !(0.0..=1.0).contains(&c) {
                return Err(material_error(
                    origin,
                    &name,
                    format!("color[{i}] must be in 0..=1, got {c}"),
                ));
            }
        }
        let jitter = raw.jitter.unwrap_or(0.0);
        if !(0.0..=1.0).contains(&jitter) {
            return Err(material_error(
                origin,
                &name,
                format!("jitter must be in 0..=1, got {jitter}"),
            ));
        }
        let Some(density) = raw.density else {
            return Err(material_error(
                origin,
                &name,
                "missing required key 'density'",
            ));
        };
        if !density.is_finite() || density <= 0.0 {
            return Err(material_error(
                origin,
                &name,
                format!("density must be > 0, got {density}"),
            ));
        }
        let Some(strength) = raw.strength else {
            return Err(material_error(
                origin,
                &name,
                "missing required key 'strength'",
            ));
        };
        if !strength.is_finite() || strength < 0.0 {
            return Err(material_error(
                origin,
                &name,
                format!("strength must be >= 0, got {strength}"),
            ));
        }
        let solid = raw.solid.unwrap_or(true);
        let fluid = raw.fluid.unwrap_or(false);
        let powder = raw.powder.unwrap_or(false);

        if self.defs.len() > usize::from(u16::MAX) {
            return Err(material_error(
                origin,
                &name,
                "registry is full: material ids are u16 (at most 65536 materials including air)",
            ));
        }
        // Fits by the cap check above.
        let id = MaterialId(self.defs.len() as u16);
        self.by_name.insert(name.clone(), id);
        self.defs.push(MaterialDef {
            name,
            color,
            jitter,
            density,
            strength,
            solid,
            fluid,
            powder,
        });
        Ok(())
    }

    /// Definition for `id`, if present.
    pub fn get(&self, id: MaterialId) -> Option<&MaterialDef> {
        self.defs.get(usize::from(id.0))
    }

    /// Id registered for `name`, if present.
    pub fn id_by_name(&self, name: &str) -> Option<MaterialId> {
        self.by_name.get(name).copied()
    }

    /// Number of materials, including the built-in air.
    pub fn len(&self) -> usize {
        self.defs.len()
    }

    /// Always false: a registry contains at least the built-in air.
    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }

    /// Iterate over `(id, definition)` pairs in id order.
    pub fn iter(&self) -> impl Iterator<Item = (MaterialId, &MaterialDef)> {
        self.defs
            .iter()
            .enumerate()
            // Indices fit u16 by the registry cap.
            .map(|(i, def)| (MaterialId(i as u16), def))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// Two well-formed materials exercising explicit and defaulted keys.
    const TWO_MATERIALS: &str = r#"
        [[material]]
        name = "stone"
        color = [0.55, 0.55, 0.57]
        jitter = 0.04
        density = 2600.0
        strength = 8.0

        [[material]]
        name = "water"
        color = [0.1, 0.3, 0.8]
        density = 1000.0
        strength = 0.0
        solid = false
    "#;

    /// Temp dir under `std::env::temp_dir()`, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("vox-core-material-{tag}-{}", std::process::id()));
            // A killed earlier run may have left the dir behind; start clean.
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }

        fn write(&self, name: &str, contents: &str) {
            fs::write(self.0.join(name), contents).expect("write temp file");
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn material(name: &str) -> String {
        format!(
            "[[material]]\nname = \"{name}\"\ncolor = [0.2, 0.2, 0.2]\ndensity = 1.0\nstrength = 0.0\n"
        )
    }

    #[test]
    fn two_materials_get_sequential_ids_with_air_at_zero() {
        let reg = MaterialRegistry::from_toml_str(TWO_MATERIALS, "inline.toml").expect("parses");
        assert_eq!(reg.len(), 3);

        let air = reg.get(MaterialId::AIR).expect("air present");
        assert_eq!(air.name, "air");
        assert!(!air.solid);
        assert_eq!(air.density, 0.0);
        assert_eq!(air.strength, 0.0);
        assert_eq!(air.jitter, 0.0);
        assert_eq!(air.color, [0.0, 0.0, 0.0]);

        assert_eq!(reg.id_by_name("stone"), Some(MaterialId(1)));
        assert_eq!(reg.id_by_name("water"), Some(MaterialId(2)));
        assert_eq!(reg.id_by_name("air"), Some(MaterialId::AIR));
        assert_eq!(reg.id_by_name("missing"), None);
        assert!(reg.get(MaterialId(3)).is_none());

        let stone = reg.get(MaterialId(1)).expect("stone present");
        assert_eq!(stone.name, "stone");
        assert_eq!(stone.color, [0.55, 0.55, 0.57]);
        assert_eq!(stone.jitter, 0.04);
        assert_eq!(stone.density, 2600.0);
        assert_eq!(stone.strength, 8.0);
        assert!(stone.solid);

        let ids: Vec<u16> = reg.iter().map(|(id, _)| id.0).collect();
        assert_eq!(ids, vec![0, 1, 2]);
        let names: Vec<&str> = reg.iter().map(|(_, def)| def.name.as_str()).collect();
        assert_eq!(names, vec!["air", "stone", "water"]);
    }

    #[test]
    fn missing_density_names_origin_material_and_key() {
        let source = r#"
            [[material]]
            name = "stone"
            color = [0.5, 0.5, 0.5]
            strength = 8.0
        "#;
        let err = MaterialRegistry::from_toml_str(source, "inline-test.toml")
            .expect_err("missing density must fail");
        let msg = err.to_string();
        assert!(msg.contains("inline-test.toml"), "must name origin: {msg}");
        assert!(msg.contains("stone"), "must name the material: {msg}");
        assert!(msg.contains("density"), "must name the key: {msg}");
    }

    #[test]
    fn duplicate_name_in_one_source_is_rejected() {
        let source = format!("{}{}", material("stone"), material("stone"));
        let err = MaterialRegistry::from_toml_str(&source, "dup.toml")
            .expect_err("duplicate name must fail");
        let msg = err.to_string();
        assert!(msg.contains("stone"), "must name the material: {msg}");
        assert!(msg.contains("duplicate"), "must say duplicate: {msg}");
    }

    #[test]
    fn declaring_air_is_rejected() {
        let err = MaterialRegistry::from_toml_str(&material("air"), "air.toml")
            .expect_err("declaring air must fail");
        let msg = err.to_string();
        assert!(msg.contains("air"), "must name the material: {msg}");
        assert!(msg.contains("reserved"), "must say reserved: {msg}");
    }

    #[test]
    fn out_of_range_color_component_is_rejected() {
        let source = r#"
            [[material]]
            name = "neon"
            color = [0.5, 1.5, 0.5]
            density = 1.0
            strength = 0.0
        "#;
        let err = MaterialRegistry::from_toml_str(source, "color.toml")
            .expect_err("color component 1.5 must fail");
        let msg = err.to_string();
        assert!(msg.contains("neon"), "must name the material: {msg}");
        assert!(msg.contains("color"), "must name the key: {msg}");
    }

    #[test]
    fn jitter_and_solid_defaults_are_applied() {
        let reg = MaterialRegistry::from_toml_str(&material("plain"), "defaults.toml")
            .expect("minimal material parses");
        let def = reg
            .get(reg.id_by_name("plain").expect("registered"))
            .expect("present");
        assert_eq!(def.jitter, 0.0, "jitter defaults to 0.0");
        assert!(def.solid, "solid defaults to true");
    }

    #[test]
    fn empty_source_yields_air_only_registry() {
        let reg = MaterialRegistry::from_toml_str("", "empty.toml").expect("empty source is fine");
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
        assert_eq!(reg.get(MaterialId::AIR).expect("air").name, "air");
        assert_eq!(reg.id_by_name("air"), Some(MaterialId::AIR));
    }

    #[test]
    fn syntax_error_wraps_toml_message_and_origin() {
        let err = MaterialRegistry::from_toml_str("not = = toml", "bad.toml")
            .expect_err("bad syntax must fail");
        let msg = err.to_string();
        assert!(msg.contains("bad.toml"), "must name origin: {msg}");
        assert!(msg.contains("TOML"), "must wrap the toml error: {msg}");
    }

    #[test]
    fn load_dir_merges_files_sorted_by_name() {
        let dir = TempDir::new("sorted");
        // Written out of order on purpose: merge order must follow filenames.
        dir.write("b.toml", &material("beta"));
        dir.write("a.toml", &material("alpha"));
        dir.write("notes.txt", "not a material file");

        let reg = MaterialRegistry::load_dir(&dir.0).expect("load_dir succeeds");
        assert_eq!(reg.len(), 3, "air + one material per toml file");
        assert_eq!(reg.id_by_name("alpha"), Some(MaterialId(1)));
        assert_eq!(reg.id_by_name("beta"), Some(MaterialId(2)));
    }

    #[test]
    fn load_dir_rejects_duplicate_name_across_files() {
        let dir = TempDir::new("dup");
        dir.write("a.toml", &material("stone"));
        dir.write("b.toml", &material("stone"));

        let err = MaterialRegistry::load_dir(&dir.0).expect_err("cross-file duplicate must fail");
        let msg = err.to_string();
        assert!(msg.contains("stone"), "must name the material: {msg}");
        assert!(msg.contains("b.toml"), "must name the second file: {msg}");
    }

    #[test]
    fn load_dir_missing_directory_is_asset_error() {
        let missing = std::env::temp_dir().join("vox-core-material-definitely-missing");
        let err = MaterialRegistry::load_dir(&missing).expect_err("missing dir must fail");
        assert!(matches!(err, CoreError::Asset { .. }), "got: {err:?}");
    }

    #[test]
    fn registry_rejects_material_ids_beyond_u16() {
        // Air occupies id 0, so u16 ids allow 65_535 declared materials; the
        // 65_536th would need id 65_536 and must be rejected.
        let mut source = String::new();
        for i in 1..=65_536_u32 {
            source.push_str(&material(&format!("m{i}")));
        }
        let err = MaterialRegistry::from_toml_str(&source, "cap.toml")
            .expect_err("65_536 declared materials must overflow u16 ids");
        let msg = err.to_string();
        assert!(msg.contains("m65536"), "must name the material: {msg}");
    }

    #[test]
    fn loads_shipped_core_materials() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/materials/core.toml");
        let source =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let reg = MaterialRegistry::from_toml_str(&source, "assets/materials/core.toml")
            .expect("shipped core.toml must parse");
        assert_eq!(reg.len(), 11, "air + 10 shipped materials");
        assert_eq!(reg.id_by_name("stone"), Some(MaterialId(1)));
        let mud = reg
            .get(reg.id_by_name("mud").expect("mud registered"))
            .expect("mud present");
        assert_eq!(mud.density, 1700.0);
        assert!(!mud.solid, "mud is non-solid (a powder, not walkable)");
        assert!(mud.powder, "mud is a powder");
        let sand = reg
            .get(reg.id_by_name("sand").expect("sand registered"))
            .expect("sand present");
        assert!(!sand.solid, "sand is non-solid (a powder, not walkable)");
        assert!(sand.powder, "sand is a powder");
        let leaves = reg
            .get(reg.id_by_name("leaves").expect("leaves registered"))
            .expect("leaves present");
        assert_eq!(leaves.jitter, 0.10);
        assert!(leaves.solid, "solid defaults to true in shipped file");
    }

    /// Pins `deny_unknown_fields`: a typo'd optional key must fail loudly
    /// instead of being silently ignored (the modding surface depends on it).
    #[test]
    fn unknown_keys_are_rejected_naming_the_typo() {
        let source = r#"
            [[material]]
            name = "typo"
            color = [0.2, 0.2, 0.2]
            density = 1.0
            strength = 0.0
            jiter = 0.1
        "#;
        let err =
            MaterialRegistry::from_toml_str(source, "typo.toml").expect_err("typo key must fail");
        let msg = err.to_string();
        assert!(msg.contains("typo.toml"), "must name origin: {msg}");
        assert!(msg.contains("jiter"), "must name the unknown key: {msg}");
    }

    #[test]
    fn out_of_range_numeric_values_are_rejected_naming_the_key() {
        let cases = [
            ("density", "density = 0.0\nstrength = 1.0"),
            ("density", "density = -5.0\nstrength = 1.0"),
            ("density", "density = nan\nstrength = 1.0"),
            ("strength", "density = 1.0\nstrength = -1.0"),
            ("strength", "density = 1.0\nstrength = nan"),
            ("jitter", "density = 1.0\nstrength = 1.0\njitter = 1.5"),
            ("jitter", "density = 1.0\nstrength = 1.0\njitter = nan"),
        ];
        for (key, body) in cases {
            let source = format!("[[material]]\nname = \"bad\"\ncolor = [0.2, 0.2, 0.2]\n{body}\n");
            let err = MaterialRegistry::from_toml_str(&source, "range.toml")
                .err()
                .unwrap_or_else(|| panic!("case `{body}` must be rejected"));
            let msg = err.to_string();
            assert!(msg.contains(key), "error must name key `{key}`: {msg}");
            assert!(msg.contains("bad"), "error must name the material: {msg}");
        }
    }

    #[test]
    fn fluid_defaults_to_false_and_can_be_set_true() {
        let reg = MaterialRegistry::from_toml_str(
            r#"
            [[material]]
            name = "stone"
            color = [0.5, 0.5, 0.5]
            density = 2000.0
            strength = 5.0

            [[material]]
            name = "water"
            color = [0.1, 0.3, 0.8]
            density = 1000.0
            strength = 0.0
            solid = false
            fluid = true
            "#,
            "test.toml",
        )
        .expect("registry");
        let stone = reg.get(reg.id_by_name("stone").unwrap()).unwrap();
        let water = reg.get(reg.id_by_name("water").unwrap()).unwrap();
        assert!(!stone.fluid, "fluid must default to false");
        assert!(water.fluid, "explicit fluid = true must round-trip");
        assert!(!water.solid, "water must not be solid");
    }

    #[test]
    fn powder_defaults_to_false_and_can_be_set_true() {
        let reg = MaterialRegistry::from_toml_str(
            r#"
            [[material]]
            name = "stone"
            color = [0.5, 0.5, 0.5]
            density = 2000.0
            strength = 5.0

            [[material]]
            name = "sand"
            color = [0.86, 0.79, 0.58]
            density = 1600.0
            strength = 1.0
            solid = false
            powder = true
            "#,
            "test.toml",
        )
        .expect("registry");
        let stone = reg.get(reg.id_by_name("stone").unwrap()).unwrap();
        let sand = reg.get(reg.id_by_name("sand").unwrap()).unwrap();
        assert!(!stone.powder, "powder must default to false");
        assert!(sand.powder, "explicit powder = true must round-trip");
        assert!(!sand.solid, "sand must not be solid");
    }

    #[test]
    fn load_dir_sort_is_case_insensitive() {
        let dir = TempDir::new("case");
        // Byte order would put `MyMod.toml` (M = 0x4D) before `core.toml`;
        // case-folded filename order must not.
        dir.write("MyMod.toml", &material("modded"));
        dir.write("core.toml", &material("base"));

        let reg = MaterialRegistry::load_dir(&dir.0).expect("load_dir succeeds");
        assert_eq!(
            reg.id_by_name("base"),
            Some(MaterialId(1)),
            "core.toml must load before MyMod.toml"
        );
        assert_eq!(reg.id_by_name("modded"), Some(MaterialId(2)));
    }
}
