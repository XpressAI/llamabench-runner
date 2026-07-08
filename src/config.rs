// SPDX-License-Identifier: GPL-3.0-or-later
//! Per-user config: the submission token, plus the persistent model **links**
//! (local GGUF path → Hugging Face repo, hash-verified — ADR-009).
//!
//! Everything lives in `dirs::config_dir()/llamabench/config.json` (e.g.
//! `~/.config/llamabench/config.json` on Linux, `~/Library/Application
//! Support/llamabench/config.json` on macOS). On unix the file is chmod 600
//! (it holds the token). `LLAMABENCH_CONFIG_DIR` overrides the directory
//! (used by tests; handy for CI).

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

/// One linked model: the local file (the map key is its canonical absolute path)
/// tied to the HF repo it claims to come from. `size`/`mtime` let a run reuse the
/// stored verification without re-hashing a multi-GB file; if they changed, the
/// file is re-hashed and re-verified. `file` is the repo filename whose LFS
/// sha256 matched (empty when nothing matched — `verified: false`).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct LinkEntry {
    pub repo: String,
    #[serde(default)]
    pub file: String,
    pub sha256: String,
    pub size: u64,
    pub mtime: i64,
    pub verified: bool,
}

/// A cached file hash (no repo claim — that's a `LinkEntry`). Lets every run
/// record `ggufSha256` (ADR-010) without re-hashing a multi-GB file: valid while
/// `size`/`mtime` are unchanged.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct HashEntry {
    pub sha256: String,
    pub size: u64,
    pub mtime: i64,
}

#[derive(Serialize, Deserialize, Default)]
struct ConfigFile {
    #[serde(default)]
    token: String,
    #[serde(default)]
    links: BTreeMap<String, LinkEntry>,
    #[serde(default)]
    hashes: BTreeMap<String, HashEntry>,
}

/// `dirs::config_dir()/llamabench/config.json` (or `$LLAMABENCH_CONFIG_DIR/config.json`).
pub fn config_path() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("LLAMABENCH_CONFIG_DIR") {
        return Ok(PathBuf::from(dir).join("config.json"));
    }
    let dir = dirs::config_dir()
        .ok_or_else(|| anyhow!("could not determine the per-user config directory"))?;
    Ok(dir.join("llamabench").join("config.json"))
}

fn load() -> ConfigFile {
    let Ok(path) = config_path() else {
        return ConfigFile::default();
    };
    fs::read_to_string(path)
        .ok()
        .and_then(|data| serde_json::from_str(&data).ok())
        .unwrap_or_default()
}

/// Write the whole config, creating parent dirs; chmod 600 on unix. Returns the path.
fn save(cfg: &ConfigFile) -> Result<PathBuf> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(cfg)?;
    fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
            .with_context(|| format!("chmod 600 {}", path.display()))?;
    }
    Ok(path)
}

/// Save the token, preserving any existing links. Returns the path.
pub fn save_token(token: &str) -> Result<PathBuf> {
    let mut cfg = load();
    cfg.token = token.to_string();
    save(&cfg)
}

/// Best-effort read of the saved token (trimmed). `None` if absent/empty/unreadable.
pub fn load_token() -> Option<String> {
    let t = load().token.trim().to_string();
    if t.is_empty() {
        None
    } else {
        Some(t)
    }
}

/// All stored links, keyed by canonical absolute model path.
pub fn links() -> BTreeMap<String, LinkEntry> {
    load().links
}

/// Insert or replace the link for `path_key`, preserving the token.
pub fn upsert_link(path_key: &str, entry: LinkEntry) -> Result<()> {
    let mut cfg = load();
    cfg.links.insert(path_key.to_string(), entry);
    save(&cfg)?;
    Ok(())
}

/// Remove the link for `path_key`. Returns whether an entry existed.
pub fn remove_link(path_key: &str) -> Result<bool> {
    let mut cfg = load();
    let existed = cfg.links.remove(path_key).is_some();
    if existed {
        save(&cfg)?;
    }
    Ok(existed)
}

/// The cached hash for `path_key`, if any.
pub fn cached_hash(path_key: &str) -> Option<HashEntry> {
    load().hashes.get(path_key).cloned()
}

/// Insert or replace the hash cache entry for `path_key`.
pub fn store_hash(path_key: &str, entry: HashEntry) -> Result<()> {
    let mut cfg = load();
    cfg.hashes.insert(path_key.to_string(), entry);
    save(&cfg)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test covers the whole round-trip: the config-dir override is process-global
    // (env var), so splitting these into parallel tests would race.
    #[test]
    fn token_and_links_round_trip() {
        let dir = std::env::temp_dir().join(format!("llamabench_cfg_{}", std::process::id()));
        std::env::set_var("LLAMABENCH_CONFIG_DIR", &dir);

        assert!(load_token().is_none());
        save_token("tok-123").unwrap();
        assert_eq!(load_token().as_deref(), Some("tok-123"));

        let entry = LinkEntry {
            repo: "unsloth/gemma-4-12b-it-GGUF".into(),
            file: "gemma-4-12b-it-UD-Q4_K_XL.gguf".into(),
            sha256: "ab".repeat(32),
            size: 42,
            mtime: 1_700_000_000,
            verified: true,
        };
        upsert_link("/models/g.gguf", entry.clone()).unwrap();
        // The token survives a link write, and vice versa.
        assert_eq!(load_token().as_deref(), Some("tok-123"));
        assert_eq!(links().get("/models/g.gguf"), Some(&entry));
        save_token("tok-456").unwrap();
        assert_eq!(links().get("/models/g.gguf"), Some(&entry));

        assert!(remove_link("/models/g.gguf").unwrap());
        assert!(!remove_link("/models/g.gguf").unwrap());
        assert!(links().is_empty());

        std::env::remove_var("LLAMABENCH_CONFIG_DIR");
        let _ = fs::remove_dir_all(&dir);
    }
}
