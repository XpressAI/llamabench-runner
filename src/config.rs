// SPDX-License-Identifier: GPL-3.0-or-later
//! Per-user token storage so `run` can submit without `--token` every time.
//!
//! The token lives in `dirs::config_dir()/llamabench/config.json` (e.g.
//! `~/.config/llamabench/config.json` on Linux, `~/Library/Application
//! Support/llamabench/config.json` on macOS). On unix the file is chmod 600.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Default)]
struct ConfigFile {
    #[serde(default)]
    token: String,
}

/// `dirs::config_dir()/llamabench/config.json`.
pub fn config_path() -> Result<PathBuf> {
    let dir = dirs::config_dir()
        .ok_or_else(|| anyhow!("could not determine the per-user config directory"))?;
    Ok(dir.join("llamabench").join("config.json"))
}

/// Write `{"token": …}`, creating parent dirs; chmod 600 on unix. Returns the path.
pub fn save_token(token: &str) -> Result<PathBuf> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let cfg = ConfigFile {
        token: token.to_string(),
    };
    let json = serde_json::to_string_pretty(&cfg)?;
    fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
            .with_context(|| format!("chmod 600 {}", path.display()))?;
    }
    Ok(path)
}

/// Best-effort read of the saved token (trimmed). `None` if absent/empty/unreadable.
pub fn load_token() -> Option<String> {
    let path = config_path().ok()?;
    let data = fs::read_to_string(path).ok()?;
    let cfg: ConfigFile = serde_json::from_str(&data).ok()?;
    let t = cfg.token.trim().to_string();
    if t.is_empty() {
        None
    } else {
        Some(t)
    }
}
