// SPDX-License-Identifier: GPL-3.0-or-later
//! Optional downloads, all cached on disk under `dirs::cache_dir()/llamabench`:
//!
//! * GGUF models from Hugging Face (`--hf-model <repo> --quant <Q>`).
//! * Prebuilt llama.cpp releases from GitHub (`--download-llama`).
//!
//! Everything **streams** to disk (ureq reader → `std::io::copy`); a multi-GB model
//! is never buffered in memory. Downloads are content-cached: a model is skipped if
//! it already exists at the expected size, and a llama.cpp build is skipped if its
//! release tag is already extracted.
//!
//! Note on llama.cpp release archives: Windows builds ship as `.zip`, but macOS and
//! Linux builds ship as `.tar.gz`, so we handle both.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

const UA: &str = concat!("llamabench-runner/", env!("CARGO_PKG_VERSION"));

fn cache_root() -> Result<PathBuf> {
    let dir =
        dirs::cache_dir().ok_or_else(|| anyhow!("could not determine the per-user cache dir"))?;
    Ok(dir.join("llamabench"))
}

// ---------------------------------------------------------------------------
// Hugging Face
// ---------------------------------------------------------------------------

/// A `.gguf` file resolved on Hugging Face for a given repo + quant.
pub struct HfFile {
    pub filename: String,
    pub url: String,
}

/// Query the HF model API and pick the `.gguf` matching `quant` (case-insensitive).
/// Does not download anything — just resolves the filename + raw URL.
pub fn hf_resolve(repo: &str, quant: &str) -> Result<HfFile> {
    let api = format!("https://huggingface.co/api/models/{repo}");
    let resp = ureq::get(&api)
        .set("User-Agent", UA)
        .call()
        .map_err(|e| anyhow!("Hugging Face API request failed for '{repo}': {e}"))?;
    let v: Value = resp.into_json().context("decoding HF model metadata")?;
    let siblings = v["siblings"]
        .as_array()
        .ok_or_else(|| anyhow!("unexpected HF API response for '{repo}' (no siblings list)"))?;
    let ggufs: Vec<String> = siblings
        .iter()
        .filter_map(|s| s["rfilename"].as_str())
        .filter(|f| f.to_lowercase().ends_with(".gguf"))
        .map(|f| f.to_string())
        .collect();
    if ggufs.is_empty() {
        bail!("no .gguf files found in '{repo}'");
    }
    let file = pick_gguf(&ggufs, quant).ok_or_else(|| {
        anyhow!(
            "no .gguf in '{repo}' matches quant '{quant}'. Available .gguf files:\n  {}",
            ggufs.join("\n  ")
        )
    })?;
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    Ok(HfFile {
        filename: file,
        url,
    })
}

/// Resolve a repo's expected SHA-256 for the `.gguf` matching `quant`, via the HF
/// **tree API** (`/api/models/<repo>/tree/main`). Each entry is `{path, size,
/// lfs:{oid, size}}`; for an LFS-tracked GGUF the `lfs.oid` IS the file's sha256, so
/// it lets us verify a local file's provenance without downloading the blob. Returns
/// `Ok(None)` when no `.gguf` matches the quant (or the match isn't LFS-tracked, so
/// no oid is published).
pub fn hf_expected_sha256(repo: &str, quant: &str) -> Result<Option<String>> {
    let api = format!("https://huggingface.co/api/models/{repo}/tree/main");
    let resp = ureq::get(&api)
        .set("User-Agent", UA)
        .call()
        .map_err(|e| anyhow!("Hugging Face tree API request failed for '{repo}': {e}"))?;
    let v: Value = resp.into_json().context("decoding HF tree metadata")?;
    let entries = v
        .as_array()
        .ok_or_else(|| anyhow!("unexpected HF tree API response for '{repo}' (not a list)"))?;
    let ggufs: Vec<String> = entries
        .iter()
        .filter_map(|e| e["path"].as_str())
        .filter(|p| p.to_lowercase().ends_with(".gguf"))
        .map(|p| p.to_string())
        .collect();
    let Some(file) = pick_gguf(&ggufs, quant) else {
        return Ok(None);
    };
    let oid = entries
        .iter()
        .find(|e| e["path"].as_str() == Some(file.as_str()))
        .and_then(|e| e["lfs"]["oid"].as_str())
        .map(str::to_string);
    Ok(oid)
}

/// Resolve, then stream the model to the cache (skipping if already present at the
/// expected size). Returns the local path to use as `--model`.
pub fn hf_download(repo: &str, quant: &str) -> Result<PathBuf> {
    let hf = hf_resolve(repo, quant)?;
    let dest = cache_root()?.join("models").join(&hf.filename);
    let expected = remote_len(&hf.url);

    if dest.exists() {
        let have = fs::metadata(&dest)?.len();
        if expected.map_or(have > 0, |e| e == have) {
            eprintln!("✓ model cached: {} ({})", dest.display(), human(have));
            return Ok(dest);
        }
        eprintln!(
            "↻ cached {} is the wrong size — re-downloading",
            hf.filename
        );
    }

    eprintln!(
        "↓ downloading {} from Hugging Face ({})…",
        hf.filename,
        expected.map(human).unwrap_or_else(|| "size unknown".into())
    );
    stream_to_file(&hf.url, &dest)?;
    if let Some(e) = expected {
        let got = fs::metadata(&dest)?.len();
        if got != e {
            bail!(
                "download size mismatch for {}: expected {e}, got {got}",
                hf.filename
            );
        }
    }
    eprintln!("✓ saved {}", dest.display());
    Ok(dest)
}

/// Pick the best `.gguf` for `quant`. Requires the quant as a substring
/// (case-insensitive); prefers a clean token-boundary match and, among ties, the
/// shortest (least-decorated) filename. Multi-part split files are de-prioritized.
fn pick_gguf(files: &[String], quant: &str) -> Option<String> {
    let q = quant.to_lowercase();
    let mut best: Option<(i32, usize, &String)> = None; // (score, len) — higher score, then shorter
    for f in files {
        let lf = f.to_lowercase();
        let Some(pos) = lf.find(&q) else { continue };
        let before_ok = pos == 0
            || !lf[..pos]
                .chars()
                .next_back()
                .unwrap()
                .is_ascii_alphanumeric();
        let after = pos + q.len();
        let after_ok = after >= lf.len() || {
            let c = lf[after..].chars().next().unwrap();
            !c.is_ascii_alphanumeric() && c != '_'
        };
        let mut score = 0;
        if before_ok {
            score += 2;
        }
        if after_ok {
            score += 2;
        }
        if lf.contains("-of-") {
            score -= 1; // de-prioritize split-file parts (…-00001-of-00010.gguf)
        }
        let len = f.len();
        let better = match best {
            None => true,
            Some((bs, bl, _)) => score > bs || (score == bs && len < bl),
        };
        if better {
            best = Some((score, len, f));
        }
    }
    best.map(|(_, _, f)| f.clone())
}

// ---------------------------------------------------------------------------
// llama.cpp prebuilt releases
// ---------------------------------------------------------------------------

/// Download (or reuse from cache) the latest prebuilt llama.cpp release for this
/// platform and return the directory containing `llama-bench`/`llama-server`.
///
/// Only the standard CPU/Metal build is selected. GPU builds (CUDA/HIP/Vulkan) are
/// **not** auto-selected — point `--llama-dir` at your own build for full speed.
pub fn download_llama_cpp() -> Result<PathBuf> {
    let api = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";
    let resp = ureq::get(api)
        .set("User-Agent", UA)
        .set("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| anyhow!("GitHub API request failed: {e}"))?;
    let v: Value = resp
        .into_json()
        .context("decoding GitHub release metadata")?;
    let tag = v["tag_name"]
        .as_str()
        .ok_or_else(|| anyhow!("no tag_name in latest llama.cpp release"))?
        .to_string();
    let assets = v["assets"]
        .as_array()
        .ok_or_else(|| anyhow!("no assets in latest llama.cpp release"))?;

    let dest_dir = cache_root()?.join("llama.cpp").join(&tag);
    if let Some(bin) = find_bin_dir(&dest_dir) {
        eprintln!("✓ llama.cpp {tag} cached at {}", bin.display());
        return Ok(bin);
    }

    let needle = asset_needle().ok_or_else(|| {
        anyhow!(
            "no prebuilt llama.cpp for this platform ({}/{}). Build llama.cpp and pass --llama-dir.",
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    })?;
    let names: Vec<String> = assets
        .iter()
        .filter_map(|a| a["name"].as_str().map(String::from))
        .collect();
    let asset = pick_asset(&names, needle).ok_or_else(|| {
        anyhow!(
            "no llama.cpp asset matched '{needle}' in release {tag}. Assets:\n  {}\n\
             Tip: point --llama-dir at your own build (required for a GPU build).",
            names.join("\n  ")
        )
    })?;
    let url = assets
        .iter()
        .find(|a| a["name"].as_str() == Some(asset.as_str()))
        .and_then(|a| a["browser_download_url"].as_str())
        .ok_or_else(|| anyhow!("asset '{asset}' has no download url"))?
        .to_string();

    fs::create_dir_all(&dest_dir).with_context(|| format!("creating {}", dest_dir.display()))?;
    let archive = dest_dir.join(&asset);
    eprintln!("↓ downloading llama.cpp {tag} ({asset})…");
    stream_to_file(&url, &archive)?;
    eprintln!("⇲ extracting {asset}…");
    extract(&archive, &dest_dir)?;
    let _ = fs::remove_file(&archive);

    let bin = find_bin_dir(&dest_dir).ok_or_else(|| {
        anyhow!(
            "extracted '{asset}' but found no llama-bench/llama-server under {}",
            dest_dir.display()
        )
    })?;
    eprintln!("✓ llama.cpp {tag} ready at {}", bin.display());
    Ok(bin)
}

/// The contiguous asset-name substring identifying the plain CPU/Metal build for
/// this OS/arch. `None` if we don't ship a prebuilt for the platform.
fn asset_needle() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => Some("macos-arm64"),
        ("macos", "x86_64") => Some("macos-x64"),
        ("linux", "x86_64") => Some("ubuntu-x64"),
        ("windows", "x86_64") => Some("win-cpu-x64"),
        _ => None,
    }
}

/// Pick the archive asset whose name contains `needle`; the contiguous needle
/// already excludes GPU variants (e.g. `ubuntu-x64` won't match `ubuntu-vulkan-x64`,
/// `macos-arm64` won't match `macos-arm64-something`). Windows ships `.zip`, macOS
/// and Linux ship `.tar.gz`. Among ties, prefer the shortest name.
fn pick_asset(names: &[String], needle: &str) -> Option<String> {
    let mut cands: Vec<&String> = names
        .iter()
        .filter(|n| {
            let l = n.to_lowercase();
            is_archive(&l) && l.contains(needle)
        })
        .collect();
    cands.sort_by_key(|n| n.len());
    cands.into_iter().next().cloned()
}

fn is_archive(lower_name: &str) -> bool {
    lower_name.ends_with(".zip") || lower_name.ends_with(".tar.gz") || lower_name.ends_with(".tgz")
}

/// Recursively find the directory containing the llama-bench binary.
fn find_bin_dir(root: &Path) -> Option<PathBuf> {
    let target = if cfg!(windows) {
        "llama-bench.exe"
    } else {
        "llama-bench"
    };
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = fs::read_dir(&dir) else { continue };
        let mut subdirs = Vec::new();
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                subdirs.push(path);
            } else if path.file_name().and_then(|s| s.to_str()) == Some(target) {
                return Some(dir);
            }
        }
        stack.extend(subdirs);
    }
    None
}

/// Extract a release archive into `dest`, dispatching on its extension.
fn extract(archive: &Path, dest: &Path) -> Result<()> {
    let name = archive
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    if name.ends_with(".zip") {
        extract_zip(archive, dest)
    } else if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        extract_tar_gz(archive, dest)
    } else {
        bail!("unsupported archive type: {}", archive.display())
    }
}

/// Extract a `.tar.gz` into `dest`. `tar`'s `unpack` preserves unix permissions and
/// guards against path-traversal (entries escaping `dest` are skipped).
fn extract_tar_gz(path: &Path, dest: &Path) -> Result<()> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let gz = flate2::read::GzDecoder::new(file);
    let mut ar = tar::Archive::new(gz);
    ar.set_preserve_permissions(true);
    ar.unpack(dest)
        .with_context(|| format!("extracting {} to {}", path.display(), dest.display()))?;
    Ok(())
}

/// Extract a zip into `dest`, preserving unix exec bits and guarding against
/// zip-slip via `enclosed_name`.
fn extract_zip(zip_path: &Path, dest: &Path) -> Result<()> {
    let file = File::open(zip_path).with_context(|| format!("opening {}", zip_path.display()))?;
    let mut archive = zip::ZipArchive::new(file).context("reading zip archive")?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let Some(rel) = entry.enclosed_name() else {
            continue;
        };
        let out = dest.join(rel);
        if entry.is_dir() {
            fs::create_dir_all(&out)?;
            continue;
        }
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut outfile =
            File::create(&out).with_context(|| format!("creating {}", out.display()))?;
        io::copy(&mut entry, &mut outfile)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Some(mode) = entry.unix_mode() {
                fs::set_permissions(&out, fs::Permissions::from_mode(mode))?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared streaming download
// ---------------------------------------------------------------------------

/// HEAD the URL (following redirects) for its Content-Length, if available.
fn remote_len(url: &str) -> Option<u64> {
    ureq::head(url)
        .set("User-Agent", UA)
        .call()
        .ok()
        .and_then(|r| r.header("Content-Length").map(str::to_string))
        .and_then(|s| s.parse().ok())
}

/// Stream `url` to `dest` via a temporary `.part` file, then rename into place.
/// Uses `io::copy` so the body never has to fit in memory.
fn stream_to_file(url: &str, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let resp = ureq::get(url)
        .set("User-Agent", UA)
        .call()
        .map_err(|e| anyhow!("download failed for {url}: {e}"))?;
    let mut reader = resp.into_reader();
    let part = dest.with_file_name(format!(
        "{}.part",
        dest.file_name().and_then(|s| s.to_str()).unwrap_or("dl")
    ));
    let mut file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    io::copy(&mut reader, &mut file).with_context(|| format!("streaming to {}", part.display()))?;
    file.sync_all().ok();
    drop(file);
    fs::rename(&part, dest).with_context(|| format!("finalizing {}", dest.display()))?;
    Ok(())
}

fn human(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{size:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_exact_quant() {
        let files = [
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf".to_string(),
            "Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf".to_string(),
            "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf".to_string(),
        ];
        assert_eq!(
            pick_gguf(&files, "Q4_K_M").unwrap(),
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        );
        // case-insensitive
        assert_eq!(
            pick_gguf(&files, "q8_0").unwrap(),
            "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
        );
        assert!(pick_gguf(&files, "Q2_K").is_none());
    }

    #[test]
    fn prefers_clean_boundary_over_superset() {
        let files = [
            "model-Q4_K_M.gguf".to_string(),
            "model-Q4_K_M_L.gguf".to_string(),
        ];
        assert_eq!(pick_gguf(&files, "Q4_K_M").unwrap(), "model-Q4_K_M.gguf");
    }

    #[test]
    fn picks_plain_cpu_asset_not_gpu() {
        // Real asset shape (b9828): macOS/Linux ship .tar.gz, Windows ships .zip, and
        // GPU variants must be excluded.
        let assets = [
            "llama-b9828-bin-macos-arm64.tar.gz".to_string(),
            "llama-b9828-bin-macos-x64.tar.gz".to_string(),
            "llama-b9828-bin-ubuntu-x64.tar.gz".to_string(),
            "llama-b9828-bin-ubuntu-arm64.tar.gz".to_string(),
            "llama-b9828-bin-ubuntu-vulkan-x64.tar.gz".to_string(),
            "llama-b9828-bin-ubuntu-rocm-7.2-x64.tar.gz".to_string(),
            "llama-b9828-bin-win-cpu-x64.zip".to_string(),
            "llama-b9828-bin-win-cpu-arm64.zip".to_string(),
            "llama-b9828-bin-win-cuda-12.4-x64.zip".to_string(),
            "llama-b9828-bin-win-vulkan-x64.zip".to_string(),
            "cudart-llama-bin-win-cuda-12.4-x64.zip".to_string(),
        ];
        assert_eq!(
            pick_asset(&assets, "macos-arm64").unwrap(),
            "llama-b9828-bin-macos-arm64.tar.gz"
        );
        assert_eq!(
            pick_asset(&assets, "macos-x64").unwrap(),
            "llama-b9828-bin-macos-x64.tar.gz"
        );
        assert_eq!(
            pick_asset(&assets, "ubuntu-x64").unwrap(),
            "llama-b9828-bin-ubuntu-x64.tar.gz"
        );
        assert_eq!(
            pick_asset(&assets, "win-cpu-x64").unwrap(),
            "llama-b9828-bin-win-cpu-x64.zip"
        );
    }

    #[test]
    fn human_sizes() {
        assert_eq!(human(512), "512 B");
        assert_eq!(human(1024), "1.0 KiB");
        assert_eq!(human(1_572_864), "1.5 MiB");
    }

    // Live smoke against the Hugging Face API. Ignored by default (network); run with
    // `cargo test -- --ignored` to confirm resolution without downloading the model.
    #[test]
    #[ignore = "network: hits the live Hugging Face API"]
    fn hf_resolve_live_1b() {
        let f = hf_resolve("bartowski/Llama-3.2-1B-Instruct-GGUF", "Q4_K_M").unwrap();
        assert!(f.filename.to_lowercase().contains("q4_k_m"));
        assert!(f.filename.to_lowercase().ends_with(".gguf"));
        assert!(f.url.contains("/resolve/main/"));
        eprintln!("resolved: {} -> {}", f.filename, f.url);
    }

    // Live smoke against the HF tree API: resolve the published sha256 (lfs.oid) for a
    // tiny GGUF *without downloading the blob*. Ignored by default (network).
    #[test]
    #[ignore = "network: hits the live Hugging Face tree API"]
    fn hf_expected_sha256_live_1b() {
        let oid = hf_expected_sha256("bartowski/Llama-3.2-1B-Instruct-GGUF", "Q4_K_M")
            .unwrap()
            .expect("Q4_K_M gguf should publish an lfs oid");
        assert_eq!(oid.len(), 64, "sha256 is 64 hex chars");
        assert!(oid.chars().all(|c| c.is_ascii_hexdigit()));
        eprintln!("expected sha256: {oid}");
    }
}
