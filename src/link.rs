// SPDX-License-Identifier: GPL-3.0-or-later
//! `llamabench link` (ADR-009): tie a local GGUF to the Hugging Face repo it came
//! from, verified by hash. The file is streamed through SHA-256 and matched against
//! the repo's published LFS oids (tree API — no blob download); the link persists in
//! the per-user config, so every later run of that file carries `hfModel` /
//! `hfVerified` provenance without re-typing (or re-hashing — size+mtime gate a
//! re-check).

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::UNIX_EPOCH;

use crate::config::{self, HashEntry, LinkEntry};
use crate::download;

/// A link resolved for a run: the repo to attribute, and whether the local bytes
/// were confirmed to be a file of that repo.
pub struct Resolved {
    pub repo: String,
    pub verified: bool,
}

/// Stream a file through SHA-256, returning lowercase hex. Uses `io::copy` into the
/// hasher so a multi-GB model is never read into memory at once.
pub fn file_sha256(path: &Path) -> Result<String> {
    let file =
        std::fs::File::open(path).with_context(|| format!("opening {} to hash", path.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = Sha256::new();
    std::io::copy(&mut reader, &mut hasher)
        .with_context(|| format!("hashing {}", path.display()))?;
    Ok(hex(&hasher.finalize()))
}

/// Same, with the download progress bar (hashing a 20 GB GGUF takes a while).
fn file_sha256_progress(path: &Path, total: u64) -> Result<String> {
    let file =
        std::fs::File::open(path).with_context(|| format!("opening {} to hash", path.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = Sha256::new();
    download::copy_with_progress(&mut reader, &mut hasher, Some(total))
        .with_context(|| format!("hashing {}", path.display()))?;
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Canonical absolute path string — the link-store key.
fn key_for(path: &str) -> Result<String> {
    let canon = std::fs::canonicalize(path)
        .with_context(|| format!("resolving {path} (does the file exist?)"))?;
    Ok(canon.to_string_lossy().into_owned())
}

fn file_meta(key: &str) -> Result<(u64, i64)> {
    let meta = std::fs::metadata(key).with_context(|| format!("reading metadata of {key}"))?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    Ok((meta.len(), mtime))
}

fn short(path: &str) -> &str {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
}

/// `llamabench link <path> <repo>` — hash, verify against the repo, persist.
pub fn cmd_link(path: &str, repo: &str) -> Result<()> {
    let key = key_for(path)?;
    let (size, mtime) = file_meta(&key)?;
    // Print the name the user typed — the canonical key may be an opaque
    // hash-named blob (e.g. inside the Hugging Face cache).
    eprintln!("  hashing {} …", short(path));
    let sha256 = file_sha256_progress(Path::new(&key), size)?;
    let (file, verified) = match download::hf_file_by_sha256(repo, &sha256)? {
        Some(f) => {
            eprintln!("✓ hash verified: {} is {repo}/{f}", short(path));
            (f, true)
        }
        None => {
            if download::hf_has_gguf(repo) {
                eprintln!(
                    "⚠ NOT verified: no file in {repo} has this sha256 — linking anyway; \
                     submissions will carry hfVerified: false"
                );
            } else {
                eprintln!(
                    "⚠ NOT verified: {repo} publishes no LFS-tracked .gguf to compare against — \
                     linking anyway; submissions will carry hfVerified: false"
                );
            }
            (String::new(), false)
        }
    };
    config::upsert_link(
        &key,
        LinkEntry {
            repo: repo.to_string(),
            file,
            sha256,
            size,
            mtime,
            verified,
        },
    )?;
    eprintln!("✓ linked {key} → {repo}");
    Ok(())
}

/// `llamabench link <path>` — show (and freshness-check) an existing link.
pub fn cmd_status(path: &str) -> Result<()> {
    let key = key_for(path)?;
    match config::links().get(&key) {
        Some(_) => match resolve(&key) {
            Some(r) => {
                println!(
                    "{key} → {} ({})",
                    r.repo,
                    if r.verified {
                        "verified"
                    } else {
                        "NOT verified"
                    }
                );
                Ok(())
            }
            None => bail!("could not re-verify the link for {key}"),
        },
        None => bail!(
            "{key} is not linked. Link it with:\n  llamabench link {path} <hf-user/repo-GGUF>"
        ),
    }
}

/// `llamabench link --list`.
pub fn cmd_list() -> Result<()> {
    let links = config::links();
    if links.is_empty() {
        println!(
            "no linked models. Link one with: llamabench link <model.gguf> <hf-user/repo-GGUF>"
        );
        return Ok(());
    }
    for (path, e) in links {
        println!(
            "{path} → {} ({})",
            e.repo,
            if e.verified {
                "verified"
            } else {
                "NOT verified"
            }
        );
    }
    Ok(())
}

/// `llamabench link --forget <path>`. Falls back to the raw string when the file no
/// longer exists (so a deleted model can still be unlinked).
pub fn cmd_forget(path: &str) -> Result<()> {
    let key = key_for(path).unwrap_or_else(|_| path.to_string());
    if config::remove_link(&key)? {
        eprintln!("✓ forgot {key}");
        Ok(())
    } else {
        bail!("no link stored for {key}")
    }
}

/// The model file's SHA-256 for `model.ggufSha256` (ADR-010) — every submission of
/// a local file carries it so the server (and one web link) can attach provenance.
/// Sources, cheapest first: a stored link whose size/mtime still match, the hash
/// cache, else a fresh streaming hash (once — cached afterwards). Never fails a
/// run: any error yields `None`.
pub fn sha256_for(model: &str) -> Option<String> {
    let key = key_for(model).ok()?;
    let (size, mtime) = file_meta(&key).ok()?;
    let fresh = |e_size: u64, e_mtime: i64| e_size == size && e_mtime == mtime;
    if let Some(e) = config::links().get(&key) {
        if fresh(e.size, e.mtime) {
            return Some(e.sha256.clone());
        }
    }
    if let Some(e) = config::cached_hash(&key) {
        if fresh(e.size, e.mtime) {
            return Some(e.sha256);
        }
    }
    eprintln!("  hashing {} for provenance (once per file)…", short(model));
    let sha256 = match file_sha256_progress(Path::new(&key), size) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("⚠ could not hash ({e}) — submitting without ggufSha256");
            return None;
        }
    };
    let _ = config::store_hash(
        &key,
        HashEntry {
            sha256: sha256.clone(),
            size,
            mtime,
        },
    );
    Some(sha256)
}

/// Resolve the stored link for a model path at run time. Size+mtime unchanged ⇒
/// reuse the stored verification (no re-hash). Changed ⇒ re-hash and re-verify
/// against the linked repo, updating the store. `None` when the path isn't linked.
/// Never fails a run: any error on the re-check path degrades to `verified: false`.
pub fn resolve(model: &str) -> Option<Resolved> {
    let key = key_for(model).ok()?;
    let entry = config::links().get(&key).cloned()?;
    let Ok((size, mtime)) = file_meta(&key) else {
        return None;
    };
    if size == entry.size && mtime == entry.mtime {
        eprintln!(
            "→ provenance: {} (linked, {})",
            entry.repo,
            if entry.verified {
                "hash-verified"
            } else {
                "NOT verified"
            }
        );
        return Some(Resolved {
            repo: entry.repo,
            verified: entry.verified,
        });
    }
    eprintln!(
        "↻ {} changed since it was linked — re-verifying against {}",
        short(&key),
        entry.repo
    );
    let sha256 = match file_sha256_progress(Path::new(&key), size) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("⚠ could not re-hash ({e}) — recording as unverified");
            return Some(Resolved {
                repo: entry.repo,
                verified: false,
            });
        }
    };
    let matched = download::hf_file_by_sha256(&entry.repo, &sha256)
        .ok()
        .flatten();
    let verified = matched.is_some();
    eprintln!(
        "{}",
        if verified {
            "✓ hash verified"
        } else {
            "⚠ hash no longer matches the linked repo — recording as unverified"
        }
    );
    let repo = entry.repo.clone();
    let _ = config::upsert_link(
        &key,
        LinkEntry {
            repo: entry.repo,
            file: matched.unwrap_or_default(),
            sha256,
            size,
            mtime,
            verified,
        },
    );
    Some(Resolved { repo, verified })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_sha256_streaming_matches_known() {
        // SHA-256("abc") — the canonical NIST test vector.
        let mut path = std::env::temp_dir();
        path.push(format!("llamabench_sha256_{}.bin", std::process::id()));
        std::fs::write(&path, b"abc").unwrap();
        let got = file_sha256(&path);
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            got.unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
