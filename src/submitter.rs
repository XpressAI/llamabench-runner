// SPDX-License-Identifier: GPL-3.0-or-later
//! Everything between "we have benchmark numbers" and "the result is on the
//! leaderboard": model identity + HF provenance, assembling the
//! `ResultSubmission`, signing, and the HTTP submit. Shared by the classic
//! subcommands (`run`/`bench`/`verify`) and the drop-in passthrough modes
//! (ADR-009), which is why nothing in here takes clap args.

use anyhow::{bail, Result};
use clap::ValueEnum;
use sha2::{Digest, Sha256};
use std::path::Path;

use crate::bench::BenchResult;
use crate::config;
use crate::contract::*;
use crate::detect;
use crate::download;
use crate::link;

pub const DEFAULT_API: &str = "https://llamabench.ai/api/results";
pub const DEFAULT_EVAL_API: &str = "https://llamabench.ai/api/evals";

pub fn positive_f64(value: &str) -> std::result::Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| "must be a number".to_string())?;
    if parsed.is_finite() && parsed > 0.0 {
        Ok(parsed)
    } else {
        Err("must be a finite number greater than zero".to_string())
    }
}

/// Which llama.cpp variant a build is from. They share the `llama-bench` /
/// `llama-server` CLI, so the runner drives them identically — but results are
/// recorded under the variant's name so they stay comparable yet distinct.
// They are all llama.cpp variants, so the shared "LlamaCpp" suffix is intentional.
#[allow(clippy::enum_variant_names)]
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Upstream ggml-org/llama.cpp (the default; the only one with prebuilt downloads).
    #[value(name = "llama.cpp")]
    LlamaCpp,
    /// ikawrakow/ik_llama.cpp — CPU/quant-focused fork.
    #[value(name = "ik_llama.cpp")]
    IkLlamaCpp,
    /// beellama.cpp.
    #[value(name = "beellama.cpp")]
    BeeLlamaCpp,
    /// Xpress AI's ve_llama.cpp — adds NEC SX-Aurora Vector Engine support.
    #[value(name = "ve_llama.cpp")]
    VeLlamaCpp,
}

impl Family {
    /// The string recorded as `backend.name` (matches the --family value).
    pub fn backend_name(self) -> &'static str {
        match self {
            Family::LlamaCpp => "llama.cpp",
            Family::IkLlamaCpp => "ik_llama.cpp",
            Family::BeeLlamaCpp => "beellama.cpp",
            Family::VeLlamaCpp => "ve_llama.cpp",
        }
    }

    /// Parse a `--family` value outside clap (the drop-in modes extract it by hand).
    pub fn parse(s: &str) -> Result<Family> {
        <Family as ValueEnum>::from_str(s, true).map_err(|_| {
            anyhow::anyhow!(
                "unknown --family '{s}' (llama.cpp, ik_llama.cpp, beellama.cpp, ve_llama.cpp)"
            )
        })
    }
}

/// The canonical model identity for a submission, resolved from a GGUF repo's HF
/// `base_model` (one level up the model tree = the unquantized finetune/base it
/// quantizes). Lets every GGUF repack of the same model group together. All fields are
/// `None` when there's no HF repo or no resolvable `base_model`, in which case the
/// caller falls back to the per-quant llama-bench label.
#[derive(Default)]
pub struct Canonical {
    /// The full canonical HF repo, e.g. `google/gemma-4-12b-it`. → `ModelInfo.base_model`.
    pub base_model: Option<String>,
    /// `slugify(<basename after '/'>)`, e.g. `gemma-4-12b-it`. → `ModelInfo.id`.
    pub id: Option<String>,
    /// The basename after '/', e.g. `gemma-4-12b-it`. → `ModelInfo.name`.
    pub name: Option<String>,
}

/// Derive the canonical model id + name from a base_model repo: the basename after the
/// last '/' is the `name`, and `slugify(basename)` is the `id`. e.g.
/// `google/gemma-4-12b-it` → (`gemma-4-12b-it`, `gemma-4-12b-it`). A repo with no '/'
/// is treated as its own basename. Returns `(id, name)`.
pub fn canonical_id_name(base_model: &str) -> (String, String) {
    let basename = base_model.rsplit('/').next().unwrap_or(base_model);
    (detect::slugify(basename), basename.to_string())
}

/// Resolve the canonical model identity for a GGUF `repo` via its HF `base_model`. On
/// success, prints a short line and returns the full base repo plus the derived
/// id/name. An absent `base_model` or any network failure yields an empty `Canonical`
/// (the run never fails over it; the caller keeps the llama-bench label).
pub fn resolve_canonical(repo: &str) -> Canonical {
    match download::hf_base_model(repo) {
        Some(base) => {
            let (id, name) = canonical_id_name(&base);
            eprintln!("→ model: {name} (base of {repo})");
            Canonical {
                base_model: Some(base),
                id: Some(id),
                name: Some(name),
            }
        }
        None => Canonical::default(),
    }
}

/// Apply an explicit canonical HF model when the GGUF repository does not publish
/// `cardData.base_model`. The source GGUF repository must still be present in the
/// provenance; this flag changes grouping identity, never artifact attribution.
pub fn explicit_canonical(mut hf: HfProvenance, base_model: Option<&str>) -> Result<HfProvenance> {
    let Some(base_model) = base_model else {
        return Ok(hf);
    };
    let base_model = base_model.trim();
    let valid_segment = |segment: &str| {
        !segment.is_empty()
            && segment.len() <= 96
            && !segment.starts_with(['-', '.'])
            && !segment.ends_with(['-', '.'])
            && !segment.contains("--")
            && !segment.contains("..")
            && !segment.ends_with(".git")
            && segment
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    };
    let valid = base_model
        .split_once('/')
        .is_some_and(|(owner, name)| valid_segment(owner) && valid_segment(name));
    if !valid {
        bail!("--base-model must be a Hugging Face repo in owner/name form");
    }
    if hf.model.is_none() {
        bail!("--base-model requires Hugging Face GGUF provenance (link the local file first, or pass --hf-model)");
    }
    if let Some(authoritative) = hf.canonical.base_model.as_deref() {
        if !authoritative.eq_ignore_ascii_case(base_model) {
            bail!(
                "--base-model {base_model} conflicts with repository metadata base_model {authoritative}"
            );
        }
        return Ok(hf);
    }
    let (id, name) = canonical_id_name(base_model);
    eprintln!(
        "→ model: {name} (explicit base of {})",
        hf.model.as_deref().unwrap_or("GGUF")
    );
    hf.canonical = Canonical {
        base_model: Some(base_model.to_string()),
        id: Some(id),
        name: Some(name),
    };
    Ok(hf)
}

/// Hugging Face provenance recorded on the model: the source repo, whether the bytes
/// are confirmed to come from it, and the canonical (base/finetune) model identity it
/// should be attributed to. Maps to `ModelInfo.hf_model` / `hf_verified` / `base_model`
/// (and the canonical `id`/`name`).
pub struct HfProvenance {
    pub model: Option<String>,
    pub verified: Option<bool>,
    pub canonical: Canonical,
}

impl HfProvenance {
    pub fn none() -> Self {
        HfProvenance {
            model: None,
            verified: None,
            canonical: Canonical::default(),
        }
    }
}

/// Where the benchmarked bytes came from, for provenance purposes.
pub enum ModelSource<'a> {
    /// A local file with no explicit repo — consults the persistent link store (ADR-009).
    LocalOnly(&'a str),
    /// A local file explicitly claimed to come from `repo` — hash-verified per run.
    LocalWithRepo(&'a str, &'a str),
    /// Downloaded straight from `repo` this run ⇒ trivially verified.
    Downloaded(&'a str),
}

/// Decide the HF provenance for this run. Whenever a repo is involved, its canonical
/// model identity is resolved from the repo's `base_model` (see `resolve_canonical`).
pub fn provenance(source: &ModelSource, quant: &str) -> HfProvenance {
    match source {
        ModelSource::Downloaded(repo) => HfProvenance {
            model: Some(repo.to_string()),
            verified: Some(true),
            canonical: resolve_canonical(repo),
        },
        ModelSource::LocalWithRepo(model, repo) => HfProvenance {
            model: Some(repo.to_string()),
            verified: Some(verify_hf_hash(model, repo, quant)),
            canonical: resolve_canonical(repo),
        },
        ModelSource::LocalOnly(model) => match link::resolve(model) {
            Some(l) => HfProvenance {
                model: Some(l.repo.clone()),
                verified: Some(l.verified),
                canonical: resolve_canonical(&l.repo),
            },
            None => HfProvenance::none(),
        },
    }
}

/// Provenance for an artifact whose current bytes have already been freshly
/// hashed. This avoids trusting a size-only download cache or a size+mtime link
/// cache when an exact-artifact evaluation is attributed.
pub fn provenance_exact(source: &ModelSource, quant: &str, sha256: &str) -> HfProvenance {
    match source {
        ModelSource::Downloaded(repo) | ModelSource::LocalWithRepo(_, repo) => HfProvenance {
            model: Some(repo.to_string()),
            verified: Some(verify_hf_sha256(repo, quant, sha256)),
            canonical: resolve_canonical(repo),
        },
        ModelSource::LocalOnly(model) => match link::resolve_with_sha256(model, sha256) {
            Some(link) => HfProvenance {
                model: Some(link.repo.clone()),
                verified: Some(link.verified),
                canonical: resolve_canonical(&link.repo),
            },
            None => HfProvenance::none(),
        },
    }
}

/// Verify a local GGUF against the HF repo it claims to be, by SHA-256. HF publishes
/// each LFS file's sha256 as its `lfs.oid` (tree API), so we stream-hash the local
/// file and compare — no re-download. Network/resolution failures are non-fatal: they
/// just mean "unverified" (`false`), so a run never fails over provenance.
fn verify_hf_hash(model: &str, repo: &str, quant: &str) -> bool {
    let local = match link::file_sha256(Path::new(model)) {
        Ok(h) => h,
        Err(e) => {
            eprintln!(
                "⚠ HF verify: could not hash local file {model} ({e}) — recording as unverified"
            );
            return false;
        }
    };
    verify_hf_sha256(repo, quant, &local)
}

fn verify_hf_sha256(repo: &str, quant: &str, local: &str) -> bool {
    match download::hf_expected_sha256(repo, quant) {
        Ok(Some(expected)) if local.eq_ignore_ascii_case(&expected) => {
            eprintln!("✓ HF hash verified: matches {repo}");
            true
        }
        Ok(Some(_)) => {
            eprintln!("⚠ HF hash MISMATCH: exact artifact differs from {repo}");
            false
        }
        Ok(None) => {
            eprintln!(
                "⚠ HF verify: no .gguf in {repo} matches quant '{quant}' — recording as unverified"
            );
            false
        }
        Err(e) => {
            eprintln!(
                "⚠ HF verify: could not fetch {repo} file hash ({e}) — recording as unverified"
            );
            false
        }
    }
}

/// llama.cpp quant from the GGUF filename, e.g. "…-Q4_K_XL.gguf" → "Q4_K_XL".
pub fn quant_from_path(model: &str) -> String {
    let stem = Path::new(model)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let toks: Vec<&str> = stem.split('-').collect();
    let is_quant = |tok: &str| {
        (tok.starts_with('Q') || tok.starts_with("IQ"))
            && tok == tok.to_uppercase()
            && tok.chars().any(|c| c.is_ascii_digit())
    };
    match toks.iter().position(|t| is_quant(t)) {
        // Keep an Unsloth "UD" (Unsloth Dynamic) prefix — it's part of the quant identity,
        // so `…-UD-Q4_K_XL.gguf` records as `UD-Q4_K_XL`, not `Q4_K_XL`.
        Some(i) if i > 0 && toks[i - 1].eq_ignore_ascii_case("UD") => {
            format!("UD-{}", toks[i])
        }
        Some(i) => toks[i].to_string(),
        None => "unknown".to_string(),
    }
}

pub fn model_name(model: &str) -> String {
    Path::new(model)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// The quant to record. The actual file's name is authoritative (it preserves variants
/// like Unsloth Dynamic `UD-Q4_K_XL`), so we parse it from the file first; a `--quant`
/// flag (which is mainly the HF selector) is only the fallback when the name has no
/// parseable quant.
pub fn resolved_quant(quant_flag: Option<&str>, model: &str) -> String {
    let from_file = quant_from_path(model);
    if from_file != "unknown" {
        return from_file;
    }
    quant_flag
        .map(str::trim)
        .filter(|q| !q.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| "unknown".to_string())
}

/// Approximate parameter count (in B) from a model/repo name, e.g.
/// "gemma-4-12b-it-UD-Q4_K_XL" → 12.0, "Llama-3.2-1B-Instruct" → 1.0. Used by the
/// llama-server drop-in mode, which has no llama-bench row to read the real count
/// from. 0.0 when nothing in the name looks like a size.
pub fn params_from_name(name: &str) -> f64 {
    let lower = name.to_lowercase();
    lower
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '.'))
        .filter_map(|tok| {
            tok.strip_suffix('b')
                .filter(|rest| !rest.is_empty())
                .and_then(|rest| rest.parse::<f64>().ok())
        })
        // MoE names often include both total and active sizes (35B-A3B). The
        // catalog's `params` is total parameters, so the largest B-valued token
        // is the least-wrong filename-only fallback.
        .fold(0.0, f64::max)
}

/// Everything the caller (classic or drop-in) decides about a submission; the
/// measured numbers ride in as a `BenchResult`.
pub struct BuildCtx<'a> {
    /// Whether the run used an accelerator (`-ngl != 0` / a device banner appeared) —
    /// gates the Apple-chip device naming and the GPU-name fallback.
    pub gpu_run: bool,
    /// Explicit llama.cpp device selector (`CUDA1`, UUID, etc.), when supplied. This
    /// keeps per-device properties tied to the GPU that actually ran the benchmark.
    pub selected_device: Option<String>,
    /// Whether the native configuration distributes work across visible GPUs.
    /// `--split-mode none` explicitly disables this even when several cards are visible.
    pub group_visible_gpus: bool,
    pub handle: &'a str,
    pub family: Family,
    /// The exact reproduce command recorded on the result (paths/keys already redacted).
    pub command: String,
    pub quant: &'a str,
    pub model_path: &'a str,
    pub active_params: Option<f64>,
    pub context_length: u32,
    pub spec_decode: String,
    pub ttft_ms: Option<u32>,
}

fn validate_active_params(active_params: Option<f64>, total_params: f64) -> Result<()> {
    let Some(active) = active_params else {
        return Ok(());
    };
    if !active.is_finite() || active <= 0.0 {
        bail!("active parameters must be a finite number greater than zero");
    }
    if total_params.is_finite() && total_params > 0.0 && active > total_params {
        bail!(
            "active parameters ({active}B) cannot exceed measured total parameters ({total_params:.3}B)"
        );
    }
    Ok(())
}

pub fn build_submission(
    ctx: &BuildCtx,
    b: &BenchResult,
    v: Option<Verification>,
    hf: HfProvenance,
) -> Result<ResultSubmission> {
    validate_active_params(ctx.active_params, b.params_b)?;
    // On Apple Silicon the GPU is the chip, and the Metal banner reads noisily as
    // "MTL0 (Apple M4)"; sysctl gives the clean canonical name, so prefer it for GPU runs.
    let mut device = ctx
        .gpu_run
        .then(detect::apple_chip)
        .flatten()
        .or_else(|| b.devices.first().cloned())
        // Banner gave nothing and it's a GPU run — ask the system (nvidia-smi) rather than
        // mislabeling a GPU run as "CPU".
        .or_else(|| {
            if ctx.gpu_run {
                detect::gpu_name()
            } else {
                None
            }
        })
        .unwrap_or_else(|| "CPU".to_string());
    // Canonical model identity (from the GGUF's HF base_model) when we resolved one, so
    // every GGUF repack of the same model groups together; otherwise fall back to the
    // per-quant llama-bench label (slugified).
    let HfProvenance {
        model: hf_model,
        verified: hf_verified,
        canonical,
    } = hf;
    let Canonical {
        base_model,
        id: canonical_id,
        name: canonical_name,
    } = canonical;
    let (model_id, name) = match (canonical_id, canonical_name) {
        (Some(id), Some(name)) => (id, name),
        _ => {
            let label = model_name(ctx.model_path);
            (detect::slugify(&label), label)
        }
    };
    let vendor = detect::vendor_of(&device);
    // Apple is unified memory (≈ usable GPU memory). NVIDIA VRAM is measured so
    // same-name capacity variants such as the RTX 4060 Ti stay distinct. The server
    // fills catalog memory for other discrete GPUs.
    let vram_gb = if vendor == "Apple" {
        detect::apple_unified_mem_gb()
    } else if vendor == "NVIDIA" {
        if let Some((count, total_gib)) = ctx
            .group_visible_gpus
            .then(|| {
                detect::nvidia_gpu_group(&device, ctx.selected_device.as_deref(), &b.backend_label)
            })
            .flatten()
        {
            if count > 1 {
                device = format!("{count}× {device}");
            }
            total_gib as f64
        } else {
            detect::nvidia_vram_gb(&device, ctx.selected_device.as_deref(), &b.backend_label)
                .map_or(0.0, |gib| gib as f64)
        }
    } else {
        0.0
    };
    Ok(ResultSubmission {
        schema_version: SCHEMA_VERSION,
        hardware: Hardware {
            id: detect::slugify(&device),
            name: device.clone(),
            vendor: vendor.to_string(),
            vram_gb,
            bandwidth_gbs: 0.0,
            cpu: detect::cpu_name(),
            system_ram_gb: detect::system_ram_gb(),
        },
        model: ModelInfo {
            id: model_id,
            name,
            params: b.params_b,
            active_params: ctx.active_params,
            base_model,
            hf_model,
            hf_verified,
            // Recorded for every local file (ADR-010): the server attaches web-side
            // provenance by this hash. None when the path isn't a readable file
            // (e.g. the server drop-in's -hf label).
            gguf_sha256: link::sha256_for(ctx.model_path),
        },
        metrics: Metrics {
            decode_tps: b.decode_tps,
            prefill_tps: b.prefill_tps,
            ttft_ms: ctx.ttft_ms,
        },
        config: Config {
            quant: ctx.quant.to_string(),
            kv_cache: b.type_k.clone(),
            context_length: ctx.context_length,
            flash_attention: b.flash_attn,
            spec_decode: ctx.spec_decode.clone(),
            command: Some(ctx.command.clone()),
        },
        backend: Backend {
            name: ctx.family.backend_name().to_string(),
            version: b.build_number.clone(),
            git_hash: b.git_hash.clone(),
        },
        verification: v,
        submitter: Submitter {
            handle: ctx.handle.to_string(),
        },
        signature: String::new(),
    })
}

pub fn emit(s: &ResultSubmission) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(s)?);
    Ok(())
}

/// Set `signature` to the sha256 of the canonical payload (with signature empty). The
/// server treats it as a content fingerprint; true per-user signing is future work.
pub fn sign(s: &mut ResultSubmission) -> Result<()> {
    s.signature = String::new();
    let canonical = serde_json::to_string(s)?;
    let mut h = Sha256::new();
    h.update(canonical.as_bytes());
    s.signature = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    Ok(())
}

pub fn submit(api: &str, token: &str, s: &ResultSubmission) -> Result<()> {
    let resp = ureq::post(api)
        .set("Authorization", &format!("Bearer {token}"))
        .send_json(s)
        .map_err(|e| anyhow::anyhow!("submit failed: {e}"))?;
    let body: serde_json::Value = resp.into_json()?;
    // The API returns { ok, id, url } with `url` an absolute result link. Print it as
    // a clean, clickable line; fall back to the raw body if it's missing.
    match body.get("url").and_then(serde_json::Value::as_str) {
        Some(url) => eprintln!("✓ Submitted: {url}"),
        None => eprintln!("✓ submitted: {body}"),
    }
    Ok(())
}

pub fn submit_eval(api: &str, token: &str, s: &EvaluationSubmission) -> Result<()> {
    let resp = ureq::post(api)
        .set("Authorization", &format!("Bearer {token}"))
        .send_json(s)
        .map_err(|e| anyhow::anyhow!("evaluation submit failed: {e}"))?;
    let body: serde_json::Value = resp.into_json()?;
    match body.get("url").and_then(serde_json::Value::as_str) {
        Some(url) => eprintln!("✓ Submitted evaluation: {url}"),
        None => eprintln!("✓ submitted evaluation: {body}"),
    }
    Ok(())
}

/// Token resolution order: explicit flag → LLAMABENCH_TOKEN env → saved config file.
/// (The classic path lets clap fold the env var into the flag; the drop-in path calls
/// this with just the extracted flag, so the env var is checked here too.)
pub fn resolve_token(explicit: Option<&str>) -> Result<String> {
    if let Some(t) = explicit.map(str::trim).filter(|t| !t.is_empty()) {
        return Ok(t.to_string());
    }
    if let Some(t) = std::env::var("LLAMABENCH_TOKEN")
        .ok()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
    {
        return Ok(t);
    }
    if let Some(t) = config::load_token() {
        return Ok(t);
    }
    bail!(
        "no token. Run `llamabench auth <token>` (get one at https://llamabench.ai/account), \
         or pass --token / set LLAMABENCH_TOKEN."
    )
}

/// Resolve the directory holding the required llama.cpp binaries: an explicit
/// `--llama-dir`, the binaries already on PATH, or a freshly downloaded prebuilt
/// (when `download_llama` is set or PATH has none). Empty string ⇒ use PATH.
pub fn resolve_llama_dir(
    llama_dir: &str,
    download_llama: bool,
    family: Family,
    required: &[&str],
) -> Result<String> {
    if !llama_dir.is_empty() {
        return Ok(llama_dir.to_string());
    }
    let on_path = required.iter().all(|b| find_on_path(b));
    // Only upstream llama.cpp has prebuilt downloads. For a fork, use its build:
    // an explicit --llama-dir (above) or its binaries on PATH — never auto-download.
    if family != Family::LlamaCpp {
        if on_path {
            return Ok(String::new());
        }
        bail!(
            "{} has no prebuilt download — build it and pass \
             --llama-dir <path-to/build/bin> (or put its {} on PATH).",
            family.backend_name(),
            required.join("/")
        );
    }
    if !download_llama {
        if on_path {
            return Ok(String::new());
        }
        eprintln!(
            "note: {} not found on PATH — fetching the prebuilt CPU/Metal llama.cpp \
             (point --llama-dir at your own build for a GPU/full-speed run)",
            required.join("/")
        );
    }
    Ok(download::download_llama_cpp()?
        .to_string_lossy()
        .into_owned())
}

/// Is `name` (or `name.exe` on Windows) an executable on PATH?
pub fn find_on_path(name: &str) -> bool {
    let exe = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|p| p.join(&exe).is_file()))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_identity_from_base_model() {
        // org/name → basename `name`; id is the slug of the basename.
        let (id, name) = canonical_id_name("google/gemma-4-12b-it");
        assert_eq!(name, "gemma-4-12b-it");
        assert_eq!(id, "gemma-4-12b-it");
        // No slash → the whole repo string is its own basename.
        let (id, name) = canonical_id_name("gemma-4-12b-it");
        assert_eq!(name, "gemma-4-12b-it");
        assert_eq!(id, "gemma-4-12b-it");
        // Only the last path segment is the basename; the id is slugified.
        let (id, name) = canonical_id_name("Org/Sub/Gemma 4 12B It");
        assert_eq!(name, "Gemma 4 12B It");
        assert_eq!(id, "gemma-4-12b-it");
    }

    #[test]
    fn explicit_canonical_requires_repo_provenance() {
        let err = explicit_canonical(HfProvenance::none(), Some("ornith-ai/Ornith-1.5-9B"))
            .err()
            .expect("missing GGUF provenance must fail");
        assert!(err
            .to_string()
            .contains("requires Hugging Face GGUF provenance"));

        let hf = HfProvenance {
            model: Some("ornith-ai/Ornith-1.5-9B-GGUF".to_string()),
            verified: Some(true),
            canonical: Canonical::default(),
        };
        let resolved = explicit_canonical(hf, Some("ornith-ai/Ornith-1.5-9B")).unwrap();
        assert_eq!(resolved.canonical.id.as_deref(), Some("ornith-1-5-9b"));
        assert_eq!(resolved.canonical.name.as_deref(), Some("Ornith-1.5-9B"));
        assert_eq!(
            resolved.canonical.base_model.as_deref(),
            Some("ornith-ai/Ornith-1.5-9B")
        );

        let authoritative = HfProvenance {
            model: Some("publisher/quantized".to_string()),
            verified: Some(true),
            canonical: Canonical {
                base_model: Some("publisher/canonical".to_string()),
                id: Some("canonical".to_string()),
                name: Some("canonical".to_string()),
            },
        };
        let err = explicit_canonical(authoritative, Some("attacker/unrelated"))
            .err()
            .expect("conflicting repository metadata must win");
        assert!(err
            .to_string()
            .contains("conflicts with repository metadata"));

        for invalid in [
            "owner/.",
            "owner/---",
            "owner/   ",
            "owner/bad--name",
            "owner/bad..name",
            "owner/name.git",
            "owner/name/extra",
        ] {
            let hf = HfProvenance {
                model: Some("publisher/quantized".to_string()),
                verified: Some(true),
                canonical: Canonical::default(),
            };
            assert!(
                explicit_canonical(hf, Some(invalid)).is_err(),
                "{invalid} must be rejected"
            );
        }
    }

    #[test]
    fn quant_parsing() {
        // Unsloth Dynamic "UD" prefix is preserved (it's a distinct quant recipe).
        assert_eq!(
            quant_from_path("/x/Qwen3.5-4B-UD-Q4_K_XL.gguf"),
            "UD-Q4_K_XL"
        );
        assert_eq!(
            quant_from_path("/x/gemma-4-12b-it-UD-Q4_K_XL.gguf"),
            "UD-Q4_K_XL"
        );
        assert_eq!(
            quant_from_path("/x/Meta-Llama-3.1-8B-Q4_K_M.gguf"),
            "Q4_K_M"
        );
        assert_eq!(quant_from_path("/x/model-IQ4_XS.gguf"), "IQ4_XS");
        assert_eq!(quant_from_path("/x/plain.gguf"), "unknown");
    }

    #[test]
    fn family_backend_names() {
        // These strings are the recorded backend.name and the --family values; the
        // leaderboard groups on them, so pin them.
        assert_eq!(Family::LlamaCpp.backend_name(), "llama.cpp");
        assert_eq!(Family::IkLlamaCpp.backend_name(), "ik_llama.cpp");
        assert_eq!(Family::BeeLlamaCpp.backend_name(), "beellama.cpp");
        assert_eq!(Family::VeLlamaCpp.backend_name(), "ve_llama.cpp");
        // The by-hand parser (drop-in wrapper flags) matches the clap values.
        assert!(matches!(
            Family::parse("ik_llama.cpp").unwrap(),
            Family::IkLlamaCpp
        ));
        assert!(Family::parse("mistral.rs").is_err());
    }

    #[test]
    fn params_heuristic() {
        assert_eq!(params_from_name("gemma-4-12b-it-UD-Q4_K_XL"), 12.0);
        assert_eq!(params_from_name("Llama-3.2-1B-Instruct-Q4_K_M"), 1.0);
        assert_eq!(params_from_name("Qwen3.5-0.6B"), 0.6);
        assert_eq!(params_from_name("Ornith-1.0-35B-A3B"), 35.0);
        // "3.1" (version) must not be read as a size; "8B" wins.
        assert_eq!(params_from_name("Meta-Llama-3.1-8B"), 8.0);
        assert_eq!(params_from_name("mystery-model"), 0.0);
    }

    #[test]
    fn active_params_cannot_exceed_measured_total() {
        assert!(validate_active_params(Some(3.0), 35.0).is_ok());
        assert!(validate_active_params(Some(40.0), 35.0)
            .unwrap_err()
            .to_string()
            .contains("cannot exceed"));
        // Server-mode filename heuristics may not recover a total; retain the
        // positive/finite validation but defer the upper bound when total is unknown.
        assert!(validate_active_params(Some(3.0), 0.0).is_ok());
        assert!(validate_active_params(Some(f64::INFINITY), 35.0).is_err());
    }
}
