// SPDX-License-Identifier: GPL-3.0-or-later
//! llamabench.ai benchmark runner (ADR-004, ADR-005).
//!
//! Drives the user's *existing* llama.cpp build — `llama-bench` for standardized
//! speed (prefill/decode), and `llama-server` for multi-turn deterministic
//! output-correctness verification — then assembles a `ResultSubmission` conforming
//! to `the llamabench result contract`. It bundles nothing.
//!
//! Phase 4: produces the result locally (print / `--dry-run`). Signing + HTTP submit
//! land alongside the API (Phase 3).

mod bench;
mod config;
mod contract;
mod detect;
mod download;
mod verify;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use sha2::{Digest, Sha256};
use std::path::Path;

use bench::{run_llama_bench, BenchOpts, BenchResult};
use contract::*;
use verify::{run_verification, VerifyOpts};

const DEFAULT_API: &str = "https://llamabench.ai/api/results";

#[derive(Parser)]
#[command(
    name = "llamabench",
    version,
    about = "Benchmark your local LLM setup and publish the result."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Save your llamabench.ai token so `run` can submit without --token.
    Auth(AuthArgs),
    /// Run llama-bench and print the result (speed only).
    Bench(RunArgs),
    /// Run the output-correctness verification against llama-server.
    Verify(RunArgs),
    /// Full run: speed + verification → a complete ResultSubmission.
    Run(RunArgs),
}

#[derive(Args)]
struct AuthArgs {
    /// The CLI token from https://llamabench.ai/account. If omitted, read from stdin.
    token: Option<String>,
}

/// Which llama.cpp variant a build is from. They share the `llama-bench` /
/// `llama-server` CLI, so the runner drives them identically — but results are
/// recorded under the variant's name so they stay comparable yet distinct.
// They are all llama.cpp variants, so the shared "LlamaCpp" suffix is intentional.
#[allow(clippy::enum_variant_names)]
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
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
    fn backend_name(self) -> &'static str {
        match self {
            Family::LlamaCpp => "llama.cpp",
            Family::IkLlamaCpp => "ik_llama.cpp",
            Family::BeeLlamaCpp => "beellama.cpp",
            Family::VeLlamaCpp => "ve_llama.cpp",
        }
    }
}

#[derive(Args, Clone)]
struct RunArgs {
    /// Directory containing llama-bench / llama-server. Default: search PATH, else
    /// auto-download a prebuilt CPU/Metal build (see --download-llama).
    #[arg(long, default_value = "")]
    llama_dir: String,
    /// Download the latest prebuilt llama.cpp (CPU/Metal) instead of using PATH.
    /// NOTE: GPU builds (CUDA/HIP/Vulkan) are NOT auto-selected — point --llama-dir
    /// at your own GPU build for full speed.
    #[arg(long)]
    download_llama: bool,
    /// Which llama.cpp variant the build is. Recorded as the backend so results
    /// stay comparable but distinct. The forks (ik_llama.cpp, beellama.cpp,
    /// ve_llama.cpp) share the same CLI — build one and point --llama-dir at it
    /// (only upstream llama.cpp can be auto-downloaded).
    #[arg(long, value_enum, default_value = "llama.cpp")]
    family: Family,
    /// Path to a local GGUF model. Combine with --hf-model --quant to record and
    /// hash-verify the file's Hugging Face provenance (the local bytes are still used).
    #[arg(long)]
    model: Option<String>,
    /// Hugging Face repo for the GGUF, e.g. bartowski/Llama-3.1-8B-Instruct-GGUF
    /// (requires --quant). WITHOUT --model the file is downloaded from here; WITH
    /// --model the local file is used but its SHA-256 is verified against this repo.
    /// The submission is attributed to the GGUF's base/finetune model (its HF
    /// base_model) so every GGUF repack of the same model groups together.
    #[arg(long)]
    hf_model: Option<String>,
    /// Quantization, e.g. Q4_K_M. Required with --hf-model (selects the .gguf to fetch).
    /// The recorded quant is read from the actual file name (so variants like
    /// UD-Q4_K_XL are preserved); --quant is only a fallback if the name has none.
    #[arg(long)]
    quant: Option<String>,
    /// Submitter handle.
    #[arg(long, default_value = "@anonymous")]
    handle: String,

    // --- llama-bench (speed) ---
    #[arg(long, default_value_t = -1)]
    ngl: i32,
    #[arg(long, default_value = "on")]
    fa: String,
    #[arg(long, default_value = "f16")]
    ctk: String,
    #[arg(long, default_value = "f16")]
    ctv: String,
    #[arg(long, default_value_t = 512)]
    n_prompt: u32,
    #[arg(long, default_value_t = 128)]
    n_gen: u32,
    #[arg(long, default_value = "none")]
    spec_decode: String,

    // --- llama-server (verification) ---
    #[arg(long, default_value_t = 8080)]
    port: u16,
    #[arg(long, default_value = "llamabench")]
    api_key: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 3)]
    turns: u32,
    #[arg(long, default_value_t = 3)]
    reps: u32,
    /// Extra args passed verbatim to llama-server (e.g. spec-decode flags).
    #[arg(long = "server-arg")]
    server_args: Vec<String>,

    /// Detect & build the result without submitting.
    #[arg(long)]
    dry_run: bool,
    /// API endpoint to submit to.
    #[arg(long, default_value = DEFAULT_API)]
    api: String,
    /// CLI token from llamabench.ai/account — required to actually submit.
    #[arg(long, env = "LLAMABENCH_TOKEN")]
    token: Option<String>,
}

fn bench_opts<'a>(a: &'a RunArgs, llama_dir: &'a str, model: &'a str) -> BenchOpts<'a> {
    BenchOpts {
        llama_bin_dir: llama_dir,
        model,
        ngl: a.ngl,
        fa: &a.fa,
        ctk: &a.ctk,
        ctv: &a.ctv,
        n_prompt: a.n_prompt,
        n_gen: a.n_gen,
    }
}

fn verify_opts<'a>(a: &'a RunArgs, llama_dir: &'a str, model: &'a str) -> VerifyOpts<'a> {
    VerifyOpts {
        server_bin_dir: llama_dir,
        model,
        port: a.port,
        api_key: &a.api_key,
        seed: a.seed,
        n_gen: a.n_gen,
        max_turns: a.turns,
        reps: a.reps,
        extra_server_args: &a.server_args,
    }
}

/// Resolve the model path. A local `--model` is always the file we benchmark; if
/// `--hf-model` is also given it only attributes provenance (see `hf_provenance`).
/// With `--hf-model` alone, download the `--quant` GGUF from the repo. At least one
/// of the two must be given.
fn resolve_model(a: &RunArgs) -> Result<String> {
    match (&a.model, &a.hf_model) {
        (None, None) => bail!(
            "no model: pass --model <path.gguf>, or --hf-model <repo> --quant <Q> \
             (e.g. --hf-model bartowski/Llama-3.1-8B-Instruct-GGUF --quant Q4_K_M)"
        ),
        // A local file always wins for the bytes we run; --hf-model alongside it is
        // recorded/verified as provenance, not downloaded.
        (Some(m), _) => Ok(m.clone()),
        (None, Some(repo)) => {
            let quant = a
                .quant
                .as_deref()
                .filter(|q| !q.trim().is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "--hf-model requires --quant <Q> to pick the .gguf (e.g. Q4_K_M)"
                    )
                })?;
            Ok(download::hf_download(repo, quant)?
                .to_string_lossy()
                .into_owned())
        }
    }
}

/// The canonical model identity for a submission, resolved from a GGUF repo's HF
/// `base_model` (one level up the model tree = the unquantized finetune/base it
/// quantizes). Lets every GGUF repack of the same model group together. All fields are
/// `None` when there's no `--hf-model` or no resolvable `base_model`, in which case the
/// caller falls back to the per-quant llama-bench label.
#[derive(Default)]
struct Canonical {
    /// The full canonical HF repo, e.g. `google/gemma-4-12b-it`. → `ModelInfo.base_model`.
    base_model: Option<String>,
    /// `slugify(<basename after '/'>)`, e.g. `gemma-4-12b-it`. → `ModelInfo.id`.
    id: Option<String>,
    /// The basename after '/', e.g. `gemma-4-12b-it`. → `ModelInfo.name`.
    name: Option<String>,
}

/// Derive the canonical model id + name from a base_model repo: the basename after the
/// last '/' is the `name`, and `slugify(basename)` is the `id`. e.g.
/// `google/gemma-4-12b-it` → (`gemma-4-12b-it`, `gemma-4-12b-it`). A repo with no '/'
/// is treated as its own basename. Returns `(id, name)`.
fn canonical_id_name(base_model: &str) -> (String, String) {
    let basename = base_model.rsplit('/').next().unwrap_or(base_model);
    (detect::slugify(basename), basename.to_string())
}

/// Resolve the canonical model identity for a GGUF `repo` via its HF `base_model`. On
/// success, prints a short line and returns the full base repo plus the derived
/// id/name. An absent `base_model` or any network failure yields an empty `Canonical`
/// (the run never fails over it; the caller keeps the llama-bench label).
fn resolve_canonical(repo: &str) -> Canonical {
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

/// Hugging Face provenance recorded on the model: the source repo, whether the bytes
/// are confirmed to come from it, and the canonical (base/finetune) model identity it
/// should be attributed to. Maps to `ModelInfo.hf_model` / `hf_verified` / `base_model`
/// (and the canonical `id`/`name`).
struct HfProvenance {
    model: Option<String>,
    verified: Option<bool>,
    canonical: Canonical,
}

/// Decide the HF provenance for this run:
/// * `--hf-model` alone (download path): the bytes came straight from the repo ⇒ verified.
/// * `--model` + `--hf-model`: hash-verify the local file against the repo (`verify_hf_hash`).
/// * `--model` alone: no provenance.
///
/// Whenever an `--hf-model` GGUF repo is given (either path), we also resolve its
/// canonical model identity from the repo's `base_model` (see `resolve_canonical`).
fn hf_provenance(a: &RunArgs, model: &str, quant: &str) -> HfProvenance {
    match (&a.model, &a.hf_model) {
        (None, Some(repo)) => HfProvenance {
            model: Some(repo.clone()),
            verified: Some(true),
            canonical: resolve_canonical(repo),
        },
        (Some(_), Some(repo)) => HfProvenance {
            model: Some(repo.clone()),
            verified: Some(verify_hf_hash(model, repo, quant)),
            canonical: resolve_canonical(repo),
        },
        _ => HfProvenance {
            model: None,
            verified: None,
            canonical: Canonical::default(),
        },
    }
}

/// Verify a local GGUF against the HF repo it claims to be, by SHA-256. HF publishes
/// each LFS file's sha256 as its `lfs.oid` (tree API), so we stream-hash the local
/// file and compare — no re-download. Network/resolution failures are non-fatal: they
/// just mean "unverified" (`false`), so a run never fails over provenance.
fn verify_hf_hash(model: &str, repo: &str, quant: &str) -> bool {
    let local = match file_sha256(Path::new(model)) {
        Ok(h) => h,
        Err(e) => {
            eprintln!(
                "⚠ HF verify: could not hash local file {model} ({e}) — recording as unverified"
            );
            return false;
        }
    };
    match download::hf_expected_sha256(repo, quant) {
        Ok(Some(expected)) if local.eq_ignore_ascii_case(&expected) => {
            eprintln!("✓ HF hash verified: matches {repo}");
            true
        }
        Ok(Some(_)) => {
            eprintln!("⚠ HF hash MISMATCH: local file differs from {repo} ({model})");
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

/// Stream a file through SHA-256, returning lowercase hex. Uses `io::copy` into the
/// hasher so a multi-GB model is never read into memory at once.
fn file_sha256(path: &Path) -> Result<String> {
    let file =
        std::fs::File::open(path).with_context(|| format!("opening {} to hash", path.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = Sha256::new();
    std::io::copy(&mut reader, &mut hasher)
        .with_context(|| format!("hashing {}", path.display()))?;
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Resolve the directory holding the required llama.cpp binaries: an explicit
/// `--llama-dir`, the binaries already on PATH, or a freshly downloaded prebuilt
/// (when `--download-llama` is set or PATH has none). Empty string ⇒ use PATH.
fn resolve_llama_dir(a: &RunArgs, required: &[&str]) -> Result<String> {
    if !a.llama_dir.is_empty() {
        return Ok(a.llama_dir.clone());
    }
    let on_path = required.iter().all(|b| find_on_path(b));
    // Only upstream llama.cpp has prebuilt downloads. For a fork, use its build:
    // an explicit --llama-dir (above) or its binaries on PATH — never auto-download.
    if a.family != Family::LlamaCpp {
        if on_path {
            return Ok(String::new());
        }
        bail!(
            "{} has no prebuilt download — build it and pass \
             --llama-dir <path-to/build/bin> (or put its {} on PATH).",
            a.family.backend_name(),
            required.join("/")
        );
    }
    if !a.download_llama {
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
fn find_on_path(name: &str) -> bool {
    let exe = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|p| p.join(&exe).is_file()))
        .unwrap_or(false)
}

/// Token resolution order: --token flag → LLAMABENCH_TOKEN env (both via `a.token`,
/// courtesy of clap) → saved config file.
fn resolve_token(a: &RunArgs) -> Result<String> {
    if let Some(t) = a.token.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
        return Ok(t.to_string());
    }
    if let Some(t) = config::load_token() {
        return Ok(t);
    }
    bail!(
        "no token. Run `llamabench auth <token>` (get one at https://llamabench.ai/account), \
         or pass --token / set LLAMABENCH_TOKEN."
    )
}

/// llama.cpp quant from the GGUF filename, e.g. "…-Q4_K_XL.gguf" → "Q4_K_XL".
fn quant_from_path(model: &str) -> String {
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

fn model_name(model: &str) -> String {
    Path::new(model)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// The quant to record. The actual file's name is authoritative (it preserves variants
/// like Unsloth Dynamic `UD-Q4_K_XL`), so we parse it from the file first; `--quant`
/// (which is mainly the HF selector) is only the fallback when the name has no parseable
/// quant. For an `--hf-model` download, `model` is the downloaded file, so this still
/// reflects exactly what ran.
fn resolved_quant(a: &RunArgs, model: &str) -> String {
    let from_file = quant_from_path(model);
    if from_file != "unknown" {
        return from_file;
    }
    a.quant
        .as_deref()
        .map(str::trim)
        .filter(|q| !q.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| "unknown".to_string())
}

fn build_submission(
    a: &RunArgs,
    model: &str,
    quant: &str,
    b: &BenchResult,
    v: Option<Verification>,
    hf: HfProvenance,
) -> ResultSubmission {
    // On Apple Silicon the GPU is the chip, and the Metal banner reads noisily as
    // "MTL0 (Apple M4)"; sysctl gives the clean canonical name, so prefer it for GPU runs.
    let device = (a.ngl != 0)
        .then(detect::apple_chip)
        .flatten()
        .or_else(|| b.devices.first().cloned())
        // Banner gave nothing and it's a GPU run — ask the system (nvidia-smi) rather than
        // mislabeling a GPU run as "CPU".
        .or_else(|| if a.ngl != 0 { detect::gpu_name() } else { None })
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
            let label = model_name(model);
            (detect::slugify(&label), label)
        }
    };
    // Record the model by file name only (./<file>.gguf) — never the submitter's local
    // absolute path, which would leak their home directory.
    let model_file = Path::new(model)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model);
    let command = format!(
        "llama-bench -m ./{} -ngl {} -fa {} -ctk {} -ctv {} -p {} -n {}",
        model_file, a.ngl, a.fa, a.ctk, a.ctv, a.n_prompt, a.n_gen
    );
    let vendor = detect::vendor_of(&device);
    // Apple is unified memory (≈ usable GPU memory); report it so the site shows real VRAM.
    // For discrete GPUs the server fills VRAM/bandwidth from its catalog.
    let vram_gb = if vendor == "Apple" {
        detect::apple_unified_mem_gb()
    } else {
        0.0
    };
    ResultSubmission {
        schema_version: SCHEMA_VERSION,
        hardware: Hardware {
            id: detect::slugify(&device),
            name: device.clone(),
            vendor: vendor.to_string(),
            vram_gb,
            bandwidth_gbs: 0.0,
        },
        model: ModelInfo {
            id: model_id,
            name,
            params: b.params_b,
            base_model,
            hf_model,
            hf_verified,
        },
        metrics: Metrics {
            decode_tps: b.decode_tps,
            prefill_tps: b.prefill_tps,
            ttft_ms: None,
        },
        config: Config {
            quant: quant.to_string(),
            kv_cache: b.type_k.clone(),
            context_length: a.n_prompt,
            flash_attention: b.flash_attn,
            spec_decode: a.spec_decode.clone(),
            command: Some(command),
        },
        backend: Backend {
            name: a.family.backend_name().to_string(),
            version: b.build_number.clone(),
            git_hash: b.git_hash.clone(),
        },
        verification: v,
        submitter: Submitter {
            handle: a.handle.clone(),
        },
        signature: String::new(),
    }
}

fn emit(s: &ResultSubmission) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(s)?);
    Ok(())
}

/// Set `signature` to the sha256 of the canonical payload (with signature empty). The
/// server treats it as a content fingerprint; true per-user signing is future work.
fn sign(s: &mut ResultSubmission) -> Result<()> {
    s.signature = String::new();
    let canonical = serde_json::to_string(s)?;
    let mut h = Sha256::new();
    h.update(canonical.as_bytes());
    s.signature = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    Ok(())
}

fn submit(api: &str, token: &str, s: &ResultSubmission) -> Result<()> {
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Auth(args) => {
            let token = match args.token {
                Some(t) => t.trim().to_string(),
                None => {
                    use std::io::Read;
                    let mut s = String::new();
                    std::io::stdin().read_to_string(&mut s)?;
                    s.trim().to_string()
                }
            };
            if token.is_empty() {
                bail!("no token provided (pass it as an argument or pipe it on stdin)");
            }
            let path = config::save_token(&token)?;
            println!("✓ token saved to {}", path.display());
        }
        Command::Bench(a) => {
            let model = resolve_model(&a)?;
            let dir = resolve_llama_dir(&a, &["llama-bench"])?;
            let b = run_llama_bench(&bench_opts(&a, &dir, &model))?;
            let quant = resolved_quant(&a, &model);
            let hf = hf_provenance(&a, &model, &quant);
            emit(&build_submission(&a, &model, &quant, &b, None, hf))?;
        }
        Command::Verify(a) => {
            let model = resolve_model(&a)?;
            let dir = resolve_llama_dir(&a, &["llama-server"])?;
            let v = run_verification(&verify_opts(&a, &dir, &model))?;
            println!("{}", serde_json::to_string_pretty(&v)?);
            if !v.valid {
                eprintln!("⚠ verification FAILED: gibberish detected — invalid submission");
            }
        }
        Command::Run(a) => {
            // Resolve the token up front (cheap) so we fail fast before any multi-GB
            // download when there's nothing to submit with. Skipped for --dry-run.
            let token = if a.dry_run {
                None
            } else {
                Some(resolve_token(&a)?)
            };
            eprintln!("\n▸ [1/4] Model — resolve & download");
            let model = resolve_model(&a)?;
            let dir = resolve_llama_dir(&a, &["llama-bench", "llama-server"])?;
            let quant = resolved_quant(&a, &model);
            eprintln!("\n▸ [2/4] Benchmark — llama-bench (prefill + decode)");
            let b = run_llama_bench(&bench_opts(&a, &dir, &model))?;
            eprintln!(
                "\n▸ [3/4] Verify — llama-server, {} turns × {} reps (the slow part)",
                a.turns, a.reps
            );
            let v = run_verification(&verify_opts(&a, &dir, &model))?;
            let valid = v.valid;
            let hf = hf_provenance(&a, &model, &quant);
            let mut submission = build_submission(&a, &model, &quant, &b, Some(v), hf);
            sign(&mut submission)?;
            emit(&submission)?;
            if !valid {
                eprintln!("⚠ verification FAILED: gibberish detected — this result is INVALID");
            }
            eprintln!("\n▸ [4/4] Submit");
            match token {
                Some(token) => submit(&a.api, &token, &submission)?,
                None => eprintln!("(dry run — not submitting)"),
            }
        }
    }
    Ok(())
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
    }

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
