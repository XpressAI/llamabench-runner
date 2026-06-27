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

use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand};
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
    /// Path to a local GGUF model. Mutually exclusive with --hf-model.
    #[arg(long)]
    model: Option<String>,
    /// Hugging Face repo to fetch the GGUF from, e.g.
    /// bartowski/Llama-3.1-8B-Instruct-GGUF. Requires --quant. Mutually exclusive
    /// with --model.
    #[arg(long)]
    hf_model: Option<String>,
    /// Quantization to select/report, e.g. Q4_K_M. Required with --hf-model; with
    /// --model it overrides the quant parsed from the filename.
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

/// Resolve the model path: a local `--model`, or download the `--hf-model` +
/// `--quant` GGUF. Exactly one of the two must be given.
fn resolve_model(a: &RunArgs) -> Result<String> {
    match (&a.model, &a.hf_model) {
        (Some(_), Some(_)) => {
            bail!("specify either --model <path> or --hf-model <repo>, not both")
        }
        (None, None) => bail!(
            "no model: pass --model <path.gguf>, or --hf-model <repo> --quant <Q> \
             (e.g. --hf-model bartowski/Llama-3.1-8B-Instruct-GGUF --quant Q4_K_M)"
        ),
        (Some(m), None) => Ok(m.clone()),
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

/// Resolve the directory holding the required llama.cpp binaries: an explicit
/// `--llama-dir`, the binaries already on PATH, or a freshly downloaded prebuilt
/// (when `--download-llama` is set or PATH has none). Empty string ⇒ use PATH.
fn resolve_llama_dir(a: &RunArgs, required: &[&str]) -> Result<String> {
    if !a.llama_dir.is_empty() {
        return Ok(a.llama_dir.clone());
    }
    let on_path = required.iter().all(|b| find_on_path(b));
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
    stem.split('-')
        .find(|tok| {
            (tok.starts_with('Q') || tok.starts_with("IQ"))
                && *tok == tok.to_uppercase()
                && tok.chars().any(|c| c.is_ascii_digit())
        })
        .unwrap_or("unknown")
        .to_string()
}

fn model_name(model: &str) -> String {
    Path::new(model)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// The quant to record: explicit `--quant` wins, else parse it from the filename.
fn resolved_quant(a: &RunArgs, model: &str) -> String {
    a.quant
        .as_deref()
        .map(str::trim)
        .filter(|q| !q.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| quant_from_path(model))
}

fn build_submission(
    a: &RunArgs,
    model: &str,
    quant: &str,
    b: &BenchResult,
    v: Option<Verification>,
) -> ResultSubmission {
    let device = b
        .devices
        .first()
        .cloned()
        .unwrap_or_else(|| "CPU".to_string());
    let name = model_name(model);
    let command = format!(
        "llama-bench -m {} -ngl {} -fa {} -ctk {} -ctv {} -p {} -n {}",
        model, a.ngl, a.fa, a.ctk, a.ctv, a.n_prompt, a.n_gen
    );
    ResultSubmission {
        schema_version: SCHEMA_VERSION,
        hardware: Hardware {
            id: detect::slugify(&device),
            name: device.clone(),
            vendor: detect::vendor_of(&device).to_string(),
            vram_gb: 0.0,
            bandwidth_gbs: 0.0,
        },
        model: ModelInfo {
            id: detect::slugify(&name),
            name,
            params: b.params_b,
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
            name: "llama.cpp".to_string(),
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
    eprintln!("✓ submitted: {body}");
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
            emit(&build_submission(&a, &model, &quant, &b, None))?;
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
            let model = resolve_model(&a)?;
            let dir = resolve_llama_dir(&a, &["llama-bench", "llama-server"])?;
            let quant = resolved_quant(&a, &model);
            eprintln!("→ running llama-bench (speed)…");
            let b = run_llama_bench(&bench_opts(&a, &dir, &model))?;
            eprintln!(
                "→ running output verification (llama-server, {} turns × {} reps)…",
                a.turns, a.reps
            );
            let v = run_verification(&verify_opts(&a, &dir, &model))?;
            let valid = v.valid;
            let mut submission = build_submission(&a, &model, &quant, &b, Some(v));
            sign(&mut submission)?;
            emit(&submission)?;
            if !valid {
                eprintln!("⚠ verification FAILED: gibberish detected — this result is INVALID");
            }
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
    fn quant_parsing() {
        assert_eq!(quant_from_path("/x/Qwen3.5-4B-UD-Q4_K_XL.gguf"), "Q4_K_XL");
        assert_eq!(
            quant_from_path("/x/gemma-4-12b-it-UD-Q4_K_XL.gguf"),
            "Q4_K_XL"
        );
        assert_eq!(quant_from_path("/x/model-IQ4_XS.gguf"), "IQ4_XS");
        assert_eq!(quant_from_path("/x/plain.gguf"), "unknown");
    }
}
