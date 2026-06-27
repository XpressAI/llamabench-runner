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
mod contract;
mod detect;
mod verify;

use anyhow::Result;
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
    /// Run llama-bench and print the result (speed only).
    Bench(RunArgs),
    /// Run the output-correctness verification against llama-server.
    Verify(RunArgs),
    /// Full run: speed + verification → a complete ResultSubmission.
    Run(RunArgs),
}

#[derive(Args, Clone)]
struct RunArgs {
    /// Directory containing llama-bench / llama-server (default: search PATH).
    #[arg(long, default_value = "")]
    llama_dir: String,
    /// Path to the GGUF model.
    #[arg(long)]
    model: String,
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

fn bench_opts(a: &RunArgs) -> BenchOpts<'_> {
    BenchOpts {
        llama_bin_dir: &a.llama_dir,
        model: &a.model,
        ngl: a.ngl,
        fa: &a.fa,
        ctk: &a.ctk,
        ctv: &a.ctv,
        n_prompt: a.n_prompt,
        n_gen: a.n_gen,
    }
}

fn verify_opts(a: &RunArgs) -> VerifyOpts<'_> {
    VerifyOpts {
        server_bin_dir: &a.llama_dir,
        model: &a.model,
        port: a.port,
        api_key: &a.api_key,
        seed: a.seed,
        n_gen: a.n_gen,
        max_turns: a.turns,
        reps: a.reps,
        extra_server_args: &a.server_args,
    }
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

fn build_submission(a: &RunArgs, b: &BenchResult, v: Option<Verification>) -> ResultSubmission {
    let device = b
        .devices
        .first()
        .cloned()
        .unwrap_or_else(|| "CPU".to_string());
    let name = model_name(&a.model);
    let command = format!(
        "llama-bench -m {} -ngl {} -fa {} -ctk {} -ctv {} -p {} -n {}",
        a.model, a.ngl, a.fa, a.ctk, a.ctv, a.n_prompt, a.n_gen
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
            quant: quant_from_path(&a.model),
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
        Command::Bench(a) => {
            let b = run_llama_bench(&bench_opts(&a))?;
            emit(&build_submission(&a, &b, None))?;
        }
        Command::Verify(a) => {
            let v = run_verification(&verify_opts(&a))?;
            println!("{}", serde_json::to_string_pretty(&v)?);
            if !v.valid {
                eprintln!("⚠ verification FAILED: gibberish detected — invalid submission");
            }
        }
        Command::Run(a) => {
            eprintln!("→ running llama-bench (speed)…");
            let b = run_llama_bench(&bench_opts(&a))?;
            eprintln!(
                "→ running output verification (llama-server, {} turns × {} reps)…",
                a.turns, a.reps
            );
            let v = run_verification(&verify_opts(&a))?;
            let valid = v.valid;
            let mut submission = build_submission(&a, &b, Some(v));
            sign(&mut submission)?;
            emit(&submission)?;
            if !valid {
                eprintln!("⚠ verification FAILED: gibberish detected — this result is INVALID");
            }
            if a.dry_run {
                eprintln!("(dry run — not submitting)");
            } else if let Some(token) = &a.token {
                submit(&a.api, token, &submission)?;
            } else {
                eprintln!("note: no --token (or LLAMABENCH_TOKEN) — not submitting. Get one at llamabench.ai/account.");
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
