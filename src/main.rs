// SPDX-License-Identifier: GPL-3.0-or-later
//! llamabench.ai benchmark runner (ADR-004, ADR-005, ADR-009).
//!
//! Two primary workflows:
//!
//! * **Speed**: runner-owned options, one `--` boundary, then an exact native
//!   `llama-bench` or `llama-server` command. See `dropin.rs`.
//! * **Evaluation**: runner-owned identity/transport options, one `--` boundary,
//!   then exact native `llama-server` tuning arguments. See `eval.rs`.
//!
//! Bare drop-in commands and classic `run`/`bench`/`verify` remain as hidden
//! compatibility surfaces for v0.4.x scripts.
//!
//! It drives the user's *existing* llama.cpp build and bundles nothing.

mod bench;
mod config;
mod contract;
mod detect;
mod download;
mod dropin;
mod eval;
mod link;
mod submitter;
mod verify;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};

use bench::{run_llama_bench, BenchOpts, BenchResult};
use contract::{
    Backend, EvaluationConfig, EvaluationModel, EvaluationSubmission, Submitter, Verification,
    EVAL_SCHEMA_VERSION, EVAL_VERSION,
};
use submitter::{
    build_submission, explicit_canonical, provenance, provenance_exact, resolved_quant, BuildCtx,
    Family, HfProvenance, ModelSource, DEFAULT_API, DEFAULT_EVAL_API,
};
use verify::{run_verification, VerifyOpts};

#[derive(Parser)]
#[command(
    name = "llamabench",
    version,
    about = "Benchmark or evaluate an exact local-LLM configuration and publish the evidence.",
    after_help = "\
SPEED — runner options before `--`, then the native command:
  llamabench speed -- llama-bench -m model.gguf -ngl 99 -fa on
  llamabench speed -- llama-server -m model.gguf -c 8192

EVAL — runner options before `--`, native llama-server arguments after it:
  llamabench eval --model ./model.gguf -- \
    -ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 2

PROVENANCE:
  llamabench link ./model.gguf unsloth/gemma-4-12b-it-GGUF

Compatibility: bare drop-in commands and classic run/bench/verify remain available."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Save your llamabench.ai token so runs can submit without --token.
    Auth(AuthArgs),
    /// Link a local GGUF to its Hugging Face repo (hash-verified, persistent).
    Link(LinkArgs),
    /// Measure and submit one exact native llama-bench or llama-server command.
    Speed(SpeedArgs),
    /// Run llama-bench and print the result (speed only).
    #[command(hide = true)]
    Bench(RunArgs),
    /// Run the output-correctness verification against llama-server.
    #[command(hide = true)]
    Verify(RunArgs),
    /// Full run: speed + verification → a complete ResultSubmission.
    #[command(hide = true)]
    Run(RunArgs),
    /// Evaluate one exact GGUF/runtime with versioned visual, tool-use, and role-play tasks.
    Eval(EvalArgs),
}

#[derive(Args, Clone)]
struct SpeedArgs {
    /// Detect, print, and sign the result without submitting it.
    #[arg(long)]
    dry_run: bool,
    /// Skip the output-correctness pass.
    #[arg(long)]
    no_verify: bool,
    /// Download the latest prebuilt upstream llama.cpp instead of using PATH.
    #[arg(long)]
    download_llama: bool,
    /// CLI token from llamabench.ai/account — required to submit.
    #[arg(long, env = "LLAMABENCH_TOKEN")]
    token: Option<String>,
    /// Submitter handle.
    #[arg(long, default_value = "@anonymous")]
    handle: String,
    /// Which llama.cpp variant the build belongs to.
    #[arg(long, value_enum, default_value = "llama.cpp")]
    family: Family,
    /// Directory containing llama-bench / llama-server.
    #[arg(long, default_value = "")]
    llama_dir: String,
    /// Result-submission API endpoint.
    #[arg(long, default_value = DEFAULT_API)]
    api: String,
    /// Canonical HF model repo when the GGUF repo omits cardData.base_model.
    #[arg(long)]
    base_model: Option<String>,
    /// Active parameters in billions for a sparse model, when officially disclosed.
    #[arg(long, value_parser = submitter::positive_f64)]
    active_params: Option<f64>,
    /// Local port used only by the runner's temporary correctness server.
    #[arg(
        long,
        default_value_t = 8080,
        value_parser = clap::value_parser!(u16).range(1..)
    )]
    verification_port: u16,
    /// Native command and arguments: `llama-bench ...` or `llama-server ...`.
    #[arg(
        last = true,
        required = true,
        num_args = 1..,
        value_name = "LLAMA_COMMAND"
    )]
    command: Vec<String>,
}

#[derive(Args)]
struct AuthArgs {
    /// The CLI token from https://llamabench.ai/account. If omitted, read from stdin.
    token: Option<String>,
}

#[derive(Args)]
struct LinkArgs {
    /// Path to the local .gguf.
    path: Option<String>,
    /// The Hugging Face repo the file comes from, e.g. unsloth/gemma-4-12b-it-GGUF.
    /// Omit to show the file's current link status.
    repo: Option<String>,
    /// List all linked models.
    #[arg(long)]
    list: bool,
    /// Remove the link for a path.
    #[arg(long, value_name = "PATH")]
    forget: Option<String>,
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
    /// hash-verify the file's Hugging Face provenance (the local bytes are still
    /// used) — or `llamabench link` it once and skip the flags for good.
    #[arg(long)]
    model: Option<String>,
    /// Hugging Face repo for the GGUF, e.g. bartowski/Llama-3.1-8B-Instruct-GGUF
    /// (requires --quant). WITHOUT --model the file is downloaded from here; WITH
    /// --model the local file is used but its SHA-256 is verified against this repo.
    /// The submission is attributed to the GGUF's base/finetune model (its HF
    /// base_model) so every GGUF repack of the same model groups together.
    #[arg(long)]
    hf_model: Option<String>,
    /// Canonical HF model repo when the GGUF repo omits cardData.base_model.
    #[arg(long)]
    base_model: Option<String>,
    /// Active parameters in billions for a sparse model, when officially disclosed.
    #[arg(long, value_parser = submitter::positive_f64)]
    active_params: Option<f64>,
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
    /// Extra arg passed verbatim to llama-server, repeatable:
    /// `--server-arg --foo --server-arg bar`. Use this when a value contains spaces.
    #[arg(long = "server-arg", allow_hyphen_values = true)]
    server_arg: Vec<String>,
    /// Extra llama-server args as ONE whitespace-delimited string — convenient for the
    /// many speculative-decoding flags, e.g.
    /// `--server-args "--spec-type draft-mtp --spec-draft-n-max 2"`. Split on whitespace
    /// and appended after any --server-arg values.
    #[arg(long = "server-args", default_value = "", allow_hyphen_values = true)]
    server_args: String,

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

#[derive(Args, Clone)]
struct EvalArgs {
    /// Directory containing llama-server. Default: search PATH, else auto-download
    /// a prebuilt CPU/Metal build (see --download-llama).
    #[arg(long, default_value = "")]
    llama_dir: String,
    /// Download the latest prebuilt upstream llama.cpp instead of using PATH.
    #[arg(long)]
    download_llama: bool,
    /// Which llama.cpp variant the build is.
    #[arg(long, value_enum, default_value = "llama.cpp")]
    family: Family,
    /// Path to the exact local GGUF to exercise.
    #[arg(long)]
    model: Option<String>,
    /// Optional Hugging Face GGUF repo for provenance. Without --model, downloads
    /// the selected --quant; with --model, verifies the local bytes against it.
    #[arg(long)]
    hf_model: Option<String>,
    /// Canonical HF model repo when the GGUF repo omits cardData.base_model.
    #[arg(long)]
    base_model: Option<String>,
    /// Quantization selector/fallback, e.g. Q4_K_M. The GGUF filename remains
    /// authoritative when it contains a quant.
    #[arg(long)]
    quant: Option<String>,
    /// Submitter handle.
    #[arg(long, default_value = "@anonymous")]
    handle: String,
    /// llama-server port used for the temporary eval session.
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// API key used only for the temporary local llama-server session.
    #[arg(long, default_value = "llamabench")]
    api_key: String,
    /// Explicit context override. Omit to fit the model-native maximum to hardware.
    #[arg(
        long,
        value_parser = clap::value_parser!(u32).range(1024..=2_147_483_647)
    )]
    context_length: Option<u32>,
    /// Native llama-server arguments. Put them after `--`; every token is forwarded
    /// in order and becomes part of the exact evaluation configuration.
    #[arg(
        last = true,
        value_name = "LLAMA_SERVER_ARG",
        allow_hyphen_values = true
    )]
    runtime_args: Vec<String>,
    /// Run and print the signed payload without submitting it.
    #[arg(long)]
    dry_run: bool,
    /// Behavior-evaluation API endpoint.
    #[arg(long, default_value = DEFAULT_EVAL_API)]
    api: String,
    /// CLI token from llamabench.ai/account — required to submit.
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
    // Repeatable --server-arg values first, then the whitespace-split --server-args string.
    let extra_server_args = a
        .server_arg
        .iter()
        .cloned()
        .chain(a.server_args.split_whitespace().map(String::from))
        .collect();
    VerifyOpts {
        server_bin_dir: llama_dir,
        model,
        port: a.port,
        api_key: &a.api_key,
        seed: a.seed,
        n_gen: a.n_gen,
        max_turns: a.turns,
        reps: a.reps,
        extra_server_args,
    }
}

/// Resolve the model path. A local `--model` is always the file we benchmark; if
/// `--hf-model` is also given it only attributes provenance. With `--hf-model`
/// alone, download the `--quant` GGUF from the repo. At least one must be given.
fn resolve_model(a: &RunArgs) -> Result<String> {
    resolve_model_parts(&a.model, &a.hf_model, a.quant.as_deref())
}

fn resolve_eval_model(a: &EvalArgs) -> Result<String> {
    resolve_model_parts(&a.model, &a.hf_model, a.quant.as_deref())
}

fn resolve_model_parts(
    model: &Option<String>,
    hf_model: &Option<String>,
    quant: Option<&str>,
) -> Result<String> {
    match (model, hf_model) {
        (None, None) => bail!(
            "no model: pass --model <path.gguf>, or --hf-model <repo> --quant <Q> \
             (e.g. --hf-model bartowski/Llama-3.1-8B-Instruct-GGUF --quant Q4_K_M)"
        ),
        // A local file always wins for the bytes we run; --hf-model alongside it is
        // recorded/verified as provenance, not downloaded.
        (Some(m), _) => Ok(m.clone()),
        (None, Some(repo)) => {
            let quant = quant.filter(|q| !q.trim().is_empty()).ok_or_else(|| {
                anyhow::anyhow!("--hf-model requires --quant <Q> to pick the .gguf (e.g. Q4_K_M)")
            })?;
            Ok(download::hf_download(repo, quant)?
                .to_string_lossy()
                .into_owned())
        }
    }
}

/// The provenance source for a classic run: an explicit `--hf-model` wins; a bare
/// `--model` consults the persistent link store (ADR-009).
fn model_source<'a>(a: &'a RunArgs, model: &'a str) -> ModelSource<'a> {
    model_source_parts(&a.model, &a.hf_model, model)
}

fn eval_model_source<'a>(a: &'a EvalArgs, model: &'a str) -> ModelSource<'a> {
    model_source_parts(&a.model, &a.hf_model, model)
}

fn model_source_parts<'a>(
    requested_model: &'a Option<String>,
    hf_model: &'a Option<String>,
    resolved_model: &'a str,
) -> ModelSource<'a> {
    match (requested_model, hf_model) {
        (Some(m), Some(repo)) => ModelSource::LocalWithRepo(m, repo),
        (None, Some(repo)) => ModelSource::Downloaded(repo),
        _ => ModelSource::LocalOnly(resolved_model),
    }
}

fn resolve_llama_dir(a: &RunArgs, required: &[&str]) -> Result<String> {
    submitter::resolve_llama_dir(&a.llama_dir, a.download_llama, a.family, required)
}

fn classic_ctx<'a>(a: &'a RunArgs, model: &'a str, quant: &'a str) -> BuildCtx<'a> {
    // Record the model by file name only (./<file>.gguf) — never the submitter's local
    // absolute path, which would leak their home directory.
    let model_file = std::path::Path::new(model)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model);
    let command = format!(
        "llama-bench -m ./{} -ngl {} -fa {} -ctk {} -ctv {} -p {} -n {}",
        model_file, a.ngl, a.fa, a.ctk, a.ctv, a.n_prompt, a.n_gen
    );
    BuildCtx {
        gpu_run: a.ngl != 0,
        selected_device: None,
        handle: &a.handle,
        family: a.family,
        command,
        quant,
        model_path: model,
        active_params: a.active_params,
        context_length: a.n_prompt,
        spec_decode: a.spec_decode.clone(),
        ttft_ms: None,
    }
}

fn classic_submission(
    a: &RunArgs,
    model: &str,
    quant: &str,
    b: &BenchResult,
    v: Option<Verification>,
    hf: HfProvenance,
    ttft_ms: Option<u32>,
) -> Result<contract::ResultSubmission> {
    let mut ctx = classic_ctx(a, model, quant);
    ctx.ttft_ms = ttft_ms;
    build_submission(&ctx, b, v, hf)
}

fn eval_server_args(a: &EvalArgs) -> Vec<String> {
    a.runtime_args.clone()
}

fn shell_word(value: &str) -> String {
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "-._/:=".contains(c))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn eval_model_file(model: &str) -> Result<String> {
    let model_file = std::path::Path::new(model)
        .file_name()
        .and_then(|value| value.to_str())
        .context("eval-v2 requires a UTF-8 GGUF filename")?;
    if model_file.is_empty()
        || matches!(model_file, "." | "..")
        || model_file.chars().count() > 255
        || model_file.contains(['/', '\\'])
        || model_file
            .chars()
            .any(|value| value.is_control() || matches!(value, '\u{2028}' | '\u{2029}'))
    {
        bail!("eval-v2 requires a path-free GGUF basename of at most 255 characters");
    }
    Ok(model_file.to_string())
}

#[derive(Clone, Copy, Default)]
struct EvalReproProvenance<'a> {
    hf_model: Option<&'a str>,
    base_model: Option<&'a str>,
}

fn eval_command(
    a: &EvalArgs,
    model: &str,
    quant: &str,
    context_length: u32,
    parallel: u32,
    runtime_args: &[String],
    provenance: EvalReproProvenance<'_>,
) -> Result<String> {
    let reproduce_context = context_length
        .checked_mul(parallel)
        .filter(|value| *value <= i32::MAX as u32)
        .context(
            "resolved per-request context and parallel count exceed the reproduce-command limit",
        )?;
    let model_file = eval_model_file(model)?;
    let mut parts = vec![
        "llamabench".to_string(),
        "eval".to_string(),
        "--model".to_string(),
        shell_word(&format!("./{model_file}")),
        "--quant".to_string(),
        shell_word(quant),
        "--context-length".to_string(),
        reproduce_context.to_string(),
    ];
    if let Some(hf_model) = provenance.hf_model {
        parts.extend(["--hf-model".to_string(), shell_word(hf_model)]);
    }
    if let Some(base_model) = provenance.base_model {
        parts.extend(["--base-model".to_string(), shell_word(base_model)]);
    }
    if a.family != Family::LlamaCpp {
        parts.extend(["--family".to_string(), a.family.backend_name().to_string()]);
    }
    if !runtime_args.is_empty() {
        parts.push("--".to_string());
        parts.extend(runtime_args.iter().map(|arg| shell_word(arg)));
    }
    Ok(parts.join(" "))
}

fn eval_submission(
    a: &EvalArgs,
    model_path: &str,
    quant: &str,
    gguf_sha256: String,
    run: eval::EvaluationRun,
    hf: HfProvenance,
    backend: Backend,
) -> Result<EvaluationSubmission> {
    let HfProvenance {
        model: hf_model,
        verified: hf_verified,
        canonical,
    } = hf;
    let submitter::Canonical {
        base_model,
        id: canonical_id,
        name: canonical_name,
    } = canonical;
    let eval::EvaluationRun {
        visuals,
        agentic_tasks,
        roleplay,
        context_length,
        context_mode,
        runtime,
    } = run;
    if quant
        .chars()
        .any(|value| value.is_control() || matches!(value, '\u{2028}' | '\u{2029}'))
    {
        bail!("eval-v2 quant cannot contain control characters");
    }
    let command = eval_command(
        a,
        model_path,
        quant,
        context_length,
        runtime.parallel,
        &runtime.args,
        EvalReproProvenance {
            hf_model: hf_model.as_deref(),
            base_model: base_model.as_deref(),
        },
    )?;
    if command.chars().count() > 8_000 {
        bail!("eval-v2 reproduce command exceeds 8000 characters");
    }
    let gguf_file = eval_model_file(model_path)?;
    let fallback_name = submitter::model_name(model_path);
    let (model_id, name) = match (canonical_id, canonical_name) {
        (Some(id), Some(name)) => (id, name),
        _ => (detect::slugify(&fallback_name), fallback_name),
    };
    let params = submitter::params_from_name(&name);

    Ok(EvaluationSubmission {
        schema_version: EVAL_SCHEMA_VERSION,
        eval_version: EVAL_VERSION.to_string(),
        model: EvaluationModel {
            id: model_id,
            name,
            params,
            base_model,
            hf_model,
            hf_verified,
            gguf_file,
            gguf_sha256,
        },
        config: EvaluationConfig {
            quant: quant.to_string(),
            context_length,
            context_mode,
            kv_cache_key: runtime.kv_cache_key,
            kv_cache_value: runtime.kv_cache_value,
            flash_attention: runtime.flash_attention,
            speculative_decoding: runtime.speculative_decoding,
            command,
            runtime_args: runtime.args,
        },
        backend,
        settings: eval::settings(),
        visuals,
        agentic_tasks,
        roleplay,
        submitter: Submitter {
            handle: a.handle.clone(),
        },
        signature: String::new(),
    })
}

fn ensure_eval_hash_unchanged(before: &str, after: &str) -> Result<()> {
    if before != after {
        bail!(
            "model artifact changed while the evaluation was running; refusing to submit ambiguous evidence"
        );
    }
    Ok(())
}

fn speed_mode(tool: &str) -> Result<dropin::Mode> {
    match tool {
        "llama-bench" => Ok(dropin::Mode::Bench),
        "llama-server" => Ok(dropin::Mode::Server),
        _ => bail!(
            "speed expects `-- llama-bench <args>` or `-- llama-server <args>`; got {tool:?}. Use --llama-dir to select a build directory"
        ),
    }
}

fn main() -> Result<()> {
    // Drop-in dispatch (ADR-009) happens before clap: `llamabench <llama-bench
    // args>` (bare flags), `llamabench llama-bench …`, `llamabench llama-server …`.
    // Everything else — subcommands, -h/-V — falls through to clap.
    let argv: Vec<String> = std::env::args().collect();
    if let Some(first) = argv.get(1).map(String::as_str) {
        match first {
            "llama-bench" => return dropin::run(dropin::Mode::Bench, &argv[2..]),
            "llama-server" => return dropin::run(dropin::Mode::Server, &argv[2..]),
            s if s.starts_with('-') && !matches!(s, "-h" | "--help" | "-V" | "--version") => {
                return dropin::run(dropin::Mode::Auto, &argv[1..])
            }
            _ => {}
        }
    }
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
        Command::Link(args) => {
            if let Some(path) = args.forget.as_deref() {
                return link::cmd_forget(path);
            }
            if args.list {
                return link::cmd_list();
            }
            match (args.path.as_deref(), args.repo.as_deref()) {
                (Some(path), Some(repo)) => link::cmd_link(path, repo)?,
                (Some(path), None) => link::cmd_status(path)?,
                (None, _) => link::cmd_list()?,
            }
        }
        Command::Speed(a) => {
            let (tool, tool_args) = a.command.split_first().expect("required by clap");
            let mode = speed_mode(tool)?;
            eval::ensure_explicit_runtime_environment()?;
            return dropin::run_with_options(
                mode,
                &dropin::WrapOpts {
                    dry_run: a.dry_run,
                    no_verify: a.no_verify,
                    download_llama: a.download_llama,
                    token: a.token,
                    handle: a.handle,
                    family: a.family,
                    llama_dir: a.llama_dir,
                    api: a.api,
                    base_model: a.base_model,
                    active_params: a.active_params,
                    verification_port: a.verification_port,
                },
                tool_args,
            );
        }
        Command::Bench(a) => {
            let model = resolve_model(&a)?;
            let dir = resolve_llama_dir(&a, &["llama-bench"])?;
            let b = run_llama_bench(&bench_opts(&a, &dir, &model))?;
            let quant = resolved_quant(a.quant.as_deref(), &model);
            let hf = explicit_canonical(
                provenance(&model_source(&a, &model), &quant),
                a.base_model.as_deref(),
            )?;
            submitter::emit(&classic_submission(&a, &model, &quant, &b, None, hf, None)?)?;
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
                Some(submitter::resolve_token(a.token.as_deref())?)
            };
            eprintln!("\n▸ [1/4] Model — resolve & download");
            let model = resolve_model(&a)?;
            let dir = resolve_llama_dir(&a, &["llama-bench", "llama-server"])?;
            let quant = resolved_quant(a.quant.as_deref(), &model);
            eprintln!("\n▸ [2/4] Benchmark — llama-bench (prefill + decode)");
            let b = run_llama_bench(&bench_opts(&a, &dir, &model))?;
            eprintln!(
                "\n▸ [3/4] Verify — llama-server, TTFT probe + {} turns × {} reps (the slow part)",
                a.turns, a.reps
            );
            let (v, ttft) = verify::run_verification_with_ttft(&verify_opts(&a, &dir, &model))?;
            let valid = v.valid;
            let hf = explicit_canonical(
                provenance(&model_source(&a, &model), &quant),
                a.base_model.as_deref(),
            )?;
            let mut submission = classic_submission(&a, &model, &quant, &b, Some(v), hf, ttft)?;
            submitter::sign(&mut submission)?;
            submitter::emit(&submission)?;
            if !valid {
                eprintln!("⚠ verification FAILED: gibberish detected — this result is INVALID");
            }
            eprintln!("\n▸ [4/4] Submit");
            match token {
                Some(token) => submitter::submit(&a.api, &token, &submission)?,
                None => eprintln!("(dry run — not submitting)"),
            }
        }
        Command::Eval(a) => {
            let token = if a.dry_run {
                None
            } else {
                Some(submitter::resolve_token(a.token.as_deref())?)
            };
            eprintln!("\n▸ Model — resolve exact GGUF artifact");
            let model = resolve_eval_model(&a)?;
            let dir = submitter::resolve_llama_dir(
                &a.llama_dir,
                a.download_llama,
                a.family,
                &["llama-server"],
            )?;
            let quant = resolved_quant(a.quant.as_deref(), &model);
            let gguf_sha256 = link::fresh_sha256_for(&model)?;
            let hf = explicit_canonical(
                provenance_exact(&eval_model_source(&a, &model), &quant, &gguf_sha256),
                a.base_model.as_deref(),
            )?;
            let (backend_version, backend_hash) = eval::server_version(&dir)?;
            let run = eval::run_eval(&eval::EvaluationOpts {
                server_bin_dir: &dir,
                model: &model,
                port: a.port,
                api_key: &a.api_key,
                context_length: a.context_length,
                runtime_args: eval_server_args(&a),
            })?;
            let final_gguf_sha256 = link::fresh_sha256_for(&model)?;
            ensure_eval_hash_unchanged(&gguf_sha256, &final_gguf_sha256)?;
            let mut submission = eval_submission(
                &a,
                &model,
                &quant,
                gguf_sha256,
                run,
                hf,
                Backend {
                    name: a.family.backend_name().to_string(),
                    version: backend_version,
                    git_hash: backend_hash,
                },
            )?;
            eval::sign(&mut submission)?;
            eprintln!("\nReproduce: {}", submission.config.command);
            println!("{}", serde_json::to_string_pretty(&submission)?);
            match token {
                Some(token) => submitter::submit_eval(&a.api, &token, &submission)?,
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
    fn server_args_combine() {
        // Repeatable --server-arg values, then the whitespace-split --server-args string.
        let cli = Cli::parse_from([
            "llamabench",
            "run",
            "--server-arg",
            "--foo",
            "--server-arg",
            "bar",
            "--server-args",
            "--spec-type draft-mtp --spec-draft-n-max 2",
            "--hf-model",
            "x/y",
            "--quant",
            "Q4_K_M",
        ]);
        let Command::Run(a) = cli.command else {
            panic!("expected run")
        };
        assert_eq!(
            verify_opts(&a, "/d", "/m.gguf").extra_server_args,
            vec![
                "--foo",
                "bar",
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                "2"
            ]
        );
    }

    #[test]
    fn link_subcommand_parses() {
        let cli = Cli::parse_from(["llamabench", "link", "./m.gguf", "unsloth/x-GGUF"]);
        let Command::Link(a) = cli.command else {
            panic!("expected link")
        };
        assert_eq!(a.path.as_deref(), Some("./m.gguf"));
        assert_eq!(a.repo.as_deref(), Some("unsloth/x-GGUF"));
        assert!(!a.list);

        let cli = Cli::parse_from(["llamabench", "link", "--forget", "./m.gguf"]);
        let Command::Link(a) = cli.command else {
            panic!("expected link")
        };
        assert_eq!(a.forget.as_deref(), Some("./m.gguf"));
    }

    #[test]
    fn speed_cli_has_one_explicit_native_command_boundary() {
        let cli = Cli::parse_from([
            "llamabench",
            "speed",
            "--dry-run",
            "--family",
            "ik_llama.cpp",
            "--base-model",
            "ornith-ai/Ornith-1.5-9B",
            "--active-params",
            "3",
            "--verification-port",
            "18080",
            "--",
            "llama-server",
            "-m",
            "model.gguf",
            "-c",
            "8192",
            "--chat-template",
            "a template with spaces",
        ]);
        let Command::Speed(a) = cli.command else {
            panic!("expected speed")
        };
        assert!(a.dry_run);
        assert!(matches!(a.family, Family::IkLlamaCpp));
        assert_eq!(a.base_model.as_deref(), Some("ornith-ai/Ornith-1.5-9B"));
        assert_eq!(a.active_params, Some(3.0));
        assert_eq!(a.verification_port, 18080);
        assert_eq!(a.command[0], "llama-server");
        assert_eq!(
            a.command.last().map(String::as_str),
            Some("a template with spaces")
        );
        assert!(matches!(
            speed_mode(&a.command[0]).unwrap(),
            dropin::Mode::Server
        ));
        assert!(speed_mode("/tmp/llama-server").is_err());
        assert!(Cli::try_parse_from([
            "llamabench",
            "speed",
            "--verification-port",
            "0",
            "--",
            "llama-bench",
            "-m",
            "model.gguf",
        ])
        .is_err());
    }

    #[test]
    fn classic_reproduce_command_redacts_path() {
        let cli = Cli::parse_from(["llamabench", "run", "--model", "/home/edu/x-Q4_K_M.gguf"]);
        let Command::Run(a) = cli.command else {
            panic!("expected run")
        };
        let ctx = classic_ctx(&a, "/home/edu/x-Q4_K_M.gguf", "Q4_K_M");
        assert!(ctx.command.starts_with("llama-bench -m ./x-Q4_K_M.gguf"));
        assert!(!ctx.command.contains("/home/edu"));
    }

    #[test]
    fn eval_cli_preserves_native_argument_boundaries() {
        let cli = Cli::parse_from([
            "llamabench",
            "eval",
            "--model",
            "/home/edu/model-Q4_K_M.gguf",
            "--family",
            "ik_llama.cpp",
            "--hf-model",
            "ornith-ai/Ornith-1.5-9B-GGUF",
            "--base-model",
            "ornith-ai/Ornith-1.5-9B",
            "--",
            "-ctk",
            "q4_0",
            "-ctv",
            "q4_0",
            "--chat-template",
            "a template with spaces",
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "2",
        ]);
        let Command::Eval(a) = cli.command else {
            panic!("expected eval")
        };
        assert_eq!(a.context_length, None);
        assert_eq!(a.runtime_args[5], "a template with spaces");
        let runtime = eval::runtime_config(&a.runtime_args).unwrap();
        let command = eval_command(
            &a,
            "/home/edu/model-Q4_K_M.gguf",
            "Q4_K_M",
            262_144,
            runtime.parallel,
            &runtime.args,
            EvalReproProvenance {
                hf_model: a.hf_model.as_deref(),
                base_model: a.base_model.as_deref(),
            },
        )
        .unwrap();
        assert!(command.contains("eval --model ./model-Q4_K_M.gguf"));
        assert!(command.contains("--context-length 262144"));
        assert!(command.contains("--family ik_llama.cpp"));
        assert!(command.contains("--hf-model ornith-ai/Ornith-1.5-9B-GGUF"));
        assert!(command.contains("--base-model ornith-ai/Ornith-1.5-9B"));
        assert!(command.contains("-- -ctk q4_0 -ctv q4_0"));
        assert!(command.contains("'a template with spaces'"));
        assert!(!command.contains("/home/edu"));
        assert_eq!(
            eval_model_file("/home/edu/model-Q4_K_M.gguf").unwrap(),
            "model-Q4_K_M.gguf"
        );
        assert!(eval_model_file("folder\\model.gguf").is_err());
        assert_eq!(runtime.kv_cache_key, "q4_0");
        assert_eq!(runtime.kv_cache_value, "q4_0");
        assert_eq!(runtime.parallel, 1);
        assert_eq!(runtime.speculative_decoding.mode, "draft-mtp");
        assert_eq!(
            runtime.speculative_decoding.parameters,
            vec!["--spec-draft-n-max=2"]
        );

        assert!(Cli::try_parse_from([
            "llamabench",
            "eval",
            "--model",
            "model.gguf",
            "--server-args",
            "-ctk q4_0",
        ])
        .is_err());
        assert!(Cli::try_parse_from([
            "llamabench",
            "eval",
            "--model",
            "model.gguf",
            "--context-length",
            "512",
        ])
        .is_err());
        let large = Cli::try_parse_from([
            "llamabench",
            "eval",
            "--model",
            "model.gguf",
            "--context-length",
            "2000000",
        ])
        .unwrap();
        let Command::Eval(large) = large.command else {
            panic!("expected eval")
        };
        assert_eq!(large.context_length, Some(2_000_000));
    }

    #[test]
    fn eval_reproduce_command_preserves_parallel_slot_context() {
        let cli = Cli::parse_from([
            "llamabench",
            "eval",
            "--model",
            "model.gguf",
            "--",
            "-np",
            "2",
        ]);
        let Command::Eval(a) = cli.command else {
            panic!("expected eval")
        };
        let runtime = eval::runtime_config(&a.runtime_args).unwrap();
        assert_eq!(runtime.parallel, 2);
        let command = eval_command(
            &a,
            "model.gguf",
            "Q4_K_M",
            262_144,
            runtime.parallel,
            &runtime.args,
            EvalReproProvenance::default(),
        )
        .unwrap();
        assert!(command.contains("--context-length 524288 -- -np 2"));
    }

    #[test]
    fn eval_submission_requires_stable_artifact_bytes() {
        assert!(ensure_eval_hash_unchanged("abc", "abc").is_ok());
        assert!(ensure_eval_hash_unchanged("abc", "def").is_err());
    }
}
