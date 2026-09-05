// SPDX-License-Identifier: GPL-3.0-or-later
//! Drop-in passthrough modes (ADR-009). The user replaces the program name in the
//! command they already run — `llama-bench …` becomes `llamabench …`, and
//! `llama-server …` becomes `llamabench llama-server …` — and we run that **exact**
//! configuration, record the literal command, and submit the result.
//!
//! * llama-bench mode appends `-oe jsonl` so the per-test JSON rides on stderr while
//!   the user's own table streams untouched on stdout. Every configuration in a
//!   matrix run (`-ngl 0,99`, `-m a.gguf,b.gguf`, …) becomes its own submission,
//!   with the config fields read from the row llama-bench itself reported.
//! * llama-server mode spawns the server with the user's args verbatim, measures
//!   prefill/decode/TTFT from the server's own `timings`, and reuses the multi-turn
//!   output-correctness verification against that same process.
//!
//! Runner-owned flags (`--dry-run`, `--no-verify`, `--token`, `--handle`,
//! `--family`, `--llama-dir`, `--api`, `--download-llama`, `--base-model`,
//! `--active-params`, `--verification-port`) are extracted before
//! passthrough; none collide with llama.cpp flag names.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::bench::{self, BenchResult};
use crate::submitter::{self, build_submission, BuildCtx, Family, ModelSource};
use crate::verify::{self, Timing, VerifyOpts, VerifySession};

#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Bench,
    Server,
    /// Bare `llamabench <args>` — decide from the flags themselves.
    Auto,
}

/// Runner-owned flags extracted from the passthrough stream.
pub struct WrapOpts {
    pub dry_run: bool,
    pub no_verify: bool,
    pub download_llama: bool,
    pub token: Option<String>,
    pub handle: String,
    pub family: Family,
    pub llama_dir: String,
    pub api: String,
    pub base_model: Option<String>,
    pub active_params: Option<f64>,
    pub verification_port: u16,
}

impl Default for WrapOpts {
    fn default() -> Self {
        WrapOpts {
            dry_run: false,
            no_verify: false,
            download_llama: false,
            token: None,
            handle: "@anonymous".to_string(),
            family: Family::LlamaCpp,
            llama_dir: String::new(),
            api: submitter::DEFAULT_API.to_string(),
            base_model: None,
            active_params: None,
            verification_port: 8080,
        }
    }
}

pub fn run(mode: Mode, args: &[String]) -> Result<()> {
    let (w, tool_args) = extract_wrapper_flags(args)?;
    run_with_options(mode, &w, &tool_args)
}

/// Explicit command path used by `llamabench speed`: Clap owns runner options and
/// the native argument vector is already separated by `--` (ADR-015).
pub fn run_with_options(mode: Mode, options: &WrapOpts, tool_args: &[String]) -> Result<()> {
    let mode = if mode == Mode::Auto {
        sniff_mode(tool_args)
    } else {
        mode
    };
    match mode {
        Mode::Server => run_server(options, tool_args),
        _ => run_bench(options, tool_args),
    }
}

fn take_value(args: &[String], i: &mut usize, name: &str, inline: Option<&str>) -> Result<String> {
    if let Some(v) = inline {
        return Ok(v.to_string());
    }
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| anyhow!("{name} needs a value"))
}

fn extract_wrapper_flags(args: &[String]) -> Result<(WrapOpts, Vec<String>)> {
    let mut w = WrapOpts::default();
    let mut rest = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let (name, inline) = match a.split_once('=') {
            Some((n, v)) => (n, Some(v)),
            None => (a.as_str(), None),
        };
        match name {
            "--dry-run" => w.dry_run = true,
            "--no-verify" => w.no_verify = true,
            "--download-llama" => w.download_llama = true,
            "--token" => w.token = Some(take_value(args, &mut i, name, inline)?),
            "--handle" => w.handle = take_value(args, &mut i, name, inline)?,
            "--family" => w.family = Family::parse(&take_value(args, &mut i, name, inline)?)?,
            "--llama-dir" => w.llama_dir = take_value(args, &mut i, name, inline)?,
            "--api" => w.api = take_value(args, &mut i, name, inline)?,
            "--base-model" => w.base_model = Some(take_value(args, &mut i, name, inline)?),
            "--active-params" => {
                let value = take_value(args, &mut i, name, inline)?;
                w.active_params = Some(
                    submitter::positive_f64(&value)
                        .map_err(|message| anyhow!("--active-params {message}"))?,
                );
            }
            "--verification-port" => {
                w.verification_port = take_value(args, &mut i, name, inline)?
                    .parse()
                    .map_err(|_| anyhow!("--verification-port must be a number from 1 to 65535"))?;
                if w.verification_port == 0 {
                    bail!("--verification-port must be a number from 1 to 65535");
                }
            }
            _ => rest.push(a.clone()),
        }
        i += 1;
    }
    Ok((w, rest))
}

/// Flags that exist on llama-server but not on llama-bench. Seeing any of them in
/// bare `llamabench <args>` means the user swapped their llama-server command in.
const SERVER_ONLY_FLAGS: &[&str] = &[
    "--port",
    "--host",
    "--api-key",
    "-c",
    "--ctx-size",
    "--jinja",
    "-np",
    "--parallel",
    "-a",
    "--alias",
    "-md",
    "--model-draft",
    "--draft",
    "--draft-max",
    "--draft-min",
    "-cd",
    "--ctx-size-draft",
    "-ngld",
    "--n-gpu-layers-draft",
    "--spec-type",
    "--no-webui",
    "--chat-template",
    "--chat-template-file",
    "--reasoning-format",
    "--slots",
    "--metrics",
    "--embedding",
    "--embeddings",
    "--reranking",
    "--pooling",
    "-cb",
    "--cont-batching",
    "-hf",
    "-hfr",
    "--hf-repo",
];

fn sniff_mode(args: &[String]) -> Mode {
    let is_server = args.iter().any(|a| {
        let name = a.split_once('=').map_or(a.as_str(), |(n, _)| n);
        SERVER_ONLY_FLAGS.contains(&name)
    });
    if is_server {
        eprintln!(
            "→ llama-server flags detected — drop-in llama-server mode \
             (force a mode with `llamabench llama-bench …` / `llamabench llama-server …`)"
        );
        Mode::Server
    } else {
        Mode::Bench
    }
}

/// The last value of any of `names` (llama.cpp semantics: later args win).
/// Supports `--flag value` and `--flag=value`. A bare flag yields `Some("")`.
/// A following token that looks like another flag is not eaten as the value —
/// unless it parses as a number (`-ngl -1`).
fn flag_value(args: &[String], names: &[&str]) -> Option<String> {
    let mut found = None;
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.split_once('=') {
            Some((n, v)) if names.contains(&n) => found = Some(v.to_string()),
            None if names.contains(&a.as_str()) => match args.get(i + 1) {
                Some(v) if !v.starts_with('-') || v.parse::<f64>().is_ok() => {
                    found = Some(v.clone());
                    i += 1;
                }
                _ => found = Some(String::new()),
            },
            _ => {}
        }
        i += 1;
    }
    found
}

fn selected_device(args: &[String]) -> Option<String> {
    flag_value(args, &["-dev", "--device"]).filter(|value| !value.is_empty())
}

fn validate_selected_device_scope(selected: Option<&str>) -> Result<()> {
    let count = selected
        .into_iter()
        .flat_map(|value| value.split(','))
        .filter(|value| !value.trim().is_empty())
        .count();
    if count > 1 {
        bail!(
            "multiple explicit --device selectors are not supported; select one device or omit --device to record all visible CUDA devices as a group"
        );
    }
    Ok(())
}

fn group_visible_gpus(args: &[String]) -> bool {
    !flag_value(args, &["-sm", "--split-mode"])
        .is_some_and(|mode| mode.eq_ignore_ascii_case("none"))
}

/// Shell-quote a token if needed (for the recorded reproduce command).
fn shell_quote(s: &str) -> String {
    let plain = !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "_-./=:,+@%".contains(c));
    if plain {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', r"'\''"))
    }
}

/// The recorded reproduce command: the tool name + the user's args verbatim, except
/// model paths are reduced to `./<file>` (never leak the submitter's home directory)
/// and secrets are removed outright.
fn redacted_command(tool: &str, args: &[String]) -> String {
    const PATH_FLAGS: &[&str] = &["-m", "--model", "-md", "--model-draft", "--mmproj"];
    const SECRET_FLAGS: &[&str] = &["--api-key", "--hf-token", "--ssl-key-file"];
    let mut out = vec![tool.to_string()];
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let (name, inline) = match a.split_once('=') {
            Some((n, v)) => (n, Some(v)),
            None => (a.as_str(), None),
        };
        let kind = if PATH_FLAGS.contains(&name) {
            Some(false)
        } else if SECRET_FLAGS.contains(&name) {
            Some(true)
        } else {
            None
        };
        match kind {
            Some(secret) => {
                let value = match inline {
                    Some(v) => Some(v.to_string()),
                    None => {
                        // The next token is this flag's value (paths/keys never start with '-').
                        i += 1;
                        args.get(i).cloned()
                    }
                };
                if secret {
                    // Drop the flag and its value entirely: a reproduce command must
                    // never carry a key, and an empty placeholder invites paste errors.
                } else if let Some(v) = value {
                    out.push(name.to_string());
                    out.push(redact_paths(&v));
                } else {
                    out.push(name.to_string());
                }
            }
            None => out.push(a.clone()),
        }
        i += 1;
    }
    out.iter()
        .map(|t| shell_quote(t))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `/home/me/models/a.gguf,/home/me/b.gguf` → `./a.gguf,./b.gguf` (llama-bench
/// accepts comma-separated model lists).
fn redact_paths(v: &str) -> String {
    v.split(',')
        .map(|p| {
            let name = Path::new(p)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(p);
            format!("./{name}")
        })
        .collect::<Vec<_>>()
        .join(",")
}

// ---------------------------------------------------------------------------
// JSON value coercion (llama-bench's jsonl types vary a little across versions)
// ---------------------------------------------------------------------------

fn jnum(row: &Value, key: &str) -> f64 {
    match &row[key] {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::String(s) => s.trim().parse().unwrap_or(0.0),
        Value::Bool(b) => f64::from(*b),
        _ => 0.0,
    }
}

fn jstr(row: &Value, key: &str) -> String {
    match &row[key] {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// llama-bench drop-in
// ---------------------------------------------------------------------------

struct RawBench {
    rows: Vec<Value>,
    stdout: String,
    stderr: String,
    devices: Vec<String>,
    status_ok: bool,
}

/// Run llama-bench with the user's args verbatim (optionally + `-oe jsonl`),
/// streaming stdout (their table) and stderr (load logs) live while capturing
/// both. JSONL result rows are peeled off the stderr stream instead of printed.
fn run_llama_bench_raw(bin_dir: &str, args: &[String], append_oe: bool) -> Result<RawBench> {
    let bin = Path::new(bin_dir).join("llama-bench");
    let mut cmd = Command::new(&bin);
    cmd.args(args);
    if append_oe {
        cmd.args(["-oe", "jsonl"]);
    }
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("running {}", bin.display()))?;

    let child_stderr = child.stderr.take().expect("piped stderr");
    let stderr_thread = thread::spawn(move || {
        let mut rows = Vec::new();
        let mut captured = String::new();
        let mut devices = Vec::new();
        for line in BufReader::new(child_stderr).lines().map_while(Result::ok) {
            let t = line.trim();
            if t.starts_with('{') {
                if let Ok(v) = serde_json::from_str::<Value>(t) {
                    if v.get("avg_ts").is_some() {
                        rows.push(v);
                        continue;
                    }
                }
            }
            if let Some(d) = bench::parse_device(&line) {
                if !devices.contains(&d) {
                    devices.push(d);
                }
            }
            eprintln!("    {line}");
            captured.push_str(&line);
            captured.push('\n');
        }
        (rows, captured, devices)
    });

    let mut stdout = String::new();
    {
        let child_stdout = child.stdout.take().expect("piped stdout");
        for line in BufReader::new(child_stdout).lines().map_while(Result::ok) {
            println!("{line}");
            stdout.push_str(&line);
            stdout.push('\n');
        }
    }
    let status = child.wait().context("waiting for llama-bench")?;
    let (rows, stderr, devices) = stderr_thread.join().unwrap_or_default();
    Ok(RawBench {
        rows,
        stdout,
        stderr,
        devices,
        status_ok: status.success(),
    })
}

/// One benchmark configuration (one submission): a pp row + a tg row that share
/// every non-test field.
struct Group {
    bench: BenchResult,
    context_length: u32,
    ngl: i64,
    threads: Option<i64>,
    model_file: String,
    /// pp/tg/pg/depth rows that didn't fit the standard pp+tg pair.
    skipped_rows: usize,
}

/// Fields that vary per test within one configuration.
const VOLATILE_FIELDS: &[&str] = &[
    "n_prompt",
    "n_gen",
    "n_depth",
    "test_time",
    "avg_ns",
    "stddev_ns",
    "avg_ts",
    "stddev_ts",
    "samples_ns",
    "samples_ts",
];

fn group_key(row: &Value) -> String {
    let mut m = row.as_object().cloned().unwrap_or_default();
    for k in VOLATILE_FIELDS {
        m.remove(*k);
    }
    Value::Object(m).to_string()
}

/// Group jsonl rows into configurations. Within a group the FIRST pp row and FIRST
/// tg row are the submission's numbers (matrix values like `-p 512,1024` produce
/// extra rows — counted in `skipped_rows` and reported, not silently dropped).
fn group_rows(rows: &[Value], banner_devices: &[String]) -> Vec<Group> {
    struct Accum {
        row: Value,
        prefill: f64,
        decode: f64,
        ctx: u32,
        skipped: usize,
    }
    let mut order: Vec<String> = Vec::new();
    let mut map: std::collections::HashMap<String, Accum> = std::collections::HashMap::new();
    for row in rows {
        let key = group_key(row);
        let acc = map.entry(key.clone()).or_insert_with(|| {
            order.push(key.clone());
            Accum {
                row: row.clone(),
                prefill: 0.0,
                decode: 0.0,
                ctx: 0,
                skipped: 0,
            }
        });
        let (np, ng, nd) = (
            jnum(row, "n_prompt"),
            jnum(row, "n_gen"),
            jnum(row, "n_depth"),
        );
        let tps = jnum(row, "avg_ts");
        if nd == 0.0 && np > 0.0 && ng == 0.0 && acc.prefill == 0.0 {
            acc.prefill = tps;
            acc.ctx = np as u32;
        } else if nd == 0.0 && ng > 0.0 && np == 0.0 && acc.decode == 0.0 {
            acc.decode = tps;
        } else {
            acc.skipped += 1;
        }
    }
    order
        .into_iter()
        .filter_map(|key| {
            let acc = map.remove(&key)?;
            if acc.prefill == 0.0 || acc.decode == 0.0 {
                eprintln!(
                    "⚠ a configuration produced only {} tests — a submission needs both a pp and \
                     a tg test; skipping it",
                    if acc.prefill > 0.0 {
                        "prompt (pp)"
                    } else {
                        "generation (tg)"
                    }
                );
                return None;
            }
            let row = &acc.row;
            let mut devices: Vec<String> = banner_devices.to_vec();
            let gpu_info = jstr(row, "gpu_info");
            if devices.is_empty() && !gpu_info.is_empty() {
                devices.push(gpu_info);
            }
            Some(Group {
                bench: BenchResult {
                    model_label: jstr(row, "model_type"),
                    params_b: jnum(row, "model_n_params") / 1e9,
                    backend_label: jstr(row, "backends"),
                    type_k: non_empty_or(jstr(row, "type_k"), "f16"),
                    type_v: non_empty_or(jstr(row, "type_v"), "f16"),
                    flash_attn: jnum(row, "flash_attn") > 0.0,
                    prefill_tps: acc.prefill,
                    decode_tps: acc.decode,
                    build_number: format!("b{}", jstr(row, "build_number")),
                    git_hash: jstr(row, "build_commit"),
                    devices,
                },
                context_length: if acc.ctx > 0 { acc.ctx } else { 512 },
                ngl: jnum(row, "n_gpu_layers") as i64,
                threads: row.get("n_threads").map(|_| jnum(row, "n_threads") as i64),
                model_file: jstr(row, "model_filename"),
                skipped_rows: acc.skipped,
            })
        })
        .collect()
}

fn non_empty_or(s: String, default: &str) -> String {
    if s.is_empty() {
        default.to_string()
    } else {
        s
    }
}

/// The llama-server flags that recreate a bench group's configuration for the
/// output-correctness verification pass.
fn verify_args_for(g: &Group) -> Vec<String> {
    let mut a = vec!["-ngl".to_string(), g.ngl.to_string()];
    a.extend(["-ctk".to_string(), g.bench.type_k.clone()]);
    a.extend(["-ctv".to_string(), g.bench.type_v.clone()]);
    if g.bench.flash_attn {
        a.extend(["-fa".to_string(), "1".to_string()]);
    }
    if let Some(t) = g.threads {
        a.extend(["-t".to_string(), t.to_string()]);
    }
    a
}

fn validate_model_metadata_scope(
    groups: &[Group],
    active_params: Option<f64>,
    base_model: Option<&str>,
) -> Result<()> {
    if active_params.is_none() && base_model.is_none() {
        return Ok(());
    }
    let Some(first_model) = groups.first().map(|group| &group.model_file) else {
        return Ok(());
    };
    if groups
        .iter()
        .any(|group| group.model_file.as_str() != first_model.as_str())
    {
        bail!(
            "--active-params and --base-model apply to one model; split a multi-model llama-bench matrix into separate commands"
        );
    }
    Ok(())
}

fn run_bench(w: &WrapOpts, args: &[String]) -> Result<()> {
    let explicit_model = flag_value(args, &["-m", "--model"]).filter(|v| !v.is_empty());
    if explicit_model.is_none() {
        bail!(
            "pass -m <model.gguf> — llamabench won't benchmark llama-bench's built-in default path"
        );
    }
    let requested_device = selected_device(args);
    validate_selected_device_scope(requested_device.as_deref())?;
    // Fail fast before a long run when there's nothing to submit with.
    let token = if w.dry_run {
        None
    } else {
        Some(submitter::resolve_token(w.token.as_deref())?)
    };
    let required: &[&str] = if w.no_verify {
        &["llama-bench"]
    } else {
        &["llama-bench", "llama-server"]
    };
    let dir = submitter::resolve_llama_dir(&w.llama_dir, w.download_llama, w.family, required)?;

    eprintln!("\n▸ [1/3] Benchmark — llama-bench, your exact command");
    let mut raw = run_llama_bench_raw(&dir, args, true)?;
    if !raw.status_ok && raw.rows.is_empty() && raw.stderr.contains("-oe") {
        // A llama-bench from before -oe existed (early 2024): run the command
        // completely untouched and read the stdout table instead.
        eprintln!("  note: this llama-bench has no -oe — re-running untouched, parsing the table");
        raw = run_llama_bench_raw(&dir, args, false)?;
    }
    let mut groups = group_rows(&raw.rows, &raw.devices);
    if groups.is_empty() {
        // Markdown fallback (old build). Only trustworthy for a single-config run.
        let mut parsed = bench::parse(&raw.stdout, &raw.stderr);
        if parsed.prefill_tps == 0.0 && parsed.decode_tps == 0.0 {
            bail!(
                "could not parse llama-bench output.\nstdout:\n{}\nstderr:\n{}",
                raw.stdout,
                raw.stderr
            );
        }
        // The table omits type_k/type_v at their default — record what the user asked
        // for (or f16), never an empty string.
        let ctk = flag_value(args, &["-ctk", "--cache-type-k"]).filter(|value| !value.is_empty());
        let ctv = flag_value(args, &["-ctv", "--cache-type-v"]).filter(|value| !value.is_empty());
        if parsed.type_k.is_empty() {
            parsed.type_k = ctk.unwrap_or_else(|| "f16".to_string());
        }
        if parsed.type_v.is_empty() {
            parsed.type_v = ctv.unwrap_or_else(|| parsed.type_k.clone());
        }
        let ngl = flag_value(args, &["-ngl", "--n-gpu-layers"])
            .and_then(|v| v.parse().ok())
            .unwrap_or(99);
        let ctx = flag_value(args, &["-p", "--n-prompt"])
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        groups.push(Group {
            bench: parsed,
            context_length: ctx,
            ngl,
            threads: None,
            model_file: explicit_model.clone().unwrap_or_default(),
            skipped_rows: 0,
        });
    }
    validate_model_metadata_scope(&groups, w.active_params, w.base_model.as_deref())?;

    // Some banners enumerate every installed GPU rather than only the selected one.
    // Ask the exact binary for its labelled devices whenever selection is explicit,
    // or when an accelerator row did not report a device at all.
    let needs_device_list = requested_device.is_some()
        || groups
            .iter()
            .any(|g| g.ngl != 0 && g.bench.devices.is_empty());
    if needs_device_list {
        let listed = bench::device_names_for_selection(
            &bench::list_devices(&Path::new(&dir).join("llama-bench")),
            requested_device.as_deref(),
        );
        if !listed.is_empty() {
            for g in &mut groups {
                if g.ngl != 0 && (requested_device.is_some() || g.bench.devices.is_empty()) {
                    g.bench.devices.clone_from(&listed);
                }
            }
        }
    }

    let command = redacted_command("llama-bench", args);
    let total = groups.len();
    for (i, g) in groups.iter().enumerate() {
        if total > 1 {
            eprintln!(
                "\n— configuration {}/{total} ({}) —",
                i + 1,
                g.bench.model_label
            );
        }
        if g.skipped_rows > 0 {
            eprintln!(
                "  note: {} extra test row(s) in this configuration (matrix -p/-n/-pg/depth) \
                 were printed above but not submitted",
                g.skipped_rows
            );
        }
        // The verification pass already pays for a running llama-server at this
        // group's config, so TTFT (standardized ~512-token prompt) rides along.
        let (verification, ttft_ms) = if w.no_verify {
            (None, None)
        } else {
            eprintln!(
                "\n▸ [2/3] Verify — llama-server, TTFT probe + 3 turns × 3 reps (the slow part)"
            );
            match verify::run_verification_with_ttft(&VerifyOpts {
                server_bin_dir: &dir,
                model: &g.model_file,
                port: w.verification_port,
                api_key: "llamabench",
                seed: 42,
                n_gen: 128,
                max_turns: 3,
                reps: 3,
                extra_server_args: verify_args_for(g),
            }) {
                Ok((v, ttft)) => (Some(v), ttft),
                Err(e) => {
                    eprintln!(
                        "⚠ verification could not run ({e}) — submitting without a verification \
                         block (pass --no-verify to skip this step explicitly)"
                    );
                    (None, None)
                }
            }
        };
        let quant = submitter::resolved_quant(None, &g.model_file);
        let hf = submitter::explicit_canonical(
            submitter::provenance(&ModelSource::LocalOnly(&g.model_file), &quant),
            w.base_model.as_deref(),
        )?;
        let ctx = BuildCtx {
            gpu_run: g.ngl != 0,
            selected_device: requested_device.clone(),
            group_visible_gpus: group_visible_gpus(args),
            handle: &w.handle,
            family: w.family,
            command: command.clone(),
            quant: &quant,
            model_path: &g.model_file,
            active_params: w.active_params,
            context_length: g.context_length,
            spec_decode: "none".to_string(),
            ttft_ms,
        };
        let invalid = verification.as_ref().is_some_and(|v| !v.valid);
        let mut s = build_submission(&ctx, &g.bench, verification, hf)?;
        submitter::sign(&mut s)?;
        submitter::emit(&s)?;
        if invalid {
            eprintln!("⚠ verification FAILED: gibberish detected — this result is INVALID");
        }
        eprintln!("\n▸ [3/3] Submit");
        match &token {
            Some(t) => submitter::submit(&w.api, t, &s)?,
            None => eprintln!("(dry run — not submitting)"),
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// llama-server drop-in
// ---------------------------------------------------------------------------

/// Kills the user's spawned server on drop so we never leak a process.
struct KillOnDrop(Child);
impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Standardized speed through the running server: the fixed ~512-token prompt,
/// 128 generated tokens, temp 0 — one warmup, then the median-by-decode of 3 reps.
/// The rates are the server's own `timings`, i.e. exactly what the user's
/// configuration delivers.
fn measure_speed(port: u16, api_key: Option<&str>) -> Result<Timing> {
    let prompt = verify::standard_prompt();
    let _ = verify::http_completion(port, api_key, &prompt, 16).context("warmup request failed")?;
    let mut reps = Vec::new();
    for i in 1..=3 {
        let t = verify::http_completion(port, api_key, &prompt, 128)?;
        eprintln!(
            "  rep {i}: prefill {:.1} t/s · decode {:.1} t/s · ttft {:.0} ms",
            t.prefill, t.decode, t.prompt_ms
        );
        reps.push(t);
    }
    reps.sort_by(|a, b| a.decode.total_cmp(&b.decode));
    Ok(reps.remove(reps.len() / 2))
}

/// "version: 6390 (a8128382)" → ("b6390", "a8128382").
fn parse_version(line: &str) -> Option<(String, String)> {
    let rest = line.trim().strip_prefix("version:")?.trim();
    let mut it = rest.split_whitespace();
    let num = it.next()?;
    if num.is_empty() || !num.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let hash = it.next()?.strip_prefix('(')?.strip_suffix(')')?;
    Some((format!("b{num}"), hash.to_string()))
}

/// The exact build of the user's llama-server, from `--version` (cheap: no model
/// load). Degrades to "unknown" — the run must not die over a banner.
fn server_version(dir: &str) -> (String, String) {
    let bin = Path::new(dir).join("llama-server");
    if let Ok(out) = Command::new(&bin).arg("--version").output() {
        for text in [&out.stderr, &out.stdout] {
            for line in String::from_utf8_lossy(text).lines() {
                if let Some(v) = parse_version(line) {
                    return v;
                }
            }
        }
    }
    ("unknown".to_string(), "unknown".to_string())
}

/// The per-request context window from `/props` (reflects `-c` after the server
/// resolved defaults and `-np` splits).
fn server_ctx(port: u16, api_key: Option<&str>) -> Option<u32> {
    let url = format!("http://127.0.0.1:{port}/props");
    let mut req = ureq::get(&url).timeout(Duration::from_secs(10));
    if let Some(key) = api_key {
        req = req.set("Authorization", &format!("Bearer {key}"));
    }
    let v: Value = req.call().ok()?.into_json().ok()?;
    v["default_generation_settings"]["n_ctx"]
        .as_u64()
        .map(|n| n as u32)
}

fn run_server(w: &WrapOpts, args: &[String]) -> Result<()> {
    let model_path = flag_value(args, &["-m", "--model"]).filter(|v| !v.is_empty());
    let hf_repo_arg = flag_value(args, &["-hf", "-hfr", "--hf-repo"]).filter(|v| !v.is_empty());
    if model_path.is_none() && hf_repo_arg.is_none() {
        bail!("pass -m <model.gguf> (or -hf <repo>[:quant]) so llamabench knows what's being benchmarked");
    }
    let requested_device = selected_device(args);
    validate_selected_device_scope(requested_device.as_deref())?;
    let token = if w.dry_run {
        None
    } else {
        Some(submitter::resolve_token(w.token.as_deref())?)
    };
    let dir =
        submitter::resolve_llama_dir(&w.llama_dir, w.download_llama, w.family, &["llama-server"])?;
    let (build_number, git_hash) = server_version(&dir);
    let port: u16 = flag_value(args, &["--port"])
        .and_then(|v| v.parse().ok())
        .unwrap_or(8080);
    let api_key = flag_value(args, &["--api-key"]).filter(|v| !v.is_empty());

    eprintln!("\n▸ [1/4] Start — llama-server, your exact command");
    let bin = Path::new(&dir).join("llama-server");
    let mut child = Command::new(&bin)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {}", bin.display()))?;
    let devices = Arc::new(Mutex::new(Vec::<String>::new()));
    for stream in [
        child
            .stdout
            .take()
            .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        child
            .stderr
            .take()
            .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
    ]
    .into_iter()
    .flatten()
    {
        let devices = Arc::clone(&devices);
        thread::spawn(move || {
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                if let Some(d) = bench::parse_device(&line) {
                    let mut devs = devices.lock().unwrap();
                    if !devs.contains(&d) {
                        devs.push(d);
                    }
                }
                eprintln!("    {line}");
            }
        });
    }
    let _guard = KillOnDrop(child);
    verify::wait_until_ready(port, api_key.as_deref(), Duration::from_secs(240))
        .context("llama-server did not become ready")?;

    eprintln!("\n▸ [2/4] Measure — prefill + decode from the server's own timings");
    let speed = measure_speed(port, api_key.as_deref())?;

    let verification = if w.no_verify {
        None
    } else {
        eprintln!("\n▸ [3/4] Verify — 3 turns × 3 reps (the slow part)");
        match verify::verify_running(&VerifySession {
            port,
            api_key: api_key.as_deref(),
            seed: 42,
            n_gen: 128,
            max_turns: 3,
            reps: 3,
        }) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!(
                    "⚠ verification could not run ({e}) — submitting without a verification block"
                );
                None
            }
        }
    };

    // Model identity: a local -m file (link-store provenance), or the server's own
    // -hf download (repo[:quant] ⇒ trivially verified provenance).
    let (model_label, quant, hf) = match (&model_path, &hf_repo_arg) {
        (Some(m), _) => {
            let quant = submitter::resolved_quant(None, m);
            let hf = submitter::explicit_canonical(
                submitter::provenance(&ModelSource::LocalOnly(m), &quant),
                w.base_model.as_deref(),
            )?;
            (submitter::model_name(m), quant, hf)
        }
        (None, Some(spec)) => {
            let (repo, tag) = match spec.split_once(':') {
                Some((r, t)) => (r.to_string(), Some(t.to_string())),
                None => (spec.clone(), None),
            };
            let label = repo.rsplit('/').next().unwrap_or(&repo).to_string();
            let quant = tag.unwrap_or_else(|| "unknown".to_string());
            let hf = submitter::explicit_canonical(
                submitter::provenance(&ModelSource::Downloaded(&repo), &quant),
                w.base_model.as_deref(),
            )?;
            (label, quant, hf)
        }
        (None, None) => unreachable!("guarded above"),
    };
    let context_length = server_ctx(port, api_key.as_deref())
        .or_else(|| flag_value(args, &["-c", "--ctx-size"]).and_then(|v| v.parse().ok()))
        .filter(|c| *c > 0)
        .unwrap_or(4096);
    let type_k = flag_value(args, &["-ctk", "--cache-type-k"])
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "f16".to_string());
    let type_v = flag_value(args, &["-ctv", "--cache-type-v"])
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| type_k.clone());
    // Recorded from the flag: a bare `-fa`, on/1/true, or auto (the modern default
    // resolves to "on" wherever the device supports it) count as enabled.
    let flash_attn = matches!(
        flag_value(args, &["-fa", "--flash-attn"]).as_deref(),
        Some("" | "on" | "1" | "true" | "enabled" | "auto")
    );
    let spec_decode = flag_value(args, &["--spec-type"])
        .filter(|v| !v.is_empty())
        .or_else(|| flag_value(args, &["-md", "--model-draft"]).map(|_| "draft".to_string()))
        .unwrap_or_else(|| "none".to_string());
    let ngl = flag_value(args, &["-ngl", "--n-gpu-layers", "--gpu-layers"])
        .and_then(|v| v.parse::<i64>().ok());
    let gpu_run = ngl.map_or_else(|| !devices.lock().unwrap().is_empty(), |n| n != 0);

    let listed_devices = if gpu_run {
        bench::list_devices(&bin)
    } else {
        Vec::new()
    };
    let mut detected_devices = devices.lock().unwrap().clone();
    let selected_names =
        bench::device_names_for_selection(&listed_devices, requested_device.as_deref());
    if gpu_run
        && !selected_names.is_empty()
        && (requested_device.is_some() || detected_devices.is_empty())
    {
        detected_devices = selected_names;
    }
    let backend_label = detected_devices
        .first()
        .and_then(|name| {
            bench::backend_for_selection(&listed_devices, requested_device.as_deref(), name)
        })
        .unwrap_or_default();
    let bench = BenchResult {
        model_label: model_label.clone(),
        params_b: submitter::params_from_name(&model_label),
        backend_label,
        type_k,
        type_v,
        flash_attn,
        prefill_tps: speed.prefill,
        decode_tps: speed.decode,
        build_number,
        git_hash,
        devices: detected_devices,
    };
    let model_path_or_label = model_path.as_deref().unwrap_or(&model_label);
    let ctx = BuildCtx {
        gpu_run,
        selected_device: requested_device,
        group_visible_gpus: group_visible_gpus(args),
        handle: &w.handle,
        family: w.family,
        command: redacted_command("llama-server", args),
        quant: &quant,
        model_path: model_path_or_label,
        active_params: w.active_params,
        context_length,
        spec_decode,
        ttft_ms: Some(speed.prompt_ms.round() as u32),
    };
    let invalid = verification.as_ref().is_some_and(|v| !v.valid);
    let mut s = build_submission(&ctx, &bench, verification, hf)?;
    submitter::sign(&mut s)?;
    submitter::emit(&s)?;
    if invalid {
        eprintln!("⚠ verification FAILED: gibberish detected — this result is INVALID");
    }
    eprintln!("\n▸ [4/4] Submit");
    match &token {
        Some(t) => submitter::submit(&w.api, t, &s)?,
        None => eprintln!("(dry run — not submitting)"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn v(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn wrapper_flags_extracted_tool_args_untouched() {
        let (w, rest) = extract_wrapper_flags(&v(&[
            "-m",
            "/x/m.gguf",
            "--dry-run",
            "-ngl",
            "99",
            "--family=ik_llama.cpp",
            "--handle",
            "@edu",
            "-fa",
            "1",
            "--llama-dir",
            "/opt/llama",
            "--verification-port",
            "18080",
        ]))
        .unwrap();
        assert!(w.dry_run);
        assert!(!w.no_verify);
        assert_eq!(w.handle, "@edu");
        assert_eq!(w.llama_dir, "/opt/llama");
        assert_eq!(w.verification_port, 18080);
        assert!(matches!(w.family, Family::IkLlamaCpp));
        // Only our flags are removed; order and values of the tool's args survive.
        assert_eq!(rest, v(&["-m", "/x/m.gguf", "-ngl", "99", "-fa", "1"]));

        for invalid in ["-1", "0", "NaN", "inf"] {
            assert!(
                extract_wrapper_flags(&v(&["--active-params", invalid, "-m", "m.gguf"])).is_err()
            );
        }
        assert!(extract_wrapper_flags(&v(&["--verification-port", "0", "-m", "m.gguf"])).is_err());
    }

    #[test]
    fn explicit_model_metadata_rejects_multi_model_matrices() {
        let group = |model_file: &str| Group {
            bench: BenchResult::default(),
            context_length: 512,
            ngl: -1,
            threads: None,
            model_file: model_file.to_string(),
            skipped_rows: 0,
        };
        assert!(validate_model_metadata_scope(
            &[group("moe.gguf"), group("moe.gguf")],
            Some(3.0),
            Some("owner/moe")
        )
        .is_ok());
        assert!(validate_model_metadata_scope(
            &[group("moe.gguf"), group("dense.gguf")],
            Some(3.0),
            None
        )
        .unwrap_err()
        .to_string()
        .contains("split a multi-model"));
        assert!(validate_model_metadata_scope(
            &[group("moe.gguf"), group("dense.gguf")],
            None,
            Some("owner/moe")
        )
        .unwrap_err()
        .to_string()
        .contains("split a multi-model"));
        assert!(validate_model_metadata_scope(
            &[group("moe.gguf"), group("dense.gguf")],
            None,
            None
        )
        .is_ok());
    }

    #[test]
    fn sniffs_server_flags() {
        assert!(matches!(
            sniff_mode(&v(&["-m", "m.gguf", "--port", "8081"])),
            Mode::Server
        ));
        assert!(matches!(
            sniff_mode(&v(&["-m", "m.gguf", "--ctx-size=8192"])),
            Mode::Server
        ));
        // llama-bench flags only → bench mode.
        assert!(matches!(
            sniff_mode(&v(&["-m", "m.gguf", "-ngl", "99", "-p", "512"])),
            Mode::Bench
        ));
    }

    #[test]
    fn flag_value_last_wins_and_forms() {
        let args = v(&["-m", "a.gguf", "--model=b.gguf", "-ngl", "-1", "-fa"]);
        assert_eq!(
            flag_value(&args, &["-m", "--model"]).as_deref(),
            Some("b.gguf")
        );
        // A negative number IS the value, not the next flag.
        assert_eq!(flag_value(&args, &["-ngl"]).as_deref(), Some("-1"));
        // Bare trailing flag → empty value, not None.
        assert_eq!(flag_value(&args, &["-fa"]).as_deref(), Some(""));
        assert_eq!(flag_value(&args, &["--port"]), None);
        assert_eq!(
            selected_device(&v(&["--device=Vulkan0", "-dev", "Vulkan1,Vulkan0"])),
            Some("Vulkan1,Vulkan0".to_string())
        );
        assert!(validate_selected_device_scope(Some("CUDA0")).is_ok());
        assert!(validate_selected_device_scope(None).is_ok());
        assert!(validate_selected_device_scope(Some("CUDA0,CUDA1"))
            .unwrap_err()
            .to_string()
            .contains("multiple explicit --device"));
        assert!(group_visible_gpus(&v(&["--split-mode", "layer"])));
        assert!(group_visible_gpus(&v(&["-sm", "row"])));
        assert!(!group_visible_gpus(&v(&["--split-mode=none"])));
        assert!(!group_visible_gpus(&v(&["-sm", "NONE"])));
    }

    #[test]
    fn command_redacts_paths_and_drops_secrets() {
        let cmd = redacted_command(
            "llama-server",
            &v(&[
                "-m",
                "/home/edu/models/gemma-4-12b-it-UD-Q4_K_XL.gguf",
                "--api-key",
                "sekrit",
                "-c",
                "8192",
                "--chat-template",
                "a b",
            ]),
        );
        assert_eq!(
            cmd,
            "llama-server -m ./gemma-4-12b-it-UD-Q4_K_XL.gguf -c 8192 --chat-template 'a b'"
        );
        // Comma-separated model lists keep the list shape.
        let cmd = redacted_command("llama-bench", &v(&["-m", "/a/x.gguf,/b/y.gguf"]));
        assert_eq!(cmd, "llama-bench -m ./x.gguf,./y.gguf");
        // --flag=value form is redacted too.
        let cmd = redacted_command("llama-bench", &v(&["-m=/a/x.gguf", "-p", "512"]));
        assert_eq!(cmd, "llama-bench -m ./x.gguf -p 512");
    }

    #[test]
    fn groups_matrix_rows_into_configs() {
        // Two configurations (ngl 0 and 99), each with a pp and a tg row — plus one
        // extra tg row in the second config that must be counted, not submitted.
        let mk = |ngl: i64, np: u32, ng: u32, ts: f64| {
            json!({
                "build_commit": "a8128382", "build_number": 6390,
                "model_filename": "/x/gemma-12b-Q4_K_M.gguf", "model_type": "gemma 12B Q4_K",
                "model_n_params": 11_900_000_000u64, "backends": "Metal",
                "gpu_info": "Apple M4", "type_k": "f16", "type_v": "f16",
                "n_gpu_layers": ngl, "n_threads": 8, "flash_attn": 1,
                "n_prompt": np, "n_gen": ng, "n_depth": 0, "avg_ts": ts,
            })
        };
        let rows = vec![
            mk(0, 512, 0, 30.0),
            mk(0, 0, 128, 10.0),
            mk(99, 512, 0, 300.0),
            mk(99, 0, 128, 60.0),
            mk(99, 0, 256, 55.0),
        ];
        let groups = group_rows(&rows, &[]);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].ngl, 0);
        assert!((groups[0].bench.prefill_tps - 30.0).abs() < 1e-9);
        assert!((groups[0].bench.decode_tps - 10.0).abs() < 1e-9);
        assert_eq!(groups[0].context_length, 512);
        assert_eq!(groups[0].skipped_rows, 0);
        assert_eq!(groups[1].ngl, 99);
        assert!((groups[1].bench.decode_tps - 60.0).abs() < 1e-9);
        assert_eq!(groups[1].skipped_rows, 1);
        // Row-derived fields land in the BenchResult.
        assert!((groups[1].bench.params_b - 11.9).abs() < 0.01);
        assert_eq!(groups[1].bench.build_number, "b6390");
        assert_eq!(groups[1].bench.git_hash, "a8128382");
        assert_eq!(groups[1].bench.devices, vec!["Apple M4".to_string()]);
        assert!(groups[1].bench.flash_attn);
        // A config with only a pp row is skipped, not half-submitted.
        assert!(group_rows(&[mk(1, 512, 0, 30.0)], &[]).is_empty());
    }

    #[test]
    fn verify_args_mirror_group_config() {
        let groups = group_rows(
            &[
                json!({"model_filename": "m.gguf", "n_gpu_layers": 99, "n_threads": 8,
                       "type_k": "q8_0", "type_v": "q8_0", "flash_attn": 1, "build_number": 1,
                       "build_commit": "x", "n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ts": 1.0}),
                json!({"model_filename": "m.gguf", "n_gpu_layers": 99, "n_threads": 8,
                       "type_k": "q8_0", "type_v": "q8_0", "flash_attn": 1, "build_number": 1,
                       "build_commit": "x", "n_prompt": 0, "n_gen": 128, "n_depth": 0, "avg_ts": 1.0}),
            ],
            &[],
        );
        assert_eq!(
            verify_args_for(&groups[0]),
            v(&["-ngl", "99", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "1", "-t", "8"])
        );
    }

    #[test]
    fn version_line_parses() {
        assert_eq!(
            parse_version("version: 6390 (a8128382)"),
            Some(("b6390".to_string(), "a8128382".to_string()))
        );
        assert_eq!(parse_version("built with Apple clang"), None);
        assert_eq!(parse_version("version: dev (x)"), None);
    }

    #[test]
    fn quoting() {
        assert_eq!(shell_quote("plain-1.0/x=y"), "plain-1.0/x=y");
        assert_eq!(shell_quote("two words"), "'two words'");
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
    }
}
