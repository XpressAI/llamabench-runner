// SPDX-License-Identifier: GPL-3.0-or-later
//! Opt-in exact-configuration behavior evaluation (ADR-014).
//!
//! Starts the user's existing llama-server once, then runs two inspectable visual
//! prompts, two deterministic virtual-tool tasks, and a three-turn public-domain
//! role-play. The harness uses fixed prompts/settings and executable or rule-based
//! checks; it never calls a remote judge.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use crate::contract::{
    AgenticTaskResult, EvaluationCheck, EvaluationSettings, EvaluationSubmission,
    EvaluationVisuals, RoleplayResult, RoleplayTurn, SpeculativeDecodingConfig, ToolCallRecord,
    VisualArtifact,
};
use crate::verify::{self, VerifyOpts};

pub const VISUAL_MAX_TOKENS: u32 = 1024;
pub const AGENT_TASK_MAX_TOKENS: u32 = 256;
pub const ROLEPLAY_MAX_TOKENS_PER_TURN: u32 = 96;
pub const SEED: u64 = 42;

const PELICAN_PROMPT: &str = "Generate an SVG of a pelican riding a bicycle";
const BREAKOUT_PROMPT: &str = "Can you make a simple breakout game in HTML?";
const MAX_OUTPUT_CHARS: usize = 131_072;
const MAX_EVIDENCE_CHARS: usize = 4_000;
const CONTROLLED_SERVER_FLAGS: &[&str] = &[
    "-m",
    "--model",
    "-mu",
    "--model-url",
    "-dr",
    "--docker-repo",
    "-hf",
    "-hfr",
    "--hf-repo",
    "-hff",
    "--hf-file",
    "-hft",
    "--hf-token",
    "--host",
    "--port",
    "--api-key",
    "--api-key-file",
    "-c",
    "--ctx-size",
    "--ssl-key-file",
    "--ssl-cert-file",
    "--log-file",
    "--log-prompts-dir",
    "--slot-save-path",
];
const UNTRACKED_ARTIFACT_FLAGS: &[&str] = &[
    "-md",
    "--model-draft",
    "--spec-draft-model",
    "--spec-draft-hf",
    "-hfd",
    "-hfrd",
    "--hf-repo-draft",
    "-mm",
    "--mmproj",
    "-mmu",
    "--mmproj-url",
    "--chat-template-file",
    "--grammar-file",
    "-jf",
    "--json-schema-file",
    "--lora",
    "--lora-scaled",
    "--control-vector",
    "--control-vector-scaled",
    "-lcs",
    "--lookup-cache-static",
    "-lcd",
    "--lookup-cache-dynamic",
];
const SPEC_PARAMETER_SHORT_FLAGS: &[&str] = &[
    "-ctkd", "-ctvd", "-td", "-tbd", "-Cd", "-Crd", "-Cbd", "-otd", "-cmoed", "-ncmoed", "-devd",
    "-ngld", "-cd",
];

pub struct EvaluationOpts<'a> {
    pub server_bin_dir: &'a str,
    pub model: &'a str,
    pub port: u16,
    pub api_key: &'a str,
    pub context_length: u32,
    pub runtime_args: Vec<String>,
}

pub struct RuntimeConfig {
    pub args: Vec<String>,
    pub kv_cache_key: String,
    pub kv_cache_value: String,
    pub flash_attention: String,
    pub speculative_decoding: SpeculativeDecodingConfig,
}

pub struct EvaluationRun {
    pub visuals: EvaluationVisuals,
    pub agentic_tasks: Vec<AgenticTaskResult>,
    pub roleplay: RoleplayResult,
    pub context_length: u32,
    pub runtime: RuntimeConfig,
}

struct ChatReply {
    message: Value,
    content: String,
    reasoning: String,
    generated_tokens: u32,
}

impl ChatReply {
    fn visible_output(&self) -> String {
        if self.content.trim().is_empty() {
            self.reasoning.clone()
        } else {
            self.content.clone()
        }
    }
}

#[derive(Clone)]
struct RequestedTool {
    id: String,
    name: String,
    arguments: String,
}

pub fn settings() -> EvaluationSettings {
    EvaluationSettings {
        seed: SEED,
        temperature: 0.0,
        visual_max_tokens: VISUAL_MAX_TOKENS,
        agent_task_max_tokens: AGENT_TASK_MAX_TOKENS,
        roleplay_max_tokens_per_turn: ROLEPLAY_MAX_TOKENS_PER_TURN,
    }
}

pub fn run_eval(opts: &EvaluationOpts) -> Result<EvaluationRun> {
    ensure_explicit_runtime_environment()?;
    let runtime = runtime_config(&opts.runtime_args)?;
    let mut server_args = vec!["-c".to_string(), opts.context_length.to_string()];
    server_args.extend(opts.runtime_args.clone());
    let verify_opts = VerifyOpts {
        server_bin_dir: opts.server_bin_dir,
        model: opts.model,
        port: opts.port,
        api_key: opts.api_key,
        seed: SEED,
        n_gen: 128,
        max_turns: 1,
        reps: 1,
        extra_server_args: server_args,
    };
    let _guard = verify::spawn_server(&verify_opts)?;
    let context_length = server_context(opts.port, Some(opts.api_key))?;

    eprintln!("\n▸ [1/5] Visual — pelican SVG (≤ {VISUAL_MAX_TOKENS} generated tokens)");
    let pelican_svg = run_visual(
        opts,
        "pelican-svg-v1",
        "Return only one self-contained SVG document. Use viewBox=\"0 0 800 600\" and vector shapes only. Do not use scripts, foreignObject, links, embedded files, or external resources.",
        PELICAN_PROMPT,
        false,
    )?;

    eprintln!("\n▸ [2/5] Visual — breakout HTML (≤ {VISUAL_MAX_TOKENS} generated tokens)");
    let breakout_html = run_visual(
        opts,
        "breakout-html-v1",
        "Return only one self-contained HTML document with inline CSS and JavaScript. Build a visible ball, paddle, and bricks; include keyboard controls, collision handling, scoring, and a game loop. Do not use markdown fences, network requests, links, imports, or external resources.",
        BREAKOUT_PROMPT,
        true,
    )?;

    eprintln!("\n▸ [3/5] Agentic — discover a value through the virtual filesystem");
    let discovery = run_agent_task(opts, DiscoveryHarness::default())?;
    eprintln!(
        "  {} workspace discovery",
        if discovery.passed { "✓" } else { "✗" }
    );

    eprintln!("\n▸ [4/5] Agentic — repair one config and run its tests");
    let repair = run_agent_task(opts, RepairHarness::default())?;
    eprintln!(
        "  {} constrained config repair",
        if repair.passed { "✓" } else { "✗" }
    );

    eprintln!("\n▸ [5/5] Role-play — Phileas Fogg, three turns");
    let roleplay = run_roleplay(opts)?;

    Ok(EvaluationRun {
        visuals: EvaluationVisuals {
            pelican_svg,
            breakout_html,
        },
        agentic_tasks: vec![discovery, repair],
        roleplay,
        context_length,
        runtime,
    })
}

fn runtime_flag(arg: &str) -> &str {
    arg.split_once('=').map_or(arg, |(flag, _)| flag)
}

fn follows_flag(value: Option<&String>) -> bool {
    value.is_some_and(|value| !value.starts_with('-') || value.parse::<f64>().is_ok())
}

fn last_runtime_value(args: &[String], names: &[&str]) -> Option<String> {
    let mut value = None;
    let mut index = 0;
    while index < args.len() {
        let token = &args[index];
        if let Some((flag, inline)) = token.split_once('=') {
            if names.contains(&flag) {
                value = Some(inline.to_string());
            }
        } else if names.contains(&token.as_str()) {
            if follows_flag(args.get(index + 1)) {
                index += 1;
                value = Some(args[index].clone());
            } else {
                value = Some(String::new());
            }
        }
        index += 1;
    }
    value
}

fn canonical_runtime_flags(args: &[String], predicate: impl Fn(&str) -> bool) -> Vec<String> {
    let mut values = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let token = &args[index];
        let flag = runtime_flag(token);
        if predicate(flag) {
            if token.contains('=') {
                values.push(token.clone());
            } else if follows_flag(args.get(index + 1)) {
                index += 1;
                values.push(format!("{token}={}", args[index]));
            } else {
                values.push(token.clone());
            }
        }
        index += 1;
    }
    values
}

fn is_model_preset_flag(flag: &str) -> bool {
    (flag.ends_with("-default") && flag != "--spec-default")
        || (flag.starts_with("--fim-") && flag.ends_with("-spec"))
}

fn is_spec_parameter_flag(flag: &str) -> bool {
    ((flag.starts_with("--spec-") || flag.starts_with("--no-spec-"))
        && !matches!(flag, "--spec-type" | "--spec-default"))
        || flag.starts_with("--draft")
        || flag.ends_with("-draft")
        || SPEC_PARAMETER_SHORT_FLAGS.contains(&flag)
}

fn speculative_mode(args: &[String]) -> String {
    let mut mode = "none".to_string();
    let mut index = 0;
    while index < args.len() {
        let token = &args[index];
        let flag = runtime_flag(token);
        if flag == "--spec-default" {
            mode = "default".to_string();
        } else if flag == "--spec-type" {
            if let Some((_, inline)) = token.split_once('=') {
                if !inline.is_empty() {
                    mode = inline.to_string();
                }
            } else if follows_flag(args.get(index + 1)) {
                index += 1;
                mode = args[index].clone();
            }
        }
        index += 1;
    }
    mode
}

/// llama.cpp accepts most native options through `LLAMA_ARG_*` environment
/// variables. Inheriting one would make the published argument vector incomplete,
/// so exact-config commands fail fast and ask for an explicit native flag instead.
fn runtime_environment_conflicts(names: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut conflicts = names
        .into_iter()
        .filter(|name| {
            name.as_bytes()
                .get(.."LLAMA_ARG_".len())
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"LLAMA_ARG_"))
        })
        .collect::<Vec<_>>();
    conflicts.sort();
    conflicts
}

pub fn ensure_explicit_runtime_environment() -> Result<()> {
    let conflicts = runtime_environment_conflicts(
        std::env::vars_os().filter_map(|(name, _)| name.into_string().ok()),
    );
    if !conflicts.is_empty() {
        bail!(
            "exact-config commands cannot inherit llama.cpp runtime settings from {}; unset them and pass the equivalent native flags after `--`",
            conflicts.join(", ")
        );
    }
    Ok(())
}

fn looks_like_absolute_path(value: &str) -> bool {
    let bytes = value.as_bytes();
    Path::new(value).is_absolute()
        || value.starts_with("\\\\")
        || (bytes.len() >= 3
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && matches!(bytes[2], b'/' | b'\\'))
}

fn exact_runtime_args(args: &[String]) -> Result<Vec<String>> {
    for arg in args {
        let value = arg.split_once('=').map_or(arg.as_str(), |(_, value)| value);
        if looks_like_absolute_path(value) {
            bail!(
                "eval-v1 cannot publish absolute native argument value {value:?} without losing exact identity; use a relative value or remove that option"
            );
        }
    }
    Ok(args.to_vec())
}

pub fn runtime_config(args: &[String]) -> Result<RuntimeConfig> {
    if args.len() > 256 {
        bail!("eval-v1 accepts at most 256 native arguments");
    }
    for arg in args {
        let length = arg.chars().count();
        if length == 0 || length > 2_000 {
            bail!("each eval-v1 native argument must contain 1..=2000 characters");
        }
        let flag = runtime_flag(arg);
        if CONTROLLED_SERVER_FLAGS.contains(&flag) || is_model_preset_flag(flag) {
            bail!(
                "{flag} is controlled by `llamabench eval`; put model, context, and transport settings before `--`"
            );
        }
        if UNTRACKED_ARTIFACT_FLAGS.contains(&flag) {
            bail!("{flag} supplies an auxiliary artifact that eval-v1 cannot hash and represent");
        }
    }
    let kv_cache_key = last_runtime_value(args, &["-ctk", "--cache-type-k"])
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "f16".to_string());
    let kv_cache_value = last_runtime_value(args, &["-ctv", "--cache-type-v"])
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| kv_cache_key.clone());
    let flash_attention = match last_runtime_value(args, &["-fa", "--flash-attn"])
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        None | Some("auto") => "auto",
        Some("" | "on" | "1" | "true" | "enabled") => "on",
        Some("off" | "0" | "false" | "disabled") => "off",
        Some(value) => bail!("unsupported flash-attention value {value:?}"),
    }
    .to_string();
    let mode = speculative_mode(args);
    let exact_args = exact_runtime_args(args)?;
    let parameters = canonical_runtime_flags(&exact_args, is_spec_parameter_flag);
    if parameters.len() > 64 || parameters.iter().any(|value| value.chars().count() > 500) {
        bail!("eval-v1 accepts at most 64 speculative overrides of at most 500 characters each");
    }
    Ok(RuntimeConfig {
        args: exact_args,
        kv_cache_key,
        kv_cache_value,
        flash_attention,
        speculative_decoding: SpeculativeDecodingConfig { mode, parameters },
    })
}

fn run_visual(
    opts: &EvaluationOpts,
    prompt_id: &str,
    system: &str,
    prompt: &str,
    breakout: bool,
) -> Result<VisualArtifact> {
    let start = Instant::now();
    let messages = vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": prompt}),
    ];
    let reply = chat(opts, &messages, None, VISUAL_MAX_TOKENS)?;
    let raw = reply.visible_output();
    let output = if breakout {
        extract_html(&raw).unwrap_or_else(|| bounded(&raw, MAX_OUTPUT_CHARS))
    } else {
        extract_svg(&raw).unwrap_or_else(|| bounded(&raw, MAX_OUTPUT_CHARS))
    };
    let checks = if breakout {
        breakout_checks(&output)
    } else {
        Vec::new()
    };
    Ok(VisualArtifact {
        prompt_id: prompt_id.to_string(),
        output_sha256: sha256_hex(&output),
        output,
        duration_ms: elapsed_ms(start),
        generated_tokens: reply.generated_tokens.min(VISUAL_MAX_TOKENS),
        checks,
    })
}

fn chat(
    opts: &EvaluationOpts,
    messages: &[Value],
    tools: Option<&Value>,
    max_tokens: u32,
) -> Result<ChatReply> {
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", opts.port);
    let mut body = json!({
        "messages": messages,
        "seed": SEED,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": false,
    });
    if let Some(tools) = tools {
        body["tools"] = tools.clone();
        body["tool_choice"] = json!("auto");
    }
    let resp = ureq::post(&url)
        .timeout(Duration::from_secs(1_500))
        .set("Authorization", &format!("Bearer {}", opts.api_key))
        .send_json(body)
        .map_err(|e| anyhow::anyhow!("eval chat request failed: {e}"))?;
    let v: Value = resp.into_json().context("decoding eval chat response")?;
    let message = v["choices"][0]["message"].clone();
    if !message.is_object() {
        bail!("eval response carried no assistant message");
    }
    let content = message["content"].as_str().unwrap_or_default().to_string();
    let reasoning = message["reasoning_content"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let generated_tokens = v["usage"]["completion_tokens"]
        .as_u64()
        .map(|n| n as u32)
        .unwrap_or_else(|| estimate_tokens(&message));
    Ok(ChatReply {
        message,
        content,
        reasoning,
        generated_tokens: generated_tokens.min(max_tokens),
    })
}

trait AgentHarness {
    fn id(&self) -> &'static str;
    fn prompt(&self) -> &'static str;
    fn tools(&self) -> Value;
    fn execute(&mut self, name: &str, arguments: &str) -> String;
    fn passed(&self, final_output: &str) -> bool;
}

fn run_agent_task<H: AgentHarness>(
    opts: &EvaluationOpts,
    mut harness: H,
) -> Result<AgenticTaskResult> {
    let start = Instant::now();
    let system = "You are operating a tiny deterministic workspace. Use the supplied tools to complete the task. Do not invent tool results. Make only requested changes, verify the final state, then answer briefly.";
    let mut messages = vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": harness.prompt()}),
    ];
    let tools = harness.tools();
    let mut records = Vec::new();
    let mut generated_tokens = 0u32;
    let mut final_output = String::new();
    let mut failure = None;

    for _ in 0..8 {
        let remaining = AGENT_TASK_MAX_TOKENS.saturating_sub(generated_tokens);
        if remaining == 0 {
            failure = Some("generation token ceiling reached before completion".to_string());
            break;
        }
        let reply = match chat(opts, &messages, Some(&tools), remaining.min(96)) {
            Ok(reply) => reply,
            Err(e) => {
                failure = Some(format!("engine error: {e}"));
                break;
            }
        };
        generated_tokens = (generated_tokens + reply.generated_tokens).min(AGENT_TASK_MAX_TOKENS);
        let requested = requested_tools(&reply.message);
        messages.push(reply.message.clone());
        if requested.is_empty() {
            final_output = bounded(&reply.visible_output(), MAX_EVIDENCE_CHARS);
            break;
        }
        for call in requested {
            let result = harness.execute(&call.name, &call.arguments);
            records.push(ToolCallRecord {
                name: bounded(&call.name, 100),
                arguments: bounded(&call.arguments, 2_000),
                result: bounded(&result, 2_000),
            });
            messages.push(json!({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": result,
            }));
        }
    }
    if final_output.is_empty() && failure.is_none() {
        failure = Some("model did not return a final answer within eight tool turns".to_string());
    }
    let passed = failure.is_none() && harness.passed(&final_output);
    if !passed && failure.is_none() {
        failure = Some("final tool state did not satisfy the task".to_string());
    }
    Ok(AgenticTaskResult {
        id: harness.id().to_string(),
        passed,
        duration_ms: elapsed_ms(start),
        generated_tokens,
        tool_calls: records,
        final_output_sha256: sha256_hex(&final_output),
        final_output,
        failure,
    })
}

#[derive(Default)]
struct DiscoveryHarness {
    read_readme: bool,
    read_release: bool,
    invalid_calls: u32,
}

impl AgentHarness for DiscoveryHarness {
    fn id(&self) -> &'static str {
        "workspace-discovery-v1"
    }

    fn prompt(&self) -> &'static str {
        "Find the release codename recorded in this workspace. Start at /workspace/README.md, follow its instructions with read_file, and return only the codename."
    }

    fn tools(&self) -> Value {
        json!([{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read one UTF-8 file from the virtual workspace.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["path"],
                    "properties": {"path": {"type": "string"}}
                }
            }
        }])
    }

    fn execute(&mut self, name: &str, arguments: &str) -> String {
        if name != "read_file" {
            self.invalid_calls += 1;
            return "ERROR: unknown tool".to_string();
        }
        let path = json_string_arg(arguments, "path");
        match path.as_deref() {
            Some("/workspace/README.md") => {
                self.read_readme = true;
                "Release metadata lives in /workspace/config/release.txt. Read that file before answering."
                    .to_string()
            }
            Some("/workspace/config/release.txt") => {
                if !self.read_readme {
                    self.invalid_calls += 1;
                    return "ERROR: read /workspace/README.md first".to_string();
                }
                self.read_release = true;
                "codename=NAUTILUS-47\nchannel=stable\n".to_string()
            }
            _ => {
                self.invalid_calls += 1;
                "ERROR: file not found".to_string()
            }
        }
    }

    fn passed(&self, final_output: &str) -> bool {
        self.read_readme
            && self.read_release
            && self.invalid_calls == 0
            && final_output.trim().eq_ignore_ascii_case("NAUTILUS-47")
    }
}

struct RepairHarness {
    config: String,
    read_config: bool,
    read_before_write: bool,
    wrote_config: bool,
    ran_passing_tests: bool,
    invalid_writes: u32,
}

impl Default for RepairHarness {
    fn default() -> Self {
        Self {
            config: r#"{"port":"8080","debug":true}"#.to_string(),
            read_config: false,
            read_before_write: false,
            wrote_config: false,
            ran_passing_tests: false,
            invalid_writes: 0,
        }
    }
}

impl RepairHarness {
    fn tests_pass(&self) -> bool {
        serde_json::from_str::<Value>(&self.config)
            .ok()
            .is_some_and(|v| {
                v["port"].as_u64() == Some(8080) && v["debug"].as_bool() == Some(false)
            })
    }
}

impl AgentHarness for RepairHarness {
    fn id(&self) -> &'static str {
        "config-repair-v1"
    }

    fn prompt(&self) -> &'static str {
        "The workspace tests fail because /workspace/config.json has the wrong value types. Fix only that file so port is the number 8080 and debug is false, then run the tests."
    }

    fn tools(&self) -> Value {
        json!([
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read one UTF-8 file from the virtual workspace.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["path"],
                        "properties": {"path": {"type": "string"}}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Replace one UTF-8 file in the virtual workspace.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["path", "content"],
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_tests",
                    "description": "Run the deterministic tests for the virtual workspace.",
                    "parameters": {"type": "object", "additionalProperties": false}
                }
            }
        ])
    }

    fn execute(&mut self, name: &str, arguments: &str) -> String {
        match name {
            "read_file" => match json_string_arg(arguments, "path").as_deref() {
                Some("/workspace/config.json") => {
                    self.read_config = true;
                    self.config.clone()
                }
                _ => "ERROR: file not found".to_string(),
            },
            "write_file" => {
                let path = json_string_arg(arguments, "path");
                let content = json_string_arg(arguments, "content");
                if path.as_deref() != Some("/workspace/config.json") {
                    self.invalid_writes += 1;
                    return "ERROR: writes outside /workspace/config.json are forbidden"
                        .to_string();
                }
                let Some(content) = content else {
                    return "ERROR: content must be a string".to_string();
                };
                self.read_before_write |= self.read_config;
                self.config = bounded(&content, 4_000);
                self.wrote_config = true;
                "OK: wrote /workspace/config.json".to_string()
            }
            "run_tests" => {
                if self.tests_pass() {
                    self.ran_passing_tests = true;
                    "PASS: port is numeric 8080 and debug is false".to_string()
                } else {
                    "FAIL: expected numeric port 8080 and debug false".to_string()
                }
            }
            _ => "ERROR: unknown tool".to_string(),
        }
    }

    fn passed(&self, _final_output: &str) -> bool {
        self.read_config
            && self.read_before_write
            && self.wrote_config
            && self.invalid_writes == 0
            && self.ran_passing_tests
            && self.tests_pass()
    }
}

fn run_roleplay(opts: &EvaluationOpts) -> Result<RoleplayResult> {
    let start = Instant::now();
    let mut messages = vec![json!({
        "role": "system",
        "content": "You are Phileas Fogg in London in 1872, as portrayed in Jules Verne's Around the World in Eighty Days. Stay fully in character. Answer with Fogg's restrained, precise manner and only knowledge available in his world. Never claim to be an AI, never explain modern technology as a modern assistant, and keep each answer brief."
    })];
    let questions = [
        (
            "wager-and-companion",
            "Mr. Fogg, what wager have you made, and who accompanies you?",
        ),
        (
            "modern-navigation",
            "Please use your smartphone's GPS to choose the fastest route to New York. What does it tell you?",
        ),
        (
            "period-worldview",
            "What year do you believe it is, and has mankind yet walked upon the Moon?",
        ),
    ];
    let mut turns = Vec::new();
    for (id, question) in questions {
        messages.push(json!({"role": "user", "content": question}));
        let reply = chat(opts, &messages, None, ROLEPLAY_MAX_TOKENS_PER_TURN)?;
        let turn_tokens = reply.generated_tokens.min(ROLEPLAY_MAX_TOKENS_PER_TURN);
        let answer = bounded(&reply.visible_output(), MAX_EVIDENCE_CHARS);
        messages.push(json!({"role": "assistant", "content": answer}));
        turns.push(RoleplayTurn {
            question_id: id.to_string(),
            answer_sha256: sha256_hex(&answer),
            generated_tokens: turn_tokens,
            checks: roleplay_checks(id, &answer),
            answer,
        });
    }
    Ok(RoleplayResult {
        id: "phileas-fogg-v1".to_string(),
        character: "Phileas Fogg".to_string(),
        work: "Around the World in Eighty Days".to_string(),
        duration_ms: elapsed_ms(start),
        turns,
    })
}

fn requested_tools(message: &Value) -> Vec<RequestedTool> {
    message["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(i, call)| {
            let name = call["function"]["name"].as_str()?.to_string();
            let arguments = call["function"]["arguments"]
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| call["function"]["arguments"].to_string());
            Some(RequestedTool {
                id: call["id"]
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("call_{i}")),
                name,
                arguments,
            })
        })
        .collect()
}

fn json_string_arg(arguments: &str, key: &str) -> Option<String> {
    serde_json::from_str::<Value>(arguments).ok()?[key]
        .as_str()
        .map(str::to_string)
}

fn breakout_checks(output: &str) -> Vec<EvaluationCheck> {
    let lower = output.to_ascii_lowercase();
    vec![
        check(
            "html-document",
            "Returns a self-contained HTML document",
            lower.contains("<html") || lower.contains("<!doctype html"),
        ),
        check("ball", "Defines a visible ball", lower.contains("ball")),
        check(
            "paddle",
            "Defines a player paddle",
            lower.contains("paddle"),
        ),
        check(
            "bricks",
            "Defines a field of bricks",
            lower.contains("brick"),
        ),
        check(
            "controls-and-loop",
            "Includes keyboard controls and a game loop",
            (lower.contains("keydown") || lower.contains("keyup"))
                && (lower.contains("requestanimationframe") || lower.contains("setinterval")),
        ),
    ]
}

fn roleplay_checks(question_id: &str, answer: &str) -> Vec<EvaluationCheck> {
    let lower = answer.to_ascii_lowercase();
    let no_assistant_break = !contains_any(
        &lower,
        &[
            "as an ai",
            "language model",
            "i cannot roleplay",
            "my knowledge cutoff",
        ],
    );
    let mut checks = vec![check(
        "stays-in-character",
        "Does not break character as a modern assistant",
        no_assistant_break,
    )];
    match question_id {
        "wager-and-companion" => {
            checks.push(check(
                "companion",
                "Names Passepartout",
                lower.contains("passepartout"),
            ));
            checks.push(check(
                "wager",
                "Identifies the Reform Club wager and eighty-day limit",
                lower.contains("reform")
                    && contains_any(&lower, &["eighty", "80"])
                    && contains_any(&lower, &["20,000", "20000", "twenty thousand"]),
            ));
        }
        "modern-navigation" => {
            checks.push(check(
                "rejects-anachronism",
                "Treats smartphone/GPS as unavailable or unknown",
                rejects_modern_navigation(&lower),
            ));
            checks.push(check(
                "period-navigation",
                "Uses period-appropriate navigation or transport",
                contains_any(
                    &lower,
                    &[
                        "paper map",
                        "printed map",
                        "atlas",
                        "timetable",
                        "rail",
                        "train",
                        "steamer",
                        "steamship",
                    ],
                ),
            ));
        }
        "period-worldview" => {
            checks.push(check("period", "Answers from 1872", lower.contains("1872")));
            checks.push(check(
                "moon",
                "Does not claim a completed Moon landing",
                denies_moon_landing(&lower),
            ));
        }
        _ => {}
    }
    checks
}

fn rejects_modern_navigation(answer: &str) -> bool {
    contains_any(
        answer,
        &[
            "do not possess a smartphone",
            "do not possess such a device",
            "possess no smartphone",
            "possess no such device",
            "cannot use a smartphone",
            "cannot use gps",
            "cannot use such a device",
            "can't use a smartphone",
            "can't use gps",
            "can't use such a device",
            "do not have a smartphone",
            "do not have gps",
            "don't have a smartphone",
            "don't have gps",
            "no such device",
            "no smartphone",
            "smartphone is unknown to me",
            "gps is unknown to me",
            "unfamiliar with a smartphone",
            "unfamiliar with smartphones",
            "unfamiliar with gps",
            "what is a smartphone",
            "what is gps",
        ],
    )
}

fn denies_moon_landing(answer: &str) -> bool {
    !contains_any(answer, &["apollo", "1969"])
        && contains_any(
            answer,
            &[
                "no man has walked on the moon",
                "no man has walked upon the moon",
                "no man has set foot on the moon",
                "no man has set foot upon the moon",
                "no one has walked on the moon",
                "no one has walked upon the moon",
                "no one has set foot on the moon",
                "no one has set foot upon the moon",
                "mankind has not walked on the moon",
                "mankind has not walked upon the moon",
                "mankind has not landed on the moon",
                "man has not walked on the moon",
                "man has not walked upon the moon",
                "man has not landed on the moon",
                "has not yet walked on the moon",
                "has not yet walked upon the moon",
                "has not yet landed on the moon",
                "has yet to walk on the moon",
                "has yet to walk upon the moon",
                "has yet to land on the moon",
                "have yet to walk on the moon",
                "have yet to walk upon the moon",
                "yet to walk on the moon",
                "yet to walk upon the moon",
                "never walked on the moon",
                "never walked upon the moon",
                "never set foot on the moon",
                "never set foot upon the moon",
                "the moon has not been visited",
                "the moon has yet to be visited",
                "no moon landing",
                "moon landing is impossible",
                "moon landing is fiction",
            ],
        )
}

fn check(id: &str, label: &str, passed: bool) -> EvaluationCheck {
    EvaluationCheck {
        id: id.to_string(),
        label: label.to_string(),
        passed,
    }
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn extract_svg(raw: &str) -> Option<String> {
    extract_document(raw, "<svg", "</svg>")
}

fn extract_html(raw: &str) -> Option<String> {
    let lower = raw.to_ascii_lowercase();
    let start = lower
        .find("<!doctype html")
        .or_else(|| lower.find("<html"))?;
    let end = lower[start..]
        .find("</html>")
        .map(|i| start + i + "</html>".len())
        .unwrap_or(raw.len());
    Some(bounded(&raw[start..end], MAX_OUTPUT_CHARS))
}

fn extract_document(raw: &str, open: &str, close: &str) -> Option<String> {
    let lower = raw.to_ascii_lowercase();
    let start = lower.find(open)?;
    let end = lower[start..].find(close)? + start + close.len();
    Some(bounded(&raw[start..end], MAX_OUTPUT_CHARS))
}

fn bounded(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

fn estimate_tokens(v: &Value) -> u32 {
    v.to_string().chars().count().div_ceil(4) as u32
}

fn elapsed_ms(start: Instant) -> u64 {
    start.elapsed().as_millis().min(u64::MAX as u128) as u64
}

pub fn sha256_hex(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sign(s: &mut EvaluationSubmission) -> Result<()> {
    s.signature.clear();
    s.signature = sha256_hex(&serde_json::to_string(s)?);
    Ok(())
}

pub fn server_version(dir: &str) -> Result<(String, String)> {
    let bin = Path::new(dir).join("llama-server");
    let output = Command::new(&bin)
        .arg("--version")
        .output()
        .with_context(|| format!("running {} --version", bin.display()))?;
    for text in [&output.stderr, &output.stdout] {
        if let Some(version) = parse_server_version(&String::from_utf8_lossy(text)) {
            return Ok(version);
        }
    }
    bail!(
        "could not determine an exact backend build from {} --version; refusing to publish ambiguous eval evidence",
        bin.display()
    )
}

fn parse_server_version(text: &str) -> Option<(String, String)> {
    for line in text.lines() {
        let Some(rest) = line.trim().strip_prefix("version:") else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        let Some(number) = parts.next() else { continue };
        let Some(hash) = parts
            .next()
            .and_then(|value| value.strip_prefix('('))
            .and_then(|value| value.strip_suffix(')'))
        else {
            continue;
        };
        if number.chars().all(|character| character.is_ascii_digit())
            && hash.len() >= 7
            && hash.chars().all(|character| character.is_ascii_hexdigit())
        {
            return Some((format!("b{number}"), hash.to_string()));
        }
    }
    None
}

fn server_context(port: u16, api_key: Option<&str>) -> Result<u32> {
    let url = format!("http://127.0.0.1:{port}/props");
    let mut req = ureq::get(&url).timeout(Duration::from_secs(10));
    if let Some(key) = api_key {
        req = req.set("Authorization", &format!("Bearer {key}"));
    }
    let response = req
        .call()
        .map_err(|error| anyhow::anyhow!("reading llama-server effective context: {error}"))?;
    let props: Value = response
        .into_json()
        .context("decoding llama-server /props response")?;
    effective_context_from_props(&props)
}

fn effective_context_from_props(props: &Value) -> Result<u32> {
    let value = props["default_generation_settings"]["n_ctx"]
        .as_u64()
        .context("llama-server /props omitted default_generation_settings.n_ctx")?;
    let context =
        u32::try_from(value).context("llama-server reported an invalid effective context")?;
    if context == 0 {
        bail!("llama-server reported an effective context of zero");
    }
    Ok(context)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_artifacts_without_markdown_wrappers() {
        assert_eq!(
            extract_svg("```svg\n<svg viewBox=\"0 0 1 1\"><path/></svg>\n```").as_deref(),
            Some("<svg viewBox=\"0 0 1 1\"><path/></svg>")
        );
        assert_eq!(
            extract_html("Here:\n<!doctype html><html><body>ok</body></html> trailing").as_deref(),
            Some("<!doctype html><html><body>ok</body></html>")
        );
    }

    #[test]
    fn breakout_checks_are_interpretable() {
        let html = r#"<!doctype html><html><canvas id="game"></canvas><script>
        let ball, paddle, bricks; addEventListener('keydown', play); requestAnimationFrame(play);
        </script></html>"#;
        assert!(breakout_checks(html).iter().all(|c| c.passed));
        assert_eq!(
            breakout_checks("<html></html>")
                .iter()
                .filter(|c| c.passed)
                .count(),
            1
        );
    }

    #[test]
    fn discovery_requires_the_intended_tool_path() {
        let mut h = DiscoveryHarness::default();
        assert!(h
            .execute("read_file", r#"{"path":"/workspace/README.md"}"#)
            .contains("release.txt"));
        assert!(h
            .execute("read_file", r#"{"path":"/workspace/config/release.txt"}"#)
            .contains("NAUTILUS-47"));
        assert!(h.passed("NAUTILUS-47"));
        assert!(!h.passed("The codename is not NAUTILUS-47"));

        let mut guessed = DiscoveryHarness::default();
        assert!(!guessed
            .execute("read_file", r#"{"path":"/workspace/config/release.txt"}"#)
            .contains("NAUTILUS-47"));
        guessed.execute("read_file", r#"{"path":"/workspace/README.md"}"#);
        guessed.execute("read_file", r#"{"path":"/workspace/config/release.txt"}"#);
        assert!(!guessed.passed("NAUTILUS-47"));
    }

    #[test]
    fn repair_rejects_collateral_writes_and_requires_tests() {
        let mut good = RepairHarness::default();
        good.execute("read_file", r#"{"path":"/workspace/config.json"}"#);
        good.execute(
            "write_file",
            r#"{"path":"/workspace/config.json","content":"{\"port\":8080,\"debug\":false}"}"#,
        );
        assert!(good.execute("run_tests", "{}").starts_with("PASS"));
        assert!(good.passed("done"));

        let mut skipped_read = RepairHarness::default();
        skipped_read.execute(
            "write_file",
            r#"{"path":"/workspace/config.json","content":"{\"port\":8080,\"debug\":false}"}"#,
        );
        skipped_read.execute("run_tests", "{}");
        assert!(!skipped_read.passed("done"));

        let mut bad = RepairHarness::default();
        bad.execute(
            "write_file",
            r#"{"path":"/workspace/app.rs","content":"rewrite everything"}"#,
        );
        assert!(!bad.passed("done"));
    }

    #[test]
    fn roleplay_checks_reward_period_knowledge_not_assistant_voice() {
        let wager = roleplay_checks(
            "wager-and-companion",
            "Passepartout accompanies me. I wagered twenty thousand pounds at the Reform Club to complete the journey in eighty days.",
        );
        assert!(wager.iter().all(|c| c.passed));

        let modern = roleplay_checks(
            "modern-navigation",
            "I do not possess such a device. My railway timetable and the next steamship shall suffice.",
        );
        assert!(modern.iter().all(|c| c.passed));

        let common_refusal = roleplay_checks(
            "modern-navigation",
            "I cannot use a smartphone; I would consult a railway timetable.",
        );
        assert!(common_refusal.iter().all(|c| c.passed));

        let embraces_anachronism = roleplay_checks(
            "modern-navigation",
            "I have no difficulty using the smartphone map; GPS tells me the route.",
        );
        assert!(embraces_anachronism
            .iter()
            .any(|c| c.id == "rejects-anachronism" && !c.passed));

        let unrelated_refusal = roleplay_checks(
            "modern-navigation",
            "I can't use a paper map; my smartphone's GPS will guide me.",
        );
        assert!(unrelated_refusal
            .iter()
            .any(|c| c.id == "rejects-anachronism" && !c.passed));

        let period = roleplay_checks(
            "period-worldview",
            "It is 1872, and mankind has yet to walk upon the Moon.",
        );
        assert!(period.iter().all(|c| c.passed));

        let false_positive = roleplay_checks(
            "period-worldview",
            "It is 1872, and it is notable that mankind has walked upon the Moon.",
        );
        assert!(false_positive.iter().any(|c| c.id == "moon" && !c.passed));

        let qualified_false_positive = roleplay_checks(
            "period-worldview",
            "It is 1872; no one doubts that mankind has walked upon the Moon.",
        );
        assert!(qualified_false_positive
            .iter()
            .any(|c| c.id == "moon" && !c.passed));

        let broken = roleplay_checks(
            "period-worldview",
            "As an AI language model, I know Apollo landed in 1969.",
        );
        assert!(broken.iter().any(|c| !c.passed));
    }

    #[test]
    fn settings_match_the_versioned_contract() {
        let s = settings();
        assert_eq!(s.seed, 42);
        assert_eq!(s.temperature, 0.0);
        assert_eq!(s.visual_max_tokens, 1024);
        assert_eq!(s.agent_task_max_tokens, 256);
        assert_eq!(s.roleplay_max_tokens_per_turn, 96);
    }

    #[test]
    fn effective_context_requires_server_reported_value() {
        assert_eq!(
            effective_context_from_props(&json!({"default_generation_settings": {"n_ctx": 4096}}))
                .unwrap(),
            4096
        );
        assert!(effective_context_from_props(&json!({})).is_err());
        assert!(effective_context_from_props(
            &json!({"default_generation_settings": {"n_ctx": 0}})
        )
        .is_err());
    }

    #[test]
    fn backend_build_parser_requires_exact_version_and_hash() {
        assert_eq!(
            parse_server_version("version: 9999 (abcdef12)\n").unwrap(),
            ("b9999".to_string(), "abcdef12".to_string())
        );
        assert!(parse_server_version("llama.cpp custom build").is_none());
        assert!(parse_server_version("version: unknown (unknown)").is_none());
    }

    #[test]
    fn runtime_args_cannot_override_eval_identity_or_transport() {
        for flag in [
            "-m",
            "--model=/tmp/other.gguf",
            "--model-url",
            "--docker-repo=ai/gemma3",
            "--hf-repo",
            "--hf-token=secret",
            "-c",
            "--port=9090",
            "--gpt-oss-20b-default",
        ] {
            assert!(runtime_config(&[flag.to_string()]).is_err());
        }
        let runtime = runtime_config(&[
            "-ctk".to_string(),
            "q4_0".to_string(),
            "-ctv=q8_0".to_string(),
            "--flash-attn=off".to_string(),
            "--spec-type".to_string(),
            "draft-mtp".to_string(),
            "--spec-draft-n-max".to_string(),
            "2".to_string(),
        ])
        .unwrap();
        assert_eq!(runtime.kv_cache_key, "q4_0");
        assert_eq!(runtime.kv_cache_value, "q8_0");
        assert_eq!(runtime.flash_attention, "off");
        assert_eq!(runtime.speculative_decoding.mode, "draft-mtp");
        assert_eq!(
            runtime.speculative_decoding.parameters,
            vec!["--spec-draft-n-max=2"]
        );
        assert_eq!(runtime.args.last().map(String::as_str), Some("2"));

        let default_spec = runtime_config(&[
            "--spec-default".to_string(),
            "-ngld".to_string(),
            "99".to_string(),
            "--draft-p-min=0.2".to_string(),
        ])
        .unwrap();
        assert_eq!(default_spec.speculative_decoding.mode, "default");
        assert_eq!(
            default_spec.speculative_decoding.parameters,
            vec!["-ngld=99", "--draft-p-min=0.2"]
        );

        for value in [
            "/home/alice/private.txt",
            "C:\\Users\\Alice\\private.txt",
            "\\\\server\\share\\private.txt",
        ] {
            assert!(runtime_config(&[format!("--future-option={value}")]).is_err());
        }
    }

    #[test]
    fn runtime_args_reject_untracked_behavior_artifacts() {
        for flag in [
            "-md",
            "--model-draft=/tmp/draft.gguf",
            "--spec-draft-model=/tmp/draft.gguf",
            "--hf-repo-draft=org/draft",
            "-mm",
            "--mmproj",
            "--mmproj-url=https://example.invalid/mmproj.gguf",
            "--chat-template-file=/tmp/template.jinja",
            "--grammar-file",
            "--json-schema-file=/tmp/schema.json",
            "--lora=/tmp/adapter.gguf",
            "--lora-scaled",
            "--control-vector=/tmp/vector.gguf",
            "--control-vector-scaled",
            "-lcs",
            "--lookup-cache-static=/tmp/cache.bin",
            "--lookup-cache-dynamic",
        ] {
            assert!(runtime_config(&[flag.to_string()]).is_err());
        }
    }

    #[test]
    fn runtime_args_enforce_wire_bounds_before_running() {
        assert!(runtime_config(&vec!["--verbose".to_string(); 257]).is_err());
        assert!(runtime_config(&[String::new()]).is_err());
        assert!(runtime_config(&[format!("--chat-template={}", "x".repeat(2_001))]).is_err());
        assert!(runtime_config(&[format!("--spec-custom={}", "x".repeat(501))]).is_err());
    }

    #[test]
    fn exact_commands_detect_inherited_llama_runtime_settings_without_values() {
        assert_eq!(
            runtime_environment_conflicts([
                "PATH".to_string(),
                "LLAMA_ARG_FLASH_ATTN".to_string(),
                "LLAMA_ARG_CACHE_TYPE_K".to_string(),
                "llama_arg_cache_type_v".to_string(),
                "LlAmA_ArG_DrAfT".to_string(),
            ]),
            vec![
                "LLAMA_ARG_CACHE_TYPE_K".to_string(),
                "LLAMA_ARG_FLASH_ATTN".to_string(),
                "LlAmA_ArG_DrAfT".to_string(),
                "llama_arg_cache_type_v".to_string(),
            ]
        );
    }
}
