// SPDX-License-Identifier: GPL-3.0-or-later
//! Output-correctness verification (ADR-005). Starts the user's `llama-server`, then
//! runs a few fixed prompts at seed + temperature 0, each at 1/2/3 conversational
//! turns and repeated, capturing each output's sha256 + a preview and a gibberish
//! verdict. Multi-turn catches KV-cache bugs that only corrupt later turns. Any
//! gibberish ⇒ the submission is invalid (small run-to-run deviations are fine).

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::{IsTerminal, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

use crate::contract::{Verification, VerificationRun};

/// A conversation script: successive user turns. We generate up to `turns` of them,
/// feeding each assistant reply back in, so a 2/3-turn bug surfaces.
struct PromptScript {
    id: &'static str,
    turns: &'static [&'static str],
}

const PROMPTS: &[PromptScript] = &[
    PromptScript {
        id: "meaning",
        turns: &[
            "What is the meaning of life? Answer in a short paragraph.",
            "Summarize that in a single sentence.",
            "Now express it in exactly three words.",
        ],
    },
    PromptScript {
        id: "count",
        turns: &[
            "List the integers from 1 to 10, comma separated.",
            "Now list them in reverse order.",
            "What is their sum?",
        ],
    },
];

pub struct VerifyOpts<'a> {
    pub server_bin_dir: &'a str,
    pub model: &'a str,
    pub port: u16,
    pub api_key: &'a str,
    pub seed: u64,
    pub n_gen: u32,
    pub max_turns: u32,
    pub reps: u32,
    pub extra_server_args: Vec<String>,
}

/// A llama-server to verify against — either one we spawned (`run_verification`) or
/// the user's own drop-in `llamabench llama-server` process (ADR-009). `api_key` is
/// only sent when the server actually requires one.
pub struct VerifySession<'a> {
    pub port: u16,
    pub api_key: Option<&'a str>,
    pub seed: u64,
    pub n_gen: u32,
    pub max_turns: u32,
    pub reps: u32,
}

/// Kills the spawned server on drop so we never leak a process.
struct ServerGuard(Child);
impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn session_of<'a>(opts: &'a VerifyOpts) -> VerifySession<'a> {
    VerifySession {
        port: opts.port,
        api_key: Some(opts.api_key),
        seed: opts.seed,
        n_gen: opts.n_gen,
        max_turns: opts.max_turns,
        reps: opts.reps,
    }
}

fn spawn_server(opts: &VerifyOpts) -> Result<ServerGuard> {
    let bin = Path::new(opts.server_bin_dir).join("llama-server");
    let port = opts.port.to_string();
    let mut cmd = Command::new(&bin);
    cmd.args([
        "-m",
        opts.model,
        "--port",
        &port,
        "--api-key",
        opts.api_key,
        "--jinja",
    ])
    .args(&opts.extra_server_args)
    .stdout(Stdio::null())
    .stderr(Stdio::null());
    let child = cmd
        .spawn()
        .with_context(|| format!("spawning {}", bin.display()))?;
    let guard = ServerGuard(child);
    wait_until_ready(opts.port, Some(opts.api_key), Duration::from_secs(240))
        .context("llama-server did not become ready")?;
    Ok(guard)
}

pub fn run_verification(opts: &VerifyOpts) -> Result<Verification> {
    let _guard = spawn_server(opts)?;
    verify_running(&session_of(opts))
}

/// Verification plus a TTFT reading from the same server (the fixed-prompt pass is
/// already paying for a running llama-server, so the standardized probe rides
/// along). Returns `None` TTFT when the probe fails — never fails the run over it.
pub fn run_verification_with_ttft(opts: &VerifyOpts) -> Result<(Verification, Option<u32>)> {
    let _guard = spawn_server(opts)?;
    let ttft = ttft_probe(opts.port, Some(opts.api_key));
    if let Some(ms) = ttft {
        eprintln!("  ttft {ms} ms (standardized ~512-token prompt via llama-server)");
    }
    let v = verify_running(&session_of(opts))?;
    Ok((v, ttft))
}

/// The verification matrix against an already-running server.
pub fn verify_running(s: &VerifySession) -> Result<Verification> {
    let total: u32 = PROMPTS
        .iter()
        .map(|script| s.max_turns.min(script.turns.len() as u32) * s.reps)
        .sum();
    let tty = std::io::stderr().is_terminal();
    let mut done = 0u32;

    let mut runs = Vec::new();
    for script in PROMPTS {
        let max = s.max_turns.min(script.turns.len() as u32);
        for turns in 1..=max {
            for rep in 1..=s.reps {
                if tty {
                    eprint!(
                        "\r  generating {}/{}  ({}, {} turn{})…   ",
                        done + 1,
                        total,
                        script.id,
                        turns,
                        if turns == 1 { "" } else { "s" }
                    );
                    let _ = std::io::stderr().flush();
                }
                // A server error/crash on a turn (e.g. the engine rejecting garbled
                // output) is itself an invalidity signal — record it as a failed run
                // rather than aborting the whole verification.
                let (output, failed) = match run_conversation(s, script, turns) {
                    Ok(o) => (o, false),
                    Err(e) => (format!("<engine error: {e}>"), true),
                };
                let gibberish = failed || is_gibberish(&output);
                runs.push(VerificationRun {
                    prompt_id: script.id.to_string(),
                    turns,
                    rep,
                    output_sha256: sha256_hex(&output),
                    output_preview: preview(&output),
                    gibberish,
                });
                done += 1;
            }
        }
    }
    if tty {
        eprintln!("\r  generated {done}/{total} verification runs            ");
    }

    let valid = !runs.iter().any(|r| r.gibberish);
    Ok(Verification {
        seed: s.seed,
        temperature: 0.0,
        n_gen: s.n_gen,
        valid,
        runs,
    })
}

struct Reply {
    content: String,
    reasoning: String,
}

/// Generate a `turns`-deep conversation, returning the final turn's full output
/// (reasoning trace + answer) for hashing/gibberish-checking.
fn run_conversation(s: &VerifySession, script: &PromptScript, turns: u32) -> Result<String> {
    let mut messages: Vec<Value> = Vec::new();
    let mut final_output = String::new();
    for i in 0..turns as usize {
        messages.push(json!({"role": "user", "content": script.turns[i]}));
        let reply = chat(s, &messages)?;
        // Conversation history carries the answer (or the reasoning if the answer is
        // empty, e.g. budget-truncated) so later turns have context.
        let history = if reply.content.is_empty() {
            &reply.reasoning
        } else {
            &reply.content
        };
        messages.push(json!({"role": "assistant", "content": history}));
        final_output = if reply.reasoning.is_empty() {
            reply.content.clone()
        } else {
            format!("{}\n{}", reply.reasoning, reply.content)
        };
    }
    Ok(final_output)
}

/// One chat completion. Reasoning models split output into `reasoning_content` +
/// `content`; we capture both.
fn chat(s: &VerifySession, messages: &[Value]) -> Result<Reply> {
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", s.port);
    let body = json!({
        "messages": messages,
        "seed": s.seed,
        "temperature": 0.0,
        "max_tokens": s.n_gen,
        "stream": false,
    });
    let mut req = ureq::post(&url);
    if let Some(key) = s.api_key {
        req = req.set("Authorization", &format!("Bearer {key}"));
    }
    let resp = req
        .send_json(body)
        .map_err(|e| anyhow::anyhow!("chat request failed: {e}"))?;
    let v: Value = resp.into_json().context("decoding chat response")?;
    let msg = &v["choices"][0]["message"];
    Ok(Reply {
        content: msg["content"].as_str().unwrap_or_default().to_string(),
        reasoning: msg["reasoning_content"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
    })
}

/// The standardized speed/TTFT prompt: fixed text, ~512 tokens once repeated, so
/// every rig measures the same work. Used by the llama-server drop-in's speed pass
/// and the TTFT probe that rides on the verification server.
pub fn standard_prompt() -> String {
    "The library at the edge of town kept its lights on long after closing, \
     because the archivist believed that somewhere between the shelves a reader was \
     always lost, tracing the margins of a book that had waited decades to be opened, \
     and she considered it her duty to keep the lamps burning until every last one of \
     them found the door. "
        .repeat(8)
}

/// The server's own timings for one `/completion` call.
pub struct Timing {
    pub prefill: f64,
    pub decode: f64,
    pub prompt_ms: f64,
}

pub fn http_completion(
    port: u16,
    api_key: Option<&str>,
    prompt: &str,
    n_predict: u32,
) -> Result<Timing> {
    let url = format!("http://127.0.0.1:{port}/completion");
    let body = json!({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": false,
    });
    let mut req = ureq::post(&url).timeout(Duration::from_secs(600));
    if let Some(key) = api_key {
        req = req.set("Authorization", &format!("Bearer {key}"));
    }
    let resp = req
        .send_json(body)
        .map_err(|e| anyhow::anyhow!("completion request failed: {e}"))?;
    let v: Value = resp.into_json().context("decoding completion response")?;
    let t = &v["timings"];
    let timing = Timing {
        prefill: t["prompt_per_second"].as_f64().unwrap_or(0.0),
        decode: t["predicted_per_second"].as_f64().unwrap_or(0.0),
        prompt_ms: t["prompt_ms"].as_f64().unwrap_or(0.0),
    };
    if timing.decode == 0.0 {
        bail!(
            "the server response carried no timings — can't measure speed (keys: {})",
            v.as_object()
                .map(|o| o.keys().cloned().collect::<Vec<_>>().join(", "))
                .unwrap_or_default()
        );
    }
    Ok(timing)
}

/// The median of `prompt_ms` samples, rounded to whole ms. `None` when empty.
pub fn median_ms(mut samples: Vec<f64>) -> Option<u32> {
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(f64::total_cmp);
    Some(samples[samples.len() / 2].round() as u32)
}

/// TTFT against a running server: one warmup, then the median `prompt_ms` of 3
/// standardized-prompt completions (`n_predict` is tiny — only the prefill matters).
/// Best-effort: any failure yields `None`, never a failed run.
pub fn ttft_probe(port: u16, api_key: Option<&str>) -> Option<u32> {
    let prompt = standard_prompt();
    http_completion(port, api_key, &prompt, 8).ok()?;
    median_ms(
        (0..3)
            .filter_map(|_| http_completion(port, api_key, &prompt, 8).ok())
            .map(|t| t.prompt_ms)
            .collect(),
    )
}

pub fn wait_until_ready(port: u16, api_key: Option<&str>, timeout: Duration) -> Result<()> {
    let url = format!("http://127.0.0.1:{port}/health");
    let start = Instant::now();
    while start.elapsed() < timeout {
        let mut req = ureq::get(&url).timeout(Duration::from_secs(5));
        if let Some(key) = api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let ok = req.call().map(|r| r.status() == 200).unwrap_or(false);
        if ok {
            return Ok(());
        }
        sleep(Duration::from_secs(2));
    }
    bail!("timed out after {:?}", timeout)
}

fn sha256_hex(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn preview(s: &str) -> String {
    let t = s.trim();
    t.chars().take(200).collect()
}

/// Heuristic gibberish gate. Conservative — catches the obvious failures (empty,
/// control-char soup, tight loops, near-zero vocabulary). The authoritative judge is
/// server-side (ADR-005); this stops a clearly-broken fast run at the source.
pub fn is_gibberish(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return true;
    }
    let total = t.chars().count();
    let bad = t
        .chars()
        .filter(|c| *c == '\u{FFFD}' || (c.is_control() && !matches!(c, '\n' | '\t' | '\r')))
        .count();
    if bad * 20 > total {
        return true; // >5% control/replacement chars
    }
    let words: Vec<&str> = t.split_whitespace().collect();
    if words.len() >= 20 {
        let uniq: HashSet<&str> = words.iter().copied().collect();
        if (uniq.len() as f64) / (words.len() as f64) < 0.18 {
            return true; // looping / tiny vocabulary
        }
        let mut run = 1usize;
        let mut max_run = 1usize;
        for i in 1..words.len() {
            if words[i] == words[i - 1] {
                run += 1;
                max_run = max_run.max(run);
            } else {
                run = 1;
            }
        }
        if max_run >= 12 {
            return true; // same token repeated many times
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_coherent_text() {
        assert!(!is_gibberish(
            "The meaning of life is a deeply personal question. Many find meaning in \
             relationships, growth, contribution, and the pursuit of understanding."
        ));
        assert!(!is_gibberish("1, 2, 3, 4, 5, 6, 7, 8, 9, 10"));
    }

    #[test]
    fn median_of_prompt_ms_samples() {
        assert_eq!(median_ms(vec![]), None);
        assert_eq!(median_ms(vec![2326.4]), Some(2326));
        // Median (not mean) so one slow outlier can't skew the recorded TTFT.
        assert_eq!(median_ms(vec![2654.0, 2326.0, 9999.0]), Some(2654));
    }

    #[test]
    fn flags_obvious_gibberish() {
        assert!(is_gibberish(""));
        assert!(is_gibberish("   "));
        assert!(is_gibberish(&"the ".repeat(40)));
        assert!(is_gibberish(&"\u{FFFD}".repeat(50)));
    }
}
