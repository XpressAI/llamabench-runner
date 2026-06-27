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
    pub extra_server_args: &'a [String],
}

/// Kills the spawned server on drop so we never leak a process.
struct ServerGuard(Child);
impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

pub fn run_verification(opts: &VerifyOpts) -> Result<Verification> {
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
    .args(opts.extra_server_args)
    .stdout(Stdio::null())
    .stderr(Stdio::null());
    let child = cmd
        .spawn()
        .with_context(|| format!("spawning {}", bin.display()))?;
    let _guard = ServerGuard(child);

    wait_until_ready(opts.port, opts.api_key, Duration::from_secs(240))
        .context("llama-server did not become ready")?;

    let mut runs = Vec::new();
    for script in PROMPTS {
        let max = opts.max_turns.min(script.turns.len() as u32);
        for turns in 1..=max {
            for rep in 1..=opts.reps {
                // A server error/crash on a turn (e.g. the engine rejecting garbled
                // output) is itself an invalidity signal — record it as a failed run
                // rather than aborting the whole verification.
                let (output, failed) = match run_conversation(opts, script, turns) {
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
            }
        }
    }

    let valid = !runs.iter().any(|r| r.gibberish);
    Ok(Verification {
        seed: opts.seed,
        temperature: 0.0,
        n_gen: opts.n_gen,
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
fn run_conversation(opts: &VerifyOpts, script: &PromptScript, turns: u32) -> Result<String> {
    let mut messages: Vec<Value> = Vec::new();
    let mut final_output = String::new();
    for i in 0..turns as usize {
        messages.push(json!({"role": "user", "content": script.turns[i]}));
        let reply = chat(opts, &messages)?;
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
fn chat(opts: &VerifyOpts, messages: &[Value]) -> Result<Reply> {
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", opts.port);
    let body = json!({
        "messages": messages,
        "seed": opts.seed,
        "temperature": 0.0,
        "max_tokens": opts.n_gen,
        "stream": false,
    });
    let resp = ureq::post(&url)
        .set("Authorization", &format!("Bearer {}", opts.api_key))
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

fn wait_until_ready(port: u16, api_key: &str, timeout: Duration) -> Result<()> {
    let url = format!("http://127.0.0.1:{port}/health");
    let start = Instant::now();
    while start.elapsed() < timeout {
        let ok = ureq::get(&url)
            .set("Authorization", &format!("Bearer {api_key}"))
            .timeout(Duration::from_secs(5))
            .call()
            .map(|r| r.status() == 200)
            .unwrap_or(false);
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
    fn flags_obvious_gibberish() {
        assert!(is_gibberish(""));
        assert!(is_gibberish("   "));
        assert!(is_gibberish(&"the ".repeat(40)));
        assert!(is_gibberish(&"\u{FFFD}".repeat(50)));
    }
}
