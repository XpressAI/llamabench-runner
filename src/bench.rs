// SPDX-License-Identifier: GPL-3.0-or-later
//! Drives `llama-bench` for the standardized speed numbers (prefill pp / decode tg)
//! and captures the exact llama.cpp build (git hash + build number).

use anyhow::{bail, Context, Result};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;

pub struct BenchOpts<'a> {
    pub llama_bin_dir: &'a str,
    pub model: &'a str,
    pub ngl: i32,
    pub fa: &'a str, // on | off | auto
    pub ctk: &'a str,
    pub ctv: &'a str,
    pub n_prompt: u32,
    pub n_gen: u32,
}

#[derive(Debug, Default)]
pub struct BenchResult {
    pub model_label: String,
    pub params_b: f64,
    pub backend_label: String,
    pub type_k: String,
    pub type_v: String,
    pub flash_attn: bool,
    pub prefill_tps: f64,
    pub decode_tps: f64,
    pub build_number: String, // "b9660"
    pub git_hash: String,     // "7dad2f1a1"
    pub devices: Vec<String>,
}

pub fn run_llama_bench(opts: &BenchOpts) -> Result<BenchResult> {
    let bin = Path::new(opts.llama_bin_dir).join("llama-bench");
    let ngl = opts.ngl.to_string();
    let np = opts.n_prompt.to_string();
    let ng = opts.n_gen.to_string();

    let mut child = Command::new(&bin)
        .args([
            "-m", opts.model, "-ngl", &ngl, "-fa", opts.fa, "-ctk", opts.ctk, "-ctv", opts.ctv,
            "-p", &np, "-n", &ng,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("running {}", bin.display()))?;

    // llama-bench prints progress (device init, model load, per-run timing) to stderr and
    // the results table to stdout. Stream stderr live so the user sees activity during the
    // (often slow) run, while we capture both for parsing.
    let child_stderr = child.stderr.take().expect("piped stderr");
    let stderr_thread = thread::spawn(move || {
        let mut captured = String::new();
        for line in BufReader::new(child_stderr).lines().map_while(Result::ok) {
            eprintln!("    {line}");
            captured.push_str(&line);
            captured.push('\n');
        }
        captured
    });

    let mut stdout = String::new();
    {
        let child_stdout = child.stdout.take().expect("piped stdout");
        for line in BufReader::new(child_stdout).lines().map_while(Result::ok) {
            stdout.push_str(&line);
            stdout.push('\n');
        }
    }
    let status = child.wait().context("waiting for llama-bench")?;
    let stderr = stderr_thread.join().unwrap_or_default();

    let mut parsed = parse(&stdout, &stderr);
    // llama-bench omits the type_k/type_v columns when they're the default (f16), so fill
    // them from what we requested — the recorded KV-cache config stays accurate.
    if parsed.type_k.is_empty() {
        parsed.type_k = opts.ctk.to_string();
    }
    if parsed.type_v.is_empty() {
        parsed.type_v = opts.ctv.to_string();
    }

    if parsed.prefill_tps == 0.0 && parsed.decode_tps == 0.0 {
        bail!(
            "could not parse llama-bench output (exit {:?}).\nstdout:\n{}\nstderr:\n{}",
            status.code(),
            stdout,
            stderr
        );
    }
    Ok(parsed)
}

/// Parse the markdown results table (stdout) and the device/build banners.
///
/// llama-bench's columns vary by backend and options — e.g. CUDA shows `ngl`, CPU shows
/// `threads`, and `type_k`/`type_v` only appear when the KV cache is non-default. So we
/// map columns by their HEADER NAME rather than a fixed position.
pub fn parse(stdout: &str, stderr: &str) -> BenchResult {
    let mut r = BenchResult::default();
    let mut headers: Option<Vec<String>> = None;

    for line in stdout.lines() {
        let t = line.trim();
        if !t.starts_with('|') {
            if let Some(b) = parse_build(t) {
                r.build_number = b.0;
                r.git_hash = b.1;
            }
            continue;
        }
        let cells: Vec<&str> = t
            .split('|')
            .map(str::trim)
            .filter(|c| !c.is_empty())
            .collect();
        if cells.is_empty() {
            continue;
        }
        // separator row: every cell is just dashes/colons
        if cells
            .iter()
            .all(|c| c.chars().all(|ch| ch == '-' || ch == ':'))
        {
            continue;
        }
        // header row defines the column layout
        if cells[0] == "model" {
            headers = Some(cells.iter().map(|s| s.to_string()).collect());
            continue;
        }
        let Some(hdr) = headers.as_ref() else {
            continue;
        };
        let col = |name: &str| -> Option<&str> {
            hdr.iter()
                .position(|h| h.as_str() == name)
                .and_then(|i| cells.get(i).copied())
        };
        let num = |s: Option<&str>| -> f64 {
            s.and_then(|v| v.split_whitespace().next())
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0)
        };

        let test = col("test").unwrap_or("");
        let tps = num(col("t/s"));
        if test.starts_with("pp") {
            r.prefill_tps = tps;
        } else if test.starts_with("tg") {
            r.decode_tps = tps;
        }

        if let Some(m) = col("model") {
            r.model_label = m.to_string();
        }
        r.params_b = num(col("params"));
        if let Some(b) = col("backend") {
            r.backend_label = b.to_string();
        }
        if let Some(k) = col("type_k") {
            r.type_k = k.to_string();
        }
        if let Some(v) = col("type_v") {
            r.type_v = v.to_string();
        }
        if let Some(f) = col("fa") {
            r.flash_attn = f == "1";
        }
    }

    for line in stderr.lines().chain(stdout.lines()) {
        if let Some(dev) = parse_device(line) {
            if !r.devices.contains(&dev) {
                r.devices.push(dev);
            }
        }
        if r.build_number.is_empty() {
            if let Some(b) = parse_build(line.trim()) {
                r.build_number = b.0;
                r.git_hash = b.1;
            }
        }
    }
    r
}

/// "build: 7dad2f1a1 (9660)" -> ("b9660", "7dad2f1a1")
fn parse_build(line: &str) -> Option<(String, String)> {
    let rest = line.strip_prefix("build:")?.trim();
    let hash = rest.split_whitespace().next()?.to_string();
    let num = rest
        .split('(')
        .nth(1)?
        .split(')')
        .next()?
        .trim()
        .to_string();
    Some((format!("b{num}"), hash))
}

/// "ggml_vulkan: 0 = AMD Radeon Pro 5500M (MoltenVK) | uma: 0 ..." -> the device name.
fn parse_device(line: &str) -> Option<String> {
    let l = line.trim();
    if !(l.starts_with("ggml_") && l.contains(" = ")) {
        return None;
    }
    let after = l.split(" = ").nth(1)?;
    let name = after.split(" | ").next()?.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Non-default KV cache (q4_0) → llama-bench includes type_k/type_v columns (10 cols).
    const STDOUT_Q4: &str = "\
| model                    |   size |  params | backend     | threads | type_k | type_v |  fa |  test |              t/s |
| ------------------------ | -----: | ------: | ----------- | ------: | -----: | -----: | --: | ----: | ---------------: |
| qwen35 4B Q4_K - Medium  | 2.70 GiB | 4.21 B | Vulkan,BLAS |       8 |   q4_0 |   q4_0 |   1 | pp512 |     33.10 ± 0.29 |
| qwen35 4B Q4_K - Medium  | 2.70 GiB | 4.21 B | Vulkan,BLAS |       8 |   q4_0 |   q4_0 |   1 | tg128 |     23.65 ± 1.96 |

build: 7dad2f1a1 (9660)";
    const STDERR: &str = "ggml_vulkan: 0 = AMD Radeon Pro 5500M (MoltenVK) | uma: 0 | fp16: 1";

    // Default KV cache (f16) → llama-bench OMITS type_k/type_v (8 cols). This is the layout
    // that broke the old fixed-index parser (the user's real gemma run).
    const STDOUT_F16: &str = "\
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.85 GiB |    11.91 B | Vulkan,BLAS |       8 |   1 |           pp512 |          8.31 ± 1.30 |
| gemma4 ?B Q4_K - Medium        |   6.85 GiB |    11.91 B | Vulkan,BLAS |       8 |   1 |           tg128 |          4.78 ± 0.14 |

build: 7dad2f1a1 (9660)";

    #[test]
    fn parses_q4_cache_layout() {
        let r = parse(STDOUT_Q4, STDERR);
        assert!((r.prefill_tps - 33.10).abs() < 0.01);
        assert!((r.decode_tps - 23.65).abs() < 0.01);
        assert!((r.params_b - 4.21).abs() < 0.01);
        assert_eq!(r.type_k, "q4_0");
        assert_eq!(r.type_v, "q4_0");
        assert!(r.flash_attn);
        assert_eq!(r.build_number, "b9660");
        assert_eq!(r.git_hash, "7dad2f1a1");
        assert_eq!(r.backend_label, "Vulkan,BLAS");
        assert_eq!(
            r.devices,
            vec!["AMD Radeon Pro 5500M (MoltenVK)".to_string()]
        );
    }

    #[test]
    fn parses_default_f16_layout() {
        // The 8-column layout (no type_k/type_v) that the old parser dropped.
        let r = parse(STDOUT_F16, STDERR);
        assert!((r.prefill_tps - 8.31).abs() < 0.01);
        assert!((r.decode_tps - 4.78).abs() < 0.01);
        assert!((r.params_b - 11.91).abs() < 0.01);
        assert!(r.flash_attn);
        assert_eq!(r.backend_label, "Vulkan,BLAS");
        // type_k/type_v are absent from the table here; run_llama_bench fills them from opts.
        assert_eq!(r.type_k, "");
    }
}
