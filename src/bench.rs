// SPDX-License-Identifier: GPL-3.0-or-later
//! Drives `llama-bench` for the standardized speed numbers (prefill pp / decode tg)
//! and captures the exact llama.cpp build (git hash + build number).

use anyhow::{bail, Context, Result};
use std::path::Path;
use std::process::Command;

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
    let out = Command::new(&bin)
        .args([
            "-m", opts.model, "-ngl", &ngl, "-fa", opts.fa, "-ctk", opts.ctk, "-ctv", opts.ctv,
            "-p", &np, "-n", &ng,
        ])
        .output()
        .with_context(|| format!("running {}", bin.display()))?;

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let parsed = parse(&stdout, &stderr);
    if parsed.prefill_tps == 0.0 && parsed.decode_tps == 0.0 {
        bail!(
            "could not parse llama-bench output (exit {:?}).\nstdout:\n{}\nstderr:\n{}",
            out.status.code(),
            stdout,
            stderr
        );
    }
    Ok(parsed)
}

/// Parse the markdown results table (stdout) and the device/build banners.
pub fn parse(stdout: &str, stderr: &str) -> BenchResult {
    let mut r = BenchResult::default();

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
        // [model, size, params, backend, threads, type_k, type_v, fa, test, t/s]
        if cells.len() < 10 || cells[0] == "model" || cells[0].starts_with("---") {
            continue;
        }
        let test = cells[8];
        let tps = cells[9]
            .split_whitespace()
            .next()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0);
        if test.starts_with("pp") {
            r.prefill_tps = tps;
        } else if test.starts_with("tg") {
            r.decode_tps = tps;
        }
        r.model_label = cells[0].to_string();
        r.params_b = cells[2]
            .split_whitespace()
            .next()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0);
        r.backend_label = cells[3].to_string();
        r.type_k = cells[5].to_string();
        r.type_v = cells[6].to_string();
        r.flash_attn = cells[7] == "1";
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

    // Real output captured from the user's machine (AMD Radeon Pro 5500M, Vulkan).
    const STDOUT: &str = "\
| model                    |   size |  params | backend     | threads | type_k | type_v |  fa |  test |              t/s |
| ------------------------ | -----: | ------: | ----------- | ------: | -----: | -----: | --: | ----: | ---------------: |
| qwen35 4B Q4_K - Medium  | 2.70 GiB | 4.21 B | Vulkan,BLAS |       8 |   q4_0 |   q4_0 |   1 | pp512 |     33.10 ± 0.29 |
| qwen35 4B Q4_K - Medium  | 2.70 GiB | 4.21 B | Vulkan,BLAS |       8 |   q4_0 |   q4_0 |   1 | tg128 |     23.65 ± 1.96 |

build: 7dad2f1a1 (9660)";
    const STDERR: &str = "ggml_vulkan: 0 = AMD Radeon Pro 5500M (MoltenVK) | uma: 0 | fp16: 1";

    #[test]
    fn parses_real_output() {
        let r = parse(STDOUT, STDERR);
        assert!((r.prefill_tps - 33.10).abs() < 0.01);
        assert!((r.decode_tps - 23.65).abs() < 0.01);
        assert!((r.params_b - 4.21).abs() < 0.01);
        assert_eq!(r.type_k, "q4_0");
        assert!(r.flash_attn);
        assert_eq!(r.build_number, "b9660");
        assert_eq!(r.git_hash, "7dad2f1a1");
        assert_eq!(
            r.devices,
            vec!["AMD Radeon Pro 5500M (MoltenVK)".to_string()]
        );
    }
}
