// SPDX-License-Identifier: GPL-3.0-or-later
//! The result-submission wire contract (ADR-005). Mirrors
//! `the llamabench result contract`; camelCase via serde rename so the JSON
//! matches the schema the server validates against.

use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
pub struct Hardware {
    pub id: String,
    pub name: String,
    pub vendor: String,
    #[serde(rename = "vramGb")]
    pub vram_gb: f64,
    #[serde(rename = "bandwidthGbs")]
    pub bandwidth_gbs: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub params: f64,
    /// Source Hugging Face repo this model is attributed to (download path, or a
    /// local `--model` paired with `--hf-model`). Omitted for an unattributed local file.
    #[serde(rename = "hfModel", skip_serializing_if = "Option::is_none")]
    pub hf_model: Option<String>,
    /// Whether the model's bytes were confirmed to come from `hf_model`: trivially
    /// true on the download path, or the result of a SHA-256 match for a local file.
    #[serde(rename = "hfVerified", skip_serializing_if = "Option::is_none")]
    pub hf_verified: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Metrics {
    #[serde(rename = "decodeTps")]
    pub decode_tps: f64,
    #[serde(rename = "prefillTps")]
    pub prefill_tps: f64,
    #[serde(rename = "ttftMs", skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub quant: String,
    #[serde(rename = "kvCache")]
    pub kv_cache: String,
    #[serde(rename = "contextLength")]
    pub context_length: u32,
    #[serde(rename = "flashAttention")]
    pub flash_attention: bool,
    #[serde(rename = "specDecode")]
    pub spec_decode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
}

/// The inference engine and its EXACT build. The git hash + build number pin the
/// llama.cpp (or other backend) revision so results are reproducible and a
/// build-specific regression is attributable (user request).
#[derive(Debug, Serialize, Deserialize)]
pub struct Backend {
    pub name: String,
    pub version: String, // build number, e.g. "b9660"
    #[serde(rename = "gitHash")]
    pub git_hash: String, // e.g. "7dad2f1a1"
}

/// One generation in the verification matrix: a specific test prompt, conversation
/// depth (turns), and repetition.
#[derive(Debug, Serialize, Deserialize)]
pub struct VerificationRun {
    #[serde(rename = "promptId")]
    pub prompt_id: String,
    pub turns: u32, // 1, 2, 3 — multi-turn catches KV-cache bugs that only break on later turns
    pub rep: u32,   // 1..=reps — temp-0 reps should match on the same build/hardware
    #[serde(rename = "outputSha256")]
    pub output_sha256: String,
    #[serde(rename = "outputPreview")]
    pub output_preview: String,
    pub gibberish: bool,
}

/// Output-correctness check (user requests): speed alone is gameable/buggy, so we
/// run a few fixed prompts at a fixed seed and temperature 0 (greedy → deterministic
/// for a given model+backend+build), each repeated and at 1/2/3 conversational turns
/// (a class of bug returns gibberish only on the 2nd/3rd turn). Small deviations
/// between reps/hardware are fine; **gibberish makes the submission invalid**. The
/// hashes also let the server compare outputs across submissions of the same config.
#[derive(Debug, Serialize, Deserialize)]
pub struct Verification {
    pub seed: u64,
    pub temperature: f64,
    #[serde(rename = "nGen")]
    pub n_gen: u32,
    /// false if ANY run produced gibberish.
    pub valid: bool,
    pub runs: Vec<VerificationRun>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResultSubmission {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    pub hardware: Hardware,
    pub model: ModelInfo,
    pub metrics: Metrics,
    pub config: Config,
    pub backend: Backend,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verification: Option<Verification>,
    pub submitter: Submitter,
    pub signature: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Submitter {
    pub handle: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_info_serializes_hf_fields() {
        let m = ModelInfo {
            id: "x".to_string(),
            name: "X".to_string(),
            params: 1.0,
            hf_model: Some("bartowski/Foo-GGUF".to_string()),
            hf_verified: Some(true),
        };
        let j = serde_json::to_value(&m).unwrap();
        assert_eq!(j["hfModel"], "bartowski/Foo-GGUF");
        assert_eq!(j["hfVerified"], true);
    }

    #[test]
    fn model_info_omits_hf_fields_when_none() {
        let m = ModelInfo {
            id: "x".to_string(),
            name: "X".to_string(),
            params: 1.0,
            hf_model: None,
            hf_verified: None,
        };
        let obj = serde_json::to_value(&m).unwrap();
        let obj = obj.as_object().unwrap();
        assert!(!obj.contains_key("hfModel"));
        assert!(!obj.contains_key("hfVerified"));
    }
}
