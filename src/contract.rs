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
    /// Host CPU model as reported by the OS — context for GPU results, the star
    /// of CPU results. Optional in the contract; omitted when detection fails.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu: Option<String>,
    /// Total host memory in GiB. Per-submission context for models that only
    /// partially fit in accelerator memory; omitted when detection fails.
    #[serde(rename = "systemRamGb", skip_serializing_if = "Option::is_none")]
    pub system_ram_gb: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub params: f64,
    /// The canonical model this submission is attributed to: the HF repo one level up
    /// the GGUF's model tree — the unquantized finetune it quantizes, or the base model
    /// when there's no finetune (e.g. `google/gemma-4-12b-it`). Lets every GGUF repack
    /// of the same model group together. Omitted when there's no `--hf-model` base.
    #[serde(rename = "baseModel", skip_serializing_if = "Option::is_none")]
    pub base_model: Option<String>,
    /// Source Hugging Face repo this model is attributed to (download path, or a
    /// local `--model` paired with `--hf-model`). Omitted for an unattributed local file.
    #[serde(rename = "hfModel", skip_serializing_if = "Option::is_none")]
    pub hf_model: Option<String>,
    /// Whether the model's bytes were confirmed to come from `hf_model`: trivially
    /// true on the download path, or the result of a SHA-256 match for a local file.
    #[serde(rename = "hfVerified", skip_serializing_if = "Option::is_none")]
    pub hf_verified: Option<bool>,
    /// SHA-256 of the benchmarked GGUF (lowercase hex). The server uses it to attach
    /// web-side Hugging Face provenance: one verified link on llamabench.ai
    /// attributes every submission of the same file (ADR-010).
    #[serde(rename = "ggufSha256", skip_serializing_if = "Option::is_none")]
    pub gguf_sha256: Option<String>,
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

// ---------------------------------------------------------------------------
// Opt-in exact-configuration behavior evaluation (ADR-014 / ADR-016).
// ---------------------------------------------------------------------------

pub const EVAL_SCHEMA_VERSION: u32 = 2;
pub const EVAL_VERSION: &str = "eval-v2";

#[derive(Debug, Serialize, Deserialize)]
pub struct EvaluationSettings {
    pub seed: u64,
    pub temperature: f64,
    #[serde(rename = "topK")]
    pub top_k: u32,
    #[serde(rename = "topP")]
    pub top_p: f64,
    #[serde(rename = "minP")]
    pub min_p: f64,
    #[serde(rename = "presencePenalty")]
    pub presence_penalty: f64,
    #[serde(rename = "frequencyPenalty")]
    pub frequency_penalty: f64,
    #[serde(rename = "repeatPenalty")]
    pub repeat_penalty: f64,
    #[serde(rename = "visualMaxTokens")]
    pub visual_max_tokens: u32,
    #[serde(rename = "agentTaskMaxTokens")]
    pub agent_task_max_tokens: u32,
    #[serde(rename = "roleplayMaxTokensPerTurn")]
    pub roleplay_max_tokens_per_turn: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvaluationCheck {
    pub id: String,
    pub label: String,
    pub passed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VisualArtifact {
    #[serde(rename = "promptId")]
    pub prompt_id: String,
    pub output: String,
    #[serde(rename = "outputSha256")]
    pub output_sha256: String,
    #[serde(rename = "durationMs")]
    pub duration_ms: u64,
    #[serde(rename = "generatedTokens")]
    pub generated_tokens: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checks: Vec<EvaluationCheck>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvaluationVisuals {
    #[serde(rename = "pelicanSvg")]
    pub pelican_svg: VisualArtifact,
    #[serde(rename = "breakoutHtml")]
    pub breakout_html: VisualArtifact,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub name: String,
    pub arguments: String,
    pub result: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AgenticTaskResult {
    pub id: String,
    pub passed: bool,
    #[serde(rename = "durationMs")]
    pub duration_ms: u64,
    #[serde(rename = "generatedTokens")]
    pub generated_tokens: u32,
    #[serde(rename = "toolCalls")]
    pub tool_calls: Vec<ToolCallRecord>,
    #[serde(rename = "finalOutput")]
    pub final_output: String,
    #[serde(rename = "finalOutputSha256")]
    pub final_output_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RoleplayTurn {
    #[serde(rename = "questionId")]
    pub question_id: String,
    pub answer: String,
    #[serde(rename = "answerSha256")]
    pub answer_sha256: String,
    #[serde(rename = "generatedTokens")]
    pub generated_tokens: u32,
    pub checks: Vec<EvaluationCheck>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RoleplayResult {
    pub id: String,
    pub character: String,
    pub work: String,
    #[serde(rename = "durationMs")]
    pub duration_ms: u64,
    pub turns: Vec<RoleplayTurn>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvaluationModel {
    pub id: String,
    pub name: String,
    pub params: f64,
    #[serde(rename = "baseModel", skip_serializing_if = "Option::is_none")]
    pub base_model: Option<String>,
    #[serde(rename = "hfModel", skip_serializing_if = "Option::is_none")]
    pub hf_model: Option<String>,
    #[serde(rename = "hfVerified", skip_serializing_if = "Option::is_none")]
    pub hf_verified: Option<bool>,
    #[serde(rename = "ggufFile")]
    pub gguf_file: String,
    #[serde(rename = "ggufSha256")]
    pub gguf_sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub quant: String,
    #[serde(rename = "contextLength")]
    pub context_length: u32,
    #[serde(rename = "contextMode")]
    pub context_mode: String,
    #[serde(rename = "kvCacheKey")]
    pub kv_cache_key: String,
    #[serde(rename = "kvCacheValue")]
    pub kv_cache_value: String,
    #[serde(rename = "flashAttention")]
    pub flash_attention: String,
    #[serde(rename = "speculativeDecoding")]
    pub speculative_decoding: SpeculativeDecodingConfig,
    #[serde(rename = "runtimeArgs")]
    pub runtime_args: Vec<String>,
    /// Local display only. Eval-v2 omits this from the signed wire payload so the
    /// server can derive the public command from the validated structured fields.
    #[serde(skip)]
    pub command: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SpeculativeDecodingConfig {
    pub mode: String,
    pub parameters: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvaluationSubmission {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    #[serde(rename = "evalVersion")]
    pub eval_version: String,
    pub model: EvaluationModel,
    pub config: EvaluationConfig,
    pub backend: Backend,
    pub settings: EvaluationSettings,
    pub visuals: EvaluationVisuals,
    #[serde(rename = "agenticTasks")]
    pub agentic_tasks: Vec<AgenticTaskResult>,
    pub roleplay: RoleplayResult,
    pub submitter: Submitter,
    pub signature: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardware_serializes_optional_system_ram() {
        let hardware = Hardware {
            id: "rtx4060".to_string(),
            name: "NVIDIA GeForce RTX 4060".to_string(),
            vendor: "NVIDIA".to_string(),
            vram_gb: 0.0,
            bandwidth_gbs: 0.0,
            cpu: Some("AMD Ryzen 9 7950X".to_string()),
            system_ram_gb: Some(128),
        };
        let json = serde_json::to_value(&hardware).unwrap();
        assert_eq!(json["systemRamGb"], 128);
    }

    #[test]
    fn model_info_serializes_hf_fields() {
        let m = ModelInfo {
            id: "gemma-4-12b-it".to_string(),
            name: "gemma-4-12b-it".to_string(),
            params: 1.0,
            base_model: Some("google/gemma-4-12b-it".to_string()),
            hf_model: Some("unsloth/gemma-4-12b-it-GGUF".to_string()),
            hf_verified: Some(true),
            gguf_sha256: Some("ab".repeat(32)),
        };
        let j = serde_json::to_value(&m).unwrap();
        assert_eq!(j["baseModel"], "google/gemma-4-12b-it");
        assert_eq!(j["hfModel"], "unsloth/gemma-4-12b-it-GGUF");
        assert_eq!(j["hfVerified"], true);
        assert_eq!(j["ggufSha256"], "ab".repeat(32));
    }

    #[test]
    fn model_info_omits_hf_fields_when_none() {
        let m = ModelInfo {
            id: "x".to_string(),
            name: "X".to_string(),
            params: 1.0,
            base_model: None,
            hf_model: None,
            hf_verified: None,
            gguf_sha256: None,
        };
        let obj = serde_json::to_value(&m).unwrap();
        let obj = obj.as_object().unwrap();
        assert!(!obj.contains_key("baseModel"));
        assert!(!obj.contains_key("hfModel"));
        assert!(!obj.contains_key("hfVerified"));
        assert!(!obj.contains_key("ggufSha256"));
    }

    #[test]
    fn roleplay_turn_serializes_its_own_token_count() {
        let turn = RoleplayTurn {
            question_id: "modern-navigation".to_string(),
            answer: "I cannot use such a device.".to_string(),
            answer_sha256: "ab".repeat(32),
            generated_tokens: 12,
            checks: Vec::new(),
        };
        let json = serde_json::to_value(&turn).unwrap();
        assert_eq!(json["generatedTokens"], 12);
    }

    #[test]
    fn evaluation_config_serializes_exact_runtime_identity() {
        let config = EvaluationConfig {
            quant: "Q4_K_M".to_string(),
            context_length: 8192,
            context_mode: "auto-fit".to_string(),
            kv_cache_key: "q4_0".to_string(),
            kv_cache_value: "q4_0".to_string(),
            flash_attention: "auto".to_string(),
            speculative_decoding: SpeculativeDecodingConfig {
                mode: "draft-mtp".to_string(),
                parameters: vec!["--spec-draft-n-max=2".to_string()],
            },
            runtime_args: vec![
                "-ctk".to_string(),
                "q4_0".to_string(),
                "--spec-type".to_string(),
                "draft-mtp".to_string(),
            ],
            command: "llamabench eval --model ./model.gguf -- -ctk q4_0".to_string(),
        };
        let json = serde_json::to_value(config).unwrap();
        assert_eq!(json["contextLength"], 8192);
        assert_eq!(json["contextMode"], "auto-fit");
        assert_eq!(json["kvCacheKey"], "q4_0");
        assert_eq!(json["kvCacheValue"], "q4_0");
        assert_eq!(json["flashAttention"], "auto");
        assert_eq!(json["speculativeDecoding"]["mode"], "draft-mtp");
        assert_eq!(json["runtimeArgs"][0], "-ctk");
        assert!(json.get("command").is_none());
    }

    #[test]
    fn evaluation_model_serializes_the_reproduce_basename() {
        let model = EvaluationModel {
            id: "model".to_string(),
            name: "Model".to_string(),
            params: 8.0,
            base_model: None,
            hf_model: None,
            hf_verified: None,
            gguf_file: "model-Q4_K_M.gguf".to_string(),
            gguf_sha256: "ab".repeat(32),
        };
        let json = serde_json::to_value(model).unwrap();
        assert_eq!(json["ggufFile"], "model-Q4_K_M.gguf");
    }
}
