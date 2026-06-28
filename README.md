# llamabench runner

The benchmark submitter for **[llamabench.ai](https://llamabench.ai)** — the crowd-sourced
local-LLM speed leaderboard.

It's a single, self-contained CLI (`llamabench`) that **bundles nothing**: it shells out to
*your existing* `llama.cpp` build (`llama-bench` for standardized prefill/decode speed, and
`llama-server` for deterministic multi-turn output-correctness checks), assembles a result, and
submits it to the leaderboard. It's open source so you can see exactly what runs on your machine
before you `curl … | sh` it.

## Install

```sh
curl -fsSL https://llamabench.ai/install.sh | sh
```

This downloads the prebuilt binary for your OS/arch from the [latest release](../../releases/latest)
and puts `llamabench` on your PATH. Prefer to do it by hand? Grab the archive for your platform
from the [Releases page](../../releases) and drop the binary somewhere on your PATH.

Supported prebuilt targets: Linux x86_64, macOS (Intel + Apple Silicon), Windows x86_64.

## Usage

```sh
llamabench --help

# 1. Save your token once — it's stored in your per-user config dir, so later
#    `run`s submit without --token. (Get one at https://llamabench.ai/account.)
llamabench auth <token>

# 2. Easiest full run: fetch the model from Hugging Face AND a prebuilt llama.cpp,
#    benchmark, verify, and submit — no local setup required.
llamabench run --hf-model bartowski/Llama-3.1-8B-Instruct-GGUF --quant Q4_K_M --download-llama

# Already have a llama.cpp build? Point at it instead of --download-llama:
llamabench run --hf-model bartowski/Llama-3.1-8B-Instruct-GGUF --quant Q4_K_M \
  --llama-dir /path/to/llama.cpp/build/bin

# Or use a local model file:
llamabench run --model /path/to/model.gguf --llama-dir /path/to/llama.cpp/build/bin

# Benchmarking a llama.cpp fork? Name it with --family so the result is recorded
# under that engine (ik_llama.cpp, beellama.cpp, or Xpress AI's ve_llama.cpp for the
# NEC Vector Engine). Forks have no prebuilt download — point --llama-dir at your build.
llamabench run --model /path/to/model.gguf \
  --family ik_llama.cpp --llama-dir /path/to/ik_llama.cpp/build/bin

# Use a LOCAL file but attribute + hash-verify its Hugging Face provenance
# (records the repo and a ✓/⚠ verified flag; the local bytes are what's benchmarked):
llamabench run --model /path/to/Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --hf-model bartowski/Llama-3.1-8B-Instruct-GGUF --quant Q4_K_M

# Speed only — run llama-bench and print the numbers:
llamabench bench --model /path/to/model.gguf --llama-dir /path/to/llama.cpp/build/bin

# Output-correctness verification against llama-server (fixed seed, temp 0, multi-turn):
llamabench verify --model /path/to/model.gguf

# Build the result without submitting (no token required):
llamabench run --model /path/to/model.gguf --dry-run
```

### Getting the model and llama.cpp

- **`--hf-model <repo> --quant <Q>`** downloads a GGUF straight from Hugging Face
  (streamed to a per-user cache, skipped if already present), picking the `.gguf`
  whose name matches the quant. `--quant` also sets the quant recorded in the result.
  Use `--model <path>` instead to point at a local file.
- **Model attribution:** when you pass `--hf-model`, the submission is attributed to the
  GGUF's **base/finetune model** (its Hugging Face `base_model`, e.g.
  `unsloth/gemma-4-12b-it-GGUF` → `google/gemma-4-12b-it`) rather than the per-quant
  llama-bench label, so every GGUF repack of the same model groups together on the
  leaderboard. The repo is still recorded as provenance in `hfModel`. If no `base_model`
  is published (or no `--hf-model` is given), the original per-quant label is kept.
- **`--model <path> --hf-model <repo> --quant <Q>`** (given *together*) benchmarks the
  **local** file but records its Hugging Face provenance and verifies it: the runner
  streams the local file through SHA-256 and compares it against the repo's published
  hash (the `lfs.oid` from HF's tree API) for the matching quant. The result carries
  `hfModel` and `hfVerified` (`✓` match / `⚠` mismatch). A provenance check that can't
  be resolved records `hfVerified: false` and never fails the run.
- **`--download-llama`** grabs the latest prebuilt llama.cpp release for your OS/arch.
  **This is the standard CPU/Metal build only — GPU builds (CUDA / HIP / Vulkan) are
  NOT auto-selected.** If you have a GPU, build llama.cpp yourself and point
  `--llama-dir` at it for full speed. With neither `--llama-dir` nor `--download-llama`,
  the runner uses `llama-bench`/`llama-server` from your `PATH`, and falls back to the
  prebuilt CPU/Metal build if they aren't found.
- **`--family <llama.cpp|ik_llama.cpp|beellama.cpp|ve_llama.cpp>`** records which
  llama.cpp variant the build is (default `llama.cpp`), so results from different engines
  stay comparable but distinct on the leaderboard. The forks share the same
  `llama-bench`/`llama-server` CLI, so the runner drives them identically — but only
  upstream llama.cpp has prebuilt downloads, so build the fork and point `--llama-dir`
  at it (or put its binaries on `PATH`). `ve_llama.cpp` is Xpress AI's fork adding NEC
  SX-Aurora Vector Engine support.

### Token resolution

`run` resolves the submission token in this order: `--token` flag →
`LLAMABENCH_TOKEN` env var → the token saved by `llamabench auth`. If none is found
(and you're not using `--dry-run`), it errors and points you at `llamabench auth`.

Common flags (see `--help` for the full list): `--ngl`, `--fa`, `--ctk`/`--ctv` (KV cache type),
`--n-prompt`/`--n-gen`, `--spec-decode`, `--seed`, `--turns`, `--reps`.

Pass extra flags straight through to `llama-server` (handy for the many speculative-decoding
options) with either:

- **`--server-args "<flags>"`** — one whitespace-delimited string, e.g.
  `--server-args "--spec-type draft-mtp --spec-draft-n-max 2"`. Easiest for a bunch at once.
- **`--server-arg <value>`** — repeatable, one value each
  (`--server-arg --foo --server-arg "two words"`). Use it when a value contains spaces.

Both are appended (repeatable `--server-arg` first, then the split `--server-args`).

## Build from source

```sh
cargo build --release
# binary at target/release/llamabench
```

Requires a stable Rust toolchain. The only dependencies are crates.io packages — no submodules,
no codegen.

## How submissions are trusted

Results are submitted under a token tied to your llamabench.ai account and land **unverified**; a
`✓ verified` badge is reserved for independently reproduced results. The runner records the exact
configuration and the `llama.cpp` revision so any result is reproducible. See the
[Methodology](https://llamabench.ai/methodology) page for details.

## License

[GPL-3.0-or-later](LICENSE). The llamabench.ai web app is a separate, proprietary project; the
runner talks to it only over the documented result API.
