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
- **`--download-llama`** grabs the latest prebuilt llama.cpp release for your OS/arch.
  **This is the standard CPU/Metal build only — GPU builds (CUDA / HIP / Vulkan) are
  NOT auto-selected.** If you have a GPU, build llama.cpp yourself and point
  `--llama-dir` at it for full speed. With neither `--llama-dir` nor `--download-llama`,
  the runner uses `llama-bench`/`llama-server` from your `PATH`, and falls back to the
  prebuilt CPU/Metal build if they aren't found.

### Token resolution

`run` resolves the submission token in this order: `--token` flag →
`LLAMABENCH_TOKEN` env var → the token saved by `llamabench auth`. If none is found
(and you're not using `--dry-run`), it errors and points you at `llamabench auth`.

Common flags (see `--help` for the full list): `--ngl`, `--fa`, `--ctk`/`--ctv` (KV cache type),
`--n-prompt`/`--n-gen`, `--spec-decode`, `--seed`, `--turns`, `--reps`, and `--server-arg` to pass
extra flags straight through to `llama-server`.

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
