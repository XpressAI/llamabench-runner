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

# Speed only — run llama-bench and print the numbers:
llamabench bench --model /path/to/model.gguf --llama-dir /path/to/llama.cpp/build/bin

# Output-correctness verification against llama-server (fixed seed, temp 0, multi-turn):
llamabench verify --model /path/to/model.gguf

# Full run: speed + verification → a complete result. Add a token to actually submit.
export LLAMABENCH_TOKEN=<token from https://llamabench.ai/account>
llamabench run --model /path/to/model.gguf

# Build the result without submitting:
llamabench run --model /path/to/model.gguf --dry-run
```

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
