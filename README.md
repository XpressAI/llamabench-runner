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

## Usage — run the exact native command

Put runner-owned options before `--` and the complete native command after it.
Your **exact configuration** is benchmarked, verified, recorded verbatim as the
reproduce command, and submitted:

```sh
# 1. Save your token once (get one at https://llamabench.ai/account).
llamabench auth <token>

# 2a. Benchmark your llama-bench configuration:
llamabench speed -- \
  llama-bench -m model.gguf -ngl 99 -fa on -ub 2048 -ot "ffn=CPU"

# 2b. Or benchmark your llama-server configuration:
llamabench speed -- \
  llama-server -m model.gguf -c 8192 -np 2 --jinja
```

Every flag is passed through to the real tool untouched (matrix runs like
`-ngl 0,99` submit one result per configuration). Runner flags live before
`--`, so they cannot collide with llama.cpp's: `--dry-run` (don't submit),
`--no-verify` (skip the output-correctness pass), `--token <t>`,
`--handle <@you>`, `--family <fork>`, `--llama-dir <bin-dir>`, `--api <url>`,
`--download-llama`.

The explicit `speed` and `eval` workflows refuse inherited `LLAMA_ARG_*`
settings because an environment-only option would be missing from the published
config. Unset it and put the equivalent native flag after `--`.

The v0.4.x drop-in forms remain compatible: bare `llamabench <flags>` selects
llama-bench unless it sees server-only flags, while explicit
`llamabench llama-bench …` and `llamabench llama-server …` force the tool. New
scripts should prefer `speed -- <native command>` because the ownership boundary
is visible and does not depend on flag sniffing.

In llama-bench mode the speed table you know streams as usual and the numbers
are read from llama-bench's own per-test output (`-oe jsonl` is appended); TTFT
is probed on the verification server with the same standardized ~512-token
prompt the server mode uses, so the two modes' TTFTs are comparable. In
llama-server mode the server runs with your args verbatim and prefill/decode/TTFT
come from the server's own `timings` on standardized requests (temp 0, ~512-token
prompt, 128 generated tokens, median of 3).

### Hugging Face provenance — automatic, verified on the site

Every submission of a local file records the GGUF's **SHA-256** (hashed once per
file, then cached by size/mtime). Linking that file to the Hugging Face repo it
came from happens **on llamabench.ai**: open the result and name the repo — the
server verifies the hash against the repo's published LFS hashes and, once one
person has linked a file, every past and future submission of the same bytes is
attributed automatically. Nothing to do in the CLI.

Prefer to pin provenance locally (offline / scripted runs)? The CLI link store
still works and takes precedence:

```sh
llamabench link ./gemma-4-12b-it-UD-Q4_K_XL.gguf unsloth/gemma-4-12b-it-GGUF
llamabench link --list            # show all links
llamabench link --forget <path>   # remove one
```

### Compatibility subcommands

The original flag-based `run`/`bench`/`verify` interface remains available for
v0.4.x scripts and the download-with-no-local-setup flow. New commands do not
copy this partial llama.cpp flag surface:

```sh
# Fetch the model from Hugging Face AND a prebuilt llama.cpp — no local setup:
llamabench run --hf-model bartowski/Llama-3.1-8B-Instruct-GGUF --quant Q4_K_M --download-llama

# Local model + your own llama.cpp build:
llamabench run --model /path/to/model.gguf --llama-dir /path/to/llama.cpp/build/bin

# Benchmarking a llama.cpp fork? Name it with --family so the result is recorded
# under that engine (ik_llama.cpp, beellama.cpp, or Xpress AI's ve_llama.cpp for the
# NEC Vector Engine). Forks have no prebuilt download — point --llama-dir at your build.
llamabench run --model /path/to/model.gguf \
  --family ik_llama.cpp --llama-dir /path/to/ik_llama.cpp/build/bin

# One-off provenance without a persistent link (hash-verified this run only):
llamabench run --model /path/to/Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --hf-model bartowski/Llama-3.1-8B-Instruct-GGUF --quant Q4_K_M

# Speed only / verification only / build-but-don't-submit:
llamabench bench --model /path/to/model.gguf
llamabench verify --model /path/to/model.gguf
llamabench run --model /path/to/model.gguf --dry-run
```

### Exact-config behavior evaluation

`eval` is a separate, opt-in fixed evaluation, not another speed measurement
and not part of `run`. Runner-owned options go before `--`; every token after
`--` is passed to `llama-server` in order and becomes part of the recorded
runtime configuration:

```sh
llamabench eval --model /path/to/model-Q4_K_M.gguf \
  --llama-dir /path/to/llama.cpp/build/bin --context-length 8192 -- \
  -ngl 99 -ctk q4_0 -ctv q4_0 -fa auto \
  --spec-type draft-mtp --spec-draft-n-max 2
```

There is no `--server-arg` repetition and no whitespace-split `--server-args`
string in `eval`; normal shell quoting determines the native argument vector.

The versioned evaluation starts `llama-server` once and runs five bounded
scenarios: a pelican-on-a-bicycle SVG, a self-contained Breakout game, two
deterministic virtual-workspace tool-use tasks, and a three-turn Phileas Fogg
role-play. It uses temperature 0, seed 42, and at most 2,848 generated tokens in
total. At 1 tok/s that is under 48 minutes; faster models finish proportionally
sooner. Use `--dry-run` to inspect the complete signed JSON without submitting.

Every evaluation is tied to the GGUF SHA-256, backend build, effective context,
KV-cache K/V types, flash-attention mode, speculative-decoding settings, and the
ordered byte-for-byte native argument vector. The server derives a configuration
fingerprint from those structured values, so Q4 KV-cache evidence never stands
in for Q8/F16 and speculative decoding never stands in for ordinary decoding.
External draft models, LoRAs, grammars, projectors, control vectors, and template
files are rejected in `eval-v1` until the contract can hash every auxiliary
artifact. Absolute native path values also fail closed rather than being shortened
into a potentially colliding config. Built-in speculative decoding and its tuning
flags are supported.

The site reports separate inspectable outcomes rather than an aggregate
"intelligence" score. The runner's virtual tools never touch the real filesystem:
they operate only on fixed in-memory fixtures. Generated SVG is rasterized by the
server before display, and generated HTML is never executed by llamabench.ai.

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

Submission commands resolve the token in this order: `--token` flag →
`LLAMABENCH_TOKEN` env var → the token saved by `llamabench auth`. If none is found
(and you're not using `--dry-run`), it errors and points you at `llamabench auth`.

Common flags (see `--help` for the full list): `--ngl`, `--fa`, `--ctk`/`--ctv` (KV cache type),
`--n-prompt`/`--n-gen`, `--spec-decode`, `--seed`, `--turns`, `--reps`.

The compatibility `run` command still accepts its historical pass-through forms:

- **`--server-args "<flags>"`** — one whitespace-delimited string.
- **`--server-arg <value>`** — repeatable, one value each
  (`--server-arg --foo --server-arg "two words"`). Use it when a value contains spaces.

Both are appended (repeatable `--server-arg` first, then the split `--server-args`).
Prefer the native argument vector after `--` in new commands; it preserves quoted
values exactly.

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
