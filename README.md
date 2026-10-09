<h1 align="center">Strata-GLM-AMD</h1>

**English** · [日本語](README.ja.md)

<p align="center"><b>GLM-5.3-Flash with 256K context on two AMD workstation cards</b><br>
a fork of <a href="https://github.com/Niko1221/Strata">Strata</a> · Linux + ROCm · free and open source</p>

Strata-GLM-AMD adds a second engine to [Strata](https://github.com/Niko1221/Strata): **`strata-glm5`**, which runs
**[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** (the `glm5-next` architecture: 45 layers, 288
experts, multi-token prediction) at a **256K-token context** on a pair of 48 GB AMD cards, or more slowly on one.
It chats, writes code, thinks when asked, calls tools, and serves the same OpenAI / Anthropic / Responses APIs
and browser app as Strata. Everything of Strata that is not about this model is unchanged; the original README
is [upstream](https://github.com/Niko1221/Strata#readme).

> **This repository is specialized for AMD RDNA 3** (gfx1100: Radeon Pro W7900, Radeon RX 7900 XTX). It is tested on
> nothing else. If you want it on another architecture (RDNA 4, CDNA, NVIDIA), ask the Codex you have at hand (or
> another coding agent): point it at this repository and [docs/GLM5_NEXT.md](docs/GLM5_NEXT.md).

The engine is built on Strata's design (the busiest experts on the GPUs, the rest in pinned RAM, misses computed
on the CPU, a guess-and-check draft model) plus work specific to this model:

- **Long context at a flat cost.** The model's DSA layers select attention cells with a learned indexer over
  the whole context. Strata-GLM-AMD keeps the indexer's choice but evaluates it window by window: the selection is
  shared by a prompt window, the selected cells are gathered into one compact int8 operand, and decoding reads
  them with a fused split-K kernel on the latent cache, so the attention cost of a window stops growing with the
  context. 128K prompts read 3x faster than with the plain path, and the needle-in-a-haystack test passes at
  236K.
- **All caches as Q8_0.** The latent KV, indexer key and gate caches are 8-bit rows: one slot's 256K context is
  about 1.2 GiB per card instead of f32's 5 GiB, and the prompt attention reads them as its int8 operand directly.
- **Q2_K experts on the tensor cores.** WMMA kernels for the 2-bit expert weights, for prompts (22-28 TMAC/s against
  the generic 10-12) and for decoding (550 GB/s against 115-160) - the quantization stays as downloaded.
- **A decode-time expert cache.** The experts the cards do not hold are computed on the CPU from pinned RAM, which is
  the memory bus's floor; while decoding, the prompt buffers are lent out as a cache of the experts that came over
  PCIe, so fresh conversations route to VRAM more often (+7-19% on new prompts). The shipped routing profile
  (`data/glm53/expert-profile.txt`) seeds the static ranking and keeps accumulating on your traffic.
- **A two-card prompt pipeline.** The layers are split across the cards; while card 1 finishes window k, card 0
  already runs window k+1. 128K prompts: 592 -> 1,060 tokens/s, outputs identical.
- **Tool calls and thinking** in GLM-5's own chat-template format (`<tool_call>name<arg_key>...`), thinking
  on/off per request.

## How fast is it?

Measured on the development machine: 2x Radeon Pro W7900 (48 GB each, gfx1100), Ryzen 7 5700X3D (8 cores, AVX2
only), 125 GB RAM, Linux, ROCm 7. The model is the 3-bit GSQ-RCO GGUF (117 GB) with the Q4_K MTP draft layer.

| | 2 cards | 1 card |
| --- | ---: | ---: |
| **Writes answers** (one chat, with the draft model) | 30-39 tokens/s | 15-16 tokens/s |
| **Two chats at once** (`"parallel": 2`, aggregate) | ~30 tokens/s | - |
| **Reads a 4K prompt** | 750 tokens/s | - |
| **Reads a 12K prompt** | 1,000-1,100 tokens/s | 490 tokens/s |
| **Reads a 128K prompt** | 1,060 tokens/s | - |
| **Needle in a haystack** (3 depths) | 3/3 up to 236K | - |

Writing speed is bound by the host RAM bandwidth of this machine (the experts that are not on the cards are
read from pinned RAM every token); a machine with faster memory channels writes faster. Prompt reading is bound
by the expert products on the cards. Every number, the experiments behind them and the ones that did not pay
off are in [docs/GLM5_NEXT.md](docs/GLM5_NEXT.md).

## What you need

| | |
| --- | --- |
| **Graphics cards** | Tested: **2x AMD Radeon Pro W7900** (48 GB, gfx1100). One W7900 works with a smaller resident expert set (see [one card](#one-card)). Other RDNA 3 / RDNA 4 cards with 48 GB should work but are untested; NVIDIA is untested. |
| **RAM** | **128 GB.** The experts that are not on the cards live in pinned RAM: 43 GiB with two cards, 86 GiB with one. |
| **Disk** | About 125 GB for the model files, on an SSD. |
| **System** | Linux with ROCm 7 (the HIP engine is compiled on your PC), Python 3.11+, a C++ compiler, CMake 3.24+, Ninja. |

## Install

### With the installer

```bash
git clone https://github.com/OpenMOSE/Strata-GLM-AMD
cd Strata-GLM-AMD
./setup.sh --family glm53
```

It checks your cards and RAM, asks three questions (both cards or one, the context, requests at once), downloads
the two model files (122 GB), compiles the engine (10-20 minutes, once), extracts the tokenizer, writes the run
config with the expert budgets sized for your cards, and starts the model. Press Enter for the recommended
answers, or pass them: `--gpus 0,1 --context 262144 --parallel 1 --yes`. Model files you already have:
`--gguf-dir <folder>`. Later, `./setup.sh` alone starts the model (or `run-glm53.sh`); `./setup.sh --family glm53`
again changes the settings. After a `git pull` run `./update.sh`: a plain start does not rebuild the engine.

### By hand

The same steps, one by one.

#### 1. Build the engine

```bash
git clone https://github.com/OpenMOSE/Strata-GLM-AMD
cd Strata-GLM-AMD
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DSTRATA_PREFILL_MMQ=ON
cmake --build build-hip --target strata-glm5 -j
```

`STRATA_PREFILL_MMQ=ON` is required (the prompt path runs on ggml's int8 MMQ kernels). Without `STRATA_GGML_DIR`
CMake fetches the pinned ggml commit; point it at a llama.cpp checkout to use your own. The first build takes
10-20 minutes.

#### 2. Download the model

Two GGUF files, from Hugging Face:

- **The model:** [`GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf`](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF)
  (117.5 GB) - the 3-bit GSQ-RCO quantization, Q2_K experts.
- **The draft layer:** [`GLM-5.3-Flash-MTP-Q4_K.gguf`](https://huggingface.co/neuralll/GLM-5.3-Flash-MTP-GGUF)
  (4.6 GB) - the model's own multi-token-prediction layer, used to guess the next tokens.

#### 3. Extract the tokenizer

The tokenizer and chat template come out of the model file into a small directory (about 20 MB):

```bash
.venv/bin/python tools/strata_tokenizer.py --gguf /path/to/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf \
      --out /path/to/strata-data/glm53/pack
```

This writes `.../glm53/pack/tokenizer/`. The chat template is patched on the way so that a request with thinking off really gets no thinking (GLM's
template always opens a think block otherwise).

#### 4. Write the server config

`strata-glm53.json` in the Strata-GLM-AMD folder (two cards, one request at a time, 256K context):

```json
{
  "model_name": "GLM-5.3-Flash",
  "exe": "/path/to/Strata-GLM-AMD/build-hip/strata-glm5",
  "cwd": "/path/to/Strata-GLM-AMD",
  "tokenizer": "/path/to/strata-data/glm53/pack/tokenizer",
  "log": "/path/to/Strata-GLM-AMD/strata-glm53.log",
  "backend": "hip",
  "gpu": [0, 1],
  "env": {"HIP_VISIBLE_DEVICES": "0,1", "STRATA_GLM5_PF_BUFS": "2",
          "STRATA_GLM5_LOD_ROUTE": "pool", "STRATA_GLM5_LOD_TOP_POOLS": "1024", "STRATA_GLM5_LOD_NOSUM": "1"},
  "engine_silence_s": 600,
  "parallel": 1,
  "args": ["--serve",
           "--gguf", "/path/to/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf",
           "--mtp", "/path/to/GLM-5.3-Flash-MTP-Q4_K.gguf",
           "--devices", "0,1",
           "--expert-budget-gib", "31.0,29.5",
           "--profile", "/path/to/strata-data/glm53/profile_chat.txt",
           "--profile-save", "/path/to/strata-data/glm53/profile_chat.txt",
           "--window", "8", "--prefill-window", "4096", "--spec", "2",
           "--ctx", "262144",
           "--attn", "lod", "--lod-top", "128", "--lod-top-decode", "64"]
}
```

`--expert-budget-gib` is how much VRAM per card the experts may take; everything else (dense weights, caches,
prompt buffers, the prefetch ring) comes first. The values above leave about 1.2-1.5 GB free per W7900 at
256K. If the engine says `out of memory` at start, lower them; the start log prints the free VRAM after every
stage. The profile file ranks the experts by use; it does not have to exist at first (the engine then takes
the lowest ids) and is written after every request; the installer seeds it with the ranking measured on real
chats (`data/glm53/expert-profile.txt`).

#### 5. Start

```bash
.venv/bin/python serve/server.py --engine strata --config strata-glm53.json --port 8096
```

Loading takes 2-3 minutes (the dense weights, 60 GB of experts onto the cards, 43 GB pinned in RAM). Then the
browser app is at `http://127.0.0.1:8096` (Chat, Monitor, About) and the APIs at `/v1/chat/completions`,
`/v1/messages` and `/v1/responses` - exactly as in Strata ([using it](https://github.com/Niko1221/Strata#using-it)).
To take requests from other machines add `--host 0.0.0.0 --api-key <secret>`.

## Configurations

Three configs were measured; sanitized copies are in [`configs/glm53/`](configs/glm53/).

| File | Cards | Context | Requests at once | `--expert-budget-gib` | Notes |
| --- | --- | --- | --- | --- | --- |
| `strata-glm53-np1.json` | 2 | 256K | 1 | 31.0,29.5 | the fastest: 60 GB of experts resident |
| `strata-glm53.json` | 2 | 256K | 2 (`"parallel": 2`) | 28.5,25.8 | two chats decode together; each slot has its own 256K cache |
| `strata-glm53-1gpu.json` | 1 | 256K | 1 | 17.9 + `--pin-gib 96` | see below |

**Context and VRAM.** Every batch slot plus the solo slot has its own cache: at 256K that is 1.2 GiB (card 0)
and 1.8 GiB (card 1) per slot, so `"parallel": 2` costs about 3 GB more of VRAM per card than one request at a
time, which comes out of the expert budget and so out of decode speed.

### One card

<a id="one-card"></a>One W7900 holds the dense weights, the caches, a 2x3.8 GB prefetch ring and 18 GB of
experts. The other 86 GB of experts must be **pinned in host RAM** (`--pin-gib 96`; the default limit of 64 GB
would leave them unpinned, and the prompt path cannot stage that many): 128 GB of RAM is the minimum. Decode
runs at 15-16 tokens/s and 12K prompts at about 490 tokens/s. `"gpu": [0]`, `HIP_VISIBLE_DEVICES=0`,
`--devices 0`.

### Engine options

| Option | Meaning |
| --- | --- |
| `--ctx N` | the context per slot (tested up to 262144) |
| `--devices 0,1` / `--layer-split` | the cards and the layer split between them (auto: by VRAM) |
| `--expert-budget-gib G[,G2]` | VRAM for resident experts per card |
| `--pin-gib G` | the pinned-RAM limit for the non-resident experts (default 64); above it they are read from the file |
| `--prefill-window T` | tokens per prompt window (4096 recommended; windows feed the two-card pipeline) |
| `--window T`, `--spec K` | the decode window and the number of draft tokens (8 and 2) |
| `--attn lod`, `--lod-top`, `--lod-top-decode`, `--lod-page`, `--lod-local`, `--lod-sink` | the windowed sparse-attention mode and its selection sizes |
| `--slots N` (or `"parallel": N` in the config) | batch slots; 2 is the recommended maximum |
| `--profile FILE`, `--profile-save FILE` | the expert-usage ranking read at start / saved after each request |

Environment variables (the config's `"env"`): `STRATA_GLM5_LOD_ROUTE=pool` (page selection by the indexer's
pools; `mean`, `tok`, `idx` are the experimental routings), `STRATA_GLM5_LOD_NOSUM=1` (no page summaries:
windowed DSA, the default for the server), `STRATA_GLM5_LOD_TOP_POOLS` (prefill pools, 1024),
`STRATA_GLM5_PF_BUFS` (prefetch ring depth, 2), `STRATA_GLM5_NOPIPE=1` (serial windows instead of the two-card
pipeline), `STRATA_GLM5_NOQ2K=1` (generic MMQ instead of the WMMA Q2_K kernel).

## Known limits

- Tested only on Linux with ROCm 7 and gfx1100. The build for NVIDIA goes through the same CUDA-style sources
  but has not been run.
- No images: GLM-5.3-Flash's vision projector is not wired into this engine.
- `"parallel"` above 2 is not recommended: decode is memory-bound and the slots share it.
- The first message of a chat is read in full (1,000+ tokens/s with two cards); a follow-up continues from the
  conversation cache. A new conversation with a long shared prefix is read again.
- With one card every prompt window prefetches almost every expert over PCIe, so prompts read at half the
  two-card speed.

## How it works, in more detail

[docs/GLM5_NEXT.md](docs/GLM5_NEXT.md) is the engineering log: the model's structure as read from the
GGUF and llama.cpp, the decode path (mHC, KDA, DSA, the MoE and its doorbell), the expert tiers and what each
variant measured, the MTP draft layer, the prompt path (int8 MMQ, the prefetch ring, the Q2_K WMMA kernel,
4096-token windows, the two-card pipeline), the windowed attention mode with its NIAH results at every size
and routing, and the server fixes (stop tokens, thinking off, tool calls). The reference implementation it was
checked against is llama.cpp's `glm5-next`; `tools/glm_oracle/` dumps its intermediate tensors and
`ref/glm5.py` is a numpy re-implementation.

## Credits and license

[Strata](https://github.com/Niko1221/Strata) by Niko1221 is the base of everything here (the server, the app,
the expert tiers, the speculation, the HIP backend). [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
is by Z.ai. The 3-bit GSQ-RCO GGUF is by [pfeifferj](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF),
the MTP GGUF by [neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-MTP-GGUF). The engine uses
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) kernels (MMQ) and follows llama.cpp's `glm5-next`
implementation. Strata-GLM-AMD is open source under the [MIT License](LICENSE); the models have
their own licenses.
