**English** · [日本語](GLM5_NEXT.ja.md)

# Running GLM-5-Next (GLM-5.3-Flash) on Strata: differences and plan

Target file: `/home/client/Projects/llm/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf` (117.5 GB, `general.architecture = glm5-next`)
Reference implementation: `llama.cpp` (`src/models/glm5-next.cpp`, commit 0faee5004 / 2026-10-04)
Dev machine: Radeon Pro W7900 ×2 (gfx1100, 48 GB each), Ryzen 7 5700X3D (8 cores, AVX2 only), 125 GB RAM

This document has two parts: "what is different" (sections 1-4) and "how to build it" (sections 5-8). Every number was read
from the GGUF metadata, the llama.cpp sources and the Strata sources. Estimated values are marked "estimated".

---

## Progress (2026-10-04)

| Phase | State | Artifacts |
| --- | --- | --- |
| P0 oracle | **done** | `tools/glm_oracle/dump_tensors.cpp` (+`build.sh`): makes llama.cpp dump named intermediate tensors as .npy. `ref/glm5.py`: numpy reference implementation. Matches CPU-run llama.cpp over 4 greedy steps, per-layer cosine 0.9999 or better |
| P1 loader | skeleton | `src/glm5/model.cpp`: `HParams` from the GGUF metadata, dense tensors into VRAM in their GGUF types as they are (9.6 GiB, 3 s). Packed formats and the tokenizer not started |
| P2 kernels | single-token version done | `src/glm5/kernels.cu`: mHC (Sinkhorn), KDA (3 convs, vector-decay recurrence), DSA (k-pool, scores, MQA over latents, wk_b absorb / wv_b expand), sigmoid router, clamp SwiGLU. Added **Q2_K** to `native_mmvq` (`native_q2k_parity`: relative 5e-3 against ggml, bit-identical for multiple columns) |
| P3 decode end to end | **done (bring-up)** | `strata-glm5 --gguf … --tokens … --n-predict K --dump DIR`. Greedy tokens match llama.cpp (3 steps, "The capital of France is" → " Paris"), final-logits cosine 0.9991 |
| P4 expert residency | **working (static)** | `src/glm5/experts.cpp` `ExpertStore`: per-layer device assignment, resident up to the VRAM budget in profile order (id order if there is no profile), a miss is H2D'd from the mmap on the spot. `--profile-save` stores the routing counts. Layer split over 2 GPUs (`--devices 0,1 --layer-split auto`): each card holds the dense weights, the state and the experts of its own layers, and R (64 KB) is peer-copied at the boundary |
| P4b doorbell | **working** | The GPU publishes the MoE ids into mapped pinned memory, the host DMAs only the misses and raises a flag, and the GPU waits in a `moe_wait` kernel (the same shape as the doorbell in Strata proper). DSA's top-k is on the GPU too (radix select). Host syncs per token: 52 → 19 (the remainder are argmax and the device boundary) |
| P6 verification window + MTP | **working** | `forward_window(T<=8)`: the dense weights are read once per window, an expert once per (layer, expert) (`native_mmvq_grouped`: pointer table + column count, bit-identical to the single-column call). KDA keeps a per-token state snapshot so that `commit(n)` can roll back. The MTP (NextN) layer is placed on the head device with `attach_mtp`, and its 4.4 GB of experts stay resident. Speculation: verify in a window of k drafts + 1, token sequence identical to greedy |
| P7 server hookup | **working (2026-10-05)** | `strata-glm5 --serve`: the line protocol of `serve/server.py` (GEN/STOP/QUIT ↔ INFO/READY/PP/RESUME/T/DONE). For the tokenizer, `glm4` preprocessing (the CHATGLM4 regex + `ignore_merges`) was added to `tools/strata_tokenizer.py` and extracted from `--gguf` (matches llama-tokenize on 6 kinds of sentences). Config `strata-glm53.json`; responds through the Chat page and the OpenAI-compatible API. Conversation cache: continue if it is a continuation of the previous one; if it merely extends the previous prompt (when the thinking has dropped out of the history), roll back to the "mark" (the KDA state at the end of the prompt + the hidden state) |
| P5 prefill | **stage 1 (2026-10-05)** | The prompt window goes from 8 to up to 2048 tokens (`--prefill-window`, default 256, 1024 in the server config). A 1,064-token prompt: 54.6 s at window 8 (19 tok/s) → 19.9 s at window 256 (53) → **13.9 s at window 1024 (77 tok/s)**, and the generated token sequence is exactly identical to window 8. Implementation: `k_moe_group` rewritten as a counting sort (any T, grouped in ascending expert order, and the host numbers the misses in ascending order as well), grouped MMVQ loops over column chunks beyond 8 columns, dense GEMV runs 8 tokens at a time, KDA snapshots only for the first 8 tokens (a prompt window needs no rollback), logits only for the last token, the MTP catch-up is windowed too (`mtp_window`), staging is the per-layer non-resident count (≤288 slots), and no swapping inside a prompt window (every expert is needed, so it would only thrash). **The doorbell moved to a dedicated host thread** (even if the enqueuing thread blocks on a launch because the queue is full, a separate thread can still write the flag the GPU is waiting on, so there is no deadlock. This was mandatory at window 256 and above) |
| P8 | not started | graph capture, CPU computation of misses, prefill stage 2 (below) |

**Measuring and reworking the expert hierarchy (2026-10-04, late night)**

The per-round breakdown (`STRATA_GLM5_ROUNDS=1`, `STRATA_GLM5_DMA=1`) showed that almost all of the time of one round (window 4) is "waiting on the PCIe transfer of the missed experts":
50-200 misses (distinct) per window, 8.26 MB each, and serialized per layer (the routing of layer L+1 is not known until layer L finishes).

| Change | Result |
| --- | --- |
| Static residency only (profile) | 103-200 ms/round. Even when the generated text repeats the same phrase, the same experts miss every round |
| Adaptive swap v1 (a staging→slot D2D at the end of the window) | **wrong**: staging is shared across layers, so the data of every layer but the last is already overwritten → NaN. Withdrawn |
| swap-in on miss (DMA the miss straight into a victim slot; after the wait the device applies the placement with `k_moe_place`) | Correct. A re-miss of an evicted expert becomes a pread (the page cache is churned, so NVMe speed) and is slow |
| Eviction pool (on eviction, D2H into a pinned pool; a re-miss is one DMA) | pread gone. But the D2H rides the same copy engine, so the effective rate is 7-8 GB/s |
| Lookahead prefetch (apply the next layer's router to this layer's FFN input: 6.1 of the top-8 hit = 76%) | Works correctly but **is slower**: the mispredicted share plus the extra evictions increase the transfer volume, and since PCIe is the limiter it backfires. Off by default (`--lookahead N`) |
| Victim = prefer the ones that hold a pinned copy | D2H gone (14 GB/s), but the resident set freezes to the profile and the working set thrashes (more misses). Withdrawn |
| **True LRU + pinned-copy holders on ties + a 20 GiB pool per card** | **37.9 ms/token (26.4 tps)**, 2.8/3 accepted, 3.84 tokens/round, token sequence matches greedy |

Lesson: on this machine the floor for decode is **PCIe bandwidth (14 GB/s measured) × the number of misses**. Lookahead prefetch only helps when
there is spare bandwidth. The remaining options are (a) computing the misses on the CPU (the design of Strata proper: do not move 8 MB, return 16 KB. Q2_K on 8 AVX2 cores takes
about 0.3-0.5 ms per expert per token, in parallel with the GPU), (b) raising the residency rate (63% today: 67 GB over the two cards once dense 10 GB + MTP 4.4 GB + KV are excluded), (c) enlarging the window to dilute
the dense weight reads (a window-8 prompt is 42 tok/s). Hang mitigation: on an exception Engine/Model/ExpertStore do not touch the device
(`faulted_`, `std::uncaught_exceptions`), and main leaks deliberately. Under rocprofv3 the doorbell version deadlocks (reason unknown).

**Real traffic through the WebUI (2026-10-05)**: the benchmark (English document sentences, a profile from the same prompt) gave 26.4 tps / 93% accepted /
99.4% hits, but in actual chat (Japanese, English, code, with thinking; 16 prompts × 350 tokens) it is **7-23 tps**
(median 15), 56-85% accepted, 85-96% hits. The per-request statistics line (`glm5: … experts: …`) identified the difference:
(1) with a working set larger than the resident set (62%), the LRU swap-in on miss **thrashes almost completely** (9,000-14,000 swaps / 350 tokens,
97% thrash), and the D2H of the eviction plus the H2D of the re-fetch eat double the PCIe. (2) The eviction pool (20 GiB per card) wraps 2-3 times in a single request, and
the experts that overflow fall back to a pread from the GGUF (5-9 GB per request, 3-5 s). (3) The MTP acceptance rate is content-dependent (low for Japanese and code).
As countermeasures: an **admission filter** (`--admit N`, default 32 windows: only an expert that missed twice within the last N windows may take a resident slot. A one-off
expert stops at staging), a profile taken from real traffic (`--profile-save` saves per request while serving too), and
**exchange swap**: every expert's home is fixed to exactly one of "a VRAM slot or a pinned host blob", and a swap is
H2D of the incoming expert into staging → D2H of the victim into the incoming expert's host blob → D2D of staging into the slot (ordering guaranteed by using the same copy
stream). The eviction pool becomes unnecessary (`--evict-pool-gib` default 0), file reads after load are zero, and the pinned host side is only
the complement (36.9 GiB). Replaying the same 16 prompts:

| Configuration | Overall tok/s | Slowest request | Median round | swaps/request | preads/request | Host RAM |
| --- | --- | --- | --- | --- | --- | --- |
| LRU, always admit, document profile, 20 GiB pool per card | 10.9 | 58.4 s | 186 ms | 8,993 | 1,304 (11 GB, 6.6 s) | 87 GB |
| admit 32, chat profile, 20 GiB pool per card | 15.4 | 41.5 s | 140 ms | 4,279 | 1,066 (9 GB, 2.7 s) | 87 GB |
| **+ exchange swap, no pool** | **17.9** | **21.4 s** | ~140 ms | ~4,500 | **0** | **47 GB** |

**The prefill breakdown and the next move (2026-10-05)**: of the ≈ 12 s of one window at window 1024, PCIe (non-resident experts ≈ 42 GB per window, nearly constant regardless of the window size) is ≈ 3 s. The remaining ≈ 9 s is compute,
broken down as (a) the MMVQ that re-reads the 10 GB of dense Q8_0 weights every 8 tokens (1.3 TB of VRAM reads for 1024 tokens ≈ 1.5 s) → replace it with a hipBLASLt BF16 GEMM
(`src/prefill/gemm.cu` already exists), (b) DSA's per-token loop (1024 × 3 launches × 11 layers = 34k launches) → a kernel that batches T,
(c) KDA's sequential scan (1024 steps × 34 layers) → chunk parallelization, (d) the column-chunk re-reads of the experts. On top of that, in a prompt window, if we prefetch
**all** of the next layer's non-resident experts (a safe superset, since a large window uses nearly all of them), the 3 s of PCIe can be overlapped with compute (staging gets two faces). The unallocated
`d.probs`/`d.wts` found while debugging (an end-of-line comment had swallowed the allocation statement) is fixed.

**Where prefill stage 1 landed (2026-10-05, evening)**: 1,064 tokens in 8.2 s (130 tok/s), token sequence matches window 8. Breakdown of one window (1024): expert MMQ 2.5 s
(gate|up 1.06 + down 1.46; 5-10× slower than expected, needs rocprof), router+group 0.92 (including the sync that fetches the row count), KDA q/k/v GEMM+conv 0.77, DSA attend 0.59,
DSA output 0.42, DSA projection 0.36, KDA output 0.23, shared expert 0.18, KDA scan 0.18 (1.4 s → 0.18 s by keeping it in registers), gather 0.16. PCIe is 0 thanks to the lookahead prefetch.
Upstream (`original/Strata` 6f32ec0, #465 batch slots and others) has been merged in as the `upstream` remote (the only conflict was CMakeLists.txt).

**decode: computing the misses on the CPU (2026-10-05, night)**: applying the design of Strata proper to GLM. In a decode window (T ≤ 8), along with the routing we publish the MoE input (T × 4096 floats)
into mapped host memory, and the doorbell thread computes the missed experts on the pinned blob with ggml-cpu's vec_dot (Q2_K/Q3_K/Q4_K × Q8_K)
(8 threads, gate/up in 128-row chunks → clamp swiglu → down in 512-row chunks), then H2Ds the result (16 KB per pair) into the device's pair-row buffer and raises the flag.
The GPU computes gate/up/down for the hits and waits only immediately before the combine. The pair ordering is the same on host and device (hits ascending → misses ascending, token order within a group:
`k_moe_group` is deterministic for n ≤ 64). Result: 170-280 ms per round → **83-88 ms**, moe_wait 90-200 ms → 0, decode **14.6 → 28.4 tok/s**
(96 tokens, 74% accepted, 2.46 tokens/round). The token sequence is the same as with GPU computation. Remaining: moe_down 24 ms (the pair-row kernel was launching with the prompt window's 8,192 rows
→ fixed to the window size), moe_hits 18 ms, about 40 ms of small kernels (to be cut with graph capture).

**decode, continued (2026-10-05, late night)**: (1) the doorbell flag was being written with a host-to-host `hipMemcpyAsync`, which blocks the calling thread
until the copy stream drains (95% of the CPU service time was this). Changed to writing it from a single-thread kernel on the copy stream
(a launch onto another device's stream makes that device current first). prefill also went 128 → 138 tok/s. (2) Splitting the misses: each layer's misses are divided between
the CPU and PCIe by a cost estimate (CPU 0.25 ms per pair, PCIe 0.65 ms per expert, the work the GPU holds before each wait 0.4 / 1.0 ms), and the PCIe side goes
through the existing swap-in (with the admission filter) so the resident set adapts to the conversation. Two flags now (DMA arrival → the miss kernel, CPU result arrival →
combine). A 200-token decode: **25.2 tok/s** (94 ms/round); all-CPU with no swapping is 19.9 tok/s (the hit rate stays at 57%).
The remaining floor: against ≈ 2 ms of the GPU's own work per layer (attention + router + hits), servicing the misses (CPU 1.5 ms / PCIe 1.8 ms) sits serialized on the
publish → compute → combine chain. It disappears as the residency rate rises. Next: graph capture of the window (about 40 ms of small kernels per round),
the efficiency of the grouped MMVQ (moe_hits 12-18 ms = a quarter of the bandwidth), and the acceptance rate.

**prefill stage 2 (2026-10-06)**: 1,064 tokens 7.7 s → **3.46 s (308 tok/s)**; the GPU breakdown of one window (1064), 3.4 s: moe_wait (PCIe) 1.38 s,
expert MMQ 1.08 s, KDA scan 0.19, KDA gate 0.16, KDA qkv 0.14, DSA attention 0.07, other 0.4. What we did (all measurements with the standard
harness, `scratchpad/mmq_bench` and so on):
1. **hipBLASLt / rocBLAS FP16 GEMM caps out at 13 TFLOPS on this GPU** (ROCm 7.0, 1024×4096×4096; the bundled tuning table is for hipBLASLt
   100100 and does not match 100401). llama.cpp's int8 MMQ reaches **36 TMAC/s (72 TOPS)** on Q8_0 dense weights → all the dense products of wide windows now go through MMQ
   (a branch in `gemv()`, revertible with `STRATA_GLM5_NODENSEMMQ=1`). Note: MMQ's activations use a k-block-major layout in which the row count is the stride →
   the quantize row count and Product.total_rows must agree.
2. **MMQ's column tile is fixed per type**: 8 for Q2_K (the 32/96 tiles are 15-100× slower, 64/128 are half speed), 24 for Q3_K/Q4_K. Automatic selection (the row count of the largest
   expert) was picking 64-128. Added `Product.ncols_opt`.
3. The combine kernel now uses a per-token table (`tok_pairs`) (scanning all pairs was costing 1 s per window).
4. Made a **Q8_0 copy** of k_b/v_b (BF16) and turned absorb/expand into an MMQ with "head = expert" (0.6 s → 0.07 s).
5. A **prefetch ring** (4 per device, storing only the non-resident experts compacted in rank order; resident experts are read straight from their slot, a 2-product arrangement)
   removed the gather (0.16 s). For medium windows (9..255) the doorbell thread DMAs only the routed non-resident ones.
6. **In the MTP layer gate/up are Q4_K and down is Q5_K**, so the blob is not a multiple of the block size and MMQ (which addresses experts in block units) was
   misreading the second and later experts → the slot/ring pitch is rounded up to a multiple of lcm(block) (`LayerExperts::stride`).
7. **Dense DSA attention** (up to 8K cells): the latent cache is requantized into Q8_0 rows (and their transpose), the scores and P·V become two MMQ products at 64 tokens per batch,
   and the selection is a softmax with a bit mask. 669 → 66 ms. Revertible with `STRATA_GLM5_NODENSEATTN=1`.
8. The prompt is cut into **equal windows** (≤ 2048): since the transfer of the non-resident experts (42 GB) does not depend on the window size, a tail window of 1024+40 cost the same as 1024.
9. Bug: the bounds table was being async-copied from pinned host memory, racing with the host rewriting it ahead → write it from a device-side kernel (`set_pair`).
   The doorbell flag write had already been turned into a kernel for the same reason. The ring's events are per layer (per buffer, you end up waiting on your own record).
The floor today: PCIe. Against 42 GB of non-resident experts per window (21 GB per GPU, 14 GB/s = 1.5 s) there is 2.0 s of compute. Since the ring only covers 4 layers,
we can only run 0.4 s ahead while the other GPU computes → about 1.4 s is exposed. At window 2048 the compute (4 s) exceeds the transfer → it should be hidden (measurements below).
**Measurements (2026-10-06, `--prefill-window 2048`)**

| Prompt | Windows | Time | tok/s | GPU in-window | of which PCIe wait |
| --- | --- | --- | --- | --- | --- |
| 1,064 | 1 × 1064 | 3.46 s | 308 | 3.40 s | 1.38 s |
| 2,047 | 1 × 2047 | 3.80 s | **539** | 3.69 s | 0.13 s |
| 4,095 | 2 × 2048 | 7.66 s | 534 | 3.70 + 3.83 s | 0.12 s |

At a window of 2048 tokens the transfer hides behind the compute (3.7 s), and the expert MMQ (1.84 s: Q2_K at 14 TMAC/s dense, 9 with scattered routing) is the largest term.
The rest: KDA scan 0.37 (sequential), KDA qkv 0.27, mHC 0.17, DSA attention 0.17, KDA gate 0.16 (small GEMMs and an l2 norm).
Debugging: `STRATA_GLM5_SYNC=1` also places sync points inside the MTP window and inside DSA, with NaN checks.

**Parallelism (batch slots, 2026-10-06)**: the same in-process slot scheme as upstream #465. `--slots N` (config `"parallel": N`) reserves N conversation states
(KDA state + DSA cache + position + mark) separately from the solo one, and `forward_rows()` decodes each row = one token of a different slot in a single window
(per-row kernels: conv/scan advance the row's slot state by one step, cache rows scatter to the slot's position, pool/score/attention use a per-row base).
`copy_slot()` moves a conversation between slots (a conversation started in solo is handed off with STOP → BGEN; the next turn of a conversation held by another slot). serve
implements `BGEN/BADM/BT/BDONE/BSTOP`, `INFO batch_slots=N slot_cache=1`. Identity: `--slots-test K` decodes K conversations solo and batched and compares them;
with GPU-only miss computation (`STRATA_GLM5_CPU=0`), 4 × 48 tokens give **0 mismatches** (with CPU-computed misses, ggml's rounding changes the ties: upstream has the same note).
Fixed along the way: the pair-combine order now follows the router's top-k order (with hits-first ordering the result depends on the residency state), the solo pool range was off by one
pool (it could not see as far as the length the indexer actually selects over), the ring's events became two faces per window (a race waiting on your own record in medium windows),
`serve_ring` recomputes the non-resident rank every time (after a swap it was stuck on a stale rank), and exceptions in the doorbell thread propagate to the main body.
The DSA cache moved to f16 (one slot 1.46 → 0.89 GiB @32K; the same as llama.cpp's default KV type). A VRAM breakdown display, `glm5 vram:`.
Performance: a batch window is dominated by servicing the misses (CPU+PCIe), a 4-row window ≈ 125-170 ms. Total throughput is 1.0-1.2× solo (with MTP),
and the benefit is mainly latency (everyone advances at once). Next: MTP drafts in slots as well (row count 4 × 2 ≤ 8), and the residency rate.

**MTP drafts in slots (2026-10-06)**: `forward_rows()` takes a "run" of consecutive rows for each slot (rows = cur + drafts).
KDA uses a grouped kernel that updates the state sequentially per run (`k_kda_conv_grp`/`k_kda_step_grp`: the convolution taps come from an earlier row of the run or from the state,
with a per-row snapshot), and `commit_rows(n_keep[])` rolls the state back to the accepted length (the DSA cache needs none, since it is overwritten by position).
The MTP layer's cache is also held for every slot (`mtp_window_slot/mtp_step_slot`; `copy_slot` copies the MTP layer too). A decode window is fixed at 8 rows (snapshots,
staging, row tables), so with S concurrent slots each slot gets `min(spec, (8-S)/S)` drafts (4-way → 1, 2-way → 2).
`--slots-test K` compares both without and with drafts against solo greedy: **0 mismatches** at 4 × 48 (CPU=0).

Performance (4 conversations, synthetic prompts, hit rate 73-80%): 4-row window 125 ms → 8-row window (1+1 drafts) 188-200 ms, 79-85% accepted, total 31.8 → 33-36.6 tok/s
(+10%). Window time breakdown (`STRATA_GLM5_PROF=1`): `moe_wait` 43 ms + the CPU row wait inside `moe_down` ≈ 15 ms = **about 60 ms is waiting on miss service**,
and most of the GPU compute (hits 21 + down 22 + other ≈ 65 ms) is in its shadow.

**Why the service is slow = host RAM bandwidth**: an expert blob is 9.2 MB, and a 4-row window has 335 non-resident ones (= 3.1 GB). CPU computation is 0.30 ms per pair
= 30 GB/s, unchanged at 8/12/16 threads (`STRATA_GLM5_CPU_THREADS`), and shifting work from DMA to the CPU (`GPU_BEFORE_CPU_MS=0.5`) is slower instead.
Host RAM reads measure **42 GB/s** (saturating at 4 threads, dual-channel DDR4), and CPU 30 + PCIe DMA ≈ 10 pins right against it.
Solo is the same: 4 misses per layer × 9.2 MB = 37 MB → 0.9 ms + GPU 0.7 ms ≈ 1.7 ms per layer × 45 = 78 ms/round (exactly as measured).
With bandwidth as the limiter, lookahead prefetch (`--lookahead`) makes the mispredicted share a pure loss (and on the slot path it busy-waits on an unpublished prediction and hangs: not used).
→ The remaining lever is "host RAM transfer per token": the residency rate. Cutting the prefetch ring from 4 to 2 (`STRATA_GLM5_PF_BUFS=2`, 2.8 GiB per unit) and raising
the budget to 30.8/28.8 (+669 experts) gives a 4-row window of 109 ms and an 8-row window of 174 ms = **38 tok/s total**, and solo 56 → 53 ms/token.

**Holding the DSA cache in Q8_0 (2026-10-06)**: each row of the latent KV (512), the indexer key (128) and the key-gate (128) is stored in ggml Q8_0 blocks
(32 values = an f16 scale + 32 int8 = 34 B) (`q8_block`/`q8_at`, `store_q8_rows`/`scatter_q8_rows`). Per token per layer,
1536 B (f16) → 816 B. The decode kernels (score / ctx / pool, row mode included) recover scale × int8 from the block on the spot.
Since the dense attention of a prompt window had already been converting the cache to Q8_0 and feeding it to MMQ, **the cache itself is now an MMQ operand**
(one conversion disappears; the transposed side `q8_0_rows_T` reads Q8_0 and requantizes). To cover MMQ's tile overread the cache is max_ctx + 512 rows.
One slot's state @32K: 0.89 → 0.47 GiB, freeing 2.1 GiB over 5 slots → into the expert budget. Accuracy: the same rounding as llama.cpp's `-ctk q8_0`.
## Windowed sparse attention — `--attn lod` (2026-10-06)

An experimental attention mode for the DSA layers of GLM-5.3-Flash, on the post-absorb MLA (64 query heads sharing a 512-dimensional latent c):
every query reads a fixed set of exact cells (a sink page, the pages it selects, the local window), so the attention cost of a window does not grow
with the context. **The k-pool / indexer (idx_q_b, idx_k, idx_gate, idx_proj, pool, score, select) is not needed by the basic mode.**
This is a different attention from the DSA at training time (the indexer's top-2048), so accuracy must be evaluated per model.

- **Pages** of 64 tokens with per-page statistics (`pmean_`, Q8_0 rows) taken at the point the page completes (`k_lod_page_mean[_rows]`).
  Being a pure function of the cache rows, they hold no state across rollback (mark / commit_rows), and are recomputed once the position rolls back.
- **Selection** (`k_lod_select`, per token): candidates = the completed pages before the local window (512). sink (page 0) + the top K of the candidates
  + all cells of the local window + the partial page go into `sel` as exact cells.
- **prompt window** (T ≥ 9): the selection is shared across the window (one score per page from the window's queries, by MMQ and `colmax`), and
  [selection | the window's own tokens] are gathered into a compressed Q8_0 operand (`k_lod_gather`) → the existing MMQ dense attention (causal mask `k_lod_mask`,
  `masked_softmax` with a column bias). The column count is ≈ K·64 + 576 + T, so it is almost independent of context length, and there is no 8192 fallback either.
- Parameters: `--attn lod`, `--lod-page 64`, `--lod-top 32` (prefill), `--lod-top-decode 32`, `--lod-local 512`, `--lod-sink 1`
  (environment variables `STRATA_GLM5_LOD*`). Debug: `STRATA_GLM5_LOD_SYNC=1` (stage-wise sync on the windowed path only; `STRATA_GLM5_SYNC=1` clashes with the doorbell).
- Correctness check: at contexts ≤ 2048 DSA is dense attention too, so the tokens of both modes are identical: 1064 tokens + 24 generated gives **24/24 identical**,
  `--slots-test 4` 0 mismatches.
- Results (stage12, ring 2, budget 30.8/28.8):

  | Prompt | DSA prefill | windowed prefill | Generation (32 tokens, greedy) |
  |---|---|---|---|
  | 4095 | 7.71 s (531 tok/s) | 7.57 s (541 tok/s) | 31/32 identical (「グラフィックメモリ」vs「グラフィックスメモリ」) |
  | 12285 | 35.7 s (344 tok/s) | **23.4 s (525 tok/s)** | DSA starts its think in English, the windowed mode outputs the same Japanese summary as at 4K (both valid; which is better has not been evaluated) |

  The window time at 12K is flat at 3.65–3.98 s (DSA takes 9.1 s past 8K). attention saturates at 0.43 s/window, and the page-score MMQ+colmax is 0.3 s/window.
  At 12K the accepted drafts are 18/26 (same as DSA). Quality must be evaluated with NIAH / long-context QA (selection accuracy on long contexts is the open question).
- top sweep (12285 tokens, with the attention buffer cap raised to 16384 cells: `STRATA_GLM5_ATTN_MAXKV`, default 16384, +200 MB/card):

  | Setting (top / top-decode) | prefill | attention/window | decode | 32 generated tokens vs DSA |
  |---|---|---|---|---|
  | 32 / 32 | 23.4 s (525 tok/s) | 0.43 s | 22.7 tok/s | A different response (skips the think and gives a Japanese summary) |
  | 128 / 32 | 25.3 s (486 tok/s) | 1.0 s | 24.3 tok/s | The first 13 tokens identical, close paraphrases after that |
  | 128 / 128 | 25.4 s (484 tok/s) | 1.1 s | 17.9 tok/s | **32/32 identical** |

  decode was dropping in proportion to the number of exact rows (128 pages = 8192 + local ≈ 9K rows/token) (a per-token kernel).
- **Fused decode kernel** (`k_lod_attn_split` / `k_lod_attn_merge`, a split-K kernel with an online softmax over the selection list, on Strata's row structure;
  being MLA, K and V are the same Q8_0 rows and are read once; 2 heads/wave tile, 8 tiles/block, `STRATA_GLM5_LOD_SPLIT` default 16, used for windows with T ≤ 8):
  decode at 12K top 128/128 goes **17.9 → 25.8 tok/s** (DSA 24.8), and the generation stays 32/32 identical to DSA; `--slots-test 4` 0 mismatches, and 1064+24 is identical to DSA too.
- Pitfalls: when queueing `cudaMemcpyAsync` on the stream of the second device, make that device current (same as the existing attention);
  the row kernels also need the slot table of the MTP layer (the DSA mode used the cache directly in MTP, so there was no table).

Conversation continuation (4 turns): from the second turn on, roll back to the "mark" and reuse the previous prompt's work (cached_tokens 24/37/66),
24–29 tok/s. The remaining floor is 110–150 ms per round = 40–70 misses in one round × 0.6 ms (PCIe) + the dense weights.

**MTP acceptance rate (2026-10-04, a 283-token prompt + 24 tokens)**: 6 rounds, 17 of 18 drafts accepted (94.4%),
**4.00 tokens/round**. The token sequence is exactly identical to greedy (the speculation is exact). The NextN formula transcribed from vLLM is correct.
In the first implementation decode was 41.8 ms/token and one MTP step was 25 ms. Templating the column count of the grouped kernel, skipping the sync for all resident layers,
and parallelizing the MoE grouping brought MTP to about 2 ms/step. The identity of the per-window time was the table above (expert miss transfers).

**Measurements (2026-10-04, all token sequences identical to llama.cpp)**

| Configuration | Resident experts | Hit rate | decode ms/token | Load |
| --- | --- | --- | --- | --- |
| 1 card, no residency (from mmap every token) | 0 | 0% | 378 | |
| 1 card, 33 GiB, id order | 3,996 (33%) | 33.7% | 220 | 9 s |
| 1 card, 33 GiB, profile from the same prompt (upper bound) | 3,866 | 82.6% | 182 | 9 s |
| 1 card, 33 GiB, profile from a different prompt | 3,866 | 36.8% | 211 | 9 s |
| **2 cards**, 36+36 GiB, profile from the same prompt (upper bound) | 8,526 (70%) | 99.4% | **103–135** | 20 s |
| **2 cards**, 36+36 GiB, profile from a different prompt | 8,526 (70%) | 74.8% | **106** | 20 s |
| 2 cards, as above, **Sinkhorn parallelization + 8 experts in one launch** (described below) | 8,526 (70%) | 99.2% | **47–57** | 20 s |
| 2 cards, as above, with the miss source in RAM pinned per layer (28 GiB) | 8,526 (70%) | 99.0% | **35–40** | 107 s |
| 2 cards, as above, doorbell (no host sync) | 8,526 (70%) | 99.0% | **35–43** | 94 s |
| 2 cards, profile collected from 34 prompts (`--prompts`) | 8,566 (71%) | 89.8% / 76.5% | 67–92 / 53–77 | 94 s |

**Profile of the doorbell version (rocprofv3, 2 cards, 99% hit)**: kernel execution for one token is GPU0 17.6 ms + GPU1 15.4 ms = 33 ms, and
of the 42 ms wall time, the 21.8 ms GPU1 spends waiting on GPU0 is exactly the serialization of the layer split. The top items of the breakdown are the Q8_0 dense GEMV 11.3 ms
(reading 9.6 GiB every token = the lower bound of the W7900's bandwidth), `moe_wait` 7.4 ms (PCIe DMA of 0.6 ms per 8.26 MB miss × about 7/token),
the batched expert kernel 6.3 ms (bandwidth lower bound 3.3 ms), and 4–5 ms/GPU of launch gaps (2,486 launches/token).
The floor on this hardware is roughly 11 + 3.3 + a few ms ≈ 16–18 ms/token (55–60 tps). The remaining means are (1) erasing the launch gaps with graph capture,
(2) keeping the hit rate at 99% with adaptive swapping (76–90% on prompts outside the profile, where the 0.6 ms per miss matters),
(3) sharing the dense weight reads across multiple tokens in a verification window (speculation). PCIe is Gen4 x16 on both cards, measured at 14 GB/s.

**The first profile (2 cards, at the 137 ms/token point, rocprofv3)**: kernel execution totals only 44 ms/token; the rest is on the host side
(4,197 kernel launches/token = 20 ms in the launch API alone, and the GPU queue drains at each of the 52 host syncs/token).
The top items on the kernel side are the Q8_0 dense GEMV 11.3 ms, the single-thread serial Sinkhorn 9.8 ms, and the Q2_K expert 6.3 ms.
First round of countermeasures: Sinkhorn on 1 warp (16 lanes + shuffle), and `native_mmvq_batched`, which merges the gate/up of 8 experts into one launch and down into one launch
(pointer table, bit-identical to the individual calls) → 103–135 ms became 47–57 ms.
Note: on HIP, `hipStreamSynchronize` after an illegal access has occurred on the GPU never returns (it looks like an "infinite loop" at 100% CPU),
so on an exception we `_Exit` without syncing.

A profile built from a single 283-token run does not transfer to another prompt (36.8%). With a profile collected from 34 runs (English documents, Japanese, code, chat)
it is 76–90%. Host pinning is fine with a per-layer `hipHostMalloc` (up to 4 GiB, about 0.7 s/GiB), while
a single 28 GiB call did not return for over 10 minutes on ROCm 7. Pinning fills via `pread`, so the page cache is not polluted.

**Measurement (1x W7900, bring-up configuration)**: 0.69–0.75 s/token. The experts are carried into VRAM from mmap by a pageable copy every token (8 × 8.26 MB × 42 layers = 2.8 GB/token), and the DSA top-k is taken on the host. Speed comes later.

**How to read the matches**: the per-layer difference comes from activation quantization (llama.cpp CPU is Q8_K/Q8_0, this engine and llama.cpp CUDA are Q8_1), and in
deep layers experts with close scores swap places (2 of the top-8 differ at layers 21 and 40), so the cosine of `ffn_moe_out` drops to 0.97,
but either selection is correct. This band is the same as in the Unsloth UD-Q4_K_XL verification (docs/UNSLOTH_Q4.md).

Reproduction steps:

```sh
# llama.cpp (HIP) and dump_tensors
cd ../llama.cpp && cmake -S . -B build-hip -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF && cmake --build build-hip --target llama-cli -j14 && cd - && tools/glm_oracle/build.sh
../llama.cpp/build-hip/bin/dump_tensors -m GLM.gguf -o oracle -p "The capital of France is" -ngl 0 --ubatch 1 \
  --n-predict 2 --filter '^l_out-' --filter '^result_output'
# The HIP build of this repo and the bring-up engine
cmake -S . -B build-hip -G Ninja -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_GGML_DIR=<llama.cpp checkout> && cmake --build build-hip --target strata-glm5 native_q2k_parity -j14
build-hip/strata-glm5 --gguf GLM.gguf --tokens $(paste -sd, oracle/tokens.txt) --n-predict 2 --dump engine
# 2 GPU, 36 GiB of experts each, routing counts saved for the next start's profile
build-hip/strata-glm5 --gguf GLM.gguf --tokens ... --devices 0,1 --layer-split auto --expert-budget-gib 36 --profile-save prof.txt
python3 tools/glm_oracle/compare_dumps.py oracle engine --step 0 --token 4        # layer outputs
python3 tools/glm_oracle/compare_dumps.py oracle engine --step 0 --token 4 --layer 3 --all
PYTHONPATH=../llama.cpp/gguf-py python ref/glm5.py --gguf GLM.gguf --compare oracle --greedy 2   # numpy reference
```

---

## 1. Model comparison

| | Qwen3.8-Flash-Next (Strata today) | GLM-5.3-Flash (`glm5-next`) |
| --- | --- | --- |
| Total parameters / active | 125B / 6B | 320B / 18B (`LLM_TYPE_320B_A18B`) |
| Layer count | 48 | **45** |
| Hidden dimension `n_embd` | 2,560 | **4,096** |
| Vocabulary | 248,320 | **154,880** (both `output` and `token_embd` are Q8_0, 0.67 GB each) |
| Linear layers (recurrence) | Gated DeltaNet ×36 (every 3rd layer is QSA) | **Kimi Delta Attention (KDA) ×34** |
| Full attention layers | QSA ×12 (layers 3,7,…,47) | **DSA (nope MLA + k-pool indexer) ×11** (layers 3,7,…,43) |
| Positional encoding | M-RoPE 64 dimensions, base 1e7 | **None** (`rope.dimension_count = 0`, nope-only MLA). Positional information comes only from KDA's conv and recurrence |
| Residual | Low-rank hyper-connection (hc=4, lr=320, element-wise sigmoid gate) | **mHC** (hc=4, pre/post/comb from 24 mixes, comb uses **20 Sinkhorn iterations**) |
| Dense FFN layers | None | **3 layers, 0 through 2** (ff 12,288, Q8_0, 160 MB per layer) |
| MoE layers | 48 layers × 512 experts, top-10, softmax router | **42 layers × 288 experts, top-8**, **sigmoid router + selection bias `exp_probs_b`**, weights normalized then **×2.5** |
| Expert shape | 2560 → 640 → 2560 | **4096 → 2048 → 4096** (1 expert = 25.2M parameters) |
| Total experts | 24,576 | **12,096** |
| Shared expert | 1, with scalar gate `ffn_gate_inp_shexp` | 1, **no gate** (plain addition) |
| SwiGLU | silu(g)·u | **with clamp**: `silu(min(g,10)) · clamp(u,−10,10)` (limit 10 for both expert and shared expert) |
| PLE (n-gram embedding, 28.8 GB on SSD) | Present (layer 1) | **None** |
| MTP draft layer | Present (obtained separately from BF16) | **No nextn tensors** in this GGUF. `DECODER_MTP` is also unimplemented on the llama.cpp side |
| Context | 262,144 | **1,048,576** |
| Tokenizer | gpt2 BPE, pre `qwen35` | gpt2 BPE, pre **`glm4`** (in llama.cpp, the `CHATGLM4` regex + `ignore_merges = true`) |
| Chat template | Qwen (`<think>`, `<tool_call>` XML) | starts with `[gMASK]<sop>`, `reasoning_effort` is low/high/max |

### 1.1 GGUF contents (quantization and size)

| Tensor group | Type | Size |
| --- | --- | --- |
| routed experts (gate/up/down), 36 layers | **Q2_K** | 85.6 GB (8.26 MB per expert) |
| routed experts, 3 layers | Q4_K | 12.2 GB (14.2 MB per expert) |
| routed experts, 3 layers | Q3_K | 9.3 GB (10.8 MB per expert) |
| **experts total** | | **107.2 GB** |
| KDA q/k/v projections (`attn_q/k/v`, 4096→8192) | Q8_0 | 3.6 GB |
| `attn_output` (8192→4096, 45 layers) | Q8_0 | 2.0 GB |
| shared expert (42 layers) | Q8_0 | 1.1 GB |
| dense FFN (3 layers) | Q8_0 | 0.48 GB |
| MLA projections (`attn_q_a` Q8_0, `attn_q_b` Q8_0, `attn_kv_a_mqa` BF16, `attn_k_b`/`attn_v_b` BF16) | | 0.78 GB |
| indexer (`indexer.attn_q_b` etc., F32) | F32 | 0.33 GB |
| KDA gate projections (`ssm_f_a/f_b/g_a/g_b/beta`, BF16), conv, norm | | 0.25 GB |
| mHC (`hc_attn_fn`, `hc_ffn_fn` 16384×24, F32) | F32 | 0.14 GB |
| router (`ffn_gate_inp` F32) | F32 | 0.20 GB |
| `output` + `token_embd` | Q8_0 | 1.35 GB |
| **non-expert total** | | **10.3 GB** |

Strata's existing GPU expert kernel (`native_mmvq`) has Q3_K/Q4_K/Q5_K/Q6_K but **does not have Q2_K**.
On the CPU side, ggml-cpu's `vec_dot` covers all types (although the AVX2 multi-token kernel `kq_avx2` only has Q4_K/Q5_1/Q8_0).
The prefill MMQ also needs Q2_K/Q3_K instances added (ggml's `mmq.cuh` has them).

### 1.2 Weight bytes read per token

| | Qwen (Q2_0) | GLM (3.0bit) |
| --- | --- | --- |
| routed experts | 10 × 48 × 1.38 MB = **0.66 GB** | 8 × 42 × 8.26 MB (Q2_K layers) ≈ **2.8 GB** |
| dense part (every token) | ≈ 3.5 GB | ≈ 10 GB |

GLM reads more than 4× as much expert data as Qwen. Using Strata's design as is (experts that don't fit on
the GPU are computed on the CPU) would mean the 8-core AVX2 CPU computing hundreds of MB of Q2_K per token,
and that becomes the biggest bottleneck. On the development machine, the right allocation is to
**make 70-80% of the experts resident on the two W7900s (96 GB) and carry the misses to the GPU over PCIe DMA to compute them there**
(estimated; section 4).

---

## 2. The computation of one layer (from llama.cpp `glm5-next.cpp`)

In every layer, the attention half and the FFN half are each wrapped in mHC.

```
R: [n_embd=4096, hc=4]  (4 residual streams)

--- attention half ---
x, post, comb = hc_pre(R, hc_attn_fn, hc_attn_scale, hc_attn_base)
x = rms_norm(x) * attn_norm
y = KDA(x)  or  DSA(x)
R = hc_post(y, R, post, comb)

--- FFN half ---
x, post, comb = hc_pre(R, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
x = rms_norm(x) * ffn_norm
y = dense_ffn(x)  (layers 0..2)   or   moe(x) + shared_expert(x)
R = hc_post(y, R, post, comb)

--- output ---
h = mean over 4 streams of R
logits = output @ rms_norm(h) * output_norm
```

### 2.1 mHC (DeepSeek V4 style hyper-connection)

The definition of `ggml_dsv4_hc_comb / hc_pre / hc_post` (`ggml/include/ggml.h:2680`):

```
flat  = R laid out as 16384 dimensions, then rms_norm(eps=1e-6)   ← RMSNorm without weights
mixes = hc_fn @ flat                                       ([24])   F32 16384×24 matrix
pre [h]      = sigmoid(mixes[h]      * scale[0] + base[h])      + eps      (h = 0..3)
post[h]      = sigmoid(mixes[4+h]    * scale[1] + base[4+h])    * 2
comb[d,s]    = mixes[8 + d + 4s] * scale[2] + base[8 + d + 4s]  → softmax over d, +eps,
               column normalize, (row normalize, column normalize) × 19   ← 20 Sinkhorn iterations
x[i]         = Σ_h R[i,h] * pre[h]                          (hc_pre)
R'[i,d]      = y[i] * post[d] + Σ_s R[i,s] * comb[d,s]      (hc_post)
```

The computation is completely different from Qwen's GR (`gr_read`/`gr_write`: a low-rank 320 bottleneck,
an element-wise sigmoid gate, inject). Strata's `fused_gr.cu` cannot be reused; this becomes
**two new fused kernels** (hc_pre+norm, hc_post). The amount of computation is small
(a 16384×24 GEMV plus a 4×4 Sinkhorn).

### 2.2 KDA (Kimi Delta Attention) layers ×34

```
q = silu(conv1d_q(attn_q @ x))     k = silu(conv1d_k(attn_k @ x))     v = silu(conv1d_v(attn_v @ x))
    * q/k/v have separate 8192-ch projections + separate convs (Qwen has one conv over 10240 ch)
    64 heads × 128 dim, h_k == h_v == 64 (no GQA)
q, k = l2_norm(q), l2_norm(k)      (eps 1e-6)
g1 = f_b @ (f_a @ x) + dt_bias                       ([8192] = 64 heads × 128)
g1 = gate_lower_bound * sigmoid(-(g1 * A))           (lower_bound = -5, A = ssm_a[h])   ← element-wise decay
beta = sigmoid(beta_w @ x)                            ([64])
recurrence (the kda branch of ggml_gated_delta_net, ggml-cpu/ops.cpp:11123):
    S[i,:] *= exp(g1[i])            ← Qwen uses a per-head scalar, KDA uses a vector over i
    d[j]   = (v[j] - Σ_i S[i,j] k[i]) * beta
    S[i,j] += k[i] d[j]
    o[j]   = Σ_i S[i,j] q[i] / sqrt(128)
g2 = g_b @ (g_a @ x)                                 ([8192])
y  = rms_norm(o, ssm_norm) * sigmoid(g2)
out = attn_output @ y                                 (8192 → 4096)
```

State: 128×128×64 = 1M float (4 MB per layer, 142 MB over 34 layers), conv state 3 × 3 × 8192.
Differences from Strata's `gdn_step`: (1) the decay is a vector, (2) three convs, (3) the gate formula,
(4) no GQA. The parallelization in `gdn.hpp` where "one thread owns a (j,h) column" can be used as is.
A **new kernel `kda_step` (single token / a multi version for the verification window) and a chunked
recurrence for prefill** are needed.

### 2.3 DSA (DeepSeek Sparse Attention, nope MLA) layers ×11

```
qr  = rms_norm(attn_q_a @ x) * attn_q_a_norm        ([1536])
q   = attn_q_b @ qr                                  ([64 heads × 256])
kvc = rms_norm(attn_kv_a_mqa @ x) * attn_kv_a_norm   ([512])  ← this is the entire KV cache (512 values per token)
q_abs[h] = attn_k_b[h] @ q[h]                        ([64 heads × 512])   wk_b absorbed into q

--- indexer (k-pool) ---
iq  = indexer.attn_q_b @ qr                          ([32 heads × 128])
ik  = layer_norm(indexer.attn_k @ x) (weight+bias)   ([128])
ig  = indexer_compressor_gate @ x                    ([128])
w   = (indexer.proj @ x) / sqrt(128*32)              ([32])
cache row = 3×128 of [ik | ig | pooled]
one pool per 4 tokens completed:
    pooled[d] = Σ_{j<4} softmax_j(ig_j[d] + ape[d,j]) * ik_j[d]      ← a softmax over the 4 positions, per dimension
score[pool] = Σ_h relu(iq_h · pooled) * w_h
select the top 512 pools (= 2048 cells) + the incomplete tail (at most 3 cells)
--- attention (MQA, 64 query heads look at the same 512-dimensional latent) ---
p   = softmax(q_abs[h] · kvc[selected] / sqrt(256))
o[h]= attn_v_b[h] @ (Σ p * kvc[selected])            (512 → 256)
out = attn_output @ concat(o)                         (64×256 = 16384 → 4096)
```

Correspondence with Qwen's QSA: the skeleton of "score blocks with compression ratio 4 using an indexer and
select the top 2048 cells" is the same. The design of Strata's `qsa_select` (block scores → top-k) and
`kv_gather` can be carried over as is. What differs is
(1) the pooling formula (QSA: pooling when a block completes / DSA: a gated softmax pool), (2) K and V are the same
512-dimensional latent with no RoPE (the cache is small: 11 layers × 512 × fp16 = 11 KB per token, 2.9 GB at 262K),
(3) the per-head `wk_b` absorption and `wv_b` expansion, (4) the score is a relu sum over 32 heads.

### 2.4 MoE router (`llama-graph.cpp:2011` `build_moe_ffn`)

```
logits = ffn_gate_inp @ x                   (F32 weights)
probs  = sigmoid(logits)
sel    = probs + exp_probs_b                ← selection includes the bias, the weights do not
ids    = top-8 of sel
w      = probs[ids];  w /= max(Σw, 2^-14);  w *= 2.5
out    = Σ w_e * down_e( silu(min(gate_e,10)) * clamp(up_e,-10,10) )  +  shared(x)
```

Strata's `router_top10` (softmax → top-k → normalize) needs a **sigmoid+bias version** added.
**clamp** has to go into the expert kernels (the GPU `native_mmvq` gate/up fusion and the CPU `native_gu_rows`).

---

## 3. What Strata can reuse / what is newly needed

| Area | Reuse | New / rework |
| --- | --- | --- |
| GGUF loading (`gguf_reader`, `gguf_split`) | ○ as is | reading the glm5-next keys |
| Pack format (`index.txt`, `native_experts.txt`, `experts.bin`) | ○ the format is generic (per-layer variable blob, `n_expert` in the header) | glm5-next's layer configuration in `tools/iq_pack.py` (3 dense layers, 288 experts, Q2_K/Q3_K/Q4_K blobs) |
| `ModelGeometry` (`layout.hpp`) | × the numbers are hardcoded | **build the model description from GGUF metadata** and hold both the Qwen and GLM lineages |
| `LayerType {GDN, QSA}` (`graph.hpp`) | extend | `{GDN, QSA, KDA, DSA}` + dense/MoE flags |
| Residual `gr.cu` / `fused_gr.cu` | × | **`mhc.cu`**: hc_pre(+rms_norm) and hc_post(+Sinkhorn), single-token/multi-token |
| `gdn.cu` / `fused_gdn.cu` / prefill `gdn_conv`/`gdn_recurrence` | design only | **`kda.cu`** (vector decay, 3 convs), chunked recurrence for prefill |
| QSA (`qsa_select`, `kv_gather`, `qsa_decode_attn`, `qsa_prompt_attn`) | the selection and gather skeleton | **`dsa_*`**: k-pool pooling, 32-head relu scoring, MQA over the latent (512), `wk_b` absorption / `wv_b` expansion |
| KV cache (fp16/int8/q4, streaming) | ○ the page table and streaming carry over | fit it to the single-latent shape (K=V, no RoPE) |
| Router `router_top10` | extend | sigmoid + bias + normalize + scale |
| Shared expert `shared_expert` | extend | a version without the scalar gate + clamp |
| GPU expert kernel `native_mmvq` (Q3_K/Q4_K) | ○ | **add Q2_K** (transcribed from ggml-cuda vecdotq), clamp |
| Multi-token expert (`native_expert_grouped`) | ○ (add the type) | Q2_K |
| CPU expert (`native_expert.cpp`, ggml-cpu) | ○ | AVX2 multi-token kernels for Q2_K/Q3_K (optional, performance) |
| Expert tiers (`expert_source`, `expert_cache`, pinned arena, profile, adaptive swap, peer, layer split) | ○ almost as is (n_layers×n_expert is generic) | make it possible to skip the dense FFN layers (no experts) |
| prefill (the chunking/streaming/MMQ of `prefill.cpp`) | ○ the skeleton | swap in GLM's layer computation, Q2_K/Q3_K for MMQ, multi-token versions of mHC/KDA/DSA |
| Verification window `verify.cpp` / `verify_kernels.cu` | ○ the skeleton | multi versions of the new kernels |
| MTP `mtp.cpp` | × (the tensors do not exist) | for now `--spec 0` + prompt lookup (`suffix_drafter`). Fetch nextn from HF later |
| PLE `ple*`, `ngram` | not needed | disabled for GLM |
| Server `serve/server.py` | ○ | the template (`[gMASK]<sop>`, the reasoning_effort values), the tool format of `OutputParser` |
| Tokenizer `tools/strata_tokenizer.py` | ○ the skeleton | the `glm4` regex + `ignore_merges` |
| `setup.py` | ○ | add the GLM family (deferred) |

---

## 4. Memory allocation on the development machine (estimated)

| Location | Contents | Size |
| --- | --- | --- |
| VRAM (2 × 48 GB) | dense weights 10.3 GB (with layer split, only each card's own layers) | ≈ 10 GB |
| | KV latent + indexer cache (262K, fp16) | ≈ 4 GB |
| | KDA state, work buffers, the prefill borrow | ≈ 4-8 GB |
| | **expert cache** | **≈ 70-75 GB → 8,500-9,000 of the 12,096 (about 70%)** |
| RAM (125 GB) | pinning all experts (107 GB) **does not fit together with everything else** | → low-RAM mode (`--resident-experts`: keep only the 30% not on the GPU, ≈ 32 GB, resident in RAM) |
| SSD | the GGUF itself (mmap) | 117 GB |

The hit rate should be 70% or better with a profile, and higher still with adaptive swap. A missed expert
(8.26 MB) takes 0.3 ms over PCIe 4.0 x16 (≈ 26 GB/s). Assuming 20-30% of the 8×42 = 336 experts per token
miss, that is ≈ 80 experts, 0.66 GB by DMA → about 25 ms. Since DMA is faster than computing the same amount
on the CPU (8-core AVX2, Q2_K), I expect that **for GLM the default becomes `--pcie-frac` near 1.0 (all misses
computed on the GPU)**. Either way, decide after measuring.

---

## 5. Implementation plan (phases)

Each phase's acceptance condition is "the same numbers come out for the same input as llama.cpp". As is
Strata's way, no kernel goes in without a parity test against the reference (oracle).

### P0. Oracle environment (1-2 days)

- Build llama.cpp with HIP (gfx1100) and run GLM-5.3-Flash on 2 GPUs + CPU (`-ngl` split). This becomes
  the reference for the token sequence and the intermediate tensors.
- With `llama-eval-callback` (or `GGML_SCHED` debugging), prepare a script as `tools/glm_oracle.py` that, for
  a short prompt, dumps per-layer `hc_attn_pre`, `attn_norm`, `kda_scan_out`, `indexer_score`, `kq_soft_max`,
  `ffn_moe_out`, `l_out`.
- `ref/glm5.py`: a reference writing one layer straightforwardly in numpy (mHC, one KDA step, k-pool, the MoE
  router). Make it match the llama.cpp dumps. The role that Strata's `ref/gdn.py` and friends played.

### P1. Generalizing the model description and the loader (3-5 days)

- Turn `ModelGeometry` into "a struct filled from GGUF metadata", holding `arch` (qwen4exp / glm5-next) and
  arrays of layer kinds (`is_recr`, `is_dense`, `has_indexer`). Qwen's behavior does not change
  (confirmed by the existing parity tests and the `--serve` goldens).
- Add glm5-next to `tools/iq_pack.py`: dense tensors keep the GGUF type (equivalent to `--native`), experts
  go into `native_experts.txt` v4 as per-layer Q2_K/Q3_K/Q4_K blobs (gate|up|down). `experts.bin` is not
  created; the GGUF is read in place (the low-RAM mode's way).
- `tools/strata_tokenizer.py`: add the `glm4` pre (the CHATGLM4 regex, `ignore_merges`) and confirm a
  round-trip match with llama.cpp's `llama-tokenize`.
- Get the new `strata-glm` startup to the point where it "reads the weights, validates the shapes, and runs
  through layer 0's mHC".

### P2. The kernels (2-3 weeks, parallelizable)

Each comes with its `src/kernels/*_parity.cpp`.

| Kernel | Contents | Reference |
| --- | --- | --- |
| `mhc.cu` | `mhc_pre` (flat rms_norm → 24 mixes → pre/post/comb(Sinkhorn) → x), `mhc_post` | `ref/glm5.py`, ggml-cpu `dsv4_hc_*` |
| `kda.cu` | three q/k/v convs + silu, l2 norm, the gate (lower bound version), `kda_step` (vector decay), output norm × sigmoid(g2) | ggml-cpu `gated_delta_net` (the kda branch) |
| `dsa_indexer.cu` | ik's layer_norm, ig, k-pool pooling (when 4 tokens complete), 32-head relu scoring + the tail | `ref/glm5.py` |
| `dsa_attn.cu` | `wk_b` absorption, MQA softmax over the selected cells, `wv_b` expansion (single token / verification window) | the non-fused ggml path |
| `router_sigmoid` | sigmoid + bias top-8, normalize, ×2.5 | `build_moe_ffn` |
| `native_mmvq` Q2_K | transcribe ggml-cuda's `vec_dot_q2_K_q8_1` + swiglu clamp | ggml-cuda |
| `shared_expert` variant | no gate + clamp | ggml |
| dense FFN | the existing Q8_0 GEMV + silu without clamp | ggml |

A HIP caveat: as with `qsa_select`'s tf32 mma, the W7900 (gfx1100) has WMMA bf16/f16, but first get the
match with a straightforward FP32 version and leave speed for later.

### P3. Single-token decode end to end (1-2 weeks)

- Add `kda_layer`, `dsa_layer`, `dense_ffn_layer` and mHC's pre/post to `layer.cpp`, and branch
  `block_layer_pre/post` by layer kind. `session.cpp`'s doorbell and handoff to the CPU pool stay as they are.
- First, with `--no-capture` (no graph) and all experts on the CPU (the inverse configuration of `--no-pool`
  is fine too), produce **the same token sequence as llama.cpp under greedy**. Speed does not matter at this
  point.
- Next, get CUDA/HIP graph capture (`session_capture_token`) working.

### P4. Integrating the expert tiers (1 week)

- Profile creation (`--dump-routing` → `tools/make_profile.py`, supporting `n_expert=288`).
- Get the expert cache (per-layer variable blobs), the PCIe DMA share, low-RAM `--resident-experts`, and the
  2-GPU layer split or peer tier working on GLM. Skip the dense FFN layers (no experts).
- Set up a cache of over 70 GB of VRAM on the development machine and measure the hit rate and tok/s.

### P5. prefill (2 weeks)

- Put GLM's layer computation on `prefill.cpp`'s chunk loop: the multi-token version of mHC, KDA's chunked
  recurrence (the KDA version of `gdn_recurrence`), DSA's prompt attention (in-chunk causal + selection),
  batched k-pool construction.
- Q2_K/Q3_K instances for MMQ (an option equivalent to `STRATA_MMQ_KQUANTS`).
- Compare logits with llama.cpp on a 32K prompt (the tolerance is documented on the same basis as the fused
  path).

### P6. Verification window and speculation (1 week)

- Hook the multi versions of the new kernels into `verify.cpp`. Since there is no MTP, drive `--spec` with
  `suffix_drafter` (prompt lookup) only.
- Later: fetch GLM-5.3-Flash's nextn (MTP) tensors from HF safetensors in the style of `tools/mtp_fetch.py`
  and build the GLM version of `mtp.cpp` (`eh_proj`, `enorm`, `hnorm`, 1 layer). This is worth 1.5-1.8×.

### P7. Server and tools (a few days)

- For the equivalent of `serve/chat_template.jinja`, use the template inside the GGUF as is. Check
  `frontend.py`'s `EFFORT` mapping (GLM has low/high/max) and `OutputParser`'s tool-call format.
- Add the `glm` family to `setup.py` (download source, RAM requirements, the `--kv` default).

### P8. Performance (ongoing)

- In order of measured effect: the fraction of missed experts computed on the GPU, a Q2_K multi-token GPU
  kernel (the verification window), the hipBLASLt table, fusing KDA, WMMA for DSA selection, fused int8 for
  prefill (a k-quant version of `moe_fused_iq`).
- Record the numbers in the style of `docs/DETAILS.md` (state the machine, settings and engine version
  together; write estimates as estimates).

---

## 6. Verification criteria

- **Token sequence**: matches llama.cpp under greedy (short, medium and long prompts; Japanese/English/code).
- **Individual kernels**: in the parity test, state the maximum absolute error against the reference in each
  header, and when it does not match, the reason (Strata's existing rule). Places where bit-exactness is
  required: the KDA recurrence (because the error accumulates) and the router (because the selection changes).
- **Expert arithmetic**: for the same bytes, the GPU/CPU results match ggml's `vec_dot` (add Q2_K to Strata's
  `native_expert_parity`).

## 7. Risks

| Risk | Response |
| --- | --- |
| At 8 MB per expert the CPU computation cannot keep up | raise the GPU resident ratio (2 GPUs), compute misses on the GPU via DMA. Prioritize the Q2_K multi-token GPU kernel |
| The experts do not fit in 125 GB of RAM | make `--resident-experts` (only the GPU's complement in RAM) the default from the start |
| Numerical match of the 20 Sinkhorn iterations | transcribe ggml-cpu's implementation line by line, F32, fixed iteration order |
| The timing of the k-pool tail / cache update | first transcribe the semantics of llama.cpp's `set_input_kpool` into `ref/glm5.py` |
| HIP-specific differences (`hipErrorInvalidValue` etc.) | reuse the workarounds hit with Qwen (`hip_compat`), verify only on gfx1100 |
| No nextn (MTP), so speculation does not help | fetch from HF in the latter half of P6. Until then, prompt lookup only |

## 8. The first step (the next task)

1. Build llama.cpp with HIP, find a configuration (`-ngl`, `--split-mode`) where GLM-5.3-Flash runs, and save
   the token sequence.
2. Write `ref/glm5.py` and make it match the `llama-eval-callback` dumps for layers 0-3 (the dense FFN and the
   first KDA).
3. Start on generalizing `ModelGeometry` (P1). The existing tests protect against regressions on the Qwen side.

---
## MTP (NextN) layer spec (finalized 2026-10-04)

File: `/home/client/Projects/llm/GLM-5.3-Flash-MTP-Q4_K.gguf` (4.58 GB, `blk.45`). Its contents are 1 DSA layer (nope MLA, with indexer,
`index_share_mtp = true`) + 288 experts top-8 (gate/up Q4_K, down Q5_K, 4.4 GB) + shared expert + `nextn.eh_proj` (8192→4096, Q8_0),
`nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm`. There is no `embed_tokens` and no `shared_head_head`; the main model's `token_embd` / `output` are used instead.
**There is no `hc_*`** (even in the HF safetensors index there are no hyper-connection weights at layer 45).

Oracle: llama.cpp has not implemented the glm5-next MTP graph (`"GLM5-Next NextN graph not implemented yet"`), and HF transformers has no MTP class either.
The formulas were established from the vLLM implementation (`ref/vllm_glm5next_mtp.py.txt`, `ref/vllm_glm5next_model.py.txt`, user-provided). Verification is by acceptance rate.

```
h_prev   = output_norm( mean_streams(R_after_layer_45) )    same tensor as the main model's result_norm (vLLM: hc_contract = mean, then self.norm)
e        = token_embd[tok_{p+1}];  e = 0 at position 0       (vLLM fused_eh_norm: zero the embedding when pos == 0, then norm)
x        = eh_proj( concat( rms_norm(e)*enorm, rms_norm(h_prev)*hnorm ) )      eps = rms_eps, order is [enorm side | hnorm side]
a        = DSA_45( rms_norm(x)*attn_norm )                   <- no mHC, an ordinary pre-norm residual (vLLM: is_mtp_layer -> the "KDA + MoE without HC" branch)
x1       = x + a
m        = MoE_45( rms_norm(x1)*ffn_norm ) + shared_expert   (sigmoid router + bias, normalization x2.5, clamp 10: same as the main model)
h_next   = rms_norm(x1 + m) * shared_head_norm               <- used as-is both for the draft logits and as h_prev for the next draft step
logits   = output @ h_next                                   (the main model's output head, no re-norm)
```

When chaining multiple draft steps, iterate with h_prev ← h_next, tok ← argmax(logits). vLLM computes the indexer top-k only at step 0 and
reuses it afterwards (`index_share_for_mtp_iteration`). The MTP layer has its own latent KV and indexer cache, so an MTP prefill that pushes
(h_p, tok_{p+1}) through for every prompt position is required (the equivalent of Strata's main-model `draft_kv`).

Implementation note: the 4.4 GB of experts is VRAM-resident (the Q4_K/Q5_K paths of `native_mmvq` and the batched version work as-is). At 2.5 accepted
tokens per pass, the 11 ms of dense weights can be divided over 2.5 tokens, so an effective 15-18 ms/token (55-65 tps) is expected (estimate).

**GLM thinking off (2026-10-06)**: the pack's `chat_template.jinja` (derived from the GGUF) always ends the generation prompt with `<|assistant|><think>`, so
even with `reasoning_effort: none` / `enable_thinking: false` the model started thinking and the server returned that as the body. Fixed the pack's template to
emit `<think></think>` when `enable_thinking is false` (`~/Projects/Strata-data/glm53/pack/tokenizer/chat_template.jinja`, original kept as `.orig`).
`reasoning_effort: low/high` already worked via the template's `Reasoning Effort:` line. The server's done line now prints the stop token id.

## solo speedups (night of 2026-10-06, in progress)

**Chunking the prompt window (`--prefill-window` up to 8192)**: in prefill with a 2048-token window, both PCIe (42 GB of non-resident per window ≈ 3.0 s) and MMQ compute
(≈2.5 s) are near the floor. Widening the window does not change the transfer volume per window, so with a 4096 window the PCIe per token halves. The engine
separates `t_max_` (window) from `t_chunk_ = min(t_max_, 2048)` (compute chunk) and walks the window chunk by chunk within each layer (scratch is chunk-sized, and
only the residual stream R / bounce / h_win / tok_rows are window-sized). Releasing the ring buffer and prefetching the next layer happen only after the window's last chunk
(`moe(..., last_chunk)`). The MTP follow-along window is also walked in chunks. VRAM: R doubles (+134 MB/device @4096).

**Redesign of decode lookahead prefetch**: the old implementation (`--lookahead P`) was dead for 3 reasons: (1) it waited for a prediction that the prompt window never
produces, and hung; (2) the `prefetch()` flags were never written; (3) eviction assumed a pool (after an exchange swap the host-side copy is lost and it falls back to reading
from file: 4.3 tok/s). The version that makes experts resident via exchange swap turns the 18 MB/expert round trip of PCIe into 1.9 GB per round, giving 19.6 tok/s (baseline 30.8).
The current version is a **one-way lookahead prefetch into staging** (`prefetch_stage`: DMA only `STRATA_GLM5_PF_CAP` (default 3) of the top-predicted non-resident experts
per layer into the upper half of staging; on a hit at service time use it via a d2d copy inside staging; residents are not changed).

Result: the one-way staging version does not help either (cap 3: 27.6 tok/s, cap 2: 30.3, none 30.7). Prefetching 2-3 experts per layer (18-28 MB) takes 1.3-2 ms at 14 GB/s of
PCIe, which exceeds the layer's processing time (≈1.7 ms) and delays the next layer's flags. Given that host RAM's 42 GB/s is shared with the CPU side (30 GB/s),
PCIe prefetch cannot beat CPU compute → **abandoned** (`--lookahead` stays at default 0; the code is kept with the hang fixed).
The 4096 window is likewise withdrawn (compute-bound, see above).

## Long-context verification: NIAH and 256K (late night 2026-10-06)

Harness `scratchpad/niah.py` (the repo's docs as filler, one needle sentence placed at depth 10/50/90%, answered directly with thinking off; CLI `--prompts`).

| Context | DSA | page-mean routing (page 64, top 128/64) | page 16 / top 512 |
|---|---|---|---|
| 8K | 3/3 (490 tok/s) | 3/3 (508 tok/s) | |
| 32K | 3/3 (273 tok/s) | 3/3 (445 tok/s) | |
| 64K | **3/3** (218 tok/s) | **0/3** (434 tok/s) | **3/3** (375 tok/s) |
| 128K | (measuring) | 0/3 (420 tok/s) | |
| 236K | | 0/3 (404 tok/s, operation itself normal) | |

The windowed mode fails to pick up the needle at 64K and above ("I don't know such a passphrase"), while DSA does. Making pages as fine as 16 without changing the exact budget (8192)
recovers it at 64K, so the cause is **dilution by the 64-token mean** (the single needle sentence is buried among the other 63 tokens and does not rank high under `max_h q·mean`).
As a countermeasure, **per-token routing** was implemented (`STRATA_GLM5_LOD_ROUTE=tok`, default): the page's score is the max of the
inner products between q̄_g (the query averaged over head groups, `STRATA_GLM5_LOD_QGROUPS`=8) and each token's latent c (prefill uses MMQ over T×G rows × all candidate cells,
16384 columns at a time; decode reads the whole cache in per-page blocks). The summary (page mean) is unchanged. The cost is ~0.1 s per prefill window at 128K, ~1 ms/layer at decode.
256K: prefill of 236K tokens in 583 s (404 tok/s, flat), decode 18 tok/s (CLI, no drafts), no crashes.

Results (per-token routing, page 64, top 128/64, G=8): **64K 3/3 (440 tok/s), 128K 3/3 (405 tok/s)** — the same correct answers as DSA (218 / 179 tok/s) at
2.0-2.3x the speed. Page-mean routing was 0/3, so per-token routing is mandatory for the windowed mode on long contexts (made the default).

**Breakdown of 128K prefill** (per-token routing, 2048 window × 58, average 5.0 s/window = 410 tok/s): moe_hits 1.26 s (25%) + moe_down 0.70 (14%) =
expert MMQ 39%, `dsa_attend` 1.22 s (24%: with top 128 the compressed operand is ≈12.4K columns, the S-matrix path), `dsa_score` 0.43 s (9%: the routing
MMQ + colmax, G=8), KDA 0.85 s (17%), mhc 0.17. The headroom in 100K prefill is in that order.

**Investigation of expert MMQ (Q2_K)**: with `scratchpad/mmq_bench_new` (288 experts, 2048 tokens, skewed distribution) Q2_K gives 9.3 TMAC/s at J=16,
1.5 at J=24-48 (6x slower), and 8.0 at J=64. Q4_K gives 15.9 at J=24. Under `-Rpass-analysis=kernel-resource-usage`, Q2_K's `mul_mat_q<J=32>` uses VGPR 256 +
1584 B/lane of scratch, J=48 uses 2676 B (register spilling), and J=16 uses 8 B. Changing the thread count / I in the RDNA3 config table (`mmq-config-rdna3.cuh`) does not
make the spilling go away (it is structural to the MMA version of vec-dot). The experiment of switching only Q2_K to the dp4a version has not been started. The current engine
uses J=8 (→ J=16) for Q2_K, and that is the ceiling.

**Routing comparison (NIAH, 3 needles, top 128 / decode 64)** — the model's trained context length is 1M (no rope):

| Context | DSA | page mean | token (G=8) | indexer routing |
|---|---|---|---|---|
| 64K | 3/3 (218 tok/s) | 0/3 | 3/3 (440) | 3/3 (436) |
| 128K | 3/3 (179) | 0/3 | 3/3 (405) | 3/3 (408) |
| 236K | 3/3 (142) | 0/3 | 1/3 (365) | **2/3 (373)** |

Experiments lowering top (token version): 64K is 3/3 even at top 32 (509 tok/s, attention 0.47 s/window), 128K is 0/3 at top 32 and 1/3 at top 64 → 128K needs top 128.
**indexer routing** (`STRATA_GLM5_LOD_ROUTE=idx`): keep DSA's indexer path (idx_q/k/gate/proj, the ik/ig cache, the k-pool) alive in the windowed mode too;
prefill is MMQ of iq (T×32 rows, zero-padded from 128 to 256) × pool keys (Q8_0) → max of `Σ_h relu(S) w_h` → page max, and decode is `dsa_score_rows` → page max.
The cost is 0.4 s/window of `dsa_score` at 128K (8%) and 0.7 s at 236K (13%). This becomes the default for long contexts.

DSA is 3/3 even at 236K (142 tok/s, 28 minutes for one prompt) → the model can pick it up. The remaining gap on the windowed side is selection granularity (top 64/128 over
64-token pages), while DSA has top 512 over 4-token pools. A version that makes the exact set pool-based as well (`STRATA_GLM5_LOD_ROUTE=pool`: the indexer's top
`--lod-top-pools` pools (prefill 2048, decode 512) are exact, and only pages where no pool opened use the summary) has been implemented and is being verified at 236K.

Attempts at the Q2_K MMQ config table: setting J=32/48 to 256/2/128 or 128/1/64 leaves the spilling the same (1.6-2.7 KB/lane), and setting J=64 to I=64 gives 6.7 TMAC/s
(worse than the original 8.0).
→ Not fixable through the config table (it is the register structure of the MMA version of vec-dot). The pinned tree was reverted. The engine stays on J=16 (9.5 TMAC/s) for Q2_K.

**Results (pool selection, 2026-10-07)**: 64K 3/3 (438 tok/s), **236K 3/3 (373 tok/s)** — the same correct answers as DSA (142 tok/s) at 2.6x. With indexer routing,
128K is 3/3 even at top 64 pages (449 tok/s), and top 32 is 2/3. Made `pool` the default route (`STRATA_GLM5_LOD_ROUTE=pool`, prefill 2048 pools = 8192
tokens exact, decode 512 pools = 2048 tokens exact, summaries for pages that did not open). MoE's Q2_K MMQ (39% of the window) remains the biggest headroom.

With 1024 prefill pools (4096 tokens exact), 128K is still 3/3 (448 tok/s) and 236K 3/3 (**405 tok/s**, vs 373 for 2048 pools) → made 1024 the default
(`STRATA_GLM5_LOD_TOP_POOLS=1024`). The 5.0 s window at 236K: MoE 39%, attention 0.89 s (18%), indexer scoring 0.71 s (14%), KDA 17%.

- The indexer key MMQ width (`STRATA_GLM5_LOD_IDXPAD`) 128 and 256 give identical results at identical speed (at 12K, dsa_score 0.13 s/window): staying at 256.
- With 512 prefill pools (2048 tokens exact = same as DSA), 128K is still 3/3 at 477 tok/s (attention 0.53 s/window). 236K is being verified.
- **Column kernel for the KDA scan** (`k_kda_step_col`, prompt windows only, `STRATA_GLM5_KDA_COL=0` for the old kernel): the old kernel used one block per head
  (32 blocks), with 4 __syncthreads per token and the gate's exp computed 128×128 times. The new kernel holds one column in two adjacent lanes (64 state registers,
  no spilling), shares exp/k/q computed once, does the inner product within the lane plus one shuffle, and syncs once per token. A/B in progress at 12K.
  Result: at 12K, kda_scan 371 → 221 ms/window (−40%), prefill 24.1 → 23.4 s (+3%). Since the reduction order changes, the output diverges near ties (default on).
- 512 pools gives 2/3 at 236K (436 tok/s) → the default for 236K stays 1024. With the KDA column kernel included, the 236K window goes 5.0 → 4.7 s.
- **Summary ablation** (`STRATA_GLM5_LOD_NOSUM=1`: indexer selection only, without adding summaries = windowed DSA) is running at 12K/236K.
  If no difference shows up, then page summaries are unnecessary on GLM, and the source of the speed is the implementation structure
  "window-shared selection + compressed operand + fused decode kernel".

**Ablation result**: without summaries (windowed DSA, 1024 pools), 236K is **3/3 at 433 tok/s** (405 tok/s with summaries, attention 0.89 → 0.69 s/window), and 12K is 527 tok/s.
On GLM-5.3-Flash (trained with DSA), the page summaries contribute nothing to NIAH and are ~7% slower for the extra columns. The source of the speedup obtained
(DSA 142 → 433 tok/s @236K) is the implementation structure "share the selection across the window and push [selected cells | the window itself] through MMQ as a compressed operand"
+ "fused decode kernel".
→ The server default is `--attn lod` + `STRATA_GLM5_LOD_ROUTE=pool` + `STRATA_GLM5_LOD_NOSUM=1` (= windowed DSA). The page summaries are kept as an option for dense-attention models.

## Per-layer selection importance (2026-10-08, 64K NIAH, 3 needles)

`STRATA_GLM5_LOD_ROUTE_LAYERS="3=pool,7=tok,..."` specifies the route per layer (pool = indexer selection, tok = token scoring without the indexer.
All-tok was 3/3 at 64K, 3/3 at 128K, 1/3 at 236K).

| Experiment | Result |
|---|---|
| keep-one (indexer on layer k only, tok on the other 10 layers) | layer 3: 2/3, each of layers 7-43 alone: **3/3** (10/10 layers) |
| leave-one-out (tok on layer k only, indexer elsewhere) | layer 23: 2/3, others: 3/3 |

→ At 64K, "one layer selecting accurately is enough". The fact that excluding only layer 23 (device 1's first DSA layer) dropped one needle needs re-measurement
(3 needles is statistically weak). Indexer scoring costs 7-14% of the window, so the speed-side significance is "can the budget for the remaining layers be made smaller?"
(next experiment: indexer on 1 layer + top 32 on the others).

**Speed test for indexer on 1 layer (23) + token scoring on the others**:

| Context | Budget on other layers | NIAH | prefill |
|---|---|---|---|
| 64K | top 32 | 3/3 | 522 tok/s (all-layer indexer, 1024 pools: 436) |
| 128K | top 64 | 3/3 | 462 (448) |
| 236K | top 128 | **1/3** | 382 (433) |

64K/128K are fast and correct, but at 236K "one accurate layer" is not enough, and on top of that the all-layer token scoring adds columns and is slower. At 236K,
all-layer indexer (pool selection) remains best. Switching by context length is practical: ≤128K uses layer 23 indexer + tok, above that all-layer pool.

**Q2_K MMQ, dp4a layout attempt (2026-10-08)**: built a translation unit with WMMA disabled for Q2_K only (dp4a tiles) and measured → the bench is identical to the WMMA version
(J=16 9.6 TMAC/s, J=24-48 1.6). Register usage is the same too (at J=32, 520 B/lane; J=48 1.9 KB; J=64 2.5 KB of scratch). The cause of the spilling is not the tile layout
but that `mul_mat_q`'s accumulator registers (the float sums of J×I/wave) exceed 256 VGPR on wave32 when J ≥ 24.
→ Not fixable via the config table or the layout. Fixing it requires either the kernel body (split J in two and do two passes, or move the accumulator to LDS) or writing a
Q2_K/WMMA version of Strata's fused expert kernel (gfx1100 WMMA is int8 16×16×16; fold the scales of Q2_K's 16-value sub-blocks into the inner product).
Note that the row count per expert averages 57 with a max of ~300, so a J=16 tile is reasonable in terms of rows, and raising J would not fill it.
The real efficiency problem is in the I (weight row) direction: the structure of walking I=64/128 tiles over 2048 rows (n_ff) × K=4096 is rate-limited by Q2_K's unpacking cost.

## WMMA kernel for Q2_K experts (2026-10-08)

`src/glm5/moe_q2k.cu`: a Strata-specific kernel that computes the prompt window's expert products (gate|up, down) with gfx11 int8 WMMA
(`wmma_i32_16x16x16_iu8`, wave32). It replaces MMQ's Q2_K (register spilling at J≥24, 9.7 TMAC/s).
- Arithmetic: w = d·sc_s·q − dmin·m_s (16-value sub-blocks). The A matrix = sc_s·q is built by **byte permute** (`__builtin_amdgcn_perm` selects from
  {0,sc,2sc,3sc} with a 2-bit code; 0..45 fits in int8), so one K=16 MMA matches a sub-block exactly. The min term Σ_s m_s·suma_s is two MMAs
  (A = m_s) over the 16-value activation sums suma split into low/high bytes. Activations are int8 in 256-value blocks (one scale + 16-value sums).
- Fragment layout (measured on gfx1100): for lane l of C, element e = Σ_k (lane 2e+l/16 of the first operand)·(lane l%16 of the second operand).
  Putting the activations as the first operand makes lane = weight row l%16 and element = activation column 2e+l/16 (it was wired the other way at first,
  giving wrong answers; found with a one-hot probe).
- Block = 8 waves × 16 weight rows = 128 rows, column tile 32 (2 × 16), grid (row tile, column tile, expert). VGPR 170, no spilling.
- Verification: max error 2.4e-5 (|ref| 245) against the double reference of ggml's `quantize_row_q2_K_ref`/`dequantize_row_q2_K`.
- Speed (288 experts, 2048 tokens × 8, skewed distribution): gate|up 28.3 → **12.3 ms**, 22.4 TMAC/s (MMQ 9.7), down 6.1 ms.
- engine: on the Q2_K layers (36/42), the products and activation quantization of `moe_mmq` are replaced (`STRATA_GLM5_NOQ2K=1` reverts to MMQ). Q3_K/Q4_K layers stay on MMQ.
- A/B in the engine (12K, pool/nosum): prefill 23.0 → **20.6 s (598 tok/s)**, window 3.81 → 3.40 s. moe_hits 1221 → 599 ms, moe_down 682 → 366 ms
  (MoE compute 1.9 → 0.97 s). In exchange, `moe_wait` 503 ms appeared: the PCIe transfer of non-resident experts (≈35 GB/window ≈ 2.5 s) no longer hides fully behind compute
  → next is the PCIe side (4 rings, or a 4096 window to double the tokens per transfer). In the 1064-token output, tokens near ties change due to the difference in the
  activation quantization block (256 vs 128) (relative to MMQ).

**4096-token window (2026-10-08)**: the hang was caused by the doorbell's per-layer tables (routing id / miss slot / prediction) being allocated at chunk size while the window size
was used as the stride (from layer 23 on it wrote out of bounds and gave "never published"). Fixed by unifying the stride to the chunk (3365985).
Since WMMA made compute faster and exposed PCIe, a 4096 window that doubles the tokens per transfer helps:

| Prompt | 2048 window + MMQ (morning) | 2048 window + WMMA | **4096 window + WMMA** |
|---|---|---|---|
| 4K | 7.6 s (531 tok/s) | | **5.4 s (764)** |
| 12K | 23.0 s (533) | 20.6 s (598) | **17.9 s (686)** |
| 128K (NIAH 3/3) | 251 s (470) | | **200 s (592)** |

Output is identical to the 2048 window (4K 32/32, 12K 16/16). The 128K window (4096) is 6.9 s: MoE 1.9 s, attention 1.0 s, PCIe waits gone.
→ Made the server default `--prefill-window 4096` (R/bounce +134 MB/device).

## Two-GPU pipeline for the prompt window (2026-10-08, 5ba5d30)

The layers are split across device 0 (0-21) and device 1 (22-44), so while device 1 is processing window k, window k+1 can start on device 0
(layer L of window k+1 only needs to run after layer L of window k, and the device boundary guarantees that ordering). Implementation:
- **Job** (`Engine::Job`: slot, pos0, T, tok, parity, tokens, per-device prefetch position, next job): the "window state" the layer code reads
  (`cur_slot_`/`tok_`/`ring_parity_`/`win_T_`) is replaced by access through `d.job` (`jslot/jtok/jparity/jT`, `stj(d, il)`). Since each device holds
  the job it is currently processing, two windows can run at once.
- **Stage** `stage_run(d, j, il0, il1)`: one device's layer range for one window (including the chunk loop). The first stage does the embedding, the last the head.
  The handoff is a host bounce of the residual stream R (2 faces by window parity) + inter-device events (2 faces). The sequential path (`forward_window`, decode windows)
  calls the same `stage_run` in order.
- **Stage threads**: one per device (`stage_main`), with jobs handed over through a queue. Each stage's host sync (MMQ's max_rows) waits only on its own
  device. The first stage never runs more than one window ahead of the last stage (because the 2 faces of the bounce/events are reused 2 windows later).
- **Continuation of ring prefetch**: `prefetch_up_to(d, j, limit)` continues into the layers of `j.next` once job j's layers are exhausted (with a per-device mutex).
  This way, while device 1 is computing window k, the experts for the first layer of window k+1 arrive in the ring. The flags are per layer (not per window), but
  the DMA waits on the previous reader's event, so it does not advance to k+1's values until window k's `moe_wait` has passed.
- **MTP follow-along**: the per-window callback (`prompt_windows(slot, toks, n, wsize, on_window)`) is called from the last stage's thread, and it queues the MTP window
  onto the same stream while `h_win` is still valid in stream order. Only the final window does `argmax_last` (a sync on device 1).
- serve: pipeline only when the slot is not in decode (otherwise the conventional per-window loop + batch_step interrupts). The CLI always pipelines.
- Disabling: `STRATA_GLM5_NOPIPE=1`. Conditions: 2 devices, prompt mode (window ≥ 256), at least 2 windows.

### Results (stage35, 72a90f7)

| Case | Sequential (stage33/34) | Pipelined | Output |
|---|---|---|---|
| 4K, window 2048 ×2 | 6.73 s (608 tok/s) | 5.47 s (748 tok/s) | 32 tokens identical |
| 12K, window 4096 ×3 | 17.9 s (686 tok/s) | 12.23 s (1005 tok/s) | 16 tokens identical |
| 128K NIAH, window 4096 | 200 s (592 tok/s) | 111 s (1060 tok/s) | 3/3 |
| 1K sequential / slots-test | no change | — | 24 tokens identical / 0 mismatches |

The more windows there are, the closer it gets to the steady state (both GPUs always computing a different window), giving 1.8x at 128K. 12K has only 3 windows, so the
single-lung portion at the head/tail remains. Decode is unaffected (as before).

Bugs hit in the first implementation (all fixed in 72a90f7):
- **Ring continuation**: the ring buffer number was `moe_seq % nbuf` (fixed per layer), so when the number of MoE layers per device is odd (19), the first layer of the next
  window was DMA'd into the same buffer as the last layer of the current window (which had not been read yet) (→ garbage experts → NaN routing → illegal memory access).
  Gave each window a per-device ring base `Job::rbase` and switched to `(rbase + moe_seq) % nbuf`. The sequential path uses the same formula (the base advances between windows).
- **Bounce reuse**: before rewriting the bounce of parity p two windows later, wait on the reader's (the next device's h2d) event `ev_in2[p]`.
  The first stage does not start window k until the last stage starts window k-1 (to guarantee the reader's enqueue ordering).
- **Window state for the final callback**: `argmax_last` decides the row count from `win_T_`, so the last stage sets `win_start_/win_T_` before the callback.
- Thread safety: mutexes on the `mmq_fits_` cache (a map) and on the loading pool for un-pinned experts (`Pool::run`).
- The issue where `STRATA_GLM5_SYNC=1` entered a doorbell wait in prompt mode and froze: in prompt mode it does not wait (so it stays usable for diagnostics).

Handling of np=2: a job carries a slot, and the state tables are per slot via `stj(d, il)`. serve pipelines only when no slot is in decode
(during decode, the conventional per-window loop + batch_step interrupts). Pushing 2 prompts into the pipeline at once (alternating windows) is not implemented.

## tool call (2026-10-08, ab27dec)

GLM-5.3 Flash emits `<tool_call>NAME<arg_key>P</arg_key><arg_value>V</arg_value>...</tool_call>` as per the template, but
serve/frontend.py's output parser only understood the `<function=NAME><parameter=P>` form, resulting in HTTP 400 "malformed tool call".
Added the GLM form (non-streaming / streaming; arguments whose schema is string stay strings, everything else is JSON-decoded; a `</arg_value>` inside a value is
judged by whether the next `<arg_key>` or `</tool_call>` follows at that position). The round trip (call → tool result `<|observation|>` → answer) and streaming were confirmed on the real machine.

## Server configurations (as of 2026-10-08)

| Config file | GPU | ctx | parallel | expert budget | Notes |
|---|---|---|---|---|---|
| strata-glm53.json | 0,1 | 262144 | 2 | 28.5,25.8 | 3 cache sets |
| strata-glm53-np1.json | 0,1 | 262144 | 1 | 31.0,29.5 | 60.5 GiB resident, 1.2/1.55 GiB free |
| strata-glm53-1gpu.json | 0 | 262144 | 1 | 17.9 (+ --pin-gib 96) | decode 15 tok/s, prefill 490 tok/s |

On 1 GPU, unless the entire non-resident portion (82 GiB) is pinned, the prompt's prefetch stalls for lack of staging slots (`--pin-gib` default 64).

## Decode: the Q2_K products on WMMA (2026-10-09)

The decode round (window of 3 rows, spec 2, ~80 ms) measured per phase: resident expert products ~30 ms (5.5 GB read at
~180 GB/s), dense layers ~25 ms, the wait for the CPU's misses 6-12 ms, the rest ~10 ms. A micro-benchmark of one layer
(20 experts x 1-3 columns, scratchpad `dec_bench`): the grouped MMVQ 161 / 114 GB/s (gate / down), the prompt WMMA
kernel 268 / 252, a D2D copy 680. Q2_K is instruction-bound, not memory-bound: the plain Q2_K MMVQ stays at 233 GB/s
even from the Infinity Cache, while Q4_K reaches 500. A dp4a kernel with integer accumulation per 32-weight block was
tried first and ran at 100 GB/s for 3 columns (6 VALU/LDS instructions per weight byte: the arithmetic alone takes that long).

The int8 WMMA packs 16 columns of dot products into one instruction, so the decode kernel is a WMMA variant (`dec_wmma`
in moe_q2k.cu): one 16-column tile per group (a decode window has at most 8 pairs on an expert), the activation sums
pre-packed per (column, super-block) by `pack_sums` (the prompt kernel repacks them per lane and super-block), the grouped
tables (`grp_w`, `col0`, `ncols`, `orow`) as its interface. 550-565 GB/s on the benchmark, bit-identical to the prompt
kernel on the same operands. In the engine the pairs' activations are quantized per 256 (`quantize_act`) as the prompt
path does, so decode and prompt now share one quantization; `STRATA_GLM5_NODECQ2K=1` restores the grouped MMVQ.

| | old | new |
|---|---|---|
| 1064 + 96 tokens, spec 2 | 28.6 tok/s (86 ms/round) | 30.8 tok/s (82 ms/round) |
| spec 3 | - | 30.8 tok/s (3.00 tokens/round) |
| slots-test, solo no drafts | 88.5 ms/token | 76.5 ms/token |
| slots-test, 2 slots no drafts | 67.3 ms/window | 57.4 ms/window |
| 1064 + 24 greedy | reference | 23/24 identical (the quantization change) |

Why the engine gains only 8% where the kernel gains 3x: the CPU's miss computation (~180 distinct experts per round,
1.5 GB from pinned RAM at ~34 GB/s = 44 ms) ran concurrently with the resident products and already hid them. With
the products at 8 ms the wait for the CPU is exposed (moe_wait 12 -> 23 ms). The round is now ~45 ms of GPU-only
work (dense layers, router, MTP) serialized layer by layer with ~44 ms of CPU work; the memory bus is the floor of
the CPU part. The remaining single-stream levers are the dense phase (25 ms for 9.8 GiB of weights, ~400 GB/s) and
the per-layer overheads; the structural lever is interleaving two requests at layer granularity so one request's CPU
phase overlaps the other's GPU phase (expected ~1.6x aggregate at np=2).

## Decode: the loan pool (2026-10-09)

**The measurement first.** With fresh prompts (8 of them: Japanese and English explanations, Python / TypeScript / Rust
tasks, thinking on) the static profile hits only 67% and decode runs at 28-30 tok/s with speculation (18 tok/s on the
`--prompts` path, which decodes without the draft model). A routing log (`STRATA_GLM5_ROUTE_LOG=<path>`: the resident
sets and every decode window's ids) replayed through residency policies (scratchpad `route_sim.py`):

| policy on the same routing | hit | distinct misses per window |
|---|---|---|
| the static profile | 66.9% | 2.65 |
| the best static set for this stream | 82.9% | |
| LRU, every miss swapped in (unlimited bandwidth) | 90.6% | 0.75 |
| EMA migration, 1-4 experts per layer per window | 90.0-90.7% | 0.74-0.80 |
| static + a loan pool of 10 / 20 / 30 slots per layer, LRU | 84.5 / 89.1 / 92.0% | 1.24 / 0.87 / 0.64 |

The working set moves with the request: no fixed set does better than 83%, a time-adaptive one passes 90%.

**The loan pool.** During decode the prompt windows' prefetch ring (2 x 1.34 + 2 x 1.20 GiB = 660 slots of 8.26 MB on
the two cards) is lent to the expert store as an LRU cache of missed experts: a slot table entry `kLoanBit | j`
addresses `loan_ptr[j]` (the grouping kernel's second address space), the store keeps `loan_of` per expert beside
`slot_of`, and the table updates go through per-layer lists applied by `k_loan_apply` before the layer's grouping.
A prompt window takes the ring back: `drop_loans()` syncs the devices and clears the tables. Protections: a slot used
in the current window is never a victim, a slot whose copy is in flight is not either, and an eviction's table entry
is applied before any addition. `STRATA_GLM5_LOAN=0` disables, `STRATA_GLM5_LOAN_PER_WIN` caps the loans per call.

**What did not work, and why.** Copying the CPU-served misses from the host into loan slots (`STRATA_GLM5_LOAN_H2D=1`)
raised the hit rate to 85% but made decode slower, in three steps of diagnosis: (1) the copies on the serve path's copy
stream delayed each layer's flag (fixed: a loan stream per device); (2) the table update was a small host-to-device
memcpy on the compute stream, which queued behind the 8 MB loan copies on the copy engine and stalled every layer
(moe_wait 11 -> 42-78 ms; fixed: the apply kernel reads mapped pinned entries, and the CPU's result rows are copied by
a kernel too); (3) what remained is the memory bus: a CPU-served miss already reads its 8.3 MB blob once, and the loan
copy reads it again, so in steady state the copies cost about what the extra hits save (the same prompt: 25.6 vs 28.5
tok/s). The default therefore loans only the misses the CPU/PCIe split sends over PCIe: their blob crosses the bus
once (into staging), and the loan is a device-to-device copy. Biasing the split toward PCIe (`STRATA_GLM5_DMA_MS`)
lengthens the PCIe side and loses again (31.9-32.4 vs 33.6).

| prompt (300 tokens, spec 2) | loans off | loans (PCIe-staged) | hit off -> on |
|---|---|---|---|
| Python CLI tool (3) | 28.5 | **33.6** | |
| Japanese seasons (1) | 36.1 | **39.2** | 86.6 -> 89.3% |
| TypeScript (5) | 30.2 | **36.0** | 74.3 -> 79.2% |
| Rust (8) | 31.9 | **34.0** | 73.6 -> 78.8% |

1064 + 24 greedy: 24/24 identical to the reference; `--slots-test 2`: 0 mismatches.

**Where decode stands.** A round (3 rows) is ~45 ms of GPU-only work (dense layers 25, router, MTP) serialized layer by
layer with the CPU's misses at ~0.2 ms each (8.3 MB from pinned RAM at the bus's 42 GB/s). The remaining levers:
more loan capacity (the prompt scratch, ~4 GiB per card more), a smaller GPU base, and for two requests the layer
interleave. Short prompts (T < 256, the mid-size window path) wait on the ring's PCIe copies (a 39-token prompt takes
1.1 s): the CPU split applies there too.

**The crash it caused (fixed, e97a32a).** The server's engine died after ~1,800 tokens of a long answer: a segfault in
serve_cpu's routing loop (the kernel log's address symbolized to it, and gdb reproduced it at window 803), i.e. a
routing id outside 0..287: the activations had turned to garbage. Two consistency holes between the host's and the
device's view of the loans: (1) the host counted a loan as resident from its allocation, the device only from the
table update at the layer's next grouping, so a hit on one side was a miss on the other and the pair rows were laid
out differently; (2) the LRU could evict a slot of a later layer of the same window, whose grouping (not yet published)
would still read it while the new copy overwrote it. Now a loan is `loan_live` only once applied, victims come only
from layers already served in the window, and an id out of range stops with a message. 2 x 2,500 tokens, the
exactness test and the eight-prompt alternation pass.
