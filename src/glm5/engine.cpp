// src/glm5/engine.cpp - the GLM-5-Next decode engine (include/strata/glm5/glm5.hpp): a verify window of T tokens through
// the trunk in the order of llama.cpp's glm5-next.cpp graph and ref/glm5.py, the rollback, and the MTP step (the
// NextN layer as vLLM computes it: docs/GLM5_NEXT.md, "MTP (NextN) 層の仕様").  Every intermediate llama.cpp names is
// offered to `dump` under the same name when T == 1, so tools/glm_oracle/compare_dumps.py can hold the two side by side.
//
// Multi-device: each layer runs on its device's stream with that device's scratch (`Dev`); at a boundary the residual
// R of the window is handed over through pinned host memory and the next stream waits on the copy.
#include "strata/glm5/glm5.hpp"
#include "strata/glm5/experts.hpp"
#include "strata/glm5/kernels.hpp"
#include "strata/glm5/moe_q2k.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <numeric>
#include <type_traits>
#include <vector>
#include <thread>
#include <stdexcept>
#include <string>

namespace strata::glm5 {
namespace {
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
constexpr uint32_t T_F32 = 0, T_BF16 = 30;
inline cudaStream_t S(void* s) { return static_cast<cudaStream_t>(s); }
inline size_t q8_cols(int n) { return strata::kernels::native_q8_1_bytes(n, 1); }
const bool kDebugSync = std::getenv("STRATA_GLM5_SYNC") != nullptr;   // synchronise after every layer and name a fault
const bool kNoDoorbell = std::getenv("STRATA_GLM5_NODOORBELL") != nullptr;   // stage the misses synchronously instead
const bool kTrace = std::getenv("STRATA_GLM5_TRACE") != nullptr;
const bool kMidCpu = !(std::getenv("STRATA_GLM5_MID_CPU") && std::atoi(std::getenv("STRATA_GLM5_MID_CPU")) == 0);   // mid-size prompt windows: the misses split between PCIe and the CPU
constexpr int kDbXRows = 2048;   // MoE input rows published for the CPU service (a mid-size window's tokens, two blocks by layer parity)
const bool kNoDecQ2K = std::getenv("STRATA_GLM5_NODECQ2K") && std::atoi(std::getenv("STRATA_GLM5_NODECQ2K")) != 0;   // the decode Q2_K products back on the grouped MMVQ
const bool kProf = std::getenv("STRATA_GLM5_PROF") != nullptr;   // per-phase GPU times of every window
constexpr int kDsaBatch = 64;    // tokens per DSA score / select / attend launch
// prompt windows attend through MMQ over the whole latent cache (masked) up to this many cells (STRATA_GLM5_ATTN_MAXKV; the S
// matrix of a 64-token batch is 64 x 64 heads x cells floats: 256 MB at 16384)
const int kIdxPad = std::getenv("STRATA_GLM5_LOD_IDXPAD") ? std::atoi(std::getenv("STRATA_GLM5_LOD_IDXPAD")) : 256;   // the indexer keys padded to this width for the MMQ routing product
const int kDenseAttnMaxKv = std::getenv("STRATA_GLM5_ATTN_MAXKV") ? std::max(512, std::atoi(std::getenv("STRATA_GLM5_ATTN_MAXKV")) / 512 * 512) : 16384;
const bool kNoDenseAttn = std::getenv("STRATA_GLM5_NODENSEATTN") != nullptr;
constexpr int kGemmMinT = 16;    // windows at least this wide take the tensor-core GEMM for the dense weights
constexpr int kQkvFuseT = 64;    // windows up to this many rows take the KDA q | k | v projections as one product ([T][3 D] scratch)
constexpr int kFabT = 16;
constexpr int kShT = 64;         // decode windows up to this many rows keep the shared experts' scratch apart from the pairs' (the early order)
static const bool kQ8Fuse = !std::getenv("STRATA_GLM5_Q8_FUSE") || std::atoi(std::getenv("STRATA_GLM5_Q8_FUSE")) != 0;   // 0: the producers do not quantize (a separate quantize per Q8_0 product)
static const bool kPackFuse = !std::getenv("STRATA_GLM5_PACK_FUSE") || std::atoi(std::getenv("STRATA_GLM5_PACK_FUSE")) != 0;   // 0: the sums packed by a separate launch
static const bool kRollbackTable = !std::getenv("STRATA_GLM5_ROLLBACK_TABLE") || std::atoi(std::getenv("STRATA_GLM5_ROLLBACK_TABLE")) != 0;   // 0: the rollback as memcpys
static const bool kNarrowBf16 = !std::getenv("STRATA_GLM5_NARROW_BF16") || std::atoi(std::getenv("STRATA_GLM5_NARROW_BF16")) != 0;   // 0: the generic BF16 kernel for narrow weights
static const bool kMoeEarly = !std::getenv("STRATA_GLM5_MOE_EARLY") || std::atoi(std::getenv("STRATA_GLM5_MOE_EARLY")) != 0;   // 0: the resident down products and shared experts after the wait
static const bool kMhcDec = !std::getenv("STRATA_GLM5_MHC_DEC") || std::atoi(std::getenv("STRATA_GLM5_MHC_DEC")) != 0;   // 0: the separate mHC kernels for small windows        // ... and the f_a | g_a | beta ones (the verify windows; the scratch is this many rows)
constexpr int kPrefetchAllMinT = 256;   // prompt windows at least this wide prefetch every non-resident expert (42 GB per window here)
constexpr int kMmqMinT = 9;      // ... and the MMQ expert path (int8 tensor cores); the verify windows (<= 8) keep the exact grouped MMVQ
const bool kNoMmq = std::getenv("STRATA_GLM5_NOMMQ") != nullptr;
const bool kNoDenseMmq = std::getenv("STRATA_GLM5_NODENSEMMQ") != nullptr;
const bool kNoCpu = std::getenv("STRATA_GLM5_CPU") && std::string(std::getenv("STRATA_GLM5_CPU")) == "0";   // narrate the doorbell: windows, served layers
void debug_sync(void* stream, int il, const char* where) {
    if (!kDebugSync) return;
    const cudaError_t e = cudaStreamSynchronize(S(stream));
    if (e != cudaSuccess) throw std::runtime_error(std::string("device fault after layer ") + std::to_string(il) + " (" + where + "): " + cudaGetErrorString(e));
}
// debug: how many of the T rows of n floats hold a NaN/Inf, and the first such row
void debug_nan(const float* dev, int T, int n, int il, const char* what) {
    if (!kDebugSync) return;
    std::vector<float> h((size_t) T * n);
    if (cudaMemcpy(h.data(), dev, h.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return;
    int bad = 0, first = -1;
    for (int t = 0; t < T; ++t) {
        bool b = false;
        for (int i = 0; i < n && !b; ++i) b = !std::isfinite(h[(size_t) t * n + i]);
        if (b) { ++bad; if (first < 0) first = t; }
    }
    if (bad) std::fprintf(stderr, "glm5: layer %d %s: %d of %d rows non-finite (first row %d)\n", il, what, bad, T, first);
}
}  // namespace

void* Engine::alloc(uint64_t bytes, int ordinal) {
    check(cudaSetDevice(ordinal), "cudaSetDevice");
    void* p = nullptr;
    check(cudaMalloc(&p, bytes ? bytes : 16), "cudaMalloc");
    check(cudaMemset(p, 0, bytes ? bytes : 16), "cudaMemset");
    allocations_.push_back(p);
    alloc_device_.push_back(ordinal);
    return p;
}
void* Engine::alloc_host(uint64_t bytes) {
    void* p = nullptr;
    check(cudaHostAlloc(&p, bytes ? bytes : 16, cudaHostAllocPortable), "cudaHostAlloc");
    std::memset(p, 0, bytes ? bytes : 16);
    host_allocations_.push_back(p);
    return p;
}
void* Engine::alloc_mapped(uint64_t bytes, int ordinal, void** dev_view) {
    check(cudaSetDevice(ordinal), "cudaSetDevice");
    void* p = nullptr;
    check(cudaHostAlloc(&p, bytes ? bytes : 16, cudaHostAllocPortable | cudaHostAllocMapped), "cudaHostAlloc mapped");
    std::memset(p, 0, bytes ? bytes : 16);
    check(cudaHostGetDevicePointer(dev_view, p, 0), "cudaHostGetDevicePointer");
    host_allocations_.push_back(p);
    return p;
}

Engine::Engine(const Model& m, int max_ctx, int t_max, int n_slots) : m_(m), hp_(m.hp()), max_ctx_(max_ctx), t_max_(t_max), n_slots_(std::max(0, n_slots)) {
    if (std::getenv("STRATA_GLM5_MMVQ_UPSTREAM") && std::atoi(std::getenv("STRATA_GLM5_MMVQ_UPSTREAM")) != 0)
        strata::kernels::native_mmvq_set_multi_exact(false);   // llama.cpp's multi-column layout (two rows per block; equal to one column to rounding)
    cur_slot_ = n_slots_;
    mtp_slot_ = n_slots_;
    // (a window wider than a chunk is wired - the layer loop runs chunk by chunk - but off: prefill is compute-bound now, a 4096
    // window gained nothing and its first decode window hung at the second device's first MoE layer; STRATA_GLM5_WIDE_WINDOW=1 re-enables)
    // the doorbell / dispatch tables are per layer with the chunk's row count as the stride (t_chunk_): a wider window
    // runs chunk by chunk and its decode windows are tiny
    const int t_cap = (!std::getenv("STRATA_GLM5_WIDE_WINDOW") || std::atoi(std::getenv("STRATA_GLM5_WIDE_WINDOW")) != 0) ? 4 * window_max_tokens() : window_max_tokens();
    if (t_max < 1 || t_max > t_cap) throw std::runtime_error("glm5: the window must be 1.." + std::to_string(t_cap) + " tokens");
    t_chunk_ = std::min(t_max_, window_max_tokens());
    // T sizes the per-chunk scratch (a prompt window wider than a chunk runs every layer chunk by chunk, so the routed
    // pairs, activations and attention scratch never exceed one chunk); TS sizes what spans the whole window: the
    // residual streams R, their bounce, the token rows, the final hidden rows
    const int E = hp_.n_embd, D = hp_.d_inner(), T = t_chunk_, TS = t_max_, K = hp_.n_expert_used, NL = m.n_layer_all();
    const int big = std::max({D, hp_.n_head * hp_.head_k_mla, hp_.n_head * hp_.kv_lora, hp_.n_ff, hp_.idx_heads * hp_.idx_dim, 2 * E});
    const int FF = hp_.n_ff_exp;
    max_pairs_ = T * K;
    pairs_alloc_ = max_pairs_ + 256;   // llama.cpp's MMQ (need_check = false) loads whole row tiles past bounds[n]
    snap_T_ = std::min(T, 8);   // the verify window; a wider (prompt) window has no per-token rollback
    if (const char* ev = std::getenv("STRATA_GLM5_LOD")) lod_.on = std::atoi(ev) != 0;
    if (const char* ev = std::getenv("STRATA_GLM5_LOD_PAGE")) lod_.ps = std::max(32, std::atoi(ev));
    if (const char* ev = std::getenv("STRATA_GLM5_LOD_TOP")) lod_.top_prefill = std::max(0, std::atoi(ev));
    if (const char* ev = std::getenv("STRATA_GLM5_LOD_TOP_DECODE")) lod_.top_decode = std::max(0, std::atoi(ev));
    if (const char* ev = std::getenv("STRATA_GLM5_LOD_LOCAL")) lod_.local = std::max(0, std::atoi(ev));
    if (const char* ev = std::getenv("STRATA_GLM5_LOD_SINK")) lod_.sink_pages = std::max(0, std::atoi(ev));
    if (const char* ev = std::getenv("STRATA_GLM5_LOD_ROUTE")) { const std::string r = ev; lod_.tok_route = r != "mean"; lod_.idx_route = r == "idx" || r == "pool"; lod_.pool_sel = r == "pool"; }
    if (const char* ev = std::getenv("STRATA_GLM5_LOD_TOP_POOLS")) lod_.top_pools = std::max(0, std::atoi(ev));
    if (const char* ev = std::getenv("STRATA_GLM5_LOD_TOP_POOLS_DECODE")) lod_.top_pools_decode = std::max(0, std::atoi(ev));
    if (const char* ev = std::getenv("STRATA_GLM5_LOD_NOSUM")) lod_.no_sum = std::atoi(ev) != 0;
    // per-layer routes: "3=tok,7=idx,11=pool,..." (a layer not named keeps the global route); storage for every route is
    // allocated when any layer uses it (the indexer caches are per layer anyway)
    if (const char* ev = std::getenv("STRATA_GLM5_LOD_ROUTE_LAYERS")) {
        lod_layer_.assign((size_t) m.n_layer_all(), lod_);
        std::string spec = ev;
        size_t i = 0;
        while (i < spec.size()) {
            size_t j = spec.find(',', i); if (j == std::string::npos) j = spec.size();
            const std::string item = spec.substr(i, j - i);
            const size_t eq = item.find('=');
            if (eq != std::string::npos) {
                const int il = std::atoi(item.substr(0, eq).c_str()); const std::string r = item.substr(eq + 1);
                if (il >= 0 && il < m.n_layer_all()) { LodCfg& c = lod_layer_[(size_t) il]; c.tok_route = r != "mean"; c.idx_route = r == "idx" || r == "pool"; c.pool_sel = r == "pool"; }
            }
            i = j + 1;
        }
        bool any_idx = lod_.idx_route, any_pool = lod_.pool_sel;
        for (const LodCfg& c : lod_layer_) { any_idx |= c.idx_route; any_pool |= c.pool_sel; }
        lod_.idx_route = any_idx; lod_.pool_sel = any_pool;   // the global flags now mean "allocated"; lod_of(il) decides per layer
        std::fprintf(stderr, "glm5: LoD per-layer routes: %s\n", ev);
    }
    if (const char* ev = std::getenv("STRATA_GLM5_LOD_QGROUPS")) lod_.qgroups = std::max(1, std::atoi(ev));
    if (lod_.on) {
        if (max_ctx / lod_.ps + 1 > 8192) throw std::runtime_error("glm5: LoD: at most 8192 pages (raise the page size for this context)");
        if (hp_.n_head % lod_.qgroups) lod_.qgroups = 1;
        std::fprintf(stderr, "glm5: LoD attention: page %d, top %d (decode %d), local %d, sink %d page(s), routing by %s\n", lod_.ps, lod_.top_prefill, lod_.top_decode, lod_.local, lod_.sink_pages,
                     lod_.pool_sel ? "the DSA indexer's pools, selected per pool" : lod_.idx_route ? "the DSA indexer's pools" : lod_.tok_route ? "tokens (head groups)" : "page means");
    }
    // the per-token selection lists: the indexer's top-k cells + a pool, or LoD's exact pages + local field + summaries
    n_sel_max_ = hp_.idx_topk + hp_.kpool;
    if (lod_.on) n_sel_max_ = std::max(n_sel_max_, (std::max(lod_.top_prefill, lod_.top_decode) + lod_.sink_pages) * lod_.ps + lod_.local + lod_.ps + max_ctx / lod_.ps + 1);
    if (lod_.on && lod_.pool_sel) n_sel_max_ = std::max(n_sel_max_, std::max(lod_.top_pools, lod_.top_pools_decode) * hp_.kpool + lod_.sink_pages * lod_.ps + lod_.local + lod_.ps + max_ctx / lod_.ps + 1);
    const int n_sel_max = n_sel_max_;
    uint64_t widest = 0;
    for (int il = hp_.n_dense_lead; il < NL; ++il) {
        const LayerWeights& L = m.layer(il);
        widest = std::max(widest, L.exp_gate.stride + L.exp_up.stride + L.exp_down.stride);
    }
    expert_stage_bytes_ = widest;
    for (int a : m.devices())
        for (int b : m.devices())
            if (a != b) {
                int can = 0;
                if (cudaDeviceCanAccessPeer(&can, a, b) == cudaSuccess && can) {
                    cudaSetDevice(a);
                    cudaDeviceEnablePeerAccess(b, 0);
                    cudaGetLastError();
                }
            }
    const int head_ord = m.layer_device()[(size_t) hp_.n_layer - 1];
    for (int ordinal : m.devices()) {
        Dev d;
        d.ordinal = ordinal;
        check(cudaSetDevice(ordinal), "cudaSetDevice");
        cudaStream_t s;
        check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate");
        d.stream = s;
        cudaEvent_t ev;
        check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        d.event = ev;
        auto f = [&](uint64_t n) { return static_cast<float*>(alloc(n * 4, ordinal)); };
        // the prompt scratch a decode window (<= 8 rows) leaves idle is lent to the decode-time expert cache: a buffer
        // whole, or past the rows a decode window writes (16 rows' worth, at least 4 MB; the lender keeps that head)
        auto lend = [&](void* p, uint64_t bytes, uint64_t row_bytes) {
            if (p && bytes) d.lend.push_back({static_cast<uint8_t*>(p), bytes, row_bytes ? std::max<uint64_t>(16 * row_bytes, 4u << 20) : 0});
        };
        d.R = f((uint64_t) TS * hp_.hc * E);
        d.x = f((uint64_t) TS * E); d.xn = f((uint64_t) T * E); d.y = f((uint64_t) T * E);
        d.mixes = f((uint64_t) T * hp_.hc_mix_dim()); d.part = f((uint64_t) T * hp_.hc_mix_dim() * mhc_mix_slices()); d.rs = f(T);
        d.pre = f((uint64_t) T * hp_.hc); d.post = f((uint64_t) T * hp_.hc); d.comb = f((uint64_t) T * hp_.hc * hp_.hc);
        d.big_a = f((uint64_t) T * big); d.big_b = f((uint64_t) T * big); d.big_c = f((uint64_t) T * big); d.big_d = f((uint64_t) T * big);
        for (float* b : {d.big_a, d.big_b, d.big_c, d.big_d}) lend(b, (uint64_t) T * big * 4, (uint64_t) big * 4);
        d.qkv = f((uint64_t) std::min(T, kQkvFuseT) * 3 * D); d.fab = f((uint64_t) kFabT * (2 * hp_.kda_head_dim + hp_.n_head));
        d.sh_g = f((uint64_t) std::min(T, kShT) * 2 * FF * std::max(1, hp_.n_expert_shared)); d.sh_h = f((uint64_t) std::min(T, kShT) * FF * std::max(1, hp_.n_expert_shared));
        d.q8_sh = static_cast<uint8_t*>(alloc((uint64_t) std::min(T, kShT) * q8_cols(FF * std::max(1, hp_.n_expert_shared)), ordinal));
        d.dense_gu = f((uint64_t) std::min(T, kShT) * 2 * hp_.n_ff);
        const int sm = std::max(hp_.q_lora, hp_.kv_lora);
        d.small_a = f((uint64_t) T * sm); d.small_b = f((uint64_t) T * sm);
        const uint64_t ffrows = std::max<uint64_t>((uint64_t) pairs_alloc_ * FF, (uint64_t) T * std::max(hp_.n_ff, FF * std::max(1, hp_.n_expert_shared)));
        d.ffn_g = f(ffrows); d.ffn_u = f(ffrows); d.ffn_h = f(ffrows);
        for (float* b : {d.ffn_g, d.ffn_u, d.ffn_h}) lend(b, ffrows * 4, (uint64_t) std::max(hp_.n_ff, K * FF) * 4);
        d.ffn_y = f((uint64_t) pairs_alloc_ * E); d.ffn_acc = f((uint64_t) T * E); d.ffn_d = f((uint64_t) T * E);
        lend(d.ffn_y, (uint64_t) pairs_alloc_ * E * 4, (uint64_t) K * E * 4);
        d.logits = f(std::max((uint64_t) snap_T_ * hp_.n_vocab, (uint64_t) T * hp_.n_expert));   // the verify window's full logits; a prompt window keeps the last token's only
        d.probs = f((uint64_t) T * hp_.n_expert); d.wts = f((uint64_t) T * K);
        d.h_win = f((uint64_t) TS * E);
        d.ids = static_cast<int32_t*>(alloc((uint64_t) T * K * 4, ordinal));
        d.q8_x = static_cast<uint8_t*>(alloc((uint64_t) T * q8_cols(E), ordinal));
        d.q8_g = static_cast<uint8_t*>(alloc((uint64_t) max_pairs_ * q8_cols(E), ordinal));
        d.q8_h = static_cast<uint8_t*>(alloc((uint64_t) max_pairs_ * q8_cols(FF), ordinal));
        lend(d.q8_g, (uint64_t) max_pairs_ * q8_cols(E), (uint64_t) K * q8_cols(E));
        lend(d.q8_h, (uint64_t) max_pairs_ * q8_cols(FF), (uint64_t) q8_cols(hp_.n_ff));
        d.q8_big = static_cast<uint8_t*>(alloc((uint64_t) T * q8_cols(big), ordinal));
        d.q_abs = f((uint64_t) T * hp_.n_head * hp_.kv_lora);
        lend(d.q_abs, (uint64_t) T * hp_.n_head * hp_.kv_lora * 4, (uint64_t) hp_.n_head * hp_.kv_lora * 4);
        d.iq = f((uint64_t) T * hp_.idx_heads * hp_.idx_dim); d.ik = f((uint64_t) T * hp_.idx_dim); d.ig = f((uint64_t) T * hp_.idx_dim); d.iw = f((uint64_t) T * hp_.idx_heads);
        d.score = f((uint64_t) kDsaBatch * ((uint64_t) max_ctx / hp_.kpool + 1));
        d.attn_scores = f((uint64_t) kDsaBatch * hp_.n_head * n_sel_max);
        lend(d.attn_scores, (uint64_t) kDsaBatch * hp_.n_head * n_sel_max * 4, (uint64_t) hp_.n_head * n_sel_max * 4);
        d.ctx = f((uint64_t) T * hp_.n_head * hp_.kv_lora);
        lend(d.ctx, (uint64_t) T * hp_.n_head * hp_.kv_lora * 4, (uint64_t) hp_.n_head * hp_.kv_lora * 4);
        d.sel = static_cast<int32_t*>(alloc((uint64_t) kDsaBatch * n_sel_max * 4, ordinal));
        d.sel_params = static_cast<int*>(alloc((uint64_t) T * 8 * 4, ordinal));
        d.expert_stage = nullptr;   // the no-store path allocates it on first use (min(max_pairs, n_expert) slots)
        d.neg_table = static_cast<int32_t*>(alloc((uint64_t) hp_.n_expert * 4, ordinal));
        check(cudaMemset(d.neg_table, 0xff, (size_t) hp_.n_expert * 4), "neg table");
        d.grp_w = static_cast<const void**>(alloc((uint64_t) max_pairs_ * sizeof(void*), ordinal));
        d.grp_wu = static_cast<const void**>(alloc((uint64_t) max_pairs_ * sizeof(void*), ordinal));
        d.grp_wd = static_cast<const void**>(alloc((uint64_t) max_pairs_ * sizeof(void*), ordinal));
        d.grp_col0 = static_cast<int*>(alloc((uint64_t) max_pairs_ * 4, ordinal));
        d.grp_ncols = static_cast<int*>(alloc((uint64_t) max_pairs_ * 4, ordinal));
        d.grp_orow = static_cast<int*>(alloc((uint64_t) max_pairs_ * 4, ordinal));
        d.pair_tok = static_cast<int*>(alloc((uint64_t) pairs_alloc_ * 4, ordinal));
        d.tok_pairs = static_cast<int*>(alloc((uint64_t) pairs_alloc_ * 4, ordinal));
        check(cudaMemset(d.pair_tok, 0, (uint64_t) pairs_alloc_ * 4), "pair_tok");
        d.pair_w = f(max_pairs_);
        d.cnt = static_cast<int*>(alloc(8 * 4, ordinal));
        d.copy_cap = 4 * NL + 4;   // (the rollback table itself is made at the first rollback)
        d.grp_e = static_cast<int*>(alloc((uint64_t) max_pairs_ * 4, ordinal));
        d.grp_miss = static_cast<int*>(alloc((uint64_t) max_pairs_ * 4, ordinal));
        d.argmax_out = static_cast<int32_t*>(alloc((uint64_t) T * 4, ordinal));
        d.am_idx = static_cast<int*>(alloc((uint64_t) T * 128 * 4, ordinal));
        d.am_val = f((uint64_t) T * 128);
        d.tok_rows = static_cast<int32_t*>(alloc((uint64_t) TS * 4, ordinal));
        d.row_slot_d = static_cast<int*>(alloc(64 * 4, ordinal)); d.row_pos_d = static_cast<int*>(alloc(64 * 4, ordinal));
        d.row_slot_h = static_cast<int*>(alloc_host(64 * 4)); d.row_pos_h = static_cast<int*>(alloc_host(64 * 4));
        d.row_k_d = static_cast<int*>(alloc(64 * 4, ordinal)); d.grp_slot_d = static_cast<int*>(alloc(64 * 4, ordinal)); d.grp_start_d = static_cast<int*>(alloc(64 * 4, ordinal)); d.grp_len_d = static_cast<int*>(alloc(64 * 4, ordinal));
        d.row_k_h = static_cast<int*>(alloc_host(64 * 4)); d.grp_slot_h = static_cast<int*>(alloc_host(64 * 4)); d.grp_start_h = static_cast<int*>(alloc_host(64 * 4)); d.grp_len_h = static_cast<int*>(alloc_host(64 * 4));
        if (T >= kGemmMinT) {
            // the tensor-core GEMM of wide (prompt) windows: a dequantization scratch of the largest dense weight
            auto* g = new strata::prefill::Gemm();
            std::string err;
            if (!g->init(d.stream, (int64_t) big * E, err)) throw std::runtime_error("glm5: prefill gemm: " + err);
            d.gemm = g;
            d.x16 = static_cast<uint16_t*>(alloc((uint64_t) T * big * 2, ordinal));
            lend(d.x16, (uint64_t) T * big * 2, 0);
            lend(g->scratch(), (uint64_t) g->scratch_elems() * 2, 0);
            // the BLAS library's first call costs ~170 ms (its kernel library loads): paid here, not by the first request
            check(cudaMemsetAsync(d.x16, 0, 64 * 64 * 2 * 2, S(d.stream)), "warm-up memset");
            g->bf16(d.x16, d.x16 + 64 * 64, d.big_a, 16, 64, 64);
            g->f16(d.x16, d.x16 + 64 * 64, d.big_a, 16, 64, 64);
            check(cudaStreamSynchronize(S(d.stream)), "warm-up sync");
        }
        if (T >= kMmqMinT && !kNoMmq && strata::prefill::mmq::built()) {
            // the MMQ expert path: the window's experts gathered into [n_expert] contiguous slots (the widest layer's sizes)
            uint64_t gu = 0, dn = 0;
            bool ok = true;
            for (int il = hp_.n_dense_lead; il < NL; ++il) {
                const LayerWeights& L = m.layer(il);
                gu = std::max(gu, L.exp_gate.stride + L.exp_up.stride);
                dn = std::max(dn, L.exp_down.stride);
                ok = ok && strata::prefill::mmq::fits((int) L.exp_gate.type, 2 * FF) && strata::prefill::mmq::fits((int) L.exp_down.type, E);
            }
            (void) gu; (void) dn;
            if (ok) {
                d.bounds_res = static_cast<int*>(alloc((uint64_t) (hp_.n_expert + 1) * 4, ordinal));
                d.bounds_miss = static_cast<int*>(alloc((uint64_t) (hp_.n_expert + 1) * 4, ordinal));
                d.mmq_xq = alloc(strata::prefill::mmq::q8_bytes(pairs_alloc_, E), ordinal);
                d.mmq_xq_h = alloc(strata::prefill::mmq::q8_bytes(pairs_alloc_, FF), ordinal);
                d.gu_out = f((uint64_t) pairs_alloc_ * 2 * FF);
                lend(d.gu_out, (uint64_t) pairs_alloc_ * 2 * FF * 4, 0);
                if (!(std::getenv("STRATA_GLM5_NOQ2K") && std::atoi(std::getenv("STRATA_GLM5_NOQ2K")) != 0)) {   // the Q2_K expert products on WMMA (gfx11); STRATA_GLM5_NOQ2K=1: the MMQ path
                    d.q2k_codes = static_cast<int8_t*>(alloc((uint64_t) pairs_alloc_ * E, ordinal));
                    lend(d.q2k_codes, (uint64_t) pairs_alloc_ * E, (uint64_t) K * E);
                    d.q2k_scales = f((uint64_t) pairs_alloc_ * (E / 256));
                    d.q2k_sums = static_cast<int32_t*>(alloc((uint64_t) pairs_alloc_ * (E / 16) * 4, ordinal));
                    // the decode windows (at most 64 pairs) on the same kernel family: packed sums, and the swiglu rows' operands
                    d.q2k_packed = static_cast<int32_t*>(alloc((uint64_t) pairs_alloc_ * (E / 256) * 8 * 4, ordinal));   // (E/256 >= FF/256: the down product's fit too)
                    d.dec_codes_h = static_cast<int8_t*>(alloc((uint64_t) 64 * FF, ordinal));
                    d.dec_scales_h = f((uint64_t) 64 * (FF / 256));
                    d.dec_sums_h = static_cast<int32_t*>(alloc((uint64_t) 64 * (FF / 16) * 4, ordinal));
                    d.dec_packed_h = static_cast<int32_t*>(alloc((uint64_t) 64 * (FF / 256) * 8 * 4, ordinal));
                }
                d.iota = static_cast<int32_t*>(alloc((uint64_t) pairs_alloc_ * 4, ordinal));
                { std::vector<int32_t> h((size_t) pairs_alloc_); std::iota(h.begin(), h.end(), 0); check(cudaMemcpy(d.iota, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "iota"); }
                d.bounds = static_cast<int*>(alloc((uint64_t) (hp_.n_expert + 1) * 4, ordinal));
                d.mmq_xd = alloc(strata::prefill::mmq::q8_bytes(T + 256, big), ordinal);
                lend(d.mmq_xd, strata::prefill::mmq::q8_bytes(T + 256, big), 0);
                d.dense_bounds = static_cast<int32_t*>(alloc(8, ordinal));
                d.dense_bounds_h = static_cast<int32_t*>(alloc_host(64));
                d.head_xq = alloc(strata::prefill::mmq::q8_bytes((int64_t) T * hp_.n_head + 256, std::max(hp_.head_k_mla, hp_.kv_lora)), ordinal);
                lend(d.head_xq, strata::prefill::mmq::q8_bytes((int64_t) T * hp_.n_head + 256, std::max(hp_.head_k_mla, hp_.kv_lora)), 0);
                d.head_ids = static_cast<int32_t*>(alloc((uint64_t) T * hp_.n_head * 4, ordinal));
                d.head_bounds = static_cast<int32_t*>(alloc((uint64_t) (hp_.n_head + 1) * 4, ordinal));
                {
                    const int64_t rows = (int64_t) kDsaBatch * hp_.n_head, npad = kDenseAttnMaxKv, KV = hp_.kv_lora;
                    d.kvcT_q8 = static_cast<uint8_t*>(alloc((uint64_t) KV * (npad / 32) * 34, ordinal));
                    lend(d.kvcT_q8, (uint64_t) KV * (npad / 32) * 34, 0);
                    if (lod_.on) {
                        d.lod_kv_q8 = static_cast<uint8_t*>(alloc((uint64_t) (npad + 512) * q8_pitch(KV), ordinal));
                        d.lod_bias = f((uint64_t) npad + 512);
                        d.lod_score = f((uint64_t) max_ctx / lod_.ps + 512);
                        d.lod_tok = f((uint64_t) max_ctx + 512);
                        d.lod_qg = f((uint64_t) T * lod_.qgroups * KV);
                        d.lod_part = f((uint64_t) 8 * hp_.n_head * 128 * (2 + KV));   // nsplit up to 128
                        d.lod_pscore = f((uint64_t) 8 * (max_ctx / lod_.ps + 512));
                        d.lod_iqp = f((uint64_t) T * hp_.idx_heads * 256);
                        d.sel_params2 = static_cast<int*>(alloc((uint64_t) T * 8 * 4, ordinal));
                    }
                    d.attn_S = f((uint64_t) rows * npad);
                    lend(d.attn_S, (uint64_t) rows * npad * 4, 0);
                    d.attn_mask = static_cast<uint32_t*>(alloc((uint64_t) kDsaBatch * (npad / 32) * 4, ordinal));
                    d.attn_xq = alloc(strata::prefill::mmq::q8_bytes(rows + 256, KV), ordinal);
                    lend(d.attn_xq, strata::prefill::mmq::q8_bytes(rows + 256, KV), 0);
                    d.attn_pq = alloc(strata::prefill::mmq::q8_bytes(rows + 256, npad), ordinal);
                    lend(d.attn_pq, strata::prefill::mmq::q8_bytes(rows + 256, npad), 0);
                    d.attn_bounds = static_cast<int32_t*>(alloc(8, ordinal));
                    d.attn_bounds_h = static_cast<int32_t*>(alloc_host(64));
                }
                d.mmq = new strata::prefill::mmq::Context();
            } else std::fprintf(stderr, "glm5: MMQ has no tile for these expert types on this GPU: prompt windows use the grouped MMVQ\n");
        }
        d.ids_host = static_cast<int32_t*>(alloc_host((uint64_t) TS * K * 4 + 64));
        for (int p = 0; p < 2; ++p) { d.ids_host2[p] = static_cast<int32_t*>(alloc_host((uint64_t) TS * 4 + 64)); cudaEvent_t e2; check(cudaEventCreateWithFlags(&e2, cudaEventDisableTiming), "cudaEventCreate"); d.ev2[p] = e2; cudaEvent_t e3; check(cudaEventCreateWithFlags(&e3, cudaEventDisableTiming), "cudaEventCreate"); d.ev_in2[p] = e3; }
        d.pf_mutex = std::make_unique<std::mutex>();
        d.cnt_host = static_cast<int*>(alloc_host(64));
        d.sel_params_host = static_cast<int*>(alloc_host((uint64_t) T * 8 * 4));
        d.sel_params2_host = static_cast<int*>(alloc_host((uint64_t) T * 8 * 4));
        d.bounce = static_cast<float*>(alloc_host((uint64_t) TS * hp_.hc * E * 4));
        d.bounce2[0] = d.bounce; d.bounce2[1] = static_cast<float*>(alloc_host((uint64_t) TS * hp_.hc * E * 4));
        {
            void* dv = nullptr;
            d.db_ids_h = static_cast<int32_t*>(alloc_mapped((uint64_t) NL * T * K * 4, ordinal, &dv)); d.db_ids_d = static_cast<int32_t*>(dv);
            d.db_pub_h = static_cast<uint32_t*>(alloc_mapped((uint64_t) NL * 4, ordinal, &dv)); d.db_pub_d = static_cast<uint32_t*>(dv);
            d.db_flag_h = static_cast<uint32_t*>(alloc_mapped((uint64_t) NL * 4, ordinal, &dv)); d.db_flag_d = static_cast<uint32_t*>(dv);
            d.tok_dev = static_cast<uint32_t*>(alloc(4, ordinal));
            d.db_x_h = static_cast<float*>(alloc_mapped((uint64_t) 2 * kDbXRows * E * 4, ordinal, &dv)); d.db_x_d = static_cast<float*>(dv);
            d.pair_rank = static_cast<int*>(alloc((uint64_t) pairs_alloc_ * 4, ordinal));
            d.cpu_skip_h = static_cast<int32_t*>(alloc_mapped((uint64_t) NL * hp_.n_expert * 4, ordinal, &dv)); d.cpu_skip_d = static_cast<int32_t*>(dv);
            d.miss_cpu_h = static_cast<int*>(alloc_mapped((uint64_t) NL * 64 * 4, ordinal, &dv)); d.miss_cpu_d = static_cast<int*>(dv);
            d.db_flag2_h = static_cast<uint32_t*>(alloc_mapped((uint64_t) NL * 4, ordinal, &dv)); d.db_flag2_d = static_cast<uint32_t*>(dv);
            d.miss_slot_h = static_cast<int*>(alloc_mapped((uint64_t) NL * T * K * 4, ordinal, &dv)); d.miss_slot_d = static_cast<int*>(dv);
            d.miss_out_h = static_cast<int*>(alloc_mapped((uint64_t) NL * T * K * 4, ordinal, &dv)); d.miss_out_d = static_cast<int*>(dv);
            const int P = 16;
            d.pred_logits = f((uint64_t) T * hp_.n_expert); d.pred_probs = f((uint64_t) T * hp_.n_expert); d.pred_wts = f((uint64_t) T * P);
            d.pred_ids = static_cast<int32_t*>(alloc((uint64_t) T * P * 4, ordinal));
            d.db_pred_h = static_cast<int32_t*>(alloc_mapped((uint64_t) NL * T * P * 4, ordinal, &dv)); d.db_pred_d = static_cast<int32_t*>(dv);
            d.db_pred_pub_h = static_cast<uint32_t*>(alloc_mapped((uint64_t) NL * 4, ordinal, &dv)); d.db_pred_pub_d = static_cast<uint32_t*>(dv);
            d.pf_flag_h = static_cast<uint32_t*>(alloc_mapped((uint64_t) NL * 4, ordinal, &dv)); d.pf_flag_d = static_cast<uint32_t*>(dv);
            d.pf_n_h = static_cast<int*>(alloc_mapped((uint64_t) NL * 4, ordinal, &dv)); d.pf_n_d = static_cast<int*>(dv);
            d.pf_e_h = static_cast<int*>(alloc_mapped((uint64_t) NL * T * P * 4, ordinal, &dv)); d.pf_e_d = static_cast<int*>(dv);
            d.pf_slot_h = static_cast<int*>(alloc_mapped((uint64_t) NL * T * P * 4, ordinal, &dv)); d.pf_slot_d = static_cast<int*>(dv);
            d.pf_out_h = static_cast<int*>(alloc_mapped((uint64_t) NL * T * P * 4, ordinal, &dv)); d.pf_out_d = static_cast<int*>(dv);
        }
        if (ordinal == head_ord && m.has_mtp()) {
            d.mtp_h = f((uint64_t) T * E); d.mtp_e = f((uint64_t) T * E); d.mtp_cat = f((uint64_t) T * 2 * E); d.mtp_x = f((uint64_t) T * E); d.mtp_x1 = f((uint64_t) T * E);
        }
        devs_.push_back(std::move(d));
    }
    dev_of_layer_.resize((size_t) NL);
    for (int il = 0; il < NL; ++il) {
        const int ord = m.layer_device()[(size_t) il];
        dev_of_layer_[(size_t) il] = (int) (std::find(m.devices().begin(), m.devices().end(), ord) - m.devices().begin());
    }
    NL_ = NL;
    const size_t n_states = (size_t) (n_slots_ + 1) * NL;
    conv_state_.assign(n_states, nullptr); ssm_state_.assign(n_states, nullptr);
    moe_seq_.assign((size_t) NL, -1);
    { std::vector<int> per_dev(devs_.size(), 0); for (int il = hp_.n_dense_lead; il < hp_.n_layer; ++il) moe_seq_[(size_t) il] = per_dev[(size_t) dev_of_layer_[(size_t) il]]++; n_dev_moe_ = per_dev; }
    ring_next_.assign(devs_.size(), 0);
    conv_snap_.assign((size_t) NL, nullptr); ssm_snap_.assign((size_t) NL, nullptr);
    conv_mark_.assign(n_states, nullptr); ssm_mark_.assign(n_states, nullptr);
    kvc_cache_.assign(n_states, nullptr); ik_cache_.assign(n_states, nullptr);
    ig_cache_.assign(n_states, nullptr); pooled_.assign(n_states, nullptr); pmean_.assign(n_states, nullptr); pooled_q8_.assign(n_states, nullptr);
    pos_s_.assign((size_t) n_slots_ + 1, 0); mark_pos_s_.assign((size_t) n_slots_ + 1, 0); h_mark_s_.assign((size_t) n_slots_ + 1, nullptr);
    row_slot_.assign((size_t) TS, 0); row_pos_.assign((size_t) TS, 0); row_k_.assign((size_t) TS, 0);
    grp_slot_.assign((size_t) T, 0); grp_start_.assign((size_t) T, 0); grp_len_.assign((size_t) T, 0);
    const int Sd = hp_.kda_head_dim;
    uint64_t slot_bytes = 0;
    for (int slot = 0; slot <= n_slots_; ++slot) {
        for (int il = 0; il < NL; ++il) {
            const int ord = m.layer_device()[(size_t) il];
            auto f = [&](uint64_t n) { if (slot == 0) slot_bytes += n * 4; return static_cast<float*>(alloc(n * 4, ord)); };
            const size_t k = st(slot, il);
            if (hp_.is_recr[(size_t) il]) {
                conv_state_[k] = f((uint64_t) 3 * (hp_.d_conv - 1) * D);
                ssm_state_[k] = f((uint64_t) hp_.n_head * Sd * Sd);
                if (slot == n_slots_ && snap_T_ > 1) {
                    conv_snap_[(size_t) il] = f((uint64_t) snap_T_ * 3 * (hp_.d_conv - 1) * D);
                    ssm_snap_[(size_t) il] = f((uint64_t) snap_T_ * hp_.n_head * Sd * Sd);
                }
                conv_mark_[k] = f((uint64_t) 3 * (hp_.d_conv - 1) * D);
                ssm_mark_[k] = f((uint64_t) hp_.n_head * Sd * Sd);
            } else {
                // Q8_0 rows; 512 spare rows: the prompt windows' dense attention reads the cache as an MMQ operand in whole tiles
                if (hp_.kv_lora % 32 || hp_.idx_dim % 32) throw std::runtime_error("glm5: the DSA cache widths must be multiples of 32");
                auto h = [&](uint64_t rows, int n) { const uint64_t b = rows * (uint64_t) q8_pitch(n); if (slot == 0) slot_bytes += b; return static_cast<uint8_t*>(alloc(b, ord)); };
                kvc_cache_[k] = h((uint64_t) max_ctx + 512, hp_.kv_lora);
                ik_cache_[k] = h((uint64_t) max_ctx + 512, hp_.idx_dim);
                ig_cache_[k] = h((uint64_t) max_ctx + 512, hp_.idx_dim);
                pooled_[k] = f((uint64_t) (max_ctx / hp_.kpool + 1) * hp_.idx_dim);
                if (lod_.on && lod_.idx_route) {   // the pooled indexer keys as a Q8_0 MMQ operand (256 columns, zero-padded)
                    const uint64_t prow = (uint64_t) ((max_ctx / hp_.kpool + 511) / 512) * 512 + 512;
                    pooled_q8_[k] = static_cast<uint8_t*>(alloc(prow * 8 * 34, ord));
                    check(cudaMemset(pooled_q8_[k], 0, prow * 8 * 34), "pooled q8 zero");
                }
                if (lod_.on) {   // the page means (zeroed; the MMQ scoring reads them in 512-row operands plus a tile beyond)
                    const uint64_t rows = (uint64_t) ((max_ctx / lod_.ps + 511) / 512) * 512 + 512;
                    pmean_[k] = h(rows, hp_.kv_lora);
                    check(cudaMemset(pmean_[k], 0, rows * (uint64_t) q8_pitch(hp_.kv_lora)), "pmean zero");
                }
            }
        }
        h_mark_s_[(size_t) slot] = static_cast<float*>(alloc((uint64_t) E * 4, head_ord));
    }
    // the per-layer slot tables (device): the batch windows' kernels index them by the row's slot
    conv_tab_.assign((size_t) NL, nullptr); ssm_tab_.assign((size_t) NL, nullptr); kvc_tab_.assign((size_t) NL, nullptr);
    ik_tab_.assign((size_t) NL, nullptr); ig_tab_.assign((size_t) NL, nullptr); pooled_tab_.assign((size_t) NL, nullptr); pmean_tab_.assign((size_t) NL, nullptr); pooled_q8_tab_.assign((size_t) NL, nullptr);
    for (int il = 0; il < NL; ++il) {   // every layer incl. the MTP's (a DSA layer: LoD's row kernels index its tables too)
        const int ord = m.layer_device()[(size_t) il];
        auto tab = [&](const auto& v) {
            using P = typename std::decay_t<decltype(v)>::value_type;
            std::vector<P> h((size_t) n_slots_ + 1);
            for (int slot = 0; slot <= n_slots_; ++slot) h[(size_t) slot] = v[st(slot, il)];
            P* dptr = static_cast<P*>(alloc(h.size() * sizeof(P), ord));
            check(cudaMemcpy(dptr, h.data(), h.size() * sizeof(P), cudaMemcpyHostToDevice), "slot table");
            return dptr;
        };
        if (il < hp_.n_layer && hp_.is_recr[(size_t) il]) { conv_tab_[(size_t) il] = tab(conv_state_); ssm_tab_[(size_t) il] = tab(ssm_state_); }
        else { kvc_tab_[(size_t) il] = tab(kvc_cache_); ik_tab_[(size_t) il] = tab(ik_cache_); ig_tab_[(size_t) il] = tab(ig_cache_); pooled_tab_[(size_t) il] = tab(pooled_); if (lod_.on) pmean_tab_[(size_t) il] = tab(pmean_); if (lod_.on && lod_.idx_route) pooled_q8_tab_[(size_t) il] = tab(pooled_q8_); }
    }
    if (n_slots_ > 0) std::fprintf(stderr, "glm5: %d batch slots + the solo slot, %.2f GiB of state each (context %d)\n", n_slots_, slot_bytes / 1073741824.0, max_ctx);
}

// a batch decode window: one token per slot, each row on its slot's state at its slot's position
void Engine::forward_rows(const int32_t* tokens, const int* slots, int n_rows) {
    if (n_rows < 1 || n_rows > 8 || n_rows > t_max_) throw std::runtime_error("glm5: forward_rows: " + std::to_string(n_rows) + " rows (1..8)");
    // the rows in runs per slot: run g = rows [grp_start, +grp_len) of slot grp_slot, at positions pos .. pos + len - 1
    n_grp_ = 0;
    for (int t = 0; t < n_rows; ++t) {
        const int sl = slots[t];
        if (sl < 0 || sl > n_slots_) throw std::runtime_error("glm5: forward_rows: no slot " + std::to_string(sl));
        if (t > 0 && slots[t - 1] == sl) { row_k_[(size_t) t] = row_k_[(size_t) t - 1] + 1; ++grp_len_[(size_t) n_grp_ - 1]; }
        else {
            for (int g = 0; g < n_grp_; ++g) if (grp_slot_[(size_t) g] == sl) throw std::runtime_error("glm5: forward_rows: slot " + std::to_string(sl) + "'s rows are not consecutive");
            grp_slot_[(size_t) n_grp_] = sl; grp_start_[(size_t) n_grp_] = t; grp_len_[(size_t) n_grp_] = 1; ++n_grp_;
            row_k_[(size_t) t] = 0;
        }
        row_slot_[(size_t) t] = sl;
        row_pos_[(size_t) t] = pos_s_[(size_t) sl] + row_k_[(size_t) t];
        if (row_pos_[(size_t) t] + 1 > max_ctx_) throw std::runtime_error("glm5: forward_rows: a slot's context is full");
    }
    for (auto& d : devs_) {
        check(cudaSetDevice(d.ordinal), "cudaSetDevice");
        for (int t = 0; t < n_rows; ++t) { d.row_slot_h[t] = row_slot_[(size_t) t]; d.row_pos_h[t] = row_pos_[(size_t) t]; d.row_k_h[t] = row_k_[(size_t) t]; }
        for (int g = 0; g < n_grp_; ++g) { d.grp_slot_h[g] = grp_slot_[(size_t) g]; d.grp_start_h[g] = grp_start_[(size_t) g]; d.grp_len_h[g] = grp_len_[(size_t) g]; }
        check(cudaMemcpyAsync(d.row_slot_d, d.row_slot_h, (size_t) n_rows * 4, cudaMemcpyHostToDevice, S(d.stream)), "row slots");
        check(cudaMemcpyAsync(d.row_pos_d, d.row_pos_h, (size_t) n_rows * 4, cudaMemcpyHostToDevice, S(d.stream)), "row positions");
        check(cudaMemcpyAsync(d.row_k_d, d.row_k_h, (size_t) n_rows * 4, cudaMemcpyHostToDevice, S(d.stream)), "row k");
        check(cudaMemcpyAsync(d.grp_slot_d, d.grp_slot_h, (size_t) n_grp_ * 4, cudaMemcpyHostToDevice, S(d.stream)), "grp slots");
        check(cudaMemcpyAsync(d.grp_start_d, d.grp_start_h, (size_t) n_grp_ * 4, cudaMemcpyHostToDevice, S(d.stream)), "grp starts");
        check(cudaMemcpyAsync(d.grp_len_d, d.grp_len_h, (size_t) n_grp_ * 4, cudaMemcpyHostToDevice, S(d.stream)), "grp lens");
    }
    rows_mode_ = true;
    cur_slot_ = n_slots_;   // the MTP / solo-only paths are not taken; st() stays valid
    pos_ = 0;               // the window's own position bookkeeping is per row
    try { forward_window_impl(tokens, n_rows, DumpFn()); } catch (...) { rows_mode_ = false; faulted_ = true; throw; }
    rows_mode_ = false;
    pos_ = pos_s_[(size_t) n_slots_];
}

// keep the first n_keep[g] rows of each slot run (1..len): a shorter keep rolls the slot's KDA states back to the snapshot
// after its last kept row (the attention caches are positional and need none); the positions advance by n_keep
void Engine::commit_rows(const int* n_keep, int n_grp) {
    if (n_grp != n_grp_) throw std::runtime_error("glm5: commit_rows: " + std::to_string(n_grp) + " groups, the window had " + std::to_string(n_grp_));
    const int D = hp_.d_inner(), Sd = hp_.kda_head_dim;
    const size_t ssm_n = (size_t) hp_.n_head * Sd * Sd, cw = (size_t) (hp_.d_conv - 1) * D, snap_stride = (size_t) win_T_ * cw;
    for (int g = 0; g < n_grp; ++g) {
        const int sl = grp_slot_[(size_t) g], len = grp_len_[(size_t) g], keep = n_keep[g];
        if (keep < 1 || keep > len) throw std::runtime_error("glm5: commit_rows: keep " + std::to_string(keep) + " of a run of " + std::to_string(len));
        if (keep < len) {
            const int row = grp_start_[(size_t) g] + keep - 1;
            for (int il = 0; il < hp_.n_layer; ++il) {
                if (!hp_.is_recr[(size_t) il]) continue;
                Dev& d = devs_[(size_t) dev_of_layer_[(size_t) il]];
                check(cudaSetDevice(d.ordinal), "cudaSetDevice");
                check(cudaMemcpyAsync(ssm_state_[st(sl, il)], ssm_snap_[(size_t) il] + (size_t) row * ssm_n, ssm_n * 4, cudaMemcpyDeviceToDevice, S(d.stream)), "rows state rollback");
                for (int which = 0; which < 3; ++which)
                    check(cudaMemcpyAsync(conv_state_[st(sl, il)] + which * cw, conv_snap_[(size_t) il] + which * snap_stride + (size_t) row * cw, cw * 4, cudaMemcpyDeviceToDevice, S(d.stream)), "rows conv rollback");
            }
        }
        pos_s_[(size_t) sl] += keep;
    }
    pos_ = pos_s_[(size_t) n_slots_];
}

Engine::~Engine() {
    { std::lock_guard<std::mutex> lk(stage_m_); stage_stop_ = true; }
    stage_cv_.notify_all();
    for (auto& t : stage_th_) if (t.joinable()) t.join();
    db_stop_.store(true, std::memory_order_release);
    if (db_thread_.joinable()) { if (faulted_) db_thread_.detach(); else db_thread_.join(); }
    // after a fault the streams may hold a doorbell wait that never ends: touching the device would hang
    if (faulted_) return;
    for (auto& d : devs_) { cudaSetDevice(d.ordinal); cudaStreamSynchronize(S(d.stream)); }
    for (size_t i = 0; i < allocations_.size(); ++i) { cudaSetDevice(alloc_device_[i]); cudaFree(allocations_[i]); }
    for (auto& d : devs_) {
        cudaSetDevice(d.ordinal);
        cudaEventDestroy(static_cast<cudaEvent_t>(d.event));
        cudaStreamDestroy(S(d.stream));
    }
    for (void* p : host_allocations_) cudaFreeHost(p);
    for (auto& d : devs_) { delete static_cast<strata::prefill::Gemm*>(d.gemm); delete static_cast<strata::prefill::mmq::Context*>(d.mmq); }
}

void* Engine::head_stream() const { return devs_[(size_t) dev_of_layer_[(size_t) hp_.n_layer - 1]].stream; }
int Engine::head_device() const { return devs_[(size_t) dev_of_layer_[(size_t) hp_.n_layer - 1]].ordinal; }
const float* Engine::logits() const { return devs_[(size_t) dev_of_layer_[(size_t) hp_.n_layer - 1]].logits; }
const float* Engine::h_win() const { return devs_[(size_t) dev_of_layer_[(size_t) hp_.n_layer - 1]].h_win; }

void Engine::reset(int slot) {
    if (slot < 0) slot = n_slots_;
    for (int il = 0; il < hp_.n_layer; ++il) {
        Dev& d = devs_[(size_t) dev_of_layer_[(size_t) il]];
        cudaSetDevice(d.ordinal);
        if (hp_.is_recr[(size_t) il]) {
            cudaMemsetAsync(conv_state_[st(slot, il)], 0, (size_t) 3 * (hp_.d_conv - 1) * hp_.d_inner() * 4, S(d.stream));
            cudaMemsetAsync(ssm_state_[st(slot, il)], 0, (size_t) hp_.n_head * hp_.kda_head_dim * hp_.kda_head_dim * 4, S(d.stream));
        }
    }
    pos_s_[(size_t) slot] = 0;
    mark_pos_s_[(size_t) slot] = 0;
    if (slot == n_slots_) { pos_ = win_start_ = win_T_ = mtp_pos_ = 0; cur_slot_ = n_slots_; }
    else if (cur_slot_ == slot) pos_ = win_start_ = win_T_ = 0;
}

// slot dst takes slot src's conversation: the KDA states, the DSA cache rows up to its position (and the pools built
// from them), the position and the mark
void Engine::copy_slot(int src, int dst) {
    if (src == dst) return;
    const int D = hp_.d_inner(), Sd = hp_.kda_head_dim, KV = hp_.kv_lora, ID = hp_.idx_dim, KP = hp_.kpool;
    const int n = pos_s_[(size_t) src];
    const int n_copy = hp_.n_layer + (m_.has_mtp() ? 1 : 0);   // the MTP layer's (DSA) caches as well
    for (int il = 0; il < n_copy; ++il) {
        Dev& d = devs_[(size_t) dev_of_layer_[(size_t) il]];
        check(cudaSetDevice(d.ordinal), "cudaSetDevice");
        const size_t a = st(src, il), b = st(dst, il);
        if (il < hp_.n_layer && hp_.is_recr[(size_t) il]) {
            check(cudaMemcpyAsync(conv_state_[b], conv_state_[a], (size_t) 3 * (hp_.d_conv - 1) * D * 4, cudaMemcpyDeviceToDevice, S(d.stream)), "copy conv");
            check(cudaMemcpyAsync(ssm_state_[b], ssm_state_[a], (size_t) hp_.n_head * Sd * Sd * 4, cudaMemcpyDeviceToDevice, S(d.stream)), "copy ssm");
            if (mark_pos_s_[(size_t) src] > 0) {
                check(cudaMemcpyAsync(conv_mark_[b], conv_mark_[a], (size_t) 3 * (hp_.d_conv - 1) * D * 4, cudaMemcpyDeviceToDevice, S(d.stream)), "copy conv mark");
                check(cudaMemcpyAsync(ssm_mark_[b], ssm_mark_[a], (size_t) hp_.n_head * Sd * Sd * 4, cudaMemcpyDeviceToDevice, S(d.stream)), "copy ssm mark");
            }
        } else if (n > 0) {
            check(cudaMemcpyAsync(kvc_cache_[b], kvc_cache_[a], (size_t) n * q8_pitch(KV), cudaMemcpyDeviceToDevice, S(d.stream)), "copy kvc");
            check(cudaMemcpyAsync(ik_cache_[b], ik_cache_[a], (size_t) n * q8_pitch(ID), cudaMemcpyDeviceToDevice, S(d.stream)), "copy ik");
            check(cudaMemcpyAsync(ig_cache_[b], ig_cache_[a], (size_t) n * q8_pitch(ID), cudaMemcpyDeviceToDevice, S(d.stream)), "copy ig");
            if (n / KP > 0) check(cudaMemcpyAsync(pooled_[b], pooled_[a], (size_t) (n / KP) * ID * 4, cudaMemcpyDeviceToDevice, S(d.stream)), "copy pooled");
            if (lod_.on && n / lod_.ps > 0) check(cudaMemcpyAsync(pmean_[b], pmean_[a], (size_t) (n / lod_.ps) * q8_pitch(KV), cudaMemcpyDeviceToDevice, S(d.stream)), "copy page means");
            if (lod_.on && lod_.idx_route && n / KP > 0) check(cudaMemcpyAsync(pooled_q8_[b], pooled_q8_[a], (size_t) (n / KP) * (kIdxPad / 32) * 34, cudaMemcpyDeviceToDevice, S(d.stream)), "copy pooled q8");
        }
    }
    if (mark_pos_s_[(size_t) src] > 0) {
        Dev& hd = devs_[(size_t) dev_of_layer_[(size_t) hp_.n_layer - 1]];
        check(cudaSetDevice(hd.ordinal), "cudaSetDevice");
        check(cudaMemcpyAsync(h_mark_s_[(size_t) dst], h_mark_s_[(size_t) src], (size_t) hp_.n_embd * 4, cudaMemcpyDeviceToDevice, S(hd.stream)), "copy h mark");
    }
    pos_s_[(size_t) dst] = n;
    mark_pos_s_[(size_t) dst] = mark_pos_s_[(size_t) src];
}

void Engine::mark(int h_row) {
    if (win_T_ < 1 || h_row < 0 || h_row >= win_T_) throw std::runtime_error("glm5: mark(" + std::to_string(h_row) + ") outside the window of " + std::to_string(win_T_));
    const int D = hp_.d_inner(), Sd = hp_.kda_head_dim;
    const size_t ssm_n = (size_t) hp_.n_head * Sd * Sd, cw = (size_t) (hp_.d_conv - 1) * D;
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!hp_.is_recr[(size_t) il]) continue;
        Dev& d = devs_[(size_t) dev_of_layer_[(size_t) il]];
        check(cudaSetDevice(d.ordinal), "cudaSetDevice");
        // the states after window token h_row: the live state when it is the last token, else that token's snapshot
        const bool live = h_row + 1 == pos_ - win_start_;
        if (!live && win_T_ > snap_T_) throw std::runtime_error("glm5: mark: a window of " + std::to_string(win_T_) + " tokens has no snapshots");
        const float* ssm = live ? ssm_state_[st(il)] : ssm_snap_[(size_t) il] + (size_t) h_row * ssm_n;
        check(cudaMemcpyAsync(ssm_mark_[st(il)], ssm, ssm_n * 4, cudaMemcpyDeviceToDevice, S(d.stream)), "mark state");
        for (int which = 0; which < 3; ++which) {
            const float* cv = live ? conv_state_[st(il)] + which * cw : conv_snap_[(size_t) il] + which * (size_t) win_T_ * cw + (size_t) h_row * cw;
            check(cudaMemcpyAsync(conv_mark_[st(il)] + which * cw, cv, cw * 4, cudaMemcpyDeviceToDevice, S(d.stream)), "mark conv");
        }
    }
    Dev& hd = devs_[(size_t) dev_of_layer_[(size_t) hp_.n_layer - 1]];
    check(cudaSetDevice(hd.ordinal), "cudaSetDevice");
    check(cudaMemcpyAsync(h_mark_s_[(size_t) cur_slot_], hd.h_win + (size_t) h_row * hp_.n_embd, (size_t) hp_.n_embd * 4, cudaMemcpyDeviceToDevice, S(hd.stream)), "mark h");
    mark_pos_s_[(size_t) cur_slot_] = win_start_ + h_row + 1;
}

bool Engine::restore_mark(int slot) {
    if (slot < 0) slot = n_slots_;
    if (mark_pos_s_[(size_t) slot] <= 0) return false;
    cur_slot_ = slot;
    const int D = hp_.d_inner(), Sd = hp_.kda_head_dim;
    const size_t ssm_n = (size_t) hp_.n_head * Sd * Sd, cw = (size_t) 3 * (hp_.d_conv - 1) * D;
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!hp_.is_recr[(size_t) il]) continue;
        Dev& d = devs_[(size_t) dev_of_layer_[(size_t) il]];
        check(cudaSetDevice(d.ordinal), "cudaSetDevice");
        check(cudaMemcpyAsync(ssm_state_[st(il)], ssm_mark_[st(il)], ssm_n * 4, cudaMemcpyDeviceToDevice, S(d.stream)), "restore state");
        check(cudaMemcpyAsync(conv_state_[st(il)], conv_mark_[st(il)], cw * 4, cudaMemcpyDeviceToDevice, S(d.stream)), "restore conv");
    }
    pos_ = mark_pos_s_[(size_t) slot];
    pos_s_[(size_t) slot] = pos_;
    win_start_ = pos_;
    win_T_ = 0;
    if (slot == n_slots_) mtp_pos_ = pos_;
    return true;
}

// y[T][n_out] = W x[T][n_in]: Q8_0 through the pinned MMVQ (Q8_1 activations, columns bitwise equal to single calls),
// BF16 and F32 in FP32 with the same per-column arithmetic as the single-token kernels
// rows [r0, r0 + n) of a row-contiguous weight as a tensor of their own
static DevTensor row_slice(const DevTensor& w, int64_t r0, int64_t n) {
    DevTensor v = w;
    const uint64_t rb = w.bytes / (uint64_t) w.ne1;
    v.data = static_cast<uint8_t*>(w.data) + (uint64_t) r0 * rb;
    v.bytes = (uint64_t) n * rb;
    v.ne1 = n;
    return v;
}

void Engine::gemv(Dev& d, int T, const DevTensor& w, const float* x, float* y, int n_in, int n_out, int ldx) {
    if ((int) w.ne0 != n_in || (int) w.ne1 != n_out)
        throw std::runtime_error("gemv: tensor is [" + std::to_string(w.ne0) + "," + std::to_string(w.ne1) + "], wanted [" +
                                 std::to_string(n_in) + "," + std::to_string(n_out) + "]");
    if (ldx && ldx != n_in && (w.type != T_BF16 || T >= kGemmMinT)) throw std::runtime_error("gemv: a strided input needs the small BF16 kernel");
    if (w.type == T_F32) {
        if (T >= kGemmMinT && d.gemm && (int64_t) n_in * n_out <= static_cast<strata::prefill::Gemm*>(d.gemm)->scratch_elems()) {
            // the F32 weight (the router) rounded to FP16 into the GEMM scratch, like a dequantized one
            auto* g = static_cast<strata::prefill::Gemm*>(d.gemm);
            strata::prefill::to_f16(w.f32(), g->scratch(), (int64_t) n_in * n_out, d.stream);
            strata::prefill::to_f16(x, d.x16, (int64_t) T * n_in, d.stream);
            g->f16(d.x16, g->scratch(), y, T, n_out, n_in);
            return;
        }
        f32_gemv_t(w.f32(), x, y, n_in, n_out, T, d.stream);
        return;
    }
    if (w.type != T_BF16 && !strata::kernels::native_mmvq_supported((int) w.type))
        throw std::runtime_error("gemv: unsupported tensor type " + std::to_string(w.type));
    if (T >= kMmqMinT && d.mmq && strata::prefill::mmq::supported((int) w.type) && n_in % 32 == 0 && n_out >= 32 && !kNoDenseMmq &&
        mmq_fits((int) w.type, n_out)) {   // fits(): the kernel has a tile for this row count (its "fallback" configs cover rows % 128 != 0)
        // a wide window on a quantized dense weight: llama.cpp's MMQ straight on the stored blocks (int8 dot products;
        // 36 TMAC/s on gfx1100 against 6.4 for the FP16 GEMM, which this GPU's BLAS libraries cap at ~13 TFLOPS)
        namespace mmq = strata::prefill::mmq;
        // the MMQ activation layout is k-block-major with the row count as its stride: the quantized rows and the
        // product's total_rows must agree (T here; the buffer keeps a tile of slack for the kernel's over-read)
        mmq::quantize(x, nullptr, d.mmq_xd, (int) w.type, n_in, n_in, T, d.stream);
        if (d.dense_T != T) { set_pair(d.dense_bounds, 0, T, d.stream); d.dense_T = T; }
        mmq::Product p;
        p.w = w.data; p.type = (int) w.type; p.w_rows = n_out; p.w_cols = n_in; p.expert_bytes = mmq::matrix_bytes((int) w.type, n_out, n_in); p.n = 1;
        p.xq = d.mmq_xd; p.bounds = d.dense_bounds; p.ids = d.iota; p.total_rows = T; p.max_rows = T; p.dst = y; p.ld_dst = n_out;
        static_cast<mmq::Context*>(d.mmq)->run(p, d.stream);
        return;
    }
    if (T >= kGemmMinT && d.gemm) {
        // a wide window: one tensor-core GEMM (the weight dequantized to FP16 once, the activations rounded to
        // FP16 / BF16), as llama.cpp's batched path does; the verify window keeps the exact MMVQ below
        auto* g = static_cast<strata::prefill::Gemm*>(d.gemm);
        if (w.type == T_BF16) {
            strata::prefill::to_bf16(x, d.x16, (int64_t) T * n_in, d.stream);
            g->bf16(d.x16, w.bf16(), y, T, n_out, n_in);
            return;
        }
        if (strata::kernels::dequant_bf16_supported((int) w.type)) {
            strata::prefill::to_f16(x, d.x16, (int64_t) T * n_in, d.stream);
            g->native(d.x16, (int) w.type, w.data, y, T, n_out, n_in);
            return;
        }
    }
    // the multi-column kernels take up to 8 activation columns: a wider window goes in chunks (the weight read once
    // per chunk); every column's result equals its single-column call
    for (int t0 = 0; t0 < T; t0 += 8) {
        const int Tc = std::min(8, T - t0);
        const float* xc = x + (size_t) t0 * n_in;
        float* yc = y + (size_t) t0 * n_out;
        if (w.type == T_BF16 && n_in <= 256 && n_in % 8 == 0 && kNarrowBf16)
            bf16_gemv_narrow_t(x + (size_t) t0 * (ldx ? ldx : n_in), ldx ? ldx : n_in, w.bf16(), yc, n_in, n_out, Tc, d.stream);
        else if (w.type == T_BF16) strata::kernels::bf16_gemv_fp32_mmvf_multi(x + (size_t) t0 * (ldx ? ldx : n_in), ldx ? ldx : n_in, w.bf16(), yc, n_out, n_in, n_out, Tc, d.stream);
        else {
            strata::kernels::native_quantize_q8_1(xc, d.q8_big, n_in, Tc, d.stream);
            strata::kernels::native_mmvq((int) w.type, w.data, d.q8_big, yc, n_in, n_out, Tc, d.stream);
        }
    }
}

// a quantized weight on q8_1 rows a producer kernel wrote (the decode windows: up to 8 columns per launch)
void Engine::gemv_q8(Dev& d, int T, const DevTensor& w, const uint8_t* xq, float* y, int n_in, int n_out) {
    if ((int) w.ne0 != n_in || (int) w.ne1 != n_out) throw std::runtime_error("gemv_q8: tensor shape mismatch");
    if (T > 8 || !strata::kernels::native_mmvq_supported((int) w.type)) throw std::runtime_error("gemv_q8: a decode window on a quantized weight only");
    strata::kernels::native_mmvq((int) w.type, w.data, xq, y, n_in, n_out, T, d.stream);
}

void Engine::mhc(Dev& d, int T, const float* R, const DevTensor& fn, const DevTensor& scale, const DevTensor& base, float* x) {
    mhc_mix_t(R, fn.f32(), scale.f32(), base.f32(), d.mixes, d.part, d.rs, d.pre, d.post, d.comb, T, hp_.n_embd, hp_.hc,
              hp_.hc_iters, hp_.rms_eps, hp_.hc_eps, d.stream);
    mhc_pre_apply_t(R, d.pre, x, T, hp_.n_embd, hp_.hc, d.stream);
}

// the pre-mix and the norm that follows it: a verify / decode window in two launches, a wider one on the separate kernels
void Engine::mhc_norm(Dev& d, int T, const float* R, const DevTensor& fn, const DevTensor& scale, const DevTensor& base, const DevTensor& norm,
                      float* x, float* xn) {
    d.xn_q8 = nullptr;
    if (T <= 8 && kMhcDec) {
        uint8_t* xq = kQ8Fuse && xn == d.xn ? d.q8_x : nullptr;   // the Q8_0 products from xn take these rows instead of quantizing again
        mhc_dec_t(R, fn.f32(), scale.f32(), base.f32(), d.mixes, d.part, d.pre, d.post, d.comb, norm.f32(), x, xn, xq, T, hp_.n_embd, hp_.hc,
                  hp_.hc_iters, hp_.rms_eps, hp_.hc_eps, d.stream);
        d.xn_q8 = xq;
        return;
    }
    mhc(d, T, R, fn, scale, base, x);
    rms_norm(x, norm.f32(), xn, T, hp_.n_embd, hp_.rms_eps, d.stream);
}

void Engine::kda_layer(Dev& d, int il, int T, const float* xn, float* out, const DumpFn& dump) {
    const LayerWeights& L = m_.layer(il);
    const int E = hp_.n_embd, H = hp_.n_head, Sd = hp_.kda_head_dim, D = hp_.d_inner();
    const int dv = d.ordinal;
    float* cs = conv_state_[stj(d, il)];
    float* csnap = T <= snap_T_ ? conv_snap_[(size_t) il] : nullptr;
    const int cs_stride = (hp_.d_conv - 1) * D, snap_stride = T * (hp_.d_conv - 1) * D;
    float *q = d.big_a, *k = d.big_b, *v = d.big_c, *tmp = d.big_d;
    const Ptr3 qkv{{q, k, v}};
    if (rows_mode_) {   // runs of rows per slot on the slot's conv state, with per-row snapshots for the rollback
        if (T > kQkvFuseT) throw std::runtime_error("kda_layer: too many rows for the fused product");
        float* rsnap = conv_snap_[(size_t) il];
        gemv(d, T, L.wqkv, xn, d.qkv, E, 3 * D);
        kda_conv_grp(d.qkv, 3 * D, 3, conv_tab_[(size_t) il], d.row_slot_d, d.row_k_d, L.conv_qkv.f32(), qkv, rsnap, snap_stride, D, hp_.d_conv, T,
                     d.grp_slot_d, d.grp_start_d, d.grp_len_d, n_grp_, d.stream);
    } else if (T <= kQkvFuseT) {   // q | k | v in one product and one conv launch
        if (d.xn_q8 && xn == d.xn && T <= 8) gemv_q8(d, T, L.wqkv, d.xn_q8, d.qkv, E, 3 * D);
        else gemv(d, T, L.wqkv, xn, d.qkv, E, 3 * D);
        kda_conv_silu_t(d.qkv, 3 * D, 3, cs, cs_stride, L.conv_qkv.f32(), qkv, csnap, snap_stride, D, hp_.d_conv, T, d.stream);
    } else {   // a prompt window: the three products one at a time (the wide [T][3 D] result would not fit the scratch)
        for (int sg = 0; sg < 3; ++sg) {
            gemv(d, T, row_slice(L.wqkv, (int64_t) sg * D, D), xn, tmp, E, D);
            const Ptr3 one{{qkv.p[sg], nullptr, nullptr}};
            kda_conv_silu_t(tmp, D, 1, cs + (size_t) sg * cs_stride, 0, L.conv_qkv.f32() + (size_t) sg * D * hp_.d_conv, one,
                            csnap ? csnap + (size_t) sg * snap_stride : nullptr, 0, D, hp_.d_conv, T, d.stream);
        }
    }
    prof_mark(d, "kda_qkv");
    if (dump) { dump("kda_q_conv", q, {H, Sd}, false, dv); dump("kda_k_conv", k, {H, Sd}, false, dv); dump("kda_v_conv", v, {H, Sd}, false, dv); }
    float *normed = d.big_a, *y = d.big_b;
    float* o = d.ffn_h;
    static const int kKdaFuse = std::getenv("STRATA_GLM5_KDA_FUSE") ? std::atoi(std::getenv("STRATA_GLM5_KDA_FUSE")) : 1;   // 0: the separate scan / gate kernels
    if (kKdaFuse && !rows_mode_ && T <= 8 && T < kGemmMinT && T <= kFabT && !dump) {
        normed = d.ffn_u; y = d.ffn_y;   // (the fused kernel reads q and k while it writes these: other buffers)
        // the verify / decode window: f_a | g_a | beta as one product ([T][2 S + H]: the f_b / g_b inputs and beta at
        // row stride 2 S + H), then the scan with its gates, norms and output gate in one launch
        const int ldf = 2 * Sd + H;
        gemv(d, T, L.fab, xn, d.fab, E, ldf);
        gemv(d, T, L.f_b, d.fab, tmp, Sd, D, ldf);            // the gate's input
        float* g2 = d.ffn_g;                                  // T x D floats, free during the attention half
        gemv(d, T, L.g_b, d.fab + Sd, g2, Sd, D, ldf);        // the output gate
        prof_mark(d, "kda_gates");
        uint8_t* yq = kQ8Fuse ? d.q8_big : nullptr;   // y's q8_1 rows for the output projection ([T][q8_cols(D)], the gemv scratch's layout)
        kda_step_fused(ssm_state_[stj(d, il)], q, k, v, tmp, L.dt_b.f32(), L.ssm_a.f32(), hp_.kda_lower, d.fab + 2 * Sd, ldf, L.ssm_norm.f32(), g2,
                       o, normed, y, yq, T <= snap_T_ ? ssm_snap_[(size_t) il] : nullptr, T, H, Sd, 1e-6f, hp_.rms_eps, d.stream);
        prof_mark(d, "kda_scan");
        if (yq) {
            gemv_q8(d, T, L.wo, yq, out, D, E);
            prof_mark(d, "kda_out");
            if (dump) dump("kda_out", out, {E}, false, dv);
            return;
        }
    } else {
        const DevTensor f_a = row_slice(L.fab, 0, Sd), g_a = row_slice(L.fab, Sd, Sd), beta_w = row_slice(L.fab, 2 * Sd, H);
        gemv(d, T, f_a, xn, d.small_a, E, Sd);
        gemv(d, T, L.f_b, d.small_a, tmp, Sd, D);
        float* g1 = d.ffn_g;   // T x D floats, free during the attention half
        kda_gate_t(tmp, L.dt_b.f32(), L.ssm_a.f32(), hp_.kda_lower, g1, T, H, Sd, d.stream);
        if (dump) dump("kda_g1", g1, {H, Sd}, false, dv);
        float* beta = d.ffn_u;
        gemv(d, T, beta_w, xn, beta, E, H);
        sigmoid_inplace(beta, T * H, d.stream);
        if (dump) dump("kda_beta", beta, {H}, false, dv);
        l2_norm_rows(q, T * H, Sd, 1e-6f, d.stream);
        l2_norm_rows(k, T * H, Sd, 1e-6f, d.stream);
        prof_mark(d, "kda_gates");
        if (rows_mode_) kda_step_grp(ssm_tab_[(size_t) il], d.grp_slot_d, d.grp_start_d, d.grp_len_d, n_grp_, q, k, v, g1, beta, o, ssm_snap_[(size_t) il], H, Sd, d.stream);
        else kda_step_t(ssm_state_[stj(d, il)], q, k, v, g1, beta, o, T <= snap_T_ ? ssm_snap_[(size_t) il] : nullptr, T, H, Sd, d.stream);
        if (dump) dump("kda_scan_out", o, {H, Sd}, false, dv);
        prof_mark(d, "kda_scan");
        gemv(d, T, g_a, xn, d.small_a, E, Sd);
        gemv(d, T, L.g_b, d.small_a, tmp, Sd, D);
        if (dump) dump("kda_g2", tmp, {H, Sd}, false, dv);
        kda_out_gate(o, L.ssm_norm.f32(), tmp, normed, y, T * H, Sd, hp_.rms_eps, d.stream);
        if (dump) dump("kda_normed", normed, {H, Sd}, false, dv);
    }
    gemv(d, T, L.wo, y, out, D, E);
    prof_mark(d, "kda_out");
    if (dump) dump("kda_out", out, {E}, false, dv);
}

// The DSA layer: the projections for the T tokens at once, the per-token cache append / pool / score / select / attend
// in order (token t sees the cells up to pos0 + t), the output projection for all T.
// the selection list length of a query at n_kv cells: the exact pages (sink + top) x ps, the local field, the summaries
int Engine::lod_n_sel(int n_kv, int n_top) const {
    const int n_cand = lod_cand(n_kv), sink = std::min(lod_.sink_pages, n_cand), top = std::min(n_top, std::max(n_cand - sink, 0));
    return (sink + top) * lod_.ps + (n_kv - n_cand * lod_.ps) + (n_cand - sink - top);
}

// windowed attention over the latent cache (on MLA): every query reads the sink page(s), its top pages and the local field
// exactly and every other complete page as one count-corrected summary (the page mean, logit + log ps).  Decode windows
// and batch rows select per token; a prompt window shares one selection (page scores pooled over its queries and heads)
// and attends through the MMQ products over a compact operand [selection | its own tokens].
void Engine::dsa_lod(Dev& d, int il, int T, int pos0, float scale) {
    static const bool kLodSync = std::getenv("STRATA_GLM5_LOD_SYNC") != nullptr;   // per-phase syncs of this path alone
    auto lsync = [&](const char* where) {
        if (!(kDebugSync || kLodSync)) return;
        const cudaError_t e = cudaStreamSynchronize(S(d.stream));
        if (e != cudaSuccess) throw std::runtime_error(std::string("glm5: device fault in layer ") + std::to_string(il) + " LoD (" + where + "): " + cudaGetErrorString(e));
    };
    const int KV = hp_.kv_lora, H = hp_.n_head, ps = lod_.ps;
    const LodCfg& lr = lod_of(il);   // this layer's route
    cudaStream_t s = S(d.stream);
    const int ld_score = max_ctx_ / ps + 1;
    const float bias = std::log((float) ps);
    uint8_t* const* kvc_tab = kvc_tab_[(size_t) il];
    uint8_t* const* pm_tab = pmean_tab_[(size_t) il];
    int* prm = d.sel_params_host;
    const bool dense = !rows_mode_ && T >= kMmqMinT && d.mmq && d.lod_kv_q8 && !kNoDenseAttn && mmq_fits(8, KV);
    if (!dense) {
        if (T > 64) throw std::runtime_error("glm5: LoD: a window of " + std::to_string(T) + " tokens needs the MMQ attention path");
        if (!rows_mode_) {   // the solo window's rows as (slot, position) pairs for the row kernels
            for (int t = 0; t < T; ++t) { d.row_slot_h[t] = jslot(d); d.row_pos_h[t] = pos0 + t; }
            check(cudaMemcpyAsync(d.row_slot_d, d.row_slot_h, (size_t) T * 4, cudaMemcpyHostToDevice, s), "lod row slots");
            check(cudaMemcpyAsync(d.row_pos_d, d.row_pos_h, (size_t) T * 4, cudaMemcpyHostToDevice, s), "lod row positions");
        }
        int n_cand_max = 0;
        for (int t = 0; t < T; ++t) {
            const int n_kv = (rows_mode_ ? row_pos_[(size_t) t] : pos0 + t) + 1, n_cand = lod_cand(n_kv);
            prm[t * 8 + 0] = n_cand; prm[t * 8 + 1] = lod_.top_decode; prm[t * 8 + 2] = n_kv; prm[t * 8 + 3] = std::min(lod_.sink_pages, n_cand); prm[t * 8 + 5] = 0;
            n_cand_max = std::max(n_cand_max, n_cand);
        }
        check(cudaMemcpyAsync(d.sel_params, prm, (size_t) T * 8 * 4, cudaMemcpyHostToDevice, s), "lod params h2d");
        const float* page_scores = d.score;
        if (lr.idx_route && n_cand_max > 0) {
            // the indexer's pool scores of each row (the DSA kernel), then the max per page
            const int KP = hp_.kpool, ppp = ps / KP, ld_pool = max_ctx_ / KP + 1;
            int* prm2 = d.sel_params2_host;
            int n_pool_max = 0;
            for (int t = 0; t < T; ++t) { prm2[t * 8 + 0] = prm[t * 8 + 0] * ppp; n_pool_max = std::max(n_pool_max, prm2[t * 8 + 0]); }
            check(cudaMemcpyAsync(d.sel_params2, prm2, (size_t) T * 8 * 4, cudaMemcpyHostToDevice, s), "lod params2 h2d");
            dsa_score_rows(pooled_tab_[(size_t) il], d.row_slot_d, d.iq, d.iw, d.score, d.sel_params2, n_pool_max, T, hp_.idx_heads, hp_.idx_dim, ld_pool, d.stream);
            if (lr.pool_sel) {
                for (int t = 0; t < T; ++t) { prm[t * 8 + 1] = lod_.top_pools_decode; prm[t * 8 + 5] = lod_.no_sum ? 1 : 0; }   // the budget in pools
                check(cudaMemcpyAsync(d.sel_params, prm, (size_t) T * 8 * 4, cudaMemcpyHostToDevice, s), "lod params h2d");
                page_scores = nullptr;
                lod_select_pools(d.score, d.sel_params, T, ps, KP, d.sel, ld_pool, n_sel_max_, d.stream);
            } else {
                pool_page_max_rows(d.score, ld_pool, d.lod_pscore, ld_score, d.sel_params, n_cand_max, T, ppp, d.stream);
                page_scores = d.lod_pscore;
            }
        } else if (lr.tok_route && n_cand_max > 0) {
            head_group_mean(d.q_abs, d.lod_qg, T, H, lod_.qgroups, KV, d.stream);
            lod_score_tok_rows(kvc_tab, d.row_slot_d, d.lod_qg, d.score, d.sel_params, n_cand_max, T, lod_.qgroups, KV, ld_score, ps, d.stream);
        } else lod_score_rows(pm_tab, d.row_slot_d, d.q_abs, d.score, d.sel_params, n_cand_max, T, H, KV, ld_score, d.stream);
        if (kDebugSync || kLodSync) lsync("lod score rows");
        prof_mark(d, "dsa_score");
        if (page_scores) lod_select(page_scores, d.sel_params, T, ps, d.sel, ld_score, n_sel_max_, d.stream);
        if (kDebugSync || kLodSync) {
            lsync("lod select rows");
            std::vector<int> pp((size_t) T * 8);
            check(cudaMemcpy(pp.data(), d.sel_params, (size_t) T * 8 * 4, cudaMemcpyDeviceToHost), "lod params d2h");
            for (int t = 0; t < T; ++t) {
                const int want = lr.pool_sel ? pp[(size_t) t * 8 + 4] : lod_n_sel(pp[(size_t) t * 8 + 2], lod_.top_decode);
                if (pp[(size_t) t * 8 + 4] != want || want > n_sel_max_)
                    std::fprintf(stderr, "glm5: layer %d LoD row %d: n_sel %d (host %d, max %d) n_kv %d cand %d\n", il, t, pp[(size_t) t * 8 + 4], want, n_sel_max_, pp[(size_t) t * 8 + 2], pp[(size_t) t * 8]);
            }
        }
        prof_mark(d, "dsa_select");
        static const int kLodSplit = std::getenv("STRATA_GLM5_LOD_SPLIT") ? std::atoi(std::getenv("STRATA_GLM5_LOD_SPLIT")) : 16;
        const bool fused = T <= 8 && d.lod_part && kLodSplit > 0 && kLodSplit <= 128 && (KV == 256 || KV == 512 || KV == 1024);
        if (fused) lod_attn_fused(kvc_tab, pm_tab, bias, d.row_slot_d, d.sel, d.sel_params, T, n_sel_max_, d.q_abs, scale, d.lod_part, d.ctx, H, KV, kLodSplit, d.stream);
        else dsa_attend_rows(kvc_tab, pm_tab, bias, d.row_slot_d, d.sel, d.sel_params, T, n_sel_max_, d.q_abs, scale, d.attn_scores, d.ctx, H, KV, d.stream);
        if (kDebugSync || kLodSync) { lsync("lod attend rows"); debug_nan(d.ctx, T, H * KV, il, "lod ctx rows"); }
        prof_mark(d, "dsa_attend");
        return;
    }
    namespace mmq = strata::prefill::mmq;
    auto* ctx = static_cast<mmq::Context*>(d.mmq);
    const uint8_t* kvc = kvc_cache_[stj(d, il)];
    const uint8_t* pm = pmean_[stj(d, il)];
    // the shared selection: the query "at pos0" with the prefill budget; page scores = max over the window's (query, head) rows
    const int n_cand = lod_cand(pos0);
    prm[0] = n_cand; prm[1] = lod_.top_prefill; prm[2] = pos0; prm[3] = std::min(lod_.sink_pages, n_cand); prm[5] = 0;
    check(cudaMemcpyAsync(d.sel_params, prm, 8 * 4, cudaMemcpyHostToDevice, s), "lod params h2d");
    if (n_cand > prm[3] && lr.idx_route) {
        // the indexer's pool scores for the window: S = iq (T*IH rows, padded to 256) x pooled keys (Q8_0, 256), per pool
        // the max over the queries of sum_h relu(S) w; then the max per page
        const int IH = hp_.idx_heads, ID = hp_.idx_dim, KP = hp_.kpool, ppp = ps / KP, n_pool = n_cand * ppp, TBq = kDsaBatch * H / IH;
        pad_rows(d.iq, d.lod_iqp, T * IH, ID, kIdxPad, d.stream);
        for (int c0 = 0; c0 < n_pool; c0 += kDenseAttnMaxKv) {
            const int nc = std::min(kDenseAttnMaxKv, n_pool - c0), nc_pad = (nc + 511) / 512 * 512;
            for (int t0 = 0; t0 < T; t0 += TBq) {
                const int TB = std::min(TBq, T - t0), rows = TB * IH;
                if (d.attn_rows != rows) { set_pair(d.attn_bounds, 0, rows, d.stream); d.attn_rows = rows; }
                mmq::quantize(d.lod_iqp + (size_t) t0 * IH * kIdxPad, nullptr, d.attn_xq, 8, kIdxPad, kIdxPad, rows, d.stream);
                mmq::Product ps_;
                ps_.w = pooled_q8_[stj(d, il)] + (size_t) c0 * (kIdxPad / 32) * 34; ps_.type = 8; ps_.w_rows = nc_pad; ps_.w_cols = kIdxPad; ps_.expert_bytes = (size_t) nc_pad * (kIdxPad / 32) * 34; ps_.n = 1;
                ps_.xq = d.attn_xq; ps_.bounds = d.attn_bounds; ps_.ids = d.iota; ps_.total_rows = rows; ps_.max_rows = rows; ps_.dst = d.attn_S; ps_.ld_dst = nc_pad;
                ctx->run(ps_, d.stream);
                idx_pool_score(d.attn_S, nc_pad, TB, IH, d.iw + (size_t) t0 * IH, nc, d.lod_tok + c0, t0 == 0, d.stream);
            }
        }
        if (!lr.pool_sel) page_max(d.lod_tok, d.lod_score, ppp, n_cand, d.stream);
        if (kDebugSync || kLodSync) lsync("lod indexer scores");
    } else if (n_cand > prm[3] && lr.tok_route) {
        // token scores: the head-group means of the window's queries (T x G rows) against every candidate cell, in column
        // chunks of the S buffer's width, the max over the rows kept per cell; then the max per page
        const int G = lod_.qgroups, rows_all = T * G, n_tok = n_cand * ps, RB = kDsaBatch * H;   // RB rows fit the S / xq buffers
        head_group_mean(d.q_abs, d.lod_qg, T, H, G, KV, d.stream);
        for (int c0 = 0; c0 < n_tok; c0 += kDenseAttnMaxKv) {
            const int nc = std::min(kDenseAttnMaxKv, n_tok - c0), nc_pad = (nc + 511) / 512 * 512;
            for (int r0 = 0; r0 < rows_all; r0 += RB) {
                const int rows = std::min(RB, rows_all - r0);
                if (d.attn_rows != rows) { set_pair(d.attn_bounds, 0, rows, d.stream); d.attn_rows = rows; }
                mmq::quantize(d.lod_qg + (size_t) r0 * KV, nullptr, d.attn_xq, 8, KV, KV, rows, d.stream);
                mmq::Product ps_;
                ps_.w = kvc + (size_t) c0 * q8_pitch(KV); ps_.type = 8; ps_.w_rows = nc_pad; ps_.w_cols = KV; ps_.expert_bytes = (size_t) nc_pad * q8_pitch(KV); ps_.n = 1;
                ps_.xq = d.attn_xq; ps_.bounds = d.attn_bounds; ps_.ids = d.iota; ps_.total_rows = rows; ps_.max_rows = rows; ps_.dst = d.attn_S; ps_.ld_dst = nc_pad;
                ctx->run(ps_, d.stream);
                colmax(d.attn_S, nc_pad, rows, nc, d.lod_tok + c0, r0 == 0, d.stream);
            }
        }
        page_max(d.lod_tok, d.lod_score, ps, n_cand, d.stream);
        if (kDebugSync || kLodSync) lsync("lod token scores");
    } else if (n_cand > prm[3]) {
        const int n_cand_pad = (n_cand + 511) / 512 * 512;
        if (n_cand_pad > kDenseAttnMaxKv || !mmq_fits(8, n_cand_pad)) throw std::runtime_error("glm5: LoD: " + std::to_string(n_cand) + " candidate pages exceed the scoring buffers");
        for (int t0 = 0; t0 < T; t0 += kDsaBatch) {
            const int TB = std::min(kDsaBatch, T - t0), rows = TB * H;
            if (d.attn_rows != rows) { set_pair(d.attn_bounds, 0, rows, d.stream); d.attn_rows = rows; }
            mmq::quantize(d.q_abs + (size_t) t0 * H * KV, nullptr, d.attn_xq, 8, KV, KV, rows, d.stream);
            mmq::Product ps_;
            ps_.w = pm; ps_.type = 8; ps_.w_rows = n_cand_pad; ps_.w_cols = KV; ps_.expert_bytes = (size_t) n_cand_pad * q8_pitch(KV); ps_.n = 1;
            ps_.xq = d.attn_xq; ps_.bounds = d.attn_bounds; ps_.ids = d.iota; ps_.total_rows = rows; ps_.max_rows = rows; ps_.dst = d.attn_S; ps_.ld_dst = n_cand_pad;
            ctx->run(ps_, d.stream);
            colmax(d.attn_S, n_cand_pad, rows, n_cand, d.lod_score, t0 == 0, d.stream);
        }
        if (kDebugSync || kLodSync) lsync("lod page scores");
    }
    prof_mark(d, "dsa_score");
    int n_sel;
    if (lr.pool_sel) {   // the exact set = the indexer's top pools (DSA's granularity); the host sizes the operand by the bound
        const int KP = hp_.kpool, ld_pool = max_ctx_ / KP + 1;
        prm[1] = lod_.top_pools; prm[5] = lod_.no_sum ? 1 : 0;
        check(cudaMemcpyAsync(d.sel_params, prm, 8 * 4, cudaMemcpyHostToDevice, s), "lod params h2d");
        lod_select_pools(d.lod_tok, d.sel_params, 1, ps, KP, d.sel, ld_pool, n_sel_max_, d.stream);
        const int sink = std::min(lod_.sink_pages, n_cand), n_top = std::min(lod_.top_pools, std::max(n_cand - sink, 0) * (ps / KP));
        n_sel = sink * ps + n_top * KP + (pos0 - n_cand * ps) + (lod_.no_sum ? 0 : n_cand - sink);
    } else {
        lod_select(d.lod_score, d.sel_params, 1, ps, d.sel, ld_score, n_sel_max_, d.stream);
        n_sel = lod_n_sel(pos0, lod_.top_prefill);
    }
    const int n_cols = n_sel + T, n_pad = (n_cols + 511) / 512 * 512;
    if (kDebugSync || kLodSync) {
        const cudaError_t le = cudaGetLastError();
        int cur = -1; cudaGetDevice(&cur);
        lsync("lod select");
        int pp[8];
        check(cudaMemcpy(pp, d.sel_params, 8 * 4, cudaMemcpyDeviceToHost), "lod params d2h");
        std::fprintf(stderr, "glm5: layer %d LoD window pos0 %d T %d: cand %d n_sel device %d host %d, n_pad %d; params dev {%d %d %d %d}, launch %s, device %d/%d\n",
                     il, pos0, T, n_cand, pp[4], n_sel, n_pad, pp[0], pp[1], pp[2], pp[3], cudaGetErrorString(le), cur, d.ordinal);
    }
    if (n_pad > kDenseAttnMaxKv || !mmq_fits(8, n_pad)) throw std::runtime_error("glm5: LoD: a compact operand of " + std::to_string(n_cols) + " columns exceeds the attention buffers");
    if (lr.pool_sel) lod_gather_dev(kvc, pm, d.sel, d.sel_params, n_sel, pos0, T, d.lod_kv_q8, d.lod_bias, bias, KV, d.stream);
    else lod_gather(kvc, pm, d.sel, n_sel, pos0, T, d.lod_kv_q8, d.lod_bias, bias, KV, d.stream);
    if (n_pad > n_cols) {
        check(cudaMemsetAsync(d.lod_kv_q8 + (size_t) n_cols * q8_pitch(KV), 0, (size_t) (n_pad - n_cols) * q8_pitch(KV), s), "lod pad");
        check(cudaMemsetAsync(d.lod_bias + n_cols, 0, (size_t) (n_pad - n_cols) * 4, s), "lod bias pad");
    }
    q8_0_rows_T(d.lod_kv_q8, KV, d.kvcT_q8, n_cols, n_pad, d.stream);
    if (kDebugSync || kLodSync) lsync("lod gather/transpose");
    prof_mark(d, "dsa_select");
    const int words = n_pad / 32;
    for (int t0 = 0; t0 < T; t0 += kDsaBatch) {
        const int TB = std::min(kDsaBatch, T - t0), rows = TB * H;
        lod_mask(d.attn_mask, words, TB, n_sel, T, t0, d.stream);
        if (d.attn_rows != rows) { set_pair(d.attn_bounds, 0, rows, d.stream); d.attn_rows = rows; }
        mmq::quantize(d.q_abs + (size_t) t0 * H * KV, nullptr, d.attn_xq, 8, KV, KV, rows, d.stream);
        mmq::Product ps_;
        ps_.w = d.lod_kv_q8; ps_.type = 8; ps_.w_rows = n_pad; ps_.w_cols = KV; ps_.expert_bytes = (size_t) n_pad * q8_pitch(KV); ps_.n = 1;
        ps_.xq = d.attn_xq; ps_.bounds = d.attn_bounds; ps_.ids = d.iota; ps_.total_rows = rows; ps_.max_rows = rows; ps_.dst = d.attn_S; ps_.ld_dst = n_pad;
        ctx->run(ps_, d.stream);
        masked_softmax(d.attn_S, n_pad, d.attn_mask, words, rows, H, n_pad, scale, d.lod_bias, d.stream);
        mmq::quantize(d.attn_S, nullptr, d.attn_pq, 8, n_pad, n_pad, rows, d.stream);
        mmq::Product pc;
        pc.w = d.kvcT_q8; pc.type = 8; pc.w_rows = KV; pc.w_cols = n_pad; pc.expert_bytes = (size_t) KV * (n_pad / 32) * 34; pc.n = 1;
        pc.xq = d.attn_pq; pc.bounds = d.attn_bounds; pc.ids = d.iota; pc.total_rows = rows; pc.max_rows = rows; pc.dst = d.ctx + (size_t) t0 * H * KV; pc.ld_dst = KV;
        ctx->run(pc, d.stream);
        if (kDebugSync || kLodSync) lsync("lod attention batch");
    }
    prof_mark(d, "dsa_attend");
}

void Engine::dsa_layer(Dev& d, int il, int T, int pos0, const float* xn, float* out, const DumpFn& dump) {
    const LayerWeights& L = m_.layer(il);
    const int E = hp_.n_embd, H = hp_.n_head, KV = hp_.kv_lora, HK = hp_.head_k_mla, HV = hp_.head_v_mla;
    const int ID = hp_.idx_dim, IH = hp_.idx_heads, KP = hp_.kpool;
    const int dv = d.ordinal;
    cudaStream_t s = S(d.stream);
    const int n_sel_max = n_sel_max_;
    if (d.xn_q8 && xn == d.xn && T <= 8) gemv_q8(d, T, L.q_a, d.xn_q8, d.small_a, E, hp_.q_lora);
    else gemv(d, T, L.q_a, xn, d.small_a, E, hp_.q_lora);
    if (kDebugSync) debug_sync(d.stream, il, "q_a");
    float* qr = d.small_b;
    rms_norm(d.small_a, L.q_a_norm.f32(), qr, T, hp_.q_lora, hp_.rms_eps, d.stream);
    if (dump) dump("q_resid", qr, {hp_.q_lora}, false, dv);
    float* q = d.big_a;
    gemv(d, T, L.q_b, qr, q, hp_.q_lora, H * HK);
    if (kDebugSync) debug_sync(d.stream, il, "q_b");
    gemv(d, T, L.kv_a, xn, d.big_b, E, KV);
    if (kDebugSync) debug_sync(d.stream, il, "kv_a");
    if (rows_mode_) {   // each row into its slot's cache at its position
        rms_norm(d.big_b, L.kv_a_norm.f32(), d.big_c, T, KV, hp_.rms_eps, d.stream);
        scatter_q8_rows(d.big_c, kvc_tab_[(size_t) il], d.row_slot_d, d.row_pos_d, T, KV, d.stream);
    } else
    { rms_norm(d.big_b, L.kv_a_norm.f32(), d.big_c, T, KV, hp_.rms_eps, d.stream); store_q8_rows(d.big_c, kvc_cache_[stj(d, il)] + (size_t) pos0 * q8_pitch(KV), T, KV, d.stream); }
    if (dump) dump("kv_cmpr", d.big_c, {1, KV}, false, dv);
    if (lod_.on) {   // LoD: the pages completed by this window's rows get their means (a function of the cache rows alone)
        if (rows_mode_) lod_page_mean_rows(kvc_tab_[(size_t) il], pmean_tab_[(size_t) il], d.row_slot_d, d.row_pos_d, T, lod_.ps, KV, d.stream);
        else { const int p0 = pos0 / lod_.ps, p_end = (pos0 + T) / lod_.ps; lod_page_mean(kvc_cache_[stj(d, il)], pmean_[stj(d, il)], p0, p_end - p0, lod_.ps, KV, d.stream); }
    }
    const bool head_mmq = T >= kMmqMinT && d.mmq && L.k_b_q8.present() && L.v_b_q8.present() && !kNoDenseMmq;
    if (head_mmq && d.head_T != T) { head_tables(d.head_ids, d.head_bounds, T, H, d.stream); d.head_T = T; }
    auto head_product = [&](const DevTensor& w8, const float* xin, int n_in, int n_out, float* out) {
        // H "experts" of [n_out][n_in]: rows head-major, source and destination rows t*H + h (the token-major layouts)
        namespace mmq = strata::prefill::mmq;
        mmq::quantize(xin, d.head_ids, d.head_xq, 8, n_in, n_in, (int64_t) T * H, d.stream);
        mmq::Product p;
        p.w = w8.data; p.type = 8; p.w_rows = n_out; p.w_cols = n_in; p.expert_bytes = mmq::matrix_bytes(8, n_out, n_in); p.n = H;
        p.xq = d.head_xq; p.bounds = d.head_bounds; p.ids = d.head_ids; p.total_rows = (int64_t) T * H; p.max_rows = T; p.dst = out; p.ld_dst = n_out;
        static_cast<mmq::Context*>(d.mmq)->run(p, d.stream);
    };
    if (kDebugSync) debug_sync(d.stream, il, "dsa q/kv projections");
    if (head_mmq) head_product(L.k_b_q8, q, HK, KV, d.q_abs);
    else dsa_absorb_t(L.k_b.bf16(), q, d.q_abs, T, H, KV, HK, d.stream);
    if (kDebugSync) debug_sync(d.stream, il, "dsa absorb");
    if (dump) dump("q_absorbed", d.q_abs, {H, KV}, false, dv);
    if (lod_.on) {
        if (lod_of(il).idx_route) {   // the indexer as the router: its queries, key/gate caches and pools exactly as in DSA mode
            gemv(d, T, L.idx_q_b, qr, d.iq, hp_.q_lora, IH * ID);
            gemv(d, T, L.idx_k, xn, d.ik, E, ID);
            layer_norm_t(d.ik, L.idx_k_norm.f32(), L.idx_k_norm_b.f32(), d.big_c, T, ID, hp_.norm_eps, d.stream);
            if (rows_mode_) scatter_q8_rows(d.big_c, ik_tab_[(size_t) il], d.row_slot_d, d.row_pos_d, T, ID, d.stream);
            else store_q8_rows(d.big_c, ik_cache_[stj(d, il)] + (size_t) pos0 * q8_pitch(ID), T, ID, d.stream);
            gemv(d, T, L.idx_gate, xn, d.ig, E, ID);
            if (rows_mode_) scatter_q8_rows(d.ig, ig_tab_[(size_t) il], d.row_slot_d, d.row_pos_d, T, ID, d.stream);
            else store_q8_rows(d.ig, ig_cache_[stj(d, il)] + (size_t) pos0 * q8_pitch(ID), T, ID, d.stream);
            gemv(d, T, L.idx_proj, xn, d.iw, E, IH);
            scale_inplace(d.iw, T * IH, 1.0f / std::sqrt((float) (ID * IH)), d.stream);
            if (rows_mode_) {
                dsa_pool_rows(ik_tab_[(size_t) il], ig_tab_[(size_t) il], pooled_tab_[(size_t) il], d.row_slot_d, d.row_pos_d, L.idx_ape.f32(), T, KP, ID, d.stream);
                rows_q8_pad_rows(pooled_tab_[(size_t) il], pooled_q8_tab_[(size_t) il], d.row_slot_d, d.row_pos_d, T, KP, ID, kIdxPad, d.stream);
            } else {
                const int p0 = pos0 / KP, p_end = (pos0 + T) / KP;
                if (p_end > p0) {
                    dsa_pool(ik_cache_[stj(d, il)], ig_cache_[stj(d, il)], L.idx_ape.f32(), pooled_[stj(d, il)] + (size_t) p0 * ID, p0 * KP, p_end - p0, KP, ID, d.stream);
                    rows_q8_pad(pooled_[stj(d, il)] + (size_t) p0 * ID, pooled_q8_[stj(d, il)] + (size_t) p0 * (kIdxPad / 32) * 34, p_end - p0, ID, kIdxPad, d.stream);
                }
            }
        }
        prof_mark(d, "dsa_proj");
        dsa_lod(d, il, T, pos0, 1.0f / std::sqrt((float) HK));
    } else {
    gemv(d, T, L.idx_q_b, qr, d.iq, hp_.q_lora, IH * ID);
    if (kDebugSync) debug_sync(d.stream, il, "idx_q_b");
    if (dump) dump("indexer_q", d.iq, {IH, ID}, false, dv);
    gemv(d, T, L.idx_k, xn, d.ik, E, ID);
    if (kDebugSync) debug_sync(d.stream, il, "idx_k");
    if (rows_mode_) { layer_norm_t(d.ik, L.idx_k_norm.f32(), L.idx_k_norm_b.f32(), d.big_c, T, ID, hp_.norm_eps, d.stream); scatter_q8_rows(d.big_c, ik_tab_[(size_t) il], d.row_slot_d, d.row_pos_d, T, ID, d.stream); }
    else { layer_norm_t(d.ik, L.idx_k_norm.f32(), L.idx_k_norm_b.f32(), d.big_c, T, ID, hp_.norm_eps, d.stream); store_q8_rows(d.big_c, ik_cache_[stj(d, il)] + (size_t) pos0 * q8_pitch(ID), T, ID, d.stream); }
    if (dump) dump("indexer_k", d.big_c, {ID}, false, dv);
    if (rows_mode_) { gemv(d, T, L.idx_gate, xn, d.ig, E, ID); scatter_q8_rows(d.ig, ig_tab_[(size_t) il], d.row_slot_d, d.row_pos_d, T, ID, d.stream); }
    else { gemv(d, T, L.idx_gate, xn, d.ig, E, ID); store_q8_rows(d.ig, ig_cache_[stj(d, il)] + (size_t) pos0 * q8_pitch(ID), T, ID, d.stream); }
    if (kDebugSync) debug_sync(d.stream, il, "idx_gate");
    if (dump) dump("indexer_gate", d.ig, {ID}, false, dv);
    gemv(d, T, L.idx_proj, xn, d.iw, E, IH);
    scale_inplace(d.iw, T * IH, 1.0f / std::sqrt((float) (ID * IH)), d.stream);
    prof_mark(d, "dsa_proj");
    if (dump) dump("indexer_weights", d.iw, {IH}, false, dv);
    // the per-token part
    int* prm = d.sel_params_host;
    for (int t = 0; t < T; ++t) {
        const int n_kv = (rows_mode_ ? row_pos_[(size_t) t] : pos0 + t) + 1;
        const int nv = n_kv / KP, n_tail = n_kv % KP;
        prm[t * 8 + 0] = nv;
        prm[t * 8 + 1] = std::min(nv, hp_.idx_topk / KP);
        prm[t * 8 + 2] = n_kv;
        prm[t * 8 + 3] = hp_.kpool_tail ? n_tail : 0;
    }
    check(cudaMemcpyAsync(d.sel_params, prm, (size_t) T * 8 * 4, cudaMemcpyHostToDevice, s), "sel params h2d");
    // the pools completed inside this window (pool p covers cells p*KP .. p*KP+KP-1), all at once; token t reads
    // only the first (pos0+t+1)/KP of them
    {
        // the pools whose last cell falls in [pos0, pos0+T): p*KP + KP - 1 >= pos0 and < pos0 + T
        const int p0 = pos0 / KP;             // the first pool whose last cell (p*KP + KP - 1) is >= pos0
        const int p_end = (pos0 + T) / KP;    // ... and < pos0 + T
        if (p_end > p0) dsa_pool(ik_cache_[stj(d, il)], ig_cache_[stj(d, il)], L.idx_ape.f32(), pooled_[stj(d, il)] + (size_t) p0 * ID, p0 * KP, p_end - p0, KP, ID, d.stream);
    }
    prof_mark(d, "dsa_pool");
    const int ld_score = max_ctx_ / KP + 1;
    const float scale = 1.0f / std::sqrt((float) HK);
    if (rows_mode_) {
        dsa_pool_rows(ik_tab_[(size_t) il], ig_tab_[(size_t) il], pooled_tab_[(size_t) il], d.row_slot_d, d.row_pos_d, L.idx_ape.f32(), T, KP, ID, d.stream);
        int n_pool_max = 0;
        for (int t = 0; t < T; ++t) n_pool_max = std::max(n_pool_max, (row_pos_[(size_t) t] + 1) / KP);
        const int ld_score_r = max_ctx_ / KP + 1;
        dsa_score_rows(pooled_tab_[(size_t) il], d.row_slot_d, d.iq, d.iw, d.score, d.sel_params, n_pool_max, T, IH, ID, ld_score_r, d.stream);
        dsa_select(d.score, d.sel_params, T, KP, d.sel, ld_score_r, n_sel_max, d.stream);
        dsa_attend_rows(kvc_tab_[(size_t) il], nullptr, 0.f, d.row_slot_d, d.sel, d.sel_params, T, n_sel_max, d.q_abs, 1.0f / std::sqrt((float) HK), d.attn_scores, d.ctx, H, KV, d.stream);
        prof_mark(d, "dsa_attend");
    } else {
    // the dense (masked) attention of a prompt window: scores = q_abs . kvc^T and ctx = P . kvc as two MMQ products per
    // batch over the whole latent cache (Q8_0 rows, re-quantized per window), the selection as a bit mask; the
    // per-token kernels stay for decode windows and long contexts
    // n_pad: the cell count padded to the MMQ K step (its tiles take 128 values; the quantizer pads rows to 512)
    const int n_kv_max = pos0 + T, n_pad = (n_kv_max + 511) / 512 * 512;
    // the Q8_0 cache itself is the scores product's operand (its rows beyond the window zeroed up to n_pad: dead rows),
    // the context product takes its transpose
    const bool dense = T >= kMmqMinT && d.mmq && d.kvcT_q8 && n_pad <= kDenseAttnMaxKv && !kNoDenseAttn && mmq_fits(8, n_pad) && mmq_fits(8, KV);
    uint8_t* kvc_q8 = kvc_cache_[stj(d, il)];
    if (dense) {
        q8_0_rows_T(kvc_q8, KV, d.kvcT_q8, n_kv_max, n_pad, d.stream);
        if (n_pad > n_kv_max) check(cudaMemsetAsync(kvc_q8 + (size_t) n_kv_max * q8_pitch(KV), 0, (size_t) (n_pad - n_kv_max) * q8_pitch(KV), s), "kvc pad");
    }
    for (int t0 = 0; t0 < T; t0 += kDsaBatch) {
        const int TB = std::min(kDsaBatch, T - t0);
        int* sp = d.sel_params + t0 * 8;
        const int n_pool_max = (pos0 + t0 + TB) / KP;   // the last token of the batch sees the most pools
        dsa_score(pooled_[stj(d, il)], d.iq + (size_t) t0 * IH * ID, d.iw + (size_t) t0 * IH, d.score, sp, n_pool_max, TB, IH, ID, ld_score, d.stream);
        if (dump) dump("indexer_score", d.score, {n_pool_max}, false, dv);
        prof_mark(d, "dsa_score");
        dsa_select(d.score, sp, TB, KP, d.sel, ld_score, n_sel_max, d.stream);
        prof_mark(d, "dsa_select");
        if (dense) {
            namespace mmq = strata::prefill::mmq;
            const int words = n_pad / 32, rows = TB * H;
            check(cudaMemsetAsync(d.attn_mask, 0, (size_t) TB * words * 4, s), "mask zero");
            sel_mask(d.sel, sp, TB, n_sel_max, d.attn_mask, words, d.stream);
            if (d.attn_rows != rows) { set_pair(d.attn_bounds, 0, rows, d.stream); d.attn_rows = rows; }
            auto* ctx = static_cast<mmq::Context*>(d.mmq);
            mmq::quantize(d.q_abs + (size_t) t0 * H * KV, nullptr, d.attn_xq, 8, KV, KV, rows, d.stream);
            mmq::Product ps;
            ps.w = kvc_q8; ps.type = 8; ps.w_rows = n_pad; ps.w_cols = KV; ps.expert_bytes = (size_t) n_pad * (KV / 32) * 34; ps.n = 1;
            ps.xq = d.attn_xq; ps.bounds = d.attn_bounds; ps.ids = d.iota; ps.total_rows = rows; ps.max_rows = rows; ps.dst = d.attn_S; ps.ld_dst = n_pad;
            ctx->run(ps, d.stream);
            masked_softmax(d.attn_S, n_pad, d.attn_mask, words, rows, H, n_pad, scale, nullptr, d.stream);
            mmq::quantize(d.attn_S, nullptr, d.attn_pq, 8, n_pad, n_pad, rows, d.stream);
            mmq::Product pc;
            pc.w = d.kvcT_q8; pc.type = 8; pc.w_rows = KV; pc.w_cols = n_pad; pc.expert_bytes = (size_t) KV * (n_pad / 32) * 34; pc.n = 1;
            pc.xq = d.attn_pq; pc.bounds = d.attn_bounds; pc.ids = d.iota; pc.total_rows = rows; pc.max_rows = rows; pc.dst = d.ctx + (size_t) t0 * H * KV; pc.ld_dst = KV;
            ctx->run(pc, d.stream);
        } else
            dsa_attend(kvc_cache_[stj(d, il)], nullptr, 0.f, d.sel, sp, TB, n_sel_max, d.q_abs + (size_t) t0 * H * KV, scale, d.attn_scores,
                       d.ctx + (size_t) t0 * H * KV, H, KV, d.stream);
        prof_mark(d, "dsa_attend");
    }
    }   // rows_mode_
    }   // lod_.on
    float* o = d.big_b;
    if (kDebugSync) debug_sync(d.stream, il, "dsa attend");
    if (head_mmq) head_product(L.v_b_q8, d.ctx, KV, HV, o);
    else dsa_expand_t(L.v_b.bf16(), d.ctx, o, T, H, KV, HV, d.stream);
    if (kDebugSync) debug_sync(d.stream, il, "dsa expand");
    gemv(d, T, L.wo, o, out, H * HV, E);
    prof_mark(d, "dsa_out");
    if (dump) dump("attn_out", out, {E}, false, dv);
}

// the lookahead prediction: layer il + 1's router on this layer's FFN input, top lookahead_ per token, published
void Engine::predict_next(Dev& d, int il, int T, const float* xn) {
    const LayerWeights& N = m_.layer(il + 1);
    const int P = lookahead_;
    gemv(d, T, N.gate_inp, xn, d.pred_logits, hp_.n_embd, hp_.n_expert);
    router_sigmoid_topk_t(d.pred_logits, N.exp_probs_b.present() ? N.exp_probs_b.f32() : nullptr, T, hp_.n_expert, P, false, 1.f,
                          d.pred_ids, d.pred_wts, d.pred_probs, d.stream);
    moe_publish(d.pred_ids, T * P, d.tok_dev, d.db_pred_d + (size_t) il * t_chunk_ * 16, d.db_pred_pub_d + il, nullptr, 0, nullptr, d.stream);
}

void Engine::dense_ffn(Dev& d, int il, int T, const float* xn, float* out, const DumpFn& dump) {
    const LayerWeights& L = m_.layer(il);
    const int E = hp_.n_embd, FF = hp_.n_ff;
    if (T <= kShT) {   // gate | up as one product; the swiglu quantizes h for the down projection
        if (d.xn_q8 && xn == d.xn && T <= 8) gemv_q8(d, T, L.ffn_gu, d.xn_q8, d.dense_gu, E, 2 * FF);
        else gemv(d, T, L.ffn_gu, xn, d.dense_gu, E, 2 * FF);
        uint8_t* hq = kQ8Fuse && T <= 8 && FF % 32 == 0 ? d.q8_h : nullptr;   // ([T][q8_cols(FF)]: the pair buffer holds max_pairs_ such rows)
        swiglu_clamp_gu_t(d.dense_gu, d.ffn_h, T, FF, hp_.clamp_shexp[(size_t) il], d.stream, hq);
        if (hq) gemv_q8(d, T, L.ffn_down, hq, out, FF, E);
        else gemv(d, T, L.ffn_down, d.ffn_h, out, FF, E);
    } else {
        gemv(d, T, row_slice(L.ffn_gu, 0, FF), xn, d.ffn_g, E, FF);
        gemv(d, T, row_slice(L.ffn_gu, FF, FF), xn, d.ffn_u, E, FF);
        swiglu_clamp(d.ffn_g, d.ffn_u, d.ffn_h, T * FF, hp_.clamp_shexp[(size_t) il], d.stream);
        gemv(d, T, L.ffn_down, d.ffn_h, out, FF, E);
    }
    if (dump) dump("ffn_out", out, {E}, false, d.ordinal);
}

// layer il predicts layer il + 1's routing when il + 1 is a trunk MoE layer on the same device (its router is there)
bool Engine::predicts_next(int il) const {
    if (lookahead_ <= 0 || !experts_ || il + 1 >= hp_.n_layer || il + 1 < hp_.n_dense_lead) return false;
    if (m_.layer_device()[(size_t) il] != m_.layer_device()[(size_t) il + 1]) return false;
    return experts_->layer(il + 1).n_slots < hp_.n_expert;
}

// One step of the doorbell service: the next pending layer, when the device has published its routing.  Dense lead
// layers only prefetch (lookahead); a MoE layer's misses are staged / swapped in and its flag written.
bool Engine::serve_step() {
    const int K = hp_.n_expert_used, T = db_T_;
    const int il = pending_.load(std::memory_order_acquire);
    if (il >= hp_.n_layer) return false;
    if (il < hp_.n_dense_lead) {
        if (predicts_next(il) && !prompt_mode_) {
            Dev& dd = devs_[(size_t) dev_of_layer_[(size_t) il]];
            if (__atomic_load_n(dd.db_pred_pub_h + il, __ATOMIC_ACQUIRE) != jtok(dd)) return false;
            prefetch_from(il);
        }
        pending_.store(il + 1, std::memory_order_release);
        return true;
    }
    Dev& d = devs_[(size_t) dev_of_layer_[(size_t) il]];
    if (__atomic_load_n(d.db_pub_h + il, __ATOMIC_ACQUIRE) != jtok(d)) return false;
    const bool all_resident = experts_->layer(il).n_slots == hp_.n_expert;
    if (mid_mode_) {
        int nm = 0;
        if (!all_resident) {
            const int k = ring_buf(d, *d.job, il);
            const int di = experts_->device_index_of_layer(il);
            void* wait_ev = ring_wait_event(d, k, *d.job);
            if (mid_split_)
                nm = experts_->serve_mid(il, d.db_ids_h + (size_t) il * t_chunk_ * K, T * K, T, d.db_x_h + (size_t) (il & 1) * kDbXRows * hp_.n_embd, jtok(d),
                                         d.db_flag_h + il, d.db_flag_d + il, d.db_flag2_h + il, d.db_flag2_d + il, d.ffn_y, hp_.n_embd,
                                         d.cpu_skip_h + (size_t) il * hp_.n_expert, experts_->win_buf(di, k), experts_->layer(il).stride, wait_ev);
            else
            nm = experts_->serve_ring(il, d.db_ids_h + (size_t) il * t_chunk_ * K, T * K, jtok(d), d.db_flag_h + il, d.db_flag_d + il, experts_->win_buf(di, k),
                                      experts_->layer(il).stride, wait_ev);
            d.ring_user[(size_t) k] = il;
            d.ring_user_tok[(size_t) k] = jtok(d);
        } else __atomic_store_n(d.db_flag_h + il, jtok(d), __ATOMIC_RELEASE);
        if (kTrace) std::fprintf(stderr, "glm5: tok %u layer %d ring-served (%d experts)\n", jtok(d), il, nm);
        pending_.store(il + 1, std::memory_order_release);
        return true;
    }
    const int nm = (cpu_mode_ && !all_resident)
        ? experts_->serve_cpu(il, d.db_ids_h + (size_t) il * t_chunk_ * K, T * K, T, d.db_x_h + (size_t) (il & 1) * kDbXRows * hp_.n_embd, jtok(d), d.db_flag_h + il, d.db_flag_d + il,
                              d.db_flag2_h + il, d.db_flag2_d + il, d.ffn_y, hp_.n_embd, d.miss_cpu_h + (size_t) il * 64,
                              d.miss_slot_h + (size_t) il * t_chunk_ * K, d.miss_out_h + (size_t) il * t_chunk_ * K, swaps_now_)
        : experts_->serve(il, d.db_ids_h + (size_t) il * t_chunk_ * K, T * K, jtok(d), d.db_flag_h + il, d.db_flag_d + il, d.miss_slot_h + (size_t) il * t_chunk_ * K,
                          d.miss_out_h + (size_t) il * t_chunk_ * K, swaps_now_);
    if (kTrace) std::fprintf(stderr, "glm5: tok %u layer %d served (%d misses)\n", jtok(d), il, nm);
    if (predicts_next(il) && !prompt_mode_) prefetch_from(il);
    pending_.store(il + 1, std::memory_order_release);
    return true;
}

// after a doorbell failure: every layer's flags set to the token the device waits for, so a kernel parked in moe_wait
// returns (the rows it then reads are garbage; the host thread throws db_err_ at its next synchronization)
void Engine::release_waits() {
    for (auto& d : devs_) {
        if (!d.db_flag_h) continue;
        const uint32_t t = d.job ? d.job->tok : tok_;
        for (int il = 0; il < hp_.n_layer; ++il) {
            __atomic_store_n(d.db_flag_h + il, t, __ATOMIC_RELEASE);
            if (d.db_flag2_h) __atomic_store_n(d.db_flag2_h + il, t, __ATOMIC_RELEASE);
        }
    }
}

void Engine::db_main() {
    using clock = std::chrono::steady_clock;
    auto idle_since = clock::now();
    bool idle = false;
    while (!db_stop_.load(std::memory_order_acquire)) {
        if (!db_active_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            idle = false;
            continue;
        }
        bool progressed = false;
        try {
            progressed = serve_step();
        } catch (const std::exception& e) {
            db_err_ = e.what();
            std::fprintf(stderr, "glm5: doorbell thread: %s\n", e.what());
            db_failed_.store(true, std::memory_order_release);
            db_active_.store(false, std::memory_order_release);
            release_waits();   // the device's parked waits open (their data is garbage); the host thread sees db_failed_ at its next sync and throws
            continue;
        }
        if (progressed) { idle = false; continue; }
        if (pending_.load(std::memory_order_acquire) >= hp_.n_layer) { std::this_thread::yield(); continue; }
        // a bounded wait for the next publish: a device fault leaves it unwritten for ever
        if (!idle) { idle = true; idle_since = clock::now(); }
        else if (kTrace && std::chrono::duration<double>(clock::now() - idle_since).count() > 5.0) {
            const int il = pending_.load();
            Dev& d = devs_[(size_t) dev_of_layer_[(size_t) il]];
            cudaSetDevice(d.ordinal);
            const cudaError_t q = cudaStreamQuery(S(d.stream));
            uint32_t tok_dev = 0;
            cudaMemcpy(&tok_dev, d.tok_dev, 4, cudaMemcpyDeviceToHost);
            std::fprintf(stderr, "glm5: doorbell idle 5 s: pending %d, published %u (want %u), flag %u, device tok %u, stream %s\n", il, d.db_pub_h[il], jtok(d),
                         d.db_flag_h[il], tok_dev, q == cudaErrorNotReady ? "busy" : cudaGetErrorString(q));
            idle_since = clock::now();
        }
        else if (std::chrono::duration<double>(clock::now() - idle_since).count() > 30.0) {
            const int il = pending_.load();
            Dev& d = devs_[(size_t) dev_of_layer_[(size_t) il]];
            db_err_ = "glm5: layer " + std::to_string(il) + " never published its routing (token " + std::to_string(jtok(d)) + ", published " +
                      std::to_string(d.db_pub_h[il]) + ")";
            db_failed_.store(true, std::memory_order_release);
            db_active_.store(false, std::memory_order_release);
            release_waits();
        }
    }
}

void Engine::wait_served(int upto) {
    const auto t0 = std::chrono::steady_clock::now();
    while (pending_.load(std::memory_order_acquire) < upto) {
        if (db_failed_.load(std::memory_order_acquire)) throw std::runtime_error(db_err_);
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 600.0)   // a cold mapped file can take minutes per window
            throw std::runtime_error("glm5: the doorbell thread served " + std::to_string(pending_.load()) + " layers in 600 s, wanted " + std::to_string(upto));
        std::this_thread::yield();
    }
}

// the host half of the lookahead: layer il published its prediction of il + 1's experts; prefetch them into il + 1
void Engine::prefetch_from(int il) {
    Dev& d = devs_[(size_t) dev_of_layer_[(size_t) il]];
    const int P = lookahead_, T = jT(d);
    while (__atomic_load_n(d.db_pred_pub_h + il, __ATOMIC_ACQUIRE) != jtok(d)) {}   // published just before the ids
    const int nl = il + 1;
    static const int kPfCap = std::getenv("STRATA_GLM5_PF_CAP") ? std::atoi(std::getenv("STRATA_GLM5_PF_CAP")) : 3;
    experts_->prefetch_stage(nl, d.db_pred_h + (size_t) il * t_chunk_ * 16, T, P, jtok(d), kPfCap);   // into staging: no pf_apply, the layer's own flag covers it
}

// The MoE of T tokens: router per token, the T*K pairs grouped by expert (resident groups first), the group kernels
// read each expert once for all of its tokens, the combine sums each token's pairs in group order.
void Engine::moe(Dev& d, int il, int T, const float* xn, float* out, bool doorbell, const DumpFn& dump, bool last_chunk) {
    const LayerWeights& L = m_.layer(il);
    const int E = hp_.n_embd, FF = hp_.n_ff_exp, K = hp_.n_expert_used;
    const int dv = d.ordinal;
    cudaStream_t s = S(d.stream);
    gemv(d, T, L.gate_inp, xn, d.logits, E, hp_.n_expert);
    if (kDebugSync) { debug_sync(d.stream, il, "router gemv"); debug_nan(xn, T, E, il, "ffn_norm"); debug_nan(d.logits, T, hp_.n_expert, il, "router logits"); }
    if (dump) dump("ffn_moe_logits", d.logits, {hp_.n_expert}, false, dv);
    router_sigmoid_topk_t(d.logits, L.exp_probs_b.present() ? L.exp_probs_b.f32() : nullptr, T, hp_.n_expert, K, hp_.expert_norm,
                          hp_.expert_scale, d.ids, d.wts, d.probs, d.stream);
    if (dump) { dump("ffn_moe_probs", d.probs, {hp_.n_expert}, false, dv); dump("ffn_moe_topk", d.ids, {K}, true, dv); dump("ffn_moe_weights", d.wts, {K, 1}, false, dv); }
    if (kDebugSync) debug_sync(d.stream, il, "router");
    const int32_t* slot_of = d.neg_table;
    const uint8_t* slots = nullptr;
    const uint8_t* stage = nullptr;
    uint64_t stage_bytes = expert_stage_bytes_, blob = L.exp_gate.stride + L.exp_up.stride + L.exp_down.stride;
    if (experts_) {
        const LayerExperts& LE = experts_->layer(il);
        slot_of = LE.slot_of_dev;
        slots = LE.slots;
        blob = LE.stride;   // the slot pitch (padded to the types' block sizes for MMQ)
        const int di = experts_->device_index_of_layer(il);
        stage = experts_->stage(di);
        stage_bytes = experts_->stage_bytes(di);
    } else {
        if (!d.expert_stage) d.expert_stage = static_cast<uint8_t*>(alloc(expert_stage_bytes_ * (uint64_t) std::min(max_pairs_, hp_.n_expert), d.ordinal));
        stage = d.expert_stage;
    }
    if (false && doorbell && !prompt_mode_ && lookahead_ > 0 && il >= hp_.n_dense_lead && il > 0 && predicts_next(il - 1)) {   // the staged lookahead needs no placement
        // layer il - 1 predicted this layer's experts: take the host's placements before the dispatch is built
        pf_apply(d.pf_flag_d + il, d.tok_dev, d.pf_n_d + il, d.pf_e_d + (size_t) il * t_chunk_ * 16, d.pf_slot_d + (size_t) il * t_chunk_ * 16,
                 d.pf_out_d + (size_t) il * t_chunk_ * 16, const_cast<int32_t*>(slot_of), d.stream);
    }
    if (doorbell && !prompt_mode_ && predicts_next(il)) predict_next(d, il, T, xn);
    const bool all_resident = experts_ && experts_->layer(il).n_slots == hp_.n_expert;
    const bool cpu = cpu_mode_ && doorbell && !all_resident;
    const bool xrows = cpu || (mid_split_ && doorbell && !all_resident);   // the CPU service reads the MoE inputs
    if (doorbell && (!prompt_mode_ || mid_mode_))
        moe_publish(d.ids, T * K, d.tok_dev, d.db_ids_d + (size_t) il * t_chunk_ * K, d.db_pub_d + il, xrows ? xn : nullptr, xrows ? T * E : 0,
                    d.db_x_d + (size_t) (il & 1) * kDbXRows * E, d.stream);
    if (T >= kMmqMinT && d.mmq && (prompt_mode_ || all_resident)) {
        moe_mmq(d, il, T, xn, out, doorbell, slot_of, slots, blob, stage, stage_bytes, all_resident, dump, last_chunk);
        return;
    }
    // the loan slots' table updates (misses the earlier windows cached in the prompt ring) before this grouping
    if (experts_ && doorbell && !prompt_mode_ && il < hp_.n_layer) experts_->apply_loans(il, d.stream);
    moe_group(d.ids, d.wts, T, K, hp_.n_expert, slot_of, slots, blob, L.exp_gate.stride, L.exp_up.stride, stage, stage_bytes,
              experts_ ? experts_->loan_ptrs(experts_->device_index_of_layer(il)) : nullptr, d.grp_w, d.grp_wu,
              d.grp_wd, d.grp_col0, d.grp_ncols, d.grp_orow, d.pair_tok, d.pair_w, d.cnt, d.grp_e, d.grp_miss, true, nullptr, d.tok_pairs, 0, nullptr, nullptr, d.stream);
    if (kDebugSync) debug_sync(d.stream, il, "group");
    const size_t col_e = q8_cols(E);
    const int rows_max = T * K;   // this window's pairs (the buffers hold max_pairs_, a prompt window's)
    // the decode window's Q2_K products on the WMMA kernel (gfx11; 2.5-4x the grouped MMVQ's bandwidth): the pairs'
    // activations quantized per 256 as the prompt path does, the sums packed once per (pair, super-block)
    const bool decq2k = d.q2k_packed && !kNoDecQ2K && rows_max <= 64 && L.exp_gate.type == 10 && L.exp_up.type == 10 && L.exp_down.type == 10 &&
                        FF % 128 == 0 && E % 256 == 0 && FF % 256 == 0 && E % 128 == 0;
    if (decq2k) {
        strata::glm5::q2k::quantize_act(xn, d.pair_tok, rows_max, E, d.q2k_codes, d.q2k_scales, d.q2k_sums, d.stream, nullptr, nullptr, kPackFuse ? d.q2k_packed : nullptr);
        if (!kPackFuse) strata::glm5::q2k::pack_sums(d.q2k_sums, rows_max, E, d.q2k_packed, d.stream);
    } else {
        if (!(d.xn_q8 == d.q8_x && xn == d.xn))   // (the kernel that wrote xn may have quantized it into q8_x already)
            strata::kernels::native_quantize_q8_1(xn, d.q8_x, E * T, 1, d.stream);   // T contiguous rows as one vector (blocks of 32 stay inside a row)
        gather_q8(d.q8_x, d.pair_tok, d.cnt + 2, d.q8_g, (int) col_e, rows_max, d.stream);
    }
    if (kDebugSync) debug_sync(d.stream, il, "gather");
    prof_mark(d, "router+group");
    if (!doorbell && !all_resident) {
        // no doorbell: the host reads the routing now and stages the distinct missed experts (first-occurrence order,
        // the order k_moe_group assigns) before the kernels that read them
        check(cudaMemcpyAsync(d.ids_host, d.ids, (size_t) T * K * 4, cudaMemcpyDeviceToHost, s), "ids d2h");
        check(cudaStreamSynchronize(s), "sync");
        if (experts_) {
            uint32_t flag = 0;
            experts_->serve(il, d.ids_host, T * K, ++adapt_tok_, &flag, nullptr, d.miss_slot_h + (size_t) il * t_chunk_ * K, d.miss_out_h + (size_t) il * t_chunk_ * K, swaps_now_);
            experts_->join(il, d.stream);
        } else {
            std::vector<int> uniq;
            for (int p = 0; p < T * K; ++p)
                if (std::find(uniq.begin(), uniq.end(), d.ids_host[p]) == uniq.end()) uniq.push_back(d.ids_host[p]);
            std::sort(uniq.begin(), uniq.end());   // k_moe_group's staging order: ascending expert id
            if (!d.expert_stage) d.expert_stage = static_cast<uint8_t*>(alloc(expert_stage_bytes_ * (uint64_t) std::min(max_pairs_, hp_.n_expert), d.ordinal));
            for (size_t m = 0; m < uniq.size(); ++m) {
                uint8_t* dst = d.expert_stage + m * expert_stage_bytes_;
                const int e = uniq[m];
                check(cudaMemcpyAsync(dst, L.exp_gate.base + (uint64_t) e * L.exp_gate.stride, L.exp_gate.stride, cudaMemcpyHostToDevice, s), "gate h2d");
                check(cudaMemcpyAsync(dst + L.exp_gate.stride, L.exp_up.base + (uint64_t) e * L.exp_up.stride, L.exp_up.stride, cudaMemcpyHostToDevice, s), "up h2d");
                check(cudaMemcpyAsync(dst + L.exp_gate.stride + L.exp_up.stride, L.exp_down.base + (uint64_t) e * L.exp_down.stride, L.exp_down.stride, cudaMemcpyHostToDevice, s), "down h2d");
            }
        }
    }
    const float limit = hp_.clamp_exp[(size_t) il];
    // zero = the start of the group list (cnt[4] == 0), cnt[0] = the first missed group, cnt[1] = the end
    const int* g0 = d.cnt + 4;
    // the hits: groups [0, cnt[0]) - gate then up
    auto gate_up = [&](const int* gb, const int* ge) {
        if (decq2k) {
            strata::glm5::q2k::dec_wmma(d.grp_w, d.grp_col0, d.grp_ncols, d.grp_orow, gb, ge, d.q2k_codes, d.q2k_scales, d.q2k_packed, d.ffn_g, E, FF, d.stream);
            strata::glm5::q2k::dec_wmma(d.grp_wu, d.grp_col0, d.grp_ncols, d.grp_orow, gb, ge, d.q2k_codes, d.q2k_scales, d.q2k_packed, d.ffn_u, E, FF, d.stream);
        } else {
            strata::kernels::native_mmvq_grouped((int) L.exp_gate.type, d.grp_w, d.q8_g, d.grp_col0, d.grp_ncols, d.grp_orow, gb, ge, d.ffn_g, E, FF, max_pairs_, d.stream);
            strata::kernels::native_mmvq_grouped((int) L.exp_up.type, d.grp_wu, d.q8_g, d.grp_col0, d.grp_ncols, d.grp_orow, gb, ge, d.ffn_u, E, FF, max_pairs_, d.stream);
        }
    };
    gate_up(g0, d.cnt);
    prof_mark(d, "moe_hits");
    const int SF = FF * std::max(1, hp_.n_expert_shared);
    // the shared experts (from xn alone) and the resident experts' down products go before the wait on the served
    // misses: the GPU computes them while the CPU serves (a decode round waits ~50 ms there), and the misses' rows
    // (pair rows [cnt[6], cnt[2]), groups [cnt[0], cnt[1])) follow the wait
    const bool early = kMoeEarly && doorbell && decq2k && !all_resident && !dump && T <= kShT;
    auto shared_experts = [&](float* g_buf, float* h_buf) {
        if (d.xn_q8 && xn == d.xn && T <= 8) gemv_q8(d, T, L.sh_gu, d.xn_q8, g_buf, E, 2 * SF);
        else gemv(d, T, L.sh_gu, xn, g_buf, E, 2 * SF);   // gate | up as one product (one launch, twice the rows per launch)
        uint8_t* hq = kQ8Fuse && T <= 8 && SF % 32 == 0 ? d.q8_sh : nullptr;
        swiglu_clamp_gu_t(g_buf, h_buf, T, SF, hp_.clamp_shexp[(size_t) il], d.stream, hq);
        if (hq) gemv_q8(d, T, L.sh_down, hq, d.ffn_d, SF, E);
        else gemv(d, T, L.sh_down, h_buf, d.ffn_d, SF, E);
    };
    if (early) {
        swiglu_clamp_rows(d.ffn_g, d.ffn_u, d.ffn_h, d.cnt + 6, rows_max, FF, limit, d.stream);
        strata::glm5::q2k::quantize_act(d.ffn_h, nullptr, rows_max, FF, d.dec_codes_h, d.dec_scales_h, d.dec_sums_h, d.stream, nullptr, d.cnt + 6, kPackFuse ? d.dec_packed_h : nullptr);
        if (!kPackFuse) strata::glm5::q2k::pack_sums(d.dec_sums_h, rows_max, FF, d.dec_packed_h, d.stream, nullptr, d.cnt + 6);
        strata::glm5::q2k::dec_wmma(d.grp_wd, d.grp_col0, d.grp_ncols, d.grp_orow, g0, d.cnt, d.dec_codes_h, d.dec_scales_h, d.dec_packed_h, d.ffn_y, FF, E, d.stream);
        shared_experts(d.sh_g, d.sh_h);
    }
    if (doorbell) moe_wait(d.db_flag_d + il, d.tok_dev, d.stream);
    prof_mark(d, "moe_wait");
    if (cpu) moe_cpu_mask(d.cnt, d.grp_miss, d.miss_cpu_d + (size_t) il * 64, d.grp_ncols, d.stream);
    if (experts_ && !all_resident)
        moe_place(d.cnt, d.grp_miss, d.grp_e, d.miss_slot_d + (size_t) il * t_chunk_ * K, d.miss_out_d + (size_t) il * t_chunk_ * K, slots, blob,
                  L.exp_gate.stride, L.exp_up.stride, d.grp_w, d.grp_wu, d.grp_wd, const_cast<int32_t*>(slot_of), d.stream);
    // the misses: groups [cnt[0], cnt[1]) (in cpu mode the CPU-served groups have ncols 0: their pair rows came from the host)
    gate_up(d.cnt, d.cnt + 1);
    prof_mark(d, "moe_misses");
    if (early) {
        swiglu_clamp_rows(d.ffn_g, d.ffn_u, d.ffn_h, d.cnt + 2, rows_max, FF, limit, d.stream, d.cnt + 6);
        strata::glm5::q2k::quantize_act(d.ffn_h, nullptr, rows_max, FF, d.dec_codes_h, d.dec_scales_h, d.dec_sums_h, d.stream, d.cnt + 6, nullptr, kPackFuse ? d.dec_packed_h : nullptr);
        if (!kPackFuse) strata::glm5::q2k::pack_sums(d.dec_sums_h, rows_max, FF, d.dec_packed_h, d.stream, d.cnt + 6, nullptr);
        strata::glm5::q2k::dec_wmma(d.grp_wd, d.grp_col0, d.grp_ncols, d.grp_orow, d.cnt, d.cnt + 1, d.dec_codes_h, d.dec_scales_h, d.dec_packed_h, d.ffn_y, FF, E, d.stream);
    } else {
        swiglu_clamp_rows(d.ffn_g, d.ffn_u, d.ffn_h, d.cnt + 2, rows_max, FF, limit, d.stream);
        if (decq2k) {
            strata::glm5::q2k::quantize_act(d.ffn_h, nullptr, rows_max, FF, d.dec_codes_h, d.dec_scales_h, d.dec_sums_h, d.stream, nullptr, nullptr, kPackFuse ? d.dec_packed_h : nullptr);
            if (!kPackFuse) strata::glm5::q2k::pack_sums(d.dec_sums_h, rows_max, FF, d.dec_packed_h, d.stream);
            strata::glm5::q2k::dec_wmma(d.grp_wd, d.grp_col0, d.grp_ncols, d.grp_orow, g0, d.cnt + 1, d.dec_codes_h, d.dec_scales_h, d.dec_packed_h, d.ffn_y, FF, E, d.stream);
        } else {
            strata::kernels::native_quantize_q8_1(d.ffn_h, d.q8_h, FF * rows_max, 1, d.stream);   // this window's pair rows in one launch
            strata::kernels::native_mmvq_grouped((int) L.exp_down.type, d.grp_wd, d.q8_h, d.grp_col0, d.grp_ncols, d.grp_orow, g0, d.cnt + 1, d.ffn_y, FF, E, max_pairs_, d.stream);
        }
    }
    if (cpu) moe_wait(d.db_flag2_d + il, d.tok_dev, d.stream);   // the CPU's rows for its pairs are in ffn_y
    // the pairs of a token summed in the router's top-k order: the same bits whichever experts happen to be resident
    // (the pair list puts the hits first, so its order moves with the residency)
    if (early) {   // the shared experts' rows added in the combine, the result straight into out
        combine_tok(d.ffn_y, d.pair_w, d.tok_pairs, K, out, T, E, d.stream, d.ffn_d);
        prof_mark(d, "moe_down");
        prof_mark(d, "shared_experts");
        return;
    }
    combine_tok(d.ffn_y, d.pair_w, d.tok_pairs, K, d.ffn_acc, T, E, d.stream);
    prof_mark(d, "moe_down");
    if (dump) dump("ffn_moe_out", d.ffn_acc, {E}, false, dv);
    shared_experts(d.ffn_g, d.ffn_h);
    if (dump) dump("ffn_shexp", d.ffn_d, {E}, false, dv);
    add_rows(d.ffn_acc, d.ffn_d, T * E, d.stream);
    check(cudaMemcpyAsync(out, d.ffn_acc, (size_t) T * E * 4, cudaMemcpyDeviceToDevice, s), "ffn out");
    prof_mark(d, "shared_experts");
    if (dump) dump("ffn_out", out, {E}, false, dv);
}

// the MoE layers of device d in order: the first after `after`, or n_layer when none
void Engine::set_experts(ExpertStore* store) {
    experts_ = store;
    if (!store) return;
    store->set_clamp(hp_.clamp_exp);
    cpu_misses_ = !kNoCpu;
    if (cpu_misses_) store->init_cpu(m_);
    const int nbuf = store->win_bufs();
    for (auto& d : devs_) {
        check(cudaSetDevice(d.ordinal), "cudaSetDevice");
        const int NL = m_.n_layer_all();
        d.layer_ev.assign((size_t) NL * 2, nullptr);
        d.layer_ev_valid.assign((size_t) NL * 2, 0);
        d.ring_user.assign((size_t) nbuf, -1);
        d.ring_user_tok.assign((size_t) nbuf, 0);
        for (int i = 0; i < NL * 2; ++i) { cudaEvent_t e; check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "layer event"); d.layer_ev[(size_t) i] = e; }
    }
    static const bool kLendScratch = !std::getenv("STRATA_GLM5_LOAN_SCRATCH") || std::atoi(std::getenv("STRATA_GLM5_LOAN_SCRATCH")) != 0;   // 0: the ring's slots only
    if (store->loans_on() && kLendScratch) {
        for (auto& d : devs_)
            for (const Dev::Lend& r : d.lend) store->add_loan_region(d.ordinal, r.base, r.bytes, r.skip);
        store->finalize_loans();
    }
}

// the event the next user of ring buffer k waits on: the previous user's record - this window's when that layer ran
// earlier in this window (recorded before the next user's publish), else the previous window's parity
void* Engine::ring_wait_event(Dev& d, int k, const Job& j) const {
    const int prev = d.ring_user[(size_t) k];
    if (prev < 0) return nullptr;
    const int parity = d.ring_user_tok[(size_t) k] == j.tok ? j.parity : j.parity ^ 1;
    const size_t i = (size_t) prev * 2 + (size_t) parity;
    return d.layer_ev_valid[i] ? d.layer_ev[i] : nullptr;
}

int Engine::ring_buf(const Dev& d, const Job& j, int il) const {
    const size_t di = (size_t) (&d - devs_.data());
    const int nbuf = experts_->win_bufs();
    return nbuf > 0 ? (j.rbase[di] + moe_seq_[(size_t) il]) % nbuf : 0;
}

// The ring position of a window's first MoE layer per device: consecutive windows continue the ring (with an odd
// MoE layer count per device, the next window's first layer takes the other buffer than this window's last one).
void Engine::assign_ring(Job& j) {
    const int nbuf = std::max(1, experts_->win_bufs());
    j.rbase.resize(devs_.size());
    for (size_t di = 0; di < devs_.size(); ++di) { j.rbase[di] = ring_next_[di]; ring_next_[di] = (ring_next_[di] + n_dev_moe_[di]) % nbuf; }
}

int Engine::next_moe_layer_on(Dev& d, int after) const {
    for (int il = std::max(after + 1, hp_.n_dense_lead); il < hp_.n_layer; ++il)
        if (devs_[(size_t) dev_of_layer_[(size_t) il]].ordinal == d.ordinal) return il;
    return hp_.n_layer;
}
// a prompt window: this device's MoE layers with sequence < seq_limit prefetched into the ring (buffer = seq % ring
// size), each after the buffer's last reader; the ring lets an idle device (the other one is computing) fill its
// first buffers at window start and every device run several layers ahead
void Engine::prefetch_up_to(Dev& d, Job& j0, int seq_limit) {
    // this device's ring: the MoE layers of job j0 with sequence < seq_limit, then (the sequence continuing) the next
    // job's first layers - the buffer's previous reader's event (recorded before this call, by this stage) orders the DMA
    std::lock_guard<std::mutex> lk(*d.pf_mutex);
    const size_t di = (size_t) (&d - devs_.data());
    const int n_dev_moe = n_dev_moe_[di];
    Job* j = &j0;
    int base = 0;   // the sequence offset of j relative to j0
    for (;;) {
        if (j->pf_next[di] >= hp_.n_layer) {
            if (!j->next) return;
            j = j->next; base += n_dev_moe;
            continue;
        }
        const int pl = j->pf_next[di];
        if (moe_seq_[(size_t) pl] + base >= seq_limit) return;
        const int k = ring_buf(d, *j, pl);
        const uint64_t blob = experts_->layer(pl).stride;
        void* wait_ev = ring_wait_event(d, k, *j);
        if (experts_->nonresident(pl) > 0)
            experts_->prefetch_all(pl, j->tok, d.db_flag_h + pl, d.db_flag_d + pl, experts_->win_buf((int) di, k), blob, wait_ev);
        else __atomic_store_n(d.db_flag_h + pl, j->tok, __ATOMIC_RELEASE);
        d.ring_user[(size_t) k] = pl;
        d.ring_user_tok[(size_t) k] = j->tok;
        j->pf_next[di] = next_moe_layer_on(d, pl);
    }
}

// The prompt windows' experts through MMQ: one ascending-order grouping (pair ranges per expert = bounds), the
// doorbell wait first (every expert of the window is needed), the window's experts gathered into contiguous
// [n_expert] slots, then two int8 tensor-core products over all experts (gate|up, then down) and the pair combine.
void Engine::moe_mmq(Dev& d, int il, int T, const float* xn, float* out, bool doorbell, const int32_t* slot_of, const uint8_t* slots,
                     uint64_t blob, const uint8_t* stage, uint64_t stage_bytes, bool all_resident, const DumpFn& dump, bool last_chunk) {
    namespace mmq = strata::prefill::mmq;
    const LayerWeights& L = m_.layer(il);
    const int E = hp_.n_embd, FF = hp_.n_ff_exp, K = hp_.n_expert_used;
    cudaStream_t s = S(d.stream);
    (void) stage; (void) stage_bytes; (void) doorbell;
    const int di = experts_->device_index_of_layer(il);
    const int nbuf = experts_->win_bufs();
    const int seq = il < hp_.n_layer ? moe_seq_[(size_t) il] : -1;
    const int k = seq >= 0 && nbuf && d.job ? ring_buf(d, *d.job, il) : 0;
    const int n_slots = experts_->layer(il).n_slots, n_nonres = hp_.n_expert - n_slots;
    uint8_t* buf = (prompt_mode_ && !all_resident) ? experts_->win_buf(di, k) : nullptr;
    const auto th0 = std::chrono::steady_clock::now();
    const auto th1 = std::chrono::steady_clock::now();
    moe_group(d.ids, d.wts, T, K, hp_.n_expert, slot_of, slots, blob, L.exp_gate.stride, L.exp_up.stride, buf, blob, nullptr, d.grp_w, d.grp_wu,
              d.grp_wd, d.grp_col0, d.grp_ncols, d.grp_orow, d.pair_tok, d.pair_w, d.cnt, d.grp_e, d.grp_miss, false, nullptr, d.tok_pairs, n_slots,
              d.bounds_res, d.bounds_miss, d.stream, mid_split_ ? 1 << 20 : 64, d.pair_rank);
    if (kDebugSync) debug_sync(d.stream, il, "group");
    // the activations of every pair, quantized for MMQ (row p = token pair_tok[p]); the stale rows past cnt[2] are harmless
    const bool q2k = d.q2k_codes && L.exp_gate.type == 10 && L.exp_up.type == 10 && L.exp_down.type == 10 && (2 * FF) % 128 == 0 && E % 256 == 0 && FF % 256 == 0;
    static bool noted = false;
    if (!noted) { noted = true; std::fprintf(stderr, "glm5: prompt experts: layer %d %s (types %u/%u/%u)\n", il, q2k ? "Q2_K on WMMA" : "MMQ", L.exp_gate.type, L.exp_up.type, L.exp_down.type); }
    if (q2k) { strata::glm5::q2k::quantize_act(xn, d.pair_tok, max_pairs_, E, d.q2k_codes, d.q2k_scales, d.q2k_sums, d.stream); strata::glm5::q2k::pack_sums(d.q2k_sums, max_pairs_, E, d.q2k_packed, d.stream); }
    else mmq::quantize(xn, d.pair_tok, d.mmq_xq, (int) L.exp_gate.type, E, E, pairs_alloc_, d.stream);
    prof_mark(d, "router+group");
    // the non-resident experts arrive by the prefetch (its flag); the resident ones are read in place in the slot buffer
    if (!all_resident) moe_wait(d.db_flag_d + il, d.tok_dev, d.stream);
    prof_mark(d, "moe_wait");
    // the widest group (rows of one expert) sizes the MMQ launch: 1024 would make every expert an 8-tile launch
    const auto th2 = std::chrono::steady_clock::now();
    check(cudaMemcpyAsync(d.cnt_host, d.cnt + 5, 4, cudaMemcpyDeviceToHost, s), "cnt d2h");
    check(cudaStreamSynchronize(s), "cnt sync");
    if (doorbell_ && db_failed_.load(std::memory_order_acquire)) throw std::runtime_error(db_err_);
    const auto th3 = std::chrono::steady_clock::now();
    const int max_rows = std::max(1, *d.cnt_host);
    if (kDebugSync) {
        std::vector<int> hc(8);
        check(cudaMemcpy(hc.data(), d.cnt, 32, cudaMemcpyDeviceToHost), "cnt d2h");
        std::fprintf(stderr, "glm5: layer %d mmq: T %d max_rows %d cnt {%d %d %d %d %d %d} n_slots %d nonres %d type %u/%u blob %llu buf %p\n",
                     il, T, max_rows, hc[0], hc[1], hc[2], hc[3], hc[4], hc[5], n_slots, n_nonres, L.exp_gate.type, L.exp_down.type, (unsigned long long) blob, (void*) buf);
    }
    // the MMQ column tile per type, measured on gfx1100 with this routing skew (docs/GLM5_NEXT.md): Q2_K 8 (its 32 / 96 tiles
    // run 15-100x slower, 64 / 128 at half speed), Q3_K / Q4_K 24; the auto choice (the widest expert's rows) picked 64-128
    static const int kJ2K = std::getenv("STRATA_GLM5_MMQ_J2K") ? std::atoi(std::getenv("STRATA_GLM5_MMQ_J2K")) : 8;   // the Q2_K tile hint (J=16 lands; 64 with the I-64 config)
    auto tile_for = [](int type) { return type == 10 ? kJ2K : 24; };
    auto* ctx = static_cast<mmq::Context*>(d.mmq);
    // two products per matrix: the resident experts in the slot buffer (bounds in slot order) and the prefetched ones in
    // the ring buffer (bounds by rank); both write the pair rows they own
    mmq::Product gu;
    gu.w = slots; gu.type = (int) L.exp_gate.type; gu.w_rows = 2 * FF; gu.w_cols = E; gu.expert_bytes = blob; gu.n = n_slots;
    gu.xq = d.mmq_xq; gu.bounds = d.bounds_res; gu.ids = d.iota; gu.total_rows = pairs_alloc_; gu.max_rows = max_rows; gu.ncols_opt = tile_for((int) L.exp_gate.type); gu.dst = d.gu_out; gu.ld_dst = 2 * FF;
    if (kDebugSync) std::fprintf(stderr, "glm5: layer %d product A: w %p type %d rows %lld cols %lld stride %zu n %d total_rows %lld max_rows %lld opt %lld dst %p ld %lld\n", il, gu.w, gu.type,
                                 (long long) gu.w_rows, (long long) gu.w_cols, gu.expert_bytes, gu.n, (long long) gu.total_rows, (long long) gu.max_rows, (long long) gu.ncols_opt, (void*) gu.dst, (long long) gu.ld_dst);
    const int32_t* skip = mid_split_ ? d.cpu_skip_d + (size_t) il * hp_.n_expert : nullptr;   // the CPU-served experts of this window
    if (q2k) {
        namespace q = strata::glm5::q2k;
        if (n_slots > 0) q::experts(slots, blob, 0, 2 * FF, E, d.bounds_res, n_slots, max_rows, d.q2k_codes, d.q2k_scales, d.q2k_packed, d.gu_out, 2 * FF, d.stream);
        if (n_nonres > 0) q::experts(buf, blob, 0, 2 * FF, E, d.bounds_miss, n_nonres, max_rows, d.q2k_codes, d.q2k_scales, d.q2k_packed, d.gu_out, 2 * FF, d.stream, skip);
    } else {
    if (n_slots > 0) ctx->run(gu, d.stream);
    if (kDebugSync) { debug_sync(d.stream, il, "mmq gate|up A"); debug_nan(d.gu_out, std::min(max_pairs_, 64), 2 * FF, il, "gate|up rows"); }
    if (n_nonres > 0) { mmq::Product g2 = gu; g2.w = buf; g2.n = n_nonres; g2.bounds = d.bounds_miss; ctx->run(g2, d.stream); }
    }
    const auto th4 = std::chrono::steady_clock::now();
    if (kDebugSync) debug_sync(d.stream, il, "mmq gate|up");
    prof_mark(d, "moe_hits");
    swiglu_clamp_gu(d.gu_out, d.ffn_h, d.cnt + 2, max_pairs_, FF, hp_.clamp_exp[(size_t) il], d.stream);
    if (kDebugSync) debug_sync(d.stream, il, "swiglu");
    if (q2k) { strata::glm5::q2k::quantize_act(d.ffn_h, nullptr, max_pairs_, FF, d.q2k_codes, d.q2k_scales, d.q2k_sums, d.stream); strata::glm5::q2k::pack_sums(d.q2k_sums, max_pairs_, FF, d.q2k_packed, d.stream); }
    else mmq::quantize(d.ffn_h, nullptr, d.mmq_xq_h, (int) L.exp_down.type, FF, FF, pairs_alloc_, d.stream);
    if (kDebugSync) debug_sync(d.stream, il, "quantize h");
    mmq::Product dn;
    dn.w = slots + L.exp_gate.stride + L.exp_up.stride; dn.type = (int) L.exp_down.type; dn.w_rows = E; dn.w_cols = FF; dn.expert_bytes = blob; dn.n = n_slots;
    dn.xq = d.mmq_xq_h; dn.bounds = d.bounds_res; dn.ids = d.iota; dn.total_rows = pairs_alloc_; dn.max_rows = max_rows; dn.ncols_opt = tile_for((int) L.exp_down.type); dn.dst = d.ffn_y; dn.ld_dst = E;
    if (q2k) {
        namespace q = strata::glm5::q2k;
        const uint64_t off = L.exp_gate.stride + L.exp_up.stride;
        if (n_slots > 0) q::experts(slots, blob, off, E, FF, d.bounds_res, n_slots, max_rows, d.q2k_codes, d.q2k_scales, d.q2k_packed, d.ffn_y, E, d.stream);
        if (n_nonres > 0) q::experts(buf, blob, off, E, FF, d.bounds_miss, n_nonres, max_rows, d.q2k_codes, d.q2k_scales, d.q2k_packed, d.ffn_y, E, d.stream, skip);
    } else {
    if (n_slots > 0) ctx->run(dn, d.stream);
    if (kDebugSync) { debug_sync(d.stream, il, "mmq down A"); debug_nan(d.ffn_y, std::min(max_pairs_, 64), E, il, "down rows"); }
    if (n_nonres > 0) { mmq::Product d2 = dn; d2.w = buf + L.exp_gate.stride + L.exp_up.stride; d2.n = n_nonres; d2.bounds = d.bounds_miss; ctx->run(d2, d.stream); }
    }
    if (kDebugSync) debug_sync(d.stream, il, "mmq down");
    if (mid_split_ && !all_resident) moe_wait(d.db_flag2_d + il, d.tok_dev, d.stream);   // the CPU's rows for its experts are in ffn_y
    combine_tok(d.ffn_y, d.pair_w, d.tok_pairs, K, d.ffn_acc, T, E, d.stream);
    // this ring buffer's last reader: the layer nbuf ahead on this device prefetches into it after this event
    if (buf && last_chunk) {   // this ring buffer's last reader (the window's last chunk of this layer)
        const size_t ei = (size_t) il * 2 + (size_t) jparity(d);
        { std::lock_guard<std::mutex> lk(*d.pf_mutex); check(cudaEventRecord(static_cast<cudaEvent_t>(d.layer_ev[ei]), s), "layer event"); d.layer_ev_valid[ei] = 1; }
        if (!mid_mode_) prefetch_up_to(d, *const_cast<Job*>(d.job), seq + nbuf + 1);
    }
    if (kProf) {
        const auto th5 = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        host_prof_[0] += ms(th0, th1); host_prof_[1] += ms(th1, th2); host_prof_[2] += ms(th2, th3); host_prof_[3] += ms(th3, th4); host_prof_[4] += ms(th4, th5);
    }
    prof_mark(d, "moe_down");
    if (dump) dump("ffn_moe_out", d.ffn_acc, {E}, false, d.ordinal);
    const int SF = FF * std::max(1, hp_.n_expert_shared);
    gemv(d, T, L.sh_gu, xn, d.ffn_g, E, 2 * SF);   // gate | up as one product (one launch, twice the rows per launch)
    swiglu_clamp_gu_t(d.ffn_g, d.ffn_h, T, SF, hp_.clamp_shexp[(size_t) il], d.stream);
    gemv(d, T, L.sh_down, d.ffn_h, d.ffn_d, SF, E);
    add_rows(d.ffn_acc, d.ffn_d, T * E, d.stream);
    check(cudaMemcpyAsync(out, d.ffn_acc, (size_t) T * E * 4, cudaMemcpyDeviceToDevice, s), "ffn out");
    prof_mark(d, "shared_experts");
}

// llama.cpp's MMQ has a tile for this (type, rows) on every visible GPU (its check is a config scan: cached)
bool Engine::mmq_fits(int type, int rows) {
    static std::mutex m;   // the stage threads of two devices consult the cache concurrently
    std::lock_guard<std::mutex> lk(m);
    const auto key = std::make_pair(type, rows);
    auto it = mmq_fits_.find(key);
    if (it != mmq_fits_.end()) return it->second;
    const bool ok = strata::prefill::mmq::fits(type, rows);
    if (!ok) std::fprintf(stderr, "glm5: MMQ has no tile for type %d x %d rows: that product keeps the GEMM path\n", type, rows);
    mmq_fits_[key] = ok;
    return ok;
}

void Engine::prof_mark(Dev& d, const char* what) {
    if (!kProf || pipe_on_) return;
    cudaEvent_t ev;
    cudaEventCreate(&ev);
    cudaEventRecord(ev, S(d.stream));
    prof_.push_back({d.ordinal, what, ev});
}

void Engine::prof_report(int T) {
    if (!kProf || prof_.empty()) return;
    for (auto& dd : devs_) { cudaSetDevice(dd.ordinal); cudaStreamSynchronize(S(dd.stream)); }
    std::map<std::string, double> acc;
    double total = 0;
    for (size_t i = 1; i < prof_.size(); ++i) {
        if (prof_[i].dev != prof_[i - 1].dev) continue;   // the chain breaks at a device hand-off
        float ms = 0;
        cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(prof_[i - 1].ev), static_cast<cudaEvent_t>(prof_[i].ev));
        acc[prof_[i].what] += ms;
        total += ms;
    }
    if (host_prof_[0] + host_prof_[2] > 0) { std::fprintf(stderr, "glm5 prof: host per window (ms): prefetch enqueue %.0f, group+quant enqueue %.0f, cnt sync wait %.0f, mmq gate|up enqueue %.0f, rest of moe %.0f\n", host_prof_[0], host_prof_[1], host_prof_[2], host_prof_[3], host_prof_[4]); for (double& v : host_prof_) v = 0; }
    std::fprintf(stderr, "glm5 prof: window T %d, %.0f ms of GPU phases:", T, total);
    std::vector<std::pair<double, std::string>> v;
    for (auto& kv : acc) v.push_back({kv.second, kv.first});
    std::sort(v.rbegin(), v.rend());
    for (auto& e : v) std::fprintf(stderr, " %s %.0f (%.0f%%)", e.second.c_str(), e.first, 100.0 * e.first / total);
    std::fprintf(stderr, "\n");
    for (auto& m : prof_) cudaEventDestroy(static_cast<cudaEvent_t>(m.ev));
    prof_.clear();
}

void Engine::forward_window(const int32_t* tokens, int T, const DumpFn& dump_in) {
    cur_slot_ = n_slots_;
    pos_ = pos_s_[(size_t) cur_slot_];
    try { forward_window_impl(tokens, T, dump_in); } catch (...) { faulted_ = true; throw; }
    pos_s_[(size_t) cur_slot_] = pos_;
}

void Engine::forward_window_slot(int slot, const int32_t* tokens, int T) {
    if (slot < 0 || slot > n_slots_) throw std::runtime_error("glm5: no slot " + std::to_string(slot));
    cur_slot_ = slot;
    pos_ = pos_s_[(size_t) slot];
    try { forward_window_impl(tokens, T, DumpFn()); } catch (...) { faulted_ = true; throw; }
    pos_s_[(size_t) slot] = pos_;
}

void Engine::forward_window_impl(const int32_t* tokens, int T, const DumpFn& dump_in) {
    const auto tw0 = std::chrono::steady_clock::now();
    const uint64_t misses0 = experts_ ? experts_->misses() : 0;
    const uint64_t swaps0 = experts_ ? experts_->swaps() : 0;
    const uint64_t pf0 = experts_ ? experts_->prefetched() : 0;
    const uint64_t dist0 = experts_ ? experts_->distinct_misses() : 0, thr0 = experts_ ? experts_->thrash() : 0, lay0 = experts_ ? experts_->layers_missed() : 0;
    if (T < 1 || T > t_max_) throw std::runtime_error("glm5: window of " + std::to_string(T) + " tokens (max " + std::to_string(t_max_) + ")");
    if (!rows_mode_ && pos_ + T > max_ctx_) throw std::runtime_error("glm5: the context (" + std::to_string(max_ctx_) + ") is full");
    const DumpFn& dump = T == 1 ? dump_in : DumpFn();
    win_start_ = pos_;
    win_T_ = T;
    if (!rows_mode_) for (int t = 0; t < T; ++t) { row_slot_[(size_t) t] = cur_slot_; row_pos_[(size_t) t] = pos_ + t; }
    swaps_now_ = T > snap_T_ ? 0 : adapt_max_;   // a prompt window touches most experts: swapping them in would only thrash
    // (cpu mode keeps the swap allowance: the PCIe side of the split admits hot experts into VRAM)
    doorbell_ = experts_ != nullptr && !dump && experts_->doorbell_ready();
    // a prompt window: the MMQ expert path, every non-resident expert of a layer prefetched two MoE layers ahead
    // into alternating buffers (no routing-driven serving, the doorbell thread stays idle); the flags still gate
    // each layer on its prefetch
    prompt_mode_ = doorbell_ && T >= kMmqMinT && devs_[0].mmq && experts_->win_bufs() > 0;
    // the mid-size window: the misses served per layer by routing (the ring filled by the doorbell), with the CPU taking
    // its share when the split is on; else windows from kPrefetchAllMinT prefetch every non-resident expert ahead
    const bool split_ok = kMidCpu && cpu_misses_ && experts_ && experts_->cpu_ready() && devs_[0].q2k_codes != nullptr;
    // the split pays while the CPU's share is worth the lost prefetch-ahead: 430 tokens 3.35 -> 2.58 s, 1064 tokens 3.42 -> 3.76 s
    static const int kMidMaxT = std::getenv("STRATA_GLM5_MID_MAX_T") ? std::atoi(std::getenv("STRATA_GLM5_MID_MAX_T")) : 640;
    mid_mode_ = prompt_mode_ && (T < kPrefetchAllMinT || (split_ok && T <= std::min({kDbXRows, t_chunk_, kMidMaxT})));
    mid_split_ = mid_mode_ && split_ok && T <= kDbXRows;
    if (prompt_mode_ && experts_ && experts_->loans_on()) drop_loans();   // the ring is the prompt path's again
    cpu_mode_ = doorbell_ && !prompt_mode_ && T <= 8 && cpu_misses_ && experts_->cpu_ready();
    if (!prompt_mode_ && kNoDoorbell) doorbell_ = false;
    if (prompt_mode_) ring_parity_ ^= 1;
    Job j;
    j.slot = cur_slot_; j.pos0 = pos_; j.T = T; j.parity = ring_parity_; j.last = true;
    j.toks.assign(tokens, tokens + T);
    j.pf_next.assign(devs_.size(), hp_.n_layer);
    j.rbase.assign(devs_.size(), 0);
    if (prompt_mode_) assign_ring(j);
    if (doorbell_) {
        if (db_failed_.load(std::memory_order_acquire)) throw std::runtime_error(db_err_);
        ++tok_;
        db_T_ = T;
        pending_.store(0, std::memory_order_release);
        for (size_t di = 0; di < devs_.size(); ++di) j.pf_next[di] = next_moe_layer_on(devs_[di], -1);
        if (!prompt_mode_ || mid_mode_) {
            if (!db_thread_.joinable()) db_thread_ = std::thread([this]() { db_main(); });
            db_active_.store(true, std::memory_order_release);
        }
    }
    j.tok = tok_;
    for (auto& dd : devs_) dd.job = &j;
    if (doorbell_ && prompt_mode_ && !mid_mode_) for (auto& dd : devs_) prefetch_up_to(dd, j, experts_->win_bufs());   // every device fills its ring at once
    // the layer stages, device by device (the layers of a device are contiguous)
    int il0 = 0;
    for (size_t di = 0; di < devs_.size() && il0 < hp_.n_layer; ++di) {
        int il1 = il0;
        while (il1 < hp_.n_layer && dev_of_layer_[(size_t) il1] == (int) di) ++il1;
        if (il1 == il0) continue;
        stage_run(devs_[di], j, il0, il1, dump);
        il0 = il1;
    }
    for (auto& dd : devs_) dd.job = nullptr;
    const auto tw1 = std::chrono::steady_clock::now();
    if (kTrace) std::fprintf(stderr, "glm5: tok %u all layers enqueued (pending %d), waiting\n", tok_, pending_.load());
    if (doorbell_ && (!prompt_mode_ || mid_mode_)) { wait_served(hp_.n_layer); db_active_.store(false, std::memory_order_release); }
    if (kTrace) std::fprintf(stderr, "glm5: tok %u served all\n", tok_);
    if (kTrace) { for (auto& dd : devs_) { cudaSetDevice(dd.ordinal); cudaStreamSynchronize(S(dd.stream)); } std::fprintf(stderr, "glm5: tok %u streams drained\n", tok_); }
    const auto tw2 = std::chrono::steady_clock::now();
    win_stats_.enqueue_ms = std::chrono::duration<double, std::milli>(tw1 - tw0).count();
    win_stats_.wait_ms = std::chrono::duration<double, std::milli>(tw2 - tw1).count();
    win_stats_.misses = experts_ ? experts_->misses() - misses0 : 0;
    win_stats_.swaps = experts_ ? (int) (experts_->swaps() - swaps0) : 0;
    if (experts_ && std::getenv("STRATA_GLM5_DMA")) { const double c0 = experts_->copy_ms(); const uint64_t b0 = experts_->copy_bytes(); experts_->collect_timing(); win_stats_.copy_ms = experts_->copy_ms() - c0; win_stats_.copy_bytes = experts_->copy_bytes() - b0; }
    if (experts_) { win_stats_.distinct = experts_->distinct_misses() - dist0; win_stats_.thrash = experts_->thrash() - thr0; win_stats_.layers = experts_->layers_missed() - lay0; win_stats_.prefetched = experts_->prefetched() - pf0; }
    prof_report(T);
    if (!rows_mode_) pos_ += T;
}

// Layers [il0, il1) of window j on device d: the embedding when il0 == 0, else the residual streams from the previous
// device's bounce (after its handoff event); the handoff out when il1 < n_layer, else the head (h_win, the logits).
// The window's state is read through d.job; the device's token counter advances here.
void Engine::stage_run(Dev& d, Job& j, int il0, int il1, const DumpFn& dump) {
    const int E = hp_.n_embd, T = j.T;
    const int p = j.parity;
    check(cudaSetDevice(d.ordinal), "cudaSetDevice");
    d.job = &j;
    if (doorbell_) u32_inc(d.tok_dev, d.stream);
    if (doorbell_ && prompt_mode_ && !mid_mode_) prefetch_up_to(d, j, experts_->win_bufs());   // (a no-op when the previous window's stage ran ahead)
    if (il0 == 0) {
        if (m_.token_embd().type != 8) throw std::runtime_error("glm5: token_embd is not Q8_0");
        std::memcpy(d.ids_host2[p], j.toks.data(), (size_t) T * 4);
        check(cudaMemcpyAsync(d.tok_rows, d.ids_host2[p], (size_t) T * 4, cudaMemcpyHostToDevice, S(d.stream)), "tok rows h2d");
        prof_mark(d, "start");
        dequant_q8_0_rows(m_.token_embd().u8(), d.tok_rows, T, d.x, E, false, d.stream);
        if (dump) dump("inp_embd", d.x, {E}, false, d.ordinal);
        init_streams(d.x, d.R, T, E, hp_.hc, d.stream);
        if (dump) dump("hc_init", d.R, {hp_.hc, E}, false, d.ordinal);
    } else {
        Dev& pd = devs_[(size_t) dev_of_layer_[(size_t) il0 - 1]];
        check(cudaStreamWaitEvent(S(d.stream), static_cast<cudaEvent_t>(pd.ev2[p]), 0), "stream wait");
        check(cudaMemcpyAsync(d.R, d.bounce2[p], (size_t) T * hp_.hc * E * 4, cudaMemcpyHostToDevice, S(d.stream)), "R h2d");
        check(cudaEventRecord(static_cast<cudaEvent_t>(d.ev_in2[p]), S(d.stream)), "event record");
        d.ev_in_valid[p] = true;
        prof_mark(d, "handoff");
    }
    for (int il = il0; il < il1; ++il) {
        if (kTrace) std::fprintf(stderr, "glm5: tok %u layer %d enqueue (pending %d)\n", j.tok, il, pending_.load());
        if (doorbell_ && db_failed_.load(std::memory_order_acquire)) throw std::runtime_error(db_err_);   // before this thread blocks in a launch behind a parked wait
        const LayerWeights& L = m_.layer(il);
        const int dv = d.ordinal;
        DumpFn dl = dump ? [&](const std::string& n, const void* ptr, const std::vector<int64_t>& sh, bool i, int) { dump(n + "-" + std::to_string(il), ptr, sh, i, dv); } : DumpFn();
        prof_mark(d, "embed/mhc_post");
        // the layer over the window in chunks of at most t_chunk_ tokens (the scratch is chunk-sized; R spans the window)
        const int TC = std::min(T, t_chunk_), n_chunks = (T + TC - 1) / TC;
        for (int c = 0; c < n_chunks; ++c) {
        const int c0 = c * TC, Tc = std::min(TC, T - c0);
        const bool last = c + 1 == n_chunks;
        float* Rc = d.R + (size_t) c0 * hp_.hc * E;
        float* xc = d.x + (size_t) c0 * E;
        mhc_norm(d, Tc, Rc, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, L.attn_norm, xc, d.xn);
        if (dl) dl("hc_attn_pre", xc, {E}, false, dv);
        if (dl) dl("attn_norm", d.xn, {E}, false, dv);
        prof_mark(d, "mhc");
        if (kDebugSync) { debug_sync(d.stream, il, "pre-attention"); debug_nan(d.xn, Tc, E, il, "attn_norm"); }
        if (hp_.is_recr[(size_t) il]) kda_layer(d, il, Tc, d.xn, d.y, dl);
        else dsa_layer(d, il, Tc, j.pos0 + c0, d.xn, d.y, dl);
        if (kDebugSync) { debug_sync(d.stream, il, hp_.is_recr[(size_t) il] ? "kda" : "dsa"); debug_nan(d.y, Tc, E, il, hp_.is_recr[(size_t) il] ? "kda out" : "dsa out"); }
        prof_mark(d, hp_.is_recr[(size_t) il] ? "kda" : "dsa");
        mhc_post_t(d.y, Rc, d.post, d.comb, Rc, Tc, E, hp_.hc, d.stream);
        if (dl) dl("hc_attn_post", Rc, {hp_.hc, E}, false, dv);
        mhc_norm(d, Tc, Rc, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, L.ffn_norm, xc, d.xn);
        if (dl) dl("hc_ffn_pre", xc, {E}, false, dv);
        if (dl) dl("ffn_norm", d.xn, {E}, false, dv);
        prof_mark(d, "mhc");
        if (il < hp_.n_dense_lead) {
            if (doorbell_ && predicts_next(il)) predict_next(d, il, Tc, d.xn);
            dense_ffn(d, il, Tc, d.xn, d.y, dl);
        } else moe(d, il, Tc, d.xn, d.y, doorbell_, dl, last);
        if (kDebugSync) debug_sync(d.stream, il, il < hp_.n_dense_lead ? "dense ffn" : "moe");
        if (il < hp_.n_dense_lead) prof_mark(d, "dense_ffn");
        mhc_post_t(d.y, Rc, d.post, d.comb, Rc, Tc, E, hp_.hc, d.stream);
        if (dl) dl("l_out", Rc, {hp_.hc, E}, false, dv);
        }   // chunks
        if (kDebugSync) {
            // serve what this layer published (it may be the only one pending), then look at the stream
            if (doorbell_ && (!prompt_mode_ || mid_mode_) && il >= hp_.n_dense_lead) wait_served(il + 1);   // (a prompt window's ring prefetch has no doorbell service)
            debug_sync(d.stream, il, "layer");
            std::fprintf(stderr, "glm5: layer %d done (T %d)\n", il, T);
        }
    }
    if (il1 < hp_.n_layer) {
        Dev& nd = devs_[(size_t) dev_of_layer_[(size_t) il1]];
        // the bounce of this parity: after the next device read its previous contents (two windows back)
        if (nd.ev_in_valid[p]) check(cudaStreamWaitEvent(S(d.stream), static_cast<cudaEvent_t>(nd.ev_in2[p]), 0), "bounce free wait");
        check(cudaMemcpyAsync(nd.bounce2[p], d.R, (size_t) T * hp_.hc * E * 4, cudaMemcpyDeviceToHost, S(d.stream)), "R d2h");
        check(cudaEventRecord(static_cast<cudaEvent_t>(d.ev2[p]), S(d.stream)), "event record");
        return;
    }
    prof_mark(d, "embed/mhc_post");
    mean_streams(d.R, d.x, T, E, hp_.hc, d.stream);
    if (dump) dump("hc_head", d.x, {E}, false, d.ordinal);
    rms_norm(d.x, m_.output_norm().f32(), d.h_win, T, E, hp_.rms_eps, d.stream);
    if (dump) dump("result_norm", d.h_win, {E}, false, d.ordinal);
    if (T > snap_T_) gemv(d, 1, m_.output(), d.h_win + (size_t) (T - 1) * E, d.logits, E, hp_.n_vocab);   // a prompt window: the last token's logits, in row 0
    else gemv(d, T, m_.output(), d.h_win, d.logits, E, hp_.n_vocab);
    if (dump) dump("result_output", d.logits, {hp_.n_vocab}, false, d.ordinal);
    prof_mark(d, "head");
}

// ---------------------------------------------------------------- the prompt pipeline
bool Engine::pipeline_ready() const {
    return devs_.size() >= 2 && experts_ != nullptr && experts_->doorbell_ready() && devs_[0].mmq && experts_->win_bufs() > 0 && n_slots_ >= 0 &&
           !(std::getenv("STRATA_GLM5_NOPIPE") && std::atoi(std::getenv("STRATA_GLM5_NOPIPE")) != 0);
}

void Engine::stage_main(size_t di) {
    Dev& d = devs_[di];
    check(cudaSetDevice(d.ordinal), "cudaSetDevice");
    const bool last_stage = di + 1 == devs_.size();
    int il0 = 0, il1 = 0;
    for (int il = 0; il < hp_.n_layer; ++il) if (dev_of_layer_[(size_t) il] == (int) di) { if (il1 == 0 && il0 == 0 && il != 0) il0 = il; il1 = il + 1; }
    if (dev_of_layer_[0] != (int) di) { il0 = 0; while (il0 < hp_.n_layer && dev_of_layer_[(size_t) il0] != (int) di) ++il0; }
    for (;;) {
        Job* j = nullptr;
        {
            std::unique_lock<std::mutex> lk(stage_m_);
            stage_cv_.wait(lk, [&] { return stage_stop_ || !stage_q_[di].empty(); });
            if (stage_stop_ && stage_q_[di].empty()) return;
            j = stage_q_[di].front();
            stage_q_[di].pop_front();
            // the first stage starts window k only once the last stage has started window k-1 (so the last stage has enqueued its
            // read of the bounce window k reuses, and the handoff event of that parity has been waited on)
            if (di == 0) stage_cv_.wait(lk, [&] { return stage_stop_ || stage_started_last_ >= j->idx; });
            if (last_stage) { stage_started_last_ = j->idx + 1; stage_cv_.notify_all(); }
        }
        try {
            stage_run(d, *j, il0, il1, DumpFn());
            if (last_stage) { win_start_ = j->pos0; win_T_ = j->T; }   // the window the logits / h_win belong to (argmax_last, mark)
            if (last_stage && win_cb_) win_cb_(j->pos0, j->T, j->last);
        } catch (...) {
            std::lock_guard<std::mutex> lk(stage_m_);
            if (!stage_err_) stage_err_ = std::current_exception();
            stage_stop_ = true;
            stage_cv_.notify_all();
            return;
        }
        {
            std::lock_guard<std::mutex> lk(stage_m_);
            if (!last_stage) stage_q_[di + 1].push_back(j);
            else ++stage_done_;
            stage_cv_.notify_all();
        }
    }
}

void Engine::drop_loans() {
    for (auto& dd : devs_) { check(cudaSetDevice(dd.ordinal), "cudaSetDevice"); check(cudaStreamSynchronize(S(dd.stream)), "loan drop sync"); }
    experts_->reset_loans();
}

void Engine::prompt_windows(int slot, const int32_t* tokens, int n, int wsize, const std::function<void(int, int, bool)>& on_window) {
    if (slot < 0 || slot > n_slots_) throw std::runtime_error("glm5: no slot " + std::to_string(slot));
    if (n <= 0) return;
    wsize = std::max(1, std::min(wsize, t_max_));
    const int n_win = (n + wsize - 1) / wsize, wlen = (n + n_win - 1) / n_win;
    cur_slot_ = slot;
    pos_ = pos_s_[(size_t) slot];
    if (pos_ + n > max_ctx_) throw std::runtime_error("glm5: the context (" + std::to_string(max_ctx_) + ") is full");
    bool pipe = pipeline_ready() && n_win >= 2;
    for (int off = 0; off < n && pipe; off += wlen) if (std::min(wlen, n - off) < kPrefetchAllMinT) pipe = false;
    if (!pipe) {
        for (int off = 0; off < n; off += wlen) {
            const int T = std::min(wlen, n - off);
            forward_window_slot(slot, tokens + off, T);
            if (on_window) on_window(pos_s_[(size_t) slot] - T, T, off + T == n);
        }
        return;
    }
    // every window a prompt window on the ring; the stages run them as a pipeline
    if (db_failed_.load(std::memory_order_acquire)) throw std::runtime_error(db_err_);
    if (experts_ && experts_->loans_on()) drop_loans();
    rows_mode_ = false; swaps_now_ = 0; doorbell_ = true; prompt_mode_ = true; mid_mode_ = false; mid_split_ = false; cpu_mode_ = false;
    db_active_.store(false, std::memory_order_release);
    std::vector<std::unique_ptr<Job>> jobs;
    int idx = 0;
    for (int off = 0; off < n; off += wlen, ++idx) {
        auto j = std::make_unique<Job>();
        j->slot = slot; j->pos0 = pos_ + off; j->T = std::min(wlen, n - off); j->idx = idx; j->last = off + j->T == n;
        j->tok = ++tok_; ring_parity_ ^= 1; j->parity = ring_parity_;
        j->toks.assign(tokens + off, tokens + off + j->T);
        j->pf_next.resize(devs_.size());
        for (size_t di = 0; di < devs_.size(); ++di) j->pf_next[di] = next_moe_layer_on(devs_[di], -1);
        assign_ring(*j);
        if (!jobs.empty()) jobs.back()->next = j.get();
        jobs.push_back(std::move(j));
    }
    if (stage_th_.empty()) {
        stage_q_.resize(devs_.size());
        stage_stop_ = false;
        for (size_t di = 0; di < devs_.size(); ++di) stage_th_.emplace_back([this, di]() { stage_main(di); });
    }
    win_cb_ = on_window;
    {
        std::lock_guard<std::mutex> lk(stage_m_);
        pipe_on_ = true;
        stage_done_ = 0; stage_started_last_ = 0; stage_err_ = nullptr;
        for (auto& j : jobs) stage_q_[0].push_back(j.get());
    }
    stage_cv_.notify_all();
    {
        std::unique_lock<std::mutex> lk(stage_m_);
        stage_cv_.wait(lk, [&] { return stage_done_ >= (int) jobs.size() || stage_err_; });
        pipe_on_ = false;
    }
    win_cb_ = nullptr;
    for (auto& dd : devs_) dd.job = nullptr;
    if (stage_err_) { faulted_ = true; std::rethrow_exception(stage_err_); }
    const Job& lastj = *jobs.back();
    pos_ = lastj.pos0 + lastj.T;
    pos_s_[(size_t) slot] = pos_;
    win_start_ = lastj.pos0;
    win_T_ = lastj.T;
    db_T_ = lastj.T;
}

void Engine::commit(int n_keep) {
    if (n_keep < 1 || n_keep > win_T_) throw std::runtime_error("glm5: commit(" + std::to_string(n_keep) + ") outside the window of " + std::to_string(win_T_));
    if (n_keep < win_T_ && win_T_ > snap_T_) throw std::runtime_error("glm5: commit: a window of " + std::to_string(win_T_) + " tokens has no rollback snapshots");
    if (n_keep < win_T_ && !kRollbackTable) {
        const int D = hp_.d_inner(), Sd = hp_.kda_head_dim;
        for (int il = 0; il < hp_.n_layer; ++il) {
            if (!hp_.is_recr[(size_t) il]) continue;
            Dev& d = devs_[(size_t) dev_of_layer_[(size_t) il]];
            check(cudaSetDevice(d.ordinal), "cudaSetDevice");
            const size_t ssm_n = (size_t) hp_.n_head * Sd * Sd;
            check(cudaMemcpyAsync(ssm_state_[st(il)], ssm_snap_[(size_t) il] + (size_t) (n_keep - 1) * ssm_n, ssm_n * 4, cudaMemcpyDeviceToDevice, S(d.stream)), "state rollback");
            const size_t cw = (size_t) (hp_.d_conv - 1) * D, snap_stride = (size_t) win_T_ * cw;
            for (int which = 0; which < 3; ++which)
                check(cudaMemcpyAsync(conv_state_[st(il)] + which * cw, conv_snap_[(size_t) il] + which * snap_stride + (size_t) (n_keep - 1) * cw, cw * 4, cudaMemcpyDeviceToDevice, S(d.stream)), "conv rollback");
        }
    } else if (n_keep < win_T_) {
        // the KDA states back to the snapshots after token n_keep - 1: one table-driven copy launch per device (the
        // table in mapped pinned memory, two sets by commit parity: the previous commit's launch has long run)
        const int D = hp_.d_inner(), Sd = hp_.kda_head_dim;
        const size_t ssm_n = (size_t) hp_.n_head * Sd * Sd;
        const size_t cw = (size_t) (hp_.d_conv - 1) * D, snap_stride = (size_t) win_T_ * cw;
        for (auto& d : devs_) {
            d.copy_n = 0;
            if (!d.copy_tab_h) {
                void* dv = nullptr;
                d.copy_tab_h = static_cast<CopyEnt*>(alloc_mapped((uint64_t) 2 * d.copy_cap * sizeof(CopyEnt), d.ordinal, &dv));
                d.copy_tab_d = static_cast<CopyEnt*>(dv);
            }
        }
        for (int il = 0; il < hp_.n_layer; ++il) {
            if (!hp_.is_recr[(size_t) il]) continue;
            Dev& d = devs_[(size_t) dev_of_layer_[(size_t) il]];
            CopyEnt* tab = d.copy_tab_h + (size_t) d.copy_par * d.copy_cap;
            tab[d.copy_n++] = {ssm_state_[st(il)], ssm_snap_[(size_t) il] + (size_t) (n_keep - 1) * ssm_n, (int) ssm_n, 0};
            for (int which = 0; which < 3; ++which)
                tab[d.copy_n++] = {conv_state_[st(il)] + which * cw, conv_snap_[(size_t) il] + which * snap_stride + (size_t) (n_keep - 1) * cw, (int) cw, 0};
        }
        for (auto& d : devs_) {
            if (!d.copy_n) continue;
            check(cudaSetDevice(d.ordinal), "cudaSetDevice");
            copy_table(d.copy_tab_d + (size_t) d.copy_par * d.copy_cap, d.copy_n, d.stream);
            d.copy_par ^= 1;
        }
    }
    pos_ = win_start_ + n_keep;
    pos_s_[(size_t) cur_slot_] = pos_;
}

std::vector<int32_t> Engine::argmax_rows() {
    Dev& d = devs_[(size_t) dev_of_layer_[(size_t) hp_.n_layer - 1]];
    check(cudaSetDevice(d.ordinal), "cudaSetDevice");
    const int rows = win_T_ > snap_T_ ? 1 : win_T_;   // a prompt window has the last token's logits only (row 0)
    strata::glm5::argmax_rows(d.logits, rows, hp_.n_vocab, d.argmax_out, d.am_idx, d.am_val, d.stream);
    std::vector<int32_t> r((size_t) rows);
    check(cudaMemcpyAsync(r.data(), d.argmax_out, (size_t) rows * 4, cudaMemcpyDeviceToHost, S(d.stream)), "argmax d2h");
    check(cudaStreamSynchronize(S(d.stream)), "sync");
    if (rows < win_T_) r.assign((size_t) win_T_, r[0]);
    return r;
}

// ---- the MTP (NextN) layer
const float* Engine::mtp_step(const float* h_prev, int32_t tok, int pos, bool want_logits) {
    try { return mtp_step_impl(h_prev, tok, pos, want_logits); } catch (...) { faulted_ = true; throw; }
}
const float* Engine::mtp_step_solo(const float* h_prev, int32_t tok, int pos, bool want_logits) { mtp_slot_ = n_slots_; return mtp_step(h_prev, tok, pos, want_logits); }

const float* Engine::mtp_step_impl(const float* h_prev, int32_t tok, int pos, bool want_logits) {
    if (!m_.has_mtp()) throw std::runtime_error("glm5: no MTP layer attached");
    const int il = hp_.n_layer;   // the NextN layer's index
    const LayerWeights& L = m_.layer(il);
    Dev& d = devs_[(size_t) dev_of_layer_[(size_t) il]];
    check(cudaSetDevice(d.ordinal), "cudaSetDevice");
    cudaStream_t s = S(d.stream);
    const int E = hp_.n_embd;
    if (pos >= max_ctx_) throw std::runtime_error("glm5: the MTP context is full");
    // e = enorm(embed(tok)) (zero at position 0), hn = hnorm(h_prev), x = eh_proj([e | hn])
    if (tok < 0 || tok >= hp_.n_vocab) throw std::runtime_error("glm5: mtp_step: token " + std::to_string(tok) + " out of range (a garbage draft)");
    d.ids_host[0] = tok;
    check(cudaMemcpyAsync(d.tok_rows, d.ids_host, 4, cudaMemcpyHostToDevice, s), "mtp tok");
    dequant_q8_0_rows(m_.token_embd_on(d.ordinal).u8(), d.tok_rows, 1, d.mtp_e, E, pos == 0, d.stream);
    rms_norm(d.mtp_e, L.enorm.f32(), d.mtp_cat, 1, E, hp_.rms_eps, d.stream);
    rms_norm(h_prev, L.hnorm.f32(), d.mtp_cat + E, 1, E, hp_.rms_eps, d.stream);
    gemv(d, 1, L.eh_proj, d.mtp_cat, d.mtp_x, 2 * E, E);
    // the decoder layer with a plain residual
    rms_norm(d.mtp_x, L.attn_norm.f32(), d.xn, 1, E, hp_.rms_eps, d.stream);
    d.xn_q8 = nullptr;
    dsa_layer(d, il, 1, pos, d.xn, d.y, DumpFn());
    check(cudaMemcpyAsync(d.mtp_x1, d.mtp_x, (size_t) E * 4, cudaMemcpyDeviceToDevice, s), "mtp x1");
    add_rows(d.mtp_x1, d.y, E, d.stream);
    rms_norm(d.mtp_x1, L.ffn_norm.f32(), d.xn, 1, E, hp_.rms_eps, d.stream);
    d.xn_q8 = nullptr;
    const int saved_T = win_T_;
    win_T_ = 1;
    moe(d, il, 1, d.xn, d.y, false, DumpFn());   // every MTP expert is resident: no doorbell
    win_T_ = saved_T;
    add_rows(d.mtp_x1, d.y, E, d.stream);
    rms_norm(d.mtp_x1, L.shared_head_norm.f32(), d.mtp_h, 1, E, hp_.rms_eps, d.stream);
    if (want_logits) gemv(d, 1, m_.output(), d.mtp_h, d.logits, E, hp_.n_vocab);
    mtp_pos_ = pos + 1;
    return d.mtp_h;
}

const float* Engine::mtp_window(const float* h_rows, const int32_t* toks, int T, int pos0, bool want_logits) {
    try { return mtp_window_impl(h_rows, toks, T, pos0, want_logits); } catch (...) { faulted_ = true; throw; }
}
const float* Engine::mtp_window_solo(const float* h_rows, const int32_t* toks, int T, int pos0, bool want_logits) { mtp_slot_ = n_slots_; return mtp_window(h_rows, toks, T, pos0, want_logits); }

const float* Engine::mtp_window_impl(const float* h_rows, const int32_t* toks, int T, int pos0, bool want_logits) {
    if (!m_.has_mtp()) throw std::runtime_error("glm5: no MTP layer attached");
    if (kTrace) std::fprintf(stderr, "glm5: mtp window T %d pos0 %d\n", T, pos0);
    if (T < 1 || T > t_max_) throw std::runtime_error("glm5: MTP window of " + std::to_string(T) + " tokens (max " + std::to_string(t_max_) + ")");
    if (T > t_chunk_) {   // a window wider than the scratch: chunk by chunk (only the last row's h and logits are kept)
        const float* last = nullptr;
        for (int c0 = 0; c0 < T; c0 += t_chunk_) {
            const int Tc = std::min(t_chunk_, T - c0);
            last = mtp_window_impl(h_rows + (size_t) c0 * hp_.n_embd, toks + c0, Tc, pos0 + c0, want_logits && c0 + Tc == T);
        }
        return last;
    }
    const int il = hp_.n_layer;
    const LayerWeights& L = m_.layer(il);
    Dev& d = devs_[(size_t) dev_of_layer_[(size_t) il]];
    check(cudaSetDevice(d.ordinal), "cudaSetDevice");
    cudaStream_t s = S(d.stream);
    const int E = hp_.n_embd;
    if (pos0 + T > max_ctx_) throw std::runtime_error("glm5: the MTP context is full");
    std::memcpy(d.ids_host, toks, (size_t) T * 4);
    check(cudaMemcpyAsync(d.tok_rows, d.ids_host, (size_t) T * 4, cudaMemcpyHostToDevice, s), "mtp toks");
    dequant_q8_0_rows(m_.token_embd_on(d.ordinal).u8(), d.tok_rows, T, d.mtp_e, E, pos0 == 0, d.stream);
    rms_norm(d.mtp_e, L.enorm.f32(), d.mtp_h, T, E, hp_.rms_eps, d.stream);      // e (mtp_h as scratch)
    rms_norm(h_rows, L.hnorm.f32(), d.mtp_x1, T, E, hp_.rms_eps, d.stream);      // hn (mtp_x1 as scratch)
    concat_rows(d.mtp_h, d.mtp_x1, d.mtp_cat, T, E, d.stream);
    if (kDebugSync) { debug_sync(d.stream, il, "mtp embed/norms"); debug_nan(h_rows, T, E, il, "mtp h_rows in"); debug_nan(d.mtp_e, T, E, il, "mtp embed"); debug_nan(d.mtp_cat, T, 2 * E, il, "mtp cat"); }
    gemv(d, T, L.eh_proj, d.mtp_cat, d.mtp_x, 2 * E, E);
    if (kDebugSync) { debug_sync(d.stream, il, "mtp eh_proj"); debug_nan(d.mtp_x, T, E, il, "mtp eh_proj out"); }
    rms_norm(d.mtp_x, L.attn_norm.f32(), d.xn, T, E, hp_.rms_eps, d.stream);
    d.xn_q8 = nullptr;
    dsa_layer(d, il, T, pos0, d.xn, d.y, DumpFn());
    if (kDebugSync) { debug_sync(d.stream, il, "mtp dsa"); debug_nan(d.y, T, E, il, "mtp dsa out"); }
    check(cudaMemcpyAsync(d.mtp_x1, d.mtp_x, (size_t) T * E * 4, cudaMemcpyDeviceToDevice, s), "mtp x1");
    add_rows(d.mtp_x1, d.y, T * E, d.stream);
    rms_norm(d.mtp_x1, L.ffn_norm.f32(), d.xn, T, E, hp_.rms_eps, d.stream);
    d.xn_q8 = nullptr;
    if (kDebugSync) { debug_sync(d.stream, il, "mtp ffn_norm"); debug_nan(d.xn, T, E, il, "mtp moe in"); }
    const int saved_T = win_T_;
    win_T_ = T;
    moe(d, il, T, d.xn, d.y, false, DumpFn());
    win_T_ = saved_T;
    add_rows(d.mtp_x1, d.y, T * E, d.stream);
    rms_norm(d.mtp_x1, L.shared_head_norm.f32(), d.mtp_h, T, E, hp_.rms_eps, d.stream);
    const float* last = d.mtp_h + (size_t) (T - 1) * E;
    if (kDebugSync) { debug_sync(d.stream, il, "mtp window moe"); debug_nan(d.mtp_h, T, E, il, "mtp h rows"); }
    if (want_logits) gemv(d, 1, m_.output(), last, d.logits, E, hp_.n_vocab);
    mtp_pos_ = pos0 + T;
    if (kTrace) std::fprintf(stderr, "glm5: mtp window enqueued\n");
    return last;
}

int32_t Engine::mtp_argmax() {
    Dev& d = devs_[(size_t) dev_of_layer_[(size_t) hp_.n_layer]];
    check(cudaSetDevice(d.ordinal), "cudaSetDevice");
    strata::glm5::argmax_rows(d.logits, 1, hp_.n_vocab, d.argmax_out, d.am_idx, d.am_val, d.stream);
    int32_t r = 0;
    check(cudaMemcpyAsync(&r, d.argmax_out, 4, cudaMemcpyDeviceToHost, S(d.stream)), "argmax d2h");
    check(cudaStreamSynchronize(S(d.stream)), "sync");
    return r;
}

}  // namespace strata::glm5
