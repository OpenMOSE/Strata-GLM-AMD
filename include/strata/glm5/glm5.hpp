// include/strata/glm5/glm5.hpp - GLM-5-Next (GLM-5.3-Flash) in Strata: the model description read from the GGUF, the
// dense weights on the devices, and a decode engine that runs a VERIFY WINDOW of T tokens (docs/GLM5_NEXT.md, P1-P6).
//
// Layers are split across devices in contiguous ranges (`layer_device`): each device holds its layers' dense weights,
// recurrent states and attention caches, and its share of the routed experts (ExpertStore).  A window of T tokens
// (T <= window_max_tokens()) reads every dense weight once and each routed expert once per distinct (layer, expert);
// the recurrent layers take the tokens in order and keep a state snapshot per token, so `commit(n)` rolls back to the
// state after window token n-1 when the speculative tail is rejected.  The MTP (NextN) layer, when attached, runs on
// the head device as `mtp_step`.  FP32 activations; the aim is to reproduce llama.cpp token for token first.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <memory>
#include <atomic>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace strata::glm5 {

class ExpertStore;

struct HParams {
    int n_layer = 0, n_embd = 0, n_ff = 0, n_head = 0, n_vocab = 0;
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_expert_shared = 0, n_dense_lead = 0;
    float rms_eps = 1e-5f, norm_eps = 1e-6f;
    float expert_scale = 1.f;
    bool expert_norm = false;
    int gating = 2;                    ///< 2 = sigmoid (llama_expert_gating_func_type)
    std::vector<float> clamp_exp, clamp_shexp;
    int d_conv = 4, kda_head_dim = 128;
    float kda_lower = -INFINITY;
    int q_lora = 0, kv_lora = 0, head_k_mla = 0, head_v_mla = 0, n_rot = 0;
    int idx_heads = 0, idx_dim = 0, idx_topk = 0, kpool = 4;
    bool kpool_tail = false;
    int hc = 4, hc_iters = 20;
    float hc_eps = 1e-6f;
    std::vector<bool> is_recr;         ///< per layer: KDA (true) or DSA (false)
    int d_inner() const { return n_head * kda_head_dim; }
    int hc_mix_dim() const { return (2 + hc) * hc; }
};

/// What the GGUF says about the architecture; throws on an unsupported file.
HParams read_hparams(const strata::GgufModel& m);

/// One tensor on a device, as the GGUF stores it (raw blocks / BF16 bits / F32).
struct DevTensor {
    void* data = nullptr;
    uint32_t type = 0;                 ///< ggml type id
    int64_t ne0 = 0, ne1 = 0, ne2 = 0;
    uint64_t bytes = 0;
    int device = 0;
    bool present() const { return data != nullptr; }
    const float* f32() const { return static_cast<const float*>(data); }
    const uint16_t* bf16() const { return static_cast<const uint16_t*>(data); }
    const uint8_t* u8() const { return static_cast<const uint8_t*>(data); }
};

/// One expert projection's host bytes: `base + e * stride` is expert e's rows, `type` their format.
struct ExpertTensor {
    const uint8_t* base = nullptr;     ///< into the memory-mapped shard
    uint64_t stride = 0;               ///< bytes per expert
    uint32_t type = 0;
    int64_t n_in = 0, n_out = 0;
    std::string path;                  ///< the shard file, and the tensor's byte offset in it (for pread)
    uint64_t file_off = 0;
};

struct LayerWeights {
    int device = 0;
    DevTensor attn_norm, ffn_norm;
    DevTensor hc_attn_fn, hc_attn_scale, hc_attn_base, hc_ffn_fn, hc_ffn_scale, hc_ffn_base;
    // KDA
    DevTensor wq, wk, wv, wo, conv_q, conv_k, conv_v, f_a, f_b, g_a, g_b, beta, ssm_a, dt_b, ssm_norm;
    // DSA
    DevTensor q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b;
    DevTensor k_b_q8, v_b_q8;          ///< Q8_0 copies of k_b / v_b (the prompt windows' per-head MMQ); absent when the source type is not BF16 / Q8_0
    DevTensor idx_k, idx_k_norm, idx_k_norm_b, idx_proj, idx_q_b, idx_gate, idx_ape;
    // FFN
    DevTensor ffn_gate, ffn_up, ffn_down;                     // the dense layers
    DevTensor gate_inp, exp_probs_b, sh_gate, sh_up, sh_down; // the MoE layers
    ExpertTensor exp_gate, exp_up, exp_down;
    // the MTP (NextN) layer only
    DevTensor eh_proj, enorm, hnorm, shared_head_norm;
};

class Model {
public:
    /// Opens every shard of the model `any_shard` belongs to and uploads each layer's dense weights to
    /// `layer_device[il]` (the token embedding to the first layer's device, the head to the last layer's).
    Model(const std::string& any_shard, const std::vector<int>& layer_device, bool verbose = true);
    /// One device for every layer.
    Model(const std::string& any_shard, int device, bool verbose = true);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    /// The NextN file: its `blk.<n_layer>` becomes layer n_layer on the head device (with a copy of the token
    /// embedding there when the first layer lives elsewhere).
    void attach_mtp(const std::string& path, bool verbose = true);
    bool has_mtp() const { return has_mtp_; }
    int n_layer_all() const { return (int) layers_.size(); }

    const HParams& hp() const { return hp_; }
    const LayerWeights& layer(int il) const { return layers_[(size_t) il]; }
    const DevTensor& token_embd() const { return token_embd_; }
    const DevTensor& token_embd_on(int ordinal) const { return token_embd_.device == ordinal || !token_embd_mtp_.present() ? token_embd_ : token_embd_mtp_; }
    const DevTensor& output() const { return output_; }
    const DevTensor& output_norm() const { return output_norm_; }
    const strata::GgufModel& gguf() const { return *gguf_; }
    uint64_t device_bytes(int ordinal) const;
    uint64_t device_bytes() const { return bytes_; }
    const std::vector<int>& layer_device() const { return layer_device_; }
    /// The distinct ordinals, in the order the layers meet them.
    const std::vector<int>& devices() const { return devices_; }

private:
    void init(const std::string& any_shard, bool verbose);
    DevTensor upload(const strata::GgufModel& g, const std::string& name, int device, bool required = true);
    ExpertTensor expert(const strata::GgufModel& g, const std::string& name);
    std::unique_ptr<strata::GgufModel> gguf_, mtp_gguf_;
    HParams hp_;
    std::vector<LayerWeights> layers_;
    DevTensor token_embd_, token_embd_mtp_, output_, output_norm_;
    std::vector<void*> allocations_;
    std::vector<int> alloc_device_;
    std::vector<uint64_t> bytes_per_device_;
    uint64_t bytes_ = 0;
    std::vector<int> layer_device_, devices_;
    bool has_mtp_ = false;
    DevTensor q8_copy(const DevTensor& t, int dev);
};

/// A named intermediate for comparison with llama.cpp's dump: `shape` in numpy order (token axis absent), f32 device
/// data or int32 when `is_int`.  The data lives on the device `device`.
using DumpFn = std::function<void(const std::string& name, const void* dev, const std::vector<int64_t>& shape,
                                  bool is_int, int device)>;

class Engine {
public:
    /// `max_ctx` bounds the attention caches (cells); `t_max` the window (<= window_max_tokens()).
    /// n_slots batch slots (each a conversation's state: KDA states, DSA caches, position) beside the solo slot (index n_slots).
    /// windowed attention (on the MLA latent cache): env STRATA_GLM5_LOD=1, _LOD_PAGE, _LOD_TOP, _LOD_TOP_DECODE, _LOD_LOCAL, _LOD_SINK
    struct LodCfg { bool on = false; int ps = 64, top_prefill = 32, top_decode = 32, local = 512, sink_pages = 1; bool tok_route = true; int qgroups = 8; bool idx_route = false; bool pool_sel = false; int top_pools = 1024, top_pools_decode = 512; bool no_sum = false; };
    Engine(const Model& m, int max_ctx, int t_max = 8, int n_slots = 0);
    int n_slots() const { return n_slots_; }
    /// The prompt's windows (equal splits of at most `wsize` tokens) through the layer stages of the devices as a pipeline:
    /// window k+1 runs on the first device while window k runs on the second.  `on_window(pos0, T, last)` is called
    /// after each window from the last device's stage thread (h_win() holds that window's rows in stream order, so an
    /// MTP catch-up enqueued from it is ordered).  Blocks until every window is done; afterwards the engine's position,
    /// logits() and h_win() are the last window's.  Falls back to serial windows when the pipeline cannot run.
    void prompt_windows(int slot, const int32_t* tokens, int n, int wsize, const std::function<void(int, int, bool)>& on_window);
    bool pipeline_ready() const;
    void drop_loans();   ///< before a prompt window: the decode loans in the prompt ring are dropped (the devices synced first)
    int solo_slot() const { return n_slots_; }
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /// T tokens at positions pos()..pos()+T-1 through the trunk.  Afterwards logits() holds [T][n_vocab] and h_win()
    /// the post-norm hidden state [T][n_embd], both on the head device; pos() has advanced by T.  `dump` is honoured
    /// for T == 1 only (the oracle comparison).
    void forward_window(const int32_t* tokens, int T, const DumpFn& dump = nullptr);   ///< on the solo slot
    void forward_window_slot(int slot, const int32_t* tokens, int T);                   ///< a prompt / decode window of one slot
    /// A batch decode window: row t is token tokens[t] of slot slots[t]; a slot's rows are consecutive (its token, then
    /// its drafts) at consecutive positions.  logits() / argmax_rows() hold the rows; commit_rows() then keeps the first
    /// n_keep[g] rows of each slot run g (1..len) and advances the slots' positions.
    void forward_rows(const int32_t* tokens, const int* slots, int S);
    void commit_rows(const int* n_keep, int n_grp);
    int n_groups() const { return n_grp_; }
    /// The MTP on a batch slot's own draft caches (the solo path's mtp_step / mtp_window use the solo slot's).
    const float* mtp_step_slot(int slot, const float* h_prev, int32_t tok, int pos, bool want_logits) { mtp_slot_ = slot; return mtp_step(h_prev, tok, pos, want_logits); }
    const float* mtp_window_slot(int slot, const float* h_rows, const int32_t* toks, int T, int pos0, bool want_logits) { mtp_slot_ = slot; return mtp_window(h_rows, toks, T, pos0, want_logits); }
    const float* mtp_step_solo(const float* h_prev, int32_t tok, int pos, bool want_logits);      ///< the solo slot's draft caches
    const float* mtp_window_solo(const float* h_rows, const int32_t* toks, int T, int pos0, bool want_logits);
    /// slot dst takes slot src's state (its tokens so far, position, mark): the next turn of a conversation held elsewhere.
    void copy_slot(int src, int dst);
    /// One token (forward_window with T = 1).
    const float* forward(int32_t token, const DumpFn& dump = nullptr) { forward_window(&token, 1, dump); return logits(); }
    /// Keep the first n_keep tokens of the last window (1..T): the recurrent states roll back to the state after window
    /// token n_keep-1 and pos() to window start + n_keep.  The attention caches need no rollback (positional).
    void commit(int n_keep);
    /// argmax of each of the last window's T rows of logits (synchronises).
    std::vector<int32_t> argmax_rows();
    int32_t argmax_last() { return argmax_rows().back(); }
    const float* logits() const;
    const float* h_win() const;
    int window_T() const { return win_T_; }
    int pos() const { return pos_; }                                   ///< the last window's slot
    int pos(int slot) const { return pos_s_[(size_t) slot]; }
    int t_max() const { return t_max_; }
    /// Host-side timing of the last window: enqueue (the layer loop), the wait for the devices' publishes, and the
    /// misses the store served (the doorbell path).
    struct WinStats { double enqueue_ms = 0, wait_ms = 0; uint64_t misses = 0, distinct = 0, thrash = 0, layers = 0, prefetched = 0, copy_bytes = 0; int swaps = 0; double copy_ms = 0; };
    WinStats last_window() const { return win_stats_; }
    void reset(int slot = -1);   ///< -1: the solo slot
    /// The conversation mark: the recurrent states after the window token h_row of the last window, pos(), and that
    /// token's post-norm hidden state (for the MTP's re-step).  One mark; restore_mark() rolls back to it (the
    /// attention caches are positional and need no rollback).  A multi-turn chat whose new prompt extends the
    /// previous one continues from the mark instead of starting over.
    void mark(int h_row);
    bool has_mark(int slot = -1) const { return mark_pos_s_[(size_t) (slot < 0 ? n_slots_ : slot)] > 0; }
    int mark_pos(int slot = -1) const { return mark_pos_s_[(size_t) (slot < 0 ? n_slots_ : slot)]; }
    bool restore_mark(int slot = -1);
    const float* h_mark(int slot = -1) const { return h_mark_s_[(size_t) (slot < 0 ? n_slots_ : slot)]; }
    /// Routed experts from a store (resident slots + staged misses) instead of per-token copies from the mapped file.
    void set_experts(ExpertStore* store);
    /// The adaptive tier: swaps per layer per window (0 = static residency).
    void set_adapt(int max_per_layer) { adapt_max_ = max_per_layer; }
    /// The lookahead prefetch: predicted experts per token (0 = off; the engine was built for up to 16).
    void set_lookahead(int n_pred) { lookahead_ = n_pred; }
    void* head_stream() const;
    int head_device() const;

    // ---- the MTP (NextN) layer, on the head device
    bool has_mtp() const { return m_.has_mtp(); }
    /// One MTP step at position `pos`: h_prev (device, n_embd floats: the trunk's post-norm hidden state at `pos`, or
    /// the previous MTP step's h_next), `tok` (the token at pos+1).  Writes the MTP's caches at `pos`.  Returns h_next
    /// (device, valid until the next step); with want_logits, mtp_argmax() gives the draft.
    const float* mtp_step(const float* h_prev, int32_t tok, int pos, bool want_logits);
    /// T MTP steps at once (the catch-up over a window): h_rows = the trunk's post-norm hidden states at positions
    /// pos0..pos0+T-1 (device, [T][n_embd]), toks[i] = the token at pos0+i+1.  Writes the MTP's caches at those
    /// positions; returns h_next of the last one, with want_logits its logits for mtp_argmax().
    const float* mtp_window(const float* h_rows, const int32_t* toks, int T, int pos0, bool want_logits);
    int32_t mtp_argmax();

    /// Everything one device needs: its stream and the activation scratch.  Public for the tests.
    /// one window in flight: the layer code reads the window's state through the device's current job
    struct Job {
        int slot = 0, pos0 = 0, T = 0, idx = 0;
        uint32_t tok = 0;
        int parity = 0;
        bool last = false;
        std::vector<int32_t> toks;
        std::vector<int> pf_next;    ///< per device: the next MoE layer to prefetch into its ring
        std::vector<int> rbase;      ///< per device: the ring position of its first MoE layer (the ring continues across windows)
        Job* next = nullptr;         ///< the window after this one (the ring prefetch continues into it)
    };
    struct Dev {
        const Job* job = nullptr;         ///< the window this device is working on
        float* bounce2[2] = {nullptr, nullptr};   ///< the residual streams' host bounce per window parity
        int32_t* ids_host2[2] = {nullptr, nullptr};
        void* ev2[2] = {nullptr, nullptr};        ///< the handoff event per window parity
        void* ev_in2[2] = {nullptr, nullptr};     ///< recorded after the bounce of a parity was read in (its next writer waits on it)
        bool ev_in_valid[2] = {false, false};
        std::unique_ptr<std::mutex> pf_mutex;     ///< the ring prefetch state of this device
        int ordinal = 0;
        void* stream = nullptr;
        void* event = nullptr;
        float *R = nullptr, *x = nullptr, *xn = nullptr, *y = nullptr, *mixes = nullptr, *part = nullptr, *rs = nullptr,
              *pre = nullptr, *post = nullptr, *comb = nullptr;
        float *big_a = nullptr, *big_b = nullptr, *big_c = nullptr, *big_d = nullptr;   // [T][big]
        float *small_a = nullptr, *small_b = nullptr;                                     // [T][q_lora]
        float *ffn_g = nullptr, *ffn_u = nullptr, *ffn_h = nullptr, *ffn_y = nullptr, *ffn_acc = nullptr, *ffn_d = nullptr;
        float *logits = nullptr, *probs = nullptr, *wts = nullptr, *h_win = nullptr;
        int32_t* ids = nullptr;
        uint8_t *q8_x = nullptr, *q8_g = nullptr, *q8_h = nullptr, *q8_big = nullptr;
        float *q_abs = nullptr, *iq = nullptr, *ik = nullptr, *ig = nullptr, *iw = nullptr, *score = nullptr,
              *attn_scores = nullptr, *ctx = nullptr;
        int32_t* sel = nullptr;
        int* sel_params = nullptr;        ///< [T][8]: {n_pool, n_top, n_kv, n_tail, n_sel}
        uint8_t* expert_stage = nullptr;  ///< the no-store fallback's staging
        int32_t* neg_table = nullptr;     ///< n_expert x -1 (the no-store fallback's slot table)
        const void **grp_w = nullptr, **grp_wu = nullptr, **grp_wd = nullptr;
        int *grp_col0 = nullptr, *grp_ncols = nullptr, *grp_orow = nullptr, *pair_tok = nullptr, *cnt = nullptr;
        int *grp_e = nullptr, *grp_miss = nullptr;
        int *miss_slot_h = nullptr, *miss_out_h = nullptr, *miss_slot_d = nullptr, *miss_out_d = nullptr;   ///< n_layer x T*k, mapped
        // the lookahead: predicted ids of the next layer (published), and the prefetch placement records
        float *pred_logits = nullptr, *pred_wts = nullptr, *pred_probs = nullptr;
        int32_t* pred_ids = nullptr;
        int32_t *db_pred_h = nullptr, *db_pred_d = nullptr;           ///< n_layer x T*P
        uint32_t *db_pred_pub_h = nullptr, *db_pred_pub_d = nullptr;  ///< n_layer
        uint32_t *pf_flag_h = nullptr, *pf_flag_d = nullptr;
        int *pf_n_h = nullptr, *pf_n_d = nullptr;
        int *pf_e_h = nullptr, *pf_e_d = nullptr, *pf_slot_h = nullptr, *pf_slot_d = nullptr, *pf_out_h = nullptr, *pf_out_d = nullptr;
        float* pair_w = nullptr;
        int32_t* argmax_out = nullptr;
        void* gemm = nullptr;             ///< strata::prefill::Gemm (hipBLASLt / cuBLAS) for wide windows
        uint16_t* x16 = nullptr;          ///< FP16 / BF16 copy of the activations for the GEMM ([T][big])
        // the MMQ expert path of prompt windows (llama.cpp's int8 tensor-core GEMM on the quantized experts)
        void* mmq = nullptr;              ///< strata::prefill::mmq::Context
        void* mmq_xq = nullptr;           ///< q8_1 (MMQ layout) of the pairs' activations [max_pairs][E]
        void* mmq_xd = nullptr;           ///< q8_1 (MMQ layout) of a wide window's activations for the dense weights [T + tile][big]
        int32_t* dense_bounds = nullptr;  ///< {0, T}: the one "expert" of a dense product
        int32_t* dense_bounds_h = nullptr; ///< its pinned source
        void* head_xq = nullptr;          ///< q8_1 (MMQ layout) of a wide window's per-head activations [T*H + tile][max(HK, KV)]
        int32_t* head_ids = nullptr;      ///< [T*H] head-major row -> token-major row
        int32_t* head_bounds = nullptr;   ///< [H + 1]
        int head_T = 0;
        // the dense (masked) DSA attention of prompt windows, in batches of kDsaBatch tokens
        uint8_t* kvcT_q8 = nullptr;       ///< [KV][n_kv_pad] Q8_0 (its transpose)
        uint8_t* lod_kv_q8 = nullptr;     ///< LoD: the prompt window's compact operand [n_pad][KV] Q8_0
        float* lod_bias = nullptr;        ///< LoD: its per-column logit bias
        float* lod_score = nullptr;       ///< LoD: the window's shared page scores
        float* lod_tok = nullptr;         ///< LoD token routing: per-token scores [max_ctx + 512]
        float* lod_qg = nullptr;          ///< LoD token routing: the head-group means of the queries [T][G][KV]
        float* lod_pscore = nullptr;      ///< LoD indexer routing: per-row page scores [8][max_ctx/ps + 512]
        float* lod_iqp = nullptr;         ///< LoD indexer routing: the indexer queries zero-padded to 256 [T*IH][256]
        int* sel_params2 = nullptr;       ///< LoD indexer routing: the pool counts per row [T][8]
        int* sel_params2_host = nullptr;
        float* lod_part = nullptr;        ///< LoD: the fused decode read's split partials [8 rows * H * nsplit][2 + KV]
        float* attn_S = nullptr;          ///< [kDsaBatch * H][n_kv_pad] scores -> probabilities
        uint32_t* attn_mask = nullptr;    ///< [kDsaBatch][n_kv_pad / 32] selection bits
        void* attn_xq = nullptr;          ///< q8_1 (MMQ) of the batch's q_abs rows [kDsaBatch * H][KV]
        void* attn_pq = nullptr;          ///< q8_1 (MMQ) of the probabilities [kDsaBatch * H][n_kv_pad]
        int32_t* attn_bounds = nullptr;   ///< {0, rows}
        int32_t* attn_bounds_h = nullptr;
        int attn_rows = 0;
        int dense_T = 0;                  ///< the T the device copy holds
        void* mmq_xq_h = nullptr;         ///< ... of the swiglu outputs [max_pairs][FF]
        float* gu_out = nullptr;          ///< [max_pairs][2 FF]: the gate | up product
        int8_t* q2k_codes = nullptr;      ///< the Q2_K WMMA path: int8 activations of the pairs [pairs_alloc][E], then of the swiglu rows [..][FF]
        float* q2k_scales = nullptr;      ///< [pairs_alloc][E/256]
        int32_t* q2k_sums = nullptr;      ///< [pairs_alloc][E/16]
        int32_t* q2k_packed = nullptr;    ///< the decode WMMA path: the pair sums packed per super-block [64][E/256][8]
        int8_t* dec_codes_h = nullptr;    ///< the decode path's swiglu rows: int8 [64][FF], scales [64][FF/256], sums [64][FF/16], packed [64][FF/256][8]
        float* dec_scales_h = nullptr;
        int32_t* dec_sums_h = nullptr;
        int32_t* dec_packed_h = nullptr;
        // the prompt windows' expert buffers alternate per MoE layer (the store's staging and its twin): buf_ev[k] is
        // recorded after the last reader of buffer k, the next prefetch into it waits for that event
        std::vector<void*> layer_ev;      ///< per layer x 2 (the window's parity): the event after its MoE's last read of its ring buffer
        std::vector<char> layer_ev_valid;
        std::vector<int> ring_user;       ///< per ring buffer: the layer that used it last (-1: none); the next user waits on that layer's event
        std::vector<uint32_t> ring_user_tok;   ///< ... and the window (tok) it used it in: this window's record or the previous window's
        int pf_next = 0;                  ///< the next MoE layer of this device to prefetch
        int32_t* iota = nullptr;          ///< 0..max_pairs-1 (MMQ's identity row table)
        int* bounds = nullptr;            ///< [n_expert + 1]: pair ranges per expert (ascending-order grouping)
        int* tok_pairs = nullptr;         ///< [max_pairs]: pair (token, k) -> its row in the pair list
        int* bounds_res = nullptr;        ///< [n_expert + 1]: the resident experts' pair ranges in slot order
        int* bounds_miss = nullptr;       ///< [n_expert + 1]: the non-resident experts' pair ranges by rank
        int* cnt_host = nullptr;          ///< pinned: the widest group's row count (the MMQ launch size)
        float* db_x_h = nullptr; float* db_x_d = nullptr;   ///< mapped [n_layers][8][n_embd]: a decode window's MoE inputs for the CPU service
        int* miss_cpu_h = nullptr; int* miss_cpu_d = nullptr; ///< mapped [n_layers][64]: per miss, 1 = computed on the CPU
        uint32_t* db_flag2_h = nullptr; uint32_t* db_flag2_d = nullptr;   ///< mapped per layer: the CPU's rows landed (waited before the combine)
        int* am_idx = nullptr;
        float* am_val = nullptr;
        int32_t* tok_rows = nullptr;      ///< [T] the window's token ids (embedding rows)
        int *row_slot_d = nullptr, *row_pos_d = nullptr;   ///< a batch window's rows: slot and position (device)
        int *row_slot_h = nullptr, *row_pos_h = nullptr;   ///< their pinned sources
        int *row_k_d = nullptr, *grp_slot_d = nullptr, *grp_start_d = nullptr, *grp_len_d = nullptr;   ///< the slot runs (device)
        int *row_k_h = nullptr, *grp_slot_h = nullptr, *grp_start_h = nullptr, *grp_len_h = nullptr;
        // pinned host
        int32_t* ids_host = nullptr;
        int* sel_params_host = nullptr;
        float* bounce = nullptr;
        // the MoE doorbell: host-mapped pinned records (one per layer) and their device views
        int32_t* db_ids_h = nullptr;      ///< n_layer x T x k
        uint32_t* db_pub_h = nullptr;
        uint32_t* db_flag_h = nullptr;
        int32_t* db_ids_d = nullptr;
        uint32_t* db_pub_d = nullptr;
        uint32_t* db_flag_d = nullptr;
        uint32_t* tok_dev = nullptr;
        // MTP
        float *mtp_h = nullptr, *mtp_e = nullptr, *mtp_cat = nullptr, *mtp_x = nullptr, *mtp_x1 = nullptr;
    };

private:
    void mhc(Dev& d, int T, const float* R, const DevTensor& fn, const DevTensor& scale, const DevTensor& base, float* x);
    void gemv(Dev& d, int T, const DevTensor& w, const float* x, float* y, int n_in, int n_out);
    void kda_layer(Dev& d, int il, int T, const float* xn, float* out, const DumpFn& dump);
    void dsa_layer(Dev& d, int il, int T, int pos0, const float* xn, float* out, const DumpFn& dump);
public:
    bool lod_on() const { return lod_.on; }
    const LodCfg& lod_cfg() const { return lod_; }
private:
    void dense_ffn(Dev& d, int il, int T, const float* xn, float* out, const DumpFn& dump);
    void moe(Dev& d, int il, int T, const float* xn, float* out, bool doorbell, const DumpFn& dump, bool last_chunk = true);
    void* alloc(uint64_t bytes, int ordinal);
    void* alloc_host(uint64_t bytes);
    void* alloc_mapped(uint64_t bytes, int ordinal, void** dev_view);
    bool serve_step();          ///< the doorbell thread: serve the next pending layer when published (true = progress)
    void db_main();
    void wait_served(int upto);  ///< the engine thread: wait until the layers < upto are served (the doorbell thread's failure rethrown)
    void prefetch_from(int il);
    void predict_next(Dev& d, int il, int T, const float* xn);
    uint32_t tok_ = 0;
    std::atomic<int> pending_{0};
    bool doorbell_ = false;
    // the doorbell thread: it alone serves the published routings, so the engine thread may block in a launch (a
    // stream whose queue filled behind a parked moe_wait) without deadlocking the device that waits for the host
    std::thread db_thread_;
    std::atomic<bool> db_stop_{false}, db_active_{false}, db_failed_{false};
    std::string db_err_;
    int db_T_ = 0;
    bool faulted_ = false;
    WinStats win_stats_;
    int adapt_max_ = 64;
    int swaps_now_ = 64;   ///< this window's swap allowance (0 in a prompt window: no admission while reading a prompt)
    int snap_T_ = 1;       ///< KDA snapshots exist for windows up to this many tokens (commit / mark rollback)
    bool prompt_mode_ = false;   ///< this window takes the MMQ expert path with prefetch-all (no routing-driven serving)
    bool cpu_mode_ = false;      ///< this window's misses are computed on the CPU (decode windows, when the store can)
    bool mid_mode_ = false;      ///< a prompt window too small for prefetch-all: the routed non-resident experts DMA'd into the ring by the doorbell thread
    bool cpu_misses_ = true;     ///< the setting (STRATA_GLM5_CPU=0 turns it off)
    void prefetch_up_to(Dev& d, Job& j, int seq_limit);   ///< this device's MoE layers with sequence < seq_limit into the ring (continuing into j.next)
    void stage_run(Dev& d, Job& j, int il0, int il1, const DumpFn& dump);   ///< layers [il0, il1) of one window on one device (embed / handoff / head as applicable)
    void stage_main(size_t di);
    int jslot(const Dev& d) const { return d.job ? d.job->slot : cur_slot_; }
    uint32_t jtok(const Dev& d) const { return d.job ? d.job->tok : tok_; }
    int jparity(const Dev& d) const { return d.job ? d.job->parity : ring_parity_; }
    int jT(const Dev& d) const { return d.job ? d.job->T : win_T_; }
    size_t stj(const Dev& d, int il) const { return st(il >= hp_.n_layer ? mtp_slot_ : jslot(d), il); }
    // the pipeline
    bool pipe_on_ = false;
    std::vector<std::thread> stage_th_;
    std::vector<std::deque<Job*>> stage_q_;
    std::mutex stage_m_;
    std::condition_variable stage_cv_;
    bool stage_stop_ = false;
    int stage_done_ = 0, stage_started_last_ = 0;   ///< jobs completed; jobs started by the last stage
    std::exception_ptr stage_err_;
    std::function<void(int, int, bool)> win_cb_;
    int next_moe_layer_on(Dev& d, int after) const;
    std::vector<int> moe_seq_;   ///< per layer: its index among its device's MoE layers (-1: dense / MTP)
    std::vector<int> n_dev_moe_, ring_next_;   ///< per device: its MoE layer count; the ring position of the next window's first MoE layer
    int ring_buf(const Dev& d, const Job& j, int il) const;   ///< the ring buffer of (window, MoE layer) on device d
    void assign_ring(Job& j);
    // STRATA_GLM5_PROF=1: GPU time per phase of a window (events on each device's stream)
    struct ProfMark { int dev; const char* what; void* ev; };
    std::vector<ProfMark> prof_;
    double host_prof_[5] = {0, 0, 0, 0, 0};
    void prof_mark(Dev& d, const char* what);
    bool mmq_fits(int type, int rows);
    std::map<std::pair<int, int>, bool> mmq_fits_;
    void moe_mmq(Dev& d, int il, int T, const float* xn, float* out, bool doorbell, const int32_t* slot_of, const uint8_t* slots, uint64_t blob,
                 const uint8_t* stage, uint64_t stage_bytes, bool all_resident, const DumpFn& dump, bool last_chunk);
    void prof_report(int T);
    int lookahead_ = 12;
    bool predicts_next(int il) const;   ///< layer il publishes a prediction for il + 1
    int pending_pf_ = 0;
    uint32_t adapt_tok_ = 0;
    void forward_window_impl(const int32_t* tokens, int T, const DumpFn& dump);
    const float* mtp_step_impl(const float* h_prev, int32_t tok, int pos, bool want_logits);
    const float* mtp_window_impl(const float* h_rows, const int32_t* toks, int T, int pos0, bool want_logits);

    const Model& m_;
    const HParams& hp_;
    ExpertStore* experts_ = nullptr;
    std::vector<Dev> devs_;
    std::vector<int> dev_of_layer_;
    int max_ctx_ = 0, t_max_ = 8;
    int t_chunk_ = 8;   ///< the compute chunk of a prompt window (<= window_max_tokens()); a wider window runs each layer chunk by chunk
    int pos_ = 0, win_start_ = 0, win_T_ = 0;
    int n_slots_ = 0, cur_slot_ = 0, NL_ = 0;
    std::vector<int> pos_s_, mark_pos_s_;     ///< per slot (the solo slot last)
    std::vector<float*> h_mark_s_;
    int mtp_slot_ = 0;   ///< the slot whose MTP caches the next MTP call uses (the solo path sets the solo slot)
    size_t st(int il) const { return (size_t) (il >= hp_.n_layer ? mtp_slot_ : cur_slot_) * NL_ + (size_t) il; }   ///< the current slot's state index (the MTP layer: mtp_slot_)
    size_t st(int slot, int il) const { return (size_t) slot * NL_ + (size_t) il; }
    std::vector<int> row_slot_, row_pos_;     ///< the window's rows: slot and position (single-slot windows: one slot, consecutive)
    bool rows_mode_ = false;
    int n_grp_ = 0;
    std::vector<int> grp_slot_, grp_start_, grp_len_, row_k_;   ///< the window's slot runs (rows mode)
    int ring_parity_ = 0;   ///< flips per prompt window: a layer's event is recorded into layer_ev[il*2 + parity], so a wait on the previous window's record never meets this window's
    void* ring_wait_event(Dev& d, int k, const Job& j) const;
    // per layer: device tables of the slots' state bases (index = slot), for the batch windows' per-row kernels
    std::vector<float**> conv_tab_, ssm_tab_, pooled_tab_;
    std::vector<uint8_t**> kvc_tab_, ik_tab_, ig_tab_;
    int mtp_pos_ = 0;
    uint64_t expert_stage_bytes_ = 0;
    int max_pairs_ = 0;
    int pairs_alloc_ = 0;   ///< max_pairs_ + an MMQ tile: its unchecked tiles read whole rows past the last expert's range
    // per-layer state, on the layer's device
    // per (slot, layer) state, on the layer's device: index st(slot, il); the snapshots are per layer (solo verify windows)
    std::vector<float*> conv_state_, ssm_state_, conv_snap_, ssm_snap_;   // KDA layers
    std::vector<float*> conv_mark_, ssm_mark_;   // the conversation mark's copy of the KDA states
    std::vector<uint8_t*> kvc_cache_, ik_cache_, ig_cache_;                // DSA layers (incl. the MTP layer), Q8_0 rows (pitch n/32*34 bytes)
    static int q8_pitch(int n) { return n / 32 * 34; }
    std::vector<float*> pooled_;
    std::vector<uint8_t*> pmean_;
    std::vector<uint8_t*> pooled_q8_;                                     ///< LoD indexer routing: the pooled indexer keys as Q8_0 rows padded to 256
    std::vector<uint8_t**> pooled_q8_tab_;                                         ///< LoD: page means of the latent cache, Q8_0 rows [max_ctx/ps + 512]
    std::vector<uint8_t**> pmean_tab_;
    /// windowed attention: env STRATA_GLM5_LOD=1, _LOD_PAGE, _LOD_TOP, _LOD_TOP_DECODE, _LOD_LOCAL, _LOD_SINK
    LodCfg lod_;
    std::vector<LodCfg> lod_layer_;   ///< per-layer route overrides (STRATA_GLM5_LOD_ROUTE_LAYERS); the global cfg otherwise
    const LodCfg& lod_of(int il) const { return (size_t) il < lod_layer_.size() ? lod_layer_[(size_t) il] : lod_; }
    int n_sel_max_ = 0;
    void dsa_lod(Dev& d, int il, int T, int pos0, float scale);
    int lod_cand(int n_kv) const { return n_kv > lod_.local ? (n_kv - lod_.local) / lod_.ps : 0; }   ///< the pages before the local field
    int lod_n_sel(int n_kv, int n_top) const;
    std::vector<void*> allocations_;
    std::vector<int> alloc_device_;
    std::vector<void*> host_allocations_;
};

}  // namespace strata::glm5
