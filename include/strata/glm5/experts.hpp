// include/strata/glm5/experts.hpp - where GLM-5-Next's routed experts live (docs/GLM5_NEXT.md, P4).
//
// 12,096 experts of 8-14 MB do not fit two 48 GB cards beside the dense weights, so each device holds the most-used
// experts of ITS layers (a ranking from a routing profile, or every layer's lowest ids when there is none) and the rest
// stay in host memory - pinned when the complement is small enough, else read from the memory-mapped GGUF - and are
// copied into a per-device staging slot when routed to.  One token routes 8 x 42 = 336 experts; every miss is one
// host-to-device copy of its blob [gate | up | down].
#pragma once

#include "strata/glm5/glm5.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include "strata/kernels/cpu/native_expert.hpp"
#include <thread>
#include <vector>

namespace strata::glm5 {

struct ExpertStoreConfig {
    std::vector<int> devices;                 ///< the ordinals used, in order
    std::vector<uint64_t> budget_bytes;       ///< per entry of `devices`: VRAM for resident experts
    std::vector<int> layer_device;            ///< per layer: index into `devices`
    std::string profile;                      ///< routing counts "layer expert count" per line (optional)
    int t_max = 8;                            ///< the window: staging slots for t_max x k distinct misses
    std::vector<int> force_resident_layers;   ///< layers whose experts are all admitted first (the MTP layer)
    uint64_t evict_pool_bytes = 8ull << 30;   ///< per device: pinned host pool holding the blobs of evicted experts
    uint32_t admit_window = 32;               ///< a missed expert is swapped in only when it was missed within this many
                                              ///< windows before (0: on the first miss); one-off experts stay in staging
    uint64_t pin_limit_bytes = 64ull << 30;   ///< pin the host complement (one allocation per layer) when it is at most
                                              ///< this large; 0 = never (misses are then staged from the mapped file)
    bool verbose = true;
};

struct LayerExperts {
    int device = -1;                          ///< ordinal
    uint64_t blob = 0;                        ///< gate + up + down bytes of one expert
    uint64_t stride = 0;                      ///< the slot / ring pitch: blob rounded up to the types' block sizes
    uint64_t gate_bytes = 0, up_bytes = 0;
    uint32_t gu_type = 0, d_type = 0;
    int n_in = 0, n_ff = 0;
    uint8_t* slots = nullptr;                 ///< device: n_slots x blob
    int32_t* slot_of_dev = nullptr;           ///< device copy of slot_of (n_expert ints)
    int n_slots = 0;
    std::vector<int32_t> slot_of;             ///< per expert: slot or -1
    std::vector<int32_t> expert_of;           ///< per slot: expert
    std::vector<const uint8_t*> host_gate, host_up, host_down;   ///< per expert: pinned complement or the mmap
    std::vector<uint32_t> slot_last_used;     ///< per slot: the window that last routed to its expert
    std::vector<uint8_t> host_pinned;         ///< per expert: host_gate points at a contiguous pinned blob (DMA-able as one)
    std::vector<uint8_t> evicted_once;        ///< per expert: was resident and got swapped out at some point (a re-miss = thrash)
    std::vector<uint32_t> last_miss_tok;      ///< per expert: 1 + the window of its last miss (0: never missed)
    std::vector<int32_t> loan_of;             ///< per expert: its loan slot on the device's loan pool, or -1 (the decode path's cache)
    std::vector<char> loan_live;              ///< per expert: the loan is in the device's slot table (a hit for both sides); set and cleared with the table
    int fd = -1;                              ///< the shard holding this layer's experts (pread for cold blobs)
    uint64_t off_gate = 0, off_up = 0, off_down = 0;   ///< file offsets of expert 0's rows
};

class ExpertStore {
public:
    ExpertStore(const Model& m, const ExpertStoreConfig& cfg);
    ~ExpertStore();
    ExpertStore(const ExpertStore&) = delete;
    ExpertStore& operator=(const ExpertStore&) = delete;

    const LayerExperts& layer(int il) const { return layers_[(size_t) il]; }
    /// The doorbell path: the host has read this layer's n routed ids (T tokens x k).  Each distinct missed expert, in
    /// order of first occurrence (= the device's miss index m), is copied on this device's copy stream straight into a
    /// resident slot it takes over - the slot used longest ago, never one routed in this window (swap-in on miss, the
    /// adaptive tier) - or, when no such slot exists or swaps are off, into staging slot m.  `miss_slot[m]` records the
    /// destination (< 0 = staging), `miss_out[m]` the evicted expert (or -1); `flag` becomes `tok` once the copies
    /// have landed.  Returns the number of misses.
    int serve(int il, const int32_t* ids, int n, uint32_t tok, uint32_t* h_flag, uint32_t* flag_dev, int* miss_slot, int* miss_out, int max_swaps);
    /// The lookahead: `ids` (n of them) are the experts the previous layer predicts layer `il` will route to in window
    /// `tok`.  Each one that is not resident and has a pinned host copy is swapped in now (victims: slots unused for
    /// two windows), the table updates go to pf_e/pf_slot/pf_out (count in *pf_n) and `flag` becomes `tok` once the
    /// copies have landed; the device applies the updates at the start of layer il.  Returns the swaps.
    int prefetch(int il, const int32_t* ids, int n, uint32_t tok, uint32_t* h_flag, uint32_t* flag_dev, int* pf_n, int* pf_e, int* pf_slot, int* pf_out,
                 int max_swaps);
    uint64_t prefetched() const { return prefetched_; }
    /// the lookahead prefetch into staging (no residency change): up to `cap` distinct predicted non-resident experts of
    /// layer il (ids [T][P], the ranks' order across the tokens) are DMA'd one way into the top half of the staging slots;
    /// serve_cpu finds them there (a staging-to-staging copy instead of a transfer).  Returns how many were started.
    int prefetch_stage(int il, const int32_t* ids, int T, int P, uint32_t tok, int cap);
    uint64_t pf_hits() const { return pf_hits_; }
    /// copy-stream time (event-measured) and bytes of the H2D/D2H copies serve()/prefetch() issued; sampled per window
    double copy_ms() const { return copy_ms_; }
    uint64_t copy_bytes() const { return copy_bytes_; }
    void copy_timing(bool on) { copy_timing_ = on; }
    void collect_timing();
    /// Make `stream` wait for the copies serve() enqueued for layer il's device (the no-doorbell path).
    void join(int il, void* stream);
    uint64_t swaps() const { return swaps_; }
    uint64_t distinct_misses() const { return distinct_misses_; }
    uint64_t thrash() const { return thrash_; }   ///< distinct misses of experts that had been resident before
    uint64_t layers_missed() const { return layers_missed_; }
    const uint8_t* stage(int device_index) const { return stage_[(size_t) device_index]; }
    /// The prompt windows' prefetch ring: win_bufs() compact buffers per device, each holding every non-resident expert of
    /// one layer at rank * blob (rank = the expert's index among the layer's non-resident ones, ascending id); 0 when the
    /// store was built for small windows.
    int win_bufs() const { return win_bufs_.empty() ? 0 : (int) win_bufs_[0].size(); }
    uint8_t* win_buf(int device_index, int k) const { return win_bufs_[(size_t) device_index][(size_t) k]; }
    int nonresident(int il) const { return (int) layers_[(size_t) il].slot_of.size() - layers_[(size_t) il].n_slots; }
    /// A mid-size prompt window (routing known, prefetch-all too costly): the routed non-resident experts of layer il
    /// DMA'd to dst + rank * stride (rank as in prefetch_all / k_moe_group), after `wait_event`, then the flag.
    int serve_ring(int il, const int32_t* ids, int n, uint32_t tok, uint32_t* h_flag, uint32_t* flag_dev, uint8_t* dst, uint64_t stride, void* wait_event);
    /// A prompt window: every non-resident expert of layer il DMA'd to dst + e * stride (its whole blob), on the layer's
    /// copy stream after `wait_event` (the buffer's last reader), then the layer's flag written.  Returns the count.
    int prefetch_all(int il, uint32_t tok, uint32_t* h_flag, uint32_t* flag_dev, uint8_t* dst, uint64_t stride, void* wait_event);
    /// The decode path's misses computed on the CPU (ggml-cpu's own dot products on the pinned blobs) instead of
    /// DMA'd: x_rows = the T tokens' MoE inputs (host, [T][n_embd]); the results of the missed pairs land in the
    /// device's pair-row buffer y_dev (rows in k_moe_group's order: hits' groups first, then the misses' in ascending
    /// expert id, token order inside a group) and the layer's flag is written after them.  Returns the miss count.
    int serve_cpu(int il, const int32_t* ids, int n, int T, const float* x_rows, uint32_t tok, uint32_t* h_flag, uint32_t* flag_dev, uint32_t* h_flag2,
                  uint32_t* flag2_dev, float* y_dev, int n_embd, int* miss_cpu, int* miss_slot, int* miss_out, int max_swaps);   ///< flag: the DMA'd misses landed; flag2: the CPU's rows landed   ///< miss_cpu[m] (mapped host ints, m = the miss's rank in ascending id): 1 = computed here, 0 = DMA'd to staging slot m
    /// A mid-size prompt window (9 <= T < 256, the MMQ path with the ring): the missed experts split between PCIe (into the
    /// ring at rank * stride, as serve_ring) and the CPU (their pair rows computed from the pinned blobs into y_dev at the
    /// MMQ layout's rows, at most kMidCpuRows of them); skip_h[rank] = 1 marks the CPU-served experts for the device's
    /// products.  flag: the ring copies landed; flag2: the CPU's rows landed.  Returns the distinct misses.
    int serve_mid(int il, const int32_t* ids, int n, int T, const float* x_rows, uint32_t tok, uint32_t* h_flag, uint32_t* flag_dev, uint32_t* h_flag2,
                  uint32_t* flag2_dev, float* y_dev, int n_embd, int32_t* skip_h, uint8_t* ring_dst, uint64_t stride, void* wait_event);
    static constexpr int kMidCpuRows = 1024;  ///< the CPU's pair rows per layer in a mid-size window (the pinned result block, by layer parity)
    /// Every layer's experts are pinned and ggml-cpu covers their formats: serve_cpu is available.
    bool cpu_ready() const { return cpu_ready_; }
    void init_cpu(const Model& m);
    /// flag_dev = the layer's device view of its mapped flag word (the engine's); null = no device-side flag (the no-doorbell path)   ///< after load(): the ggml-cpu geometry, the scratch, the pinned output rows
    void set_clamp(const std::vector<float>& per_layer) { clamp_limit_ = per_layer; }
    uint64_t stage_bytes(int device_index) const { return stage_slot_bytes_[(size_t) device_index]; }
    int device_index_of_layer(int il) const;
    bool doorbell_ready() const { return doorbell_ready_; }
    /// Routing counts since the start, as "layer expert count" lines (a profile for the next start).
    bool save_profile(const std::string& path, std::string& err) const;
    uint64_t resident_bytes(int device_index) const { return resident_bytes_[(size_t) device_index]; }
    int resident_experts() const { return n_resident_; }
    int total_experts() const { return n_total_; }
    uint64_t file_reads() const { return file_reads_; }      ///< evicted experts read back from the GGUF (no pinned copy)
    uint64_t file_bytes() const { return file_bytes_; }
    double file_ms() const { return file_ms_; }                 ///< wall time of those reads (the serving thread waits)
    double cpu_ms() const { return cpu_ms_; }                   ///< wall time of serve_cpu (the doorbell thread computing)
    uint64_t cpu_pairs() const { return cpu_pairs_; }
    const double* cpu_phase_ms() const { return cpu_t_; }
    uint64_t cpu_calls() const { return cpu_calls_; }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t host_pinned_bytes() const { return pinned_bytes_ + pool_bytes_; }
    /// The loan pool (decode windows): the prompt windows' prefetch ring lent as an LRU cache of missed experts.  A miss the
    /// CPU computes (or PCIe staged) is also copied into a loan slot; from the next window on the device reads it there
    /// (the slot table entry kLoanBit | j, applied by apply_loans before the layer's grouping).  Reset before a prompt
    /// window (the ring is the prompt path's).  STRATA_GLM5_LOAN=0 disables, STRATA_GLM5_LOAN_PER_WIN caps the loans per call.
    bool loans_on() const { return loan_on_; }
    const uint8_t* const* loan_ptrs(int device_index) const { return loan_ptr_d_.empty() ? nullptr : loan_ptr_d_[(size_t) device_index]; }
    void apply_loans(int il, void* stream);        ///< the pending slot table updates of layer il (on the layer's device, before its grouping)
    void reset_loans();                            ///< every loan dropped (the host syncs the devices first)
    uint64_t loan_hits() const { return loan_hits_; }
    uint64_t loans_made() const { return loans_made_; }

private:
    void plan(const Model& m, const ExpertStoreConfig& cfg);
    void read_blob(const LayerExperts& E, int e, uint8_t* dst) const;   ///< pread gate|up|down of expert e into dst
    std::vector<int> fds_;
    void load(const Model& m, const ExpertStoreConfig& cfg);
    std::vector<LayerExperts> layers_;
    std::vector<int> devices_;
    std::vector<uint8_t*> stage_;             ///< per device index: k_max x max blob
    std::vector<std::vector<uint8_t*>> win_bufs_;   ///< per device index: the prompt windows' prefetch ring
    std::vector<void*> copy_streams_;         ///< per device index: the DMA stream of the doorbell path
    std::vector<uint32_t*> tok_pinned_;       ///< per device index: the source of the flag write
    std::vector<void*> copy_events_;
    // the eviction pool: evicted experts' blobs are copied device-to-host into pinned slots (a ring; the oldest entry
    // is dropped when the ring wraps and that expert goes back to the file)
    struct Pool_ {
        std::vector<uint8_t*> chunks;
        uint64_t slot_bytes = 0;
        int n_slots = 0, next = 0;
        std::vector<std::pair<int, int>> holder;   ///< per pool slot: (layer, expert) or (-1, -1)
        uint8_t* at(int s) const { const uint64_t per = chunk_slots; return chunks[(size_t) (s / per)] + (uint64_t) (s % per) * slot_bytes; }
        uint64_t chunk_slots = 0;
    };
    std::vector<Pool_> evict_;                 ///< per device index
    uint64_t pool_bytes_ = 0;
    std::vector<uint8_t*> stage_host_;        ///< per device index: pinned k_max x max blob, the DMA source of a miss
    // a small pool that memcpy's missed blobs from the mapped file into the pinned staging (a pageable
    // hipMemcpyAsync from the mapping took 1-2 ms per blob; memcpy on several threads plus one DMA takes ~0.3 ms)
    struct Pool {
        std::vector<std::thread> threads;
        std::mutex m, run_m;
        std::condition_variable cv, done;
        std::vector<std::function<void()>> jobs;
        size_t next = 0, finished = 0, total = 0;
        bool stop = false;
        void start(int n);
        void run(std::vector<std::function<void()>>&& js);
        ~Pool();
    } pool_;
    std::vector<uint64_t> stage_slot_bytes_;
    std::vector<uint64_t> resident_bytes_;
    std::vector<std::vector<uint32_t>> counts_;   ///< per layer, per expert
    std::vector<void*> device_allocs_;
    std::vector<void*> pinned_;              ///< per layer (or null): that layer's non-resident experts, page-locked
    uint64_t pinned_bytes_ = 0;
    std::vector<bool> layer_pinned_;
    int k_max_ = 0, n_resident_ = 0, n_total_ = 0;
    uint64_t hits_ = 0, misses_ = 0, swaps_ = 0, distinct_misses_ = 0, thrash_ = 0, layers_missed_ = 0, prefetched_ = 0;
    uint64_t pf_hits_ = 0;
    std::vector<std::vector<std::pair<int, int>>> pf_stage_;   ///< per layer: (expert, staging slot) prefetched for the current token
    uint64_t file_reads_ = 0, file_bytes_ = 0;
    uint32_t admit_window_ = 32;
    bool cpu_ready_ = false;
    bool wide_ = false;
    std::vector<std::vector<int32_t>> nonres_rank_;   ///< per layer, per expert: its rank among the non-resident ones (-1: resident)
    double cpu_ms_ = 0;
    uint64_t cpu_pairs_ = 0;
    double cpu_t_[6] = {0, 0, 0, 0, 0, 0};   ///< split, quant x, phase A, quant h, phase B, enqueue
    uint64_t cpu_calls_ = 0;
    int cpu_threads_ = 8;
    // the loan pool, per device: slot j = loan_ptr_h_[di][j] (loan_stride_ bytes), its holder (layer, expert) and last window
    bool loan_on_ = false;
    uint64_t loan_stride_ = 0;
    int loan_per_win_ = 8;
    bool loan_h2d_ = false;
    std::vector<std::vector<uint8_t*>> loan_ptr_h_;
    std::vector<const uint8_t**> loan_ptr_d_;
    std::vector<std::vector<std::pair<int, int>>> loan_holder_;   ///< per device, per slot: (layer, expert) or (-1, -1)
    std::vector<std::vector<uint32_t>> loan_last_used_;
    std::vector<std::vector<int>> loan_free_;
    std::vector<int> loan_clock_;                                  ///< per device: the LRU scan position
    // per layer: the pending table updates (e, j) in two pinned buffers (the device copies one while the host fills the other)
    std::vector<int32_t*> loan_apply_h_[2];
    std::vector<int32_t*> loan_apply_dev_[2];                       ///< the device views of the two mapped buffers
    std::vector<int> loan_apply_n_;
    std::vector<int> loan_apply_side_;
    std::vector<void*> loan_streams_;                              ///< per device: the loan copies' own stream (never ahead of a layer's flags)
    std::vector<std::vector<char>> loan_busy_;                     ///< per device, per slot: a copy in flight (not a victim until applied)
    struct LoanBatch { void* ev; size_t di; std::vector<std::pair<int, int>> adds; };   ///< the additions of one serve call and the event after their copies
    std::vector<std::deque<LoanBatch>> loan_batches_;              ///< per layer: in copy order
    std::vector<std::vector<void*>> loan_ev_pool_;                 ///< per device (events are device-bound)
    std::mutex loan_m_;
    uint64_t loan_hits_ = 0, loans_made_ = 0;
    static constexpr int kLoanApplyMax = 256;
    void init_loans(int n_layers);
    int loan_alloc(size_t di, int il, int e, uint32_t tok);        ///< a free or LRU loan slot (not one used in window tok), or -1
    void loan_push(int il, int e, int j);                           ///< queue the table update (under loan_m_)
    void* loan_event(size_t di);                                   ///< a pooled event of that device
    void loan_event_done(size_t di, void* ev);
    FILE* route_log_ = nullptr;   ///< STRATA_GLM5_ROUTE_LOG=<path>: the decode routing per window (T tok layer ids...) and the resident sets (S layer ids...)
    double cpu_pair_ms_ = 0.9;    ///< the CPU's cost of one missed pair (ms of all threads together); tuned by measurement
    double dma_ms_ = 0.65;         ///< one expert blob over PCIe
    double gpu_before_dma_ms_ = 0.4, gpu_before_cpu_ms_ = 1.0;   ///< the device's own work before each wait (the overlap budgets)
    bool cpu_pair_fixed_ = false;
    std::vector<float> clamp_limit_;   ///< per layer: the swiglu clamp (0 = none)
    std::vector<strata::kernels::cpu::NativeFmt> fmt_;   ///< per layer: ggml-cpu geometry of its expert blobs
    std::vector<std::vector<uint8_t>> act_;              ///< per token: the quantized MoE input (vec_dot_type)
    std::vector<std::vector<float>> ff_;                 ///< per pair: swiglu output (n_ff)
    std::vector<std::vector<uint8_t>> hq_;               ///< per pair: quantized swiglu output
    float* y_pin_ = nullptr;                             ///< pinned [n_layers][64][n_embd]: the pairs' outputs, DMA source
    float* y_pin_dev_ = nullptr;                         ///< its device pointer (mapped): the rows are read by a kernel, not the copy engine
    int y_pin_rows_ = 0;
    std::vector<uint8_t> seen_;   ///< scratch of serve(): distinct misses
    struct Miss { int e; std::vector<int> toks; int row0; int ff0; };   ///< a CPU-served expert: its tokens (token order), its first pair row in the device buffer, its first row in ff_/hq_/y_pin
    /// the CPU's products for `misses` of layer il (x_rows: the window's MoE inputs, T of them), their rows copied into
    /// y_dev and flag2 written behind them; returns the pair count
    int cpu_run(int il, const std::vector<Miss>& misses, int T, const float* x_rows, float* y_dev, int n_embd, uint32_t tok, uint32_t* h_flag2,
                uint32_t* flag2_dev, void* cs, std::chrono::steady_clock::time_point& tl);
    int y_rows_ = 64;             ///< the pinned result block's rows per layer (kMidCpuRows when the mid-size split is on)
    double file_ms_ = 0;
    int32_t pick_victim(LayerExperts& E, uint32_t tok, uint32_t protect);
    void dma_miss(LayerExperts& E, int il, size_t di, size_t m, int e, uint32_t tok, int* miss_slot, int* miss_out, int max_swaps, int& swapped,
                  std::vector<std::function<void()>>& jobs, std::vector<int>& staged, void* cs);
    void finish_staged(LayerExperts& E, size_t di, int* miss_slot, std::vector<std::function<void()>>& jobs, std::vector<int>& staged, void* cs);
    void evict_to_pool(LayerExperts& E, int il, size_t di, int32_t victim, void* cs, int keep_e);
    std::vector<void*> adapt_events_;
    std::vector<bool> adapt_pending_;
    bool copy_timing_ = false;
    double copy_ms_ = 0;
    uint64_t copy_bytes_ = 0;
    std::vector<std::pair<void*, void*>> timing_events_;   ///< (start, end) pairs in flight
    std::vector<uint64_t> timing_bytes_;
    bool doorbell_ready_ = false;
};

}  // namespace strata::glm5
