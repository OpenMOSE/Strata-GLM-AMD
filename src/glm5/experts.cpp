// src/glm5/experts.cpp - see include/strata/glm5/experts.hpp.
#include "strata/glm5/experts.hpp"
#include "strata/glm5/kernels.hpp"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <map>

#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <mutex>
#include <exception>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace strata::glm5 {
namespace {
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
}  // namespace

ExpertStore::ExpertStore(const Model& m, const ExpertStoreConfig& cfg) {
    if (const char* rl = std::getenv("STRATA_GLM5_ROUTE_LOG")) route_log_ = std::fopen(rl, "w");
    admit_window_ = cfg.admit_window;
    const HParams& hp = m.hp();
    if (cfg.budget_bytes.size() != cfg.devices.size()) throw std::runtime_error("ExpertStore: one budget per device");
    devices_ = cfg.devices;
    k_max_ = hp.n_expert_used * 8;   // the staging: distinct misses of a verify window (<= 8 tokens); wider windows take the prefetch ring
    wide_ = cfg.t_max > 8;
    seen_.assign((size_t) hp.n_expert, 0);
    const int NL = m.n_layer_all();
    if ((int) cfg.layer_device.size() < NL) throw std::runtime_error("ExpertStore: layer_device must name every layer (incl. the MTP layer)");
    layers_.resize((size_t) NL);
    pf_stage_.resize((size_t) NL);
    counts_.assign((size_t) NL, std::vector<uint32_t>((size_t) hp.n_expert, 0));
    for (int il = hp.n_dense_lead; il < NL; ++il) {
        const LayerWeights& L = m.layer(il);
        LayerExperts& E = layers_[(size_t) il];
        E.device = devices_[(size_t) cfg.layer_device[(size_t) il]];
        E.gate_bytes = L.exp_gate.stride;
        E.up_bytes = L.exp_up.stride;
        E.blob = L.exp_gate.stride + L.exp_up.stride + L.exp_down.stride;
        // the slot pitch: MMQ addresses experts in whole blocks of the matrix's type, so the pitch is a multiple of the gate's
        // and the down's block sizes (a layer mixing Q4_K and Q5_K, as the MTP does, needs the padding; same-type layers are exact)
        {
            const uint64_t bg = ggml_type_size((ggml_type) L.exp_gate.type), bd = ggml_type_size((ggml_type) L.exp_down.type);
            const uint64_t l = std::lcm(bg, bd);
            E.stride = (E.blob + l - 1) / l * l;
        }
        E.gu_type = L.exp_gate.type;
        E.d_type = L.exp_down.type;
        E.n_in = (int) L.exp_gate.n_in;
        E.n_ff = (int) L.exp_gate.n_out;
        E.slot_of.assign((size_t) hp.n_expert, -1);
        E.host_gate.resize((size_t) hp.n_expert);
        E.host_up.resize((size_t) hp.n_expert);
        E.host_down.resize((size_t) hp.n_expert);
        E.host_pinned.assign((size_t) hp.n_expert, 0);
        E.evicted_once.assign((size_t) hp.n_expert, 0);
        E.last_miss_tok.assign((size_t) hp.n_expert, 0);
        for (int e = 0; e < hp.n_expert; ++e) {
            E.host_gate[(size_t) e] = L.exp_gate.base + (uint64_t) e * L.exp_gate.stride;
            E.host_up[(size_t) e] = L.exp_up.base + (uint64_t) e * L.exp_up.stride;
            E.host_down[(size_t) e] = L.exp_down.base + (uint64_t) e * L.exp_down.stride;
        }
        if (L.exp_gate.path != L.exp_up.path || L.exp_gate.path != L.exp_down.path)
            throw std::runtime_error("ExpertStore: a layer's experts split across shards is not handled yet");
        static std::map<std::string, int> open_fds;
        auto it = open_fds.find(L.exp_gate.path);
        if (it == open_fds.end()) {
            const int fd = ::open(L.exp_gate.path.c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("cannot open " + L.exp_gate.path);
            it = open_fds.emplace(L.exp_gate.path, fd).first;
        }
        E.fd = it->second;
        E.off_gate = L.exp_gate.file_off;
        E.off_up = L.exp_up.file_off;
        E.off_down = L.exp_down.file_off;
        n_total_ += hp.n_expert;
    }
    plan(m, cfg);
    load(m, cfg);
    init_loans(m.hp().n_layer);
}

ExpertStore::~ExpertStore() {
    if (std::uncaught_exceptions() > 0) return;   // a fault in flight: do not touch the devices
    for (void* p : device_allocs_) cudaFree(p);
    for (uint8_t* p : stage_host_) if (p) cudaFreeHost(p);
    for (void* p : pinned_) if (p) cudaFreeHost(p);
    for (auto& P : evict_) for (uint8_t* c : P.chunks) cudaFreeHost(c);
    if (y_pin_) cudaFreeHost(y_pin_);
    if (copy_desc_) cudaFreeHost(copy_desc_);
}

// pread is used instead of memcpy from the mapping: a cold mapping faults page by page (observed ~50 MB/s on six
// threads after the resident uploads had flushed the page cache), a 2.7 MB pread reads ahead at the SSD's rate.
void ExpertStore::read_blob(const LayerExperts& E, int e, uint8_t* dst) const {
    auto rd = [&](uint64_t off, uint8_t* to, uint64_t n) {
        while (n) {
            const ssize_t got = ::pread(E.fd, to, n, (off_t) off);
            if (got <= 0) throw std::runtime_error("pread of an expert failed");
            to += got; off += (uint64_t) got; n -= (uint64_t) got;
        }
    };
    rd(E.off_gate + (uint64_t) e * E.gate_bytes, dst, E.gate_bytes);
    rd(E.off_up + (uint64_t) e * E.up_bytes, dst + E.gate_bytes, E.up_bytes);
    const uint64_t down_bytes = E.blob - E.gate_bytes - E.up_bytes;
    rd(E.off_down + (uint64_t) e * down_bytes, dst + E.gate_bytes + E.up_bytes, down_bytes);
}

void ExpertStore::Pool::start(int n) {
    for (int t = 0; t < n; ++t)
        threads.emplace_back([this]() {
            for (;;) {
                std::function<void()> job;
                {
                    std::unique_lock<std::mutex> lk(m);
                    cv.wait(lk, [&] { return stop || next < jobs.size(); });
                    if (stop) return;
                    job = jobs[next++];
                }
                job();
                std::lock_guard<std::mutex> lk(m);
                if (++finished == total) done.notify_all();
            }
        });
}
void ExpertStore::Pool::run(std::vector<std::function<void()>>&& js) {
    if (js.empty()) return;
    std::lock_guard<std::mutex> outer(run_m);   // one batch at a time (the stage threads of two devices may both miss pins)
    std::unique_lock<std::mutex> lk(m);
    jobs = std::move(js);
    next = finished = 0;
    total = jobs.size();
    cv.notify_all();
    done.wait(lk, [&] { return finished == total; });
    jobs.clear();
}
ExpertStore::Pool::~Pool() {
    { std::lock_guard<std::mutex> lk(m); stop = true; }
    cv.notify_all();
    for (auto& t : threads) t.join();
}

// Which experts are resident: walk the ranking (profile counts, else by id round-robin over the layers) and admit a
// pair while its device's budget holds its blob.
void ExpertStore::plan(const Model& m, const ExpertStoreConfig& cfg) {
    const HParams& hp = m.hp();
    std::vector<std::tuple<uint32_t, int, int>> ranked;   // (count, layer, expert), sorted descending
    bool have_profile = false;
    if (!cfg.profile.empty()) {
        std::ifstream f(cfg.profile);
        if (!f && cfg.verbose) std::fprintf(stderr, "glm5 experts: no expert profile at %s yet (the lowest ids are resident until one is saved)\n", cfg.profile.c_str());
        std::string line;
        while (f && std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            int il, e;
            unsigned c;
            if (std::sscanf(line.c_str(), "%d %d %u", &il, &e, &c) != 3) continue;
            if (il >= hp.n_dense_lead && il < hp.n_layer && e >= 0 && e < hp.n_expert) { ranked.emplace_back(c, il, e); counts_[(size_t) il][(size_t) e] = c; }   // the trunk's layers only; the counts carry on from the profile (a save accumulates sessions)
        }
        have_profile = !ranked.empty();
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
    }
    if (!have_profile) {
        for (int e = 0; e < hp.n_expert; ++e)
            for (int il = hp.n_dense_lead; il < hp.n_layer; ++il) ranked.emplace_back(0u, il, e);
    }
    std::vector<uint64_t> left = cfg.budget_bytes;
    resident_bytes_.assign(devices_.size(), 0);
    const int NL = (int) layers_.size();
    std::vector<int> n_slots((size_t) NL, 0);
    std::vector<std::vector<int>> admitted((size_t) NL);
    for (int il : cfg.force_resident_layers) {
        LayerExperts& E = layers_[(size_t) il];
        const size_t di = (size_t) cfg.layer_device[(size_t) il];
        for (int e = 0; e < hp.n_expert; ++e) {
            if (left[di] < E.blob) throw std::runtime_error("ExpertStore: the budget of device " + std::to_string(devices_[di]) + " cannot hold layer " + std::to_string(il) + "'s experts");
            left[di] -= E.blob;
            resident_bytes_[di] += E.blob;
            E.slot_of[(size_t) e] = n_slots[(size_t) il]++;
            admitted[(size_t) il].push_back(e);
            ++n_resident_;
        }
    }
    for (const auto& [c, il, e] : ranked) {
        LayerExperts& E = layers_[(size_t) il];
        if (E.slot_of[(size_t) e] != -1) continue;
        const size_t di = (size_t) cfg.layer_device[(size_t) il];
        if (left[di] < E.blob) continue;
        left[di] -= E.blob;
        resident_bytes_[di] += E.blob;
        E.slot_of[(size_t) e] = n_slots[(size_t) il]++;
        admitted[(size_t) il].push_back(e);
        ++n_resident_;
    }
    for (int il = hp.n_dense_lead; il < NL; ++il) {
        LayerExperts& E = layers_[(size_t) il];
        E.n_slots = n_slots[(size_t) il];
        E.expert_of = admitted[(size_t) il];
        E.slot_last_used.assign((size_t) E.n_slots, 0u);
    }
    if (route_log_) {
        for (int il = hp.n_dense_lead; il < NL; ++il) {
            std::fprintf(route_log_, "S %d %d", il, layers_[(size_t) il].n_slots);
            for (int e : layers_[(size_t) il].expert_of) std::fprintf(route_log_, " %d", e);
            std::fprintf(route_log_, "\n");
        }
        std::fflush(route_log_);
    }
    if (cfg.verbose) {
        std::fprintf(stderr, "glm5 experts: %d of %d resident (%s)", n_resident_, n_total_, have_profile ? "profile ranking" : "no profile: lowest ids first");
        for (size_t d = 0; d < devices_.size(); ++d) std::fprintf(stderr, "; device %d %.1f GiB", devices_[d], resident_bytes_[d] / 1073741824.0);
        std::fprintf(stderr, "\n");
    }
}

void ExpertStore::load(const Model& m, const ExpertStoreConfig& cfg) {
    const HParams& hp = m.hp();
    const auto t0 = std::chrono::steady_clock::now();
    // the complement (every non-resident expert) goes into page-locked host memory, one allocation per layer, so a
    // miss is one DMA.  One 28 GiB hipHostMalloc did not return in 10 minutes on ROCm 7; per-layer pieces of up to
    // 4 GiB take ~0.7 s per GiB.
    const int NL = (int) layers_.size();
    uint64_t complement = 0, widest = 0;
    for (int il = hp.n_dense_lead; il < NL; ++il) {
        const LayerExperts& E = layers_[(size_t) il];
        complement += (uint64_t) (hp.n_expert - E.n_slots) * E.blob;
        widest = std::max(widest, E.blob);
    }
    const bool pin = complement > 0 && complement <= cfg.pin_limit_bytes;
    if (!pin && cfg.verbose && complement > 0)
        std::fprintf(stderr, "glm5 experts: the host complement (%.1f GiB) is over the pin limit; misses are staged from the mapped file\n", complement / 1073741824.0);
    pinned_.assign((size_t) NL, nullptr);
    layer_pinned_.assign((size_t) NL, false);
    // staging slots per device, and their pinned host twins
    stage_.assign(devices_.size(), nullptr);
    win_bufs_.assign(devices_.size(), {});
    stage_host_.assign(devices_.size(), nullptr);
    stage_slot_bytes_.assign(devices_.size(), widest);
    for (size_t d = 0; d < devices_.size(); ++d) {
        check(cudaSetDevice(devices_[d]), "cudaSetDevice");
        void* p = nullptr;
        check(cudaMalloc(&p, widest * (uint64_t) k_max_), "cudaMalloc staging");
        device_allocs_.push_back(p);
        stage_[d] = static_cast<uint8_t*>(p);
        if (wide_) {   // built for wide windows: the prompt path's prefetch ring (non-resident experts only)
            uint64_t nonres_max = 0;
            for (int il = hp.n_dense_lead; il < NL; ++il) {
                const LayerExperts& E = layers_[(size_t) il];
                if (E.device != devices_[d]) continue;
                nonres_max = std::max(nonres_max, (uint64_t) (hp.n_expert - E.n_slots) * E.stride);
            }
            const int nbuf = std::getenv("STRATA_GLM5_PF_BUFS") ? std::max(2, std::atoi(std::getenv("STRATA_GLM5_PF_BUFS"))) : 4;
            for (int k = 0; k < nbuf; ++k) {
                void* p2 = nullptr;
                check(cudaMalloc(&p2, std::max<uint64_t>(nonres_max, 16)), "cudaMalloc prefetch ring");
                device_allocs_.push_back(p2);
                win_bufs_[d].push_back(static_cast<uint8_t*>(p2));
            }
            if (cfg.verbose) std::fprintf(stderr, "glm5 experts: device %d prefetch ring %d x %.2f GiB\n", devices_[d], nbuf, nonres_max / 1073741824.0);
        }
        void* hp = nullptr;
        check(cudaHostAlloc(&hp, widest * (uint64_t) k_max_, cudaHostAllocPortable), "cudaHostAlloc staging");
        stage_host_[d] = static_cast<uint8_t*>(hp);
        cudaStream_t cs;
        check(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "copy stream");
        copy_streams_.push_back(cs);
        void* tp = nullptr;
        check(cudaHostAlloc(&tp, 64 * sizeof(uint32_t), cudaHostAllocPortable), "tok pinned");
        tok_pinned_.push_back(static_cast<uint32_t*>(tp));
        cudaEvent_t ce;
        check(cudaEventCreateWithFlags(&ce, cudaEventDisableTiming), "copy event");
        copy_events_.push_back(ce);
        cudaEvent_t ae;
        check(cudaEventCreateWithFlags(&ae, cudaEventDisableTiming), "adapt event");
        adapt_events_.push_back(ae);
        adapt_pending_.push_back(false);
    }
    cpu_threads_ = std::getenv("STRATA_GLM5_CPU_THREADS") ? std::max(1, std::atoi(std::getenv("STRATA_GLM5_CPU_THREADS"))) : 8;
    if (std::getenv("STRATA_GLM5_CPU_PAIR_MS")) { cpu_pair_ms_ = std::atof(std::getenv("STRATA_GLM5_CPU_PAIR_MS")); cpu_pair_fixed_ = true; }
    if (std::getenv("STRATA_GLM5_DMA_MS")) dma_ms_ = std::atof(std::getenv("STRATA_GLM5_DMA_MS"));
    if (std::getenv("STRATA_GLM5_GPU_BEFORE_DMA_MS")) gpu_before_dma_ms_ = std::atof(std::getenv("STRATA_GLM5_GPU_BEFORE_DMA_MS"));
    if (std::getenv("STRATA_GLM5_GPU_BEFORE_CPU_MS")) gpu_before_cpu_ms_ = std::atof(std::getenv("STRATA_GLM5_GPU_BEFORE_CPU_MS"));
    pool_.start(cpu_threads_);
    evict_.resize(devices_.size());
    for (size_t d = 0; d < devices_.size(); ++d) {
        Pool_& P = evict_[d];
        P.slot_bytes = widest;
        const uint64_t chunk_bytes = 2ull << 30;   // 2 GiB chunks pin quickly (one big allocation did not return)
        P.chunk_slots = chunk_bytes / widest;
        uint64_t left = cfg.evict_pool_bytes;
        while (left >= widest && P.n_slots < 1 << 20) {
            const uint64_t n = std::min<uint64_t>(P.chunk_slots, left / widest);
            void* p = nullptr;
            if (cudaHostAlloc(&p, n * widest, cudaHostAllocPortable) != cudaSuccess) break;
            P.chunks.push_back(static_cast<uint8_t*>(p));
            P.n_slots += (int) n;
            pool_bytes_ += n * widest;
            left -= n * widest;
            if (n < P.chunk_slots) break;
        }
        P.holder.assign((size_t) P.n_slots, {-1, -1});
        if (cfg.verbose) std::fprintf(stderr, "glm5 experts: device %d eviction pool %d slots (%.1f GiB pinned)\n", devices_[d], P.n_slots, P.n_slots * widest / 1073741824.0);
    }
    // pass 1: the complement into page-locked memory, read with pread while the page cache is still free of the
    // resident experts (pinning under a full page cache ran at ~3 GB/min)
    if (pin) {
        for (int il = hp.n_dense_lead; il < NL; ++il) {
            LayerExperts& E = layers_[(size_t) il];
            std::vector<int> missing;
            for (int e = 0; e < hp.n_expert; ++e) if (E.slot_of[(size_t) e] == -1) missing.push_back(e);
            const uint64_t bytes = (uint64_t) missing.size() * E.blob;
            if (!bytes) continue;
            void* p = nullptr;
            if (cudaHostAlloc(&p, bytes, cudaHostAllocPortable) != cudaSuccess) {
                if (cfg.verbose) std::fprintf(stderr, "glm5 experts: layer %d: could not pin %.2f GiB; its misses are staged from the file\n", il, bytes / 1073741824.0);
                continue;
            }
            pinned_[(size_t) il] = p;
            layer_pinned_[(size_t) il] = true;
            pinned_bytes_ += bytes;
            std::vector<std::function<void()>> jobs;
            for (size_t i = 0; i < missing.size(); ++i) {
                uint8_t* dst = static_cast<uint8_t*>(p) + (uint64_t) i * E.blob;
                const int e = missing[i];
                jobs.emplace_back([this, &E, e, dst]() { read_blob(E, e, dst); });
            }
            pool_.run(std::move(jobs));
            for (size_t i = 0; i < missing.size(); ++i) {
                const int e = missing[i];
                const uint8_t* dst = static_cast<const uint8_t*>(p) + (uint64_t) i * E.blob;
                E.host_gate[(size_t) e] = dst;
                E.host_up[(size_t) e] = dst + E.gate_bytes;
                E.host_down[(size_t) e] = dst + E.gate_bytes + E.up_bytes;
                E.host_pinned[(size_t) e] = 1;
            }
            if (cfg.verbose && (il % 8 == 0 || il == NL - 1))
                std::fprintf(stderr, "glm5 experts: layer %2d: %zu experts pinned, %.1f GiB pinned so far, %.0f s\n", il, missing.size(),
                             pinned_bytes_ / 1073741824.0, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    // pass 2: the resident slots, read with pread into a pinned bounce and DMA'd (one bounce per device)
    std::vector<uint8_t*> bounce(devices_.size(), nullptr);
    for (size_t d = 0; d < devices_.size(); ++d) {
        void* b = nullptr;
        check(cudaHostAlloc(&b, widest * 16, cudaHostAllocPortable), "cudaHostAlloc bounce");
        bounce[d] = static_cast<uint8_t*>(b);
    }
    uint64_t copied = 0;
    for (int il = hp.n_dense_lead; il < NL; ++il) {
        LayerExperts& E = layers_[(size_t) il];
        check(cudaSetDevice(E.device), "cudaSetDevice");
        size_t di = 0;
        while (devices_[di] != E.device) ++di;
        {
            void* st = nullptr;
            check(cudaMalloc(&st, (size_t) hp.n_expert * 4), "cudaMalloc slot table");
            device_allocs_.push_back(st);
            check(cudaMemcpy(st, E.slot_of.data(), (size_t) hp.n_expert * 4, cudaMemcpyHostToDevice), "slot table");
            E.slot_of_dev = static_cast<int32_t*>(st);
        }
        if (E.n_slots) {
            void* p = nullptr;
            check(cudaMalloc(&p, E.stride * (uint64_t) E.n_slots), "cudaMalloc expert slots");
            device_allocs_.push_back(p);
            E.slots = static_cast<uint8_t*>(p);
            for (int s0 = 0; s0 < E.n_slots; s0 += 16) {
                const int n = std::min(16, E.n_slots - s0);
                std::vector<std::function<void()>> jobs;
                for (int i = 0; i < n; ++i) {
                    const int e = E.expert_of[(size_t) (s0 + i)];
                    uint8_t* dst = bounce[di] + (uint64_t) i * widest;
                    jobs.emplace_back([this, &E, e, dst]() { read_blob(E, e, dst); });
                }
                pool_.run(std::move(jobs));
                for (int i = 0; i < n; ++i)
                    check(cudaMemcpyAsync(E.slots + (uint64_t) (s0 + i) * E.stride, bounce[di] + (uint64_t) i * widest, E.blob,
                                          cudaMemcpyHostToDevice, nullptr), "expert upload");
                check(cudaStreamSynchronize(nullptr), "expert upload sync");
                copied += (uint64_t) n * E.blob;
            }
        }
        if (cfg.verbose && (il % 8 == 0 || il == NL - 1))
            std::fprintf(stderr, "glm5 experts: layer %2d: %d resident on device %d, %.1f GiB copied so far, %.0f s\n", il, E.n_slots,
                         E.device, copied / 1073741824.0, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    for (uint8_t* b : bounce) cudaFreeHost(b);
    doorbell_ready_ = true;
    if (cfg.verbose)
        std::fprintf(stderr, "glm5 experts: %.1f GiB resident, %.1f GiB pinned on the host, in %.0f s\n", copied / 1073741824.0,
                     pinned_bytes_ / 1073741824.0, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

int ExpertStore::device_index_of_layer(int il) const {
    const int dev = layers_[(size_t) il].device;
    for (size_t d = 0; d < devices_.size(); ++d) if (devices_[d] == dev) return (int) d;
    return 0;
}

bool ExpertStore::promote_loan(LayerExperts& E, int il, size_t di, int e, uint32_t tok, int max_swaps, int& swapped, void* cs_) {
    cudaStream_t cs = static_cast<cudaStream_t>(cs_);
    const uint32_t lm = E.last_miss_tok[(size_t) e];
    const bool admit = admit_window_ == 0 || (lm != 0 && tok + 1 - lm <= admit_window_);
    E.last_miss_tok[(size_t) e] = tok + 1;
    if (!admit || swapped >= max_swaps || evict_[di].n_slots != 0 || !E.host_pinned[(size_t) e] || E.host_gate[(size_t) e] == nullptr) return false;
    const int32_t victim = pick_victim(E, tok, tok);   // (a slot no row of this window reads)
    if (victim < 0) return false;
    int j;
    {
        std::lock_guard<std::mutex> lk(loan_m_);
        j = E.loan_of[(size_t) e];
        if (loan_apply_n_[(size_t) il] + 2 > kLoanApplyMax) j = -1;   // both table entries must land: no promotion when the list is full
    }
    if (j < 0) return false;
    const int e_out = E.expert_of[(size_t) victim];
    uint8_t* slot_mem = E.slots + (uint64_t) victim * E.stride;
    uint8_t* hb = const_cast<uint8_t*>(E.host_gate[(size_t) e]);
    // the exchange of homes: the victim's blob into e's pinned host blob (which it inherits), e's loan copy into the slot
    check(cudaMemcpyAsync(hb, slot_mem, E.blob, cudaMemcpyDeviceToHost, cs), "promote d2h");
    check(cudaMemcpyAsync(slot_mem, loan_ptr_h_[di][(size_t) j], E.blob, cudaMemcpyDeviceToDevice, cs), "promote d2d");
    copy_bytes_ += 2 * E.blob;
    E.host_gate[(size_t) e_out] = hb; E.host_up[(size_t) e_out] = hb + E.gate_bytes; E.host_down[(size_t) e_out] = hb + E.gate_bytes + E.up_bytes;
    E.host_pinned[(size_t) e_out] = 1;
    E.host_gate[(size_t) e] = E.host_up[(size_t) e] = E.host_down[(size_t) e] = nullptr;
    E.host_pinned[(size_t) e] = 0;
    E.slot_of[(size_t) e_out] = -1;
    E.evicted_once[(size_t) e_out] = 1;
    E.slot_of[(size_t) e] = victim;
    E.expert_of[(size_t) victim] = e;
    E.slot_last_used[(size_t) victim] = tok;
    E.loan_hits_n[(size_t) e] = 0;
    ++swapped; ++promotions_;
    {
        // the device's table at the layer's next grouping (after this window's flag, which the copies precede): e -> the
        // slot, e_out -> missing; the loan slot stays out of the free list (this window may read it) until the LRU takes it
        std::lock_guard<std::mutex> lk(loan_m_);
        loan_push(il, e, -2 - victim);
        loan_push(il, e_out, -1);
        E.loan_of[(size_t) e] = -1;
        E.loan_live[(size_t) e] = 0;
        loan_holder_[di][(size_t) j] = {-1, -1};
        loan_last_used_[di][(size_t) j] = tok;
        loan_busy_[di][(size_t) j] = 0;
    }
    return true;
}

int32_t ExpertStore::pick_victim(LayerExperts& E, uint32_t tok, uint32_t protect) {
    // The slot whose expert was routed longest ago, never one routed in a window >= protect; among equally old slots
    // the one whose expert has a pinned host copy (no device-to-host save needed).  A policy that always evicted the
    // pinned ones first kept the resident set frozen at the profile and thrashed the conversation's working set.
    int32_t victim = -1;
    uint32_t oldest = protect;
    bool victim_pinned = false;
    for (int32_t sl = 0; sl < E.n_slots; ++sl) {
        const uint32_t u = E.slot_last_used[(size_t) sl];
        if (u >= oldest && !(u == oldest && victim >= 0 && !victim_pinned)) continue;
        const bool pinned = E.host_pinned[(size_t) E.expert_of[(size_t) sl]] != 0;
        if (u < oldest || pinned) { oldest = u; victim = sl; victim_pinned = pinned; }
    }
    (void) tok;
    return victim;
}

// the evicted blob goes to the pinned pool (device-to-host, the other PCIe direction), so a later miss on it is one
// DMA instead of a file read
void ExpertStore::evict_to_pool(LayerExperts& E, int il, size_t di, int32_t victim, void* stream, int keep_e) {
    const int e_out = E.expert_of[(size_t) victim];
    Pool_& P = evict_[di];
    if (P.n_slots == 0 || E.host_pinned[(size_t) e_out]) return;
    // never recycle the pool slot holding the expert that is about to be copied in from it
    if (P.holder[(size_t) P.next] == std::make_pair(il, keep_e)) P.next = (P.next + 1) % P.n_slots;
    const int ps = P.next;
    P.next = (P.next + 1) % P.n_slots;
    const auto [hl, he] = P.holder[(size_t) ps];
    if (hl >= 0) {   // the ring wrapped: that expert goes back to the file
        LayerExperts& H = layers_[(size_t) hl];
        H.host_pinned[(size_t) he] = 0;
        H.host_gate[(size_t) he] = H.host_up[(size_t) he] = H.host_down[(size_t) he] = nullptr;
    }
    uint8_t* hp = P.at(ps);
    check(cudaMemcpyAsync(hp, E.slots + (uint64_t) victim * E.stride, E.blob, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream)), "evict d2h");
    P.holder[(size_t) ps] = {il, e_out};
    E.host_gate[(size_t) e_out] = hp;
    E.host_up[(size_t) e_out] = hp + E.gate_bytes;
    E.host_down[(size_t) e_out] = hp + E.gate_bytes + E.up_bytes;
    E.host_pinned[(size_t) e_out] = 1;
}

namespace {
// the flag kernel on the layer's copy stream: a kernel launch needs that device current (a memcpy does not)
void flag_kernel(int device, uint32_t* flag_dev, uint32_t tok, void* cs) {
    int cur = 0;
    cudaGetDevice(&cur);
    if (cur != device) cudaSetDevice(device);
    strata::glm5::write_flag(flag_dev, tok, cs);
    if (cur != device) cudaSetDevice(cur);
}
// the loan copies by the shaders when the blob allows it (STRATA_GLM5_LOAN_MEMCPY=1: the copy engine, as before)
const bool kLoanMemcpy = std::getenv("STRATA_GLM5_LOAN_MEMCPY") && std::atoi(std::getenv("STRATA_GLM5_LOAN_MEMCPY")) != 0;
void copy_dev_kernel(int device, void* dst, const void* src, size_t bytes, void* ls) {
    int cur = 0;
    cudaGetDevice(&cur);
    if (cur != device) cudaSetDevice(device);
    strata::glm5::copy_dev(dst, src, bytes, ls);
    if (cur != device) cudaSetDevice(cur);
}
}  // namespace

int ExpertStore::prefetch(int il, const int32_t* ids, int n, uint32_t tok, uint32_t* h_flag, uint32_t* flag_dev, int* pf_n, int* pf_e, int* pf_slot,
                          int* pf_out, int max_swaps) {
    // the lookahead prefetch: the predicted non-resident experts of layer il are exchanged into resident slots (the
    // same exchange as dma_miss: in through a staging slot, the victim out into the incoming expert's pinned blob) on
    // the copy stream, and the layer's prefetch flag is written behind the copies (flag_dev) for the device's pf_apply.
    // Staging slots from the top half: the layer being served uses the bottom ones (its unplaced misses are read
    // from staging by the device) and the copy stream orders every reuse.
    LayerExperts& E = layers_[(size_t) il];
    const size_t di = (size_t) device_index_of_layer(il);
    cudaStream_t cs = static_cast<cudaStream_t>(copy_streams_[di]);
    const int stage_lo = k_max_ / 2, stage_n = k_max_ - stage_lo;
    int done = 0;
    std::vector<int> seen;
    for (int j = 0; j < n && done < max_swaps && done < stage_n; ++j) {
        const int e = ids[j];
        if (e < 0 || e >= (int) E.slot_of.size()) continue;
        if (E.slot_of[(size_t) e] >= 0) { E.slot_last_used[(size_t) E.slot_of[(size_t) e]] = tok; continue; }   // keep it: predicted
        if (std::find(seen.begin(), seen.end(), e) != seen.end()) continue;
        seen.push_back(e);
        if (!E.host_pinned[(size_t) e] || E.host_gate[(size_t) e] == nullptr) continue;   // only the pinned complement is worth a prefetch
        const int32_t victim = pick_victim(E, tok, tok);   // not a slot this window touched
        if (victim < 0) break;
        const int e_out = E.expert_of[(size_t) victim];
        uint8_t* slot_mem = E.slots + (uint64_t) victim * E.stride;
        uint8_t* hb = const_cast<uint8_t*>(E.host_gate[(size_t) e]);
        uint8_t* st = stage_[di] + (size_t) (stage_lo + done) * stage_slot_bytes_[di];
        check(cudaMemcpyAsync(st, hb, E.blob, cudaMemcpyHostToDevice, cs), "prefetch dma");
        copy_bytes_ += E.blob;
        check(cudaMemcpyAsync(hb, slot_mem, E.blob, cudaMemcpyDeviceToHost, cs), "prefetch exchange d2h");
        check(cudaMemcpyAsync(slot_mem, st, E.blob, cudaMemcpyDeviceToDevice, cs), "prefetch exchange d2d");
        E.host_gate[(size_t) e_out] = hb;
        E.host_up[(size_t) e_out] = hb + E.gate_bytes;
        E.host_down[(size_t) e_out] = hb + E.gate_bytes + E.up_bytes;
        E.host_pinned[(size_t) e_out] = 1;
        E.host_gate[(size_t) e] = E.host_up[(size_t) e] = E.host_down[(size_t) e] = nullptr;
        E.host_pinned[(size_t) e] = 0;
        pf_e[done] = e;
        pf_slot[done] = victim;
        pf_out[done] = e_out;
        E.slot_of[(size_t) e_out] = -1;
        E.evicted_once[(size_t) e_out] = 1;
        E.slot_of[(size_t) e] = victim;
        E.expert_of[(size_t) victim] = e;
        E.slot_last_used[(size_t) victim] = tok;
        ++done;
    }
    *pf_n = done;
    prefetched_ += (uint64_t) done;
    if (done == 0) __atomic_store_n(h_flag, tok, __ATOMIC_RELEASE);
    else if (flag_dev) flag_kernel(E.device, flag_dev, tok, cs);
    else __atomic_store_n(h_flag, tok, __ATOMIC_RELEASE);
    return done;
}

int ExpertStore::prefetch_stage(int il, const int32_t* ids, int T, int P, uint32_t tok, int cap) {
    LayerExperts& E = layers_[(size_t) il];
    const size_t di = (size_t) device_index_of_layer(il);
    cudaStream_t cs = static_cast<cudaStream_t>(copy_streams_[di]);
    const int stage_lo = k_max_ / 2;
    std::vector<std::pair<int, int>>& pf = pf_stage_[(size_t) il];
    pf.clear();
    cap = std::min(cap, k_max_ - stage_lo);
    for (int r = 0; r < P && (int) pf.size() < cap; ++r)
        for (int t = 0; t < T && (int) pf.size() < cap; ++t) {
            const int e = ids[t * P + r];
            if (e < 0 || e >= (int) E.slot_of.size() || E.slot_of[(size_t) e] >= 0) continue;
            if (!E.host_pinned[(size_t) e] || E.host_gate[(size_t) e] == nullptr) continue;
            bool dup = false;
            for (const auto& q : pf) if (q.first == e) { dup = true; break; }
            if (dup) continue;
            const int p = stage_lo + (int) pf.size();
            check(cudaMemcpyAsync(stage_[di] + (size_t) p * stage_slot_bytes_[di], E.host_gate[(size_t) e], E.blob, cudaMemcpyHostToDevice, cs), "prefetch stage dma");
            copy_bytes_ += E.blob;
            pf.emplace_back(e, p);
        }
    (void) tok;
    prefetched_ += (uint64_t) pf.size();
    return (int) pf.size();
}

void ExpertStore::dma_miss(LayerExperts& E, int il, size_t di, size_t m, int e, uint32_t tok, int* miss_slot, int* miss_out, int max_swaps, int& swapped,
                           std::vector<std::function<void()>>& jobs, std::vector<int>& staged, void* cs_) {
    cudaStream_t cs = static_cast<cudaStream_t>(cs_);
    // the destination: the least recently used slot not touched by this window, else staging slot m.  Admission:
    // only an expert missed again within admit_window windows displaces a resident one (LRU over a resident set
    // smaller than the working set thrashed: nearly every swapped-in expert displaced one needed again)
    const uint32_t lm = E.last_miss_tok[(size_t) e];
    const bool admit = admit_window_ == 0 || (lm != 0 && tok + 1 - lm <= admit_window_);
    E.last_miss_tok[(size_t) e] = tok + 1;
    int32_t victim = (admit && swapped < max_swaps) ? pick_victim(E, tok, tok) : -1;
    uint8_t* dst;
    if (victim >= 0) {
        const int e_out = E.expert_of[(size_t) victim];
        miss_slot[m] = victim;
        miss_out[m] = e_out;
        uint8_t* slot_mem = E.slots + (uint64_t) victim * E.stride;
        if (evict_[di].n_slots == 0 && E.host_pinned[(size_t) e]) {
            // the exchange: every expert has one home, a VRAM slot or its pinned host blob.  The incoming expert
            // lands in staging, the victim is saved into the incoming expert's host blob (which it inherits), and
            // staging is copied into the slot.  No pool, no file reads after the load: pinned host = the complement.
            uint8_t* hb = const_cast<uint8_t*>(E.host_gate[(size_t) e]);
            uint8_t* st = stage_[di] + m * stage_slot_bytes_[di];
            check(cudaMemcpyAsync(st, hb, E.blob, cudaMemcpyHostToDevice, cs), "serve dma");
            copy_bytes_ += E.blob;   // the doorbell path has no timing events: the bytes alone
            check(cudaMemcpyAsync(hb, slot_mem, E.blob, cudaMemcpyDeviceToHost, cs), "exchange d2h");
            check(cudaMemcpyAsync(slot_mem, st, E.blob, cudaMemcpyDeviceToDevice, cs), "exchange d2d");
            E.host_gate[(size_t) e_out] = hb;
            E.host_up[(size_t) e_out] = hb + E.gate_bytes;
            E.host_down[(size_t) e_out] = hb + E.gate_bytes + E.up_bytes;
            E.host_pinned[(size_t) e_out] = 1;
            E.host_gate[(size_t) e] = E.host_up[(size_t) e] = E.host_down[(size_t) e] = nullptr;
            E.host_pinned[(size_t) e] = 0;
            dst = nullptr;   // copied already
        } else {
            evict_to_pool(E, il, di, victim, cs, e);
            dst = slot_mem;
        }
        E.slot_of[(size_t) e_out] = -1;
        E.evicted_once[(size_t) e_out] = 1;
        E.slot_of[(size_t) e] = victim;
        E.expert_of[(size_t) victim] = e;
        E.slot_last_used[(size_t) victim] = tok;
        ++swapped;
    } else {
        miss_slot[m] = -1;
        miss_out[m] = -1;
        dst = stage_[di] + m * stage_slot_bytes_[di];
    }
    if (!dst) {
    } else if (E.host_pinned[(size_t) e]) {
        check(cudaMemcpyAsync(dst, E.host_gate[(size_t) e], E.blob, cudaMemcpyHostToDevice, cs), "serve dma");
    } else {
        // no contiguous pinned copy (an expert that was resident at the start, since evicted): pread the three
        // parts into the pinned staging twin, DMA from there
        uint8_t* h = stage_host_[di] + m * stage_slot_bytes_[di];
        jobs.emplace_back([this, &E, e, h]() { read_blob(E, e, h); });
        staged.push_back((int) m);
    }
}

void ExpertStore::finish_staged(LayerExperts& E, size_t di, int* miss_slot, std::vector<std::function<void()>>& jobs, std::vector<int>& staged, void* cs_) {
    cudaStream_t cs = static_cast<cudaStream_t>(cs_);
    if (!jobs.empty()) {
        const auto t0 = std::chrono::steady_clock::now();
        pool_.run(std::move(jobs));
        file_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        file_reads_ += staged.size();
        file_bytes_ += (uint64_t) staged.size() * E.blob;
        for (int m : staged) {
            const int32_t dst_slot = miss_slot[(size_t) m];
            uint8_t* dst = dst_slot >= 0 ? E.slots + (uint64_t) dst_slot * E.stride : stage_[di] + (uint64_t) m * stage_slot_bytes_[di];
            check(cudaMemcpyAsync(dst, stage_host_[di] + (uint64_t) m * stage_slot_bytes_[di], E.blob, cudaMemcpyHostToDevice, cs), "serve dma");
        }
    }
}

int ExpertStore::serve(int il, const int32_t* ids, int n, uint32_t tok, uint32_t* h_flag, uint32_t* flag_dev, int* miss_slot, int* miss_out, int max_swaps) {
    LayerExperts& E = layers_[(size_t) il];
    if (route_log_) { std::fprintf(route_log_, "T %u %d", tok, il); for (int j = 0; j < n; ++j) std::fprintf(route_log_, " %d", ids[j]); std::fprintf(route_log_, "\n"); }
    const size_t di = (size_t) device_index_of_layer(il);
    cudaStream_t cs = static_cast<cudaStream_t>(copy_streams_[di]);
    std::vector<std::function<void()>> jobs;
    std::vector<int> missed;   // distinct missed experts, first-occurrence order = the device's miss index
    int swapped = 0, n_promoted = 0;
    for (int j = 0; j < n; ++j) {
        const int e = ids[j];
        if (e < 0 || e >= (int) E.slot_of.size())
            throw std::runtime_error("serve: layer " + std::to_string(il) + " routed expert " + std::to_string(e) + " at pair " + std::to_string(j) + " of " + std::to_string(n));
        ++counts_[(size_t) il][(size_t) e];
        const int32_t slot = E.slot_of[(size_t) e];
        if (slot >= 0) { ++hits_; E.slot_last_used[(size_t) slot] = tok; continue; }
        if (loan_on_ && E.loan_live[(size_t) e]) {   // a loan the device's table holds: a hit for both sides (the device numbers the staging slots by its table)
            ++hits_; ++loan_hits_;
            if (!seen_[(size_t) e]) {
                seen_[(size_t) e] = 2;
                uint8_t& hn = E.loan_hits_n[(size_t) e];
                if (hn < 255) ++hn;
                const bool want = promote_hits_ > 0 && hn >= promote_hits_ && n_promoted < promote_max_;
                if (want && promote_loan(E, il, di, e, tok, max_swaps, swapped, cs)) ++n_promoted;
                else loan_last_used_[di][(size_t) E.loan_of[(size_t) e]] = tok;
            }
            continue;
        }
        ++misses_;
        if (seen_[(size_t) e]) continue;
        seen_[(size_t) e] = 1;
        missed.push_back(e);
    }
    for (int e : missed) seen_[(size_t) e] = 0;
    for (int j = 0; j < n; ++j) seen_[(size_t) ids[j]] = 0;   // (the loan hits' marks too)
    std::sort(missed.begin(), missed.end());   // k_moe_group numbers the staging slots in ascending expert id
    if ((int) missed.size() > k_max_) throw std::runtime_error("serve: " + std::to_string(missed.size()) + " distinct misses, staging holds " + std::to_string(k_max_));
    distinct_misses_ += missed.size();
    if (!missed.empty()) ++layers_missed_;
    for (int e : missed) if (E.evicted_once[(size_t) e]) ++thrash_;
    std::vector<int> staged;   // misses that stay in staging (read from the file first when unpinned)
    cudaEvent_t t_start = nullptr, t_end = nullptr;
    if (copy_timing_ && !missed.empty()) {
        // the events must belong to the copy stream's device; the host thread's current device is the engine's
        int cur = 0;
        cudaGetDevice(&cur);
        cudaSetDevice(E.device);
        cudaEventCreateWithFlags(&t_start, 0); cudaEventCreateWithFlags(&t_end, 0);
        cudaSetDevice(cur);
        cudaEventRecord(t_start, cs);
    }
    for (size_t m = 0; m < missed.size(); ++m) dma_miss(E, il, di, m, missed[m], tok, miss_slot, miss_out, max_swaps, swapped, jobs, staged, cs);
    finish_staged(E, di, miss_slot, jobs, staged, cs);
    swaps_ += (uint64_t) swapped;
    // the loans, as the CPU-split path makes them: a miss that stays in its staging slot is also copied into a loan
    // slot (device to device, behind the staging copies), a hit from this layer's next window on
    if (loan_on_ && E.stride <= loan_stride_ && !missed.empty()) {
        cudaStream_t ls = static_cast<cudaStream_t>(loan_streams_[di]);
        LoanBatch batch{nullptr, di, {}};
        bool ordered = false;
        for (size_t m = 0; m < missed.size() && (int) batch.adds.size() < loan_per_win_; ++m) {
            if (miss_slot[m] >= 0) continue;   // swapped into a resident slot: resident already
            const int e = missed[m];
            int j;
            { std::lock_guard<std::mutex> lk(loan_m_); j = loan_alloc(di, il, e, tok); }
            if (j < 0) break;
            if (!ordered) {
                std::lock_guard<std::mutex> lk(loan_m_);
                cudaEvent_t se = static_cast<cudaEvent_t>(loan_event(di));
                check(cudaEventRecord(se, cs), "loan stage event"); check(cudaStreamWaitEvent(ls, se, 0), "loan stage wait");
                loan_event_done(di, se); ordered = true;
            }
            uint8_t* dst = loan_ptr_h_[di][(size_t) j];
            if (!kLoanMemcpy && E.blob % 16 == 0) copy_dev_kernel(E.device, dst, stage_[di] + m * stage_slot_bytes_[di], E.blob, ls);
            else check(cudaMemcpyAsync(dst, stage_[di] + m * stage_slot_bytes_[di], E.blob, cudaMemcpyDeviceToDevice, ls), "loan d2d");
            copy_bytes_ += E.blob;
            batch.adds.emplace_back(e, j);
        }
        if (!batch.adds.empty()) {
            std::lock_guard<std::mutex> lk(loan_m_);
            batch.ev = loan_event(di);
            check(cudaEventRecord(static_cast<cudaEvent_t>(batch.ev), ls), "loan batch event");
            loans_made_ += batch.adds.size();
            loan_batches_[(size_t) il].push_back(std::move(batch));
        }
    }
    if (t_start) {
        cudaEventRecord(t_end, cs);
        timing_events_.emplace_back(t_start, t_end);
        timing_bytes_.push_back((uint64_t) missed.size() * E.blob);
    }
    if (missed.empty()) {
        __atomic_store_n(h_flag, tok, __ATOMIC_RELEASE);
    } else {
        if (flag_dev) flag_kernel(E.device, flag_dev, tok, cs);
    }
    return (int) missed.size();
}

namespace {
// native_expert.cpp's native_fmt without its activation-size caps (those are the Qwen pool's buffers; GLM's n_embd 4096
// quantizes to 4672 bytes of Q8_K and this store owns its scratch)
bool glm_native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, strata::kernels::cpu::NativeFmt& f, std::string& err) {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
    const ggml_type_traits_cpu* tg = ggml_get_type_traits_cpu((ggml_type) gu_type);
    const ggml_type_traits_cpu* td = ggml_get_type_traits_cpu((ggml_type) d_type);
    if (!tg || !tg->vec_dot || !td || !td->vec_dot) { err = "ggml-cpu has no dot product for type " + std::to_string(tg && tg->vec_dot ? d_type : gu_type); return false; }
    const ggml_type_traits_cpu* ag = ggml_get_type_traits_cpu(tg->vec_dot_type);
    const ggml_type_traits_cpu* ad = ggml_get_type_traits_cpu(td->vec_dot_type);
    if (!ag || !ag->from_float || !ad || !ad->from_float) { err = "ggml-cpu cannot quantize the activation"; return false; }
    if (n_embd % ggml_blck_size((ggml_type) gu_type) || n_ff % ggml_blck_size((ggml_type) d_type) || n_embd % ggml_blck_size(tg->vec_dot_type) ||
        n_ff % ggml_blck_size(td->vec_dot_type)) { err = "expert geometry is not whole blocks"; return false; }
    f.gu_type = gu_type; f.d_type = d_type;
    f.gu_act = (int) tg->vec_dot_type; f.d_act = (int) td->vec_dot_type;
    f.n_embd = n_embd; f.n_ff = n_ff;
    f.gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    f.d_row = ggml_row_size((ggml_type) d_type, n_ff);
    f.up_off = (size_t) n_ff * f.gu_row;
    f.down_off = 2 * f.up_off;
    f.bytes = f.down_off + (size_t) n_embd * f.d_row;
    f.act_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    f.h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    return true;
}
}  // namespace

// ---- the CPU service of the decode path's misses
void ExpertStore::init_cpu(const Model& m) {
    const HParams& hp = m.hp();
    const int NL = (int) layers_.size();
    fmt_.assign((size_t) NL, {});
    bool ok = strata::kernels::cpu::native_experts_available();
    std::string why;
    for (int il = hp.n_dense_lead; il < NL && ok; ++il) {
        const LayerExperts& E = layers_[(size_t) il];
        if (!layer_pinned_[(size_t) il] && E.n_slots < hp.n_expert) { ok = false; why = "layer " + std::to_string(il) + " is not pinned"; break; }
        if (!glm_native_fmt((int) E.gu_type, (int) E.d_type, E.n_in, E.n_ff, fmt_[(size_t) il], why)) { ok = false; break; }
        if (fmt_[(size_t) il].gu_row * (size_t) E.n_ff != E.gate_bytes || fmt_[(size_t) il].bytes != E.blob) { ok = false; why = "blob layout differs from ggml-cpu's"; break; }
    }
    cpu_ready_ = ok;
    if (!ok) { std::fprintf(stderr, "glm5 experts: CPU misses unavailable (%s): misses are DMA'd\n", why.c_str()); return; }
    size_t act_max = 0, h_max = 0;
    int64_t ff_max = 0;
    for (int il = hp.n_dense_lead; il < NL; ++il) { act_max = std::max(act_max, fmt_[(size_t) il].act_bytes); h_max = std::max(h_max, fmt_[(size_t) il].h_bytes); ff_max = std::max(ff_max, fmt_[(size_t) il].n_ff); }
    y_rows_ = kMidCpuRows;   // the decode windows use at most 64 rows; the mid-size split up to kMidCpuRows
    act_.assign(2048, std::vector<uint8_t>(act_max + 64));
    ff_.assign((size_t) y_rows_, std::vector<float>((size_t) ff_max));
    hq_.assign((size_t) y_rows_, std::vector<uint8_t>(h_max + 64));
    // two blocks by layer parity: layer il's rows are copied (and waited on at its combine) before layer il + 2's are written
    y_pin_rows_ = 2 * y_rows_;
    void* p = nullptr;
    check(cudaHostAlloc(&p, (size_t) y_pin_rows_ * hp.n_embd * 4, cudaHostAllocPortable | cudaHostAllocMapped), "cudaHostAlloc y_pin");
    y_pin_ = static_cast<float*>(p);
    { void* dp = nullptr; if (cudaHostGetDevicePointer(&dp, p, 0) == cudaSuccess) y_pin_dev_ = static_cast<float*>(dp); else cudaGetLastError(); }
    if (y_pin_dev_) {   // the copy descriptors (row runs per served expert), two sets by layer parity like the result blocks
        void* q = nullptr;
        check(cudaHostAlloc(&q, (size_t) 2 * kCopyDescMax * 3 * 4, cudaHostAllocPortable | cudaHostAllocMapped), "cudaHostAlloc copy desc");
        copy_desc_ = static_cast<int32_t*>(q);
        void* dq = nullptr;
        if (cudaHostGetDevicePointer(&dq, q, 0) == cudaSuccess) copy_desc_dev_ = static_cast<int32_t*>(dq);
        else { cudaGetLastError(); cudaFreeHost(q); copy_desc_ = nullptr; }
    }
    std::fprintf(stderr, "glm5 experts: decode misses computed on the CPU (ggml-cpu, %d threads)\n", cpu_threads_);
}

void ExpertStore::init_loans(int n_layers) {
    n_layers = (int) layers_.size();   // every layer entry (the MTP layer included: its tables stay empty)
    loan_on_ = wide_ && !win_bufs_.empty() && !(std::getenv("STRATA_GLM5_LOAN") && std::atoi(std::getenv("STRATA_GLM5_LOAN")) == 0);
    if (std::getenv("STRATA_GLM5_LOAN_PER_WIN")) loan_per_win_ = std::max(0, std::atoi(std::getenv("STRATA_GLM5_LOAN_PER_WIN")));
    // default: loans only from the PCIe-staged misses (their blob crosses the memory bus once, the loan is a device copy);
    // STRATA_GLM5_LOAN_H2D=1 also copies the CPU-served misses from the host (a second read of each: measured slower)
    loan_h2d_ = std::getenv("STRATA_GLM5_LOAN_H2D") && std::atoi(std::getenv("STRATA_GLM5_LOAN_H2D")) != 0;
    for (auto& E : layers_) { E.loan_of.assign(E.slot_of.size(), -1); E.loan_live.assign(E.slot_of.size(), 0); E.loan_hits_n.assign(E.slot_of.size(), 0); }
    if (std::getenv("STRATA_GLM5_PROMOTE_HITS")) promote_hits_ = std::max(0, std::atoi(std::getenv("STRATA_GLM5_PROMOTE_HITS")));
    if (std::getenv("STRATA_GLM5_PROMOTE_MAX")) promote_max_ = std::max(0, std::atoi(std::getenv("STRATA_GLM5_PROMOTE_MAX")));
    if (!loan_on_) return;
    // the slot pitch: the most common layer stride (layers with a larger one do not loan)
    std::map<uint64_t, int> strides;
    for (int il = 0; il < n_layers; ++il) if (layers_[(size_t) il].n_slots || layers_[(size_t) il].stride) ++strides[layers_[(size_t) il].stride];
    for (const auto& kv : strides) if (kv.second >= strides[loan_stride_]) loan_stride_ = kv.first;
    loan_ptr_h_.assign(devices_.size(), {}); loan_ptr_d_.assign(devices_.size(), nullptr);
    loan_holder_.assign(devices_.size(), {}); loan_last_used_.assign(devices_.size(), {}); loan_free_.assign(devices_.size(), {});
    loan_clock_.assign(devices_.size(), 0);
    int total = 0;
    for (size_t d = 0; d < devices_.size(); ++d) {
        check(cudaSetDevice(devices_[d]), "cudaSetDevice");
        uint64_t ring = 0;
        for (int il = 0; il < n_layers; ++il) {
            const LayerExperts& E = layers_[(size_t) il];
            if (E.device != devices_[d] || E.n_slots == 0) continue;
            ring = std::max(ring, (uint64_t) ((int) E.slot_of.size() - E.n_slots) * E.stride);
        }
        for (uint8_t* buf : win_bufs_[d])
            for (uint64_t off = 0; off + loan_stride_ <= ring; off += loan_stride_) loan_ptr_h_[d].push_back(buf + off);
        const int n = (int) loan_ptr_h_[d].size();
        loan_holder_[d].assign((size_t) n, {-1, -1}); loan_last_used_[d].assign((size_t) n, 0u);
        for (int j = n - 1; j >= 0; --j) loan_free_[d].push_back(j);
        total += n;
    }
    loan_ring_n_.assign(devices_.size(), 0);
    for (size_t d = 0; d < devices_.size(); ++d) loan_ring_n_[d] = (int) loan_ptr_h_[d].size();
    finalize_loans();
    loan_apply_n_.assign((size_t) n_layers, 0); loan_apply_side_.assign((size_t) n_layers, 0);
    for (int side = 0; side < 2; ++side) loan_apply_dev_[side].assign((size_t) n_layers, nullptr);
    loan_batches_.assign((size_t) n_layers, {});
    loan_streams_.assign(devices_.size(), nullptr); loan_busy_.assign(devices_.size(), {});
    loan_ev_pool_.assign(devices_.size(), {});
    for (size_t d = 0; d < devices_.size(); ++d) {
        check(cudaSetDevice(devices_[d]), "cudaSetDevice");
        cudaStream_t ls; check(cudaStreamCreateWithFlags(&ls, cudaStreamNonBlocking), "loan stream"); loan_streams_[d] = ls;
        loan_busy_[d].assign(loan_holder_[d].size(), 0);
    }
    for (int side = 0; side < 2; ++side) loan_apply_h_[side].assign((size_t) n_layers, nullptr);
    for (int il = 0; il < n_layers; ++il) {
        const LayerExperts& E = layers_[(size_t) il];
        if (E.device < 0) continue;
        check(cudaSetDevice(E.device), "cudaSetDevice");
        // the entries in mapped pinned memory: the apply kernel reads them from the host directly (a memcpy on the compute
        // stream would queue behind the loan copies on the copy engine and stall the layer)
        for (int side = 0; side < 2; ++side) {
            void* hp = nullptr; check(cudaHostAlloc(&hp, kLoanApplyMax * 8, cudaHostAllocPortable | cudaHostAllocMapped), "cudaHostAlloc loan apply");
            loan_apply_h_[side][(size_t) il] = static_cast<int32_t*>(hp);
            void* dp = nullptr; check(cudaHostGetDevicePointer(&dp, hp, 0), "loan apply device pointer");
            loan_apply_dev_[side][(size_t) il] = static_cast<int32_t*>(dp);
        }
    }
    std::fprintf(stderr, "glm5 experts: loan pool: %d slots of %.2f MB on the prefetch ring (decode windows cache their misses there; at most %d per window, %s)\n",
                 total, loan_stride_ / 1e6, loan_per_win_, loan_h2d_ ? "CPU-served misses copied too" : "PCIe-staged misses only");
}

void ExpertStore::add_loan_region(int device_ordinal, uint8_t* base, size_t bytes, size_t skip) {
    if (!loan_on_ || !base) return;
    size_t di = devices_.size();
    for (size_t d = 0; d < devices_.size(); ++d) if (devices_[d] == device_ordinal) di = d;
    if (di == devices_.size()) return;
    for (size_t off = (skip + 255) / 256 * 256; off + loan_stride_ <= bytes; off += loan_stride_) {
        loan_ptr_h_[di].push_back(base + off);
        loan_holder_[di].push_back({-1, -1});
        loan_last_used_[di].push_back(0u);
        loan_busy_[di].push_back(0);
        loan_free_[di].insert(loan_free_[di].begin(), (int) loan_ptr_h_[di].size() - 1);   // (the ring's slots are taken first)
    }
}

// the device tables (slot -> pointer) from the host lists; called by init_loans and again after regions were added
void ExpertStore::finalize_loans() {
    if (!loan_on_) return;
    int ring = 0, extra = 0;
    for (size_t d = 0; d < devices_.size(); ++d) {
        check(cudaSetDevice(devices_[d]), "cudaSetDevice");
        const int n = (int) loan_ptr_h_[d].size();
        if (loan_ptr_d_[d]) {
            for (size_t i = 0; i < device_allocs_.size(); ++i)
                if (device_allocs_[i] == loan_ptr_d_[d]) { device_allocs_.erase(device_allocs_.begin() + (std::ptrdiff_t) i); break; }
            check(cudaFree(const_cast<uint8_t**>(loan_ptr_d_[d])), "cudaFree loan table");
            loan_ptr_d_[d] = nullptr;
        }
        void* p = nullptr;
        check(cudaMalloc(&p, (size_t) std::max(n, 1) * sizeof(uint8_t*)), "cudaMalloc loan table");
        device_allocs_.push_back(p);
        if (n) check(cudaMemcpy(p, loan_ptr_h_[d].data(), (size_t) n * sizeof(uint8_t*), cudaMemcpyHostToDevice), "loan table");
        loan_ptr_d_[d] = static_cast<const uint8_t**>(p);
        if (!loan_busy_.empty()) loan_busy_[d].resize((size_t) n, 0);
        ring += loan_ring_n_.empty() ? n : loan_ring_n_[d];
        extra += n - (loan_ring_n_.empty() ? n : loan_ring_n_[d]);
    }
    if (extra) std::fprintf(stderr, "glm5 experts: loan pool: %d slots on the prefetch ring + %d in the prompt scratch (%.2f GB more)\n", ring, extra, extra * (double) loan_stride_ / 1e9);
}

int ExpertStore::loan_alloc(size_t di, int il, int e, uint32_t tok) {
    int j = -1;
    if (!loan_free_[di].empty()) { j = loan_free_[di].back(); loan_free_[di].pop_back(); }
    else {
        // the LRU victim: the clock hand walks the pool; a slot used in this window (any layer) is skipped
        const int n = (int) loan_holder_[di].size();
        if (n == 0) return -1;
        int best = -1; uint32_t oldest = tok;
        for (int k = 0; k < n; ++k) {
            const int cand = (loan_clock_[di] + k) % n;
            if (loan_busy_[di][(size_t) cand]) continue;   // a copy in flight
            // only a slot of a layer already served in this window (its grouping is known not to read the slot: last used
            // before this window); a later layer's grouping of this window has not been published yet
            if (loan_holder_[di][(size_t) cand].first > il) continue;
            if (loan_last_used_[di][(size_t) cand] < oldest) { oldest = loan_last_used_[di][(size_t) cand]; best = cand; if (oldest + 8 < tok) break; }
        }
        if (best < 0) return -1;
        loan_clock_[di] = (best + 1) % n;
        const auto [hl, he] = loan_holder_[di][(size_t) best];
        if (hl >= 0) {
            if (loan_apply_n_[(size_t) hl] >= kLoanApplyMax - 1) return -1;   // its layer's update list is full: no eviction now
            layers_[(size_t) hl].loan_of[(size_t) he] = -1;
            layers_[(size_t) hl].loan_live[(size_t) he] = 0;   // (its device entry is cleared at its next grouping, before any use)
            loan_push(hl, he, -1);
        }
        j = best;
    }
    loan_holder_[di][(size_t) j] = {il, e};
    loan_last_used_[di][(size_t) j] = tok;
    loan_busy_[di][(size_t) j] = 1;
    layers_[(size_t) il].loan_of[(size_t) e] = j;
    layers_[(size_t) il].loan_hits_n[(size_t) e] = 0;
    return j;
}

void ExpertStore::loan_push(int il, int e, int j) {
    // (called under loan_m_) the immediate list: evictions (j < 0) and landed additions
    int& n = loan_apply_n_[(size_t) il];
    if (n >= kLoanApplyMax) return;
    int32_t* buf = loan_apply_h_[loan_apply_side_[(size_t) il]][(size_t) il];
    buf[2 * n] = e; buf[2 * n + 1] = j;
    ++n;
}

void* ExpertStore::loan_event(size_t di) {
    std::vector<void*>& pool = loan_ev_pool_[di];
    if (!pool.empty()) { void* e = pool.back(); pool.pop_back(); return e; }
    int cur = 0; cudaGetDevice(&cur);
    if (cur != devices_[di]) check(cudaSetDevice(devices_[di]), "cudaSetDevice");
    cudaEvent_t ev; check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "loan event create");
    if (cur != devices_[di]) check(cudaSetDevice(cur), "cudaSetDevice");
    return ev;
}
void ExpertStore::loan_event_done(size_t di, void* ev) { loan_ev_pool_[di].push_back(ev); }

void ExpertStore::apply_loans(int il, void* stream) {
    if (!loan_on_) return;
    int n; int32_t* src_dev;
    {
        std::lock_guard<std::mutex> lk(loan_m_);
        // the batches whose copies have landed (in order), their slots no longer busy
        std::deque<LoanBatch>& q = loan_batches_[(size_t) il];
        const size_t di = (size_t) device_index_of_layer(il);
        while (!q.empty() && cudaEventQuery(static_cast<cudaEvent_t>(q.front().ev)) == cudaSuccess) {
            LoanBatch& b = q.front();
            for (const auto& [e, j] : b.adds) {
                if (layers_[(size_t) il].loan_of[(size_t) e] == j) { loan_push(il, e, j); layers_[(size_t) il].loan_live[(size_t) e] = 1; }   // live from this grouping on
                loan_busy_[di][(size_t) j] = 0;
            }
            loan_event_done(b.di, b.ev);
            q.pop_front();
        }
        n = loan_apply_n_[(size_t) il];
        if (n == 0) return;
        src_dev = loan_apply_dev_[loan_apply_side_[(size_t) il]][(size_t) il];
        loan_apply_side_[(size_t) il] ^= 1;      // the host fills the other buffer from now on
        loan_apply_n_[(size_t) il] = 0;
    }
    strata::glm5::loan_apply(src_dev, n, layers_[(size_t) il].slot_of_dev, stream);   // reads the mapped entries
}

void ExpertStore::reset_loans() {
    if (!loan_on_) return;
    std::lock_guard<std::mutex> lk(loan_m_);
    bool any_loan = false;
    for (size_t d = 0; d < devices_.size() && !any_loan; ++d) any_loan = loan_free_[d].size() != loan_holder_[d].size();
    for (size_t il = 0; il < layers_.size() && !any_loan; ++il) any_loan = loan_apply_n_[il] != 0 || !loan_batches_[il].empty();
    if (!any_loan) return;
    for (size_t d = 0; d < devices_.size(); ++d) {
        check(cudaSetDevice(devices_[d]), "cudaSetDevice");
        check(cudaStreamSynchronize(static_cast<cudaStream_t>(copy_streams_[d])), "loan reset sync");
        check(cudaStreamSynchronize(static_cast<cudaStream_t>(loan_streams_[d])), "loan reset sync");
        const int n = (int) loan_holder_[d].size();
        loan_holder_[d].assign((size_t) n, {-1, -1}); loan_last_used_[d].assign((size_t) n, 0u); loan_busy_[d].assign((size_t) n, 0);
        loan_free_[d].clear();
        for (int j = n - 1; j >= 0; --j) loan_free_[d].push_back(j);
        loan_clock_[d] = 0;
    }
    for (size_t il = 0; il < layers_.size(); ++il) {
        LayerExperts& E = layers_[il];
        if (E.device < 0 || E.loan_of.empty()) continue;
        bool any = loan_apply_n_[il] > 0;   // entries not yet applied (a promotion's, an eviction's): the table from the host mirror
        for (auto& v : E.loan_of) if (v >= 0) { v = -1; any = true; }
        for (auto& v : E.loan_live) v = 0;
        for (auto& v : E.loan_hits_n) v = 0;
        loan_apply_n_[il] = 0;
        for (auto& b : loan_batches_[il]) loan_event_done(b.di, b.ev);
        loan_batches_[il].clear();
        if (any) {   // the device table back to the resident slots (the host mirror of slot_of is current)
            check(cudaSetDevice(E.device), "cudaSetDevice");
            check(cudaMemcpy(E.slot_of_dev, E.slot_of.data(), E.slot_of.size() * 4, cudaMemcpyHostToDevice), "loan reset table");
        }
    }
}

int ExpertStore::cpu_run(int il, const std::vector<Miss>& misses, int T, const float* x_rows, float* y_dev, int n_embd, uint32_t tok, uint32_t* h_flag2,
                         uint32_t* flag2_dev, void* cs_, std::chrono::steady_clock::time_point& tl) {
    LayerExperts& E = layers_[(size_t) il];
    const strata::kernels::cpu::NativeFmt& f = fmt_[(size_t) il];
    cudaStream_t cs = static_cast<cudaStream_t>(cs_);
    auto lap = [&](int i, std::chrono::steady_clock::time_point& t) { const auto n = std::chrono::steady_clock::now(); cpu_t_[i] += std::chrono::duration<double, std::milli>(n - t).count(); t = n; };
    // the tokens' inputs quantized for the gate/up dot products
    for (int t = 0; t < T; ++t) strata::kernels::cpu::native_quant_act(f, x_rows + (size_t) t * n_embd, act_[(size_t) t].data());
    lap(1, tl);
    float* y_layer = y_pin_ + (size_t) ((il & 1) * y_rows_) * n_embd;   // row r of this layer's pinned block = ff row r
    const auto* tr = ggml_get_type_traits_cpu((ggml_type) f.gu_type);
    const float limit = clamp_limit_[(size_t) il];
    int n_pairs = 0;
    // phase A: ff[pair] = swiglu_clamp(gate . a, up . a) for the rows of each expert, in 128-row chunks
    std::vector<std::function<void()>> jobs;
    const int n_ff = (int) f.n_ff;
    for (const Miss& ms : misses) {
        const uint8_t* blob = E.host_gate[(size_t) ms.e];
        n_pairs += (int) ms.toks.size();
        for (int r0 = 0; r0 < n_ff; r0 += 128) {
            const int r1 = std::min(n_ff, r0 + 128);
            jobs.emplace_back([this, &ms, blob, r0, r1, tr, &f, limit]() {
                for (size_t i = 0; i < ms.toks.size(); ++i) {
                    const void* a = act_[(size_t) ms.toks[i]].data();
                    float* ff = ff_[(size_t) (ms.ff0 + (int) i)].data();
                    for (int r = r0; r < r1; ++r) {
                        float g = 0.f, u = 0.f;
                        tr->vec_dot((int) f.n_embd, &g, 0, blob + (size_t) r * f.gu_row, 0, a, 0, 1);
                        tr->vec_dot((int) f.n_embd, &u, 0, blob + f.up_off + (size_t) r * f.gu_row, 0, a, 0, 1);
                        if (limit > 0.f) { g = std::min(g, limit); u = std::min(std::max(u, -limit), limit); }
                        ff[r] = g / (1.f + std::exp(-g)) * u;
                    }
                }
            });
        }
    }
    pool_.run(std::move(jobs));
    lap(2, tl);
    // the swiglu outputs quantized for the down dot products
    for (const Miss& ms : misses)
        for (size_t i = 0; i < ms.toks.size(); ++i) {
            const size_t pr = (size_t) (ms.ff0 + (int) i);
            strata::kernels::cpu::native_quant_h(f, ff_[pr].data(), hq_[pr].data());
        }
    lap(3, tl);
    // phase B: y[pair] = down . hq, in 512-row chunks
    jobs.clear();
    for (const Miss& ms : misses) {
        const uint8_t* blob = E.host_gate[(size_t) ms.e];
        for (int r0 = 0; r0 < n_embd; r0 += 512) {
            const int r1 = std::min(n_embd, r0 + 512);
            jobs.emplace_back([this, &ms, blob, r0, r1, &f, y_layer, n_embd]() {
                std::vector<const void*> hq(ms.toks.size());
                std::vector<float*> out(ms.toks.size());
                for (size_t i = 0; i < ms.toks.size(); ++i) {
                    const size_t pr = (size_t) (ms.ff0 + (int) i);
                    hq[i] = hq_[pr].data();
                    out[i] = y_layer + pr * n_embd;
                }
                strata::kernels::cpu::native_down_rows(f, blob, hq.data(), (int) ms.toks.size(), out.data(), r0, r1);
            });
        }
    }
    pool_.run(std::move(jobs));
    lap(4, tl);
    // the CPU-served pairs' rows to the device (per expert: a contiguous run of rows), then the flag
    if (y_pin_dev_ && !misses.empty()) {
        // one kernel reads the whole layer's rows from the mapped pinned block (a kernel launch needs the device current)
        int cur = 0; cudaGetDevice(&cur);
        if (cur != E.device) check(cudaSetDevice(E.device), "cudaSetDevice");
        const float* src = y_pin_dev_ + (size_t) ((il & 1) * y_rows_) * n_embd;
        if (copy_desc_dev_ && misses.size() <= (size_t) kCopyDescMax) {   // one launch: the served experts' row runs from a descriptor list
            int32_t* desc = copy_desc_ + (size_t) (il & 1) * kCopyDescMax * 3;
            for (size_t i = 0; i < misses.size(); ++i) { desc[3 * i] = misses[i].ff0; desc[3 * i + 1] = misses[i].row0; desc[3 * i + 2] = (int32_t) misses[i].toks.size(); }
            strata::glm5::copy_host_rows_multi(src, y_dev, copy_desc_dev_ + (size_t) (il & 1) * kCopyDescMax * 3, (int) misses.size(), n_embd, cs);
        } else
        for (const Miss& ms : misses)
            strata::glm5::copy_host_rows(src + (size_t) ms.ff0 * n_embd, y_dev + (size_t) ms.row0 * n_embd, ms.toks.size() * (size_t) n_embd, cs);
        if (cur != E.device) check(cudaSetDevice(cur), "cudaSetDevice");
    } else
    for (const Miss& ms : misses)
        check(cudaMemcpyAsync(y_dev + (size_t) ms.row0 * n_embd, y_layer + (size_t) ms.ff0 * n_embd, ms.toks.size() * (size_t) n_embd * 4,
                              cudaMemcpyHostToDevice, cs), "cpu y h2d");
    if (misses.empty()) __atomic_store_n(h_flag2, tok, __ATOMIC_RELEASE);
    else if (flag2_dev) flag_kernel(E.device, flag2_dev, tok, cs);
    lap(5, tl);
    return n_pairs;
}

int ExpertStore::serve_mid(int il, const int32_t* ids, int n, int T, const float* x_rows, uint32_t tok, uint32_t* h_flag, uint32_t* flag_dev, uint32_t* h_flag2,
                           uint32_t* flag2_dev, float* y_dev, int n_embd, int32_t* skip_h, uint8_t* ring_dst, uint64_t stride, void* wait_event) {
    LayerExperts& E = layers_[(size_t) il];
    if (route_log_) { std::fprintf(route_log_, "T %u %d", tok, il); for (int j2 = 0; j2 < n; ++j2) std::fprintf(route_log_, " %d", ids[j2]); std::fprintf(route_log_, "\n"); }
    const size_t di = (size_t) device_index_of_layer(il);
    cudaStream_t cs = static_cast<cudaStream_t>(copy_streams_[di]);
    const int K = n / T, ne = (int) E.slot_of.size();
    // the ranks of the non-resident experts (k_moe_group's order for the ring), the counts per expert
    if (nonres_rank_.empty()) nonres_rank_.resize(layers_.size());
    std::vector<int32_t>& rank = nonres_rank_[(size_t) il];
    rank.assign((size_t) ne, -1);
    int n_nonres = 0;
    for (int e = 0; e < ne; ++e) if (E.slot_of[(size_t) e] < 0) rank[(size_t) e] = n_nonres++;
    std::vector<int> count((size_t) ne, 0);
    for (int j = 0; j < n; ++j) {
        const int e = ids[j];
        if (e < 0 || e >= ne) throw std::runtime_error("glm5 experts: a routing id out of range (corrupt activations)");
        ++counts_[(size_t) il][(size_t) e]; ++count[(size_t) e];
        if (E.slot_of[(size_t) e] >= 0) ++hits_; else ++misses_;
    }
    for (int r = 0; r < n_nonres; ++r) skip_h[r] = 0;
    // the MMQ layout: the resident experts' pairs first (slot order), then the non-resident ones' by rank
    int col = 0;
    for (int sl = 0; sl < E.n_slots; ++sl) col += count[(size_t) E.expert_of[(size_t) sl]];
    std::vector<int> miss_e, col0((size_t) ne, -1);
    for (int e = 0; e < ne; ++e) if (E.slot_of[(size_t) e] < 0 && count[(size_t) e]) { col0[(size_t) e] = col; col += count[(size_t) e]; miss_e.push_back(e); }
    distinct_misses_ += miss_e.size();
    if (!miss_e.empty()) ++layers_missed_;
    if (miss_e.empty()) { __atomic_store_n(h_flag, tok, __ATOMIC_RELEASE); __atomic_store_n(h_flag2, tok, __ATOMIC_RELEASE); return 0; }
    const auto t_cpu0 = std::chrono::steady_clock::now();
    auto tl = t_cpu0;
    ++cpu_calls_;
    // the split (as serve_cpu): every miss starts on the CPU, the experts with the most pairs move to PCIe while that
    // shortens the longer side; then the CPU's rows capped at kMidCpuRows (the largest groups move over first)
    std::vector<int> pairs_of((size_t) miss_e.size());
    for (size_t i = 0; i < miss_e.size(); ++i) pairs_of[i] = count[(size_t) miss_e[i]];
    // only the Q2_K layers' products honor the skip mask (the WMMA kernel); the other layers' experts all go over PCIe
    const bool cpu_ok = E.gu_type == 10 && E.d_type == 10;
    std::vector<char> on_cpu(miss_e.size(), cpu_ok ? 1 : 0);
    double cpu_t = 0, dma_t = 0;
    for (size_t i = 0; i < miss_e.size(); ++i) if (on_cpu[i]) cpu_t += pairs_of[i] * cpu_pair_ms_; else dma_t += dma_ms_;
    auto move_largest = [&]() {
        int best = -1;
        for (size_t i = 0; i < miss_e.size(); ++i) if (on_cpu[i] && (best < 0 || pairs_of[i] > pairs_of[(size_t) best])) best = (int) i;
        if (best < 0) return false;
        on_cpu[(size_t) best] = 0; cpu_t -= pairs_of[(size_t) best] * cpu_pair_ms_; dma_t += dma_ms_;
        return true;
    };
    for (;;) {
        int best = -1;
        for (size_t i = 0; i < miss_e.size(); ++i) if (on_cpu[i] && (best < 0 || pairs_of[i] > pairs_of[(size_t) best])) best = (int) i;
        if (best < 0) break;
        if (std::max(cpu_t - pairs_of[(size_t) best] * cpu_pair_ms_, dma_t + dma_ms_) >= std::max(cpu_t, dma_t)) break;
        move_largest();
    }
    for (;;) {
        int rows = 0;
        for (size_t i = 0; i < miss_e.size(); ++i) if (on_cpu[i]) rows += pairs_of[i];
        if (rows <= kMidCpuRows || !move_largest()) break;
    }
    // the PCIe side into the ring (after the buffer's last reader), its flag behind the copies
    int n_dma = 0;
    if (wait_event) check(cudaStreamWaitEvent(cs, static_cast<cudaEvent_t>(wait_event), 0), "ring wait");
    for (size_t i = 0; i < miss_e.size(); ++i) {
        if (on_cpu[i]) continue;
        const int e = miss_e[i];
        if (!E.host_pinned[(size_t) e]) throw std::runtime_error("serve_mid: an unpinned expert (the pin limit)");
        check(cudaMemcpyAsync(ring_dst + (uint64_t) rank[(size_t) e] * stride, E.host_gate[(size_t) e], E.blob, cudaMemcpyHostToDevice, cs), "ring dma");
        ++n_dma;
    }
    prefetched_ += (uint64_t) n_dma;
    std::vector<Miss> misses;
    int ff0 = 0;
    for (size_t i = 0; i < miss_e.size(); ++i) {
        if (!on_cpu[i]) continue;
        const int e = miss_e[i];
        if (!E.host_pinned[(size_t) e]) throw std::runtime_error("serve_mid: an unpinned expert (the pin limit)");
        skip_h[rank[(size_t) e]] = 1;
        Miss ms{e, {}, col0[(size_t) e], ff0};
        for (int j = 0; j < n; ++j) if (ids[j] == e) ms.toks.push_back(j / K);
        ff0 += (int) ms.toks.size();
        misses.push_back(std::move(ms));
    }
    if (n_dma > 0) { if (flag_dev) flag_kernel(E.device, flag_dev, tok, cs); } else __atomic_store_n(h_flag, tok, __ATOMIC_RELEASE);
    cpu_t_[0] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tl).count(); tl = std::chrono::steady_clock::now();
    const int n_cpu_pairs = cpu_run(il, misses, T, x_rows, y_dev, n_embd, tok, h_flag2, flag2_dev, cs, tl);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_cpu0).count();
    cpu_ms_ += ms;
    cpu_pairs_ += (uint64_t) n_cpu_pairs;
    if (n_cpu_pairs > 0 && !cpu_pair_fixed_) cpu_pair_ms_ = 0.9 * cpu_pair_ms_ + 0.1 * (ms / n_cpu_pairs);
    return (int) miss_e.size();
}

int ExpertStore::serve_cpu(int il, const int32_t* ids, int n, int T, const float* x_rows, uint32_t tok, uint32_t* h_flag, uint32_t* flag_dev, uint32_t* h_flag2,
                           uint32_t* flag2_dev, float* y_dev, int n_embd, int* miss_cpu, int* miss_slot, int* miss_out, int max_swaps) {
    LayerExperts& E = layers_[(size_t) il];
    if (route_log_) { std::fprintf(route_log_, "T %u %d", tok, il); for (int j2 = 0; j2 < n; ++j2) std::fprintf(route_log_, " %d", ids[j2]); std::fprintf(route_log_, "\n"); }
    const strata::kernels::cpu::NativeFmt& f = fmt_[(size_t) il];
    const size_t di = (size_t) device_index_of_layer(il);
    cudaStream_t cs = static_cast<cudaStream_t>(copy_streams_[di]);
    const int K = n / T;
    const int ne = (int) E.slot_of.size();
    // k_moe_group's pair layout: counts per expert; the hits' groups first (ascending id), then the misses' (ascending);
    // inside a group the pairs in token order
    std::vector<int> count((size_t) ne, 0);
    auto resident = [&](int e) { return E.slot_of[(size_t) e] >= 0 || (loan_on_ && E.loan_live[(size_t) e]); };   // live = in the device's table too
    std::vector<char> promoted((size_t) ne, 0);   // a loan hit's promotion tried once per window
    int n_promoted = 0, n_promo_calls = 0;
    for (int j = 0; j < n; ++j) {
        const int e = ids[j];
        if (e < 0 || e >= ne) {   // a routing id outside the experts: the router saw NaN / garbage activations
            std::fprintf(stderr, "glm5 experts: layer %d window %u: routing id %d of %d (pair %d of %d) is out of range: the activations are corrupt\n", il, tok, e, ne, j, n);
            throw std::runtime_error("glm5 experts: a routing id out of range (corrupt activations)");
        }
        ++counts_[(size_t) il][(size_t) e]; ++count[(size_t) e];
        if (resident(e)) {
            ++hits_;
            if (E.slot_of[(size_t) e] < 0) {
                ++loan_hits_;
                if (!promoted[(size_t) e]) {
                    promoted[(size_t) e] = 1;
                    uint8_t& hn = E.loan_hits_n[(size_t) e];
                    if (hn < 255) ++hn;
                    const bool want = promote_hits_ > 0 && hn >= promote_hits_ && n_promo_calls < promote_max_;
                    if (want && promote_loan(E, il, di, e, tok, max_swaps, n_promoted, cs)) ++n_promo_calls;
                    else loan_last_used_[di][(size_t) E.loan_of[(size_t) e]] = tok;
                }
            }
        } else ++misses_;
    }
    int col = 0;
    for (int e = 0; e < ne; ++e) if (count[(size_t) e] && resident(e)) col += count[(size_t) e];
    const int first_miss_row = col;
    std::vector<int> miss_e;
    std::vector<int> col0((size_t) ne, -1);
    for (int e = 0; e < ne; ++e) if (count[(size_t) e] && !resident(e)) { col0[(size_t) e] = col; col += count[(size_t) e]; miss_e.push_back(e); }
    const int n_pairs = col, n_miss_rows = col - first_miss_row;
    distinct_misses_ += miss_e.size();
    if (!miss_e.empty()) ++layers_missed_;
    for (int e : miss_e) if (E.evicted_once[(size_t) e]) ++thrash_;
    if (miss_e.empty()) { __atomic_store_n(h_flag, tok, __ATOMIC_RELEASE); __atomic_store_n(h_flag2, tok, __ATOMIC_RELEASE); return 0; }
    const auto t_cpu0 = std::chrono::steady_clock::now();
    auto lap = [&](int i, std::chrono::steady_clock::time_point& t) { const auto n = std::chrono::steady_clock::now(); cpu_t_[i] += std::chrono::duration<double, std::milli>(n - t).count(); t = n; };
    auto tl = t_cpu0;
    ++cpu_calls_;
    // the split: every miss starts on the CPU; the experts with the most pairs move to PCIe one by one while that
    // shortens the longer of the two (the CPU's time falls by pairs x cpu_pair_ms, PCIe grows by dma_ms per expert)
    std::vector<int> pairs_of((size_t) miss_e.size());
    for (size_t i = 0; i < miss_e.size(); ++i) pairs_of[i] = count[(size_t) miss_e[i]];
    std::vector<char> on_cpu(miss_e.size(), 1);
    // the misses the lookahead staged already: on the PCIe side at no cost (a staging-to-staging copy)
    std::vector<int> staged_at(miss_e.size(), -1);
    {
        std::vector<std::pair<int, int>>& pf = pf_stage_[(size_t) il];
        for (size_t i = 0; i < miss_e.size(); ++i)
            for (const auto& q : pf) if (q.first == miss_e[i]) { staged_at[i] = q.second; on_cpu[i] = 0; ++pf_hits_; break; }
        pf.clear();
    }
    {
        // the device waits max(DMA time - its work before that wait, CPU time - its work before the combine, 0)
        auto wait_of = [&](double cpu_t, double dma_t) { return std::max({dma_t - gpu_before_dma_ms_, cpu_t - gpu_before_cpu_ms_, 0.0}); };
        double cpu_t = 0;
        for (size_t i = 0; i < miss_e.size(); ++i) if (on_cpu[i]) cpu_t += pairs_of[i] * cpu_pair_ms_;
        double dma_t = 0;
        for (;;) {
            int best = -1;
            // the most pairs first (the CPU time they save); among equals the expert routed most often so far (the
            // PCIe side's blob becomes a loan: a frequent expert's loan is the one that gets hit again)
            for (size_t i = 0; i < miss_e.size(); ++i) {
                if (!on_cpu[i]) continue;
                if (best < 0 || pairs_of[i] > pairs_of[(size_t) best] ||
                    (pairs_of[i] == pairs_of[(size_t) best] && counts_[(size_t) il][(size_t) miss_e[i]] > counts_[(size_t) il][(size_t) miss_e[(size_t) best]]))
                    best = (int) i;
            }
            if (best < 0) break;
            const double cpu_after = cpu_t - pairs_of[(size_t) best] * cpu_pair_ms_, dma_after = dma_t + dma_ms_;
            if (wait_of(cpu_after, dma_after) >= wait_of(cpu_t, dma_t)) break;
            on_cpu[(size_t) best] = 0; cpu_t = cpu_after; dma_t = dma_after;
        }
    }
    int n_dma = 0, n_cpu_pairs = 0, swapped = n_promoted;   // (the promotions count against the window's swap budget)
    std::vector<std::function<void()>> dma_jobs;
    std::vector<int> dma_staged;
    for (size_t i = 0; i < miss_e.size(); ++i) {
        miss_cpu[i] = on_cpu[i] ? 1 : 0;
        if (!on_cpu[i] && staged_at[i] >= 0) {
            ++n_dma;   // staged by the lookahead: into this miss's staging slot, no residency change
            check(cudaMemcpyAsync(stage_[di] + i * stage_slot_bytes_[di], stage_[di] + (size_t) staged_at[i] * stage_slot_bytes_[di], E.blob, cudaMemcpyDeviceToDevice, cs), "staged d2d");
            miss_slot[i] = -1;
            miss_out[i] = -1;
        } else if (!on_cpu[i]) {
            ++n_dma;
            // the PCIe side goes through the swap-in machinery (an admitted expert becomes resident; else staging slot i)
            dma_miss(E, il, di, i, miss_e[i], tok, miss_slot, miss_out, max_swaps, swapped, dma_jobs, dma_staged, cs);
        } else {
            miss_slot[i] = -1;
            miss_out[i] = -1;
            n_cpu_pairs += pairs_of[i];
        }
    }
    finish_staged(E, di, miss_slot, dma_jobs, dma_staged, cs);
    swaps_ += (uint64_t) swapped;
    // the DMA'd misses' flag, queued behind their copies: the device computes them while the CPU works
    if (n_dma > 0) { if (flag_dev) flag_kernel(E.device, flag_dev, tok, cs); } else __atomic_store_n(h_flag, tok, __ATOMIC_RELEASE);
    // the loans: a missed expert with a pinned host copy is also copied into a loan slot (CPU-served: from the host; PCIe
    // staged: a device-to-device copy from its staging slot), usable from this layer's next window
    if (loan_on_ && E.stride <= loan_stride_) {
        cudaStream_t ls = static_cast<cudaStream_t>(loan_streams_[di]);
        LoanBatch batch{nullptr, di, {}};
        bool staged_any = false;
        for (size_t i = 0; i < miss_e.size() && (int) batch.adds.size() < loan_per_win_; ++i) {
            const int e = miss_e[i];
            const bool staged = !on_cpu[i] && miss_slot[i] < 0 && staged_at[i] < 0;
            if (!on_cpu[i] && !staged) continue;                    // swapped into a resident slot: resident already
            if (on_cpu[i] && (!loan_h2d_ || !E.host_pinned[(size_t) e] || E.host_gate[(size_t) e] == nullptr)) continue;
            int j;
            { std::lock_guard<std::mutex> lk(loan_m_); j = loan_alloc(di, il, e, tok); }
            if (j < 0) break;
            uint8_t* dst = loan_ptr_h_[di][(size_t) j];
            if (on_cpu[i]) check(cudaMemcpyAsync(dst, E.host_gate[(size_t) e], E.blob, cudaMemcpyHostToDevice, ls), "loan h2d");
            else {
                if (!staged_any) {   // the staging copies are on the copy stream: order the loan stream behind them once
                    std::lock_guard<std::mutex> lk(loan_m_);
                    cudaEvent_t se = static_cast<cudaEvent_t>(loan_event(di));
                    check(cudaEventRecord(se, cs), "loan stage event"); check(cudaStreamWaitEvent(ls, se, 0), "loan stage wait");
                    loan_event_done(di, se); staged_any = true;
                }
                if (!kLoanMemcpy && E.blob % 16 == 0) copy_dev_kernel(E.device, dst, stage_[di] + i * stage_slot_bytes_[di], E.blob, ls);
                else check(cudaMemcpyAsync(dst, stage_[di] + i * stage_slot_bytes_[di], E.blob, cudaMemcpyDeviceToDevice, ls), "loan d2d");
            }
            copy_bytes_ += E.blob;
            batch.adds.emplace_back(e, j);
        }
        if (!batch.adds.empty()) {
            std::lock_guard<std::mutex> lk(loan_m_);
            batch.ev = loan_event(di);
            check(cudaEventRecord(static_cast<cudaEvent_t>(batch.ev), ls), "loan batch event");
            loans_made_ += batch.adds.size();
            loan_batches_[(size_t) il].push_back(std::move(batch));
        }
    }
    lap(0, tl);
    std::vector<Miss> misses;
    for (size_t i = 0; i < miss_e.size(); ++i) {
        if (!on_cpu[i]) continue;
        const int e = miss_e[i];
        Miss ms{e, {}, col0[(size_t) e], col0[(size_t) e] - first_miss_row};
        for (int j = 0; j < n; ++j) if (ids[j] == e) ms.toks.push_back(j / K);
        misses.push_back(std::move(ms));
    }
    cpu_run(il, misses, T, x_rows, y_dev, n_embd, tok, h_flag2, flag2_dev, cs, tl);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_cpu0).count();
    cpu_ms_ += ms;
    cpu_pairs_ += (uint64_t) n_cpu_pairs;
    if (n_cpu_pairs > 0 && !cpu_pair_fixed_) cpu_pair_ms_ = 0.9 * cpu_pair_ms_ + 0.1 * (ms / n_cpu_pairs);   // the split follows the machine's current CPU speed
    (void) n_pairs; (void) n_miss_rows; (void) n_dma;
    return (int) miss_e.size();
}


int ExpertStore::serve_ring(int il, const int32_t* ids, int n, uint32_t tok, uint32_t* h_flag, uint32_t* flag_dev, uint8_t* dst, uint64_t stride,
                            void* wait_event) {
    LayerExperts& E = layers_[(size_t) il];
    const size_t di = (size_t) device_index_of_layer(il);
    cudaStream_t cs = static_cast<cudaStream_t>(copy_streams_[di]);
    // the ranks of this layer's non-resident experts now (the decode windows' swaps change the set): as k_moe_group and
    // prefetch_all number them, ascending id
    if (nonres_rank_.empty()) nonres_rank_.resize(layers_.size());
    std::vector<int32_t>& rank = nonres_rank_[(size_t) il];
    rank.assign(E.slot_of.size(), -1);
    { int r = 0; for (size_t e = 0; e < E.slot_of.size(); ++e) if (E.slot_of[e] < 0) rank[e] = r++; }
    std::vector<int> missed;
    for (int j = 0; j < n; ++j) {
        const int e = ids[j];
        ++counts_[(size_t) il][(size_t) e];
        if (E.slot_of[(size_t) e] >= 0) { ++hits_; continue; }
        ++misses_;
        if (seen_[(size_t) e]) continue;
        seen_[(size_t) e] = 1;
        missed.push_back(e);
    }
    for (int e : missed) seen_[(size_t) e] = 0;
    distinct_misses_ += missed.size();
    if (!missed.empty()) ++layers_missed_;
    if (missed.empty()) { __atomic_store_n(h_flag, tok, __ATOMIC_RELEASE); return 0; }
    if (wait_event) check(cudaStreamWaitEvent(cs, static_cast<cudaEvent_t>(wait_event), 0), "ring wait");
    for (int e : missed) {
        if (!E.host_pinned[(size_t) e]) throw std::runtime_error("serve_ring: an unpinned expert (the pin limit)");
        check(cudaMemcpyAsync(dst + (uint64_t) rank[(size_t) e] * stride, E.host_gate[(size_t) e], E.blob, cudaMemcpyHostToDevice, cs), "ring dma");
    }
    prefetched_ += (uint64_t) missed.size();
    if (flag_dev) flag_kernel(E.device, flag_dev, tok, cs);
    return (int) missed.size();
}

int ExpertStore::prefetch_all(int il, uint32_t tok, uint32_t* h_flag, uint32_t* flag_dev, uint8_t* dst, uint64_t stride, void* wait_event) {
    LayerExperts& E = layers_[(size_t) il];
    const size_t di = (size_t) device_index_of_layer(il);
    cudaStream_t cs = static_cast<cudaStream_t>(copy_streams_[di]);
    if (wait_event) check(cudaStreamWaitEvent(cs, static_cast<cudaEvent_t>(wait_event), 0), "prefetch wait");
    std::vector<int> unpinned, unpinned_rank;
    int n = 0;
    for (int e = 0; e < (int) E.slot_of.size(); ++e) {
        if (E.slot_of[(size_t) e] >= 0) continue;
        const int rank = n++;   // compact: the expert's index among the non-resident ones (k_moe_group numbers them the same)
        if (E.host_pinned[(size_t) e]) check(cudaMemcpyAsync(dst + (uint64_t) rank * stride, E.host_gate[(size_t) e], E.blob, cudaMemcpyHostToDevice, cs), "prefetch dma");
        else { unpinned.push_back(e); unpinned_rank.push_back(rank); }
    }
    if (!unpinned.empty()) {
        // no pinned copy (the pin limit): through the pinned staging twin, which earlier DMAs may still be reading
        if ((int) unpinned.size() > k_max_) throw std::runtime_error("prefetch_all: more unpinned experts than staging slots");
        check(cudaStreamSynchronize(cs), "prefetch sync");
        std::vector<std::function<void()>> jobs;
        for (size_t i = 0; i < unpinned.size(); ++i) {
            uint8_t* h = stage_host_[di] + i * stage_slot_bytes_[di];
            const int e = unpinned[i];
            jobs.emplace_back([this, &E, e, h]() { read_blob(E, e, h); });
        }
        const auto t0 = std::chrono::steady_clock::now();
        pool_.run(std::move(jobs));
        file_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        file_reads_ += unpinned.size();
        file_bytes_ += (uint64_t) unpinned.size() * E.blob;
        for (size_t i = 0; i < unpinned.size(); ++i)
            check(cudaMemcpyAsync(dst + (uint64_t) unpinned_rank[i] * stride, stage_host_[di] + i * stage_slot_bytes_[di], E.blob, cudaMemcpyHostToDevice, cs), "prefetch dma");
    }
    prefetched_ += (uint64_t) n;
    if (n == 0) {
        __atomic_store_n(h_flag, tok, __ATOMIC_RELEASE);
    } else {
        if (flag_dev) flag_kernel(E.device, flag_dev, tok, cs);
    }
    return n;
}

void ExpertStore::collect_timing() {
    for (size_t i = 0; i < timing_events_.size(); ++i) {
        cudaEvent_t a = static_cast<cudaEvent_t>(timing_events_[i].first), b = static_cast<cudaEvent_t>(timing_events_[i].second);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        copy_ms_ += ms;
        copy_bytes_ += timing_bytes_[i];
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    timing_events_.clear();
    timing_bytes_.clear();
}

void ExpertStore::join(int il, void* stream) {
    const size_t di = (size_t) device_index_of_layer(il);
    check(cudaEventRecord(static_cast<cudaEvent_t>(copy_events_[di]), static_cast<cudaStream_t>(copy_streams_[di])), "copy event record");
    check(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), static_cast<cudaEvent_t>(copy_events_[di]), 0), "copy join");
}

bool ExpertStore::save_profile(const std::string& path, std::string& err) const {
    std::ofstream f(path);
    if (!f) { err = "cannot write " + path; return false; }
    f << "# glm5 routing counts: layer expert count\n";
    for (size_t il = 0; il < counts_.size(); ++il)
        for (size_t e = 0; e < counts_[il].size(); ++e)
            if (counts_[il][e]) f << il << ' ' << e << ' ' << counts_[il][e] << '\n';
    return true;
}

}  // namespace strata::glm5
