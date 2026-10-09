// src/program/glm5_main.cpp - `strata-glm5`, the GLM-5-Next driver (docs/GLM5_NEXT.md P3-P6).
//
//     strata-glm5 --gguf MODEL.gguf --tokens 785,6722,315 [--n-predict K] [--dump DIR] [--ctx N]
//                 [--devices 0,1] [--layer-split auto|L1[,L2]] [--expert-budget-gib G[,G2]]
//                 [--profile FILE] [--profile-save FILE] [--pin-gib G]
//                 [--window T] [--mtp FILE] [--spec K] [--prompts FILE]
//
// The prompt goes through the trunk in windows of --window tokens (the dense weights are read once per window).
// With --mtp, decoding is speculative: the NextN layer drafts --spec tokens, one trunk window checks them, the
// longest agreeing prefix plus the trunk's own next token are kept, the recurrent states roll back for the rest.
// The result is the greedy output, token for token.  Without --mtp, decoding is one token per window.
//
// --dump DIR writes every named intermediate (single-token windows) as DIR/s<step>_<name>[__k].npy in the layout
// tools/glm_oracle/dump_tensors.cpp uses with --ubatch 1, for tools/glm_oracle/compare_dumps.py.
#include "strata/glm5/glm5.hpp"
#include "strata/glm5/experts.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

void write_npy(const std::string& path, const void* host, const std::vector<int64_t>& shape, bool is_int) {
    std::string sh = "(";
    size_t n = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        sh += std::to_string(shape[i]) + (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
        n *= (size_t) shape[i];
    }
    sh += ")";
    std::string hdr = std::string("{'descr': '") + (is_int ? "<i4" : "<f4") + "', 'fortran_order': False, 'shape': " + sh + ", }";
    while ((10 + hdr.size() + 1) % 64 != 0) hdr += ' ';
    hdr += '\n';
    std::ofstream f(path, std::ios::binary);
    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    f.write((const char*) magic, 8);
    const uint16_t hl = (uint16_t) hdr.size();
    f.write((const char*) &hl, 2);
    f.write(hdr.data(), hdr.size());
    f.write((const char*) host, n * 4);
}

template <typename T>
std::vector<T> parse_list(const std::string& s, T (*conv)(const char*)) {
    std::vector<T> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) out.push_back(conv(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return out;
}
int to_int(const char* s) { return atoi(s); }
double to_double(const char* s) { return atof(s); }
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct SpecStats {
    int rounds = 0, accepted = 0, drafted = 0, generated = 0;
    double trunk_s = 0, mtp_s = 0;
};

// The drafts per round, chosen from the measured rate: per draft count k an EMA of the tokens a round produced and of
// its wall time; the next round uses the k with the best tokens/ms, one round in eight trying a neighbour so both
// estimates stay fresh (the third draft pays on some answers - TypeScript +5% - and costs on others - Rust -7%).
// STRATA_GLM5_SPEC_ADAPT=0 drafts the configured count every round.
struct SpecPolicy {
    int k_max = 0, round = 0, k_cur = 2;
    double tok[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, ms[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    int seen[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    static bool on() { static const bool v = !(std::getenv("STRATA_GLM5_SPEC_ADAPT") && atoi(std::getenv("STRATA_GLM5_SPEC_ADAPT")) == 0); return v; }
    double rate(int k) const { return ms[k] > 0 ? tok[k] / ms[k] : 0.0; }
    int choose() {
        if (!on() || k_max <= 2) return k_max;
        ++round;
        if (k_cur > k_max) k_cur = k_max;
        const int lo = std::max(1, k_cur - 1), hi = std::min(k_max, k_cur + 1);
        if (seen[hi] < 4 && hi != k_cur) return hi;              // an untried count: try it a few times first
        if (round % 8 == 0) return (round / 8) % 2 ? hi : lo;    // the exploratory round
        int best = k_cur;
        for (int k = lo; k <= hi; ++k) if (rate(k) > rate(best) * 1.02) best = k;   // a 2% margin against noise
        k_cur = best;
        return best;
    }
    void update(int k, int n, double round_ms) {   // k drafted, n accepted, the round's wall time
        if (k < 1 || k > 8 || round_ms <= 0) return;
        const double a = seen[k] < 10 ? 0.3 : 0.1;
        tok[k] += a * ((double) (n + 1) - tok[k]);
        ms[k] += a * (round_ms - ms[k]);
        ++seen[k];
    }
};

// The MTP catch-up over the window positions whose next token is known, then the drafts for the next round.
// h rows: the trunk's post-norm hidden states of the last window; next[i] = the token at position pos0 + i + 1.
// Returns the drafts (k of them, or fewer when an EOS appears).
std::vector<int32_t> mtp_draft(strata::glm5::Engine& eng, int slot, const float* h_rows, int n_embd, int pos0, const std::vector<int32_t>& next,
                               int k, SpecStats& st) {
    const double t0 = now_s();
    // the catch-up as one MTP window (its DSA / MoE batched over the tokens, like the trunk's window) on the slot's MTP caches
    const float* h = eng.mtp_window_slot(slot, h_rows, next.data(), (int) next.size(), pos0, true);
    std::vector<int32_t> drafts;
    int pos = pos0 + (int) next.size();
    for (int j = 0; j < k; ++j) {
        const int32_t d = eng.mtp_argmax();
        drafts.push_back(d);
        if (j + 1 < k) h = eng.mtp_step_slot(slot, h, d, pos++, true);
    }
    st.mtp_s += now_s() - t0;
    return drafts;
}


// ---------------------------------------------------------------- --serve: the line protocol serve/server.py speaks
// reads  GEN <max_new> [key=value ...] <id,id,...>   |  STOP  |  QUIT
// writes INFO ..., READY <ctx> stop, PP <read> <total> <ms> <tok/s>, RESUME <n>, T <id>, DONE ... / ERR <message>
struct StdinLines {
    std::deque<std::string> lines;
    std::mutex m;
    std::condition_variable cv;
    std::atomic<bool> stop{false}, quit{false};
    std::thread th;
    void start() {
        th = std::thread([this]() {
            std::string buf;
            char c;
            for (;;) {
                const ssize_t n = ::read(0, &c, 1);   // raw read: glibc's exit() takes stdin's lock
                if (n <= 0) { quit = true; cv.notify_all(); return; }
                if (c != '\n') { buf.push_back(c); continue; }
                if (buf == "STOP") { stop = true; buf.clear(); continue; }
                if (buf == "QUIT") { quit = true; cv.notify_all(); return; }
                { std::lock_guard<std::mutex> lk(m); lines.push_back(buf); }
                buf.clear();
                cv.notify_all();
            }
        });
        th.detach();
    }
    bool try_next(std::string& line) {   // a line if one is waiting (no wait)
        std::lock_guard<std::mutex> lk(m);
        if (lines.empty()) return false;
        line = std::move(lines.front());
        lines.pop_front();
        return true;
    }
    bool pending() { std::lock_guard<std::mutex> lk(m); return !lines.empty() || quit.load(); }
    bool next(std::string& line) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return quit.load() || !lines.empty(); });
        if (lines.empty()) return false;
        line = std::move(lines.front());
        lines.pop_front();
        return true;
    }
};

// one conversation's bookkeeping: a batch slot, or the solo conversation (slot n_slots)
struct SlotSt {
    bool active = false, stop = false;
    std::vector<int32_t> processed;   // the tokens the trunk has taken on this slot, in order (positions 0..pos-1)
    int32_t cur = -1;                 // the token to feed next (a batch slot)
    int64_t max_new = 0, produced = 0;
    double t0 = 0;
    std::vector<int32_t> drafts;      // the MTP's drafts for the tokens after `cur` (valid when mtp_ready)
    SpecPolicy pol;                   // the drafts per round for this slot
    bool mtp_ready = false;
    SpecStats st;
};
constexpr int kRowsMax = 8;   // the decode window: the KDA snapshots, the expert staging and the row tables are sized for 8 rows

int serve_loop(strata::glm5::Engine& eng, strata::glm5::ExpertStore* store, const strata::glm5::HParams& hp, int ctx, int window, int pwin, int spec,
               const std::vector<int32_t>& eos_ids, const std::string& profile_save) {
    StdinLines in;
    in.start();
    const int NS = eng.n_slots(), SOLO = eng.solo_slot();
    std::vector<SlotSt> slots((size_t) NS + 1);
    const int E = hp.n_embd;
    const bool use_mtp = eng.has_mtp() && spec > 0;
    std::printf("INFO engine=glm5 kv=q8_0 attn=%s expert_slots=%d%s\n", eng.lod_on() ? "lod" : "dsa", store ? store->resident_experts() : 0,
                NS > 0 ? (" batch_slots=" + std::to_string(NS) + " slot_cache=1").c_str() : "");
    std::printf("READY %d stop\n", ctx);
    std::fflush(stdout);
    auto is_eos = [&](int32_t tk) { return std::find(eos_ids.begin(), eos_ids.end(), tk) != eos_ids.end(); };
    auto split = [](const std::string& line) {
        std::vector<std::string> f;
        size_t i = 0;
        while (i < line.size()) { size_t j = line.find(' ', i); if (j == std::string::npos) j = line.size(); if (j > i) f.push_back(line.substr(i, j - i)); i = j + 1; }
        return f;
    };
    // the conversation cache across every slot: `target` continues after the longest prefix of the prompt any slot holds
    // (another slot's state is copied over), else after its own mark, else from scratch.  Returns the reused count.
    auto place = [&](int target, const std::vector<int32_t>& ids) -> size_t {
        int best = -1;
        size_t best_p = 0;
        for (int c = 0; c <= NS; ++c) {
            const std::vector<int32_t>& pr = slots[(size_t) c].processed;
            if (pr.empty() || pr.size() >= ids.size() || slots[(size_t) c].active) continue;
            if (std::equal(pr.begin(), pr.end(), ids.begin()) && pr.size() > best_p) { best = c; best_p = pr.size(); }
        }
        SlotSt& S0 = slots[(size_t) target];
        if (best >= 0 && best != target) {
            std::fprintf(stderr, "glm5: conversation cache: slot %d takes slot %d's %zu tokens\n", target, best, best_p);
            eng.copy_slot(best, target);
            S0.processed = slots[(size_t) best].processed;
            S0.mtp_ready = false;
            return best_p;
        }
        size_t p = 0;
        while (p < S0.processed.size() && p < ids.size() && S0.processed[p] == ids[p]) ++p;
        const size_t mk = eng.has_mark(target) ? (size_t) eng.mark_pos(target) : 0;
        if (p == S0.processed.size() && p < ids.size()) return p;   // the prompt continues what the slot holds
        if (mk > 0 && p >= mk && ids.size() > mk) {
            // the prompt extends the previous one but not its reply (the reasoning dropped from the history): back to the mark
            std::fprintf(stderr, "glm5: conversation cache: slot %d: the prompt diverges at token %zu of %zu (%zu in the prompt): back to the mark at %zu\n",
                         target, p, S0.processed.size(), ids.size(), mk);
            eng.restore_mark(target);
            S0.processed.resize(mk);
            S0.mtp_ready = false;
            if (use_mtp) eng.mtp_step_slot(target, eng.h_mark(target), ids[mk], (int) mk - 1, false);   // the MTP's entry at mk-1 took the old reply's first token
            return mk;
        }
        if (!S0.processed.empty())
            std::fprintf(stderr, "glm5: conversation cache: slot %d: the prompt diverges at token %zu of %zu (%zu in the prompt): reset\n", target, p, S0.processed.size(), ids.size());
        eng.reset(target);
        S0.processed.clear();
        S0.mtp_ready = false;
        return 0;
    };
    // one batch window: every active slot feeds its token and (the window's 8 rows shared out) its MTP drafts; the
    // accepted prefix of each slot's run is kept (the KDA states of the rest rolled back); BT lines, BDONE for the ones that end
    auto batch_step = [&]() {
        const double tb = now_s();
        struct Grp { int slot, start, len, pos0; std::vector<int32_t> next; };
        std::vector<int32_t> toks;
        std::vector<int> rows;
        std::vector<Grp> grps;
        int n_act = 0;
        for (int b = 0; b < NS; ++b) {
            SlotSt& S0 = slots[(size_t) b];
            if (!S0.active) continue;
            if (S0.stop) {   // ended before its next window: the slot holds everything but the last token (as the server expects)
                std::printf("BDONE %d %lld cancel %.1f\n", b, (long long) S0.produced, (now_s() - S0.t0) * 1e3);
                S0.active = false; S0.stop = false;
                continue;
            }
            ++n_act;
        }
        std::fflush(stdout);
        if (n_act == 0) return;
        const int k_max = use_mtp ? std::min(spec, (kRowsMax - n_act) / n_act) : 0;   // drafts per slot this window
        for (int b = 0; b < NS; ++b) {
            SlotSt& S0 = slots[(size_t) b];
            if (!S0.active) continue;
            Grp g{b, (int) toks.size(), 1, eng.pos(b), {}};
            toks.push_back(S0.cur); rows.push_back(b);
            const int k = S0.mtp_ready ? std::min<int>({k_max, (int) S0.drafts.size(), std::max(0, ctx - (int) S0.processed.size() - 2)}) : 0;
            for (int j = 0; j < k; ++j) { toks.push_back(S0.drafts[(size_t) j]); rows.push_back(b); ++g.len; }
            grps.push_back(std::move(g));
        }
        eng.forward_rows(toks.data(), rows.data(), (int) rows.size());
        const std::vector<int32_t> am = eng.argmax_rows();
        std::vector<int> keep(grps.size());
        for (size_t gi = 0; gi < grps.size(); ++gi) {
            Grp& g = grps[gi];
            SlotSt& S0 = slots[(size_t) g.slot];
            int n = 0;   // the accepted drafts
            while (n < g.len - 1 && toks[(size_t) (g.start + 1 + n)] == am[(size_t) (g.start + n)]) ++n;
            keep[gi] = n + 1;
            ++S0.st.rounds; S0.st.drafted += g.len - 1; S0.st.accepted += n;
            S0.pol.update(g.len - 1, n, (now_s() - tb) * 1e3);
            for (int i = 0; i <= n; ++i) S0.processed.push_back(toks[(size_t) (g.start + i)]);
            for (int i = 0; i < n; ++i) g.next.push_back(toks[(size_t) (g.start + 1 + i)]);
            g.next.push_back(am[(size_t) (g.start + n)]);
            for (int32_t y : g.next) {
                ++S0.produced;
                std::printf("BT %d %d\n", g.slot, (int) y);
                const char* fin = is_eos(y) ? "stop" : S0.produced >= S0.max_new ? "length" : nullptr;
                if (!fin && y == g.next.back() && (int) S0.processed.size() + 2 > ctx) fin = "length";
                if (fin) {
                    std::printf("BDONE %d %lld %s %.1f\n", g.slot, (long long) S0.produced, fin, (now_s() - S0.t0) * 1e3);
                    const double dt = now_s() - S0.t0;
                    std::fprintf(stderr, "glm5: slot %d done: %lld tokens in %.0f ms = %.1f tok/s, %s; spec %d/%d in %d rounds\n", g.slot, (long long) S0.produced, dt * 1e3,
                                 dt > 0 ? (double) S0.produced / dt : 0.0, fin, S0.st.accepted, S0.st.drafted, S0.st.rounds);
                    S0.active = false;
                    break;
                }
                S0.cur = y;
            }
        }
        std::fflush(stdout);
        eng.commit_rows(keep.data(), (int) grps.size());
        if (use_mtp)   // the MTP catch-up over each slot's kept rows, then its drafts for the next window
            for (const Grp& g : grps) {
                SlotSt& S0 = slots[(size_t) g.slot];
                if (!S0.active) continue;
                if (S0.pol.k_max == 0) S0.pol.k_max = spec;
                S0.drafts = mtp_draft(eng, g.slot, eng.h_win() + (size_t) g.start * E, E, g.pos0, g.next, S0.pol.choose(), S0.st);
                S0.mtp_ready = true;
            }
    };
    auto any_active = [&]() { for (int b = 0; b < NS; ++b) if (slots[(size_t) b].active) return true; return false; };
    // the prompt's new tokens through the trunk in equal windows on `target` (the solo slot: with the MTP catch-up);
    // returns the first token; between windows a batch window for the slots decoding meanwhile
    auto read_prompt = [&](int target, const std::vector<int32_t>& ids, size_t p, SpecStats& st) -> int32_t {
        SlotSt& S0 = slots[(size_t) target];
        const std::vector<int32_t> fresh(ids.begin() + (long) p, ids.end());
        const int wmax = any_active() ? std::min(pwin, 1024) : pwin;
        const size_t n_win = (fresh.size() + (size_t) wmax - 1) / (size_t) wmax, wsize = n_win ? (fresh.size() + n_win - 1) / n_win : 0;
        int32_t first = -1;
        if (!any_active() && eng.pipeline_ready() && n_win >= 2) {
            // no slot decoding meanwhile: the windows through the device pipeline (the PP lines and the MTP catch-up from
            // the last stage, in window order)
            const double ta0 = now_s();
            double t_prev = ta0;
            eng.prompt_windows(target, fresh.data(), (int) fresh.size(), wmax, [&](int pos0, int T, bool last) {
                const size_t off = (size_t) pos0 - p;
                if (last) first = eng.argmax_last();
                const double now = now_s(), dt = now - t_prev;
                t_prev = now;
                std::printf("PP %zu %zu %.1f %.1f\n", p + off + (size_t) T, ids.size(), dt * 1e3, T / dt);
                std::fflush(stdout);
                if (use_mtp) {
                    std::vector<int32_t> next;
                    for (int i = 0; i < T; ++i) { const size_t q = off + (size_t) i + 1; next.push_back(q < fresh.size() ? fresh[q] : first); }
                    S0.drafts = mtp_draft(eng, target, eng.h_win(), E, (int) (p + off), next, last ? spec : 0, st);
                    S0.mtp_ready = last;
                }
            });
            eng.mark((int) (fresh.size() - 1 - ((fresh.size() - 1) / wsize) * wsize));   // the last window's last row
            for (int32_t t : fresh) S0.processed.push_back(t);
            return first;
        }
        for (size_t off = 0; off < fresh.size(); off += wsize) {
            if (target == SOLO && in.stop) break;
            const int T = (int) std::min(wsize, fresh.size() - off);
            const double ta = now_s();
            if (target == SOLO) eng.forward_window(fresh.data() + off, T);
            else eng.forward_window_slot(target, fresh.data() + off, T);
            const bool last = off + (size_t) T == fresh.size();
            if (last) { first = eng.argmax_last(); eng.mark(T - 1); }
            for (int i = 0; i < T; ++i) S0.processed.push_back(fresh[off + (size_t) i]);
            const double dt = now_s() - ta;
            std::printf("PP %zu %zu %.1f %.1f\n", p + off + (size_t) T, ids.size(), dt * 1e3, T / dt);
            std::fflush(stdout);
            if (use_mtp) {
                std::vector<int32_t> next;
                for (int i = 0; i < T; ++i) { const size_t q = off + (size_t) i + 1; next.push_back(q < fresh.size() ? fresh[q] : first); }
                S0.drafts = mtp_draft(eng, target, eng.h_win(), E, (int) (p + off), next, last ? spec : 0, st);
                S0.mtp_ready = last;
            }
            if (!last && any_active()) batch_step();   // the slots decoding meanwhile get a window between the prompt's
        }
        return first;
    };
    std::string line;
    for (;;) {
        // between requests: the batch slots decode; a waiting line is taken between windows
        if (any_active() && !in.pending()) { batch_step(); continue; }
        if (!in.next(line)) break;
        std::vector<std::string> f = split(line);
        if (f.empty()) continue;
        if (f[0] == "BSTOP") { if (f.size() >= 2) { const int b = atoi(f[1].c_str()); if (b >= 0 && b < NS && slots[(size_t) b].active) slots[(size_t) b].stop = true; } continue; }
        if (f[0] == "BYIELD") continue;   // the prompt reads here are not interruptible
        const bool bgen = f[0] == "BGEN";
        if (!bgen && f[0] != "GEN") { std::printf("ERR unknown command\n"); std::fflush(stdout); continue; }
        int target = SOLO;
        size_t fi = 1;
        if (bgen) {
            if (f.size() < 4) { std::printf("ERR malformed BGEN\n"); std::fflush(stdout); continue; }
            target = atoi(f[1].c_str());
            if (target < 0 || target >= NS || slots[(size_t) target].active) { std::printf("ERR BGEN: no such free slot (--slots %d)\n", NS); std::fflush(stdout); continue; }
            fi = 2;
        } else if (f.size() < 3) { std::printf("ERR malformed GEN\n"); std::fflush(stdout); continue; }
        in.stop = false;
        int64_t max_new = atoll(f[fi].c_str());
        std::vector<int32_t> ids = parse_list<int>(f.back(), to_int);
        if (ids.empty()) { std::printf("ERR empty prompt\n"); std::fflush(stdout); continue; }
        if (max_new <= 0) max_new = ctx;
        try {
            const double t0 = now_s();
            const size_t p = place(target, ids);
            if (p) std::printf("RESUME %zu\n", p);
            std::fflush(stdout);
            if ((int) ids.size() + max_new > ctx) max_new = ctx - (int) ids.size();
            if (max_new <= 0) { std::printf("ERR the prompt fills the context\n"); std::fflush(stdout); if (bgen) { std::printf("BADM %d 0\n", target); std::fflush(stdout); } continue; }
            SpecStats st;
            struct Snap { uint64_t hits, misses, swaps, distinct, thrash, bytes; double ms; uint64_t freads, fbytes; double fms; double cms; uint64_t cpairs; uint64_t lhits, lmade; };
            auto snap = [&]() {
                Snap x{};
                if (store) x = {store->hits(), store->misses(), store->swaps(), store->distinct_misses(), store->thrash(), store->copy_bytes(), store->copy_ms(), store->file_reads(), store->file_bytes(), store->file_ms(), store->cpu_ms(), store->cpu_pairs(), store->loan_hits(), store->loans_made()};
                return x;
            };
            const Snap s0 = snap();
            int32_t cur = read_prompt(target, ids, p, st);
            const double prompt_ms = (now_s() - t0) * 1e3;
            const double tg = now_s();
            std::string finish = "length";
            std::vector<int32_t> out;
            auto emit_eos = [&](int32_t tk) { std::printf("T %d\n", tk); std::fflush(stdout); };
            if (bgen) {
                // the admission: the first token through the control lines (as a GEN 1 would), then BADM <slot> <continues>
                SlotSt& S0 = slots[(size_t) target];
                bool cont = false;
                if (cur >= 0 && is_eos(cur)) { finish = "stop"; emit_eos(cur); }
                else if (cur >= 0) { std::printf("T %d\n", cur); out.push_back(cur); cont = max_new > 1; }
                std::printf("DONE %zu %zu %.1f %.1f %s 0 0 %zu\n", out.size(), ids.size(), prompt_ms, 0.0, finish.c_str(), p);
                std::printf("BADM %d %d\n", target, cont ? 1 : 0);
                std::fflush(stdout);
                if (cont) { S0.active = true; S0.stop = false; S0.cur = cur; S0.max_new = max_new; S0.produced = 1; S0.t0 = now_s(); S0.st = SpecStats(); }
                std::fprintf(stderr, "glm5: slot %d admitted: %zu prompt (%zu reused) %.0f ms%s\n", target, ids.size(), p, prompt_ms, cont ? "" : " (ended at its first token)");
                continue;
            }
            // the solo request: speculative decoding with the MTP's drafts
            SlotSt& S0 = slots[(size_t) SOLO];
            if (in.stop) finish = "cancel";
            else if (cur >= 0 && is_eos(cur)) { finish = "stop"; emit_eos(cur); cur = -1; }
            else if (cur >= 0) { std::printf("T %d\n", cur); std::fflush(stdout); out.push_back(cur); }
            while (cur >= 0 && (int64_t) out.size() < max_new) {
                if (in.stop) { finish = "cancel"; break; }
                if (use_mtp && S0.mtp_ready) {
                    std::vector<int32_t>& drafts = S0.drafts;
                    std::vector<int32_t> win;
                    win.push_back(cur);
                    for (int32_t d : drafts) win.push_back(d);
                    const int T = (int) win.size();
                    const int pos0 = eng.pos(SOLO);
                    const double tr = now_s();
                    eng.forward_window(win.data(), T);
                    const std::vector<int32_t> am = eng.argmax_rows();
                    int n = 0;
                    while (n < (int) drafts.size() && drafts[(size_t) n] == am[(size_t) n]) ++n;
                    eng.commit(n + 1);
                    ++st.rounds; st.drafted += (int) drafts.size(); st.accepted += n;
                    S0.pol.update((int) drafts.size(), n, (now_s() - tr) * 1e3);
                    for (int i = 0; i <= n; ++i) S0.processed.push_back(win[(size_t) i]);
                    std::vector<int32_t> next;
                    for (int i = 0; i < n; ++i) next.push_back(drafts[(size_t) i]);
                    next.push_back(am[(size_t) n]);
                    bool ended = false;
                    for (int32_t tk : next) {
                        if (is_eos(tk)) { finish = "stop"; ended = true; emit_eos(tk); break; }
                        std::printf("T %d\n", tk);
                        out.push_back(tk);
                        if ((int64_t) out.size() >= max_new) break;
                    }
                    std::fflush(stdout);
                    if (ended) { cur = -1; break; }
                    cur = am[(size_t) n];
                    if (S0.pol.k_max == 0) S0.pol.k_max = spec;
                    drafts = mtp_draft(eng, SOLO, eng.h_win(), E, pos0, next, S0.pol.choose(), st);
                } else {
                    eng.forward_window(&cur, 1);
                    S0.processed.push_back(cur);
                    cur = eng.argmax_last();
                    if (is_eos(cur)) { finish = "stop"; emit_eos(cur); cur = -1; break; }
                    std::printf("T %d\n", cur); std::fflush(stdout);
                    out.push_back(cur);
                }
            }
            const double decode_ms = (now_s() - tg) * 1e3;
            const uint64_t hits = store ? store->hits() : 0, lookups = store ? store->hits() + store->misses() : 0;
            // DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> <drafts_accepted> <drafts_offered> <reused> <hits> <lookups>
            std::printf("DONE %zu %zu %.1f %.1f %s %d %d %zu %llu %llu\n", out.size(), ids.size(), prompt_ms, decode_ms, finish.c_str(),
                        st.accepted, st.drafted, p, (unsigned long long) hits, (unsigned long long) lookups);
            std::fflush(stdout);
            if (store && !profile_save.empty()) {   // the routing counts so far (every request: the server ends us with a kill)
                std::string err;
                if (!store->save_profile(profile_save, err)) std::fprintf(stderr, "glm5 experts: %s\n", err.c_str());
            }
            const Snap s1 = snap();
            std::fprintf(stderr, "glm5: %zu prompt (%zu reused) %.0f ms, %zu tokens %.0f ms = %.1f tok/s, %s; spec %d/%d in %d rounds (%.0f ms/round, mtp %.0f ms);"
                         " experts: %.1f%% hit, %llu misses, %llu swaps (%llu distinct, %llu thrash), %.0f MB in %.0f ms, file reads %llu (%.0f MB, %.0f ms), cpu %llu pairs in %.0f ms; loans %llu hits, %llu made\n", ids.size(), p, prompt_ms,
                         out.size(), decode_ms, out.empty() ? 0.0 : out.size() / (decode_ms / 1e3), finish.c_str(), st.accepted, st.drafted, st.rounds,
                         st.rounds ? decode_ms / st.rounds : 0.0, st.mtp_s * 1e3,
                         (s1.hits + s1.misses > s0.hits + s0.misses) ? 100.0 * (double) (s1.hits - s0.hits) / (double) ((s1.hits + s1.misses) - (s0.hits + s0.misses)) : 0.0,
                         (unsigned long long) (s1.misses - s0.misses), (unsigned long long) (s1.swaps - s0.swaps), (unsigned long long) (s1.distinct - s0.distinct),
                         (unsigned long long) (s1.thrash - s0.thrash), (double) (s1.bytes - s0.bytes) / 1e6, s1.ms - s0.ms,
                         (unsigned long long) (s1.freads - s0.freads), (double) (s1.fbytes - s0.fbytes) / 1e6, s1.fms - s0.fms, (unsigned long long) (s1.cpairs - s0.cpairs), s1.cms - s0.cms,
                         (unsigned long long) (s1.lhits - s0.lhits), (unsigned long long) (s1.lmade - s0.lmade));
        } catch (const std::exception& e) {
            std::printf("ERR %s\n", e.what());
            std::fflush(stdout);
            std::fprintf(stderr, "glm5: %s\n", e.what());
            std::fflush(stderr);
            std::_Exit(1);
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf, tokens_arg, dump_dir, profile, profile_save, devices_arg = "0", split_arg = "auto", budget_arg, prompts_file, mtp_path, eos_arg;
    bool serve = false;
    int admit_window = 32;
    int n_predict = 0, ctx = 4096, window = 8, pwin = 2048, spec = -1, adapt = 64, lookahead = 0, n_slots = 0, slots_test = 0;
    double pin_gib = 64, evict_pool_gib = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--gguf") gguf = next();
        else if (a == "--tokens") tokens_arg = next();
        else if (a == "--n-predict") n_predict = atoi(next());
        else if (a == "--dump") dump_dir = next();
        else if (a == "--device" || a == "--devices") devices_arg = next();
        else if (a == "--layer-split") split_arg = next();
        else if (a == "--ctx") ctx = atoi(next());
        else if (a == "--expert-budget-gib") budget_arg = next();
        else if (a == "--profile") profile = next();
        else if (a == "--profile-save") profile_save = next();
        else if (a == "--pin-gib") pin_gib = atof(next());
        else if (a == "--prompts") prompts_file = next();
        else if (a == "--window") window = atoi(next());
        else if (a == "--prefill-window") pwin = atoi(next());
        else if (a == "--mtp") mtp_path = next();
        else if (a == "--spec") spec = atoi(next());
        else if (a == "--attn") { const std::string v = next(); if (v == "lod") setenv("STRATA_GLM5_LOD", "1", 1); else if (v == "dsa") setenv("STRATA_GLM5_LOD", "0", 1); else { std::fprintf(stderr, "--attn: dsa or lod\n"); return 1; } }
        else if (a == "--lod-page") setenv("STRATA_GLM5_LOD_PAGE", next(), 1);
        else if (a == "--lod-top") setenv("STRATA_GLM5_LOD_TOP", next(), 1);
        else if (a == "--lod-top-decode") setenv("STRATA_GLM5_LOD_TOP_DECODE", next(), 1);
        else if (a == "--lod-local") setenv("STRATA_GLM5_LOD_LOCAL", next(), 1);
        else if (a == "--lod-sink") setenv("STRATA_GLM5_LOD_SINK", next(), 1);
        else if (a == "--adapt") adapt = atoi(next());
        else if (a == "--evict-pool-gib") evict_pool_gib = atof(next());
        else if (a == "--lookahead") lookahead = atoi(next());
        else if (a == "--serve") serve = true;
        else if (a == "--slots" || a == "--batch") n_slots = atoi(next());
        else if (a == "--slots-test") slots_test = atoi(next());
        else if (a == "--admit") admit_window = atoi(next());
        else if (a == "--eos") eos_arg = next();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (gguf.empty() || (tokens_arg.empty() && prompts_file.empty() && !serve)) {
        std::fprintf(stderr, "usage: strata-glm5 --gguf MODEL --tokens 1,2,3 [--n-predict K] [--dump DIR] [--ctx N] [--devices 0,1] "
                             "[--layer-split auto|L1[,L2]] [--expert-budget-gib G[,G2]] [--profile FILE] [--profile-save FILE] "
                             "[--pin-gib G] [--window T] [--mtp FILE] [--spec K] [--prompts FILE]\n");
        return 2;
    }
    if (!dump_dir.empty()) window = 1;
    if (lookahead > 16) lookahead = 16;
    if (spec < 0) spec = mtp_path.empty() ? 0 : 3;
    if (!mtp_path.empty() && spec + 1 > window) { std::fprintf(stderr, "--spec %d needs --window >= %d\n", spec, spec + 1); return 2; }
    try {
        const std::vector<int> devices = parse_list<int>(devices_arg, to_int);
        if (devices.empty()) { std::fprintf(stderr, "--devices needs at least one ordinal\n"); return 2; }
        strata::GgufModel probe = strata::GgufModel::open(gguf);
        const int n_layer = (int) (int64_t) probe.meta().get("glm5-next.block_count")->u;
        std::vector<int> starts;
        if (split_arg == "auto") {
            for (size_t k = 1; k < devices.size(); ++k) starts.push_back((int) ((int64_t) n_layer * (int64_t) k / (int64_t) devices.size()));
        } else {
            starts = parse_list<int>(split_arg, to_int);
        }
        if (starts.size() + 1 != devices.size()) { std::fprintf(stderr, "--layer-split needs %zu value(s) for %zu devices\n", devices.size() - 1, devices.size()); return 2; }
        std::vector<int> layer_device((size_t) n_layer);
        for (int il = 0; il < n_layer; ++il) {
            size_t k = 0;
            while (k < starts.size() && il >= starts[k]) ++k;
            layer_device[(size_t) il] = devices[k];
        }
        const double t0 = now_s();
        // model, engine and store are leaked on purpose when an exception escapes: a faulted stream never joins
        strata::glm5::Model& model = *new strata::glm5::Model(gguf, layer_device);
        const auto& hp = model.hp();
        if (!mtp_path.empty()) model.attach_mtp(mtp_path);
        std::fprintf(stderr, "glm5: %d layers, n_embd %d, %d experts x top-%d, vocab %d; dense weights:", hp.n_layer, hp.n_embd,
                     hp.n_expert, hp.n_expert_used, hp.n_vocab);
        for (size_t k = 0; k < devices.size(); ++k) {
            const int first = k == 0 ? 0 : starts[k - 1], last = k + 1 < devices.size() ? starts[k] - 1 : n_layer - 1;
            std::fprintf(stderr, " device %d layers %d-%d %.2f GiB;", devices[k], first, last, model.device_bytes(devices[k]) / 1073741824.0);
        }
        std::fprintf(stderr, " %.1f s\n", now_s() - t0);
        // the engine is leaked on purpose when an exception escapes (see Engine::~Engine): a faulted stream never joins
        if (pwin < window) pwin = window;
        strata::glm5::Engine& eng = *new strata::glm5::Engine(model, ctx, std::max(window, pwin), n_slots);
        auto vram_report = [&](const char* when) {
            for (int dv : model.devices()) {
                size_t fb = 0, tb = 0;
                if (cudaSetDevice(dv) == cudaSuccess && cudaMemGetInfo(&fb, &tb) == cudaSuccess)
                    std::fprintf(stderr, "glm5 vram: device %d %s: %.2f GiB free of %.2f\n", dv, when, fb / 1073741824.0, tb / 1073741824.0);
            }
        };
        vram_report("after the engine's buffers");
        strata::glm5::ExpertStore* store = nullptr;
        if (!budget_arg.empty()) {
            std::vector<double> budgets = parse_list<double>(budget_arg, to_double);
            if (budgets.size() == 1) budgets.assign(devices.size(), budgets[0]);
            if (budgets.size() != devices.size()) { std::fprintf(stderr, "--expert-budget-gib needs one value or one per device\n"); return 2; }
            strata::glm5::ExpertStoreConfig cfg;
            cfg.devices = model.devices();
            for (int dev : model.devices()) {
                size_t k = 0;
                while (devices[k] != dev) ++k;
                cfg.budget_bytes.push_back((uint64_t) (budgets[k] * 1073741824.0));
            }
            cfg.layer_device.resize((size_t) model.n_layer_all());
            for (int il = 0; il < model.n_layer_all(); ++il) {
                size_t k = 0;
                while (model.devices()[k] != model.layer_device()[(size_t) il]) ++k;
                cfg.layer_device[(size_t) il] = (int) k;
            }
            if (model.has_mtp()) cfg.force_resident_layers.push_back(hp.n_layer);
            cfg.profile = profile;
            cfg.t_max = std::max(window, pwin);
            cfg.pin_limit_bytes = (uint64_t) (pin_gib * 1073741824.0);
            cfg.evict_pool_bytes = (uint64_t) (evict_pool_gib * 1073741824.0);
            cfg.admit_window = (uint32_t) std::max(0, admit_window);
            store = new strata::glm5::ExpertStore(model, cfg);
            eng.set_experts(store);
            eng.set_adapt(adapt);
            eng.set_lookahead(lookahead);
            if (std::getenv("STRATA_GLM5_DMA")) store->copy_timing(true);
        } else if (model.has_mtp()) {
            std::fprintf(stderr, "--mtp needs --expert-budget-gib (the MTP layer's experts live in VRAM)\n");
            return 2;
        }
        vram_report("after the expert store");
        if (slots_test > 0) {
            // --slots-test K: K short prompts (prefixes of --tokens) decoded alone (greedy, no drafts) and then together
            // in one batch window per step; the batch must produce every conversation's own tokens
            if (slots_test > eng.n_slots()) { std::fprintf(stderr, "slots-test: needs --slots >= %d\n", slots_test); return 1; }
            std::vector<int32_t> toks = parse_list<int>(tokens_arg, to_int);
            const int K = slots_test, steps = n_predict > 0 ? n_predict : 32;
            std::vector<std::vector<int32_t>> prompts((size_t) K), solo((size_t) K);
            for (int i = 0; i < K; ++i) prompts[(size_t) i].assign(toks.begin(), toks.begin() + std::min<size_t>(toks.size(), (size_t) (16 + 9 * i)));
            double solo_s = 0;
            for (int i = 0; i < K; ++i) {
                eng.reset();
                eng.forward_window(prompts[(size_t) i].data(), (int) prompts[(size_t) i].size());
                int32_t cur = eng.argmax_last();
                const double t0 = now_s();
                for (int st = 0; st < steps; ++st) { solo[(size_t) i].push_back(cur); eng.forward_window(&cur, 1); cur = eng.argmax_last(); }
                solo_s += now_s() - t0;
                std::fprintf(stderr, "slots-test: solo %d done (%zu prompt tokens)\n", i, prompts[(size_t) i].size());
            }
            // the batch: without drafts (one row per slot) and, with the MTP, with each slot's drafts sharing the window
            int worst = 0;
            auto run_batch = [&](int k_draft) {
                std::vector<std::vector<int32_t>> got((size_t) K), drafts((size_t) K);
                std::vector<int32_t> curs((size_t) K);
                SpecStats sst;
                for (int i = 0; i < K; ++i) {
                    const std::vector<int32_t>& pr = prompts[(size_t) i];
                    eng.reset(i); eng.forward_window_slot(i, pr.data(), (int) pr.size()); curs[(size_t) i] = eng.argmax_last();
                    if (k_draft > 0) { std::vector<int32_t> next(pr.begin() + 1, pr.end()); next.push_back(curs[(size_t) i]); drafts[(size_t) i] = mtp_draft(eng, i, eng.h_win(), hp.n_embd, 0, next, k_draft, sst); }
                    std::fprintf(stderr, "slots-test: slot %d primed\n", i);
                }
                const double t1 = now_s();
                int windows = 0;
                struct Snap { uint64_t hits, misses, distinct, bytes, cpairs; double cms, copy_ms; };
                auto snap = [&]() { Snap x{}; if (store) x = {store->hits(), store->misses(), store->distinct_misses(), store->copy_bytes(), store->cpu_pairs(), store->cpu_ms(), store->copy_ms()}; return x; };
                const Snap s0 = snap();
                for (;;) {
                    std::vector<int32_t> toks; std::vector<int> rows, starts, lens, pos0;
                    for (int i = 0; i < K; ++i) {
                        if ((int) got[(size_t) i].size() >= steps) continue;
                        starts.push_back((int) toks.size()); pos0.push_back(eng.pos(i)); rows.push_back(i); toks.push_back(curs[(size_t) i]);
                        int len = 1;
                        for (int32_t d : drafts[(size_t) i]) { rows.push_back(i); toks.push_back(d); ++len; }
                        lens.push_back(len);
                    }
                    if (rows.empty()) break;
                    eng.forward_rows(toks.data(), rows.data(), (int) rows.size());
                    const std::vector<int32_t> am = eng.argmax_rows();
                    std::vector<int> keep(starts.size());
                    std::vector<std::vector<int32_t>> nexts(starts.size());
                    for (size_t g = 0; g < starts.size(); ++g) {
                        const int i = rows[(size_t) starts[g]];
                        int n = 0;
                        while (n < lens[g] - 1 && toks[(size_t) (starts[g] + 1 + n)] == am[(size_t) (starts[g] + n)]) ++n;
                        keep[g] = n + 1;
                        for (int j = 0; j <= n; ++j) got[(size_t) i].push_back(toks[(size_t) (starts[g] + j)]);
                        for (int j = 0; j < n; ++j) nexts[g].push_back(toks[(size_t) (starts[g] + 1 + j)]);
                        nexts[g].push_back(am[(size_t) (starts[g] + n)]);
                        curs[(size_t) i] = am[(size_t) (starts[g] + n)];
                        sst.drafted += lens[g] - 1; sst.accepted += n;
                    }
                    eng.commit_rows(keep.data(), (int) starts.size());
                    if (k_draft > 0)
                        for (size_t g = 0; g < starts.size(); ++g) {
                            const int i = rows[(size_t) starts[g]];
                            drafts[(size_t) i] = mtp_draft(eng, i, eng.h_win() + (size_t) starts[g] * hp.n_embd, hp.n_embd, pos0[g], nexts[g], k_draft, sst);
                        }
                    ++windows;
                    if (windows < 3 || windows % 16 == 0) std::fprintf(stderr, "slots-test: batch window %d done (%zu rows)\n", windows, rows.size());
                }
                const double batch_s = now_s() - t1;
                int mism = 0;
                for (int i = 0; i < K; ++i)
                    for (int st = 0; st < steps; ++st)
                        if (got[(size_t) i][(size_t) st] != solo[(size_t) i][(size_t) st]) { ++mism; if (mism <= 5) std::printf("drafts %d: step %d slot %d: batch %d solo %d\n", k_draft, st, i, got[(size_t) i][(size_t) st], solo[(size_t) i][(size_t) st]); }
                std::printf("slots-test: %d conversations x %d steps, %d drafts/slot: %d mismatches; solo %.1f ms/token, batch %d windows of %.1f ms = %.1f ms/token (%.1f tok/s aggregate), spec %d/%d, mtp %.0f ms\n",
                            K, steps, k_draft, mism, solo_s / (K * steps) * 1e3, windows, batch_s / windows * 1e3, batch_s / (K * steps) * 1e3, K * steps / batch_s, sst.accepted, sst.drafted, sst.mtp_s * 1e3);
                const Snap s1 = snap();
                if (store)
                    std::printf("slots-test: experts per window: %.1f%% hit, %.1f misses (%.1f distinct), %.1f MB DMA (%.1f ms), %.1f cpu pairs (%.1f ms)\n",
                                100.0 * (double) (s1.hits - s0.hits) / std::max<double>(1.0, (double) ((s1.hits + s1.misses) - (s0.hits + s0.misses))),
                                (double) (s1.misses - s0.misses) / windows, (double) (s1.distinct - s0.distinct) / windows, (double) (s1.bytes - s0.bytes) / 1e6 / windows,
                                (s1.copy_ms - s0.copy_ms) / windows, (double) (s1.cpairs - s0.cpairs) / windows, (s1.cms - s0.cms) / windows);
                std::fflush(stdout);
                worst = std::max(worst, mism);
            };
            run_batch(0);
            if (eng.has_mtp() && spec > 0) run_batch(std::min(spec, (8 - K) / K));
            std::_Exit(worst ? 3 : 0);
        }
        if (serve) {
            std::vector<int32_t> eos = parse_list<int>(eos_arg, to_int);
            if (eos.empty()) {   // the GGUF's eos / eot / eom
                for (const char* k : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"})
                    if (const auto* v = probe.meta().get(k)) eos.push_back((int32_t) v->u);
            }
            std::fprintf(stderr, "glm5: serving (ctx %d, window %d, spec %d, eos", ctx, window, model.has_mtp() ? spec : 0);
            for (int32_t t : eos) std::fprintf(stderr, " %d", t);
            std::fprintf(stderr, ")\n");
            vram_report("serving");
            return serve_loop(eng, store, hp, ctx, window, pwin, model.has_mtp() ? spec : 0, eos, profile_save);
        }
        if (!dump_dir.empty()) {
            const std::string mk = "mkdir -p '" + dump_dir + "'";
            if (std::system(mk.c_str()) != 0) return 1;
        }
        auto report_store = [&]() {
            if (!store) return;
            const double total = (double) (store->hits() + store->misses());
            std::fprintf(stderr, "glm5 experts: %llu hits, %llu misses (%.1f%% from VRAM), %llu adaptive swaps\n", (unsigned long long) store->hits(),
                         (unsigned long long) store->misses(), total > 0 ? 100.0 * store->hits() / total : 0.0, (unsigned long long) store->swaps());
            if (!profile_save.empty()) {
                std::string err;
                if (!store->save_profile(profile_save, err)) std::fprintf(stderr, "%s\n", err.c_str());
                else std::fprintf(stderr, "glm5 experts: routing counts written to %s\n", profile_save.c_str());
            }
        };
        // the prompt through the trunk in windows; with the MTP, its catch-up after every window
        auto run_prompt = [&](const std::vector<int32_t>& toks, SpecStats& st, std::vector<int32_t>& drafts) {
            const int E = hp.n_embd;
            drafts.clear();
            int32_t first = -1;
            const int pos_base = eng.pos(eng.solo_slot());
            const double ta = now_s();
            // the windows through the device pipeline; the MTP catch-up per window from the last stage (stream-ordered)
            eng.prompt_windows(eng.solo_slot(), toks.data(), (int) toks.size(), pwin, [&](int pos0, int T, bool last) {
                const size_t off = (size_t) (pos0 - pos_base);
                if (last) first = eng.argmax_last();
                if (model.has_mtp()) {
                    std::vector<int32_t> next;
                    for (int i = 0; i < T; ++i) {
                        const size_t p = off + (size_t) i + 1;
                        next.push_back(p < toks.size() ? toks[p] : first);
                    }
                    drafts = mtp_draft(eng, eng.solo_slot(), eng.h_win(), E, pos0, next, last ? spec : 0, st);
                }
            });
            st.trunk_s += now_s() - ta;
            return first;
        };
        if (!prompts_file.empty()) {
            std::ifstream pf(prompts_file);
            std::string line;
            int n = 0;
            const double tp = now_s();
            while (std::getline(pf, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::vector<int32_t> toks = parse_list<int>(line, to_int);
                if (toks.empty()) continue;
                eng.reset();
                SpecStats st;
                std::vector<int32_t> drafts;
                const double t0 = now_s();
                int32_t cur = run_prompt(toks, st, drafts);
                const double t1 = now_s();
                std::vector<int32_t> out;
                out.push_back(cur);
                for (int step = 0; step + 1 < n_predict; ++step) { eng.forward_window(&cur, 1); cur = eng.argmax_last(); out.push_back(cur); }
                ++n;
                const double t2 = now_s();
                std::printf("prompt %d: %zu tokens, prefill %.2f s (%.0f tok/s), decode %d in %.2f s (%.1f tok/s); tokens:", n, toks.size(), t1 - t0,
                            toks.size() / std::max(1e-9, t1 - t0), n_predict, t2 - t1, n_predict / std::max(1e-9, t2 - t1));
                for (int32_t t : out) std::printf(" %d", t);
                std::printf("\n");
                std::fflush(stdout);
                std::fprintf(stderr, "prompt %d: %zu tokens + %d, %.0f s so far\n", n, toks.size(), n_predict, now_s() - tp);
            }
            report_store();
            return 0;
        }
        std::vector<int32_t> toks = parse_list<int>(tokens_arg, to_int);
        std::vector<char> host;
        if (!dump_dir.empty()) {
            // the oracle run: single-token windows, every intermediate written
            std::vector<int32_t> cur = toks;
            for (int step = 0; step <= n_predict; ++step) {
                std::map<std::string, int> seen;
                for (size_t k = 0; k < cur.size(); ++k) {
                    strata::glm5::DumpFn dump = [&](const std::string& name, const void* dev, const std::vector<int64_t>& shape, bool is_int, int device) {
                        size_t n = 1;
                        for (auto dd : shape) n *= (size_t) dd;
                        host.resize(n * 4);
                        if (cudaSetDevice(device) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
                            cudaMemcpy(host.data(), dev, n * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
                            throw std::runtime_error("dump: copy of " + name + " failed");
                        const int dup = seen[name]++;
                        write_npy(dump_dir + "/s" + std::to_string(step) + "_" + name + (dup ? "__" + std::to_string(dup) : "") + ".npy", host.data(), shape, is_int);
                    };
                    eng.forward(cur[k], dump);
                }
                const int32_t best = eng.argmax_last();
                std::printf("step %d: argmax %d\n", step, best);
                cur.assign(1, best);
            }
            return 0;
        }
        SpecStats st;
        std::vector<int32_t> drafts;
        const double tpr = now_s();
        int32_t cur = run_prompt(toks, st, drafts);
        const double prompt_s = now_s() - tpr;
        std::printf("prompt: %zu tokens in %.2f s (%.0f tok/s, window %d); first token %d\n", toks.size(), prompt_s, toks.size() / prompt_s, window, cur);
        std::vector<int32_t> out;
        out.push_back(cur);
        const double tg = now_s();
        const int E = hp.n_embd;
        if (!model.has_mtp() || spec == 0) {
            for (int step = 1; step < n_predict; ++step) {
                eng.forward_window(&cur, 1);
                cur = eng.argmax_last();
                out.push_back(cur);
            }
        } else {
            SpecPolicy pol; pol.k_max = spec;
            while ((int) out.size() < n_predict) {
                // the window: the newest committed token, then the drafts
                std::vector<int32_t> win;
                win.push_back(cur);
                for (int32_t d : drafts) win.push_back(d);
                const int T = (int) win.size();
                const int pos0 = eng.pos();
                const double ta = now_s();
                eng.forward_window(win.data(), T);
                const std::vector<int32_t> am = eng.argmax_rows();
                int n = 0;
                while (n < (int) drafts.size() && drafts[(size_t) n] == am[(size_t) n]) ++n;
                eng.commit(n + 1);
                const double round_ms = (now_s() - ta) * 1e3;
                st.trunk_s += now_s() - ta;
                if (std::getenv("STRATA_GLM5_ROUNDS")) {
                    const auto ws = eng.last_window();
                    std::fprintf(stderr, "round %d: T %d accepted %d; %.1f ms (enqueue %.1f, publish wait %.1f, misses %llu = %llu distinct in %llu layers, %llu thrash, swaps %d, prefetched %llu; miss DMA %.1f ms for %.0f MB = %.1f GB/s)\n", st.rounds + 1, T, n,
                                 round_ms, ws.enqueue_ms, ws.wait_ms, (unsigned long long) ws.misses, (unsigned long long) ws.distinct, (unsigned long long) ws.layers, (unsigned long long) ws.thrash, ws.swaps, (unsigned long long) ws.prefetched,
                                 ws.copy_ms, ws.copy_bytes / 1e6, ws.copy_ms > 0 ? ws.copy_bytes / 1e6 / ws.copy_ms : 0.0);
                }
                ++st.rounds;
                st.drafted += (int) drafts.size();
                st.accepted += n;
                pol.update((int) drafts.size(), n, round_ms);
                std::vector<int32_t> next;   // the token after each kept window position
                for (int i = 0; i < n; ++i) next.push_back(drafts[(size_t) i]);
                next.push_back(am[(size_t) n]);
                for (int32_t t : next) out.push_back(t);
                cur = am[(size_t) n];
                drafts = mtp_draft(eng, eng.solo_slot(), eng.h_win(), E, pos0, next, pol.choose(), st);
            }
            if ((int) out.size() > n_predict) out.resize((size_t) n_predict);
        }
        const double gen_s = now_s() - tg;
        std::printf("tokens:");
        for (int32_t t : out) std::printf(" %d", t);
        std::printf("\n%zu tokens in %.2f s = %.1f tok/s (%.1f ms/token)\n", out.size(), gen_s, out.size() / gen_s, gen_s / out.size() * 1e3);
        if (store) std::printf("experts: %llu distinct misses, %llu thrash; cpu %llu pairs in %.0f ms (%.2f ms/pair); file reads %llu\n", (unsigned long long) store->distinct_misses(), (unsigned long long) store->thrash(), (unsigned long long) store->cpu_pairs(), store->cpu_ms(), store->cpu_pairs() ? store->cpu_ms() / store->cpu_pairs() : 0.0, (unsigned long long) store->file_reads());
        if (store) { const double* ph = store->cpu_phase_ms(); std::printf("cpu phases over %llu calls: split %.0f, quant x %.0f, gate/up %.0f, quant h %.0f, down %.0f, enqueue %.0f ms\n", (unsigned long long) store->cpu_calls(), ph[0], ph[1], ph[2], ph[3], ph[4], ph[5]); }
        if (st.rounds)
            std::printf("speculation: %d rounds, %d drafted, %d accepted (%.2f per round, %.1f%%), %.2f tokens/round; trunk %.2f s, mtp %.2f s\n",
                        st.rounds, st.drafted, st.accepted, (double) st.accepted / st.rounds, 100.0 * st.accepted / std::max(1, st.drafted),
                        (double) out.size() / st.rounds, st.trunk_s, st.mtp_s);
        report_store();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-glm5: %s\n", e.what());
        std::fflush(stderr);
        std::_Exit(1);
    }
}
