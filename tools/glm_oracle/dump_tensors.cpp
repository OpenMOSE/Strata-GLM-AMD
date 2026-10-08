// tools/glm_oracle/dump_tensors.cpp - run a prompt through llama.cpp and save the named intermediate tensors.
//
// This is the ORACLE for the GLM-5-Next port (docs/GLM5_NEXT.md, P0): every Strata kernel and ref/glm5.py are
// judged against what llama.cpp computes on the same GGUF and the same token ids.  It links only libllama and
// libggml (no `common`), so it builds against any llama.cpp build tree: see build.sh beside it.
//
//   dump_tensors -m MODEL.gguf -o OUTDIR [-p "text" | --tokens 1,2,3] [--filter REGEX ...]
//                [-ngl N] [--n-cpu-moe N] [--ubatch N] [--n-predict K] [--ctx N] [--threads N]
//
// Every graph node whose name matches one of the --filter regexes (all nodes when none is given) is written
// as `<OUTDIR>/s<step>_<name>.npy` (float32 for F32/F16/BF16, int32 for I32; other types are skipped), where
// <step> counts llama_decode calls: step 0 is the prompt, steps 1..K the greedy continuation one token at a
// time.  The npy shape is ggml's ne reversed, so a [n_embd, n_tokens] node lands as (n_tokens, n_embd).
// `tokens.txt` holds the prompt ids and `greedy.txt` the argmax of every step, one per line.
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <vector>

namespace {

struct Dumper {
    std::string outdir;
    std::vector<std::regex> filters;
    int step = 0;
    std::map<std::string, int> seen;   // per-step duplicate names
    std::vector<uint8_t> buf;
    size_t written = 0;

    bool wants(const char* name) const {
        if (filters.empty()) return true;
        for (const auto& f : filters)
            if (std::regex_search(name, f)) return true;
        return false;
    }

    static float to_f32(const uint8_t* p, ggml_type t) {
        switch (t) {
            case GGML_TYPE_F32: { float v; memcpy(&v, p, 4); return v; }
            case GGML_TYPE_F16: { ggml_fp16_t h; memcpy(&h, p, 2); return ggml_fp16_to_fp32(h); }
            case GGML_TYPE_BF16: { ggml_bf16_t b; memcpy(&b, p, 2); return ggml_bf16_to_fp32(b); }
            default: return 0.f;
        }
    }

    void dump(ggml_tensor* t) {
        const ggml_type ty = t->type;
        const bool is_i32 = ty == GGML_TYPE_I32;
        if (!(ty == GGML_TYPE_F32 || ty == GGML_TYPE_F16 || ty == GGML_TYPE_BF16 || is_i32)) return;
        const size_t nbytes = ggml_nbytes(t);
        const uint8_t* data;
        if (ggml_backend_buffer_is_host(t->buffer)) {
            data = (const uint8_t*) t->data;
        } else {
            buf.resize(nbytes);
            ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
            data = buf.data();
        }
        std::string name = t->name;
        for (auto& c : name) if (c == '/' || c == ' ') c = '_';
        const int dup = seen[name]++;
        char fn[1024];
        if (dup == 0) snprintf(fn, sizeof fn, "%s/s%d_%s.npy", outdir.c_str(), step, name.c_str());
        else snprintf(fn, sizeof fn, "%s/s%d_%s__%d.npy", outdir.c_str(), step, name.c_str(), dup);

        // the data span is nb-strided; nbytes already accounts for views, but the element walk uses nb
        const int64_t* ne = t->ne;
        const size_t* nb = t->nb;
        std::vector<char> out;
        out.reserve((size_t) ne[0] * ne[1] * ne[2] * ne[3] * 4);
        for (int64_t i3 = 0; i3 < ne[3]; ++i3)
            for (int64_t i2 = 0; i2 < ne[2]; ++i2)
                for (int64_t i1 = 0; i1 < ne[1]; ++i1)
                    for (int64_t i0 = 0; i0 < ne[0]; ++i0) {
                        const uint8_t* p = data + i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                        char v[4];
                        if (is_i32) memcpy(v, p, 4);
                        else { float f = to_f32(p, ty); memcpy(v, &f, 4); }
                        out.insert(out.end(), v, v + 4);
                    }
        // npy v1.0 header
        char shape[256];
        snprintf(shape, sizeof shape, "(%lld, %lld, %lld, %lld)", (long long) ne[3], (long long) ne[2],
                 (long long) ne[1], (long long) ne[0]);
        std::string hdr = std::string("{'descr': '") + (is_i32 ? "<i4" : "<f4") +
                          "', 'fortran_order': False, 'shape': " + shape + ", }";
        while ((10 + hdr.size() + 1) % 64 != 0) hdr += ' ';
        hdr += '\n';
        std::ofstream f(fn, std::ios::binary);
        const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
        f.write((const char*) magic, 8);
        const uint16_t hl = (uint16_t) hdr.size();
        f.write((const char*) &hl, 2);
        f.write(hdr.data(), hdr.size());
        f.write(out.data(), out.size());
        written += out.size();
    }
};

bool cb_eval(ggml_tensor* t, bool ask, void* user) {
    auto* d = (Dumper*) user;
    if (ask) return d->wants(t->name);
    if (d->wants(t->name)) d->dump(t);
    return true;
}

std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int32_t> ids;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) ids.push_back(atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return ids;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, prompt, tokens_arg, outdir = "oracle_out";
    int ngl = 999, ncmoe = 0, ubatch = 512, n_predict = 0, n_ctx = 4096, threads = 8;
    Dumper d;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "-m") model = next();
        else if (a == "-p") prompt = next();
        else if (a == "--tokens") tokens_arg = next();
        else if (a == "-o") outdir = next();
        else if (a == "--filter") d.filters.emplace_back(next());
        else if (a == "-ngl") ngl = atoi(next());
        else if (a == "--n-cpu-moe") ncmoe = atoi(next());
        else if (a == "--ubatch") ubatch = atoi(next());
        else if (a == "--n-predict") n_predict = atoi(next());
        else if (a == "--ctx") n_ctx = atoi(next());
        else if (a == "--threads") threads = atoi(next());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model.empty() || (prompt.empty() && tokens_arg.empty())) {
        fprintf(stderr, "usage: dump_tensors -m MODEL -o OUTDIR (-p TEXT | --tokens IDS) [--filter RE]... [-ngl N] "
                        "[--n-cpu-moe N] [--ubatch N] [--n-predict K] [--ctx N] [--threads N]\n");
        return 2;
    }
    d.outdir = outdir;
    std::string mk = "mkdir -p '" + outdir + "'";
    if (system(mk.c_str()) != 0) return 1;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    // keep the first N layers' experts on the CPU: the 117 GB file does not fit two 48 GB cards
    std::vector<std::string> pats;
    std::vector<llama_model_tensor_buft_override> ov;
    for (int l = 0; l < ncmoe; ++l) pats.push_back("blk\\." + std::to_string(l) + "\\.ffn_(up|down|gate)_exps");
    for (auto& p : pats) ov.push_back({p.c_str(), ggml_backend_cpu_buffer_type()});
    ov.push_back({nullptr, nullptr});
    mp.tensor_buft_overrides = ov.data();

    llama_model* lm = llama_model_load_from_file(model.c_str(), mp);
    if (!lm) { fprintf(stderr, "could not load %s\n", model.c_str()); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(lm);

    std::vector<llama_token> toks;
    if (!tokens_arg.empty()) {
        for (int32_t id : parse_ids(tokens_arg)) toks.push_back(id);
    } else {
        const bool add_bos = llama_vocab_get_add_bos(vocab);
        const int n = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, add_bos, true);
        toks.resize(n);
        if (llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), toks.data(), n, add_bos, true) < 0) {
            fprintf(stderr, "tokenize failed\n"); return 1;
        }
    }
    {
        std::ofstream f(outdir + "/tokens.txt");
        for (auto t : toks) f << t << "\n";
    }
    fprintf(stderr, "prompt: %zu tokens\n", toks.size());

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = ubatch;
    cp.n_ubatch = ubatch;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &d;
    cp.no_perf = false;
    llama_context* ctx = llama_init_from_model(lm, cp);
    if (!ctx) { fprintf(stderr, "could not create the context\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::ofstream greedy(outdir + "/greedy.txt");
    std::vector<llama_token> cur = toks;
    for (int step = 0; step <= n_predict; ++step) {
        d.step = step;
        d.seen.clear();
        // the prompt in slices of ubatch (so every slice's nodes carry the same step number)
        for (size_t off = 0; off < cur.size(); off += (size_t) ubatch) {
            const int n = (int) std::min((size_t) ubatch, cur.size() - off);
            llama_batch batch = llama_batch_get_one(cur.data() + off, n);
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at step %d\n", step); return 1; }
        }
        const float* logits = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = i;
        // the last position's logits, as the ref computes them
        {
            char fn[1024];
            snprintf(fn, sizeof fn, "%s/s%d_logits_last.npy", outdir.c_str(), step);
            std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(n_vocab) + ",), }";
            while ((10 + hdr.size() + 1) % 64 != 0) hdr += ' ';
            hdr += '\n';
            std::ofstream f(fn, std::ios::binary);
            const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
            f.write((const char*) magic, 8);
            const uint16_t hl = (uint16_t) hdr.size();
            f.write((const char*) &hl, 2);
            f.write(hdr.data(), hdr.size());
            f.write((const char*) logits, (size_t) n_vocab * 4);
        }
        greedy << best << "\n";
        greedy.flush();
        char piece[256];
        const int pn = llama_token_to_piece(vocab, best, piece, sizeof piece, 0, true);
        fprintf(stderr, "step %d: argmax %d '%.*s' (%zu bytes of tensors so far)\n", step, best, pn > 0 ? pn : 0,
                piece, d.written);
        cur.assign(1, best);
    }
    llama_free(ctx);
    llama_model_free(lm);
    llama_backend_free();
    return 0;
}
