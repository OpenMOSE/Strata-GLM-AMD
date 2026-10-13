// src/glm5/model.cpp - the GGUF's hparams and the dense weights on the devices (include/strata/glm5/glm5.hpp).
#include "strata/glm5/glm5.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <exception>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace strata::glm5 {
namespace {

const strata::MetaValue& need(const strata::GgufFile& g, const std::string& key) {
    const strata::MetaValue* v = g.get(key);
    if (!v) throw std::runtime_error("GGUF: missing " + key);
    return *v;
}
int geti(const strata::GgufFile& g, const std::string& key, int def, bool required = false) {
    const strata::MetaValue* v = g.get(key);
    if (!v) { if (required) throw std::runtime_error("GGUF: missing " + key); return def; }
    return (int) (int64_t) v->u;
}
float getf(const strata::GgufFile& g, const std::string& key, float def) {
    const strata::MetaValue* v = g.get(key);
    return v ? (float) v->num() : def;
}
std::vector<float> getfa(const strata::GgufFile& g, const std::string& key, int n, float def) {
    std::vector<float> out((size_t) n, def);
    const strata::MetaValue* v = g.get(key);
    if (!v) return out;
    if (v->type == strata::MetaType::ARRAY) {
        for (size_t i = 0; i < out.size() && i < v->items.size(); ++i) out[i] = (float) v->items[i].num();
    } else {
        std::fill(out.begin(), out.end(), (float) v->num());
    }
    return out;
}
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

HParams read_hparams(const strata::GgufModel& m) {
    const strata::GgufFile& g = m.meta();
    const std::string arch = need(g, "general.architecture").s;
    if (arch != "glm5-next") throw std::runtime_error("this engine runs glm5-next; the file is " + arch);
    const std::string a = "glm5-next.";
    HParams h;
    h.n_layer = geti(g, a + "block_count", 0, true);
    h.n_embd = geti(g, a + "embedding_length", 0, true);
    h.n_ff = geti(g, a + "feed_forward_length", 0, true);
    h.n_head = geti(g, a + "attention.head_count", 0, true);
    h.n_vocab = geti(g, a + "vocab_size", 0);
    h.rms_eps = getf(g, a + "attention.layer_norm_rms_epsilon", 1e-5f);
    h.norm_eps = getf(g, a + "attention.layer_norm_epsilon", 1e-5f);
    h.n_expert = geti(g, a + "expert_count", 0, true);
    h.n_expert_used = geti(g, a + "expert_used_count", 0, true);
    h.n_ff_exp = geti(g, a + "expert_feed_forward_length", 0, true);
    h.n_expert_shared = geti(g, a + "expert_shared_count", 0);
    h.n_dense_lead = geti(g, a + "leading_dense_block_count", 0);
    h.expert_scale = getf(g, a + "expert_weights_scale", 1.f);
    h.expert_norm = geti(g, a + "expert_weights_norm", 0) != 0;
    h.gating = geti(g, a + "expert_gating_func", 2);
    h.clamp_exp = getfa(g, a + "swiglu_clamp_exp", h.n_layer, 0.f);
    h.clamp_shexp = g.get(a + "swiglu_clamp_shexp") ? getfa(g, a + "swiglu_clamp_shexp", h.n_layer, 0.f) : h.clamp_exp;
    h.d_conv = geti(g, a + "ssm.conv_kernel", 4, true);
    h.kda_head_dim = geti(g, a + "kda.head_dim", 128, true);
    h.kda_lower = getf(g, a + "kda.gate_lower_bound", -INFINITY);
    h.q_lora = geti(g, a + "attention.q_lora_rank", 0, true);
    h.kv_lora = geti(g, a + "attention.kv_lora_rank", 0, true);
    h.head_k_mla = geti(g, a + "attention.key_length_mla", 0, true);
    h.head_v_mla = geti(g, a + "attention.value_length_mla", 0, true);
    h.n_rot = geti(g, a + "rope.dimension_count", 0);
    h.idx_heads = geti(g, a + "attention.indexer.head_count", 0, true);
    h.idx_dim = geti(g, a + "attention.indexer.key_length", 0, true);
    h.idx_topk = geti(g, a + "attention.indexer.top_k", 0, true);
    h.kpool = geti(g, a + "attention.indexer.kpool", 0, true);
    h.kpool_tail = geti(g, a + "attention.indexer.kpool_select_tail", 0) != 0;
    h.hc = geti(g, a + "hyper_connection.count", 4, true);
    h.hc_iters = geti(g, a + "hyper_connection.sinkhorn_iterations", 20, true);
    h.hc_eps = getf(g, a + "hyper_connection.epsilon", 1e-6f);
    const strata::MetaValue& kv = need(g, a + "attention.head_count_kv");
    h.is_recr.assign((size_t) h.n_layer, true);
    if (kv.type == strata::MetaType::ARRAY) {
        for (size_t i = 0; i < h.is_recr.size() && i < kv.items.size(); ++i) h.is_recr[i] = kv.items[i].u == 0;
    } else {
        std::fill(h.is_recr.begin(), h.is_recr.end(), kv.u == 0);
    }
    if (h.hc != 4) throw std::runtime_error("mHC with hyper_connection.count != 4 is not supported (llama.cpp neither)");
    if (h.n_rot != 0) throw std::runtime_error("glm5-next with RoPE dimensions is not the nope-only MLA this engine runs");
    if (h.gating != 2) throw std::runtime_error("only the sigmoid router (expert_gating_func 2) is implemented");
    if (h.kpool < 2 || h.idx_topk % h.kpool) throw std::runtime_error("indexer.kpool must be > 1 and divide top_k");
    if (const strata::TensorInfo* t = m.find("token_embd.weight")) {
        if (t->shape.size() == 2) h.n_vocab = (int) t->shape[1];
    }
    return h;
}


namespace {
uint16_t f32_to_f16_bits(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    int32_t ex = (int32_t) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = u & 0x7fffffu;
    if (((u >> 23) & 0xff) == 0xff) return (uint16_t) (sign | 0x7c00u | (mant ? 0x200u : 0));
    if (ex >= 31) return (uint16_t) (sign | 0x7c00u);
    if (ex <= 0) {
        if (ex < -10) return (uint16_t) sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - ex);
        uint32_t h = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return (uint16_t) (sign | h);
    }
    uint32_t h = (uint32_t) (ex << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return (uint16_t) (sign | h);
}
float f16_bits_to_f32(uint16_t dh) {
    const uint32_t sgn = (dh & 0x8000u) << 16, ex = (dh >> 10) & 0x1f, mant = dh & 0x3ff;
    float d;
    if (ex == 0) d = (sgn ? -1.f : 1.f) * std::ldexp((float) mant, -24);
    else if (ex == 31) d = mant ? NAN : (sgn ? -INFINITY : INFINITY);
    else { uint32_t u = sgn | ((ex + 112) << 23) | (mant << 13); std::memcpy(&d, &u, 4); }
    return d;
}
}  // namespace

// A Q8_0 copy of a BF16 or Q8_0 tensor (ggml's quantize_row_q8_0 reference arithmetic): the prompt windows run the
// per-head attention matrices through MMQ; the decode kernels keep the original.
DevTensor Model::q8_copy(const DevTensor& t, int dev) {
    DevTensor q;
    if (!t.present() || (t.type != 30 && t.type != 8)) return q;
    const int64_t n = t.ne0 * t.ne1 * t.ne2;
    if (n % 32) return q;
    std::vector<uint8_t> out((size_t) (n / 32) * 34);
    check(cudaSetDevice(dev), "cudaSetDevice");
    if (t.type == 8) {
        check(cudaMemcpy(out.data(), t.data, out.size(), cudaMemcpyDeviceToHost), "q8 d2h");
    } else {
        std::vector<uint16_t> bf((size_t) n);
        check(cudaMemcpy(bf.data(), t.data, bf.size() * 2, cudaMemcpyDeviceToHost), "bf16 d2h");
        for (int64_t b = 0; b < n / 32; ++b) {
            float x[32], amax = 0.f;
            for (int i = 0; i < 32; ++i) { const uint32_t u = (uint32_t) bf[(size_t) b * 32 + i] << 16; std::memcpy(&x[i], &u, 4); amax = std::max(amax, std::fabs(x[i])); }
            const float d = amax / 127.f, id = d ? 1.f / d : 0.f;
            const uint16_t dh = f32_to_f16_bits(d);
            uint8_t* blk = out.data() + (size_t) b * 34;
            std::memcpy(blk, &dh, 2);
            for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t) (int8_t) std::lround(x[i] * id);
        }
    }
    void* dptr = nullptr;
    check(cudaMalloc(&dptr, out.size()), "cudaMalloc q8 copy");
    allocations_.push_back(dptr);
    alloc_device_.push_back(dev);
    check(cudaMemcpy(dptr, out.data(), out.size(), cudaMemcpyHostToDevice), "q8 h2d");
    q = t;
    q.data = dptr; q.type = 8; q.bytes = out.size();
    bytes_ += q.bytes;
    return q;
}

Model::Model(const std::string& any_shard, int device, bool verbose) {
    gguf_ = std::make_unique<strata::GgufModel>(strata::GgufModel::open(any_shard));
    hp_ = read_hparams(*gguf_);
    layer_device_.assign((size_t) hp_.n_layer, device);
    init(any_shard, verbose);
}

Model::Model(const std::string& any_shard, const std::vector<int>& layer_device, bool verbose) {
    gguf_ = std::make_unique<strata::GgufModel>(strata::GgufModel::open(any_shard));
    hp_ = read_hparams(*gguf_);
    if ((int) layer_device.size() != hp_.n_layer) throw std::runtime_error("glm5: layer_device must name every layer");
    layer_device_ = layer_device;
    init(any_shard, verbose);
}

void Model::init(const std::string&, bool verbose) {
    for (int d : layer_device_)
        if (std::find(devices_.begin(), devices_.end(), d) == devices_.end()) devices_.push_back(d);
    int max_ord = 0;
    for (int d : devices_) max_ord = std::max(max_ord, d);
    bytes_per_device_.assign((size_t) max_ord + 1, 0);
    layers_.resize((size_t) hp_.n_layer);
    token_embd_ = upload(*gguf_, "token_embd.weight", layer_device_.front());
    output_ = upload(*gguf_, "output.weight", layer_device_.back());
    output_norm_ = upload(*gguf_, "output_norm.weight", layer_device_.back());
    for (int il = 0; il < hp_.n_layer; ++il) {
        LayerWeights& L = layers_[(size_t) il];
        const int dev = layer_device_[(size_t) il];
        L.device = dev;
        const std::string b = "blk." + std::to_string(il) + ".";
        auto up = [&](const std::string& n, bool required = true) { return upload(*gguf_, b + n, dev, required); };
        L.attn_norm = up("attn_norm.weight");
        L.ffn_norm = up("ffn_norm.weight");
        L.hc_attn_fn = up("hc_attn_fn.weight");
        L.hc_attn_scale = up("hc_attn_scale.weight");
        L.hc_attn_base = up("hc_attn_base.weight");
        L.hc_ffn_fn = up("hc_ffn_fn.weight");
        L.hc_ffn_scale = up("hc_ffn_scale.weight");
        L.hc_ffn_base = up("hc_ffn_base.weight");
        if (hp_.is_recr[(size_t) il]) {
            {   // q | k | v as one product; their conv taps stacked the same way; f_a | g_a | beta as one (all from the normed input)
                DevTensor wq = up("attn_q.weight"), wk = up("attn_k.weight"), wv = up("attn_v.weight");
                L.wqkv = stack_rows({&wq, &wk, &wv});
                DevTensor cq = up("ssm_conv1d_q.weight"), ck = up("ssm_conv1d_k.weight"), cv = up("ssm_conv1d_v.weight");
                L.conv_qkv = stack_rows({&cq, &ck, &cv});
                DevTensor fa = up("ssm_f_a.weight"), ga = up("ssm_g_a.weight"), be = up("ssm_beta.weight");
                L.fab = stack_rows({&fa, &ga, &be});
            }
            L.wo = up("attn_output.weight");
            L.f_b = up("ssm_f_b.weight");
            L.g_b = up("ssm_g_b.weight");
            L.ssm_a = up("ssm_a");
            L.dt_b = up("ssm_dt.bias");
            L.ssm_norm = up("ssm_norm.weight");
        } else {
            L.wo = up("attn_output.weight");
            L.q_a = up("attn_q_a.weight");
            L.q_a_norm = up("attn_q_a_norm.weight");
            L.q_b = up("attn_q_b.weight");
            L.kv_a = up("attn_kv_a_mqa.weight");
            L.kv_a_norm = up("attn_kv_a_norm.weight");
            L.k_b = up("attn_k_b.weight");
            L.v_b = up("attn_v_b.weight");
            L.k_b_q8 = q8_copy(L.k_b, dev);
            L.v_b_q8 = q8_copy(L.v_b, dev);
            L.idx_k = up("indexer.attn_k.weight");
            L.idx_k_norm = up("indexer.k_norm.weight");
            L.idx_k_norm_b = up("indexer.k_norm.bias");
            L.idx_proj = up("indexer.proj.weight");
            L.idx_q_b = up("indexer.attn_q_b.weight");
            L.idx_gate = up("indexer_compressor_gate.weight");
            L.idx_ape = up("indexer_compressor_ape.weight");
        }
        if (il < hp_.n_dense_lead) {
            {
                DevTensor fg = up("ffn_gate.weight"), fu = up("ffn_up.weight");
                L.ffn_gu = stack_rows({&fg, &fu});
            }
            L.ffn_down = up("ffn_down.weight");
        } else {
            L.gate_inp = up("ffn_gate_inp.weight");
            L.exp_probs_b = up("exp_probs_b.bias", false);
            {
                DevTensor sg = up("ffn_gate_shexp.weight"), su = up("ffn_up_shexp.weight");
                L.sh_gu = stack_rows({&sg, &su});
            }
            L.sh_down = up("ffn_down_shexp.weight");
            L.exp_gate = expert(*gguf_, b + "ffn_gate_exps.weight");
            L.exp_up = expert(*gguf_, b + "ffn_up_exps.weight");
            L.exp_down = expert(*gguf_, b + "ffn_down_exps.weight");
        }
        if (verbose && (il % 8 == 0 || il == hp_.n_layer - 1))
            std::fprintf(stderr, "glm5: layer %2d uploaded to device %d, %.2f GiB on the devices\n", il, dev, bytes_ / 1073741824.0);
    }
}

Model::~Model() {
    if (std::uncaught_exceptions() > 0) return;   // a device fault in flight: cudaFree would wait on a spinning stream
    for (size_t i = 0; i < allocations_.size(); ++i) {
        cudaSetDevice(alloc_device_[i]);
        cudaFree(allocations_[i]);
    }
}

uint64_t Model::device_bytes(int ordinal) const {
    return ordinal >= 0 && (size_t) ordinal < bytes_per_device_.size() ? bytes_per_device_[(size_t) ordinal] : 0;
}

// the parts (same type, same row length, same device) stacked as one matrix [sum ne1][ne0] (row-contiguous types:
// the stacked bytes are the parts' bytes in order); the parts' own buffers are freed (the whole replaces them)
DevTensor Model::stack_rows(std::initializer_list<DevTensor*> parts) {
    const DevTensor& a = **parts.begin();
    uint64_t bytes = 0;
    int64_t ne1 = 0;
    for (const DevTensor* t : parts) {
        if (!t->present() || t->type != a.type || t->ne0 != a.ne0 || t->device != a.device)
            throw std::runtime_error("stack_rows: the parts differ in type, row length or device");
        bytes += t->bytes;
        ne1 += t->ne1 * std::max<int64_t>(t->ne2, 1);
    }
    check(cudaSetDevice(a.device), "cudaSetDevice");
    void* d = nullptr;
    check(cudaMalloc(&d, bytes), "cudaMalloc stack");
    allocations_.push_back(d);
    alloc_device_.push_back(a.device);
    uint64_t off = 0;
    for (const DevTensor* t : parts) {
        check(cudaMemcpy(static_cast<uint8_t*>(d) + off, t->data, t->bytes, cudaMemcpyDeviceToDevice), "stack part");
        off += t->bytes;
    }
    bytes_ += bytes;
    bytes_per_device_[(size_t) a.device] += bytes;
    DevTensor r = a;
    r.data = d; r.bytes = bytes; r.ne1 = ne1; r.ne2 = 1;
    for (DevTensor* t : parts) release(*t);
    return r;
}

// frees one uploaded tensor's device buffer (an original that a stacked copy replaced)
void Model::release(DevTensor& t) {
    if (!t.present()) return;
    for (size_t i = 0; i < allocations_.size(); ++i) {
        if (allocations_[i] != t.data) continue;
        check(cudaSetDevice(alloc_device_[i]), "cudaSetDevice");
        check(cudaFree(allocations_[i]), "cudaFree release");
        allocations_.erase(allocations_.begin() + (std::ptrdiff_t) i);
        alloc_device_.erase(alloc_device_.begin() + (std::ptrdiff_t) i);
        bytes_ -= t.bytes;
        bytes_per_device_[(size_t) t.device] -= t.bytes;
        t = DevTensor{};
        return;
    }
    throw std::runtime_error("release: not an uploaded tensor");
}

DevTensor Model::upload(const strata::GgufModel& gg, const std::string& name, int device, bool required) {
    size_t shard = 0;
    const strata::TensorInfo* t = gg.find(name, &shard);
    if (!t) {
        if (required) throw std::runtime_error("GGUF: missing tensor " + name);
        return {};
    }
    const uint64_t bytes = strata::tensor_payload_bytes(*t);
    if (!bytes || !gg.in_bounds(*t, shard)) throw std::runtime_error("GGUF: " + name + " is out of bounds or unsized");
    check(cudaSetDevice(device), "cudaSetDevice");
    void* d = nullptr;
    check(cudaMalloc(&d, bytes), ("cudaMalloc " + name).c_str());
    allocations_.push_back(d);
    alloc_device_.push_back(device);
    check(cudaMemcpy(d, gg.shard(shard).tensor_data(*t), bytes, cudaMemcpyHostToDevice), ("upload " + name).c_str());
    bytes_ += bytes;
    bytes_per_device_[(size_t) device] += bytes;
    DevTensor r;
    r.data = d;
    r.type = t->type;
    r.bytes = bytes;
    r.device = device;
    r.ne0 = (int64_t) t->shape[0];
    r.ne1 = t->shape.size() > 1 ? (int64_t) t->shape[1] : 1;
    r.ne2 = t->shape.size() > 2 ? (int64_t) t->shape[2] : 1;
    return r;
}

ExpertTensor Model::expert(const strata::GgufModel& gg, const std::string& name) {
    size_t shard = 0;
    const strata::TensorInfo* t = gg.find(name, &shard);
    if (!t || t->shape.size() != 3) throw std::runtime_error("GGUF: missing 3-D expert tensor " + name);
    const uint64_t bytes = strata::tensor_payload_bytes(*t);
    if (!bytes || !gg.in_bounds(*t, shard)) throw std::runtime_error("GGUF: " + name + " is out of bounds or unsized");
    ExpertTensor e;
    e.base = gg.shard(shard).tensor_data(*t);
    e.n_in = (int64_t) t->shape[0];
    e.n_out = (int64_t) t->shape[1];
    e.stride = bytes / t->shape[2];
    e.type = t->type;
    e.path = gg.shard(shard).path();
    e.file_off = gg.shard(shard).data_start() + t->offset;
    return e;
}


void Model::attach_mtp(const std::string& path, bool verbose) {
    if (has_mtp_) throw std::runtime_error("glm5: an MTP layer is already attached");
    mtp_gguf_ = std::make_unique<strata::GgufModel>(strata::GgufModel::open(path));
    const int il = hp_.n_layer;
    const std::string b = "blk." + std::to_string(il) + ".";
    if (!mtp_gguf_->find(b + "nextn.eh_proj.weight")) throw std::runtime_error(path + " has no " + b + "nextn.eh_proj.weight (not this model's NextN layer)");
    const int dev = layer_device_.back();
    LayerWeights L;
    L.device = dev;
    auto up = [&](const std::string& n, bool required = true) { return upload(*mtp_gguf_, b + n, dev, required); };
    L.attn_norm = up("attn_norm.weight");
    L.ffn_norm = up("ffn_norm.weight");
    L.wo = up("attn_output.weight");
    L.q_a = up("attn_q_a.weight");
    L.q_a_norm = up("attn_q_a_norm.weight");
    L.q_b = up("attn_q_b.weight");
    L.kv_a = up("attn_kv_a_mqa.weight");
    L.kv_a_norm = up("attn_kv_a_norm.weight");
    L.k_b = up("attn_k_b.weight");
    L.v_b = up("attn_v_b.weight");
    L.idx_k = up("indexer.attn_k.weight");
    L.idx_k_norm = up("indexer.k_norm.weight");
    L.idx_k_norm_b = up("indexer.k_norm.bias");
    L.idx_proj = up("indexer.proj.weight");
    L.idx_q_b = up("indexer.attn_q_b.weight");
    L.idx_gate = up("indexer_compressor_gate.weight");
    L.idx_ape = up("indexer_compressor_ape.weight");
    L.gate_inp = up("ffn_gate_inp.weight");
    L.exp_probs_b = up("exp_probs_b.bias", false);
    {
        DevTensor sg = up("ffn_gate_shexp.weight"), su = up("ffn_up_shexp.weight");
        L.sh_gu = stack_rows({&sg, &su});
    }
    L.sh_down = up("ffn_down_shexp.weight");
    L.exp_gate = expert(*mtp_gguf_, b + "ffn_gate_exps.weight");
    L.exp_up = expert(*mtp_gguf_, b + "ffn_up_exps.weight");
    L.exp_down = expert(*mtp_gguf_, b + "ffn_down_exps.weight");
    L.eh_proj = up("nextn.eh_proj.weight");
    L.enorm = up("nextn.enorm.weight");
    L.hnorm = up("nextn.hnorm.weight");
    L.shared_head_norm = up("nextn.shared_head_norm.weight");
    L.k_b_q8 = q8_copy(L.k_b, dev);
    L.v_b_q8 = q8_copy(L.v_b, dev);
    if (L.k_b.type == 8 || L.v_b.type == 8) {
        // the MTP file stores attn_k_b / attn_v_b as Q8_0; the DSA kernels read them as BF16 rows: dequantize once
        for (DevTensor* t : {&L.k_b, &L.v_b}) {
            if (t->type != 8) continue;
            const int64_t n = t->ne0 * t->ne1 * t->ne2;
            std::vector<uint8_t> raw((size_t) t->bytes);
            check(cudaMemcpy(raw.data(), t->data, (size_t) t->bytes, cudaMemcpyDeviceToHost), "k_b d2h");
            std::vector<uint16_t> bf((size_t) n);
            for (int64_t i = 0; i < n; ++i) {
                const uint8_t* blk = raw.data() + (size_t) (i / 32) * 34;
                uint16_t dh; std::memcpy(&dh, blk, 2);
                // fp16 -> f32
                const uint32_t sgn = (dh & 0x8000u) << 16, ex = (dh >> 10) & 0x1f, mant = dh & 0x3ff;
                float d;
                if (ex == 0) d = (sgn ? -1.f : 1.f) * std::ldexp((float) mant, -24);
                else if (ex == 31) d = mant ? NAN : (sgn ? -INFINITY : INFINITY);
                else { uint32_t u = sgn | ((ex + 112) << 23) | (mant << 13); std::memcpy(&d, &u, 4); }
                const float v = d * (float) (int8_t) blk[2 + (i % 32)];
                uint32_t u; std::memcpy(&u, &v, 4);
                // round to nearest even into bf16
                const uint32_t lsb = (u >> 16) & 1u;
                u += 0x7fffu + lsb;
                bf[(size_t) i] = (uint16_t) (u >> 16);
            }
            void* dptr = nullptr;
            check(cudaSetDevice(dev), "cudaSetDevice");
            check(cudaMalloc(&dptr, bf.size() * 2), "cudaMalloc k_b bf16");
            allocations_.push_back(dptr);
            alloc_device_.push_back(dev);
            check(cudaMemcpy(dptr, bf.data(), bf.size() * 2, cudaMemcpyHostToDevice), "k_b h2d");
            t->data = dptr;
            t->type = 30;
            t->bytes = bf.size() * 2;
            bytes_ += t->bytes;
        }
    }
    layers_.push_back(L);
    layer_device_.push_back(dev);
    hp_.is_recr.push_back(false);
    hp_.clamp_exp.push_back(hp_.clamp_exp.empty() ? 0.f : hp_.clamp_exp.back());
    hp_.clamp_shexp.push_back(hp_.clamp_shexp.empty() ? 0.f : hp_.clamp_shexp.back());
    if (token_embd_.device != dev) token_embd_mtp_ = upload(*gguf_, "token_embd.weight", dev);
    has_mtp_ = true;
    if (verbose) std::fprintf(stderr, "glm5: MTP layer %d attached on device %d (%s)\n", il, dev, path.c_str());
}

}  // namespace strata::glm5
