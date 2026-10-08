// src/glm5/kernels.cu - see include/strata/glm5/kernels.hpp.  Plain FP32 kernels, written for correctness against
// ref/glm5.py and llama.cpp first; the measured fast paths come after they agree.
#include <stdexcept>
#include "strata/glm5/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace strata::glm5 {
namespace {

constexpr int WARP = 32;

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// block-wide sum for blockDim.x <= 1024, result valid in every thread
__device__ float block_sum(float v, float* sh /* >= 32 */) {
    v = warp_sum(v);
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + WARP - 1) / WARP;
    float t = threadIdx.x < nw ? sh[threadIdx.x] : 0.f;
    if (wid == 0) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    return sh[0];
}
__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float(((uint32_t) b) << 16); }
__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
// the DSA caches are stored as Q8_0 rows (ggml blocks: f16 scale + 32 int8 = 34 bytes per 32 values; row pitch n/32*34)
__device__ __forceinline__ int q8_pitch(int n) { return n / 32 * 34; }
__device__ __forceinline__ float q8_at(const uint8_t* row, int j) {
    const uint8_t* b = row + (j >> 5) * 34;
    return __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(b))) * (float) (int8_t) b[2 + (j & 31)];
}
// one Q8_0 block from 32 floats
__device__ __forceinline__ void q8_block(const float* v, uint8_t* blk) {
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
    const float dd = amax / 127.f, id = dd ? 1.f / dd : 0.f;
    *reinterpret_cast<uint16_t*>(blk) = __half_as_ushort(__float2half(dd));
    for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t) (int8_t) rintf(v[i] * id);
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.f + expf(-x)); }

inline cudaStream_t S(void* s) { return static_cast<cudaStream_t>(s); }

// ---------------------------------------------------------------- gemv f32
__global__ void k_f32_gemv(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int n_in,
                           int n_out) {
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    if (row >= n_out) return;
    const float* w = W + (size_t) row * n_in;
    float acc = 0.f;
    if ((n_in & 3) == 0) {
        const float4* w4 = reinterpret_cast<const float4*>(w);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int i = lane; i < n_in / 4; i += WARP) {
            const float4 a = w4[i], b = x4[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (int i = lane; i < n_in; i += WARP) acc += w[i] * x[i];
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

// ---------------------------------------------------------------- norms
__global__ void k_rms_norm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int cols,
                           float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t) blockIdx.x * cols;
    float* yr = y + (size_t) blockIdx.x * cols;
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, sh);
    const float scale = 1.0f / sqrtf(ss / cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) yr[i] = xr[i] * scale * (w ? w[i] : 1.f);
}
__global__ void k_layer_norm(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                             float* __restrict__ y, int n, float eps) {
    __shared__ float sh[32];
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i];
    const float mean = block_sum(s, sh) / n;
    float v = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { const float d = x[i] - mean; v += d * d; }
    const float var = block_sum(v, sh) / n;
    const float scale = 1.0f / sqrtf(var + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = (x[i] - mean) * scale * w[i] + b[i];
}

// ---------------------------------------------------------------- mHC
__global__ void k_mhc_rs(const float* __restrict__ R, float* rs, int n, float eps) {
    __shared__ float sh[32];
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += R[i] * R[i];
    ss = block_sum(ss, sh);
    if (threadIdx.x == 0) *rs = 1.0f / sqrtf(ss / n + eps);
}
constexpr int kMixSlices = 8;
// block (row, slice): the dot of mix row `row` with the normalised R over slice `slice` of the hc*n_embd axis
__global__ void k_mhc_mixes(const float* __restrict__ R, const float* __restrict__ fn, const float* __restrict__ rs,
                            float* __restrict__ part, int n) {
    const int row = blockIdx.x, slice = blockIdx.y;
    __shared__ float sh[32];
    const float* w = fn + (size_t) row * n;
    const int per = n / kMixSlices, i0 = slice * per;
    float acc = 0.f;
    const float s = *rs;
    const float4* w4 = reinterpret_cast<const float4*>(w + i0);
    const float4* r4 = reinterpret_cast<const float4*>(R + i0);
    for (int i = threadIdx.x; i < per / 4; i += blockDim.x) {
        const float4 a = w4[i], b = r4[i];
        acc += a.x * (b.x * s) + a.y * (b.y * s) + a.z * (b.z * s) + a.w * (b.w * s);
    }
    acc = block_sum(acc, sh);
    if (threadIdx.x == 0) part[row * kMixSlices + slice] = acc;
}
__global__ void k_mhc_nonlin(const float* __restrict__ part, const float* __restrict__ scale,
                             const float* __restrict__ base, float* mixes, float* pre, float* post, float* comb, int hc,
                             int iters, float eps) {
    // one warp: lanes 0..hc-1 the pre/post scalars, lanes 0..hc*hc-1 the comb entries e = dst + hc*src
    // (hc == 4: dst = lane & 3, src = lane >> 2; shuffles over xor 1,2 reduce over dst, over xor 4,8 over src)
    const int lane = threadIdx.x;
    if (lane < (2 + hc) * hc) {   // the mix = the sum of its slices, in slice order
        float m = 0.f;
        for (int q = 0; q < kMixSlices; ++q) m += part[lane * kMixSlices + q];
        mixes[lane] = m;
    }
    __syncwarp();
    if (lane < hc) {
        pre[lane] = sigmoidf_(mixes[lane] * scale[0] + base[lane]) + eps;
        post[lane] = 2.0f * sigmoidf_(mixes[hc + lane] * scale[1] + base[hc + lane]);
    }
    const bool on = lane < hc * hc;
    float c = on ? mixes[2 * hc + lane] * scale[2] + base[2 * hc + lane] : -INFINITY;
    // softmax over dst within each src group of 4 lanes
    float m = c;
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    float ex = on ? expf(c - m) : 0.f;
    float sum = ex;
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    c = on ? ex / sum + eps : 0.f;
    auto norm_cols = [&]() {   // for each dst: divide by the sum over src (+eps)
        float t = c;
        t += __shfl_xor_sync(0xffffffffu, t, 4);
        t += __shfl_xor_sync(0xffffffffu, t, 8);
        c = c / (t + eps);
    };
    auto norm_rows = [&]() {   // for each src: divide by the sum over dst (+eps)
        float t = c;
        t += __shfl_xor_sync(0xffffffffu, t, 1);
        t += __shfl_xor_sync(0xffffffffu, t, 2);
        c = c / (t + eps);
    };
    norm_cols();
    for (int it = 1; it < iters; ++it) { norm_rows(); norm_cols(); }
    if (on) comb[lane] = c;   // comb[src*hc + dst]
}
__global__ void k_mhc_pre_apply(const float* __restrict__ R, const float* __restrict__ pre, float* __restrict__ x,
                                int n_embd, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float acc = 0.f;
    for (int h = 0; h < hc; ++h) acc += R[(size_t) h * n_embd + i] * pre[h];
    x[i] = acc;
}
__global__ void k_mhc_post(const float* __restrict__ y, const float* __restrict__ R_in, const float* __restrict__ post,
                           const float* __restrict__ comb, float* __restrict__ R_out, int n_embd, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float r[8];
    for (int s = 0; s < hc; ++s) r[s] = R_in[(size_t) s * n_embd + i];
    const float yi = y[i];
    for (int d = 0; d < hc; ++d) {
        float acc = yi * post[d];
        for (int s = 0; s < hc; ++s) acc += r[s] * comb[s * hc + d];
        R_out[(size_t) d * n_embd + i] = acc;
    }
}

// ---------------------------------------------------------------- KDA
__global__ void k_kda_conv(const float* __restrict__ x, float* __restrict__ st, const float* __restrict__ w,
                           float* __restrict__ out, int d_inner, int d_conv) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d_inner) return;
    float win[8];
    for (int k = 0; k < d_conv - 1; ++k) win[k] = st[(size_t) k * d_inner + c];
    win[d_conv - 1] = x[c];
    float acc = 0.f;
    for (int k = 0; k < d_conv; ++k) acc += w[(size_t) c * d_conv + k] * win[k];
    for (int k = 0; k < d_conv - 1; ++k) st[(size_t) k * d_inner + c] = win[k + 1];
    out[c] = siluf_(acc);
}
__global__ void k_l2_norm_rows(float* __restrict__ x, int S, float eps) {
    const int h = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    float* r = x + (size_t) h * S;
    float ss = 0.f;
    for (int i = lane; i < S; i += WARP) ss += r[i] * r[i];
    ss = warp_sum(ss);
    const float scale = 1.0f / fmaxf(sqrtf(ss), eps);
    for (int i = lane; i < S; i += WARP) r[i] *= scale;
}
__global__ void k_kda_gate(const float* __restrict__ g_in, const float* __restrict__ dt_b, const float* __restrict__ A,
                           float lower, float* __restrict__ g, int n, int S) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = A[i / S];
    const float v = g_in[i] + dt_b[i];
    if (lower > -INFINITY) g[i] = lower * sigmoidf_(-(v * a));
    else g[i] = log1pf(expf(v)) * a;
}
__global__ void k_sigmoid(float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = sigmoidf_(x[i]);
}
// block per head, 4 x S threads: thread (quarter, j) walks i over its quarter of S reading S[h][i][j] (coalesced
// across j); the sums over i are completed across the quarters through shared memory
constexpr int kKdaQuarters = 4;
__global__ void k_kda_step(float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k,
                           const float* __restrict__ v, const float* __restrict__ g, const float* __restrict__ beta,
                           float* __restrict__ o, int S) {
    extern __shared__ float red[];   // kKdaQuarters x S
    const int h = blockIdx.x;
    const int j = threadIdx.x % S, qu = threadIdx.x / S;
    const int i0 = qu * (S / kKdaQuarters), i1 = i0 + S / kKdaQuarters;
    float* st = state + (size_t) h * S * S;
    const float* qh = q + (size_t) h * S;
    const float* kh = k + (size_t) h * S;
    const float* gh = g + (size_t) h * S;
    float sk = 0.f;
    for (int i = i0; i < i1; ++i) {
        const float s = st[(size_t) i * S + j] * expf(gh[i]);
        st[(size_t) i * S + j] = s;
        sk += s * kh[i];
    }
    red[qu * S + j] = sk;
    __syncthreads();
    sk = 0.f;
    for (int t = 0; t < kKdaQuarters; ++t) sk += red[t * S + j];
    const float d = (v[(size_t) h * S + j] - sk) * beta[h];
    float acc = 0.f;
    for (int i = i0; i < i1; ++i) {
        const float s = st[(size_t) i * S + j] + kh[i] * d;
        st[(size_t) i * S + j] = s;
        acc += s * qh[i];
    }
    __syncthreads();
    red[qu * S + j] = acc;
    __syncthreads();
    if (qu == 0) {
        acc = 0.f;
        for (int t = 0; t < kKdaQuarters; ++t) acc += red[t * S + j];
        o[(size_t) h * S + j] = acc * rsqrtf((float) S);
    }
}
__global__ void k_kda_out_gate(const float* __restrict__ o, const float* __restrict__ w, const float* __restrict__ g2,
                               float* __restrict__ normed, float* __restrict__ y, int S, float eps) {
    const int h = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    const float* r = o + (size_t) h * S;
    float ss = 0.f;
    for (int i = lane; i < S; i += WARP) ss += r[i] * r[i];
    ss = warp_sum(ss);
    const float scale = 1.0f / sqrtf(ss / S + eps);
    for (int i = lane; i < S; i += WARP) {
        const float nv = r[i] * scale * w[i];
        normed[(size_t) h * S + i] = nv;
        y[(size_t) h * S + i] = nv * sigmoidf_(g2[(size_t) h * S + i]);
    }
}

// ---------------------------------------------------------------- DSA
// warp per (h, j): q_abs[h][j] = sum_i wk_b[h][j][i] q[h][i]
__global__ void k_dsa_absorb(const uint16_t* __restrict__ wk_b, const float* __restrict__ q, float* __restrict__ q_abs,
                             int kv_lora, int head_k) {
    const int h = blockIdx.y;
    const int j = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    if (j >= kv_lora) return;
    const uint16_t* w = wk_b + ((size_t) h * kv_lora + j) * head_k;
    const float* qh = q + (size_t) h * head_k;
    float acc = 0.f;
    for (int i = lane; i < head_k; i += WARP) acc += bf16f(w[i]) * qh[i];
    acc = warp_sum(acc);
    if (lane == 0) q_abs[(size_t) h * kv_lora + j] = acc;
}
__global__ void k_dsa_pool(const uint8_t* __restrict__ ik, const uint8_t* __restrict__ ig, const float* __restrict__ ape,
                           float* __restrict__ pooled, int j0, int kpool, int D) {
    const int pitch = q8_pitch(D);
    // blockIdx.y = the pool within a range launch (pool p0 + y covers cells j0 + y * kpool ..)
    const int y = blockIdx.y;
    j0 += y * kpool;
    pooled += (size_t) y * D;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float lg[8], m = -INFINITY;
    for (int j = 0; j < kpool; ++j) { lg[j] = q8_at(ig + (size_t) (j0 + j) * pitch, d) + ape[(size_t) j * D + d]; m = fmaxf(m, lg[j]); }
    float sum = 0.f;
    for (int j = 0; j < kpool; ++j) { lg[j] = expf(lg[j] - m); sum += lg[j]; }
    float acc = 0.f;
    for (int j = 0; j < kpool; ++j) acc += lg[j] / sum * q8_at(ik + (size_t) (j0 + j) * pitch, d);
    pooled[d] = acc;
}
// warp per pool; blockIdx.y = the token of the batch (its own n_pool from params[t*8], its own iq / w rows, score row)
__global__ void k_dsa_score(const float* __restrict__ pooled, const float* __restrict__ iq, const float* __restrict__ w,
                            float* __restrict__ score, const int* __restrict__ params, int n_head, int D, int ld_score) {
    const int t = blockIdx.y;
    const int n_pool = params[t * 8 + 0];
    iq += (size_t) t * n_head * D;
    w += (size_t) t * n_head;
    score += (size_t) t * ld_score;
    const int p = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    if (p >= n_pool) return;
    const float* pk = pooled + (size_t) p * D;
    float total = 0.f;
    for (int h = 0; h < n_head; ++h) {
        const float* qh = iq + (size_t) h * D;
        float acc = 0.f;
        for (int i = lane; i < D; i += WARP) acc += qh[i] * pk[i];
        acc = warp_sum(acc);
        total += fmaxf(acc, 0.f) * w[h];
    }
    if (lane == 0) score[p] = total;
}

// ---------------------------------------------------------------- DSA selection on the device
// One block of 1024 threads per token (blockIdx.x): the n_top largest of score[0..n_pool) by a 4-pass radix select
// on the order-preserving uint32 image of the float, then the selected pools in ascending order (ties at the
// threshold: lower index first, as a stable descending sort followed by an index sort gives), expanded to kpool
// cells each, then the tail cells.  Writes sel[0..n_sel) and params[4] = n_sel.  Every per-token argument lives in
// device memory, so a captured graph replays it.
__device__ __forceinline__ uint32_t f32_key(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__global__ void k_dsa_select(const float* __restrict__ score, int* __restrict__ params /* per token: n_pool, n_top, n_kv, n_tail, [4] = n_sel out */,
                             int kpool, int32_t* __restrict__ sel, int ld_score, int n_sel_max) {
    __shared__ int hist[256];
    __shared__ int scan[1024];
    __shared__ uint32_t s_prefix;
    __shared__ int s_remaining, s_base;
    const int t = blockIdx.x;
    score += (size_t) t * ld_score;
    params += t * 8;
    sel += (size_t) t * n_sel_max;
    int* n_sel_out = params + 4;
    const int n_pool = params[0], n_top = params[1], n_kv = params[2], n_tail = params[3];
    const int tid = threadIdx.x, nt = blockDim.x;
    uint32_t prefix = 0;      // the bits decided so far (high to low)
    int remaining = n_top;    // how many of the candidates matching the prefix we still want
    if (n_top > 0 && n_pool > 0) {
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            for (int b = tid; b < 256; b += nt) hist[b] = 0;
            __syncthreads();
            for (int p = tid; p < n_pool; p += nt) {
                const uint32_t key = f32_key(score[p]);
                if (pass == 0 || (key >> (shift + 8)) == (prefix >> (shift + 8)))
                    atomicAdd(&hist[(key >> shift) & 255], 1);
            }
            __syncthreads();
            if (tid == 0) {
                // walk the bins from the top: the bin holding the remaining-th largest decides the next 8 bits
                int acc = 0, b = 255;
                for (; b > 0; --b) {
                    if (acc + hist[b] >= remaining) break;
                    acc += hist[b];
                }
                s_prefix = prefix | ((uint32_t) b << shift);
                s_remaining = remaining - acc;
            }
            __syncthreads();
            prefix = s_prefix;
            remaining = s_remaining;
            __syncthreads();
        }
    }
    const uint32_t thr = prefix;   // the n_top-th largest key; `remaining` of the pools equal to it are taken
    // pass A: every pool with key > thr, in index order; pass B: the first `remaining` pools with key == thr
    if (tid == 0) s_base = 0;
    __syncthreads();
    for (int round = 0; round < 2; ++round) {
        for (int p0 = 0; p0 < n_pool; p0 += nt) {
            const int p = p0 + tid;
            int flag = 0;
            if (p < n_pool && n_top > 0) {
                const uint32_t key = f32_key(score[p]);
                flag = round == 0 ? key > thr : key == thr;
            }
            scan[tid] = flag;
            __syncthreads();
            for (int o = 1; o < nt; o <<= 1) {        // inclusive scan
                const int v = tid >= o ? scan[tid - o] : 0;
                __syncthreads();
                scan[tid] += v;
                __syncthreads();
            }
            const int total = scan[nt - 1];
            if (flag && (round == 0 || scan[tid] - 1 < remaining)) {
                const int rank = s_base + scan[tid] - 1;
                for (int j = 0; j < kpool; ++j) sel[rank * kpool + j] = p * kpool + j;
            }
            __syncthreads();
            if (tid == 0) {
                s_base += round == 0 ? total : min(total, remaining);
                if (round == 1) s_remaining = remaining - min(total, remaining);
            }
            __syncthreads();
            if (round == 1) remaining = s_remaining;
            __syncthreads();
        }
    }
    const int n_cells = s_base * kpool;
    for (int j = tid; j < n_tail; j += nt) sel[n_cells + j] = n_kv - n_tail + j;
    if (tid == 0) *n_sel_out = n_cells + n_tail;
}

// warp per (h, c); blockIdx.z = the token of the batch
// a selected entry: a cache cell (>= 0) or, in LoD mode, a page summary (-(p+1): row p of `means`, logit + bias)
__device__ __forceinline__ const uint8_t* sel_row(const uint8_t* kvc, const uint8_t* means, int32_t c, int pitch) {
    return c >= 0 ? kvc + (size_t) c * pitch : means + (size_t) (-c - 1) * pitch;
}
__global__ void k_dsa_scores_warp(const uint8_t* __restrict__ kvc, const uint8_t* __restrict__ means, float bias, const int32_t* __restrict__ sel,
                             const int* __restrict__ params, const float* __restrict__ q_abs, float scale, float* __restrict__ scores, int kv_lora, int ld,
                             int n_head) {
    const int t = blockIdx.z;
    const int n_sel = params[t * 8 + 4];
    sel += (size_t) t * ld;
    q_abs += (size_t) t * n_head * kv_lora;
    scores += (size_t) t * n_head * ld;
    const int h = blockIdx.y;
    const int c = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    if (c >= n_sel) return;
    const int32_t sc = sel[c];
    const uint8_t* k = sel_row(kvc, means, sc, q8_pitch(kv_lora));
    const float* qh = q_abs + (size_t) h * kv_lora;
    float acc = 0.f;
    for (int i = lane; i < kv_lora; i += WARP) acc += qh[i] * q8_at(k, i);
    acc = warp_sum(acc);
    if (lane == 0) scores[(size_t) h * ld + c] = acc * scale + (sc < 0 ? bias : 0.f);
}
// block per token: the selected rows in tiles of 8 in shared memory, thread (head, tile cell) = one dot product.
// Each selected row is read once per token instead of once per head.
__global__ void k_dsa_scores(const uint8_t* __restrict__ kvc, const uint8_t* __restrict__ means, float bias, const int32_t* __restrict__ sel,
                             const int* __restrict__ params, const float* __restrict__ q_abs, float scale, float* __restrict__ scores, int kv_lora, int ld, int n_head) {
    extern __shared__ float krow[];   // [8][kv_lora]
    const int t = blockIdx.x;
    const int n_sel = params[t * 8 + 4];
    sel += (size_t) t * ld;
    q_abs += (size_t) t * n_head * kv_lora;
    scores += (size_t) t * n_head * ld;
    const int h = threadIdx.x / 8, cc = threadIdx.x % 8;
    const float* qh = q_abs + (size_t) h * kv_lora;
    for (int c0 = 0; c0 < n_sel; c0 += 8) {
        __syncthreads();
        for (int i = threadIdx.x; i < 8 * kv_lora; i += blockDim.x) {
            const int r = i / kv_lora, c = c0 + r;
            krow[i] = c < n_sel ? q8_at(sel_row(kvc, means, sel[c], q8_pitch(kv_lora)), i - r * kv_lora) : 0.f;
        }
        __syncthreads();
        const int c = c0 + cc;
        if (c < n_sel) {
            const float* kr = krow + cc * kv_lora;
            float acc = 0.f;
            for (int i = 0; i < kv_lora; ++i) acc += qh[i] * kr[i];
            scores[(size_t) h * ld + c] = acc * scale + (sel[c] < 0 ? bias : 0.f);
        }
    }
}
// block per (head, token): softmax over the first n_sel of a row of stride ld, in place
__global__ void k_softmax_rows(float* __restrict__ s, const int* __restrict__ params, int ld, int n_head) {
    __shared__ float sh[32];
    const int t = blockIdx.y;
    const int n = params[t * 8 + 4];
    float* r = s + ((size_t) t * n_head + blockIdx.x) * ld;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, r[i]);
    m = warp_max(m);
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    if (lane == 0) sh[wid] = m;
    __syncthreads();
    const int nw = (blockDim.x + WARP - 1) / WARP;
    float tt = threadIdx.x < nw ? sh[threadIdx.x] : -INFINITY;
    if (wid == 0) tt = warp_max(tt);
    if (threadIdx.x == 0) sh[0] = tt;
    __syncthreads();
    m = sh[0];
    __syncthreads();
    float sum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { r[i] = expf(r[i] - m); sum += r[i]; }
    sum = block_sum(sum, sh);
    for (int i = threadIdx.x; i < n; i += blockDim.x) r[i] /= sum;
}
// block per (head, token), thread per j: ctx[t][h][j] = sum_c p[t][h][c] kvc[sel[t][c]][j]
__global__ void k_dsa_ctx_warp(const uint8_t* __restrict__ kvc, const uint8_t* __restrict__ means, const int32_t* __restrict__ sel, const int* __restrict__ params,
                          const float* __restrict__ p, float* __restrict__ ctx, int kv_lora, int ld, int n_head) {
    const int t = blockIdx.y;
    const int n_sel = params[t * 8 + 4];
    sel += (size_t) t * ld;
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    if (j >= kv_lora) return;
    const float* ph = p + ((size_t) t * n_head + h) * ld;
    float acc = 0.f;
    const int pitch = q8_pitch(kv_lora);
    for (int c = 0; c < n_sel; ++c) acc += ph[c] * q8_at(sel_row(kvc, means, sel[c], pitch), j);
    ctx[((size_t) t * n_head + h) * kv_lora + j] = acc;
}
// block per token, thread per j, every head at once: ctx[t][h][j] = sum_c p[t][h][c] kvc[sel[c]][j]; each selected
// row is read once per token
template <int NH>
__global__ void k_dsa_ctx(const uint8_t* __restrict__ kvc, const uint8_t* __restrict__ means, const int32_t* __restrict__ sel, const int* __restrict__ params,
                          const float* __restrict__ p, float* __restrict__ ctx, int kv_lora, int ld) {
    const int t = blockIdx.x;
    const int n_sel = params[t * 8 + 4];
    sel += (size_t) t * ld;
    p += (size_t) t * NH * ld;
    ctx += (size_t) t * NH * kv_lora;
    const int j = threadIdx.x;
    if (j >= kv_lora) return;
    float acc[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) acc[h] = 0.f;
    for (int c = 0; c < n_sel; ++c) {
        const float kv = q8_at(sel_row(kvc, means, sel[c], q8_pitch(kv_lora)), j);
#pragma unroll
        for (int h = 0; h < NH; ++h) acc[h] += p[(size_t) h * ld + c] * kv;
    }
#pragma unroll
    for (int h = 0; h < NH; ++h) ctx[(size_t) h * kv_lora + j] = acc[h];
}
// warp per (h, v): out[h][v] = sum_j wv_b[h][v][j] ctx[h][j]
__global__ void k_dsa_expand(const uint16_t* __restrict__ wv_b, const float* __restrict__ ctx, float* __restrict__ out,
                             int kv_lora, int head_v) {
    const int h = blockIdx.y;
    const int v = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    if (v >= head_v) return;
    const uint16_t* w = wv_b + ((size_t) h * head_v + v) * kv_lora;
    const float* c = ctx + (size_t) h * kv_lora;
    float acc = 0.f;
    for (int j = lane; j < kv_lora; j += WARP) acc += bf16f(w[j]) * c[j];
    acc = warp_sum(acc);
    if (lane == 0) out[(size_t) h * head_v + v] = acc;
}

// ---------------------------------------------------------------- the MoE doorbell (host-mapped pinned memory)
// publish: the routed ids and the token counter into host-visible memory, fenced so the host sees the ids first
__global__ void k_moe_publish(const int32_t* __restrict__ ids, int k, const uint32_t* __restrict__ tok,
                              int32_t* __restrict__ h_ids, volatile uint32_t* h_published, const float* __restrict__ x, int nx,
                              float* __restrict__ h_x) {
    for (int i = threadIdx.x; i < k; i += blockDim.x) h_ids[i] = ids[i];
    for (int i = threadIdx.x; i < nx; i += blockDim.x) h_x[i] = x[i];   // the MoE inputs for the CPU service (nx = 0: none)
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) { *h_published = *tok; __threadfence_system(); }
}
// select: the dispatch tables of this layer's routed experts, hits first.  With m the compacted position (hits
// 0..nh-1, then the misses), the tables are
//   tabA [2 nh]: gate(m), up(m) of the hits, rowsA = m, K + m           (computed before the wait)
//   tabB [2 nm]: gate, up of the misses (their staging slot j), rowsB     (after the wait)
//   tabC [K]   : down(m) for every m                                    (after the wait)
//   wperm[m]   : the router weight of the expert at position m;  cnt = {2 nh, 2 nm}
// One thread does it all: k <= 8.
__global__ void k_moe_select(const int32_t* __restrict__ ids, const float* __restrict__ wts, int k,
                             const int32_t* __restrict__ slot_of, const uint8_t* __restrict__ slots, uint64_t blob,
                             uint64_t gate_bytes, uint64_t up_bytes, const uint8_t* __restrict__ stage, uint64_t stage_bytes,
                             const void** __restrict__ tabA, const void** __restrict__ tabB, const void** __restrict__ tabC,
                             int* __restrict__ rowsA, int* __restrict__ rowsB, float* __restrict__ wperm, int* __restrict__ cnt) {
    if (threadIdx.x != 0) return;
    int nh = 0, nm = 0;
    int order[16];
    for (int j = 0; j < k; ++j) if (slot_of[ids[j]] >= 0) order[nh++] = j;
    for (int j = 0; j < k; ++j) if (slot_of[ids[j]] < 0) order[nh + nm++] = j;
    for (int m = 0; m < k; ++m) {
        const int j = order[m];
        const int32_t slot = slot_of[ids[j]];
        const uint8_t* b = slot >= 0 ? slots + (uint64_t) slot * blob : stage + (uint64_t) j * stage_bytes;
        if (m < nh) {
            tabA[m] = b;
            tabA[nh + m] = b + gate_bytes;
            rowsA[m] = m;
            rowsA[nh + m] = k + m;
        } else {
            const int t = m - nh;
            tabB[t] = b;
            tabB[nm + t] = b + gate_bytes;
            rowsB[t] = m;
            rowsB[nm + t] = k + m;
        }
        tabC[m] = b + gate_bytes + up_bytes;
        wperm[m] = wts[j];
    }
    cnt[0] = 2 * nh;
    cnt[1] = 2 * nm;
}
// wait: spin until the host has served this token's misses for this layer (flag == tok)
__global__ void k_moe_wait(const volatile uint32_t* h_flag, const uint32_t* __restrict__ tok) {
    if (threadIdx.x != 0) return;
    const uint32_t want = *tok;
    while (*h_flag != want) { __builtin_amdgcn_s_sleep(1); }
    __threadfence_system();
}
__global__ void k_u32_inc(uint32_t* v) { if (threadIdx.x == 0) *v += 1u; }
// the lookahead prefetch's placement: wait until the host has issued this layer's prefetch for token `tok`
// (pf_flag == tok), then apply its n table updates: slot_of[out] = -1, slot_of[e] = slot
__global__ void k_pf_apply(const volatile uint32_t* pf_flag, const uint32_t* __restrict__ tok, const volatile int* pf_n,
                           const int* __restrict__ pf_e, const int* __restrict__ pf_slot, const int* __restrict__ pf_out,
                           int32_t* __restrict__ slot_of) {
    const uint32_t want = *tok;
    if (threadIdx.x == 0) { while (*pf_flag != want) { __builtin_amdgcn_s_sleep(1); } }
    __syncthreads();
    __threadfence_system();
    const int n = *pf_n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int out = pf_out[i];
        if (out >= 0) slot_of[out] = -1;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) slot_of[pf_e[i]] = pf_slot[i];
}
__global__ void k_swap_slot(int32_t* __restrict__ slot_of, int e_out, int e_in, int32_t slot) {
    if (threadIdx.x != 0) return;
    if (e_out >= 0) slot_of[e_out] = -1;
    slot_of[e_in] = slot;
}

// ---------------------------------------------------------------- MoE
__global__ void k_router(const float* __restrict__ logits, const float* __restrict__ bias, int n_expert, int k,
                         bool norm, float scale, int32_t* __restrict__ ids, float* __restrict__ wts,
                         float* __restrict__ probs_out) {
    extern __shared__ float sh[];
    float* probs = sh;              // n_expert
    float* sel = sh + n_expert;     // n_expert
    __shared__ float red_v[32];
    __shared__ int red_i[32];
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        const float p = sigmoidf_(logits[e]);
        probs[e] = p;
        sel[e] = bias ? p + bias[e] : p;
        if (probs_out) probs_out[e] = p;
    }
    __syncthreads();
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, nw = (blockDim.x + WARP - 1) / WARP;
    __shared__ float wsum;
    if (threadIdx.x == 0) wsum = 0.f;
    for (int t = 0; t < k; ++t) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int e = threadIdx.x; e < n_expert; e += blockDim.x)
            if (sel[e] > bv || (sel[e] == bv && e < bi)) { bv = sel[e]; bi = e; }
        for (int o = WARP / 2; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { red_v[wid] = bv; red_i[wid] = bi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < nw; ++w)
                if (red_v[w] > bv || (red_v[w] == bv && red_i[w] < bi)) { bv = red_v[w]; bi = red_i[w]; }
            ids[t] = bi;
            wts[t] = probs[bi];
            wsum += probs[bi];
            sel[bi] = -INFINITY;
        }
        __syncthreads();
    }
    if (threadIdx.x < k) {
        float w = wts[threadIdx.x];
        if (norm) w /= fmaxf(wsum, 6.103515625e-5f);
        wts[threadIdx.x] = w * scale;
    }
}
__global__ void k_swiglu_clamp(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ h, int n,
                               float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (limit > 0.f) h[i] = siluf_(fminf(g[i], limit)) * fminf(fmaxf(u[i], -limit), limit);
    else h[i] = siluf_(g[i]) * u[i];
}
__global__ void k_weighted_sum(const float* __restrict__ y, const float* __restrict__ w, int k, int n,
                               float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    for (int j = 0; j < k; ++j) acc += w[j] * y[(size_t) j * n + i];
    out[i] = acc;
}
__global__ void k_axpy_dev(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ a, int idx,
                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a[idx] * x[i];
}
__global__ void k_add(float* __restrict__ y, const float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
__global__ void k_scale(float* __restrict__ x, int n, float s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}
__global__ void k_mean_streams(const float* __restrict__ R, float* __restrict__ h, int n_embd, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float acc = 0.f;
    for (int s = 0; s < hc; ++s) acc += R[(size_t) s * n_embd + i];
    h[i] = acc / hc;
}
__global__ void k_argmax1(const float* __restrict__ x, int n, int* __restrict__ idx, float* __restrict__ val) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (x[i] > bv) { bv = x[i]; bi = i; }
    sv[threadIdx.x] = bv;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float ov = sv[threadIdx.x + s];
            const int oi = si[threadIdx.x + s];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) {
                sv[threadIdx.x] = ov;
                si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { idx[blockIdx.x] = si[0]; val[blockIdx.x] = sv[0]; }
}
__global__ void k_argmax2(const int* __restrict__ idx, const float* __restrict__ val, int n, int32_t* out) {
    if (threadIdx.x != 0) return;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = 0; i < n; ++i)
        if (val[i] > bv || (val[i] == bv && idx[i] < bi)) { bv = val[i]; bi = idx[i]; }
    *out = bi;
}
__global__ void k_dequant_q8_0_row(const uint8_t* __restrict__ w, int row, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t* blk = w + (size_t) row * (n / 32) * 34 + (size_t) (i / 32) * 34;
    half d;
    __builtin_memcpy(&d, blk, 2);
    const int8_t qv = (int8_t) blk[2 + (i % 32)];
    out[i] = __half2float(d) * (float) qv;
}

inline unsigned blocks_for(int n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

}  // namespace

int mhc_mix_slices() { return kMixSlices; }

void f32_gemv(const float* W, const float* x, float* y, int n_in, int n_out, void* s) {
    k_f32_gemv<<<blocks_for(n_out, 8), 256, 0, S(s)>>>(W, x, y, n_in, n_out);
}
void rms_norm(const float* x, const float* w, float* y, int rows, int cols, float eps, void* s) {
    k_rms_norm<<<rows, 256, 0, S(s)>>>(x, w, y, cols, eps);
}
void layer_norm(const float* x, const float* w, const float* b, float* y, int n, float eps, void* s) {
    k_layer_norm<<<1, 128, 0, S(s)>>>(x, w, b, y, n, eps);
}
void mhc_mix(const float* R, const float* fn, const float* scale, const float* base, float* mixes, float* part, float* rs,
             float* pre, float* post, float* comb, int n_embd, int hc, int iters, float eps_rms, float eps_hc,
             void* s) {
    const int n = hc * n_embd;
    k_mhc_rs<<<1, 1024, 0, S(s)>>>(R, rs, n, eps_rms);
    k_mhc_mixes<<<dim3((2 + hc) * hc, kMixSlices), 128, 0, S(s)>>>(R, fn, rs, part, n);
    k_mhc_nonlin<<<1, 32, 0, S(s)>>>(part, scale, base, mixes, pre, post, comb, hc, iters, eps_hc);
}
void mhc_pre_apply(const float* R, const float* pre, float* x, int n_embd, int hc, void* s) {
    k_mhc_pre_apply<<<blocks_for(n_embd), 256, 0, S(s)>>>(R, pre, x, n_embd, hc);
}
void mhc_post(const float* y, const float* R_in, const float* post, const float* comb, float* R_out, int n_embd,
              int hc, void* s) {
    k_mhc_post<<<blocks_for(n_embd), 256, 0, S(s)>>>(y, R_in, post, comb, R_out, n_embd, hc);
}
void kda_conv_silu(const float* x, float* conv_state, const float* w, float* out, int d_inner, int d_conv, void* s) {
    k_kda_conv<<<blocks_for(d_inner), 256, 0, S(s)>>>(x, conv_state, w, out, d_inner, d_conv);
}
void l2_norm_rows(float* x, int n_head, int S_, float eps, void* s) {
    k_l2_norm_rows<<<blocks_for(n_head, 8), 256, 0, S(s)>>>(x, S_, eps);
}
void kda_gate(const float* g_in, const float* dt_b, const float* A, float lower, float* g, int n_head, int S_,
              void* s) {
    k_kda_gate<<<blocks_for(n_head * S_), 256, 0, S(s)>>>(g_in, dt_b, A, lower, g, n_head * S_, S_);
}
void sigmoid_inplace(float* x, int n, void* s) { k_sigmoid<<<blocks_for(n), 256, 0, S(s)>>>(x, n); }
void kda_step(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
              float* o, int n_head, int S_, void* s) {
    k_kda_step<<<n_head, kKdaQuarters * S_, kKdaQuarters * S_ * sizeof(float), S(s)>>>(state, q, k, v, g, beta, o, S_);
}
void kda_out_gate(const float* o, const float* w, const float* g2, float* normed, float* y, int n_head, int S_,
                  float eps, void* s) {
    k_kda_out_gate<<<blocks_for(n_head, 8), 256, 0, S(s)>>>(o, w, g2, normed, y, S_, eps);
}
void dsa_absorb(const uint16_t* wk_b, const float* q, float* q_abs, int n_head, int kv_lora, int head_k, void* s) {
    k_dsa_absorb<<<dim3(blocks_for(kv_lora, 8), n_head), 256, 0, S(s)>>>(wk_b, q, q_abs, kv_lora, head_k);
}
void dsa_pool(const uint8_t* ik_cache, const uint8_t* ig_cache, const float* ape, float* pooled, int j0, int n_pools, int kpool, int D,
              void* s) {
    if (n_pools <= 0) return;
    k_dsa_pool<<<dim3(blocks_for(D), n_pools), 256, 0, S(s)>>>(ik_cache, ig_cache, ape, pooled, j0, kpool, D);
}
void dsa_score(const float* pooled, const float* iq, const float* w, float* score, const int* params, int n_pool_max, int TB, int n_head, int D,
               int ld_score, void* s) {
    if (n_pool_max <= 0) return;
    k_dsa_score<<<dim3(blocks_for(n_pool_max, 8), TB), 256, 0, S(s)>>>(pooled, iq, w, score, params, n_head, D, ld_score);
}
void dsa_select(const float* score, int* params, int TB, int kpool, int32_t* sel, int ld_score, int n_sel_max, void* s) {
    k_dsa_select<<<TB, 1024, 0, S(s)>>>(score, params, kpool, sel, ld_score, n_sel_max);
}
void dsa_attend(const uint8_t* kvc_cache, const uint8_t* means, float bias, const int32_t* sel, const int* params, int TB, int n_sel_max, const float* q_abs, float scale,
                float* scores, float* ctx, int n_head, int kv_lora, void* s) {
    const bool wide = TB >= 16 && n_head * 8 <= 1024 && kv_lora <= 1024 && (n_head == 16 || n_head == 32 || n_head == 64);
    if (wide) {
        k_dsa_scores<<<TB, n_head * 8, (size_t) 8 * kv_lora * sizeof(float), S(s)>>>(kvc_cache, means, bias, sel, params, q_abs, scale, scores, kv_lora, n_sel_max, n_head);
    } else {
        k_dsa_scores_warp<<<dim3(blocks_for(n_sel_max, 8), n_head, TB), 256, 0, S(s)>>>(kvc_cache, means, bias, sel, params, q_abs, scale, scores, kv_lora, n_sel_max, n_head);
    }
    k_softmax_rows<<<dim3(n_head, TB), 256, 0, S(s)>>>(scores, params, n_sel_max, n_head);
    if (wide) {
        if (n_head == 16) k_dsa_ctx<16><<<TB, kv_lora, 0, S(s)>>>(kvc_cache, means, sel, params, scores, ctx, kv_lora, n_sel_max);
        else if (n_head == 32) k_dsa_ctx<32><<<TB, kv_lora, 0, S(s)>>>(kvc_cache, means, sel, params, scores, ctx, kv_lora, n_sel_max);
        else k_dsa_ctx<64><<<TB, kv_lora, 0, S(s)>>>(kvc_cache, means, sel, params, scores, ctx, kv_lora, n_sel_max);
    } else {
        k_dsa_ctx_warp<<<dim3(n_head, TB), kv_lora, 0, S(s)>>>(kvc_cache, means, sel, params, scores, ctx, kv_lora, n_sel_max, n_head);
    }
}
void dsa_expand(const uint16_t* wv_b, const float* ctx, float* out, int n_head, int kv_lora, int head_v, void* s) {
    k_dsa_expand<<<dim3(blocks_for(head_v, 8), n_head), 256, 0, S(s)>>>(wv_b, ctx, out, kv_lora, head_v);
}
void router_sigmoid_topk(const float* logits, const float* bias, int n_expert, int k, bool norm, float scale,
                         int32_t* ids, float* weights, float* probs, void* s) {
    k_router<<<1, 256, 2 * n_expert * sizeof(float), S(s)>>>(logits, bias, n_expert, k, norm, scale, ids, weights,
                                                              probs);
}
void swiglu_clamp(const float* g, const float* u, float* h, int n, float limit, void* s) {
    k_swiglu_clamp<<<blocks_for(n), 256, 0, S(s)>>>(g, u, h, n, limit);
}
void moe_publish(const int32_t* ids, int k, const uint32_t* tok, int32_t* h_ids, uint32_t* h_published, const float* x, int nx, float* h_x, void* s) {
    k_moe_publish<<<1, 256, 0, S(s)>>>(ids, k, tok, h_ids, h_published, x, nx, h_x);
}
// the layer's doorbell flag written from the copy stream (ordered after the copies queued before it; the host thread
// does not block, unlike a host-to-host hipMemcpyAsync)
__global__ void k_write_flag(volatile uint32_t* flag, uint32_t v) { *flag = v; __threadfence_system(); }
void write_flag(uint32_t* flag_dev, uint32_t v, void* s) { k_write_flag<<<1, 1, 0, S(s)>>>(flag_dev, v); }
// the T tokens' MoE inputs to the host (the CPU computes the misses on them); before moe_publish on the same stream
__global__ void k_publish_x(const float* __restrict__ x, float* __restrict__ h_x, int n) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) h_x[(size_t) t * n + i] = x[(size_t) t * n + i];
}
void publish_x(const float* x, int T, int n, float* h_x, void* s) {
    k_publish_x<<<dim3(16, T), 256, 0, S(s)>>>(x, h_x, n);
}
void moe_select(const int32_t* ids, const float* wts, int k, const int32_t* slot_of, const uint8_t* slots, uint64_t blob,
                uint64_t gate_bytes, uint64_t up_bytes, const uint8_t* stage, uint64_t stage_bytes, const void** tabA,
                const void** tabB, const void** tabC, int* rowsA, int* rowsB, float* wperm, int* cnt, void* s) {
    k_moe_select<<<1, 32, 0, S(s)>>>(ids, wts, k, slot_of, slots, blob, gate_bytes, up_bytes, stage, stage_bytes, tabA, tabB, tabC,
                                     rowsA, rowsB, wperm, cnt);
}
void moe_wait(const uint32_t* h_flag, const uint32_t* tok, void* s) { k_moe_wait<<<1, 32, 0, S(s)>>>(h_flag, tok); }
void u32_inc(uint32_t* v, void* s) { k_u32_inc<<<1, 32, 0, S(s)>>>(v); }
void pf_apply(const uint32_t* pf_flag, const uint32_t* tok, const int* pf_n, const int* pf_e, const int* pf_slot, const int* pf_out,
              int32_t* slot_of, void* s) {
    k_pf_apply<<<1, 64, 0, S(s)>>>(pf_flag, tok, pf_n, pf_e, pf_slot, pf_out, slot_of);
}
void swap_slot(int32_t* slot_of, int e_out, int e_in, int32_t slot, void* s) { k_swap_slot<<<1, 32, 0, S(s)>>>(slot_of, e_out, e_in, slot); }
void weighted_sum(const float* y, const float* w, int k, int n, float* out, void* s) {
    k_weighted_sum<<<blocks_for(n), 256, 0, S(s)>>>(y, w, k, n, out);
}
void axpy_dev(float* y, const float* x, const float* a, int idx, int n, void* s) {
    k_axpy_dev<<<blocks_for(n), 256, 0, S(s)>>>(y, x, a, idx, n);
}
void add_inplace(float* y, const float* x, int n, void* s) { k_add<<<blocks_for(n), 256, 0, S(s)>>>(y, x, n); }
void scale_inplace(float* x, int n, float sc, void* s) { k_scale<<<blocks_for(n), 256, 0, S(s)>>>(x, n, sc); }
void mean_streams(const float* R, float* h, int n_embd, int hc, void* s) {
    k_mean_streams<<<blocks_for(n_embd), 256, 0, S(s)>>>(R, h, n_embd, hc);
}
void argmax(const float* x, int n, int32_t* out, int* scratch_idx, float* scratch_val, void* s) {
    constexpr int B = 128;
    k_argmax1<<<B, 1024, 0, S(s)>>>(x, n, scratch_idx, scratch_val);
    k_argmax2<<<1, 32, 0, S(s)>>>(scratch_idx, scratch_val, B, out);
}
void dequant_q8_0_row(const uint8_t* w, int row, float* out, int n, void* s) {
    k_dequant_q8_0_row<<<blocks_for(n), 256, 0, S(s)>>>(w, row, out, n);
}


// ================================================================ multi-token (verify window) kernels
// Token-batched twins of the kernels above: the same per-token arithmetic, the token axis outermost.  T <= kTmax.
namespace {
constexpr int kTmax = 2048;   // the window; the KDA snapshots (commit rollback) cover the first kSnapMax tokens only

__global__ void k_f32_gemv_t(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int n_in,
                             int n_out) {
    const int t = blockIdx.y;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    if (row >= n_out) return;
    const float* w = W + (size_t) row * n_in;
    const float* xt = x + (size_t) t * n_in;
    float acc = 0.f;
    if ((n_in & 3) == 0) {
        const float4* w4 = reinterpret_cast<const float4*>(w);
        const float4* x4 = reinterpret_cast<const float4*>(xt);
        for (int i = lane; i < n_in / 4; i += WARP) {
            const float4 a = w4[i], b = x4[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (int i = lane; i < n_in; i += WARP) acc += w[i] * xt[i];
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) t * n_out + row] = acc;
}
__global__ void k_layer_norm_t(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                               float* __restrict__ y, int n, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i];
    const float mean = block_sum(s, sh) / n;
    float v = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { const float d = xr[i] - mean; v += d * d; }
    const float var = block_sum(v, sh) / n;
    const float scale = 1.0f / sqrtf(var + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = (xr[i] - mean) * scale * w[i] + b[i];
}
// mHC over T tokens: rs[t], part[t][24][slices], mixes[t][24], pre/post[t][hc], comb[t][hc*hc]
__global__ void k_mhc_rs_t(const float* __restrict__ R, float* rs, int n, float eps) {
    __shared__ float sh[32];
    const float* r = R + (size_t) blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += r[i] * r[i];
    ss = block_sum(ss, sh);
    if (threadIdx.x == 0) rs[blockIdx.x] = 1.0f / sqrtf(ss / n + eps);
}
__global__ void k_mhc_mixes_t(const float* __restrict__ R, const float* __restrict__ fn, const float* __restrict__ rs,
                              float* __restrict__ part, int n, int n_mix) {
    const int row = blockIdx.x, slice = blockIdx.y, t = blockIdx.z;
    __shared__ float sh[32];
    const float* w = fn + (size_t) row * n;
    const float* Rt = R + (size_t) t * n;
    const int per = n / kMixSlices, i0 = slice * per;
    float acc = 0.f;
    const float s = rs[t];
    const float4* w4 = reinterpret_cast<const float4*>(w + i0);
    const float4* r4 = reinterpret_cast<const float4*>(Rt + i0);
    for (int i = threadIdx.x; i < per / 4; i += blockDim.x) {
        const float4 a = w4[i], b = r4[i];
        acc += a.x * (b.x * s) + a.y * (b.y * s) + a.z * (b.z * s) + a.w * (b.w * s);
    }
    acc = block_sum(acc, sh);
    if (threadIdx.x == 0) part[((size_t) t * n_mix + row) * kMixSlices + slice] = acc;
}
__global__ void k_mhc_nonlin_t(const float* __restrict__ part, const float* __restrict__ scale,
                               const float* __restrict__ base, float* mixes, float* pre, float* post, float* comb, int hc,
                               int iters, float eps) {
    const int t = blockIdx.x;
    const int n_mix = (2 + hc) * hc;
    const float* pt = part + (size_t) t * n_mix * kMixSlices;
    float* mt = mixes + (size_t) t * n_mix;
    const int lane = threadIdx.x;
    if (lane < n_mix) {
        float m = 0.f;
        for (int q = 0; q < kMixSlices; ++q) m += pt[lane * kMixSlices + q];
        mt[lane] = m;
    }
    __syncwarp();
    if (lane < hc) {
        pre[t * hc + lane] = sigmoidf_(mt[lane] * scale[0] + base[lane]) + eps;
        post[t * hc + lane] = 2.0f * sigmoidf_(mt[hc + lane] * scale[1] + base[hc + lane]);
    }
    const bool on = lane < hc * hc;
    float c = on ? mt[2 * hc + lane] * scale[2] + base[2 * hc + lane] : -INFINITY;
    float m = c;
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    float ex = on ? expf(c - m) : 0.f;
    float sum = ex;
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    c = on ? ex / sum + eps : 0.f;
    auto norm_cols = [&]() {
        float tt = c;
        tt += __shfl_xor_sync(0xffffffffu, tt, 4);
        tt += __shfl_xor_sync(0xffffffffu, tt, 8);
        c = c / (tt + eps);
    };
    auto norm_rows = [&]() {
        float tt = c;
        tt += __shfl_xor_sync(0xffffffffu, tt, 1);
        tt += __shfl_xor_sync(0xffffffffu, tt, 2);
        c = c / (tt + eps);
    };
    norm_cols();
    for (int it = 1; it < iters; ++it) { norm_rows(); norm_cols(); }
    if (on) comb[t * hc * hc + lane] = c;
}
__global__ void k_mhc_pre_apply_t(const float* __restrict__ R, const float* __restrict__ pre, float* __restrict__ x,
                                  int n_embd, int hc) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    const float* Rt = R + (size_t) t * hc * n_embd;
    float acc = 0.f;
    for (int h = 0; h < hc; ++h) acc += Rt[(size_t) h * n_embd + i] * pre[t * hc + h];
    x[(size_t) t * n_embd + i] = acc;
}
__global__ void k_mhc_post_t(const float* __restrict__ y, const float* __restrict__ R_in, const float* __restrict__ post,
                             const float* __restrict__ comb, float* __restrict__ R_out, int n_embd, int hc) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    const float* Rt = R_in + (size_t) t * hc * n_embd;
    float* Ro = R_out + (size_t) t * hc * n_embd;
    float r[8];
    for (int s = 0; s < hc; ++s) r[s] = Rt[(size_t) s * n_embd + i];
    const float yi = y[(size_t) t * n_embd + i];
    for (int d = 0; d < hc; ++d) {
        float acc = yi * post[t * hc + d];
        for (int s = 0; s < hc; ++s) acc += r[s] * comb[t * hc * hc + s * hc + d];
        Ro[(size_t) d * n_embd + i] = acc;
    }
}
// KDA conv over T tokens, sequential per channel; the window state after every token goes to snap[t] (rollback)
__global__ void k_kda_conv_t(const float* __restrict__ x, float* __restrict__ st, const float* __restrict__ w,
                             float* __restrict__ out, float* __restrict__ snap, int d_inner, int d_conv, int T) {
    // thread (channel c, token t): the causal window ends at t and starts in the state for t < d_conv - 1; the same
    // tap order as the sequential form.  The state after the window = the last d_conv - 1 inputs.
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= d_inner || t >= T) return;
    float win[8];
    for (int k = 0; k < d_conv; ++k) {
        const int src = t - (d_conv - 1) + k;   // input index of tap k
        win[k] = src >= 0 ? x[(size_t) src * d_inner + c] : st[(size_t) (src + d_conv - 1) * d_inner + c];
    }
    float acc = 0.f;
    for (int k = 0; k < d_conv; ++k) acc += w[(size_t) c * d_conv + k] * win[k];
    out[(size_t) t * d_inner + c] = siluf_(acc);
    if (snap) for (int k = 0; k < d_conv - 1; ++k) snap[((size_t) t * (d_conv - 1) + k) * d_inner + c] = win[k + 1];
}
// the conv state after a window: the last d_conv - 1 inputs (from the old state when the window is shorter)
__global__ void k_kda_conv_state(const float* __restrict__ x, float* __restrict__ st, int d_inner, int d_conv, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d_inner) return;
    float nw[8];
    for (int k = 0; k < d_conv - 1; ++k) {
        const int src = T - (d_conv - 1) + k;
        nw[k] = src >= 0 ? x[(size_t) src * d_inner + c] : st[(size_t) (src + d_conv - 1) * d_inner + c];
    }
    for (int k = 0; k < d_conv - 1; ++k) st[(size_t) k * d_inner + c] = nw[k];
}
__global__ void k_kda_gate_t(const float* __restrict__ g_in, const float* __restrict__ dt_b, const float* __restrict__ A,
                             float lower, float* __restrict__ g, int n_tot, int D, int S) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tot) return;
    const int c = i % D;
    const float a = A[c / S];
    const float v = g_in[i] + dt_b[c];
    if (lower > -INFINITY) g[i] = lower * sigmoidf_(-(v * a));
    else g[i] = log1pf(expf(v)) * a;
}
// KDA recurrence over T tokens (sequential), block per head, 4 x S threads; the state after token t goes to snap[t]
__global__ void k_kda_step_t(float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k,
                             const float* __restrict__ v, const float* __restrict__ g, const float* __restrict__ beta,
                             float* __restrict__ o, float* __restrict__ snap, int H, int S, int T) {
    // thread (quarter, column j) owns the 32 state elements of its quarter's rows in registers for the whole window;
    // the per-element arithmetic and the reduction order are those of the single-token kernel
    constexpr int NI = 32;   // S / kKdaQuarters (S == 128)
    extern __shared__ float red[];
    const int h = blockIdx.x;
    const int j = threadIdx.x % S, qu = threadIdx.x / S;
    const int i0 = qu * NI;
    float* st = state + (size_t) h * S * S;
    const int HS = H * S;
    float sreg[NI];
#pragma unroll
    for (int r = 0; r < NI; ++r) sreg[r] = st[(size_t) (i0 + r) * S + j];
    for (int t = 0; t < T; ++t) {
        const float* qh = q + (size_t) t * HS + (size_t) h * S;
        const float* kh = k + (size_t) t * HS + (size_t) h * S;
        const float* gh = g + (size_t) t * HS + (size_t) h * S;
        float sk = 0.f;
#pragma unroll
        for (int r = 0; r < NI; ++r) {
            const float s = sreg[r] * expf(gh[i0 + r]);
            sreg[r] = s;
            sk += s * kh[i0 + r];
        }
        red[qu * S + j] = sk;
        __syncthreads();
        sk = 0.f;
        for (int tt = 0; tt < kKdaQuarters; ++tt) sk += red[tt * S + j];
        const float d = (v[(size_t) t * HS + (size_t) h * S + j] - sk) * beta[t * H + h];
        float acc = 0.f;
#pragma unroll
        for (int r = 0; r < NI; ++r) {
            const float s = sreg[r] + kh[i0 + r] * d;
            sreg[r] = s;
            acc += s * qh[i0 + r];
            if (snap) snap[((size_t) t * H + h) * S * S + (size_t) (i0 + r) * S + j] = s;
        }
        __syncthreads();
        red[qu * S + j] = acc;
        __syncthreads();
        if (qu == 0) {
            acc = 0.f;
            for (int tt = 0; tt < kKdaQuarters; ++tt) acc += red[tt * S + j];
            o[(size_t) t * HS + (size_t) h * S + j] = acc * rsqrtf((float) S);
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < NI; ++r) st[(size_t) (i0 + r) * S + j] = sreg[r];
}
// warp per (head, j) over every token: w row in registers (lane-strided, as the per-token kernel reads it)
__global__ void k_dsa_absorb_b(const uint16_t* __restrict__ wk_b, const float* __restrict__ q, float* __restrict__ q_abs,
                               int n_head, int kv_lora, int head_k, int T) {
    const int h = blockIdx.y;
    const int j = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    if (j >= kv_lora) return;
    const uint16_t* w = wk_b + ((size_t) h * kv_lora + j) * head_k;
    float wr[8];   // head_k <= 256
    const int nw = (head_k - lane + WARP - 1) / WARP;
    for (int r = 0; r < 8; ++r) wr[r] = r < nw ? bf16f(w[lane + r * WARP]) : 0.f;
    for (int t = 0; t < T; ++t) {
        const float* qh = q + ((size_t) t * n_head + h) * head_k;
        float acc = 0.f;
        for (int r = 0; r < nw; ++r) acc += wr[r] * qh[lane + r * WARP];
        acc = warp_sum(acc);
        if (lane == 0) q_abs[((size_t) t * n_head + h) * kv_lora + j] = acc;
    }
}
__global__ void k_dsa_expand_b(const uint16_t* __restrict__ wv_b, const float* __restrict__ ctx, float* __restrict__ out,
                               int n_head, int kv_lora, int head_v, int T) {
    const int h = blockIdx.y;
    const int v = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    if (v >= head_v) return;
    const uint16_t* w = wv_b + ((size_t) h * head_v + v) * kv_lora;
    float wr[16];   // kv_lora <= 512
    const int nw = (kv_lora - lane + WARP - 1) / WARP;
    for (int r = 0; r < 16; ++r) wr[r] = r < nw ? bf16f(w[lane + r * WARP]) : 0.f;
    for (int t = 0; t < T; ++t) {
        const float* c = ctx + ((size_t) t * n_head + h) * kv_lora;
        float acc = 0.f;
        for (int r = 0; r < nw; ++r) acc += wr[r] * c[lane + r * WARP];
        acc = warp_sum(acc);
        if (lane == 0) out[((size_t) t * n_head + h) * head_v + v] = acc;
    }
}
__global__ void k_dsa_absorb_t(const uint16_t* __restrict__ wk_b, const float* __restrict__ q, float* __restrict__ q_abs,
                               int n_head, int kv_lora, int head_k) {
    const int t = blockIdx.z, h = blockIdx.y;
    const int j = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    if (j >= kv_lora) return;
    const uint16_t* w = wk_b + ((size_t) h * kv_lora + j) * head_k;
    const float* qh = q + ((size_t) t * n_head + h) * head_k;
    float acc = 0.f;
    for (int i = lane; i < head_k; i += WARP) acc += bf16f(w[i]) * qh[i];
    acc = warp_sum(acc);
    if (lane == 0) q_abs[((size_t) t * n_head + h) * kv_lora + j] = acc;
}
__global__ void k_dsa_expand_t(const uint16_t* __restrict__ wv_b, const float* __restrict__ ctx, float* __restrict__ out,
                               int n_head, int kv_lora, int head_v) {
    const int t = blockIdx.z, h = blockIdx.y;
    const int v = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    if (v >= head_v) return;
    const uint16_t* w = wv_b + ((size_t) h * head_v + v) * kv_lora;
    const float* c = ctx + ((size_t) t * n_head + h) * kv_lora;
    float acc = 0.f;
    for (int j = lane; j < kv_lora; j += WARP) acc += bf16f(w[j]) * c[j];
    acc = warp_sum(acc);
    if (lane == 0) out[((size_t) t * n_head + h) * head_v + v] = acc;
}
// the router over T tokens: block per token
__global__ void k_router_t(const float* __restrict__ logits, const float* __restrict__ bias, int n_expert, int k,
                           bool norm, float scale, int32_t* __restrict__ ids, float* __restrict__ wts,
                           float* __restrict__ probs_out) {
    extern __shared__ float sh[];
    const int t = blockIdx.x;
    const float* lg = logits + (size_t) t * n_expert;
    float* probs = sh;
    float* sel = sh + n_expert;
    __shared__ float red_v[32];
    __shared__ int red_i[32];
    __shared__ float wsum;
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        const float p = sigmoidf_(lg[e]);
        probs[e] = p;
        sel[e] = bias ? p + bias[e] : p;
        if (probs_out) probs_out[(size_t) t * n_expert + e] = p;
    }
    if (threadIdx.x == 0) wsum = 0.f;
    __syncthreads();
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, nw = (blockDim.x + WARP - 1) / WARP;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int e = threadIdx.x; e < n_expert; e += blockDim.x)
            if (sel[e] > bv || (sel[e] == bv && e < bi)) { bv = sel[e]; bi = e; }
        for (int o = WARP / 2; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { red_v[wid] = bv; red_i[wid] = bi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < nw; ++w)
                if (red_v[w] > bv || (red_v[w] == bv && red_i[w] < bi)) { bv = red_v[w]; bi = red_i[w]; }
            ids[t * k + r] = bi;
            wts[t * k + r] = probs[bi];
            wsum += probs[bi];
            sel[bi] = -INFINITY;
        }
        __syncthreads();
    }
    if (threadIdx.x < k) {
        float w = wts[t * k + threadIdx.x];
        if (norm) w /= fmaxf(wsum, 6.103515625e-5f);
        wts[t * k + threadIdx.x] = w * scale;
    }
}
// The MoE dispatch of a window: the T*K routed (token, expert) pairs grouped by expert, resident experts' groups first.
// One thread (T*K <= 64).  Outputs:
//   grp_w[g] (gate ptr), grp_wu[g] (up), grp_wd[g] (down): the blob of group g's expert (slot, or stage slot m for the
//   m-th distinct missed expert in order of first occurrence - the host assigns the same slots)
//   grp_col0[g], grp_ncols[g]: the pair range of group g;  grp_orow[g] = grp_col0[g]
//   pair_tok[p], pair_w[p]: the token and router weight of pair p (pairs ordered by group)
//   cnt[0] = n_hit_groups, cnt[1] = n_groups, cnt[2] = n_pairs, cnt[3] = n_missed_experts
__global__ void k_moe_group(const int32_t* __restrict__ ids, const float* __restrict__ wts, int T, int K, int n_expert,
                            const int32_t* __restrict__ slot_of, const uint8_t* __restrict__ slots, uint64_t blob,
                            uint64_t gate_bytes, uint64_t up_bytes, const uint8_t* __restrict__ stage, uint64_t stage_bytes,
                            const void** __restrict__ grp_w, const void** __restrict__ grp_wu, const void** __restrict__ grp_wd,
                            int* __restrict__ grp_col0, int* __restrict__ grp_ncols, int* __restrict__ grp_orow,
                            int* __restrict__ pair_tok, float* __restrict__ pair_w, int* __restrict__ cnt,
                            int* __restrict__ grp_e, int* __restrict__ grp_miss, int split_hits, int* __restrict__ bounds,
                            int* __restrict__ tok_pairs, int n_slots, int* __restrict__ bounds_res, int* __restrict__ bounds_miss) {
    // A counting sort of the T*K pairs by expert (one block, any T): the groups are the routed experts in ascending
    // id, hits before misses; a missed expert's staging slot = its rank among the missed experts (the host sorts its
    // miss list the same way).  A pair's column within its group = its arrival order (the order does not matter:
    // the pairs of a token are summed).
    constexpr int kMaxExperts = 512;
    __shared__ int s_count[kMaxExperts], s_col0[kMaxExperts], s_fill[kMaxExperts], s_inv[kMaxExperts];
    const int n = T * K;
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) { s_count[e] = 0; s_fill[e] = 0; }
    for (int sl = threadIdx.x; sl < kMaxExperts; sl += blockDim.x) s_inv[sl] = -1;
    __syncthreads();
    for (int p = threadIdx.x; p < n; p += blockDim.x) atomicAdd(&s_count[ids[p]], 1);
    __syncthreads();
    if (!split_hits) {
        // the prompt path: resident experts' groups in slot order (one MMQ product over the slot buffer), then the
        // non-resident ones by their rank among all non-resident experts (the compact prefetch buffer)
        for (int e = threadIdx.x; e < n_expert; e += blockDim.x)
            if (s_count[e] && slot_of[e] >= 0) s_inv[slot_of[e]] = e;
        __syncthreads();
        if (threadIdx.x == 0) {
            int g = 0, col = 0;
            for (int sl = 0; sl < n_slots; ++sl) {
                bounds_res[sl] = col;
                const int e = s_inv[sl];
                if (e < 0) continue;
                const uint8_t* b = slots + (uint64_t) sl * blob;
                s_col0[e] = col;
                grp_w[g] = b; grp_wu[g] = b + gate_bytes; grp_wd[g] = b + gate_bytes + up_bytes;
                grp_col0[g] = col; grp_orow[g] = col; grp_ncols[g] = s_count[e]; grp_e[g] = e; grp_miss[g] = -1;
                col += s_count[e];
                ++g;
            }
            bounds_res[n_slots] = col;
            cnt[0] = g;
            int rank = 0;
            for (int e = 0; e < n_expert; ++e) {
                if (slot_of[e] >= 0) continue;
                bounds_miss[rank] = col;
                if (s_count[e]) {
                    const uint8_t* b = stage + (uint64_t) rank * stage_bytes;
                    s_col0[e] = col;
                    grp_w[g] = b; grp_wu[g] = b + gate_bytes; grp_wd[g] = b + gate_bytes + up_bytes;
                    grp_col0[g] = col; grp_orow[g] = col; grp_ncols[g] = s_count[e]; grp_e[g] = e; grp_miss[g] = rank;
                    col += s_count[e];
                    ++g;
                }
                ++rank;
            }
            bounds_miss[rank] = col;
            int widest = 0;
            for (int e = 0; e < n_expert; ++e) widest = max(widest, s_count[e]);
            cnt[5] = widest;
            cnt[1] = g; cnt[2] = col; cnt[3] = rank; cnt[4] = 0;
        }
    } else if (threadIdx.x == 0) {
        // split_hits: the resident experts' groups first, then the missed ones (the decode path computes the hits
        // before the doorbell wait).  Otherwise one pass in ascending expert id (the MMQ path: its row bounds per
        // expert need the pairs laid out by expert id); bounds[e] = the first pair of expert e, bounds[n_expert] = all
        int g = 0, col = 0, nmiss = 0;
        const int passes = split_hits ? 2 : 1;
        for (int pass = 0; pass < passes; ++pass) {
            for (int e = 0; e < n_expert; ++e) {
                if (!split_hits && bounds) bounds[e] = col;
                if (s_count[e] == 0) continue;
                const int slot = slot_of[e];
                if (split_hits && (slot >= 0) != (pass == 0)) continue;
                const uint8_t* b = slot >= 0 ? slots + (uint64_t) slot * blob : stage + (uint64_t) nmiss * stage_bytes;
                if (slot < 0) ++nmiss;
                s_col0[e] = col;
                grp_w[g] = b;
                grp_wu[g] = b + gate_bytes;
                grp_wd[g] = b + gate_bytes + up_bytes;
                grp_col0[g] = col;
                grp_orow[g] = col;
                grp_ncols[g] = s_count[e];
                grp_e[g] = e;
                grp_miss[g] = slot < 0 ? nmiss - 1 : -1;
                col += s_count[e];
                ++g;
            }
            if (pass == 0) cnt[0] = split_hits ? g : 0;
        }
        if (!split_hits && bounds) bounds[n_expert] = col;
        int widest = 0;
        for (int e = 0; e < n_expert; ++e) widest = max(widest, s_count[e]);
        cnt[5] = widest;
        cnt[1] = g;
        cnt[2] = col;
        cnt[3] = nmiss;
        cnt[4] = 0;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < n; p += blockDim.x) {
        const int e = ids[p];
        int rank;
        if (n <= 64) {   // a decode window: the pairs of a group in token order (the host's CPU service lays its rows out the same way)
            rank = 0;
            for (int q = 0; q < p; ++q) rank += ids[q] == e;
        } else rank = atomicAdd(&s_fill[e], 1);
        const int slot = s_col0[e] + rank;
        pair_tok[slot] = p / K;
        pair_w[slot] = wts[p];
        if (tok_pairs) tok_pairs[p] = slot;   // the inverse table: pair (token p/K, k = p%K) -> its row
    }
}
// after the host served the misses: miss m went to slot miss_slot[m] (>= 0; a negative value = it stayed in staging
// slot m); point the miss groups at their slots and update the device's slot table (evicting miss_out[m])
__global__ void k_moe_place(const int* __restrict__ cnt, const int* __restrict__ grp_miss, const int* __restrict__ grp_e,
                            const int* __restrict__ miss_slot, const int* __restrict__ miss_out, const uint8_t* __restrict__ slots,
                            uint64_t blob, uint64_t gate_bytes, uint64_t up_bytes, const void** __restrict__ grp_w,
                            const void** __restrict__ grp_wu, const void** __restrict__ grp_wd, int32_t* __restrict__ slot_of) {
    for (int g = cnt[0] + threadIdx.x; g < cnt[1]; g += blockDim.x) {
        const int m = grp_miss[g];
        if (m < 0) continue;
        const int dst = miss_slot[m];
        if (dst < 0) continue;
        const uint8_t* b = slots + (uint64_t) dst * blob;
        grp_w[g] = b;
        grp_wu[g] = b + gate_bytes;
        grp_wd[g] = b + gate_bytes + up_bytes;
        const int out = miss_out[m];
        if (out >= 0) slot_of[out] = -1;
        slot_of[grp_e[g]] = dst;
    }
}
// the MMQ path: the resident groups' expert blobs (gate | up | down) into the window buffer at the slot of their
// expert id, dst + e * stride (the non-resident ones were DMA'd there by the prefetch).  blockIdx.x = group,
// blockIdx.y strides the 16-byte vectors.
__global__ void k_moe_gather_experts(const void* const* __restrict__ grp_w, const int* __restrict__ grp_e, const int* __restrict__ grp_miss,
                                     const int* __restrict__ n_groups, uint64_t blob, uint8_t* __restrict__ dst, uint64_t stride) {
    const int g = blockIdx.x;
    if (g >= *n_groups || grp_miss[g] >= 0) return;
    const uint4* src = static_cast<const uint4*>(grp_w[g]);
    uint4* d = reinterpret_cast<uint4*>(dst + (uint64_t) grp_e[g] * stride);
    const uint64_t n = blob / 16, step = (uint64_t) gridDim.y * blockDim.x;
    for (uint64_t i = (uint64_t) blockIdx.y * blockDim.x + threadIdx.x; i < n; i += step) d[i] = src[i];
}
// h[r][i] = swiglu_clamp(gu[r][i], gu[r][FF + i]) for r < *n_rows (the MMQ gate|up product's layout)
__global__ void k_swiglu_clamp_gu(const float* __restrict__ gu, float* __restrict__ h, const int* __restrict__ n_rows, int n, float limit) {
    const int r = blockIdx.y;
    if (r >= *n_rows) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gu[(size_t) r * 2 * n + i], u = gu[(size_t) r * 2 * n + n + i];
    const size_t o = (size_t) r * n + i;
    if (limit > 0.f) h[o] = siluf_(fminf(g, limit)) * fminf(fmaxf(u, -limit), limit);
    else h[o] = siluf_(g) * u;
}
// the hybrid split: miss m served by the CPU (miss_cpu[m] != 0) -> its group computes nothing on the device (its pair
// rows arrive from the host); the DMA'd misses keep their staging-slot groups
__global__ void k_moe_cpu_mask(const int* __restrict__ cnt, const int* __restrict__ grp_miss, const int* __restrict__ miss_cpu,
                               int* __restrict__ grp_ncols) {
    for (int g = cnt[0] + threadIdx.x; g < cnt[1]; g += blockDim.x) {
        const int m = grp_miss[g];
        if (m >= 0 && miss_cpu[m]) grp_ncols[g] = 0;
    }
}
// gather the pairs' Q8_1 activation columns: xg[p] = xq[pair_tok[p]] (col_bytes each), p < cnt[2]
__global__ void k_gather_q8(const uint8_t* __restrict__ xq, const int* __restrict__ pair_tok, const int* __restrict__ n_pairs,
                            uint8_t* __restrict__ xg, int col_bytes) {
    const int p = blockIdx.x;
    if (p >= *n_pairs) return;
    const uint32_t* src = reinterpret_cast<const uint32_t*>(xq + (size_t) pair_tok[p] * col_bytes);
    uint32_t* dst = reinterpret_cast<uint32_t*>(xg + (size_t) p * col_bytes);
    for (int i = threadIdx.x; i < col_bytes / 4; i += blockDim.x) dst[i] = src[i];
}
// out[t][i] = sum over pairs p with pair_tok[p] == t of pair_w[p] * y[p][i]   (pairs in group order: deterministic)
__global__ void k_combine_pairs(const float* __restrict__ y, const float* __restrict__ pair_w, const int* __restrict__ pair_tok,
                                const int* __restrict__ n_pairs, float* __restrict__ out, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int np = *n_pairs;
    float acc = 0.f;
    for (int p = 0; p < np; ++p)
        if (pair_tok[p] == t) acc += pair_w[p] * y[(size_t) p * n + i];
    out[(size_t) t * n + i] = acc;
}
// out[t][i] = sum_k pair_w[s] y[s][i], s = tok_pairs[t*K + k]: the per-token table instead of a scan of every pair
__global__ void k_combine_tok(const float* __restrict__ y, const float* __restrict__ pair_w, const int* __restrict__ tok_pairs, int K,
                              float* __restrict__ out, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const int s = tok_pairs[t * K + k];
        acc += pair_w[s] * y[(size_t) s * n + i];
    }
    out[(size_t) t * n + i] = acc;
}
__global__ void k_swiglu_clamp_rows(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ h,
                                    const int* __restrict__ n_rows, int n, float limit) {
    const int r = blockIdx.y;
    if (r >= *n_rows) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t o = (size_t) r * n + i;
    if (limit > 0.f) h[o] = siluf_(fminf(g[o], limit)) * fminf(fmaxf(u[o], -limit), limit);
    else h[o] = siluf_(g[o]) * u[o];
}
__global__ void k_argmax_rows1(const float* __restrict__ x, int n, int* __restrict__ idx, float* __restrict__ val) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    const float* xr = x + (size_t) blockIdx.y * n;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (xr[i] > bv) { bv = xr[i]; bi = i; }
    sv[threadIdx.x] = bv;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float ov = sv[threadIdx.x + s];
            const int oi = si[threadIdx.x + s];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) { sv[threadIdx.x] = ov; si[threadIdx.x] = oi; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { idx[blockIdx.y * gridDim.x + blockIdx.x] = si[0]; val[blockIdx.y * gridDim.x + blockIdx.x] = sv[0]; }
}
__global__ void k_argmax_rows2(const int* __restrict__ idx, const float* __restrict__ val, int n, int32_t* out) {
    if (threadIdx.x != 0) return;
    const int r = blockIdx.x;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = 0; i < n; ++i)
        if (val[r * n + i] > bv || (val[r * n + i] == bv && idx[r * n + i] < bi)) { bv = val[r * n + i]; bi = idx[r * n + i]; }
    out[r] = bi;
}
__global__ void k_dequant_q8_0_rows(const uint8_t* __restrict__ w, const int32_t* __restrict__ rows, float* __restrict__ out,
                                    int n, int zero_first) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (zero_first && t == 0) { out[(size_t) t * n + i] = 0.f; return; }
    const uint8_t* blk = w + (size_t) rows[t] * (n / 32) * 34 + (size_t) (i / 32) * 34;
    half d;
    __builtin_memcpy(&d, blk, 2);
    out[(size_t) t * n + i] = __half2float(d) * (float) (int8_t) blk[2 + (i % 32)];
}
__global__ void k_concat_rows(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * n) return;
    out[(size_t) t * 2 * n + i] = i < n ? a[(size_t) t * n + i] : b[(size_t) t * n + i - n];
}
__global__ void k_add_rows(float* __restrict__ y, const float* __restrict__ x, int n_tot) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_tot) y[i] += x[i];
}
}  // namespace

int window_max_tokens() { return kTmax; }
void f32_gemv_t(const float* W, const float* x, float* y, int n_in, int n_out, int T, void* s) {
    k_f32_gemv_t<<<dim3(blocks_for(n_out, 8), T), 256, 0, S(s)>>>(W, x, y, n_in, n_out);
}
void layer_norm_t(const float* x, const float* w, const float* b, float* y, int rows, int n, float eps, void* s) {
    k_layer_norm_t<<<rows, 128, 0, S(s)>>>(x, w, b, y, n, eps);
}
void mhc_mix_t(const float* R, const float* fn, const float* scale, const float* base, float* mixes, float* part, float* rs,
               float* pre, float* post, float* comb, int T, int n_embd, int hc, int iters, float eps_rms, float eps_hc, void* s) {
    const int n = hc * n_embd, n_mix = (2 + hc) * hc;
    k_mhc_rs_t<<<T, 1024, 0, S(s)>>>(R, rs, n, eps_rms);
    k_mhc_mixes_t<<<dim3(n_mix, kMixSlices, T), 128, 0, S(s)>>>(R, fn, rs, part, n, n_mix);
    k_mhc_nonlin_t<<<T, 32, 0, S(s)>>>(part, scale, base, mixes, pre, post, comb, hc, iters, eps_hc);
}
void mhc_pre_apply_t(const float* R, const float* pre, float* x, int T, int n_embd, int hc, void* s) {
    k_mhc_pre_apply_t<<<dim3(blocks_for(n_embd), T), 256, 0, S(s)>>>(R, pre, x, n_embd, hc);
}
void mhc_post_t(const float* y, const float* R_in, const float* post, const float* comb, float* R_out, int T, int n_embd, int hc,
                void* s) {
    k_mhc_post_t<<<dim3(blocks_for(n_embd), T), 256, 0, S(s)>>>(y, R_in, post, comb, R_out, n_embd, hc);
}
void kda_conv_silu_t(const float* x, float* conv_state, const float* w, float* out, float* snap, int d_inner, int d_conv, int T,
                     void* s) {
    k_kda_conv_t<<<dim3(blocks_for(d_inner), T), 256, 0, S(s)>>>(x, conv_state, w, out, snap, d_inner, d_conv, T);
    k_kda_conv_state<<<blocks_for(d_inner), 256, 0, S(s)>>>(x, conv_state, d_inner, d_conv, T);
}
void kda_gate_t(const float* g_in, const float* dt_b, const float* A, float lower, float* g, int T, int n_head, int S_, void* s) {
    const int D = n_head * S_;
    k_kda_gate_t<<<blocks_for(T * D), 256, 0, S(s)>>>(g_in, dt_b, A, lower, g, T * D, D, S_);
}
// the prompt-window scan with one thread per state column: the 128 state elements of column j in registers, the gates'
// exp shared per step (128 exps instead of 128 x 128), the two dot products over i inside the thread (no cross-thread
// reduction): one __syncthreads per token.  No snapshots (the verify windows keep the quarter kernel).
__global__ void k_kda_step_col(float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                               const float* __restrict__ g, const float* __restrict__ beta, float* __restrict__ o, int H, int T) {
    // two adjacent lanes per column (rows 0..63 / 64..127): 64 state registers each, the two partial dot products joined
    // by one lane shuffle; the gates' exp and the k/q rows shared per step
    constexpr int S = 128, HALF = 64;
    __shared__ float eg[S], kk[S], qq[S];
    const int h = blockIdx.x, j = threadIdx.x >> 1, half = threadIdx.x & 1, i0 = half * HALF;
    float* st = state + (size_t) h * S * S;
    const int HS = H * S;
    float sreg[HALF];
#pragma unroll
    for (int r = 0; r < HALF; ++r) sreg[r] = st[(size_t) (i0 + r) * S + j];
    const float inv = rsqrtf((float) S);
    for (int t = 0; t < T; ++t) {
        const size_t base = (size_t) t * HS + (size_t) h * S;
        __syncthreads();
        if (threadIdx.x < S) { eg[threadIdx.x] = expf(g[base + threadIdx.x]); kk[threadIdx.x] = k[base + threadIdx.x]; qq[threadIdx.x] = q[base + threadIdx.x]; }
        __syncthreads();
        float sk = 0.f;
#pragma unroll
        for (int r = 0; r < HALF; ++r) { const float sv = sreg[r] * eg[i0 + r]; sreg[r] = sv; sk += sv * kk[i0 + r]; }
        sk += __shfl_xor(sk, 1, WARP);
        const float d = (v[base + j] - sk) * beta[t * H + h];
        float acc = 0.f;
#pragma unroll
        for (int r = 0; r < HALF; ++r) { const float sv = sreg[r] + kk[i0 + r] * d; sreg[r] = sv; acc += sv * qq[i0 + r]; }
        acc += __shfl_xor(acc, 1, WARP);
        if (half == 0) o[base + j] = acc * inv;
    }
#pragma unroll
    for (int r = 0; r < HALF; ++r) st[(size_t) (i0 + r) * S + j] = sreg[r];
}
void kda_step_t(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta, float* o,
                float* snap, int T, int n_head, int S_, void* s) {
    if (S_ != 128) throw std::runtime_error("kda_step_t: head dim must be 128");
    static const bool kCol = !std::getenv("STRATA_GLM5_KDA_COL") || std::atoi(std::getenv("STRATA_GLM5_KDA_COL")) != 0;
    if (!snap && kCol && T > 8) k_kda_step_col<<<n_head, 2 * S_, 0, S(s)>>>(state, q, k, v, g, beta, o, n_head, T);
    else k_kda_step_t<<<n_head, kKdaQuarters * S_, kKdaQuarters * S_ * sizeof(float), S(s)>>>(state, q, k, v, g, beta, o, snap, n_head, S_, T);
}
void dsa_absorb_t(const uint16_t* wk_b, const float* q, float* q_abs, int T, int n_head, int kv_lora, int head_k, void* s) {
    if (T >= 16 && head_k <= 256) k_dsa_absorb_b<<<dim3(blocks_for(kv_lora, 8), n_head), 256, 0, S(s)>>>(wk_b, q, q_abs, n_head, kv_lora, head_k, T);
    else k_dsa_absorb_t<<<dim3(blocks_for(kv_lora, 8), n_head, T), 256, 0, S(s)>>>(wk_b, q, q_abs, n_head, kv_lora, head_k);
}
void dsa_expand_t(const uint16_t* wv_b, const float* ctx, float* out, int T, int n_head, int kv_lora, int head_v, void* s) {
    if (T >= 16 && kv_lora <= 512) k_dsa_expand_b<<<dim3(blocks_for(head_v, 8), n_head), 256, 0, S(s)>>>(wv_b, ctx, out, n_head, kv_lora, head_v, T);
    else k_dsa_expand_t<<<dim3(blocks_for(head_v, 8), n_head, T), 256, 0, S(s)>>>(wv_b, ctx, out, n_head, kv_lora, head_v);
}
void router_sigmoid_topk_t(const float* logits, const float* bias, int T, int n_expert, int k, bool norm, float scale, int32_t* ids,
                           float* weights, float* probs, void* s) {
    k_router_t<<<T, 256, 2 * n_expert * sizeof(float), S(s)>>>(logits, bias, n_expert, k, norm, scale, ids, weights, probs);
}
void moe_group(const int32_t* ids, const float* wts, int T, int K, int n_expert, const int32_t* slot_of, const uint8_t* slots, uint64_t blob,
               uint64_t gate_bytes, uint64_t up_bytes, const uint8_t* stage, uint64_t stage_bytes, const void** grp_w,
               const void** grp_wu, const void** grp_wd, int* grp_col0, int* grp_ncols, int* grp_orow, int* pair_tok,
               float* pair_w, int* cnt, int* grp_e, int* grp_miss, bool split_hits, int* bounds, int* tok_pairs, int n_slots, int* bounds_res,
               int* bounds_miss, void* s) {
    if (n_expert > 512) throw std::runtime_error("moe_group: more than 512 experts");
    k_moe_group<<<1, 1024, 0, S(s)>>>(ids, wts, T, K, n_expert, slot_of, slots, blob, gate_bytes, up_bytes, stage, stage_bytes, grp_w, grp_wu,
                                      grp_wd, grp_col0, grp_ncols, grp_orow, pair_tok, pair_w, cnt, grp_e, grp_miss, split_hits ? 1 : 0, bounds, tok_pairs,
                                      n_slots, bounds_res, bounds_miss);
}
void moe_gather_experts(const void* const* grp_w, const int* grp_e, const int* grp_miss, const int* n_groups, int n_expert, uint64_t blob,
                        uint8_t* dst, uint64_t stride, void* s) {
    k_moe_gather_experts<<<dim3(n_expert, 64), 256, 0, S(s)>>>(grp_w, grp_e, grp_miss, n_groups, blob, dst, stride);
}
void moe_cpu_mask(const int* cnt, const int* grp_miss, const int* miss_cpu, int* grp_ncols, void* s) {
    k_moe_cpu_mask<<<1, 256, 0, S(s)>>>(cnt, grp_miss, miss_cpu, grp_ncols);
}
void swiglu_clamp_gu(const float* gu, float* h, const int* n_rows, int max_rows, int n, float limit, void* s) {
    k_swiglu_clamp_gu<<<dim3(blocks_for(n), max_rows), 256, 0, S(s)>>>(gu, h, n_rows, n, limit);
}
void moe_place(const int* cnt, const int* grp_miss, const int* grp_e, const int* miss_slot, const int* miss_out, const uint8_t* slots,
               uint64_t blob, uint64_t gate_bytes, uint64_t up_bytes, const void** grp_w, const void** grp_wu, const void** grp_wd,
               int32_t* slot_of, void* s) {
    k_moe_place<<<1, 256, 0, S(s)>>>(cnt, grp_miss, grp_e, miss_slot, miss_out, slots, blob, gate_bytes, up_bytes, grp_w, grp_wu, grp_wd, slot_of);
}
void gather_q8(const uint8_t* xq, const int* pair_tok, const int* n_pairs, uint8_t* xg, int col_bytes, int max_pairs, void* s) {
    k_gather_q8<<<max_pairs, 256, 0, S(s)>>>(xq, pair_tok, n_pairs, xg, col_bytes);
}
void combine_pairs(const float* y, const float* pair_w, const int* pair_tok, const int* n_pairs, float* out, int T, int n, void* s) {
    k_combine_pairs<<<dim3(blocks_for(n), T), 256, 0, S(s)>>>(y, pair_w, pair_tok, n_pairs, out, n);
}
// the per-head MMQ tables: rows head-major (h*T + t) -> source / destination row t*H + h; bounds[h] = h*T
__global__ void k_head_tables(int32_t* __restrict__ ids, int32_t* __restrict__ bounds, int T, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * H) ids[i] = (i % T) * H + i / T;
    if (i <= H) bounds[i] = i * T;
}
void head_tables(int32_t* ids, int32_t* bounds, int T, int H, void* s) {
    k_head_tables<<<blocks_for(std::max(T * H, H + 1)), 256, 0, S(s)>>>(ids, bounds, T, H);
}
// ---- the prompt windows' dense (masked) DSA attention through MMQ: the latent cache as a Q8_0 "weight"
// the transpose of a Q8_0 cache: row j (j < cols) of the output = column j of x over the cells c < n_cells (zero beyond),
// re-quantized along the cells in blocks of 32; n_pad % 32 == 0
__global__ void k_q8_0_rows_T(const uint8_t* __restrict__ x, int cols, uint8_t* __restrict__ out, int n_cells, int n_pad) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int bpr = n_pad / 32, pitch = q8_pitch(cols);
    if (b >= cols * bpr) return;
    const int j = b / bpr, c0 = (b % bpr) * 32;
    float v[32];
    for (int i = 0; i < 32; ++i) { const int c = c0 + i; v[i] = c < n_cells ? q8_at(x + (size_t) c * pitch, j) : 0.f; }
    q8_block(v, out + (size_t) b * 34);
}
// the selection masks: bit c of row tb set for every cell in sel[tb][0..n_sel) (mask zeroed by the caller)
__global__ void k_sel_mask(const int32_t* __restrict__ sel, const int* __restrict__ params, int n_sel_max, uint32_t* __restrict__ mask, int words) {
    const int tb = blockIdx.x;
    const int n_sel = params[tb * 8 + 4];
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) {
        const int c = sel[(size_t) tb * n_sel_max + i];
        atomicOr(&mask[(size_t) tb * words + c / 32], 1u << (c % 32));
    }
}
// the masked softmax in place: row r = (tb, h), cells with the bit set take s*scale, the rest 0 probability
__global__ void k_masked_softmax(float* __restrict__ S, int ld, const uint32_t* __restrict__ mask, int words, int H, int n_pad, float scale,
                                 const float* __restrict__ bias) {
    __shared__ float sh[32];
    const int r = blockIdx.x, tb = r / H;
    float* row = S + (size_t) r * ld;
    const uint32_t* m = mask + (size_t) tb * words;
    float mx = -INFINITY;
    for (int c = threadIdx.x; c < n_pad; c += blockDim.x) if (m[c / 32] >> (c % 32) & 1u) mx = fmaxf(mx, row[c] * scale + (bias ? bias[c] : 0.f));
    mx = warp_max(mx);
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, nw = (blockDim.x + WARP - 1) / WARP;
    if (lane == 0) sh[wid] = mx;
    __syncthreads();
    float t = threadIdx.x < nw ? sh[threadIdx.x] : -INFINITY;
    if (wid == 0) t = warp_max(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    mx = sh[0];
    __syncthreads();
    float sum = 0.f;
    for (int c = threadIdx.x; c < n_pad; c += blockDim.x) {
        const bool on = m[c / 32] >> (c % 32) & 1u;
        const float e = on ? expf(row[c] * scale + (bias ? bias[c] : 0.f) - mx) : 0.f;
        row[c] = e;
        sum += e;
    }
    sum = block_sum(sum, sh);
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    for (int c = threadIdx.x; c < n_pad; c += blockDim.x) row[c] *= inv;
}
__global__ void k_set_pair(int32_t* p, int a, int b) { p[0] = a; p[1] = b; }
void set_pair(int32_t* p, int a, int b, void* s) { k_set_pair<<<1, 1, 0, S(s)>>>(p, a, b); }
void q8_0_rows_T(const uint8_t* x, int cols, uint8_t* out, int n_cells, int n_pad, void* s) { k_q8_0_rows_T<<<blocks_for(cols * (n_pad / 32)), 256, 0, S(s)>>>(x, cols, out, n_cells, n_pad); }
void sel_mask(const int32_t* sel, const int* params, int TB, int n_sel_max, uint32_t* mask, int words, void* s) { k_sel_mask<<<TB, 256, 0, S(s)>>>(sel, params, n_sel_max, mask, words); }
void masked_softmax(float* S_, int ld, const uint32_t* mask, int words, int rows, int H, int n_pad, float scale, const float* bias, void* s) { k_masked_softmax<<<rows, 256, 0, S(s)>>>(S_, ld, mask, words, H, n_pad, scale, bias); }
// ---- batch decode windows: row t belongs to slot row_slot[t] at position row_pos[t]; tab[slot] = that slot's state base
// one conv step per row on its slot's conv state (the same arithmetic as k_kda_conv)
__global__ void k_kda_conv_rows(const float* __restrict__ x, float* const* __restrict__ tab, const int* __restrict__ row_slot, int which,
                                const float* __restrict__ w, float* __restrict__ out, int d_inner, int d_conv) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= d_inner) return;
    float* st = tab[row_slot[t]] + (size_t) which * (d_conv - 1) * d_inner;
    float win[8];
    for (int k = 0; k < d_conv - 1; ++k) win[k] = st[(size_t) k * d_inner + c];
    win[d_conv - 1] = x[(size_t) t * d_inner + c];
    float acc = 0.f;
    for (int k = 0; k < d_conv; ++k) acc += w[(size_t) c * d_conv + k] * win[k];
    for (int k = 0; k < d_conv - 1; ++k) st[(size_t) k * d_inner + c] = win[k + 1];
    out[(size_t) t * d_inner + c] = siluf_(acc);
}
// one recurrence step per row on its slot's state: block (head h, row t), the single-token kernel's arithmetic
__global__ void k_kda_step_rows(float* const* __restrict__ tab, const int* __restrict__ row_slot, const float* __restrict__ q,
                                const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ g,
                                const float* __restrict__ beta, float* __restrict__ o, int H, int S) {
    extern __shared__ float red[];
    const int h = blockIdx.x, t = blockIdx.y;
    const int j = threadIdx.x % S, qu = threadIdx.x / S;
    const int i0 = qu * (S / kKdaQuarters), i1 = i0 + S / kKdaQuarters;
    float* st = tab[row_slot[t]] + (size_t) h * S * S;
    const size_t HS = (size_t) H * S;
    const float* qh = q + t * HS + (size_t) h * S;
    const float* kh = k + t * HS + (size_t) h * S;
    const float* gh = g + t * HS + (size_t) h * S;
    float sk = 0.f;
    for (int i = i0; i < i1; ++i) {
        const float s = st[(size_t) i * S + j] * expf(gh[i]);
        st[(size_t) i * S + j] = s;
        sk += s * kh[i];
    }
    red[qu * S + j] = sk;
    __syncthreads();
    sk = 0.f;
    for (int tt = 0; tt < kKdaQuarters; ++tt) sk += red[tt * S + j];
    const float d = (v[t * HS + (size_t) h * S + j] - sk) * beta[t * H + h];
    float acc = 0.f;
    for (int i = i0; i < i1; ++i) {
        const float s = st[(size_t) i * S + j] + kh[i] * d;
        st[(size_t) i * S + j] = s;
        acc += s * qh[i];
    }
    __syncthreads();
    red[qu * S + j] = acc;
    __syncthreads();
    if (qu == 0) {
        acc = 0.f;
        for (int tt = 0; tt < kKdaQuarters; ++tt) acc += red[tt * S + j];
        o[t * HS + (size_t) h * S + j] = acc * rsqrtf((float) S);
    }
}
// row t of src ([T][n]) quantized to Q8_0 into tab[row_slot[t]]'s cache row row_pos[t]; thread per block of 32
__global__ void k_scatter_q8_rows(const float* __restrict__ src, uint8_t* const* __restrict__ tab, const int* __restrict__ row_slot,
                                  const int* __restrict__ row_pos, int n) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (b >= n / 32) return;
    q8_block(src + (size_t) t * n + b * 32, tab[row_slot[t]] + (size_t) row_pos[t] * q8_pitch(n) + b * 34);
}
// rows of f32 stored as Q8_0 (the DSA caches): dst row t = src[t]; thread per block of 32
__global__ void k_store_q8_rows(const float* __restrict__ src, uint8_t* __restrict__ dst, int n) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (b >= n / 32) return;
    q8_block(src + (size_t) t * n + b * 32, dst + (size_t) t * q8_pitch(n) + b * 34);
}
// the pool completed by row t (its position ends one): pool (pos+1)/KP - 1 of its slot from its slot's caches
__global__ void k_dsa_pool_rows(uint8_t* const* __restrict__ ik_tab, uint8_t* const* __restrict__ ig_tab, float* const* __restrict__ pooled_tab,
                                const int* __restrict__ row_slot, const int* __restrict__ row_pos, const float* __restrict__ ape, int kpool, int D) {
    const int t = blockIdx.y, pos = row_pos[t];
    if ((pos + 1) % kpool != 0) return;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const int p = (pos + 1) / kpool - 1, j0 = p * kpool, sl = row_slot[t];
    const uint8_t* ik = ik_tab[sl];
    const uint8_t* ig = ig_tab[sl];
    const int pitch = q8_pitch(D);
    float lg[8], m = -INFINITY;
    for (int j = 0; j < kpool; ++j) { lg[j] = q8_at(ig + (size_t) (j0 + j) * pitch, d) + ape[(size_t) j * D + d]; m = fmaxf(m, lg[j]); }
    float sum = 0.f;
    for (int j = 0; j < kpool; ++j) { lg[j] = expf(lg[j] - m); sum += lg[j]; }
    float acc = 0.f;
    for (int j = 0; j < kpool; ++j) acc += lg[j] / sum * q8_at(ik + (size_t) (j0 + j) * pitch, d);
    pooled_tab[sl][(size_t) p * D + d] = acc;
}
// the indexer scores of row t over its slot's pools (k_dsa_score with a per-row pool base)
__global__ void k_dsa_score_rows(float* const* __restrict__ pooled_tab, const int* __restrict__ row_slot, const float* __restrict__ iq,
                                 const float* __restrict__ w, float* __restrict__ score, const int* __restrict__ params, int n_head, int D,
                                 int ld_score) {
    const int t = blockIdx.y;
    const int n_pool = params[t * 8 + 0];
    const float* pooled = pooled_tab[row_slot[t]];
    iq += (size_t) t * n_head * D;
    w += (size_t) t * n_head;
    score += (size_t) t * ld_score;
    const int p = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    if (p >= n_pool) return;
    const float* pk = pooled + (size_t) p * D;
    float total = 0.f;
    for (int h = 0; h < n_head; ++h) {
        const float* qh = iq + (size_t) h * D;
        float acc = 0.f;
        for (int i = lane; i < D; i += WARP) acc += qh[i] * pk[i];
        acc = warp_sum(acc);
        total += fmaxf(acc, 0.f) * w[h];
    }
    if (lane == 0) score[p] = total;
}
// the attention of row t over its slot's latent cache (k_dsa_scores_warp / k_dsa_ctx_warp with a per-row cache base)
__global__ void k_dsa_scores_rows(uint8_t* const* __restrict__ kvc_tab, uint8_t* const* __restrict__ means_tab, float bias, const int* __restrict__ row_slot,
                                  const int32_t* __restrict__ sel, const int* __restrict__ params, const float* __restrict__ q_abs, float scale,
                                  float* __restrict__ scores, int kv_lora, int ld, int n_head) {
    const int t = blockIdx.z;
    const int n_sel = params[t * 8 + 4];
    const uint8_t* kvc = kvc_tab[row_slot[t]];
    const uint8_t* means = means_tab ? means_tab[row_slot[t]] : nullptr;
    sel += (size_t) t * ld;
    q_abs += (size_t) t * n_head * kv_lora;
    scores += (size_t) t * n_head * ld;
    const int h = blockIdx.y;
    const int c = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    if (c >= n_sel) return;
    const int32_t sc = sel[c];
    const uint8_t* k = sel_row(kvc, means, sc, q8_pitch(kv_lora));
    const float* qh = q_abs + (size_t) h * kv_lora;
    float acc = 0.f;
    for (int i = lane; i < kv_lora; i += WARP) acc += qh[i] * q8_at(k, i);
    acc = warp_sum(acc);
    if (lane == 0) scores[(size_t) h * ld + c] = acc * scale + (sc < 0 ? bias : 0.f);
}
__global__ void k_dsa_ctx_rows(uint8_t* const* __restrict__ kvc_tab, uint8_t* const* __restrict__ means_tab, const int* __restrict__ row_slot,
                               const int32_t* __restrict__ sel, const int* __restrict__ params, const float* __restrict__ p, float* __restrict__ ctx,
                               int kv_lora, int ld, int n_head) {
    const int t = blockIdx.y;
    const int n_sel = params[t * 8 + 4];
    const uint8_t* kvc = kvc_tab[row_slot[t]];
    const uint8_t* means = means_tab ? means_tab[row_slot[t]] : nullptr;
    sel += (size_t) t * ld;
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    if (j >= kv_lora) return;
    const float* ph = p + ((size_t) t * n_head + h) * ld;
    float acc = 0.f;
    const int pitch = q8_pitch(kv_lora);
    for (int c = 0; c < n_sel; ++c) acc += ph[c] * q8_at(sel_row(kvc, means, sel[c], pitch), j);
    ctx[((size_t) t * n_head + h) * kv_lora + j] = acc;
}
// rows of one slot in a run (row_k[t] = the row's index within its slot's run): the conv taps come from the slot's
// state for indices before the run and from the run's earlier rows otherwise; snap[t] = the window after row t
__global__ void k_kda_conv_grp(const float* __restrict__ x, float* const* __restrict__ tab, const int* __restrict__ row_slot,
                               const int* __restrict__ row_k, int which, const float* __restrict__ w, float* __restrict__ out,
                               float* __restrict__ snap, int d_inner, int d_conv) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= d_inner) return;
    const float* st = tab[row_slot[t]] + (size_t) which * (d_conv - 1) * d_inner;
    const int kk = row_k[t];
    float win[8];
    for (int k = 0; k < d_conv; ++k) {
        const int idx = kk - (d_conv - 1) + k;   // the run's row the tap reads (negative: the state)
        win[k] = idx >= 0 ? x[(size_t) (t - kk + idx) * d_inner + c] : st[(size_t) (idx + d_conv - 1) * d_inner + c];
    }
    float acc = 0.f;
    for (int k = 0; k < d_conv; ++k) acc += w[(size_t) c * d_conv + k] * win[k];
    out[(size_t) t * d_inner + c] = siluf_(acc);
    if (snap) for (int k = 0; k < d_conv - 1; ++k) snap[((size_t) t * (d_conv - 1) + k) * d_inner + c] = win[k + 1];
}
// the conv state after a run: the last d_conv - 1 inputs of the run (from the old state when the run is shorter); block.y = group
__global__ void k_kda_conv_state_grp(const float* __restrict__ x, float* const* __restrict__ tab, const int* __restrict__ grp_slot,
                                     const int* __restrict__ grp_start, const int* __restrict__ grp_len, int which, int d_inner, int d_conv) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, g = blockIdx.y;
    if (c >= d_inner) return;
    float* st = tab[grp_slot[g]] + (size_t) which * (d_conv - 1) * d_inner;
    const int r0 = grp_start[g], len = grp_len[g];
    float nw[8];
    for (int k = 0; k < d_conv - 1; ++k) {
        const int idx = len - (d_conv - 1) + k;
        nw[k] = idx >= 0 ? x[(size_t) (r0 + idx) * d_inner + c] : st[(size_t) (idx + d_conv - 1) * d_inner + c];
    }
    for (int k = 0; k < d_conv - 1; ++k) st[(size_t) k * d_inner + c] = nw[k];
}
// the recurrence over a slot's run of rows: block (head, group), the rows in order, snap[row] = the state after it
__global__ void k_kda_step_grp(float* const* __restrict__ tab, const int* __restrict__ grp_slot, const int* __restrict__ grp_start,
                               const int* __restrict__ grp_len, const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                               const float* __restrict__ g, const float* __restrict__ beta, float* __restrict__ o, float* __restrict__ snap, int H, int S) {
    constexpr int NI = 32;
    extern __shared__ float red[];
    const int h = blockIdx.x, grp = blockIdx.y;
    const int j = threadIdx.x % S, qu = threadIdx.x / S;
    const int i0 = qu * NI;
    float* st = tab[grp_slot[grp]] + (size_t) h * S * S;
    const int HS = H * S, r0 = grp_start[grp], len = grp_len[grp];
    float sreg[NI];
#pragma unroll
    for (int r = 0; r < NI; ++r) sreg[r] = st[(size_t) (i0 + r) * S + j];
    for (int tt = 0; tt < len; ++tt) {
        const int t = r0 + tt;
        const float* qh = q + (size_t) t * HS + (size_t) h * S;
        const float* kh = k + (size_t) t * HS + (size_t) h * S;
        const float* gh = g + (size_t) t * HS + (size_t) h * S;
        float sk = 0.f;
#pragma unroll
        for (int r = 0; r < NI; ++r) { const float s = sreg[r] * expf(gh[i0 + r]); sreg[r] = s; sk += s * kh[i0 + r]; }
        red[qu * S + j] = sk;
        __syncthreads();
        sk = 0.f;
        for (int u = 0; u < kKdaQuarters; ++u) sk += red[u * S + j];
        const float d = (v[(size_t) t * HS + (size_t) h * S + j] - sk) * beta[t * H + h];
        float acc = 0.f;
#pragma unroll
        for (int r = 0; r < NI; ++r) {
            const float s = sreg[r] + kh[i0 + r] * d;
            sreg[r] = s;
            acc += s * qh[i0 + r];
            if (snap) snap[((size_t) t * H + h) * S * S + (size_t) (i0 + r) * S + j] = s;
        }
        __syncthreads();
        red[qu * S + j] = acc;
        __syncthreads();
        if (qu == 0) { acc = 0.f; for (int u = 0; u < kKdaQuarters; ++u) acc += red[u * S + j]; o[(size_t) t * HS + (size_t) h * S + j] = acc * rsqrtf((float) S); }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < NI; ++r) st[(size_t) (i0 + r) * S + j] = sreg[r];
}
void kda_conv_grp(const float* x, float* const* tab, const int* row_slot, const int* row_k, int which, const float* w, float* out, float* snap,
                  int d_inner, int d_conv, int T, const int* grp_slot, const int* grp_start, const int* grp_len, int n_grp, void* s) {
    k_kda_conv_grp<<<dim3(blocks_for(d_inner), T), 256, 0, S(s)>>>(x, tab, row_slot, row_k, which, w, out, snap, d_inner, d_conv);
    k_kda_conv_state_grp<<<dim3(blocks_for(d_inner), n_grp), 256, 0, S(s)>>>(x, tab, grp_slot, grp_start, grp_len, which, d_inner, d_conv);
}
void kda_step_grp(float* const* tab, const int* grp_slot, const int* grp_start, const int* grp_len, int n_grp, const float* q, const float* k,
                  const float* v, const float* g, const float* beta, float* o, float* snap, int n_head, int S_, void* s) {
    if (S_ != 128) throw std::runtime_error("kda_step_grp: head dim must be 128");
    k_kda_step_grp<<<dim3(n_head, n_grp), kKdaQuarters * S_, kKdaQuarters * S_ * sizeof(float), S(s)>>>(tab, grp_slot, grp_start, grp_len, q, k, v, g, beta, o, snap, n_head, S_);
}
void kda_conv_rows(const float* x, float* const* tab, const int* row_slot, int which, const float* w, float* out, int d_inner, int d_conv, int T, void* s) {
    k_kda_conv_rows<<<dim3(blocks_for(d_inner), T), 256, 0, S(s)>>>(x, tab, row_slot, which, w, out, d_inner, d_conv);
}
void kda_step_rows(float* const* tab, const int* row_slot, const float* q, const float* k, const float* v, const float* g, const float* beta,
                   float* o, int T, int n_head, int S_, void* s) {
    k_kda_step_rows<<<dim3(n_head, T), kKdaQuarters * S_, kKdaQuarters * S_ * sizeof(float), S(s)>>>(tab, row_slot, q, k, v, g, beta, o, n_head, S_);
}
void scatter_q8_rows(const float* src, uint8_t* const* tab, const int* row_slot, const int* row_pos, int T, int n, void* s) {
    k_scatter_q8_rows<<<dim3(blocks_for(n / 32), T), 256, 0, S(s)>>>(src, tab, row_slot, row_pos, n);
}
void store_q8_rows(const float* src, uint8_t* dst, int T, int n, void* s) { k_store_q8_rows<<<dim3(blocks_for(n / 32), T), 256, 0, S(s)>>>(src, dst, n); }
void dsa_pool_rows(uint8_t* const* ik_tab, uint8_t* const* ig_tab, float* const* pooled_tab, const int* row_slot, const int* row_pos, const float* ape,
                   int T, int kpool, int D, void* s) {
    k_dsa_pool_rows<<<dim3(blocks_for(D), T), 256, 0, S(s)>>>(ik_tab, ig_tab, pooled_tab, row_slot, row_pos, ape, kpool, D);
}
void dsa_score_rows(float* const* pooled_tab, const int* row_slot, const float* iq, const float* w, float* score, const int* params, int n_pool_max,
                    int T, int n_head, int D, int ld_score, void* s) {
    if (n_pool_max <= 0) return;
    k_dsa_score_rows<<<dim3(blocks_for(n_pool_max, 8), T), 256, 0, S(s)>>>(pooled_tab, row_slot, iq, w, score, params, n_head, D, ld_score);
}
void dsa_attend_rows(uint8_t* const* kvc_tab, uint8_t* const* means_tab, float bias, const int* row_slot, const int32_t* sel, const int* params, int T,
                     int n_sel_max, const float* q_abs, float scale, float* scores, float* ctx, int n_head, int kv_lora, void* s) {
    k_dsa_scores_rows<<<dim3(blocks_for(n_sel_max, 8), n_head, T), 256, 0, S(s)>>>(kvc_tab, means_tab, bias, row_slot, sel, params, q_abs, scale, scores, kv_lora, n_sel_max, n_head);
    k_softmax_rows<<<dim3(n_head, T), 256, 0, S(s)>>>(scores, params, n_sel_max, n_head);
    k_dsa_ctx_rows<<<dim3(n_head, T), kv_lora, 0, S(s)>>>(kvc_tab, means_tab, row_slot, sel, params, scores, ctx, kv_lora, n_sel_max, n_head);
}

// ---------------------------------------------------------------- LoD attention (indexer-free): page means of the latent cache,
// page scores, the selection (exact pages + summaries), the prompt windows' compact operand
// the mean of a slot's complete page (ps rows of the Q8_0 cache) as one Q8_0 row of `means`; block per page, 512 threads
__global__ void k_lod_page_mean(const uint8_t* __restrict__ kvc, uint8_t* __restrict__ means, int p0, int ps, int n) {
    __shared__ float m[1024];
    const int p = p0 + blockIdx.x, pitch = q8_pitch(n);
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float acc = 0.f;
        for (int r = 0; r < ps; ++r) acc += q8_at(kvc + (size_t) (p * ps + r) * pitch, j);
        m[j] = acc / (float) ps;
    }
    __syncthreads();
    if (threadIdx.x < n / 32) q8_block(m + threadIdx.x * 32, means + (size_t) p * pitch + threadIdx.x * 34);
}
// rows mode: row t completes page (pos+1)/ps - 1 of its slot when (pos+1) % ps == 0
__global__ void k_lod_page_mean_rows(uint8_t* const* __restrict__ kvc_tab, uint8_t* const* __restrict__ means_tab, const int* __restrict__ row_slot,
                                     const int* __restrict__ row_pos, int ps, int n) {
    __shared__ float m[1024];
    const int t = blockIdx.x, pos = row_pos[t];
    if ((pos + 1) % ps != 0) return;
    const int p = (pos + 1) / ps - 1, sl = row_slot[t], pitch = q8_pitch(n);
    const uint8_t* kvc = kvc_tab[sl];
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float acc = 0.f;
        for (int r = 0; r < ps; ++r) acc += q8_at(kvc + (size_t) (p * ps + r) * pitch, j);
        m[j] = acc / (float) ps;
    }
    __syncthreads();
    if (threadIdx.x < n / 32) q8_block(m + threadIdx.x * 32, means_tab[sl] + (size_t) p * pitch + threadIdx.x * 34);
}
// page scores of row t: score[t][p] = max over the heads of q_abs[t][h] . mean[p] for the pages p < params[t*8+0] (the
// candidates); warp per (t, p).  The ranking is what matters (max over heads of the raw dot products).
__global__ void k_lod_score_rows(uint8_t* const* __restrict__ means_tab, const int* __restrict__ row_slot, const float* __restrict__ q_abs,
                                 float* __restrict__ score, const int* __restrict__ params, int n_head, int n, int ld_score) {
    const int t = blockIdx.y, n_cand = params[t * 8 + 0];
    const int p = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP, lane = threadIdx.x % WARP;
    if (p >= n_cand) return;
    const uint8_t* row = means_tab[row_slot[t]] + (size_t) p * q8_pitch(n);
    float mv[16];   // the lane's values of the mean row (n <= 16 * WARP)
    for (int e = 0; e < 16; ++e) { const int i = lane + e * WARP; mv[e] = i < n ? q8_at(row, i) : 0.f; }
    const float* q = q_abs + (size_t) t * n_head * n;
    float best = -INFINITY;
    for (int h = 0; h < n_head; ++h) {
        const float* qh = q + (size_t) h * n;
        float acc = 0.f;
        for (int e = 0; e < 16; ++e) { const int i = lane + e * WARP; if (i < n) acc += qh[i] * mv[e]; }
        acc = warp_sum(acc);
        best = fmaxf(best, acc);
    }
    if (lane == 0) score[(size_t) t * ld_score + p] = best;
}
// the column maxima of S [rows][ld] over the first n_cols columns folded into score[] (running max across batches)
__global__ void k_colmax(const float* __restrict__ S_, int ld, int rows, int n_cols, float* __restrict__ score, int first) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cols) return;
    float m = first ? -INFINITY : score[c];
    for (int r = 0; r < rows; ++r) m = fmaxf(m, S_[(size_t) r * ld + c]);
    score[c] = m;
}
// the LoD selection of token t (params[t*8]: [0] n_cand = the pages before the local field, [1] n_top, [2] n_kv, [3] sink pages;
// out [4] n_sel): the top n_top candidate pages (never the sink) and the sink pages exactly, every cell from the local
// field's first page to n_kv - 1 exactly, then the other candidate pages as summaries -(p+1).  One block per token.
__global__ void k_lod_select(const float* __restrict__ score, int* __restrict__ params, int ps, int32_t* __restrict__ sel, int ld_score, int n_sel_max) {
    __shared__ int hist[256];
    __shared__ int scan[1024];
    __shared__ uint32_t s_prefix;
    __shared__ int s_remaining, s_base;
    __shared__ unsigned char open[8192];   // candidate pages (n_cand <= 8192: 512K cells at 64/page): 1 = read exactly
    const int t = blockIdx.x;
    score += (size_t) t * ld_score;
    params += t * 8;
    sel += (size_t) t * n_sel_max;
    const int n_cand = params[0], n_kv = params[2], sink = params[3];
    const int n_top = min(params[1], max(n_cand - sink, 0));
    const int tid = threadIdx.x, nt = blockDim.x;
    for (int p = tid; p < n_cand; p += nt) open[p] = p < sink ? 1 : 0;
    __syncthreads();
    uint32_t prefix = 0;
    int remaining = n_top;
    if (n_top > 0) {
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            for (int b = tid; b < 256; b += nt) hist[b] = 0;
            __syncthreads();
            for (int p = sink + tid; p < n_cand; p += nt) {
                const uint32_t key = f32_key(score[p]);
                if (pass == 0 || (key >> (shift + 8)) == (prefix >> (shift + 8))) atomicAdd(&hist[(key >> shift) & 255], 1);
            }
            __syncthreads();
            if (tid == 0) {
                int acc = 0, b = 255;
                for (; b > 0; --b) { if (acc + hist[b] >= remaining) break; acc += hist[b]; }
                s_prefix = prefix | ((uint32_t) b << shift);
                s_remaining = remaining - acc;
            }
            __syncthreads();
            prefix = s_prefix; remaining = s_remaining;
            __syncthreads();
        }
        // open every candidate above the threshold, and the first `remaining` equal to it (index order)
        const uint32_t thr = prefix;
        if (tid == 0) s_base = 0;
        __syncthreads();
        for (int round = 0; round < 2; ++round) {
            for (int p0 = sink; p0 < n_cand; p0 += nt) {
                const int p = p0 + tid;
                int flag = 0;
                if (p < n_cand) { const uint32_t key = f32_key(score[p]); flag = round == 0 ? key > thr : key == thr; }
                scan[tid] = flag;
                __syncthreads();
                for (int o = 1; o < nt; o <<= 1) { const int v = tid >= o ? scan[tid - o] : 0; __syncthreads(); scan[tid] += v; __syncthreads(); }
                const int total = scan[nt - 1];
                if (flag && (round == 0 || scan[tid] - 1 < remaining)) open[p] = 1;
                __syncthreads();
                if (tid == 0) { if (round == 1) s_remaining = remaining - min(total, remaining); }
                __syncthreads();
                if (round == 1) remaining = s_remaining;
                __syncthreads();
            }
        }
    }
    // the list: exact pages (open candidates) in index order, the local field, then the summaries
    if (tid == 0) s_base = 0;
    __syncthreads();
    for (int p0 = 0; p0 < n_cand; p0 += nt) {
        const int p = p0 + tid;
        const int flag = p < n_cand ? open[p] : 0;
        scan[tid] = flag;
        __syncthreads();
        for (int o = 1; o < nt; o <<= 1) { const int v = tid >= o ? scan[tid - o] : 0; __syncthreads(); scan[tid] += v; __syncthreads(); }
        if (flag) { const int rank = s_base + scan[tid] - 1; for (int j = 0; j < ps; ++j) sel[rank * ps + j] = p * ps + j; }
        __syncthreads();
        if (tid == 0) s_base += scan[nt - 1];
        __syncthreads();
    }
    const int n_exact_pages = s_base;
    const int loc0 = n_cand * ps, n_loc = n_kv - loc0;   // the local field: every cell from the first non-candidate page
    int base = n_exact_pages * ps;
    for (int j = tid; j < n_loc; j += nt) sel[base + j] = loc0 + j;
    base += n_loc;
    __syncthreads();
    if (tid == 0) s_base = 0;
    __syncthreads();
    for (int p0 = 0; p0 < n_cand; p0 += nt) {
        const int p = p0 + tid;
        const int flag = p < n_cand ? !open[p] : 0;
        scan[tid] = flag;
        __syncthreads();
        for (int o = 1; o < nt; o <<= 1) { const int v = tid >= o ? scan[tid - o] : 0; __syncthreads(); scan[tid] += v; __syncthreads(); }
        if (flag) sel[base + s_base + scan[tid] - 1] = -(p + 1);
        __syncthreads();
        if (tid == 0) s_base += scan[nt - 1];
        __syncthreads();
    }
    if (tid == 0) params[4] = base + s_base;
}
// the prompt window's compact operand: rows [0, n_sel) = the shared selection's rows (cache cells / page means), rows
// [n_sel, n_sel + T) = the window's own tokens pos0 + j; bias[c] = `bias` for the summary rows, else 0
__global__ void k_lod_gather(const uint8_t* __restrict__ kvc, const uint8_t* __restrict__ means, const int32_t* __restrict__ sel, int n_sel, int pos0, int T,
                             uint8_t* __restrict__ out, float* __restrict__ bias_col, float bias, int n) {
    const int c = blockIdx.x, pitch = q8_pitch(n);
    if (c >= n_sel + T) return;
    const int32_t e = c < n_sel ? sel[c] : pos0 + (c - n_sel);
    const uint8_t* src = sel_row(kvc, means, e, pitch);
    uint8_t* dst = out + (size_t) c * pitch;
    for (int i = threadIdx.x; i < pitch; i += blockDim.x) dst[i] = src[i];
    if (threadIdx.x == 0) bias_col[c] = e < 0 ? bias : 0.f;
}
// the mask rows of the window's tokens t0 .. t0 + TB - 1: every fixed column, and own-token column j <= t
__global__ void k_lod_mask(uint32_t* __restrict__ mask, int words, int n_sel, int T, int t0) {
    const int tb = blockIdx.x, t = t0 + tb;
    uint32_t* m = mask + (size_t) tb * words;
    for (int w = threadIdx.x; w < words; w += blockDim.x) {
        uint32_t bits = 0;
        for (int b = 0; b < 32; ++b) {
            const int c = w * 32 + b;
            if (c < n_sel || (c < n_sel + T && c - n_sel <= t)) bits |= 1u << b;
        }
        m[w] = bits;
    }
}
// ---- the fused selection read (a split-K kernel with an online softmax, on Strata's row structure): one online softmax over row t's
// selection list for a pair of heads; the K and V of a column are the same Q8_0 latent row, read once.  Block = 256 lanes =
// 8 wave tiles, a tile owns one column at a time with every lane holding 2 consecutive elements (EPL2 pairs per lane).
// part[((t*H + h)*nsplit + spl)][2 + KV] = {m, den, acc}; the merge kernel folds the splits.
constexpr int kLodHpt = 2, kLodBlock = 256, kLodNt = kLodBlock / WARP;
template <int EPL2>
__global__ void k_lod_attn_split(uint8_t* const* __restrict__ kvc_tab, uint8_t* const* __restrict__ means_tab, float bias,
                                 const int* __restrict__ row_slot, const int32_t* __restrict__ sel_in, const int* __restrict__ params,
                                 const float* __restrict__ q_abs, float scale, float* __restrict__ part, int H, int KV, int n_sel_max, int nsplit) {
    constexpr int DVT = 2 * EPL2 * WARP;
    __shared__ float sm_q[kLodHpt][DVT];
    __shared__ float sm_m[kLodNt], sm_den[kLodNt];
    __shared__ float sm_acc[kLodNt][DVT];
    const int row = blockIdx.x, spl = blockIdx.y;
    const int nhp = H / kLodHpt, t = row / nhp, h0 = (row % nhp) * kLodHpt;
    const int lane = threadIdx.x % WARP, tile = threadIdx.x / WARP;
    const int n_sel = params[t * 8 + 4];
    const int32_t* sel = sel_in + (size_t) t * n_sel_max;
    const int sl = row_slot[t];
    const uint8_t* kvc = kvc_tab[sl];
    const uint8_t* means = means_tab ? means_tab[sl] : nullptr;
    const int pitch = q8_pitch(KV);
    for (int j = 0; j < kLodHpt; ++j)
        for (int d = threadIdx.x; d < KV; d += blockDim.x) sm_q[j][d] = q_abs[((size_t) t * H + h0 + j) * KV + d];
    __syncthreads();
    float m[kLodHpt], den[kLodHpt], acc[kLodHpt][2 * EPL2];
    for (int j = 0; j < kLodHpt; ++j) { m[j] = -INFINITY; den[j] = 0.f; for (int i = 0; i < 2 * EPL2; ++i) acc[j][i] = 0.f; }
    for (int c = spl * kLodNt + tile; c < n_sel; c += nsplit * kLodNt) {
        const int32_t e = sel[c];
        const uint8_t* rp = e >= 0 ? kvc + (size_t) e * pitch : means + (size_t) (-e - 1) * pitch;
        const float cb = e < 0 ? bias : 0.f;
        float2 kv2[EPL2];
        float sj[kLodHpt] = {0.f, 0.f};
#pragma unroll
        for (int ep = 0; ep < EPL2; ++ep) {
            const int i2 = lane + ep * WARP;                       // the pair index: elements 2*i2, 2*i2+1 (one Q8_0 block)
            const uint8_t* b = rp + (i2 >> 4) * 34;                // 16 pairs per 32-element block
            const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(b)));
            kv2[ep] = make_float2(d * (float) (int8_t) b[2 + 2 * (i2 & 15)], d * (float) (int8_t) b[3 + 2 * (i2 & 15)]);
            for (int j = 0; j < kLodHpt; ++j) sj[j] += sm_q[j][2 * i2] * kv2[ep].x + sm_q[j][2 * i2 + 1] * kv2[ep].y;
        }
#pragma unroll
        for (int off = WARP / 2; off > 0; off >>= 1)
            for (int j = 0; j < kLodHpt; ++j) sj[j] += __shfl_xor(sj[j], off, WARP);
        float w[kLodHpt];
        for (int j = 0; j < kLodHpt; ++j) {
            const float logit = sj[j] * scale + cb;
            if (logit > m[j]) {
                const float ex = expf(m[j] - logit);
                den[j] *= ex;
#pragma unroll
                for (int i = 0; i < 2 * EPL2; ++i) acc[j][i] *= ex;
                m[j] = logit; w[j] = 1.f;
            } else w[j] = expf(logit - m[j]);
            den[j] += w[j];
        }
#pragma unroll
        for (int ep = 0; ep < EPL2; ++ep)
            for (int j = 0; j < kLodHpt; ++j) { acc[j][2 * ep] += w[j] * kv2[ep].x; acc[j][2 * ep + 1] += w[j] * kv2[ep].y; }
    }
    // fold the block's tiles, one head at a time
    for (int j = 0; j < kLodHpt; ++j) {
        __syncthreads();
        if (lane == 0) sm_m[tile] = m[j];
        __syncthreads();
        float m_b = -INFINITY;
        for (int u = 0; u < kLodNt; ++u) m_b = fmaxf(m_b, sm_m[u]);
        const float e_t = m[j] == -INFINITY ? 0.f : expf(m[j] - m_b);
        if (lane == 0) sm_den[tile] = den[j] * e_t;
#pragma unroll
        for (int ep = 0; ep < EPL2; ++ep) { const int i2 = lane + ep * WARP; sm_acc[tile][2 * i2] = acc[j][2 * ep] * e_t; sm_acc[tile][2 * i2 + 1] = acc[j][2 * ep + 1] * e_t; }
        __syncthreads();
        float* out = part + (((size_t) t * H + h0 + j) * nsplit + spl) * (2 + KV);
        if (threadIdx.x == 0) { float db = 0.f; for (int u = 0; u < kLodNt; ++u) db += sm_den[u]; out[0] = m_b; out[1] = db; }
        for (int d = threadIdx.x; d < KV; d += blockDim.x) { float a = 0.f; for (int u = 0; u < kLodNt; ++u) a += sm_acc[u][d]; out[2 + d] = a; }
    }
}
// the merge: ctx[t][h] = sum over splits of e^(m_s - m) acc_s / sum e^(m_s - m) den_s; block per (t, h)
__global__ void k_lod_attn_merge(const float* __restrict__ part, float* __restrict__ ctx, int KV, int nsplit) {
    __shared__ float sm_w[128];
    const int row = blockIdx.x;
    const float* p = part + (size_t) row * nsplit * (2 + KV);
    float mx = -INFINITY;
    for (int sp = 0; sp < nsplit; ++sp) mx = fmaxf(mx, p[(size_t) sp * (2 + KV)]);
    for (int sp = threadIdx.x; sp < nsplit; sp += blockDim.x) { const float ms = p[(size_t) sp * (2 + KV)]; sm_w[sp] = ms > -INFINITY ? expf(ms - mx) : 0.f; }
    __syncthreads();
    float den = 0.f;
    for (int sp = 0; sp < nsplit; ++sp) den += p[(size_t) sp * (2 + KV) + 1] * sm_w[sp];
    const float id = den > 0.f ? 1.f / den : 0.f;
    float* out = ctx + (size_t) row * KV;
    for (int d = threadIdx.x; d < KV; d += blockDim.x) {
        float a = 0.f;
        for (int sp = 0; sp < nsplit; ++sp) { const float w = sm_w[sp]; if (w > 0.f) a += p[(size_t) sp * (2 + KV) + 2 + d] * w; }
        out[d] = a * id;
    }
}
void lod_attn_fused(uint8_t* const* kvc_tab, uint8_t* const* means_tab, float bias, const int* row_slot, const int32_t* sel, const int* params, int T,
                    int n_sel_max, const float* q_abs, float scale, float* part, float* ctx, int n_head, int kv_lora, int nsplit, void* s) {
    const dim3 grid((unsigned) (T * n_head / kLodHpt), (unsigned) nsplit);
    if (kv_lora == 512) k_lod_attn_split<8><<<grid, kLodBlock, 0, S(s)>>>(kvc_tab, means_tab, bias, row_slot, sel, params, q_abs, scale, part, n_head, kv_lora, n_sel_max, nsplit);
    else if (kv_lora == 256) k_lod_attn_split<4><<<grid, kLodBlock, 0, S(s)>>>(kvc_tab, means_tab, bias, row_slot, sel, params, q_abs, scale, part, n_head, kv_lora, n_sel_max, nsplit);
    else if (kv_lora == 1024) k_lod_attn_split<16><<<grid, kLodBlock, 0, S(s)>>>(kvc_tab, means_tab, bias, row_slot, sel, params, q_abs, scale, part, n_head, kv_lora, n_sel_max, nsplit);
    k_lod_attn_merge<<<T * n_head, 256, 0, S(s)>>>(part, ctx, kv_lora, nsplit);
}
// ---- token-level LoD routing: pages scored by the best token inside them against head-group means of the query
// (the page mean dilutes one distinctive token among 63; the summaries still use the means)
// qg[t][g][n] = the mean over the heads of group g of q[t][h][n]; block.x over n, block.y = t
__global__ void k_head_group_mean(const float* __restrict__ q, float* __restrict__ qg, int H, int G, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    const int per = H / G;
    for (int g = 0; g < G; ++g) {
        float acc = 0.f;
        for (int h = g * per; h < (g + 1) * per; ++h) acc += q[((size_t) t * H + h) * n + i];
        qg[((size_t) t * G + g) * n + i] = acc / (float) per;
    }
}
// page_score[p] = max over the page's tokens of tok_score; the first n_tok tokens (pages p0 .. p0 + n_pages)
__global__ void k_page_max(const float* __restrict__ tok_score, float* __restrict__ page_score, int ps, int n_pages) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pages) return;
    float m = -INFINITY;
    for (int j = 0; j < ps; ++j) m = fmaxf(m, tok_score[(size_t) p * ps + j]);
    page_score[p] = m;
}
// decode rows: block per page (ps threads, one token each), every row t: score[t][p] = max over the page's tokens of the
// max over the row's head groups of qg[t][g] . c_tok; the page's candidate count per row is params[t*8+0]
__global__ void k_lod_score_tok_rows(uint8_t* const* __restrict__ kvc_tab, const int* __restrict__ row_slot, const float* __restrict__ qg,
                                     float* __restrict__ score, const int* __restrict__ params, int T, int G, int n, int ld_score, int ps) {
    extern __shared__ float sq[];   // [G][n]: one row's query groups at a time
    const int p = blockIdx.x;
    __shared__ float red[256];
    for (int t = 0; t < T; ++t) {
        __syncthreads();
        for (int i = threadIdx.x; i < G * n; i += blockDim.x) sq[i] = qg[(size_t) t * G * n + i];
        __syncthreads();
        const int n_cand = params[t * 8 + 0];
        if (p >= n_cand) continue;
        const uint8_t* row = kvc_tab[row_slot[t]] + (size_t) (p * ps + threadIdx.x) * q8_pitch(n);
        float best = -INFINITY;
        for (int g = 0; g < G; ++g) {
            const float* qq = sq + (size_t) g * n;
            float acc = 0.f;
            for (int b = 0; b < n; b += 32) {   // block by block: one scale read per 32 values
                const uint8_t* blk = row + (b >> 5) * 34;
                const float dsc = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(blk)));
                float a = 0.f;
                for (int i = 0; i < 32; ++i) a += qq[b + i] * (float) (int8_t) blk[2 + i];
                acc += dsc * a;
            }
            best = fmaxf(best, acc);
        }
        red[threadIdx.x] = best;
        __syncthreads();
        for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]); __syncthreads(); }
        if (threadIdx.x == 0) score[(size_t) t * ld_score + p] = red[0];
        __syncthreads();
    }
}
void head_group_mean(const float* q, float* qg, int T, int H, int G, int n, void* s) { k_head_group_mean<<<dim3(blocks_for(n), T), 256, 0, S(s)>>>(q, qg, H, G, n); }
void page_max(const float* tok_score, float* page_score, int ps, int n_pages, void* s) {
    if (n_pages <= 0) return;
    k_page_max<<<blocks_for(n_pages), 256, 0, S(s)>>>(tok_score, page_score, ps, n_pages);
}
void lod_score_tok_rows(uint8_t* const* kvc_tab, const int* row_slot, const float* qg, float* score, const int* params, int n_cand_max, int T, int G, int n,
                        int ld_score, int ps, void* s) {
    if (n_cand_max <= 0) return;
    k_lod_score_tok_rows<<<n_cand_max, ps, (size_t) G * n * sizeof(float), S(s)>>>(kvc_tab, row_slot, qg, score, params, T, G, n, ld_score, ps);
}
// ---- indexer-routed LoD: the DSA indexer's pool scores route the pages
// rows of f32 [n][D] zero-padded to DP columns as Q8_0 rows (the MMQ operand); thread per output row
__global__ void k_rows_q8_pad(const float* __restrict__ x, uint8_t* __restrict__ out, int n, int D, int DP) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    float v[32];
    for (int b = 0; b < DP / 32; ++b) {
        for (int i = 0; i < 32; ++i) { const int c = b * 32 + i; v[i] = c < D ? x[(size_t) r * D + c] : 0.f; }
        q8_block(v, out + (size_t) r * (DP / 32) * 34 + b * 34);
    }
}
__global__ void k_rows_q8_pad_rows(float* const* __restrict__ pooled_tab, uint8_t* const* __restrict__ q8_tab, const int* __restrict__ row_slot,
                                   const int* __restrict__ row_pos, int kpool, int D, int DP) {
    const int t = blockIdx.x, pos = row_pos[t];
    if ((pos + 1) % kpool != 0 || threadIdx.x != 0) return;
    const int p = (pos + 1) / kpool - 1, sl = row_slot[t];
    const float* x = pooled_tab[sl] + (size_t) p * D;
    uint8_t* out = q8_tab[sl] + (size_t) p * (DP / 32) * 34;
    float v[32];
    for (int b = 0; b < DP / 32; ++b) {
        for (int i = 0; i < 32; ++i) { const int c = b * 32 + i; v[i] = c < D ? x[c] : 0.f; }
        q8_block(v, out + b * 34);
    }
}
// f32 rows [n][D] -> [n][DP] zero-padded (the quantizer's input)
__global__ void k_pad_rows(const float* __restrict__ x, float* __restrict__ out, int n, int D, int DP) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * DP) return;
    const int r = i / DP, c = i % DP;
    out[i] = c < D ? x[(size_t) r * D + c] : 0.f;
}
// the indexer's pool scores of a batch of TB queries from the product S [TB*IH][ld]: score_p = max over t of
// sum_h relu(S[t*IH + h][p]) w[t][h]; folded into pool_score (running max unless `first`)
__global__ void k_idx_pool_score(const float* __restrict__ S_, int ld, int TB, int IH, const float* __restrict__ w, int n_pool,
                                 float* __restrict__ pool_score, int first) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pool) return;
    float m = first ? -INFINITY : pool_score[p];
    for (int t = 0; t < TB; ++t) {
        float tot = 0.f;
        for (int h = 0; h < IH; ++h) tot += fmaxf(S_[((size_t) t * IH + h) * ld + p], 0.f) * w[t * IH + h];
        m = fmaxf(m, tot);
    }
    pool_score[p] = m;
}
// page_score[p] = max over the page's ppp pools; rows variant per token row with its own candidate count
__global__ void k_pool_page_max_rows(const float* __restrict__ pool_score, int ld_pool, float* __restrict__ page_score, int ld_page,
                                     const int* __restrict__ params, int ppp) {
    const int t = blockIdx.y, p = blockIdx.x * blockDim.x + threadIdx.x, n_cand = params[t * 8 + 0];
    if (p >= n_cand) return;
    float m = -INFINITY;
    for (int j = 0; j < ppp; ++j) m = fmaxf(m, pool_score[(size_t) t * ld_pool + p * ppp + j]);
    page_score[(size_t) t * ld_page + p] = m;
}
void rows_q8_pad(const float* x, uint8_t* out, int n, int D, int DP, void* s) { if (n > 0) k_rows_q8_pad<<<blocks_for(n), 256, 0, S(s)>>>(x, out, n, D, DP); }
void rows_q8_pad_rows(float* const* pooled_tab, uint8_t* const* q8_tab, const int* row_slot, const int* row_pos, int T, int kpool, int D, int DP, void* s) {
    k_rows_q8_pad_rows<<<T, 32, 0, S(s)>>>(pooled_tab, q8_tab, row_slot, row_pos, kpool, D, DP);
}
void pad_rows(const float* x, float* out, int n, int D, int DP, void* s) { k_pad_rows<<<blocks_for(n * DP), 256, 0, S(s)>>>(x, out, n, D, DP); }
void idx_pool_score(const float* S_, int ld, int TB, int IH, const float* w, int n_pool, float* pool_score, bool first, void* s) {
    if (n_pool > 0) k_idx_pool_score<<<blocks_for(n_pool), 256, 0, S(s)>>>(S_, ld, TB, IH, w, n_pool, pool_score, first ? 1 : 0);
}
void pool_page_max_rows(const float* pool_score, int ld_pool, float* page_score, int ld_page, const int* params, int n_cand_max, int T, int ppp, void* s) {
    if (n_cand_max > 0) k_pool_page_max_rows<<<dim3(blocks_for(n_cand_max), T), 256, 0, S(s)>>>(pool_score, ld_pool, page_score, ld_page, params, ppp);
}
// ---- pool-level LoD selection (DSA's granularity): the top n_top pools of the candidate pages are read exactly (their
// KP cells), the sink pages and the local field exactly, and every candidate page without an open pool as a summary.
// params[t*8]: [0] n_cand pages, [1] n_top pools, [2] n_kv, [3] sink pages; out [4] n_sel.  One block per token.
__global__ void k_lod_select_pools(const float* __restrict__ pool_score, int* __restrict__ params, int ps, int KP, int32_t* __restrict__ sel, int ld_pool, int n_sel_max) {
    __shared__ int hist[256];
    __shared__ int scan[1024];
    __shared__ uint32_t s_prefix;
    __shared__ int s_remaining, s_base;
    __shared__ unsigned char open[8192];       // pages touched by an open pool
    const int t = blockIdx.x;
    pool_score += (size_t) t * ld_pool;
    params += t * 8;
    sel += (size_t) t * n_sel_max;
    const int n_cand = params[0], n_kv = params[2], sink = min(params[3], n_cand), ppp = ps / KP;   // sink pages exist only among the candidates
    const int pool_lo = sink * ppp, n_pool = n_cand * ppp;
    const int n_top = min(params[1], max(n_pool - pool_lo, 0));
    const int tid = threadIdx.x, nt = blockDim.x;
    for (int p = tid; p < n_cand; p += nt) open[p] = p < sink ? 1 : 0;
    __syncthreads();
    uint32_t prefix = 0;
    int remaining = n_top;
    if (n_top > 0) {
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            for (int b = tid; b < 256; b += nt) hist[b] = 0;
            __syncthreads();
            for (int q = pool_lo + tid; q < n_pool; q += nt) {
                const uint32_t key = f32_key(pool_score[q]);
                if (pass == 0 || (key >> (shift + 8)) == (prefix >> (shift + 8))) atomicAdd(&hist[(key >> shift) & 255], 1);
            }
            __syncthreads();
            if (tid == 0) {
                int acc = 0, b = 255;
                for (; b > 0; --b) { if (acc + hist[b] >= remaining) break; acc += hist[b]; }
                s_prefix = prefix | ((uint32_t) b << shift);
                s_remaining = remaining - acc;
            }
            __syncthreads();
            prefix = s_prefix; remaining = s_remaining;
            __syncthreads();
        }
    }
    const uint32_t thr = prefix;
    // the open pools' cells, in pool order: pass A (key > thr) then pass B (the first `remaining` with key == thr)
    if (tid == 0) s_base = sink * ps;   // the sink pages' cells come first
    for (int j = tid; j < sink * ps && j < n_kv; j += nt) sel[j] = j;
    __syncthreads();
    for (int round = 0; round < 2 && n_top > 0; ++round) {
        for (int q0 = pool_lo; q0 < n_pool; q0 += nt) {
            const int q = q0 + tid;
            int flag = 0;
            if (q < n_pool) { const uint32_t key = f32_key(pool_score[q]); flag = round == 0 ? key > thr : key == thr; }
            scan[tid] = flag;
            __syncthreads();
            for (int o = 1; o < nt; o <<= 1) { const int v = tid >= o ? scan[tid - o] : 0; __syncthreads(); scan[tid] += v; __syncthreads(); }
            const int total = scan[nt - 1];
            const bool take = flag && (round == 0 || scan[tid] - 1 < remaining);
            if (take) {
                const int rank = s_base + (scan[tid] - 1) * KP;
                for (int j = 0; j < KP; ++j) sel[rank + j] = q * KP + j;
                open[q / ppp] = 1;
            }
            __syncthreads();
            if (tid == 0) { s_base += (round == 0 ? total : min(total, remaining)) * KP; if (round == 1) s_remaining = remaining - min(total, remaining); }
            __syncthreads();
            if (round == 1) remaining = s_remaining;
            __syncthreads();
        }
    }
    // the local field
    const int loc0 = n_cand * ps, n_loc = n_kv - loc0;
    int base = s_base;
    for (int j = tid; j < n_loc; j += nt) sel[base + j] = loc0 + j;
    base += n_loc;
    __syncthreads();
    // the summaries: candidate pages (beyond the sink) with no open pool (none when params[5] says so: plain windowed DSA)
    const int no_sum = params[5];
    if (tid == 0) s_base = 0;
    __syncthreads();
    for (int p0 = 0; p0 < n_cand && !no_sum; p0 += nt) {
        const int p = p0 + tid;
        const int flag = p < n_cand ? !open[p] : 0;
        scan[tid] = flag;
        __syncthreads();
        for (int o = 1; o < nt; o <<= 1) { const int v = tid >= o ? scan[tid - o] : 0; __syncthreads(); scan[tid] += v; __syncthreads(); }
        if (flag) sel[base + s_base + scan[tid] - 1] = -(p + 1);
        __syncthreads();
        if (tid == 0) s_base += scan[nt - 1];
        __syncthreads();
    }
    if (tid == 0) params[4] = base + s_base;
}
// the gather for a device-sized selection: rows beyond params[4] (up to n_sel_host) are zero with bias -1e30
__global__ void k_lod_gather_dev(const uint8_t* __restrict__ kvc, const uint8_t* __restrict__ means, const int32_t* __restrict__ sel, const int* __restrict__ params,
                                 int n_sel_host, int pos0, int T, uint8_t* __restrict__ out, float* __restrict__ bias_col, float bias, int n) {
    const int c = blockIdx.x, pitch = q8_pitch(n), n_sel_dev = params[4];
    if (c >= n_sel_host + T) return;
    uint8_t* dst = out + (size_t) c * pitch;
    if (c < n_sel_host && c >= n_sel_dev) {
        for (int i = threadIdx.x; i < pitch; i += blockDim.x) dst[i] = 0;
        if (threadIdx.x == 0) bias_col[c] = -1e30f;
        return;
    }
    const int32_t e = c < n_sel_host ? sel[c] : pos0 + (c - n_sel_host);
    const uint8_t* src = sel_row(kvc, means, e, pitch);
    for (int i = threadIdx.x; i < pitch; i += blockDim.x) dst[i] = src[i];
    if (threadIdx.x == 0) bias_col[c] = e < 0 ? bias : 0.f;
}
void lod_select_pools(const float* pool_score, int* params, int T, int ps, int KP, int32_t* sel, int ld_pool, int n_sel_max, void* s) {
    k_lod_select_pools<<<T, 1024, 0, S(s)>>>(pool_score, params, ps, KP, sel, ld_pool, n_sel_max);
}
void lod_gather_dev(const uint8_t* kvc, const uint8_t* means, const int32_t* sel, const int* params, int n_sel_host, int pos0, int T, uint8_t* out, float* bias_col,
                    float bias, int n, void* s) {
    k_lod_gather_dev<<<n_sel_host + T, 256, 0, S(s)>>>(kvc, means, sel, params, n_sel_host, pos0, T, out, bias_col, bias, n);
}
void lod_page_mean(const uint8_t* kvc, uint8_t* means, int p0, int n_pages, int ps, int n, void* s) {
    if (n_pages <= 0) return;
    k_lod_page_mean<<<n_pages, 512, 0, S(s)>>>(kvc, means, p0, ps, n);
}
void lod_page_mean_rows(uint8_t* const* kvc_tab, uint8_t* const* means_tab, const int* row_slot, const int* row_pos, int T, int ps, int n, void* s) {
    k_lod_page_mean_rows<<<T, 512, 0, S(s)>>>(kvc_tab, means_tab, row_slot, row_pos, ps, n);
}
void lod_score_rows(uint8_t* const* means_tab, const int* row_slot, const float* q_abs, float* score, const int* params, int n_cand_max, int T, int n_head, int n,
                    int ld_score, void* s) {
    if (n_cand_max <= 0) return;
    k_lod_score_rows<<<dim3(blocks_for(n_cand_max, 8), T), 256, 0, S(s)>>>(means_tab, row_slot, q_abs, score, params, n_head, n, ld_score);
}
void colmax(const float* S_, int ld, int rows, int n_cols, float* score, bool first, void* s) {
    if (n_cols <= 0) return;
    k_colmax<<<blocks_for(n_cols), 256, 0, S(s)>>>(S_, ld, rows, n_cols, score, first ? 1 : 0);
}
void lod_select(const float* score, int* params, int T, int ps, int32_t* sel, int ld_score, int n_sel_max, void* s) {
    k_lod_select<<<T, 1024, 0, S(s)>>>(score, params, ps, sel, ld_score, n_sel_max);
}
void lod_gather(const uint8_t* kvc, const uint8_t* means, const int32_t* sel, int n_sel, int pos0, int T, uint8_t* out, float* bias_col, float bias, int n, void* s) {
    k_lod_gather<<<n_sel + T, 256, 0, S(s)>>>(kvc, means, sel, n_sel, pos0, T, out, bias_col, bias, n);
}
void lod_mask(uint32_t* mask, int words, int TB, int n_sel, int T, int t0, void* s) { k_lod_mask<<<TB, 256, 0, S(s)>>>(mask, words, n_sel, T, t0); }
void combine_tok(const float* y, const float* pair_w, const int* tok_pairs, int K, float* out, int T, int n, void* s) {
    k_combine_tok<<<dim3(blocks_for(n), T), 256, 0, S(s)>>>(y, pair_w, tok_pairs, K, out, n);
}
void swiglu_clamp_rows(const float* g, const float* u, float* h, const int* n_rows, int max_rows, int n, float limit, void* s) {
    k_swiglu_clamp_rows<<<dim3(blocks_for(n), max_rows), 256, 0, S(s)>>>(g, u, h, n_rows, n, limit);
}
void argmax_rows(const float* x, int rows, int n, int32_t* out, int* scratch_idx, float* scratch_val, void* s) {
    constexpr int B = 128;
    k_argmax_rows1<<<dim3(B, rows), 1024, 0, S(s)>>>(x, n, scratch_idx, scratch_val);
    k_argmax_rows2<<<rows, 32, 0, S(s)>>>(scratch_idx, scratch_val, B, out);
}
void dequant_q8_0_rows(const uint8_t* w, const int32_t* rows, int T, float* out, int n, bool zero_first, void* s) {
    k_dequant_q8_0_rows<<<dim3(blocks_for(n), T), 256, 0, S(s)>>>(w, rows, out, n, zero_first ? 1 : 0);
}
void concat_rows(const float* a, const float* b, float* out, int T, int n, void* s) {
    k_concat_rows<<<dim3(blocks_for(2 * n), T), 256, 0, S(s)>>>(a, b, out, n);
}
void add_rows(float* y, const float* x, int n_tot, void* s) { k_add_rows<<<blocks_for(n_tot), 256, 0, S(s)>>>(y, x, n_tot); }

}  // namespace strata::glm5
