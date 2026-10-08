// src/glm5/moe_q2k.cu - see include/strata/glm5/moe_q2k.hpp
#include "strata/glm5/moe_q2k.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace strata::glm5::q2k {
namespace {
inline cudaStream_t S(void* s) { return static_cast<cudaStream_t>(s); }
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define STRATA_Q2K_WMMA 1
#else
#define STRATA_Q2K_WMMA 0
#endif
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));

constexpr int kSb = 84;          // bytes per Q2_K super-block: scales[16] qs[64] d dmin
constexpr int kRowsPerWave = 16, kWaves = 8, kThreads = 32 * kWaves, kRowTile = kRowsPerWave * kWaves;

// ---- the activation quantizer: block per (row, 256-block), thread per value
__global__ void k_quant_act(const float* __restrict__ x, const int32_t* __restrict__ ids, int K, int8_t* __restrict__ codes,
                            float* __restrict__ scales, int32_t* __restrict__ sums) {
    __shared__ float red[8];
    const int row = blockIdx.y, blk = blockIdx.x, k = blk * 256 + threadIdx.x;
    const int src = ids ? ids[row] : row;
    const float v = x[(size_t) src * K + k];
    float am = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor(am, o, 32));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = am;
    __syncthreads();
    am = red[0];
    for (int w = 1; w < 8; ++w) am = fmaxf(am, red[w]);
    const float d = am / 127.f, inv = am > 0.f ? 127.f / am : 0.f;
    const int q = __float2int_rn(v * inv);
    codes[(size_t) row * K + k] = (int8_t) q;
    int s = q;
    for (int o = 8; o > 0; o >>= 1) s += __shfl_xor(s, o, 32);   // the 16-lane sums (lanes 0..15 and 16..31 of a wave)
    if ((threadIdx.x & 15) == 0) sums[(size_t) row * (K / 16) + k / 16] = s;
    if (threadIdx.x == 0) scales[(size_t) row * (K / 256) + blk] = d;
}

// ---- the expert products: block = 8 waves x 16 weight rows (128 rows), NT column tiles of 16; grid (row tiles, col tiles, experts)
template <int NT>
__global__ void __launch_bounds__(kThreads) k_experts(const uint8_t* __restrict__ w, uint64_t expert_bytes, uint64_t mat_off, int n_rows, int K,
                                                        const int32_t* __restrict__ bounds, const int8_t* __restrict__ codes, const float* __restrict__ scales,
                                                        const int32_t* __restrict__ sums, float* __restrict__ dst, int ld_dst) {
#if STRATA_Q2K_WMMA
    const int e = blockIdx.z, c_first = bounds[e], n_cols = bounds[e + 1] - c_first;
    const int col0 = blockIdx.y * (16 * NT);
    if (col0 >= n_cols) return;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, r = lane & 15, half = lane >> 4;
    const int row = blockIdx.x * kRowTile + wave * kRowsPerWave + r;
    if (row >= n_rows) return;   // n_rows % 16 == 0 keeps whole waves
    const int nsb = K / 256, rowbytes = nsb * kSb;
    const uint8_t* wrow = w + (uint64_t) e * expert_bytes + mat_off + (uint64_t) row * rowbytes;
    // the B rows this lane reads: column col0 + 16 n + r of the expert (clamped to the last one)
    const int8_t* brow[NT];
    const int32_t* srow[NT];
    const float* xrow[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int c = min(col0 + 16 * n + r, n_cols - 1) + c_first;
        brow[n] = codes + (size_t) c * K; srow[n] = sums + (size_t) c * (K / 16); xrow[n] = scales + (size_t) c * nsb;
    }
    float accf[NT][8];
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int q = 0; q < 8; ++q) accf[n][q] = 0.f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint32_t* sbw = reinterpret_cast<const uint32_t*>(wrow + (size_t) sb * kSb);   // 4-byte aligned (84 = 21 words)
        uint32_t scw[4], qsw[16];
#pragma unroll
        for (int i = 0; i < 4; ++i) scw[i] = sbw[i];
#pragma unroll
        for (int i = 0; i < 16; ++i) qsw[i] = sbw[4 + i];
        const uint32_t dmw = sbw[20];
        const float d = __half2float(__ushort_as_half((unsigned short) (dmw & 0xffff)));
        const float dmin = __half2float(__ushort_as_half((unsigned short) (dmw >> 16)));
        i32x8 acc[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n) acc[n] = i32x8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            const uint32_t sc = (scw[s >> 2] >> (8 * (s & 3))) & 0x0f;
            const uint32_t lut = sc * 0x03020100u;                               // bytes {0, sc, 2sc, 3sc}
            const int wi = 8 * (s >> 3) + 4 * (s & 1), sh = 2 * ((s & 7) >> 1);
            i32x4 a;
#pragma unroll
            for (int t = 0; t < 4; ++t) a[t] = (int) __builtin_amdgcn_perm(0u, lut, (qsw[wi + t] >> sh) & 0x03030303u);
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const i32x4 b = *reinterpret_cast<const i32x4*>(brow[n] + sb * 256 + 16 * s);
                // the operand order: lane l's result element e is sum_k (first operand of lane 2e + l/16) . (second operand of lane l%16),
                // so the activations go first: lane l holds weight row l%16 and activation columns 2e + l/16 (measured on gfx1100)
                acc[n] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, b, true, a, acc[n], false);
            }
        }
        // the min term: A = m_s (16 bytes), B = suma_s split into a low byte (unsigned) and a high byte (signed)
        i32x4 am;
#pragma unroll
        for (int i = 0; i < 4; ++i) am[i] = (int) ((scw[i] >> 4) & 0x0f0f0f0fu);
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int32_t* sp = srow[n] + sb * 16;
            i32x4 blo, bhi;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t lo = 0, hi = 0;
#pragma unroll
                for (int t = 0; t < 4; ++t) { const int v = sp[4 * i + t]; lo |= (uint32_t) (v & 0xff) << (8 * t); hi |= (uint32_t) ((v >> 8) & 0xff) << (8 * t); }
                blo[i] = (int) lo; bhi[i] = (int) hi;
            }
            i32x8 mlo = i32x8{0, 0, 0, 0, 0, 0, 0, 0}, mhi = i32x8{0, 0, 0, 0, 0, 0, 0, 0};
            mlo = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, blo, false, am, mlo, false);
            mhi = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, bhi, false, am, mhi, false);
            // flush: column 2q + half of this tile
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const int cc = min(col0 + 16 * n + 2 * q + half, n_cols - 1) + c_first;
                const float dx = scales[(size_t) cc * nsb + sb];
                accf[n][q] += dx * (d * (float) acc[n][q] - dmin * (float) (mhi[q] * 256 + mlo[q]));
            }
        }
    }
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int c = col0 + 16 * n + 2 * q + half;
            if (c < n_cols) dst[(size_t) (c_first + c) * ld_dst + row] = accf[n][q];
        }
#else
    (void) w; (void) expert_bytes; (void) mat_off; (void) n_rows; (void) K; (void) bounds; (void) codes; (void) scales; (void) sums; (void) dst; (void) ld_dst;
#endif
}
}  // namespace

bool available() { return STRATA_Q2K_WMMA != 0 || true; }

void quantize_act(const float* x, const int32_t* ids, int rows, int K, int8_t* codes, float* scales, int32_t* sums, void* s) {
    if (K % 256) throw std::runtime_error("q2k: K must be a multiple of 256");
    if (rows <= 0) return;
    k_quant_act<<<dim3((unsigned) (K / 256), (unsigned) rows), 256, 0, S(s)>>>(x, ids, K, codes, scales, sums);
}

void experts(const uint8_t* w, uint64_t expert_bytes, uint64_t mat_off, int n_rows, int K, const int32_t* bounds, int n, int max_cols,
             const int8_t* codes, const float* scales, const int32_t* sums, float* dst, int ld_dst, void* s) {
    if (n <= 0 || max_cols <= 0) return;
    if (n_rows % kRowTile) throw std::runtime_error("q2k: n_rows must be a multiple of " + std::to_string(kRowTile));
    const dim3 grid((unsigned) (n_rows / kRowTile), (unsigned) ((max_cols + 31) / 32), (unsigned) n);
    k_experts<2><<<grid, kThreads, 0, S(s)>>>(w, expert_bytes, mat_off, n_rows, K, bounds, codes, scales, sums, dst, ld_dst);
}
}  // namespace strata::glm5::q2k
