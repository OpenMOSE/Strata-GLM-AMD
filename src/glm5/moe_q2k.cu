// src/glm5/moe_q2k.cu - see include/strata/glm5/moe_q2k.hpp
#include "strata/glm5/moe_q2k.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

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
                            float* __restrict__ scales, int32_t* __restrict__ sums, const int* __restrict__ row_lo, const int* __restrict__ row_hi,
                            int32_t* __restrict__ packed) {
    __shared__ float red[8];
    __shared__ int s16[16];
    const int row = blockIdx.y, blk = blockIdx.x, k = blk * 256 + threadIdx.x;
    if ((row_lo && row < *row_lo) || (row_hi && row >= *row_hi)) return;
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
    if (threadIdx.x == 0) scales[(size_t) row * (K / 256) + blk] = d;
    if (!packed) {
        if ((threadIdx.x & 15) == 0) sums[(size_t) row * (K / 16) + k / 16] = s;
        return;
    }
    // this super-block's 16 sums packed here (k_pack_sums' layout): thread i < 4 the low bytes of sums 4 i .. 4 i + 3, i >= 4 the high bytes
    if ((threadIdx.x & 15) == 0) s16[threadIdx.x >> 4] = s;
    __syncthreads();
    if (threadIdx.x < 8) {
        const int i = threadIdx.x;
        uint32_t v = 0;
#pragma unroll
        for (int t = 0; t < 4; ++t) { const int xq = s16[4 * (i & 3) + t]; v |= (uint32_t) ((i < 4 ? xq : (xq >> 8)) & 0xff) << (8 * t); }
        packed[((size_t) row * (K / 256) + blk) * 8 + i] = (int32_t) v;
    }
}

// ---- the expert products: block = 8 waves x 16 weight rows (128 rows), NT column tiles of 16; grid (row tiles, col tiles, experts)
template <int NT>
__global__ void __launch_bounds__(kThreads) k_experts(const uint8_t* __restrict__ w, uint64_t expert_bytes, uint64_t mat_off, int n_rows, int K,
                                                        const int32_t* __restrict__ bounds, const int8_t* __restrict__ codes, const float* __restrict__ scales,
                                                        const int32_t* __restrict__ packed, float* __restrict__ dst, int ld_dst, const int32_t* __restrict__ skip) {
#if STRATA_Q2K_WMMA
    const int e = blockIdx.z, c_first = bounds[e], n_cols = bounds[e + 1] - c_first;
    const int col0 = blockIdx.y * (16 * NT);
    if (col0 >= n_cols) return;
    if (skip && skip[e]) return;   // its rows come from the CPU
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, r = lane & 15, half = lane >> 4;
    const int row = blockIdx.x * kRowTile + wave * kRowsPerWave + r;
    if (row >= n_rows) return;   // n_rows % 16 == 0 keeps whole waves
    const int nsb = K / 256, rowbytes = nsb * kSb;
    const uint8_t* wrow = w + (uint64_t) e * expert_bytes + mat_off + (uint64_t) row * rowbytes;
    // the B rows this lane reads: column col0 + 16 n + r of the expert (clamped to the last one)
    const int8_t* brow[NT];
    const int32_t* prow[NT];   // the packed sums of column c: [nsb][8] (lo bytes x4, hi bytes x4)
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int c = min(col0 + 16 * n + r, n_cols - 1) + c_first;
        brow[n] = codes + (size_t) c * K; prow[n] = packed + (size_t) c * nsb * 8;
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
            const i32x4 blo = *reinterpret_cast<const i32x4*>(prow[n] + sb * 8), bhi = *reinterpret_cast<const i32x4*>(prow[n] + sb * 8 + 4);
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
    (void) w; (void) expert_bytes; (void) mat_off; (void) n_rows; (void) K; (void) bounds; (void) codes; (void) scales; (void) packed; (void) dst; (void) ld_dst; (void) skip;
#endif
}

// ---- the decode path: lanes stride over a row's super-blocks (coalesced row streaming), the dot products in integers
// per 32-weight block (sc*q by byte permute, dp4a with the int8 activations; the min term from the per-16 activation sums),
// two FMAs per block, one wave reduction per row and column.  Block = 8 waves x kDecRowsPerWave rows of one group.
constexpr int kDecRowsPerWave = 8, kDecWaves = 8, kDecThreads = 32 * kDecWaves, kDecRowTile = kDecRowsPerWave * kDecWaves, kDecMaxCols = 8;

__global__ void k_dec_prepare(const uint8_t* __restrict__ q8, int K, int pitch, int8_t* __restrict__ codes, float* __restrict__ d8, int32_t* __restrict__ usum) {
    const int b = blockIdx.x, c = blockIdx.y, i = threadIdx.x;   // block b of 32 values of column c
    const uint8_t* blk = q8 + (size_t) c * pitch + (size_t) b * 36;
    const int8_t q = (int8_t) blk[4 + i];
    codes[(size_t) c * K + b * 32 + i] = q;
    int s = q;
    for (int o = 8; o > 0; o >>= 1) s += __shfl_xor(s, o, 32);
    if ((i & 15) == 0) usum[(size_t) c * (K / 16) + 2 * b + (i >> 4)] = s;
    if (i == 0) d8[(size_t) c * (K / 32) + b] = __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(blk)));
}

__device__ __forceinline__ int dec_dp4a(int a, int b, int c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    return __dp4a(a, b, c);
#endif
}

template <int SPL, int NT>   // SPL: sub-blocks (of 16 weights) per lane - 8 for K = 4096 (2 lanes per super-block), 4 for 2048, 16 for 8192; NT: columns per tile
__global__ void __launch_bounds__(kDecThreads) k_dec(const void* const* __restrict__ wp, const int* __restrict__ col0, const int* __restrict__ ncols,
                                                     const int* __restrict__ orow, const int* __restrict__ g_begin, const int* __restrict__ g_end,
                                                     const int8_t* __restrict__ codes, const float* __restrict__ d8, const int32_t* __restrict__ usum,
                                                     float* __restrict__ y, int K, int n_out) {
    extern __shared__ int s_mem[];
    constexpr int LPS = 16 / SPL;                         // lanes per super-block
    const int nsb = K / 256, rowbytes = nsb * kSb;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int sb = lane / LPS, s0 = (lane % LPS) * SPL;   // this lane's super-block and first sub-block
    const int gb = *g_begin, ge = *g_end;
    const int kw = K / 4, ku = K / 16, kd = K / 32;
    for (int g = gb + (int) blockIdx.y; g < ge; g += (int) gridDim.y) {
        const int nc = ncols[g];
        if (nc <= 0) continue;
        const int c0 = col0[g];
        int* s_codes = s_mem;                               // [nc][K/4] words
        int* s_usum = s_mem + nc * kw;                      // [nc][K/16]
        float* s_d8 = reinterpret_cast<float*>(s_mem + nc * (kw + ku));   // [nc][K/32]
        __syncthreads();
        {
            const int* src = reinterpret_cast<const int*>(codes) + (size_t) c0 * kw;
            for (int i = threadIdx.x; i < nc * kw; i += kDecThreads) s_codes[i] = src[i];
            const int* su = usum + (size_t) c0 * ku;
            for (int i = threadIdx.x; i < nc * ku; i += kDecThreads) s_usum[i] = su[i];
            const float* sd = d8 + (size_t) c0 * kd;
            for (int i = threadIdx.x; i < nc * kd; i += kDecThreads) s_d8[i] = sd[i];
        }
        __syncthreads();
        const uint8_t* w = static_cast<const uint8_t*>(wp[g]);
        const int row0 = (int) blockIdx.x * kDecRowTile + wave * kDecRowsPerWave;
        const int wb = 4 + 8 * (s0 >> 3);                   // the 8 code words of this lane's 128-weight half
        // the row's 13 words, the next row's loaded while this one is computed
        uint32_t scw[4], qsw[8], dmw;
        auto load = [&](int row) {
            const uint32_t* sbw = reinterpret_cast<const uint32_t*>(w + (size_t) row * rowbytes + (size_t) sb * kSb);
#pragma unroll
            for (int i = 0; i < 4; ++i) scw[i] = sbw[i];
#pragma unroll
            for (int i = 0; i < 8; ++i) qsw[i] = sbw[wb + i];
            dmw = sbw[20];
        };
        if (row0 < n_out) load(row0);
        for (int rr = 0; rr < kDecRowsPerWave; ++rr) {
            const int row = row0 + rr;
            if (row >= n_out) break;
            uint32_t scw_c[4], qsw_c[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) scw_c[i] = scw[i];
#pragma unroll
            for (int i = 0; i < 8; ++i) qsw_c[i] = qsw[i];
            const float d = __half2float(__ushort_as_half((unsigned short) (dmw & 0xffff)));
            const float dmin = __half2float(__ushort_as_half((unsigned short) (dmw >> 16)));
            if (rr + 1 < kDecRowsPerWave && row + 1 < n_out) load(row + 1);
            // the sc*q bytes of this lane's sub-blocks (shared by every column), and their min nibbles
            int aq[SPL][4];
            int mq[SPL];
#pragma unroll
            for (int sp = 0; sp < SPL; ++sp) {
                const int s = s0 + sp;
                const uint32_t sc = (scw_c[s >> 2] >> (8 * (s & 3))) & 0x0f;
                mq[sp] = (int) ((scw_c[s >> 2] >> (8 * (s & 3) + 4)) & 0x0f);
                const uint32_t lut = sc * 0x03020100u;
                const int wi = 4 * (s & 1), sh = 2 * ((s & 7) >> 1);
#pragma unroll
                for (int t = 0; t < 4; ++t) aq[sp][t] = (int) __builtin_amdgcn_perm(0u, lut, (qsw_c[wi + t] >> sh) & 0x03030303u);
            }
            for (int cb = 0; cb < nc; cb += NT) {
                float acc[NT];
#pragma unroll
                for (int j = 0; j < NT; ++j) acc[j] = 0.f;
#pragma unroll
                for (int sp = 0; sp < SPL; sp += 2) {       // one Q8_1 block (32 weights = 2 sub-blocks) per step
                    int dot[NT], mn[NT];
#pragma unroll
                    for (int j = 0; j < NT; ++j) { dot[j] = 0; mn[j] = 0; }
#pragma unroll
                    for (int ss = 0; ss < 2; ++ss) {
                        const int s = s0 + sp + ss, wofs = (sb * 256 + 16 * s) / 4;
#pragma unroll
                        for (int j = 0; j < NT; ++j) {
                            const int c = cb + j;
                            if (NT == 1 || c < nc) {
                                const int* act = s_codes + c * kw + wofs;
#pragma unroll
                                for (int t = 0; t < 4; ++t) dot[j] = dec_dp4a(aq[sp + ss][t], act[t], dot[j]);
                                mn[j] += mq[sp + ss] * s_usum[c * ku + sb * 16 + s];
                            }
                        }
                    }
                    const int b = sb * 8 + (s0 + sp) / 2;
#pragma unroll
                    for (int j = 0; j < NT; ++j)
                        if (NT == 1 || cb + j < nc) acc[j] += s_d8[(cb + j) * kd + b] * (d * (float) dot[j] - dmin * (float) mn[j]);
                }
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    if (NT == 1 || cb + j < nc) {
                        float v = acc[j];
                        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
                        if (lane == 0) y[(size_t) (orow[g] + cb + j) * n_out + row] = v;
                    }
                }
            }
        }
    }
}

// ---- the decode WMMA kernel: one 16-column tile per group (a decode window has at most 8 pairs on one expert), the
// activation sums pre-packed per (column, super-block) as the min term's B operands; block = 8 waves x 16 rows of one group.
__global__ void k_pack_sums(const int32_t* __restrict__ sums, int nsb, int32_t* __restrict__ packed, const int* __restrict__ row_lo, const int* __restrict__ row_hi) {
    // grid (nsb, cols), 8 threads: thread i < 4 packs the low bytes of sums[16 sb + 4 i .. + 3], i >= 4 the high bytes
    const int sb = blockIdx.x, c = blockIdx.y, i = threadIdx.x;
    if ((row_lo && c < *row_lo) || (row_hi && c >= *row_hi)) return;
    const int32_t* sp = sums + ((size_t) c * nsb + sb) * 16 + 4 * (i & 3);
    uint32_t v = 0;
#pragma unroll
    for (int t = 0; t < 4; ++t) { const int x = sp[t]; v |= (uint32_t) ((i < 4 ? x : (x >> 8)) & 0xff) << (8 * t); }
    packed[((size_t) c * nsb + sb) * 8 + i] = (int32_t) v;
}

__global__ void __launch_bounds__(kThreads) k_dec_wmma(const void* const* __restrict__ wp, const int* __restrict__ col0, const int* __restrict__ ncols,
                                                        const int* __restrict__ orow, const int* __restrict__ g_begin, const int* __restrict__ g_end,
                                                        const int8_t* __restrict__ codes, const float* __restrict__ scales, const int32_t* __restrict__ packed,
                                                        float* __restrict__ y, int K, int n_out) {
#if STRATA_Q2K_WMMA
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, r = lane & 15, half = lane >> 4;
    const int nsb = K / 256, rowbytes = nsb * kSb;
    const int gb = *g_begin, ge = *g_end;
    for (int g = gb + (int) blockIdx.y; g < ge; g += (int) gridDim.y) {
        const int nc = ncols[g];
        if (nc <= 0) continue;
        const int row = (int) blockIdx.x * kRowTile + wave * kRowsPerWave + r;
        if (row >= n_out) continue;
        const int cf = col0[g];
        const uint8_t* wrow = static_cast<const uint8_t*>(wp[g]) + (size_t) row * rowbytes;
        const int cb = cf + min(r, nc - 1);                          // the B row this lane reads (clamped to the last column)
        const int8_t* brow = codes + (size_t) cb * K;
        const int32_t* prow = packed + (size_t) cb * nsb * 8;
        float accf[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) accf[q] = 0.f;
        const int cq = cf + min(2 * 0 + half, nc - 1);              // (per q below)
        (void) cq;
        for (int sb = 0; sb < nsb; ++sb) {
            const uint32_t* sbw = reinterpret_cast<const uint32_t*>(wrow + (size_t) sb * kSb);
            uint32_t scw[4], qsw[16];
#pragma unroll
            for (int i = 0; i < 4; ++i) scw[i] = sbw[i];
#pragma unroll
            for (int i = 0; i < 16; ++i) qsw[i] = sbw[4 + i];
            const uint32_t dmw = sbw[20];
            const float d = __half2float(__ushort_as_half((unsigned short) (dmw & 0xffff)));
            const float dmin = __half2float(__ushort_as_half((unsigned short) (dmw >> 16)));
            i32x8 acc = i32x8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < 16; ++s) {
                const uint32_t sc = (scw[s >> 2] >> (8 * (s & 3))) & 0x0f;
                const uint32_t lut = sc * 0x03020100u;
                const int wi = 8 * (s >> 3) + 4 * (s & 1), sh = 2 * ((s & 7) >> 1);
                i32x4 a;
#pragma unroll
                for (int t = 0; t < 4; ++t) a[t] = (int) __builtin_amdgcn_perm(0u, lut, (qsw[wi + t] >> sh) & 0x03030303u);
                const i32x4 b = *reinterpret_cast<const i32x4*>(brow + sb * 256 + 16 * s);
                acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, b, true, a, acc, false);
            }
            i32x4 am;
#pragma unroll
            for (int i = 0; i < 4; ++i) am[i] = (int) ((scw[i] >> 4) & 0x0f0f0f0fu);
            const i32x4 blo = *reinterpret_cast<const i32x4*>(prow + sb * 8), bhi = *reinterpret_cast<const i32x4*>(prow + sb * 8 + 4);
            i32x8 mlo = i32x8{0, 0, 0, 0, 0, 0, 0, 0}, mhi = i32x8{0, 0, 0, 0, 0, 0, 0, 0};
            mlo = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, blo, false, am, mlo, false);
            mhi = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, bhi, false, am, mhi, false);
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const int cc = cf + min(2 * q + half, nc - 1);
                const float dx = scales[(size_t) cc * nsb + sb];
                accf[q] += dx * (d * (float) acc[q] - dmin * (float) (mhi[q] * 256 + mlo[q]));
            }
        }
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int c = 2 * q + half;
            if (c < nc) y[(size_t) (orow[g] + c) * n_out + row] = accf[q];
        }
    }
#else
    (void) wp; (void) col0; (void) ncols; (void) orow; (void) g_begin; (void) g_end; (void) codes; (void) scales; (void) packed; (void) y; (void) K; (void) n_out;
#endif
}
}  // namespace

bool available() { return STRATA_Q2K_WMMA != 0 || true; }

void quantize_act(const float* x, const int32_t* ids, int rows, int K, int8_t* codes, float* scales, int32_t* sums, void* s, const int* row_lo,
                  const int* row_hi, int32_t* packed) {
    if (K % 256) throw std::runtime_error("q2k: K must be a multiple of 256");
    if (rows <= 0) return;
    k_quant_act<<<dim3((unsigned) (K / 256), (unsigned) rows), 256, 0, S(s)>>>(x, ids, K, codes, scales, sums, row_lo, row_hi, packed);
}

void experts(const uint8_t* w, uint64_t expert_bytes, uint64_t mat_off, int n_rows, int K, const int32_t* bounds, int n, int max_cols,
             const int8_t* codes, const float* scales, const int32_t* packed, float* dst, int ld_dst, void* s, const int32_t* skip) {
    if (n <= 0 || max_cols <= 0) return;
    if (n_rows % kRowTile) throw std::runtime_error("q2k: n_rows must be a multiple of " + std::to_string(kRowTile));
    const dim3 grid((unsigned) (n_rows / kRowTile), (unsigned) ((max_cols + 31) / 32), (unsigned) n);
    k_experts<2><<<grid, kThreads, 0, S(s)>>>(w, expert_bytes, mat_off, n_rows, K, bounds, codes, scales, packed, dst, ld_dst, skip);
}

void dec_prepare(const void* x_q8_1, int K, int n_cols, int8_t* codes, float* d8, int32_t* usum, void* s) {
    if (n_cols <= 0) return;
    if (K % 256) throw std::runtime_error("q2k dec: K must be a multiple of 256");
    k_dec_prepare<<<dim3((unsigned) (K / 32), (unsigned) n_cols), 32, 0, S(s)>>>(static_cast<const uint8_t*>(x_q8_1), K, (K / 32) * 36, codes, d8, usum);
}

void dec_grouped(const void* const* wp, const int* col0, const int* ncols, const int* orow, const int* g_begin, const int* g_end,
                 const int8_t* codes, const float* d8, const int32_t* usum, float* y, int K, int n_out, int max_cols, void* s) {
    if (max_cols <= 0 || n_out <= 0) return;
    if (max_cols > kDecMaxCols) throw std::runtime_error("q2k dec: at most 8 columns per group");
    const size_t smem = (size_t) max_cols * ((size_t) K + K / 4 + K / 8);
    const dim3 grid((unsigned) ((n_out + kDecRowTile - 1) / kDecRowTile), 32u);
    auto launch = [&](auto spl, auto nt) {
        k_dec<decltype(spl)::value, decltype(nt)::value><<<grid, kDecThreads, smem, S(s)>>>(wp, col0, ncols, orow, g_begin, g_end, codes, d8, usum, y, K, n_out);
    };
    auto by_nt = [&](auto spl) {
        if (max_cols <= 1) launch(spl, std::integral_constant<int, 1>{});
        else if (max_cols <= 2) launch(spl, std::integral_constant<int, 2>{});
        else if (max_cols <= 4) launch(spl, std::integral_constant<int, 4>{});
        else launch(spl, std::integral_constant<int, 8>{});
    };
    switch (K) {
    case 2048: by_nt(std::integral_constant<int, 4>{}); break;
    case 4096: by_nt(std::integral_constant<int, 8>{}); break;
    case 8192: by_nt(std::integral_constant<int, 16>{}); break;
    default: throw std::runtime_error("q2k dec: K must be 2048, 4096 or 8192");
    }
}

void pack_sums(const int32_t* sums, int cols, int K, int32_t* packed, void* s, const int* row_lo, const int* row_hi) {
    if (cols <= 0) return;
    k_pack_sums<<<dim3((unsigned) (K / 256), (unsigned) cols), 8, 0, S(s)>>>(sums, K / 256, packed, row_lo, row_hi);
}

void dec_wmma(const void* const* wp, const int* col0, const int* ncols, const int* orow, const int* g_begin, const int* g_end,
              const int8_t* codes, const float* scales, const int32_t* packed, float* y, int K, int n_out, void* s) {
    if (n_out <= 0) return;
    if (n_out % kRowTile) throw std::runtime_error("q2k dec: n_out must be a multiple of " + std::to_string(kRowTile));
    const dim3 grid((unsigned) (n_out / kRowTile), 32u);
    k_dec_wmma<<<grid, kThreads, 0, S(s)>>>(wp, col0, ncols, orow, g_begin, g_end, codes, scales, packed, y, K, n_out);
}
}  // namespace strata::glm5::q2k
