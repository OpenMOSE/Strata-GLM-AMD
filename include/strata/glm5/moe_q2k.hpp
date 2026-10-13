// include/strata/glm5/moe_q2k.hpp - the Q2_K expert products of a prompt window on gfx11 WMMA (int8 matrix cores).
//
// A Q2_K row is super-blocks of 256 weights: 16 sub-blocks of 16 with 4-bit scale sc_s and min m_s, a super-block
// scale d and dmin (f16): w = d*sc_s*q - dmin*m_s, q in 0..3.  The activations are int8 per 256-value block (scale dx)
// with the sum of every 16 (int).  Over one super-block of row r and column c:
//     sum_k w_k x_k = dx * ( d * sum_s sc_s (q.a)_s  -  dmin * sum_s m_s suma_s )
// The first term is one int8 MMA chain with A = sc_s * q (0..45 fits int8: a byte permute of {0, sc, 2sc, 3sc} by the
// 2-bit codes), the second two MMAs with A = m_s and B = suma split into a low byte and a high byte.
#pragma once
#include <cstdint>

namespace strata::glm5::q2k {
bool available();   ///< the device code is built (gfx11 WMMA)
/// int8 rows [rows][K] (row = ids ? x row ids[row] : row), scales [rows][K/256] f32, sums [rows][K/16] int32.  K % 256 == 0.
/// row_lo / row_hi (nullable, device): only the rows in [*row_lo, *row_hi) are written
/// packed (nullable): the sums packed per (row, super-block) as pack_sums does, written here instead of `sums`
void quantize_act(const float* x, const int32_t* ids, int rows, int K, int8_t* codes, float* scales, int32_t* sums, void* stream,
                  const int* row_lo = nullptr, const int* row_hi = nullptr, int32_t* packed = nullptr);
/// dst[(bounds[e] + c) * ld_dst + r] = w_e[r] . act[bounds[e] + c] for expert e in [0, n), its columns [bounds[e], bounds[e+1]),
/// its matrix at w + e * expert_bytes + mat_off ([n_rows][K] Q2_K, row pitch K/256*84 bytes).  max_cols bounds the grid.
void experts(const uint8_t* w, uint64_t expert_bytes, uint64_t mat_off, int n_rows, int K, const int32_t* bounds, int n, int max_cols,
             const int8_t* codes, const float* scales, const int32_t* packed, float* dst, int ld_dst, void* stream, const int32_t* skip = nullptr);
///< packed: the activation sums packed per (column, super-block) by pack_sums
///< skip: per expert e (nullable), nonzero = its columns are computed elsewhere (the CPU); the kernel leaves them

/// The decode path (a window of at most 8 rows): the Q8_1 columns of the window (the grouped MMVQ's operand, pitch
/// (K/32)*36 bytes) unpacked once into int8 codes [cols][K], per-32 scales d8 [cols][K/32] and per-16 sums [cols][K/16].
void dec_prepare(const void* x_q8_1, int K, int n_cols, int8_t* codes, float* d8, int32_t* usum, void* stream);
/// y[(orow[g] + c) * n_out + r] = w_g[r] . act[col0[g] + c] for the groups [*g_begin, *g_end), c < ncols[g] (<= max_cols <= 8);
/// w_g = wp[g] ([n_out][K] Q2_K); K in {2048, 4096, 8192}.  Lanes stride over a row's super-blocks: bandwidth-bound.
void dec_grouped(const void* const* wp, const int* col0, const int* ncols, const int* orow, const int* g_begin, const int* g_end,
                 const int8_t* codes, const float* d8, const int32_t* usum, float* y, int K, int n_out, int max_cols, void* stream);
/// The decode products on WMMA (gfx11): the window's activations quantized by quantize_act (codes [cols][K], scales
/// [cols][K/256], sums [cols][K/16]) with the sums packed once per (column, super-block) by pack_sums (packed [cols][K/256][8]);
/// then y[(orow[g] + c) * n_out + r] = w_g[r] . act[col0[g] + c] for the groups [*g_begin, *g_end), c < ncols[g] <= 16.
void pack_sums(const int32_t* sums, int cols, int K, int32_t* packed, void* stream, const int* row_lo = nullptr, const int* row_hi = nullptr);
void dec_wmma(const void* const* wp, const int* col0, const int* ncols, const int* orow, const int* g_begin, const int* g_end,
              const int8_t* codes, const float* scales, const int32_t* packed, float* y, int K, int n_out, void* stream);
}  // namespace strata::glm5::q2k
