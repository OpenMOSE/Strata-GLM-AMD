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
void quantize_act(const float* x, const int32_t* ids, int rows, int K, int8_t* codes, float* scales, int32_t* sums, void* stream);
/// dst[(bounds[e] + c) * ld_dst + r] = w_e[r] . act[bounds[e] + c] for expert e in [0, n), its columns [bounds[e], bounds[e+1]),
/// its matrix at w + e * expert_bytes + mat_off ([n_rows][K] Q2_K, row pitch K/256*84 bytes).  max_cols bounds the grid.
void experts(const uint8_t* w, uint64_t expert_bytes, uint64_t mat_off, int n_rows, int K, const int32_t* bounds, int n, int max_cols,
             const int8_t* codes, const float* scales, const int32_t* sums, float* dst, int ld_dst, void* stream);
}  // namespace strata::glm5::q2k
