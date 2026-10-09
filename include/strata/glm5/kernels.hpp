// include/strata/glm5/kernels.hpp - the GLM-5-Next (GLM-5.3-Flash) decode kernels, transcribed from llama.cpp's
// glm5-next.cpp and the ggml CPU ops it calls (docs/GLM5_NEXT.md P2).  Reference: ref/glm5.py.
//
// Single-token decode, FP32 activations, every call enqueues on the given stream and allocates nothing.  All pointers
// are device pointers.  Weight layouts are the GGUF's own: a [n_in, n_out] GGUF tensor is n_out contiguous rows of n_in.
#pragma once

#include <cstdint>

namespace strata::glm5 {

/// y[o] = sum_i W[o * n_in + i] * x[i], F32 weights (the mHC mixes, the router, the indexer projections).
void f32_gemv(const float* W, const float* x, float* y, int n_in, int n_out, void* stream);

/// ggml_rms_norm (+ ggml_mul by w when w != null), out of place, `rows` rows of `cols`.
void rms_norm(const float* x, const float* w, float* y, int rows, int cols, float eps, void* stream);
/// ggml_norm (LLM_NORM): (x - mean) / sqrt(var + eps) * w + b, one row.
void layer_norm(const float* x, const float* w, const float* b, float* y, int n, float eps, void* stream);

/// mHC: the 24 mixes of rms_norm(R flat) and their nonlinearities.
///   R [hc][n_embd], fn [(2+hc)*hc][hc*n_embd], scale [3], base [(2+hc)*hc]
///   pre [hc] = sigmoid(mix*scale0 + base) + eps_hc;  post [hc] = 2 sigmoid(mix*scale1 + base)
///   comb [hc*hc], comb[src*hc + dst]: softmax over dst, + eps, normalize over src, then (over dst, over src) x (iters-1)
///   mixes [(2+hc)*hc] scratch (written), rs [1] scratch
/// `part` scratch: (2+hc)*hc * mhc_mix_slices() floats
int mhc_mix_slices();
void mhc_mix(const float* R, const float* fn, const float* scale, const float* base, float* mixes, float* part, float* rs,
             float* pre, float* post, float* comb, int n_embd, int hc, int iters, float eps_rms, float eps_hc,
             void* stream);
/// x[i] = sum_h R[h][i] * pre[h]
void mhc_pre_apply(const float* R, const float* pre, float* x, int n_embd, int hc, void* stream);
/// R_out[d][i] = y[i] * post[d] + sum_s R_in[s][i] * comb[s*hc + d]  (R_out may alias R_in)
void mhc_post(const float* y, const float* R_in, const float* post, const float* comb, float* R_out, int n_embd,
              int hc, void* stream);

/// KDA.  conv: out[c] = silu(sum_k w[c*d_conv + k] * window_k[c]), window = (state[0..d_conv-2][c], x[c]); the state
/// shifts by one.  w is the GGUF [d_conv, 1, d_inner] tensor (d_conv contiguous).
void kda_conv_silu(const float* x, float* conv_state, const float* w, float* out, int d_inner, int d_conv,
                   void* stream);
/// x[h][.] /= max(||x[h]||, eps) for n_head rows of S (ggml_l2_norm)
void l2_norm_rows(float* x, int n_head, int S, float eps, void* stream);
/// g[h][i] = lower * sigmoid(-((g_in[h][i] + dt_b[h*S+i]) * A[h]))   (lower > -inf)
///         = softplus(g_in + dt_b) * A[h]                            (otherwise)
void kda_gate(const float* g_in, const float* dt_b, const float* A, float lower, float* g, int n_head, int S,
              void* stream);
void sigmoid_inplace(float* x, int n, void* stream);
/// One step of the KDA recurrence, in place on state [n_head][S_i][S_j] (ggml's S[i][j]: i = key dim, j = value dim):
///   S[i,:] *= exp(g[i]);  d[j] = (v[j] - sum_i S[i,j] k[i]) * beta;  S[i,j] += k[i] d[j];  o[j] = sum_i S[i,j] q[i] / sqrt(S)
void kda_step(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
              float* o, int n_head, int S, void* stream);
/// normed[h][.] = rms_norm(o[h]) * w;  y = normed * sigmoid(g2)
void kda_out_gate(const float* o, const float* w, const float* g2, float* normed, float* y, int n_head, int S,
                  float eps, void* stream);

/// DSA.  q_abs[h][j] = sum_i wk_b[h][j][i] * q[h][i]   (wk_b bf16, GGUF [head_k, kv_lora, n_head])
void dsa_absorb(const uint16_t* wk_b, const float* q, float* q_abs, int n_head, int kv_lora, int head_k,
                void* stream);
/// pooled[d] = sum_j softmax_j(ig[j0+j][d] + ape[j][d]) * ik[j0+j][d] over the kpool rows starting at j0
void dsa_pool(const uint8_t* ik_cache, const uint8_t* ig_cache, const float* ape, float* pooled, int j0, int n_pools, int kpool, int D,
              void* stream);   ///< pools p0.. (n_pools of them) over cells j0 + y*kpool: pooled[y]
/// score[p] = sum_h relu(iq[h] . pooled[p]) * w[h]
void dsa_score(const float* pooled, const float* iq, const float* w, float* score, const int* params, int n_pool_max, int TB, int n_head,
               int D, int ld_score, void* stream);   ///< TB tokens: params[t*8] = n_pool of token t; iq/w rows per token; score rows of ld_score
/// The selection, on the device: params = {n_pool, n_top, n_kv, n_tail} (device ints); the n_top best pools in
/// ascending index order (ties: lower index) expanded to kpool cells, then the n_tail newest cells; *n_sel = the count.
void dsa_select(const float* score, int* params, int TB, int kpool, int32_t* sel, int ld_score, int n_sel_max, void* stream);   ///< per token: params[t*8+4] = n_sel
/// Attention over the selected cells: scores[h][c] = q_abs[h] . kvc[sel[c]] * scale, softmax over c, ctx[h] = sum p kvc.
/// n_sel is read from the device; scores is scratch of n_head * n_sel_max floats (row stride n_sel_max).
void dsa_attend(const uint8_t* kvc_cache, const uint8_t* means, float bias, const int32_t* sel, const int* params, int TB, int n_sel_max, const float* q_abs, float scale,
                float* scores, float* ctx, int n_head, int kv_lora, void* stream);   ///< TB tokens; scores [TB][n_head][n_sel_max], ctx rows per token
/// out[h][v] = sum_j wv_b[h][v][j] * ctx[h][j]   (wv_b bf16, GGUF [kv_lora, head_v, n_head])
void dsa_expand(const uint16_t* wv_b, const float* ctx, float* out, int n_head, int kv_lora, int head_v,
                void* stream);

/// MoE router: probs = sigmoid(logits); selection by probs + bias (top-k, ties to the lower id); weights = probs[ids],
/// / max(sum, 2^-14) when norm, * scale.  One block.
void router_sigmoid_topk(const float* logits, const float* bias, int n_expert, int k, bool norm, float scale,
                         int32_t* ids, float* weights, float* probs, void* stream);
/// h = silu(min(g, limit)) * clamp(u, -limit, limit); limit <= 0 means plain silu(g) * u
void swiglu_clamp(const float* g, const float* u, float* h, int n, float limit, void* stream);
/// The MoE doorbell.  `h_*` are DEVICE views of host-mapped pinned memory; `tok` is a device counter.
void moe_publish(const int32_t* ids, int k, const uint32_t* tok, int32_t* h_ids, uint32_t* h_published, const float* x, int nx, float* h_x, void* stream);   ///< ids (and nx floats of x) to the host, then the flag
/// The doorbell flag (device view of the mapped word) written by a kernel on `stream`, after the copies queued before it.
void write_flag(uint32_t* flag_dev, uint32_t v, void* stream);
/// n floats (n % 4 == 0) from mapped pinned host memory (its device pointer) into device memory, by a kernel
void copy_host_rows(const float* src_mapped, float* dst, size_t n, void* stream);
/// The T tokens' MoE inputs ([T][n] floats) to mapped host memory (before moe_publish on the same stream).
void publish_x(const float* x, int T, int n, float* h_x, void* stream);
/// The dispatch tables, hits first (see kernels.cu): tabA/rowsA the hits' gate+up, tabB/rowsB the misses', tabC the downs,
/// wperm the weights in that order, cnt = {2 n_hit, 2 n_miss}.
void moe_select(const int32_t* ids, const float* wts, int k, const int32_t* slot_of, const uint8_t* slots, uint64_t blob,
                uint64_t gate_bytes, uint64_t up_bytes, const uint8_t* stage, uint64_t stage_bytes, const void** tabA,
                const void** tabB, const void** tabC, int* rowsA, int* rowsB, float* wperm, int* cnt, void* stream);
void moe_wait(const uint32_t* h_flag, const uint32_t* tok, void* stream);
void u32_inc(uint32_t* v, void* stream);
/// The lookahead prefetch's placement (device views of mapped records): wait for pf_flag == *tok, apply n swaps.
void pf_apply(const uint32_t* pf_flag, const uint32_t* tok, const int* pf_n, const int* pf_e, const int* pf_slot, const int* pf_out,
              int32_t* slot_of, void* stream);
/// the adaptive swap's table update: slot_of[e_out] = -1 (e_out >= 0), slot_of[e_in] = slot
void swap_slot(int32_t* slot_of, int e_out, int e_in, int32_t slot, void* stream);
/// out[i] = sum_j w[j] * y[j][i]  (the weighted combine of k expert outputs)
void weighted_sum(const float* y, const float* w, int k, int n, float* out, void* stream);
/// y += a * x, a read from device memory (weights[idx])
void axpy_dev(float* y, const float* x, const float* a, int idx, int n, void* stream);
void add_inplace(float* y, const float* x, int n, void* stream);
void scale_inplace(float* x, int n, float s, void* stream);
/// h[i] = mean_s R[s][i]
void mean_streams(const float* R, float* h, int n_embd, int hc, void* stream);
/// out[0] = argmax(x[0..n)), ties to the lower index
void argmax(const float* x, int n, int32_t* out, int* scratch_idx, float* scratch_val, void* stream);
/// Q8_0 row r of a [n, rows] GGUF tensor to f32 (the token embedding)
void dequant_q8_0_row(const uint8_t* w, int row, float* out, int n, void* stream);

// ---- multi-token (verify window) twins: T tokens, token axis outermost; see kernels.cu
int window_max_tokens();
void f32_gemv_t(const float* W, const float* x, float* y, int n_in, int n_out, int T, void* stream);
void layer_norm_t(const float* x, const float* w, const float* b, float* y, int rows, int n, float eps, void* stream);
void mhc_mix_t(const float* R, const float* fn, const float* scale, const float* base, float* mixes, float* part, float* rs,
               float* pre, float* post, float* comb, int T, int n_embd, int hc, int iters, float eps_rms, float eps_hc, void* stream);
void mhc_pre_apply_t(const float* R, const float* pre, float* x, int T, int n_embd, int hc, void* stream);
void mhc_post_t(const float* y, const float* R_in, const float* post, const float* comb, float* R_out, int T, int n_embd, int hc,
                void* stream);
/// snap (may be null): the conv window after each token, [T][(d_conv-1)][d_inner], for the rollback
void kda_conv_silu_t(const float* x, float* conv_state, const float* w, float* out, float* snap, int d_inner, int d_conv, int T,
                     void* stream);
void kda_gate_t(const float* g_in, const float* dt_b, const float* A, float lower, float* g, int T, int n_head, int S, void* stream);
/// snap (may be null): the state after each token, [T][H][S][S]
void kda_step_t(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta, float* o,
                float* snap, int T, int n_head, int S, void* stream);
void dsa_absorb_t(const uint16_t* wk_b, const float* q, float* q_abs, int T, int n_head, int kv_lora, int head_k, void* stream);
void dsa_expand_t(const uint16_t* wv_b, const float* ctx, float* out, int T, int n_head, int kv_lora, int head_v, void* stream);
void router_sigmoid_topk_t(const float* logits, const float* bias, int T, int n_expert, int k, bool norm, float scale, int32_t* ids,
                           float* weights, float* probs, void* stream);
/// The window's MoE dispatch (see kernels.cu k_moe_group): T*K pairs grouped by expert, resident groups first.
/// cnt = {n_hit_groups, n_groups, n_pairs, n_missed_experts, 0}
/// slot table updates for the loan slots: entries (e, j) -> slot_of[e] = kLoanBit | j, (e, -1) -> -1
void loan_apply(const int32_t* entries, int n, int32_t* slot_of, void* stream);
void moe_group(const int32_t* ids, const float* wts, int T, int K, int n_expert, const int32_t* slot_of, const uint8_t* slots, uint64_t blob,
               uint64_t gate_bytes, uint64_t up_bytes, const uint8_t* stage, uint64_t stage_bytes, const uint8_t* const* loan_ptr, const void** grp_w,
               const void** grp_wu, const void** grp_wd, int* grp_col0, int* grp_ncols, int* grp_orow, int* pair_tok,
               float* pair_w, int* cnt, int* grp_e, int* grp_miss, bool split_hits, int* bounds, int* tok_pairs, int n_slots, int* bounds_res,
               int* bounds_miss, void* stream);   ///< !split_hits: residents in slot order (bounds_res[n_slots+1]), then the non-resident by rank (bounds_miss)
/// out[t] = sum over the token's K pairs (tok_pairs[t*K+k] = pair row) of pair_w * y[row]
void combine_tok(const float* y, const float* pair_w, const int* tok_pairs, int K, float* out, int T, int n, void* stream);
/// The prompt windows' dense DSA attention: the latent cache rows (and their transpose) as Q8_0 blocks, the selection
/// masks from the per-token cell lists, and the masked softmax of a [rows][n_pad] score block in place.
void set_pair(int32_t* p, int a, int b, void* stream);   ///< p[0] = a, p[1] = b on the stream (a bounds table without a host-side copy)
void q8_0_rows_T(const uint8_t* x, int cols, uint8_t* out, int n_cells, int n_pad, void* stream);   ///< the transpose of a Q8_0 cache, re-quantized along the cells
// LoD attention (indexer-free): see docs/GLM5_NEXT.md
void lod_page_mean(const uint8_t* kvc, uint8_t* means, int p0, int n_pages, int ps, int n, void* stream);
void lod_page_mean_rows(uint8_t* const* kvc_tab, uint8_t* const* means_tab, const int* row_slot, const int* row_pos, int T, int ps, int n, void* stream);
void lod_score_rows(uint8_t* const* means_tab, const int* row_slot, const float* q_abs, float* score, const int* params, int n_cand_max, int T, int n_head, int n,
                    int ld_score, void* stream);
void colmax(const float* S_, int ld, int rows, int n_cols, float* score, bool first, void* stream);
void head_group_mean(const float* q, float* qg, int T, int H, int G, int n, void* stream);
void page_max(const float* tok_score, float* page_score, int ps, int n_pages, void* stream);
void rows_q8_pad(const float* x, uint8_t* out, int n, int D, int DP, void* stream);
void rows_q8_pad_rows(float* const* pooled_tab, uint8_t* const* q8_tab, const int* row_slot, const int* row_pos, int T, int kpool, int D, int DP, void* stream);
void pad_rows(const float* x, float* out, int n, int D, int DP, void* stream);
void idx_pool_score(const float* S_, int ld, int TB, int IH, const float* w, int n_pool, float* pool_score, bool first, void* stream);
void lod_select_pools(const float* pool_score, int* params, int T, int ps, int KP, int32_t* sel, int ld_pool, int n_sel_max, void* stream);
void lod_gather_dev(const uint8_t* kvc, const uint8_t* means, const int32_t* sel, const int* params, int n_sel_host, int pos0, int T, uint8_t* out, float* bias_col,
                    float bias, int n, void* stream);
void pool_page_max_rows(const float* pool_score, int ld_pool, float* page_score, int ld_page, const int* params, int n_cand_max, int T, int ppp, void* stream);
void lod_score_tok_rows(uint8_t* const* kvc_tab, const int* row_slot, const float* qg, float* score, const int* params, int n_cand_max, int T, int G, int n,
                        int ld_score, int ps, void* stream);
void lod_select(const float* score, int* params, int T, int ps, int32_t* sel, int ld_score, int n_sel_max, void* stream);
void lod_gather(const uint8_t* kvc, const uint8_t* means, const int32_t* sel, int n_sel, int pos0, int T, uint8_t* out, float* bias_col, float bias, int n, void* stream);
void lod_mask(uint32_t* mask, int words, int TB, int n_sel, int T, int t0, void* stream);
/// the fused LoD read: part = [T*n_head*nsplit][2 + kv_lora] scratch; kv_lora in {256, 512, 1024}; nsplit <= 128
void lod_attn_fused(uint8_t* const* kvc_tab, uint8_t* const* means_tab, float bias, const int* row_slot, const int32_t* sel, const int* params, int T,
                    int n_sel_max, const float* q_abs, float scale, float* part, float* ctx, int n_head, int kv_lora, int nsplit, void* stream);
void sel_mask(const int32_t* sel, const int* params, int TB, int n_sel_max, uint32_t* mask, int words, void* stream);
void masked_softmax(float* S, int ld, const uint32_t* mask, int words, int rows, int H, int n_pad, float scale, const float* bias, void* stream);   ///< bias: per column (nullable)
/// Batch decode windows (row t: slot row_slot[t] at position row_pos[t]; tab[slot] = that slot's state base for the layer).
void kda_conv_rows(const float* x, float* const* tab, const int* row_slot, int which, const float* w, float* out, int d_inner, int d_conv, int T, void* stream);
/// Runs of consecutive rows per slot (groups g: rows [grp_start[g], +grp_len[g]) of slot grp_slot[g]; row_k[t] = the row's index in its run):
/// the conv and the recurrence in order within a run, with per-row snapshots for the rollback of rejected drafts.
void kda_conv_grp(const float* x, float* const* tab, const int* row_slot, const int* row_k, int which, const float* w, float* out, float* snap,
                  int d_inner, int d_conv, int T, const int* grp_slot, const int* grp_start, const int* grp_len, int n_grp, void* stream);
void kda_step_grp(float* const* tab, const int* grp_slot, const int* grp_start, const int* grp_len, int n_grp, const float* q, const float* k,
                  const float* v, const float* g, const float* beta, float* o, float* snap, int n_head, int S, void* stream);
void kda_step_rows(float* const* tab, const int* row_slot, const float* q, const float* k, const float* v, const float* g, const float* beta,
                   float* o, int T, int n_head, int S, void* stream);
void scatter_q8_rows(const float* src, uint8_t* const* tab, const int* row_slot, const int* row_pos, int T, int n, void* stream);
void store_q8_rows(const float* src, uint8_t* dst, int T, int n, void* stream);   ///< f32 rows into a Q8_0 cache (n % 32 == 0)
void dsa_pool_rows(uint8_t* const* ik_tab, uint8_t* const* ig_tab, float* const* pooled_tab, const int* row_slot, const int* row_pos, const float* ape,
                   int T, int kpool, int D, void* stream);
void dsa_score_rows(float* const* pooled_tab, const int* row_slot, const float* iq, const float* w, float* score, const int* params, int n_pool_max,
                    int T, int n_head, int D, int ld_score, void* stream);
void dsa_attend_rows(uint8_t* const* kvc_tab, uint8_t* const* means_tab, float bias, const int* row_slot, const int32_t* sel, const int* params, int T,
                     int n_sel_max, const float* q_abs,
                     float scale, float* scores, float* ctx, int n_head, int kv_lora, void* stream);
/// The per-head MMQ tables for T tokens x H heads: ids[h*T+t] = t*H+h, bounds[h] = h*T.
void head_tables(int32_t* ids, int32_t* bounds, int T, int H, void* stream);
/// The MMQ path: the resident groups' expert blobs into the window buffer at dst + e * stride (grp_miss[g] < 0 only).
void moe_gather_experts(const void* const* grp_w, const int* grp_e, const int* grp_miss, const int* n_groups, int n_expert, uint64_t blob,
                        uint8_t* dst, uint64_t stride, void* stream);
/// The hybrid split: the CPU-served misses' groups (miss_cpu[m] != 0) get ncols = 0 (nothing to compute on the device).
void moe_cpu_mask(const int* cnt, const int* grp_miss, const int* miss_cpu, int* grp_ncols, void* stream);
/// h[r] = swiglu_clamp(gu[r][0..n), gu[r][n..2n)) for the first *n_rows rows
void swiglu_clamp_gu(const float* gu, float* h, const int* n_rows, int max_rows, int n, float limit, void* stream);
/// After the misses were served: miss m lives in slot miss_slot[m] (< 0: staging slot m); repoint the miss groups and
/// update the device slot table (miss_out[m] is the evicted expert or -1).  Runs after moe_wait / join.
void moe_place(const int* cnt, const int* grp_miss, const int* grp_e, const int* miss_slot, const int* miss_out, const uint8_t* slots,
               uint64_t blob, uint64_t gate_bytes, uint64_t up_bytes, const void** grp_w, const void** grp_wu, const void** grp_wd,
               int32_t* slot_of, void* stream);
void gather_q8(const uint8_t* xq, const int* pair_tok, const int* n_pairs, uint8_t* xg, int col_bytes, int max_pairs, void* stream);
void combine_pairs(const float* y, const float* pair_w, const int* pair_tok, const int* n_pairs, float* out, int T, int n, void* stream);
void swiglu_clamp_rows(const float* g, const float* u, float* h, const int* n_rows, int max_rows, int n, float limit, void* stream);
void argmax_rows(const float* x, int rows, int n, int32_t* out, int* scratch_idx, float* scratch_val, void* stream);
/// rows[t] of a Q8_0 [n, V] tensor to f32 [T][n]; zero_first: row 0 of the output is zero (the MTP's position-0 rule)
void dequant_q8_0_rows(const uint8_t* w, const int32_t* rows, int T, float* out, int n, bool zero_first, void* stream);
void concat_rows(const float* a, const float* b, float* out, int T, int n, void* stream);
void add_rows(float* y, const float* x, int n_tot, void* stream);

}  // namespace strata::glm5
