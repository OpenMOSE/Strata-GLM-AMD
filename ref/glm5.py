"""ref/glm5.py - a plain numpy reference of GLM-5-Next (GLM-5.3-Flash), transcribed from llama.cpp's
`src/models/glm5-next.cpp` (commit 0faee5004) and the ggml CPU ops it calls.

This is the specification Strata's GLM kernels are written against (docs/GLM5_NEXT.md, P0).  It is slow and
it is meant to be: every line maps to one ggml op, in the same order, with the same epsilons, so that a
disagreement with llama.cpp's own dump (tools/glm_oracle/dump_tensors) points at a misunderstanding of the
model rather than at an optimisation.

    python ref/glm5.py --gguf MODEL.gguf --tokens 1,2,3 [--layers N] [--greedy K]
                       [--compare DUMPDIR]            # against dump_tensors' s<step>_<name>.npy files

Stateful: `Model.forward(tokens)` appends to the recurrent states and the attention caches, so a prompt
followed by one token at a time reproduces llama.cpp's decode exactly as Strata will run it.  The whole chunk
goes through each layer at once (the way llama.cpp processes a prompt ubatch) so a layer's weights are
dequantized once per call.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from load import Gguf  # noqa: E402

F = np.float32


# ----------------------------------------------------------------------------- small ggml ops
def rms_norm(x, w=None, eps=1e-5):
    """ggml_rms_norm: x * 1/sqrt(mean(x^2) + eps), optionally * w (ggml_mul)."""
    x = x.astype(F)
    s = np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=np.float64) + eps).astype(F)
    y = x / s
    return y * w if w is not None else y


def layer_norm(x, w, b, eps):
    """ggml_norm (LLM_NORM): (x - mean) / sqrt(var + eps) * w + b."""
    x = x.astype(F)
    m = np.mean(x, axis=-1, keepdims=True, dtype=np.float64)
    v = np.mean((x - m) ** 2, axis=-1, keepdims=True, dtype=np.float64)
    return (((x - m) / np.sqrt(v + eps)).astype(F)) * w + b


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(F)


def silu(x):
    return x * sigmoid(x)


def softmax(x, axis=-1):
    x = x.astype(np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(F)


def l2_norm(x, eps=1e-6):
    """ggml_l2_norm: x * 1/max(||x||, eps), over the last axis."""
    n = np.sqrt(np.sum(x.astype(np.float64) ** 2, axis=-1, keepdims=True))
    return (x / np.maximum(n, eps)).astype(F)


def swiglu_clamp(gate, up, limit):
    """ggml_swiglu_clamp: silu(min(gate, limit)) * clamp(up, -limit, limit)."""
    g = np.minimum(gate, limit)
    u = np.clip(up, -limit, limit)
    return silu(g) * u


# ----------------------------------------------------------------------------- the model
class HParams:
    def __init__(self, kv: dict):
        a = "glm5-next"
        g = lambda k, d=None: kv.get(f"{a}.{k}", d)  # noqa: E731
        self.n_layer = int(g("block_count"))
        self.n_embd = int(g("embedding_length"))
        self.n_ff = int(g("feed_forward_length"))
        self.n_head = int(g("attention.head_count"))
        self.head_count_kv = list(g("attention.head_count_kv"))
        self.rms_eps = float(g("attention.layer_norm_rms_epsilon"))
        self.norm_eps = float(g("attention.layer_norm_epsilon", 1e-5))
        self.n_expert = int(g("expert_count"))
        self.n_expert_used = int(g("expert_used_count"))
        self.n_ff_exp = int(g("expert_feed_forward_length"))
        self.n_expert_shared = int(g("expert_shared_count"))
        self.n_dense_lead = int(g("leading_dense_block_count", 0))
        self.expert_scale = float(g("expert_weights_scale", 1.0))
        self.expert_norm = bool(g("expert_weights_norm", False))
        self.gating = int(g("expert_gating_func", 2))  # 2 = sigmoid
        clamp = g("swiglu_clamp_exp")
        self.clamp_exp = [float(c) for c in clamp] if clamp is not None else [0.0] * self.n_layer
        clamp = g("swiglu_clamp_shexp")
        self.clamp_shexp = [float(c) for c in clamp] if clamp is not None else list(self.clamp_exp)
        self.d_conv = int(g("ssm.conv_kernel"))
        self.kda_head_dim = int(g("kda.head_dim"))
        self.kda_lower = float(g("kda.gate_lower_bound", -np.inf))
        self.q_lora = int(g("attention.q_lora_rank"))
        self.kv_lora = int(g("attention.kv_lora_rank"))
        self.head_k_mla = int(g("attention.key_length_mla"))
        self.head_v_mla = int(g("attention.value_length_mla"))
        self.n_rot = int(g("rope.dimension_count", 0))
        self.idx_heads = int(g("attention.indexer.head_count"))
        self.idx_dim = int(g("attention.indexer.key_length"))
        self.idx_topk = int(g("attention.indexer.top_k"))
        self.kpool = int(g("attention.indexer.kpool"))
        self.kpool_tail = bool(g("attention.indexer.kpool_select_tail", False))
        self.hc = int(g("hyper_connection.count"))
        self.hc_iters = int(g("hyper_connection.sinkhorn_iterations"))
        self.hc_eps = float(g("hyper_connection.epsilon"))
        assert self.hc == 4 and self.n_rot == 0, "this reference follows llama.cpp: hc == 4, nope-only MLA"

    def is_recurrent(self, il):
        return self.head_count_kv[il] == 0


class Weights:
    """Dequantized tensors with a byte-bounded LRU, because a layer's dense weights are ~0.6 GB in f32."""

    def __init__(self, gg: Gguf, cache_gb: float):
        self.gg = gg
        self.cap = int(cache_gb * (1 << 30))
        self.cache: dict[str, np.ndarray] = {}
        self.used = 0

    def _put(self, key, arr):
        self.used += arr.nbytes
        self.cache[key] = arr
        while self.used > self.cap and len(self.cache) > 1:
            k = next(iter(self.cache))
            self.used -= self.cache.pop(k).nbytes
        return arr

    def __call__(self, name):
        if name in self.cache:
            a = self.cache.pop(name)
            self.cache[name] = a
            return a
        return self._put(name, np.ascontiguousarray(self.gg.tensor(name), dtype=F))

    def has(self, name):
        return name in self.gg._by_name

    def expert(self, name, e):
        key = f"{name}#{e}"
        if key in self.cache:
            a = self.cache.pop(key)
            self.cache[key] = a
            return a
        return self._put(key, np.ascontiguousarray(self.gg.expert(name, e), dtype=F))


class Model:
    def __init__(self, gguf_path, layers=None, cache_gb=12.0, record=True):
        self.gg = Gguf(gguf_path)
        kv = self.gg.g.kv if hasattr(self.gg.g, "kv") else self.gg.g.metadata
        self.hp = HParams(kv)
        self.w = Weights(self.gg, cache_gb)
        self.n_layers = self.hp.n_layer if layers is None else min(layers, self.hp.n_layer)
        self.record = record
        self.rec: dict[str, np.ndarray] = {}
        hp = self.hp
        # per-layer state
        self.pos = 0
        d_inner = hp.n_head * hp.kda_head_dim
        self.conv_state = {}   # il -> (3, d_conv-1, d_inner)   q|k|v windows
        self.ssm_state = {}    # il -> (n_head, S, S)  S[i, j] as ggml: state[i][j], i = key dim, j = value dim
        self.kvc = {}          # il -> list of latent rows (kv_lora,)
        self.idx_k = {}        # il -> list of indexer keys (idx_dim,)
        self.idx_g = {}        # il -> list of indexer gates (idx_dim,)
        self.pooled = {}       # il -> list of pooled keys (idx_dim,), one per complete pool
        for il in range(self.n_layers):
            if hp.is_recurrent(il):
                self.conv_state[il] = np.zeros((3, hp.d_conv - 1, d_inner), F)
                self.ssm_state[il] = np.zeros((hp.n_head, hp.kda_head_dim, hp.kda_head_dim), F)
            else:
                self.kvc[il], self.idx_k[il], self.idx_g[il], self.pooled[il] = [], [], [], []

    # -- recording, in dump_tensors' shape convention (ggml ne reversed)
    def cb(self, name, il, arr):
        if self.record:
            self.rec[f"{name}-{il}" if il >= 0 else name] = np.asarray(arr, dtype=arr.dtype)

    def t(self, name, il=None):
        return self.w(f"blk.{il}.{name}" if il is not None else name)

    # -- mHC
    def hc_pre(self, R, prefix, il):
        """R (T, hc, n_embd) -> x (T, n_embd), post (T, hc), comb (T, src, dst)."""
        hp = self.hp
        T = R.shape[0]
        fn = self.t(f"{prefix}_fn.weight", il)        # (24, hc*n_embd)
        scale = self.t(f"{prefix}_scale.weight", il)  # (3,)
        base = self.t(f"{prefix}_base.weight", il)    # (24,)
        hc = hp.hc
        flat = R.reshape(T, hc * hp.n_embd)
        flat_n = rms_norm(flat, None, hp.rms_eps)
        mixes = flat_n @ fn.T                          # (T, 24)
        self.cb("hc_mixes", il, mixes)
        pre = sigmoid(mixes[:, 0:hc] * scale[0] + base[0:hc]) + hp.hc_eps
        post = sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc]) * 2.0
        comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).reshape(T, hc, hc)   # [t, src, dst] (dst fastest)
        # Sinkhorn: softmax over dst, + eps, normalize over src, then (over dst, over src) x (iters - 1)
        comb = softmax(comb, axis=2) + hp.hc_eps
        eps = hp.hc_eps

        def norm_cols(c):   # for each dst: divide by the sum over src (+eps)
            return c / (np.sum(c, axis=1, keepdims=True) + eps)

        def norm_rows(c):   # for each src: divide by the sum over dst (+eps)
            return c / (np.sum(c, axis=2, keepdims=True) + eps)

        comb = norm_cols(comb)
        for _ in range(1, hp.hc_iters):
            comb = norm_cols(norm_rows(comb))
        x = np.einsum("thi,th->ti", R, pre).astype(F)
        return x, post.astype(F), comb.astype(F)

    @staticmethod
    def hc_post(y, R, post, comb):
        """result[t, dst, i] = y[t, i] * post[t, dst] + sum_src R[t, src, i] * comb[t, src, dst]."""
        return (y[:, None, :] * post[:, :, None] + np.einsum("tsi,tsd->tdi", R, comb)).astype(F)

    # -- KDA
    def conv1d(self, il, which, x_proj):
        hp = self.hp
        T = x_proj.shape[0]
        w = self.t(f"ssm_conv1d_{'qkv'[which]}.weight", il).reshape(-1, hp.d_conv)   # (d_inner, d_conv)
        st = self.conv_state[il][which]                                             # (d_conv-1, d_inner)
        pad = np.concatenate([st, x_proj], axis=0)                                  # (d_conv-1+T, d_inner)
        y = np.zeros_like(x_proj)
        for k in range(hp.d_conv):
            y += pad[k:k + T] * w[:, k]
        self.conv_state[il][which] = pad[T:]
        return silu(y)

    def kda_layer(self, x, il):
        hp = self.hp
        T = x.shape[0]
        H, S = hp.n_head, hp.kda_head_dim
        q = self.conv1d(il, 0, x @ self.t("attn_q.weight", il).T).reshape(T, H, S)
        k = self.conv1d(il, 1, x @ self.t("attn_k.weight", il).T).reshape(T, H, S)
        v = self.conv1d(il, 2, x @ self.t("attn_v.weight", il).T).reshape(T, H, S)
        self.cb("kda_q_conv", il, q); self.cb("kda_k_conv", il, k); self.cb("kda_v_conv", il, v)

        g1 = (x @ self.t("ssm_f_a.weight", il).T) @ self.t("ssm_f_b.weight", il).T + self.t("ssm_dt.bias", il)
        A = self.t("ssm_a", il)                                                     # (H,) = -exp(A_log)
        g1 = g1.reshape(T, H, S)
        if hp.kda_lower > -np.inf:
            g1 = hp.kda_lower * sigmoid(-(g1 * A[None, :, None]))
        else:
            g1 = np.log1p(np.exp(g1.astype(np.float64))).astype(F) * A[None, :, None]
        self.cb("kda_g1", il, g1)
        beta = sigmoid(x @ self.t("ssm_beta.weight", il).T)                        # (T, H)
        self.cb("kda_beta", il, beta)

        q = l2_norm(q); k = l2_norm(k)
        state = self.ssm_state[il]                                                  # (H, S_i, S_j)
        o = np.zeros((T, H, S), F)
        scale = F(1.0 / np.sqrt(S))
        for t in range(T):
            dec = np.exp(g1[t])                                                      # (H, S_i)
            state *= dec[:, :, None]                                                 # S[i, :] *= exp(g[i])
            kt = k[t]                                                                # (H, S_i)
            sk = np.einsum("hij,hi->hj", state, kt)                                  # sum_i S[i,j] k[i]
            d = (v[t] - sk) * beta[t][:, None]                                       # (H, S_j)
            state += kt[:, :, None] * d[:, None, :]                                  # S[i,j] += k[i] d[j]
            o[t] = np.einsum("hij,hi->hj", state, q[t]) * scale
        self.ssm_state[il] = state
        self.cb("kda_scan_out", il, o)

        g2 = ((x @ self.t("ssm_g_a.weight", il).T) @ self.t("ssm_g_b.weight", il).T).reshape(T, H, S)
        self.cb("kda_g2", il, g2)
        normed = rms_norm(o, self.t("ssm_norm.weight", il), hp.rms_eps)
        self.cb("kda_normed", il, normed)
        y = (normed * sigmoid(g2)).reshape(T, H * S)
        out = y @ self.t("attn_output.weight", il).T
        self.cb("kda_out", il, out)
        return out

    # -- DSA
    def dsa_layer(self, x, il):
        hp = self.hp
        T = x.shape[0]
        H = hp.n_head
        qr = rms_norm(x @ self.t("attn_q_a.weight", il).T, self.t("attn_q_a_norm.weight", il), hp.rms_eps)
        self.cb("q_resid", il, qr)
        q = (qr @ self.t("attn_q_b.weight", il).T).reshape(T, H, hp.head_k_mla)
        kvc = rms_norm(x @ self.t("attn_kv_a_mqa.weight", il).T, self.t("attn_kv_a_norm.weight", il), hp.rms_eps)
        self.cb("kv_cmpr", il, kvc[:, None, :])
        wk_b = self.t("attn_k_b.weight", il)                                        # (H, kv_lora, head_k)
        wv_b = self.t("attn_v_b.weight", il)                                        # (H, head_v, kv_lora)
        q_abs = np.einsum("hji,thi->thj", wk_b, q).astype(F)                        # (T, H, kv_lora)
        self.cb("q_absorbed", il, q_abs)

        # indexer
        iq = (qr @ self.t("indexer.attn_q_b.weight", il).T).reshape(T, hp.idx_heads, hp.idx_dim)
        self.cb("indexer_q", il, iq)
        ik = layer_norm(x @ self.t("indexer.attn_k.weight", il).T, self.t("indexer.k_norm.weight", il),
                        self.t("indexer.k_norm.bias", il), hp.norm_eps)
        self.cb("indexer_k", il, ik)
        ig = x @ self.t("indexer_compressor_gate.weight", il).T
        self.cb("indexer_gate", il, ig)
        wgt = (x @ self.t("indexer.proj.weight", il).T) * F(1.0 / np.sqrt(hp.idx_dim * hp.idx_heads))
        self.cb("indexer_weights", il, wgt)
        ape = self.t("indexer_compressor_ape.weight", il)                           # (kpool, idx_dim)

        cache_k, cache_g, cache_c, pooled = self.idx_k[il], self.idx_g[il], self.kvc[il], self.pooled[il]
        kp = hp.kpool
        out = np.zeros((T, H * hp.head_v_mla), F)
        scores_rec, sel_rec = [], []
        for t in range(T):
            cache_k.append(ik[t]); cache_g.append(ig[t]); cache_c.append(kvc[t])
            n_kv = len(cache_c)
            # a pool completes when its last member arrives
            if n_kv % kp == 0:
                j0 = n_kv - kp
                pk = np.stack(cache_k[j0:n_kv])                                      # (kpool, D)
                pg = np.stack(cache_g[j0:n_kv]) + ape                                # (kpool, D)
                probs = softmax(pg, axis=0)
                pooled.append(np.sum(probs * pk, axis=0).astype(F))
            nv = n_kv // kp                                                          # pools visible to this token
            n_tail = n_kv % kp
            cells = []
            score = np.full((len(pooled),), -np.inf, F)
            if nv:
                P = np.stack(pooled[:nv])                                            # (nv, D)
                s = np.maximum(np.einsum("hd,pd->hp", iq[t], P), 0.0)                # relu, (heads, nv)
                score[:nv] = np.einsum("hp,h->p", s, wgt[t])
                n_top = min(nv, hp.idx_topk // kp)
                top = np.argsort(-score[:nv], kind="stable")[:n_top]
                for p in sorted(top):
                    cells.extend(range(p * kp, p * kp + kp))
            if hp.kpool_tail:
                cells.extend(range(n_kv - n_tail, n_kv))
            cells = np.array(sorted(cells), dtype=np.int64)
            scores_rec.append(score); sel_rec.append(cells)
            K = np.stack([cache_c[c] for c in cells])                                # (n_sel, kv_lora)
            kq = (q_abs[t] @ K.T) * F(1.0 / np.sqrt(hp.head_k_mla))                  # (H, n_sel)
            p = softmax(kq, axis=1)
            ctx = p @ K                                                              # (H, kv_lora)
            o = np.einsum("hvj,hj->hv", wv_b, ctx)                                   # (H, head_v)
            out[t] = o.reshape(-1)
        n_pool = max(len(s) for s in scores_rec)
        sc = np.full((T, n_pool), -np.inf, F)
        for t, s in enumerate(scores_rec):
            sc[t, :len(s)] = s
        self.cb("indexer_score", il, sc)
        self.rec_sel = sel_rec
        out = out @ self.t("attn_output.weight", il).T
        self.cb("attn_out", il, out)
        return out

    # -- FFN
    def dense_ffn(self, x, il):
        g = x @ self.t("ffn_gate.weight", il).T
        u = x @ self.t("ffn_up.weight", il).T
        h = swiglu_clamp(g, u, self.hp.clamp_shexp[il]) if self.hp.clamp_shexp[il] > 1e-6 else silu(g) * u
        out = h @ self.t("ffn_down.weight", il).T
        self.cb("ffn_out", il, out)
        return out

    def moe(self, x, il):
        hp = self.hp
        T = x.shape[0]
        logits = x @ self.t("ffn_gate_inp.weight", il).T                             # (T, n_expert)
        self.cb("ffn_moe_logits", il, logits)
        assert hp.gating == 2, "only the sigmoid router is transcribed"
        probs = sigmoid(logits)
        self.cb("ffn_moe_probs", il, probs)
        sel = probs + self.t("exp_probs_b.bias", il) if self.w.has(f"blk.{il}.exp_probs_b.bias") else probs
        k = hp.n_expert_used
        ids = np.argsort(-sel, axis=1, kind="stable")[:, :k]                         # (T, k)
        wts = np.take_along_axis(probs, ids, axis=1)
        if hp.expert_norm:
            wts = wts / np.maximum(np.sum(wts, axis=1, keepdims=True), F(6.103515625e-5))
        if hp.expert_scale not in (0.0, 1.0):
            wts = wts * F(hp.expert_scale)
        self.cb("ffn_moe_topk", il, ids.astype(np.int32))
        self.cb("ffn_moe_weights", il, wts[:, :, None])
        limit = hp.clamp_exp[il]
        out = np.zeros_like(x)
        for t in range(T):
            for j in range(k):
                e = int(ids[t, j])
                g = x[t] @ self.w.expert(f"blk.{il}.ffn_gate_exps.weight", e).T
                u = x[t] @ self.w.expert(f"blk.{il}.ffn_up_exps.weight", e).T
                h = swiglu_clamp(g, u, limit) if limit > 1e-6 else silu(g) * u
                out[t] += wts[t, j] * (h @ self.w.expert(f"blk.{il}.ffn_down_exps.weight", e).T)
        self.cb("ffn_moe_out", il, out)
        g = x @ self.t("ffn_gate_shexp.weight", il).T
        u = x @ self.t("ffn_up_shexp.weight", il).T
        lim = hp.clamp_shexp[il]
        h = swiglu_clamp(g, u, lim) if lim > 1e-6 else silu(g) * u
        sh = h @ self.t("ffn_down_shexp.weight", il).T
        self.cb("ffn_shexp", il, sh)
        out = out + sh
        self.cb("ffn_out", il, out)
        return out

    # -- the forward pass over one chunk of tokens
    def forward(self, tokens):
        hp = self.hp
        tokens = np.asarray(tokens, dtype=np.int64)
        T = len(tokens)
        emb = self.gg.rows("token_embd.weight", tokens).astype(F)                    # (T, n_embd)
        self.cb("inp_embd", -1, emb)
        R = np.repeat(emb[:, None, :], hp.hc, axis=1)                                # (T, hc, n_embd)
        self.cb("hc_init", -1, R)
        for il in range(self.n_layers):
            t0 = time.time()
            x, post, comb = self.hc_pre(R, "hc_attn", il)
            self.cb("hc_attn_pre", il, x)
            x = rms_norm(x, self.t("attn_norm.weight", il), hp.rms_eps)
            self.cb("attn_norm", il, x)
            y = self.kda_layer(x, il) if hp.is_recurrent(il) else self.dsa_layer(x, il)
            R = self.hc_post(y, R, post, comb)
            self.cb("hc_attn_post", il, R)
            x, post, comb = self.hc_pre(R, "hc_ffn", il)
            self.cb("hc_ffn_pre", il, x)
            x = rms_norm(x, self.t("ffn_norm.weight", il), hp.rms_eps)
            self.cb("ffn_norm", il, x)
            y = self.dense_ffn(x, il) if il < hp.n_dense_lead else self.moe(x, il)
            R = self.hc_post(y, R, post, comb)
            self.cb("l_out", il, R)
            print(f"  layer {il:2d} {'KDA' if hp.is_recurrent(il) else 'DSA'}{' dense' if il < hp.n_dense_lead else ''}"
                  f" {time.time() - t0:5.1f}s", file=sys.stderr)
        self.pos += T
        if self.n_layers < hp.n_layer:
            return None
        h = np.mean(R, axis=1, dtype=np.float64).astype(F)
        self.cb("hc_head", -1, h)
        h = rms_norm(h, self.t("output_norm.weight"), hp.rms_eps)
        self.cb("result_norm", -1, h)
        logits = h @ self.t("output.weight").T
        self.cb("result_output", -1, logits)
        return logits


# ----------------------------------------------------------------------------- comparison
def compare(rec: dict, dump_dir: pathlib.Path, step: int):
    rows = []
    for name, a in rec.items():
        f = dump_dir / f"s{step}_{name}.npy"
        if not f.exists():
            continue
        b = np.load(f)
        a2, b2 = np.squeeze(a), np.squeeze(b)
        note = ""
        if name.startswith("indexer_score"):           # the dump pads the pool axis
            a2 = a.reshape(a.shape[0], -1)
            b2 = b.reshape(-1, b.shape[-1])[:a2.shape[0], :a2.shape[1]]
            fin = np.isfinite(a2)
            a2, b2 = np.where(fin, a2, 0), np.where(fin, b2, 0)
            note = "visible pools only"
        if name.startswith("ffn_moe_weights"):         # ggml_top_k's order is unspecified: compare as sets
            a2 = -np.sort(-a2.reshape(a2.shape[0], -1), axis=1)
            b2 = -np.sort(-b2.reshape(b2.shape[0], -1), axis=1)
            note = "sorted"
        if a2.shape != b2.shape:
            rows.append((name, f"shape ref {a2.shape} vs dump {b2.shape}", "", "", ""))
            continue
        if a2.dtype.kind == "i":
            rows.append((name, "ids", f"{np.mean(a2 == b2) * 100:.1f}% equal", "", note))
            continue
        d = np.abs(a2.astype(np.float64) - b2.astype(np.float64))
        ref = np.abs(b2.astype(np.float64))
        cos = float(np.dot(a2.ravel(), b2.ravel()) / (np.linalg.norm(a2) * np.linalg.norm(b2) + 1e-30))
        rows.append((name, f"{d.max():.3e}", f"{d.max() / (ref.max() + 1e-30):.3e}", f"{cos:.6f}", note))
    w = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'tensor':{w}s}  {'max|diff|':>10s}  {'rel to max':>10s}  {'cosine':>9s}")
    for r in rows:
        print(f"{r[0]:{w}s}  {r[1]:>10s}  {r[2]:>10s}  {r[3]:>9s}  {r[4]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokens", help="comma-separated ids; default: DUMPDIR/tokens.txt when --compare is given")
    ap.add_argument("--layers", type=int, help="run only the first N layers (no logits)")
    ap.add_argument("--greedy", type=int, default=0, help="continue K tokens greedily, one per step")
    ap.add_argument("--compare", help="dump_tensors output directory")
    ap.add_argument("--cache-gb", type=float, default=12.0)
    ap.add_argument("--save", help="save the recorded tensors of step 0 to this .npz")
    a = ap.parse_args()
    dump = pathlib.Path(a.compare) if a.compare else None
    if a.tokens:
        toks = [int(x) for x in a.tokens.split(",") if x]
    elif dump and (dump / "tokens.txt").exists():
        toks = [int(x) for x in (dump / "tokens.txt").read_text().split()]
    else:
        sys.exit("--tokens or --compare with tokens.txt is required")
    m = Model(a.gguf, layers=a.layers, cache_gb=a.cache_gb)
    print(f"{m.hp.n_layer} layers, running {m.n_layers}; {len(toks)} prompt tokens", file=sys.stderr)
    cur = toks
    for step in range(a.greedy + 1):
        m.rec = {}
        logits = m.forward(cur)
        if logits is not None:
            best = int(np.argmax(logits[-1]))
            print(f"step {step}: argmax {best}", file=sys.stderr)
            if dump and (dump / "greedy.txt").exists():
                g = [int(x) for x in (dump / "greedy.txt").read_text().split()]
                if step < len(g):
                    print(f"         llama.cpp {g[step]} -> {'MATCH' if g[step] == best else 'DIFFERENT'}", file=sys.stderr)
            cur = [best]
        if dump:
            print(f"--- step {step} vs {dump}")
            compare(m.rec, dump, step)
            f = dump / f"s{step}_logits_last.npy"
            if logits is not None and f.exists():
                b = np.load(f)
                print(f"logits_last: max|diff| {np.abs(logits[-1] - b).max():.3e}, argmax ref {int(np.argmax(logits[-1]))} "
                      f"llama.cpp {int(np.argmax(b))}")
        if a.save and step == 0:
            np.savez(a.save, **m.rec)
        if logits is None:
            break


if __name__ == "__main__":
    main()
