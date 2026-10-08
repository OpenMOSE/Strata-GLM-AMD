// src/kernels/native_q2k_parity.cpp - the Q2_K native MMVQ kernel (GLM-5.3-Flash's expert format, 36 of 42 layers)
// against ggml: ggml quantizes random rows to Q2_K, (a) ggml's `to_float` + a float dot is the reference, (b) ggml-cpu's
// own `vec_dot` (Q8_K activations) is the CPU contract, (c) `native_q2_k_f32` (Q8_1 activations) is the GPU path.
// (b) and (c) each differ from (a) by their activation rounding only; ncols 2..8 must be BITWISE equal to ncols 1.
//
//     build/native_q2k_parity [--selftest]
#include "strata/kernels/native_mmvq.hpp"
#include "ggml-cpu.h"
#include "ggml.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {
double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

int check(int n_in, int n_out, int seed) {
    int failures = 0;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> W((size_t) n_in * n_out);
    for (auto& w : W) w = nd(rng);
    const auto* tr = ggml_get_type_traits(GGML_TYPE_Q2_K);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q2_K, n_in);
    std::vector<uint8_t> q(row_bytes * n_out);
    ggml_quantize_chunk(GGML_TYPE_Q2_K, W.data(), q.data(), 0, n_out, n_in, nullptr);
    // (a) the float reference from the dequantized rows
    std::vector<float> Wq((size_t) n_in * n_out);
    for (int r = 0; r < n_out; ++r) tr->to_float(q.data() + (size_t) r * row_bytes, Wq.data() + (size_t) r * n_in, n_in);
    constexpr int NC = 8;
    std::vector<float> X((size_t) NC * n_in);
    for (auto& x : X) x = nd(rng);
    std::vector<float> ref((size_t) NC * n_out), cpu((size_t) NC * n_out);
    for (int c = 0; c < NC; ++c)
        for (int r = 0; r < n_out; ++r) {
            double s = 0;
            for (int i = 0; i < n_in; ++i) s += (double) Wq[(size_t) r * n_in + i] * X[(size_t) c * n_in + i];
            ref[(size_t) c * n_out + r] = (float) s;
        }
    // (b) ggml-cpu's vec_dot with its own activation format
    const auto* tc = ggml_get_type_traits_cpu(GGML_TYPE_Q2_K);
    const ggml_type at = tc->vec_dot_type;
    const size_t act_bytes = ggml_row_size(at, n_in);
    std::vector<uint8_t> act(act_bytes);
    for (int c = 0; c < NC; ++c) {
        ggml_get_type_traits_cpu(at)->from_float(X.data() + (size_t) c * n_in, act.data(), n_in);
        for (int r = 0; r < n_out; ++r)
            tc->vec_dot(n_in, &cpu[(size_t) c * n_out + r], 0, q.data() + (size_t) r * row_bytes, 0, act.data(), 0, 1);
    }
    // (c) the GPU path, ncols 1 and 2..8
    void *d_w, *d_x, *d_y, *d_s;
    cudaMalloc(&d_w, q.size());
    cudaMalloc(&d_x, X.size() * 4);
    cudaMalloc(&d_y, (size_t) NC * n_out * 4);
    cudaMalloc(&d_s, strata::kernels::native_q8_1_bytes(n_in, NC));
    cudaMemcpy(d_w, q.data(), q.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, X.data(), X.size() * 4, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);
    std::vector<float> gpu1((size_t) NC * n_out), gpun((size_t) NC * n_out);
    for (int c = 0; c < NC; ++c) {
        strata::kernels::native_q2_k_f32(d_w, (const float*) d_x + (size_t) c * n_in, d_s, (float*) d_y, n_in, n_out, 1, s);
        cudaStreamSynchronize(s);
        cudaMemcpy(gpu1.data() + (size_t) c * n_out, d_y, (size_t) n_out * 4, cudaMemcpyDeviceToHost);
    }
    const double r_cpu = rel(cpu, ref), r_gpu = rel(gpu1, ref);
    std::printf("Q2_K %dx%d: cpu vec_dot vs float %.3e, gpu ncols=1 vs float %.3e, gpu vs cpu %.3e\n", n_in, n_out,
                r_cpu, r_gpu, rel(gpu1, cpu));
    if (r_gpu > 2e-2 || !(r_gpu < 2 * r_cpu + 5e-3)) { std::printf("  FAIL: GPU error out of the activation-rounding band\n"); ++failures; }
    for (int nc = 2; nc <= NC; ++nc) {
        strata::kernels::native_q2_k_f32(d_w, (const float*) d_x, d_s, (float*) d_y, n_in, n_out, nc, s);
        cudaStreamSynchronize(s);
        cudaMemcpy(gpun.data(), d_y, (size_t) nc * n_out * 4, cudaMemcpyDeviceToHost);
        size_t diff = 0;
        for (size_t i = 0; i < (size_t) nc * n_out; ++i) diff += gpun[i] != gpu1[i];
        if (diff) { std::printf("  FAIL: ncols=%d differs from ncols=1 in %zu values\n", nc, diff); ++failures; }
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("  FAIL: CUDA error %s\n", cudaGetErrorString(e)); ++failures; }
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_y); cudaFree(d_s);
    cudaStreamDestroy(s);
    return failures;
}
}  // namespace

int main() {
    ggml_cpu_init();
    int failures = 0;
    failures += check(4096, 48, 1);   // gate / up rows (4096 in)
    failures += check(2048, 48, 2);   // down rows (2048 in)
    failures += check(256, 9, 3);     // one block per row, the small-K kernel
    std::printf(failures ? "native_q2k_parity: %d FAILURE(S)\n" : "native_q2k_parity: ok\n", failures);
    return failures ? 1 : 0;
}
