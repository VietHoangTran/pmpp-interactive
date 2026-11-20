// Lab 20.4 driver (milestone M7): tests and benchmarks the KV-cache append and the
// single-token decode attention. You do not need to edit this file; your work goes in work/.
// The tests themselves are in ../shared-g12/decode_test.hpp (lab 20.5 runs them too); the
// CPU references are below.
#include "pmpp/harness.hpp"
#include "../shared-g12/decode_test.hpp"

using pmpp::Buf;
using m7::HEAD_DIM;

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeKvAppend(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> kNew,
                    id<MTLBuffer> vNew, id<MTLBuffer> Kc, id<MTLBuffer> Vc, id<MTLBuffer> pos,
                    uint32_t batch, uint32_t maxSeq, uint32_t hKv);
void encodeDecodeAttention(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> q,
                           id<MTLBuffer> Kc, id<MTLBuffer> Vc, id<MTLBuffer> len, id<MTLBuffer> out,
                           uint32_t batch, uint32_t hQ, uint32_t hKv, uint32_t maxSeq, float scale);
using pmpp::metal::buffer;
static id<MTLComputePipelineState> appendPso() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "kv_append"); }
static id<MTLComputePipelineState> decodePso() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "decode_attention"); }

// The kernel gives each lane HEAD_DIM / 32 dimensions and each threadgroup 8 SIMD-groups,
// so it needs SIMD-groups of 32 lanes (the book's WARP_SIZE) and 256 threads. Apple does
// not document the width as a number (notes/today/ch20-facts.md C3): ask the pipeline.
static std::string pipelineProblem() {
    if (!appendPso()) return "kv_append did not build: " + pmpp::take_error();
    id<MTLComputePipelineState> p = decodePso();
    if (!p) return "decode_attention did not build: " + pmpp::take_error();
    const NSUInteger logicalWarp = 32, block = 256;
    if (p.threadExecutionWidth != logicalWarp)
        return m7::fmt("decode_attention assumes 32-lane SIMD-groups; this pipeline's threadExecutionWidth is %.0f",
                       (double)p.threadExecutionWidth);
    if (p.maxTotalThreadsPerThreadgroup < block)
        return m7::fmt("decode_attention needs 256 threads per threadgroup; this pipeline allows %.0f",
                       (double)p.maxTotalThreadsPerThreadgroup);
    return "";
}
static pmpp::Timing runAppend(Buf* kNew, Buf* vNew, Buf* Kc, Buf* Vc, Buf* pos, int batch, int maxSeq, int hKv) {
    id<MTLComputePipelineState> p = appendPso();
    if (!p) return {0, 0};
    return pmpp::run([&](void* v) {
        encodeKvAppend(((pmpp::metal::Pass*)v)->encoder, p, buffer(kNew), buffer(vNew), buffer(Kc), buffer(Vc),
                       buffer(pos), (uint32_t)batch, (uint32_t)maxSeq, (uint32_t)hKv);
    });
}
static pmpp::Timing runDecode(Buf* q, Buf* Kc, Buf* Vc, Buf* len, Buf* out, int batch, int hQ, int hKv, int maxSeq,
                              float scale) {
    id<MTLComputePipelineState> p = decodePso();
    if (!p) return {0, 0};
    return pmpp::run([&](void* v) {
        encodeDecodeAttention(((pmpp::metal::Pass*)v)->encoder, p, buffer(q), buffer(Kc), buffer(Vc), buffer(len),
                              buffer(out), (uint32_t)batch, (uint32_t)hQ, (uint32_t)hKv, (uint32_t)maxSeq, scale);
    });
}
static pmpp::Timing runStep(Buf* kNew, Buf* vNew, Buf* pos, Buf* q, Buf* Kc, Buf* Vc, Buf* len, Buf* out,
                            int batch, int hQ, int hKv, int maxSeq, float scale) {
    id<MTLComputePipelineState> pa = appendPso(), pd = decodePso();
    if (!pa || !pd) return {0, 0};
    return pmpp::run([&](void* v) {   // one encoder: the attention runs after the append (ch02-facts C9)
        id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
        encodeKvAppend(enc, pa, buffer(kNew), buffer(vNew), buffer(Kc), buffer(Vc), buffer(pos), (uint32_t)batch,
                       (uint32_t)maxSeq, (uint32_t)hKv);
        encodeDecodeAttention(enc, pd, buffer(q), buffer(Kc), buffer(Vc), buffer(len), buffer(out), (uint32_t)batch,
                              (uint32_t)hQ, (uint32_t)hKv, (uint32_t)maxSeq, scale);
    });
}
static double memoryBudget() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
static double largestBuffer() { return pmpp::device_prop("maxBufferLength", 0); }
#else
void launchKvAppend(const float* k_new, const float* v_new, float* Kc, float* Vc, const int* pos, int batch,
                    int max_seq, int h_kv);
void launchDecodeAttention(const float* q, const float* Kc, const float* Vc, const int* len, float* out, int batch,
                           int h_q, int h_kv, int max_seq, float scale);
static float* f(Buf* b) { return (float*)pmpp::raw(b); }
static int* i32(Buf* b) { return (int*)pmpp::raw(b); }
static std::string pipelineProblem() { return ""; }
static pmpp::Timing runAppend(Buf* kNew, Buf* vNew, Buf* Kc, Buf* Vc, Buf* pos, int batch, int maxSeq, int hKv) {
    return pmpp::run([&](void*) { launchKvAppend(f(kNew), f(vNew), f(Kc), f(Vc), i32(pos), batch, maxSeq, hKv); });
}
static pmpp::Timing runDecode(Buf* q, Buf* Kc, Buf* Vc, Buf* len, Buf* out, int batch, int hQ, int hKv, int maxSeq,
                              float scale) {
    return pmpp::run([&](void*) {
        launchDecodeAttention(f(q), f(Kc), f(Vc), i32(len), f(out), batch, hQ, hKv, maxSeq, scale);
    });
}
static pmpp::Timing runStep(Buf* kNew, Buf* vNew, Buf* pos, Buf* q, Buf* Kc, Buf* Vc, Buf* len, Buf* out,
                            int batch, int hQ, int hKv, int maxSeq, float scale) {
    return pmpp::run([&](void*) {   // one stream: the attention runs after the append
        launchKvAppend(f(kNew), f(vNew), f(Kc), f(Vc), i32(pos), batch, maxSeq, hKv);
        launchDecodeAttention(f(q), f(Kc), f(Vc), i32(len), f(out), batch, hQ, hKv, maxSeq, scale);
    });
}
static double memoryBudget() { return pmpp::device_prop("totalGlobalMem", 0); }
static double largestBuffer() { return memoryBudget(); }
#endif

// ---------------------------------------------------------------- CPU references
namespace m7 {
// One query per (b, h) over the first len[b] rows of K/V head h / (h_q / h_kv): Fig. 20.6's
// S' = Q'K^T (scaled), P' = softmax(S') (Eq. 20.2, max subtracted), O' = P'V. Double.
void cpuDecodeAttention(const float* q, const float* Kc, const float* Vc, const int* len, int batch, int h_q,
                        int h_kv, int max_seq, double scale, std::vector<double>& out) {
    out.assign((size_t)batch * h_q * HEAD_DIM, 0.0);
    const size_t rowLen = (size_t)h_kv * HEAD_DIM;
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < h_q; ++h) {
            const int g = h / (h_q / h_kv);
            const float* qr = q + ((size_t)b * h_q + h) * HEAD_DIM;
            std::vector<double> s(len[b]);
            double m = -INFINITY;
            for (int t = 0; t < len[b]; ++t) {
                const float* kr = Kc + ((size_t)b * max_seq + t) * rowLen + (size_t)g * HEAD_DIM;
                double dot = 0;
                for (int e = 0; e < HEAD_DIM; ++e) dot += (double)qr[e] * kr[e];
                s[t] = dot * scale;
                m = std::max(m, s[t]);
            }
            double D = 0;
            for (int t = 0; t < len[b]; ++t) D += std::exp(s[t] - m);
            double* o = &out[((size_t)b * h_q + h) * HEAD_DIM];
            for (int t = 0; t < len[b]; ++t) {
                const double p = std::exp(s[t] - m) / D;
                const float* vr = Vc + ((size_t)b * max_seq + t) * rowLen + (size_t)g * HEAD_DIM;
                for (int e = 0; e < HEAD_DIM; ++e) o[e] += p * vr[e];
            }
        }
    }
}

// Eq. 20.1 for one head, every row: S = scale Q K^T, plus M (-infinity where column > row),
// softmax by rows with the row max subtracted (Eq. 20.2), O = P V. Double.
void cpuCausalAttention(const double* Q, const double* K, const double* V, int N, int d, double scale,
                        std::vector<double>& O) {
    O.assign((size_t)N * d, 0.0);
    std::vector<double> row(N);
    for (int r = 0; r < N; ++r) {
        double m = -INFINITY;
        for (int c = 0; c < N; ++c) {
            double s = 0;
            for (int e = 0; e < d; ++e) s += Q[(size_t)r * d + e] * K[(size_t)c * d + e];
            row[c] = s * scale + (c > r ? -INFINITY : 0.0);   // + M
            m = std::max(m, row[c]);
        }
        double D = 0;
        for (int c = 0; c < N; ++c) D += std::exp(row[c] - m);   // e^(-inf) = 0 for masked columns
        for (int c = 0; c < N; ++c) {
            const double p = std::exp(row[c] - m) / D;
            for (int e = 0; e < d; ++e) O[(size_t)r * d + e] += p * V[(size_t)c * d + e];
        }
    }
}
}  // namespace m7

// ---------------------------------------------------------------- tests
// Outputs are averages of V rows, all in [-1, 1] or a few times that, so the tolerance
// 3e-5 * (1 + |reference|) is close to an absolute 3e-5. It leaves room for float32
// accumulation over 4097 keys and for Metal's fast exp ("<= 3 + floor(fabs(2 * x)) ulp",
// ch10-facts C4): the reference solution's largest error, measured on Metal when the lab
// was written, was below 3e-7. It leaves none for a missing rescale, a wrong row, or one
// key of 4097 left out (that moves an output by about 1/4097 of a V value).
static const double kRtol = 3e-5;

static int tests(const pmpp::Args& a) {
    pmpp::Tests t(a);
    const std::string problem = pipelineProblem();
    t.check("kernels build; decode_attention gets 32-lane SIMD-groups and 256 threads (Metal only)",
            [&] { return problem; });

    m7::appendTests(t, runAppend);
    m7::decodeTests(t, runDecode, {{8, 8}, {8, 2}, {8, 1}, {6, 2}}, kRtol);

    const std::vector<int> prompt = m7::stepPrompts();
    std::string name = "16-step decode loop, h_q 8, h_kv 2, prompts " + m7::lensText(prompt).substr(4) +
                       ": each step equals the last row of full causal attention";
    t.check(name, [&] { return m7::stepLoop(runStep, 8, 2, prompt, 16, kRtol, 900); });
    return t.finish();
}

// ---------------------------------------------------------------- benchmark
// Decode attention alone at h_q = h_kv = 32, HEAD_DIM 128 (MHA, a 7B-class shape, our
// choice), for each cache length and batch. GB/s counts the cache bytes one call must
// read: 2 x len x h_kv x 128 x 4 per sequence (q and out are 0.2% of that at len 512).
// A size is skipped when its two caches would take more than half of the memory the
// device reports, or one cache more than its largest buffer (our margins).
// The caches are not filled: filling gigabytes from the host takes longer than the
// measurement, and this kernel has no branch that depends on the values. Metal hands out
// buffers filled with 0xA5 bytes; CUDA and HIP leave whatever was there.
static int bench(const pmpp::Args& a) {
    if (!pipelineProblem().empty()) {
        std::printf("bench skipped: %s\n", pipelineProblem().c_str());
        return 1;
    }
    const int hQ = 32, hKv = 32;
    const double budget = memoryBudget(), maxBuf = largestBuffer();
    std::printf("decode attention, h_q = h_kv = %d, HEAD_DIM %d, fp32; best and median of %d calls\n", hQ, HEAD_DIM,
                a.reps);
    std::printf("  %6s %6s %12s %12s %12s %10s\n", "batch", "len", "cache MiB", "best ms", "median ms", "GB/s");
    for (int batch : {1, 8}) {
        for (int len : {512, 2048, 8192, 32768}) {
            const double cacheBytes = (double)batch * len * hKv * HEAD_DIM * 4;   // one of K, V
            if (2 * cacheBytes > 0.5 * budget || cacheBytes > maxBuf) {
                std::printf("  %6d %6d %12.0f   skipped: needs %.2f GB, device reports %.2f GB (largest buffer %.2f GB)\n",
                            batch, len, 2 * cacheBytes / 1048576.0, 2 * cacheBytes / 1e9, budget / 1e9, maxBuf / 1e9);
                pmpp::bench_row(a, "decode_skipped").n("batch", batch).n("len", len).n("h_q", hQ).n("h_kv", hKv)
                    .n("cache_bytes", 2 * cacheBytes).n("device_bytes", budget).s("why", "does not fit").emit();
                continue;
            }
            m7::Bufs bufs;
            auto qh = pmpp::random_floats((size_t)batch * hQ * HEAD_DIM, 77);
            Buf* q = bufs.add(m7::upload(qh));
            Buf* Kc = bufs.add(pmpp::alloc((size_t)cacheBytes));
            Buf* Vc = bufs.add(pmpp::alloc((size_t)cacheBytes));
            Buf* lenb = bufs.add(m7::upload(std::vector<int>(batch, len)));
            Buf* out = bufs.add(pmpp::alloc((size_t)batch * hQ * HEAD_DIM * 4));
            const pmpp::Stat st = pmpp::time_it(a.reps, [&] {
                return runDecode(q, Kc, Vc, lenb, out, batch, hQ, hKv, len, m7::headScale());
            });
            const std::string err = pmpp::take_error();
            if (!err.empty() || st.min_s <= 0) {
                std::printf("  %6d %6d   failed: %s\n", batch, len, err.c_str());
                return 1;
            }
            const double gbs = 2 * cacheBytes / st.min_s / 1e9;
            std::printf("  %6d %6d %12.0f %12.4f %12.4f %10.1f\n", batch, len, 2 * cacheBytes / 1048576.0,
                        st.min_s * 1e3, st.median_s * 1e3, gbs);
            pmpp::bench_row(a, "decode").n("batch", batch).n("len", len).n("h_q", hQ).n("h_kv", hKv)
                .n("head_dim", HEAD_DIM).n("cache_bytes", 2 * cacheBytes).n("seconds", st.min_s)
                .n("median_seconds", st.median_s).n("ms", st.min_s * 1e3).n("gb_per_s", gbs).emit();
        }
    }

    // The append alone: one token per sequence, a few hundred kilobytes at most. Metal
    // reports the command buffer's GPU time; CUDA and HIP report host wall-clock time around
    // the launch and the wait, so there the launch is included.
    std::printf("KV append, h_kv %d, HEAD_DIM %d (best of %d calls; %s)\n", hKv, HEAD_DIM, a.reps,
                std::string(pmpp::backend()) == "metal" ? "GPU time of the command buffer"
                                                        : "host time around launch and wait");
    for (int batch : {1, 8}) {
        const int maxSeq = 2048;
        const size_t rowBytes = (size_t)batch * hKv * HEAD_DIM * 4;
        m7::Bufs bufs;
        Buf* kn = bufs.add(m7::upload(pmpp::random_floats(rowBytes / 4, 5)));
        Buf* vn = bufs.add(m7::upload(pmpp::random_floats(rowBytes / 4, 6)));
        Buf* Kc = bufs.add(pmpp::alloc((size_t)maxSeq * rowBytes));
        Buf* Vc = bufs.add(pmpp::alloc((size_t)maxSeq * rowBytes));
        Buf* pos = bufs.add(m7::upload(std::vector<int>(batch, maxSeq / 2)));
        const pmpp::Stat st = pmpp::time_it(a.reps, [&] { return runAppend(kn, vn, Kc, Vc, pos, batch, maxSeq, hKv); });
        std::printf("  batch %d: %.1f us per call (moves %.0f KiB)\n", batch, st.min_s * 1e6, 4.0 * rowBytes / 1024);
        pmpp::bench_row(a, "append").n("batch", batch).n("h_kv", hKv).n("bytes_moved", 4.0 * rowBytes)
            .n("seconds", st.min_s).n("us", st.min_s * 1e6).emit();
    }
    std::printf("These are measurements of this machine, not expectations. Compare the GB/s with lab 5.1's.\n");
    return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
