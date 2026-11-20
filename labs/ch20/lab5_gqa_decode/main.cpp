// Lab 20.5 driver: tests the grouped-query decode kernel with lab 20.4's tests, and times it
// against lab 20.4's per-head kernel. You do not need to edit this file; your work goes in
// work/. The tests are in ../shared-g12/decode_test.hpp; the CPU references are below.
// Lab 20.4's reference solution is built beside your kernel (provided/perhead.cmake): its
// per-head kernel is the benchmark's baseline, and its KV append feeds the 16-step loop.
#include "pmpp/harness.hpp"
#include "../shared-g12/decode_test.hpp"

using pmpp::Buf;
using m7::HEAD_DIM;

// ---- entry points: yours in <variant>/, lab 20.4's from the build directory ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeDecodeAttentionGrouped(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> q,
                                  id<MTLBuffer> Kc, id<MTLBuffer> Vc, id<MTLBuffer> len, id<MTLBuffer> out,
                                  uint32_t batch, uint32_t hQ, uint32_t hKv, uint32_t maxSeq, float scale);
void encodeDecodeAttentionPerHead(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> q,
                                  id<MTLBuffer> Kc, id<MTLBuffer> Vc, id<MTLBuffer> len, id<MTLBuffer> out,
                                  uint32_t batch, uint32_t hQ, uint32_t hKv, uint32_t maxSeq, float scale);
void encodeKvAppend(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> kNew,
                    id<MTLBuffer> vNew, id<MTLBuffer> Kc, id<MTLBuffer> Vc, id<MTLBuffer> pos,
                    uint32_t batch, uint32_t maxSeq, uint32_t hKv);
using pmpp::metal::buffer;
// Your kernel.metal is compiled once per group size g_q = hQ / hKv, with GQ defined as that
// number (the counterpart of kernel.cu's template parameter). The runtime caches each one.
static const int kMaxGq = 8;
static id<MTLComputePipelineState> groupedPso(int gq) {
    return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "decode_attention_grouped", {{"GQ", std::to_string(gq)}});
}
static id<MTLComputePipelineState> perHeadPso() { return pmpp::metal::pipeline(PMPP_CH20_PERHEAD_METAL, "decode_attention"); }
static id<MTLComputePipelineState> appendPso() { return pmpp::metal::pipeline(PMPP_CH20_PERHEAD_METAL, "kv_append"); }

// Your kernel gives each lane HEAD_DIM / 32 dimensions and each threadgroup 8 SIMD-groups:
// it needs 32-lane SIMD-groups (the book's WARP_SIZE) and 256 threads. Apple does not
// document the width as a number (notes/today/ch20-facts.md C3): ask the pipeline.
static std::string checkPso(id<MTLComputePipelineState> p, const char* name) {
    if (!p) return std::string(name) + " did not build: " + pmpp::take_error();
    const NSUInteger logicalWarp = 32, block = 256;
    if (p.threadExecutionWidth != logicalWarp)
        return std::string(name) + m7::fmt(" assumes 32-lane SIMD-groups; this pipeline's threadExecutionWidth is %.0f",
                                           (double)p.threadExecutionWidth);
    if (p.maxTotalThreadsPerThreadgroup < block)
        return std::string(name) + m7::fmt(" needs 256 threads per threadgroup; this pipeline allows %.0f",
                                           (double)p.maxTotalThreadsPerThreadgroup);
    return "";
}
static std::string pipelineProblem() {
    for (int gq = 1; gq <= kMaxGq; ++gq) {
        std::string p = checkPso(groupedPso(gq), "decode_attention_grouped");
        if (!p.empty()) return p + m7::fmt(" (compiled with GQ = %.0f)", gq);
    }
    return "";
}
static std::string providedProblem() {
    if (!appendPso()) return "lab 20.4's kv_append did not build: " + pmpp::take_error();
    return checkPso(perHeadPso(), "lab 20.4's decode_attention");
}
static pmpp::Timing runDecodeWith(id<MTLComputePipelineState> p, bool grouped, Buf* q, Buf* Kc, Buf* Vc, Buf* len,
                                  Buf* out, int batch, int hQ, int hKv, int maxSeq, float scale) {
    if (!p) return {0, 0};
    return pmpp::run([&](void* v) {
        id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
        (grouped ? encodeDecodeAttentionGrouped : encodeDecodeAttentionPerHead)(
            enc, p, buffer(q), buffer(Kc), buffer(Vc), buffer(len), buffer(out), (uint32_t)batch, (uint32_t)hQ,
            (uint32_t)hKv, (uint32_t)maxSeq, scale);
    });
}
static pmpp::Timing runGrouped(Buf* q, Buf* Kc, Buf* Vc, Buf* len, Buf* out, int batch, int hQ, int hKv, int maxSeq,
                               float scale) {
    return runDecodeWith(groupedPso(hQ / hKv), true, q, Kc, Vc, len, out, batch, hQ, hKv, maxSeq, scale);
}
static pmpp::Timing runPerHead(Buf* q, Buf* Kc, Buf* Vc, Buf* len, Buf* out, int batch, int hQ, int hKv, int maxSeq,
                               float scale) {
    return runDecodeWith(perHeadPso(), false, q, Kc, Vc, len, out, batch, hQ, hKv, maxSeq, scale);
}
static pmpp::Timing runStep(Buf* kNew, Buf* vNew, Buf* pos, Buf* q, Buf* Kc, Buf* Vc, Buf* len, Buf* out,
                            int batch, int hQ, int hKv, int maxSeq, float scale) {
    id<MTLComputePipelineState> pa = appendPso(), pd = groupedPso(hQ / hKv);
    if (!pa || !pd) return {0, 0};
    return pmpp::run([&](void* v) {   // one encoder: the attention runs after the append (ch02-facts C9)
        id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
        encodeKvAppend(enc, pa, buffer(kNew), buffer(vNew), buffer(Kc), buffer(Vc), buffer(pos), (uint32_t)batch,
                       (uint32_t)maxSeq, (uint32_t)hKv);
        encodeDecodeAttentionGrouped(enc, pd, buffer(q), buffer(Kc), buffer(Vc), buffer(len), buffer(out),
                                     (uint32_t)batch, (uint32_t)hQ, (uint32_t)hKv, (uint32_t)maxSeq, scale);
    });
}
static double memoryBudget() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
static double largestBuffer() { return pmpp::device_prop("maxBufferLength", 0); }
#else
void launchDecodeAttentionGrouped(const float* q, const float* Kc, const float* Vc, const int* len, float* out,
                                  int batch, int h_q, int h_kv, int max_seq, float scale);
void launchDecodeAttentionPerHead(const float* q, const float* Kc, const float* Vc, const int* len, float* out,
                                  int batch, int h_q, int h_kv, int max_seq, float scale);
void launchKvAppend(const float* k_new, const float* v_new, float* Kc, float* Vc, const int* pos, int batch,
                    int max_seq, int h_kv);
static float* f(Buf* b) { return (float*)pmpp::raw(b); }
static int* i32(Buf* b) { return (int*)pmpp::raw(b); }
static std::string pipelineProblem() { return ""; }
static std::string providedProblem() { return ""; }
static pmpp::Timing runGrouped(Buf* q, Buf* Kc, Buf* Vc, Buf* len, Buf* out, int batch, int hQ, int hKv, int maxSeq,
                               float scale) {
    return pmpp::run([&](void*) {
        launchDecodeAttentionGrouped(f(q), f(Kc), f(Vc), i32(len), f(out), batch, hQ, hKv, maxSeq, scale);
    });
}
static pmpp::Timing runPerHead(Buf* q, Buf* Kc, Buf* Vc, Buf* len, Buf* out, int batch, int hQ, int hKv, int maxSeq,
                               float scale) {
    return pmpp::run([&](void*) {
        launchDecodeAttentionPerHead(f(q), f(Kc), f(Vc), i32(len), f(out), batch, hQ, hKv, maxSeq, scale);
    });
}
static pmpp::Timing runStep(Buf* kNew, Buf* vNew, Buf* pos, Buf* q, Buf* Kc, Buf* Vc, Buf* len, Buf* out,
                            int batch, int hQ, int hKv, int maxSeq, float scale) {
    return pmpp::run([&](void*) {   // one stream: the attention runs after the append
        launchKvAppend(f(kNew), f(vNew), f(Kc), f(Vc), i32(pos), batch, maxSeq, hKv);
        launchDecodeAttentionGrouped(f(q), f(Kc), f(Vc), i32(len), f(out), batch, hQ, hKv, maxSeq, scale);
    });
}
static double memoryBudget() { return pmpp::device_prop("totalGlobalMem", 0); }
static double largestBuffer() { return memoryBudget(); }
#endif

// ---------------------------------------------------------------- CPU references
// The same two references as lab 20.4's main.cpp.
namespace m7 {
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
        for (int c = 0; c < N; ++c) D += std::exp(row[c] - m);
        for (int c = 0; c < N; ++c) {
            const double p = std::exp(row[c] - m) / D;
            for (int e = 0; e < d; ++e) O[(size_t)r * d + e] += p * V[(size_t)c * d + e];
        }
    }
}
}  // namespace m7

// ---------------------------------------------------------------- tests
static const double kRtol = 3e-5;   // as lab 20.4 (its main.cpp says why)

static int tests(const pmpp::Args& a) {
    pmpp::Tests t(a);
    const std::string provided = providedProblem();
    t.check("provided: lab 20.4's per-head kernel, h_q 8, h_kv 2, len {33, 1, 9} (a failure here is the course's bug)",
            [&]() -> std::string {
                if (!provided.empty()) return provided;
                return m7::runDecodeCase(runPerHead, m7::makeDecodeData(8, 2, {33, 1, 9}, true, 7), kRtol);
            });
    const std::string problem = pipelineProblem();
    t.check("your kernel builds for GQ = 1 to 8; decode_attention_grouped gets 32-lane SIMD-groups and 256 threads "
            "(Metal only)",
            [&] { return problem; });

    m7::decodeTests(t, runGrouped, {{8, 8}, {8, 4}, {8, 2}, {8, 1}, {6, 2}}, kRtol, "grouped decode");

    const std::vector<int> prompt = m7::stepPrompts();
    std::string name = "16-step decode loop (lab 20.4's append, your kernel), h_q 8, h_kv 2, prompts " +
                       m7::lensText(prompt).substr(4) + ": each step equals the last row of full causal attention";
    t.check(name, [&] { return m7::stepLoop(runStep, 8, 2, prompt, 16, kRtol, 950); });
    return t.finish();
}

// ---------------------------------------------------------------- benchmark
// h_q = 32, HEAD_DIM 128, len 8192, batch 1 and 4, h_kv = 32, 8 and 4 (g_q 1, 4, 8): lab
// 20.4's per-head kernel and yours take turns in every round; the best round counts. GB/s
// counts the UNIQUE cache bytes, 2 x len x h_kv x 128 x 4 per sequence, for both kernels:
// the per-head kernel asks for every row g_q times, and the bytes column is what a kernel
// that read each row once would move. The caches are not filled (lab 20.4's driver says
// why). Sizes whose caches would take more than half of the device's reported memory are
// skipped.
static int bench(const pmpp::Args& a) {
    const std::string why = providedProblem().empty() ? pipelineProblem() : providedProblem();
    if (!why.empty()) {
        std::printf("bench skipped: %s\n", why.c_str());
        return 1;
    }
    const int hQ = 32, len = 8192;
    const double budget = memoryBudget(), maxBuf = largestBuffer();
    std::printf("decode attention, h_q %d, HEAD_DIM %d, len %d, fp32; best and median of %d rounds\n", hQ, HEAD_DIM,
                len, a.reps);
    std::printf("  %5s %5s %4s %10s %8s %10s %10s %10s %10s %9s\n", "batch", "h_kv", "g_q", "cache MiB", "blocks",
                "per-head", "GB/s", "grouped", "GB/s", "speed-up");
    for (int batch : {1, 4}) {
        for (int hKv : {32, 8, 4}) {
            const int gq = hQ / hKv;
            const double cacheBytes = (double)batch * len * hKv * HEAD_DIM * 4;   // one of K, V
            if (2 * cacheBytes > 0.5 * budget || cacheBytes > maxBuf) {
                std::printf("  %5d %5d %4d   skipped: needs %.2f GB, device reports %.2f GB\n", batch, hKv, gq,
                            2 * cacheBytes / 1e9, budget / 1e9);
                pmpp::bench_row(a, "skipped").n("batch", batch).n("h_kv", hKv).n("cache_bytes", 2 * cacheBytes)
                    .s("why", "does not fit").emit();
                continue;
            }
            m7::Bufs bufs;
            Buf* q = bufs.add(m7::upload(pmpp::random_floats((size_t)batch * hQ * HEAD_DIM, 31)));
            Buf* Kc = bufs.add(pmpp::alloc((size_t)cacheBytes));
            Buf* Vc = bufs.add(pmpp::alloc((size_t)cacheBytes));
            Buf* lenb = bufs.add(m7::upload(std::vector<int>(batch, len)));
            Buf* out = bufs.add(pmpp::alloc((size_t)batch * hQ * HEAD_DIM * 4));
            auto ph = [&] { return runPerHead(q, Kc, Vc, lenb, out, batch, hQ, hKv, len, m7::headScale()).best(); };
            auto gr = [&] { return runGrouped(q, Kc, Vc, lenb, out, batch, hQ, hKv, len, m7::headScale()).best(); };
            ph(); gr();                                     // warm-up
            std::vector<double> tp, tg;
            for (int r = 0; r < a.reps; ++r) { tp.push_back(ph()); tg.push_back(gr()); }
            std::sort(tp.begin(), tp.end());
            std::sort(tg.begin(), tg.end());
            const std::string err = pmpp::take_error();
            if (!err.empty() || tp[0] <= 0 || tg[0] <= 0) {
                std::printf("  %5d %5d   failed: %s\n", batch, hKv, err.c_str());
                return 1;
            }
            const double gbP = 2 * cacheBytes / tp[0] / 1e9, gbG = 2 * cacheBytes / tg[0] / 1e9;
            std::printf("  %5d %5d %4d %10.0f %4d/%-3d %8.3f ms %10.1f %7.3f ms %10.1f %8.2fx\n", batch, hKv, gq,
                        2 * cacheBytes / 1048576.0, batch * hQ, batch * hKv, tp[0] * 1e3, gbP, tg[0] * 1e3, gbG,
                        tp[0] / tg[0]);
            pmpp::bench_row(a, "perhead").n("batch", batch).n("len", len).n("h_q", hQ).n("h_kv", hKv).n("g_q", gq)
                .n("blocks", batch * hQ).n("cache_bytes", 2 * cacheBytes).n("seconds", tp[0])
                .n("median_seconds", tp[tp.size() / 2]).n("ms", tp[0] * 1e3).n("gb_per_s", gbP).emit();
            pmpp::bench_row(a, "grouped").n("batch", batch).n("len", len).n("h_q", hQ).n("h_kv", hKv).n("g_q", gq)
                .n("blocks", batch * hKv).n("cache_bytes", 2 * cacheBytes).n("seconds", tg[0])
                .n("median_seconds", tg[tg.size() / 2]).n("ms", tg[0] * 1e3).n("gb_per_s", gbG)
                .n("speedup_vs_perhead", tp[0] / tg[0]).emit();
        }
    }
    std::printf("blocks = thread blocks launched, per-head / grouped. These are measurements of this machine,\n"
                "not expectations.\n");
    return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
