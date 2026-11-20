// Labs 20.4 and 20.5 (milestone M7): the tests both drivers run. Provided; you do not edit
// this file. Each lab's main.cpp wraps its device calls for its backend, defines the two
// CPU references declared below (the project keeps CPU references in main.cpp), and calls
// these functions.
//
// Layouts (row-major float32, batch first, as every milestone of the project):
//   q, out        [batch, h_q, HEAD_DIM]
//   k_new, v_new  [batch, h_kv, HEAD_DIM]
//   Kc, Vc        [batch, max_seq, h_kv, HEAD_DIM]
//   pos, len      [batch] int32
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "pmpp/harness.hpp"

namespace m7 {

constexpr int HEAD_DIM = 128;

// ---- defined in each lab's main.cpp ----
// out[(b*h_q + h)*HEAD_DIM + d] = sum over t < len[b] of softmax_t(scale * q[b][h] . K[b][t][g]) * V[b][t][g][d],
// g = h / (h_q / h_kv), accumulated in double. Rows t >= len[b] are never read.
void cpuDecodeAttention(const float* q, const float* Kc, const float* Vc, const int* len,
                        int batch, int h_q, int h_kv, int max_seq, double scale, std::vector<double>& out);
// Eq. 20.1 for one head over N tokens: O = softmax(scale * Q K^T + M) V with the causal
// mask M (-infinity above the diagonal) and the max-subtracted softmax of Eq. 20.2.
// Q, K, V are N x d row-major; O (N x d) is written. In double.
void cpuCausalAttention(const double* Q, const double* K, const double* V, int N, int d, double scale,
                        std::vector<double>& O);

// ---- the device calls, wrapped by main.cpp; each launches and waits ----
using pmpp::Buf;
using AppendRun = std::function<pmpp::Timing(Buf* kNew, Buf* vNew, Buf* Kc, Buf* Vc, Buf* pos,
                                             int batch, int maxSeq, int hKv)>;
using DecodeRun = std::function<pmpp::Timing(Buf* q, Buf* Kc, Buf* Vc, Buf* len, Buf* out,
                                             int batch, int hQ, int hKv, int maxSeq, float scale)>;
// One decode step in one submission: the append, then the attention over the cache it
// has just written (one stream on CUDA and HIP, one compute encoder on Metal).
using StepRun = std::function<pmpp::Timing(Buf* kNew, Buf* vNew, Buf* pos, Buf* q, Buf* Kc, Buf* Vc,
                                           Buf* len, Buf* out, int batch, int hQ, int hKv, int maxSeq,
                                           float scale)>;

inline float headScale() { return 1.0f / std::sqrt((float)HEAD_DIM); }   // the book's scaling, 1/sqrt(d)
inline double headScaleD() { return 1.0 / std::sqrt((double)HEAD_DIM); }

// Values that no kernel computes here: what a cache element must still hold if nobody
// wrote it, and the band after `out`.
inline float canaryK(size_t i) { return -1000.0f - (float)(i % 997); }
inline float canaryV(size_t i) { return -3000.0f - (float)(i % 991); }
constexpr float kGuard = 12345.0f;
constexpr int kGuardLen = 64;
inline float qnan() { return std::numeric_limits<float>::quiet_NaN(); }

inline std::string fmt(const char* f, double a = 0, double b = 0, double c = 0, double d = 0, double e = 0, double g = 0) {
    char buf[320];
    std::snprintf(buf, sizeof buf, f, a, b, c, d, e, g);
    return buf;
}

inline Buf* upload(const void* p, size_t bytes) {
    Buf* b = pmpp::alloc(bytes);
    pmpp::upload(b, p, bytes);
    return b;
}
template <class T>
inline Buf* upload(const std::vector<T>& v) { return upload(v.data(), v.size() * sizeof(T)); }

struct Bufs {
    std::vector<Buf*> list;
    Buf* add(Buf* b) { list.push_back(b); return b; }
    ~Bufs() { for (Buf* b : list) pmpp::release(b); }
};

// `out` starts as NaN (an element nobody writes fails) and is followed by kGuardLen floats of
// kGuard that must survive. Returns an empty string or the first problem.
inline std::string checkOut(const std::vector<float>& got, const std::vector<double>& ref, int batch, int hQ,
                            double rtol) {
    const size_t n = (size_t)batch * hQ * HEAD_DIM;
    size_t bad = 0, first = 0;
    double worst = 0;
    for (size_t i = 0; i < n; ++i) {
        const double e = std::fabs((double)got[i] - ref[i]);
        if (!(e <= rtol * (1.0 + std::fabs(ref[i])))) {   // also catches NaN
            if (bad++ == 0) first = i;
        }
        if (e > worst || std::isnan(e)) worst = e;
    }
    if (bad) {
        const size_t row = first / HEAD_DIM;
        return fmt("%.0f of %.0f outputs wrong; first: sequence %.0f, head %.0f, dim %.0f: got %.7g",
                   (double)bad, (double)n, (double)(row / hQ), (double)(row % hQ), (double)(first % HEAD_DIM),
                   (double)got[first]) +
               fmt(", want %.7g (max abs error %.3g)", ref[first], worst);
    }
    for (int i = 0; i < kGuardLen; ++i) {
        if (!(got[n + i] == kGuard)) {
            return fmt("wrote past the end of out: guard float %.0f after the last output was changed", (double)i);
        }
    }
    return "";
}

// ---------------------------------------------------------------- append
// Two appends into caches full of canaries: the second at different rows, with new data.
// Afterwards row pos1[b] of sequence b holds the first call's row, row pos2[b] the second's,
// and every other element of both caches its canary. Compared bit for bit: an append copies.
inline std::string appendCase(const AppendRun& append, int batch, int maxSeq, int hKv,
                              const std::vector<int>& pos1, const std::vector<int>& pos2, uint32_t seed) {
    const size_t rowLen = (size_t)hKv * HEAD_DIM;
    const size_t cacheN = (size_t)batch * maxSeq * rowLen;
    std::vector<float> K(cacheN), V(cacheN);
    for (size_t i = 0; i < cacheN; ++i) { K[i] = canaryK(i); V[i] = canaryV(i); }
    Bufs bufs;
    Buf* Kc = bufs.add(upload(K));
    Buf* Vc = bufs.add(upload(V));
    std::vector<std::vector<float>> kNew, vNew;
    for (int call = 0; call < 2; ++call) {
        const std::vector<int>& pos = call == 0 ? pos1 : pos2;
        kNew.push_back(pmpp::random_floats((size_t)batch * rowLen, seed + 10 * call));
        vNew.push_back(pmpp::random_floats((size_t)batch * rowLen, seed + 10 * call + 1));
        Buf* kb = bufs.add(upload(kNew.back()));
        Buf* vb = bufs.add(upload(vNew.back()));
        Buf* pb = bufs.add(upload(pos));
        append(kb, vb, Kc, Vc, pb, batch, maxSeq, hKv);
        for (int b = 0; b < batch; ++b) {
            for (size_t e = 0; e < rowLen; ++e) {
                const size_t dst = ((size_t)b * maxSeq + pos[b]) * rowLen + e;
                K[dst] = kNew.back()[(size_t)b * rowLen + e];
                V[dst] = vNew.back()[(size_t)b * rowLen + e];
            }
        }
        std::vector<float> gotK(cacheN), gotV(cacheN);
        pmpp::download(Kc, gotK.data(), cacheN * 4);
        pmpp::download(Vc, gotV.data(), cacheN * 4);
        for (int which = 0; which < 2; ++which) {
            const std::vector<float>& got = which == 0 ? gotK : gotV;
            const std::vector<float>& want = which == 0 ? K : V;
            for (size_t i = 0; i < cacheN; ++i) {
                if (std::memcmp(&got[i], &want[i], 4) != 0) {
                    const size_t row = i / rowLen;
                    const int b = (int)(row / maxSeq), t = (int)(row % maxSeq);
                    const bool isNew = (t == pos1[b]) || (call == 1 && t == pos2[b]);
                    return fmt("after call %.0f, ", call + 1) + (which == 0 ? "K" : "V") +
                           fmt(" cache, sequence %.0f, row %.0f, K/V head %.0f, dim %.0f: ", b, t,
                               (double)((i % rowLen) / HEAD_DIM), (double)(i % HEAD_DIM)) +
                           fmt("got %.7g, want %.7g", got[i], want[i]) +
                           (isNew ? " (a row an append should have written)"
                                  : " (a row no append should touch: it held a canary)");
                }
            }
        }
    }
    return "";
}

inline void appendTests(pmpp::Tests& t, const AppendRun& append) {
    struct C { int batch, maxSeq, hKv; std::vector<int> pos1, pos2; };
    const std::vector<C> cases = {
        {1, 2, 1, {0}, {1}},
        {3, 7, 2, {0, 6, 3}, {1, 5, 4}},
        {4, 33, 8, {32, 0, 17, 5}, {31, 1, 16, 6}},
        {2, 5, 3, {4, 2}, {0, 3}},
    };
    for (const C& c : cases) {
        std::string name = fmt("append: batch %.0f, max_seq %.0f, h_kv %.0f, two calls", c.batch, c.maxSeq, c.hKv);
        t.check(name, [&]() -> std::string {
            return appendCase(append, c.batch, c.maxSeq, c.hKv, c.pos1, c.pos2, 50 + c.batch * 7 + c.hKv);
        });
    }
}

// ---------------------------------------------------------------- decode
// q and the cache rows below len[b] are uniform in [-1, 1]; rows at or past len[b] are NaN,
// so a kernel that reads one poisons its output. With `lateMax`, the last key of every
// sequence and K/V head is 1.6 times the sum of the queries that read it: each of those
// queries scores it at about 6 while every other score is near 0, so the running max jumps
// at the very end, in one warp only, and the merge must scale every other warp down by
// about e^-6 (and a missing rescale shows).
struct DecodeData {
    int batch, hQ, hKv, maxSeq;
    std::vector<int> lens;
    std::vector<float> q, K, V;
};

inline DecodeData makeDecodeData(int hQ, int hKv, const std::vector<int>& lens, bool lateMax, uint32_t seed) {
    DecodeData d;
    d.batch = (int)lens.size(); d.hQ = hQ; d.hKv = hKv; d.lens = lens;
    d.maxSeq = *std::max_element(lens.begin(), lens.end()) + 3;   // 3 rows nobody may read
    const size_t rowLen = (size_t)hKv * HEAD_DIM;
    d.q = pmpp::random_floats((size_t)d.batch * hQ * HEAD_DIM, seed);
    d.K = pmpp::random_floats((size_t)d.batch * d.maxSeq * rowLen, seed + 1);
    d.V = pmpp::random_floats((size_t)d.batch * d.maxSeq * rowLen, seed + 2);
    const int gq = hQ / hKv;
    for (int b = 0; b < d.batch; ++b) {
        for (int t = lens[b]; t < d.maxSeq; ++t) {
            for (size_t e = 0; e < rowLen; ++e) {
                d.K[((size_t)b * d.maxSeq + t) * rowLen + e] = qnan();
                d.V[((size_t)b * d.maxSeq + t) * rowLen + e] = qnan();
            }
        }
        if (!lateMax) continue;
        const int t = lens[b] - 1;
        for (int g = 0; g < hKv; ++g) {
            for (int e = 0; e < HEAD_DIM; ++e) {
                float sum = 0;
                for (int j = 0; j < gq; ++j) sum += d.q[((size_t)b * hQ + g * gq + j) * HEAD_DIM + e];
                d.K[((size_t)b * d.maxSeq + t) * rowLen + (size_t)g * HEAD_DIM + e] = 1.6f * sum;
            }
        }
    }
    return d;
}

inline std::string runDecodeCase(const DecodeRun& decode, const DecodeData& d, double rtol) {
    Bufs bufs;
    Buf* q = bufs.add(upload(d.q));
    Buf* Kc = bufs.add(upload(d.K));
    Buf* Vc = bufs.add(upload(d.V));
    Buf* len = bufs.add(upload(d.lens));
    const size_t n = (size_t)d.batch * d.hQ * HEAD_DIM;
    std::vector<float> got(n + kGuardLen, qnan());
    std::fill(got.begin() + n, got.end(), kGuard);
    Buf* out = bufs.add(upload(got));
    decode(q, Kc, Vc, len, out, d.batch, d.hQ, d.hKv, d.maxSeq, headScale());
    pmpp::download(out, got.data(), got.size() * 4);
    std::vector<double> ref;
    cpuDecodeAttention(d.q.data(), d.K.data(), d.V.data(), d.lens.data(), d.batch, d.hQ, d.hKv, d.maxSeq,
                       headScaleD(), ref);
    return checkOut(got, ref, d.batch, d.hQ, rtol);
}

inline std::string lensText(const std::vector<int>& lens) {
    std::string s = "len {";
    for (size_t i = 0; i < lens.size(); ++i) s += (i ? ", " : "") + std::to_string(lens[i]);
    return s + "}";
}

// The sequence lengths: 1 to 4 sequences per call, each with its own length. The
// emulators pay for every warp shuffle with a hand-over between OS threads, so they get
// short sequences, and late-max data only (it is random apart from the last key).
inline std::vector<std::vector<int>> decodeLengths() {
    const std::string be = pmpp::backend();
    if (be == "race") return {{1}, {2, 40}, {33, 1, 9, 17}};
    if (be == "emu") return {{1}, {2, 9}, {33, 1, 3}};
    return {{1}, {2, 31}, {33, 1, 100}, {1000, 4097, 2, 31}};
}

// Every (h_q, h_kv) in `shapes`, every set of lengths, random keys and late-max keys.
// On the emulators: late-max keys only, plus one long case (len 200 and 5) on the last
// shape, so that a warp runs through 25 keys under AddressSanitizer.
inline void decodeTests(pmpp::Tests& t, const DecodeRun& decode, const std::vector<std::pair<int, int>>& shapes,
                        double rtol, const std::string& prefix = "decode") {
    const bool emulated = pmpp::is_emu();
    const bool emu = std::string(pmpp::backend()) == "emu";
    uint32_t seed = 300;
    for (size_t si = 0; si < shapes.size(); ++si) {
        const auto& s = shapes[si];
        const std::string kind = s.first == s.second ? "MHA" : (s.second == 1 ? "MQA" : "GQA");
        std::vector<std::vector<int>> lengths = decodeLengths();
        if (emu && si + 1 == shapes.size()) lengths.push_back({200, 5});
        for (const auto& lens : lengths) {
            for (int late = emulated ? 1 : 0; late <= 1; ++late) {
                const DecodeData d = makeDecodeData(s.first, s.second, lens, late == 1, seed += 11);
                std::string name = prefix + fmt(", h_q %.0f, h_kv %.0f (", s.first, s.second) + kind +
                                   fmt(", g_q %.0f), ", s.first / s.second) + lensText(lens) +
                                   (late ? ", late max" : ", random keys");
                t.check(name, [&]() -> std::string { return runDecodeCase(decode, d, rtol); });
            }
        }
    }
}

// ---------------------------------------------------------------- the 16-step loop
// The milestone: an autoregressive loop. Each sequence starts with `prompt[b]` tokens
// already in the cache (written by the host, standing in for prefill). Every step then
// appends one token's K and V per sequence at pos[b] = prompt[b] + step and attends with
// len[b] = pos[b] + 1, in one submission. The step's q, k_new and v_new are a fixed
// function of the previous step's output, as a model's next layer input would be:
//   q[b][h][e]     = 2 x[b][h][e] + a_q[h][e]
//   k_new[b][g][e] = 2 x[b][g*g_q][(e + 1) % 128] + a_k[g][e]
//   v_new[b][g][e] = 0.9 x[b][g*g_q + g_q - 1][127 - e] + a_v[g][e]
// where x is the device's own previous output and a_* are fixed. The host keeps every
// token's Q, K and V. At every step each output must equal the LAST row of a full causal
// attention (Eq. 20.1, mask and all) recomputed from scratch over all tokens so far:
// section 20.4's claim that the cached decode needs only the new row. At the end the
// caches must hold exactly the history, and the rows past it must still be NaN.
inline std::string stepLoop(const StepRun& step, int hQ, int hKv, const std::vector<int>& prompt, int steps,
                            double rtol, uint32_t seed) {
    const int batch = (int)prompt.size();
    const int gq = hQ / hKv;
    const int maxSeq = *std::max_element(prompt.begin(), prompt.end()) + steps + 2;
    const size_t rowLen = (size_t)hKv * HEAD_DIM, qLen = (size_t)hQ * HEAD_DIM;
    const size_t cacheN = (size_t)batch * maxSeq * rowLen;

    // Host history: per sequence, one row per token.
    std::vector<std::vector<float>> histQ(batch), histK(batch), histV(batch);
    std::vector<float> K(cacheN, qnan()), V(cacheN, qnan());
    for (int b = 0; b < batch; ++b) {
        const auto pk = pmpp::random_floats((size_t)prompt[b] * rowLen, seed + 3 * b);
        const auto pv = pmpp::random_floats((size_t)prompt[b] * rowLen, seed + 3 * b + 1);
        const auto pq = pmpp::random_floats((size_t)prompt[b] * qLen, seed + 3 * b + 2);
        histK[b] = pk; histV[b] = pv; histQ[b] = pq;
        std::copy(pk.begin(), pk.end(), K.begin() + (size_t)b * maxSeq * rowLen);
        std::copy(pv.begin(), pv.end(), V.begin() + (size_t)b * maxSeq * rowLen);
    }
    auto aq = pmpp::random_floats(qLen, seed + 100), ak = pmpp::random_floats(rowLen, seed + 101),
         av = pmpp::random_floats(rowLen, seed + 102);
    for (auto* a : {&aq, &ak, &av}) for (float& v : *a) v *= 0.5f;
    std::vector<float> x = pmpp::random_floats((size_t)batch * qLen, seed + 103);

    Bufs bufs;
    Buf* Kc = bufs.add(upload(K));
    Buf* Vc = bufs.add(upload(V));
    Buf* qb = bufs.add(pmpp::alloc((size_t)batch * qLen * 4));
    Buf* kb = bufs.add(pmpp::alloc((size_t)batch * rowLen * 4));
    Buf* vb = bufs.add(pmpp::alloc((size_t)batch * rowLen * 4));
    Buf* posb = bufs.add(pmpp::alloc((size_t)batch * 4));
    Buf* lenb = bufs.add(pmpp::alloc((size_t)batch * 4));
    const size_t outN = (size_t)batch * qLen;
    Buf* out = bufs.add(pmpp::alloc((outN + kGuardLen) * 4));

    std::vector<float> q(batch * qLen), kn(batch * rowLen), vn(batch * rowLen), got(outN + kGuardLen);
    std::vector<int> pos(batch), len(batch);
    for (int s = 0; s < steps; ++s) {
        for (int b = 0; b < batch; ++b) {
            pos[b] = prompt[b] + s;
            len[b] = pos[b] + 1;
            const float* xb = &x[(size_t)b * qLen];
            for (int h = 0; h < hQ; ++h)
                for (int e = 0; e < HEAD_DIM; ++e)
                    q[(size_t)b * qLen + h * HEAD_DIM + e] = 2.0f * xb[h * HEAD_DIM + e] + aq[h * HEAD_DIM + e];
            for (int g = 0; g < hKv; ++g)
                for (int e = 0; e < HEAD_DIM; ++e) {
                    kn[(size_t)b * rowLen + g * HEAD_DIM + e] =
                        2.0f * xb[(g * gq) * HEAD_DIM + (e + 1) % HEAD_DIM] + ak[g * HEAD_DIM + e];
                    vn[(size_t)b * rowLen + g * HEAD_DIM + e] =
                        0.9f * xb[(g * gq + gq - 1) * HEAD_DIM + (HEAD_DIM - 1 - e)] + av[g * HEAD_DIM + e];
                }
        }
        std::fill(got.begin(), got.begin() + outN, qnan());
        std::fill(got.begin() + outN, got.end(), kGuard);
        pmpp::upload(qb, q.data(), q.size() * 4);
        pmpp::upload(kb, kn.data(), kn.size() * 4);
        pmpp::upload(vb, vn.data(), vn.size() * 4);
        pmpp::upload(posb, pos.data(), pos.size() * 4);
        pmpp::upload(lenb, len.data(), len.size() * 4);
        pmpp::upload(out, got.data(), got.size() * 4);
        step(kb, vb, posb, qb, Kc, Vc, lenb, out, batch, hQ, hKv, maxSeq, headScale());
        pmpp::download(out, got.data(), got.size() * 4);

        // The host's copy of every token so far, and the full causal attention over it.
        std::vector<double> ref(outN);
        for (int b = 0; b < batch; ++b) {
            histQ[b].insert(histQ[b].end(), q.begin() + (size_t)b * qLen, q.begin() + (size_t)(b + 1) * qLen);
            histK[b].insert(histK[b].end(), kn.begin() + (size_t)b * rowLen, kn.begin() + (size_t)(b + 1) * rowLen);
            histV[b].insert(histV[b].end(), vn.begin() + (size_t)b * rowLen, vn.begin() + (size_t)(b + 1) * rowLen);
            const int N = len[b];
            std::vector<double> Qd((size_t)N * HEAD_DIM), Kd((size_t)N * HEAD_DIM), Vd((size_t)N * HEAD_DIM), O;
            for (int h = 0; h < hQ; ++h) {
                const int g = h / gq;
                for (int tok = 0; tok < N; ++tok)
                    for (int e = 0; e < HEAD_DIM; ++e) {
                        Qd[(size_t)tok * HEAD_DIM + e] = histQ[b][(size_t)tok * qLen + h * HEAD_DIM + e];
                        Kd[(size_t)tok * HEAD_DIM + e] = histK[b][(size_t)tok * rowLen + g * HEAD_DIM + e];
                        Vd[(size_t)tok * HEAD_DIM + e] = histV[b][(size_t)tok * rowLen + g * HEAD_DIM + e];
                    }
                cpuCausalAttention(Qd.data(), Kd.data(), Vd.data(), N, HEAD_DIM, headScaleD(), O);
                for (int e = 0; e < HEAD_DIM; ++e)
                    ref[((size_t)b * hQ + h) * HEAD_DIM + e] = O[(size_t)(N - 1) * HEAD_DIM + e];
            }
        }
        std::string err = checkOut(got, ref, batch, hQ, rtol);
        if (!err.empty()) return fmt("step %.0f of %.0f: ", s + 1, steps) + err;
        std::copy(got.begin(), got.begin() + outN, x.begin());
    }

    // The caches after the loop: exactly the history, then NaN.
    std::vector<float> gotK(cacheN), gotV(cacheN);
    pmpp::download(Kc, gotK.data(), cacheN * 4);
    pmpp::download(Vc, gotV.data(), cacheN * 4);
    for (int b = 0; b < batch; ++b) {
        const int filled = prompt[b] + steps;
        for (int tok = 0; tok < maxSeq; ++tok)
            for (size_t e = 0; e < rowLen; ++e) {
                const size_t i = ((size_t)b * maxSeq + tok) * rowLen + e;
                for (int which = 0; which < 2; ++which) {
                    const float g = which == 0 ? gotK[i] : gotV[i];
                    if (tok < filled) {
                        const float w = (which == 0 ? histK[b] : histV[b])[(size_t)tok * rowLen + e];
                        if (std::memcmp(&g, &w, 4) != 0)
                            return std::string(which == 0 ? "K" : "V") +
                                   fmt(" cache after the loop: sequence %.0f, token %.0f, element %.0f: got %.7g, want %.7g",
                                       b, tok, (double)e, g, w);
                    } else if (!std::isnan(g)) {
                        return std::string(which == 0 ? "K" : "V") +
                               fmt(" cache after the loop: sequence %.0f, row %.0f is past the last token but was written",
                                   b, tok);
                    }
                }
            }
    }
    return "";
}

// The prompts: one sequence starts empty (its first step attends to one key), the others
// at different lengths, so pos and len differ per sequence at every step.
inline std::vector<int> stepPrompts() {
    const std::string be = pmpp::backend();
    if (be == "race") return {0, 3, 5};
    if (be == "emu") return {0, 2};
    return {0, 3, 9, 100};
}

}  // namespace m7
