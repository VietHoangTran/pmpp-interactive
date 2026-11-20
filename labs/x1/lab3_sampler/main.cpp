// Lab X1.3 driver (milestone M5 complete): tests your batched GPU sampler against a CPU
// reference sampler, piece by piece and as a whole, runs a toy decode loop, and benchmarks the
// sampler stage by stage and against one CPU thread. You do not need to edit this file; your
// work goes in work/.
#include <chrono>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <set>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// ---------------------------------------------------------------- settings
struct Settings {
  const char* name;
  int topK;
  float topP, minP, temperature, rep, freq, pres;
};
// llama.cpp's defaults (notes/today/x1-facts.md D1) with the penalties switched on.
static const Settings kDefaults = {"llama.cpp defaults, penalties 1.1/0.1/0.1", 40, 0.95f, 0.05f, 0.8f, 1.1f, 0.1f, 0.1f};
static std::vector<Settings> settingsList(bool withWholeRow) {
  std::vector<Settings> v = {kDefaults};
  Settings s = kDefaults;
  s.name = "topK 1"; s.topK = 1; v.push_back(s);
  if (withWholeRow) { s = kDefaults; s.name = "topK 0 (whole row, full sort)"; s.topK = 0; v.push_back(s); }
  s = kDefaults; s.name = "topP 1"; s.topP = 1.0f; v.push_back(s);
  s = kDefaults; s.name = "minP 0"; s.minP = 0.0f; v.push_back(s);
  s = kDefaults; s.name = "temperature 0 (greedy)"; s.temperature = 0.0f; v.push_back(s);
  s = kDefaults; s.name = "temperature 2"; s.temperature = 2.0f; v.push_back(s);
  s = kDefaults; s.name = "no penalties (1, 0, 0)"; s.rep = 1.0f; s.freq = 0.0f; s.pres = 0.0f; v.push_back(s);
  return v;
}
static int effectiveK(int vocab, int topK) { return topK <= 0 || topK >= vocab ? vocab : topK; }

static const int kGuard = 64;
static const uint32_t kGuardBits = 0xDEADBEEFu;
static const int kGuardIdx = -77;
static const int kUnwrittenToken = -1;
static const uint32_t kUnwrittenU = 0xFFC0FFEEu;   // uOut before a launch: a NaN

// ---------------------------------------------------------------- the entry points
// Buffers for the stages, used by the benchmark to time each stage alone.
struct StageBufs {
  Buf *logits, *counts, *W, *candIdx, *candVal, *selScr, *sortedVal, *sortedPos, *sortScr, *probs, *mass, *keep,
      *kept, *n, *seeds, *streams, *steps, *tokens;
};
enum Stage { StPenalties, StSelect, StSort, StSoftmax, StTopP, StMinP, StDraw, StageCount };
static const char* kStageNames[StageCount] = {"penalties", "top-k select", "sort of k", "softmax", "top-p", "min-p + combine", "draw"};

#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
#include "sampler_metal.h"
using pmpp::metal::buffer;
static id<MTLBuffer> bufOrNil(Buf* b) { return b ? buffer(b) : nil; }

static std::string dirOf(const std::string& p) { return p.substr(0, p.find_last_of('/')); }
static const std::string kLabs = dirOf(dirOf(dirOf(dirOf(PMPP_METAL_SOURCE))));   // .../labs

// Your kernels from your kernel.metal; the earlier labs' from their solution kernel.metal files
// and from labs/x1/shared-g11/, loaded by path.
static SamplerPipelines pipes() {
  auto P = [](const std::string& f, const char* fn) { return pmpp::metal::pipeline(f, fn); };
  const std::string mine = PMPP_METAL_SOURCE;
  const std::string c9 = kLabs + "/ch09/lab3_tokencounts/solution/kernel.metal";
  const std::string c10 = kLabs + "/ch10/lab3_softmax_argmax/solution/kernel.metal";
  const std::string c11 = kLabs + "/ch11/lab6_topp/solution/kernel.metal";
  const std::string c14 = kLabs + "/ch14/lab3_sortrows/solution/kernel.metal";
  const std::string g11 = kLabs + "/x1/shared-g11/";
  SamplerPipelines p;
  p.copyRows = P(mine, "copy_rows");
  p.applyRepetition = P(mine, "apply_repetition");
  p.drawRows = P(mine, "draw_rows");
  p.philoxKat = P(mine, "philox_kat");
  p.applyPenalties = P(c9, "apply_penalties");
  p.appendTokens = P(c9, "append_tokens");
  p.argmax = P(c10, "argmax_rows");
  p.softmax = P(c10, "softmax_rows");
  p.topkThreshold = P(g11 + "topk_g11.metal", "topk_threshold");
  p.topkGather = P(g11 + "topk_g11.metal", "topk_gather");
  p.descKeys = P(c14, "desc_keys");
  p.sortPass = P(c14, "sort_rows_pass");
  p.sortGather = P(c14, "gather_rows");
  p.topP = P(c11, "top_p_mask");
  p.minP = P(g11 + "minp_g11.metal", "minp_combine");
  return p;
}

static std::string pipelineProblem() {
  SamplerPipelines p = pipes();
  if (!p.copyRows || !p.applyRepetition || !p.drawRows || !p.philoxKat)
    return "your Metal kernels did not build: " + pmpp::take_error();
  if (!p.applyPenalties || !p.appendTokens || !p.argmax || !p.softmax || !p.topkThreshold || !p.topkGather ||
      !p.descKeys || !p.sortPass || !p.sortGather || !p.topP || !p.minP)
    return "an earlier lab's Metal kernels did not build: " + pmpp::take_error();
  const NSUInteger w = p.drawRows.threadExecutionWidth;
  if (p.drawRows.maxTotalThreadsPerThreadgroup < 256 || w == 0 || 256 % w != 0)
    return "draw_rows cannot run 256 threads in whole SIMD-groups on this device";
  return "";
}

struct Ws { SampleWorkspace ws; };
static Ws* makeWs(int rows, int vocab, int topK) {
  return new Ws{makeSampleWorkspace(pmpp::metal::device(), (uint32_t)rows, (uint32_t)vocab, topK)};
}
static void freeWs(Ws* w) { delete w; }
static std::string checkWsGuard(Ws*) { return ""; }   // separate buffers: no guard (README)

static std::string runKat(Buf* in, Buf* out, int count) {
  SamplerPipelines p = pipes();
  bool ok = true;
  pmpp::run([&](void* v) { ok = encodePhiloxKat(((pmpp::metal::Pass*)v)->encoder, p.philoxKat, buffer(in), buffer(out), (uint32_t)count); });
  return ok ? "" : "your encodePhiloxKat refused to dispatch";
}
static std::string runRepetition(Buf* logits, Buf* counts, int rows, int vocab, float penalty) {
  SamplerPipelines p = pipes();
  bool ok = true;
  pmpp::run([&](void* v) {
    ok = encodeApplyRepetition(((pmpp::metal::Pass*)v)->encoder, p.applyRepetition, buffer(logits), buffer(counts),
                               (uint32_t)rows, (uint32_t)vocab, penalty);
  });
  return ok ? "" : "your encodeApplyRepetition refused to dispatch";
}
// u from `uIn` when it is not null, else from Philox with seeds, streams, steps.
static std::string runDraw(Buf* sl, Buf* sp, Buf* ci, Buf* n, Buf* seeds, Buf* streams, Buf* steps, Buf* uIn,
                           Buf* tokens, Buf* uOut, int rows, int stride, float invT, int repeat = 1,
                           pmpp::Timing* tm = nullptr) {
  SamplerPipelines p = pipes();
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    for (int i = 0; i < repeat && ok; ++i)
      ok = encodeDrawRows(((pmpp::metal::Pass*)v)->encoder, p.drawRows, buffer(sl), buffer(sp), buffer(ci), buffer(n),
                          bufOrNil(seeds), bufOrNil(streams), bufOrNil(steps), bufOrNil(uIn), buffer(tokens),
                          bufOrNil(uOut), (uint32_t)rows, (uint32_t)stride, invT);
  });
  if (tm) *tm = t;
  return ok ? "" : "your encodeDrawRows refused to dispatch";
}
// One sampling step (repeated `repeat` times).
static std::string runSample(Ws* w, Buf* logits, Buf* counts, Buf* seeds, Buf* streams, Buf* steps, Buf* tokens,
                             Buf* uOut, int rows, int vocab, const Settings& s, int repeat = 1,
                             pmpp::Timing* tm = nullptr) {
  SamplerPipelines p = pipes();
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    for (int i = 0; i < repeat && ok; ++i) {
      ok = encodeSampleRows(enc, p, w->ws, buffer(logits), buffer(counts), buffer(seeds), buffer(streams), buffer(steps),
                            buffer(tokens), bufOrNil(uOut), (uint32_t)rows, (uint32_t)vocab, s.topK, s.topP, s.minP,
                            s.temperature, s.rep, s.freq, s.pres);
    }
  });
  if (tm) *tm = t;
  return ok ? "" : "your encodeSampleRows refused to dispatch";
}
// Lab 9.3's appendTokens: counts[r][tokens[r]] += 1.
static std::string runAppend(Buf* tokens, Buf* counts, int rows, int vocab) {
  SamplerPipelines p = pipes();
  pmpp::run([&](void* v) {
    encodeAppendTokens(((pmpp::metal::Pass*)v)->encoder, p.appendTokens, buffer(tokens), buffer(counts), (uint32_t)rows, (uint32_t)vocab);
  });
  return "";
}
static std::string runStage(Stage st, StageBufs& b, int rows, int vocab, int K, const Settings& s, int repeat,
                            pmpp::Timing* tm) {
  SamplerPipelines p = pipes();
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> e = ((pmpp::metal::Pass*)v)->encoder;
    const uint32_t R = (uint32_t)rows, V = (uint32_t)vocab, k = (uint32_t)K;
    for (int i = 0; i < repeat && ok; ++i) switch (st) {
      case StPenalties:
        ok = encodeCopyRows(e, p.copyRows, buffer(b.logits), buffer(b.W), R * V) &&
             encodeApplyRepetition(e, p.applyRepetition, buffer(b.W), buffer(b.counts), R, V, s.rep);
        if (ok) encodeApplyPenalties(e, p.applyPenalties, buffer(b.W), buffer(b.counts), R, V, s.freq, s.pres);
        break;
      case StSelect: ok = encodeTopKRows(e, p.topkThreshold, p.topkGather, buffer(b.W), buffer(b.candIdx), buffer(b.candVal), buffer(b.selScr), R, V, k); break;
      case StSort: ok = encodeSortRowsDesc(e, p.descKeys, p.sortPass, p.sortGather, buffer(b.candVal), buffer(b.sortedVal), buffer(b.sortedPos), buffer(b.sortScr), R, k); break;
      case StSoftmax: ok = encodeSoftmaxRows(e, p.softmax, buffer(b.sortedVal), buffer(b.probs), R, k, 1.0f); break;
      case StTopP: ok = encodeTopPMask(e, p.topP, buffer(b.probs), buffer(b.mass), buffer(b.keep), buffer(b.kept), R, k, s.topP); break;
      case StMinP: ok = encodeMinPCombine(e, p.minP, buffer(b.sortedVal), buffer(b.kept), buffer(b.n), R, k, s.minP); break;
      case StDraw:
        ok = encodeDrawRows(e, p.drawRows, buffer(b.sortedVal), buffer(b.sortedPos), buffer(b.candIdx), buffer(b.n), buffer(b.seeds),
                            buffer(b.streams), buffer(b.steps), nil, buffer(b.tokens), nil, R, k, 1.0f / s.temperature);
        break;
      default: break;
    }
  });
  if (tm) *tm = t;
  return ok ? "" : std::string("a ") + kStageNames[st] + " encoder refused to dispatch";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
// Yours (work/kernel.cu)
void launchPhiloxKat(const unsigned int* in, unsigned int* out, int count);
void launchCopyRows(const float* in, float* out, size_t n);
void launchApplyRepetition(float* logits, const unsigned int* counts, int rows, int vocab, float penalty);
void launchDrawRows(const float* sortedLogits, const int* sortedPos, const int* candIdx, const int* n,
                    const unsigned long long* seeds, const unsigned long long* streams,
                    const unsigned long long* steps, int* tokens, float* uOut, int rows, int stride, float invT);
void launchDrawRowsU(const float* sortedLogits, const int* sortedPos, const int* candIdx, const int* n,
                     const float* u, int* tokens, int rows, int stride, float invT);
size_t sampleWorkspaceBytes(int rows, int vocab, int topK);
void launchSampleRows(const float* logits, const unsigned int* counts, const unsigned long long* seeds,
                      const unsigned long long* streams, const unsigned long long* steps, int* tokens, float* uOut,
                      void* workspace, int rows, int vocab, int topK, float topP, float minP, float temperature,
                      float repeatPenalty, float frequencyPenalty, float presencePenalty);
// Provided (labs/x1/shared-g11/)
void launchAppendTokens(const int* newTokens, unsigned int* counts, int batch, int vocab);
void launchApplyPenalties(float* logits, const unsigned int* counts, int batch, int vocab, float f, float p);
void launchTopKRows(const float* logits, int* outIdx, float* outVal, unsigned int* scratch, int rows, int cols, int k);
void launchSortRowsDesc(const float* probs, float* sortedProbs, int* sortedIdx, unsigned int* scratch, int rows, int cols);
void launchSoftmaxRows(const float* x, float* y, int rows, int cols, float scale);
void launchTopPMask(const float* probs, float* massBefore, unsigned char* keep, int* kept, int rows, int cols, float p);
void launchMinPCombine(const float* sortedLogits, const int* kept, int* n, int rows, int cols, float minP);

template <class T> static T* P(Buf* b) { return b ? (T*)pmpp::raw(b) : nullptr; }
static std::string pipelineProblem() {
  const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  return 256 <= most ? "" : "this device runs at most " + std::to_string(most) + " threads per block; the lab needs 256";
}

// The workspace: sampleWorkspaceBytes as you compute it, plus a guard the driver checks.
struct Ws { Buf* buf; size_t bytes; };
static Ws* makeWs(int rows, int vocab, int topK) {
  const size_t bytes = sampleWorkspaceBytes(rows, vocab, topK);
  Ws* w = new Ws{pmpp::alloc(bytes + kGuard * 4), bytes};
  std::vector<uint32_t> g(kGuard, kGuardBits);
  std::vector<unsigned char> all(bytes + kGuard * 4, 0xCD);
  std::memcpy(all.data() + bytes, g.data(), kGuard * 4);
  pmpp::upload(w->buf, all.data(), all.size());
  return w;
}
static void freeWs(Ws* w) { pmpp::release(w->buf); delete w; }
static std::string checkWsGuard(Ws* w) {
  std::vector<unsigned char> all(w->bytes + kGuard * 4);
  pmpp::download(w->buf, all.data(), all.size());
  std::vector<uint32_t> g(kGuard);
  std::memcpy(g.data(), all.data() + w->bytes, kGuard * 4);
  for (int i = 0; i < kGuard; ++i)
    if (g[i] != kGuardBits)
      return "wrote past the sampleWorkspaceBytes(...) = " + std::to_string(w->bytes) + " bytes of the workspace";
  return "";
}

static std::string runKat(Buf* in, Buf* out, int count) {
  pmpp::run([&](void*) { launchPhiloxKat(P<unsigned int>(in), P<unsigned int>(out), count); });
  return "";
}
static std::string runRepetition(Buf* logits, Buf* counts, int rows, int vocab, float penalty) {
  pmpp::run([&](void*) { launchApplyRepetition(P<float>(logits), P<unsigned int>(counts), rows, vocab, penalty); });
  return "";
}
static std::string runDraw(Buf* sl, Buf* sp, Buf* ci, Buf* n, Buf* seeds, Buf* streams, Buf* steps, Buf* uIn,
                           Buf* tokens, Buf* uOut, int rows, int stride, float invT, int repeat = 1,
                           pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) {
    for (int i = 0; i < repeat; ++i) {
      if (uIn)
        launchDrawRowsU(P<float>(sl), P<int>(sp), P<int>(ci), P<int>(n), P<float>(uIn), P<int>(tokens), rows, stride, invT);
      else
        launchDrawRows(P<float>(sl), P<int>(sp), P<int>(ci), P<int>(n), P<unsigned long long>(seeds),
                       P<unsigned long long>(streams), P<unsigned long long>(steps), P<int>(tokens), P<float>(uOut), rows,
                       stride, invT);
    }
  });
  if (tm) *tm = t;
  return "";
}
static std::string runSample(Ws* w, Buf* logits, Buf* counts, Buf* seeds, Buf* streams, Buf* steps, Buf* tokens,
                             Buf* uOut, int rows, int vocab, const Settings& s, int repeat = 1,
                             pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) {
    for (int i = 0; i < repeat; ++i) {
      launchSampleRows(P<float>(logits), P<unsigned int>(counts), P<unsigned long long>(seeds),
                       P<unsigned long long>(streams), P<unsigned long long>(steps), P<int>(tokens), P<float>(uOut),
                       pmpp::raw(w->buf), rows, vocab, s.topK, s.topP, s.minP, s.temperature, s.rep, s.freq, s.pres);
    }
  });
  if (tm) *tm = t;
  return "";
}
static std::string runAppend(Buf* tokens, Buf* counts, int rows, int vocab) {
  pmpp::run([&](void*) { launchAppendTokens(P<int>(tokens), P<unsigned int>(counts), rows, vocab); });
  return "";
}
static std::string runStage(Stage st, StageBufs& b, int rows, int vocab, int K, const Settings& s, int repeat,
                            pmpp::Timing* tm) {
  pmpp::Timing t = pmpp::run([&](void*) {
    for (int i = 0; i < repeat; ++i) switch (st) {
      case StPenalties:
        launchCopyRows(P<float>(b.logits), P<float>(b.W), (size_t)rows * vocab);
        launchApplyRepetition(P<float>(b.W), P<unsigned int>(b.counts), rows, vocab, s.rep);
        launchApplyPenalties(P<float>(b.W), P<unsigned int>(b.counts), rows, vocab, s.freq, s.pres);
        break;
      case StSelect: launchTopKRows(P<float>(b.W), P<int>(b.candIdx), P<float>(b.candVal), P<unsigned int>(b.selScr), rows, vocab, K); break;
      case StSort: launchSortRowsDesc(P<float>(b.candVal), P<float>(b.sortedVal), P<int>(b.sortedPos), P<unsigned int>(b.sortScr), rows, K); break;
      case StSoftmax: launchSoftmaxRows(P<float>(b.sortedVal), P<float>(b.probs), rows, K, 1.0f); break;
      case StTopP: launchTopPMask(P<float>(b.probs), P<float>(b.mass), P<unsigned char>(b.keep), P<int>(b.kept), rows, K, s.topP); break;
      case StMinP: launchMinPCombine(P<float>(b.sortedVal), P<int>(b.kept), P<int>(b.n), rows, K, s.minP); break;
      case StDraw:
        launchDrawRows(P<float>(b.sortedVal), P<int>(b.sortedPos), P<int>(b.candIdx), P<int>(b.n), P<unsigned long long>(b.seeds),
                       P<unsigned long long>(b.streams), P<unsigned long long>(b.steps), P<int>(b.tokens), nullptr, rows, K,
                       1.0f / s.temperature);
        break;
      default: break;
    }
  });
  if (tm) *tm = t;
  return "";
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- bits, Philox, u on the host
static uint32_t bitsOf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float fromBits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static std::string hex(uint32_t u) { char s[16]; std::snprintf(s, sizeof s, "0x%08x", u); return s; }
static double ulpOf(double x) {   // the float spacing at |x|
  const float a = (float)std::fabs(x);
  return (double)std::nextafter(a, std::numeric_limits<float>::infinity()) - (double)a;
}
static uint32_t orderedKey(uint32_t u) {   // lab 14.3's key, ascending
  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Random123's Philox4x32-10 (x1-facts A4), the CPU's own copy.
struct Words { uint32_t x, y, z, w; };
static Words hostPhilox(Words c, uint32_t k0, uint32_t k1) {
  for (int r = 0; r < 10; ++r) {
    if (r > 0) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
    const uint64_t p0 = (uint64_t)0xD2511F53u * c.x, p1 = (uint64_t)0xCD9E8D57u * c.z;
    c = Words{(uint32_t)(p1 >> 32) ^ c.y ^ k0, (uint32_t)p1, (uint32_t)(p0 >> 32) ^ c.w ^ k1, (uint32_t)p0};
  }
  return c;
}
static uint32_t hostRowBits(uint64_t seed, uint64_t stream, uint64_t step) {
  return hostPhilox(Words{(uint32_t)step, (uint32_t)(step >> 32), (uint32_t)stream, (uint32_t)(stream >> 32)},
                    (uint32_t)seed, (uint32_t)(seed >> 32)).x;
}
static float hostUniform(uint32_t w) { return (float)(w >> 8) * (1.0f / 16777216.0f); }

// Random123's tests/kat_vectors, 10 rounds (x1-facts A4): counter, key, expected.
struct Kat { uint32_t ctr[4], key[2], want[4]; };
static const Kat kKats[] = {
    {{0, 0, 0, 0}, {0, 0}, {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}},
    {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, {0xffffffffu, 0xffffffffu},
     {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}},
    {{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u},
     {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}},
};

// ---------------------------------------------------------------- data
static uint32_t mix(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}
static double unit(uint32_t h) { return (double)(h >> 8) / 16777216.0; }

// Rows shaped like a language model's logits (ours, synthetic): a tail of small values and a
// head of up to 48 plausible tokens between 4 and 14. Every row also holds an exact tie of two
// unpenalised head tokens, +0.0 and -0.0 among seen tokens, and in odd rows two tokens tied at
// the row's maximum (so greedy decoding meets a tie). Counts: 64 seen tokens per row, half of
// them in the head, counts 1 to 4.
static void makeRows(int rows, int vocab, uint32_t seed, std::vector<float>& x, std::vector<uint32_t>& counts) {
  x.assign((size_t)rows * vocab, 0.0f);
  counts.assign((size_t)rows * vocab, 0u);
  for (int r = 0; r < rows; ++r) {
    float* row = &x[(size_t)r * vocab];
    uint32_t* cnt = &counts[(size_t)r * vocab];
    const uint32_t base = seed * 7919u + (uint32_t)r * 104729u + 1u;
    for (int c = 0; c < vocab; ++c) {
      const uint32_t h = mix(base ^ (uint32_t)c * 2654435761u);
      row[c] = (float)(2.5 * (unit(h) + unit(mix(h + 1)) + unit(mix(h + 2)) - 1.5));
    }
    const int heads = std::min(48, std::max(1, vocab / 4));
    std::vector<int> head;
    for (int j = 0; (int)head.size() < heads && j < 4 * heads; ++j) {
      const int c = (int)(mix(base + 977u * (uint32_t)j) % (uint32_t)vocab);
      if (std::find(head.begin(), head.end(), c) == head.end()) head.push_back(c);
    }
    for (size_t j = 0; j < head.size(); ++j) row[head[j]] = (float)(4.0 + 10.0 * std::sqrt(unit(mix(base + 31u * (uint32_t)j))));
    // Seen tokens: every other head token, then tail tokens.
    for (size_t j = 0; j < head.size(); j += 2) cnt[head[j]] = 1u + mix(base + 5u * (uint32_t)j) % 4u;
    for (int j = 0; j < 40; ++j) cnt[mix(base + 13u * (uint32_t)j + 7u) % (uint32_t)vocab] += 1u + (uint32_t)j % 3u;
    // A tie of two unpenalised head tokens (lowest id first wherever they land).
    if (head.size() >= 4) {
      cnt[head[1]] = cnt[head[3]] = 0u;
      row[head[3]] = row[head[1]];
    }
    // Zeros of both signs among seen tokens: the repetition rule at exactly 0.
    for (int j = 0; j < 4 && j < vocab; ++j) {
      const int c = (int)(mix(base + 101u * (uint32_t)j + 3u) % (uint32_t)vocab);
      if (std::find(head.begin(), head.end(), c) != head.end()) continue;
      row[c] = j % 2 ? -0.0f : 0.0f;
      cnt[c] = 1u;
    }
    // Odd rows: two unseen tokens tied at the maximum.
    if (r % 2 == 1 && vocab >= 2) {
      float m = row[0];
      for (int c = 1; c < vocab; ++c) m = std::max(m, row[c]);
      const int a = (int)(mix(base + 4242u) % (uint32_t)vocab), b = (int)(mix(base + 4243u) % (uint32_t)vocab);
      if (a != b) { row[a] = row[b] = m + 1.0f; cnt[a] = cnt[b] = 0u; }
    }
  }
}

// ---------------------------------------------------------------- the CPU reference sampler
// Penalties in float, in llama.cpp's order (repetition, then frequency and presence, x1-facts
// D2), so that an unseen token keeps its logit bit for bit; everything after that in double.
// Every decision records how far it is from flipping; a "band" says how far float arithmetic on
// a device may move it (README.md, "Which draws must match").
struct Plan {
  bool greedy = false;
  int greedyToken = -1;
  int K = 0, kept = 0, countMinP = 0, n = 0;
  std::vector<int> ids;        // vocabulary ids of the K candidates, largest first, ties by id
  std::vector<float> vals;     // their penalised logits
  std::vector<double> errs;    // how far a device's penalised logit may be from vals
  std::vector<double> cdf;     // inclusive sums of the first n weights
  double total = 0, band = 0;  // band: how far a device's sums may be from cdf
  std::vector<char> close;     // positions in an adjacent pair a device may order differently
  std::string unsure;          // why a decision before the draw is inside its band, or ""
};

static void penalise(const float* x, const uint32_t* c, int vocab, const Settings& s, std::vector<float>& v,
                     std::vector<double>& e) {
  v.resize((size_t)vocab);
  e.assign((size_t)vocab, 0.0);
  const bool rep = s.rep != 1.0f, fp = s.freq != 0.0f || s.pres != 0.0f;
  for (int i = 0; i < vocab; ++i) {
    const float a = x[i];
    float r = a;
    if (rep && c[i] > 0u) r = a <= 0.0f ? a * s.rep : a / s.rep;
    float out = r, sub = 0.0f;
    if (fp) {
      const float f = (float)c[i] * s.freq;
      const float p = (float)(c[i] > 0u) * s.pres;
      sub = f + p;
      out = r - sub;
    }
    v[i] = out;
    // A seen token's value went through a division, a multiplication and a subtraction that a
    // device may round differently (or contract): 8 units in the last place of the largest
    // magnitude involved. An unseen token's value is the logit itself, exactly, everywhere.
    if (c[i] > 0u && (rep || fp))
      e[i] = 8.0 * ulpOf(std::max(std::max(std::fabs((double)a), std::fabs((double)r)),
                                  std::max(std::fabs((double)sub), std::fabs((double)out))));
  }
}

static Plan planRow(const float* x, const uint32_t* c, int vocab, const Settings& s) {
  Plan pl;
  std::vector<float> v;
  std::vector<double> e;
  penalise(x, c, vocab, s, v, e);
  if (s.temperature <= 0.0f) {   // greedy: lab 10.3's argmax, the lowest id on ties
    pl.greedy = true;
    int best = 0;
    for (int i = 1; i < vocab; ++i)
      if (v[i] > v[best]) best = i;
    pl.greedyToken = best;
    for (int i = 0; i < vocab && pl.unsure.empty(); ++i)
      if (i != best && e[i] + e[best] > 0 && (double)v[i] + e[i] >= (double)v[best] - e[best])
        pl.unsure = "argmax: token " + std::to_string(i) + " is within rounding of the maximum";
    return pl;
  }
  const int K = effectiveK(vocab, s.topK);
  pl.K = K;
  std::vector<int> order((size_t)vocab);
  std::iota(order.begin(), order.end(), 0);
  std::vector<uint32_t> key((size_t)vocab);
  for (int i = 0; i < vocab; ++i) key[i] = orderedKey(bitsOf(v[i]));
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key[a] > key[b]; });
  if (K < vocab) {   // can a device's rounding swap a token across the top-k boundary?
    double A = 1e300, B = -1e300, C = 1e300, D = -1e300;
    for (int j = 0; j < vocab; ++j) {
      const int t = order[j];
      if (j < K) { C = std::min(C, v[t] - e[t]); if (e[t] > 0) A = std::min(A, v[t] - e[t]); }
      else { B = std::max(B, v[t] + e[t]); if (e[t] > 0) D = std::max(D, v[t] + e[t]); }
    }
    if (A <= B || C <= D) pl.unsure = "top-k: a penalised logit is within rounding of the k-th";
  }
  for (int j = 0; j < K; ++j) {
    pl.ids.push_back(order[j]);
    pl.vals.push_back(v[order[j]]);
    pl.errs.push_back(e[order[j]]);
  }
  // Softmax at scale 1 and top-p (lab 11.6's rule: keep i when i == 0, p >= 1, or the mass
  // before it is below p).
  std::vector<double> pr((size_t)K);
  double z = 0;
  for (int j = 0; j < K; ++j) z += (pr[j] = std::exp((double)pl.vals[j] - (double)pl.vals[0]));
  std::vector<double> before((size_t)K);
  double m = 0;
  for (int j = 0; j < K; ++j) { before[j] = m; m += pr[j] / z; }
  pl.kept = 0;
  for (int j = 0; j < K; ++j)
    if (j == 0 || s.topP >= 1.0f || before[j] < (double)s.topP) ++pl.kept;
  // Each count is a prefix. Positions whose decision is within rounding of its threshold could
  // go either way on a device, so each count lies in a range; n is certain only if every pair of
  // counts in those ranges gives the same n.
  int keptLo = pl.kept, keptHi = pl.kept;
  if (s.topP < 1.0f) {
    const double band = std::ldexp(512.0 + K / 256.0, -24);
    for (int j = 1; j < K; ++j)
      if (std::fabs(before[j] - (double)s.topP) <= band) { keptLo = std::min(keptLo, j); keptHi = std::max(keptHi, j + 1); }
  }
  // Min-p as a prefix count (minp_g11.cu): logits at least the first plus ln(minP), with
  // ln(minP) in float as the host computes it.
  pl.countMinP = K;
  int minLo = K, minHi = K;
  if (s.minP > 0.0f) {
    const float lnP = std::log(s.minP);
    const double thr = (double)pl.vals[0] + (double)lnP;
    pl.countMinP = 0;
    for (int j = 0; j < K; ++j)
      if ((double)pl.vals[j] >= thr) ++pl.countMinP;
    minLo = minHi = pl.countMinP;
    const double tb = 2.0 * ulpOf(std::max(std::max(std::fabs((double)pl.vals[0]), std::fabs((double)lnP)), std::fabs(thr)));
    for (int j = 1; j < K; ++j)
      if (std::fabs((double)pl.vals[j] - thr) <= tb + pl.errs[j] + pl.errs[0]) { minLo = std::min(minLo, j); minHi = std::max(minHi, j + 1); }
  }
  pl.n = std::max(1, std::min(pl.kept, pl.countMinP));
  if (pl.unsure.empty() && std::max(1, std::min(keptLo, minLo)) != std::max(1, std::min(keptHi, minHi)))
    pl.unsure = keptLo != keptHi ? "top-p: the mass before a survivor is within rounding of p"
                                 : "min-p: a survivor is within rounding of the threshold";
  // Weights in double from the device's own 1 / T (a float); their inclusive sums.
  const double invT = (double)(1.0f / s.temperature);
  double maxE = 0;
  for (int j = 0; j < pl.n; ++j) {
    pl.total += std::exp(((double)pl.vals[j] - (double)pl.vals[0]) * invT);
    pl.cdf.push_back(pl.total);
    maxE = std::max(maxE, pl.errs[j]);
  }
  pl.band = pl.total * (std::ldexp(1.0, -14) + pl.n * std::ldexp(1.0, -22) + 2.0 * invT * (pl.errs[0] + maxE));
  // Adjacent candidates whose order a device's rounding could reverse.
  pl.close.assign((size_t)K, 0);
  for (int j = 0; j + 1 < K && j < pl.n; ++j)
    if (pl.errs[j] + pl.errs[j + 1] > 0 && (double)pl.vals[j] - pl.errs[j] <= (double)pl.vals[j + 1] + pl.errs[j + 1])
      pl.close[j] = pl.close[j + 1] = 1;
  return pl;
}

// The draw: the first position whose inclusive sum is greater than u * total.
struct Draw { int token; bool decisive; std::string why; };
static Draw drawFrom(const Plan& pl, float u) {
  if (pl.greedy) return Draw{pl.greedyToken, pl.unsure.empty(), pl.unsure};
  const double target = (double)u * pl.total;
  const int j = (int)(std::upper_bound(pl.cdf.begin(), pl.cdf.end(), target) - pl.cdf.begin());
  Draw d{pl.ids[std::min(j, pl.n - 1)], true, ""};
  if (!pl.unsure.empty()) { d.decisive = false; d.why = pl.unsure; }
  else if (pl.close[j]) { d.decisive = false; d.why = "the chosen candidate's order is within rounding of its neighbour's"; }
  else if ((j > 0 && target - pl.cdf[j - 1] <= pl.band) || pl.cdf[j] - target <= pl.band) {
    d.decisive = false;
    d.why = "u * total is within rounding of a boundary of the cumulative weights";
  }
  return d;
}

// ---------------------------------------------------------------- 1. Philox known answers
static std::string checkKat() {
  for (const Kat& k : kKats) {   // the driver's own Philox first
    const Words w = hostPhilox(Words{k.ctr[0], k.ctr[1], k.ctr[2], k.ctr[3]}, k.key[0], k.key[1]);
    if (w.x != k.want[0] || w.y != k.want[1] || w.z != k.want[2] || w.w != k.want[3])
      return "the driver's host Philox fails a known answer (a bug in main.cpp, not in your kernel)";
  }
  const int n = (int)(sizeof kKats / sizeof kKats[0]);
  std::vector<uint32_t> in;
  for (const Kat& k : kKats) in.insert(in.end(), {k.ctr[0], k.ctr[1], k.ctr[2], k.ctr[3], k.key[0], k.key[1]});
  Buf* bin = pmpp::alloc(in.size() * 4);
  Buf* bout = pmpp::alloc(((size_t)4 * n + kGuard) * 4);
  pmpp::upload(bin, in.data(), in.size() * 4);
  std::vector<uint32_t> out((size_t)4 * n + kGuard, kGuardBits);
  pmpp::upload(bout, out.data(), out.size() * 4);
  std::string err = runKat(bin, bout, n);
  if (err.empty()) err = pmpp::take_error();
  if (err.empty()) {
    pmpp::download(bout, out.data(), out.size() * 4);
    for (int i = 0; i < n && err.empty(); ++i)
      for (int j = 0; j < 4; ++j)
        if (out[(size_t)4 * i + j] != kKats[i].want[j]) {
          err = "known-answer vector " + std::to_string(i + 1) + " (counter " + hex(kKats[i].ctr[0]) + "..., key " +
                hex(kKats[i].key[0]) + "...): word " + std::to_string(j) + " is " + hex(out[(size_t)4 * i + j]) +
                ", Random123 gives " + hex(kKats[i].want[j]);
          break;
        }
    for (int g = 0; g < kGuard && err.empty(); ++g)
      if (out[(size_t)4 * n + g] != kGuardBits) err = "philoxKat wrote past its 4 x count words";
  }
  pmpp::release(bin); pmpp::release(bout);
  return err;
}

// ---------------------------------------------------------------- 2. u, bit for bit
// One-candidate rows through the draw with Philox: token = the row's id, and uOut = u.
static std::string checkU(int count) {
  std::vector<unsigned long long> seeds((size_t)count), streams((size_t)count), steps((size_t)count);
  for (int i = 0; i < count; ++i) {
    const uint32_t h = mix((uint32_t)i * 2246822519u + 1u);
    seeds[i] = i == 0 ? 0ull : ((unsigned long long)mix(h) << 32 | mix(h + 1)) >> (i % 3 == 1 ? 32 : 0);
    streams[i] = i == 0 ? 0ull : (unsigned long long)(mix(h + 2) % 64u) << (i % 5 == 3 ? 32 : 0);
    steps[i] = i == 0 ? 0ull : (unsigned long long)(mix(h + 3) % 1000u) | (i % 7 == 6 ? 1ull << 33 : 0ull);
  }
  std::vector<float> logits((size_t)count, 1.5f);
  std::vector<int> pos((size_t)count, 0), ids((size_t)count), n((size_t)count, 1);
  for (int i = 0; i < count; ++i) ids[i] = 1000 + i;
  Buf *bl = pmpp::alloc(count * 4), *bp = pmpp::alloc(count * 4), *bi = pmpp::alloc(count * 4), *bn = pmpp::alloc(count * 4);
  Buf *bs = pmpp::alloc(count * 8), *bst = pmpp::alloc(count * 8), *bstep = pmpp::alloc(count * 8);
  Buf *bt = pmpp::alloc((count + kGuard) * 4), *bu = pmpp::alloc((count + kGuard) * 4);
  pmpp::upload(bl, logits.data(), count * 4); pmpp::upload(bp, pos.data(), count * 4);
  pmpp::upload(bi, ids.data(), count * 4); pmpp::upload(bn, n.data(), count * 4);
  pmpp::upload(bs, seeds.data(), count * 8); pmpp::upload(bst, streams.data(), count * 8);
  pmpp::upload(bstep, steps.data(), count * 8);
  std::vector<int> tok((size_t)count + kGuard, kGuardIdx);
  std::vector<uint32_t> ub((size_t)count + kGuard, kGuardBits);
  std::fill(tok.begin(), tok.begin() + count, kUnwrittenToken);
  std::fill(ub.begin(), ub.begin() + count, kUnwrittenU);
  pmpp::upload(bt, tok.data(), tok.size() * 4); pmpp::upload(bu, ub.data(), ub.size() * 4);
  std::string err = runDraw(bl, bp, bi, bn, bs, bst, bstep, nullptr, bt, bu, count, 1, 1.0f);
  if (err.empty()) err = pmpp::take_error();
  if (err.empty()) {
    pmpp::download(bt, tok.data(), tok.size() * 4);
    pmpp::download(bu, ub.data(), ub.size() * 4);
    for (int i = 0; i < count && err.empty(); ++i) {
      const uint32_t want = bitsOf(hostUniform(hostRowBits(seeds[i], streams[i], steps[i])));
      if (ub[i] != want) {
        char b[260];
        std::snprintf(b, sizeof b, "row %d (seed %llu, stream %llu, step %llu): u has the bits %s (%.9g), the CPU's %s (%.9g)%s",
                      i, seeds[i], streams[i], steps[i], hex(ub[i]).c_str(), (double)fromBits(ub[i]), hex(want).c_str(),
                      (double)fromBits(want), ub[i] == kUnwrittenU ? ": never written" : "");
        err = b;
        if (i == 0) err += "; seed 0, stream 0, step 0 must give word 0 = 0x6627e8d5, so u = 0x6627e8 * 2^-24";
      } else if (tok[i] != ids[i]) {
        err = "row " + std::to_string(i) + ": one candidate, so the token must be its id " + std::to_string(ids[i]) +
              ", got " + std::to_string(tok[i]);
      }
    }
    for (int g = 0; g < kGuard && err.empty(); ++g)
      if (tok[count + g] != kGuardIdx || ub[count + g] != kGuardBits) err = "the draw wrote past rows entries of tokens or uOut";
  }
  for (Buf* b : {bl, bp, bi, bn, bs, bst, bstep, bt, bu}) pmpp::release(b);
  return err;
}

// ---------------------------------------------------------------- 3. the draw with planted u
struct DrawCase { std::string what; std::vector<float> logits; int n; float u; int want; };
static std::vector<DrawCase> drawCases(bool small) {
  std::vector<DrawCase> v;
  auto equal = [](int n) { return std::vector<float>((size_t)n, 0.5f); };
  // Equal weights, u = m * 2^-24 with u * n exact in float: the token is floor(u * n).
  const std::vector<int> ns = small ? std::vector<int>{1, 2, 64, 1024} : std::vector<int>{1, 2, 64, 1000, 1024, 2048, 4096};
  for (int n : ns) {
    std::vector<uint32_t> ms = {0u, 1u, (1u << 23), (1u << 24) - 1u};
    for (uint32_t j = 0; j < 6; ++j) ms.push_back(mix(n * 31u + j) & 0xFFFFFFu);
    if (n == 2048) { ms.push_back(1023u << 13); ms.push_back(1024u << 13); }   // the seam of the first chunk of 1024
    if (n == 4096) { ms.push_back(2047u << 12); ms.push_back(2048u << 12); ms.push_back(3071u << 12); }
    for (uint32_t m : ms) {
      const double u = (double)m / 16777216.0, t = u * n;
      if ((double)(float)t != t) continue;                // keep only exact products
      v.push_back({"equal weights, n " + std::to_string(n) + ", u = " + std::to_string(m) + " x 2^-24", equal(n), n,
                   (float)u, (int)std::floor(t)});
    }
  }
  // Zero weights (logit -1e4 against 0.5: exp underflows to exactly 0) are never chosen.
  std::vector<float> z = {0.5f, -1e4f, -1e4f, 0.5f, -1e4f, -1e4f};
  v.push_back({"u = 0, weights 1 0 0 1 0 0", z, 6, 0.0f, 0});
  v.push_back({"u = 0.5, weights 1 0 0 1 0 0 (target 1: the zeros at sum 1 are not above it)", z, 6, 0.5f, 3});
  v.push_back({"u = 1 - 2^-24, weights 1 0 0 1 0 0", z, 6, 1.0f - std::ldexp(1.0f, -24), 3});
  v.push_back({"u = 1 - 2^-24, n 1", {0.5f}, 1, 1.0f - std::ldexp(1.0f, -24), 0});
  // The fallback: u = 1 cannot come from (w >> 8) * 2^-24, but it puts the target at the total,
  // as rounding can: no sum is above it, so the last positive weight wins, not a zero after it.
  v.push_back({"u = 1 (fallback), weights 1 1 1 0 0", {0.5f, 0.5f, 0.5f, -1e4f, -1e4f}, 5, 1.0f, 2});
  v.push_back({"u = 1 (fallback), n 1", {0.5f}, 1, 1.0f, 0});
  if (!small) v.push_back({"u = 1 (fallback), 1500 equal weights then 3 zeros", [] {
                             std::vector<float> l(1503, 0.5f); l[1500] = l[1501] = l[1502] = -1e4f; return l; }(), 1503, 1.0f, 1499});
  return v;
}

static std::string checkDrawCases(bool small) {
  const std::vector<DrawCase> cs = drawCases(small);
  const int rows = (int)cs.size();
  int stride = 1;
  for (const DrawCase& c : cs) stride = std::max(stride, (int)c.logits.size());
  stride += 3;   // room for candidates past n: they hold a huge weight that must never be read
  std::vector<float> l((size_t)rows * stride, 50.0f), u((size_t)rows);
  std::vector<int> pos((size_t)rows * stride), ids((size_t)rows * stride), n((size_t)rows);
  for (int r = 0; r < rows; ++r) {
    std::copy(cs[r].logits.begin(), cs[r].logits.end(), l.begin() + (size_t)r * stride);
    for (int i = 0; i < stride; ++i) { pos[(size_t)r * stride + i] = (i * 7 + 3) % stride; ids[(size_t)r * stride + i] = 100000 + i; }
    n[r] = cs[r].n;
    u[r] = cs[r].u;
  }
  // pos permutes the positions and ids names them: the token for position p is
  // ids[pos[p]] = 100000 + pos[p].
  auto tokenAt = [&](int r, int p) { return ids[(size_t)r * stride + pos[(size_t)r * stride + p]]; };
  Buf *bl = pmpp::alloc(l.size() * 4), *bp = pmpp::alloc(pos.size() * 4), *bi = pmpp::alloc(ids.size() * 4);
  Buf *bn = pmpp::alloc(rows * 4), *bu = pmpp::alloc(rows * 4), *bt = pmpp::alloc((rows + kGuard) * 4);
  pmpp::upload(bl, l.data(), l.size() * 4); pmpp::upload(bp, pos.data(), pos.size() * 4);
  pmpp::upload(bi, ids.data(), ids.size() * 4); pmpp::upload(bn, n.data(), rows * 4); pmpp::upload(bu, u.data(), rows * 4);
  std::vector<int> tok((size_t)rows + kGuard, kGuardIdx);
  std::fill(tok.begin(), tok.begin() + rows, kUnwrittenToken);
  pmpp::upload(bt, tok.data(), tok.size() * 4);
  std::string err = runDraw(bl, bp, bi, bn, nullptr, nullptr, nullptr, bu, bt, nullptr, rows, stride, 1.0f);
  if (err.empty()) err = pmpp::take_error();
  if (err.empty()) {
    pmpp::download(bt, tok.data(), tok.size() * 4);
    for (int r = 0; r < rows && err.empty(); ++r) {
      const int want = tokenAt(r, cs[r].want);
      if (tok[r] == want) continue;
      int gotPos = -1;
      for (int p = 0; p < stride; ++p) if (tokenAt(r, p) == tok[r]) gotPos = p;
      err = cs[r].what + ": expected position " + std::to_string(cs[r].want) + " (token " + std::to_string(want) +
            "), got " + (gotPos >= 0 ? "position " + std::to_string(gotPos) : "token " + std::to_string(tok[r]));
      if (gotPos >= cs[r].n) err += ": past the n survivors (read only the first n[r] candidates)";
      else if (gotPos >= 0 && gotPos == cs[r].want + 1) err += " (the rule is: the first inclusive sum strictly greater than u * total)";
      else if (gotPos >= 0 && gotPos == cs[r].want - 1) err += " (is the sum inclusive?)";
    }
    for (int g = 0; g < kGuard && err.empty(); ++g)
      if (tok[rows + g] != kGuardIdx) err = "the draw wrote past rows entries of tokens";
  }
  for (Buf* b : {bl, bp, bi, bn, bu, bt}) pmpp::release(b);
  return err;
}

// ---------------------------------------------------------------- 4. repetition alone
static std::string checkRepetition(int rows, int vocab, float penalty) {
  std::vector<float> x;
  std::vector<uint32_t> c;
  makeRows(rows, vocab, 99u, x, c);
  for (size_t i = 0; i < x.size(); i += 5) x[i] = -x[i];   // more negatives among seen tokens
  for (size_t i = 0; i < x.size(); i += 3) c[i] = (uint32_t)(i % 4);
  const size_t n = x.size();
  Buf *bl = pmpp::alloc((n + kGuard) * 4), *bc = pmpp::alloc(n * 4);
  std::vector<uint32_t> all(n + kGuard, kGuardBits);
  for (size_t i = 0; i < n; ++i) all[i] = bitsOf(x[i]);
  pmpp::upload(bl, all.data(), all.size() * 4);
  pmpp::upload(bc, c.data(), n * 4);
  std::string err = runRepetition(bl, bc, rows, vocab, penalty);
  if (err.empty()) err = pmpp::take_error();
  if (err.empty()) {
    pmpp::download(bl, all.data(), all.size() * 4);
    for (size_t i = 0; i < n && err.empty(); ++i) {
      const float a = x[i], got = fromBits(all[i]);
      if (c[i] == 0u || penalty == 1.0f) {
        if (all[i] != bitsOf(a))
          err = std::string(c[i] == 0u ? "an unseen token" : "penalty 1") + " must leave the logit bit-identical: logit " +
                std::to_string(i) + " was " + hex(bitsOf(a)) + ", now " + hex(all[i]);
        continue;
      }
      const float want = a <= 0.0f ? a * penalty : a / penalty;
      const bool zero = a == 0.0f;
      if (zero ? got != 0.0f : !(std::fabs((double)got - (double)want) <= 2.0 * ulpOf(want))) {
        char b[240];
        std::snprintf(b, sizeof b, "logit %zu = %.9g, count %u, penalty %g: got %.9g, llama.cpp's rule gives %.9g "
                      "(a logit <= 0 is multiplied by the penalty, a positive one divided)", i, (double)a, c[i],
                      (double)penalty, (double)got, (double)want);
        err = b;
      }
    }
    for (int g = 0; g < kGuard && err.empty(); ++g)
      if (all[n + g] != kGuardBits) err = "applyRepetition wrote past rows x vocab logits";
  }
  pmpp::release(bl); pmpp::release(bc);
  return err;
}

// ---------------------------------------------------------------- 5. the whole sampler
struct Tally {
  long draws = 0, decisive = 0, near = 0;
  uint64_t digest = 1469598103934665603ull;
  std::string firstNear;
  // The core: the decisive draws of the first coreRows rows, coreSeeds seeds and coreSteps steps,
  // hashed in the same order on their own. Rows depend only on their index (makeRows), so a run
  // over more rows, seeds or steps has the same core as a run of exactly the core's shape.
  int coreRows = 0, coreSeeds = 0, coreSteps = 0;
  long coreDecisive = 0;
  uint64_t core = 1469598103934665603ull;
};
static void digestAdd(uint64_t& h, uint64_t v) { for (int i = 0; i < 8; ++i) { h ^= (v >> (8 * i)) & 0xFF; h *= 1099511628211ull; } }

// Runs every (seed, step) of the lists on one batch and compares with the CPU's plans.
static std::string checkSampler(int rows, int vocab, const Settings& s, const std::vector<uint64_t>& seedList,
                                const std::vector<uint64_t>& stepList, uint32_t dataSeed, Tally& tally) {
  std::vector<float> x;
  std::vector<uint32_t> c;
  makeRows(rows, vocab, dataSeed, x, c);
  std::vector<Plan> plans;
  for (int r = 0; r < rows; ++r) plans.push_back(planRow(&x[(size_t)r * vocab], &c[(size_t)r * vocab], vocab, s));
  const size_t n = (size_t)rows * vocab;
  Buf *bl = pmpp::alloc(n * 4), *bc = pmpp::alloc(n * 4);
  Buf *bs = pmpp::alloc(rows * 8), *bst = pmpp::alloc(rows * 8), *bstep = pmpp::alloc(rows * 8);
  Buf *bt = pmpp::alloc((rows + kGuard) * 4), *bu = pmpp::alloc((rows + kGuard) * 4);
  pmpp::upload(bl, x.data(), n * 4);
  pmpp::upload(bc, c.data(), n * 4);
  Ws* ws = makeWs(rows, vocab, s.topK);
  std::vector<unsigned long long> seeds((size_t)rows), streams((size_t)rows), steps((size_t)rows);
  for (int r = 0; r < rows; ++r) streams[r] = 1000ull + (unsigned long long)r * 7919ull;   // one stream per row
  pmpp::upload(bst, streams.data(), rows * 8);
  std::string err;
  for (size_t si = 0; si < seedList.size() && err.empty(); ++si)
    for (size_t ti = 0; ti < stepList.size() && err.empty(); ++ti) {
      for (int r = 0; r < rows; ++r) { seeds[r] = seedList[si] + (uint64_t)r * 0x9E3779B97F4A7C15ull; steps[r] = stepList[ti]; }
      pmpp::upload(bs, seeds.data(), rows * 8);
      pmpp::upload(bstep, steps.data(), rows * 8);
      std::vector<int> tok((size_t)rows + kGuard, kGuardIdx);
      std::vector<uint32_t> ub((size_t)rows + kGuard, kGuardBits);
      std::fill(tok.begin(), tok.begin() + rows, kUnwrittenToken);
      std::fill(ub.begin(), ub.begin() + rows, kUnwrittenU);
      pmpp::upload(bt, tok.data(), tok.size() * 4);
      pmpp::upload(bu, ub.data(), ub.size() * 4);
      err = runSample(ws, bl, bc, bs, bst, bstep, bt, bu, rows, vocab, s);
      if (err.empty()) err = pmpp::take_error();
      if (!err.empty()) break;
      pmpp::download(bt, tok.data(), tok.size() * 4);
      pmpp::download(bu, ub.data(), ub.size() * 4);
      for (int r = 0; r < rows && err.empty(); ++r) {
        const float u = hostUniform(hostRowBits(seeds[r], streams[r], steps[r]));
        const Draw d = drawFrom(plans[r], u);
        const std::string at = "row " + std::to_string(r) + ", seed " + std::to_string(seeds[r]) + ", step " + std::to_string(steps[r]);
        if (s.temperature <= 0.0f) {
          if (ub[r] != kUnwrittenU) err = at + ": greedy decoding draws nothing, so uOut must not be written";
        } else if (ub[r] != bitsOf(u)) {
          err = at + ": uOut has the bits " + hex(ub[r]) + ", the CPU's u is " + hex(bitsOf(u)) +
                (ub[r] == kUnwrittenU ? " (never written)" : "");
        }
        if (!err.empty()) break;
        if (tok[r] < 0 || tok[r] >= vocab) {
          err = at + ": token " + std::to_string(tok[r]) + " is not a vocabulary id" + (tok[r] == kUnwrittenToken ? " (never written)" : "");
          break;
        }
        ++tally.draws;
        if (!d.decisive) {
          ++tally.near;
          if (tally.firstNear.empty()) tally.firstNear = at + ": " + d.why;
          continue;
        }
        ++tally.decisive;
        digestAdd(tally.digest, (uint64_t)seeds[r] ^ ((uint64_t)steps[r] << 40) ^ ((uint64_t)r << 56));
        digestAdd(tally.digest, (uint64_t)(uint32_t)tok[r]);
        if (r < tally.coreRows && (int)si < tally.coreSeeds && (int)ti < tally.coreSteps) {
          ++tally.coreDecisive;
          digestAdd(tally.core, (uint64_t)seeds[r] ^ ((uint64_t)steps[r] << 40) ^ ((uint64_t)r << 56));
          digestAdd(tally.core, (uint64_t)(uint32_t)tok[r]);
        }
        if (tok[r] != d.token) {
          const Plan& p = plans[r];
          err = at + " (" + s.name + "): token " + std::to_string(tok[r]) + ", the CPU's " + std::to_string(d.token);
          if (p.greedy) { err += " (greedy: the largest penalised logit, lowest id on ties)"; break; }
          int gp = -1;
          for (int j = 0; j < p.K; ++j) if (p.ids[j] == tok[r]) gp = j;
          const int want = (int)(std::find(p.ids.begin(), p.ids.end(), d.token) - p.ids.begin());
          char b[200];
          std::snprintf(b, sizeof b, "; the CPU kept %d candidates (top-p %d, min-p %d) and drew position %d with u = %.9g",
                        p.n, p.kept, p.countMinP, want, (double)u);
          err += b;
          if (gp < 0) err += "; your token is not among the CPU's top-k candidates";
          else if (gp >= p.n) err += "; your token is at position " + std::to_string(gp) + ", past the survivors";
          else err += "; yours is at position " + std::to_string(gp);
        }
      }
      for (int g = 0; g < kGuard && err.empty(); ++g)
        if (tok[rows + g] != kGuardIdx || ub[rows + g] != kGuardBits) err = "wrote past rows entries of tokens or uOut";
    }
  if (err.empty()) err = checkWsGuard(ws);
  if (err.empty()) {   // the inputs are read-only
    std::vector<float> x2(n);
    std::vector<uint32_t> c2(n);
    pmpp::download(bl, x2.data(), n * 4);
    pmpp::download(bc, c2.data(), n * 4);
    if (std::memcmp(x2.data(), x.data(), n * 4) != 0) err = "the logits changed: penalise a copy in the workspace";
    else if (c2 != c) err = "the counts changed: they are read-only";
  }
  freeWs(ws);
  for (Buf* b : {bl, bc, bs, bst, bstep, bt, bu}) pmpp::release(b);
  return err;
}

// ---------------------------------------------------------------- 6. a toy decode loop
// 4 rows, 16 steps (on emu 2 rows, 4 steps); logits from a fixed table (makeRows with the step
// as seed), counts kept on the device by lab 9.3's appendTokens after every step, as M7's decode
// loop will. The CPU first finds a seed whose draws are all decisive along its own trajectory.
static std::string checkDecodeLoop(int rows, int steps, int vocab) {
  const Settings s = kDefaults;
  std::vector<std::vector<float>> table;
  for (int t = 0; t < steps; ++t) {
    std::vector<float> x;
    std::vector<uint32_t> unused;
    makeRows(rows, vocab, 5000u + (uint32_t)t, x, unused);
    table.push_back(x);
  }
  auto seedOf = [](uint64_t base, int r) { return base + (uint64_t)r * 0x9E3779B97F4A7C15ull; };
  uint64_t chosen = 0;
  std::vector<int> want;
  bool found = false;
  for (uint64_t base = 1; base <= 64 && !found; ++base) {
    std::vector<uint32_t> counts((size_t)rows * vocab, 0u);
    std::vector<int> toks;
    bool ok = true;
    for (int t = 0; t < steps && ok; ++t)
      for (int r = 0; r < rows && ok; ++r) {
        const Plan pl = planRow(&table[t][(size_t)r * vocab], &counts[(size_t)r * vocab], vocab, s);
        const Draw d = drawFrom(pl, hostUniform(hostRowBits(seedOf(base, r), (uint64_t)r, (uint64_t)t)));
        ok = d.decisive;
        toks.push_back(d.token);
        counts[(size_t)r * vocab + d.token] += 1u;
      }
    if (ok) { found = true; chosen = base; want = toks; }
  }
  if (!found) return "no seed among the first 64 gives only decisive draws (a problem with main.cpp's data, not your kernel)";

  Buf *bl = pmpp::alloc((size_t)rows * vocab * 4), *bc = pmpp::alloc((size_t)rows * vocab * 4);
  Buf *bs = pmpp::alloc(rows * 8), *bst = pmpp::alloc(rows * 8), *bstep = pmpp::alloc(rows * 8), *bt = pmpp::alloc(rows * 4);
  std::vector<uint32_t> zero((size_t)rows * vocab, 0u);
  pmpp::upload(bc, zero.data(), zero.size() * 4);
  std::vector<unsigned long long> seeds((size_t)rows), streams((size_t)rows), stepv((size_t)rows);
  for (int r = 0; r < rows; ++r) { seeds[r] = seedOf(chosen, r); streams[r] = (unsigned long long)r; }
  pmpp::upload(bs, seeds.data(), rows * 8);
  pmpp::upload(bst, streams.data(), rows * 8);
  Ws* ws = makeWs(rows, vocab, s.topK);
  std::string err;
  std::vector<int> got;
  for (int t = 0; t < steps && err.empty(); ++t) {
    pmpp::upload(bl, table[t].data(), table[t].size() * 4);
    std::fill(stepv.begin(), stepv.end(), (unsigned long long)t);
    pmpp::upload(bstep, stepv.data(), rows * 8);
    std::vector<int> tok((size_t)rows, kUnwrittenToken);
    pmpp::upload(bt, tok.data(), rows * 4);
    err = runSample(ws, bl, bc, bs, bst, bstep, bt, nullptr, rows, vocab, s);
    if (err.empty()) err = pmpp::take_error();
    pmpp::download(bt, tok.data(), rows * 4);
    for (int r = 0; r < rows && err.empty(); ++r) {
      got.push_back(tok[r]);
      if (tok[r] < 0 || tok[r] >= vocab)
        err = "step " + std::to_string(t) + ", row " + std::to_string(r) + ": token " + std::to_string(tok[r]) +
              " is not a vocabulary id" + (tok[r] == kUnwrittenToken ? " (never written)" : "");
      else if (tok[r] != want[(size_t)t * rows + r])
        err = "seed " + std::to_string(chosen) + ", step " + std::to_string(t) + ", row " + std::to_string(r) + ": token " +
              std::to_string(tok[r]) + ", the CPU's " + std::to_string(want[(size_t)t * rows + r]) +
              (t > 0 ? " (every earlier step matched, so the counts the penalties read were the same)" : "");
    }
    // The tokens are checked, so lab 9.3's appendTokens can count them on the device.
    if (err.empty()) err = runAppend(bt, bc, rows, vocab);
    if (err.empty()) err = pmpp::take_error();
  }
  if (err.empty()) err = checkWsGuard(ws);
  freeWs(ws);
  for (Buf* b : {bl, bc, bs, bst, bstep, bt}) pmpp::release(b);
  return err;
}

// ---------------------------------------------------------------- tests
static std::vector<int> pick(const std::vector<int>& race, const std::vector<int>& emu, const std::vector<int>& dev) {
  const std::string b = pmpp::backend();
  return b == "race" ? race : b == "emu" ? emu : dev;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string why = pipelineProblem();
  if (!why.empty()) {
    t.check("sampler: kernels usable", [&] { return why; });
    return t.finish();
  }
  const std::string be = pmpp::backend();
  const bool emu = be == "emu", race = be == "race";

  t.check("1. Philox4x32-10: Random123's three known-answer vectors", [] { return checkKat(); });
  const int uCount = emu ? 64 : race ? 32 : 4096;
  t.check("2. u bit for bit for " + std::to_string(uCount) + " (seed, stream, step), seed 0 stream 0 step 0 first",
          [&] { return checkU(uCount); });
  t.check("3. the draw with planted u (equal weights, zeros, seams, u near 1, fallback)",
          [&] { return checkDrawCases(pmpp::is_emu()); });
  for (float pen : {1.1f, 1.0f, 2.0f})
    t.check("4. repetition penalty " + std::to_string(pen).substr(0, 3), [&] { return checkRepetition(2, emu ? 300 : 1000, pen); });

  // 5. The whole sampler. Devices: 64 seeds x 4 steps per shape. The emulators: far fewer, and
  // top-k at most 64 (no whole-row sort), because every launch costs hundreds of thread switches.
  const std::vector<int> rowCounts = pick({2}, {1, 3}, {1, 3, 8});
  const std::vector<int> vocabs = pick({300, 1000}, {1000, 4099}, {1000, 4099, 32000, 151936});
  std::vector<uint64_t> seedList, stepList;
  const int nSeeds = emu ? 1 : race ? 2 : 64, nSteps = emu ? 1 : race ? 2 : 4;
  for (int i = 0; i < nSeeds; ++i) seedList.push_back(0x1234567ull * (uint64_t)(i + 1));
  for (int i = 0; i < nSteps; ++i) stepList.push_back((uint64_t)i * 3);
  const std::vector<Settings> all = settingsList(!emu);   // emu: no whole-row sort
  long draws = 0, near = 0;
  int turn = 0;
  for (int rows : rowCounts)
    for (int vocab : vocabs)
      for (size_t si = 0; si < all.size(); ++si) {
        if (emu && (turn++ % 4) != 0) continue;   // emu: every fourth (shape, setting)
        const Settings& s = all[si];
        t.check("5. " + std::to_string(rows) + " rows x " + std::to_string(vocab) + ", " + s.name, [&] {
          Tally ty;
          std::string err = checkSampler(rows, vocab, s, seedList, stepList, 7u * (uint32_t)vocab + (uint32_t)rows, ty);
          draws += ty.draws;
          near += ty.near;
          if (!err.empty()) return err;
          if (ty.decisive * 2 < ty.draws)
            return "only " + std::to_string(ty.decisive) + " of " + std::to_string(ty.draws) +
                   " draws are decisive: the test proves too little (a problem with main.cpp's data, not your kernel). First: " +
                   ty.firstNear;
          return std::string();
        });
      }
  std::printf("  near-boundary draws, counted and not compared: %ld of %ld\n", near, draws);

  const int loopRows = emu ? 2 : 4, loopSteps = emu ? 4 : 16;
  t.check("6. toy decode loop: " + std::to_string(loopRows) + " rows, " + std::to_string(loopSteps) +
              " steps, counts kept on the device",
          [&] { return checkDecodeLoop(loopRows, loopSteps, emu ? 300 : 1000); });

  // 7. Digests of the decisive draws, compared across backends on the labs page. Devices hash
  // 8 rows x 16 seeds x 8 steps = 1,024 draws, so a backend that differs on one draw in a few
  // hundred changes the digest. The emulators run one thread at a time, far too slowly for that,
  // so they run only the core, 2 rows x 2 seeds x 2 steps, which devices hash separately as well.
  const bool emulated = emu || race;
  const int dRows = emulated ? 2 : 8, nDSeeds = emulated ? 2 : 16, nDSteps = emulated ? 2 : 8;
  t.check("7. tokens digest: " + std::to_string(dRows) + " rows x 1000, llama.cpp defaults, " + std::to_string(nDSeeds) +
              " seeds x " + std::to_string(nDSteps) + " steps" + (emulated ? " (the core only)" : ", and its 2 x 2 x 2 core"),
          [&] {
    Tally ty;
    ty.coreRows = 2;
    ty.coreSeeds = 2;
    ty.coreSteps = 2;
    std::vector<uint64_t> dSeeds, dSteps;
    for (int i = 0; i < nDSeeds; ++i) dSeeds.push_back(11ull * (uint64_t)(i + 1));
    for (int i = 0; i < nDSteps; ++i) dSteps.push_back((uint64_t)i);
    std::string err = checkSampler(dRows, 1000, kDefaults, dSeeds, dSteps, 4242u, ty);
    char d[32], c[32];
    std::snprintf(d, sizeof d, "%016llx", (unsigned long long)ty.digest);
    std::snprintf(c, sizeof c, "%016llx", (unsigned long long)ty.core);
    std::printf("  core_digest %s over %ld decisive draws\n", c, ty.coreDecisive);
    if (!emulated) std::printf("  tokens_digest %s over %ld decisive draws of %ld\n", d, ty.decisive, ty.draws);
    pmpp::Json j;
    j.s("kind", "test").s("lab", a.lab).s("variant", a.variant).s("backend", pmpp::backend())
     .s("case", "tokens digest").b("pass", err.empty()).s("core_digest", c).n("core_draws", (double)ty.coreDecisive);
    if (!emulated) j.s("tokens_digest", d).n("decisive_draws", (double)ty.decisive).n("draws", (double)ty.draws);
    j.emit();
    if (err.empty() && ty.decisive * 2 < ty.draws)
      err = "only " + std::to_string(ty.decisive) + " of " + std::to_string(ty.draws) +
            " draws are decisive: the digest covers too little (a problem with main.cpp's data, not your kernel)";
    return err;
  });
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
static double perCall(int reps, const std::function<std::string(int, pmpp::Timing*)>& run, int* calls, std::string* err) {
  *calls = 1;
  pmpp::Timing tm{0, 0};
  for (;;) {
    *err = run(*calls, &tm);
    if (err->empty()) *err = pmpp::take_error();
    if (!err->empty()) return -1;
    if (tm.best() >= 0.002 || *calls >= 65536) break;
    *calls *= 2;
  }
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    *err = run(*calls, &tm);
    if (err->empty()) *err = pmpp::take_error();
    if (!err->empty()) return -1;
    best = std::min(best, tm.best());
  }
  return best / *calls;
}

// One CPU thread sampling every row in float, as llama.cpp's host samplers do it by default
// (x1-facts D1; top-k by std::partial_sort, ch14-facts D3): penalties, top-k, softmax, top-p,
// min-p, temperature, the draw. Ties by id, so that its tokens can be checked.
struct Cand { int id; float logit; };
static void cpuSampleFloat(const std::vector<float>& x, const std::vector<uint32_t>& c, int rows, int vocab,
                           const Settings& s, const std::vector<float>& u, std::vector<int>& out, std::vector<Cand>& work) {
  out.resize((size_t)rows);
  work.resize((size_t)vocab);
  const int K = effectiveK(vocab, s.topK);
  for (int r = 0; r < rows; ++r) {
    const float* xr = &x[(size_t)r * vocab];
    const uint32_t* cr = &c[(size_t)r * vocab];
    for (int i = 0; i < vocab; ++i) {
      float v = xr[i];
      if (cr[i] > 0u) {
        v = v <= 0.0f ? v * s.rep : v / s.rep;
        v -= (float)cr[i] * s.freq + s.pres;
      }
      work[i] = Cand{i, v};
    }
    auto cmp = [](const Cand& a, const Cand& b) { return a.logit > b.logit || (a.logit == b.logit && a.id < b.id); };
    std::partial_sort(work.begin(), work.begin() + K, work.end(), cmp);
    float sum = 0.0f;
    std::vector<float> p((size_t)K);
    for (int j = 0; j < K; ++j) sum += (p[j] = std::exp(work[j].logit - work[0].logit));
    int kept = 0;
    float mass = 0.0f;
    for (int j = 0; j < K; ++j) {
      if (!(j == 0 || s.topP >= 1.0f || mass < s.topP)) break;
      ++kept;
      mass += p[j] / sum;
    }
    int nMin = K;
    if (s.minP > 0.0f) {
      const float thr = work[0].logit + std::log(s.minP);
      nMin = 0;
      while (nMin < K && work[nMin].logit >= thr) ++nMin;
    }
    const int n = std::max(1, std::min(kept, nMin));
    const float invT = 1.0f / s.temperature;
    float total = 0.0f;
    for (int j = 0; j < n; ++j) total += (p[j] = std::exp((work[j].logit - work[0].logit) * invT));
    const float target = u[r] * total;
    float run = 0.0f;
    int pick = n - 1;
    for (int j = 0; j < n; ++j) {
      run += p[j];
      if (run > target) { pick = j; break; }
    }
    out[r] = work[pick].id;
  }
}

static int bench(const pmpp::Args& a) {
  const std::string why = pipelineProblem();
  if (!why.empty()) { std::printf("sampler: %s\n", why.c_str()); return 1; }
  const double mem = deviceMemory();
  const double budget = mem > 0 ? mem / 4 : 1e9;
  int rc = 0;
  const Settings s = kDefaults;
  Settings whole = kDefaults;
  whole.topK = 0;
  whole.name = "topK 0";

  std::printf("\none sampling step, llama.cpp's defaults with penalties on (top-k 40, top-p 0.95, min-p 0.05, T 0.8,\n"
              "penalties 1.1/0.1/0.1); best of %d runs, each repeated until it lasts at least 2 ms\n", a.reps);
  std::printf("  %4s %7s %6s | %10s %10s | %10s %10s | %8s\n", "rows", "vocab", "top-k", "GPU ms", "tokens/s", "CPU ms",
              "CPU tok/s", "GPU/CPU");
  struct Shape { int rows, vocab; const Settings* s; };
  std::vector<Shape> shapes;
  for (int vocab : {32000, 151936})
    for (int rows : {1, 8, 64}) shapes.push_back({rows, vocab, &s});
  for (int rows : {1, 8}) shapes.push_back({rows, 151936, &whole});
  std::vector<std::string> stageLines;
  for (const Shape& sh : shapes) {
    const int rows = sh.rows, vocab = sh.vocab, K = effectiveK(vocab, sh.s->topK);
    const size_t n = (size_t)rows * vocab;
    const double bytes = (double)n * 12 + (double)rows * K * 33;
    if (bytes > budget) {
      std::printf("  %4d %7d %6d | skipped: larger than a quarter of the device memory reported\n", rows, vocab, sh.s->topK);
      continue;
    }
    std::vector<float> x;
    std::vector<uint32_t> c;
    makeRows(rows, vocab, 77u + (uint32_t)vocab, x, c);
    std::vector<unsigned long long> seeds((size_t)rows), streams((size_t)rows), steps((size_t)rows, 5ull);
    std::vector<float> u((size_t)rows);
    for (int r = 0; r < rows; ++r) {
      seeds[r] = 42ull + (unsigned long long)r;
      streams[r] = (unsigned long long)r;
      u[r] = hostUniform(hostRowBits(seeds[r], streams[r], steps[r]));
    }
    std::vector<Draw> want;
    for (int r = 0; r < rows; ++r)
      want.push_back(drawFrom(planRow(&x[(size_t)r * vocab], &c[(size_t)r * vocab], vocab, *sh.s), u[r]));

    Buf *bl = pmpp::alloc(n * 4), *bc = pmpp::alloc(n * 4), *bs = pmpp::alloc(rows * 8), *bst = pmpp::alloc(rows * 8);
    Buf *bstep = pmpp::alloc(rows * 8), *bt = pmpp::alloc(rows * 4);
    pmpp::upload(bl, x.data(), n * 4); pmpp::upload(bc, c.data(), n * 4); pmpp::upload(bs, seeds.data(), rows * 8);
    pmpp::upload(bst, streams.data(), rows * 8); pmpp::upload(bstep, steps.data(), rows * 8);
    Ws* ws = makeWs(rows, vocab, sh.s->topK);
    int calls = 0;
    std::string err, bad;
    const double gpu = perCall(a.reps, [&](int k, pmpp::Timing* tm) {
      return runSample(ws, bl, bc, bs, bst, bstep, bt, nullptr, rows, vocab, *sh.s, k, tm); }, &calls, &err);
    if (gpu <= 0) bad = "failed: " + err;
    std::vector<int> tok((size_t)rows);
    if (bad.empty()) {   // check what the timed calls produced before reporting their time
      pmpp::download(bt, tok.data(), rows * 4);
      for (int r = 0; r < rows && bad.empty(); ++r)
        if (want[r].decisive && tok[r] != want[r].token) bad = "row " + std::to_string(r) + " drew the wrong token";
    }
    // One CPU thread, float, timed with a steady clock (a host measurement), checked too.
    std::vector<int> cpuTok;
    std::vector<Cand> work;
    cpuSampleFloat(x, c, rows, vocab, *sh.s, u, cpuTok, work);
    double cpu = 1e30;
    for (int r = 0; r < a.reps; ++r) {
      int inner = 0;
      const auto t0 = std::chrono::steady_clock::now();
      double el = 0;
      do {
        cpuSampleFloat(x, c, rows, vocab, *sh.s, u, cpuTok, work);
        ++inner;
        el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      } while (el < 0.002);
      cpu = std::min(cpu, el / inner);
    }
    for (int r = 0; r < rows && bad.empty(); ++r)
      if (want[r].decisive && cpuTok[r] != want[r].token) bad = "the CPU baseline drew a wrong token in row " + std::to_string(r);

    // Each stage alone, on buffers prepared by running the stages once in order.
    double stage[StageCount] = {0};
    if (bad.empty()) {
      StageBufs b{bl, bc, pmpp::alloc(n * 4), pmpp::alloc((size_t)rows * K * 4), pmpp::alloc((size_t)rows * K * 4),
                  pmpp::alloc((size_t)rows * 16), pmpp::alloc((size_t)rows * K * 4), pmpp::alloc((size_t)rows * K * 4),
                  pmpp::alloc((size_t)rows * K * 16), pmpp::alloc((size_t)rows * K * 4), pmpp::alloc((size_t)rows * K * 4),
                  pmpp::alloc((size_t)rows * K), pmpp::alloc((size_t)rows * 4), pmpp::alloc((size_t)rows * 4), bs, bst, bstep,
                  pmpp::alloc((size_t)rows * 4)};
      for (int st = 0; st < StageCount && bad.empty(); ++st) {
        err = runStage((Stage)st, b, rows, vocab, K, *sh.s, 1, nullptr);
        if (err.empty()) err = pmpp::take_error();
        if (!err.empty()) bad = err;
      }
      std::vector<int> st((size_t)rows);
      if (bad.empty()) {
        pmpp::download(b.tokens, st.data(), rows * 4);
        for (int r = 0; r < rows && bad.empty(); ++r)
          if (want[r].decisive && st[r] != want[r].token) bad = "the stages run one by one drew a wrong token in row " + std::to_string(r);
      }
      for (int st2 = 0; st2 < StageCount && bad.empty(); ++st2) {
        int c2 = 0;
        stage[st2] = perCall(a.reps, [&](int k, pmpp::Timing* tm) { return runStage((Stage)st2, b, rows, vocab, K, *sh.s, k, tm); }, &c2, &err);
        if (stage[st2] <= 0) bad = std::string(kStageNames[st2]) + " failed: " + err;
      }
      for (Buf* q : {b.W, b.candIdx, b.candVal, b.selScr, b.sortedVal, b.sortedPos, b.sortScr, b.probs, b.mass, b.keep, b.kept, b.n, b.tokens})
        pmpp::release(q);
    }
    freeWs(ws);
    for (Buf* q : {bl, bc, bs, bst, bstep, bt}) pmpp::release(q);
    if (!bad.empty()) { std::printf("  %4d %7d %6d | %s; not reported\n", rows, vocab, sh.s->topK, bad.c_str()); rc = 1; continue; }

    std::printf("  %4d %7d %6d | %10.4f %10.0f | %10.4f %10.0f | %8.3f\n", rows, vocab, sh.s->topK, gpu * 1e3, rows / gpu,
                cpu * 1e3, rows / cpu, gpu / cpu);
    char line[400];
    int at = std::snprintf(line, sizeof line, "  %4d %7d %6d |", rows, vocab, sh.s->topK);
    for (int st = 0; st < StageCount; ++st) at += std::snprintf(line + at, sizeof line - at, " %9.4f", stage[st] * 1e3);
    stageLines.push_back(line);
    pmpp::Json j = pmpp::bench_row(a, sh.s->topK == 0 ? "sample_step_whole_row" : "sample_step");
    j.n("rows", rows).n("vocab", vocab).n("top_k", sh.s->topK).n("seconds", gpu).n("tokens_per_s", rows / gpu).n("calls", calls)
     .n("gpu_over_cpu", gpu / cpu);
    for (int st = 0; st < StageCount; ++st) j.n(std::string("stage_") + std::to_string(st) + "_s", stage[st]);
    j.emit();
    pmpp::bench_row(a, sh.s->topK == 0 ? "cpu_sample_step_whole_row" : "cpu_sample_step").n("rows", rows).n("vocab", vocab)
        .n("top_k", sh.s->topK).n("seconds", cpu).n("tokens_per_s", rows / cpu).emit();
  }
  std::printf("\neach stage alone, ms per call (stage_0 .. stage_6 in the JSON rows):\n  %4s %7s %6s |", "rows", "vocab", "top-k");
  for (int st = 0; st < StageCount; ++st) std::printf(" %9.9s", kStageNames[st]);
  std::printf("\n");
  for (const std::string& l : stageLines) std::printf("%s\n", l.c_str());
  std::printf("\nGPU ms: one whole launchSampleRows (encodeSampleRows on Metal) per call. Metal reports the command\n"
              "buffer's GPU time; CUDA and HIP report host wall-clock time around the launches and a synchronise.\n"
              "CPU: the same chain in float on one CPU thread (llama.cpp samples on the host by default), timed on\n"
              "the host with a steady clock. GPU/CPU below 1 means the GPU took less time. Every timed result was\n"
              "checked against the CPU reference first. These are measurements of this machine, at these sizes.\n");
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
