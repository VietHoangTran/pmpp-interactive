// Lab 20.1 driver (milestone M6, part one): tests your causal mask-and-softmax kernel
// against a CPU reference, then times it inside the unfused attention pipeline, between the
// project's M3 (S = Q K^T) and M2 (O = P V) kernels (../shared-g11/unfused.hpp). You do not
// need to edit this file; your work goes in work/.
#include <cstring>
#include <limits>

#include "../shared-g11/unfused.hpp"
using pmpp::Buf;

static const int kBlock = 256;   // threads per block (BLOCK_SIZE in kernel.cu, kBlock in launch.mm)

// One launch's buffers: S in, P and L out.
struct Job { Buf *S, *P, *L; };

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
bool encodeCausalSoftmax(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> S, id<MTLBuffer> P,
                         id<MTLBuffer> L, uint32_t batch, uint32_t nq, uint32_t nk, float scale);
using pmpp::metal::buffer;

static id<MTLComputePipelineState> pso() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "causal_softmax"); }

// Why this pipeline cannot run the kernel as the lab specifies it, or "". The driver's own
// copy of the rule a correct launch.mm applies (lab 10.3's): 256 threads per threadgroup, in
// whole SIMD-groups, and no more SIMD-groups than a SIMD-group has lanes.
static std::string pipelineProblem() {
  id<MTLComputePipelineState> p = pso();
  if (!p) return "the Metal kernel did not build: " + pmpp::take_error();
  char b[200];
  if ((NSUInteger)kBlock > p.maxTotalThreadsPerThreadgroup) {
    std::snprintf(b, sizeof b, "this pipeline runs at most %lu threads per threadgroup; the lab needs %d",
                  (unsigned long)p.maxTotalThreadsPerThreadgroup, kBlock);
    return b;
  }
  const NSUInteger w = p.threadExecutionWidth;
  if (w == 0 || kBlock % w != 0 || (NSUInteger)kBlock / w > w) {
    std::snprintf(b, sizeof b, "SIMD-group width %lu: %d threads are not 1 to %lu whole SIMD-groups",
                  (unsigned long)w, kBlock, (unsigned long)w);
    return b;
  }
  return "";
}

// Enqueues `repeat` rounds of one launch per job, in one timed run. "" when it ran.
static std::string launch(const std::vector<Job>& jobs, int batch, int nq, int nk, float scale, int repeat = 1,
                          pmpp::Timing* tm = nullptr) {
  std::string why = pipelineProblem();
  if (!why.empty()) return why;
  id<MTLComputePipelineState> p = pso();
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    for (int i = 0; i < repeat && ok; ++i)
      for (const Job& j : jobs)
        ok = ok && encodeCausalSoftmax(enc, p, buffer(j.S), buffer(j.P), buffer(j.L), (uint32_t)batch, (uint32_t)nq,
                                       (uint32_t)nk, scale);
  });
  if (tm) *tm = t;
  if (!ok) return "your launch.mm refused to dispatch, although this pipeline can run 256 threads per threadgroup";
  return "";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchCausalSoftmax(const float* S, float* P, float* L, int batch, int nq, int nk, float scale);
static float* f(Buf* b) { return (float*)pmpp::raw(b); }

static std::string pipelineProblem() {
  const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  return kBlock <= most ? "" : "this device runs at most " + std::to_string(most) + " threads per block; the lab needs 256";
}
static std::string launch(const std::vector<Job>& jobs, int batch, int nq, int nk, float scale, int repeat = 1,
                          pmpp::Timing* tm = nullptr) {
  std::string why = pipelineProblem();
  if (!why.empty()) return why;
  pmpp::Timing t = pmpp::run([&](void*) {
    for (int i = 0; i < repeat; ++i)
      for (const Job& j : jobs) launchCausalSoftmax(f(j.S), f(j.P), f(j.L), batch, nq, nk, scale);
  });
  if (tm) *tm = t;
  return "";
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- test data
// Each row of a batch follows one recipe; consecutive rows (across items too) follow
// consecutive recipes, so a kernel that reads the wrong row or the wrong item is caught.
enum Recipe { Uniform, PlusNinety, MinusNinety, MaxLast, HugeMasked, AllEqual, kRecipes };
static const char* recipeName(int r) {
  switch (r) {
    case Uniform: return "scores uniform in [-8, 8]";
    case PlusNinety: return "scores offset by +90";
    case MinusNinety: return "scores offset by -90";
    case MaxLast: return "the row's largest score at its last visible column";
    case HugeMasked: return "1e30 in every masked column";
    default: return "every score equal";
  }
}

static void fillRow(int recipe, float* row, int nk, int visible, uint32_t seed) {
  std::vector<float> u = pmpp::random_floats((size_t)nk, seed);   // uniform in [-1, 1]
  for (int c = 0; c < nk; ++c) {
    switch (recipe) {
      case PlusNinety: row[c] = 8.0f * u[c] + 90.0f; break;
      case MinusNinety: row[c] = 8.0f * u[c] - 90.0f; break;
      case HugeMasked: row[c] = c < visible ? 8.0f * u[c] : 1e30f; break;
      case AllEqual: row[c] = 3.25f; break;
      default: row[c] = 8.0f * u[c]; break;   // Uniform, and the background of MaxLast
    }
  }
  if (recipe == MaxLast) row[visible - 1] = 9.0f;   // above every other score of the row
}

// ---------------------------------------------------------------- one launch, checked
static const int kGuard = 64;                    // canaries past the end of P and of L
static const float kGuardF = -12345.5f;          // what they hold before the launch
static const float kNaN = std::numeric_limits<float>::quiet_NaN();

static std::string fmtShape(int batch, int nq, int nk, float scale) {
  char b[120];
  std::snprintf(b, sizeof b, "batch %d, nq %d, nk %d, scale %g", batch, nq, nk, scale);
  return b;
}

// Builds a batch whose global row g follows recipe (start + g) % kRecipes, runs your kernel,
// and checks every row, every masked zero and every canary.
static std::string checkCase(int batch, int nq, int nk, float scale, int start) {
  const size_t total = (size_t)batch * nq * nk, rows = (size_t)batch * nq;
  std::vector<float> s(total);
  std::vector<int> recipe(rows);
  for (size_t g = 0; g < rows; ++g) {
    const int r = (int)(g % nq);
    recipe[g] = (int)((start + g) % kRecipes);
    fillRow(recipe[g], &s[g * nk], nk, r + (nk - nq) + 1, 7919u * (uint32_t)nk + 104729u * (uint32_t)(start + g) + 3u);
  }
  Buf* S = pmpp::alloc(total * 4);
  Buf* P = pmpp::alloc((total + kGuard) * 4);
  Buf* L = pmpp::alloc((rows + kGuard) * 4);
  pmpp::upload(S, s.data(), total * 4);
  std::vector<float> p(total + kGuard, kGuardF), l(rows + kGuard, kGuardF);
  std::fill(p.begin(), p.begin() + total, kNaN);   // an element nobody writes shows as NaN
  std::fill(l.begin(), l.begin() + rows, kNaN);
  pmpp::upload(P, p.data(), p.size() * 4);
  pmpp::upload(L, l.data(), l.size() * 4);

  std::string err = launch({{S, P, L}}, batch, nq, nk, scale);
  if (err.empty()) {
    pmpp::download(P, p.data(), p.size() * 4);
    pmpp::download(L, l.data(), l.size() * 4);
    for (size_t g = 0; g < rows && err.empty(); ++g) {
      const int r = (int)(g % nq), item = (int)(g / nq), visible = r + (nk - nq) + 1;
      std::string e = causal::checkRow(&s[g * nk], &p[g * nk], l[g], visible, nk, scale);
      if (e.empty()) continue;
      err = fmtShape(batch, nq, nk, scale) + "; item " + std::to_string(item) + ", row " + std::to_string(r) + " (" +
            recipeName(recipe[g]) + "): " + e;
      if (recipe[g] == HugeMasked) err += ". This row holds 1e30 in its masked columns: was a masked score read?";
    }
    for (size_t i = total; i < total + kGuard && err.empty(); ++i)
      if (p[i] != kGuardF) err = "wrote P[" + std::to_string(i) + "], past the end of P";
    for (size_t i = rows; i < rows + kGuard && err.empty(); ++i)
      if (l[i] != kGuardF) err = "wrote L[" + std::to_string(i) + "], past the end of L";
  }
  pmpp::release(S); pmpp::release(P); pmpp::release(L);
  return err;
}

// Runs checkCase from enough starting recipes that every recipe appears in every shape.
static std::string checkShape(int batch, int nq, int nk, float scale) {
  for (int start = 0; start < kRecipes; start += batch * nq) {
    std::string e = checkCase(batch, nq, nk, scale, start);
    if (!e.empty()) return e;
  }
  return "";
}

// ---------------------------------------------------------------- tests
static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string b = pmpp::backend();
  struct Shape { int nq, nk; };
  std::vector<int> batches;
  std::vector<Shape> shapes;
  std::vector<float> scales = {1.0f, 1.0f / std::sqrt(128.0f)};
  // The emulators run every block's 256 threads one by one (emu, three orders) or as OS
  // threads under ThreadSanitizer (race): a fraction of a second per block. Their shapes keep
  // what matters to one row's logic (a decode row, the 256-column coarsening boundary, a
  // square item, nq < nk) with few rows.
  if (b == "race") {
    batches = {1};
    shapes = {{1, 1}, {1, 100}, {7, 100}, {5, 5}};
    scales = {1.0f};
  } else if (b == "emu") {
    batches = {1, 2};
    shapes = {{1, 1}, {1, 2}, {1, 300}, {7, 100}, {9, 9}, {4, 260}};
    scales = {1.0f / std::sqrt(128.0f)};
  } else {
    batches = {1, 3};
    shapes = {{1, 1},     {1, 2},     {1, 4097},  {7, 100},   {31, 31},     {32, 32},
              {33, 33},   {255, 255}, {256, 256}, {257, 257}, {1000, 1000}, {2048, 2048}};
  }

  std::string why = pipelineProblem();
  if (!why.empty()) {
    t.check("causal softmax: kernel usable", [&] { return why; });
    return t.finish();
  }
  for (const Shape& s : shapes)
    for (int batch : batches) {
      char name[120];
      std::snprintf(name, sizeof name, "causal softmax, batch %d, nq %d, nk %d%s", batch, s.nq, s.nk,
                    s.nq == 1 ? "  (one decode step)" : s.nq == s.nk ? "  (square, the book's case)" : "");
      t.check(name, [&] {
        for (float scale : scales) {
          std::string e = checkShape(batch, s.nq, s.nk, scale);
          if (!e.empty()) return e;
        }
        return std::string();
      });
    }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// The unfused attention pipeline at 8 heads, d = 128: S = Q K^T (M3), your causal softmax,
// O = P V (M2), each stage timed on its own and each stage's output checked before its
// time is reported.
static int bench(const pmpp::Args& a) {
  const int heads = 8, d = unfused::kD;
  const float scale = 1.0f / std::sqrt((float)d);
  const double mem = deviceMemory();
  const double budget = mem > 0 ? mem / 4 : 1e9;   // bytes one size may take
  std::string why = pipelineProblem();
  if (!why.empty()) { std::printf("causal softmax: %s\n", why.c_str()); return 1; }
  const std::string sWhy = unfused::sProblem(), oWhy = unfused::oProblem();
  if (!sWhy.empty()) std::printf("S = Q K^T stage absent: %s\n", sWhy.c_str());
  if (!oWhy.empty()) std::printf("O = P V stage absent: %s\n", oWhy.c_str());

  std::printf("\nunfused attention, %d heads, d = %d, causal, scale 1/sqrt(d); best of %d timed runs per stage\n",
              heads, d, a.reps);
  std::printf("  %5s  %10s %10s %10s %10s  %8s  %10s  %12s\n", "N", "S ms", "softmax ms", "O ms", "total ms",
              "softmax", "softmax", "S and P");
  std::printf("  %5s  %10s %10s %10s %10s  %8s  %10s  %12s\n", "", "(M3)", "(yours)", "(M2)", "", "share",
              "GB/s", "bytes");
  int rc = 0;
  for (int n : {1024, 2048, 4096}) {
    if (unfused::Problem::totalBytes(heads, n) > budget) {
      std::printf("  %5d  skipped: needs %.2f GB, more than a quarter of the device memory reported\n", n,
                  unfused::Problem::totalBytes(heads, n) / 1e9);
      continue;
    }
    unfused::Problem pr(heads, n, 900u + (uint32_t)n);
    std::vector<Job> jobs;
    for (auto& x : pr.h) jobs.push_back({x.S, x.P, x.L});
    int calls = 0;
    std::string err;

    // Stage 1: S = Q K^T. Without M3, S is filled with stand-in scores so that the softmax
    // still has something to read; its time does not depend on the values.
    double sS = -1;
    if (sWhy.empty()) {
      sS = unfused::perCall(a.reps, [&](int c, pmpp::Timing* tm) { return unfused::runS(pr, c, tm); }, &calls, &err);
      if (sS > 0) err = unfused::checkS(pr);
      if (!err.empty()) { std::printf("  %5d  S stage: %s; not reported\n", n, err.c_str()); rc = 1; sS = -1; }
    } else {
      for (int i = 0; i < heads; ++i) {
        std::vector<float> fake = pmpp::random_floats((size_t)n * n, 5000u + (uint32_t)i);
        for (float& x : fake) x *= 4.0f;
        pmpp::upload(pr.h[i].S, fake.data(), fake.size() * 4);
      }
    }

    // Stage 2: your causal softmax, one launch per head (batch 1, nq = nk = n).
    double sP = unfused::perCall(a.reps, [&](int c, pmpp::Timing* tm) { return launch(jobs, 1, n, n, scale, c, tm); },
                                 &calls, &err);
    if (sP > 0) err = unfused::checkP(pr, scale);
    if (!err.empty()) { std::printf("  %5d  softmax stage: %s; not reported\n", n, err.c_str()); rc = 1; continue; }

    // Stage 3: O = P V.
    double sO = -1;
    if (oWhy.empty()) {
      sO = unfused::perCall(a.reps, [&](int c, pmpp::Timing* tm) { return unfused::runO(pr, c, tm); }, &calls, &err);
      if (sO > 0) err = unfused::checkO(pr);
      if (!err.empty()) { std::printf("  %5d  O stage: %s; not reported\n", n, err.c_str()); rc = 1; sO = -1; }
    }

    // Minimum traffic of the softmax: read the visible half of S once, write all of P once.
    const double minBytes = heads * (4.0 * n * (n + 1.0) / 2.0 + 4.0 * n * (double)n);
    const double gbps = minBytes / sP / 1e9;
    const double scoreBytes = unfused::Problem::scoreBytes(heads, n);
    auto ms = [](double s) { char b[32]; if (s > 0) std::snprintf(b, sizeof b, "%10.3f", s * 1e3); else std::snprintf(b, sizeof b, "%10s", "-"); return std::string(b); };
    const bool all = sS > 0 && sO > 0;
    const double total = all ? sS + sP + sO : -1;
    char share[32];
    if (all) std::snprintf(share, sizeof share, "%7.1f%%", 100.0 * sP / total);
    else std::snprintf(share, sizeof share, "%8s", "-");
    std::printf("  %5d  %s %s %s %s  %s  %10.1f  %12.0f\n", n, ms(sS).c_str(), ms(sP).c_str(), ms(sO).c_str(),
                ms(total).c_str(), share, gbps, scoreBytes);
    pmpp::Json j = pmpp::bench_row(a, "unfused_attention");
    j.n("n", n).n("heads", heads).n("d", d).n("softmax_seconds", sP).n("softmax_gb_per_s", gbps)
        .n("score_bytes", scoreBytes);
    if (sS > 0) j.n("s_seconds", sS);
    if (sO > 0) j.n("o_seconds", sO);
    if (all) j.n("total_seconds", total).n("softmax_share", sP / total);
    j.emit();
  }
  std::printf("\nEach stage runs once per head (8 launches; every head has its own buffers). Each time is per call\n"
              "of the stage: the best of %d timed runs, each repeating the stage until it lasts at least 2 ms.\n"
              "Softmax GB/s counts its minimum traffic: the visible half of S read once and all of P written once.\n"
              "Your kernel reads the visible half three times, so it moves more than it is credited with.\n"
              "\"S and P bytes\" is what the unfused path keeps in device memory for the scores of all heads.\n"
              "Every stage's output was spot-checked (6 rows of every head) before its time was reported.\n"
              "These are measurements of this machine, at these sizes.\n", a.reps);
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
