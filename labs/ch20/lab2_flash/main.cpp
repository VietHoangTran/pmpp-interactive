// Lab 20.2 driver (milestone M6, part two): tests your causal flash attention forward kernel
// against a CPU reference (../shared-g11/flash_test.hpp, shared with lab 20.3), then times it
// against the unfused pipeline of lab 20.1: S = Q K^T with M3, lab 20.1's reference causal
// softmax, O = P V with M2 (../shared-g11/unfused.hpp). You do not need to edit this file;
// your work goes in work/.
#include "../shared-g11/flash_test.hpp"
#include "../shared-g11/unfused.hpp"

#include <memory>
using pmpp::Buf;

// ---- the provided reference softmax (lab 20.1's solution, renamed; see unfused.cmake) ----
#if defined(PMPP_CH20_HAVE_SOFTMAX_REF)
#if defined(PMPP_BACKEND_METAL)
bool encodeCausalSoftmaxRef(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> S, id<MTLBuffer> P,
                            id<MTLBuffer> L, uint32_t batch, uint32_t nq, uint32_t nk, float scale);
static id<MTLComputePipelineState> softmaxPipeline() {
  return pmpp::metal::pipeline(PMPP_CH20_SOFTMAX_REF_METAL, "causal_softmax");
}
static std::string softmaxProblem() {
  return unfused::pipelineProblem(softmaxPipeline(), 256, "lab 20.1's reference causal_softmax");
}
static bool encodeSoftmax(void* v, unfused::Problem& pr, float scale) {
  bool ok = true;
  for (auto& x : pr.h)
    ok = ok && encodeCausalSoftmaxRef(((pmpp::metal::Pass*)v)->encoder, softmaxPipeline(), pmpp::metal::buffer(x.S),
                                      pmpp::metal::buffer(x.P), pmpp::metal::buffer(x.L), 1, (uint32_t)pr.n,
                                      (uint32_t)pr.n, scale);
  return ok;
}
#else
void launchCausalSoftmaxRef(const float* S, float* P, float* L, int batch, int nq, int nk, float scale);
static std::string softmaxProblem() { return ""; }
static bool encodeSoftmax(void*, unfused::Problem& pr, float scale) {
  for (auto& x : pr.h) launchCausalSoftmaxRef(unfused::f(x.S), unfused::f(x.P), unfused::f(x.L), 1, pr.n, pr.n, scale);
  return true;
}
#endif
#else
static std::string softmaxProblem() { return "lab 20.1's solution was not found when the lab was configured"; }
static bool encodeSoftmax(void*, unfused::Problem&, float) { return false; }
#endif

// The whole unfused pipeline, `calls` times, in one timed run: per head S = Q K^T (M3), then
// the causal softmax of every head (lab 20.1's reference), then O = P V (M2). Each stage
// writes its own buffers, so the S and P of every head exist in device memory at once.
static std::string runUnfused(unfused::Problem& pr, float scale, int calls, pmpp::Timing* tm) {
  for (const std::string& why : {unfused::sProblem(), softmaxProblem(), unfused::oProblem()})
    if (!why.empty()) return why;
  bool ok = true;
  *tm = pmpp::run([&](void* v) {
    for (int c = 0; c < calls; ++c) {
      unfused::encodeS(unfused::pass(v), pr);
      ok = ok && encodeSoftmax(v, pr, scale);
      unfused::encodeO(unfused::pass(v), pr);
    }
  });
  return ok ? "" : "lab 20.1's reference launch.mm refused to dispatch";
}

// Spot-checks rows of every head of the flash output, with the tests' tolerances.
static std::string checkFlashRows(int heads, int n, const std::vector<float>& q, const std::vector<float>& k,
                                  const std::vector<float>& v, Buf* O, Buf* L, float scale, int bc) {
  const size_t nd = (size_t)n * flash::kD;
  std::vector<float> o = unfused::fetch(O, (size_t)heads * nd), l = unfused::fetch(L, (size_t)heads * n);
  flash::RowRef ref;
  for (int h = 0; h < heads; ++h)
    for (int r : unfused::spotRows(n)) {
      flash::refRow(&q[h * nd], &k[h * nd], &v[h * nd], r, n, scale, bc, ref);
      for (int x = 0; x < flash::kD; ++x)
        if (!(std::fabs((double)o[h * nd + (size_t)r * flash::kD + x] - ref.o[x]) <= 2 * ref.e * ref.omag[x] + 1e-7))
          return "flash O, head " + std::to_string(h) + ", row " + std::to_string(r) + " wrong";
      if (!(std::fabs((double)l[(size_t)h * n + r] - ref.L) <= 2 * ref.e + 1e-5 + std::ldexp(1.0, -22) * std::fabs(ref.L)))
        return "flash L, head " + std::to_string(h) + ", row " + std::to_string(r) + " wrong";
    }
  return "";
}

static int tests(const pmpp::Args& a) { return flash::runTests(a); }

// ---------------------------------------------------------------- benchmark
static int bench(const pmpp::Args& a) {
  const int heads = 8, d = flash::kD;
  const float scale = 1.0f / std::sqrt((float)d);
  flash::Kernel& kn = flash::kernel();
  if (!kn.why.empty()) { std::printf("flash attention: %s\n", kn.why.c_str()); return 1; }
  const double mem = flash::deviceMemory();
  const double budget = mem > 0 ? mem / 4 : 1e9;   // bytes one size may take
  std::string unfusedWhy;
  for (const std::string& why : {unfused::sProblem(), softmaxProblem(), unfused::oProblem()})
    if (unfusedWhy.empty() && !why.empty()) unfusedWhy = why;
  if (!unfusedWhy.empty()) std::printf("unfused pipeline absent: %s\n", unfusedWhy.c_str());

  std::printf("\ncausal attention forward, batch 1, %d heads, d = %d, scale 1/sqrt(d); flash with B_c = %d; best of %d "
              "timed runs\n", heads, d, kn.bc, a.reps);
  std::printf("  %5s  %10s %10s  %10s  %10s %10s  %14s\n", "n", "flash ms", "unfused ms", "flash /", "flash", "unfused",
              "unfused S and P");
  std::printf("  %5s  %10s %10s  %10s  %10s %10s  %14s\n", "", "", "", "unfused", "GFLOP/s", "GFLOP/s", "bytes");
  int rc = 0;
  for (int n : {512, 1024, 2048, 4096}) {
    const double flashBytes = heads * (4.0 * n * d + n) * 4;
    const bool unfusedFits = unfused::Problem::totalBytes(heads, n) + flashBytes <= budget;
    if (flashBytes > budget) {
      std::printf("  %5d  skipped: needs %.2f GB, more than a quarter of the device memory reported\n", n, flashBytes / 1e9);
      continue;
    }
    // Causal attention's useful work: the visible half of two n x n x d products, per head.
    const double flop = heads * 2.0 * n * (double)n * d;

    // The same data for both paths: the unfused problem's per-head Q, K, V, laid out
    // [1 x heads x n x d] for the flash kernel. Without room for the unfused buffers, flash
    // gets data of its own.
    std::unique_ptr<unfused::Problem> pr;
    if (unfusedFits) pr.reset(new unfused::Problem(heads, n, 700u + (uint32_t)n));
    const size_t nd = (size_t)n * d;
    std::vector<float> q(heads * nd), k(heads * nd), v(heads * nd);
    if (pr) {
      for (int h = 0; h < heads; ++h) {
        std::copy(pr->q[h].begin(), pr->q[h].end(), q.begin() + h * nd);
        std::copy(pr->k[h].begin(), pr->k[h].end(), k.begin() + h * nd);
        std::copy(pr->v[h].begin(), pr->v[h].end(), v.begin() + h * nd);
      }
    } else {
      q = pmpp::random_floats(heads * nd, 71u); k = pmpp::random_floats(heads * nd, 72u); v = pmpp::random_floats(heads * nd, 73u);
    }
    Buf *Q = pmpp::alloc(heads * nd * 4), *K = pmpp::alloc(heads * nd * 4), *V = pmpp::alloc(heads * nd * 4),
        *O = pmpp::alloc(heads * nd * 4), *L = pmpp::alloc((size_t)heads * n * 4);
    pmpp::upload(Q, q.data(), q.size() * 4);
    pmpp::upload(K, k.data(), k.size() * 4);
    pmpp::upload(V, v.data(), v.size() * 4);

    int calls = 0;
    std::string err;
    double sF = unfused::perCall(a.reps, [&](int c, pmpp::Timing* tm) { return flash::run(Q, K, V, O, L, 1, heads, n, scale, c, tm); },
                                 &calls, &err);
    if (sF > 0) err = checkFlashRows(heads, n, q, k, v, O, L, scale, kn.bc);
    if (!err.empty()) { std::printf("  %5d  flash: %s; not reported\n", n, err.c_str()); rc = 1; sF = -1; }

    double sU = -1;
    if (unfusedWhy.empty() && pr) {
      sU = unfused::perCall(a.reps, [&](int c, pmpp::Timing* tm) { return runUnfused(*pr, scale, c, tm); }, &calls, &err);
      if (sU > 0) err = unfused::checkS(*pr);
      if (err.empty() && sU > 0) err = unfused::checkP(*pr, scale);
      if (err.empty() && sU > 0) err = unfused::checkO(*pr);
      if (!err.empty()) { std::printf("  %5d  unfused: %s; not reported\n", n, err.c_str()); rc = 1; sU = -1; }
    } else if (!unfusedFits) {
      std::printf("  %5d  unfused: skipped, its S and P need %.2f GB, more than a quarter of the device memory reported\n",
                  n, unfused::Problem::scoreBytes(heads, n) / 1e9);
    }
    pmpp::release(Q); pmpp::release(K); pmpp::release(V); pmpp::release(O); pmpp::release(L);
    if (sF <= 0) continue;

    const double scoreBytes = unfused::Problem::scoreBytes(heads, n);
    if (sU > 0)
      std::printf("  %5d  %10.3f %10.3f  %10.2f  %10.1f %10.1f  %14.0f\n", n, sF * 1e3, sU * 1e3, sF / sU,
                  flop / sF / 1e9, flop / sU / 1e9, scoreBytes);
    else
      std::printf("  %5d  %10.3f %10s  %10s  %10.1f %10s  %14.0f\n", n, sF * 1e3, "-", "-", flop / sF / 1e9, "-",
                  scoreBytes);
    pmpp::Json j = pmpp::bench_row(a, "attention");
    j.n("n", n).n("heads", heads).n("d", d).n("b_c", kn.bc).n("flash_seconds", sF).n("flash_gflops", flop / sF / 1e9)
        .n("unfused_score_bytes", scoreBytes);
    if (sU > 0) j.n("unfused_seconds", sU).n("unfused_gflops", flop / sU / 1e9).n("flash_over_unfused", sF / sU);
    j.emit();
  }
  std::printf("\nGFLOP/s counts causal attention's useful work, 2 n^2 d per head: half of the 4 n^2 d of the two\n"
              "full products, since half of S is masked. The unfused path computes all of S and all of P V anyway.\n"
              "\"flash / unfused\" below 1 means flash attention is faster. The unfused path is lab 20.1's pipeline:\n"
              "M3 for S, lab 20.1's reference softmax, M2 for O, every head in buffers of its own; \"S and P bytes\"\n"
              "is the device memory those two take, which flash attention never allocates.\n"
              "Each time is the best of %d timed runs of one call, each run repeating the call until it lasts at\n"
              "least 2 ms; both outputs were spot-checked (6 rows of every head) before their times were reported.\n"
              "These are measurements of this machine, at these sizes.\n", a.reps);
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
