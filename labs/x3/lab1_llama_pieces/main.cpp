// Lab X3.1 driver (milestone M10's missing pieces): tests your RMSNorm, RoPE and SwiGLU against
// CPU references in double, tests the port of part B for its three defects, and benchmarks
// part A. You do not need to edit this file; your work goes in work/.
//
// Not in the book: module X3 is ours. The conventions follow notes/today/x3-facts.md (B1, B2,
// B6, D1, D4); README.md states the contract, the tolerances and why.
#include <cfloat>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include "pmpp/harness.hpp"
using pmpp::Buf;

enum Kernel { RmsNorm, Rope, SwiGLU, Port };

// One call of one kernel. Buffers by kernel:
//   RmsNorm, Port  a = x, b = w, c = y          RoPE  a = x, b = pos, c = cosT, d = sinT
//   SwiGLU         a = gate, b = up, c = out
struct Call {
  Kernel k;
  Buf *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
  int rows = 0, dim = 0, heads = 0, headDim = 0, mode = 0, n = 0;
  float eps = 0.0f;
};

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
bool encodeRmsNorm(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                   id<MTLBuffer> x, id<MTLBuffer> w, id<MTLBuffer> y, uint32_t rows, uint32_t d, float eps);
void encodeRope(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                id<MTLBuffer> x, id<MTLBuffer> pos, id<MTLBuffer> cosT, id<MTLBuffer> sinT,
                uint32_t rows, uint32_t heads, uint32_t headDim, uint32_t mode);
void encodeSwiGLU(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                  id<MTLBuffer> gate, id<MTLBuffer> up, id<MTLBuffer> out, uint32_t n);
bool encodeRmsNormPort(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                       id<MTLBuffer> x, id<MTLBuffer> w, id<MTLBuffer> y, uint32_t rows, uint32_t d, float eps);
using pmpp::metal::buffer;

static id<MTLComputePipelineState> pipelineFor(Kernel k) {
  static const char* fn[] = {"rms_norm_rows", "rope_rows", "swiglu", "rms_norm_port"};
  return pmpp::metal::pipeline(PMPP_METAL_SOURCE, fn[k]);
}

// Why this pipeline cannot run the kernel as the lab specifies it, or "". The driver's own copy
// of the rule a correct launch.mm applies.
static std::string pipelineProblem(Kernel k) {
  id<MTLComputePipelineState> p = pipelineFor(k);
  if (!p) return "the Metal kernel did not build: " + pmpp::take_error();
  if (k == Rope || k == SwiGLU) return "";
  const NSUInteger w = p.threadExecutionWidth;
  char b[220];
  if ((NSUInteger)256 > p.maxTotalThreadsPerThreadgroup) {
    std::snprintf(b, sizeof b, "this pipeline runs at most %lu threads per threadgroup; the lab needs 256",
                  (unsigned long)p.maxTotalThreadsPerThreadgroup);
    return b;
  }
  if (w == 0 || 256 % w != 0 || (NSUInteger)256 / w > w) {
    std::snprintf(b, sizeof b, "SIMD-group width %lu: 256 threads are not 1 to %lu whole SIMD-groups",
                  (unsigned long)w, (unsigned long)w);
    return b;
  }
  return "";
}

// Enqueues `repeat` launches of one call in one timed run. Empty string when it ran.
static std::string launch(const Call& c, int repeat = 1, pmpp::Timing* tm = nullptr) {
  std::string why = pipelineProblem(c.k);
  if (!why.empty()) return why;
  id<MTLComputePipelineState> p = pipelineFor(c.k);
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    for (int i = 0; i < repeat && ok; ++i) {
      switch (c.k) {
        case RmsNorm:
          ok = encodeRmsNorm(enc, p, buffer(c.a), buffer(c.b), buffer(c.c), c.rows, c.dim, c.eps);
          break;
        case Rope:
          encodeRope(enc, p, buffer(c.a), buffer(c.b), buffer(c.c), buffer(c.d), c.rows, c.heads, c.headDim, c.mode);
          break;
        case SwiGLU: encodeSwiGLU(enc, p, buffer(c.a), buffer(c.b), buffer(c.c), c.n); break;
        default:
          ok = encodeRmsNormPort(enc, p, buffer(c.a), buffer(c.b), buffer(c.c), c.rows, c.dim, c.eps);
          break;
      }
    }
  });
  if (tm) *tm = t;
  if (!ok) return "your launch.mm refused to dispatch, although this pipeline can run the kernel";
  return "";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchRmsNorm(const float* x, const float* w, float* y, int rows, int d, float eps);
void launchRope(float* x, const int* pos, const float* cosT, const float* sinT, int rows, int heads, int headDim,
                int mode);
void launchSwiGLU(const float* gate, const float* up, float* out, int n);
void launchRmsNormPort(const float* x, const float* w, float* y, int rows, int d, float eps);
static float* f(Buf* b) { return (float*)pmpp::raw(b); }
static int* i32(Buf* b) { return (int*)pmpp::raw(b); }

static std::string pipelineProblem(Kernel k) {
  if (k == Rope || k == SwiGLU) return "";
  const int need = k == Port ? 1024 : 256;   // the port runs 1024 threads per block for d >= 1024
  const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  return need <= most ? "" : "this device runs at most " + std::to_string(most) + " threads per block; the lab needs " +
                                 std::to_string(need);
}
static std::string launch(const Call& c, int repeat = 1, pmpp::Timing* tm = nullptr) {
  std::string why = pipelineProblem(c.k);
  if (!why.empty()) return why;
  pmpp::Timing t = pmpp::run([&](void*) {
    for (int i = 0; i < repeat; ++i) {
      switch (c.k) {
        case RmsNorm: launchRmsNorm(f(c.a), f(c.b), f(c.c), c.rows, c.dim, c.eps); break;
        case Rope: launchRope(f(c.a), i32(c.b), f(c.c), f(c.d), c.rows, c.heads, c.headDim, c.mode); break;
        case SwiGLU: launchSwiGLU(f(c.a), f(c.b), f(c.c), c.n); break;
        default: launchRmsNormPort(f(c.a), f(c.b), f(c.c), c.rows, c.dim, c.eps); break;
      }
    }
  });
  if (tm) *tm = t;
  return "";
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- tolerances (README.md)
// |got - ref| <= k * FLT_EPSILON * (1 + |ref|), ref computed in double from the same float inputs.
static const double kEpsF = FLT_EPSILON;   // 2^-23
static const double kRmsK = 32;            // the sum of up to ~30 roundings, sqrt, a division, two products
static const double kRopeK = 8;            // two products and an add per output
static const double kSwiK = 16;            // exp, an add, a division, two products

static const float kEps = 1e-5f;           // llama2.c's and SmolLM2's RMSNorm epsilon (x3-facts D1, D4)
static const double kRopeBase = 10000.0;   // llama2.c's RoPE base (x3-facts D1)

// ---------------------------------------------------------------- buffers
static const int kGuard = 64;                    // elements past the end of every output
static const float kGuardF = -12345.5f;          // what they hold before the launch
static const float kBand = 1000.0f;              // inputs' guard band on devices: reading it changes the result
static const float kNaN = std::numeric_limits<float>::quiet_NaN();

static Buf* put(const std::vector<float>& v) {
  Buf* b = pmpp::alloc(v.size() * 4);
  pmpp::upload(b, v.data(), v.size() * 4);
  return b;
}
static Buf* putInts(const std::vector<int>& v) {
  Buf* b = pmpp::alloc(v.size() * 4);
  pmpp::upload(b, v.data(), v.size() * 4);
  return b;
}
static std::vector<float> get(Buf* b, size_t n) {
  std::vector<float> v(n);
  pmpp::download(b, v.data(), n * 4);
  return v;
}

// The emulator (AddressSanitizer) gets inputs allocated to their exact size, so that a read past
// the end stops the run and names the line. The other backends cannot see that, so their inputs
// carry a guard band whose value changes any result that reads it.
static bool inputBands() { return std::string(pmpp::backend()) != "emu"; }

static std::string fmt(const char* f, double a, double b = 0, double c = 0) {
  char buf[240];
  std::snprintf(buf, sizeof buf, f, a, b, c);
  return buf;
}

// Compares [rows x cols] results. `where` names the case in the message.
static std::string compareRows(const float* got, const std::vector<double>& ref, int rows, int cols, double k,
                               const std::string& where) {
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < (size_t)rows * cols; ++i) {
    const double e = std::fabs((double)got[i] - ref[i]);
    if (!(e <= k * kEpsF * (1.0 + std::fabs(ref[i])))) {   // also catches NaN
      if (bad == 0) first = i;
      ++bad;
    }
  }
  if (!bad) return "";
  std::string err = where + ": " + std::to_string(bad) + " of " + std::to_string((size_t)rows * cols) +
                    " values wrong; first at row " + std::to_string(first / cols) + ", column " +
                    std::to_string(first % cols) + fmt(": got %.9g, expected %.9g", got[first], ref[first]);
  if (std::isnan(got[first])) err += " (NaN: never written, or a sum that read memory nobody wrote)";
  return err;
}

static std::string checkGuard(const std::vector<float>& v, size_t from, const char* name) {
  for (size_t i = from; i < v.size(); ++i)
    if (v[i] != kGuardF) return std::string("wrote ") + name + "[" + std::to_string(i) + "], past the end";
  return "";
}

static bool sameBits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

// ---------------------------------------------------------------- RMSNorm (part A and the port)
// Row r is uniform in [-s, s] with s = 0.25 + 0.75 * (r % 4), so rows have different scales and a
// kernel that uses another row's sum is caught. Row `zeroRow` (if >= 0) is all zeros.
static std::vector<float> rmsInput(int rows, int d, int zeroRow) {
  std::vector<float> x = pmpp::random_floats((size_t)rows * d, 101u + 7u * (uint32_t)d + 13u * (uint32_t)rows);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < d; ++c) x[(size_t)r * d + c] *= r == zeroRow ? 0.0f : 0.25f + 0.75f * (r % 4);
  return x;
}
static std::vector<float> rmsWeight(int d) {
  std::vector<float> w = pmpp::random_floats((size_t)d, 977u + (uint32_t)d);
  for (float& v : w) v = 1.0f + 0.5f * v;   // around 1, as trained RMSNorm weights are
  return w;
}
static void refRms(const float* x, const float* w, int rows, int d, float eps, std::vector<double>& out) {
  out.assign((size_t)rows * d, 0.0);
  for (int r = 0; r < rows; ++r) {
    double ss = 0;
    for (int c = 0; c < d; ++c) ss += (double)x[(size_t)r * d + c] * x[(size_t)r * d + c];
    const double scale = 1.0 / std::sqrt(ss / d + (double)eps);
    for (int c = 0; c < d; ++c) out[(size_t)r * d + c] = (double)x[(size_t)r * d + c] * scale * w[c];
  }
}

// One launch of RmsNorm or Port, checked: values, guard after y, w (and x, out of place) unchanged.
static std::string checkRms(Kernel k, int rows, int d, bool inPlace, int zeroRow = -1) {
  const size_t total = (size_t)rows * d;
  const bool bands = inputBands();
  std::vector<float> x = rmsInput(rows, d, zeroRow);
  std::vector<float> w = rmsWeight(d);
  std::vector<double> ref;
  refRms(x.data(), w.data(), rows, d, kEps, ref);

  std::vector<float> xb = x, wb = w;
  if (inPlace) xb.resize(total + kGuard, kGuardF);   // y is x: its guard is the output guard
  else if (bands) xb.resize(total + kGuard, kBand);
  if (bands) wb.resize((size_t)d + kGuard, kBand);
  Buf* X = put(xb);
  Buf* W = put(wb);
  Buf* Y = X;
  if (!inPlace) {
    std::vector<float> yb(total + kGuard, kGuardF);
    std::fill(yb.begin(), yb.begin() + total, kNaN);   // a value nobody writes shows as NaN
    Y = put(yb);
  }
  Call c{k, X, W, Y};
  c.rows = rows; c.dim = d; c.eps = kEps;
  std::string err = launch(c);
  const std::string where = std::to_string(rows) + (rows == 1 ? " row" : " rows") + " of " + std::to_string(d) +
                            (inPlace ? ", in place" : "");
  if (err.empty()) {
    std::vector<float> got = get(Y, total + kGuard);
    err = compareRows(got.data(), ref, rows, d, kRmsK, where);
    if (err.empty()) err = checkGuard(got, total, "y");
    if (err.empty() && !sameBits(get(W, wb.size()), wb)) err = where + ": the weights w were changed";
    if (err.empty() && !inPlace && !sameBits(get(X, xb.size()), xb)) err = where + ": the input x was changed";
  }
  pmpp::release(X); pmpp::release(W);
  if (Y != X) pmpp::release(Y);
  return err;
}

// ---------------------------------------------------------------- RoPE
struct Tables {
  int maxPos = 0, half = 0;
  std::vector<float> cosT, sinT;
};
// theta(p, i) = p * base^(-2i / headDim), the angle llama.cpp, Hugging Face and llama2.c use
// (x3-facts B6), in double; cos and sin rounded to float once.
static Tables makeTables(int maxPos, int headDim, double base) {
  Tables t;
  t.maxPos = maxPos; t.half = headDim / 2;
  t.cosT.resize((size_t)maxPos * t.half);
  t.sinT.resize((size_t)maxPos * t.half);
  for (int p = 0; p < maxPos; ++p)
    for (int i = 0; i < t.half; ++i) {
      const double theta = p * std::pow(base, -2.0 * i / headDim);
      t.cosT[(size_t)p * t.half + i] = (float)std::cos(theta);
      t.sinT[(size_t)p * t.half + i] = (float)std::sin(theta);
    }
  return t;
}
// In double from the float inputs and the float tables the kernel reads.
static void refRope(const std::vector<float>& x, const std::vector<int>& pos, const Tables& t, int rows, int heads,
                    int headDim, int mode, std::vector<double>& out) {
  out.assign(x.begin(), x.begin() + (size_t)rows * heads * headDim);
  const int half = headDim / 2;
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < heads; ++h) {
      double* v = &out[((size_t)r * heads + h) * headDim];
      for (int i = 0; i < half; ++i) {
        const int a = mode == 0 ? 2 * i : i, b = mode == 0 ? 2 * i + 1 : i + half;
        const double c = t.cosT[(size_t)pos[r] * half + i], s = t.sinT[(size_t)pos[r] * half + i];
        const double x0 = v[a], x1 = v[b];
        v[a] = x0 * c - x1 * s;
        v[b] = x0 * s + x1 * c;
      }
    }
}

static int maxPos() {
  const std::string b = pmpp::backend();
  return b == "race" ? 64 : b == "emu" ? 512 : 4096;
}
// Distinct positions: 0 and maxPos - 1 always among them once rows >= 2.
static std::vector<int> positions(int rows, int mp) {
  if (rows == 1) return {mp - 1};
  std::vector<int> p(rows);
  for (int r = 0; r < rows; ++r) p[r] = r == 0 ? 0 : r == rows - 1 ? mp - 1 : (int)((37LL * r) % (mp - 1));
  return p;
}

// Runs the RoPE kernel in place on a copy of x (with a guard after it). Returns the launch error
// or a guard error; `out` receives the rotated values.
static std::string runRope(const std::vector<float>& x, const std::vector<int>& pos, const Tables& t, int rows,
                           int heads, int headDim, int mode, std::vector<float>& out) {
  const size_t total = (size_t)rows * heads * headDim;
  std::vector<float> xb = x;
  xb.resize(total + kGuard, kGuardF);
  Buf* X = put(xb);
  Buf* P = putInts(pos);
  Buf* C = put(t.cosT);
  Buf* S = put(t.sinT);
  Call c{Rope, X, P, C, S};
  c.rows = rows; c.heads = heads; c.headDim = headDim; c.mode = mode;
  std::string err = launch(c);
  if (err.empty()) {
    out = get(X, total + kGuard);
    err = checkGuard(out, total, "x");
    out.resize(total);
  }
  pmpp::release(X); pmpp::release(P); pmpp::release(C); pmpp::release(S);
  return err;
}

static std::string ropeWhere(int rows, int heads, int headDim, int mode) {
  return std::to_string(rows) + (rows == 1 ? " row" : " rows") + " x " + std::to_string(heads) +
         (heads == 1 ? " head" : " heads") + " of " + std::to_string(headDim) + ", mode " + std::to_string(mode);
}

static std::string checkRope(int rows, int heads, int headDim, int mode) {
  const Tables t = makeTables(maxPos(), headDim, kRopeBase);
  const std::vector<int> pos = positions(rows, t.maxPos);
  const std::vector<float> x =
      pmpp::random_floats((size_t)rows * heads * headDim, 300u + (uint32_t)(rows * 31 + heads * 7 + headDim));
  std::vector<float> got;
  std::string err = runRope(x, pos, t, rows, heads, headDim, mode, got);
  if (!err.empty()) return err;
  std::vector<double> ref;
  refRope(x, pos, t, rows, heads, headDim, mode, ref);
  return compareRows(got.data(), ref, rows, heads * headDim, kRopeK, ropeWhere(rows, heads, headDim, mode));
}

// At position 0 every angle is 0: cos 1, sin 0, so the output must equal the input exactly.
static std::string checkRopeIdentity(int mode) {
  const int rows = 3, heads = 2, headDim = 64;
  const Tables t = makeTables(maxPos(), headDim, kRopeBase);
  const std::vector<int> pos(rows, 0);
  const std::vector<float> x = pmpp::random_floats((size_t)rows * heads * headDim, 4242u);
  std::vector<float> got;
  std::string err = runRope(x, pos, t, rows, heads, headDim, mode, got);
  if (!err.empty()) return err;
  for (size_t i = 0; i < x.size(); ++i)
    if (!(got[i] == x[i]))
      return "element " + std::to_string(i) + fmt(" changed from %.9g to %.9g at position 0", x[i], got[i]);
  return "";
}

// The two pairings must give different answers on random input, or the tests could not tell a
// kernel that ignores `mode` from a correct one.
static std::string checkRopeModesDiffer() {
  const int rows = 3, heads = 2, headDim = 64;
  const Tables t = makeTables(maxPos(), headDim, kRopeBase);
  const std::vector<int> pos = positions(rows, t.maxPos);
  const std::vector<float> x = pmpp::random_floats((size_t)rows * heads * headDim, 777u);
  std::vector<float> g0, g1;
  std::string err = runRope(x, pos, t, rows, heads, headDim, 0, g0);
  if (err.empty()) err = runRope(x, pos, t, rows, heads, headDim, 1, g1);
  if (!err.empty()) return err;
  std::vector<double> r0, r1;
  refRope(x, pos, t, rows, heads, headDim, 0, r0);
  refRope(x, pos, t, rows, heads, headDim, 1, r1);
  const int cols = heads * headDim;
  // The test's own premise: the references differ.
  if (compareRows(g0.data(), r1, rows, cols, kRopeK, "").empty() && compareRows(g1.data(), r0, rows, cols, kRopeK, "").empty())
    return "the references of the two modes agree on this input; the test cannot tell them apart";
  size_t differ = 0;
  for (size_t i = 0; i < g0.size(); ++i) differ += std::fabs((double)g0[i] - g1[i]) > 1e-3;
  if (differ == 0) return "mode 0 and mode 1 gave the same output: does the kernel read `mode`?";
  std::string e0 = compareRows(g0.data(), r0, rows, cols, kRopeK, "mode 0 against its reference");
  if (!e0.empty()) return e0;
  return compareRows(g1.data(), r1, rows, cols, kRopeK, "mode 1 against its reference");
}

// llama.cpp's converter (x3-facts B6, conversion/llama.py), written out for a [R x cols] matrix:
//   if n_head_kv is not None and n_head != n_head_kv: n_head = n_head_kv
//   weights.reshape(n_head, 2, R // n_head // 2, cols).swapaxes(1, 2).reshape(R, cols)
// so new row h*hd + 2j + k is old row h*hd + k*(hd/2) + j, where hd = R / n_head.
static int permutedRow(int oldRow, int R, int nHead, int nHeadKv) {
  if (nHeadKv > 0 && nHead != nHeadKv) nHead = nHeadKv;
  const int hd = R / nHead, half = hd / 2;
  const int h = oldRow / hd, o = oldRow % hd;
  const int k = o / half, j = o % half;
  return h * hd + 2 * j + k;
}
// y[r][o] = sum_j W[o][j] x[r][j]: nn.Linear's convention, W is [out x in] (x3-facts D1).
static std::vector<float> linear(const std::vector<float>& W, const std::vector<float>& x, int rows, int out, int in) {
  std::vector<float> y((size_t)rows * out);
  for (int r = 0; r < rows; ++r)
    for (int o = 0; o < out; ++o) {
      double s = 0;
      for (int j = 0; j < in; ++j) s += (double)W[(size_t)o * in + j] * x[(size_t)r * in + j];
      y[(size_t)r * out + o] = (float)s;
    }
  return y;
}

// rope mode 1 on (W x) == the inverse row permutation of rope mode 0 on (P W x), with P
// llama.cpp's permute(W, nHead, permKv). `heads` is how many heads W's rows hold. With
// expectMatch false the identity must fail (a permutation built with the wrong head count).
static std::string checkPermutation(int heads, int nHead, int permKv, bool expectMatch) {
  const int rows = 3, headDim = pmpp::is_emu() ? 8 : 64;
  const int in = 2 * headDim, R = heads * headDim;
  const Tables t = makeTables(maxPos(), headDim, kRopeBase);
  const std::vector<int> pos = positions(rows, t.maxPos);
  const std::vector<float> W = pmpp::random_floats((size_t)R * in, 9000u + (uint32_t)(heads * 10 + permKv));
  const std::vector<float> x = pmpp::random_floats((size_t)rows * in, 9100u);
  std::vector<float> PW(W.size());
  for (int o = 0; o < R; ++o)
    std::memcpy(&PW[(size_t)permutedRow(o, R, nHead, permKv) * in], &W[(size_t)o * in], (size_t)in * 4);
  const std::vector<float> q = linear(W, x, rows, R, in), pq = linear(PW, x, rows, R, in);

  std::vector<float> neox, norm;
  std::string err = runRope(q, pos, t, rows, heads, headDim, 1, neox);
  if (err.empty()) err = runRope(pq, pos, t, rows, heads, headDim, 0, norm);
  if (!err.empty()) return err;
  std::vector<double> back((size_t)rows * R);   // norm with P undone
  for (int r = 0; r < rows; ++r)
    for (int o = 0; o < R; ++o) back[(size_t)r * R + o] = norm[(size_t)r * R + permutedRow(o, R, nHead, permKv)];
  const std::string where = "rope mode 1 on W x vs P^-1 rope mode 0 on P W x, " + std::to_string(heads) +
                            " heads of " + std::to_string(headDim);
  const std::string cmp = compareRows(neox.data(), back, rows, R, kRopeK, where);
  if (expectMatch) return cmp;
  return cmp.empty() ? "the identity held with the wrong head count: the test cannot see this mistake" : "";
}

// ---------------------------------------------------------------- SwiGLU
static std::string checkSwiGLU(int n, bool alias) {
  std::vector<float> g = pmpp::random_floats((size_t)n, 555u + (uint32_t)n);
  std::vector<float> u = pmpp::random_floats((size_t)n, 556u + (uint32_t)n);
  for (float& v : g) v *= 8.0f;
  for (float& v : u) v *= 2.0f;
  g[0] = -100.0f;                    // exp(100) overflows float
  if (n > 1) g[n - 1] = 100.0f;
  std::vector<double> ref(n);
  for (int i = 0; i < n; ++i) ref[i] = (double)g[i] / (1.0 + std::exp(-(double)g[i])) * u[i];

  const bool bands = inputBands();
  std::vector<float> gb = g, ub = u;
  if (alias) gb.resize((size_t)n + kGuard, kGuardF);
  else if (bands) gb.resize((size_t)n + kGuard, kBand);
  if (bands) ub.resize((size_t)n + kGuard, kBand);
  Buf* G = put(gb);
  Buf* U = put(ub);
  Buf* O = G;
  if (!alias) {
    std::vector<float> ob((size_t)n + kGuard, kGuardF);
    std::fill(ob.begin(), ob.begin() + n, kNaN);
    O = put(ob);
  }
  Call c{SwiGLU, G, U, O};
  c.n = n;
  std::string err = launch(c);
  const std::string where = "n " + std::to_string(n) + (alias ? ", out == gate" : "");
  if (err.empty()) {
    std::vector<float> got = get(O, (size_t)n + kGuard);
    for (int i = 0; i < n && err.empty(); ++i)
      if (!std::isfinite(got[i]) && std::isfinite(ref[i]))
        err = where + ": element " + std::to_string(i) + fmt(" is %g for gate %g", got[i], g[i]) +
              (std::isnan(got[i]) ? " (NaN: never written, or infinity / infinity)" : "");
    if (err.empty()) err = compareRows(got.data(), ref, 1, n, kSwiK, where);
    if (err.empty()) err = checkGuard(got, n, "out");
    if (err.empty() && !sameBits(get(U, ub.size()), ub)) err = where + ": up was changed";
    if (err.empty() && !alias && !sameBits(get(G, gb.size()), gb)) err = where + ": gate was changed";
  }
  pmpp::release(G); pmpp::release(U);
  if (O != G) pmpp::release(O);
  return err;
}

// ---------------------------------------------------------------- tests
static std::vector<int> pick(const std::vector<int>& race, const std::vector<int>& emu, const std::vector<int>& dev) {
  const std::string b = pmpp::backend();
  return b == "race" ? race : b == "emu" ? emu : dev;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  auto runnable = [&](Kernel k, const std::string& label) {
    std::string why = pipelineProblem(k);
    if (why.empty()) return true;
    t.check(label, [&] { return why; });
    return false;
  };

  // ---- part A: RMSNorm
  if (runnable(RmsNorm, "RMSNorm: kernel usable")) {
    const std::vector<int> rowsList = pick({1, 3}, {1, 3, 17}, {1, 3, 17});
    for (int d : pick({1, 7, 64, 1000}, {1, 7, 64, 128, 576, 1000, 4097}, {1, 7, 64, 128, 576, 1000, 4097}))
      for (int inPlace = 0; inPlace < 2; ++inPlace)
        t.check("RMSNorm, d " + std::to_string(d) + (inPlace ? ", in place" : ", out of place"), [&] {
          for (int rows : rowsList) {
            std::string err = checkRms(RmsNorm, rows, d, inPlace != 0);
            if (!err.empty()) return err;
          }
          return std::string();
        });
    for (int d : pick({64}, {64, 4097}, {64, 4097}))
      t.check("RMSNorm, an all-zero row gives 0, d " + std::to_string(d),
              [&] { return checkRms(RmsNorm, 3, d, false, 1); });
  }

  // ---- part A: RoPE
  if (runnable(Rope, "RoPE: kernel usable")) {
    for (int headDim : pick({8, 64}, {8, 64, 128}, {8, 64, 128}))
      for (int mode = 0; mode < 2; ++mode)
        t.check("RoPE mode " + std::to_string(mode) + (mode == 0 ? " (pairs 2i, 2i+1)" : " (pairs i, i+headDim/2)") +
                    ", headDim " + std::to_string(headDim),
                [&] {
                  for (int heads : {1, 2, 4})
                    for (int rows : {1, 3}) {
                      std::string err = checkRope(rows, heads, headDim, mode);
                      if (!err.empty()) return err;
                    }
                  return std::string();
                });
    for (int mode = 0; mode < 2; ++mode)
      t.check("RoPE mode " + std::to_string(mode) + ": position 0 is the identity",
              [&] { return checkRopeIdentity(mode); });
    t.check("RoPE: the two modes give different outputs", [&] { return checkRopeModesDiffer(); });
    // Q: 4 heads, permuted with (n_head, n_head). K: 2 KV heads, permuted with (n_head, n_head_kv).
    t.check("RoPE: NeoX on W x = NORM on llama.cpp's permuted rows, Q (n_head 4)",
            [&] { return checkPermutation(4, 4, 4, true); });
    t.check("RoPE: NeoX on W x = NORM on llama.cpp's permuted rows, K (n_head 4, n_head_kv 2)",
            [&] { return checkPermutation(2, 4, 2, true); });
    t.check("RoPE: K permuted with n_head instead of n_head_kv breaks the identity (x3-facts D6)",
            [&] { return checkPermutation(2, 4, 4, false); });
  }

  // ---- part A: SwiGLU
  if (runnable(SwiGLU, "SwiGLU: kernel usable"))
    for (int n : pick({1, 31, 1000}, {1, 31, 1000, 4099}, {1, 31, 1000, 4099}))
      for (int alias = 0; alias < 2; ++alias)
        t.check("SwiGLU, n " + std::to_string(n) + (alias ? ", out == gate" : ""),
                [&] { return checkSwiGLU(n, alias != 0); });

  // ---- part B: the port. Ordered so that defects (a) and (b) are reported before (c), whose
  // out-of-row read stops an emulator run (AddressSanitizer) at the first case that has it.
  if (runnable(Port, "port: kernel usable")) {
    const int many = pick({6}, {9}, {17})[0];   // more rows than the port's grid of 4 blocks
    struct Case { int rows, d; const char* why; };
    const std::vector<Case> cases =
        std::string(pmpp::backend()) == "race"
            ? std::vector<Case>{{3, 64, "one row per block, 256 threads"},
                                {many, 64, "several rows per block"},
                                {many, 577, "several rows per block, d % 4 != 0"}}
            : std::vector<Case>{{3, 64, "one row per block, 256 threads"},
                                {3, 4096, "one row per block, 1024 threads, d % 4 == 0"},
                                {many, 64, "several rows per block, 256 threads"},
                                {many, 4096, "several rows per block, 1024 threads"},
                                {1, 577, "d % 4 != 0"},
                                {3, 1001, "d % 4 != 0"},
                                {3, 4097, "d % 4 != 0, 1024 threads"},
                                {many, 1001, "several rows per block, d % 4 != 0"}};
    for (const Case& c : cases)
      t.check("port: rows " + std::to_string(c.rows) + ", d " + std::to_string(c.d) + " (" + c.why + ")",
              [&] { return checkRms(Port, c.rows, c.d, false); });
  }
  return t.finish();
}

// ---------------------------------------------------------------- lab 5.1's bandwidth
// The bandwidth lab 5.1 measured on this backend, read from reports/runs/<backend>.json (the
// file labs/pmpp.py writes), preferring your work over the solution. 0 when absent.
static std::string repoRoot() {
  std::string p = __FILE__;   // <repo>/labs/x3/lab1_llama_pieces/main.cpp
  for (int i = 0; i < 4; ++i) {
    const size_t s = p.find_last_of('/');
    if (s == std::string::npos) return "";
    p = p.substr(0, s);
  }
  return p;
}
static double lab51Bandwidth(std::string* variant) {
  std::ifstream in(repoRoot() + "/reports/runs/" + pmpp::backend() + ".json");
  if (!in) return 0;
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string s = ss.str();
  double best = 0;
  std::string bestVariant;
  for (size_t pos = s.find('{'); pos != std::string::npos; pos = s.find('{', pos + 1)) {
    const size_t end = s.find('}', pos);
    if (end == std::string::npos) break;
    const size_t inner = s.find('{', pos + 1);
    if (inner != std::string::npos && inner < end) continue;   // not an innermost object
    const std::string obj = s.substr(pos, end - pos);
    if (obj.find("\"ch05/lab1_roofline\"") == std::string::npos) continue;
    if (obj.find("\"name\": \"bandwidth\"") == std::string::npos) continue;
    const size_t g = obj.find("\"gb_per_s\":");
    if (g == std::string::npos) continue;
    const double v = std::atof(obj.c_str() + g + 11);
    const std::string var = obj.find("\"variant\": \"work\"") != std::string::npos ? "work" : "solution";
    if (v > 0 && (bestVariant.empty() || (var == "work" && bestVariant != "work"))) {
      best = v;
      bestVariant = var;
    }
  }
  if (variant) *variant = bestVariant;
  return best;
}

// ---------------------------------------------------------------- benchmark
// Time per call: best of `reps` timed runs after one untimed run; each run repeats the call
// `calls` times back to back, `calls` doubled until one run lasts at least 2 ms, so that short
// calls are not lost in the cost of submitting the work (lab 10.3's method).
static double perCall(int reps, const Call& c, int* calls, std::string* err) {
  *calls = 1;
  pmpp::Timing tm{0, 0};
  for (;;) {
    *err = launch(c, *calls, &tm);
    if (err->empty()) *err = pmpp::take_error();
    if (!err->empty()) return -1;
    if (tm.best() >= 0.002 || *calls >= 4096) break;
    *calls *= 2;
  }
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    *err = launch(c, *calls, &tm);
    if (err->empty()) *err = pmpp::take_error();
    if (!err->empty()) return -1;
    best = std::min(best, tm.best());
  }
  return best / *calls;
}

static int bench(const pmpp::Args& a) {
  std::string from;
  const double bw51 = lab51Bandwidth(&from);
  if (bw51 > 0)
    std::printf("lab 5.1 measured %.1f GB/s on this backend (%s; reports/runs/%s.json)\n", bw51, from.c_str(),
                pmpp::backend());
  else
    std::printf("no lab 5.1 bandwidth in reports/runs/%s.json: run `python3 labs/pmpp.py run 5.1 --bench`\n"
                "to see each row as a fraction of it.\n",
                pmpp::backend());
  const double mem = deviceMemory();
  const double budget = mem > 0 ? mem / 4 : 1e9;
  const std::vector<int> dims = {576, 4096}, rowsList = {1, 16, 256};
  int rc = 0;

  for (int fam = 0; fam < 3; ++fam) {
    const Kernel k = fam == 0 ? RmsNorm : fam == 1 ? Rope : SwiGLU;
    const char* name = fam == 0 ? "rmsnorm" : fam == 1 ? "rope" : "swiglu";
    std::string why = pipelineProblem(k);
    if (!why.empty()) { std::printf("%s: %s\n", name, why.c_str()); rc = 1; continue; }
    std::printf("\n%s, best of %d timed runs\n", fam == 0 ? "RMSNorm (y = x * scale * w, out of place)"
                                               : fam == 1 ? "RoPE (mode 0, in place)"
                                                          : "SwiGLU (n = rows x 4d, out of place)",
                a.reps);
    std::printf("  %5s %5s %10s %6s %12s %9s %10s\n", "rows", "d", "MB moved", "calls", "us per call", "GB/s",
                bw51 > 0 ? "of lab 5.1" : "");
    for (int d : dims)
      for (int rows : rowsList) {
        const int headDim = d % 128 == 0 ? 128 : 64, heads = d / headDim;
        const size_t elems = fam == 2 ? (size_t)rows * 4 * d : (size_t)rows * d;
        // Unique bytes, each counted once: RMSNorm reads x and w and writes y; RoPE reads and
        // writes x and reads pos and the cos and sin rows of the rows' (distinct) positions;
        // SwiGLU reads gate and up and writes out.
        const double bytes = fam == 0 ? 4.0 * (2.0 * elems + d)
                           : fam == 1 ? 8.0 * elems + 4.0 * rows + 4.0 * rows * headDim
                                      : 12.0 * elems;
        if (bytes > budget) {
          std::printf("  %5d %5d  skipped: larger than a quarter of the device memory reported\n", rows, d);
          continue;
        }
        std::vector<Buf*> bufs;
        Call c{k};
        c.rows = rows; c.dim = d; c.eps = kEps; c.n = (int)elems; c.heads = heads; c.headDim = headDim;
        std::vector<float> x = pmpp::random_floats(elems, 60u + (uint32_t)(rows + d + fam));
        std::vector<float> w = rmsWeight(d), up = pmpp::random_floats(elems, 61u + (uint32_t)rows);
        Tables tab;
        std::vector<int> pos(rows);
        if (fam == 0) {
          c.a = put(x); c.b = put(w); c.c = pmpp::alloc(elems * 4);
          bufs = {c.a, c.b, c.c};
        } else if (fam == 1) {
          tab = makeTables(std::max(256, rows), headDim, kRopeBase);
          for (int r = 0; r < rows; ++r) pos[r] = r;
          c.a = put(x); c.b = putInts(pos); c.c = put(tab.cosT); c.d = put(tab.sinT);
          bufs = {c.a, c.b, c.c, c.d};
        } else {
          for (float& v : x) v *= 8.0f;
          c.a = put(x); c.b = put(up); c.c = pmpp::alloc(elems * 4);
          bufs = {c.a, c.b, c.c};
        }
        int calls = 0;
        std::string err;
        const double s = perCall(a.reps, c, &calls, &err);
        // Check a result before reporting a time. RoPE ran in place many times, so it is run
        // once more on fresh data.
        std::string bad;
        if (s > 0) {
          std::vector<double> ref;
          if (fam == 0) {
            refRms(x.data(), w.data(), rows, d, kEps, ref);
            bad = compareRows(get(c.c, elems).data(), ref, rows, d, kRmsK, "result");
          } else if (fam == 1) {
            pmpp::upload(c.a, x.data(), elems * 4);
            if ((err = launch(c)).empty()) {
              refRope(x, pos, tab, rows, heads, headDim, 0, ref);
              bad = compareRows(get(c.a, elems).data(), ref, rows, d, kRopeK, "result");
            }
          } else {
            ref.resize(elems);
            for (size_t i = 0; i < elems; ++i) ref[i] = (double)x[i] / (1.0 + std::exp(-(double)x[i])) * up[i];
            bad = compareRows(get(c.c, elems).data(), ref, 1, (int)elems, kSwiK, "result");
          }
        }
        for (Buf* b : bufs) pmpp::release(b);
        if (s <= 0 || !err.empty()) { std::printf("  %5d %5d  failed: %s\n", rows, d, err.c_str()); rc = 1; continue; }
        if (!bad.empty()) { std::printf("  %5d %5d  wrong (%s); not reported\n", rows, d, bad.c_str()); rc = 1; continue; }
        const double gbps = bytes / s / 1e9;
        std::printf("  %5d %5d %10.3f %6d %12.2f %9.2f", rows, d, bytes / 1e6, calls, s * 1e6, gbps);
        if (bw51 > 0) std::printf(" %9.1f%%", 100.0 * gbps / bw51);
        std::printf("\n");
        pmpp::Json j = pmpp::bench_row(a, name);
        j.n("rows", rows).n("d", d).n("bytes", bytes).n("seconds", s).n("gb_per_s", gbps).n("calls", calls);
        if (fam == 1) j.n("heads", heads).n("head_dim", headDim);
        if (bw51 > 0) j.n("lab51_gb_per_s", bw51);
        j.emit();
      }
  }
  std::printf("\nGB/s counts the unique bytes each call must read and write, once. Time per call is the best of\n"
              "%d timed runs after an untimed one, each run repeating the call until it lasts at least 2 ms.\n"
              "RoPE uses heads of 128 where d is a multiple of 128, else 64 (576 = 9 x 64).\n"
              "These are measurements of this machine, at these sizes.\n",
              a.reps);
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
