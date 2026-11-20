// Test and benchmark helpers shared by the labs. Header-only, C++17.
// Results are printed twice: readable text for you, and machine-readable lines
// starting with "@pmpp " that labs/pmpp.py collects into the report.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pmpp/backend.hpp"

namespace pmpp {

// ---------- tiny JSON line writer ----------
class Json {
 public:
  Json& s(const std::string& k, const std::string& v) { key(k); o_ << '"' << esc(v) << '"'; return *this; }
  Json& n(const std::string& k, double v) {
    key(k);
    if (std::isfinite(v)) { char b[64]; std::snprintf(b, sizeof b, "%.9g", v); o_ << b; } else o_ << "null";
    return *this;
  }
  Json& b(const std::string& k, bool v) { key(k); o_ << (v ? "true" : "false"); return *this; }
  void emit() { std::printf("@pmpp {%s}\n", o_.str().c_str()); std::fflush(stdout); }
 private:
  void key(const std::string& k) { if (!first_) o_ << ','; first_ = false; o_ << '"' << esc(k) << "\":"; }
  static std::string esc(const std::string& in) {
    std::string out;
    for (char c : in) {
      if (c == '"' || c == '\\') { out += '\\'; out += c; }
      else if (c == '\n') out += "\\n";
      else if ((unsigned char)c < 0x20) out += ' ';
      else out += c;
    }
    return out;
  }
  std::ostringstream o_;
  bool first_ = true;
};

// ---------- command line ----------
struct Args {
  bool test = false, bench = false, device = false;
  std::string lab = PMPP_LAB, variant = PMPP_VARIANT;
  int reps = 5;
};
inline Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s == "--test") a.test = true;
    else if (s == "--bench") a.bench = true;
    else if (s == "--device") a.device = true;
    else if (s == "--reps" && i + 1 < argc) a.reps = std::atoi(argv[++i]);
    else { std::fprintf(stderr, "usage: %s [--test] [--bench] [--device] [--reps N]\n", argv[0]); std::exit(2); }
  }
  if (!a.test && !a.bench && !a.device) a.test = true;
  return a;
}

inline void emit_device() {
  Json j;
  j.s("kind", "device").s("backend", backend());
  std::string why;
  bool ok = device_available(&why);
  j.b("available", ok);
  if (!ok) j.s("why", why);
  std::printf("backend: %s%s\n", backend(), ok ? "" : (" (no device: " + why + ")").c_str());
  if (ok) for (auto& kv : device_props()) {
    std::printf("  %-34s %s\n", kv.first.c_str(), kv.second.c_str());
    j.s(kv.first, kv.second);
  }
  j.emit();
}

// ---------- deterministic data (identical on every platform) ----------
inline std::vector<float> random_floats(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t x = seed * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;                 // xorshift32
    v[i] = (float)((x >> 8) & 0xFFFF) / 32767.5f - 1.0f;     // uniform in [-1, 1]
  }
  return v;
}

// C[j x l] = A[j x k] * B[k x l], row-major, accumulated in double.
// `mag` (optional) receives sum_k |a*b| per output, used to scale the tolerance.
inline void cpu_gemm(const float* A, const float* B, int j, int k, int l,
                     std::vector<float>& C, std::vector<float>* mag = nullptr) {
  C.assign((size_t)j * l, 0.f);
  if (mag) mag->assign((size_t)j * l, 0.f);
  std::vector<double> acc(l), am(l);
  for (int r = 0; r < j; ++r) {
    std::fill(acc.begin(), acc.end(), 0.0); std::fill(am.begin(), am.end(), 0.0);
    for (int x = 0; x < k; ++x) {
      const double a = A[(size_t)r * k + x];
      const float* brow = B + (size_t)x * l;
      for (int c = 0; c < l; ++c) { const double p = a * brow[c]; acc[c] += p; am[c] += std::fabs(p); }
    }
    for (int c = 0; c < l; ++c) {
      C[(size_t)r * l + c] = (float)acc[c];
      if (mag) (*mag)[(size_t)r * l + c] = (float)am[c];
    }
  }
}

struct Cmp { bool ok; size_t bad, first_bad; double max_err; };
// |got - ref| <= rtol * (1 + mag). `mag` may be null (then |ref| is used).
inline Cmp compare(const float* got, const float* ref, const float* mag, size_t n, double rtol = 1e-4) {
  Cmp c{true, 0, 0, 0.0};
  for (size_t i = 0; i < n; ++i) {
    const double e = std::fabs((double)got[i] - (double)ref[i]);
    const double tol = rtol * (1.0 + (mag ? mag[i] : std::fabs(ref[i])));
    if (!(e <= tol)) { if (c.bad == 0) c.first_bad = i; c.bad++; c.ok = false; }   // also catches NaN
    if (e > c.max_err || std::isnan(e)) c.max_err = e;
  }
  return c;
}

// ---------- test bookkeeping ----------
class Tests {
 public:
  explicit Tests(const Args& a) : a_(a) {}
  // Runs `body` once per schedule (several on the emulator, one on real devices).
  // `body` returns an empty string on success, else a description of the failure.
  void check(const std::string& name, const std::function<std::string()>& body) {
    for (int s = 0; s < schedules(); ++s) {
      use_schedule(s);
      std::string full = name;
      if (schedules() > 1) full += std::string(" [") + schedule_name(s) + "]";
      std::string err = body();
      std::string launch_err = take_error();
      if (err.empty() && !launch_err.empty()) err = "launch error: " + launch_err;
      const bool ok = err.empty();
      std::printf("  %s  %s%s%s\n", ok ? "PASS" : "FAIL", full.c_str(), ok ? "" : "  -> ", err.c_str());
      Json j;
      j.s("kind", "test").s("lab", a_.lab).s("variant", a_.variant).s("backend", backend())
       .s("case", full).b("pass", ok);
      if (!ok) j.s("detail", err);
      j.emit();
      total_++; failed_ += !ok;
    }
    use_schedule(0);
  }
  int finish() {
    std::printf("%s: %d/%d test cases passed (%s, %s)\n", failed_ ? "FAILED" : "OK",
                total_ - failed_, total_, a_.variant.c_str(), backend());
    return failed_ ? 1 : 0;
  }
 private:
  Args a_;
  int total_ = 0, failed_ = 0;
};

inline std::string describe(const Cmp& c, int cols) {
  if (c.ok) return "";
  char b[200];
  std::snprintf(b, sizeof b, "%zu wrong elements; first at row %zu, col %zu; max abs error %.4g",
                c.bad, c.first_bad / (size_t)cols, c.first_bad % (size_t)cols, c.max_err);
  return b;
}

// ---------- timing ----------
struct Stat { double min_s, median_s; int reps; };
// One untimed warm-up, then `reps` timed runs.
inline Stat time_it(int reps, const std::function<Timing()>& once) {
  once();
  std::vector<double> t;
  for (int i = 0; i < reps; ++i) t.push_back(once().best());
  std::sort(t.begin(), t.end());
  return Stat{t.front(), t[t.size() / 2], reps};
}

inline Json bench_row(const Args& a, const std::string& name) {
  Json j;
  j.s("kind", "bench").s("lab", a.lab).s("variant", a.variant).s("backend", backend()).s("name", name);
  return j;
}


// ---------- a matrix-multiplication problem instance on the device ----------
// C[j x l] = A[j x k] * B[k x l], row-major. With `with_data` false the device
// buffers are allocated but left with arbitrary contents (enough for timing).
struct GemmProblem {
  int j, k, l;
  std::vector<float> a, b, ref, mag;
  Buf *A = nullptr, *B = nullptr, *C = nullptr;
  GemmProblem(int j_, int k_, int l_, uint32_t seed, bool with_data = true) : j(j_), k(k_), l(l_) {
    A = alloc((size_t)j * k * 4); B = alloc((size_t)k * l * 4); C = alloc((size_t)j * l * 4);
    if (with_data) {
      a = random_floats((size_t)j * k, seed);
      b = random_floats((size_t)k * l, seed + 1);
      upload(A, a.data(), a.size() * 4);
      upload(B, b.data(), b.size() * 4);
    }
  }
  GemmProblem(const GemmProblem&) = delete;
  ~GemmProblem() { release(A); release(B); release(C); }
  double flop() const { return 2.0 * j * (double)k * l; }
  // Downloads C and compares it with a CPU reference. Empty string = correct.
  std::string check() {
    if (ref.empty()) cpu_gemm(a.data(), b.data(), j, k, l, ref, &mag);
    std::vector<float> got((size_t)j * l);
    download(C, got.data(), got.size() * 4);
    return describe(compare(got.data(), ref.data(), mag.data(), got.size()), l);
  }
};

// True on both CPU emulators ("emu" and the ThreadSanitizer flavour "race"): drivers use
// it to pick small problem sizes and to skip benchmarks, which mean nothing there.
inline bool is_emu() { const std::string b = backend(); return b == "emu" || b == "race"; }

// Standard main(): device info, then tests and/or benchmarks.
inline int lab_main(int argc, char** argv, const std::function<int(const Args&)>& tests,
                    const std::function<int(const Args&)>& bench) {
  Args a = parse_args(argc, argv);
  if (a.device) { emit_device(); return 0; }
  std::string why;
  if (!device_available(&why)) { std::printf("no %s device: %s\n", backend(), why.c_str()); return 3; }
  int rc = 0;
  if (a.test) rc |= tests(a);
  if (a.bench) {
    if (is_emu()) std::printf("bench skipped: the CPU emulator models correctness only; its timings mean nothing.\n");
    else rc |= bench(a);
  }
  return rc;
}

}  // namespace pmpp
