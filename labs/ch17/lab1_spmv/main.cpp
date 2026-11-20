// Lab 17.1 driver: tests and benchmarks sparse matrix-vector multiply-and-accumulate,
// y = A·x + y, with your CSR and ELL kernels, and the hybrid ELL + COO format: the provided
// COO kernel (provided/, Fig. 17.5) for the spilled entries, then your ELL kernel.
// You do not need to edit this file; your work goes in work/.
//
// Every format is built here, on the host, from one generated matrix, and the CPU reference
// (accumulated in double) lives here. A result must match within a tolerance scaled by the
// size of each row's terms, |y0| + sum |a_ij x_j|, so it is never relative-only near zero:
// Metal does not default to strict IEEE arithmetic (notes/today/ch05-facts.md C7), and a
// float atomic add on global memory flushes subnormals to zero (ch10-facts A6).
//
// What the tests plant, so that each common mistake fails with a message that names it:
//   - y starts as random non-zero values y0 and must end as y0 + A·x: a kernel that assigns
//     instead of adding fails.
//   - y has kGuard elements after its last row, set to a sentinel that must survive.
//   - Every array a kernel reads has kGuard elements after its end. A thread past the last
//     row then stays inside the buffers, and what those elements hold makes it change y's
//     guard: CSR rows of one entry (column 0, value 1), nnzPerRow 1, ELL slots of column 0
//     and value 1.
//   - CUDA, HIP, emu, race: x is passed as buffer + 1 and the element before x[0] is NaN,
//     so reading x at a pad's column -1 poisons the row. Metal gets no such guard: its
//     default fast math makes NaN propagation unreliable, and the encode functions take
//     whole buffers, so there is no checked way to hand the kernel x + 1.
//   - ELL slots of the alignment rows numRows..stride-1 hold value NaN and column 0.
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <limits>
#include <numeric>

#include "pmpp/harness.hpp"
#include "spmv.h"
using pmpp::Buf;

// A grid of whole blocks has at most SPMV_BLOCK - 1 threads past the last row.
static const int kGuard = SPMV_BLOCK;
static const float kSentinel = -7777.25f;
static const float kNaN = std::numeric_limits<float>::quiet_NaN();
static const double kRtol = 1e-4;

// ---------------------------------------------------------------- arrays on the device
struct DevCsr { Buf *rowPtrs = nullptr, *colIdx = nullptr, *values = nullptr; int numRows = 0; };
struct DevEll { Buf *colIdx = nullptr, *values = nullptr, *nnzPerRow = nullptr; int numRows = 0, width = 0, stride = 0; };
struct DevCoo { Buf *rowIdx = nullptr, *colIdx = nullptr, *values = nullptr; int nnz = 0; };
struct DevX { Buf* buf = nullptr; int lead = 0; };   // x[0] is element `lead` of buf

// ---- entry points implemented in <variant>/ and provided/ ----
// Each run* returns "" when the work was submitted, else why it was not. `dispatches`, when
// given (the numRows = 0 tests), receives the number of dispatches the encode function made;
// on Metal those are counted, not sent to the GPU. CUDA-family launchers ignore it: there a
// launch of zero blocks shows up in the error state instead.
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeSpmvCsr(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> rowPtrs,
                   id<MTLBuffer> colIdx, id<MTLBuffer> values, id<MTLBuffer> x, id<MTLBuffer> y, uint32_t numRows);
void encodeSpmvEll(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> colIdx,
                   id<MTLBuffer> values, id<MTLBuffer> nnzPerRow, id<MTLBuffer> x, id<MTLBuffer> y,
                   uint32_t numRows, uint32_t width, uint32_t stride);
void encodeSpmvCoo(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> rowIdx,
                   id<MTLBuffer> colIdx, id<MTLBuffer> values, id<MTLBuffer> x, id<MTLBuffer> y,
                   uint32_t numNonzeros);
using pmpp::metal::buffer;

// Takes the compute encoder's place in the numRows = 0 tests: passes every call on to the
// real encoder, except dispatches, which it counts and drops.
@interface SpmvDispatchCounter : NSProxy
@property(nonatomic, strong) id target;
@property(nonatomic) int dispatches;
@end
@implementation SpmvDispatchCounter
- (NSMethodSignature*)methodSignatureForSelector:(SEL)sel { return [self.target methodSignatureForSelector:sel]; }
- (void)forwardInvocation:(NSInvocation*)inv {
  if ([NSStringFromSelector(inv.selector) hasPrefix:@"dispatch"]) { self.dispatches = self.dispatches + 1; return; }
  [inv invokeWithTarget:self.target];
}
@end

static const int kXLead = 0;
static const int kBenchXLead = 0;
static const char* kNothing = "nothing encoded";
static std::string dirOf(const std::string& p) { return p.substr(0, p.find_last_of('/')); }
static id<MTLComputePipelineState> pso(const char* fn) { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, fn); }
static id<MTLComputePipelineState> cooPso() {
  return pmpp::metal::pipeline(dirOf(dirOf(PMPP_METAL_SOURCE)) + "/provided/coo_spmv.metal", "spmv_coo");
}
static std::string notBuilt(const char* fn) {
  return std::string("the Metal kernel ") + fn + " did not build: " + pmpp::take_error();
}
static id<MTLComputeCommandEncoder> encoderFor(void* pass, SpmvDispatchCounter* counter) {
  id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)pass)->encoder;
  if (!counter) return enc;
  counter.target = enc;
  return (id<MTLComputeCommandEncoder>)counter;
}
static std::string runCsr(const DevCsr& a, const DevX& x, Buf* y, pmpp::Timing* tm = nullptr,
                          int* dispatches = nullptr) {
  id<MTLComputePipelineState> p = pso("spmv_csr");
  if (!p) return notBuilt("spmv_csr");
  SpmvDispatchCounter* counter = dispatches ? [SpmvDispatchCounter alloc] : nil;
  pmpp::Timing t = pmpp::run([&](void* v) {
    encodeSpmvCsr(encoderFor(v, counter), p, buffer(a.rowPtrs), buffer(a.colIdx), buffer(a.values),
                  buffer(x.buf), buffer(y), (uint32_t)a.numRows);
  });
  if (dispatches) *dispatches = counter.dispatches;
  if (tm) *tm = t;
  return "";
}
static std::string runEll(const DevEll& e, const DevX& x, Buf* y, pmpp::Timing* tm = nullptr,
                          int* dispatches = nullptr) {
  id<MTLComputePipelineState> p = pso("spmv_ell");
  if (!p) return notBuilt("spmv_ell");
  SpmvDispatchCounter* counter = dispatches ? [SpmvDispatchCounter alloc] : nil;
  pmpp::Timing t = pmpp::run([&](void* v) {
    encodeSpmvEll(encoderFor(v, counter), p, buffer(e.colIdx), buffer(e.values), buffer(e.nnzPerRow),
                  buffer(x.buf), buffer(y), (uint32_t)e.numRows, (uint32_t)e.width, (uint32_t)e.stride);
  });
  if (dispatches) *dispatches = counter.dispatches;
  if (tm) *tm = t;
  return "";
}
static std::string runCoo(const DevCoo& c, const DevX& x, Buf* y, pmpp::Timing* tm = nullptr) {
  id<MTLComputePipelineState> p = cooPso();
  if (!p) return notBuilt("spmv_coo (provided)");
  pmpp::Timing t = pmpp::run([&](void* v) {
    encodeSpmvCoo(((pmpp::metal::Pass*)v)->encoder, p, buffer(c.rowIdx), buffer(c.colIdx), buffer(c.values),
                  buffer(x.buf), buffer(y), (uint32_t)c.nnz);
  });
  if (tm) *tm = t;
  return "";
}
// The hybrid: the provided COO kernel, then your ELL kernel, in one compute pass. The pass is
// serial, so the ELL kernel sees the COO kernel's writes (notes/today/ch02-facts.md C9).
static std::string runHybrid(const DevCoo& c, const DevEll& e, const DevX& x, Buf* y, pmpp::Timing* tm = nullptr) {
  id<MTLComputePipelineState> pc = cooPso();
  if (!pc) return notBuilt("spmv_coo (provided)");
  id<MTLComputePipelineState> pe = pso("spmv_ell");
  if (!pe) return notBuilt("spmv_ell");
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    encodeSpmvCoo(enc, pc, buffer(c.rowIdx), buffer(c.colIdx), buffer(c.values), buffer(x.buf), buffer(y),
                  (uint32_t)c.nnz);
    encodeSpmvEll(enc, pe, buffer(e.colIdx), buffer(e.values), buffer(e.nnzPerRow), buffer(x.buf), buffer(y),
                  (uint32_t)e.numRows, (uint32_t)e.width, (uint32_t)e.stride);
  });
  if (tm) *tm = t;
  return "";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
static const char* kMemKey = "recommendedMaxWorkingSetSize";
static double maxBufferBytes() { return pmpp::device_prop("maxBufferLength", 0); }
static const char* kCsrNanHint = "Nothing in this test holds NaN, so it came from a read outside the arrays.";
static const char* kEllNanHint =
    "Inside the buffers, NaN sits in one place only: the slots of the alignment rows numRows..stride-1. A kernel "
    "that computes a slot's index wrongly (stepping by numRows instead of stride, for instance) reaches them. "
    "Outside the buffers anything can sit, NaN included: reading x at a pad's column -1 reads before the start of "
    "x. Stop at nnzPerRow[row], or skip slots whose column is negative.";
#else
void launchSpmvCsr(const int* rowPtrs, const int* colIdx, const float* values, const float* x, float* y, int numRows);
void launchSpmvEll(const int* colIdx, const float* values, const int* nnzPerRow, const float* x, float* y,
                   int numRows, int width, int stride);
void launchSpmvCoo(const int* rowIdx, const int* colIdx, const float* values, const float* x, float* y,
                   int numNonzeros);

static const int kXLead = 1;
// The benchmark keeps x 256-byte aligned (64 floats), with NaN still just before x[0].
static const int kBenchXLead = 64;
static const char* kNothing = "nothing launched";
static const int* ints(Buf* b) { return (const int*)pmpp::raw(b); }
static const float* floats(Buf* b) { return (const float*)pmpp::raw(b); }
static const float* xs(const DevX& x) { return floats(x.buf) + x.lead; }
static float* ys(Buf* b) { return (float*)pmpp::raw(b); }
static std::string runCsr(const DevCsr& a, const DevX& x, Buf* y, pmpp::Timing* tm = nullptr, int* = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) {
    launchSpmvCsr(ints(a.rowPtrs), ints(a.colIdx), floats(a.values), xs(x), ys(y), a.numRows);
  });
  if (tm) *tm = t;
  return "";
}
static std::string runEll(const DevEll& e, const DevX& x, Buf* y, pmpp::Timing* tm = nullptr, int* = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) {
    launchSpmvEll(ints(e.colIdx), floats(e.values), ints(e.nnzPerRow), xs(x), ys(y), e.numRows, e.width, e.stride);
  });
  if (tm) *tm = t;
  return "";
}
static std::string runCoo(const DevCoo& c, const DevX& x, Buf* y, pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) {
    launchSpmvCoo(ints(c.rowIdx), ints(c.colIdx), floats(c.values), xs(x), ys(y), c.nnz);
  });
  if (tm) *tm = t;
  return "";
}
// The hybrid: the provided COO kernel, then your ELL kernel. Two launches on one stream run
// in order.
static std::string runHybrid(const DevCoo& c, const DevEll& e, const DevX& x, Buf* y, pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) {
    launchSpmvCoo(ints(c.rowIdx), ints(c.colIdx), floats(c.values), xs(x), ys(y), c.nnz);
    launchSpmvEll(ints(e.colIdx), floats(e.values), ints(e.nnzPerRow), xs(x), ys(y), e.numRows, e.width, e.stride);
  });
  if (tm) *tm = t;
  return "";
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
static const char* kMemKey = "totalGlobalMem";
static double maxBufferBytes() { return 0; }   // no per-allocation limit reported
static const char* kCsrNanHint =
    "In this test only x[-1], the element just before x[0], holds NaN: something read x below index 0.";
static const char* kEllNanHint =
    "In this test NaN sits in two places only. (1) x[-1], the element just before x[0]: that is what reading x at "
    "a pad's column -1 gives. Stop at nnzPerRow[row], or skip slots whose column is negative. (2) The slots of the "
    "alignment rows numRows..stride-1: a kernel that computes a slot's index wrongly (stepping by numRows instead "
    "of stride, for instance) reaches them.";
#endif

// ---------------------------------------------------------------- matrices on the host
struct Rng {   // xorshift32: the same numbers on every platform
  uint32_t s;
  explicit Rng(uint32_t seed) : s(seed * 2654435761u + 1u) { if (!s) s = 1; }
  uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
  int below(int n) { return (int)(((uint64_t)next() * (uint32_t)n) >> 32); }
  float value() { return (float)((next() >> 8) & 0xFFFF) / 32767.5f - 1.0f; }   // in [-1, 1], never 0
};

struct Csr {
  int numRows = 0, numCols = 0;
  std::vector<int> rowPtrs = {0}, colIdx;
  std::vector<float> values;
  int nnz() const { return (int)colIdx.size(); }
  int len(int r) const { return rowPtrs[(size_t)r + 1] - rowPtrs[(size_t)r]; }
  int maxLen() const { int m = 0; for (int r = 0; r < numRows; ++r) m = std::max(m, len(r)); return m; }
  int minLen() const { int m = numRows ? len(0) : 0; for (int r = 0; r < numRows; ++r) m = std::min(m, len(r)); return m; }
  int medianLen() const {
    if (numRows == 0) return 0;
    std::vector<int> l((size_t)numRows);
    for (int r = 0; r < numRows; ++r) l[(size_t)r] = len(r);
    std::nth_element(l.begin(), l.begin() + numRows / 2, l.end());
    return l[(size_t)numRows / 2];
  }
};

// Rows of the given lengths. A row's columns are distinct and sorted, drawn from
// [r - band, r + band] clipped to the matrix (every column when band < 0); a row that asks
// for more columns than that range holds gets all of them. Values uniform in [-1, 1].
static Csr makeMatrix(int numRows, int numCols, const std::vector<int>& lens, int band, uint32_t seed) {
  Csr a;
  a.numRows = numRows;
  a.numCols = numCols;
  a.rowPtrs.reserve((size_t)numRows + 1);
  size_t total = 0;
  for (int l : lens) total += (size_t)l;
  a.colIdx.reserve(total);
  a.values.reserve(total);
  std::vector<int> mark((size_t)std::max(numCols, 1), -1), cols;
  Rng g(seed);
  for (int r = 0; r < numRows; ++r) {
    int lo = 0, hi = numCols - 1;
    if (band >= 0) { lo = std::max(0, r - band); hi = std::min(numCols - 1, r + band); }
    const int range = std::max(hi - lo + 1, 0);
    const int want = std::min(lens[(size_t)r], range);
    cols.clear();
    if (2 * want > range) {   // dense: a partial shuffle of the whole range
      for (int c = lo; c <= hi; ++c) cols.push_back(c);
      for (int i = 0; i < want; ++i) std::swap(cols[(size_t)i], cols[(size_t)(i + g.below(range - i))]);
      cols.resize((size_t)want);
    } else {
      while ((int)cols.size() < want) {
        const int c = lo + g.below(range);
        if (mark[(size_t)c] != r) { mark[(size_t)c] = r; cols.push_back(c); }
      }
    }
    std::sort(cols.begin(), cols.end());
    for (int c : cols) { a.colIdx.push_back(c); a.values.push_back(g.value()); }
    a.rowPtrs.push_back((int)a.colIdx.size());
  }
  return a;
}

// The same rows, longest first, equal lengths in their original order (a stable sort, as in
// Fig. 17.14): the JDS permutation of section 17.6. Each row keeps its columns.
static Csr sortedByLength(const Csr& a) {
  std::vector<int> order((size_t)a.numRows);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int p, int q) { return a.len(p) > a.len(q); });
  Csr s;
  s.numRows = a.numRows;
  s.numCols = a.numCols;
  s.rowPtrs.reserve(a.rowPtrs.size());
  s.colIdx.reserve(a.colIdx.size());
  s.values.reserve(a.values.size());
  for (int r : order) {
    for (int i = a.rowPtrs[(size_t)r]; i < a.rowPtrs[(size_t)r + 1]; ++i) {
      s.colIdx.push_back(a.colIdx[(size_t)i]);
      s.values.push_back(a.values[(size_t)i]);
    }
    s.rowPtrs.push_back((int)s.colIdx.size());
  }
  return s;
}

struct Ell {
  int numRows = 0, width = 0, stride = 0;
  std::vector<int> colIdx, nnzPerRow;
  std::vector<float> values;
  long long entries = 0;   // real entries stored (the rest of the width * stride slots are pads)
};
struct Coo {
  std::vector<int> rowIdx, colIdx;
  std::vector<float> values;
  int nnz() const { return (int)values.size(); }
};

// ELL of the given width and stride, built from CSR. Each row's first `width` entries go to
// the ELL part; with `spill`, the rest go there (the hybrid's COO part, in row order).
// Without `spill`, `width` must cover every row.
static Ell toEll(const Csr& a, int width, int stride, Coo* spill = nullptr) {
  Ell e;
  e.numRows = a.numRows;
  e.width = width;
  e.stride = stride;
  const size_t slots = (size_t)width * (size_t)stride;
  e.colIdx.assign(slots, -1);     // pads: column -1, value 0, as cuSPARSE and rocSPARSE pad
  e.values.assign(slots, 0.0f);   // (notes/today/ch17-facts.md A6, B3)
  for (int t = 0; t < width; ++t) {
    for (int r = a.numRows; r < stride; ++r) {   // alignment rows: column 0, value NaN
      e.colIdx[(size_t)t * stride + r] = 0;
      e.values[(size_t)t * stride + r] = kNaN;
    }
  }
  e.nnzPerRow.assign((size_t)a.numRows, 0);
  for (int r = 0; r < a.numRows; ++r) {
    for (int k = 0; k < a.len(r); ++k) {
      const size_t src = (size_t)a.rowPtrs[(size_t)r] + (size_t)k;
      if (k < width) {
        const size_t i = (size_t)k * stride + r;   // column-major: slot k of row r
        e.colIdx[i] = a.colIdx[src];
        e.values[i] = a.values[src];
        e.nnzPerRow[(size_t)r]++;
        e.entries++;
      } else if (spill) {
        spill->rowIdx.push_back(r);
        spill->colIdx.push_back(a.colIdx[src]);
        spill->values.push_back(a.values[src]);
      } else {
        std::fprintf(stderr, "internal error: ELL width %d is shorter than row %d (%d)\n", width, r, a.len(r));
        std::exit(2);
      }
    }
  }
  return e;
}
static int spilledEntries(const Csr& a, int width) {
  int n = 0;
  for (int r = 0; r < a.numRows; ++r) n += std::max(a.len(r) - width, 0);
  return n;
}

// ---------------------------------------------------------------- CPU reference
struct Expect { std::vector<float> ax, ref, mag; };
// ref = y0 + A·x, accumulated in double. mag = |y0| + sum |a_ij x_j| scales the tolerance.
static Expect cpuSpmv(const Csr& a, const std::vector<float>& x, const std::vector<float>& y0) {
  Expect e;
  e.ax.resize((size_t)a.numRows);
  e.ref.resize((size_t)a.numRows);
  e.mag.resize((size_t)a.numRows);
  for (int r = 0; r < a.numRows; ++r) {
    double s = 0, m = std::fabs((double)y0[(size_t)r]);
    for (int i = a.rowPtrs[(size_t)r]; i < a.rowPtrs[(size_t)r + 1]; ++i) {
      const double p = (double)a.values[(size_t)i] * x[(size_t)a.colIdx[(size_t)i]];
      s += p;
      m += std::fabs(p);
    }
    e.ax[(size_t)r] = (float)s;
    e.ref[(size_t)r] = (float)((double)y0[(size_t)r] + s);
    e.mag[(size_t)r] = (float)m;
  }
  return e;
}

struct Inputs { std::vector<float> x, y0; Expect ex; };
static Inputs inputsFor(const Csr& a, uint32_t seed) {
  Inputs in;
  in.x = pmpp::random_floats((size_t)a.numCols, seed);           // uniform in [-1, 1], never 0
  in.y0 = pmpp::random_floats((size_t)a.numRows, seed + 7919u);
  in.ex = cpuSpmv(a, in.x, in.y0);
  return in;
}

// ---------------------------------------------------------------- uploads
// Owns the device buffers of one test or one benchmark step.
struct Bufs {
  std::vector<Buf*> all;
  Bufs() = default;
  Bufs(const Bufs&) = delete;
  ~Bufs() { for (Buf* b : all) pmpp::release(b); }
  // `lead` copies of leadFill, then v, then tail. Never fewer than one element.
  template <class T>
  Buf* put(const std::vector<T>& v, const std::vector<T>& tail = {}, size_t lead = 0, T leadFill = T()) {
    Buf* b = nullptr;
    if (lead == 0 && tail.empty()) {   // no host copy: the benchmark's arrays can be gigabytes
      b = pmpp::alloc(std::max<size_t>(v.size(), 1) * sizeof(T));
      if (!v.empty()) pmpp::upload(b, v.data(), v.size() * sizeof(T));
    } else {
      std::vector<T> h(lead, leadFill);
      h.insert(h.end(), v.begin(), v.end());
      h.insert(h.end(), tail.begin(), tail.end());
      if (h.empty()) h.push_back(T());
      b = pmpp::alloc(h.size() * sizeof(T));
      pmpp::upload(b, h.data(), h.size() * sizeof(T));
    }
    all.push_back(b);
    return b;
  }
};

static DevCsr putCsr(Bufs& b, const Csr& a, bool guards) {
  std::vector<int> ptrTail, colTail;
  std::vector<float> valTail;
  if (guards) {   // a phantom row numRows + j has one entry: column 0, value 1
    for (int j = 0; j < kGuard; ++j) ptrTail.push_back(a.nnz() + 1 + j);
    colTail.assign(kGuard, 0);
    valTail.assign(kGuard, 1.0f);
  }
  return DevCsr{b.put(a.rowPtrs, ptrTail), b.put(a.colIdx, colTail), b.put(a.values, valTail), a.numRows};
}
static DevEll putEll(Bufs& b, const Ell& e, bool guards) {
  std::vector<int> colTail, nnzTail;
  std::vector<float> valTail;
  if (guards) {
    // A phantom row reads at least its slot 0, index `row`, which lies past the slots when
    // width is 0: the tail then covers a whole stride as well.
    const int tail = kGuard + (e.width == 0 ? e.stride : 0);
    colTail.assign((size_t)tail, 0);
    valTail.assign((size_t)tail, 1.0f);
    nnzTail.assign(kGuard, 1);
  }
  return DevEll{b.put(e.colIdx, colTail), b.put(e.values, valTail), b.put(e.nnzPerRow, nnzTail),
                e.numRows, e.width, e.stride};
}
static DevCoo putCoo(Bufs& b, const Coo& c) {
  return DevCoo{b.put(c.rowIdx), b.put(c.colIdx), b.put(c.values), c.nnz()};
}
static DevX putX(Bufs& b, const std::vector<float>& x, int lead) {
  return DevX{b.put(x, {}, (size_t)lead, kNaN), lead};
}
static Buf* putY(Bufs& b, const std::vector<float>& y0, bool guards) {
  return b.put(y0, guards ? std::vector<float>(kGuard, kSentinel) : std::vector<float>{});
}

// ---------------------------------------------------------------- checking y
static std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char* f, ...) {
  char b[1200];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(b, sizeof b, f, ap);
  va_end(ap);
  return b;
}

// What came back in y, against the reference. Empty string = right.
static std::string checkY(Buf* Y, int numRows, const std::vector<float>& y0, const Expect& ex, bool guards,
                          const char* nanHint) {
  const size_t n = (size_t)numRows + (guards ? kGuard : 0);
  std::vector<float> got(std::max<size_t>(n, 1));
  if (n) pmpp::download(Y, got.data(), n * sizeof(float));
  std::string out;
  auto add = [&](const std::string& s) { out += (out.empty() ? "" : "; ") + s; };

  if (guards) {
    int changed = 0, first = -1;
    for (int g = 0; g < kGuard; ++g) {
      if (std::memcmp(&got[(size_t)numRows + g], &kSentinel, sizeof(float)) != 0) {
        if (first < 0) first = g;
        ++changed;
      }
    }
    if (changed) {
      add(fmt("%d of the %d elements of y after the last row were changed (first: y[numRows + %d] became %g). "
              "A thread whose row is numRows or more must not touch y; the grid is whole blocks, so such "
              "threads exist", changed, kGuard, first, (double)got[(size_t)numRows + first]));
    }
  }

  int nanRows = 0, firstNan = -1, bad = 0, firstBad = -1, assigned = 0, untouched = 0;
  for (int r = 0; r < numRows; ++r) {
    const float v = got[(size_t)r];
    if (std::isnan(v)) {
      if (firstNan < 0) firstNan = r;
      ++nanRows;
      continue;
    }
    const double tol = kRtol * (1.0 + ex.mag[(size_t)r]);
    if (std::fabs((double)v - ex.ref[(size_t)r]) <= tol) continue;
    if (firstBad < 0) firstBad = r;
    ++bad;
    if (std::fabs((double)v - ex.ax[(size_t)r]) <= tol) ++assigned;
    if (std::memcmp(&v, &y0[(size_t)r], sizeof(float)) == 0) ++untouched;
  }
  if (nanRows) add(fmt("y is NaN in %d of %d rows (first: row %d). %s", nanRows, numRows, firstNan, nanHint));
  if (bad) {
    const size_t r = (size_t)firstBad;
    std::string s = fmt("%d of %d rows wrong; first: row %d, got %.7g, expected %.7g (y0 %.7g plus A·x %.7g)",
                        bad, numRows, firstBad, (double)got[r], (double)ex.ref[r], (double)y0[r], (double)ex.ax[r]);
    if (assigned == bad) {
      s += ". Every wrong row holds A·x alone, without y0: the kernel wrote y[row] = sum (as Fig. 17.12 prints) "
           "instead of adding to y";
    } else if (untouched == bad) {
      s += ". Every wrong row still holds y0: the kernel did not run, or wrote nothing";
    }
    add(s);
  }
  return out;
}

// ---------------------------------------------------------------- tests
static int roundUp(int n, int m) { return (n + m - 1) / m * m; }

static std::string testCsr(const Csr& a, const Inputs& in) {
  Bufs b;
  const DevCsr d = putCsr(b, a, true);
  const DevX x = putX(b, in.x, kXLead);
  Buf* Y = putY(b, in.y0, true);
  const std::string e = runCsr(d, x, Y);
  if (!e.empty()) return e;
  return checkY(Y, a.numRows, in.y0, in.ex, true, kCsrNanHint);
}
static std::string testEll(const Csr& a, const Inputs& in, int width, int stride) {
  Bufs b;
  const DevEll d = putEll(b, toEll(a, width, stride), true);
  const DevX x = putX(b, in.x, kXLead);
  Buf* Y = putY(b, in.y0, true);
  const std::string e = runEll(d, x, Y);
  if (!e.empty()) return e;
  return checkY(Y, a.numRows, in.y0, in.ex, true, kEllNanHint);
}
static std::string testHybrid(const Csr& a, const Inputs& in, int k, int stride) {
  Bufs b;
  Coo spill;
  const DevEll d = putEll(b, toEll(a, k, stride, &spill), true);
  const DevCoo c = putCoo(b, spill);
  const DevX x = putX(b, in.x, kXLead);
  Buf* Y = putY(b, in.y0, true);
  const std::string e = runHybrid(c, d, x, Y);
  if (!e.empty()) return e;
  return checkY(Y, a.numRows, in.y0, in.ex, true, kEllNanHint);
}
// The provided kernel alone on every entry, so that a broken copy of provided/ shows.
static std::string testCoo(const Csr& a, const Inputs& in) {
  Bufs b;
  Coo all;
  toEll(a, 0, a.numRows, &all);
  const DevCoo c = putCoo(b, all);
  const DevX x = putX(b, in.x, kXLead);
  Buf* Y = putY(b, in.y0, true);
  const std::string e = runCoo(c, x, Y);
  if (!e.empty()) return e;
  const std::string r = checkY(Y, a.numRows, in.y0, in.ex, true, "");
  return r.empty() ? "" : "the provided COO kernel is wrong (has provided/ been edited?): " + r;
}
// numRows = 0: the launcher must launch (encode) nothing, and nothing may change.
static std::string testNoRows(bool ell) {
  Csr a;
  a.numCols = 5;
  const Inputs in = inputsFor(a, 1699);
  Bufs b;
  const DevX x = putX(b, in.x, kXLead);
  Buf* Y = putY(b, in.y0, true);
  int dispatches = 0;
  pmpp::take_error();
  std::string e = ell ? runEll(putEll(b, toEll(a, 0, 0), true), x, Y, nullptr, &dispatches)
                      : runCsr(putCsr(b, a, true), x, Y, nullptr, &dispatches);
  if (!e.empty()) return e;
  const std::string launch = pmpp::take_error();
  if (!launch.empty()) {
    return fmt("numRows is 0, yet %s launched a kernel, and the launch failed (%s). HIP rejects a grid of zero "
               "blocks and so does the emulator; return before launching", ell ? "launchSpmvEll" : "launchSpmvCsr",
               launch.c_str());
  }
  if (dispatches) {
    return fmt("numRows is 0, yet %s dispatched %d grid(s): when there are no rows, encode nothing",
               ell ? "encodeSpmvEll" : "encodeSpmvCsr", dispatches);
  }
  return checkY(Y, 0, in.y0, in.ex, true, "");
}

// ---------------------------------------------------------------- the benchmark's matrices
static const int kBand = 2048;   // columns within +-2048 of the diagonal: x reads stay local
enum Gen { Uniform, Spread, SpreadSorted, Skewed };
static const Gen kGens[4] = {Uniform, Spread, SpreadSorted, Skewed};
static const char* genName(Gen g) {
  return g == Uniform ? "uniform" : g == Spread ? "spread" : g == SpreadSorted ? "spread_sorted" : "skewed";
}
static Csr benchMatrix(Gen g, int n) {
  if (g == SpreadSorted) return sortedByLength(benchMatrix(Spread, n));
  std::vector<int> lens((size_t)n, 16);   // uniform: every row 16
  Rng r(g == Uniform ? 1701 : g == Spread ? 1702 : 1704);
  for (int i = 0; i < n; ++i) {
    if (g == Spread) lens[(size_t)i] = 1 + r.below(31);         // 1..31, mean 16, random order
    if (g == Skewed && i % 512 == 0) lens[(size_t)i] = 256;      // one row in 512 is 16 times longer
  }
  return makeMatrix(n, n, lens, kBand, g == Uniform ? 11 : g == Spread ? 12 : 14);
}

struct MatrixCase { std::string name; Csr a; bool hybrid, big, cooAlone; };

static std::vector<int> randomLens(int n, int hi, uint32_t seed) {   // uniform on 0..hi
  std::vector<int> l((size_t)n);
  Rng g(seed);
  for (int& v : l) v = g.below(hi + 1);
  return l;
}

static std::vector<MatrixCase> testMatrices() {
  std::vector<MatrixCase> m;
  uint32_t seed = 300;
  auto add = [&](const std::string& name, int rows, int cols, const std::vector<int>& lens, bool hybrid,
                 bool cooAlone = false) {
    m.push_back({fmt("%d x %d, %s", rows, cols, name.c_str()), makeMatrix(rows, cols, lens, -1, ++seed), hybrid,
                 false, cooAlone});
  };
  auto randomRows = [&](int rows, int cols) {
    const int hi = std::min(12, cols);
    return std::make_pair(fmt("random row lengths 0-%d", hi), randomLens(rows, hi, 1000u + (uint32_t)rows));
  };
  if (std::string(pmpp::backend()) == "race") {   // ThreadSanitizer is slow: two small matrices
    for (int rows : {33, 65}) {
      auto r = randomRows(rows, rows);
      add(r.first, rows, rows, r.second, true, rows == 65);
    }
    return m;
  }
  add("one row of 12", 1, 20, {12}, false);
  for (int rows : {7, 33, 65, 129}) {   // none a multiple of the block
    auto r = randomRows(rows, rows);
    add(r.first, rows, rows, r.second, rows != 7);
  }
  {
    auto r = randomRows(300, 305);
    add(r.first + " (rectangular)", 300, 305, r.second, false);
  }
  {
    std::vector<int> l = randomLens(60, 9, 77);
    for (int i = 0; i < 60; ++i) l[(size_t)i] = (i % 3 == 0) ? 0 : 1 + l[(size_t)i] % 9;
    add("every third row empty, the others 1-9 long", 60, 60, l, false);
  }
  add("every row empty (nnz 0)", 20, 20, std::vector<int>(20, 0), false);
  add("every row 5 long", 64, 64, std::vector<int>(64, 5), false);
  {
    std::vector<int> l = randomLens(100, 2, 78);
    for (int& v : l) v += 1;   // 1..3
    l[37] = 40;
    add("rows of 1-3, row 37 of 40", 100, 100, l, true);
  }
  {
    auto r = randomRows(200, 200);
    add(r.first, 200, 200, r.second, true, true);
  }
  if (!pmpp::is_emu()) {   // real devices: the benchmark's generators at its smallest size
    const int n = (1 << 16) + 5;
    for (Gen g : kGens) {
      m.push_back({fmt("%d x %d, the benchmark's '%s' matrix", n, n, genName(g)), benchMatrix(g, n),
                   g == Spread || g == Skewed, true, false});
    }
  }
  return m;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  if (std::string(pmpp::backend()) != "race") {
    t.check(std::string("CSR, numRows 0: ") + kNothing, [] { return testNoRows(false); });
    t.check(std::string("ELL, numRows 0: ") + kNothing, [] { return testNoRows(true); });
  }
  uint32_t seed = 1700;
  for (const MatrixCase& mc : testMatrices()) {
    const Csr& m = mc.a;
    const Inputs in = inputsFor(m, ++seed);
    const std::string what = fmt("%s (nnz %d)", mc.name.c_str(), m.nnz());
    const int n = m.numRows, w = m.maxLen();

    t.check("CSR, " + what, [&] { return testCsr(m, in); });

    std::vector<std::pair<int, std::string>> strides = {
        {n, "numRows"}, {roundUp(n, 8), "numRows rounded up to 8"}, {n + 37, "numRows + 37"}};
    if (mc.big) strides.push_back({roundUp(n, 32), "numRows rounded up to 32"});
    std::vector<int> seen;
    for (const auto& s : strides) {
      if (std::find(seen.begin(), seen.end(), s.first) != seen.end()) continue;   // numRows already a multiple
      seen.push_back(s.first);
      t.check(fmt("ELL, %s; width %d, stride %d (%s)", what.c_str(), w, s.first, s.second.c_str()),
              [&] { return testEll(m, in, w, s.first); });
    }
    if (!mc.big) {
      t.check(fmt("ELL, %s; width %d, longer than any row (whole columns of pads), stride %d", what.c_str(), w + 3,
                  roundUp(n, 8)),
              [&] { return testEll(m, in, w + 3, roundUp(n, 8)); });
    }
    if (mc.hybrid) {
      std::vector<std::pair<int, std::string>> ks = {
          {0, "all entries in COO"}, {1, "1"}, {m.medianLen(), "the median row length"}, {w, "the longest row: no COO part"}};
      std::vector<int> done;
      const int stride = roundUp(n, 8);
      for (const auto& k : ks) {
        if (std::find(done.begin(), done.end(), k.first) != done.end()) continue;
        done.push_back(k.first);
        t.check(fmt("hybrid, %s; K = %d (%s): the provided COO kernel on %d spilled entries, then your ELL kernel, "
                    "stride %d", what.c_str(), k.first, k.second.c_str(), spilledEntries(m, k.first), stride),
                [&] { return testHybrid(m, in, k.first, stride); });
      }
    }
    if (mc.cooAlone) t.check("provided COO kernel alone, " + what, [&] { return testCoo(m, in); });
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// Bytes each format stores, and the least traffic any format could make: every value and
// column once, one pointer or length per row, x once, y read and written.
static double csrBytes(const Csr& a) { return 4.0 * (a.numRows + 1) + 8.0 * a.nnz(); }
static double ellBytes(int numRows, int width, int stride) { return 8.0 * width * (double)stride + 4.0 * numRows; }
static double usefulBytes(const Csr& a) {
  return 8.0 * a.nnz() + 4.0 * a.numRows + 4.0 * a.numCols + 8.0 * a.numRows;   // numCols = numRows here
}

struct Measured { double s = -1; std::string err; };
// Checks one format's result against the CPU reference, then times it: best of reps after a
// warm-up. y starts from y0 for the check; the timed runs keep adding to it.
static Measured measure(const pmpp::Args& a, Buf* Y, const std::vector<float>& y0, const Expect& ex, int n,
                        const std::function<std::string(pmpp::Timing*)>& once) {
  Measured m;
  pmpp::upload(Y, y0.data(), (size_t)n * sizeof(float));
  m.err = once(nullptr);
  if (m.err.empty()) m.err = pmpp::take_error();
  if (m.err.empty()) m.err = checkY(Y, n, y0, ex, false, "");
  if (!m.err.empty()) return m;
  pmpp::Timing tm{0, 0};
  m.s = pmpp::time_it(a.reps, [&] { once(&tm); return tm; }).min_s;
  m.err = pmpp::take_error();
  if (!m.err.empty()) m.s = -1;
  return m;
}

static void emitRow(const pmpp::Args& a, Gen g, const char* format, const Csr& m, double width, double stride,
                    double k, long long coo, double s, double matrixBytes, double pad) {
  pmpp::Json j = pmpp::bench_row(a, "spmv");
  j.s("matrix", genName(g)).s("format", format).n("rows", m.numRows).n("nnz", m.nnz()).n("width", width)
      .n("stride", stride);
  if (k >= 0) j.n("k", k);
  j.n("coo_entries", (double)coo).n("seconds", s).n("matrix_bytes", matrixBytes).n("pad_fraction", pad)
      .n("gbps", usefulBytes(m) / s / 1e9);
  j.emit();
}
static void printRow(Gen g, const char* format, const std::string& width, const std::string& stride, double pad,
                     long long coo, double matrixBytes, double s, const Csr& m) {
  std::printf("  %-14s %-12s %6s %8s %6.1f%% %8lld %10.1f %10.4f %9.1f\n", genName(g), format, width.c_str(),
              stride.c_str(), 100.0 * pad, coo, matrixBytes / 1e6, s * 1e3, usefulBytes(m) / s / 1e9);
}

static int bench(const pmpp::Args& a) {
  const double mem = deviceMemory(), budget = mem > 0 ? mem / 4 : 0, maxBuf = maxBufferBytes();
  // A format runs if its arrays plus x and y take at most a quarter of the memory the device
  // reports, and (Metal) its largest array fits in one buffer.
  auto fits = [&](double total, double largest) {
    return (budget <= 0 || total <= budget) && (maxBuf <= 0 || largest <= maxBuf);
  };
  auto skipped = [&](Gen g, const char* format, double total) {
    std::printf("  %-14s %-12s skipped: needs %.1f MB, more than a quarter of the %.0f MB this device reports "
                "(%s)%s\n", genName(g), format, total / 1e6, mem / 1e6, kMemKey,
                maxBuf > 0 ? ", or more than one buffer may hold" : "");
  };

  // Size: from 2^16 + 5 rows, double while CSR on 'uniform' takes under 2 ms, the row count
  // stays at most 2^21 + 5, and the skewed matrix in ELL (aligned stride) still fits.
  int k = 16;
  double sizingS = 0;
  for (;;) {
    const int n = (1 << k) + 5;
    const Csr u = benchMatrix(Uniform, n);
    Bufs b;
    const DevCsr d = putCsr(b, u, false);
    const DevX x = putX(b, pmpp::random_floats((size_t)n, 5), kBenchXLead);
    Buf* Y = putY(b, pmpp::random_floats((size_t)n, 6), false);
    if (k == 16) {   // untimed work first, so the first timing does not also pay for waking the device
      const auto t0 = std::chrono::steady_clock::now();
      while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.25) runCsr(d, x, Y);
    }
    pmpp::Timing tm{0, 0};
    sizingS = pmpp::time_it(a.reps, [&] { runCsr(d, x, Y, &tm); return tm; }).min_s;
    const std::string err = pmpp::take_error();
    if (!err.empty()) { std::printf("CSR failed while sizing the benchmark: %s\n", err.c_str()); return 1; }
    const int next = (1 << (k + 1)) + 5, nextStride = roundUp(next, 32);
    const double skewedNext = ellBytes(next, 256, nextStride) + 8.0 * next;
    if (sizingS >= 2e-3 || k + 1 > 21 || !fits(skewedNext, 4.0 * 256 * (double)nextStride)) break;
    ++k;
  }
  const int n = (1 << k) + 5;
  std::printf("numRows = numCols = 2^%d + 5 = %d (CSR on 'uniform' took %.3f ms at this size). 2^k + 5 rows, so\n"
              "stride = numRows is not a multiple of 8; ell_aligned rounds the stride up to %d.\n",
              k, n, sizingS * 1e3, roundUp(n, 32));
  std::printf("Columns distinct and sorted within each row, within +-%d of the diagonal; values and x uniform in\n"
              "[-1, 1]. Every result is checked against the CPU first; times are the best of %d runs after a "
              "warm-up.\n", kBand, a.reps);
  if (budget > 0) std::printf("Formats are skipped above a quarter of the %.0f MB this device reports (%s).\n",
                              mem / 1e6, kMemKey);
  std::printf("GB/s counts useful bytes, 8 per entry + 16 per row (a pointer or length, x once, y read and\n"
              "written): the least any format could move, the same for every format, not what moved.\n\n");

  const std::vector<float> xh = pmpp::random_floats((size_t)n, 77), y0 = pmpp::random_floats((size_t)n, 78);
  Bufs vb;
  const DevX X = putX(vb, xh, kBenchXLead);
  Buf* Y = putY(vb, y0, false);

  std::printf("  %-14s %-12s %6s %8s %7s %8s %10s %10s %9s\n", "matrix", "format", "width", "stride", "pad",
              "COO", "matrix MB", "time ms", "GB/s");
  int rc = 0;
  for (Gen g : kGens) {
    const Csr m = benchMatrix(g, n);
    const Expect ex = cpuSpmv(m, xh, y0);
    const int width = m.maxLen();
    double csrS = -1, ellS = -1, alignedS = -1, hybridS = -1;
    std::printf("  %s: nnz %d, row lengths %d to %d, median %d\n", genName(g), m.nnz(), m.minLen(), width,
                m.medianLen());

    // csr
    if (!fits(csrBytes(m) + 8.0 * n, 4.0 * m.nnz())) {
      skipped(g, "csr", csrBytes(m) + 8.0 * n);
    } else {
      Bufs b;
      const DevCsr d = putCsr(b, m, false);
      const Measured r = measure(a, Y, y0, ex, n, [&](pmpp::Timing* tm) { return runCsr(d, X, Y, tm); });
      if (r.s < 0) { std::printf("  %-14s csr: wrong, not timed: %s\n", genName(g), r.err.c_str()); return 1; }
      csrS = r.s;
      printRow(g, "csr", "-", "-", 0, 0, csrBytes(m), csrS, m);
      emitRow(a, g, "csr", m, NAN, NAN, -1, 0, csrS, csrBytes(m), 0);
    }

    // ell (stride = numRows) and ell_aligned (stride a multiple of 32 floats: 128 bytes)
    for (bool aligned : {false, true}) {
      const char* name = aligned ? "ell_aligned" : "ell";
      const int stride = aligned ? roundUp(n, 32) : n;
      const double bytes = ellBytes(n, width, stride);
      if (!fits(bytes + 8.0 * n, 4.0 * width * (double)stride)) { skipped(g, name, bytes + 8.0 * n); continue; }
      Bufs b;
      double pad = 0;
      DevEll d;
      {
        const Ell e = toEll(m, width, stride);   // freed once on the device
        pad = 1.0 - (double)e.entries / ((double)width * stride);
        d = putEll(b, e, false);
      }
      const Measured r = measure(a, Y, y0, ex, n, [&](pmpp::Timing* tm) { return runEll(d, X, Y, tm); });
      if (r.s < 0) { std::printf("  %-14s %s: wrong, not timed: %s\n", genName(g), name, r.err.c_str()); return 1; }
      (aligned ? alignedS : ellS) = r.s;
      printRow(g, name, std::to_string(width), std::to_string(stride), pad, 0, bytes, r.s, m);
      emitRow(a, g, name, m, width, stride, -1, 0, r.s, bytes, pad);
    }

    // hybrid: K = 16, the typical row length; the provided COO pass and your ELL kernel timed together
    if (g == Spread || g == Skewed) {
      const int K = 16, stride = n;
      const long long cooN = spilledEntries(m, K);
      const double bytes = ellBytes(n, K, stride) + 12.0 * cooN;
      if (!fits(bytes + 8.0 * n, std::max(4.0 * K * (double)stride, 4.0 * cooN))) {
        skipped(g, "hybrid", bytes + 8.0 * n);
      } else {
        Bufs b;
        double pad = 0;
        DevEll d;
        DevCoo c;
        {
          Coo spill;
          const Ell e = toEll(m, K, stride, &spill);   // host copies freed once on the device
          pad = 1.0 - (double)e.entries / ((double)K * stride);
          d = putEll(b, e, false);
          c = putCoo(b, spill);
        }
        const Measured r = measure(a, Y, y0, ex, n, [&](pmpp::Timing* tm) { return runHybrid(c, d, X, Y, tm); });
        if (r.s < 0) { std::printf("  %-14s hybrid: wrong, not timed: %s\n", genName(g), r.err.c_str()); return 1; }
        hybridS = r.s;
        printRow(g, "hybrid", std::to_string(K), std::to_string(stride), pad, cooN, bytes, r.s, m);
        emitRow(a, g, "hybrid", m, K, stride, K, cooN, r.s, bytes, pad);
      }
    }

    if (csrS > 0) {
      std::printf("  %-14s CSR time / ELL time:", genName(g));
      if (ellS > 0) std::printf(" %.2f (ell)", csrS / ellS);
      if (alignedS > 0) std::printf(" %.2f (ell_aligned)", csrS / alignedS);
      if (hybridS > 0) std::printf(" %.2f (hybrid)", csrS / hybridS);
      if (g == Skewed && ellS > 0) std::printf("; that is, ELL time / CSR time %.2f (ell)", ellS / csrS);
      std::printf("\n");
    }
  }
  std::printf("These are measurements of this machine at this size, not properties of the formats.\n");
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
