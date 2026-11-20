// Lab 21.1 driver: tests your three direct-Coulomb-summation kernels (scatter, gather,
// coarsened) and the provided adjacent kernel against a CPU reference, then times them.
// You do not need to edit this file; your work goes in work/.
//
// What a test case does (README, "What the tests do"):
//   - The grid holds nz slices in exactly nz * ny * nx floats, so on `emu` a write past its
//     end is an AddressSanitizer report naming the line.
//   - Slice k starts at 0. Every other slice holds a sentinel that must survive bit for bit.
//   - Atoms: x and y uniform over the slice plus a margin of two spacings; z = (m + 1/2) *
//     spacing for a whole number m, so every atom is at least half a spacing from every
//     slice plane and no distance is zero; charges uniform in [-1, 1], both signs.
//   - Each point of slice k is compared with the sum in double:
//         |got - ref| <= 1e-4 * (sum over atoms of |charge| / distance)
//     Never bitwise: Metal does not default to strict IEEE arithmetic (ch21-facts C3), and
//     float atomics add in no fixed order (ch21-facts A6).
#include "pmpp/harness.hpp"
#include "dcs.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using pmpp::Buf;

enum Kind { kScatter = 0, kGather = 1, kCoarsened = 2, kAdjacent = 3 };
static const char* kindName(int kind) {
    static const char* names[] = {"scatter", "gather", "coarse_strided", "coarse_adjacent"};
    return names[kind];
}
static const char* kindWhat(int kind) {
    static const char* what[] = {"scatter (yours, Fig. 21.5)", "gather (yours, Fig. 21.6)",
                                 "coarsened (yours, Fig. 21.10)", "coarse_adjacent (provided, Fig. 21.8 corrected)"};
    return what[kind];
}
// Threads per block (threadgroup) that each launch uses (dcs.h).
static int threadsNeeded(int kind) { return kind == kScatter ? BLOCK_X : BLOCK_X * BLOCK_Y; }

// ---------------------------------------------------------------- backends
// Each backend section defines:
//   kernelProblem(kind)   empty if that kernel can run here, else why not
//   enqueue(kind, pass, ...)  one full slice: every chunk of every atom
//   groupWidth(), widthSource()  the warp (SIMD-group) width, and where it came from
//   kWantDeviceAtoms      whether the launches read the atoms from a device buffer
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
using pmpp::metal::buffer;

static const bool kWantDeviceAtoms = true;   // Metal: one buffer, bound chunk by chunk

static id<MTLComputePipelineState> pso(int kind) {
    const std::vector<std::pair<std::string, std::string>> macros = {{"COARSEN", std::to_string(COARSEN)}};
    switch (kind) {
        case kScatter: return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "dcs_scatter", macros);
        case kGather: return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "dcs_gather", macros);
        case kCoarsened: return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "dcs_coarsened", macros);
        default:
            return pmpp::metal::pipeline(std::string(PMPP_METAL_SHARED_DIR) + "/coarse_adjacent.metal",
                                         "dcs_coarse_adjacent", macros);
    }
}
static std::string kernelProblem(int kind) {
    id<MTLComputePipelineState> p = pso(kind);
    if (!p) return std::string(kindName(kind)) + ": the Metal kernel did not build: " + pmpp::take_error();
    if (p.maxTotalThreadsPerThreadgroup < (NSUInteger)threadsNeeded(kind))
        return std::string(kindName(kind)) + ": this pipeline allows " +
               std::to_string((int)p.maxTotalThreadsPerThreadgroup) + " threads per threadgroup; the launch uses " +
               std::to_string(threadsNeeded(kind));
    return "";
}
static void enqueue(int kind, void* vp, Buf* G, const std::vector<float>&, Buf* A, int nx, int ny, int nz, float h,
                    int k, int natoms, int chunk) {
    id<MTLComputePipelineState> p = pso(kind);
    if (!p) return;
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)vp)->encoder;
    switch (kind) {
        case kScatter: encodeDcsScatter(enc, p, buffer(G), nx, ny, nz, h, k, buffer(A), natoms, chunk); break;
        case kGather: encodeDcsGather(enc, p, buffer(G), nx, ny, nz, h, k, buffer(A), natoms, chunk); break;
        case kCoarsened: encodeDcsCoarsened(enc, p, buffer(G), nx, ny, nz, h, k, buffer(A), natoms, chunk); break;
        default: encodeDcsCoarseAdjacent(enc, p, buffer(G), nx, ny, nz, h, k, buffer(A), natoms, chunk); break;
    }
}
// Apple documents no fixed SIMD-group width; the pipeline reports it (ch05-facts C3).
static int groupWidth() {
    id<MTLComputePipelineState> p = pso(kCoarsened);
    return p ? (int)p.threadExecutionWidth : 0;
}
static std::string widthSource() { return "threadExecutionWidth of the dcs_coarsened pipeline"; }

#else  // CUDA, HIP and the emulators
static const bool kWantDeviceAtoms = false;  // CUDA: the launches copy from host memory

static std::string kernelProblem(int kind) {
    const int limit = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
    if (limit < threadsNeeded(kind))
        return std::string(kindName(kind)) + ": this device allows " + std::to_string(limit) +
               " threads per block; the launch uses " + std::to_string(threadsNeeded(kind));
    return "";
}
static void enqueue(int kind, void*, Buf* G, const std::vector<float>& atoms, Buf*, int nx, int ny, int nz, float h,
                    int k, int natoms, int chunk) {
    float* g = (float*)pmpp::raw(G);
    const float* a = atoms.data();   // host memory: the launch copies each chunk to constant memory
    switch (kind) {
        case kScatter: launchDcsScatter(g, nx, ny, nz, h, k, a, natoms, chunk); break;
        case kGather: launchDcsGather(g, nx, ny, nz, h, k, a, natoms, chunk); break;
        case kCoarsened: launchDcsCoarsened(g, nx, ny, nz, h, k, a, natoms, chunk); break;
        default: launchDcsCoarseAdjacent(g, nx, ny, nz, h, k, a, natoms, chunk); break;
    }
}
static int groupWidth() { return (int)pmpp::device_prop("warpSize", 0); }
static std::string widthSource() { return "warpSize from the device properties"; }
#endif

// ---------------------------------------------------------------- a problem on the device
static const float kSentinel = -77777.0f;   // no correct launch writes outside slice k
static const double kRelTol = 1e-4;         // times sum |charge| / distance, per point

struct Case {
    int nx, ny, nz, k, natoms, chunk;
    float h;            // grid spacing
    const char* why;
};

static std::string caseText(const Case& c) {
    char b[300];
    std::snprintf(b, sizeof b, "%d x %d, slice %d of %d, %d atom%s, chunk %d, spacing %g  (%s)", c.nx, c.ny, c.k, c.nz,
                  c.natoms, c.natoms == 1 ? "" : "s", c.chunk, (double)c.h, c.why);
    return b;
}

// x, y, z, charge per atom. Deterministic for a seed, on every platform.
static std::vector<float> makeAtoms(const Case& c, uint32_t seed) {
    const std::vector<float> r = pmpp::random_floats((size_t)c.natoms * 4, seed);   // each in [-1, 1]
    std::vector<float> a((size_t)c.natoms * 4);
    for (int n = 0; n < c.natoms; ++n) {
        const float ux = 0.5f * (r[4 * n] + 1.0f), uy = 0.5f * (r[4 * n + 1] + 1.0f), uz = 0.5f * (r[4 * n + 2] + 1.0f);
        a[4 * n] = (-2.0f + ux * (float)(c.nx + 3)) * c.h;    // from 2 spacings before the slice to 2 past it
        a[4 * n + 1] = (-2.0f + uy * (float)(c.ny + 3)) * c.h;
        int m = -2 + (int)std::floor(uz * (float)(c.nz + 4));  // a plane index from -2 to nz + 1
        m = std::min(std::max(m, -2), c.nz + 1);
        a[4 * n + 2] = ((float)m + 0.5f) * c.h;                // half-way between two slice planes
        a[4 * n + 3] = r[4 * n + 3];                           // charge, either sign
    }
    return a;
}

// The potential at every point of slice k, and sum |charge| / distance (the tolerance's
// scale), both in double. The point coordinates are i * h, j * h, k * h, as in the kernels.
static void cpuSlice(const Case& c, const std::vector<float>& atoms, std::vector<double>& ref,
                     std::vector<double>& mag) {
    ref.assign((size_t)c.nx * c.ny, 0.0);
    mag.assign((size_t)c.nx * c.ny, 0.0);
    const double h = c.h, z = h * c.k;
    for (int n = 0; n < c.natoms; ++n) {
        const double ax = atoms[4 * n], ay = atoms[4 * n + 1], az = atoms[4 * n + 2], q = atoms[4 * n + 3];
        const double dz2 = (z - az) * (z - az);
        for (int j = 0; j < c.ny; ++j) {
            const double dy = h * j - ay, dyz = dy * dy + dz2;
            double* rr = ref.data() + (size_t)j * c.nx;
            double* mm = mag.data() + (size_t)j * c.nx;
            for (int i = 0; i < c.nx; ++i) {
                const double dx = h * i - ax;
                const double inv = 1.0 / std::sqrt(dx * dx + dyz);
                rr[i] += q * inv;
                mm[i] += std::fabs(q) * inv;
            }
        }
    }
}

struct Problem {
    Case c;
    std::vector<float> atoms;
    std::vector<double> ref, mag;
    Buf* G = nullptr;   // the grid, nz * ny * nx floats exactly
    Buf* A = nullptr;   // the atoms on the device (Metal only)

    Problem(const Case& cs, uint32_t seed, bool withReference) : c(cs) {
        atoms = makeAtoms(c, seed);
        if (withReference) cpuSlice(c, atoms, ref, mag);
        G = pmpp::alloc(gridFloats() * 4);
        if (kWantDeviceAtoms) {
            A = pmpp::alloc(atoms.size() * 4);
            pmpp::upload(A, atoms.data(), atoms.size() * 4);
        }
    }
    Problem(const Problem&) = delete;
    ~Problem() {
        pmpp::release(G);
        if (A) pmpp::release(A);
    }
    size_t gridFloats() const { return (size_t)c.nz * c.ny * c.nx; }
    size_t sliceFloats() const { return (size_t)c.ny * c.nx; }

    // Slice k = 0, every other slice = the sentinel.
    void reset() {
        std::vector<float> v(gridFloats(), kSentinel);
        std::fill(v.begin() + (size_t)c.k * sliceFloats(), v.begin() + (size_t)(c.k + 1) * sliceFloats(), 0.0f);
        pmpp::upload(G, v.data(), v.size() * 4);
    }
    pmpp::Timing run(int kind, int chunk) {
        return pmpp::run([&](void* p) { enqueue(kind, p, G, atoms, A, c.nx, c.ny, c.nz, c.h, c.k, c.natoms, chunk); });
    }
    std::vector<float> grid() {
        std::vector<float> g(gridFloats());
        pmpp::download(G, g.data(), g.size() * 4);
        return g;
    }

    // Empty if slice k of `got` matches `want` (the reference, or another kernel's result)
    // within the tolerance and every other slice still holds the sentinel.
    std::string compare(const std::vector<float>& got, const std::function<double(size_t)>& want,
                        const char* wantName) const {
        std::string err;
        const size_t off = (size_t)c.k * sliceFloats();
        size_t bad = 0, first = 0;
        for (size_t p = 0; p < sliceFloats(); ++p) {
            const double e = std::fabs((double)got[off + p] - want(p));
            if (!(e <= kRelTol * mag[p])) {   // also catches NaN
                if (bad == 0) first = p;
                ++bad;
            }
        }
        if (bad) {
            char b[400];
            std::snprintf(b, sizeof b,
                          "%zu of the %zu points of slice %d are wrong; the first at x %zu, y %zu: got %.7g, %s %.7g "
                          "(tolerance %.3g)",
                          bad, sliceFloats(), c.k, first % c.nx, first / c.nx, (double)got[off + first], wantName,
                          want(first), kRelTol * mag[first]);
            err = b;
        }
        size_t changed = 0, firstChanged = 0;
        for (size_t q = 0; q < gridFloats(); ++q) {
            if (q >= off && q < off + sliceFloats()) continue;
            if (std::memcmp(&got[q], &kSentinel, 4) != 0) {
                if (changed == 0) firstChanged = q;
                ++changed;
            }
        }
        if (changed) {
            const size_t s = firstChanged / sliceFloats(), r = firstChanged % sliceFloats();
            char b[300];
            std::snprintf(b, sizeof b,
                          "%s%zu floats outside slice %d were overwritten; the first is slice %zu, x %zu, y %zu",
                          err.empty() ? "" : "; ", changed, c.k, s, r % c.nx, r / c.nx);
            err += b;
        }
        return err;
    }
    std::string checkAgainstReference(const std::vector<float>& got) const {
        return compare(got, [&](size_t p) { return ref[p]; }, "expected");
    }
};

// ---------------------------------------------------------------- tests
static std::vector<Case> testCases() {
    std::vector<Case> cs = {
        {37, 11, 1, 0, 150, 64, 0.5f, "three launches of 64, 64 and 22 atoms; rows of 37 points"},
        {300, 5, 1, 0, 150, 64, 0.5f, "a row is one whole span of 4 x 64 points and 44 more"},
        {37, 11, 3, 1, 150, 64, 0.5f, "slices 0 and 2 hold a sentinel that must survive"},
        {300, 5, 3, 2, 1, 64, 0.25f, "a single atom, in the last slice"},
        {37, 11, 2, 0, 150, CHUNK_MAX, 0.5f, "all atoms in one launch"},
    };
    if (!pmpp::is_emu()) cs.push_back({1000, 333, 1, 0, 2000, 1024, 0.5f, "two launches, of 1,024 and 976 atoms"});
    return cs;
}

static int tests(const pmpp::Args& a) {
    pmpp::Tests t(a);
    const std::vector<Case> cs = testCases();
    std::vector<Problem*> ps;
    for (size_t n = 0; n < cs.size(); ++n) ps.push_back(new Problem(cs[n], 2100u + 37u * (uint32_t)n, true));

    for (int kind : {kScatter, kGather, kCoarsened, kAdjacent}) {
        for (Problem* p : ps) {
            t.check(std::string(kindWhat(kind)) + ": " + caseText(p->c), [&]() -> std::string {
                const std::string why = kernelProblem(kind);
                if (!why.empty()) return why;
                p->reset();
                p->run(kind, p->c.chunk);
                return p->checkAgainstReference(p->grid());
            });
        }
    }
    // Gather and coarsened compute the same sums, each point's atoms in the same order; they
    // must agree with each other as closely as each agrees with the reference. (Two kernels
    // that write nothing agree too, so gather must first match the reference.)
    for (Problem* p : ps) {
        t.check("gather and coarsened agree: " + caseText(p->c), [&]() -> std::string {
            for (int kind : {kGather, kCoarsened}) {
                const std::string why = kernelProblem(kind);
                if (!why.empty()) return why;
            }
            p->reset();
            p->run(kGather, p->c.chunk);
            const std::vector<float> g = p->grid();
            const std::string gatherErr = p->checkAgainstReference(g);
            if (!gatherErr.empty()) return "gather does not match the reference yet: " + gatherErr;
            p->reset();
            p->run(kCoarsened, p->c.chunk);
            const std::vector<float> co = p->grid();
            const size_t off = (size_t)p->c.k * p->sliceFloats();
            return p->compare(co, [&](size_t q) { return (double)g[off + q]; }, "gather gave");
        });
    }
    for (Problem* p : ps) delete p;
    return t.finish();
}

// ---------------------------------------------------------------- benchmark
struct Timed {
    int kind;
    double best = std::numeric_limits<double>::infinity();
};

// One untimed round, then `reps` rounds in which the kernels take turns; each keeps its
// best. A "run" is one whole slice: every chunk of every atom.
static void timeAll(const pmpp::Args& a, Problem& p, int chunk, std::vector<Timed>& ks) {
    for (Timed& k : ks) p.run(k.kind, chunk);
    for (int r = 0; r < a.reps; ++r)
        for (Timed& k : ks) k.best = std::min(k.best, p.run(k.kind, chunk).best());
}

static void emitRow(const pmpp::Args& a, const char* comparison, const Case& c, int chunk, int kind, double s,
                    int width) {
    const double pairs = (double)c.nx * c.ny * c.natoms;
    pmpp::bench_row(a, kindName(kind)).s("comparison", comparison).n("nx", c.nx).n("ny", c.ny).n("atoms", c.natoms)
        .n("chunk", chunk).n("seconds", s).n("pairs_per_s", pairs / s).n("group_width", width).emit();
}

static int bench(const pmpp::Args& a) {
    for (int kind : {kScatter, kGather, kCoarsened, kAdjacent}) {
        const std::string why = kernelProblem(kind);
        if (!why.empty()) {
            std::printf("bench: %s\n", why.c_str());
            return 1;
        }
    }
    const int width = groupWidth();
    std::printf("group width: %d (%s)\n", width, widthSource().c_str());
    std::printf("Each time is one whole slice (every chunk), the best of %d runs. A pair is one (atom, point).\n",
                a.reps);

    // Comparison A: small, because scatter has one thread per atom of a chunk and each of
    // them walks the whole slice.
    {
        const Case c = {256, 256, 1, 0, 1024, 256, 0.5f, "comparison A"};
        Problem p(c, 2150u, false);
        p.reset();
        std::vector<Timed> ks = {{kScatter}, {kGather}};
        timeAll(a, p, c.chunk, ks);
        const std::string err = pmpp::take_error();
        if (!err.empty()) {
            std::printf("comparison A: a launch failed: %s\n", err.c_str());
            return 1;
        }
        std::printf("\nA. scatter against gather: %d x %d slice, %d atoms, chunk %d (%d launches)\n", c.nx, c.ny,
                    c.natoms, c.chunk, (c.natoms + c.chunk - 1) / c.chunk);
        std::printf("   %-16s %12s %12s\n", "kernel", "ms/slice", "Gpairs/s");
        for (const Timed& k : ks) {
            std::printf("   %-16s %12.4f %12.3f\n", kindName(k.kind), k.best * 1e3,
                        (double)c.nx * c.ny * c.natoms / k.best / 1e9);
            emitRow(a, "A", c, c.chunk, k.kind, k.best, width);
        }
        std::printf("   gather ran %.1f times as fast as scatter on this device\n", ks[0].best / ks[1].best);
    }

    // Comparison B: gather, the provided adjacent kernel and your strided kernel, at four
    // chunk sizes. 512 atoms are 8 KB, NVIDIA's documented constant-cache working set per
    // SM (ch21-facts A5); whether anything changes there is for you to see.
    {
        const Case c = {1024, 1024, 1, 0, 4096, 0, 0.5f, "comparison B"};
        Problem p(c, 2151u, false);
        p.reset();
        std::printf("\nB. gather, coarse_adjacent and coarse_strided: %d x %d slice, %d atoms\n", c.nx, c.ny, c.natoms);
        std::printf("   %-6s %-16s %12s %12s %16s\n", "chunk", "kernel", "ms/slice", "Gpairs/s", "strided/adjacent");
        for (int chunk : {64, 256, 512, 1024}) {
            std::vector<Timed> ks = {{kGather}, {kAdjacent}, {kCoarsened}};
            timeAll(a, p, chunk, ks);
            const std::string err = pmpp::take_error();
            if (!err.empty()) {
                std::printf("   chunk %d: a launch failed: %s\n", chunk, err.c_str());
                continue;
            }
            for (const Timed& k : ks) {
                char ratio[32] = "";
                if (k.kind == kCoarsened) std::snprintf(ratio, sizeof ratio, "%.2fx", ks[1].best / k.best);
                std::printf("   %-6d %-16s %12.4f %12.3f %16s\n", chunk, kindName(k.kind), k.best * 1e3,
                            (double)c.nx * c.ny * c.natoms / k.best / 1e9, ratio);
                emitRow(a, "B", c, chunk, k.kind, k.best, width);
            }
        }
        std::printf("   strided/adjacent is how many times as fast the strided kernel ran as the adjacent one.\n");
    }
    std::printf("\nThese are measurements of this device at these sizes, not properties of the kernels.\n");
    return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
