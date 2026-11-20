// Runs one chapter 21 snippet, chosen when it is compiled (-DKIND_SCATTER, -DKIND_GATHER,
// -DKIND_ADJACENT, -DKIND_STRIDED or -DKIND_PRINTED), with the host side p. 517 describes:
// for each slice, the slice is copied in, the atoms go into constant memory one chunk at a
// time with one launch per chunk adding into the slice, and the slice is copied back. The
// result is compared with a float64 sum on the CPU, and nothing may be written outside the
// grid. For the printed Fig. 21.8 (KIND_PRINTED) it checks instead that the result is wrong
// in exactly the way section 21.2 says: at blockDim.x = 32, points 0 to 34 of every block's
// stretch are written (3 to 31 four times) and 35 to 127 never are.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "check.h"

#if !defined(KIND_SCATTER) && !defined(KIND_GATHER) && !defined(KIND_ADJACENT) && !defined(KIND_STRIDED) && \
    !defined(KIND_PRINTED)
#define KIND_GATHER  // the CUDA front-end check parses this file with no KIND given
#endif
#ifndef CHUNK_SIZE
#define CHUNK_SIZE 256
#endif
extern __constant__ float atoms[CHUNK_SIZE * 4];
__global__ void cenergy(float* energygrid, dim3 grid, float gridspacing, float z, int numatoms);

static const float kSpacing = 0.5f;  // the book's 0.5 (p. 525): z / gridspacing is exact
static const int kPad = 64;          // floats after the grid that must stay untouched

struct Problem {
    int W, H, slices, natoms;
    std::vector<float> xyzq;  // four floats per atom
};

static Problem makeProblem(int W, int H, int slices, int natoms, uint32_t seed) {
    Problem p{W, H, slices, natoms, std::vector<float>(4 * natoms)};
    uint32_t s = seed;
    auto next = [&s]() { s = s * 1664525u + 1013904223u; return (float)((s >> 8) & 0xffff) / 65536.0f; };
    for (int a = 0; a < natoms; ++a) {
        p.xyzq[4 * a + 0] = next() * W * kSpacing;
        p.xyzq[4 * a + 1] = next() * H * kSpacing;
        p.xyzq[4 * a + 2] = (float)slices * kSpacing + 0.25f + next();  // above every slice: no zero distance
        p.xyzq[4 * a + 3] = next() * 2.0f - 1.0f;
    }
    return p;
}

// float64 sums, and the sum of |contribution| at each point (the scale for the tolerance).
static void reference(const Problem& p, std::vector<double>& ref, std::vector<double>& scale) {
    ref.assign((size_t)p.W * p.H * p.slices, 0.0);
    scale.assign(ref.size(), 0.0);
    for (int k = 0; k < p.slices; ++k)
        for (int j = 0; j < p.H; ++j)
            for (int i = 0; i < p.W; ++i)
                for (int a = 0; a < p.natoms; ++a) {
                    double dx = i * (double)kSpacing - p.xyzq[4 * a];
                    double dy = j * (double)kSpacing - p.xyzq[4 * a + 1];
                    double dz = k * (double)kSpacing - p.xyzq[4 * a + 2];
                    double c = p.xyzq[4 * a + 3] / std::sqrt(dx * dx + dy * dy + dz * dz);
                    size_t at = ((size_t)k * p.H + j) * p.W + i;
                    ref[at] += c;
                    scale[at] += std::fabs(c);
                }
}

static dim3 gridFor(const Problem& p, dim3 block, int inChunk) {
#if defined(KIND_SCATTER)
    (void)p;
    return dim3((inChunk + block.x - 1) / block.x);  // one thread per atom of the chunk
#elif defined(KIND_GATHER)
    (void)inChunk;
    return dim3((p.W + block.x - 1) / block.x, (p.H + block.y - 1) / block.y);
#else
    (void)inChunk;  // coarsened by 4: a block covers 4 * blockDim.x points of a row
    return dim3((p.W + 4 * block.x - 1) / (4 * block.x), (p.H + block.y - 1) / block.y);
#endif
}

// The host side of p. 517. Returns the grid followed by kPad floats that must stay 0.
static std::vector<float> run(const Problem& p, dim3 block) {
    const size_t n = (size_t)p.W * p.H * p.slices;
    std::vector<float> host(n + kPad, 0.0f);
    float* d = nullptr;
    cudaMalloc((void**)&d, host.size() * sizeof(float));
    cudaMemcpy(d + n, host.data() + n, kPad * sizeof(float), cudaMemcpyHostToDevice);
    for (int k = 0; k < p.slices; ++k) {
        const size_t off = (size_t)k * p.W * p.H;
        cudaMemcpy(d + off, host.data() + off, (size_t)p.W * p.H * sizeof(float), cudaMemcpyHostToDevice);
        for (int first = 0; first < p.natoms; first += CHUNK_SIZE) {
            int inChunk = std::min(CHUNK_SIZE, p.natoms - first);
            cudaMemcpyToSymbol(atoms, p.xyzq.data() + 4 * first, (size_t)inChunk * 4 * sizeof(float));
            cenergy<<<gridFor(p, block, inChunk), block>>>(d, dim3(p.W, p.H, p.slices), kSpacing, k * kSpacing, inChunk);
        }
        cudaMemcpy(host.data() + off, d + off, (size_t)p.W * p.H * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(host.data() + n, d + n, kPad * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);
    return host;
}

#if !defined(KIND_PRINTED)
static void checkCase(const char* kind, int W, int H, int slices, int natoms, dim3 block) {
    Problem p = makeProblem(W, H, slices, natoms, 2021u + W * 7 + H);
    std::vector<double> ref, scale;
    reference(p, ref, scale);
    for (int o = 0; o < kOrders; ++o) {
        const char* order = useOrder(o);
        std::vector<float> got = run(p, block);
        bool close = true;
        for (size_t at = 0; at < ref.size(); ++at)
            if (std::fabs(got[at] - ref[at]) > 1e-5 * scale[at] + 1e-12) close = false;
        bool clean = std::all_of(got.begin() + ref.size(), got.end(), [](float v) { return v == 0.0f; });
        char what[240];
        std::snprintf(what, sizeof what,
                      "%s: %d x %d points, %d slices, %d atoms in chunks of %d, blocks of %u x %u, %s: "
                      "matches the float64 sum, nothing written past the grid",
                      kind, W, H, slices, natoms, CHUNK_SIZE, block.x, block.y, order);
        expect(close && clean, what);
    }
}
#endif

int main() {
#if defined(KIND_SCATTER)
    std::printf("scatter.cu (Fig. 21.5 with an atom bound)\n");
    checkCase("scatter", 37, 5, 2, 150, dim3(32));
    checkCase("scatter", 16, 16, 1, 9, dim3(4));
    checkCase("scatter", 8, 3, 1, CHUNK_SIZE, dim3(64));
#elif defined(KIND_GATHER)
    std::printf("gather.cu (Fig. 21.6 with bound checks)\n");
    checkCase("gather", 77, 5, 2, 150, dim3(32, 2));
    checkCase("gather", 16, 16, 1, 9, dim3(8, 8));
    checkCase("gather", 33, 3, 1, 70, dim3(16, 4));
#elif defined(KIND_ADJACENT) || defined(KIND_STRIDED)
#if defined(KIND_ADJACENT)
    const char* kind = "coarsen_adjacent";
    std::printf("coarsen_adjacent.cu (Fig. 21.8 corrected)\n");
#else
    const char* kind = "coarsen_strided";
    std::printf("coarsen_strided.cu (Fig. 21.10 with bound checks)\n");
#endif
    checkCase(kind, 77, 5, 2, 150, dim3(16, 2));
    checkCase(kind, 128, 2, 1, 40, dim3(32, 1));
    checkCase(kind, 130, 3, 1, 70, dim3(32, 2));
    checkCase(kind, 9, 9, 1, 12, dim3(4, 4));
#elif defined(KIND_PRINTED)
    std::printf("coarsen_printed.cu (Fig. 21.8 as printed: expected to be wrong)\n");
    const int W = 128, H = 2;
    Problem p = makeProblem(W, H, 1, 20, 77u);
    std::vector<double> ref, scale;
    reference(p, ref, scale);
    for (int o = 0; o < kOrders; ++o) {
        const char* order = useOrder(o);
        std::vector<float> got = run(p, dim3(32, 1));
        bool tailUntouched = true, fourTimes = true, edges = true, wrong = false;
        for (int j = 0; j < H; ++j)
            for (int i = 0; i < W; ++i) {
                double r = ref[(size_t)j * W + i], g = got[(size_t)j * W + i], tol = 4e-5 * scale[(size_t)j * W + i] + 1e-12;
                int writers = i < 3 ? i + 1 : i <= 31 ? 4 : i <= 34 ? 35 - i : 0;
                if (i >= 35 && g != 0.0) tailUntouched = false;
                if (i >= 3 && i <= 31 && std::fabs(g - 4 * r) > tol) fourTimes = false;
                if ((i < 3 || (i >= 32 && i <= 34)) && std::fabs(g - writers * r) > tol) edges = false;
                if (std::fabs(g - r) > 1e-5 * scale[(size_t)j * W + i]) wrong = true;
            }
        char what[200];
        std::snprintf(what, sizeof what, "128 x 2 points, blocks of 32, %s: points 35 to 127 of each row never written", order);
        expect(tailUntouched, what);
        std::snprintf(what, sizeof what, "%s: points 3 to 31 written four times (four times the true sum, as every += lands here)", order);
        expect(fourTimes, what);
        std::snprintf(what, sizeof what, "%s: points 0, 1, 2 and 34, 33, 32 written 1, 2, 3 times", order);
        expect(edges, what);
        std::snprintf(what, sizeof what, "%s: the result differs from the float64 sum", order);
        expect(wrong, what);
    }
#endif
    return finish();
}
