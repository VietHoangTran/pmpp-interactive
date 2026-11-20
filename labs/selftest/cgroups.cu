// Self-test: the cooperative-groups subset.
#include <cooperative_groups.h>
#include "selftest_cu.h"
namespace cg = cooperative_groups;

// ---- thread_block: the same information as the built-in variables, and a barrier ------------
__global__ void kBlock(int* rank, int* ok, int* swapped) {
    __shared__ int stage[96];
    cg::thread_block block = cg::this_thread_block();
    int r = (int)block.thread_rank();
    int g = blockIdx.x * 96 + r;
    rank[g] = r;
    dim3 ti = block.thread_index(), gi = block.group_index(), dt = block.dim_threads();
    ok[g] = ti.x == threadIdx.x && ti.y == threadIdx.y && ti.z == threadIdx.z &&
            gi.x == blockIdx.x && gi.y == blockIdx.y && gi.z == blockIdx.z &&
            dt.x == blockDim.x && dt.y == blockDim.y && dt.z == blockDim.z &&
            block.size() == 96 && block.num_threads() == 96 &&
            r == (int)(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);
    stage[r] = g;
    block.sync();
    swapped[g] = stage[95 - r];
    cg::sync(block);
}
static std::string tBlock() {
    const int blocks = 3, T = 96, n = blocks * T;
    StBuf<int> rank(n, -1), ok(n, 0), swapped(n, -1);
    kBlock<<<blocks, dim3(8, 4, 3)>>>(rank.dev, ok.dev, swapped.dev);
    ST_LAUNCHED();
    std::vector<int> r = rank.host(), o = ok.host(), s = swapped.host();
    for (int g = 0; g < n; ++g) {
        ST_REQUIRE(r[g] == g % T, "thread_rank is %d for thread %d", r[g], g % T);
        ST_REQUIRE(o[g] == 1, "thread_block disagrees with threadIdx/blockIdx/blockDim in thread %d", g);
        ST_REQUIRE(s[g] == g / T * T + 95 - g % T, "after sync thread %d read %d", g, s[g]);
    }
    return "";
}

// ---- tiled_partition<32>: one tile per warp, and the reduction built on it -----------------
__global__ void kTile(const int* in, int* sums, int* meta) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int v = in[g];
    for (int offset = (int)warp.size() / 2; offset > 0; offset /= 2) v += warp.shfl_down(v, offset);
    int lead = warp.shfl(g, 0);                 // global index of the tile's rank-0 thread
    if (warp.thread_rank() == 0) sums[g / 32] = v;
    meta[g] = (int)warp.meta_group_rank() * 1000 + (int)warp.meta_group_size() * 100 + (lead == g / 32 * 32) * 10 +
              (warp.all(1) != 0) + (warp.any(0) != 0);
}
__global__ void kTile8(int* out) {
    cg::thread_block_tile<8> tile = cg::tiled_partition<8>(cg::this_thread_block());
    int t = threadIdx.x;
    unsigned b = tile.ballot(t % 3 == 0);       // bit r = the tile's rank-r thread
    int x = tile.shfl_xor(t, 1);
    int u = tile.shfl_up(t, 3);
    tile.sync();
    out[t] = (int)b * 10000 + x * 100 + u;
}
static std::string tTile() {
    const int blocks = 2, T = 128, n = blocks * T;
    std::vector<int> h(n);
    for (int i = 0; i < n; ++i) h[i] = (i * 11) % 53 - 20;
    StBuf<int> in(h), sums(n / 32, 0), meta(n, -1);
    kTile<<<blocks, T>>>(in.dev, sums.dev, meta.dev);
    ST_LAUNCHED();
    std::vector<int> s = sums.host(), m = meta.host();
    for (int w = 0; w < n / 32; ++w) {
        int want = 0;
        for (int i = 0; i < 32; ++i) want += h[w * 32 + i];
        ST_REQUIRE(s[w] == want, "tile %d sums to %d, expected %d", w, s[w], want);
    }
    for (int g = 0; g < n; ++g) {
        int want = (g % T / 32) * 1000 + 4 * 100 + 10 + 1 + 0;
        ST_REQUIRE(m[g] == want, "tile properties in thread %d are %d, expected %d", g, m[g], want);
    }
    StBuf<int> out(64, -1);
    kTile8<<<1, 64>>>(out.dev);
    ST_LAUNCHED();
    std::vector<int> o = out.host();
    for (int t = 0; t < 64; ++t) {
        unsigned b = 0;
        for (int r = 0; r < 8; ++r) if ((t / 8 * 8 + r) % 3 == 0) b |= 1u << r;
        int want = (int)b * 10000 + (t ^ 1) * 100 + (t % 8 >= 3 ? t - 3 : t);
        ST_REQUIRE(o[t] == want, "8-thread tile, thread %d: got %d, expected %d", t, o[t], want);
    }
    return "";
}

// ---- coalesced_threads: the threads of a warp that took the same branch ---------------------
__global__ void kCoalesced(int* size, int* rank, int* fromFirst, unsigned* votes) {
    int t = threadIdx.x;
    size[t] = rank[t] = fromFirst[t] = -1;
    if (t % 4 == 1) {                           // lanes 1, 5, 9, ... of each warp
        cg::coalesced_group active = cg::coalesced_threads();
        size[t] = (int)active.size();
        rank[t] = (int)active.thread_rank();
        fromFirst[t] = active.shfl(t, 0);       // the value held by the group's rank-0 thread
        votes[t] = active.ballot(t % 8 == 1);   // packed by rank
        active.sync();
    }
}
static std::string tCoalesced() {
    const int T = 64;
    StBuf<int> size(T), rank(T), first(T); StBuf<unsigned> votes(T, 0u);
    kCoalesced<<<1, T>>>(size.dev, rank.dev, first.dev, votes.dev);
    ST_LAUNCHED();
    std::vector<int> s = size.host(), r = rank.host(), f = first.host(); std::vector<unsigned> v = votes.host();
    for (int t = 0; t < T; ++t) {
        if (t % 4 != 1) { ST_REQUIRE(s[t] == -1, "thread %d should not have joined the group", t); continue; }
        // The size a GPU reports may be smaller; the ranks must be consistent with it.
        ST_REQUIRE(s[t] >= 1 && s[t] <= 8, "group size %d in thread %d", s[t], t);
        ST_REQUIRE(r[t] >= 0 && r[t] < s[t], "rank %d of %d in thread %d", r[t], s[t], t);
#ifdef PMPP_EMU
        ST_REQUIRE(s[t] == 8 && r[t] == t % 32 / 4, "emulator: thread %d has rank %d of %d, expected %d of 8", t, r[t], s[t], t % 32 / 4);
        ST_REQUIRE(f[t] == t / 32 * 32 + 1, "emulator: shfl(0) in thread %d gave %d", t, f[t]);
        ST_REQUIRE(v[t] == 0x55u, "emulator: packed ballot in thread %d is 0x%x, expected 0x55", t, v[t]);
#endif
    }
    return "";
}

// ---- grid_group: a barrier across every block of the grid -----------------------------------
// Phase 1: each block sums its slice into perBlock[]. Phase 2, after grid.sync():
// every thread reads ALL the block sums. Without the grid-wide barrier phase 2
// would read sums that other blocks have not written yet.
__global__ void kGrid(const int* in, int* perBlock, long long* seen, int* info) {
    __shared__ int partial[32];
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    int t = threadIdx.x, T = blockDim.x, g = blockIdx.x * T + t;
    for (int round = 0; round < 3; ++round) {
        partial[t] = in[g] + round;
        block.sync();
        if (t == 0) { int s = 0; for (int i = 0; i < T; ++i) s += partial[i]; perBlock[blockIdx.x] = s; }
        grid.sync();
        long long total = 0;
        for (unsigned b = 0; b < gridDim.x; ++b) total += perBlock[b];
        seen[round * (int)grid.size() + (int)grid.thread_rank()] = total;
        grid.sync();                            // nobody overwrites perBlock while it is being read
    }
    info[g] = grid.is_valid() * 1000 + (grid.block_rank() == blockIdx.x) * 100 + (grid.num_blocks() == gridDim.x) * 10 +
              (grid.dim_blocks().x == gridDim.x && grid.block_index().x == blockIdx.x && grid.num_threads() == grid.size());
}
static std::string tGrid() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!prop.cooperativeLaunch) return "";    // nothing to test on a device without cooperative launch
    const int T = 32;
    int perSm = 0;
    ST_REQUIRE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, kGrid, T, 0) == cudaSuccess && perSm >= 1,
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor gave %d blocks", perSm);
    int blocks = prop.multiProcessorCount * perSm;
    if (blocks > 24) blocks = 24;
    const int n = blocks * T;
    std::vector<int> h(n);
    for (int i = 0; i < n; ++i) h[i] = (i * 5) % 19;
    StBuf<int> in(h), perBlock(blocks, -1), info(n, 0); StBuf<long long> seen(3 * n, -1);
    const int* a0 = in.dev; int* a1 = perBlock.dev; long long* a2 = seen.dev; int* a3 = info.dev;
    void* args[] = {&a0, &a1, &a2, &a3};
    cudaError_t e = cudaLaunchCooperativeKernel((void*)kGrid, dim3(blocks), dim3(T), args);
    ST_REQUIRE(e == cudaSuccess, "cudaLaunchCooperativeKernel failed: %s", cudaGetErrorString(e));
    ST_LAUNCHED();
    long long base = 0;
    for (int i = 0; i < n; ++i) base += h[i];
    std::vector<long long> s = seen.host(); std::vector<int> inf = info.host();
    for (int round = 0; round < 3; ++round)
        for (int g = 0; g < n; ++g)
            ST_REQUIRE(s[round * n + g] == base + (long long)round * n, "round %d, thread %d saw a grid total of %lld, expected %lld",
                       round, g, s[round * n + g], base + (long long)round * n);
    for (int g = 0; g < n; ++g) ST_REQUIRE(inf[g] == 1111, "grid_group properties in thread %d are %d, expected 1111", g, inf[g]);
    return "";
}

#ifdef PMPP_EMU
// ---- emulator only: its limits and its refusals ------------------------------------------------
__global__ void kGridOnly(int* out) {
    cg::grid_group grid = cg::this_grid();
    out[blockIdx.x * blockDim.x + threadIdx.x] = grid.is_valid();
    grid.sync();
}
static std::string tGridRefusals() {
    StBuf<int> out(4096, -1);
    int* a0 = out.dev;
    void* args[] = {&a0};
    // 1. More threads than the emulator keeps alive at once.
    cudaError_t e = cudaLaunchCooperativeKernel((void*)kGridOnly, dim3(64), dim3(64), args);
    ST_REQUIRE(e == cudaErrorCooperativeLaunchTooLarge, "a 4096-thread cooperative launch returned: %s", cudaGetErrorString(e));
    ST_REQUIRE(cudaGetLastError() == cudaErrorCooperativeLaunchTooLarge, "the error was not recorded");
    // 2. The largest launch that fits works.
    e = cudaLaunchCooperativeKernel((void*)kGridOnly, dim3(16), dim3(64), args);
    ST_REQUIRE(e == cudaSuccess, "a 1024-thread cooperative launch failed: %s", cudaGetErrorString(e));
    ST_LAUNCHED();
    ST_REQUIRE(out.host()[1023] == 1, "is_valid() was %d in a cooperative launch", out.host()[1023]);
    // 3. grid.sync() in a kernel launched with <<< >>> is stopped with an explanation.
    kGridOnly<<<2, 32>>>(out.dev);
    e = cudaGetLastError();
    ST_REQUIRE(e == cudaErrorLaunchFailure && pmpp_emu::last_message().find("cudaLaunchCooperativeKernel") != std::string::npos,
               "grid.sync() under <<< >>> gave: %s", cudaGetErrorString(e));
    return "";
}
#endif

extern const pmpp_selftest_case pmpp_selftest_cases[] = {
    {"this_thread_block: rank, indices, dimensions, sync", tBlock},
    {"tiled_partition<32> and <8>: shfl, shfl_down, shfl_up, shfl_xor, ballot, any, all", tTile},
    {"coalesced_threads: size, thread_rank, shfl, ballot", tCoalesced},
    {"this_grid().sync() under cudaLaunchCooperativeKernel", tGrid},
#ifdef PMPP_EMU
    {"emulator: cooperative launch limit, and grid.sync() without a cooperative launch", tGridRefusals},
#endif
};
ST_END_CASES
