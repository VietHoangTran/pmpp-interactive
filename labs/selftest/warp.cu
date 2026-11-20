// Self-test: warps and warp-level intrinsics.
#include "selftest_cu.h"

static const unsigned FULL = 0xffffffffu;

__device__ int linearTid() { return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y; }

// ---- a warp is 32 consecutive threads in linear thread order ----------------------
__global__ void kLayout(int* lead, int* count, int* ws) {
    int tid = linearTid();
    lead[tid] = __shfl_sync(FULL, tid, 0);            // the linear index of lane 0 of my warp
    count[tid] = __popc(__ballot_sync(FULL, 1));       // how many threads share my warp
    ws[tid] = warpSize;
}
static std::string tLayout() {
    const int T = 8 * 8 * 2;
    StBuf<int> lead(T, -1), count(T, -1), ws(T, -1);
    kLayout<<<1, dim3(8, 8, 2)>>>(lead.dev, count.dev, ws.dev);
    ST_LAUNCHED();
    std::vector<int> l = lead.host(), c = count.host(), w = ws.host();
    for (int t = 0; t < T; ++t) {
        ST_REQUIRE(w[t] == 32, "warpSize is %d", w[t]);
        ST_REQUIRE(l[t] == t / 32 * 32, "thread %d: lane 0 of its warp is thread %d, expected %d", t, l[t], t / 32 * 32);
        ST_REQUIRE(c[t] == 32, "thread %d: %d threads in its warp", t, c[t]);
    }
    return "";
}

// ---- the four shuffles, with and without a width -----------------------------------
enum { S_IDX, S_IDX_W8, S_IDX_WRAP, S_UP, S_UP_W16, S_DOWN, S_XOR, S_XOR_W8, S_COUNT };
__global__ void kShuffle(int* out) {
    int lane = threadIdx.x, v = 100 + lane;
    out[S_IDX * 32 + lane] = __shfl_sync(FULL, v, 5);
    out[S_IDX_W8 * 32 + lane] = __shfl_sync(FULL, v, 3, 8);
    out[S_IDX_WRAP * 32 + lane] = __shfl_sync(FULL, v, 35);
    out[S_UP * 32 + lane] = __shfl_up_sync(FULL, v, 1);
    out[S_UP_W16 * 32 + lane] = __shfl_up_sync(FULL, v, 2, 16);
    out[S_DOWN * 32 + lane] = __shfl_down_sync(FULL, v, 16);
    out[S_XOR * 32 + lane] = __shfl_xor_sync(FULL, v, 1);
    out[S_XOR_W8 * 32 + lane] = __shfl_xor_sync(FULL, v, 4, 8);
}
static std::string tShuffle() {
    StBuf<int> out(S_COUNT * 32, -1);
    kShuffle<<<1, 32>>>(out.dev);
    ST_LAUNCHED();
    std::vector<int> o = out.host();
    const char* names[S_COUNT] = {"shfl(5)", "shfl(3, width 8)", "shfl(35)", "shfl_up(1)", "shfl_up(2, width 16)",
                                  "shfl_down(16)", "shfl_xor(1)", "shfl_xor(4, width 8)"};
    for (int lane = 0; lane < 32; ++lane) {
        int want[S_COUNT];
        want[S_IDX] = 105;
        want[S_IDX_W8] = 100 + lane / 8 * 8 + 3;
        want[S_IDX_WRAP] = 103;                                    // 35 % 32
        want[S_UP] = lane >= 1 ? 100 + lane - 1 : 100 + lane;      // the lowest lane keeps its own value
        want[S_UP_W16] = lane % 16 >= 2 ? 100 + lane - 2 : 100 + lane;
        want[S_DOWN] = lane < 16 ? 100 + lane + 16 : 100 + lane;   // the upper lanes keep their own value
        want[S_XOR] = 100 + (lane ^ 1);
        want[S_XOR_W8] = 100 + (lane ^ 4);
        for (int k = 0; k < S_COUNT; ++k)
            ST_REQUIRE(o[k * 32 + lane] == want[k], "%s in lane %d returned %d, expected %d", names[k], lane, o[k * 32 + lane], want[k]);
    }
    return "";
}

// ---- operand types -------------------------------------------------------------------
__global__ void kTypes(float* f, double* d, long long* ll, unsigned* u, unsigned long long* ull) {
    int lane = threadIdx.x;
    f[lane] = __shfl_xor_sync(FULL, 0.5f + lane, 1);
    d[lane] = __shfl_xor_sync(FULL, 0.25 + lane, 1);
    ll[lane] = __shfl_xor_sync(FULL, -5000000000LL * (lane + 1), 1);
    u[lane] = __shfl_xor_sync(FULL, 4000000000u + (unsigned)lane, 1);
    ull[lane] = __shfl_xor_sync(FULL, 0xF000000000000000ULL + (unsigned long long)lane, 1);
}
static std::string tTypes() {
    StBuf<float> f(32); StBuf<double> d(32); StBuf<long long> ll(32); StBuf<unsigned> u(32); StBuf<unsigned long long> ull(32);
    kTypes<<<1, 32>>>(f.dev, d.dev, ll.dev, u.dev, ull.dev);
    ST_LAUNCHED();
    std::vector<float> hf = f.host(); std::vector<double> hd = d.host(); std::vector<long long> hll = ll.host();
    std::vector<unsigned> hu = u.host(); std::vector<unsigned long long> hull = ull.host();
    for (int lane = 0; lane < 32; ++lane) {
        int o = lane ^ 1;
        ST_REQUIRE(hf[lane] == 0.5f + o, "float shuffle wrong in lane %d", lane);
        ST_REQUIRE(hd[lane] == 0.25 + o, "double shuffle wrong in lane %d", lane);
        ST_REQUIRE(hll[lane] == -5000000000LL * (o + 1), "long long shuffle wrong in lane %d", lane);
        ST_REQUIRE(hu[lane] == 4000000000u + (unsigned)o, "unsigned shuffle wrong in lane %d", lane);
        ST_REQUIRE(hull[lane] == 0xF000000000000000ULL + (unsigned long long)o, "unsigned long long shuffle wrong in lane %d", lane);
    }
    return "";
}

// ---- votes, including two halves of a warp voting separately --------------------------
__global__ void kVotes(unsigned* ballot, int* any, int* all, unsigned* half, int* allHalf) {
    int lane = threadIdx.x;
    int pred = lane % 3 == 0;
    ballot[lane] = __ballot_sync(FULL, pred);
    any[lane] = __any_sync(FULL, pred);
    all[lane] = __all_sync(FULL, pred);
    unsigned mine = lane < 16 ? 0x0000ffffu : 0xffff0000u;     // disjoint masks: two independent groups
    half[lane] = __ballot_sync(mine, lane < 16 ? (lane & 1) : 1);
    allHalf[lane] = __all_sync(mine, lane >= 16);
}
static std::string tVotes() {
    StBuf<unsigned> ballot(32), half(32); StBuf<int> any(32, -1), all(32, -1), allHalf(32, -1);
    kVotes<<<1, 32>>>(ballot.dev, any.dev, all.dev, half.dev, allHalf.dev);
    ST_LAUNCHED();
    unsigned want = 0;
    for (int l = 0; l < 32; l += 3) want |= 1u << l;
    std::vector<unsigned> b = ballot.host(), h = half.host();
    std::vector<int> an = any.host(), al = all.host(), ah = allHalf.host();
    for (int lane = 0; lane < 32; ++lane) {
        ST_REQUIRE(b[lane] == want, "ballot in lane %d is 0x%08x, expected 0x%08x", lane, b[lane], want);
        ST_REQUIRE(an[lane] != 0 && al[lane] == 0, "any/all in lane %d are %d/%d, expected non-zero/0", lane, an[lane], al[lane]);
        unsigned wantHalf = lane < 16 ? 0x0000aaaau : 0xffff0000u;
        ST_REQUIRE(h[lane] == wantHalf, "half-warp ballot in lane %d is 0x%08x, expected 0x%08x", lane, h[lane], wantHalf);
        ST_REQUIRE((ah[lane] != 0) == (lane >= 16), "half-warp all in lane %d is %d", lane, ah[lane]);
    }
    return "";
}

// ---- the textbook warp reduction --------------------------------------------------------
__global__ void kWarpSum(const int* in, int* sums) {
    int tid = threadIdx.x, g = blockIdx.x * blockDim.x + tid;
    int v = in[g];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) v += __shfl_down_sync(FULL, v, offset);
    if (tid % warpSize == 0) sums[g / warpSize] = v;
}
static std::string tWarpSum() {
    const int blocks = 3, T = 64, n = blocks * T;
    std::vector<int> h(n);
    for (int i = 0; i < n; ++i) h[i] = (i * 37) % 101 - 50;
    StBuf<int> in(h), sums(n / 32, 0);
    kWarpSum<<<blocks, T>>>(in.dev, sums.dev);
    ST_LAUNCHED();
    std::vector<int> s = sums.host();
    for (int w = 0; w < n / 32; ++w) {
        int want = 0;
        for (int i = 0; i < 32; ++i) want += h[w * 32 + i];
        ST_REQUIRE(s[w] == want, "warp %d sums to %d, expected %d", w, s[w], want);
    }
    return "";
}

// ---- a partial last warp, and threads that have exited -----------------------------------
__global__ void kPartial(unsigned* active, unsigned* ballot) {
    int tid = threadIdx.x;
    active[tid] = __activemask();          // before any divergence: every lane that exists
    if (tid >= 36) return;                 // lanes 4..7 of the second warp leave
    ballot[tid] = __ballot_sync(FULL, 1);  // exited and absent lanes are ignored
}
static std::string tPartial() {
    StBuf<unsigned> active(40, 0), ballot(40, 0);
    kPartial<<<1, 40>>>(active.dev, ballot.dev);
    ST_LAUNCHED();
    std::vector<unsigned> a = active.host(), b = ballot.host();
    for (int t = 0; t < 40; ++t) {
        unsigned wantA = t < 32 ? FULL : 0xffu, wantB = t < 32 ? FULL : (t < 36 ? 0xfu : 0u);
        ST_REQUIRE(a[t] == wantA, "__activemask in thread %d is 0x%08x, expected 0x%08x", t, a[t], wantA);
        ST_REQUIRE(b[t] == wantB, "__ballot_sync in thread %d is 0x%08x, expected 0x%08x", t, b[t], wantB);
    }
    return "";
}

// ---- __syncwarp orders memory accesses inside a warp ---------------------------------------
__global__ void kSyncWarp(int* out) {
    __shared__ int scratch[64];
    int tid = threadIdx.x;
    scratch[tid] = 1000 * (int)blockIdx.x + tid;
    __syncwarp();
    out[blockIdx.x * blockDim.x + tid] = scratch[tid ^ 1];     // a neighbour in the same warp
}
static std::string tSyncWarp() {
    StBuf<int> out(2 * 64, -1);
    kSyncWarp<<<2, 64>>>(out.dev);
    ST_LAUNCHED();
    std::vector<int> o = out.host();
    for (int b = 0; b < 2; ++b)
        for (int t = 0; t < 64; ++t)
            ST_REQUIRE(o[b * 64 + t] == 1000 * b + (t ^ 1), "block %d thread %d read %d", b, t, o[b * 64 + t]);
    return "";
}

// ---- integer intrinsics ----------------------------------------------------------------------
__global__ void kBits(const unsigned* x, int* popc, int* ffs, int* clz, unsigned* brev, int* popcll, int* ffsll) {
    int i = threadIdx.x;
    popc[i] = __popc(x[i]);
    ffs[i] = __ffs((int)x[i]);
    clz[i] = __clz((int)x[i]);
    brev[i] = __brev(x[i]);
    unsigned long long wide = ((unsigned long long)x[i] << 32) | x[i];
    popcll[i] = __popcll(wide);
    ffsll[i] = __ffsll((long long)((unsigned long long)x[i] << 20));
}
static std::string tBits() {
    std::vector<unsigned> h = {0u, 1u, 8u, 0xF0F0u, 0x80000000u, 0xffffffffu, 0x00010000u, 12345678u};
    const int n = (int)h.size();
    StBuf<unsigned> x(h), brev(n); StBuf<int> popc(n), ffs(n), clz(n), popcll(n), ffsll(n);
    kBits<<<1, n>>>(x.dev, popc.dev, ffs.dev, clz.dev, brev.dev, popcll.dev, ffsll.dev);
    ST_LAUNCHED();
    std::vector<int> p = popc.host(), f = ffs.host(), c = clz.host(), pl = popcll.host(), fl = ffsll.host();
    std::vector<unsigned> r = brev.host();
    for (int i = 0; i < n; ++i) {
        int wp = 0, wf = 0, wc = 32; unsigned wr = 0;
        for (int b = 0; b < 32; ++b) {
            if (h[i] >> b & 1) { wp++; if (!wf) wf = b + 1; wc = 31 - b; wr |= 1u << (31 - b); }
        }
        ST_REQUIRE(p[i] == wp, "__popc(0x%x) = %d, expected %d", h[i], p[i], wp);
        ST_REQUIRE(f[i] == wf, "__ffs(0x%x) = %d, expected %d", h[i], f[i], wf);
        ST_REQUIRE(c[i] == wc, "__clz(0x%x) = %d, expected %d", h[i], c[i], wc);
        ST_REQUIRE(r[i] == wr, "__brev(0x%x) = 0x%x, expected 0x%x", h[i], r[i], wr);
        ST_REQUIRE(pl[i] == 2 * wp, "__popcll gave %d, expected %d", pl[i], 2 * wp);
        ST_REQUIRE(fl[i] == (wf ? wf + 20 : 0), "__ffsll gave %d, expected %d", fl[i], wf ? wf + 20 : 0);
    }
    return "";
}

#ifdef PMPP_EMU
// ---- emulator only: mistakes that are undefined on a GPU must end the launch with a message ----
static std::string expectStopped(const char* needle1, const char* needle2) {
    cudaError_t e = cudaGetLastError();
    ST_REQUIRE(e == cudaErrorLaunchFailure, "expected the emulator to stop the kernel, got: %s", cudaGetErrorString(e));
    const std::string& msg = pmpp_emu::last_message();
    ST_REQUIRE(msg.find(needle1) != std::string::npos && msg.find(needle2) != std::string::npos,
               "the message does not mention '%s' and '%s': %s", needle1, needle2, msg.c_str());
    return "";
}

__global__ void kMismatch(int* out) {
    int lane = threadIdx.x;
    if (lane < 16) out[lane] = __shfl_sync(FULL, lane, 0);     // names all 32 lanes ...
    else __syncthreads();                                      // ... but half of them are here
}
static std::string tMismatch() {
    StBuf<int> out(32, 0);
    kMismatch<<<1, 32>>>(out.dev);
    return expectStopped("__shfl_sync", "__syncthreads");
}

__global__ void kNoSelf(unsigned* out) {
    if (threadIdx.x == 0) out[0] = __ballot_sync(0x2u, 1);     // lane 0 is not in its own mask
}
static std::string tNoSelf() {
    StBuf<unsigned> out(1, 0);
    kNoSelf<<<1, 32>>>(out.dev);
    return expectStopped("__ballot_sync", "own lane");
}

__global__ void kBadWidth(int* out) { out[threadIdx.x] = __shfl_sync(FULL, 1, 0, 12); }
static std::string tBadWidth() {
    StBuf<int> out(32, 0);
    kBadWidth<<<1, 32>>>(out.dev);
    return expectStopped("__shfl_sync", "power of two");
}

__global__ void kOneSidedBarrier(int* out) {
    if (threadIdx.x == 0) return;
    if (threadIdx.x % 2) __syncthreads();
    else out[threadIdx.x] = __shfl_up_sync(FULL, 7, 1);        // waits for lanes that sit at the barrier
}
static std::string tOneSidedBarrier() {
    StBuf<int> out(32, 0);
    kOneSidedBarrier<<<1, 32>>>(out.dev);
    return expectStopped("__shfl_up_sync", "__syncthreads");
}

__global__ void kReadsAbsentLane(int* out) { out[threadIdx.x] = __shfl_down_sync(FULL, (int)threadIdx.x, 4); }
static std::string tReadsAbsentLane() {
    StBuf<int> out(8, 0);
    kReadsAbsentLane<<<1, 8>>>(out.dev);       // lanes 8..31 do not exist in this block
    ST_LAUNCHED();
    std::vector<int> o = out.host();
    for (int t = 0; t < 8; ++t) {
        int want = t < 4 ? t + 4 : (int)0xCDCDCDCD;
        ST_REQUIRE(o[t] == want, "lane %d got 0x%08x, expected 0x%08x", t, (unsigned)o[t], (unsigned)want);
    }
    return "";
}
#endif

extern const pmpp_selftest_case pmpp_selftest_cases[] = {
    {"a warp is 32 consecutive threads in linear order", tLayout},
    {"shuffles: index, up, down, xor, width", tShuffle},
    {"shuffles: float, double, 64-bit operands", tTypes},
    {"votes: ballot, any, all, disjoint masks", tVotes},
    {"warp reduction with __shfl_down_sync", tWarpSum},
    {"partial warp and exited lanes", tPartial},
    {"__syncwarp", tSyncWarp},
    {"__popc, __ffs, __clz, __brev, 64-bit forms", tBits},
#ifdef PMPP_EMU
    {"emulator stops: shuffle names lanes that are at a barrier", tMismatch},
    {"emulator stops: mask without the caller's own lane", tNoSelf},
    {"emulator stops: width that is not a power of two", tBadWidth},
    {"emulator stops: half a warp at a barrier, half in a shuffle", tOneSidedBarrier},
    {"emulator marks a read from a lane that does not exist", tReadsAbsentLane},
#endif
};
ST_END_CASES
