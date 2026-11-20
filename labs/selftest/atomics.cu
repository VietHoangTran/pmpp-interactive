// Self-test: the legacy atomic functions and cuda::atomic_ref from <cuda/atomic>.
#include <cuda/atomic>
#include "selftest_cu.h"

// ---- atomicAdd on every supported type, from many blocks at once -------------------
__global__ void kAdd(int* i, unsigned* u, unsigned long long* ull, float* f, double* d) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(i, g % 5 - 2);
    atomicAdd(u, 3u);
    atomicAdd(ull, 1ULL << 33);
    atomicAdd(f, 0.5f);
    atomicAdd(d, 0.25);
    atomicSub(i + 1, 2);
    atomicSub(u + 1, 1u);
}
static std::string tAdd() {
    const int blocks = 6, T = 96, n = blocks * T;
    StBuf<int> i(2, 0); StBuf<unsigned> u(2, 1000000u); StBuf<unsigned long long> ull(1, 0); StBuf<float> f(1, 0.f); StBuf<double> d(1, 0.0);
    kAdd<<<blocks, T>>>(i.dev, u.dev, ull.dev, f.dev, d.dev);
    ST_LAUNCHED();
    int wi = 0;
    for (int g = 0; g < n; ++g) wi += g % 5 - 2;
    ST_REQUIRE(i.host()[0] == wi, "atomicAdd(int) gave %d, expected %d", i.host()[0], wi);
    ST_REQUIRE(i.host()[1] == -2 * n, "atomicSub(int) gave %d, expected %d", i.host()[1], -2 * n);
    ST_REQUIRE(u.host()[0] == 1000000u + 3u * n, "atomicAdd(unsigned) gave %u", u.host()[0]);
    ST_REQUIRE(u.host()[1] == 1000000u - n, "atomicSub(unsigned) gave %u", u.host()[1]);
    ST_REQUIRE(ull.host()[0] == (unsigned long long)n << 33, "atomicAdd(unsigned long long) is wrong");
    ST_REQUIRE(f.host()[0] == 0.5f * n, "atomicAdd(float) gave %g, expected %g", f.host()[0], 0.5f * n);
    ST_REQUIRE(d.host()[0] == 0.25 * n, "atomicAdd(double) gave %g, expected %g", d.host()[0], 0.25 * n);
    return "";
}

// ---- min, max, and, or, xor, inc, dec ------------------------------------------------
__global__ void kMinMax(const int* in, int* lo, int* hi, long long* lo64, unsigned long long* hi64,
                        unsigned* bitsOr, unsigned* bitsAnd, unsigned* bitsXor, unsigned* inc, unsigned* dec) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    atomicMin(lo, in[g]);
    atomicMax(hi, in[g]);
    atomicMin(lo64, (long long)in[g] * 4000000000LL);
    atomicMax(hi64, (unsigned long long)(in[g] + 1000));
    atomicOr(bitsOr, 1u << (g % 32));
    atomicAnd(bitsAnd, ~(1u << (g % 7)));
    atomicXor(bitsXor, 1u << (g % 32));
    atomicInc(inc, 9u);                 // counts 0..9 and wraps to 0
    atomicDec(dec, 9u);                 // counts down and wraps to 9
}
static std::string tMinMax() {
    const int blocks = 4, T = 64, n = blocks * T;       // n = 256
    std::vector<int> h(n);
    int wlo = 1 << 30, whi = -(1 << 30);
    for (int g = 0; g < n; ++g) { h[g] = (g * 7919) % 2003 - 1000; wlo = h[g] < wlo ? h[g] : wlo; whi = h[g] > whi ? h[g] : whi; }
    StBuf<int> in(h), lo(1, 1 << 30), hi(1, -(1 << 30));
    StBuf<long long> lo64(1, 0); StBuf<unsigned long long> hi64(1, 0);
    StBuf<unsigned> bo(1, 0u), ba(1, 0xffffffffu), bx(1, 0u), inc(1, 0u), dec(1, 0u);
    kMinMax<<<blocks, T>>>(in.dev, lo.dev, hi.dev, lo64.dev, hi64.dev, bo.dev, ba.dev, bx.dev, inc.dev, dec.dev);
    ST_LAUNCHED();
    ST_REQUIRE(lo.host()[0] == wlo && hi.host()[0] == whi, "atomicMin/Max gave %d/%d, expected %d/%d", lo.host()[0], hi.host()[0], wlo, whi);
    ST_REQUIRE(lo64.host()[0] == (long long)wlo * 4000000000LL, "atomicMin(long long) is wrong");
    ST_REQUIRE(hi64.host()[0] == (unsigned long long)(whi + 1000), "atomicMax(unsigned long long) is wrong");
    ST_REQUIRE(bo.host()[0] == 0xffffffffu, "atomicOr gave 0x%08x", bo.host()[0]);
    ST_REQUIRE(ba.host()[0] == 0xffffff80u, "atomicAnd gave 0x%08x", ba.host()[0]);
    ST_REQUIRE(bx.host()[0] == 0u, "atomicXor gave 0x%08x (each bit is toggled an even number of times)", bx.host()[0]);
    ST_REQUIRE(inc.host()[0] == (unsigned)(n % 10), "atomicInc gave %u, expected %u", inc.host()[0], (unsigned)(n % 10));
    ST_REQUIRE(dec.host()[0] == (unsigned)((10 - n % 10) % 10), "atomicDec gave %u", dec.host()[0]);
    return "";
}

// ---- exchange and compare-and-swap: exactly one thread wins ---------------------------
__global__ void kClaim(int* owner, unsigned* winners, unsigned long long* token, float* last, unsigned short* small) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (atomicCAS(owner, -1, g) == -1) atomicAdd(winners, 1u);          // only the first CAS sees -1
    if (atomicExch(token, 0ULL) == 77ULL) atomicAdd(winners + 1, 1u);   // only one thread takes the token
    atomicExch(last, (float)g);
    atomicCAS(small, (unsigned short)500, (unsigned short)g);
}
static std::string tClaim() {
    const int blocks = 5, T = 64, n = blocks * T;
    StBuf<int> owner(1, -1); StBuf<unsigned> winners(2, 0u); StBuf<unsigned long long> token(1, 77ULL);
    StBuf<float> last(1, -1.f); StBuf<unsigned short> small(1, (unsigned short)500);
    kClaim<<<blocks, T>>>(owner.dev, winners.dev, token.dev, last.dev, small.dev);
    ST_LAUNCHED();
    ST_REQUIRE(winners.host()[0] == 1u, "%u threads won the compare-and-swap, expected 1", winners.host()[0]);
    ST_REQUIRE(winners.host()[1] == 1u, "%u threads took the token, expected 1", winners.host()[1]);
    ST_REQUIRE(owner.host()[0] >= 0 && owner.host()[0] < n, "owner is %d", owner.host()[0]);
    ST_REQUIRE(last.host()[0] >= 0.f && last.host()[0] < (float)n, "atomicExch(float) left %g", last.host()[0]);
    ST_REQUIRE(small.host()[0] < n, "atomicCAS(unsigned short) left %u", (unsigned)small.host()[0]);
    return "";
}

// ---- cuda::atomic_ref, the interface the book's 5th edition uses ------------------------
__global__ void kRef(unsigned* bins, int* lo, int* hi, float* sum, int* first, unsigned* flags, int nbins) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> bin(bins[g % nbins]);
    bin.fetch_add(1u, cuda::memory_order_relaxed);
    cuda::atomic_ref<int, cuda::thread_scope_device> l(*lo), h(*hi);
    l.fetch_min(g - 100, cuda::std::memory_order_relaxed);
    h.fetch_max(g - 100);
    cuda::atomic_ref<float, cuda::thread_scope_system> s(*sum);
    s.fetch_add(0.5f);
    cuda::atomic_ref<int, cuda::thread_scope_device> f(*first);
    int expected = -1;
    f.compare_exchange_strong(expected, g, cuda::memory_order_acq_rel, cuda::memory_order_relaxed);
    cuda::atomic_ref<unsigned, cuda::thread_scope_block> fl(flags[blockIdx.x]);   // one per block
    fl.fetch_or(1u << (threadIdx.x % 32), cuda::memory_order_relaxed);
    unsigned seen = fl.load(cuda::memory_order_relaxed);
    if (!(seen >> (threadIdx.x % 32) & 1u)) fl.store(0xdeadu);                    // never happens
}
static std::string tRef() {
    const int blocks = 4, T = 128, n = blocks * T, nbins = 7;
    StBuf<unsigned> bins(nbins, 0u), flags(blocks, 0u); StBuf<int> lo(1, 0), hi(1, 0), first(1, -1); StBuf<float> sum(1, 0.f);
    kRef<<<blocks, T>>>(bins.dev, lo.dev, hi.dev, sum.dev, first.dev, flags.dev, nbins);
    ST_LAUNCHED();
    std::vector<unsigned> b = bins.host(), fl = flags.host();
    for (int k = 0; k < nbins; ++k) {
        unsigned want = (unsigned)(n / nbins + (k < n % nbins ? 1 : 0));
        ST_REQUIRE(b[k] == want, "bin %d holds %u, expected %u", k, b[k], want);
    }
    ST_REQUIRE(lo.host()[0] == -100 && hi.host()[0] == n - 1 - 100, "fetch_min/fetch_max gave %d/%d", lo.host()[0], hi.host()[0]);
    ST_REQUIRE(sum.host()[0] == 0.5f * n, "fetch_add(float) gave %g", sum.host()[0]);
    ST_REQUIRE(first.host()[0] >= 0 && first.host()[0] < n, "compare_exchange_strong left %d", first.host()[0]);
    for (int k = 0; k < blocks; ++k) ST_REQUIRE(fl[k] == 0xffffffffu, "fetch_or in block %d gave 0x%08x", k, fl[k]);
    return "";
}

// ---- cuda::atomic_ref on shared memory, exchange ----------------------------------------------
__global__ void kShared(unsigned* perBlock, int* order) {
    __shared__ unsigned counter;
    if (threadIdx.x == 0) counter = 0;
    __syncthreads();
    cuda::atomic_ref<unsigned, cuda::thread_scope_block> c(counter);
    unsigned ticket = c.fetch_add(1u);                   // every thread gets a different ticket
    order[blockIdx.x * blockDim.x + ticket] = 1;
    __syncthreads();
    if (threadIdx.x == 0) perBlock[blockIdx.x] = c.exchange(0u);
}
static std::string tShared() {
    const int blocks = 3, T = 100;
    StBuf<unsigned> perBlock(blocks, 0u); StBuf<int> order(blocks * T, 0);
    kShared<<<blocks, T>>>(perBlock.dev, order.dev);
    ST_LAUNCHED();
    std::vector<unsigned> p = perBlock.host(); std::vector<int> o = order.host();
    for (int b = 0; b < blocks; ++b) ST_REQUIRE(p[b] == (unsigned)T, "block %d counted %u, expected %d", b, p[b], T);
    for (int k = 0; k < blocks * T; ++k) ST_REQUIRE(o[k] == 1, "ticket %d was never handed out", k);
    return "";
}

extern const pmpp_selftest_case pmpp_selftest_cases[] = {
    {"atomicAdd / atomicSub on int, unsigned, unsigned long long, float, double", tAdd},
    {"atomicMin, Max, And, Or, Xor, Inc, Dec", tMinMax},
    {"atomicCAS and atomicExch: one winner", tClaim},
    {"cuda::atomic_ref: fetch_add, fetch_min, fetch_max, compare_exchange, scopes, orders", tRef},
    {"cuda::atomic_ref on a __shared__ counter", tShared},
};
ST_END_CASES
