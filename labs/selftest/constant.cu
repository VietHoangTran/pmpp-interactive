// Self-test: constant memory and the vector types / qualifiers that come with the runtime.
#include "selftest_cu.h"

#define TAPS 5
__constant__ float taps[TAPS];
__constant__ int4 window;                      // a single struct in constant memory

__global__ void kStencil(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0;
    #pragma unroll
    for (int t = 0; t < TAPS; ++t) {
        int j = i + t - TAPS / 2;
        if (j >= window.x && j < window.y) acc += taps[t] * in[j];
    }
    out[i] = acc;
}
static std::string tConstant() {
    const int n = 200;
    std::vector<float> h(n), w = {0.5f, -1.f, 2.f, -1.f, 0.5f};
    for (int i = 0; i < n; ++i) h[i] = (float)((i * 13) % 17) - 8.f;
    int4 win = make_int4(3, n - 2, 0, 0);
    ST_REQUIRE(cudaMemcpyToSymbol(taps, w.data(), TAPS * sizeof(float)) == cudaSuccess, "cudaMemcpyToSymbol failed");
    ST_REQUIRE(cudaMemcpyToSymbol(window, &win, sizeof(win)) == cudaSuccess, "cudaMemcpyToSymbol (struct) failed");
    StBuf<float> in(h), out(n, -99.f);
    kStencil<<<(n + 63) / 64, 64>>>(in.dev, out.dev, n);
    ST_LAUNCHED();
    std::vector<float> o = out.host();
    for (int i = 0; i < n; ++i) {
        float want = 0;
        for (int t = 0; t < TAPS; ++t) { int j = i + t - TAPS / 2; if (j >= 3 && j < n - 2) want += w[t] * h[j]; }
        ST_REQUIRE(o[i] == want, "element %d is %g, expected %g", i, o[i], want);
    }
    return "";
}

static std::string tOffsetAndBack() {
    float two[2] = {7.f, 9.f}, back[TAPS] = {0, 0, 0, 0, 0}, init[TAPS] = {1, 2, 3, 4, 5};
    ST_REQUIRE(cudaMemcpyToSymbol(taps, init, sizeof(init)) == cudaSuccess, "cudaMemcpyToSymbol failed");
    // offset is in bytes: overwrite taps[3] and taps[4]
    ST_REQUIRE(cudaMemcpyToSymbol(taps, two, sizeof(two), 3 * sizeof(float)) == cudaSuccess, "cudaMemcpyToSymbol with offset failed");
    ST_REQUIRE(cudaMemcpyFromSymbol(back, taps, sizeof(back)) == cudaSuccess, "cudaMemcpyFromSymbol failed");
    ST_REQUIRE(back[0] == 1 && back[2] == 3 && back[3] == 7 && back[4] == 9, "read back %g %g %g %g %g", back[0], back[1], back[2], back[3], back[4]);
    float one = 0;
    ST_REQUIRE(cudaMemcpyFromSymbol(&one, taps, sizeof(float), 4 * sizeof(float)) == cudaSuccess && one == 9.f, "offset read gave %g", one);
    return "";
}

// ---- vector types, make_ functions, inlining qualifiers, __restrict__ ----------------------
__forceinline__ __device__ float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__noinline__ __device__ float4 scale(float4 v, float s) { return make_float4(v.x * s, v.y * s, v.z * s, v.w * s); }
__device__ void axpy(const float* __restrict__ x, float* __restrict__ y, float a, int i) { y[i] += a * x[i]; }

__global__ void kVectors(const float3* a, const float3* b, float* dots, float4* scaled, int2* pair, uint3* where,
                         const float* x, float* y) {
    int i = threadIdx.x;
    dots[i] = dot3(a[i], b[i]);
    scaled[i] = scale(make_float4(a[i].x, a[i].y, a[i].z, 1.f), 2.f);
    pair[i] = make_int2(i, -i);
    where[i] = make_uint3(threadIdx.x, blockDim.x, gridDim.x);
    float2 p = make_float2(1.5f, 2.5f);
    uint2 q = make_uint2(3u, 4u);
    int3 r = make_int3(1, 2, 3);
    uint4 s = make_uint4(1u, 2u, 3u, 4u);
    if (p.x + p.y + q.x + q.y + r.x + r.y + r.z + s.w != 21.f) dots[i] = -1e9f;
    axpy(x, y, 3.f, i);
}
static std::string tVectors() {
    const int n = 48;
    std::vector<float3> ha(n), hb(n);
    std::vector<float> hx(n), hy(n);
    for (int i = 0; i < n; ++i) {
        ha[i] = make_float3((float)i, 1.f, -2.f); hb[i] = make_float3(2.f, (float)i, 0.5f);
        hx[i] = (float)i; hy[i] = 1.f;
    }
    StBuf<float3> a(ha), b(hb); StBuf<float> dots(n), x(hx), y(hy); StBuf<float4> scaled(n); StBuf<int2> pair(n); StBuf<uint3> where(n);
    kVectors<<<1, n>>>(a.dev, b.dev, dots.dev, scaled.dev, pair.dev, where.dev, x.dev, y.dev);
    ST_LAUNCHED();
    std::vector<float> d = dots.host(), yy = y.host(); std::vector<float4> s = scaled.host();
    std::vector<int2> p = pair.host(); std::vector<uint3> w = where.host();
    ST_REQUIRE(sizeof(float2) == 8 && sizeof(float3) == 12 && sizeof(float4) == 16 && sizeof(int4) == 16 && sizeof(uint3) == 12,
               "vector type sizes are wrong");
    for (int i = 0; i < n; ++i) {
        ST_REQUIRE(d[i] == 2.f * i + i - 1.f, "dot product %d is %g", i, d[i]);
        ST_REQUIRE(s[i].x == 2.f * i && s[i].y == 2.f && s[i].z == -4.f && s[i].w == 2.f, "float4 %d is wrong", i);
        ST_REQUIRE(p[i].x == i && p[i].y == -i, "int2 %d is wrong", i);
        ST_REQUIRE(w[i].x == (unsigned)i && w[i].y == (unsigned)n && w[i].z == 1u, "uint3 %d is wrong", i);
        ST_REQUIRE(yy[i] == 1.f + 3.f * i, "axpy %d gave %g", i, yy[i]);
    }
    dim3 grid(4, 5);
    ST_REQUIRE(grid.x == 4 && grid.y == 5 && grid.z == 1, "dim3 defaults are wrong");
    return "";
}

extern const pmpp_selftest_case pmpp_selftest_cases[] = {
    {"__constant__ array and struct, cudaMemcpyToSymbol, #pragma unroll", tConstant},
    {"cudaMemcpyToSymbol with an offset, cudaMemcpyFromSymbol", tOffsetAndBack},
    {"float2/3/4, int2/3/4, uint2/3/4, make_*, __forceinline__, __noinline__, __restrict__", tVectors},
};
ST_END_CASES
