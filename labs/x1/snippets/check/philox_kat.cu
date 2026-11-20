// Runs ../philox.cu (section X1.7) on the course's CPU emulator and on the host, and prints
// what app/src/sim/philox.test.ts compares with app/src/sim/philox.ts:
//   kat <name> <four words>            Random123's three 10-round known-answer vectors (x1-facts
//                                      A4), computed by a kernel, one vector per thread
//   host <name> <four words>           the same, computed on the host by the same functions
//   u <seed> <stream> <step> <bits> <float bits>
//                                      uniformKernel for 64 rows: (0, 0, 0), all ones, then
//                                      rows from a fixed 64-bit generator
// The test puts the snippet's text in place of the #include below before translating, so
// that the emulator registers both files' kernels in one translation unit.
#include <cstdio>
#include <cstring>
#include <vector>
#include "../philox.cu"

__global__ void katKernel(const unsigned int* in, unsigned int* out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned int* v = in + 6 * i;
    PhiloxWords c = {v[0], v[1], v[2], v[3]};
    const PhiloxWords r = philox4x32_10(c, v[4], v[5]);
    out[4 * i + 0] = r.x;
    out[4 * i + 1] = r.y;
    out[4 * i + 2] = r.z;
    out[4 * i + 3] = r.w;
}

static unsigned int floatBits(float f) {
    unsigned int b;
    std::memcpy(&b, &f, sizeof b);
    return b;
}

int main() {
    const char* names[3] = {"zero", "ones", "pi"};
    const unsigned int kat[18] = {0u, 0u, 0u, 0u, 0u, 0u,
                                  0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
                                  0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u, 0xa4093822u, 0x299f31d0u};
    unsigned int *in_d, *out_d;
    cudaMalloc((void**)&in_d, sizeof kat);
    cudaMalloc((void**)&out_d, 12 * sizeof(unsigned int));
    cudaMemcpy(in_d, kat, sizeof kat, cudaMemcpyHostToDevice);
    katKernel<<<1, 32>>>(in_d, out_d, 3);
    unsigned int out[12];
    cudaMemcpy(out, out_d, sizeof out, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 3; ++i)
        std::printf("kat %s %08x %08x %08x %08x\n", names[i], out[4 * i], out[4 * i + 1], out[4 * i + 2], out[4 * i + 3]);
    for (int i = 0; i < 3; ++i) {
        PhiloxWords c = {kat[6 * i], kat[6 * i + 1], kat[6 * i + 2], kat[6 * i + 3]};
        const PhiloxWords r = philox4x32_10(c, kat[6 * i + 4], kat[6 * i + 5]);
        std::printf("host %s %08x %08x %08x %08x\n", names[i], r.x, r.y, r.z, r.w);
    }

    const int rows = 64;
    std::vector<unsigned long long> seed(rows), stream(rows), step(rows);
    unsigned long long s = 2026;
    for (int r = 0; r < rows; ++r) {
        unsigned long long v[3];
        for (int k = 0; k < 3; ++k) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            v[k] = s ^ (s >> 29);
        }
        seed[r] = v[0];
        stream[r] = r < 32 ? (unsigned long long)r : v[1];   // the first rows are consecutive streams
        step[r] = r < 32 ? 7ull : v[2] >> (r % 40);
    }
    seed[0] = stream[0] = step[0] = 0;
    seed[1] = stream[1] = step[1] = ~0ull;
    unsigned long long *seed_d, *stream_d, *step_d;
    float* u_d;
    cudaMalloc((void**)&seed_d, rows * sizeof(unsigned long long));
    cudaMalloc((void**)&stream_d, rows * sizeof(unsigned long long));
    cudaMalloc((void**)&step_d, rows * sizeof(unsigned long long));
    cudaMalloc((void**)&u_d, rows * sizeof(float));
    cudaMemcpy(seed_d, seed.data(), rows * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(stream_d, stream.data(), rows * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(step_d, step.data(), rows * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    launchUniform(seed_d, stream_d, step_d, u_d, rows);
    std::vector<float> u(rows);
    cudaMemcpy(u.data(), u_d, rows * sizeof(float), cudaMemcpyDeviceToHost);
    for (int r = 0; r < rows; ++r)
        std::printf("u %llu %llu %llu %08x %08x\n", seed[r], stream[r], step[r], m5Bits(seed[r], stream[r], step[r]),
                    floatBits(u[r]));
    cudaFree(in_d);
    cudaFree(out_d);
    cudaFree(seed_d);
    cudaFree(stream_d);
    cudaFree(step_d);
    cudaFree(u_d);
    std::printf("DONE\n");
    return 0;
}
