// Runs ../launch_checked.cu and asserts what section 3.1 says about it. On a CUDA device
// the same file checks the documented behaviour instead of the emulator's.
#include <vector>
#include "check.h"

__global__ void fillOnes(float* out, int n);
cudaError_t launchOneBlock(float* out_d, int n, dim3 dimBlock);
bool blockShapeIsRejected(float* out_d, int n);

int main() {
    const int n = 2048;
    std::vector<float> host(n, 7.0f);
    float* out_d = nullptr;
    cudaMalloc((void**)&out_d, n * sizeof(float));
    cudaMemcpy(out_d, host.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    std::printf("launch_checked.cu\n");
    expect(blockShapeIsRejected(out_d, n), "blockShapeIsRejected returns true: dim3(32, 32, 2) is rejected");
    cudaDeviceSynchronize();
    cudaMemcpy(host.data(), out_d, n * sizeof(float), cudaMemcpyDeviceToHost);
    bool untouched = true;
    for (float v : host) untouched = untouched && v == 7.0f;
    expect(untouched, "the rejected launch left the buffer untouched");
    expect(cudaGetLastError() == cudaSuccess, "cudaGetLastError reset the error state");

    // 128 threads, but z is over NVIDIA's documented cap of 64.
    cudaError_t e = launchOneBlock(out_d, 128, dim3(1, 1, 128));
    cudaDeviceSynchronize();
#if defined(PMPP_EMU)
    expect(e == cudaSuccess, "the emulator runs a (1, 1, 128) block");
    cudaMemcpy(host.data(), out_d, n * sizeof(float), cudaMemcpyDeviceToHost);
    bool ones = true;
    for (int i = 0; i < 128; ++i) ones = ones && host[i] == 1.0f;
    expect(ones, "and its 128 threads wrote their elements");
#else
    expect(e == cudaErrorInvalidConfiguration, "a (1, 1, 128) block is rejected: z is over 64");
#endif

    e = launchOneBlock(out_d, n, dim3(0, 1, 1));
    expect(e == cudaErrorInvalidConfiguration, "a block with a zero dimension is rejected");

#if defined(PMPP_EMU)
    // The emulator reports grid caps but does not enforce them.
    fillOnes<<<dim3(1, 65536, 1), dim3(1, 1, 1)>>>(out_d, 1);
    expect(cudaGetLastError() == cudaSuccess, "the emulator runs a grid of 65,536 blocks along y");
    cudaDeviceSynchronize();
    // It does enforce its own limit on dynamic shared memory per block.
    fillOnes<<<1, 32, 1 << 20>>>(out_d, 32);
    expect(cudaGetLastError() == cudaErrorInvalidConfiguration, "the emulator rejects 1 MiB of dynamic shared memory");
#endif
    cudaFree(out_d);
    return finish();
}
