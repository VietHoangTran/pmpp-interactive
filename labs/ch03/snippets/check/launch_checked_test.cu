// Runs ../launch_checked.cu and asserts what section 3.1 says about it. On a CUDA device
// the same file checks the same documented limits; only the dynamic-shared-memory check is
// emulator-only. Not for HIP: AMD documents z up to 1024, so (1, 1, 128) is a valid block there.
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

    // 128 threads, but z is over NVIDIA's documented cap of 64. The emulator enforces the
    // per-dimension caps too, so the expectation is the same on both.
    cudaError_t e = launchOneBlock(out_d, 128, dim3(1, 1, 128));
    cudaDeviceSynchronize();
    expect(e == cudaErrorInvalidConfiguration, "a (1, 1, 128) block is rejected: z is over 64");

    e = launchOneBlock(out_d, n, dim3(0, 1, 1));
    expect(e == cudaErrorInvalidConfiguration, "a block with a zero dimension is rejected");

    // 65,536 blocks along y is one over the documented grid cap.
    fillOnes<<<dim3(1, 65536, 1), dim3(1, 1, 1)>>>(out_d, 1);
    expect(cudaGetLastError() == cudaErrorInvalidConfiguration, "a grid of 65,536 blocks along y is rejected");
    cudaDeviceSynchronize();

#if defined(PMPP_EMU)
    // The emulator's own limit on dynamic shared memory per block.
    fillOnes<<<1, 32, 1 << 20>>>(out_d, 32);
    expect(cudaGetLastError() == cudaErrorInvalidConfiguration, "the emulator rejects 1 MiB of dynamic shared memory");
#endif
    cudaFree(out_d);
    return finish();
}
