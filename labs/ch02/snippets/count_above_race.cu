// Count the probabilities above a threshold t. *count is 0 before the launch.
__global__ void countAbove(const float* p, float t, int* count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && p[i] > t) {
        *count = *count + 1;
    }
}
