// For a question on chapter 19's summary page (Exercise 3, the backward pass); not built
// or run. The kernel is ours, not the book's. It computes dE/dX, the gradient with respect
// to the layer's input, by giving each thread one element of dE/dY and scattering that
// element's contributions into dE/dX, which is zeroed before the launch. Grid: x covers
// the H_out * W_out pixels of one output map, y the M output maps, z the N samples.
// Layouts as in lab 19.1: NCHW, row-major float. The question asks what is wrong with it.
__global__ void convBackwardXScatter(const float* dY, const float* F, float* dX,
                                     int C, int H, int W, int M, int K) {
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    int n = blockIdx.z;
    if (pixel >= H_out * W_out) return;
    int h = pixel / W_out;
    int w = pixel % W_out;
    float g = dY[((n * M + m) * H_out + h) * W_out + w];
    for (int c = 0; c < C; ++c)
        for (int p = 0; p < K; ++p)
            for (int q = 0; q < K; ++q)
                dX[((n * C + c) * H + h + p) * W + w + q] += g * F[((m * C + c) * K + p) * K + q];
}
