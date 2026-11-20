// Attention scores for one query per head: s[h][t] = dot(q[h], K[t][h]).
// The key cache K is stored (token, head, dim), row-major. One thread per (head, token).
__global__ void scoresKernel(const float* q, const float* K, float* s, int T, int H, int D) {
    int h = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T) {
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) {
            acc += q[h * D + d] * K[(t * H + h) * D + d];
        }
        s[h * T + t] = acc;
    }
}
