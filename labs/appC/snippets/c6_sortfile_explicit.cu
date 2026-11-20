// sortfile ported the Chapter 2 way, with a device buffer and two explicit copies.
// Ours, not the book's: Fig. C.1 (p. 627) shows only the CPU original and the managed
// port. This is the third column the lesson sets beside them.
// Host code only; the sort kernel and use_data are declared and never defined, and the
// file is parsed, not built or run. The second cudaMemcpy also waits for the kernel: a
// blocking copy does not return until it has completed, and it runs after the kernel
// launched before it (notes/today/ch02-facts.md A4, A5).
#include <cstdio>
#include <cstdlib>

__global__ void qsort_char(char *data, int n, int elemSize);
void use_data(char *data);

static const dim3 grid(1), block(1);  // stand-ins for the figure's <<<...>>>

void sortfile(FILE *fp, int N) {
    char *data, *d_data;
    data = (char *)malloc(N);
    cudaMalloc(&d_data, N);
    fread(data, 1, N, fp);
    cudaMemcpy(d_data, data, N, cudaMemcpyHostToDevice);
    qsort_char<<<grid, block>>>(d_data, N, 1);
    cudaMemcpy(data, d_data, N, cudaMemcpyDeviceToHost);
    use_data(data);
    cudaFree(d_data);
    free(data);
}
