// Fig. C.1, right box (book p. 627): sortfile ported with Unified Memory.
// Host code only. The book shows no definitions of the sort kernel or use_data, so they
// are declared here and never defined; the file is parsed, not built or run.
// Where this differs from the page:
//   - the figure elides the launch configuration as <<<...>>>; grid and block below
//     stand in for whatever configuration a parallel sort would choose;
//   - the figure prints cudaDeviceSynchronize() without its semicolon; it is added.
#include <cstdio>

__global__ void qsort_char(char *data, int n, int elemSize);
void use_data(char *data);

static const dim3 grid(1), block(1);  // stand-ins for the figure's <<<...>>>

void sortfile(FILE *fp, int N) {
    char *data;
    cudaMallocManaged(&data, N);
    fread(data, 1, N, fp);
    qsort_char<<<grid, block>>>(data, N, 1);
    cudaDeviceSynchronize();
    use_data(data);
    cudaFree(data);
}
