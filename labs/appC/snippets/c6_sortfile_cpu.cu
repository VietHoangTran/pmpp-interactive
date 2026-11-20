// Fig. C.1, left box (book p. 627): the CPU original of sortfile.
// Host code only. The book shows no definitions of qsort_char or use_data, so they are
// declared here and never defined; the file is parsed, not built or run.
// The blank line inside the function is the figure's: the managed port puts
// cudaDeviceSynchronize() there.
#include <cstdio>
#include <cstdlib>

void qsort_char(char *data, int n, int elemSize);
void use_data(char *data);

void sortfile(FILE *fp, int N) {
    char *data;
    data = (char *)malloc(N);
    fread(data, 1, N, fp);
    qsort_char(data, N, 1);

    use_data(data);
    free(data);
}
