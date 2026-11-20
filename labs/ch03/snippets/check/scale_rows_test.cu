// Runs ../scale_rows_defect.cu, the summary page's in-place kernel, on a 4 x 50 matrix of
// 2.0 and counts the elements left at 1.0 under each thread order. Column 0 is the only
// part the page calls guaranteed; the emulator's counts are the ones the page quotes.
#include <vector>
#include "check.h"

void launchScaleRows(float* X, int rows, int cols);

int main() {
    const int rows = 4, cols = 50, n = rows * cols;
#if defined(PMPP_EMU)
    const int quoted[3] = {4, 64, 37};   // app/src/chapters/ch03/Wrap.tsx, q-inplace-rows
#endif
    std::printf("scale_rows_defect.cu\n");
    float* X = nullptr;
    cudaMalloc((void**)&X, n * sizeof(float));
    for (int o = 0; o < kOrders; ++o) {
        const char* order = useOrder(o);
        std::vector<float> host(n, 2.0f);
        cudaMemcpy(X, host.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        launchScaleRows(X, rows, cols);
        cudaDeviceSynchronize();
        cudaMemcpy(host.data(), X, n * sizeof(float), cudaMemcpyDeviceToHost);
        int ones = 0;
        bool column0 = true, onlyOneOrTwo = true;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                const float v = host[r * cols + c];
                ones += v == 1.0f;
                onlyOneOrTwo = onlyOneOrTwo && (v == 1.0f || v == 2.0f);
                if (c == 0) column0 = column0 && v == 1.0f;
            }
        char what[160];
        std::snprintf(what, sizeof what, "%s: column 0 is 1.0 and every element is 1.0 or 2.0", order);
        expect(column0 && onlyOneOrTwo, what);
#if defined(PMPP_EMU)
        std::snprintf(what, sizeof what, "%s: %d of %d elements are 1.0 (the page says %d)", order, ones, n, quoted[o]);
        expect(ones == quoted[o], what);
#else
        std::printf("  %s: %d of %d elements are 1.0\n", order, ones, n);
#endif
    }
    cudaFree(X);
    return finish();
}
