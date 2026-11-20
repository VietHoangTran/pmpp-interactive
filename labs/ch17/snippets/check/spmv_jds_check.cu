// Runs ../spmv_jds.cu, the summary page's answer to exercise 5, on the course's CPU
// emulator and prints one line per run, which app/src/chapters/ch17/Wrap.snippets.test.ts
// reads. The JDS arrays are built here on the host (our code, not the book's): rows sorted
// longest first, ties in their original order, then stored column-major without padding.
// Values and x are small integers, so every correct order of float additions gives the
// exact answer and a comparison can be exact.
//
//   ./spmv_jds_check
//     arrays fig17_14: row <list> iterPtr <list>
//                           the host-built arrays of Fig. 17.14, to compare with p. 416
//     check jds <matrix> <order>: ok | wrong <rows>
//                           the kernel on Fig. 17.1, Fig. 17.14 (values a..o = 1..15, ours),
//                           exercise 1's matrix, a skewed 40 x 33 matrix and a matrix with
//                           empty rows, in the emulator's three thread orders
//     unpermuted: wrong <rows> of 6
//                           Fig. 17.14 with x all ones and y starting at 0, through a copy of
//                           the kernel whose last line writes y[r] instead of y[row[r]]
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "../spmv_jds.cu"

// The kernel with the result written to the sorted position: for the "unpermuted" line only.
__global__ void spmv_jds_unpermuted_kernel(JDSMatrix jdsMatrix, float* x, float* y) {
    unsigned int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r < jdsMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int t = 0; t < jdsMatrix.numIters && r < jdsMatrix.iterPtr[t + 1] - jdsMatrix.iterPtr[t]; ++t) {
            unsigned int i = jdsMatrix.iterPtr[t] + r;
            sum += x[jdsMatrix.colIdx[i]]*jdsMatrix.value[i];
        }
        y[r] += sum;
    }
}

#if defined(PMPP_EMU) && !defined(PMPP_EMU_RACE)
static const int kOrders = 3;
static const char* useOrder(int i) {
    using pmpp_emu::Schedule;
    pmpp_emu::set_schedule(i == 0 ? Schedule::Forward : i == 1 ? Schedule::Reverse : Schedule::Shuffle, 17);
    return i == 0 ? "forward" : i == 1 ? "reverse" : "shuffle";
}
#else
static const int kOrders = 1;
static const char* useOrder(int) { return "device"; }
#endif

struct Nz { unsigned r, c; float v; };
struct Host {
    const char* name;
    unsigned rows, cols;
    std::vector<Nz> nz;   // row order, columns ascending
};

static Host fig17_1() {
    return {"fig17_1", 4, 4, {{0, 0, 1}, {0, 1, 7}, {1, 0, 5}, {1, 2, 3}, {1, 3, 9}, {2, 1, 2}, {2, 2, 8}, {3, 3, 6}}};
}
// Fig. 17.14: a at (0,0); b..e in row 1 at columns 0, 1, 3, 4; f, g in row 2 at 2, 4; h..j in
// row 3 at 0, 3, 5; k, l in row 4 at 1, 4; m..o in row 5 at 0, 2, 5. Letters a..o are 1..15.
static Host fig17_14() {
    const unsigned cols[6][4] = {{0}, {0, 1, 3, 4}, {2, 4}, {0, 3, 5}, {1, 4}, {0, 2, 5}};
    const unsigned lens[6] = {1, 4, 2, 3, 2, 3};
    Host h{"fig17_14", 6, 6, {}};
    float v = 0;
    for (unsigned r = 0; r < 6; ++r)
        for (unsigned k = 0; k < lens[r]; ++k) h.nz.push_back({r, cols[r][k], ++v});
    return h;
}
static Host exercise1() {
    return {"exercise1", 4, 4, {{0, 0, 1}, {0, 2, 7}, {1, 2, 8}, {2, 1, 4}, {2, 2, 3}, {3, 0, 2}, {3, 3, 1}}};
}
// Row r has (7r mod 5) non-zeros, and row 13 twenty more: one long row, as on p. 412.
static Host skewed() {
    Host h{"skewed", 40, 33, {}};
    for (unsigned r = 0; r < h.rows; ++r) {
        unsigned len = (7 * r) % 5 + (r == 13 ? 20 : 0);
        std::vector<bool> used(h.cols, false);
        std::vector<unsigned> cs;
        for (unsigned k = 0; cs.size() < len; ++k) {
            unsigned c = (3 * r + 5 * k) % h.cols;
            if (!used[c]) { used[c] = true; cs.push_back(c); }
        }
        std::sort(cs.begin(), cs.end());
        for (unsigned k = 0; k < cs.size(); ++k) h.nz.push_back({r, cs[k], (float)(1 + (r + k) % 4)});
    }
    return h;
}
// Every third row empty, the others of length 1 to 3: empty rows sort to the end.
static Host withEmptyRows() {
    Host h{"empty_rows", 12, 7, {}};
    for (unsigned r = 0; r < h.rows; ++r) {
        if (r % 3 == 0) continue;
        unsigned len = 1 + r % 3 + (r % 4 == 1 ? 1 : 0);
        for (unsigned k = 0; k < len; ++k) h.nz.push_back({r, (r + 2 * k) % h.cols, (float)(1 + (r + k) % 5)});
    }
    std::sort(h.nz.begin(), h.nz.end(), [](const Nz& a, const Nz& b) { return a.r != b.r ? a.r < b.r : a.c < b.c; });
    return h;
}

static std::vector<float> reference(const Host& h, const std::vector<float>& x, const std::vector<float>& y0) {
    std::vector<float> y = y0;
    for (const Nz& e : h.nz) y[e.r] += e.v * x[e.c];
    return y;
}

template <class T>
static T* toDevice(const std::vector<T>& v) {
    T* p = nullptr;
    cudaMalloc((void**)&p, (v.empty() ? 1 : v.size()) * sizeof(T));
    if (!v.empty()) cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}

struct HostJds { unsigned numIters; std::vector<unsigned> row, iterPtr, colIdx; std::vector<float> value; };
static HostJds buildJds(const Host& h) {
    std::vector<std::vector<Nz>> lists(h.rows);
    for (const Nz& e : h.nz) lists[e.r].push_back(e);
    HostJds j{0, {}, {0}, {}, {}};
    for (unsigned r = 0; r < h.rows; ++r) j.row.push_back(r);
    std::stable_sort(j.row.begin(), j.row.end(), [&](unsigned a, unsigned b) { return lists[a].size() > lists[b].size(); });
    for (const auto& l : lists) j.numIters = std::max<unsigned>(j.numIters, (unsigned)l.size());
    for (unsigned t = 0; t < j.numIters; ++t) {
        for (unsigned r : j.row)
            if (lists[r].size() > t) { j.colIdx.push_back(lists[r][t].c); j.value.push_back(lists[r][t].v); }
        j.iterPtr.push_back((unsigned)j.colIdx.size());
    }
    return j;
}
static JDSMatrix upload(const Host& h, const HostJds& j) {
    return {h.rows, j.numIters, toDevice(j.row), toDevice(j.iterPtr), toDevice(j.colIdx), toDevice(j.value)};
}

static int wrongRows(const std::vector<float>& got, const std::vector<float>& want) {
    int n = 0;
    for (size_t i = 0; i < want.size(); ++i) n += !(got[i] == want[i]);
    return n;
}

static std::string join(const std::vector<unsigned>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s;
}

int main() {
    {
        const HostJds j = buildJds(fig17_14());
        std::printf("arrays fig17_14: row %s iterPtr %s\n", join(j.row).c_str(), join(j.iterPtr).c_str());
    }

    const unsigned block = 4;   // several blocks even on the small matrices
    for (Host h : {fig17_1(), fig17_14(), exercise1(), skewed(), withEmptyRows()}) {
        std::vector<float> x(h.cols), y0(h.rows);
        for (unsigned c = 0; c < h.cols; ++c) x[c] = (float)(1 + c % 5);
        for (unsigned r = 0; r < h.rows; ++r) y0[r] = (float)(10 * (r % 7));
        const std::vector<float> want = reference(h, x, y0);
        const JDSMatrix d = upload(h, buildJds(h));
        float* x_d = toDevice(x);
        float* y_d = toDevice(y0);
        for (int o = 0; o < kOrders; ++o) {
            const char* order = useOrder(o);
            cudaMemcpy(y_d, y0.data(), h.rows * sizeof(float), cudaMemcpyHostToDevice);
            spmv_jds_kernel<<<(h.rows + block - 1) / block, block>>>(d, x_d, y_d);
            cudaDeviceSynchronize();
            std::vector<float> got(h.rows);
            cudaMemcpy(got.data(), y_d, h.rows * sizeof(float), cudaMemcpyDeviceToHost);
            const int bad = wrongRows(got, want);
            if (bad) std::printf("check jds %s %s: wrong %d\n", h.name, order, bad);
            else std::printf("check jds %s %s: ok\n", h.name, order);
        }
    }

    // The result written to the sorted position instead of the original row.
    {
        useOrder(0);
        const Host h = fig17_14();
        std::vector<float> x(6, 1.0f), y0(6, 0.0f);
        const std::vector<float> want = reference(h, x, y0);
        const JDSMatrix d = upload(h, buildJds(h));
        float* x_d = toDevice(x);
        float* y_d = toDevice(y0);
        spmv_jds_unpermuted_kernel<<<2, 4>>>(d, x_d, y_d);
        cudaDeviceSynchronize();
        std::vector<float> got(6);
        cudaMemcpy(got.data(), y_d, 6 * sizeof(float), cudaMemcpyDeviceToHost);
        std::printf("unpermuted: wrong %d of 6\n", wrongRows(got, want));
    }
    std::printf("DONE\n");
    return 0;
}
