// Runs the four kernels page 17.2 shows (../spmv_coo.cu, ../spmv_csr.cu, ../spmv_ell.cu,
// ../spmv_csc.cu) on the course's CPU emulator and prints one line per run, which
// app/src/chapters/ch17/snippets-s172.test.ts reads. The formats are built here on the host
// (our code, not the book's). Values and x are small integers, so every correct order of
// float additions gives the exact answer and a comparison can be exact.
//
//   ./spmv_check            all kernels on Fig. 17.1 and on a skewed 40 x 33 matrix, in the
//                           emulator's three thread orders, then two demonstrations:
//     check <kernel> <matrix> <order>: ok | wrong <rows>
//     assign <variant>: wrong <rows> of <rows>
//                           the hybrid of p. 414 run COO part first, then the ELL part,
//                           with the page's y[row] = sum (a copy below) or the snippet's +=
//     padnan <marker>: nan <rows>
//                           Fig. 17.1 in ELL with pads holding column <marker>, x[0] = +Inf,
//                           the full-width kernel: rows that come out NaN
//   ./spmv_check race-atomic | race-plain
//                           one COO run on the skewed matrix with the snippet's atomic, or
//                           with a plain y[row] += (a copy below), for the race emulator
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../spmv_coo.cu"
#include "../spmv_csr.cu"
#include "../spmv_ell.cu"
#include "../spmv_csc.cu"

// The page's last line of Fig. 17.12, for the "assign" demonstration only.
__global__ void spmv_ell_assign_kernel(ELLMatrix ellMatrix, float* x, float* y) {
    unsigned int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < ellMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int t = 0; t < ellMatrix.nnzPerRow[row]; ++t) {
            unsigned int i = t*ellMatrix.stride + row;
            sum += x[ellMatrix.colIdx[i]]*ellMatrix.value[i];
        }
        y[row] = sum;
    }
}

// SpMV/COO without the atomic, for the race emulator only.
__global__ void spmv_coo_plain_kernel(COOMatrix cooMatrix, float* x, float* y) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < cooMatrix.numNonzeros) {
        unsigned int row = cooMatrix.rowIdx[i];
        y[row] += x[cooMatrix.colIdx[i]]*cooMatrix.value[i];
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

struct HostEll { unsigned stride, width; std::vector<unsigned> nnzPerRow; std::vector<int> colIdx; std::vector<float> value; };
// Rows truncated to `keep` elements (a hybrid's ELL part) when keep is given.
static HostEll buildEll(const Host& h, unsigned stride, int padMarker, unsigned keep = ~0u) {
    HostEll e{stride, 0, std::vector<unsigned>(h.rows, 0), {}, {}};
    for (const Nz& n : h.nz) if (e.nnzPerRow[n.r] < keep) e.nnzPerRow[n.r]++;
    for (unsigned r = 0; r < h.rows; ++r) e.width = e.nnzPerRow[r] > e.width ? e.nnzPerRow[r] : e.width;
    e.colIdx.assign(stride * e.width, padMarker);
    e.value.assign(stride * e.width, 0.0f);
    std::vector<unsigned> t(h.rows, 0);
    for (const Nz& n : h.nz) {
        if (t[n.r] >= keep) continue;
        unsigned i = t[n.r]++ * stride + n.r;
        e.colIdx[i] = (int)n.c;
        e.value[i] = n.v;
    }
    return e;
}

struct Device {
    COOMatrix coo; CSRMatrix csr; CSCMatrix csc; ELLMatrix ell, ellAligned;
};

static ELLMatrix ellOnDevice(const Host& h, const HostEll& e) {
    return {h.rows, e.stride, e.width, toDevice(e.nnzPerRow), toDevice(e.colIdx), toDevice(e.value)};
}

static Device upload(const Host& h) {
    Device d{};
    std::vector<unsigned> rowIdx, colIdx, rowPtrs(h.rows + 1, 0), colPtrs(h.cols + 1, 0), cscRow;
    std::vector<float> value, cscVal;
    for (const Nz& e : h.nz) { rowIdx.push_back(e.r); colIdx.push_back(e.c); value.push_back(e.v); rowPtrs[e.r + 1]++; colPtrs[e.c + 1]++; }
    for (unsigned r = 0; r < h.rows; ++r) rowPtrs[r + 1] += rowPtrs[r];
    for (unsigned c = 0; c < h.cols; ++c) colPtrs[c + 1] += colPtrs[c];
    cscRow.resize(h.nz.size());
    cscVal.resize(h.nz.size());
    std::vector<unsigned> cur(colPtrs.begin(), colPtrs.end() - 1);
    for (const Nz& e : h.nz) { cscRow[cur[e.c]] = e.r; cscVal[cur[e.c]++] = e.v; }   // rows ascending within a column
    d.coo = {(unsigned)h.nz.size(), toDevice(rowIdx), toDevice(colIdx), toDevice(value)};
    d.csr = {h.rows, toDevice(rowPtrs), toDevice(colIdx), toDevice(value)};
    d.csc = {h.cols, toDevice(colPtrs), toDevice(cscRow), toDevice(cscVal)};
    d.ell = ellOnDevice(h, buildEll(h, h.rows, -1));
    d.ellAligned = ellOnDevice(h, buildEll(h, (h.rows + 7) / 8 * 8, -1));   // every iteration starts on a multiple of 8 floats
    return d;
}

static int wrongRows(const std::vector<float>& got, const std::vector<float>& want) {
    int n = 0;
    for (size_t i = 0; i < want.size(); ++i) n += !(got[i] == want[i]);
    return n;
}

static void runAll() {
    const unsigned block = 8;
    for (Host h : {fig17_1(), skewed()}) {
        std::vector<float> x(h.cols), y0(h.rows);
        for (unsigned c = 0; c < h.cols; ++c) x[c] = (float)(1 + c % 5);
        for (unsigned r = 0; r < h.rows; ++r) y0[r] = (float)(10 * (r % 7));
        const std::vector<float> want = reference(h, x, y0);
        Device d = upload(h);
        float* x_d = toDevice(x);
        float* y_d = toDevice(y0);
        const char* names[] = {"coo", "csr", "ell", "ell-full", "ell-aligned", "ell-aligned-full", "csc"};
        for (int k = 0; k < 7; ++k)
            for (int o = 0; o < kOrders; ++o) {
                const char* order = useOrder(o);
                cudaMemcpy(y_d, y0.data(), h.rows * sizeof(float), cudaMemcpyHostToDevice);
                const unsigned n = k == 0 ? d.coo.numNonzeros : k == 6 ? h.cols : h.rows;
                const unsigned grid = (n + block - 1) / block;
                switch (k) {
                    case 0: spmv_coo_kernel<<<grid, block>>>(d.coo, x_d, y_d); break;
                    case 1: spmv_csr_kernel<<<grid, block>>>(d.csr, x_d, y_d); break;
                    case 2: spmv_ell_kernel<<<grid, block>>>(d.ell, x_d, y_d); break;
                    case 3: spmv_ell_full_kernel<<<grid, block>>>(d.ell, x_d, y_d); break;
                    case 4: spmv_ell_kernel<<<grid, block>>>(d.ellAligned, x_d, y_d); break;
                    case 5: spmv_ell_full_kernel<<<grid, block>>>(d.ellAligned, x_d, y_d); break;
                    default: spmv_csc_kernel<<<grid, block>>>(d.csc, x_d, y_d); break;
                }
                cudaDeviceSynchronize();
                std::vector<float> got(h.rows);
                cudaMemcpy(got.data(), y_d, h.rows * sizeof(float), cudaMemcpyDeviceToHost);
                const int bad = wrongRows(got, want);
                if (bad) std::printf("check %s %s %s: wrong %d\n", names[k], h.name, order, bad);
                else std::printf("check %s %s %s: ok\n", names[k], h.name, order);
            }
    }

    // The hybrid of p. 414 on Fig. 17.1 with K = 2: row 1's third non-zero goes to the COO part.
    // COO part first, then the ELL part.
    {
        useOrder(0);
        Host h = fig17_1();
        std::vector<float> x = {1, 2, 3, 4}, y0 = {10, 20, 30, 40};
        const std::vector<float> want = reference(h, x, y0);
        Host tail{"tail", 4, 4, {{1, 3, 9}}};
        Device dt = upload(tail);
        ELLMatrix part = ellOnDevice(h, buildEll(h, h.rows, -1, 2));
        float* x_d = toDevice(x);
        float* y_d = toDevice(y0);
        for (int v = 0; v < 2; ++v) {
            cudaMemcpy(y_d, y0.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);
            spmv_coo_kernel<<<1, 8>>>(dt.coo, x_d, y_d);
            if (v == 0) spmv_ell_assign_kernel<<<1, 8>>>(part, x_d, y_d);
            else spmv_ell_kernel<<<1, 8>>>(part, x_d, y_d);
            cudaDeviceSynchronize();
            std::vector<float> got(4);
            cudaMemcpy(got.data(), y_d, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            std::printf("assign %s: wrong %d of 4\n", v == 0 ? "page" : "snippet", wrongRows(got, want));
        }
    }

    // Pads holding column 0 instead of -1, with x[0] infinite: 0 * Inf is NaN.
    for (int marker : {0, -1}) {
        useOrder(0);
        Host h = fig17_1();
        std::vector<float> x = {INFINITY, 1, 1, 1}, y0 = {0, 0, 0, 0};
        ELLMatrix e = ellOnDevice(h, buildEll(h, h.rows, marker));
        float* x_d = toDevice(x);
        float* y_d = toDevice(y0);
        spmv_ell_full_kernel<<<1, 8>>>(e, x_d, y_d);
        cudaDeviceSynchronize();
        std::vector<float> got(4);
        cudaMemcpy(got.data(), y_d, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        int nan = 0;
        for (float v : got) nan += std::isnan(v);
        std::printf("padnan %d: nan %d\n", marker, nan);
    }
}

static void runRace(bool atomic) {
    Host h = skewed();
    std::vector<float> x(h.cols, 1.0f), y0(h.rows, 0.0f);
    Device d = upload(h);
    float* x_d = toDevice(x);
    float* y_d = toDevice(y0);
    const unsigned grid = (d.coo.numNonzeros + 31) / 32;
    if (atomic) spmv_coo_kernel<<<grid, 32>>>(d.coo, x_d, y_d);
    else spmv_coo_plain_kernel<<<grid, 32>>>(d.coo, x_d, y_d);
    cudaDeviceSynchronize();
    std::printf("race %s: launched\n", atomic ? "atomic" : "plain");
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "race-atomic") == 0) runRace(true);
    else if (argc > 1 && std::strcmp(argv[1], "race-plain") == 0) runRace(false);
    else runAll();
    std::printf("DONE\n");
    return 0;
}
