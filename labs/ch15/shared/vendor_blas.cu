// Labs 15.1 to 15.5, provided and driver-side: you do not edit or call this.
// The vendor's single-precision GEMM, the baseline milestone M3 is measured against
// (notes/today/ch15-facts.md A10, B6). The book names cuBLAS (p. 368); hipBLAS is ours.
//   CUDA:      cublasSgemm  (cublas_v2.h, linked as CUDA::cublas)
//   HIP:       hipblasSgemm (hipblas/hipblas.h, linked as roc::hipblas)
//   emulators: none; the benchmark does not run there.
//
// Both libraries are column-major (ch03-facts A7, B6). A row-major M x N matrix read as
// column-major is its transpose, so the row-major C = A B is computed as the column-major
// C^T = B^T A^T: B goes first and the sizes are (N, M, K) (A10; llama.cpp does the same,
// ch03-facts D2). The benchmark checks the result against the CPU reference before it
// times anything, so a wrong argument order cannot pass unnoticed.
//
// The handle is left in its default math mode (CUBLAS_DEFAULT_MATH, HIPBLAS_DEFAULT_MATH),
// in which a single-precision GEMM computes in FP32; TF32, BF16x9 and XF32 are opt-in
// (A10, B6). NVIDIA documents two environment variables that change this without any
// code: NVIDIA_TF32_OVERRIDE and CUBLAS_EMULATE_SINGLE_PRECISION. ch15VendorName() says
// whether either is set.
//
// UNVERIFIED on CUDA and HIP: written on a machine with neither, from the declarations
// quoted in the facts file. If it does not build, configure once with
//   cmake -S labs -B <your build directory> -DPMPP_CH15_VENDOR=OFF
// and the benchmark runs without the vendor's column. ch15.cmake defines
// PMPP_CH15_VENDOR_MISSING when it cannot find the library on this machine.
#include <cstdlib>
#include <string>

#if defined(PMPP_BACKEND_CUDA) && !defined(PMPP_CH15_NO_VENDOR) && !defined(PMPP_CH15_VENDOR_MISSING)
#define CH15_CUBLAS 1
#include <cublas_v2.h>
#elif defined(PMPP_BACKEND_HIP) && !defined(PMPP_CH15_NO_VENDOR) && !defined(PMPP_CH15_VENDOR_MISSING)
#define CH15_HIPBLAS 1
#include <hipblas/hipblas.h>
#endif

#if defined(CH15_CUBLAS)

static std::string envNote(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string("; ") + name + "=" + v + " is set" : std::string();
}

std::string ch15VendorName() {
    return std::string("cuBLAS cublasSgemm, math mode CUBLAS_DEFAULT_MATH (this lab never changes it)") +
           envNote("NVIDIA_TF32_OVERRIDE") + envNote("CUBLAS_EMULATE_SINGLE_PRECISION");
}

std::string ch15VendorGemm(const float* A, const float* B, float* C, int M, int K, int N) {
    static cublasHandle_t handle = nullptr;   // created once, kept for the life of the program
    if (!handle && cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        handle = nullptr;
        return "cublasCreate failed";
    }
    const float one = 1.0f, zero = 0.0f;
    // Column-major C^T (N x M) = B^T (N x K) times A^T (K x M).
    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N) !=
        CUBLAS_STATUS_SUCCESS) {
        return "cublasSgemm failed";
    }
    return "";
}

#elif defined(CH15_HIPBLAS)

std::string ch15VendorName() {
    return "hipBLAS hipblasSgemm, math mode HIPBLAS_DEFAULT_MATH (this lab never changes it)";
}

std::string ch15VendorGemm(const float* A, const float* B, float* C, int M, int K, int N) {
    static hipblasHandle_t handle = nullptr;
    if (!handle && hipblasCreate(&handle) != HIPBLAS_STATUS_SUCCESS) {
        handle = nullptr;
        return "hipblasCreate failed";
    }
    const float one = 1.0f, zero = 0.0f;
    if (hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N) !=
        HIPBLAS_STATUS_SUCCESS) {
        return "hipblasSgemm failed";
    }
    return "";
}

#else

static std::string why() {
#if defined(PMPP_CH15_NO_VENDOR)
    return "switched off at configure time (-DPMPP_CH15_VENDOR=OFF)";
#elif defined(PMPP_CH15_VENDOR_MISSING)
    return "the vendor library was not found when the build was configured";
#else
    return "not available on the emulator";
#endif
}

std::string ch15VendorName() { return "none: " + why(); }

std::string ch15VendorGemm(const float*, const float*, float*, int, int, int) { return why(); }

#endif
