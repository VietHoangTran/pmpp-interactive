// Lab 15.6, provided: the vendor library's GEMM as the baseline for milestone M3, on CUDA
// (cuBLAS) and HIP (hipBLAS). You do not edit or call this; the benchmark does.
//
//   Y[batch x out] = X[batch x in] * W^T,  W stored (out, in) row-major.
//
// cuBLAS and hipBLAS are column-major (ch03-facts A7, B6). A row-major array read
// column-major is its transpose, so the call asks for Y^T = W * X^T (our derivation, the
// standard trick, ch15-facts A10):
//   - Y (batch x out, row-major) is, column-major, Y^T: out x batch, leading dimension out.
//   - W (out x in, row-major) is, column-major, W^T: in x out, ld in. OP_T turns it back
//     into W: out x in. So m = out, k = in.
//   - X (batch x in, row-major) is, column-major, X^T: in x batch, ld in. OP_N. n = batch.
//   cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, out, batch, in, &1, W, in, X, in, &0, Y, out)
// The tests check this call against the CPU reference, on every GPU backend.
//
// Precision: a handle never given a math mode computes FP32 in FP32 (ch15-facts A10,
// B6); this file never sets one. NVIDIA's TF32 and BF16x9 modes and the environment
// variables that switch them on are reported by vendorMath().
//
// HIP: the cuBLAS names are mapped to hipBLAS by #define, as llama.cpp's vendors/hip.h
// does (ch05-facts B9, ch15-facts B6); hipblasSgemm takes the same arguments.
//
// Built into every backend of the lab. Where no vendor library was found at configure
// time (and on the emulators and in the CUDA front-end check) PMPP_HAVE_VENDOR_BLAS is
// not defined and the functions below report that there is none.
//
// UNVERIFIED on CUDA and HIP: written on a machine with neither. If it does not build,
// configure with -DPMPP_CH15_VENDOR_BLAS=OFF and the lab works without the vendor rows.
#include <cstdlib>
#include <string>

#if defined(PMPP_HAVE_VENDOR_BLAS) && defined(PMPP_BACKEND_HIP)
#include <hipblas/hipblas.h>
#define cublasHandle_t        hipblasHandle_t
#define cublasStatus_t        hipblasStatus_t
#define cublasCreate          hipblasCreate
#define cublasSgemm           hipblasSgemm
#define CUBLAS_OP_N           HIPBLAS_OP_N
#define CUBLAS_OP_T           HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#elif defined(PMPP_HAVE_VENDOR_BLAS)
#include <cublas_v2.h>
#endif

#if defined(PMPP_HAVE_VENDOR_BLAS)

// Returns false if the library could not be set up or refused the call.
bool launchLinearVendor(const float* X, const float* W, float* Y, int batch, int in, int out) {
    static cublasHandle_t handle = nullptr;
    if (!handle && cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        handle = nullptr;
        return false;
    }
    const float one = 1.0f, zero = 0.0f;
    const cublasStatus_t s = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out, batch, in,
                                         &one, W, in, X, in, &zero, Y, out);
    return s == CUBLAS_STATUS_SUCCESS;
}

static std::string envValue(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(name) + "=" + v : std::string(name) + " not set";
}

#if defined(PMPP_BACKEND_HIP)
const char* vendorName() { return "hipBLAS hipblasSgemm"; }
std::string vendorMath() {
    return "HIPBLAS_DEFAULT_MATH (the handle's math mode is never changed here; XF32 is opt-in)";
}
#else
const char* vendorName() { return "cuBLAS cublasSgemm"; }
std::string vendorMath() {
    return "CUBLAS_DEFAULT_MATH (the handle's math mode is never changed here); " +
           envValue("NVIDIA_TF32_OVERRIDE") + "; " + envValue("CUBLAS_EMULATE_SINGLE_PRECISION");
}
#endif

#else  // no vendor library on this backend

bool launchLinearVendor(const float*, const float*, float*, int, int, int) { return false; }
const char* vendorName() { return "none"; }
std::string vendorMath() { return "no vendor library on this backend"; }

#endif
