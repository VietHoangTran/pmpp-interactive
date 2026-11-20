// Chapter 13, section 13.3: the sequential merge, co_rank and the basic merge kernel as
// the lesson shows them. Corrected from Figs. 13.2, 13.5 and 13.9 (pp. 305, 310, 313):
//   - __device__ on both helpers: an unqualified function is host-only on CUDA and HIP.
//   - const int* for the inputs.
//   - min and max written as ternaries, as Fig. 13.5 already does: the integer min/max
//     overloads differ between CUDA, HIP and the course's emulator (ch13-facts A5, B1).
//   - Fig. 13.5's comments fixed: the j_low line's comment said i_low, and the delta
//     comment had unbalanced parentheses.
//   - Fig. 13.9 line 03: integer ceiling division. As printed, ceil() receives an integer
//     quotient that has already been truncated, so it rounds nothing up (ch13-facts A4).
//   - Fig. 13.9 line 04: k_curr is clamped to m + n as k_next is (ours, not the book's).
// Each function keeps the book's line numbering. The lesson's copies are tied to this file
// by app/src/chapters/ch13/snippets-s133.test.ts; check.py runs it on the CPU emulator.

__device__ void merge_sequential(const int* A, int m, const int* B, int n, int* C) {
    int i = 0;  // index into A
    int j = 0;  // index into B
    int k = 0;  // index into C
    while ((i < m) && (j < n)) {  // both inputs have elements left
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    if (i == m) {  // A is used up: copy the rest of B
        while (j < n) {
            C[k++] = B[j++];
        }
    } else {  // B is used up: copy the rest of A
        while (i < m) {
            C[k++] = A[i++];
        }
    }
}

__device__ int co_rank(int k, const int* A, int m, const int* B, int n) {
    int i = k < m ? k : m;                // i = min(k, m)
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : k - n;  // i_low = max(0, k - n)
    int j_low = 0 > (k - m) ? 0 : k - m;  // j_low = max(0, k - m)
    int delta;
    bool active = true;
    while (active) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            delta = ((i - i_low + 1) >> 1);  // ceil((i - i_low) / 2)
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            delta = ((j - j_low + 1) >> 1);  // ceil((j - j_low) / 2)
            i_low = i;
            i = i + delta;
            j = j - delta;
        } else {
            active = false;
        }
    }
    return i;
}

__global__ void merge_basic_kernel(const int* A, int m, const int* B, int n, int* C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elementsPerThread = (m + n + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
    int k_curr = tid * elementsPerThread < m + n ? tid * elementsPerThread : m + n;
    int k_next = (tid + 1) * elementsPerThread < m + n ? (tid + 1) * elementsPerThread : m + n;
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}
