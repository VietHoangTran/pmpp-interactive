// Vector addition on the host alone: the book's Fig. 2.4 with this course's
// naming, where _h marks data that lives in host memory. There is no device
// code in this file, and it is still a CUDA C++ program (book p. 24).

// Compute vector sum C_h = A_h + B_h
void vecAdd(const float* A_h, const float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; ++i) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

int main() {
    // Stands for allocating A, B and C and reading A and B: N elements each.
    const int N = 8;
    float A[N] = {3, 1, 4, 1, 5, 9, 2, 6};
    float B[N] = {2, 7, 1, 8, 2, 8, 1, 8};
    float C[N];

    vecAdd(A, B, C, N);
    return 0;
}
