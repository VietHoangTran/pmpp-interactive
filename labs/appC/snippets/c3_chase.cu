// A linked list that the host built. The kernel can follow it only if every
// node is in memory a kernel may dereference; under UVA (CUDA 4) that means
// cudaHostAlloc or device memory, never plain malloc memory.
struct Node {
    float value;
    Node* next;
};

// One thread walks the list and sums it: pure pointer chasing, no parallelism.
__global__ void sumList(const Node* head, float* sum) {
    float s = 0.0f;
    for (const Node* p = head; p != nullptr; p = p->next) s += p->value;
    *sum = s;
}

void launchSumList(const Node* head, float* sum) {
    sumList<<<1, 1>>>(head, sum);
}
