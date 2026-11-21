// Fig. 23.14 (p. 557): the host loop of the MPI version that overlaps computation and
// communication with three streams and three events, together with the setup the book shows
// in pieces in the text. The course page 23.3–23.5 (route ch23/23.4) shows the two marked regions; the loop keeps the
// book's line numbers 01 to 43, and app/src/sim/haloloop.ts cites the same numbers.
//
// Differences from the page:
//   - line 19 is printed with an extra ')' ("computeTop, 0));"), which does not compile;
//     Fig. 23.20 prints the same line correctly.
//   - the comment lines 03, 09, 18, 24 and 36 are ours.
//   - the setup gathers what the book shows in the text: the priority range and the three
//     streams (p. 558), the events (p. 560; the book shows the call for resetL2 only, the
//     other two are created the same way here), the pinned norm (p. 562). The starting
//     value of l2norm is ours: the book never shows it.
//
// Status: front-end-checked (clang's CUDA parser against the course's stub headers and
// check/mpi_nccl_decls.h). Never run: it needs MPI and one GPU per rank.
#include <cmath>
#include <utility>
#include "check/mpi_nccl_decls.h"

void jacobi_overlap(float* output, float* input, float* l2norm_d, unsigned int nx, unsigned int ny,
                    float tol, unsigned int iter_max, int rank, int numRanks) {
    // [setup]
    int low, high;                                    // least, then greatest priority
    cudaDeviceGetStreamPriorityRange(&low, &high);    // a lower number is a higher priority
    cudaStream_t topStream, bottomStream, internalStream;
    cudaStreamCreateWithPriority(&internalStream, cudaStreamDefault, low);
    cudaStreamCreateWithPriority(&topStream, cudaStreamDefault, high);
    cudaStreamCreateWithPriority(&bottomStream, cudaStreamDefault, high);

    cudaEvent_t resetL2, computeTop, computeBottom;   // only ever waited on: no timing
    cudaEventCreateWithFlags(&resetL2, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&computeTop, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&computeBottom, cudaEventDisableTiming);

    float* l2norm_h;                                  // pinned, so line 21 stays asynchronous
    cudaMallocHost(&l2norm_h, sizeof(float));
    float l2norm = 1.0f;                              // anything above tol
    // [/setup]
    // [loop]
    for(unsigned int iter = 0; l2norm > tol && iter < iter_max; ++iter) {

        // zero the norm, then make both boundary streams wait for that
        cudaMemsetAsync(l2norm_d, 0, sizeof(float), internalStream);
        cudaEventRecord(resetL2, internalStream);
        cudaStreamWaitEvent(topStream, resetL2, 0);
        cudaStreamWaitEvent(bottomStream, resetL2, 0);

        // one kernel, three launches: row 1, row ny - 2, rows 2 to ny - 3
        launch_jacobi_kernel(output, input, l2norm_d, 3, nx, topStream);
        cudaEventRecord(computeTop, topStream);
        launch_jacobi_kernel(output + (ny - 3)*nx, input + (ny - 3)*nx, l2norm_d, 3,
                             nx, bottomStream);
        cudaEventRecord(computeBottom, bottomStream);
        launch_jacobi_kernel(output + nx, input + nx, l2norm_d, ny - 2, nx,
                             internalStream);

        // the copy of the norm waits for all three kernels
        cudaStreamWaitEvent(internalStream, computeTop, 0);
        cudaStreamWaitEvent(internalStream, computeBottom, 0);
        cudaMemcpyAsync(l2norm_h, l2norm_d, sizeof(float), cudaMemcpyDeviceToHost,
                        internalStream);

        // halo exchange: the host waits for each boundary row, then sends it
        const int top = rank > 0 ? rank - 1 : (numRanks - 1);
        const int bottom = (rank + 1) % numRanks;
        cudaStreamSynchronize(topStream);
        MPI_Sendrecv(output + nx, nx, MPI_FLOAT, top, 0,
                     output + (ny - 1)*nx, nx, MPI_FLOAT, bottom, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cudaStreamSynchronize(bottomStream);
        MPI_Sendrecv(output + (ny - 2) * nx, nx, MPI_FLOAT, bottom, 0,
                     output, nx, MPI_FLOAT, top, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // the host needs the norm: wait for the copy, then reduce across ranks
        cudaStreamSynchronize(internalStream);
        MPI_Allreduce(l2norm_h, &l2norm, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        l2norm = std::sqrt(l2norm);

        std::swap(output, input);

    }
    // [/loop]
    cudaFreeHost(l2norm_h);
    cudaEventDestroy(resetL2);
    cudaEventDestroy(computeTop);
    cudaEventDestroy(computeBottom);
    cudaStreamDestroy(topStream);
    cudaStreamDestroy(bottomStream);
    cudaStreamDestroy(internalStream);
}
