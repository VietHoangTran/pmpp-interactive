// Fig. 23.20 (p. 568), lines 27 to 44: the NCCL version's halo exchange and the end of its
// loop body. Lines 01 to 26 are those of Fig. 23.14 (overlap_loop.cu); line 46 swaps the
// buffers. Section 23.4 shows the marked region, numbered from 27 as on the page, and
// app/src/sim/haloloop.ts cites the same numbers.
//
// Setup this region assumes (Fig. 23.17, p. 565; notes/today/ch23-facts.md A12):
//   rank 0 creates the id with ncclGetUniqueId and every rank receives it with MPI_Bcast;
//   each rank then selects its own GPU with cudaSetDevice BEFORE calling
//   ncclCommInitRank(&nccl_comm, numRanks, id, rank). NCCL's reference requires the device to
//   be set first; Fig. 23.17 does not show it. The events exchangeTop and exchangeBottom are
//   created like the others, with cudaEventDisableTiming.
//
// Differences from the page: the comment on line 38 is ours. The page calls the communicator
// nccl_comm here and ncclComm in Fig. 23.17; this file keeps nccl_comm.
//
// Status: front-end-checked (clang's CUDA parser against the course's stub headers and
// check/mpi_nccl_decls.h). Never run: it needs MPI, NCCL and one GPU per rank.
#include <cmath>
#include "check/mpi_nccl_decls.h"

void nccl_exchange(float* output, float* l2norm_h, float& l2norm, unsigned int nx, unsigned int ny,
                   int top, int bottom, ncclComm_t nccl_comm,
                   cudaStream_t topStream, cudaStream_t bottomStream, cudaStream_t internalStream,
                   cudaEvent_t exchangeTop, cudaEvent_t exchangeBottom) {
    // [exchange]
    ncclGroupStart();
    ncclSend(output + nx,          nx, ncclFloat, top,    nccl_comm, topStream);
    ncclRecv(output + (ny - 1)*nx, nx, ncclFloat, bottom, nccl_comm, topStream);
    ncclGroupEnd();
    cudaEventRecord(exchangeTop, topStream);
    ncclGroupStart();
    ncclSend(output + (ny - 2)*nx, nx, ncclFloat, bottom, nccl_comm, bottomStream);
    ncclRecv(output,               nx, ncclFloat, top,    nccl_comm, bottomStream);
    ncclGroupEnd();
    cudaEventRecord(exchangeBottom, bottomStream);

    // the host's one wait: it needs the norm
    cudaStreamSynchronize(internalStream);
    MPI_Allreduce(l2norm_h, &l2norm, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    l2norm = std::sqrt(l2norm);

    cudaStreamWaitEvent(internalStream, exchangeTop, 0);
    cudaStreamWaitEvent(internalStream, exchangeBottom, 0);
    // [/exchange]
}
