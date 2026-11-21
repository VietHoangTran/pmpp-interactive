// Declarations that let ../overlap_loop.cu and ../nccl_exchange.cu pass the CUDA front-end
// check (clang's CUDA parser against labs/common/include/pmpp/check/cuda_stub.h). They are
// NOT the real <mpi.h> or <nccl.h> and nothing here is ever linked or run: the snippets need
// MPI, NCCL and several GPUs, which this course does not assume.
//
// Signatures: MPI_Sendrecv and MPI_Allreduce from the MPI 5.0 report (notes/today/
// ch23-facts.md, A10); ncclSend, ncclRecv, ncclGroupStart, ncclGroupEnd and ncclCommInitRank
// from the NCCL 2.32 API reference (A12); cudaDeviceGetStreamPriorityRange and
// cudaStreamCreateWithPriority from the CUDA 13.5 runtime reference (A1). The opaque types
// and constant values are placeholders: only their spelling is checked.
#pragma once
#include <cstddef>

// ---- CUDA calls the course's stub header once lacked (ch23-facts, "Notes for the lab runtime");
// the stub now declares them too, and these identical redeclarations still compile
extern "C" {
cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority);
}

// ---- MPI (A10)
typedef struct pmpp_mpi_comm* MPI_Comm;
typedef struct pmpp_mpi_datatype* MPI_Datatype;
typedef struct pmpp_mpi_op* MPI_Op;
struct MPI_Status { int MPI_SOURCE, MPI_TAG, MPI_ERROR; };
extern MPI_Comm MPI_COMM_WORLD;
extern MPI_Datatype MPI_FLOAT;
extern MPI_Op MPI_SUM;
#define MPI_STATUS_IGNORE ((MPI_Status*)0)
int MPI_Sendrecv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                 void* recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
                 MPI_Comm comm, MPI_Status* status);
int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

// ---- NCCL (A12)
typedef enum { ncclSuccess = 0 } ncclResult_t;
typedef enum { ncclFloat = 7 } ncclDataType_t;
typedef struct pmpp_nccl_comm* ncclComm_t;
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclGroupStart();
ncclResult_t ncclGroupEnd();

// ---- the book's host wrapper, used in Figs. 23.14 and 23.20 but never shown (notes/ch23.md,
// "Code listings"): launches Fig. 23.1's jacobi_kernel on `ny` rows starting at `out`/`in`.
void launch_jacobi_kernel(float* out, float* in, float* l2norm, unsigned int ny, unsigned int nx, cudaStream_t stream);
