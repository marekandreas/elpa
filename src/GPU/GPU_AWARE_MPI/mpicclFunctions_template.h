#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#ifdef DEBUG_CUDA
#define debugmessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)
#else
#define debugmessage(x, ...)
#endif

#include <mpi.h>

typedef MPI_Comm mpicclComm_t;

typedef struct {
    intptr_t internal[16];
} ncclUniqueId;

#ifdef WITH_GPU_AWARE_MPICCL
extern "C" {

int mpicclGroupStartFromC() {
  return 1;
}

int mpicclGroupEndFromC() {
  return 1;
}


int mpicclGetUniqueIdFromC(ncclUniqueId *mpicclID)
{
  return 1;
}

size_t mpicclRequestSizeFromC(void) {
  return sizeof(MPI_Request);
}

int mpicclCommInitRankFromC(mpicclComm_t *mpicclComm, int nRanks, ncclUniqueId *mpicclID, int myRank) {
  return 1;
}

int mpicclCommDestroyFromC(mpicclComm_t mpicclComm){
  return 1;
}


int mpicclRedOpSumFromC(void) {
  int val = MPI_Op_c2f(MPI_SUM); // converts MPI_Op to int
  return val;
}

int mpicclRedOpProdFromC(void) {
  int val = MPI_Op_c2f(MPI_PROD);
  return val;
}

int mpicclRedOpMinFromC(void) {
  int val = MPI_Op_c2f(MPI_MIN);
  return val;
}

int mpicclRedOpMaxFromC(void) {
  int val = MPI_Op_c2f(MPI_MAX);
  return val;
}

int mpicclRedOpAvgFromC(void) {
  int val = MPI_Op_c2f(MPI_OP_NULL); // no AVG in MPI
  return val;
}

int mpicclDataTypeMpicclIntFromC(void) {
  int val = MPI_Type_c2f(MPI_INT);
  return val;
}

int mpicclDataTypeMpicclInt32FromC(void) {
  int val = MPI_Type_c2f(MPI_INT32_T);
  return val;
}

int mpicclDataTypeMpicclInt64FromC(void) {
  int val = MPI_Type_c2f(MPI_INT64_T);
  return val;
}

int mpicclDataTypeMpicclFloat32FromC(void) {
  int val = MPI_Type_c2f(MPI_FLOAT);
  return val;
}

int mpicclDataTypeMpicclFloatFromC(void) {
  int val = MPI_Type_c2f(MPI_FLOAT);
  return val;
}

// For NCCL: ncclFloat64=ncclDouble
int mpicclDataTypeMpicclFloat64FromC(void) {
  int val = MPI_Type_c2f(MPI_DOUBLE);
  return val;
}

int mpicclDataTypeMpicclDoubleFromC(void) {
  int val = MPI_Type_c2f(MPI_DOUBLE);
  return val;
}

int mpicclAllReduceFromC(const void *sendbuff, void *recvbuff, size_t count, int mpicclDatatype, int mpicclOp, intptr_t mpicclComm, gpuStream_t gpuStream) {
  int mpicclError;
  void *sendbuff_safe; // to handle the case when sendbuff and recvbuff are the same (MPI_IN_PLACE)

#ifdef WITH_GPU_STREAMS
  gpuStreamSynchronize(gpuStream);
#endif

  sendbuff_safe = (sendbuff == recvbuff) ? MPI_IN_PLACE : (void *)sendbuff;
  mpicclError = MPI_Allreduce(sendbuff_safe, recvbuff, count, MPI_Type_f2c(mpicclDatatype), MPI_Op_f2c(mpicclOp), MPI_Comm_f2c(mpicclComm));
  if (mpicclError != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(mpicclError, errstr, &len);
    errormessage("Error in mpicclAllReduce: %s\n", errstr);
    return 0;
  }

  return 1;
}

int mpicclReduceFromC(const void *sendbuff, void *recvbuff, size_t count, int mpicclDatatype, int mpicclOp, int root, intptr_t mpicclComm, gpuStream_t gpuStream) {
  int mpicclError;
  int rank;
  MPI_Comm comm;
  void *sendbuff_safe;

#ifdef WITH_GPU_STREAMS
  gpuStreamSynchronize(gpuStream);
#endif

  comm = MPI_Comm_f2c(mpicclComm);
  sendbuff_safe = (void *)sendbuff;
  if (sendbuff == recvbuff) {
    mpicclError = MPI_Comm_rank(comm, &rank);
    if (mpicclError != MPI_SUCCESS) {
      char errstr[MPI_MAX_ERROR_STRING];
      int len = 0;
      MPI_Error_string(mpicclError, errstr, &len);
      errormessage("Error in mpicclReduce (MPI_Comm_rank): %s\n", errstr);
      return 0;
    }
    if (rank == root) {
      sendbuff_safe = MPI_IN_PLACE;
    }
  }

  mpicclError = MPI_Reduce(sendbuff_safe, recvbuff, count, MPI_Type_f2c(mpicclDatatype), MPI_Op_f2c(mpicclOp), root, comm);
  if (mpicclError != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(mpicclError, errstr, &len);
    errormessage("Error in mpicclReduce: %s\n", errstr);
    return 0;
  }

  return 1;
}

int mpicclBroadcastFromC(const void* sendbuff, void* recvbuff, size_t count, int mpicclDatatype, int root, intptr_t mpicclComm, gpuStream_t gpuStream) {
  int mpicclError;
  gpuError_t gpuerr = gpuSuccess;
  
  int rank;
  MPI_Comm_rank(MPI_Comm_f2c(mpicclComm), &rank);

  // Treat special case when recvbuff != sendbuff:
  // not directly supported by MPI_Bcast, so we need to copy sendbuff to recvbuff on root
  if (rank==root && recvbuff != sendbuff){
    int typesize;
    MPI_Datatype dtype = MPI_Type_f2c(mpicclDatatype);
    mpicclError = MPI_Type_size(dtype, &typesize);
    if (mpicclError != MPI_SUCCESS) {
      char errstr[MPI_MAX_ERROR_STRING];
      int len = 0;
      MPI_Error_string(mpicclError, errstr, &len);
      errormessage("Error in mpicclBcastFromC (MPI_Type_size): %s\n", errstr);
      return 0;
    }

    size_t nbytes = (size_t)typesize * count;

#ifdef WITH_SYCL_GPU_VERSION
    sycl::queue queue = getQueueOrDefault(gpuStream);
    queue.memcpy(recvbuff, sendbuff, nbytes);
#else
#ifdef WITH_GPU_STREAMS
    gpuerr = gpuMemcpyAsync(recvbuff, sendbuff, nbytes, gpuMemcpyDeviceToDevice, gpuStream);
#else
    gpuerr = gpuMemcpy     (recvbuff, sendbuff, nbytes, gpuMemcpyDeviceToDevice);
#endif
#endif
  }

// The sync is needed since MPI_Bcast doesn't carry a stream
#ifdef WITH_GPU_STREAMS
  gpuerr = gpuStreamSynchronize(gpuStream);

  if (gpuerr != gpuSuccess) {
    errormessage("Error in executing mpicclBroadcastFromC (gpuMemcpy): %s\n",gpuGetErrorString(gpuerr));
    return 0;
  }
#endif

  mpicclError = MPI_Bcast(recvbuff, (int)count, MPI_Type_f2c(mpicclDatatype), root, MPI_Comm_f2c(mpicclComm));
  if (mpicclError != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(mpicclError, errstr, &len);
    errormessage("Error in mpicclBroadcast: %s\n", errstr);
    return 0;
  }

#ifdef WITH_GPU_STREAMS
  gpuerr = gpuStreamSynchronize(gpuStream);
#endif
  return 1;
}

int mpicclSendFromC(const void* sendbuff, size_t count, int mpicclDatatype, int peer, intptr_t mpicclComm, gpuStream_t gpuStream) {
  int mpicclError;

#ifdef WITH_GPU_STREAMS
  gpuStreamSynchronize(gpuStream);
#endif

  int tag_dummy = 0;

  mpicclError = MPI_Send(sendbuff, count, MPI_Type_f2c(mpicclDatatype), peer, tag_dummy, MPI_Comm_f2c(mpicclComm));
  if (mpicclError != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(mpicclError, errstr, &len);
    errormessage("Error in mpicclSend: %s\n", errstr);
    return 0;
  }

  return 1;
}

int mpicclRecvFromC(void* recvbuff, size_t count, int mpicclDatatype, int peer, intptr_t mpicclComm, gpuStream_t gpuStream) {
  int mpicclError;

#ifdef WITH_GPU_STREAMS
  gpuStreamSynchronize(gpuStream);
#endif

  int tag_dummy = 0;

  mpicclError = MPI_Recv(recvbuff, count, MPI_Type_f2c(mpicclDatatype), peer, tag_dummy, MPI_Comm_f2c(mpicclComm), MPI_STATUS_IGNORE);
  if (mpicclError != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(mpicclError, errstr, &len);
    errormessage("Error in mpicclRecv: %s\n", errstr);
    return 0;
  }

  return 1;
}

int mpicclSendrecvFromC(const void* sendbuff, size_t sendcount, int sendMpicclDatatype, int dest, 
                        void*       recvbuff, size_t recvcount, int recvMpicclDatatype, int source,
                        intptr_t mpicclComm, gpuStream_t gpuStream) {
  int mpicclError;

#ifdef WITH_GPU_STREAMS
  gpuStreamSynchronize(gpuStream);
#endif

  int tag_dummy = 0;

  mpicclError = MPI_Sendrecv(sendbuff, sendcount, MPI_Type_f2c(sendMpicclDatatype), dest  , tag_dummy,
                             recvbuff, recvcount, MPI_Type_f2c(recvMpicclDatatype), source, tag_dummy,
                             MPI_Comm_f2c(mpicclComm), MPI_STATUS_IGNORE);
  if (mpicclError != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(mpicclError, errstr, &len);
    errormessage("Error in mpicclSendrecv: %s\n", errstr);
    return 0;
  }

  return 1;
}

int mpicclIsendFromC(const void* sendbuff, size_t count, int mpicclDatatype, int peer, intptr_t mpicclComm, intptr_t request, gpuStream_t gpuStream) {
  int mpicclError;

#ifdef WITH_GPU_STREAMS
  gpuStreamSynchronize(gpuStream);
#endif

  int tag_dummy = 0;

  mpicclError = MPI_Isend(sendbuff, count, MPI_Type_f2c(mpicclDatatype), peer, tag_dummy, MPI_Comm_f2c(mpicclComm), (MPI_Request*)request);
  if (mpicclError != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(mpicclError, errstr, &len);
    errormessage("Error in mpicclIsend: %s\n", errstr);
    return 0;
  }

  return 1;
}

int mpicclIrecvFromC(void* recvbuff, size_t count, int mpicclDatatype, int peer, intptr_t mpicclComm, intptr_t request, gpuStream_t gpuStream) {
  int mpicclError;

#ifdef WITH_GPU_STREAMS
  gpuStreamSynchronize(gpuStream);
#endif

  int tag_dummy = 0;

  mpicclError = MPI_Irecv(recvbuff, count, MPI_Type_f2c(mpicclDatatype), peer, tag_dummy, MPI_Comm_f2c(mpicclComm), (MPI_Request*)request);
  if (mpicclError != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(mpicclError, errstr, &len);
    errormessage("Error in mpicclIrecv: %s\n", errstr);
    return 0;
  }

  return 1;
}

int mpicclWaitallFromC(int count, intptr_t request_handles, gpuStream_t gpuStream) {
  int mpicclError;
  
  mpicclError = MPI_Waitall(count, (MPI_Request*)request_handles, MPI_STATUSES_IGNORE);
  if (mpicclError != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(mpicclError, errstr, &len);
    errormessage("Error in mpicclWaitall: %s\n", errstr);
    return 0;
  }

  return 1;
}


} // extern "C" 
#endif
