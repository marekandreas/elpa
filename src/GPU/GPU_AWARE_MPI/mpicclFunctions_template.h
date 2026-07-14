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

int mpicclCommInitRankFromC(mpicclComm_t *mpicclComm, int nRanks, ncclUniqueId *mpicclID, int myRank) {
  return 1;
}


int mpicclCommDestroyFromC(mpicclComm_t mpicclComm){
  return 1;
}

// PETERDEBUG111: cleanup after testing?
// https://www.mpi-forum.org/docs/mpi-5.0/mpi50-report.pdf
// see page 905 (pdf)/853 (printed doc), Sec 20.4.5,  Handle Serialization and tables there
// MPI_Op_toint from MPI 5.0

// use intptr_t here and for NCCL/RCCL/oneCCL too?
// or create or own enum table? key-value pairs?
int mpicclRedOpSumFromC(void) {
  //int val = MPI_Op_toint(MPI_SUM) // >MPI 5.0
  //MPI_Op op = MPI_SUM;
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
  errormessage("Error in mpicclRedOpAvgFromC: %s\n", "MPI does not have an AVG operation");
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

  //printf("mpicclAllreduceFromC, start\n"); // PETERDEBUG111: cleanup after testing

  sendbuff_safe = (sendbuff == recvbuff) ? MPI_IN_PLACE : (void *)sendbuff;
  mpicclError = MPI_Allreduce(sendbuff_safe, recvbuff, count, MPI_Type_f2c(mpicclDatatype), MPI_Op_f2c(mpicclOp), MPI_Comm_f2c(mpicclComm));
  if (mpicclError != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(mpicclError, errstr, &len);
    errormessage("Error in mpicclAllReduce: %s\n", errstr);
    return 0;
  }
  //gpuDeviceSynchronize(); // PETERDEBUG111: cleanup after testing

  //printf("mpicclAllreduceFromC, done\n"); // PETERDEBUG111: cleanup after testing
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
  //gpuDeviceSynchronize(); // PETERDEBUG111: cleanup after testing

  return 1;
}

int mpicclBroadcastFromC(const void* sendbuff, void* recvbuff, size_t count, int mpicclDatatype, int root, intptr_t mpicclComm, gpuStream_t gpuStream) {
  int mpicclError;

  //printf("sizeof(intptr_t)=%zu, sizeof(MPI_Comm)=%zu\n", sizeof(intptr_t), sizeof(MPI_Comm));
  gpuError_t gpuerr = gpuSuccess;
  
  int rank;
  //MPI_Comm_rank(MPI_COMM_WORLD, &rank); // PETERDEBUG111: cleanup after testing
  //MPI_Comm      MPI_COMM = (MPI_Comm)mpicclComm_;

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

    //gpuDeviceSynchronize(); // PETERDEBUG111: cleanup after testing

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
  //printf("mpicclSendFromC, start\n"); // PETERDEBUG111: cleanup after testing

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
  //gpuDeviceSynchronize(); // PETERDEBUG111: cleanup after testing

  //printf("mpicclSendFromC, done\n"); // PETERDEBUG111: cleanup after testing
  return 1;
}

int mpicclRecvFromC(void* recvbuff, size_t count, int mpicclDatatype, int peer, intptr_t mpicclComm, gpuStream_t gpuStream) {
  int mpicclError;
  //printf("mpicclRecvFromC, start\n"); // PETERDEBUG111: cleanup after testing

#ifdef WITH_GPU_STREAMS
  gpuStreamSynchronize(gpuStream);
#endif

  int tag_dummy = 0;
  MPI_Status* status;

  mpicclError = MPI_Recv(recvbuff, count, MPI_Type_f2c(mpicclDatatype), peer, tag_dummy, MPI_Comm_f2c(mpicclComm), status);
  if (mpicclError != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(mpicclError, errstr, &len);
    errormessage("Error in mpicclRecv: %s\n", errstr);
    return 0;
  }
  //gpuDeviceSynchronize(); // PETERDEBUG111: cleanup after testing
  //printf("mpicclRecvFromC, done\n"); // PETERDEBUG111: cleanup after testing
  return 1;
}

int mpicclSendrecvFromC(const void* sendbuff, size_t sendcount, int sendMpicclDatatype, int dest, 
                        void*       recvbuff, size_t recvcount, int recvMpicclDatatype, int source,
                        intptr_t mpicclComm, gpuStream_t gpuStream) {
  int mpicclError;
  //printf("mpicclSendrecvFromC, start\n"); // PETERDEBUG111: cleanup after testing

#ifdef WITH_GPU_STREAMS
  gpuStreamSynchronize(gpuStream);
#endif

  int tag_dummy = 0;
  MPI_Status* status;

  mpicclError = MPI_Sendrecv(sendbuff, sendcount, MPI_Type_f2c(sendMpicclDatatype), dest  , tag_dummy, 
                             recvbuff, recvcount, MPI_Type_f2c(recvMpicclDatatype), source, tag_dummy,
                             MPI_COMM_WORLD, status);
                             //MPI_Comm_f2c(mpicclComm), status); // PETERDEBUG111
  if (mpicclError != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(mpicclError, errstr, &len);
    errormessage("Error in mpicclSendrecv: %s\n", errstr);
    return 0;
  }

  //printf("mpicclSendrecvFromC, done\n"); // PETERDEBUG111: cleanup after testing
  return 1;
}

int mpicclIsendFromC(const void* sendbuff, size_t count, int mpicclDatatype, int peer, intptr_t mpicclComm, intptr_t request, gpuStream_t gpuStream) {
  int mpicclError;
  //printf("mpicclIsendFromC, start\n"); // PETERDEBUG111: cleanup after testing

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
  //gpuDeviceSynchronize(); // PETERDEBUG111: cleanup after testing

  //printf("mpicclIsendFromC, done\n"); // PETERDEBUG111: cleanup after testing
  return 1;
}

int mpicclIrecvFromC(void* recvbuff, size_t count, int mpicclDatatype, int peer, intptr_t mpicclComm, intptr_t request, gpuStream_t gpuStream) {
  int mpicclError;
  //printf("mpicclIrecvFromC, start\n"); // PETERDEBUG111: cleanup after testing

#ifdef WITH_GPU_STREAMS
  gpuStreamSynchronize(gpuStream);
#endif

  int tag_dummy = 0;
  MPI_Status* status;

  mpicclError = MPI_Irecv(recvbuff, count, MPI_Type_f2c(mpicclDatatype), peer, tag_dummy, MPI_Comm_f2c(mpicclComm), (MPI_Request*)request);
  if (mpicclError != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(mpicclError, errstr, &len);
    errormessage("Error in mpicclIrecv: %s\n", errstr);
    return 0;
  }
  //gpuDeviceSynchronize(); // PETERDEBUG111: cleanup after testing

  //printf("mpicclIrecvFromC, done\n"); // PETERDEBUG111: cleanup after testing
  return 1;
}

int mpicclWaitallFromC(int count, intptr_t request_handles, gpuStream_t gpuStream) {
  int mpicclError;
  //printf("mpicclWaitallFromC, start\n"); // PETERDEBUG111: cleanup after testing

  //printf("sizeof(int)=%zu, sizeof(MPI_Request)=%zu\n", sizeof(int), sizeof(MPI_Request)); // PETERDEBUG111: cleanup after testing
  //gpuDeviceSynchronize(); // PETERDEBUG111: cleanup after testing
  
  mpicclError = MPI_Waitall(count, (MPI_Request*)request_handles, MPI_STATUSES_IGNORE);
  if (mpicclError != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(mpicclError, errstr, &len);
    errormessage("Error in mpicclWaitall: %s\n", errstr);
    return 0;
  }

  //printf("mpicclWaitallFromC, done\n"); // PETERDEBUG111: cleanup after testing
  return 1;
}


} // extern "C" 
#endif
