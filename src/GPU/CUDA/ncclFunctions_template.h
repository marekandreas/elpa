#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#ifdef DEBUG_CUDA
#define debugmessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)
#else
#define debugmessage(x, ...)
#endif
  
ncclUniqueId globalIDFixThis;

#ifdef WITH_NVIDIA_NCCL
extern "C" {
  int ncclGroupStartFromC() {
    ncclResult_t ncclError;

    ncclError = ncclGroupStart();
    if (ncclError != ncclSuccess) {
      if (ncclError == ncclUnhandledCudaError) {
        errormessage("Error in ncclGroupStart: %s\n", "ncclUnhandledCudaError");
      } else if (ncclError == ncclSystemError) {
        errormessage("Error in ncclGroupStart: %s\n", "ncclSystemError");
      } else if (ncclError == ncclInternalError) {
        errormessage("Error in ncclGroupStart: %s\n", "ncclInternalError");
      } else if (ncclError == ncclInvalidArgument) {
        errormessage("Error in ncclGroupStart: %s\n", "ncclInvalidArguments");
      } else if (ncclError == ncclInvalidUsage) {
        errormessage("Error in ncclGroupStart: %s\n", "ncclInvalidUsage");
      } else if (ncclNumResults) {
        errormessage("Error in ncclGroupStart: %s\n", "ncclNumResults");
      } else {
        errormessage("Error in ncclGroupStart: %s\n", "unknown error");
      }
      return 0;
    }
    return 1;
  }

  int ncclGroupEndFromC() {
    ncclResult_t ncclError;

    ncclError = ncclGroupEnd();
    if (ncclError != ncclSuccess) {
      if (ncclError == ncclUnhandledCudaError) {
        errormessage("Error in ncclGroupEnd: %s\n", "ncclUnhandledCudaError");
      } else if (ncclError == ncclSystemError) {
        errormessage("Error in ncclGroupEnd: %s\n", "ncclSystemError");
      } else if (ncclError == ncclInternalError) {
        errormessage("Error in ncclGroupEnd: %s\n", "ncclInternalError");
      } else if (ncclError == ncclInvalidArgument) {
        errormessage("Error in ncclGroupEnd: %s\n", "ncclInvalidArguments");
      } else if (ncclError == ncclInvalidUsage) {
        errormessage("Error in ncclGroupEnd: %s\n", "ncclInvalidUsage");
      } else if (ncclNumResults) {
        errormessage("Error in ncclGroupEnd: %s\n", "ncclNumResults");
      } else {
        errormessage("Error in ncclGroupEnd: %s\n", "unknown error");
      }
      return 0;
    }
    return 1;
  }


  int ncclGetUniqueIdFromC(ncclUniqueId *ncclID) {
    ncclResult_t ncclError;
    ncclUniqueId id_dummy;
    ncclError = ncclGetUniqueId(&id_dummy);
    for (int i=0; i<sizeof(ncclUniqueId);i++) {
       ncclID->internal[i] =  id_dummy.internal[i];
       //DEBUG
       //printf("a %c \n",ncclID->internal[i]);
    }
    
    if (ncclError != ncclSuccess) {
      if (ncclError == ncclUnhandledCudaError) {
        errormessage("Error in ncclGetUniqueId: %s\n", "ncclUnhandledCudaError");
      } else if (ncclError == ncclSystemError) {
        errormessage("Error in ncclGetUniqueId: %s\n", "ncclSystemError");
      } else if (ncclError == ncclInternalError) {
        errormessage("Error in ncclGetUniqueId: %s\n", "ncclInternalError");
      } else if (ncclError == ncclInvalidArgument) {
        errormessage("Error in ncclGetUniqueId: %s\n", "ncclInvalidArguments");
      } else if (ncclError == ncclInvalidUsage) {
        errormessage("Error in ncclGetUniqueId: %s\n", "ncclInvalidUsage");
      } else if (ncclNumResults) {
        errormessage("Error in ncclGetUniqueId: %s\n", "ncclNumResults");
      } else {
        errormessage("Error in ncclGetUniqueId: %s\n", "unknown error");
      }
      return 0;

    }
    return 1;
  }

  int ncclCommInitRankFromC(ncclComm_t *ncclComm, int nRanks, ncclUniqueId *ncclID, int myRank) {
    ncclResult_t ncclError;

    ncclUniqueId id_dummy;
    for (int i=0; i<sizeof(ncclUniqueId);i++) {
       // debug
       //printf("j %c \n",ncclID->internal[i]);
       id_dummy.internal[i] = ncclID->internal[i];
    }
    if (sizeof(ncclUniqueId) != 16*sizeof(intptr_t)) {
     printf("sizes of ncclUniqueId changed \n");
     return 0;
    }

    ncclError = ncclCommInitRank(ncclComm, nRanks, id_dummy, myRank);
    if (ncclError != ncclSuccess) {
      if (ncclError == ncclUnhandledCudaError) {
        errormessage("Error in ncclCommInitRank: %s\n", "ncclUnhandledCudaError");
      } else if (ncclError == ncclSystemError) {
        errormessage("Error in ncclCommInitRank: %s\n", "ncclSystemError");
      } else if (ncclError == ncclInternalError) {
        errormessage("Error in ncclCommInitRank: %s\n", "ncclInternalError");
      } else if (ncclError == ncclInvalidArgument) {
        errormessage("Error in ncclCommInitRank: %s\n", "ncclInvalidArguments");
      } else if (ncclError == ncclInvalidUsage) {
        errormessage("Error in ncclCommInitRank: %s\n", "ncclInvalidUsage");
      } else if (ncclNumResults) {
        errormessage("Error in ncclCommInitRank: %s\n", "ncclNumResults");
      } else {
        errormessage("Error in ncclCommInitRank: %s\n", "unknown error");
      }
      return 0;
    }
    return 1;
  }

  // only for version >= 2.13
  //int ncclCommFinalizeFromC(ncclComm_t ncclComm) {
  //  ncclResult_t ncclError;

  //  ncclError = ncclCommFinalize(ncclComm);
  //  if (ncclError != ncclSuccess) {
  //    errormessage("Error in ncclCommFinalize: %s\n", "unknown error");
  //    return 0;

  //  }
  //  return 1;
  //}

  int ncclCommDestroyFromC(ncclComm_t ncclComm) {
    ncclResult_t ncclError;

    //signature: ncclResult_t ncclCommDestroy(ncclComm_t comm)
    ncclError = ncclCommDestroy(ncclComm);
    if (ncclError != ncclSuccess) {
      if (ncclError == ncclUnhandledCudaError) {
        errormessage("Error in ncclCommDestroy: %s\n", "ncclUnhandledCudaError");
      } else if (ncclError == ncclSystemError) {
        errormessage("Error in ncclCommDestroy: %s\n", "ncclSystemError");
      } else if (ncclError == ncclInternalError) {
        errormessage("Error in ncclCommDestroy: %s\n", "ncclInternalError");
      } else if (ncclError == ncclInvalidArgument) {
        errormessage("Error in ncclCommDestroy: %s\n", "ncclInvalidArguments");
      } else if (ncclError == ncclInvalidUsage) {
        errormessage("Error in ncclCommDestroy: %s\n", "ncclInvalidUsage");
      } else if (ncclNumResults) {
        errormessage("Error in ncclCommDestroy: %s\n", "ncclNumResults");
      } else {
        errormessage("Error in ncclCommDestroy: %s\n", "unknown error");
      }
      return 0;
    }
    return 1;
  }

  int ncclRedOpSumFromC(void) {
    int val = ncclSum;
    return val;
  }

  int ncclRedOpProdFromC(void) {
    int val = ncclProd;
    return val;
  }

  int ncclRedOpMinFromC(void) {
    int val = ncclMin;
    return val;
  }

  int ncclRedOpMaxFromC(void) {
    int val = ncclMax;
    return val;
  }

  int ncclRedOpAvgFromC(void) {
    int val = ncclAvg;
    return val;
  }

  int ncclDataTypeNcclIntFromC(void) {
    int val = ncclInt;
    return val;
  }

  int ncclDataTypeNcclInt32FromC(void) {
    int val = ncclInt32;
    return val;
  }

  int ncclDataTypeNcclInt64FromC(void) {
    int val = ncclInt64;
    return val;
  }

  int ncclDataTypeNcclFloat32FromC(void) {
    int val = ncclFloat32;
    return val;
  }

  int ncclDataTypeNcclFloatFromC(void) {
    int val = ncclFloat;
    return val;
  }

  int ncclDataTypeNcclFloat64FromC(void) {
    int val = ncclFloat64;
    return val;
  }

  int ncclDataTypeNcclDoubleFromC(void) {
    int val = ncclDouble;
    return val;
  }

  int ncclAllReduceFromC(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t ncclDatatype, ncclRedOp_t ncclOp, ncclComm_t ncclComm, cudaStream_t cudaStream) {
    ncclResult_t ncclError;

    ncclError = ncclAllReduce(sendbuff, recvbuff, count, ncclDatatype, ncclOp, ncclComm, cudaStream);
    if (ncclError != ncclSuccess) {
      if (ncclError == ncclUnhandledCudaError) {
        errormessage("Error in ncclAllReduce: %s\n", "ncclUnhandledCudaError");
      } else if (ncclError == ncclSystemError) {
        errormessage("Error in ncclAllReduce: %s\n", "ncclSystemError");
      } else if (ncclError == ncclInternalError) {
        errormessage("Error in ncclAllReduce: %s\n", "ncclInternalError");
      } else if (ncclError == ncclInvalidArgument) {
        errormessage("Error in ncclAllReduce: %s\n", "ncclInvalidArguments");
      } else if (ncclError == ncclInvalidUsage) {
        errormessage("Error in ncclAllReduce: %s\n", "ncclInvalidUsage");
      } else if (ncclNumResults) {
        errormessage("Error in ncclAllReduce: %s\n", "ncclNumResults");
      } else {
        errormessage("Error in ncclAllReduce: %s\n", "unknown error");
      }
      return 0;
    }
    return 1;
  }

  int ncclReduceFromC(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t ncclDatatype, ncclRedOp_t ncclOp, int root, ncclComm_t ncclComm, cudaStream_t cudaStream) {
    ncclResult_t ncclError;

    ncclError = ncclReduce(sendbuff, recvbuff, count, ncclDatatype, ncclOp, root, ncclComm, cudaStream);
    if (ncclError != ncclSuccess) {
      if (ncclError == ncclUnhandledCudaError) {
        errormessage("Error in ncclReduce: %s\n", "ncclUnhandledCudaError");
      } else if (ncclError == ncclSystemError) {
        errormessage("Error in ncclReduce: %s\n", "ncclSystemError");
      } else if (ncclError == ncclInternalError) {
        errormessage("Error in ncclReduce: %s\n", "ncclInternalError");
      } else if (ncclError == ncclInvalidArgument) {
        errormessage("Error in ncclReduce: %s\n", "ncclInvalidArguments");
      } else if (ncclError == ncclInvalidUsage) {
        errormessage("Error in ncclReduce: %s\n", "ncclInvalidUsage");
      } else if (ncclNumResults) {
        errormessage("Error in ncclReduce: %s\n", "ncclNumResults");
      } else {
        errormessage("Error in ncclReduce: %s\n", "unknown error");
      }
      return 0;
    }
    return 1;
  }

  int ncclBroadcastFromC(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t ncclDatatype, int root, ncclComm_t ncclComm, cudaStream_t cudaStream) {
    ncclResult_t ncclError;

    ncclError = ncclBroadcast(sendbuff, recvbuff, count, ncclDatatype, root, ncclComm, cudaStream);

    if (ncclError != ncclSuccess) {
      if (ncclError == ncclUnhandledCudaError) {
        errormessage("Error in ncclBroadcast: %s\n", "ncclUnhandledCudaError");
      } else if (ncclError == ncclSystemError) {
        errormessage("Error in ncclBroadcast: %s\n", "ncclSystemError");
      } else if (ncclError == ncclInternalError) {
        errormessage("Error in ncclBroadcast: %s\n", "ncclInternalError");
      } else if (ncclError == ncclInvalidArgument) {
        errormessage("Error in ncclBroadcast: %s\n", "ncclInvalidArguments");
      } else if (ncclError == ncclInvalidUsage) {
        errormessage("Error in ncclBroadcast: %s\n", "ncclInvalidUsage");
      } else if (ncclNumResults) {
        errormessage("Error in ncclBroadcast: %s\n", "ncclNumResults");
      } else {
        errormessage("Error in ncclBroadcast: %s\n", "unknown error");
      }
      return 0;
    }
    return 1;
  }

  int ncclSendFromC(const void* sendbuff, size_t count, ncclDataType_t ncclDatatype, int peer, ncclComm_t ncclComm, cudaStream_t cudaStream) {
    ncclResult_t ncclError;

    ncclError = ncclSend(sendbuff, count, ncclDatatype, peer, ncclComm, cudaStream);
    if (ncclError != ncclSuccess) {
      if (ncclError == ncclUnhandledCudaError) {
        errormessage("Error in ncclSend: %s\n", "ncclUnhandledCudaError");
      } else if (ncclError == ncclSystemError) {
        errormessage("Error in ncclSend: %s\n", "ncclSystemError");
      } else if (ncclError == ncclInternalError) {
        errormessage("Error in ncclSend: %s\n", "ncclInternalError");
      } else if (ncclError == ncclInvalidArgument) {
        errormessage("Error in ncclSend: %s\n", "ncclInvalidArguments");
      } else if (ncclError == ncclInvalidUsage) {
        errormessage("Error in ncclSend: %s\n", "ncclInvalidUsage");
      } else if (ncclNumResults) {
        errormessage("Error in ncclSend: %s\n", "ncclNumResults");
      } else {
        errormessage("Error in ncclSend: %s\n", "unknown error");
      }
      return 0;
    }
    return 1;
  }

  int ncclRecvFromC(void* recvbuff, size_t count, ncclDataType_t ncclDatatype, int peer, ncclComm_t ncclComm, cudaStream_t cudaStream) {
    ncclResult_t ncclError;

    ncclError = ncclRecv(recvbuff, count, ncclDatatype, peer, ncclComm, cudaStream);
    if (ncclError != ncclSuccess) {
      if (ncclError == ncclUnhandledCudaError) {
        errormessage("Error in ncclRecv: %s\n", "ncclUnhandledCudaError");
      } else if (ncclError == ncclSystemError) {
        errormessage("Error in ncclRecv: %s\n", "ncclSystemError");
      } else if (ncclError == ncclInternalError) {
        errormessage("Error in ncclRecv: %s\n", "ncclInternalError");
      } else if (ncclError == ncclInvalidArgument) {
        errormessage("Error in ncclRecv: %s\n", "ncclInvalidArguments");
      } else if (ncclError == ncclInvalidUsage) {
        errormessage("Error in ncclRecv: %s\n", "ncclInvalidUsage");
      } else if (ncclNumResults) {
        errormessage("Error in ncclRecv: %s\n", "ncclNumResults");
      } else {
        errormessage("Error in ncclRecv: %s\n", "unknown error");
      }
      return 0;
    }
    return 1;
  }


}
#endif
