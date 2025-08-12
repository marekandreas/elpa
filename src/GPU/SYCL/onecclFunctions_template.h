#include <cstring>
#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#ifdef DEBUG_SYCL
#define debugmessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)
#else
#define debugmessage(x, ...)
#endif

#include "syclCommon.hpp"

#ifdef WITH_ONEAPI_ONECCL

#include <oneapi/ccl.hpp>
#include <mpi.h>

using namespace sycl_be;

extern "C" {

  int onecclGroupStartFromC() {
    ccl::group_start();
    return 1;
  }

  int onecclGroupEndFromC() {
    ccl::group_end();
    return 1;
  }

  int onecclInitFromC() {
    ccl::init();
    return 1;
  }

  /**
   * Create a main Key-Value Store to create a oneCCL communicator.
   * Only call from Rank 0!
   */
  int onecclGetUniqueIdFromC(void *kvsAddress) {
    #ifdef WITH_MPI
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return 1;
      }
    #endif
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type tmpAddr;
    kvs = ccl::create_main_kvs();

    tmpAddr = kvs->get_address();
    std::memcpy(kvsAddress, tmpAddr.data(), tmpAddr.max_size());
    // So that we can retrieve the KVS later and it doesn't get deconstructed.
    SyclState::defaultState().registerKvs(kvsAddress, kvs);
    return 1;
  }


  int onecclCommInitRankFromC(ccl::communicator **onecclComm, int nRanks, void *kvsAddress, int myRank) {
    SyclState &ss = SyclState::defaultState();
    std::optional<cclKvsHandle> kvsOpt = ss.retrieveKvs(kvsAddress);
    cclKvsHandle kvs;
    if (!kvsOpt.has_value()) {
      kvs = ccl::create_kvs(*static_cast<ccl::kvs::address_type *>(kvsAddress));
      ss.registerKvs(kvsAddress, kvs);
    } else {
      kvs = kvsOpt.value();
    }
    // oneCCL doesn't return an opaque pointer to the communicator, but an interface object instead.
    *onecclComm = ss.getDefaultDeviceHandle().initCclCommunicator(nRanks, myRank, kvs);

    return 1;
  }

  int onecclCommDestroyFromC(ccl::communicator *onecclComm, QueueData *qd) {
    QueueData *qData = getQueueDataOrDefault(qd);
    return 1;
  }

  int onecclStreamSynchronizeFromC(QueueData *qd) {
    sycl::queue q = getQueueOrDefault(qd);
    q.wait();
    return 1;
  }

  int onecclRedOpSumFromC() {
    return static_cast<int>(ccl::reduction::sum);
  }

  int onecclRedOpProdFromC() {
    return static_cast<int>(ccl::reduction::prod);
  }

  int onecclRedOpMinFromC() {
    return static_cast<int>(ccl::reduction::min);
  }

  int onecclRedOpMaxFromC() {
    return static_cast<int>(ccl::reduction::max);
  }

  int onecclRedOpAvgFromC(void) {
    return static_cast<int>(ccl::reduction::avg);
  }

  int onecclDataTypeOnecclIntFromC(void) {
    // According to the NVIDIA Docs, this is supposed to be an int32.
    // There is no direct match in oneCCL, thus return the int32 enum value.
    return static_cast<int>(ccl::datatype::int32);
  }

  int onecclDataTypeOnecclInt32FromC(void) {
    return static_cast<int>(ccl::datatype::int32);
  }

  int onecclDataTypeOnecclInt64FromC(void) {
    return static_cast<int>(ccl::datatype::int64);
  }

  int onecclDataTypeOnecclFloat32FromC(void) {
    return static_cast<int>(ccl::datatype::float32);
  }

  int onecclDataTypeOnecclFloatFromC(void) {
    // Same as above, no direct match, return value according to specification in the NVIDIA docs.
    return static_cast<int>(ccl::datatype::float32);
  }

  int onecclDataTypeOnecclFloat64FromC(void) {
    return static_cast<int>(ccl::datatype::float64);
  }

  int onecclDataTypeOnecclDoubleFromC(void) {
    // Same as above, no direct match, return value according to specification in the NVIDIA docs.
    return static_cast<int>(ccl::datatype::float64);
  }

  size_t onecclSizeForDatatypeFromC(ccl::datatype onecclDatatype) {
    switch (onecclDatatype) {
      case ccl::datatype::int32:
        return sizeof(int32_t);
      case ccl::datatype::int64:
        return sizeof(int64_t);
      case ccl::datatype::float32:
        return sizeof(float);
      case ccl::datatype::float64:
        return sizeof(double);
      default:
      errormessage("%s\n", "Error in onecclSizeForDatatype: Unknown datatype.");
        return 0;
    }
  }

  int onecclAllReduceFromC(const void *sendbuff, void *recvbuff, size_t count, ccl::datatype onecclDatatype, ccl::reduction onecclOp, ccl::communicator *onecclComm, QueueData *qd) {
    QueueData *qData = getQueueDataOrDefault(qd);
    if (onecclOp == ccl::reduction::custom) {
      errormessage("%s\n", "Error in onecclAllReduce: ccl::reduction::custom is not supported in ELPA.");
      return 0;
    }

    try {
      auto attributes = ccl::create_operation_attr<ccl::allreduce_attr>();
      ccl::allreduce(sendbuff, recvbuff, count, onecclDatatype, onecclOp, *onecclComm, qData->cclStream, attributes).wait();
    } catch (const ccl::exception &e) {
      errormessage("Error in onecclAllReduce: %s\n", e.what());
      return 0;
    }
    return 1;
  }

  int onecclReduceFromC(const void *sendbuff, void *recvbuff, size_t count, ccl::datatype onecclDatatype, ccl::reduction onecclOp, int root, ccl::communicator *onecclComm, QueueData *qd) {
    QueueData *qData = getQueueDataOrDefault(qd);
    if (onecclOp == ccl::reduction::custom) {
      errormessage("%s\n", "Error in onecclReduce: ccl::reduction::custom is not supported in ELPA. (Likely you wanted avg, which oneCCL doesn't have)");
      return 0;
    }

    try {
      auto attributes = ccl::create_operation_attr<ccl::reduce_attr>();
      ccl::reduce(sendbuff, recvbuff, count, onecclDatatype, onecclOp, root, *onecclComm, qData->cclStream, attributes).wait();
    } catch (const ccl::exception &e) {
      errormessage("Error in onecclReduce: %s\n", e.what());
      return 0;
    }
    return 1;
  }

  int onecclBroadcastFromC(const void* sendbuff, void* recvbuff, size_t count, ccl::datatype onecclDatatype, int root, ccl::communicator *onecclComm, QueueData *qd) {
    QueueData *qData = getQueueDataOrDefault(qd);
    try {
      std::vector<ccl::event> deps;
      if (sendbuff != recvbuff && root == onecclComm->rank()) {
        auto q = qData->queue;
        auto e = q.memcpy(recvbuff, sendbuff, count * onecclSizeForDatatypeFromC(onecclDatatype));
        deps.push_back(ccl::create_event(e));
      }
      auto attr = ccl::create_operation_attr<ccl::broadcast_attr>();
      ccl::broadcast(recvbuff, count, onecclDatatype, root, *onecclComm, attr, deps).wait();
    } catch (const ccl::exception &e) {
      errormessage("Error in onecclBroadcast: %s\n", e.what());
      return 0;
    }
    return 1;
  }

  int onecclSendFromC(void* sendbuff, size_t count, ccl::datatype onecclDatatype, int peer, ccl::communicator *onecclComm, QueueData *qd) {
    try {
      QueueData *qData = getQueueDataOrDefault(qd);
      ccl::stream &stream = qData->cclStream;
      ccl::send(sendbuff, count, onecclDatatype, peer, *onecclComm, stream).wait();
    } catch (const ccl::exception &e) {
      errormessage("Error in onecclSend: %s\n", e.what());
      return 0;
    }
    return 1;
  }

  int onecclRecvFromC(void* recvbuff, size_t count, ccl::datatype onecclDatatype, int peer, ccl::communicator *onecclComm, QueueData *qd) {
    try {
      QueueData *qData = getQueueDataOrDefault(qd);
      ccl::recv(recvbuff, count, onecclDatatype, peer, *onecclComm, qData->cclStream).wait();
    } catch (const ccl::exception &e) {
      errormessage("Error in onecclRecv: %s\n", e.what());
      return 0;
    }
    return 1;
  }

}
#endif
