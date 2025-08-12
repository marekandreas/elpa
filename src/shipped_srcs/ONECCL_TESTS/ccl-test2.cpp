#include <cstdint>
#include <cstdlib>
#include <memory>


#include <sycl/sycl.hpp>
#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <type_traits>

#include "ccl/types.hpp"
#include "elpa-oneccl-wrapper.h"
#include "syclCommon.hpp"

void testOnecclDirectly();

void mpi_finalize() {
    int is_finalized = 0;
    MPI_Finalized(&is_finalized);

    if (!is_finalized) {
	MPI_Finalize();
    }
}


ccl::communicator* testOnecclInitializationWithElpaWrappers(char *addr_space) {
    int myrank, totalRanks, numDevices;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);
    elpa::gpu::sycl::collectGpuDevices(false);
    numDevices = elpa::gpu::sycl::getNumDevices();
    elpa::gpu::sycl::selectGpuDevice(myrank % numDevices);
    sycl::queue q = elpa::gpu::sycl::getQueue();
    
    if (!myrank) {
	onecclGetUniqueIdFromC(addr_space);
        MPI_Bcast(addr_space, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(addr_space, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    ccl::communicator *comm;
    onecclCommInitRankFromC(&comm, totalRanks, addr_space, myrank);
    return comm;
}

template<typename T> struct always_false : std::false_type {};

template<typename DT> bool verifyAllreduceResult(DT result, int totalRanks, std::string reductionName) {
    using namespace std::string_literals;
    DT expectedResult;
    if (reductionName == "sum"s) {
	expectedResult = static_cast<DT>(0);
	for (int i = 0; i < totalRanks; i++) {
	    expectedResult += static_cast<DT>(i + 1) * static_cast<DT>(1000);
	}
    } else if (reductionName == "prod"s) {
	expectedResult = static_cast<DT>(1);
	for (int i = 0; i < totalRanks; i++) {
	    expectedResult *= static_cast<DT>(i + 1) * static_cast<DT>(1000);
	}
    } else if (reductionName == "min"s) {
	expectedResult = static_cast<DT>(1000);
    } else if (reductionName == "max"s) {
	expectedResult = static_cast<DT>(1000 * totalRanks);
    } else {
	expectedResult = 0;
	return false;
    }
    return result == expectedResult;
}

template<typename DT> std::pair<int(*)(), std::string> getCclTypes() {
    using namespace std::string_literals;
    int (*cclDatatype)();
    std::string cclDatatypeStr;
    if constexpr (std::is_same_v<float, DT>) {
	cclDatatype = onecclDataTypeOnecclFloatFromC;
	cclDatatypeStr = "float"s;
    } else if constexpr (std::is_same_v<double, DT>) {
	cclDatatype = onecclDataTypeOnecclDoubleFromC;
	cclDatatypeStr = "double"s;
    } else if constexpr (std::is_same_v<int, DT>) {
	cclDatatype = onecclDataTypeOnecclIntFromC;
	cclDatatypeStr = "int"s;
    } else if constexpr (std::is_same_v<int64_t, DT>) {
	cclDatatype = onecclDataTypeOnecclInt64FromC;
	cclDatatypeStr = "int64_t"s;
    } else {
	static_assert(always_false<DT>::value, "Unsupported data type");
    }
    return std::make_pair(cclDatatype, cclDatatypeStr);
}

template<typename DT> void testSendRecvWithElpaWrappers(ccl::communicator * comm) {
    int myrank, totalRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);

    sycl::queue q = elpa::gpu::sycl::getQueue();
    auto freeLambda = [=](void *ptr) { sycl::free(ptr, q); };
    auto sendBuffer = std::unique_ptr<DT[], decltype(freeLambda)>(sycl::malloc_shared<DT>(1, q), freeLambda);
    auto recvBuffer = std::unique_ptr<DT[], decltype(freeLambda)>(sycl::malloc_shared<DT>(1, q), freeLambda);

    q.parallel_for(1, [sb = sendBuffer.get(), rb = recvBuffer.get(), myrank](sycl::id<1>) {
	sb[0] = (myrank + 1) * 1000.0f;
	rb[0] = 0.0f;
    }).wait_and_throw();

    int sendPartner = (myrank + totalRanks + 1) % totalRanks;
    int recvPartner = (myrank + totalRanks - 1) % totalRanks;

    auto [cclDatatype, cclDatatypeStr] = getCclTypes<DT>();

    if (!myrank) std::cout << "== Test Send/Recv with " << cclDatatypeStr << " ==" << std::endl;
    onecclSendFromC(sendBuffer.get(), 1, static_cast<ccl::datatype>(cclDatatype()), sendPartner, comm, elpa::gpu::sycl::getCclStream());
    onecclRecvFromC(recvBuffer.get(), 1, static_cast<ccl::datatype>(cclDatatype()), recvPartner, comm, elpa::gpu::sycl::getCclStream());
    q.wait();

    int expectedResult = (recvPartner + 1)  * 1000;
    bool correct = recvBuffer[0] == static_cast<DT>(expectedResult);
    std::cout << "  Result: " << recvBuffer[0] << " is " << (correct ? "\033[1;32mCORRECT" : "\033[1;31mINCORRECT") << "\033[0m" << std::endl;
}

template<typename DT> void testReduceWithElpaWrappers(ccl::communicator *comm) {
    using namespace std::string_literals;
    int myrank, totalRanks, numDevices;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);


    sycl::queue q = elpa::gpu::sycl::getQueue();
    auto freeLambda = [=](void *ptr) { sycl::free(ptr, q); };
    auto sendBuffer = std::unique_ptr<DT[], decltype(freeLambda)>(sycl::malloc_shared<DT>(1, q), freeLambda);
    auto recvBuffer = std::unique_ptr<DT[], decltype(freeLambda)>(sycl::malloc_shared<DT>(1, q), freeLambda);

    q.parallel_for(1, [sb = sendBuffer.get(), rb = recvBuffer.get(), myrank](sycl::id<1> idx) {
	sb[0] = (myrank + 1) * 1000.0f;
	rb[0] = 0.0f;
    }).wait_and_throw();

    auto [cclDatatype, cclDatatypeStr] = getCclTypes<DT>();

    std::vector<std::pair<int(*)(), std::string>> reductions = {
	std::pair{&onecclRedOpSumFromC, "sum"s},
	std::pair{&onecclRedOpProdFromC, "prod"s},
	std::pair{&onecclRedOpMinFromC, "min"s},
	std::pair{&onecclRedOpMaxFromC, "max"s},
    };

    if (!myrank) std::cout << "== Test Reduce with " << cclDatatypeStr << " ==" << std::endl;
    for (auto [reduction, reductionName] : reductions) {
	if (myrank == 0) {
	    std::cout << " - Test " << reductionName << " with " << cclDatatypeStr << ": ";
	}
	onecclReduceFromC(
	    sendBuffer.get(),
	    recvBuffer.get(),
	    1,
	    static_cast<ccl::datatype>(cclDatatype()),
	    static_cast<ccl::reduction>(reduction()),
	    0,
	    comm, 
	    elpa::gpu::sycl::getCclStream()
	);
	if (myrank == 0) {
	    bool correct = verifyAllreduceResult<DT>(recvBuffer[0], totalRanks, reductionName);
	    std::cout << "  Result: " << recvBuffer[0] << " is " << (correct ? "\033[1;32mCORRECT" : "\033[1;31mINCORRECT") << "\033[0m" << std::endl;
	}
	q.wait();
    }
}

template<typename DT> void testBroadcastWithElpaWrappers(ccl::communicator *comm) {
    using namespace std::string_literals;
    
    int myrank, totalRanks, numDevices;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);


    sycl::queue q = elpa::gpu::sycl::getQueue();
    auto freeLambda = [=](void *ptr) { sycl::free(ptr, q); };
    auto sendBuffer = std::unique_ptr<DT[], decltype(freeLambda)>(sycl::malloc_shared<DT>(1, q), freeLambda);
    auto recvBuffer = std::unique_ptr<DT[], decltype(freeLambda)>(sycl::malloc_shared<DT>(1, q), freeLambda);

    q.parallel_for(1, [sb = sendBuffer.get(), rb = recvBuffer.get(), myrank](sycl::id<1> idx) {
	sb[0] = (myrank + 1) * 1000.0f;
	rb[0] = 0.0f;
    }).wait_and_throw();

    auto [cclDatatype, cclDatatypeStr] = getCclTypes<DT>();

    if (myrank == 0) std::cout << "== Test Broadcast with " << cclDatatypeStr << " ==" << std::endl;

    for (int i = 0; i < totalRanks; i++) {
	if (!myrank) std::cout << " - Test Broadcast from rank "<< i << " with " << cclDatatypeStr << ": ";
	onecclBroadcastFromC(
	    sendBuffer.get(),
	    recvBuffer.get(),
	    1,
	    static_cast<ccl::datatype>(cclDatatype()),
	    i,
	    comm, 
	    elpa::gpu::sycl::getCclStream()
	);
	if (myrank == 0) {
	    bool correct = recvBuffer[0] == static_cast<DT>(1000 * (i+1));
	    std::cout << "  Result: " << recvBuffer[0] << " is " << (correct ? "\033[1;32mCORRECT" : "\033[1;31mINCORRECT") << "\033[0m" << std::endl;
	}
	q.wait();
    }
}
template<typename DT> void testAllReduceWithElpaWrappers(ccl::communicator *comm) {
    using namespace std::string_literals;
    
    int myrank, totalRanks, numDevices;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);


    sycl::queue q = elpa::gpu::sycl::getQueue();
    auto freeLambda = [=](void *ptr) { sycl::free(ptr, q); };
    auto sendBuffer = std::unique_ptr<DT[], decltype(freeLambda)>(sycl::malloc_shared<DT>(1, q), freeLambda);
    auto recvBuffer = std::unique_ptr<DT[], decltype(freeLambda)>(sycl::malloc_shared<DT>(1, q), freeLambda);

    q.parallel_for(1, [sb = sendBuffer.get(), rb = recvBuffer.get(), myrank](sycl::id<1> idx) {
	sb[0] = (myrank + 1) * 1000.0f;
	rb[0] = 0.0f;
    }).wait_and_throw();

    auto [cclDatatype, cclDatatypeStr] = getCclTypes<DT>();

    std::vector<std::pair<int(*)(), std::string>> reductions = {
	std::pair{&onecclRedOpSumFromC, "sum"s},
	std::pair{&onecclRedOpProdFromC, "prod"s},
	std::pair{&onecclRedOpMinFromC, "min"s},
	std::pair{&onecclRedOpMaxFromC, "max"s},
    };

    if (!myrank) std::cout << "== Test Allreduce with " << cclDatatypeStr << " ==" << std::endl;
    for (auto [reduction, reductionName] : reductions) {
	if (myrank == 0) {
	    std::cout << " - Test " << reductionName << " with " << cclDatatypeStr << ": ";
	}
	onecclAllReduceFromC(
	    sendBuffer.get(),
	    recvBuffer.get(),
	    1,
	    static_cast<ccl::datatype>(cclDatatype()),
	    static_cast<ccl::reduction>(reduction()),
	    comm,
	    elpa::gpu::sycl::getCclStream()
	);
	if (myrank == 0) {
	    bool correct = verifyAllreduceResult<DT>(recvBuffer[0], totalRanks, reductionName);
	    std::cout << "  Result: " << recvBuffer[0] << " is " << (correct ? "\033[1;32mCORRECT" : "\033[1;31mINCORRECT") << "\033[0m" << std::endl;
	}
	q.wait();
    }
}

void testOnecclDirectly() {
    using namespace std::string_literals;
    atexit(mpi_finalize);
    ccl::init();
    MPI_Init(nullptr, nullptr);

    int myrank, totalRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);
    
    ccl::kvs::address_type addr;
    std::unordered_map<void *, ccl::shared_ptr_class<ccl::kvs>> kvsMap;

    if (myrank == 0) {
	auto tmpKvs = ccl::create_main_kvs();
	addr = tmpKvs->get_address();
	kvsMap.insert({&addr, tmpKvs});
	std::cout << addr.max_size() << std::endl;
	MPI_Bcast(addr.data(), addr.max_size(), MPI_CHAR, 0, MPI_COMM_WORLD);
    } else {
	MPI_Bcast(addr.data(), addr.max_size(), MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    sycl::queue q;
    ccl::device cclDevice = ccl::create_device(q.get_device());
    ccl::context cclContext = ccl::create_context(q.get_context());
    
    ccl::shared_ptr_class<ccl::kvs> kvs;
    if (kvsMap.find(&addr) != kvsMap.end()) {
	kvs = kvsMap[&addr];
    } else {
        kvs = ccl::create_kvs(addr);
	kvsMap.insert({&addr, kvs});
    }

    ccl::communicator *cclComm = new ccl::communicator(std::move(ccl::create_communicator(totalRanks, myrank, cclDevice, cclContext, kvs)));
    std::cout << "Rank " << myrank << ": " << addr.size() << " ~ " << std::endl;

    ccl::stream stream = ccl::create_stream(q);

    auto freeLambda = [=](void *ptr) { sycl::free(ptr, q); };
    auto sendBuffer = std::unique_ptr<float[], decltype(freeLambda)>(sycl::malloc_shared<float>(1, q), freeLambda);
    auto recvBuffer = std::unique_ptr<float[], decltype(freeLambda)>(sycl::malloc_shared<float>(1, q), freeLambda);

    q.parallel_for(1, [sb = sendBuffer.get(), rb = recvBuffer.get(), myrank](sycl::id<1> idx) {
	sb[0] = (myrank + 1) * 1000.0f;
	rb[0] = 0.0f;
    }).wait_and_throw();


    std::vector<std::pair<ccl::reduction, std::string>> reductions = {
	std::pair{ccl::reduction::sum, "sum"s},
	std::pair{ccl::reduction::prod, "prod"s},
	std::pair{ccl::reduction::min, "min"s},
	std::pair{ccl::reduction::max, "max"s},
    };

    for (auto [reduction, reductionName] : reductions) {
	auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();
	ccl::allreduce(sendBuffer.get(), recvBuffer.get(), 1, reduction, *cclComm, stream, attr).wait();
	
	if (myrank == 0) {
	    std::cout << "Rank " << myrank << " got result for " << reductionName << ": " << recvBuffer[0] << std::endl;
	}
    }
}

int main() {
    // testOnecclDirectly();
    atexit(mpi_finalize);
    ccl::init();
    MPI_Init(nullptr, nullptr);

    std::unique_ptr<char []> addr_space = std::make_unique<char[]>(256);
    ccl::communicator *comm = testOnecclInitializationWithElpaWrappers(addr_space.get());
    
    testAllReduceWithElpaWrappers<float>(comm);
    testAllReduceWithElpaWrappers<double>(comm);
    testAllReduceWithElpaWrappers<int>(comm);
    testAllReduceWithElpaWrappers<int64_t>(comm);
    
    testReduceWithElpaWrappers<float>(comm);
    testReduceWithElpaWrappers<double>(comm);
    testReduceWithElpaWrappers<int>(comm);
    testReduceWithElpaWrappers<int64_t>(comm);
    
    testBroadcastWithElpaWrappers<float>(comm);
    testBroadcastWithElpaWrappers<double>(comm);
    testBroadcastWithElpaWrappers<int>(comm);
    testBroadcastWithElpaWrappers<int64_t>(comm);
    
    testSendRecvWithElpaWrappers<float>(comm);
    testSendRecvWithElpaWrappers<double>(comm);
    testSendRecvWithElpaWrappers<int>(comm);
    testSendRecvWithElpaWrappers<int64_t>(comm);
    
    onecclCommDestroyFromC(comm);
}
