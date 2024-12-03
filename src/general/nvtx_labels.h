#ifdef WITH_NVTX
#define NVTX_RANGE_PUSH(x) call nvtxRangePush(x)
#define NVTX_RANGE_POP(x) call nvtxRangePop()
#else
#define NVTX_RANGE_PUSH(x)
#define NVTX_RANGE_POP(x)
#endif
