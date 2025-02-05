#if defined(WITH_NVTX)
#define NVTX_RANGE_PUSH(x) call nvtxRangePush(x)
#define NVTX_RANGE_POP(x) call nvtxRangePop()
#elif defined(WITH_ROCTX)
#define NVTX_RANGE_PUSH(x) call roctxRangePush(x)
#define NVTX_RANGE_POP(x) call roctxRangePop()
#else
#define NVTX_RANGE_PUSH(x)
#define NVTX_RANGE_POP(x)
#endif
