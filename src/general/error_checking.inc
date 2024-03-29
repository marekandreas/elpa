!cannot use __FILE__ because filename with path can be too long for gfortran (max line length)
#define check_memcpy_gpu(file, success) call check_memcpy_GPU_f(file, __LINE__, success)
#define check_stream_synchronize_gpu(file, success) call check_memcpy_GPU_f(file, __LINE__, success)
#define check_alloc_gpu(file, success) call check_alloc_GPU_f(file, __LINE__, success)
#define check_dealloc_gpu(file, success) call check_dealloc_GPU_f(file, __LINE__, success)
#define check_host_register_gpu(file, success) call check_host_register_GPU_f(file, __LINE__, success)
#define check_host_unregister_gpu(file, success) call check_host_unregister_GPU_f(file, __LINE__, success)
#define check_host_alloc_gpu(file, success) call check_host_alloc_GPU_f(file, __LINE__, success)
#define check_host_dealloc_gpu(file, success) call check_host_dealloc_GPU_f(file, __LINE__, success)
#define check_memset_gpu(file, success) call check_memset_GPU_f(file, __LINE__, success)
#define check_allocate(file, success, errorMessage) call check_allocate_f(file, __LINE__, success, errorMessage)
#define check_deallocate(file, success, errorMessage) call check_deallocate_f(file, __LINE__, success, errorMessage)

