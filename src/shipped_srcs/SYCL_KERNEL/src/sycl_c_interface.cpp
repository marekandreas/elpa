#include "syclCommon.hpp"

#include <CL/sycl.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

extern "C" {

int sycl_set_device(int i_gpu) try {
    elpa::gpu::sycl::selectGpuDevice(i_gpu);
    return 0;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int sycl_get_device_count(int *n_gpu) try {
  elpa::gpu::sycl::collectGpuDevices(false);
  *n_gpu = elpa::gpu::sycl::getNumDevices();
  return 0;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int sycl_device_synchronize() try {
    elpa::gpu::sycl::getQueue().wait_and_throw();
    return 0;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int sycl_malloc(intptr_t *a, size_t size) try {
    auto &q = elpa::gpu::sycl::getQueue();
    *a = (intptr_t)sycl::malloc_device(size, q);
    return 0;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int sycl_free(intptr_t *a) try {
    auto &q = elpa::gpu::sycl::getQueue();
    sycl::free(a, q);
    return 0;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int sycl_memcpy(intptr_t *dest, intptr_t *src, size_t count, int dir) try {
    auto &q = elpa::gpu::sycl::getQueue();
    q.memcpy(dest, src, count).wait();
    return 0;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}
