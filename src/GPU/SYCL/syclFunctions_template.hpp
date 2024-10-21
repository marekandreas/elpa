/*
//    Copyright 2022, A. Marek
//
//    This file is part of ELPA.
//
//    The ELPA library was originally created by the ELPA consortium,
//    consisting of the following organizations:
//
//    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
//    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
//      Informatik,
//    - Technische Universität München, Lehrstuhl für Informatik mit
//      Schwerpunkt Wissenschaftliches Rechnen ,
//    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
//    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
//      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
//      and
//    - IBM Deutschland GmbH
//
//
//    More information can be found here:
//    http://elpa.mpcdf.mpg.de/
//
//    ELPA is free software: you can redistribute it and/or modify
//    it under the terms of the version 3 of the license of the
//    GNU Lesser General Public License as published by the Free
//    Software Foundation.
//
//    ELPA is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU Lesser General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
//
//    ELPA reflects a substantial effort on the part of the original
//    ELPA consortium, and we ask you to respect the spirit of the
//    license that we chose: i.e., please contribute any changes you
//    may have back to the original ELPA library distribution, and keep
//    any derivatives of ELPA under the same license that we chose for
//    the original distribution, the GNU Lesser General Public License.
//
// This file was written by A. Marek, MPCDF (2022)
// it is based on a prototype implementation developed for MPCDF
// by A. Poeppl, Intel Corporation (2022)
*/

#include <sycl/sycl.hpp>

#include <complex>
#include <oneapi/mkl.hpp>

#include <iostream>
#include <cstdint>
#include <vector>
#include <optional>
#include <cstdint>

#include "config-f90.h"

#include "syclCommon.hpp"

#ifdef WITH_SYCL_GPU_VERSION

using namespace sycl_be;

extern "C" {

static void collectGpuDevices(bool onlyGpus) {
  SyclState::initialize(onlyGpus);
}

bool isCPU=0;

static oneapi::mkl::transpose transposeFromChar(char c) {
  switch (c) {
    case 'C': [[fallthrough]];
    case 'c': return oneapi::mkl::transpose::conjtrans;
    case 'T': [[fallthrough]];
    case 't': return oneapi::mkl::transpose::trans;
    case 'N': [[fallthrough]];
    case 'n': [[fallthrough]];
    default:  return oneapi::mkl::transpose::nontrans;
  }
}

static oneapi::mkl::uplo uploFromChar(char c) {
  switch (c) {
    case 'U': [[fallthrough]];
    case 'u': return oneapi::mkl::uplo::upper;
    case 'L': [[fallthrough]];
    case 'l': [[fallthrough]];
    default: return oneapi::mkl::uplo::lower;

  }
}

static oneapi::mkl::diag diagFromChar(char c) {
  switch (c) {
    case 'n': [[fallthrough]];
    case 'N': return oneapi::mkl::diag::nonunit;
    case 'U': [[fallthrough]];
    case 'u': [[fallthrough]];
    default: return oneapi::mkl::diag::unit;

  }
}

static oneapi::mkl::side sideFromChar(char c) {
  switch (c) {
    case 'l': [[fallthrough]];
    case 'L': return oneapi::mkl::side::left;
    case 'r': [[fallthrough]];
    case 'R': [[fallthrough]];
    default: return oneapi::mkl::side::right;

  }
}


  int syclMemcpyHostToDevice = 20;
  int syclMemcpyDeviceToHost = 200;
  int syclMemcpyDeviceToDevice = 2000;


  int syclPrintGpuInfoFromC() {
    SyclState::defaultState().printGpuInfo();
    return 0;
  }

  int syclPrintDevicesFromC() {
    SyclState::defaultState().printGpuInfo();
    return 0;
  }

  int syclMemcpyDeviceToDeviceFromC(){
    return syclMemcpyDeviceToDevice;
  }

  int syclMemcpyHostToDeviceFromC(){
    return syclMemcpyHostToDevice;
  }
  int syclMemcpyDeviceToHostFromC(){
    return syclMemcpyDeviceToHost;
  }

  int syclStateInitializeFromC(int onlyL0Gpus) {
    bool isInitializedSuccessfully = SyclState::initialize(onlyL0Gpus != 0);
    return isInitializedSuccessfully ? 1 : 0;
  }

  int syclGetDeviceCountFromC(int *count) {
    *count = SyclState::defaultState().getNumDevices();
    return 1;
  }

  int syclSetDeviceFromC(int targetDeviceId) {
    try {
      SyclState::defaultState().selectGpuDevice(targetDeviceId);
    } catch(std::runtime_error &e) {
      std::cout << "Device #" << targetDeviceId << " cannot be selected." << std::endl;
      return 0;
    }
    return 1;
  }

  int syclHostRegisterDefaultFromC() {
    return 0;
  }

  int syclHostRegisterPortableFromC() {
    return 1;
  }

  int syclHostRegisterMappedFromC() {
    return 2;
  }

  int syclGetLastErrorFromC() {
    return 1;
  }

  int syclDeviceGetAttributeFromC(int *value, int attribute) {
    namespace sid = sycl::info::device;
    DeviceSelection &devSel = SyclState::defaultState().getDefaultDeviceHandle();
    sycl::device &dev = devSel.device;
    sycl::range<3> maxWgDim = dev.get_info<sid::max_work_item_sizes<3>>();
    auto sgSizes = dev.get_info<sid::sub_group_sizes>();
    auto maxSgSize = *std::max_element(sgSizes.begin(), sgSizes.end());
    switch(attribute) {
      case 0: return dev.get_info<sid::max_work_group_size>();
      case 1: return maxWgDim[2];
      case 2: return maxWgDim[1];
      case 3: return maxWgDim[0];
      case 4: return UINT32_MAX / maxWgDim[2];
      case 5: return UINT32_MAX / maxWgDim[1];
      case 6: return UINT32_MAX / maxWgDim[0];
      case 7: return maxSgSize;        
      case 8: return dev.get_info<sid::max_compute_units>();
      default: return 0;
    }
  }

  int syclblasGetVersionFromC(QueueData *blasHandle, int *version) {
    // This may not be 100% portable if a non-intel MKL is targeted. 
    // If that should ever be the case, one should have a look at this.
    MKLVersion mklVersion;
    mkl_get_version(&mklVersion);
    return mklVersion.MajorVersion * 10000 + mklVersion.MinorVersion * 100 + mklVersion.UpdateVersion;
  }

  int syclStreamCreateFromC(QueueData **my_stream) {
    QueueData *queueHandle = SyclState::defaultState().getDefaultDeviceHandle().createQueue();
    *my_stream = queueHandle;
    return 1;
  }

  int syclStreamDestroyFromC(QueueData *my_stream) {
    bool isSuccessfullyDeleted = SyclState::defaultState().getDefaultDeviceHandle().destroyQueue(my_stream);
    return isSuccessfullyDeleted ? 1 : 0;
  }

  int syclStreamSynchronizeExplicitFromC(QueueData *my_stream) {
    sycl::queue q = getQueueOrDefault(my_stream);
    q.wait();
    return 1;
  }

  int syclStreamSynchronizeImplicitFromC() {
    sycl::queue q = getQueueOrDefault(nullptr);
    q.wait();
    return 1;
  }

  int syclDeviceSynchronizeFromC() {
    sycl::queue q = getQueueOrDefault(nullptr);
    q.wait();
    return 1;
  }

/*
    interface
    function sycl_devicesynchronize_c() result(istat) &
             bind(C,name="syclDeviceSynchronizeFromC")

      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)                       :: istat
    end function
  end interface
*/


  int syclblasSetStreamFromC(QueueData *syclBlasHandle, QueueData *syclStream) {
    // This does nothing, as QueueData is being used for everything here, and it contains both queue and scratchpad.
    return 1;
  }

  void syclSetGpuParamsFromC() {
    // These do not really have any meaning, as there is an universal address space,
    // and as long as the pointer is either a host pointer, or of the chosen GPU, there
    // is no need to indicate the direction. However, they are used by ELPA, and we can
    // use them as an indication to help detect programming errors such as switched src
    // and dst.
    syclMemcpyHostToDevice = 0;
    syclMemcpyDeviceToHost = 1;
    syclMemcpyDeviceToDevice = 2;
  }

  int syclblasCreateFromC(QueueData **handle) {
    //stub function
    return 1;
  }

  int syclblasDestroyFromC(QueueData **handle) {
    //stub function
    return 1;
  }

  int syclsolverCreateFromC(QueueData **handle) {
    //stub function
    return 1;
  }

  int syclsolverDestroyFromC(QueueData **handle) {
    //stub function
    return 1;
  }

  int syclMallocFromC(intptr_t *a, size_t elems) {
    DeviceSelection &devSel = SyclState::defaultState().getDefaultDeviceHandle();
    *a = reinterpret_cast<intptr_t>(sycl::malloc_device(elems, devSel.device, devSel.context));
    char *bytes = reinterpret_cast<char *>(*a);

    using sycl::usm::alloc;
    auto allocStr = [] (alloc al) {
      switch (al) {
        case alloc::host: return "alloc::host";
        case alloc::device: return "alloc::device";
        case alloc::unknown: return "alloc::unknown";
        default: return "alloc::????";
      }
    };

    auto allocT = sycl::get_pointer_type(bytes, devSel.context);
    // queue.wait();
    std::cerr << "ALLOC |" << "SYCL USM" << "| ~> void *: " << (size_t (*a)) << " -> " << allocStr(allocT) << "\n";

    if (*a) {
      //std::cout << "Allocated " << elems << "B starting at address " << *a << std::endl;
      return 1;
    } else {
      std::cout << "Failed to allocate " << elems << "B on device." << std::endl;
      return 0;
    }
  }

    int syclMallocHostFromC(intptr_t *a, size_t elems) {
    DeviceSelection &devSel = SyclState::defaultState().getDefaultDeviceHandle();
    *a = reinterpret_cast<intptr_t>(sycl::malloc_host(elems, devSel.context));
    char *bytes = reinterpret_cast<char *>(*a);

    using sycl::usm::alloc;
    auto allocStr = [] (alloc al) {
      switch (al) {
        case alloc::host: return "alloc::host";
        case alloc::device: return "alloc::device";
        case alloc::unknown: return "alloc::unknown";
        default: return "alloc::????";
      }
    };

    auto allocT = sycl::get_pointer_type(bytes, devSel.context);
    // queue.wait();
    std::cerr << "ALLOC |" << "SYCL USM" << "| ~> void *: " << (size_t (*a)) << " -> " << allocStr(allocT) << "\n";


    if (*a) {
      //std::cout << "Allocated " << elems << "B starting at address " << *a << std::endl;
      return 1;
    } else {
      std::cout << "Failed to allocate " << elems << "B on device." << std::endl;
      return 0;
    }
  }


  int syclFreeFromC(void *a) {
    DeviceSelection &devSel = SyclState::defaultState().getDefaultDeviceHandle();
    using sycl::usm::alloc;
    auto allocStr = [] (alloc a) {
      switch (a) {
        case alloc::host: return "alloc::host";
        case alloc::device: return "alloc::device";
        case alloc::unknown: return "alloc::unknown";
        default: return "alloc::????";
      }
    };

    auto allocT = sycl::get_pointer_type(a, devSel.context);
    // queue.wait();
    std::cerr << "FREE |" << "syclFree" << "| ~> void **: " << ((size_t) a) << " -> " << allocStr(allocT) << "\n";
    sycl::free(a, devSel.context);
    return 1;
  }

  int syclFreeHostFromC(void *a) {
    return syclFreeFromC(a);
  }

  bool checkPointerValidity(void *dst, void *src, int direction, sycl::queue queue) {
    using sycl::usm::alloc;
    auto allocStr = [] (alloc a) {
      switch (a) {
        case alloc::host: return "alloc::host";
        case alloc::device: return "alloc::device";
        case alloc::unknown: return "alloc::unknown";
        default: return "alloc::????";
      }
    };
    auto dirStr = [] (int dir) {
      if    (dir == syclMemcpyDeviceToDevice) { return "Devc->Devc"; }
      else if (dir == syclMemcpyDeviceToHost) { return "Devc->Host"; }
      else if (dir == syclMemcpyHostToDevice) { return "Host->Devc"; }
      else { return "Invalid"; }
    };
    sycl::context c = queue.get_context();
    auto dstAllocT = sycl::get_pointer_type(dst, c);
    auto srcAllocT = sycl::get_pointer_type(src, c);
    std::cerr << "MEMCPY |" << dirStr(direction) << "| ~> Dst: " << ((size_t) dst) << " -> " << allocStr(dstAllocT) << ", Src: " << ((size_t) src) << " -> " << allocStr(srcAllocT) << "\n";
    bool isFailed = false;
    /*if (isCPU == 1) {
      if (direction == syclMemcpyDeviceToDevice) {
        if (sycl::get_pointer_type(dst, c) != alloc::host) {
          std::cerr << "Pointer dst (" << reinterpret_cast<intptr_t>(dst) << ") is not a device pointer in the context of the chosen CPU queue." << std::endl;
          isFailed = true;
        }
        if (sycl::get_pointer_type(src, c) != alloc::host) {
          std::cerr << "Pointer src (" << reinterpret_cast<intptr_t>(src) << ") is not a device pointer in the context for the chosen CPU queue." << std::endl;
          isFailed = true;
        }
      } else if (direction == syclMemcpyDeviceToHost) {
        if (sycl::get_pointer_type(dst, c) != alloc::host) {
          std::cerr << "Pointer dst (" << reinterpret_cast<intptr_t>(dst) << ") is likely not a host pointer!." << std::endl;
          isFailed = true;
        }
        if (sycl::get_pointer_type(src, c) != alloc::host) {
          std::cerr << "Pointer src (" << reinterpret_cast<intptr_t>(src) << ") is not a device pointer in the context of the chosen CPU queue." << std::endl;
          isFailed = true;
        }
      } else if (direction == syclMemcpyHostToDevice) {
        if (sycl::get_pointer_type(dst, c) != alloc::host) {
          std::cerr << "Pointer dst (" << reinterpret_cast<intptr_t>(dst) << ") is not a device pointer in the context of the chosen CPU queue." << std::endl;
          isFailed = true;
        }
        if (sycl::get_pointer_type(src, c) != alloc::host) {
          std::cerr << "Pointer src (" << reinterpret_cast<intptr_t>(src) << ") is likely not a host pointer!." << std::endl;
          isFailed = true;
        }
      } else {
        std::cerr << "Direction of transfer for memcpy unknown" << std::endl;
        isFailed = true;
      }
    } 
    else */ {
      if (direction == syclMemcpyDeviceToDevice) {
        if (sycl::get_pointer_type(dst, c) != alloc::device) {
          std::cerr << "Pointer dst (" << reinterpret_cast<intptr_t>(dst) << ") is not a device pointer in the context of the chosen GPU queue." << std::endl;
          isFailed = true;
        }
        if (sycl::get_pointer_type(src, c) != alloc::device) {
          std::cerr << "Pointer src (" << reinterpret_cast<intptr_t>(src) << ") is not a device pointer in the context for the chosen GPU queue." << std::endl;
          isFailed = true;
        }
      } else if (direction == syclMemcpyDeviceToHost) {
        if (sycl::get_pointer_type(dst, c) == alloc::device) {
          std::cerr << "Pointer dst (" << reinterpret_cast<intptr_t>(dst) << ") is a device pointer (but expected host/unknown)!." << std::endl;
          isFailed = true;
        }
        if (sycl::get_pointer_type(src, c) != alloc::device) {
          std::cerr << "Pointer src (" << reinterpret_cast<intptr_t>(src) << ") is not a device pointer in the context of the chosen GPU queue." << std::endl;
          isFailed = true;
        }
      } else if (direction == syclMemcpyHostToDevice) {
        if (sycl::get_pointer_type(dst, c) != alloc::device) {
          std::cerr << "Pointer dst (" << reinterpret_cast<intptr_t>(dst) << ") is not a device pointer in the context of the chosen GPU queue." << std::endl;
          isFailed = true;
        }
        if (sycl::get_pointer_type(src, c) == alloc::device) {
          std::cerr << "Pointer src (" << reinterpret_cast<intptr_t>(src) << ") is a device pointer (but expected host/unknown)!." << std::endl;
          isFailed = true;
        }
      } else {
        std::cerr << "Direction of transfer for memcpy unknown" << std::endl;
        isFailed = true;
      }
    }
    return isFailed;
  }

  int syclMemcpyFromC(void *dst, void *src, size_t size, int direction) {
    QueueData *qHandle = getQueueDataOrDefault(nullptr);
    sycl::queue queue = qHandle->queue;
    bool isFailed = false;
#ifndef NDEBUG
    isFailed = checkPointerValidity(dst, src, direction, queue);
#endif
    if (!isFailed) {
      queue.memcpy(dst, src, size).wait();
      return 1;
    } else {
      return 0;
    }
  }

  int syclMemcpyAsyncFromC(void *dst, void *src, size_t size, int direction, QueueData *queue_handle) {
    auto queue = queue_handle->queue;
    bool isFailed = false;
#ifndef NDEBUG
    isFailed = checkPointerValidity(dst, src, direction, queue);
#endif
    if (!isFailed) {
      queue.memcpy(dst, src, size);
      return 1;
    } else {
      return 0;
    }
  }

  int syclMemcpy2dFromC(void *dst, size_t dpitch, void *src, size_t spitch, size_t width, size_t height, int direction) {
    QueueData *qHandle = getQueueDataOrDefault(nullptr);
    sycl::queue queue = qHandle->queue;
    bool isFailed = false;
#ifndef NDEBUG
    isFailed = checkPointerValidity(dst, src, direction, queue);
#endif
    if (!isFailed) {
      // Note that this operation currently relies on an Intel SYCL extension. This may or may not become part of the next SYCL standard.
      // For now, it is only supported by DPC++ and the Intel C++ Compiler. This should be okay, since there are implementations for the other vendors. 
      queue.ext_oneapi_memcpy2d(dst, dpitch, src, spitch, width, height).wait();
      return 1;
    } else {
      return 0;
    }
  }

  int syclMemcpy2dAsyncFromC(void *dst, size_t dpitch, void *src, size_t spitch, size_t width, size_t height, int direction, QueueData *queue_handle) {
    auto queue = queue_handle->queue;
    bool isFailed = false;
#ifndef NDEBUG
    isFailed = checkPointerValidity(dst, src, direction, queue);
#endif
    if (!isFailed) {
      // Note that this operation currently relies on an Intel SYCL extension. This may or may not become part of the next SYCL standard.
      // For now, it is only supported by DPC++ and the Intel C++ Compiler. This should be okay, since there are implementations for the other vendors. 
      queue.ext_oneapi_memcpy2d(dst, dpitch, src, spitch, width, height);
      return 1;
    } else {
      return 0;
    }
  }


  int syclMemsetFromC(void *mem, int32_t val, size_t size) {
    // No handle passed, use default queue handle.
    QueueData *qHandle = getQueueDataOrDefault(nullptr);
    sycl::queue queue = qHandle->queue;
  #ifndef NDEBUG
    if (isCPU == 1) {
      if (sycl::get_pointer_type(mem, queue.get_context()) != sycl::usm::alloc::host) {
        std::cerr << "Pointer (" << reinterpret_cast<intptr_t>(mem) << ") is not a device pointer in the context of the chosen CPU queue." << std::endl;
        return 0;
      }
    } else {
      if (sycl::get_pointer_type(mem, queue.get_context()) != sycl::usm::alloc::device) {
        std::cerr << "Pointer (" << reinterpret_cast<intptr_t>(mem) << ") is not a device pointer in the context of the chosen GPU queue." << std::endl;
        return 0;
      }
    }
#endif
    queue.memset(mem, val, size).wait();
    return 1;
  }

  int syclMemsetAsyncFromC(void *mem, int32_t val, size_t size, QueueData *queue_handle) {
    sycl::queue queue = queue_handle->queue;
#ifndef NDEBUG
    if (isCPU == 1) {
      if (sycl::get_pointer_type(mem, queue.get_context()) != sycl::usm::alloc::host) {
        std::cerr << "Pointer (" << reinterpret_cast<intptr_t>(mem) << ") is not a device pointer in the context of the chosen CPU queue." << std::endl;
        return 0;
      }
    } else {
      if (sycl::get_pointer_type(mem, queue.get_context()) != sycl::usm::alloc::device) {
        std::cerr << "Pointer (" << reinterpret_cast<intptr_t>(mem) << ") is not a device pointer in the context of the chosen GPU queue." << std::endl;
        return 0;
      }
    }
#endif
    queue.memset(mem, val, size);
    return 1;
  }

  int syclHostRegisterFromC(void *ptr, size_t length, int flags) {
#if defined(SYCL_EXT_ONEAPI_COPY_OPTIMIZE) && SYCL_EXT_ONEAPI_COPY_OPTIMIZE == 1
    // oneAPI SYCL extension is available, may not always be the case, especially with other implementations, such as AdaptiveCpp
    DeviceSelection &devSel = SyclState::defaultState().getDefaultDeviceHandle();
    sycl::ext::oneapi::experimental::prepare_for_device_copy(ptr, length, devSel.context);
#else
    // Do nothing, as SYCL standard does not support the operation, it only affects performance.
#endif
    return 1;
  }

  int syclHostUnregisterFromC(void *ptr) {
    #if defined(SYCL_EXT_ONEAPI_COPY_OPTIMIZE) && SYCL_EXT_ONEAPI_COPY_OPTIMIZE == 1
    // oneAPI SYCL extension is available, may not always be the case, especially with other implementations, such as AdaptiveCpp
    DeviceSelection &devSel = SyclState::defaultState().getDefaultDeviceHandle();
    sycl::ext::oneapi::experimental::release_from_device_copy(ptr, devSel.context);
#else
    // Do nothing, as SYCL standard does not support the operation, it only affects performance.
#endif
    return 1;
  }


  void syclblasDgemm_elpa_wrapper(QueueData *handle, char cta, char ctb, int m, int n, int k, double alpha, void *a, int lda, void *b, int ldb, double beta, void *c, int ldc) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t m_, n_, k_, lda_, ldb_, ldc_;
    m_ = (std::int64_t) m;
    n_ = (std::int64_t) n;
    k_ = (std::int64_t) k;
    lda_ = (std::int64_t) lda;
    ldb_ = (std::int64_t) ldb;
    ldc_ = (std::int64_t) ldc;
    using oneapi::mkl::blas::column_major::gemm;
    auto ta = transposeFromChar(cta);
    auto tb = transposeFromChar(ctb);
    gemm(queue, ta, tb, m_, n_, k_, alpha, reinterpret_cast<double *>(a), lda_, reinterpret_cast<double *>(b), ldb_, beta, reinterpret_cast<double *>(c), ldc_);
  }

  void syclblasSgemm_elpa_wrapper(QueueData *handle, char cta, char ctb, int m, int n, int k, float alpha, void *a, int lda, void *b, int ldb, float beta, void *c, int ldc) {
    // handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t m_, n_, k_, lda_, ldb_, ldc_;
    m_ = (std::int64_t) m;
    n_ = (std::int64_t) n;
    k_ = (std::int64_t) k;
    lda_ = (std::int64_t) lda;
    ldb_ = (std::int64_t) ldb;
    ldc_ = (std::int64_t) ldc;
    using oneapi::mkl::blas::column_major::gemm;
    auto ta = transposeFromChar(cta);
    auto tb = transposeFromChar(ctb);
    gemm(queue, ta, tb, m_, n_, k_, alpha, reinterpret_cast<float *>(a), lda_, reinterpret_cast<float *>(b), ldb_, beta, reinterpret_cast<float *>(c), ldc_);
  }

  void syclblasZgemm_elpa_wrapper(QueueData *handle, char cta, char ctb, int m, int n, int k, std::complex<double> alpha, void *a, int lda, void *b, int ldb, std::complex<double> beta, void *c, int ldc) {
    // handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t m_, n_, k_, lda_, ldb_, ldc_;
    m_ = (std::int64_t) m;
    n_ = (std::int64_t) n;
    k_ = (std::int64_t) k;
    lda_ = (std::int64_t) lda;
    ldb_ = (std::int64_t) ldb;
    ldc_ = (std::int64_t) ldc;
    using oneapi::mkl::blas::column_major::gemm;
    auto ta = transposeFromChar(cta);
    auto tb = transposeFromChar(ctb);
    gemm(queue, ta, tb, m_, n_, k_, alpha, reinterpret_cast<std::complex<double> *>(a), lda_, reinterpret_cast<std::complex<double> *>(b), ldb_, beta, reinterpret_cast<std::complex<double> *>(c), ldc_);
  }

  void syclblasCgemm_elpa_wrapper(QueueData *handle, char cta, char ctb, int m, int n, int k, std::complex<float> alpha, void *a, int lda, void *b, int ldb, std::complex<float> beta, void *c, int ldc) {
    // handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t m_, n_, k_, lda_, ldb_, ldc_;
    m_ = (std::int64_t) m;
    n_ = (std::int64_t) n;
    k_ = (std::int64_t) k;
    lda_ = (std::int64_t) lda;
    ldb_ = (std::int64_t) ldb;
    ldc_ = (std::int64_t) ldc;
    using oneapi::mkl::blas::column_major::gemm;
    auto ta = transposeFromChar(cta);
    auto tb = transposeFromChar(ctb);
    gemm(queue, ta, tb, m_, n_, k_, alpha, reinterpret_cast<std::complex<float> *>(a), lda_, reinterpret_cast<std::complex<float> *>(b), ldb_, beta, reinterpret_cast<std::complex<float> *>(c), ldc_);
  }

  // implemented in mkl???
  //
  void syclblasDtrtri_elpa_wrapper(QueueData *handle, char uplo, char diag, int n, void *a, int lda, int info) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    //using oneapi::mkl::blas::column_major::gemm;
    auto up = uploFromChar(uplo);
    auto di = diagFromChar(diag);

    throw std::runtime_error("Not Implemented, do not call.");
    // FIXME trtri is currently unavailable on the GPU!
    // dtrtri(&uplo, &diag, &n, reinterpret_cast<float *>(a), &lda, &info);
  }

  void syclblasStrtri_elpa_wrapper(QueueData *handle, char uplo, char diag, int n, void *a, int lda, int info) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    //using oneapi::mkl::blas::column_major::gemm;
    auto up = uploFromChar(uplo);
    auto di = diagFromChar(diag);

    throw std::runtime_error("Not Implemented, do not call.");
    // FIXME trtri is currently unavailable on the GPU!
    // dtrtri(&uplo, &diag, &n, reinterpret_cast<float *>(a), &lda, &info);
  }

  void syclblasZtrtri_elpa_wrapper(QueueData *handle, char uplo, char diag, int n, void *a, int lda, int info) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    //using oneapi::mkl::blas::column_major::gemm;
    auto up = uploFromChar(uplo);
    auto di = diagFromChar(diag);
    throw std::runtime_error("Not Implemented, do not call.");

    // FIXME trtri is currently unavailable on the GPU!
    // dtrtri(&uplo, &diag, &n, reinterpret_cast<float *>(a), &lda, &info);
  }

  void syclblasCtrtri_elpa_wrapper(QueueData *handle, char uplo, char diag, int n, void *a, int lda, int info) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    //using oneapi::mkl::blas::column_major::gemm;
    auto up = uploFromChar(uplo);
    auto di = diagFromChar(diag);
    throw std::runtime_error("Not Implemented, do not call.");
    // FIXME trtri is currently unavailable on the GPU!
    //dtrtri(&uplo, &diag, &n, reinterpret_cast<float *>(a), &lda, &info);
  }

  // different API!!
  void syclblasDpotrf_elpa_wrapper(QueueData *handle, char uplo, int n, void *a, int lda, int info) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    using oneapi::mkl::lapack::potrf;
    auto up = uploFromChar(uplo);
    throw std::runtime_error("Not Implemented, do not call.");
    //void potrf( ..., int lda, &scratchpad, std::int64_t scratchpad_size)
    //potrf(queue, up, &n, reinterpret_cast<double *>(a), &lda, &info);
  }

  void syclblasSpotrf_elpa_wrapper(QueueData *handle, char uplo, int n, void *a, int lda, int info) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    using oneapi::mkl::lapack::potrf;
    auto up = uploFromChar(uplo);
    throw std::runtime_error("Not Implemented, do not call.");
    //void potrf( ..., int lda, &scratchpad, std::int64_t scratchpad_size)
    //potrf(queue, up, &n, reinterpret_cast<float *>(a), &lda, &info);
  }

  void syclblasZpotrf_elpa_wrapper(QueueData *handle, char uplo, int n, void *a, int lda, int info) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    using oneapi::mkl::lapack::potrf;
    auto up = uploFromChar(uplo);
    throw std::runtime_error("Not Implemented, do not call.");
    //void potrf( ..., int lda, &scratchpad, std::int64_t scratchpad_size)
    //potrf(queue, up, &n, reinterpret_cast<std::complex<double> *>(c), &lda, &info);
  }

  void syclblasCpotrf_elpa_wrapper(QueueData *handle, char uplo, int n, void *a, int lda, int info) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    using oneapi::mkl::lapack::potrf;
    auto up = uploFromChar(uplo);
    throw std::runtime_error("Not Implemented, do not call.");
    //void potrf( ..., int lda, &scratchpad, std::int64_t scratchpad_size)
    //potrf(queue, up, &n, reinterpret_cast<std::complex<float> *>(c), &lda, &info);
  }

  void syclblasDcopy_elpa_wrapper(QueueData *handle, int n, void *x, int incx, void *y, int incy) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t n_, incx_, incy_;
    n_ = (std::int64_t) n;
    incx_ = (std::int64_t) incx;
    incy_ = (std::int64_t) incy;
    using oneapi::mkl::blas::column_major::copy;
    copy(queue, n_, reinterpret_cast<double *>(x), incx_, reinterpret_cast<double *>(y), incy_);
  }

  void syclblasScopy_elpa_wrapper(QueueData *handle, int n, void *x, int incx, void *y, int incy) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t n_, incx_, incy_;
    n_ = (std::int64_t) n;
    incx_ = (std::int64_t) incx;
    incy_ = (std::int64_t) incy;
    using oneapi::mkl::blas::column_major::copy;
    copy(queue, n_, reinterpret_cast<float *>(x), incx_, reinterpret_cast<float *>(y), incy_);
  }

  void syclblasZcopy_elpa_wrapper(QueueData *handle, int n, void *x, int incx, void *y, int incy) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t n_, incx_, incy_;
    n_ = (std::int64_t) n;
    incx_ = (std::int64_t) incx;
    incy_ = (std::int64_t) incy;
    using oneapi::mkl::blas::column_major::copy;
    copy(queue, n_, reinterpret_cast<std::complex<double> *>(x), incx_, reinterpret_cast<std::complex<double> *>(y), incy_);
  }

  void syclblasCcopy_elpa_wrapper(QueueData *handle, int n, void *x, int incx, void *y, int incy) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t n_, incx_, incy_;
    n_ = (std::int64_t) n;
    incx_ = (std::int64_t) incx;
    incy_ = (std::int64_t) incy;
    using oneapi::mkl::blas::column_major::copy;
    copy(queue, n_, reinterpret_cast<std::complex<float> *>(x), incx_, reinterpret_cast<std::complex<float> *>(y), incy_);
  }

  void syclblasDtrmm_elpa_wrapper(QueueData *handle, char side, char uplo, char trans, char diag, int m, int n, double alpha, void *a, int lda, void *b, int ldb) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t m_, n_, lda_, ldb_;
    m_ = (std::int64_t) m;
    n_ = (std::int64_t) n;
    lda_ = (std::int64_t) lda;
    ldb_ = (std::int64_t) ldb;
    using oneapi::mkl::blas::column_major::trmm;
    auto sd = sideFromChar(side);
    auto up = uploFromChar(uplo);
    auto ta = transposeFromChar(trans);
    auto di = diagFromChar(diag);
    trmm(queue, sd, up, ta, di, m_, n_, alpha, reinterpret_cast<double *>(a), lda_, reinterpret_cast<double *>(b), ldb_);
  }

  void syclblasStrmm_elpa_wrapper(QueueData *handle, char side, char uplo, char trans, char diag, int m, int n, float alpha, void *a, int lda, void *b, int ldb) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t m_, n_, lda_, ldb_;
    m_ = (std::int64_t) m;
    n_ = (std::int64_t) n;
    lda_ = (std::int64_t) lda;
    ldb_ = (std::int64_t) ldb;
    using oneapi::mkl::blas::column_major::trmm;
    auto sd = sideFromChar(side);
    auto up = uploFromChar(uplo);
    auto ta = transposeFromChar(trans);
    auto di = diagFromChar(diag);
    trmm(queue, sd, up, ta, di, m_, n_, alpha, reinterpret_cast<float *>(a), lda_, reinterpret_cast<float *>(b), ldb_);
  }

  void syclblasZtrmm_elpa_wrapper(QueueData *handle, char side, char uplo, char trans, char diag, int m, int n, std::complex<double> alpha, void *a, int lda, void *b, int ldb) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t m_, n_, lda_, ldb_;
    m_ = (std::int64_t) m;
    n_ = (std::int64_t) n;
    lda_ = (std::int64_t) lda;
    ldb_ = (std::int64_t) ldb;
    using oneapi::mkl::blas::column_major::trmm;
    auto sd = sideFromChar(side);
    auto up = uploFromChar(uplo);
    auto ta = transposeFromChar(trans);
    auto di = diagFromChar(diag);
    trmm(queue, sd, up, ta, di, m_, n_, alpha, reinterpret_cast<std::complex<double> *>(a), lda_, reinterpret_cast<std::complex<double> *>(b), ldb_);
  }

  void syclblasCtrmm_elpa_wrapper(QueueData *handle, char side, char uplo, char trans, char diag, int m, int n, std::complex<float> alpha, void *a, int lda, void *b, int ldb) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t m_, n_, lda_, ldb_;
    m_ = (std::int64_t) m;
    n_ = (std::int64_t) n;
    lda_ = (std::int64_t) lda;
    ldb_ = (std::int64_t) ldb;
    using oneapi::mkl::blas::column_major::trmm;
    auto sd = sideFromChar(side);
    auto up = uploFromChar(uplo);
    auto ta = transposeFromChar(trans);
    auto di = diagFromChar(diag);
    trmm(queue, sd, up, ta, di, m_, n_, alpha, reinterpret_cast<std::complex<float> *>(a), lda_, reinterpret_cast<std::complex<float> *>(b), ldb_);
  }

  void syclblasDtrsm_elpa_wrapper(QueueData *handle, char side, char uplo, char trans, char diag, int m, int n, double alpha, void *a, int lda, void *b, int ldb) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t m_, n_, lda_, ldb_;
    m_ = (std::int64_t) m;
    n_ = (std::int64_t) n;
    lda_ = (std::int64_t) lda;
    ldb_ = (std::int64_t) ldb;
    using oneapi::mkl::blas::column_major::trsm;
    auto sd = sideFromChar(side);
    auto up = uploFromChar(uplo);
    auto ta = transposeFromChar(trans);
    auto di = diagFromChar(diag);
    trsm(queue, sd, up, ta, di, m_, n_, alpha, reinterpret_cast<double *>(a), lda_, reinterpret_cast<double *>(b), ldb_);
  }

  void syclblasStrsm_elpa_wrapper(QueueData *handle, char side, char uplo, char trans, char diag, int m, int n, float alpha, void *a, int lda, void *b, int ldb) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t m_, n_, lda_, ldb_;
    m_ = (std::int64_t) m;
    n_ = (std::int64_t) n;
    lda_ = (std::int64_t) lda;
    ldb_ = (std::int64_t) ldb;
    using oneapi::mkl::blas::column_major::trsm;
    auto sd = sideFromChar(side);
    auto up = uploFromChar(uplo);
    auto ta = transposeFromChar(trans);
    auto di = diagFromChar(diag);
    trsm(queue, sd, up, ta, di, m_, n_, alpha, reinterpret_cast<float *>(a), lda_, reinterpret_cast<float *>(b), ldb_);
  }

  void syclblasZtrsm_elpa_wrapper(QueueData *handle, char side, char uplo, char trans, char diag, int m, int n, std::complex<double> alpha, void *a, int lda, void *b, int ldb) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t m_, n_, lda_, ldb_;
    m_ = (std::int64_t) m;
    n_ = (std::int64_t) n;
    lda_ = (std::int64_t) lda;
    ldb_ = (std::int64_t) ldb;
    using oneapi::mkl::blas::column_major::trsm;
    auto sd = sideFromChar(side);
    auto up = uploFromChar(uplo);
    auto ta = transposeFromChar(trans);
    auto di = diagFromChar(diag);
    trsm(queue, sd, up, ta, di, m_, n_, alpha, reinterpret_cast<std::complex<double> *>(a), lda_, reinterpret_cast<std::complex<double> *>(b), ldb_);
  }

  void syclblasCtrsm_elpa_wrapper(QueueData *handle, char side, char uplo, char trans, char diag, int m, int n, std::complex<float> alpha, void *a, int lda, void *b, int ldb) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t m_, n_, lda_, ldb_;
    m_ = (std::int64_t) m;
    n_ = (std::int64_t) n;
    lda_ = (std::int64_t) lda;
    ldb_ = (std::int64_t) ldb;
    using oneapi::mkl::blas::column_major::trsm;
    auto sd = sideFromChar(side);
    auto up = uploFromChar(uplo);
    auto ta = transposeFromChar(trans);
    auto di = diagFromChar(diag);
    trsm(queue, sd, up, ta, di, m_, n_, alpha, reinterpret_cast<std::complex<float> *>(a), lda_, reinterpret_cast<std::complex<float> *>(b), ldb_);
  }

  void syclblasDgemv_elpa_wrapper(QueueData *handle, char cta, int m, int n, double alpha, void *a, int lda, void *x, int incx, double beta, void *y, int incy) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t m_, n_, lda_, incx_, incy_;
    m_ = (std::int64_t) m;
    n_ = (std::int64_t) n;
    lda_ = (std::int64_t) lda;
    incx_ = (std::int64_t) incx;
    incy_ = (std::int64_t) incy;
    using oneapi::mkl::blas::column_major::gemv;
    auto ta = transposeFromChar(cta);
    gemv(queue, ta, m_, n_, alpha, reinterpret_cast<double *>(a), lda_, reinterpret_cast<double *>(x), incx_, beta, reinterpret_cast<double *>(y), incy_);
  }

  void syclblasSgemv_elpa_wrapper(QueueData *handle, char cta, int m, int n, float alpha, void *a, int lda, void *x, int incx, float beta, void *y, int incy) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t m_, n_, lda_, incx_, incy_;
    m_ = (std::int64_t) m;
    n_ = (std::int64_t) n;
    lda_ = (std::int64_t) lda;
    incx_ = (std::int64_t) incx;
    incy_ = (std::int64_t) incy;
    using oneapi::mkl::blas::column_major::gemv;
    auto ta = transposeFromChar(cta);
    gemv(queue, ta, m_, n_, alpha, reinterpret_cast<float *>(a), lda_, reinterpret_cast<float *>(x), incx_, beta, reinterpret_cast<float *>(y), incy_);
  }

  void syclblasZgemv_elpa_wrapper(QueueData *handle, char cta, int m, int n, std::complex<double> alpha, void *a, int lda, void *x, int incx, std::complex<double> beta, void *y, int incy) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t m_, n_, lda_, incx_, incy_;
    m_ = (std::int64_t) m;
    n_ = (std::int64_t) n;
    lda_ = (std::int64_t) lda;
    incx_ = (std::int64_t) incx;
    incy_ = (std::int64_t) incy;
    using oneapi::mkl::blas::column_major::gemv;
    auto ta = transposeFromChar(cta);
    gemv(queue, ta, m_, n_, alpha, reinterpret_cast<std::complex<double> *>(a), lda_, reinterpret_cast<std::complex<double> *>(x), incx_, beta, reinterpret_cast<std::complex<double> *>(y), incy_);
  }

  void syclblasCgemv_elpa_wrapper(QueueData *handle, char cta, int m, int n, std::complex<float> alpha, void *a, int lda, void *x, int incx, std::complex<float> beta, void *y, int incy) {
    //handle not needed
    QueueData *qHandle = getQueueDataOrDefault(handle);
    sycl::queue queue = qHandle->queue;
    std::int64_t m_, n_, lda_, incx_, incy_;
    m_ = (std::int64_t) m;
    n_ = (std::int64_t) n;
    lda_ = (std::int64_t) lda;
    incx_ = (std::int64_t) incx;
    incy_ = (std::int64_t) incy;
    using oneapi::mkl::blas::column_major::gemv;
    auto ta = transposeFromChar(cta);
    gemv(queue, ta, m_, n_, alpha, reinterpret_cast<std::complex<float> *>(a), lda_, reinterpret_cast<std::complex<float> *>(x), incx_, beta, reinterpret_cast<std::complex<float> *>(y), incy_);
  }
} // extern C
#endif /* WITH_SYCL_GPU_VERSION */
