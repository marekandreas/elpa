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
// by A. Poeppl, Intel (2022) 
*/

#include <CL/sycl.hpp>

#include <complex>
#include <oneapi/mkl.hpp>

#include <complex>
#include <iostream>
#include <cstdint>
#include <vector>
#include <optional> 
     
#include "config-f90.h"

#ifdef WITH_SYCL_GPU_VERSION
using namespace cl;

static std::vector<sycl::device> devices;
static std::optional<sycl::queue> chosenDeviceQueue{};

bool deviceCollectionFlag = false;

extern "C" {

static void collectGpuDevices() {
  if (deviceCollectionFlag) {
    return;
  }

  for (auto const &p : sycl::platform::get_platforms()) {
    for (auto dev : p.get_devices()) {
      using namespace sycl::info;
// why this?
#ifndef ENABLE_ALL_DEVICES
      if (dev.get_info<device::device_type>() == device_type::gpu)
#endif
      {
        devices.push_back(dev);
      }

    }
  }
}

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
    default: oneapi::mkl::uplo::lower;

  }
}

static oneapi::mkl::diag diagFromChar(char c) {
  switch (c) {
    case 'n': [[fallthrough]];
    case 'N': return oneapi::mkl::diag::nonunit;
    case 'U': [[fallthrough]];
    case 'u': [[fallthrough]];
    default: oneapi::mkl::diag::unit;

  }
}

static oneapi::mkl::side sideFromChar(char c) {
  switch (c) {
    case 'l': [[fallthrough]];
    case 'L': return oneapi::mkl::side::left;
    case 'r': [[fallthrough]];
    case 'R': [[fallthrough]];
    default: oneapi::mkl::side::right;

  }
}


  int syclChosenGpu;

  int syclMemcpyHostToDevice = 20;
  int syclMemcpyDeviceToHost = 200;
  int syclMemcpyDeviceToDevice = 2000;


  void syclPrintGpuInfoFromC() {
    using namespace sycl::info;
    collectGpuDevices();

    std::cout << "~~~~~~~~~~~~~~~~~~~ ELPA GPU Info ~~~~~~~~~~~~~~~~~~~~" << std::endl;
    std::cout << "GPU Backend:       Intel oneAPI SYCL" << std::endl;
    std::cout << "# GPU devices:     " << devices.size() << std::endl;
    std::cout << "Eligible devices: " << std::endl;
    for (size_t i = 0; i < devices.size(); i++) {
      bool hasDpSupport = devices[i].has(sycl::aspect::fp64);
      std::cout << " - Device #" << i << ": "
                << devices[i].get_info<device::name>() << " ("
                << devices[i].get_info<device::max_compute_units>() << " EUs"
                << (hasDpSupport ? "" : ", SP only") << ")" << std::endl;
    }
    std::cout << "~~~~~~~~~ Display Verbose SYCL Platform Info ~~~~~~~~~" << std::endl;
    std::cout << std::endl;
    for (auto const &platform : sycl::platform::get_platforms()) {
      std::cout << " - Platform: " << platform.get_info<platform::name>() << std::endl;
      std::cout << "   * Vendor:  " << platform.get_info<platform::vendor>() << std::endl;
      std::cout << "   * Version: " << platform.get_info<platform::version>() << std::endl;
      std::cout << "   * Profile: " << platform.get_info<platform::vendor>() << std::endl;
      for (auto const &device : platform.get_devices()) {
        std::cout << "    -- Device: " << device.get_info<device::name>() << std::endl;
        std::cout << "       * Device Type:                             ";
        auto deviceType = device.get_info<device::device_type>();
        switch (deviceType) {
          case device_type::cpu:
            std::cout << "CPU" << std::endl;
            break;
          case device_type::gpu:
            std::cout << "GPU" << std::endl;
            break;
          case device_type::accelerator:
            std::cout << "Accelerator" << std::endl;
            break;
          case device_type::custom:
            std::cout << "CUSTOM" << std::endl;
            break;
          case device_type::automatic:
            std::cout << "AUTOMATIC" << std::endl;
            break;
          case device_type::host:
            std::cout << "HOST" << std::endl;
            break;
          case device_type::all:
            default:
            std::cout << "UNKNOWN" << std::endl;
            break;
        }
        std::cout << "       * Max Compute Units:                       " << device.get_info<device::max_compute_units>() << std::endl;
        std::cout << "       * Double Precision Floating Point support: " << ((device.has(sycl::aspect::fp64)) ? "Yes" : "No") << std::endl;
        std::cout << "       * Max Work Item Dimensions:                " << device.get_info<device::max_work_item_dimensions>() << std::endl;
        auto maxWorkItemSize = device.get_info<device::max_work_item_sizes>();
        std::cout << "       * Max Work Item Sizes:                     " << "{" << maxWorkItemSize[0] << ", " << maxWorkItemSize[1] << ", " << maxWorkItemSize[2] << "}" << std::endl;
        std::cout << "       * Max Work Group Sizes:                    " << device.get_info<device::max_work_group_size>() << std::endl;
        std::cout << "       * Max Memory Alloc size:                   " << device.get_info<device::max_mem_alloc_size>() << std::endl;
        std::cout << "       * Max Parameter size:                      " << device.get_info<device::max_parameter_size>() << std::endl;
        std::cout << "       * Global Mem Cache Type:                   ";
        auto globalMemoryCacheType = device.get_info<device::global_mem_cache_type>();
        switch (globalMemoryCacheType) {
          case global_mem_cache_type::none:
            std::cout << "None" << std::endl;
            break;
          case global_mem_cache_type::read_only:
            std::cout << "Read-Only";
            break;
          case global_mem_cache_type::read_write:
            std::cout << "Read-Write";
            break;
          default:
            std::cout << "UNKNOWN ENTRY!";
            break;
          }
          std::cout << std::endl;
          std::cout << "       * Local Mem Type:                          ";
          auto localMemType = device.get_info<device::local_mem_type>();
          switch (localMemType) {
            case local_mem_type::none:
              std::cout << "None";
              break;
            case local_mem_type::local:
              std::cout << "Local";
              break;
            case local_mem_type::global:
              std::cout << "Global";
              break;
            default:
              std::cout << "UNKNOWN ENTRY!";
              break;
          }
          std::cout << std::endl;
          std::cout << "       * Local Mem Size:                          " << device.get_info<device::local_mem_size>() << std::endl;
          std::cout << "       * Host Unified Memory:                     " << device.get_info<device::host_unified_memory>() << std::endl;
        }
    }
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

  int syclGetDeviceCountFromC() {
    collectGpuDevices();
    return devices.size();
  }



  void syclSetDeviceFromC(int targetGpuDeviceId) {
    collectGpuDevices();
    if (targetGpuDeviceId >= devices.size()){
      std::cerr << "Invalid device ID selected, only " << devices.size() << " devices available." << std::endl;
      abort();
    }
    syclChosenGpu = targetGpuDeviceId;
    chosenDeviceQueue = std::make_optional<sycl::queue>(devices[syclChosenGpu]);
  }

  void syclSetGpuParamsFromC() {
    collectGpuDevices();
    // These do not really have any meaning, as there is an universal address space,
    // and as long as the pointer is either a host pointer, or of the chosen GPU, there
    // is no need to indicate the direction. However, they are used by ELPA, and we can
    // use them as an indication to help detect programming errors such as switched src
    // and dst.
    syclMemcpyHostToDevice = 0;
    syclMemcpyDeviceToHost = 1;
    syclMemcpyDeviceToDevice = 2;
  }

  int syclblasCreateFromC(intptr_t* handle){
    //stub function
    return 1;
  }

  int syclsolverCreateFromC(intptr_t* handle){
    //stub function
    return 1;
  }


  int syclMallocFromC(intptr_t *a, size_t elems) {
    if (chosenDeviceQueue) {
      *a = reinterpret_cast<intptr_t>(sycl::malloc_device(elems, *chosenDeviceQueue));
      if (*a) {
        std::cout << "Allocated " << elems << "B starting at address " << *a << std::endl;
        return 1;
      } else {
        std::cout << "Allocation failed!" << std::endl;
        return 0;
      }
    } else {
      std::cerr << "No device selected for allocation." << std::endl;
      return 0;
    }
  }

  int syclFreeFromC(intptr_t *a) {
    if (chosenDeviceQueue) {
      void * ptr = reinterpret_cast<void *>(*a);
      sycl::free(ptr, *chosenDeviceQueue);
      return 1;
    } else {
      std::cerr << "No device selected for deallocation." << std::endl;
      return 0;
    }
  }


  int syclMemcpyFromC(void *dst, void *src, size_t size, int direction) {
    bool isFailed = false;
    using sycl::usm::alloc;
    if (chosenDeviceQueue) {
      if (direction == syclMemcpyDeviceToDevice) {
        if (sycl::get_pointer_type(dst, chosenDeviceQueue->get_context()) != alloc::device) {
          std::cerr << "Pointer dst (" << reinterpret_cast<intptr_t>(dst) << ") is not a device pointer in the context of the chosen GPU queue." << std::endl;
          isFailed = true;
        }
        if (sycl::get_pointer_type(src, chosenDeviceQueue->get_context()) != alloc::device) {
          std::cerr << "Pointer src (" << reinterpret_cast<intptr_t>(src) << ") is not a device pointer in the context for the chosen GPU queue." << std::endl;
          isFailed = true;
        }
      } else if (direction == syclMemcpyDeviceToHost) {
        if (sycl::get_pointer_type(dst, chosenDeviceQueue->get_context()) != alloc::unknown) {
          std::cerr << "Pointer dst (" << reinterpret_cast<intptr_t>(dst) << ") is likely not a host pointer!." << std::endl;
          isFailed = true;
        }
        if (sycl::get_pointer_type(src, chosenDeviceQueue->get_context()) != alloc::device) {
          std::cerr << "Pointer src (" << reinterpret_cast<intptr_t>(src) << ") is not a device pointer in the context of the chosen GPU queue." << std::endl;
          isFailed = true;
        }
      } else if (direction == syclMemcpyHostToDevice) {
        if (sycl::get_pointer_type(dst, chosenDeviceQueue->get_context()) != alloc::device) {
          std::cerr << "Pointer dst (" << reinterpret_cast<intptr_t>(dst) << ") is not a device pointer in the context of the chosen GPU queue." << std::endl;
          isFailed = true;
        }
        if (sycl::get_pointer_type(src, chosenDeviceQueue->get_context()) != alloc::unknown) {
          std::cerr << "Pointer src (" << reinterpret_cast<intptr_t>(src) << ") is likely not a host pointer!." << std::endl;
          isFailed = true;
        }
      } else {
        std::cerr << "Direction of transfer for memcpy unknown" << std::endl;
        isFailed = true;
      }
      if (!isFailed) {
        chosenDeviceQueue->memcpy(dst, src, size).wait();
        return 1;
      } else {
        return 0;
      }
    } else {
      std::cerr << "No device selected for memcopy operation." << std::endl;
      return 0;
    }
  }

  int syclMemsetFromC(void *mem, size_t size, int32_t val) {
    if (chosenDeviceQueue) {
      if (sycl::get_pointer_type(mem, chosenDeviceQueue->get_context()) != sycl::usm::alloc::device) {
        std::cerr << "Pointer (" << reinterpret_cast<intptr_t>(mem) << ") is not a device pointer in the context of the chosen GPU queue." << std::endl;
        return 0;
      }
      chosenDeviceQueue->memset(mem, val, size).wait();
      return 1;
    } else {
      std::cerr << "No device selected for memset operation." << std::endl;
      return 0;
    }
  }

  void mklSyclDgemmFromC(intptr_t *handle, char cta, char ctb, int m, int n, int k, double alpha, void *a, int lda, void *b, int ldb, double beta, void *c, int ldc) {
    //handle not needed
    if (chosenDeviceQueue) {
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
      gemm(*chosenDeviceQueue, ta, tb, m_, n_, k_, alpha, reinterpret_cast<double *>(a), lda_, reinterpret_cast<double *>(b), ldb_, beta, reinterpret_cast<double *>(c), ldc_);
    } else {
      std::cerr << "No device selected for DGEMM operation." << std::endl;
    }
  }

  void mklSyclSgemmFromC(intptr_t *handle, char cta, char ctb, int m, int n, int k, float alpha, void *a, int lda, void *b, int ldb, float beta, void *c, int ldc) {
    // handle not needed
    if (chosenDeviceQueue) {
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
      gemm(*chosenDeviceQueue, ta, tb, m_, n_, k_, alpha, reinterpret_cast<float *>(a), lda_, reinterpret_cast<float *>(b), ldb_, beta, reinterpret_cast<float *>(c), ldc_);
    } else {
      std::cerr << "No device selected for SGEMM operation." << std::endl;
    }
  }

  void mklSyclZgemmFromC(intptr_t *handle, char cta, char ctb, int m, int n, int k, std::complex<double> alpha, void *a, int lda, void *b, int ldb, std::complex<double> beta, void *c, int ldc) {
    // handle not needed
    if (chosenDeviceQueue) {
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
      gemm(*chosenDeviceQueue, ta, tb, m_, n_, k_, alpha, reinterpret_cast<std::complex<double> *>(a), lda_, reinterpret_cast<std::complex<double> *>(b), ldb_, beta, reinterpret_cast<std::complex<double> *>(c), ldc_);
    } else {
      std::cerr << "No device selected for ZGEMM operation." << std::endl;
    }
  }

  void mklSyclCgemmFromC(intptr_t *handle, char cta, char ctb, int m, int n, int k, std::complex<float> alpha, void *a, int lda, void *b, int ldb, std::complex<float> beta, void *c, int ldc) {
    // handle not needed
    if (chosenDeviceQueue) {
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
      gemm(*chosenDeviceQueue, ta, tb, m_, n_, k_, alpha, reinterpret_cast<std::complex<float> *>(a), lda_, reinterpret_cast<std::complex<float> *>(b), ldb_, beta, reinterpret_cast<std::complex<float> *>(c), ldc_);
    } else {
      std::cerr << "No device selected for CGEMM operation." << std::endl;
    }
  }

  // implemented in mkl???
  //
  void mklSyclDtrtriFromC(intptr_t *handle, char uplo, char diag, int n, void *a, int lda, int info) {
    //handle not needed
    if (chosenDeviceQueue) {
      //using oneapi::mkl::blas::column_major::gemm;
      auto up = uploFromChar(uplo);
      auto di = diagFromChar(diag);

//      dtrtri(&uplo, &diag, &n, reinterpret_cast<double *>(a), &lda, &info);
    } else {
      std::cerr << "No device selected for DTRTRI operation." << std::endl;
    }
  }  

  void mklSyclStrtriFromC(intptr_t *handle, char uplo, char diag, int n, void *a, int lda, int info) {
    //handle not needed
    if (chosenDeviceQueue) {
      //using oneapi::mkl::blas::column_major::gemm;
      auto up = uploFromChar(uplo);
      auto di = diagFromChar(diag);

//      dtrtri(&uplo, &diag, &n, reinterpret_cast<float *>(a), &lda, &info);
    } else {
      std::cerr << "No device selected for STRTRI operation." << std::endl;
    }
  }

  void mklSyclZtrtriFromC(intptr_t *handle, char uplo, char diag, int n, void *a, int lda, int info) {
    //handle not needed
    if (chosenDeviceQueue) {
      //using oneapi::mkl::blas::column_major::gemm;
      auto up = uploFromChar(uplo);
      auto di = diagFromChar(diag);

//      dtrtri(&uplo, &diag, &n, reinterpret_cast<float *>(a), &lda, &info);
    } else {
      std::cerr << "No device selected for ZTRTRI operation." << std::endl;
    }
  }  

  void mklSyclCtrtriFromC(intptr_t *handle, char uplo, char diag, int n, void *a, int lda, int info) {
    //handle not needed
    if (chosenDeviceQueue) {
      //using oneapi::mkl::blas::column_major::gemm;
      auto up = uploFromChar(uplo);
      auto di = diagFromChar(diag);

//      dtrtri(&uplo, &diag, &n, reinterpret_cast<float *>(a), &lda, &info);
    } else {
      std::cerr << "No device selected for CTRTRI operation." << std::endl;
    }
  }  

  // different API!!
  void mklSyclDpotrfFromC(intptr_t *handle, char uplo, int n, void *a, int lda, int info) {
      //handle not needed
      if (chosenDeviceQueue) {
        using oneapi::mkl::lapack::potrf;
        auto up = uploFromChar(uplo);
	//void potrf( ..., int lda, &scratchpad, std::int64_t scratchpad_size)
        //potrf(*chosenDeviceQueue, up, &n, reinterpret_cast<double *>(a), &lda, &info);
      } else {
        std::cerr << "No device selected for DPOTRF operation." << std::endl;
      }
  }  

  void mklSyclSpotrfFromC(intptr_t *handle, char uplo, int n, void *a, int lda, int info) {
      //handle not needed
      if (chosenDeviceQueue) {
        using oneapi::mkl::lapack::potrf;
        auto up = uploFromChar(uplo);
	//void potrf( ..., int lda, &scratchpad, std::int64_t scratchpad_size)
        //potrf(*chosenDeviceQueue, up, &n, reinterpret_cast<float *>(a), &lda, &info);
      } else {
        std::cerr << "No device selected for SPOTRF operation." << std::endl;
      }
  }  

  void mklSyclZpotrfFromC(intptr_t *handle, char uplo, int n, void *a, int lda, int info) {
      //handle not needed
      if (chosenDeviceQueue) {
        using oneapi::mkl::lapack::potrf;
        auto up = uploFromChar(uplo);
	//void potrf( ..., int lda, &scratchpad, std::int64_t scratchpad_size)
        //potrf(*chosenDeviceQueue, up, &n, reinterpret_cast<std::complex<double> *>(c), &lda, &info);
      } else {
        std::cerr << "No device selected for ZPOTRF operation." << std::endl;
      }
  }  

  void mklSyclCpotrfFromC(intptr_t *handle, char uplo, int n, void *a, int lda, int info) {
      //handle not needed
      if (chosenDeviceQueue) {
        using oneapi::mkl::lapack::potrf;
        auto up = uploFromChar(uplo);
	//void potrf( ..., int lda, &scratchpad, std::int64_t scratchpad_size)
        //potrf(*chosenDeviceQueue, up, &n, reinterpret_cast<std::complex<float> *>(c), &lda, &info);
      } else {
        std::cerr << "No device selected for CPOTRF operation." << std::endl;
      }
  }  

  void mklSyclDcopyFromC(intptr_t *handle, int n, void *x, int incx, void *y, int incy) {
      //handle not needed
      if (chosenDeviceQueue) {
	std::int64_t n_, incx_, incy_;
        n_ = (std::int64_t) n;
        incx_ = (std::int64_t) incx;
        incy_ = (std::int64_t) incy;
        using oneapi::mkl::blas::column_major::copy;
        copy(*chosenDeviceQueue, n_, reinterpret_cast<double *>(x), incx_, reinterpret_cast<double *>(y), incy_);
      } else {
        std::cerr << "No device selected for DCOPY operation." << std::endl;
      }
  }  

  void mklSyclScopyFromC(intptr_t *handle, int n, void *x, int incx, void *y, int incy) {
      //handle not needed
      if (chosenDeviceQueue) {
	std::int64_t n_, incx_, incy_;
        n_ = (std::int64_t) n;
        incx_ = (std::int64_t) incx;
        incy_ = (std::int64_t) incy;
        using oneapi::mkl::blas::column_major::copy;
        copy(*chosenDeviceQueue, n_, reinterpret_cast<float *>(x), incx_, reinterpret_cast<float *>(y), incy_);
      } else {
        std::cerr << "No device selected for SCOPY operation." << std::endl;
      }
  }  

  void mklSyclZcopyFromC(intptr_t *handle, int n, void *x, int incx, void *y, int incy) {
      //handle not needed
      if (chosenDeviceQueue) {
	std::int64_t n_, incx_, incy_;
        n_ = (std::int64_t) n;
        incx_ = (std::int64_t) incx;
        incy_ = (std::int64_t) incy;
        using oneapi::mkl::blas::column_major::copy;
        copy(*chosenDeviceQueue, n_, reinterpret_cast<std::complex<double> *>(x), incx_, reinterpret_cast<std::complex<double> *>(y), incy_);
      } else {
        std::cerr << "No device selected for ZCOPY operation." << std::endl;
      }
  }  

  void mklSyclCcopyFromC(intptr_t *handle, int n, void *x, int incx, void *y, int incy) {
      //handle not needed
      if (chosenDeviceQueue) {
	std::int64_t n_, incx_, incy_;
        n_ = (std::int64_t) n;
        incx_ = (std::int64_t) incx;
        incy_ = (std::int64_t) incy;
        using oneapi::mkl::blas::column_major::copy;
        copy(*chosenDeviceQueue, n_, reinterpret_cast<std::complex<float> *>(x), incx_, reinterpret_cast<std::complex<float> *>(y), incy_);
      } else {
        std::cerr << "No device selected for CCOPY operation." << std::endl;
      }
  }  

  void mklSyclDtrmmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int m, int n, double alpha, void *a, int lda, void *b, int ldb) {
      //handle not needed
      if (chosenDeviceQueue) {
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
        trmm(*chosenDeviceQueue, sd, up, ta, di, m_, n_, alpha, reinterpret_cast<double *>(a), lda_, reinterpret_cast<double *>(b), ldb_);
      } else {
        std::cerr << "No device selected for DTRMM operation." << std::endl;
      }
  }  

  void mklSyclStrmmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int m, int n, double alpha, void *a, int lda, void *b, int ldb) {
      //handle not needed
      if (chosenDeviceQueue) {
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
        trmm(*chosenDeviceQueue, sd, up, ta, di, m_, n_, alpha, reinterpret_cast<float *>(a), lda_, reinterpret_cast<float *>(b), ldb_);
      } else {
        std::cerr << "No device selected for STRMM operation." << std::endl;
      }
  }  

  void mklSyclZtrmmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int m, int n, double alpha, void *a, int lda, void *b, int ldb) {
      //handle not needed
      if (chosenDeviceQueue) {
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
        trmm(*chosenDeviceQueue, sd, up, ta, di, m_, n_, alpha, reinterpret_cast<std::complex<double> *>(a), lda_, reinterpret_cast<std::complex<double> *>(b), ldb_);
      } else {
        std::cerr << "No device selected for ZTRMM operation." << std::endl;
      }
  }  

  void mklSyclCtrmmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int m, int n, double alpha, void *a, int lda, void *b, int ldb) {
      //handle not needed
      if (chosenDeviceQueue) {
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
        trmm(*chosenDeviceQueue, sd, up, ta, di, m_, n_, alpha, reinterpret_cast<std::complex<float> *>(a), lda_, reinterpret_cast<std::complex<float> *>(b), ldb_);
      } else {
        std::cerr << "No device selected for CTRMM operation." << std::endl;
      }
  }  

  void mklSyclDtrsmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int m, int n, double alpha, void *a, int lda, void *b, int ldb) {
      //handle not needed
      if (chosenDeviceQueue) {
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
        trsm(*chosenDeviceQueue, sd, up, ta, di, m_, n_, alpha, reinterpret_cast<double *>(a), lda_, reinterpret_cast<double *>(b), ldb_);
      } else {
        std::cerr << "No device selected for DTRSM operation." << std::endl;
      }
  }  

  void mklSyclStrsmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int m, int n, double alpha, void *a, int lda, void *b, int ldb) {
      //handle not needed
      if (chosenDeviceQueue) {
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
        trsm(*chosenDeviceQueue, sd, up, ta, di, m_, n_, alpha, reinterpret_cast<float *>(a), lda_, reinterpret_cast<float *>(b), ldb_);
      } else {
        std::cerr << "No device selected for STRSM operation." << std::endl;
      }
  }  

  void mklSyclZtrsmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int m, int n, double alpha, void *a, int lda, void *b, int ldb) {
      //handle not needed
      if (chosenDeviceQueue) {
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
        trsm(*chosenDeviceQueue, sd, up, ta, di, m_, n_, alpha, reinterpret_cast<std::complex<double> *>(a), lda_, reinterpret_cast<std::complex<double> *>(b), ldb_);
      } else {
        std::cerr << "No device selected for ZTRSM operation." << std::endl;
      }
  }  

  void mklSyclCtrsmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int m, int n, double alpha, void *a, int lda, void *b, int ldb) {
      //handle not needed
      if (chosenDeviceQueue) {
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

        trsm(*chosenDeviceQueue, sd, up, ta, di, m_, n_, alpha, reinterpret_cast<std::complex<float> *>(a), lda_, reinterpret_cast<std::complex<float> *>(b), ldb_);
      } else {
        std::cerr << "No device selected for CTRSM operation." << std::endl;
      }
  }  

  // compile error here; fix this
  //
  void mklSyclDgemvFromC(intptr_t *handle, char cta, int m, int n, double alpha, void *a, int lda, void *x, int incx, double beta, void *y, int incy) {
      //handle not needed
      if (chosenDeviceQueue) {
	std::int64_t m_, n_, lda_, incx_, incy_;
        m_ = (std::int64_t) m;
        n_ = (std::int64_t) n;
        lda_ = (std::int64_t) lda;
        incx_ = (std::int64_t) incx;
        incy_ = (std::int64_t) incy;
	using oneapi::mkl::blas::column_major::gemv;
        auto ta = transposeFromChar(cta);
        gemv(*chosenDeviceQueue, ta, m_, n_, alpha, reinterpret_cast<double *>(a), lda_, reinterpret_cast<double *>(x), incx_, beta, reinterpret_cast<double *>(y), incy_);
      } else {
        std::cerr << "No device selected for DGEMV operation." << std::endl;
      }
  }  

  void mklSyclSgemvFromC(intptr_t *handle, char cta, int m, int n, float alpha, void *a, int lda, void *x, int incx, float beta, void *y, int incy) {
      //handle not needed
      if (chosenDeviceQueue) {
	std::int64_t m_, n_, lda_, incx_, incy_;
        m_ = (std::int64_t) m;
        n_ = (std::int64_t) n;
        lda_ = (std::int64_t) lda;
        incx_ = (std::int64_t) incx;
        incy_ = (std::int64_t) incy;
	using oneapi::mkl::blas::column_major::gemv;
        auto ta = transposeFromChar(cta);
        gemv(*chosenDeviceQueue, ta, m_, n_, alpha, reinterpret_cast<float *>(a), lda_, reinterpret_cast<float *>(x), incx_, beta, reinterpret_cast<float *>(y), incy_);
      } else {
        std::cerr << "No device selected for SGEMV operation." << std::endl;
      }
  }  

  void mklSyclZgemvFromC(intptr_t *handle, char cta, int m, int n, std::complex<double> alpha, void *a, int lda, void *x, int incx, std::complex<double> beta, void *y, int incy) {
      //handle not needed
      if (chosenDeviceQueue) {
	std::int64_t m_, n_, lda_, incx_, incy_;
        m_ = (std::int64_t) m;
        n_ = (std::int64_t) n;
        lda_ = (std::int64_t) lda;
        incx_ = (std::int64_t) incx;
        incy_ = (std::int64_t) incy;
	using oneapi::mkl::blas::column_major::gemv;
        auto ta = transposeFromChar(cta);
        gemv(*chosenDeviceQueue, ta, m_, n_, alpha, reinterpret_cast<std::complex<double> *>(a), lda_, reinterpret_cast<std::complex<double> *>(x), incx_, beta, reinterpret_cast<std::complex<double> *>(y), incy_);
      } else {
        std::cerr << "No device selected for ZGEMV operation." << std::endl;
      }
  }  

  void mklSyclCgemvFromC(intptr_t *handle, char cta, int m, int n, std::complex<float> alpha, void *a, int lda, void *x, int incx, std::complex<float> beta, void *y, int incy) {
      //handle not needed
      if (chosenDeviceQueue) {
	std::int64_t m_, n_, lda_, incx_, incy_;
        m_ = (std::int64_t) m;
        n_ = (std::int64_t) n;
        lda_ = (std::int64_t) lda;
        incx_ = (std::int64_t) incx;
        incy_ = (std::int64_t) incy;
	using oneapi::mkl::blas::column_major::gemv;
        auto ta = transposeFromChar(cta);
        gemv(*chosenDeviceQueue, ta, m_, n_, alpha, reinterpret_cast<std::complex<float> *>(a), lda_, reinterpret_cast<std::complex<float> *>(x), incx_, beta, reinterpret_cast<std::complex<float> *>(y), incy_);
      } else {
        std::cerr << "No device selected for ZGEMV operation." << std::endl;
      }
  }  
} // extern C
#endif /* WITH_SYCL_GPU_VERSION */
