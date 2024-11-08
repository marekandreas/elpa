//    This file is part of ELPA.
//
//    The ELPA library was originally created by the ELPA consortium,
//    consisting of the following organizations:
//
//    - Max Planck Computing and Data Facility (MPCDF), formerly known as
//      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
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
//    This particular source code file contains additions, changes and
//    enhancements authored by Intel Corporation which is not part of
//    the ELPA consortium.
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
// This file was written by A. Poeppl, Intel Corporation (2022) for MPCDF

#ifndef SYCL_COMMON_HPP
#define SYCL_COMMON_HPP

#pragma once

#include <sycl/sycl.hpp>

#ifdef WITH_ONEAPI_ONECCL
#include <oneapi/ccl.hpp>
#endif

namespace sycl_be {

struct QueueData {
  friend class DeviceSelection;
  private:
  void *oneMklScratchpad; 
  
  public:
  sycl::queue queue;
#ifdef WITH_ONEAPI_ONECCL
  ccl::stream cclStream;
#endif  
  size_t oneMklScratchpadSize;


  QueueData(sycl::device device, sycl::context context);
  ~QueueData();

  template <typename T> inline T* getScratchpadFor(size_t numElements);
  
#ifdef WITH_ONEAPI_ONECCL
  ccl::stream* getCclStreamRef();
#endif
};


struct DeviceSelection {
  int deviceId;
  sycl::device device;
  sycl::context context;
#ifdef WITH_ONEAPI_ONECCL
  ccl::device cclDevice;
#endif  
  QueueData defaultQueueHandle;
  std::vector<QueueData> queueHandles;

  DeviceSelection(int deviceId, sycl::device device);
  QueueData* createQueue();
  bool destroyQueue(QueueData *handle);
  QueueData* getQueue(int id);
  QueueData* getDefaultQueueRef();
};


class SyclState {
  static std::optional<SyclState> _staticState;

  bool isManagingOnlyL0Gpus;
  public:
  bool isDebugEnabled;
  private:
  std::vector<sycl::device> devices;
  std::unordered_map<int, DeviceSelection> deviceData;
  int defaultDevice;

#ifdef WITH_ONEAPI_ONECCL
  using cclKvsHandle = ccl::shared_ptr_class<ccl::kvs>;
  ccl::context cclContext;
  std::unordered_map<void *, egs::cclKvsHandle> kvsMap;
#endif
  
  SyclState(bool onlyL0Gpus = false, bool isDebugEnabled = false);
  DeviceSelection& getDeviceHandle(int deviceNum);
  
  public:
  
  void printGpuInfo();
  DeviceSelection& selectGpuDevice(int deviceNum);
  DeviceSelection& getDefaultDeviceHandle();
  
  size_t getNumDevices();

  static SyclState& defaultState();
  static bool initialize(bool onlyL0Gpus = false, bool isDebugEnabled = false);

#ifdef WITH_ONEAPI_ONECCL
  void registerKvs(void *kvsAddr, cclKvsHandle kvs);
  std::optional<cclKvsHandle> retrieveKvs(void *kvsAddress);
#endif
};
  
  sycl::queue getQueueOrDefault(QueueData *my_stream);
  QueueData* getQueueDataOrDefault(QueueData *my_stream);
  template<int numDims> sycl::range<numDims> maxWorkgroupSize(sycl::queue d);
}

template <typename T>
inline T* sycl_be::QueueData::getScratchpadFor(size_t numElements) {
  if (numElements > oneMklScratchpadSize * sizeof(T)) {
    if (oneMklScratchpad != nullptr) {
      sycl::free(oneMklScratchpad, queue);
    }
    oneMklScratchpad = sycl::malloc_device(numElements * sizeof(T), this->queue);
    oneMklScratchpadSize = numElements * sizeof(T);
  }
  return static_cast<T*>(oneMklScratchpad);
}

template <int numDims>
inline sycl::range<numDims> sycl_be::maxWorkgroupSize(sycl::queue q) {
  static_assert(numDims > 0 && numDims <= 3, "numDims must be either 1, 2, or 3.");
  sycl::device d = q.get_device();
  int maxWgSize = d.get_info<sycl::info::device::max_work_group_size>();
  if constexpr (numDims == 1) {
    return sycl::range<1>(maxWgSize);
  } else if constexpr (numDims == 2) {
    int res[2] {128, 128};
    int dim = 0;
    while (res[0] * res[1] > maxWgSize) {
      res[dim] /= 2;
      dim = (dim + 1) % 2;
    }
    return sycl::range<2>(res[0], res[1]);
  } else {
    int res[3] {32, 32, 32};
    int dim = 0;
    while (res[0] * res[1] * res[2] > maxWgSize) {
      res[dim] /= 2;
      dim = (dim + 1) % 3;
    }
    return sycl::range<3>(res[0], res[1], res[2]);
  }
}
#endif
