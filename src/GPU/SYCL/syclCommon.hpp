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
  sycl::queue queue;
  size_t oneMklScratchpadSize;
  void *oneMklScratchpad;

#ifdef WITH_ONEAPI_ONECCL
  ccl::device cclDevice;
  ccl::context cclContext;
  ccl::stream cclStream;
#endif  

  QueueData(sycl::device device);
  ~QueueData();

  void increaseScratchpadSize(size_t newSize);

#ifdef WITH_ONEAPI_ONECCL
  ccl::stream* getCclStreamRef();
#endif
};

struct DeviceSelection {
  int deviceId;
  sycl::device device;
  std::vector<QueueData> queueHandles;
  
#ifdef WITH_ONEAPI_ONECCL
  using cclKvsHandle = ccl::shared_ptr_class<ccl::kvs>;
  std::unordered_map<void *, egs::cclKvsHandle> kvsMap;
#endif

  DeviceSelection(int deviceId, sycl::device device);
  QueueData* createQueue();
  QueueData* getQueue(int id);
#ifdef WITH_ONEAPI_ONECCL
  void registerKvs(void *kvsAddr, cclKvsHandle kvs);
  std::optional<cclKvsHandle> retrieveKvs(void *kvsAddress);
#endif
};


class SyclState {
  static std::optional<SyclState> _staticState;

  std::vector<sycl::device> devices;
  
  SyclState(bool onlyL0Gpus = false);
  
  public:
  
  void printGpuInfo();
  int selectGpuDevice(int deviceNum);
  size_t getNumDevices();
  DeviceSelection *getDeviceHandle();


  static SyclState& defaultState();
  static void initialize(bool onlyL0Gpus = false);

};

/*
  void collectGpuDevices(bool onlyGpus);
  void collectCpuDevices();
  void printGpuInfo();
  int selectGpuDevice(int deviceNum);
  int selectCpuDevice(int deviceNum);
  void selectDefaultGpuDevice();
  size_t getNumDevices();
  size_t getNumCpuDevices();
  sycl::device getDevice();
  sycl::queue getQueue();
  sycl::queue* getQueueRef();
*/
  
}
#endif
