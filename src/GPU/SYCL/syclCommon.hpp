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

#include <CL/sycl.hpp>

#ifdef WITH_ONEAPI_ONECCL
#include <oneapi/ccl.hpp>
#endif

namespace elpa {
namespace gpu {
namespace sycl {

  void collectGpuDevices(bool onlyGpus);
  void collectCpuDevices();
  void printGpuInfo();
  int selectGpuDevice(int deviceNum);
  int selectCpuDevice(int deviceNum);
  void selectDefaultGpuDevice();
  size_t getNumDevices();
  size_t getNumCpuDevices();
  cl::sycl::device getDevice();
  cl::sycl::queue getQueue();
  cl::sycl::queue* getQueueRef();

  

#ifdef WITH_ONEAPI_ONECCL
  // oneCCL deals with objects rather than opaque handles. Thus, ELPA becomes responsible for keeping them.
  // To keep the interface uniform, ELPA mostly deals with pointers. Where possible, I use them, but in some
  // cases, memoizing the values is necessary.

  using cclKvsHandle = ccl::shared_ptr_class<ccl::kvs>;
  void registerKvs(void *kvsAddr, cclKvsHandle kvs);
  std::optional<cclKvsHandle> retrieveKvs(void *kvsAddress);

  ccl::device& getCclDevice();
  ccl::context& getCclContext();
  ccl::stream& getCclStream();
  ccl::stream* getCclStreamRef();
#endif

}
}
}
#endif
