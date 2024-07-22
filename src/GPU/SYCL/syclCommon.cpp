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

#include "config-f90.h"

#include "syclCommon.hpp"

#include <vector>
#include <optional>
#include <unordered_map>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#ifdef WITH_ONEAPI_ONECCL
#include <oneapi/ccl.hpp>    
#endif



void sycl_be::SyclState::initialize(bool onlyL0Gpus) {
  if (sycl_be::SyclState::_staticState) {
    std::cout << "SyclStaticState already initialized! This will not query devices again." << std::endl;
  } else {
    SyclState::_staticState = std::make_optional(SyclState(onlyL0Gpus));
  }
}

/**
 * @param onlyL0Gpus: ELPA is a library, and hence we cannot assume the user's environment.
 * Therefore, we may need to be opinionated about the device selection.
 * Currently, devices are displayed in duplicate, if they are supported by multiple platforms.
 * For now, the default behavior is to only utilize the Intel Level Zero platform and and GPUs.
 * This will have to be changed later as we move towards generalizing the backend.
 * Alternatively, one can pass an enviromental variable to let ELPA reveal and circle through
 * all available devices. The displayed devices can then be limited through SYCL device filters
 * expressed in SYCL env variables.
 */
sycl_be::SyclState::SyclState(bool onlyL0Gpus) {
  namespace si = sycl::info;
  for (auto const &p : sycl::platform::get_platforms()) {
    if (!onlyL0Gpus || (p.get_info<si::platform::name>().find("Level-Zero") != std::string::npos)) {
      auto deviceType = (onlyL0Gpus) ? si::device_type::gpu : si::device_type::all;
      for (auto dev : p.get_devices(deviceType)) {
          devices.push_back(dev);
      }
    }
  }
}

void sycl_be::SyclState& sycl_be::SyclState::defaultState() {
  if (_staticState) {
    return *_static_state;
  } else {
    throw std::runtime_error("SyclState not initialized!");
  }
}

size_t sycl_be::SyclState::getNumDevices() {
  return devices.size();
}

void sycl_be::SyclState::printGpuInfo() {
#ifdef WITH_MPI
  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  if (mpiRank > 0) {
    return;
  }
#endif

  std::cout << "~~~~~~~~~~~~~~~~~~~ ELPA SYCL Backend Info ~~~~~~~~~~~~~~~~~~~~" << std::endl;
  std::cout << "GPU Backend:       Intel oneAPI SYCL" << std::endl;
  std::cout << "# GPU devices:     " << devices.size() << std::endl;
  std::cout << "Eligible devices: " << std::endl;
  for (size_t i = 0; i < devices.size(); i++) {
    bool hasDpSupport = devices[i].has(sycl::aspect::fp64);
    std::cout << " - Device #" << i << ": "
      << devices[i].get_platform().get_info<sycl::info::platform::name>() << " -> "
      << devices[i].get_info<sycl::info::device::name>() << " ("
      << devices[i].get_info<sycl::info::device::max_compute_units>() << " EUs"
      << (hasDpSupport ? "" : ", SP only") << ")" << std::endl;
  }
  std::cout << "~~~~~~~~~~~~~~~~ END ELPA SYCL Backend Info ~~~~~~~~~~~~~~~~~~~" << std::endl;
}

sycl_be::DeviceSelection& sycl_be::SyclState::selectGpuDevice(int deviceNum) {

}

sycl_be::DeviceSelection& sycl_be::SyclState::getDeviceHandle(deviceNum) {

}

sycl_be::DeviceSelection& sycl_be::SyclState::getDefaultDeviceHandle() {
  
}


#ifdef WITH_ONEAPI_ONECCL
ccl::device& egs::getCclDevice() {
  if (!chosenQueue) {
    egs::selectDefaultGpuDevice();
  }
  return chosenQueue->cclDevice;
}

ccl::context& egs::getCclContext() {
  if (!chosenQueue) {
    egs::selectDefaultGpuDevice();
  }
  return chosenQueue->cclContext;
}

ccl::stream& egs::getCclStream() {
  if (!chosenQueue) {
    egs::selectDefaultGpuDevice();
  }
  return chosenQueue->cclStream;
}

ccl::stream* egs::getCclStreamRef() {
  if (!chosenQueue) {
    egs::selectDefaultGpuDevice();
  }
  return &(chosenQueue->cclStream);
}



std::optional<egs::cclKvsHandle> egs::retrieveKvs(void *kvsAddress) {
  if (kvsMap.find(kvsAddress) != kvsMap.end()) {
    return kvsMap[kvsAddress];
  }
  return std::nullopt;
}

void egs::registerKvs(void *kvsAddr, egs::cclKvsHandle kvs) {
  kvsMap.insert({kvsAddr, kvs});
}
#endif
