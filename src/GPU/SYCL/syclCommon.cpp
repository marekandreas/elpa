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

using namespace sycl_be;

//--------------------------------------------------------------------------------------------
// SyclState
//--------------------------------------------------------------------------------------------

std::optional<SyclState> SyclState::_staticState;

bool SyclState::initialize(bool onlyL0Gpus, bool isDebugEnabled) {
  if (SyclState::_staticState && onlyL0Gpus != SyclState::_staticState->isManagingOnlyL0Gpus) {
    std::cout << "SyclStaticState already initialized " 
              << ((SyclState::_staticState->isManagingOnlyL0Gpus) ? "with only L0 GPUs" : "all SYCL devices") << "."
              << " You are trying to re-initialize with " << ((onlyL0Gpus) ? "with only L0 GPUs" : "all SYCL devices")
              << " which doesn't match the previous selection. " << std::endl;
    return false;
  } else if (SyclState::_staticState && isDebugEnabled != SyclState::_staticState->isDebugEnabled) {
    std::cout << "SyclStaticState already initialized " 
              << ((SyclState::_staticState->isDebugEnabled) ? "with debug enabled" : "with debug disabled") << "."
              << " You are trying to re-initialize with " << ((isDebugEnabled) ? "with debug enabled" : "with debug disabled")
              << " which doesn't match the previous selection. " << std::endl;
    return false;
  } else if (SyclState::_staticState) {
    return true;     // SyclState already initialized, but with the same parameters.
  } else {
    SyclState::_staticState = std::make_optional(SyclState(onlyL0Gpus, isDebugEnabled));
    return true;
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
SyclState::SyclState(bool onlyL0Gpus, bool isDebugEnabled)
  : isManagingOnlyL0Gpus(onlyL0Gpus), isDebugEnabled(isDebugEnabled), defaultDevice(-1) {
  namespace si = sycl::info;
  auto platforms = sycl::platform::get_platforms();
  if (onlyL0Gpus) {
    for (auto &p : platforms) {
        if (p.get_info<si::platform::name>().find("Level-Zero") != std::string::npos) {
          devices = p.get_devices(sycl::info::device_type::gpu);
          break;
        }
    }    
  } else {
    for (auto &p : platforms) {
      auto platformDevices = p.get_devices(sycl::info::device_type::all);
      devices.insert(devices.end(), platformDevices.begin(), platformDevices.end());
    }    
  }
}

SyclState& SyclState::defaultState() {
  if (_staticState) {
    return *_staticState;
  } else {
    throw std::runtime_error("ELPA GPU SYCL Backend (This is likely a programming error in ELPA.): SyclState not initialized!");
  }
}

size_t SyclState::getNumDevices() {
  return devices.size();
}

void SyclState::printGpuInfo() {
  if (this->isDebugEnabled) {
    auto deviceTypeToString = [](sycl::info::device_type dt) {
      switch (dt) {
        case sycl::info::device_type::cpu: return "CPU";
        case sycl::info::device_type::gpu: return "GPU";
        case sycl::info::device_type::accelerator: return "Accelerator";
        default: return "Other/Unknown/Error";
      }
    };

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
        << deviceTypeToString(devices[i].get_info<sycl::info::device::device_type>()) << ", "
        << devices[i].get_info<sycl::info::device::max_compute_units>() << " EUs"
        << (hasDpSupport ? "" : ", SP only") << ")" << std::endl;
    }
    std::cout << "~~~~~~~~~~~~~~~~ END ELPA SYCL Backend Info ~~~~~~~~~~~~~~~~~~~" << std::endl;
  }
}

DeviceSelection& SyclState::selectGpuDevice(int deviceNum) {
  if (deviceNum < this->devices.size()) {
    this->defaultDevice = deviceNum;
    return this->getDeviceHandle(deviceNum);
  } else {
    throw std::runtime_error("ELPA GPU SYCL Backend (This is likely a programming error in ELPA.): Device number out of range!");
  }
}

DeviceSelection& SyclState::getDeviceHandle(int deviceNum) {
  if (deviceData.find(deviceNum) != deviceData.end()) {
    return this->deviceData.at(deviceNum);
  } else if (deviceNum < this->devices.size() && deviceNum >= 0) {
    this->deviceData.insert({deviceNum, DeviceSelection(deviceNum, this->devices[deviceNum])});
    return this->deviceData.at(deviceNum);
  } else {
    throw std::runtime_error("ELPA GPU SYCL Backend (This is likely a programming error in ELPA.): No GPU device chosen yet. No handle available.");     
  }
}

DeviceSelection& SyclState::getDefaultDeviceHandle() {
  if (this->defaultDevice >= 0) {
    return this->getDeviceHandle(this->defaultDevice);
  } else {
    throw std::runtime_error("ELPA GPU SYCL Backend (This is likely a programming error in ELPA.): No GPU device chosen yet. No default handle available. ");
  }
}

#ifdef WITH_ONEAPI_ONECCL
std::optional<cclKvsHandle> SyclState::retrieveKvs(void *kvsAddress) {
  if (kvsMap.find(kvsAddress) != kvsMap.end()) {
    return kvsMap[kvsAddress];
  }
  return std::nullopt;
}

void SyclState::registerKvs(void *kvsAddr, cclKvsHandle kvs) {
  kvsMap.insert({kvsAddr, kvs});
}

void SyclState::teardownCclStack() {
  for (auto &deviceData : this->deviceData) {
    for (auto &queueData : deviceData.second.queueHandles) {
    }
  }
  kvsMap.clear();
}
#endif

//--------------------------------------------------------------------------------------------
// DeviceSelection
//--------------------------------------------------------------------------------------------

DeviceSelection::DeviceSelection(int deviceId, sycl::device device) 
  : deviceId(deviceId),
    device(device),
#if defined(SYCL_EXT_ONEAPI_DEFAULT_CONTEXT) && SYCL_EXT_ONEAPI_DEFAULT_CONTEXT == 1
    context(device.get_platform().ext_oneapi_get_default_context()),
#else
    context(sycl::queue(device).get_context()),
#endif
#ifdef WITH_ONEAPI_ONECCL
    cclDevice(ccl::create_device(this->device)),
    cclContext(ccl::create_context(this->context)),
#endif
    defaultQueueHandle(device, context) {
  queueHandles.push_back(defaultQueueHandle);
}

QueueData* DeviceSelection::createQueue() {
  queueHandles.emplace_back(device, context);
  return &queueHandles.back();
}

bool DeviceSelection::destroyQueue(QueueData *handle) {
  bool isQueueFound = false;
  for (auto it = queueHandles.begin(); it != queueHandles.end(); it++) {
    if (&(*it) == handle) {
      queueHandles.erase(it);
      isQueueFound = true;
      break;
    }
  }
  return isQueueFound;
}

QueueData* DeviceSelection::getQueue(int id) {
  if (id < queueHandles.size()) {
    return &queueHandles[id];
  } else {
    throw std::runtime_error("ELPA GPU SYCL Backend (This is likely a programming error in ELPA.): Queue ID does not exist.");
  }
}

QueueData* DeviceSelection::getDefaultQueueRef() {
  return &(this->defaultQueueHandle);
}

bool DeviceSelection::isCpuDevice() {
  return (device.get_info<sycl::info::device::device_type>() == sycl::info::device_type::cpu);
}

#ifdef WITH_ONEAPI_ONECCL
ccl::communicator* DeviceSelection::initCclCommunicator(int nRanks, int myRank, cclKvsHandle kvs) {
  this->cclComms.emplace_back(ccl::create_communicator(nRanks, myRank, this->cclDevice, this->cclContext, kvs));
  return &(cclComms.back());
}
#endif

//--------------------------------------------------------------------------------------------
// QueueData
//--------------------------------------------------------------------------------------------

QueueData::QueueData(sycl::device device, sycl::context context) 
  : queue(context, device, sycl::property_list(sycl::property::queue::in_order())),
#ifdef WITH_ONEAPI_ONECCL
    cclStream(ccl::create_stream(queue)),
#endif
    oneMklScratchpadSize(0),
    oneMklScratchpad(nullptr) {}

 QueueData::~QueueData() {
    if (oneMklScratchpad) {
      sycl::free(oneMklScratchpad, queue);
    }
 }

#ifdef WITH_ONEAPI_ONECCL
ccl::stream* QueueData::getCclStreamRef() {
  return &cclStream;
}
#endif




QueueData* sycl_be::getQueueDataOrDefault(QueueData *handle) {
#ifdef WITH_GPU_STREAMS
  return (handle == nullptr) ? &(SyclState::defaultState().getDefaultDeviceHandle().defaultQueueHandle) : handle;
#else 
  return SyclState::defaultState().getDefaultDeviceHandle().getDefaultQueueRef();
#endif
}

sycl::queue sycl_be::getQueueOrDefault(QueueData *qh) {
  return getQueueDataOrDefault(qh)->queue;
}