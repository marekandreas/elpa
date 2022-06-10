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

#include "syclCommon.hpp"

#include <vector>
#include <optional>

static bool deviceCollectionFlag = false;

std::vector<cl::sycl::device> devices;
std::optional<cl::sycl::queue> chosenQueue;

void elpa::gpu::sycl::collectGpuDevices() {
  if (deviceCollectionFlag) {
    return;
  } else {
    deviceCollectionFlag = true;
  }

  // We need to be opinionated about the device selection. Currently, devices are displayed in duplicate, if they are supported
  // by multiple platforms. For now, a first step could be only supporting level zero and Intel GPUs. This will have to be
  // changed later as we move towards generalizing the backend.
  for (auto const &p : cl::sycl::platform::get_platforms()) {
    if (p.get_info<cl::sycl::info::platform::name>().find("Level-Zero") != std::string::npos) {
      for (auto dev : p.get_devices()) {
        if (dev.get_info<cl::sycl::info::device::device_type>() == cl::sycl::info::device_type::gpu) {
          devices.push_back(dev);
        }
      }
    }
  }
}

int elpa::gpu::sycl::selectGpuDevice(int deviceId) {
  collectGpuDevices();
  if (deviceId >= devices.size()){
    std::cerr << "Invalid device ID selected, only " << devices.size() << " devices available." << std::endl;
    return 0;
  }
  cl::sycl::property::queue::in_order io;
  cl::sycl::property_list props(io);
  chosenQueue = std::make_optional<cl::sycl::queue>(devices[deviceId], props);
  return 1;
}

void elpa::gpu::sycl::selectDefaultGpuDevice() {
  cl::sycl::gpu_selector gpuSelector;
  cl::sycl::property::queue::in_order io;
  cl::sycl::property_list props(io);
  chosenQueue = std::make_optional<cl::sycl::queue>(gpuSelector, props);
}

cl::sycl::queue & elpa::gpu::sycl::getQueue() {
  if (!chosenQueue) {
    elpa::gpu::sycl::selectDefaultGpuDevice();
  }
  return *chosenQueue;
}

cl::sycl::device elpa::gpu::sycl::getDevice() {
  if (!chosenQueue) {
    elpa::gpu::sycl::selectDefaultGpuDevice();
  }
  return chosenQueue->get_device();
}

size_t elpa::gpu::sycl::getNumDevices() {
  collectGpuDevices();
  return devices.size();
}

void elpa::gpu::sycl::printGpuInfo() {
  collectGpuDevices();
  std::cout << "~~~~~~~~~~~~~~~~~~~ ELPA GPU Info ~~~~~~~~~~~~~~~~~~~~" << std::endl;
  std::cout << "GPU Backend:       Intel oneAPI SYCL" << std::endl;
  std::cout << "# GPU devices:     " << devices.size() << std::endl;
  std::cout << "Eligible devices: " << std::endl;
  for (size_t i = 0; i < devices.size(); i++) {
    bool hasDpSupport = devices[i].has(cl::sycl::aspect::fp64);
    std::cout << " - Device #" << i << ": "
      << devices[i].get_info<cl::sycl::info::device::name>() << " ("
      << devices[i].get_info<cl::sycl::info::device::max_compute_units>() << " EUs"
      << (hasDpSupport ? "" : ", SP only") << ")" << std::endl;
  }
  std::cout << "~~~~~~~~~ Display Verbose SYCL Platform Info ~~~~~~~~~" << std::endl;
  std::cout << std::endl;
  for (auto const &platform : cl::sycl::platform::get_platforms()) {
    std::cout << " - Platform: " << platform.get_info<cl::sycl::info::platform::name>() << std::endl;
    std::cout << "   * Vendor:  " << platform.get_info<cl::sycl::info::platform::vendor>() << std::endl;
    std::cout << "   * Version: " << platform.get_info<cl::sycl::info::platform::version>() << std::endl;
    std::cout << "   * Profile: " << platform.get_info<cl::sycl::info::platform::vendor>() << std::endl;
    for (auto const &device : platform.get_devices()) {
      std::cout << "    -- Device: " << device.get_info<cl::sycl::info::device::name>() << std::endl;
      std::cout << "       * Device Type:                             ";
      auto deviceType = device.get_info<cl::sycl::info::device::device_type>();
      switch (deviceType) {
        case cl::sycl::info::device_type::cpu:
          std::cout << "CPU" << std::endl;
          break;
        case cl::sycl::info::device_type::gpu:
          std::cout << "GPU" << std::endl;
          break;
        case cl::sycl::info::device_type::accelerator:
          std::cout << "Accelerator" << std::endl;
          break;
        case cl::sycl::info::device_type::custom:
          std::cout << "CUSTOM" << std::endl;
          break;
        case cl::sycl::info::device_type::automatic:
          std::cout << "AUTOMATIC" << std::endl;
          break;
        case cl::sycl::info::device_type::host:
          std::cout << "HOST" << std::endl;
          break;
        case cl::sycl::info::device_type::all:
        default:
          std::cout << "UNKNOWN" << std::endl;
          break;
      }
      std::cout << "       * Max Compute Units:                       " << device.get_info<cl::sycl::info::device::max_compute_units>() << std::endl;
      std::cout << "       * Double Precision Floating Point support: " << ((device.has(cl::sycl::aspect::fp64)) ? "Yes" : "No") << std::endl;
      std::cout << "       * Max Work Item Dimensions:                " << device.get_info<cl::sycl::info::device::max_work_item_dimensions>() << std::endl;
      auto maxWorkItemSize = device.get_info<cl::sycl::info::device::max_work_item_sizes>();
      std::cout << "       * Max Work Item Sizes:                     " << "{" << maxWorkItemSize[0] << ", " << maxWorkItemSize[1] << ", " << maxWorkItemSize[2] << "}" << std::endl;
      std::cout << "       * Max Work Group Sizes:                    " << device.get_info<cl::sycl::info::device::max_work_group_size>() << std::endl;
      std::cout << "       * Max Memory Alloc size:                   " << device.get_info<cl::sycl::info::device::max_mem_alloc_size>() << std::endl;
      std::cout << "       * Max Parameter size:                      " << device.get_info<cl::sycl::info::device::max_parameter_size>() << std::endl;
      std::cout << "       * Global Mem Cache Type:                   ";
      auto globalMemoryCacheType = device.get_info<cl::sycl::info::device::global_mem_cache_type>();
      switch (globalMemoryCacheType) {
        case cl::sycl::info::global_mem_cache_type::none:
          std::cout << "None" << std::endl;
          break;
        case cl::sycl::info::global_mem_cache_type::read_only:
          std::cout << "Read-Only";
          break;
        case cl::sycl::info::global_mem_cache_type::read_write:
          std::cout << "Read-Write";
          break;
        default:
          std::cout << "UNKNOWN ENTRY!";
          break;
      }
      std::cout << std::endl;
      std::cout << "       * Local Mem Type:                          ";
      auto localMemType = device.get_info<cl::sycl::info::device::local_mem_type>();
      switch (localMemType) {
        case cl::sycl::info::local_mem_type::none:
          std::cout << "None";
          break;
        case cl::sycl::info::local_mem_type::local:
          std::cout << "Local";
          break;
        case cl::sycl::info::local_mem_type::global:
          std::cout << "Global";
          break;
        default:
          std::cout << "UNKNOWN ENTRY!";
          break;
      }
      std::cout << std::endl;
      std::cout << "       * Local Mem Size:                          " << device.get_info<cl::sycl::info::device::local_mem_size>() << std::endl;
      std::cout << "       * Host Unified Memory:                     " << device.get_info<cl::sycl::info::device::host_unified_memory>() << std::endl;
    }
  }
}
