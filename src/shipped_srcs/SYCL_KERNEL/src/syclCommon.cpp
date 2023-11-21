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

#ifdef WITH_MPI
#include <mpi.h>
#endif


static bool deviceCollectionFlag = false;

std::vector<cl::sycl::device> devices;
std::optional<cl::sycl::queue> chosenQueue;

void elpa::gpu::sycl::collectGpuDevices(bool onlyGpus) {
  if (deviceCollectionFlag) {
    return;
  } else {
    deviceCollectionFlag = true;
  }

  // ELPA is a library, and hence we cannot assume the user's environment.
  // Therefore, we may need to be opinionated about the device selection.
  // Currently, devices are displayed in duplicate, if they are supported by multiple platforms.
  // For now, the default behavior is to only utilize the Intel Level Zero platform and and GPUs.
  // This will have to be changed later as we move towards generalizing the backend.
  // Alternatively, one can pass an enviromental variable to let ELPA reveal and circle through
  // all available devices. The displayed devices can then be limited through SYCL device filters
  // expressed in SYCL env variables.
  namespace si = cl::sycl::info;
  for (auto const &p : cl::sycl::platform::get_platforms()) {
    if (!onlyGpus || (p.get_info<si::platform::name>().find("Level-Zero") != std::string::npos)) {
      for (auto dev : p.get_devices((onlyGpus) ? si::device_type::gpu : si::device_type::all)) {
          devices.push_back(dev);
      }
    }
  }
}

void elpa::gpu::sycl::collectCpuDevices() {
  if (deviceCollectionFlag) {
    return;
  } else {
    deviceCollectionFlag = true;
  }

  std::cout << "DO NOT CALL THIS!!!!!!!!!!!!!" << std::endl;
  std::cout << "DO NOT CALL THIS!!!!!!!!!!!!!" << std::endl;
  std::cout << "DO NOT CALL THIS!!!!!!!!!!!!!" << std::endl;
  std::cout << "DO NOT CALL THIS!!!!!!!!!!!!!" << std::endl;
  std::cout << "DO NOT CALL THIS!!!!!!!!!!!!!" << std::endl;
  std::cout << "DO NOT CALL THIS!!!!!!!!!!!!!" << std::endl;
  std::cout << "DO NOT CALL THIS!!!!!!!!!!!!!" << std::endl;
  std::cout << "DO NOT CALL THIS!!!!!!!!!!!!!" << std::endl;

  // We need to be opinionated about the device selection. Currently, devices are displayed in duplicate, if they are supported
  // by multiple platforms. For now, a first step could be only supporting level zero and Intel GPUs. This will have to be
  // changed later as we move towards generalizing the backend.
  for (auto const &p : cl::sycl::platform::get_platforms()) {
    for (auto dev : p.get_devices()) {
      devices.push_back(dev);
    }
  }
}

int elpa::gpu::sycl::selectGpuDevice(int deviceId) {
  if (deviceId >= devices.size()){
    std::cerr << "Invalid device ID selected, only " << devices.size() << " devices available." << std::endl;
    return 0;
  }
  cl::sycl::property::queue::in_order io;
  cl::sycl::property_list props(io);
  chosenQueue = std::make_optional<cl::sycl::queue>(devices[deviceId], props);
  auto platform = chosenQueue->get_device().get_platform().get_info<cl::sycl::info::platform::name>();
  auto deviceName = chosenQueue->get_device().get_info<cl::sycl::info::device::name>();
  std::cout << "Selected device: (" << platform << ") " << deviceName << std::endl;
  return 1;
}

int elpa::gpu::sycl::selectCpuDevice(int deviceId) {
  collectCpuDevices();
  elpa::gpu::sycl::selectGpuDevice(deviceId);
  return 1;
}

void elpa::gpu::sycl::selectDefaultGpuDevice() {
  auto gpuSelector = cl::sycl::gpu_selector_v;
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
  return devices.size();
}

size_t elpa::gpu::sycl::getNumCpuDevices() {
  collectCpuDevices();
  return devices.size();
}

void elpa::gpu::sycl::printGpuInfo() {
#ifdef WITH_MPI
  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  if (mpiRank > 0) {
    return;
  }
#endif

  std::cout << "~~~~~~~~~~~~~~~~~~~ ELPA GPU Info ~~~~~~~~~~~~~~~~~~~~" << std::endl;
  std::cout << "GPU Backend:       Intel oneAPI SYCL" << std::endl;
  std::cout << "# GPU devices:     " << devices.size() << std::endl;
  std::cout << "Eligible devices: " << std::endl;
  for (size_t i = 0; i < devices.size(); i++) {
    bool hasDpSupport = devices[i].has(cl::sycl::aspect::fp64);
    std::cout << " - Device #" << i << ": "
      << devices[i].get_platform().get_info<cl::sycl::info::platform::name>() << " -> "
      << devices[i].get_info<cl::sycl::info::device::name>() << " ("
      << devices[i].get_info<cl::sycl::info::device::max_compute_units>() << " EUs"
      << (hasDpSupport ? "" : ", SP only") << ")" << std::endl;
  }
  std::cout << "~~~~~~~~~~~~~~~~ END ELPA GPU Info ~~~~~~~~~~~~~~~~~~~" << std::endl;
}
