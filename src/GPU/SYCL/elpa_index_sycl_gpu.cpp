// Copyright 2021, A. Marek MPCDF
//
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
//    along with ELPA. If not, see <http://www.gnu.org/licenses/>
//
//    ELPA reflects a substantial effort on the part of the original
//    ELPA consortium, and we ask you to respect the spirit of the
//    license that we chose: i.e., please contribute any changes you
//    may have back to the original ELPA library distribution, and keep
//    any derivatives of ELPA under the same license that we chose for
//    the original distribution, the GNU Lesser General Public License.
//
#include <CL/sycl.hpp>


static std::vector<sycl::device> devices_index;
static std::optional<sycl::queue> chosenDeviceQueue_index{};

bool deviceCollectionFlag_index = false;

extern "C" {

  static void collectGpuDevices_index() {
    if (deviceCollectionFlag_index) {
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
          devices_index.push_back(dev);
        }
      }
    }
  }

  int sycl_gpu_count() {
    int count = -1000;
    collectGpuDevices_index();
    count = devices_index.size();
    return count;
  }
}

