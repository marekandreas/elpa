//    Copyright 2024, A. Marek
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
//    This file was written by A. Poeppl, Intel Corporation

#include "src/GPU/SYCL/syclCommon.hpp"

#include <sycl/sycl.hpp>
#include <math.h>
#include <stdlib.h>
#include <alloca.h>
#include <stdint.h>
#include <complex>

#include "config-f90.h"

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

using namespace sycl_be;

template<typename T>
void sycl_scale_qmat_complex(int *ldq_in, int *l_cols_in, std::complex<T> *q_dev, std::complex<T> *tau_dev, QueueData *my_stream) {
  int ldq = *ldq_in;
  int l_cols = *l_cols_in;

  sycl::queue q = getQueueOrDefault(my_stream);
  sycl::range<1> threadsPerBlock = maxWorkgroupSize<1>(q);
  sycl::range<1> blocks((l_cols + threadsPerBlock - 1) / threadsPerBlock);
  
  q.parallel_for(sycl::nd_range<1>(blocks * threadsPerBlock, threadsPerBlock), [=](sycl::nd_item<1> it) {
      T one(1.0);
      T zero(0.0);
      std::complex<T> c_one(one, zero);
      int col = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);
      int index = ldq * col;

      if (col < l_cols) {
        q_dev[index] *= (c_one - tau_dev[1]);
      }
    });
}

extern "C" void sycl_scale_qmat_double_complex_FromC(int *ldq_in, int *l_cols_in, std::complex<double> *q_dev, std::complex<double> *tau_dev, QueueData *my_stream) {
  sycl_scale_qmat_complex<double>(ldq_in, l_cols_in, q_dev, tau_dev, my_stream);
}

extern "C" void sycl_scale_qmat_float_complex_FromC(int *ldq_in, int *l_cols_in, std::complex<float> *q_dev, std::complex<float> *tau_dev, QueueData *my_stream) {
  sycl_scale_qmat_complex<float>(ldq_in, l_cols_in, q_dev, tau_dev, my_stream);
}
