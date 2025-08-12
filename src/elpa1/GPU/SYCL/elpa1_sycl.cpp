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
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <alloca.h>
#include <ccomplex>
#include <cstdint>
#include <complex>


#include "config-f90.h"

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

using namespace sycl_be;

template<typename T>
void sycl_copy_real_part_to_q_complex(std::complex<T> *q_dev, T *q_real_dev, int *matrixRows_in, int *l_rows_in, int *l_cols_nev_in, QueueData *my_stream) {
  int l_rows = *l_rows_in;
  int l_cols_nev = *l_cols_nev_in;
  int matrixRows = *matrixRows_in;

  sycl::queue q = getQueueOrDefault(my_stream);
  sycl::range<2> threadsPerBlock = maxWorkgroupSize<2>(q);
  sycl::range<2> blocks((l_cols_nev + threadsPerBlock[0] - 1) / threadsPerBlock[0],
                        (l_rows     + threadsPerBlock[1] - 1) / threadsPerBlock[1]);

  q.parallel_for(sycl::nd_range<2>(blocks * threadsPerBlock, threadsPerBlock), [=](sycl::nd_item<2> it) {
    int row = it.get_group(1) * it.get_local_range(1) + it.get_local_id(1);
    int col = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);

    if (row < l_rows && col < l_cols_nev) {
      int index = row + matrixRows * col;
      q_dev[index] = std::complex<T>(q_real_dev[index], 0.0);
    }
  });

  q.wait_and_throw();
}

extern "C" void sycl_copy_real_part_to_q_double_complex_FromC(std::complex<double> *q_dev, double *q_real_dev, int *matrixRows_in, int *l_rows_in, int *l_cols_nev_in, QueueData *my_stream) {
  sycl_copy_real_part_to_q_complex<double>(q_dev, q_real_dev, matrixRows_in, l_rows_in, l_cols_nev_in, my_stream);
}

extern "C" void sycl_copy_real_part_to_q_float_complex_FromC(std::complex<float> *q_dev, float *q_real_dev, int *matrixRows_in, int *l_rows_in, int *l_cols_nev_in, QueueData *my_stream) {
  sycl_copy_real_part_to_q_complex<float>(q_dev, q_real_dev, matrixRows_in, l_rows_in, l_cols_nev_in, my_stream);
}


template<typename T> 
void sycl_zero_skewsymmetric_q(T *q_dev, int *matrixRows_in, int *matrixCols_in, QueueData *my_stream) {
  int matrixCols = *matrixCols_in;
  int matrixRows = *matrixRows_in;

  sycl::queue q = getQueueOrDefault(my_stream);
  sycl::range<2> threadsPerBlock = maxWorkgroupSize<2>(q);
  sycl::range<2> blocks((matrixCols + threadsPerBlock[0] - 1) / threadsPerBlock[0],
                        (matrixRows + threadsPerBlock[1] - 1) / threadsPerBlock[1]);

  q.parallel_for(sycl::nd_range<2>(blocks * threadsPerBlock, threadsPerBlock), [=](sycl::nd_item<2> it) {
    int row = it.get_group(1) * it.get_local_range(1) + it.get_local_id(1);
    int col = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);

    if (row < matrixRows && col >= matrixCols && col < 2 * matrixCols) {
        int index = row + (matrixRows) * col;
        q_dev[index] = 0.0;
    }
  });
  
  q.wait_and_throw();
}

extern "C" void sycl_zero_skewsymmetric_q_double_FromC(double *q_dev, int *matrixRows_in, int *matrixCols_in, QueueData *my_stream) {
  sycl_zero_skewsymmetric_q<double>(q_dev, matrixRows_in, matrixCols_in, my_stream);
}

extern "C" void sycl_zero_skewsymmetric_q_float_FromC(float *q_dev, int *matrixRows_in, int *matrixCols_in, QueueData *my_stream) {
  sycl_zero_skewsymmetric_q<float>(q_dev, matrixRows_in, matrixCols_in, my_stream);
}

template<typename T, bool isPlus>
void sycl_copy_skewsymmetric_second_half_q_kernel(T *q_dev, const int i, const int matrixRows, const int matrixCols, const sycl::nd_item<1> &it) {
    int col = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);

    if (col >= matrixCols && col < 2 * matrixCols) {  // geht nivht
        int    index = (i-1) + matrixRows * (col);
        int indexLow = (i-1) + matrixRows * (col-matrixCols);
        if constexpr (isPlus) {
            q_dev[index] = q_dev[indexLow];
        } else {
            q_dev[index] = -q_dev[indexLow];
        }
        q_dev[indexLow] = 0.0;
    }	
}

template<typename T> void sycl_copy_skewsymmetric_second_half_q(T *q_dev, int *i_in, int *matrixRows_in, int *matrixCols_in, int *negative_or_positive_in, QueueData *my_stream) {
  int matrixCols = *matrixCols_in;
  int matrixRows = *matrixRows_in;
  int negative_or_positive = *negative_or_positive_in;
  int i = *i_in;

  sycl::queue q = getQueueOrDefault(my_stream);
  sycl::range<1> threadsPerBlock = maxWorkgroupSize<1>(q);
  sycl::range<1> blocks((2 * matrixCols + threadsPerBlock[0] - 1) / threadsPerBlock[0]);

  sycl::nd_range<1> r(blocks * threadsPerBlock, threadsPerBlock);
  if (negative_or_positive == 1) {
    q.parallel_for(r, [=](sycl::nd_item<1> it) {
      sycl_copy_skewsymmetric_second_half_q_kernel<T, true>(q_dev, i, matrixRows, matrixCols, it);
    });
  } else {
    q.parallel_for(r, [=](sycl::nd_item<1> it) {
      sycl_copy_skewsymmetric_second_half_q_kernel<T, false>(q_dev, i, matrixRows, matrixCols, it);
    });
  }
  
  q.wait_and_throw();
}

extern "C" void sycl_copy_skewsymmetric_second_half_q_double_FromC(double *q_dev, int *i_in, int *matrixRows_in, int *matrixCols_in, int *negative_or_positive_in, QueueData *my_stream) {
  sycl_copy_skewsymmetric_second_half_q<double>(q_dev, i_in, matrixRows_in, matrixCols_in, negative_or_positive_in, my_stream);
}
extern "C" void sycl_copy_skewsymmetric_second_half_q_float_FromC(float *q_dev, int *i_in, int *matrixRows_in, int *matrixCols_in, int *negative_or_positive_in, QueueData *my_stream) {
  sycl_copy_skewsymmetric_second_half_q<float>(q_dev, i_in, matrixRows_in, matrixCols_in, negative_or_positive_in, my_stream);
}


template<typename T>
void sycl_copy_skewsymmetric_first_half_q_FromC(T *q_dev, int *i_in, int *matrixRows_in, int *matrixCols_in, int *negative_or_positive_in, QueueData *my_stream) {
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;
  int negative_or_positive = *negative_or_positive_in;
  int i = *i_in;

  sycl::queue q = getQueueOrDefault(my_stream);
  sycl::range<1> threadsPerBlock = maxWorkgroupSize<1>(q);
  sycl::range<1> blocks((matrixCols + threadsPerBlock[0] - 1) / threadsPerBlock[0]);

  q.parallel_for(sycl::nd_range<1>(blocks * threadsPerBlock, threadsPerBlock), [=](sycl::nd_item<1> it) {
    int col = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);
    if (col < matrixCols) {
      int index = (i-1) + matrixRows * col;
      q_dev[index] = -q_dev[index];
    }
  });
  
  q.wait_and_throw();
}

extern "C" void sycl_copy_skewsymmetric_first_half_q_double_FromC(double *q_dev, int *i_in, int *matrixRows_in, int *matrixCols_in, int *negative_or_positive_in, QueueData *my_stream) {
  sycl_copy_skewsymmetric_first_half_q_FromC<double>(q_dev, i_in, matrixRows_in, matrixCols_in, negative_or_positive_in, my_stream);
}

extern "C" void sycl_copy_skewsymmetric_first_half_q_float_FromC(float *q_dev, int *i_in, int *matrixRows_in, int *matrixCols_in, int *negative_or_positive_in, QueueData *my_stream) {
  sycl_copy_skewsymmetric_first_half_q_FromC<float>(q_dev, i_in, matrixRows_in, matrixCols_in, negative_or_positive_in, my_stream);
}


template<typename T> void sycl_get_skewsymmetric_second_half_q_FromC(T *q_dev, T *q_2_dev, int *matrixRows_in, int *matrixCols_in, QueueData *my_stream) {
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;

  sycl::queue q = getQueueOrDefault(my_stream);
  sycl::range<2> threadsPerBlock = maxWorkgroupSize<2>(q);
  sycl::range<2> blocks((matrixCols + threadsPerBlock[0] - 1) / threadsPerBlock[0], (matrixRows + threadsPerBlock[1] - 1) / threadsPerBlock[1]);

  q.parallel_for(sycl::nd_range<2>(blocks * threadsPerBlock, threadsPerBlock), [=](sycl::nd_item<2> item_ct1) {
    int row = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    int col = item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0);

    if (row < matrixRows && col < matrixCols) {
      int index  = row + matrixRows * col;
      int index2 = row + matrixRows * (col + matrixCols);
      q_2_dev[index] = q_dev[index2];
    }
  });

  q.wait_and_throw();
}

extern "C" void sycl_get_skewsymmetric_second_half_q_double_FromC(double *q_dev, double *q_2_dev, int *matrixRows_in, int *matrixCols_in, QueueData *my_stream) {
  sycl_get_skewsymmetric_second_half_q_FromC<double>(q_dev, q_2_dev, matrixRows_in, matrixCols_in, my_stream);
}

extern "C" void sycl_get_skewsymmetric_second_half_q_float_FromC(float *q_dev, float *q_2_dev, int *matrixRows_in, int *matrixCols_in, QueueData *my_stream) {
  sycl_get_skewsymmetric_second_half_q_FromC<float>(q_dev, q_2_dev, matrixRows_in, matrixCols_in, my_stream);
}

template<typename T> void sycl_put_skewsymmetric_second_half_q_FromC(T *q_dev, T *q2_dev, int *matrixRows_in, int *matrixCols_in, QueueData *my_stream) {
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;

  sycl::queue q = getQueueOrDefault(my_stream);
  sycl::range<2> threadsPerBlock = maxWorkgroupSize<2>(q);
  sycl::range<2> blocks((matrixCols + threadsPerBlock[0] - 1) / threadsPerBlock[0], (matrixRows + threadsPerBlock[1] - 1) / threadsPerBlock[1]);

  q.parallel_for(sycl::nd_range<2>(blocks * threadsPerBlock, threadsPerBlock), [=](sycl::nd_item<2> it) {
    int row = it.get_group(1) * it.get_local_range(1) + it.get_local_id(1);
    int col = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);

    if (row < matrixRows && col < matrixCols) {
      int index  = row + matrixRows * col;
      int index2 = row + matrixRows * (col + matrixCols);
      q_dev[index2] = q2_dev[index];
    }
  });
  q.wait_and_throw();
}


extern "C" void sycl_put_skewsymmetric_second_half_q_double_FromC(double *q_dev, double *q2_dev, int *matrixRows_in, int *matrixCols_in, QueueData *my_stream) {
  sycl_put_skewsymmetric_second_half_q_FromC<double>(q_dev, q2_dev, matrixRows_in, matrixCols_in, my_stream);
}

extern "C" void sycl_put_skewsymmetric_second_half_q_float_FromC(float *q_dev, float *q2_dev, int *matrixRows_in, int *matrixCols_in, QueueData *my_stream) {
  sycl_put_skewsymmetric_second_half_q_FromC<float>(q_dev, q2_dev, matrixRows_in, matrixCols_in, my_stream);
}