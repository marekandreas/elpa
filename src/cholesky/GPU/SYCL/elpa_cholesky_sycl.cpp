//    Copyright 2023, P. Karpov
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
//    This file was written by P. Karpov, MPCDF

#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <stdint.h>
#include <complex>
#include <type_traits>
#include "config-f90.h"

#include <CL/sycl.hpp>
#include "src/GPU/SYCL/syclCommon.hpp"

#include "../../../GPU/common_device_functions.h"

#define MAX_THREADS_PER_BLOCK 1024

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

// PETERDEBUG -- is it possible to inline this?
// template <typename T>  T elpaDeviceNumber2(double number);
// template <>  double elpaDeviceNumber2<double>(double number) {return number;}
// template <>  float  elpaDeviceNumber2<float> (double number) {return (float) number;}
// template <> std::complex<double> elpaDeviceNumber2<std::complex<double>>(double number) {return std::complex<double>(number, 0.0);}
// template <> std::complex<float>  elpaDeviceNumber2<std::complex<float>> (double number) {return std::complex<float>((float)number, 0.0f);}

// inline double elpaDeviceComplexConjugate(double number) {return number;}
// inline float elpaDeviceComplexConjugate(float  number) {return number;}
// inline std::complex<double> elpaDeviceComplexConjugate(std::complex<double> number) {return std::conj(number);}
// inline std::complex<float>  elpaDeviceComplexConjugate(std::complex<float> number)  {return std::conj(number);}

// inline int pcol(int I_gl, int nblk, int np_cols) {
//   // C-style 0-based indexing in assumed
//   return (I_gl/nblk)%np_cols;
// }

// inline int local_index(int I_gl, int my_proc, int num_procs,
//                                 int nblk) {

// //  local_index: returns the local index for a given global index
// //               If the global index has no local index on the
// //               processor my_proc, return next local index after that row/col
// //               C-style 0-based indexing in assumed
// //  Parameters
// //
// //  I_gl        Global index
// //  my_proc     Processor row/column for which to calculate the local index
// //  num_procs   Total number of processors along row/column
// //  nblk        Blocksize
// //
// // Behavior corresponds to Fortran's local_index() with iflag> 0 : Return next local index after that row/col
// //
// // L_block_gl = I_gl/nblk; // global ordinal number of the nblk-block among other blocks
// // l_block_loc = L_block_gl/num_procs =  I_gl/(num_procs*nblk); // local ordinal number of the nblk-block among other blocks
// // x = I_gl%nblk; // local coordinate within the block
// // local_index = l_block*nblk + x;

//   if ((I_gl/nblk)%num_procs == my_proc) // (L_block_gl%num_procs == my_proc), block is local
//     {
//     return I_gl/(num_procs*nblk)* nblk + I_gl%nblk; // local_index = l_block_loc * nblk + x
//     }
//   else if ((I_gl/nblk)%num_procs < my_proc) // block is non-local
//     {
//     return I_gl/(num_procs*nblk)* nblk;
//     }
//   else // ((I_gl/nblk)%num_procs > my_proc)
//     {
//     return (I_gl/(num_procs*nblk) + 1)* nblk;
//     }
// }

//________________________________________________________________

template <typename T>
void sycl_copy_a_tmatc_kernel(T *a_dev, T *tmatc_dev, const int l_cols, const int matrixRows, const int l_colx, const int l_row1, const int nblk, const sycl::nd_item<1> &it){

  int ii_index = it.get_local_id(0) + 1; // range 1..nblk
  int jj_index = it.get_group(0) + 1;    // range 1..l_cols-l_colx+1
  
  if constexpr (std::is_same<T, std::complex<float>>::value || std::is_same<T, std::complex<double>>::value) {
    tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] = std::conj(a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows]);
  } 
  else {
    tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] =           a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows];
  }
}

template <typename T>
void sycl_copy_a_tmatc_FromC(T *a_dev, T *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){

  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;

  sycl::range<1> global_range = sycl::range<1>(nblk*(l_cols - l_colx + 1));
  sycl::range<1> local_range  = sycl::range<1>(nblk);
  
  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  queue.parallel_for(
    sycl::nd_range<1>(global_range, local_range),
    [=](sycl::nd_item<1> it) {
      sycl_copy_a_tmatc_kernel(a_dev, tmatc_dev, l_cols, matrixRows, l_colx, l_row1, nblk, it);
    });
  queue.wait_and_throw();

}

extern "C" void sycl_copy_double_a_tmatc_FromC(double *a_dev, double *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){
  sycl_copy_a_tmatc_FromC(a_dev, tmatc_dev, nblk_in, matrixRows_in, l_cols_in, l_colx_in, l_row1_in, my_stream);
}

extern "C" void sycl_copy_float_a_tmatc_FromC(float *a_dev, float *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){
  sycl_copy_a_tmatc_FromC(a_dev, tmatc_dev, nblk_in, matrixRows_in, l_cols_in, l_colx_in, l_row1_in, my_stream);
}

extern "C" void sycl_copy_double_complex_a_tmatc_FromC(std::complex<double> *a_dev, std::complex<double> *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){
  sycl_copy_a_tmatc_FromC(a_dev, tmatc_dev, nblk_in, matrixRows_in, l_cols_in, l_colx_in, l_row1_in, my_stream);
}

extern "C" void sycl_copy_float_complex_a_tmatc_FromC(std::complex<float> *a_dev, std::complex<float> *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){
  sycl_copy_a_tmatc_FromC(a_dev, tmatc_dev, nblk_in, matrixRows_in, l_cols_in, l_colx_in, l_row1_in, my_stream);
}

//________________________________________________________________


template <typename T>
void sycl_set_a_lower_to_zero_kernel (T *a_dev, int na, int matrixRows, int my_pcol, int np_cols, int my_prow, int np_rows, int nblk,
                                      const sycl::nd_item<1> &it) {

  int J_gl_0 = it.get_group(0);      // 0..nblk-1
  int di_loc_0 = it.get_local_id(0); // 0..MAX_THREADS_PER_BLOCK-1

  T Zero = 0;

  for (int J_gl = J_gl_0; J_gl < na; J_gl += it.get_group_range(0))
    {
    if (my_pcol == pcol(J_gl, nblk, np_cols))
      {
      // Calculate local column and row indices of the first element below the diagonal (that has to be set to zero)
      int l_col1 = local_index(J_gl  , my_pcol, np_cols, nblk);
      int l_row1 = local_index(J_gl+1, my_prow, np_rows, nblk); // I_gl = J_gl + 1

      // Calculate the offset and number of elements to zero out
      //int offset = l_row1 + matrixRows*l_col1;
      //int num = (matrixRows - l_row1);

      // Set to zero in the GPU memory
      for (int di_loc=di_loc_0; di_loc < (matrixRows-l_row1); di_loc += it.get_local_range(0))
        a_dev[(l_row1+di_loc) + matrixRows*l_col1] = Zero;
      }
    }
}

template <typename T>
void sycl_set_a_lower_to_zero(T *a_dev, int *na_in, int *matrixRows_in,
                              int *my_pcol_in, int *np_cols_in, int *my_prow_in,
                              int *np_rows_in, int *nblk_in, int *wantDebug_in,
                              intptr_t my_stream) {
  int na = *na_in;
  int matrixRows = *matrixRows_in;
  int my_pcol = *my_pcol_in;
  int np_cols = *np_cols_in;
  int my_prow = *my_prow_in;
  int np_rows = *np_rows_in;
  int nblk = *nblk_in;
  int wantDebug = *wantDebug_in;

  sycl::range<1> global_range = sycl::range<1>(MAX_THREADS_PER_BLOCK*nblk);
  sycl::range<1> local_range  = sycl::range<1>(MAX_THREADS_PER_BLOCK);

  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  queue.parallel_for(
    sycl::nd_range<1>(global_range, local_range),
    [=](sycl::nd_item<1> it) {
      sycl_set_a_lower_to_zero_kernel(a_dev, na, matrixRows, my_pcol, np_cols, my_prow, np_rows, nblk, it);
    });
  queue.wait_and_throw();

}

extern "C" void sycl_set_a_lower_to_zero_FromC(char dataType, intptr_t a_dev, int *na_in, int *matrixRows_in,
                                               int *my_pcol_in, int *np_cols_in, int *my_prow_in, int *np_rows_in,
                                               int *nblk_in, int *wantDebug_in, intptr_t my_stream) {

  if (dataType=='D') sycl_set_a_lower_to_zero<double>((double *) a_dev, na_in, matrixRows_in, my_pcol_in, np_cols_in, my_prow_in, np_rows_in, nblk_in, wantDebug_in, my_stream);
  if (dataType=='S') sycl_set_a_lower_to_zero<float> ((float *) a_dev, na_in, matrixRows_in, my_pcol_in, np_cols_in, my_prow_in, np_rows_in, nblk_in, wantDebug_in, my_stream);
  if (dataType=='Z') sycl_set_a_lower_to_zero<std::complex<double>>((std::complex<double> *)a_dev, na_in, matrixRows_in, my_pcol_in, np_cols_in, my_prow_in, np_rows_in, nblk_in, wantDebug_in, my_stream);
  if (dataType=='C') sycl_set_a_lower_to_zero<std::complex<float>> ((std::complex<float>  *)a_dev, na_in, matrixRows_in, my_pcol_in, np_cols_in, my_prow_in, np_rows_in, nblk_in, wantDebug_in, my_stream);
}