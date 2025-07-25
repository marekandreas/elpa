//    Copyright 2025, P. Karpov
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
//      Schwerpunkt Wissenschaftliches Rechnen,
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

#include "src/GPU/SYCL/syclCommon.hpp"
#include <sycl/sycl.hpp>
#include <math.h>
#include <stdlib.h>
#include <alloca.h>
#include <stdint.h>
#include <complex>

#include "config-f90.h"

#include "src/GPU/common_device_functions.h"
#include "src/GPU/gpu_to_cuda_and_hip_interface.h"

using namespace sycl_be;

extern "C" int syclDeviceSynchronizeFromC();

template <typename T>
void gpu_distribute_global_column_kernel (T *g_col, T *l_col, 
                                          const int g_col_dim1, const int g_col_dim2, const int ldq, const int matrixCols, 
                                          const int noff_in, const int noff, const int nlen, const int my_prow, const int np_rows, const int nblk, 
                                          const sycl::nd_item<3> &it) {
    int i = it.get_group(2) * it.get_local_range(2) + it.get_local_id(2) + 1;
    int ind = it.get_group(1) * it.get_local_range(1) + it.get_local_id(1);

    //variant 2
    int ind2 = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);

    int nbs = noff/(nblk*np_rows);
    int nbe = (noff+nlen-1)/(nblk*np_rows);

    int number_of_entries;
    int entries_in_started_col;
    int entries_in_sub_matrix;
    int columns_in_sub_matrix;

    int index_sub_matrix;
    int col_sub_matrix;
    int row_sub_matrix;

    int l_col_global_row2;
    int l_col_global_col2;

    int g_col_global_row2;
    int g_col_global_col2;


    if (i>= 1 && i < nlen+1) {
      int g_col_offset1 = 1;
      int g_col_offset2 = i;
             
      int l_col_offset1 = 1;
      int l_col_offset2 = noff_in+i;

      int jb = ind + nbs;

      if (jb >= nbs && jb<=nbe){	
        int g_off2 = jb*nblk*np_rows + nblk*my_prow;
        int l_off2 = jb*nblk;

        int js2 = std::max(noff+1-g_off2, 1);
        int je2 = std::min(noff+nlen-g_off2, nblk);

        if (je2>=js2) {

          //for (int index4=0;index4<=je2-js2;index4++) {
	  //  int index=index4+js2;
	    int index = ind2 + js2;


	    if (index >= js2 && index <= je2) {
		    
              number_of_entries = (g_col_dim2-g_col_offset2)*g_col_dim1+g_col_dim1-g_col_offset1+1;
              entries_in_started_col=g_col_dim1-g_col_offset1+1;
              entries_in_sub_matrix=number_of_entries-entries_in_started_col;
              columns_in_sub_matrix=entries_in_sub_matrix/g_col_dim1;

              if (g_off2-noff+index > entries_in_started_col) {
                index_sub_matrix = g_off2-noff+index - entries_in_started_col;
                col_sub_matrix = (index_sub_matrix -1)/g_col_dim1 + 1;
                row_sub_matrix = index_sub_matrix % g_col_dim1;

                if (row_sub_matrix == 0) { row_sub_matrix = g_col_dim1;}

                g_col_global_col2 = col_sub_matrix + g_col_offset2;
                g_col_global_row2 = row_sub_matrix;
              } else {

                g_col_global_col2 = g_col_offset2;
                g_col_global_row2 = g_col_offset1 + g_off2-noff+index -1;

              }


              number_of_entries = (matrixCols-l_col_offset2)*ldq+ldq-l_col_offset1+1;
              entries_in_started_col=ldq-l_col_offset1+1;
              entries_in_sub_matrix=number_of_entries-entries_in_started_col;
              columns_in_sub_matrix=entries_in_sub_matrix/ldq;

              if (l_off2+index > entries_in_started_col) {
                index_sub_matrix = l_off2+index - entries_in_started_col;
                col_sub_matrix = (index_sub_matrix -1)/ ldq + 1;
                row_sub_matrix = index_sub_matrix % ldq;

                if (row_sub_matrix == 0) { row_sub_matrix = ldq;}

                l_col_global_col2 = col_sub_matrix + l_col_offset2;
                l_col_global_row2 = row_sub_matrix;
              } else {

                l_col_global_col2 = l_col_offset2;
                l_col_global_row2 = l_col_offset1 + l_off2+index -1;
              }


              l_col[(l_col_global_row2-1) + ldq*(l_col_global_col2-1) ] = g_col[(g_col_global_row2-1) + g_col_dim1 * (g_col_global_col2-1)];
	    }
          //}
        }
      }
    }
}


template <typename T>
void gpu_distribute_global_column(T *g_col_dev, T *l_col_dev, int g_col_dim1, int g_col_dim2, int ldq, int matrixCols, 
                                  int noff_in, int noff, int nlen, int my_prow, int np_rows, int nblk,
                                  int debug, QueueData *my_stream) {

  int nbs = noff/(nblk*np_rows);
  int nbe = (noff+nlen-1)/(nblk*np_rows);

  sycl::queue q = getQueueOrDefault(my_stream);
  sycl::range<3> threadsPerBlock(2,16,32);
  sycl::range<3> blocks( (matrixCols + threadsPerBlock.get(0) - 1) / threadsPerBlock.get(0),
                        (nbe-nbs+1 + threadsPerBlock.get(1) - 1) / threadsPerBlock.get(1), 
                        (nlen + threadsPerBlock.get(2) - 1) / threadsPerBlock.get(2));

  q.parallel_for(sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock), [=](sycl::nd_item<3> it) {
        gpu_distribute_global_column_kernel(g_col_dev, l_col_dev, 
                                            g_col_dim1, g_col_dim2, ldq, matrixCols, 
                                            noff_in, noff, nlen, my_prow, np_rows, nblk, it);
  });
  if (debug) syclDeviceSynchronizeFromC();
}

extern "C" void CONCATENATE(ELPA_GPU,  _distribute_global_column_FromC)(char dataType, intptr_t g_col_dev, intptr_t l_col_dev,
                                                                        int g_col_dim1, int g_col_dim2, int ldq, int matrixCols, 
                                                                        int noff_in, int noff, int nlen, 
                                                                        int my_prow, int np_rows, int nblk,
                                                                        int debug, gpuStream_t my_stream) {
  if      (dataType=='D') gpu_distribute_global_column<double>((double *) g_col_dev, (double *) l_col_dev, 
                                                              g_col_dim1, g_col_dim2, ldq, matrixCols, 
                                                              noff_in, noff, nlen, my_prow, np_rows, nblk,
                                                              debug, my_stream);
  else if (dataType=='S') gpu_distribute_global_column<float> ((float  *) g_col_dev, (float  *) l_col_dev, 
                                                              g_col_dim1, g_col_dim2, ldq, matrixCols, 
                                                              noff_in, noff, nlen, my_prow, np_rows, nblk,
                                                              debug, my_stream);
  else {
    printf("Error in elpa_distribute_global_column: Unsupported data type\n");
  }
}
