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
//    This file was written by A. Marek, MPCDF

#include "config-f90.h"

#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <complex.h>
#include <cuComplex.h>
#include <stdint.h>
#include "config-f90.h"
    
#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)


__global__ void cuda_distribute_global_column_double_kernel(double *g_col, const int g_col_dim1, const int g_col_dim2, double *l_col, const int ldq, const int matrixCols, const int noff_in, const int noff, const int nlen, const int my_prow, const int np_rows, const int nblk) {

    int i    = blockIdx.x * blockDim.x + threadIdx.x +1;
    int ind  = blockIdx.y * blockDim.y + threadIdx.y;

    //variant 2
    int ind2  = blockIdx.z * blockDim.z + threadIdx.z;

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

        int js2 = max(noff+1-g_off2, 1);
        int je2 = min(noff+nlen-g_off2, nblk);

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

extern "C" void cuda_distribute_global_column_double_FromC(double *g_col_dev, int *g_col_dim1_in, int *g_col_dim2_in,  double *l_col_dev, int *ldq_in, int *matrixCols_in, int *noff_in_in, int *noff_in, int *nlen_in, int * my_prow_in, int *np_rows_in, int *nblk_in, cudaStream_t  my_stream){

  int noffin = *noff_in_in;
  int noff = *noff_in;
  int nlen = *nlen_in;
  int my_prow = *my_prow_in;
  int np_rows = *np_rows_in;
  int nblk = *nblk_in;
  int matrixCols = *matrixCols_in;
  int ldq = *ldq_in;
  int g_col_dim1 = *g_col_dim1_in;
  int g_col_dim2 = *g_col_dim2_in;


  int nbs = noff/(nblk*np_rows);
  int nbe = (noff+nlen-1)/(nblk*np_rows);

  //dim3 threadsPerBlock(32,32);
  //dim3 blocks((nlen + threadsPerBlock.x - 1) / threadsPerBlock.x, ((nbe-nbs+1) + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // variant 2
  dim3 threadsPerBlock(32,16,2);
  dim3 blocks((nlen + threadsPerBlock.x - 1) / threadsPerBlock.x, ((nbe-nbs+1) + threadsPerBlock.y - 1) / threadsPerBlock.y, ((matrixCols) + threadsPerBlock.z - 1) / threadsPerBlock.z);

#ifdef WITH_GPU_STREAMS
  cuda_distribute_global_column_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(g_col_dev, g_col_dim1, g_col_dim2, l_col_dev, ldq, matrixCols, noffin, noff, nlen, my_prow, np_rows, nblk);
#else
  cuda_distribute_global_column_double_kernel<<<blocks, threadsPerBlock>>>(g_col_dev, g_col_dim1, g_col_dim2, l_col_dev, ldq, matrixCols, noffin, noff, nlen, my_prow, np_rows, nblk);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_distribute_global_column_double_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}




__global__ void cuda_distribute_global_column_float_kernel(float *g_col, const int g_col_dim1, const int g_col_dim2, float *l_col, const int ldq, const int matrixCols, const int noff_in, const int noff, const int nlen, const int my_prow, const int np_rows, const int nblk) {

    int i    = blockIdx.x * blockDim.x + threadIdx.x +1;
    int ind  = blockIdx.y * blockDim.y + threadIdx.y;

    //variant 2
    int ind2  = blockIdx.z * blockDim.z + threadIdx.z;

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

        int js2 = max(noff+1-g_off2, 1);
        int je2 = min(noff+nlen-g_off2, nblk);

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

extern "C" void cuda_distribute_global_column_float_FromC(float *g_col_dev, int *g_col_dim1_in, int *g_col_dim2_in,  float *l_col_dev, int *ldq_in, int *matrixCols_in, int *noff_in_in, int *noff_in, int *nlen_in, int * my_prow_in, int *np_rows_in, int *nblk_in, cudaStream_t  my_stream){

  int noffin = *noff_in_in;
  int noff = *noff_in;
  int nlen = *nlen_in;
  int my_prow = *my_prow_in;
  int np_rows = *np_rows_in;
  int nblk = *nblk_in;
  int matrixCols = *matrixCols_in;
  int ldq = *ldq_in;
  int g_col_dim1 = *g_col_dim1_in;
  int g_col_dim2 = *g_col_dim2_in;


  int nbs = noff/(nblk*np_rows);
  int nbe = (noff+nlen-1)/(nblk*np_rows);

  //dim3 threadsPerBlock(32,32);
  //dim3 blocks((nlen + threadsPerBlock.x - 1) / threadsPerBlock.x, ((nbe-nbs+1) + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // variant 2
  dim3 threadsPerBlock(32,16,2);
  dim3 blocks((nlen + threadsPerBlock.x - 1) / threadsPerBlock.x, ((nbe-nbs+1) + threadsPerBlock.y - 1) / threadsPerBlock.y, ((matrixCols) + threadsPerBlock.z - 1) / threadsPerBlock.z);

#ifdef WITH_GPU_STREAMS
  cuda_distribute_global_column_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(g_col_dev, g_col_dim1, g_col_dim2, l_col_dev, ldq, matrixCols, noffin, noff, nlen, my_prow, np_rows, nblk);
#else
  cuda_distribute_global_column_float_kernel<<<blocks, threadsPerBlock>>>(g_col_dev, g_col_dim1, g_col_dim2, l_col_dev, ldq, matrixCols, noffin, noff, nlen, my_prow, np_rows, nblk);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_distribute_global_column_float_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}
