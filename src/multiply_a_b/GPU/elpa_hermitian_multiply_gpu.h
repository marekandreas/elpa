//    Copyright 2023, A. Marek
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

//________________________________________________________________

template <typename T>
__global__ void gpu_copy_tmp2_c_kernel(T *tmp2_dev, T *c_dev, 
                                       const int nr_done, const int nstor, const int lcs, const int lce, const int ldc, const int ldcCols){

  int idex = threadIdx.x + 1; // range 1..nstor
  int jdex = blockIdx.x  + 1; // range 1..lce-lse+1

  //base 1 index
  c_dev[nr_done+(idex-1) + ldc*(lcs-1+jdex-1)] = tmp2_dev[0+(idex-1)+nstor*(jdex-1)];

}

template <typename T>
void gpu_copy_tmp2_c (T *tmp2_dev, T *c_dev, 
                      int nr_done, int nstor, int lcs, int lce, int ldc, int ldcCols, gpuStream_t my_stream) { 

  dim3 blocks = dim3(lce-lcs+1,1,1);
  dim3 threadsPerBlock = dim3(nr_done+nstor-(nr_done+1)+1,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_tmp2_c_kernel<<<blocks,threadsPerBlock, 0, my_stream>>>(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols);
#else
  gpu_copy_tmp2_c_kernel<<<blocks,threadsPerBlock>>>(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols);
#endif
  gpuError_t gpuerr = gpuGetLastError();
  if (gpuerr != gpuSuccess){
    printf("Error in executing copy_tmp2_c_kernel: %s\n",gpuGetErrorString(gpuerr));
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _copy_tmp2_c_FromC) (char dataType, intptr_t tmp2_dev, intptr_t c_dev,
                                                        int nr_done, int nstor, int lcs, int lce, int ldc, int ldcCols, gpuStream_t my_stream) {
  if      (dataType=='D') gpu_copy_tmp2_c<double>((double *) tmp2_dev, (double *) c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream);
  else if (dataType=='S') gpu_copy_tmp2_c<float> ((float  *) tmp2_dev, (float  *) c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream);
  else if (dataType=='Z') gpu_copy_tmp2_c<gpuDoubleComplex>((gpuDoubleComplex *) tmp2_dev, (gpuDoubleComplex *) c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream);
  else if (dataType=='C') gpu_copy_tmp2_c<gpuFloatComplex> ((gpuFloatComplex  *) tmp2_dev, (gpuFloatComplex  *) c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream);
  else printf("Error in gpu_copy_tmp2_c: Unsupported data type\n");
}

//________________________________________________________________

template <typename T>
__global__ void gpu_copy_a_aux_bc_kernel(T *a_dev, T *aux_bc_dev, 
                                        const int n_aux_bc, const int nvals, const int lrs, const int lre, const int noff, const int nblk, const int n, const int l_rows, const int lda, const int ldaCols){

  int idex    = blockIdx.x +1; // range 1..lre-lrs+1
  //int jdex = threadIdx.x + 1; // range 1..1
  aux_bc_dev[(n_aux_bc+1-1)+(idex-1)] = a_dev[(lrs-1)+(idex-1)+lda*(noff*nblk+n-1)];
}

template <typename T>
void gpu_copy_a_aux_bc(T *a_dev, T *aux_bc_dev, 
                      int n_aux_bc, int nvals, int lrs, int lre, int noff, int nblk, int n, int l_rows, int lda, int ldaCols, gpuStream_t my_stream) { 
		
  dim3 blocks = dim3(lre-lrs+1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_a_aux_bc_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(a_dev, aux_bc_dev, 
                                                                   n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols);
#else
  gpu_copy_a_aux_bc_kernel<<<blocks,threadsPerBlock>>>             (a_dev, aux_bc_dev, 
                                                                   n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols);
#endif
  gpuError_t gpuerr = gpuGetLastError();
  if (gpuerr != gpuSuccess){
    printf("Error in executing gpu_copy_a_aux_bc_kernel: %s\n",gpuGetErrorString(gpuerr));
  }
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_a_aux_bc_FromC)(char dataType, intptr_t a_dev, intptr_t aux_bc_dev,
                                                            int n_aux_bc, int nvals, int lrs, int lre, int noff, int nblk, int n, 
                                                            int l_rows, int lda, int ldaCols, gpuStream_t my_stream) {
  if      (dataType=='D') gpu_copy_a_aux_bc<double>((double *) a_dev, (double *) aux_bc_dev, 
                                                   n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols, my_stream);
  else if (dataType=='S') gpu_copy_a_aux_bc<float> ((float  *) a_dev, (float  *) aux_bc_dev,
                                                   n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols, my_stream);
  else if (dataType=='Z') gpu_copy_a_aux_bc<gpuDoubleComplex>((gpuDoubleComplex *) a_dev, (gpuDoubleComplex *) aux_bc_dev,
                                                   n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols, my_stream);
  else if (dataType=='C') gpu_copy_a_aux_bc<gpuFloatComplex> ((gpuFloatComplex  *) a_dev, (gpuFloatComplex  *) aux_bc_dev,
                                                   n_aux_bc, nvals, lrs, lre, noff, nblk, n ,l_rows ,lda ,ldaCols ,my_stream);
  else printf("Error in gpu_copy_a_aux_bc: Unsupported data type\n");
}
//________________________________________________________________

template <typename T>
__global__ void gpu_copy_aux_bc_aux_mat_kernel(T *aux_bc_dev, T *aux_mat_dev, 
                                               const int lrs, const int lre, const int nstor, const int n_aux_bc, const int nvals, const int l_rows, const int nblk_mult, const int nblk) {
	
  //aux_mat(lrs:lre,nstor) = aux_bc(n_aux_bc+1:n_aux_bc+nvals)

  int idex = threadIdx.x + 1; // range 1..1
  int jdex = blockIdx.x  + 1; // range 1..lre-lrs+1
  aux_mat_dev[lrs-1+(jdex-1)+l_rows*(nstor-1)] = aux_bc_dev[n_aux_bc+(jdex-1)];

}

template <typename T>
void gpu_copy_aux_bc_aux_mat(T *aux_bc_dev, T *aux_mat_dev, 
                             int lrs, int lre, int nstor, int n_aux_bc, int nvals, int l_rows, int nblk, int nblk_mult, gpuStream_t my_stream) {
  
  dim3 blocks = dim3(lre-lrs+1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_aux_bc_aux_mat_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(aux_bc_dev, aux_mat_dev, 
                                                                         lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult);
#else
  gpu_copy_aux_bc_aux_mat_kernel<<<blocks,threadsPerBlock>>>            (aux_bc_dev, aux_mat_dev,
                                                                         lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult);
#endif
  gpuError_t gpuerr = gpuGetLastError();
  if (gpuerr != gpuSuccess){
    printf("Error in executing copy_aux_bc_aux_mat_kernel: %s\n",gpuGetErrorString(gpuerr));
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _copy_aux_bc_aux_mat_FromC) (char dataType, intptr_t aux_bc_dev, intptr_t aux_mat_dev,
                                        int lrs, int lre, int nstor, int n_aux_bc, int nvals, int l_rows, int nblk, int nblk_mult, gpuStream_t my_stream) {
  if      (dataType=='D') gpu_copy_aux_bc_aux_mat<double>((double *) aux_bc_dev, (double *) aux_mat_dev, 
                                                                   lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult, my_stream);
  else if (dataType=='S') gpu_copy_aux_bc_aux_mat<float> ((float  *) aux_bc_dev, (float  *) aux_mat_dev,
                                                                   lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult, my_stream);
  else if (dataType=='Z') gpu_copy_aux_bc_aux_mat<gpuDoubleComplex>((gpuDoubleComplex *) aux_bc_dev, (gpuDoubleComplex *) aux_mat_dev,
                                                                   lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult, my_stream);
  else if (dataType=='C') gpu_copy_aux_bc_aux_mat<gpuFloatComplex> ((gpuFloatComplex  *) aux_bc_dev, (gpuFloatComplex  *) aux_mat_dev,
                                                                   lrs, lre, nstor, n_aux_bc, nvals, l_rows ,nblk ,nblk_mult ,my_stream);
  else printf("Error in gpu_copy_aux_bc_aux_mat: Unsupported data type\n");
}