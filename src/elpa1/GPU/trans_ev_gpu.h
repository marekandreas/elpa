//    Copyright 2024, P. Karpov
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

//_________________________________________________________________________________________________

template <typename T>
__global__ void gpu_copy_hvb_a_kernel(T *hvb_dev, T *a_dev, int ld_hvb, int lda, int my_prow, int np_rows,
                                      int my_pcol, int np_cols, int nblk, int ics, int ice) {
  // nb = 0
  // do ic = ics, ice
  //   l_colh = local_index(ic  , my_pcol, np_cols, nblk, -1) ! Column of Householder Vector
  //   l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1) ! # rows of Householder Vector

  //   if (my_pcol == cur_pcol) then
  //     hvb(nb+1:nb+l_rows) = a_mat(1:l_rows,l_colh)
  //     if (my_prow == prow(ic-1, nblk, np_rows)) then
  //       hvb(nb+l_rows) = 1.
  //     endif
  //   endif

  //   nb = nb+l_rows
  // enddo

  int i0   = threadIdx.x; // 0..l_rows-1
  int ic_0 = blockIdx.x ; // 0..l_cowl-1

  T One = elpaDeviceNumber<T>(1.0);
  //int nb=0; // PETERDEBUG: cleanup

  for (int ic = ics; ic <= ice; ic++) {
    int l_colh = local_index(ic  , my_pcol, np_cols, nblk, -1); // Column of Householder Vector
    int l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1); // Number of rows of Householder Vector

    // if (my_pcol == cur_pcol) // already true
    for (int i = 0; i < l_rows; i++) {
      hvb_dev[i + ld_hvb*(ic-ics)] = a_dev[i + (l_colh-1)*lda]; // nb -> ld_hvb*(ic-ics), no compression
      // hvb_dev[i + nb] = a_dev[i + (l_colh-1)*lda];
    }
    
    if (my_prow == prow(ic-1, nblk, np_rows)) {
      hvb_dev[(l_rows-1) + ld_hvb*(ic-ics)] = One;
    }
    
    //nb += l_rows;
  }
}

template <typename T>
void gpu_copy_hvb_a(T *hvb_dev, T *a_dev, int *ld_hvb_in, int *lda_in, int *my_prow_in, int *np_rows_in,
                    int *my_pcol_in, int *np_cols_in, int *nblk_in, int *ics_in, int *ice_in, int *SM_count_in, int *debug_in, gpuStream_t my_stream){
  int ld_hvb = *ld_hvb_in;
  int lda = *lda_in;
  int my_prow = *my_prow_in;
  int np_rows = *np_rows_in;
  int my_pcol = *my_pcol_in;
  int np_cols = *np_cols_in;
  int nblk = *nblk_in;
  int ics = *ics_in;
  int ice = *ice_in;
  int SM_count = *SM_count_in;
  int debug = *debug_in;

  //dim3 blocks = dim3(SM_count, 1, 1); // PETERDEBUG
  //dim3 blocks = dim3(l_cols,1,1);
  //dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

  dim3 blocks = dim3(1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_hvb_a_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(hvb_dev, a_dev, ld_hvb, lda, my_prow, np_rows, my_pcol, np_cols, nblk, ics, ice);
#else
  gpu_copy_hvb_a_kernel<<<blocks,threadsPerBlock>>>            (hvb_dev, a_dev, ld_hvb, lda, my_prow, np_rows, my_pcol, np_cols, nblk, ics, ice);
#endif

  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_copy_hvb_a: %s\n", gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _copy_hvb_a_FromC) (char dataType, intptr_t hvb_dev, intptr_t a_dev,
                                      int *ld_hvb_in, int *lda_in, int *my_prow_in, int *np_rows_in,
                                      int *my_pcol_in, int *np_cols_in, int *nblk_in, int *ics_in, int *ice_in, 
                                      int *SM_count_in, int *debug_in, gpuStream_t my_stream){
  if      (dataType=='D') gpu_copy_hvb_a<double>((double *) hvb_dev, (double *) a_dev, ld_hvb_in, lda_in, my_prow_in, np_rows_in, my_pcol_in, np_cols_in, nblk_in, ics_in, ice_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='S') gpu_copy_hvb_a<float> ((float  *) hvb_dev, (float  *) a_dev, ld_hvb_in, lda_in, my_prow_in, np_rows_in, my_pcol_in, np_cols_in, nblk_in, ics_in, ice_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='Z') gpu_copy_hvb_a<cuDoubleComplex>((gpuDoubleComplex *) hvb_dev, (gpuDoubleComplex *) a_dev, ld_hvb_in, lda_in, my_prow_in, np_rows_in, my_pcol_in, np_cols_in, nblk_in, ics_in, ice_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='C') gpu_copy_hvb_a<cuFloatComplex> ((gpuFloatComplex  *) hvb_dev, (gpuFloatComplex  *) a_dev, ld_hvb_in, lda_in, my_prow_in, np_rows_in, my_pcol_in, np_cols_in, nblk_in, ics_in, ice_in, SM_count_in, debug_in, my_stream);
}

//_________________________________________________________________________________________________

template <typename T>
__global__ void gpu_copy_hvm_hvb_kernel(T *hvm_dev, T *hvb_dev, int ld_hvm, int ld_hvb, int my_prow, int np_rows,
                                        int nstor, int nblk, int ics, int ice) {
  // nb = 0
  // NVTX_RANGE_PUSH("loop: copy hvm <- hvb")
  // do ic = ics, ice
  //   l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1) ! # rows of Householder Vector
  //   hvm(1:l_rows,nstor+1) = hvb(nb+1:nb+l_rows)
  //   if (useGPU) then
  //     hvm_ubnd = l_rows
  //   endif
  //   nstor = nstor+1
  //   nb = nb+l_rows
  // enddo

  int i0   = threadIdx.x; // 0..l_rows-1
  int ic_0 = blockIdx.x ; // 0..l_cowl-1

  T Zero = elpaDeviceNumber<T>(0.0);
  //int nb=0; // PETERDEBUG: cleanup

  for (int ic = ics; ic <= ice; ic++) {
    int l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1);

    for (int i=0; i < l_rows; i++) {
      hvm_dev[i + ld_hvm*(ic-ics+nstor)] = hvb_dev[i + ld_hvb*(ic-ics)]; // nb -> ld_hvb*(ic-ics), no compression
    }
    
    for (int i=l_rows; i < ld_hvm; i++) {
      hvm_dev[i + ld_hvm*(ic-ics+nstor)] = Zero; // since we're not compressing, we need to take extra care to clear from previous iterations
    }

  }
}

template <typename T>
void gpu_copy_hvm_hvb(T *hvm_dev, T *hvb_dev, int *ld_hvm_in, int *ld_hvb_in, int *my_prow_in, int *np_rows_in,
                      int *nstor_in, int *nblk_in, int *ics_in, int *ice_in, int *SM_count_in, int *debug_in, gpuStream_t my_stream){
  int ld_hvm = *ld_hvm_in;
  int ld_hvb = *ld_hvb_in;
  int my_prow = *my_prow_in;
  int np_rows = *np_rows_in;
  int nstor = *nstor_in;
  int nblk = *nblk_in;
  int ics = *ics_in;
  int ice = *ice_in;
  int SM_count = *SM_count_in;
  int debug = *debug_in;

  //dim3 blocks = dim3(SM_count, 1, 1); // PETERDEBUG
  //dim3 blocks = dim3(l_cols,1,1);
  //dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

  dim3 blocks = dim3(1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_hvm_hvb_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(hvm_dev, hvb_dev, ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, ics, ice);
#else
  gpu_copy_hvm_hvb_kernel<<<blocks,threadsPerBlock>>>            (hvm_dev, hvb_dev, ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, ics, ice);
#endif
  
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_copy_hvm_hvb: %s\n", gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _copy_hvm_hvb_FromC) (char dataType, intptr_t hvm_dev, intptr_t hvb_dev,
                                      int *ld_hvm_in, int *ld_hvb_in, int *my_prow_in, int *np_rows_in,
                                      int *nstor_in, int *nblk_in, int *ics_in, int *ice_in, 
                                      int *SM_count_in, int *debug_in, gpuStream_t my_stream){
  if      (dataType=='D') gpu_copy_hvm_hvb<double>((double *) hvm_dev, (double *) hvb_dev, ld_hvm_in, ld_hvb_in, my_prow_in, np_rows_in, nstor_in, nblk_in, ics_in, ice_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='S') gpu_copy_hvm_hvb<float> ((float  *) hvm_dev, (float  *) hvb_dev, ld_hvm_in, ld_hvb_in, my_prow_in, np_rows_in, nstor_in, nblk_in, ics_in, ice_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='Z') gpu_copy_hvm_hvb<cuDoubleComplex>((gpuDoubleComplex *) hvm_dev, (gpuDoubleComplex *) hvb_dev, ld_hvm_in, ld_hvb_in, my_prow_in, np_rows_in, nstor_in, nblk_in, ics_in, ice_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='C') gpu_copy_hvm_hvb<cuFloatComplex> ((gpuFloatComplex  *) hvm_dev, (gpuFloatComplex  *) hvb_dev, ld_hvm_in, ld_hvb_in, my_prow_in, np_rows_in, nstor_in, nblk_in, ics_in, ice_in, SM_count_in, debug_in, my_stream);
}