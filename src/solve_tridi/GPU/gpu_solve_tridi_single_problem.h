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

//________________________________________________________________

template <typename T>
__global__ void gpu_check_monotony_kernel(T *d, T *q, T *qtmp, const int nlen, const int ldq) {
  //int i = blockIdx.x * blockDim.x + threadIdx.x;
  //int j = blockIdx.y * blockDim.y + threadIdx.y;

  int j;

  //if (i>=0 && i<gemm_dim_k) {
  //  if (j>=0 && j<gemm_dim_l) {
  //    qtmp1_tmp[i+gemm_dim_k*j] = qtmp1[i+gemm_dim_k*j];
  //  }    
  //}
  for (int i=0; i<nlen-1;i++) {
    if (d[i+1] < d[i]) {
      double dtmp = d[i+1];
      for (j=0; j<nlen; j++) {
        qtmp[j] = q[j+ldq*(i+1)];
      }
      for (j=i; j>=0; j--){
        if (dtmp < d[j]) {
          d[j+1] = d[j];
          for (int k=0;k<nlen;k++) {
            q[k+ldq*(j+1)] = q[k+ldq*j];
          }
        } else {
          break;
        }
      }
      d[j+1] = dtmp;
      for (int k=0;k<nlen;k++) {
        q[k+ldq*(j+1)] = qtmp[k];
      }
    }
  }
}


template <typename T>
void gpu_check_monotony(T *d_dev, T *q_dev, T *qtmp_dev, int nlen, int ldq, int debug, gpuStream_t my_stream) {

  dim3 threadsPerBlock(1);
  //dim3 blocks((gemm_dim_k + threadsPerBlock.x - 1) / threadsPerBlock.x, (gemm_dim_l + threadsPerBlock.y - 1) / threadsPerBlock.y);
  dim3 blocks(1);

#ifdef WITH_GPU_STREAMS
  gpu_check_monotony_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(d_dev, q_dev, qtmp_dev, nlen, ldq);
#else
  gpu_check_monotony_kernel<<<blocks,threadsPerBlock>>>            (d_dev, q_dev, qtmp_dev, nlen, ldq);
#endif
  
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_check_monotony: %s\n",gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _check_monotony_FromC)(char dataType, intptr_t d_dev, intptr_t q_dev, intptr_t qtmp_dev,
                                                              int nlen, int ldq, int debug, gpuStream_t my_stream) {
  if      (dataType=='D') gpu_check_monotony<double>((double *) d_dev, (double *) q_dev, (double *) qtmp_dev, nlen, ldq, debug, my_stream);
  else if (dataType=='S') gpu_check_monotony<float> ((float  *) d_dev, (float  *) q_dev, (float  *) qtmp_dev, nlen, ldq, debug, my_stream);
  else {
    printf("Error in elpa_check_monotony: Unsupported data type\n");
  }
}

//________________________________________________________________

template <typename T>
__global__ void gpu_construct_tridi_matrix_kernel(T *q, T *d, T *e, const int nlen, const int ldq) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // q has been zeroed outside of kernel
  if (i>=0 && i<nlen) {
    // diagonal
    q[i+ldq*i] = d[i];
  }

  // sub-diagonal
  if (i>=0 && i<nlen-1) {
    q[i+1 + ldq * i] = e[i];
    q[i + ldq * (i+1)] = e[i];
  }

}


template <typename T>
void gpu_construct_tridi_matrix(T *q_dev, T *d_dev, T *e_dev, int nlen, int ldq, int debug, gpuStream_t my_stream) {

  dim3 threadsPerBlock(1024);
  dim3 blocks((nlen + threadsPerBlock.x - 1) / threadsPerBlock.x);

#ifdef WITH_GPU_STREAMS
  gpu_construct_tridi_matrix_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(q_dev, d_dev, e_dev, nlen, ldq);
#else
  gpu_construct_tridi_matrix_kernel<<<blocks,threadsPerBlock>>>            (q_dev, d_dev, e_dev, nlen, ldq);
#endif
  
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_construct_tridi_matrix: %s\n",gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _construct_tridi_matrix_FromC)(char dataType, intptr_t q_dev, intptr_t d_dev, intptr_t e_dev,
                                                              int nlen, int ldq, int debug, gpuStream_t my_stream) {
  if      (dataType=='D') gpu_construct_tridi_matrix<double>((double *) q_dev, (double *) d_dev, (double *) e_dev, nlen, ldq, debug, my_stream);
  else if (dataType=='S') gpu_construct_tridi_matrix<float> ((float  *) q_dev, (float  *) d_dev, (float  *) e_dev, nlen, ldq, debug, my_stream);
  else {
    printf("Error in elpa_construct_tridi_matrix: Unsupported data type\n");
  }
}