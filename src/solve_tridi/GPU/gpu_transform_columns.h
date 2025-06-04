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

//________________________________________________________________

template <typename T>
__global__ void gpu_transform_one_column_kernel (T *a_dev, T *b_dev, T *c_dev, T *alpha_dev, T *beta_dev, int n_elements){
  
  // c = alpha*a + beta*b

  int i0 = threadIdx.x + blockIdx.x*blockDim.x;
  
  for (int i=i0; i<n_elements; i+=gridDim.x*blockDim.x)
    {
    c_dev[i] = alpha_dev[0]*a_dev[i] + beta_dev[0]*b_dev[i];
    }
}

template <typename T>
void gpu_transform_one_column(T *a_dev, T *b_dev, T *c_dev, T *alpha_dev, T *beta_dev, 
                              int n_elements, int SM_count, int debug, gpuStream_t my_stream){

  dim3 blocks = dim3(SM_count,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_transform_one_column_kernel<<<blocks,threadsPerBlock,0,my_stream>>>((T *) a_dev, (T *) b_dev, (T *) c_dev, 
                                                                          (T *) alpha_dev, (T *) beta_dev, n_elements);
#else
  gpu_transform_one_column_kernel<<<blocks,threadsPerBlock>>>            ((T *) a_dev, (T *) b_dev, (T *) c_dev, 
                                                                          (T *) alpha_dev, (T *) beta_dev, n_elements);
#endif
    
    if (debug)
      {
      gpuDeviceSynchronize();
      gpuError_t gpuerr = gpuGetLastError();
      if (gpuerr != gpuSuccess){
        printf("Error in executing gpu_transform_one_column: %s\n",gpuGetErrorString(gpuerr));
      }
    }
  }


extern "C" void CONCATENATE(ELPA_GPU,  _transform_one_column_FromC)(char dataType, intptr_t a_dev, intptr_t b_dev, intptr_t c_dev, 
                                                      intptr_t alpha_dev, intptr_t beta_dev, 
                                                      int n_elements, int SM_count, int debug, gpuStream_t my_stream){

  if      (dataType=='D') gpu_transform_one_column<double>((double *) a_dev, (double *) b_dev, (double *) c_dev, 
                                             (double *) alpha_dev, (double *) beta_dev, 
                                             n_elements, SM_count, debug, my_stream);
  else if (dataType=='S') gpu_transform_one_column<float> ((float  *) a_dev, (float  *) b_dev, (float  *) c_dev, 
                                             (float  *) alpha_dev, (float  *) beta_dev, 
                                             n_elements, SM_count, debug, my_stream);
  else {
    printf("Error in elpa_transform_one_column: Unsupported data type\n");
  }
}

//________________________________________________________________

template <typename T>
__global__ void gpu_transform_two_columns_kernel (T *q_dev, T *qtrans_dev, T *tmp_dev,
                        int ldq, int l_rows, int l_rqs, int l_rqe, int lc1, int lc2){
  
  // tmp(1:l_rows)      = q(l_rqs:l_rqe,lc1)*qtrans(1,1) + q(l_rqs:l_rqe,lc2)*qtrans(2,1)
  // q(l_rqs:l_rqe,lc2) = q(l_rqs:l_rqe,lc1)*qtrans(1,2) + q(l_rqs:l_rqe,lc2)*qtrans(2,2)
  // q(l_rqs:l_rqe,lc1) = tmp(1:l_rows)

  int i0 = threadIdx.x + blockIdx.x*blockDim.x;
  
  T qtrans11 = qtrans_dev[0];
  T qtrans21 = qtrans_dev[1];
  T qtrans12 = qtrans_dev[2];
  T qtrans22 = qtrans_dev[3];

  for (int i=i0; i<l_rows; i+=gridDim.x*blockDim.x)
    {
    tmp_dev[i] = q_dev[l_rqs-1+i + (lc1-1)*ldq]*qtrans11 + q_dev[l_rqs-1+i + (lc2-1)*ldq]*qtrans21;
    q_dev[l_rqs-1+i + (lc2-1)*ldq] = q_dev[l_rqs-1+i + (lc1-1)*ldq]*qtrans12 + q_dev[l_rqs-1+i + (lc2-1)*ldq]*qtrans22;
    q_dev[l_rqs-1+i + (lc1-1)*ldq] = tmp_dev[i];
    }
}

template <typename T>
void gpu_transform_two_columns(T *q_dev, T *qtrans_dev, T *tmp_dev, int ldq, int l_rows, int l_rqs, int l_rqe, int lc1, int lc2, 
                               int SM_count, int debug, gpuStream_t my_stream){

  dim3 blocks = dim3(SM_count,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_transform_two_columns_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(q_dev, qtrans_dev, tmp_dev, ldq, l_rows, l_rqs, l_rqe, lc1, lc2);
#else
  gpu_transform_two_columns_kernel<<<blocks,threadsPerBlock>>>            (q_dev, qtrans_dev, tmp_dev, ldq, l_rows, l_rqs, l_rqe, lc1, lc2);
#endif
  
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_transform_columns: %s\n",gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _transform_two_columns_FromC)(char dataType, intptr_t q_dev, intptr_t qtrans_dev, intptr_t tmp_dev,
                                                                     int ldq, int l_rows, int l_rqs, int l_rqe, int lc1, int lc2, 
                                                                     int SM_count, int debug, gpuStream_t my_stream){

  if      (dataType=='D') gpu_transform_two_columns<double>((double *) q_dev, (double *) qtrans_dev, (double *) tmp_dev, 
                                             ldq, l_rows, l_rqs, l_rqe, lc1, lc2, SM_count, debug, my_stream);
  else if (dataType=='S') gpu_transform_two_columns<float> ((float  *) q_dev, (float  *) qtrans_dev, (float  *) tmp_dev,
                                             ldq, l_rows, l_rqs, l_rqe, lc1, lc2, SM_count, debug, my_stream);
  else {
    printf("Error in elpa_transform_two_columns: Unsupported data type\n");
  }
}
