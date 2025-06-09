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
__global__ void gpu_update_d_kernel(T *d, T *e, int *limits, const int ndiv, const int na) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  for (int ii=0; ii<ndiv-1; ii++) {
    int n = limits[ii]-1;
    d[n]   = d[n]   - fabs(e[n]);
    d[n+1] = d[n+1] - fabs(e[n]);
  }
}


template <typename T>
void gpu_update_d(T *d_dev, T *e_dev, int *limits_dev, int ndiv, int na, int debug, gpuStream_t my_stream) {

  dim3 threadsPerBlock(1);
  dim3 blocks(1);

#ifdef WITH_GPU_STREAMS
  gpu_update_d_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(d_dev, e_dev, limits_dev, ndiv, na);
#else
  gpu_update_d_kernel<<<blocks,threadsPerBlock>>>            (d_dev, e_dev, limits_dev, ndiv, na);
#endif
  
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_update_d: %s\n",gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _update_d_FromC)(char dataType, intptr_t d_dev, intptr_t e_dev, intptr_t limits_dev,
                                                        int ndiv, int na, int debug, gpuStream_t my_stream) {
  if      (dataType=='D') gpu_update_d<double>((double *) d_dev, (double *) e_dev, (int *) limits_dev, ndiv, na, debug, my_stream);
  else if (dataType=='S') gpu_update_d<float> ((float  *) d_dev, (float  *) e_dev, (int *) limits_dev, ndiv, na, debug, my_stream);
  else {
    printf("Error in elpa_update_d: Unsupported data type\n");
  }
}

//________________________________________________________________

template <typename T>
__global__ void gpu_copy_qmat1_to_qmat2_kernel(T *qmat1, T *qmat2, const int max_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= 0 && i < max_size) {
    if ( j>=0 && j < max_size) {
      qmat2[i + max_size*j] = qmat1[i + max_size*j];
    }
  }
}

template <typename T>
void gpu_copy_qmat1_to_qmat2(T *qmat1_dev, T *qmat2_dev, int max_size, int debug, gpuStream_t my_stream) {

  dim3 threadsPerBlock(32,32);
  dim3 blocks((max_size + threadsPerBlock.x - 1) / threadsPerBlock.x, (max_size + threadsPerBlock.y - 1) / threadsPerBlock.y);


#ifdef WITH_GPU_STREAMS
  gpu_copy_qmat1_to_qmat2_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(qmat1_dev, qmat2_dev, max_size);
#else
  gpu_copy_qmat1_to_qmat2_kernel<<<blocks,threadsPerBlock>>>            (qmat1_dev, qmat2_dev, max_size);
#endif
  
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_copy_qmat1_to_qmat2: %s\n",gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _copy_qmat1_to_qmat2_FromC)(char dataType, intptr_t qmat1_dev, intptr_t qmat2_dev, 
                                                                   int max_size, int debug, gpuStream_t my_stream) {
  if      (dataType=='D') gpu_copy_qmat1_to_qmat2<double>((double *) qmat1_dev, (double *) qmat2_dev, max_size, debug, my_stream);
  else if (dataType=='S') gpu_copy_qmat1_to_qmat2<float> ((float  *) qmat1_dev, (float  *) qmat2_dev, max_size, debug, my_stream);
  else {
    printf("Error in elpa_copy_qmat1_to_qmat2: Unsupported data type\n");
  }
}
