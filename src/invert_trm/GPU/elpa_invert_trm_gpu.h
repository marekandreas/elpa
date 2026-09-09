//    Copyright 2021, A. Marek
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

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

// Set a device element to zero; for complex types both components are zeroed.
template <typename T>
__device__ inline void set_to_zero(T &v)
{
    if constexpr (std::is_same_v<T, gpuDoubleComplex> || std::is_same_v<T, gpuFloatComplex>) {
        v.x = 0;
        v.y = 0;
    } else {
        v = (T)0;
    }
}

//_____________________________________________________________________________
// gpu_copy_a_tmat2

template <typename T>
__global__ void gpu_copy_a_tmat2_kernel(T *a_dev, T *tmat2_dev, const int nblk, const int matrixRows, const int l_colx, const int l_row1, const int nb)
{
  int l_col_index = blockIdx.x  + 1; // range 1..l_colx-l_cols-1

  for (int nb_index = threadIdx.x + 1; nb_index <= nb; nb_index += blockDim.x) {
    tmat2_dev[nb_index-1 + (l_colx-1 + l_col_index-1) * nblk] =
      a_dev[l_row1-1 + nb_index-1 + (l_colx-1 + l_col_index-1) * matrixRows];
  }
}

#ifdef WITH_AMD_GPU_VERSION
#define BLK_X_DIM 32
#define BLK_Y_DIM 4

template <typename T>
__global__ void gpu_copy_a_tmat2_block_kernel(T const* __restrict__ a_dev, T* __restrict__ tmat2_dev,
                                               const int nblk, const int matrixRows, const int l_colx,
                                               const int l_row1)
{
  int x = blockIdx.x * BLK_X_DIM + threadIdx.x;
  int y = blockIdx.y * BLK_X_DIM + threadIdx.y;

  for (int j = 0; j < BLK_X_DIM; j += BLK_Y_DIM) {
    tmat2_dev[(l_colx - 1 + y + j) * nblk + x] =
      a_dev[(l_colx - 1 + y + j) * matrixRows + x + l_row1 - 1];
  }
}
#endif

template <typename T>
void gpu_copy_a_tmat2(T *a_dev, T *tmat2_dev, const int nblk, const int matrixRows,
                      const int l_cols, const int l_colx, const int l_row1, const int nb,
                      gpuStream_t my_stream)
{
#ifdef WITH_AMD_GPU_VERSION
  if ((l_cols-l_colx+1) % BLK_X_DIM == 0 && nb % BLK_X_DIM == 0) {
    dim3 blocks = dim3(nb / BLK_X_DIM, (l_cols-l_colx+1) / BLK_X_DIM);
    dim3 threadsPerBlock = dim3(BLK_X_DIM, BLK_Y_DIM);
#ifdef WITH_GPU_STREAMS
    gpu_copy_a_tmat2_block_kernel<T><<<blocks,threadsPerBlock,0,my_stream>>>(a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#else
    gpu_copy_a_tmat2_block_kernel<T><<<blocks,threadsPerBlock>>>(a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#endif
  }
  else
#endif
  {
    dim3 blocks = dim3(l_cols-l_colx+1,1,1);
    dim3 threadsPerBlock = dim3(min(nb, MAX_THREADS_PER_BLOCK), 1, 1);
#ifdef WITH_GPU_STREAMS
    gpu_copy_a_tmat2_kernel<T><<<blocks,threadsPerBlock,0,my_stream>>>(a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1, nb);
#else
    gpu_copy_a_tmat2_kernel<T><<<blocks,threadsPerBlock>>>(a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1, nb);
#endif
  }

  gpuError_t cuerr = gpuGetLastError();
  if (cuerr != gpuSuccess){
    printf("Error in executing gpu_copy_a_tmat2_kernel: %s\n",gpuGetErrorString(cuerr));
  }
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_double_a_tmat2_FromC)(double *a_dev, double *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, gpuStream_t my_stream){
  gpu_copy_a_tmat2<double>(a_dev, tmat2_dev, *nblk_in, *matrixRows_in, *l_cols_in, *l_colx_in, *l_row1_in, *nb_in, my_stream);
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_float_a_tmat2_FromC)(float *a_dev, float *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, gpuStream_t my_stream){
  gpu_copy_a_tmat2<float>(a_dev, tmat2_dev, *nblk_in, *matrixRows_in, *l_cols_in, *l_colx_in, *l_row1_in, *nb_in, my_stream);
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_double_complex_a_tmat2_FromC)(double _Complex *a_dev, double _Complex *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, gpuStream_t my_stream){
  gpu_copy_a_tmat2<gpuDoubleComplex>((gpuDoubleComplex*) a_dev, (gpuDoubleComplex*) tmat2_dev, *nblk_in, *matrixRows_in, *l_cols_in, *l_colx_in, *l_row1_in, *nb_in, my_stream);
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_float_complex_a_tmat2_FromC)(float _Complex *a_dev, float _Complex *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, gpuStream_t my_stream){
  gpu_copy_a_tmat2<gpuFloatComplex>((gpuFloatComplex*) a_dev, (gpuFloatComplex*) tmat2_dev, *nblk_in, *matrixRows_in, *l_cols_in, *l_colx_in, *l_row1_in, *nb_in, my_stream);
}

//_____________________________________________________________________________
// gpu_copy_tmp2_tmat2

template <typename T>
__device__ void gpu_copy_tmp2_tmat2_kernel_body(T *tmp2_dev, T *tmat2_dev, const int nblk, const int l_col1, const int nb)
{
  int l_col_index = blockIdx.x  + 1; // range 1..nb

  for (int nb_index = threadIdx.x + 1; nb_index <= nb; nb_index += blockDim.x) {
    tmat2_dev[nb_index-1 + (l_col1-1 + l_col_index-1) * nblk] =
      tmp2_dev[nb_index-1 + (1-1 + l_col_index-1) * nblk];
  }
}

__global__ void gpu_copy_double_tmp2_tmat2_kernel(double *tmp2_dev, double *tmat2_dev, const int nblk, const int l_col1, const int nb)
{ gpu_copy_tmp2_tmat2_kernel_body(tmp2_dev, tmat2_dev, nblk, l_col1, nb); }

__global__ void gpu_copy_float_tmp2_tmat2_kernel(float *tmp2_dev, float *tmat2_dev, const int nblk, const int l_col1, const int nb)
{ gpu_copy_tmp2_tmat2_kernel_body(tmp2_dev, tmat2_dev, nblk, l_col1, nb); }

__global__ void gpu_copy_double_complex_tmp2_tmat2_kernel(gpuDoubleComplex *tmp2_dev, gpuDoubleComplex *tmat2_dev, const int nblk, const int l_col1, const int nb)
{ gpu_copy_tmp2_tmat2_kernel_body(tmp2_dev, tmat2_dev, nblk, l_col1, nb); }

__global__ void gpu_copy_float_complex_tmp2_tmat2_kernel(gpuFloatComplex *tmp2_dev, gpuFloatComplex *tmat2_dev, const int nblk, const int l_col1, const int nb)
{ gpu_copy_tmp2_tmat2_kernel_body(tmp2_dev, tmat2_dev, nblk, l_col1, nb); }

extern "C" void CONCATENATE(ELPA_GPU, _copy_double_tmp2_tmat2_FromC)(double *tmp2_dev, double *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, gpuStream_t my_stream){
  int nblk   = *nblk_in;
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;

  dim3 blocks = dim3(nb,1,1);
  dim3 threadsPerBlock = dim3(min(nb, MAX_THREADS_PER_BLOCK), 1, 1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_double_tmp2_tmat2_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(tmp2_dev, tmat2_dev, nblk, l_col1, nb);
#else
  gpu_copy_double_tmp2_tmat2_kernel<<<blocks,threadsPerBlock>>>(tmp2_dev, tmat2_dev, nblk, l_col1, nb);
#endif
  gpuError_t cuerr = gpuGetLastError();
  if (cuerr != gpuSuccess){
    printf("Error in executing gpu_copy_double_tmp2_tmat2_kernel: %s\n",gpuGetErrorString(cuerr));
  }
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_float_tmp2_tmat2_FromC)(float *tmp2_dev, float *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, gpuStream_t my_stream){
  int nblk   = *nblk_in;
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;

  dim3 blocks = dim3(nb,1,1);
  dim3 threadsPerBlock = dim3(min(nb, MAX_THREADS_PER_BLOCK), 1, 1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_float_tmp2_tmat2_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(tmp2_dev, tmat2_dev, nblk, l_col1, nb);
#else
  gpu_copy_float_tmp2_tmat2_kernel<<<blocks,threadsPerBlock>>>(tmp2_dev, tmat2_dev, nblk, l_col1, nb);
#endif
  gpuError_t cuerr = gpuGetLastError();
  if (cuerr != gpuSuccess){
    printf("Error in executing gpu_copy_float_tmp2_tmat2_kernel: %s\n",gpuGetErrorString(cuerr));
  }
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_double_complex_tmp2_tmat2_FromC)(double _Complex *tmp2_dev, double _Complex *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, gpuStream_t my_stream){
  int nblk   = *nblk_in;
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;

  gpuDoubleComplex* tmp2_casted  = (gpuDoubleComplex*) tmp2_dev;
  gpuDoubleComplex* tmat2_casted = (gpuDoubleComplex*) tmat2_dev;

  dim3 blocks = dim3(nb,1,1);
  dim3 threadsPerBlock = dim3(min(nb, MAX_THREADS_PER_BLOCK), 1, 1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_double_complex_tmp2_tmat2_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(tmp2_casted, tmat2_casted, nblk, l_col1, nb);
#else
  gpu_copy_double_complex_tmp2_tmat2_kernel<<<blocks,threadsPerBlock>>>(tmp2_casted, tmat2_casted, nblk, l_col1, nb);
#endif
  gpuError_t cuerr = gpuGetLastError();
  if (cuerr != gpuSuccess){
    printf("Error in executing gpu_copy_double_complex_tmp2_tmat2_kernel: %s\n",gpuGetErrorString(cuerr));
  }
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_float_complex_tmp2_tmat2_FromC)(float _Complex *tmp2_dev, float _Complex *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, gpuStream_t my_stream){
  int nblk   = *nblk_in;
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;

  gpuFloatComplex* tmp2_casted  = (gpuFloatComplex*) tmp2_dev;
  gpuFloatComplex* tmat2_casted = (gpuFloatComplex*) tmat2_dev;

  dim3 blocks = dim3(nb,1,1);
  dim3 threadsPerBlock = dim3(min(nb, MAX_THREADS_PER_BLOCK), 1, 1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_float_complex_tmp2_tmat2_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(tmp2_casted, tmat2_casted, nblk, l_col1, nb);
#else
  gpu_copy_float_complex_tmp2_tmat2_kernel<<<blocks,threadsPerBlock>>>(tmp2_casted, tmat2_casted, nblk, l_col1, nb);
#endif
  gpuError_t cuerr = gpuGetLastError();
  if (cuerr != gpuSuccess){
    printf("Error in executing gpu_copy_float_complex_tmp2_tmat2_kernel: %s\n",gpuGetErrorString(cuerr));
  }
}

//_____________________________________________________________________________
// gpu_copy_a_tmat1
// Copies a block of a_dev into tmat1_dev and zeros the source elements.
// The zeroing uses set_to_zero<T> to handle both real and complex types.

template <typename T>
__global__ void gpu_copy_a_tmat1_kernel(T *a_dev, T *tmat1_dev, const int l_rows, const int matrixRows, const int l_col1, const int nb, const int l_row1)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y;

  if (row < l_row1 - 1) {
    tmat1_dev[row + col * l_rows] =
      a_dev[row + (l_col1-1 + col) * matrixRows];
    set_to_zero(a_dev[row + (l_col1-1 + col) * matrixRows]);
  }
}

template <typename T>
void gpu_copy_a_tmat1(T *a_dev, T *tmat1_dev, const int l_rows, const int matrixRows,
                      const int nb, const int l_row1, const int l_col1, gpuStream_t my_stream)
{
  dim3 threadsPerBlock = dim3(min(l_row1-1, MAX_THREADS_PER_BLOCK), 1, 1);
  dim3 blocks = dim3((l_row1-2)/threadsPerBlock.x+1,nb,1);
#ifdef WITH_GPU_STREAMS
  gpu_copy_a_tmat1_kernel<T><<<blocks,threadsPerBlock,0,my_stream>>>(a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#else
  gpu_copy_a_tmat1_kernel<T><<<blocks,threadsPerBlock>>>(a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#endif

  gpuError_t cuerr = gpuGetLastError();
  if (cuerr != gpuSuccess){
    printf("Error in executing gpu_copy_a_tmat1_kernel: %s\n",gpuGetErrorString(cuerr));
  }
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_double_a_tmat1_FromC)(double *a_dev, double *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, gpuStream_t my_stream){
  gpu_copy_a_tmat1<double>(a_dev, tmat1_dev, *l_rows_in, *matrixRows_in, *nb_in, *l_row1_in, *l_col1_in, my_stream);
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_float_a_tmat1_FromC)(float *a_dev, float *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, gpuStream_t my_stream){
  gpu_copy_a_tmat1<float>(a_dev, tmat1_dev, *l_rows_in, *matrixRows_in, *nb_in, *l_row1_in, *l_col1_in, my_stream);
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_double_complex_a_tmat1_FromC)(double _Complex *a_dev, double _Complex *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, gpuStream_t my_stream){
  gpu_copy_a_tmat1<gpuDoubleComplex>((gpuDoubleComplex*) a_dev, (gpuDoubleComplex*) tmat1_dev, *l_rows_in, *matrixRows_in, *nb_in, *l_row1_in, *l_col1_in, my_stream);
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_float_complex_a_tmat1_FromC)(float _Complex *a_dev, float _Complex *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, gpuStream_t my_stream){
  gpu_copy_a_tmat1<gpuFloatComplex>((gpuFloatComplex*) a_dev, (gpuFloatComplex*) tmat1_dev, *l_rows_in, *matrixRows_in, *nb_in, *l_row1_in, *l_col1_in, my_stream);
}

//_____________________________________________________________________________
// gpu_copy_tmp1_tmp2
// Scatter the lower-triangular packed tmp1 into the full column-major tmp2.

template <typename T>
__device__ void gpu_copy_tmp1_tmp2_kernel_body(T *tmp1_dev, T *tmp2_dev, const int nblk, const int nb)
{
  int i_index = blockIdx.x + 1; // range 1..nb

  for (int j_index = threadIdx.x + 1; j_index <= i_index; j_index += blockDim.x) {
    tmp2_dev[1-1 + j_index-1 + (i_index-1) * nblk] =
      tmp1_dev[(i_index*(i_index+1) - 2*i_index) / 2 + 1 - 1 + j_index - 1];
  }
}

__global__ void gpu_copy_double_tmp1_tmp2_kernel(double *tmp1_dev, double *tmp2_dev, const int nblk, const int nb)
{ gpu_copy_tmp1_tmp2_kernel_body(tmp1_dev, tmp2_dev, nblk, nb); }

__global__ void gpu_copy_float_tmp1_tmp2_kernel(float *tmp1_dev, float *tmp2_dev, const int nblk, const int nb)
{ gpu_copy_tmp1_tmp2_kernel_body(tmp1_dev, tmp2_dev, nblk, nb); }

__global__ void gpu_copy_double_complex_tmp1_tmp2_kernel(gpuDoubleComplex *tmp1_dev, gpuDoubleComplex *tmp2_dev, const int nblk, const int nb)
{ gpu_copy_tmp1_tmp2_kernel_body(tmp1_dev, tmp2_dev, nblk, nb); }

__global__ void gpu_copy_float_complex_tmp1_tmp2_kernel(gpuFloatComplex *tmp1_dev, gpuFloatComplex *tmp2_dev, const int nblk, const int nb)
{ gpu_copy_tmp1_tmp2_kernel_body(tmp1_dev, tmp2_dev, nblk, nb); }

extern "C" void CONCATENATE(ELPA_GPU, _copy_double_tmp1_tmp2_FromC)(double *tmp1_dev, double *tmp2_dev, int *nblk_in, int *nb_in, gpuStream_t my_stream){
  int nblk = *nblk_in;
  int nb = *nb_in;

  dim3 threadsPerBlock = dim3(min(nb, MAX_THREADS_PER_BLOCK), 1, 1);
  dim3 blocks = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_double_tmp1_tmp2_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(tmp1_dev, tmp2_dev, nblk, nb);
#else
  gpu_copy_double_tmp1_tmp2_kernel<<<blocks,threadsPerBlock>>>(tmp1_dev, tmp2_dev, nblk, nb);
#endif
  gpuError_t cuerr = gpuGetLastError();
  if (cuerr != gpuSuccess){
    printf("Error in executing gpu_copy_double_tmp1_tmp2_kernel: %s\n",gpuGetErrorString(cuerr));
  }
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_float_tmp1_tmp2_FromC)(float *tmp1_dev, float *tmp2_dev, int *nblk_in, int *nb_in, gpuStream_t my_stream){
  int nblk = *nblk_in;
  int nb = *nb_in;

  dim3 threadsPerBlock = dim3(min(nb, MAX_THREADS_PER_BLOCK), 1, 1);
  dim3 blocks = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_float_tmp1_tmp2_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(tmp1_dev, tmp2_dev, nblk, nb);
#else
  gpu_copy_float_tmp1_tmp2_kernel<<<blocks,threadsPerBlock>>>(tmp1_dev, tmp2_dev, nblk, nb);
#endif
  gpuError_t cuerr = gpuGetLastError();
  if (cuerr != gpuSuccess){
    printf("Error in executing gpu_copy_float_tmp1_tmp2_kernel: %s\n",gpuGetErrorString(cuerr));
  }
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_double_complex_tmp1_tmp2_FromC)(double _Complex *tmp1_dev, double _Complex *tmp2_dev, int *nblk_in, int *nb_in, gpuStream_t my_stream){
  int nblk = *nblk_in;
  int nb = *nb_in;

  gpuDoubleComplex* tmp1_casted = (gpuDoubleComplex*) tmp1_dev;
  gpuDoubleComplex* tmp2_casted = (gpuDoubleComplex*) tmp2_dev;

  dim3 threadsPerBlock = dim3(min(nb, MAX_THREADS_PER_BLOCK), 1, 1);
  dim3 blocks = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_double_complex_tmp1_tmp2_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(tmp1_casted, tmp2_casted, nblk, nb);
#else
  gpu_copy_double_complex_tmp1_tmp2_kernel<<<blocks,threadsPerBlock>>>(tmp1_casted, tmp2_casted, nblk, nb);
#endif
  gpuError_t cuerr = gpuGetLastError();
  if (cuerr != gpuSuccess){
    printf("Error in executing gpu_copy_double_complex_tmp1_tmp2_kernel: %s\n",gpuGetErrorString(cuerr));
  }
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_float_complex_tmp1_tmp2_FromC)(float _Complex *tmp1_dev, float _Complex *tmp2_dev, int *nblk_in, int *nb_in, gpuStream_t my_stream){
  int nblk = *nblk_in;
  int nb = *nb_in;

  gpuFloatComplex* tmp1_casted = (gpuFloatComplex*) tmp1_dev;
  gpuFloatComplex* tmp2_casted = (gpuFloatComplex*) tmp2_dev;

  dim3 threadsPerBlock = dim3(min(nb, MAX_THREADS_PER_BLOCK), 1, 1);
  dim3 blocks = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_float_complex_tmp1_tmp2_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(tmp1_casted, tmp2_casted, nblk, nb);
#else
  gpu_copy_float_complex_tmp1_tmp2_kernel<<<blocks,threadsPerBlock>>>(tmp1_casted, tmp2_casted, nblk, nb);
#endif
  gpuError_t cuerr = gpuGetLastError();
  if (cuerr != gpuSuccess){
    printf("Error in executing gpu_copy_float_complex_tmp1_tmp2_kernel: %s\n",gpuGetErrorString(cuerr));
  }
}

//_____________________________________________________________________________
// gpu_copy_a_tmp1
// Gather the lower-triangular block from a_dev into packed tmp1_dev.

template <typename T>
__device__ void gpu_copy_a_tmp1_kernel_body(T *a_dev, T *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb)
{
  int i_index = blockIdx.x + 1; // range 1..nb

  for (int j_index = threadIdx.x + 1; j_index <= i_index; j_index += blockDim.x) {
    tmp1_dev[(i_index*(i_index+1) - 2*i_index) / 2 + 1 - 1 + j_index - 1] =
      a_dev[l_row1-1 + j_index-1 + (l_col1-1 + i_index-1) * matrixRows];
  }
}

__global__ void gpu_copy_double_a_tmp1_kernel(double *a_dev, double *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb)
{ gpu_copy_a_tmp1_kernel_body(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb); }

__global__ void gpu_copy_float_a_tmp1_kernel(float *a_dev, float *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb)
{ gpu_copy_a_tmp1_kernel_body(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb); }

__global__ void gpu_copy_double_complex_a_tmp1_kernel(gpuDoubleComplex *a_dev, gpuDoubleComplex *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb)
{ gpu_copy_a_tmp1_kernel_body(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb); }

__global__ void gpu_copy_float_complex_a_tmp1_kernel(gpuFloatComplex *a_dev, gpuFloatComplex *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb)
{ gpu_copy_a_tmp1_kernel_body(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb); }

extern "C" void CONCATENATE(ELPA_GPU, _copy_double_a_tmp1_FromC)(double *a_dev, double *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, gpuStream_t my_stream){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;

  dim3 threadsPerBlock = dim3(min(nb, MAX_THREADS_PER_BLOCK), 1, 1);
  dim3 blocks = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_double_a_tmp1_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb);
#else
  gpu_copy_double_a_tmp1_kernel<<<blocks,threadsPerBlock>>>(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb);
#endif
  gpuError_t cuerr = gpuGetLastError();
  if (cuerr != gpuSuccess){
    printf("Error in executing gpu_copy_double_a_tmp1_kernel: %s\n",gpuGetErrorString(cuerr));
  }
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_float_a_tmp1_FromC)(float *a_dev, float *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, gpuStream_t my_stream){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;

  dim3 threadsPerBlock = dim3(min(nb, MAX_THREADS_PER_BLOCK), 1, 1);
  dim3 blocks = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_float_a_tmp1_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb);
#else
  gpu_copy_float_a_tmp1_kernel<<<blocks,threadsPerBlock>>>(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb);
#endif
  gpuError_t cuerr = gpuGetLastError();
  if (cuerr != gpuSuccess){
    printf("Error in executing gpu_copy_float_a_tmp1_kernel: %s\n",gpuGetErrorString(cuerr));
  }
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_double_complex_a_tmp1_FromC)(double _Complex *a_dev, double _Complex *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, gpuStream_t my_stream){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;

  gpuDoubleComplex* a_casted    = (gpuDoubleComplex*) a_dev;
  gpuDoubleComplex* tmp1_casted = (gpuDoubleComplex*) tmp1_dev;

  dim3 threadsPerBlock = dim3(min(nb, MAX_THREADS_PER_BLOCK), 1, 1);
  dim3 blocks = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_double_complex_a_tmp1_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(a_casted, tmp1_casted, l_row1, l_col1, matrixRows, nb);
#else
  gpu_copy_double_complex_a_tmp1_kernel<<<blocks,threadsPerBlock>>>(a_casted, tmp1_casted, l_row1, l_col1, matrixRows, nb);
#endif
  gpuError_t cuerr = gpuGetLastError();
  if (cuerr != gpuSuccess){
    printf("Error in executing gpu_copy_double_complex_a_tmp1_kernel: %s\n",gpuGetErrorString(cuerr));
  }
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_float_complex_a_tmp1_FromC)(float _Complex *a_dev, float _Complex *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, gpuStream_t my_stream){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;

  gpuFloatComplex* a_casted    = (gpuFloatComplex*) a_dev;
  gpuFloatComplex* tmp1_casted = (gpuFloatComplex*) tmp1_dev;

  dim3 threadsPerBlock = dim3(min(nb, MAX_THREADS_PER_BLOCK), 1, 1);
  dim3 blocks = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_float_complex_a_tmp1_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(a_casted, tmp1_casted, l_row1, l_col1, matrixRows, nb);
#else
  gpu_copy_float_complex_a_tmp1_kernel<<<blocks,threadsPerBlock>>>(a_casted, tmp1_casted, l_row1, l_col1, matrixRows, nb);
#endif
  gpuError_t cuerr = gpuGetLastError();
  if (cuerr != gpuSuccess){
    printf("Error in executing gpu_copy_float_complex_a_tmp1_kernel: %s\n",gpuGetErrorString(cuerr));
  }
}
