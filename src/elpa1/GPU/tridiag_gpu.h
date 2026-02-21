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

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#define MIN(x, y) (((x) < (y)) ? (x) : (y))


// A helper struct to determine if a type is a pointer
template <typename T>
struct is_pointer { static const bool value = false; };

template <typename T>
struct is_pointer<T*> { static const bool value = true; };

//_________________________________________________________________________________________________

template <typename T, typename T_real>
__global__ void gpu_copy_and_set_zeros_kernel(T *v_row_dev, T *u_col_dev, T const *a_dev,
                                              T *aux1_dev, T *vav_dev, T_real *d_vec_dev,
                                              int l_rows, int l_cols, int matrixRows, int istep,
                                              int isOurProcessRow, int isOurProcessCol, int isOurProcessCol_prev, int isSkewsymmetric, int useCCL){
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

  if (isOurProcessCol_prev)
    {
    for (int i_row=tid; i_row<l_rows; i_row+=blockDim.x*gridDim.x)
      v_row_dev[i_row] = a_dev[i_row + matrixRows*l_cols];
    }

  //u_col_dev
  if (l_cols>l_rows)
    {
    T zero = elpaDeviceNumber<T>(0.0);
    for (int i=l_rows+tid; i<l_cols; i+=blockDim.x*gridDim.x)
      u_col_dev[i] = zero;
    }

  // set zeros for aux1_dev and vav_dev to be summed with atomicAdd
  if (tid==0)
    {
    aux1_dev[0] = elpaDeviceNumber<T>(0.0);
    if (useCCL) *vav_dev = elpaDeviceNumber<T>(0.0);

    if (isOurProcessRow && isOurProcessCol)
      {
      if (isSkewsymmetric)
        d_vec_dev[istep-1-1] = 0.0;
      else
        d_vec_dev[istep-1-1] = elpaDeviceRealPart(a_dev[(l_rows-1) + matrixRows*(l_cols-1)]);
      }
    }
}

template <typename T, typename T_real>
void gpu_copy_and_set_zeros(T *v_row_dev, T *u_col_dev, T *a_dev, 
                            T *aux1_dev, T *vav_dev, T_real *d_vec_dev, 
                            int l_rows, int l_cols, int matrixRows, int istep, 
                            int isOurProcessRow,  int isOurProcessCol, int isOurProcessCol_prev, 
                            int isSkewsymmetric, int useCCL, int wantDebug, int SM_count, gpuStream_t my_stream){

  int threads, blocks;

  threads = MIN_THREADS_PER_BLOCK;
  blocks = SM_count;

  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(threads,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_and_set_zeros_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(v_row_dev, u_col_dev, a_dev, aux1_dev, vav_dev, d_vec_dev, 
                                                                        l_rows, l_cols, matrixRows, istep, isOurProcessRow, isOurProcessCol, 
                                                                        isOurProcessCol_prev, isSkewsymmetric, useCCL);
#else
  gpu_copy_and_set_zeros_kernel<<<blocks,threadsPerBlock>>>             (v_row_dev, u_col_dev, a_dev, aux1_dev, vav_dev, d_vec_dev, 
                                                                        l_rows, l_cols, matrixRows, istep, isOurProcessRow, isOurProcessCol, 
                                                                        isOurProcessCol_prev, isSkewsymmetric, useCCL);
#endif

  if (wantDebug){
#ifdef WITH_GPU_STREAMS
    gpuStreamSynchronize(my_stream);
#else
    gpuDeviceSynchronize();
#endif
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess) printf("Error in executing gpu_copy_and_set_zeros: %s\n", gpuGetErrorString(gpuerr));
  }
}


extern "C" void CONCATENATE(ELPA_GPU,  _copy_and_set_zeros_FromC)(char dataType, intptr_t v_row_dev, intptr_t u_col_dev, intptr_t a_dev, 
                            double *aux1_dev, double *vav_dev, double *d_vec_dev, 
                            int l_rows, int l_cols, int matrixRows, int istep,
                            int isOurProcessRow, int isOurProcessCol, int isOurProcessCol_prev, 
                            int isSkewsymmetric, int useCCL, int wantDebug, int SM_count, gpuStream_t my_stream){
  if      (dataType=='D') gpu_copy_and_set_zeros<double, double> ((double *)v_row_dev, (double *)u_col_dev, (double *)a_dev,
                            (double *)aux1_dev, (double *)vav_dev, (double *)d_vec_dev, 
                            l_rows, l_cols, matrixRows, istep, isOurProcessRow, isOurProcessCol, 
                            isOurProcessCol_prev, isSkewsymmetric, useCCL, wantDebug, SM_count, my_stream);
  else if (dataType=='S') gpu_copy_and_set_zeros<float, float>   ((float  *)v_row_dev, (float  *)u_col_dev, (float  *)a_dev,
                            (float  *)aux1_dev, (float  *)vav_dev, (float  *)d_vec_dev, 
                            l_rows, l_cols, matrixRows, istep, isOurProcessRow, isOurProcessCol, 
                            isOurProcessCol_prev, isSkewsymmetric, useCCL, wantDebug, SM_count, my_stream);
  else if (dataType=='Z') gpu_copy_and_set_zeros<gpuDoubleComplex, double> ((gpuDoubleComplex *)v_row_dev, (gpuDoubleComplex *)u_col_dev, (gpuDoubleComplex *)a_dev,
                            (gpuDoubleComplex *)aux1_dev, (gpuDoubleComplex *)vav_dev, (double *)d_vec_dev, 
                            l_rows, l_cols, matrixRows, istep, isOurProcessRow, isOurProcessCol, 
                            isOurProcessCol_prev, isSkewsymmetric, useCCL, wantDebug, SM_count, my_stream);
  else if (dataType=='C') gpu_copy_and_set_zeros<gpuFloatComplex, float>   ((gpuFloatComplex  *)v_row_dev, (gpuFloatComplex  *)u_col_dev, (gpuFloatComplex  *)a_dev,
                            (gpuFloatComplex  *)aux1_dev, (gpuFloatComplex  *)vav_dev, (float  *)d_vec_dev, 
                            l_rows, l_cols, matrixRows, istep, isOurProcessRow, isOurProcessCol, 
                            isOurProcessCol_prev, isSkewsymmetric, useCCL, wantDebug, SM_count, my_stream);
  else {
    printf("Error in gpu_copy_and_set_zeros_FromC: Unsupported data type\n");
  }
}

//_________________________________________________________________________________________________
// device syncronization is needed afterwards, e.g. gpu_memcpy
// the kernel used only in NCCL codepath

// result_dev[0] must be initialized to zero before calling the kernel
// Initializing result_dev[0]=0 inside the kernel can be unsafe: 
// if block-0 is already finished, but block-1 (or block-n) is just starting, it will overwrite the result of block-0

template <typename T>
__global__ void gpu_dot_product_kernel(int n, T *x_dev, int incx, T *y_dev, int incy, T *result_dev){
  __shared__ T cache[MAX_THREADS_PER_BLOCK]; // extra space of fixed size is reserved for a speedup
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

  T temp = elpaDeviceNumber<T>(0.0);
  int i = tid;
  while (i < n) {
    temp = elpaDeviceAdd(temp, elpaDeviceMultiply(elpaDeviceComplexConjugate(x_dev[i*incx]), y_dev[i*incy])); // temp += x_dev[i*incx] * y_dev[i*incy];
    i += blockDim.x * gridDim.x;
  }

  // set the cache values
  cache[threadIdx.x] = temp;
  // synchronize threads in this block
  __syncthreads();

  // for reductions, threadsPerBlock=blockDim.x must be a power of 2
  i = blockDim.x/2;
  while (i > 0) {
    if (threadIdx.x < i) cache[threadIdx.x] = elpaDeviceAdd(cache[threadIdx.x], cache[threadIdx.x + i]);
    __syncthreads();
    i /= 2;
  }

  if (threadIdx.x==0) atomicAdd(&result_dev[0], cache[0]);
  
}

template <typename T>
void gpu_dot_product (int n, T *x_dev, int incx, T *y_dev, int incy, T *result_dev, 
                      int wantDebug, int SM_count, gpuStream_t my_stream){

  int blocks = SM_count;
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1); 

#ifdef WITH_GPU_STREAMS
  gpu_dot_product_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(n, x_dev, incx, y_dev, incy, result_dev);
#else
  gpu_dot_product_kernel<<<blocks,threadsPerBlock>>>            (n, x_dev, incx, y_dev, incy, result_dev);
#endif

  if (wantDebug){
#ifdef WITH_GPU_STREAMS
    gpuStreamSynchronize(my_stream);
#else
    gpuDeviceSynchronize();
#endif
    gpuError_t cuerr = gpuGetLastError();
    if (cuerr != gpuSuccess){
      printf("Error in executing gpu_dot_product_kernel: %s\n",gpuGetErrorString(cuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU, _dot_product_FromC)(char dataType, int n, intptr_t x_dev, int incx, intptr_t y_dev, int incy, intptr_t result_dev, 
                                                          int wantDebug, int SM_count, gpuStream_t my_stream){
  if      (dataType=='D') gpu_dot_product<double>(n, (double *)x_dev, incx, (double *)y_dev, incy, (double *)result_dev, wantDebug, SM_count, my_stream);
  else if (dataType=='S') gpu_dot_product<float> (n, (float  *)x_dev, incx, (float  *)y_dev, incy, (float  *)result_dev, wantDebug, SM_count, my_stream);
  else if (dataType=='Z') gpu_dot_product<gpuDoubleComplex>(n, (gpuDoubleComplex *)x_dev, incx, (gpuDoubleComplex *)y_dev, incy, (gpuDoubleComplex *)result_dev, wantDebug, SM_count, my_stream);
  else if (dataType=='C') gpu_dot_product<gpuFloatComplex> (n, (gpuFloatComplex  *)x_dev, incx, (gpuFloatComplex  *)y_dev, incy, (gpuFloatComplex  *)result_dev, wantDebug, SM_count, my_stream);
  else {
    printf("Error in gpu_dot_product_FromC: Unsupported data type\n");
  }
}

//_________________________________________________________________________________________________

template <typename T>
__global__ void gpu_dot_product_and_assign_kernel(T *v_row_dev, int l_rows, int isOurProcessRow, T *aux1_dev){
  const int threadsPerBlock = MAX_THREADS_PER_BLOCK;
  __shared__ T cache[threadsPerBlock];
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

/*
  if (isOurProcessRow) {
    aux1(1) = dot_product(v_row(1:l_rows-1),v_row(1:l_rows-1)) ! = "q"
    aux1(2) = v_row(l_rows) ! = "a_11" (or rather a_nn)
    }
  else{
    aux1(1) = dot_product(v_row(1:l_rows),v_row(1:l_rows))
    aux1(2) = 0.
  }
*/

  T temp = elpaDeviceNumber<T>(0.0);
  int index_global = tid;
  while (index_global < l_rows-1) {
    temp = elpaDeviceAdd(temp, elpaDeviceMultiply(elpaDeviceComplexConjugate(v_row_dev[index_global]), v_row_dev[index_global])); // temp += v_row_dev[index_global]*v_row_dev[index_global];
    index_global += blockDim.x * gridDim.x;
  }

  // set the cache values
  cache[threadIdx.x] = temp;
  // synchronize threads in this block
  __syncthreads();

  // for reductions, threadsPerBlock must be a power of 2
  int i = blockDim.x/2;
  while (i > 0) {
    if (threadIdx.x < i) cache[threadIdx.x] = elpaDeviceAdd(cache[threadIdx.x], cache[threadIdx.x + i]);
    __syncthreads();
    i /= 2;
  }

  if (threadIdx.x==0) atomicAdd(&aux1_dev[0], cache[0]);
  
  if (tid==0)
    {
    if (isOurProcessRow) 
      {
      aux1_dev[1] = v_row_dev[l_rows-1];
      }
    else
      {
      if (l_rows>0) atomicAdd(&aux1_dev[0], elpaDeviceMultiply(elpaDeviceComplexConjugate(v_row_dev[l_rows-1]), v_row_dev[l_rows-1]));
      aux1_dev[1] = elpaDeviceNumber<T>(0.0);
      }
    }
}

template <typename T>
void gpu_dot_product_and_assign(T *v_row_dev, int l_rows, int isOurProcessRow, T *aux1_dev, 
                                int wantDebug, int SM_count, gpuStream_t my_stream){
  
  dim3 blocksPerGrid = dim3(SM_count,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_dot_product_and_assign_kernel<<<blocksPerGrid,threadsPerBlock,0,my_stream>>>(v_row_dev, l_rows, isOurProcessRow, aux1_dev);
#else
  gpu_dot_product_and_assign_kernel<<<blocksPerGrid,threadsPerBlock>>>            (v_row_dev, l_rows, isOurProcessRow, aux1_dev);
#endif

  if (wantDebug){
#ifdef WITH_GPU_STREAMS
    gpuStreamSynchronize(my_stream);
#else
    gpuDeviceSynchronize();
#endif
    gpuError_t cuerr = gpuGetLastError();
    if (cuerr != gpuSuccess){
      printf("Error in executing gpu_dot_product_and_assign_kernel: %s\n",gpuGetErrorString(cuerr));
    }
  }

}

extern "C" void CONCATENATE(ELPA_GPU, _dot_product_and_assign_FromC)(char dataType, intptr_t v_row_dev, int l_rows, int isOurProcessRow, intptr_t aux1_dev, int wantDebug, int SM_count, gpuStream_t my_stream){
  if      (dataType=='D') gpu_dot_product_and_assign<double>((double *)v_row_dev, l_rows, isOurProcessRow, (double *)aux1_dev, wantDebug, SM_count, my_stream);
  else if (dataType=='S') gpu_dot_product_and_assign<float> ((float  *)v_row_dev, l_rows, isOurProcessRow, (float  *)aux1_dev, wantDebug, SM_count, my_stream);
  else if (dataType=='Z') gpu_dot_product_and_assign<gpuDoubleComplex>((gpuDoubleComplex *)v_row_dev, l_rows, isOurProcessRow, (gpuDoubleComplex *)aux1_dev, wantDebug, SM_count, my_stream);
  else if (dataType=='C') gpu_dot_product_and_assign<gpuFloatComplex> ((gpuFloatComplex  *)v_row_dev, l_rows, isOurProcessRow, (gpuFloatComplex  *)aux1_dev, wantDebug, SM_count, my_stream);
  else {
    printf("Error in gpu_dot_product_and_assign_FromC: Unsupported data type\n");
  }
}

//_________________________________________________________________________________________________

template <typename T, typename T_real, typename T_value_or_pointer>
__global__ void gpu_set_e_vec_scale_set_one_store_v_row_kernel(T_real *e_vec_dev, T *vrl_dev, T *a_dev, T *v_row_dev, T *tau_dev, T_value_or_pointer xf_host_or_dev, 
                                                      int l_rows, int l_cols,  int matrixRows, int istep, int isOurProcessRow, int useCCL){
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

/*
  if (my_prow == prow(istep-1, nblk, np_rows)) then
    if (.not. useCCL) then
#if REALCASE == 1
      e_vec(istep-1) = vrl
#endif
#if COMPLEXCASE == 1
      e_vec(istep-1) = real(vrl,kind=rk)
#endif
    endif ! useCCL
  endif

  call nvtxRangePush("scale v_row *= xf")
  ! Scale v_row and store Householder Vector for back transformation
  v_row(1:l_rows) = v_row(1:l_rows) * xf
  call nvtxRangePop()

  if (my_prow == prow(istep-1, nblk, np_rows)) then
    v_row(l_rows) = 1.
  endif

  ! store Householder Vector for back transformation
  call nvtxRangePush("cpu copy: v_row->a_mat")
  ! update a_mat
  a_mat(1:l_rows,l_cols+1) = v_row(1:l_rows)
  call nvtxRangePop()

  if (.not. useCCL) then
    ! add tau after the end of actuall v_row, to be broadcasted with it
    v_row(l_rows+1) = tau(istep)
  endif
*/

  T xf = convert_to_device(xf_host_or_dev, typename std::conditional<std::is_pointer<T_value_or_pointer>::value, std::true_type, std::false_type>::type());

  if (useCCL && tid==0)
    {
    if (isOurProcessRow) e_vec_dev[istep-1-1] = elpaDeviceRealPart(*vrl_dev);
    v_row_dev[l_rows+1-1] = tau_dev[istep-1];
    }

  int index_global = tid;
  while (index_global < l_rows) {
    v_row_dev[index_global] = elpaDeviceMultiply(v_row_dev[index_global], xf);
    index_global += blockDim.x * gridDim.x;
  }

  if (isOurProcessRow && (index_global - blockDim.x*gridDim.x == l_rows-1)) // last element
    {
    v_row_dev[l_rows-1] = elpaDeviceNumber<T>(1.0);
    }
    
  int i_row = tid;
  while (i_row < l_rows) 
    {
    a_dev[i_row + matrixRows*l_cols] = v_row_dev[i_row];
    i_row += blockDim.x * gridDim.x;
    }

}

template <typename T, typename T_real>
void gpu_set_e_vec_scale_set_one_store_v_row (T_real *e_vec_dev, T *vrl_dev, T *a_dev, 
                                              T *v_row_dev, T *tau_dev, T *xf_host_or_dev, 
                                              int l_rows, int l_cols,  int matrixRows, int istep, 
                                              int isOurProcessRow, int useCCL, int wantDebug, gpuStream_t my_stream){

  int blocks = std::max((l_rows+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK, 1);
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1); // TODO_23_11: change to NB?

  gpuPointerAttributes attributes;
  gpuError_t error = gpuPointerGetAttributes(&attributes, xf_host_or_dev);

  if (error == gpuSuccess) 
    {
#ifdef HAVE_OLD_HIP_TYPESTRUCTURE
    if (attributes.memoryType == gpuMemoryTypeHost)
#else
    if (attributes.type == gpuMemoryTypeHost)
#endif
      {
      T xf_host_value = *xf_host_or_dev;
#ifdef WITH_GPU_STREAMS
      gpu_set_e_vec_scale_set_one_store_v_row_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_value,
                                                                                             l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL);
#else 
      gpu_set_e_vec_scale_set_one_store_v_row_kernel<<<blocks,threadsPerBlock>>>            (e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_value,
                                                                                            l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL);
#endif
      if (wantDebug)
        {
        gpuError_t cuerr = gpuGetLastError();
        if (cuerr != gpuSuccess) printf("Error in executing gpu_set_e_vec_scale_set_one_store_v_row_kernel: %s\n",gpuGetErrorString(cuerr));
        }
      }
#ifdef HAVE_OLD_HIP_TYPESTRUCTURE
    else if (attributes.memoryType == gpuMemoryTypeDevice)
#else
    else if (attributes.type == gpuMemoryTypeDevice)
#endif
      {
#ifdef WITH_GPU_STREAMS
      gpu_set_e_vec_scale_set_one_store_v_row_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev,
                                                                                             l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL);
#else
      gpu_set_e_vec_scale_set_one_store_v_row_kernel<<<blocks,threadsPerBlock>>>            (e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev,
                                                                                             l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL);
#endif
      if (wantDebug)
        {
#ifdef WITH_GPU_STREAMS
        gpuStreamSynchronize(my_stream);
#else
        gpuDeviceSynchronize();
#endif
        gpuError_t cuerr = gpuGetLastError();
        if (cuerr != gpuSuccess) printf("Error in executing gpu_set_e_vec_scale_set_one_store_v_row_kernel: %s\n",gpuGetErrorString(cuerr));
        }
      }
    } 
  
  else 
    {
    printf("Error: Pointer type is unknown\n");
    }

}

extern "C" void CONCATENATE(ELPA_GPU, _set_e_vec_scale_set_one_store_v_row_FromC) (char dataType, intptr_t e_vec_dev, intptr_t vrl_dev, intptr_t a_dev, intptr_t v_row_dev, intptr_t tau_dev, intptr_t xf_host_or_dev, 
                                                                      int l_rows, int l_cols,  int matrixRows, int istep, int isOurProcessRow, int useCCL, int wantDebug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_set_e_vec_scale_set_one_store_v_row<double,double>((double *)e_vec_dev, (double *)vrl_dev, (double *)a_dev, (double *)v_row_dev, (double *)tau_dev, (double *)xf_host_or_dev, l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, wantDebug, my_stream);
  else if (dataType=='S') gpu_set_e_vec_scale_set_one_store_v_row<float, float> ((float  *)e_vec_dev, (float  *)vrl_dev, (float  *)a_dev, (float  *)v_row_dev, (float  *)tau_dev, (float  *)xf_host_or_dev, l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, wantDebug, my_stream);
  else if (dataType=='Z') gpu_set_e_vec_scale_set_one_store_v_row<gpuDoubleComplex, double>((double *)e_vec_dev, (gpuDoubleComplex *)vrl_dev, (gpuDoubleComplex *)a_dev, (gpuDoubleComplex *)v_row_dev, (gpuDoubleComplex *)tau_dev, (gpuDoubleComplex *)xf_host_or_dev, l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, wantDebug, my_stream);
  else if (dataType=='C') gpu_set_e_vec_scale_set_one_store_v_row<gpuFloatComplex , float> ((float  *)e_vec_dev, (gpuFloatComplex  *)vrl_dev, (gpuFloatComplex  *)a_dev, (gpuFloatComplex  *)v_row_dev, (gpuFloatComplex  *)tau_dev, (gpuFloatComplex  *)xf_host_or_dev, l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, wantDebug, my_stream);
  else {
    printf("Error in gpu_set_e_vec_scale_set_one_store_v_row_FromC: Unsupported data type\n");
  }
}

//_________________________________________________________________________________________________

template <typename T, typename T_value_or_pointer>
__global__ void gpu_store_u_v_in_uv_vu_kernel(T *vu_stored_rows_dev, T *uv_stored_cols_dev, T *v_row_dev, T *u_row_dev,
                T *v_col_dev, T *u_col_dev, T *tau_dev, T *aux_complex_dev, T_value_or_pointer vav_host_or_dev, T_value_or_pointer tau_host_or_dev,
                int l_rows, int l_cols, int n_stored_vecs, int max_local_rows, int max_local_cols, int istep, int useCCL){
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

  T conjg_tau = convert_to_device(tau_host_or_dev, typename std::conditional<std::is_pointer<T_value_or_pointer>::value, std::true_type, std::false_type>::type());
  conjg_tau = elpaDeviceComplexConjugate(conjg_tau);
  
  T conjg_tau_v_row_dev, conjg_tau_v_col_dev;

  // recover tau_dev(istep) after broadcasting
  if (useCCL && tid==0)
    {
    tau_dev[istep-1] = v_row_dev[l_rows+1-1];
    }

/*
  // istep
  if (l_rows > 0) then
    ! update vu_stored_rows
    vu_stored_rows(1:l_rows,2*n_stored_vecs+1) = conjg_tau*v_row(1:l_rows)
    vu_stored_rows(1:l_rows,2*n_stored_vecs+2) = 0.5*conjg_tau*vav*v_row(1:l_rows) - u_row(1:l_rows)
  endif
  if (l_cols > 0) then
    ! update uv_stored_cols
    uv_stored_cols(1:l_cols,2*n_stored_vecs+1) = 0.5*conjg_tau*vav*v_col(1:l_cols) - u_col(1:l_cols)
    uv_stored_cols(1:l_cols,2*n_stored_vecs+2) = conjg_tau*v_col(1:l_cols)
  endif
...

// istep = istep-1
// n_stored_vecs = n_stored_vecs+1
#if COMPLEXCASE == 1
          aux(1:2*n_stored_vecs) = conjg(uv_stored_cols(l_cols+1,1:2*n_stored_vecs))
#endif
*/
  T vav =  convert_to_device(vav_host_or_dev, typename std::conditional<std::is_pointer<T_value_or_pointer>::value, std::true_type, std::false_type>::type());

  int i_row = tid;
  while (i_row < l_rows)
    {
    conjg_tau_v_row_dev = elpaDeviceMultiply(conjg_tau, v_row_dev[i_row]);
    vu_stored_rows_dev[i_row + max_local_rows*(2*n_stored_vecs+0)] = conjg_tau_v_row_dev;
    conjg_tau_v_row_dev = elpaDeviceMultiply(conjg_tau_v_row_dev, elpaDeviceNumber<T>(0.5));
    vu_stored_rows_dev[i_row + max_local_rows*(2*n_stored_vecs+1)] =  elpaDeviceSubtract( elpaDeviceMultiply(conjg_tau_v_row_dev, vav) , u_row_dev[i_row] );
    
    i_row += blockDim.x * gridDim.x;
    }

  int i_col = tid;
  while (i_col < l_cols)
    {
    conjg_tau_v_col_dev = elpaDeviceMultiply(conjg_tau, v_col_dev[i_col]);
    uv_stored_cols_dev[i_col + max_local_cols*(2*n_stored_vecs+1)] = conjg_tau_v_col_dev;
    conjg_tau_v_col_dev = elpaDeviceMultiply(conjg_tau_v_col_dev, elpaDeviceNumber<T>(0.5));
    uv_stored_cols_dev[i_col + max_local_cols*(2*n_stored_vecs+0)] = elpaDeviceSubtract( elpaDeviceMultiply(conjg_tau_v_col_dev, vav) , u_col_dev[i_col] );

    i_col += blockDim.x * gridDim.x;
    }

  if ((std::is_same<T, gpuDoubleComplex>::value || std::is_same<T, gpuFloatComplex>::value) && l_cols>0)
    {
    int j = tid;
    while (j < 2*(n_stored_vecs+0)) // whole vector aux_complex_dev has to be copied and not two last elements only because l_cols has changed since last istep
      {
      aux_complex_dev[j] = elpaDeviceComplexConjugate(uv_stored_cols_dev[l_cols-1 + max_local_cols*j]);
      j += blockDim.x * gridDim.x;
      }
    
    // two last elements should be treated by the respective threads inorder to avoid sync problems
    i_col -= blockDim.x * gridDim.x;
    if (i_col == l_cols-1) 
      {
      aux_complex_dev[2*n_stored_vecs+0] = elpaDeviceComplexConjugate(uv_stored_cols_dev[l_cols-1 + max_local_cols*(2*n_stored_vecs+0)]);
      aux_complex_dev[2*n_stored_vecs+1] = elpaDeviceComplexConjugate(uv_stored_cols_dev[l_cols-1 + max_local_cols*(2*n_stored_vecs+1)]);
      }
    }

}


template <typename T>
void gpu_store_u_v_in_uv_vu(T *vu_stored_rows_dev, T *uv_stored_cols_dev, T *v_row_dev, T *u_row_dev,
                            T *v_col_dev, T *u_col_dev, T *tau_dev, T *aux_complex_dev, T *vav_host_or_dev, T *tau_host_or_dev,
                            int l_rows, int l_cols, int n_stored_vecs, int max_local_rows, int max_local_cols, 
                            int istep, int useCCL, int wantDebug, gpuStream_t my_stream){
  
  int threads = MAX_THREADS_PER_BLOCK/2; // the kernel has many local variables, for which we need memory registers. So we use less threads here to save memory.
  int blocks = std::max({(l_rows+threads-1)/threads, (l_cols+threads-1)/threads, 1});
  
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(threads,1,1);

  gpuPointerAttributes attributes;
  gpuError_t error = gpuPointerGetAttributes(&attributes, vav_host_or_dev);

  if (error == gpuSuccess) 
    {
#ifdef HAVE_OLD_HIP_TYPESTRUCTURE
    if (attributes.memoryType == gpuMemoryTypeHost)
#else
    if (attributes.type == gpuMemoryTypeHost)
#endif
      {
      T vav_host_value = *vav_host_or_dev;
      T tau_host_value = *tau_host_or_dev;
#ifdef WITH_GPU_STREAMS
      gpu_store_u_v_in_uv_vu_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, 
                                       v_col_dev, u_col_dev, tau_dev, aux_complex_dev, vav_host_value, tau_host_value, 
                                       l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, useCCL);
#else
      gpu_store_u_v_in_uv_vu_kernel<<<blocks,threadsPerBlock>>>            (vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, 
                                       v_col_dev, u_col_dev, tau_dev, aux_complex_dev, vav_host_value, tau_host_value, 
                                       l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, useCCL);
#endif
      if (wantDebug)
        {
        gpuError_t gpuerr = gpuGetLastError();
        if (gpuerr != gpuSuccess) printf("Error in executing gpu_store_u_v_in_uv_vu_kernel: %s\n",gpuGetErrorString(gpuerr));
        }
      } 
    
#ifdef HAVE_OLD_HIP_TYPESTRUCTURE
    else if (attributes.memoryType == gpuMemoryTypeDevice) 
#else
    else if (attributes.type == gpuMemoryTypeDevice) 
#endif
      {
#ifdef WITH_GPU_STREAMS
      gpu_store_u_v_in_uv_vu_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, 
                                       v_col_dev, u_col_dev, tau_dev, aux_complex_dev, vav_host_or_dev, tau_host_or_dev, 
                                       l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, useCCL);
#else
      gpu_store_u_v_in_uv_vu_kernel<<<blocks,threadsPerBlock>>>            (vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, 
                                       v_col_dev, u_col_dev, tau_dev, aux_complex_dev, vav_host_or_dev, tau_host_or_dev, 
                                       l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, useCCL);
#endif

      if (wantDebug)
        {
#ifdef WITH_GPU_STREAMS
    gpuStreamSynchronize(my_stream);
#else
    gpuDeviceSynchronize();
#endif
        gpuError_t gpuerr = gpuGetLastError();
        if (gpuerr != gpuSuccess) printf("Error in executing gpu_store_u_v_in_uv_vu_kernel: %s\n",gpuGetErrorString(gpuerr));
        }
      } 
    
    } 
  
  else 
    {
    printf("Error: Pointer type is unknown\n");
    }

}

extern "C" void CONCATENATE(ELPA_GPU, _store_u_v_in_uv_vu_FromC) (char dataType, intptr_t vu_stored_rows_dev, intptr_t uv_stored_cols_dev, intptr_t v_row_dev, intptr_t u_row_dev,
                                                      intptr_t v_col_dev, intptr_t u_col_dev, intptr_t tau_dev, intptr_t aux_complex_dev, intptr_t vav_host_or_dev, intptr_t tau_host_or_dev,
                                                      int l_rows, int l_cols, int n_stored_vecs, int max_local_rows, int max_local_cols, int istep, int useCCL, int wantDebug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_store_u_v_in_uv_vu<double>((double *)vu_stored_rows_dev, (double *)uv_stored_cols_dev, (double *)v_row_dev, (double *)u_row_dev, (double *)v_col_dev, (double *)u_col_dev, (double *)tau_dev, (double *)aux_complex_dev, (double *)vav_host_or_dev, (double *)tau_host_or_dev, 
                                                                l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, useCCL, wantDebug, my_stream);
  else if (dataType=='S') gpu_store_u_v_in_uv_vu<float> ((float  *)vu_stored_rows_dev, (float  *)uv_stored_cols_dev, (float  *)v_row_dev, (float  *)u_row_dev, (float  *)v_col_dev, (float  *)u_col_dev, (float  *)tau_dev, (float  *)aux_complex_dev, (float  *)vav_host_or_dev, (float  *)tau_host_or_dev,
                                                                l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, useCCL, wantDebug, my_stream);
  else if (dataType=='Z') gpu_store_u_v_in_uv_vu<gpuDoubleComplex>((gpuDoubleComplex *)vu_stored_rows_dev, (gpuDoubleComplex *)uv_stored_cols_dev, (gpuDoubleComplex *)v_row_dev, (gpuDoubleComplex *)u_row_dev, (gpuDoubleComplex *)v_col_dev, (gpuDoubleComplex *)u_col_dev, (gpuDoubleComplex *)tau_dev, (gpuDoubleComplex *)aux_complex_dev, (gpuDoubleComplex *)vav_host_or_dev, (gpuDoubleComplex *)tau_host_or_dev,
                                                                l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, useCCL, wantDebug, my_stream);
  else if (dataType=='C') gpu_store_u_v_in_uv_vu<gpuFloatComplex> ((gpuFloatComplex  *)vu_stored_rows_dev, (gpuFloatComplex  *)uv_stored_cols_dev, (gpuFloatComplex  *)v_row_dev, (gpuFloatComplex  *)u_row_dev, (gpuFloatComplex  *)v_col_dev, (gpuFloatComplex  *)u_col_dev, (gpuFloatComplex  *)tau_dev, (gpuFloatComplex  *)aux_complex_dev, (gpuFloatComplex  *)vav_host_or_dev, (gpuFloatComplex  *)tau_host_or_dev,
                                                                l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, useCCL, wantDebug, my_stream);
  else {
    printf("Error in gpu_store_u_v_in_uv_vu_double_FromC: Unsupported data type\n");
  }
}

//_________________________________________________________________________________________________

template <typename T, typename T_real>
__global__ void gpu_update_matrix_element_add_kernel(T *vu_stored_rows_dev, T *uv_stored_cols_dev, T *a_dev, T_real *d_vec_dev, 
                                                      int l_rows, int l_cols, int matrixRows, int max_local_rows, int max_local_cols, int istep, int n_stored_vecs, int isSkewsymmetric){
/*
      if (n_stored_vecs > 0) then
        ! update a_mat (only one elememt!)
        dot_prod = dot_product(vu_stored_rows(l_rows,1:2*n_stored_vecs), uv_stored_cols(l_cols,1:2*n_stored_vecs))
        a_mat(l_rows,l_cols) = a_mat(l_rows,l_cols) + dot_prod
      endif
#if REALCASE == 1
      if (isSkewsymmetric) then
        d_vec(istep-1) = 0.0_rk
      else
        d_vec(istep-1) = a_mat(l_rows,l_cols)
      endif
#endif
#if COMPLEXCASE == 1
      d_vec(istep-1) = real(a_mat(l_rows,l_cols),kind=rk)
#endif
*/

  const int threadsPerBlock = MAX_THREADS_PER_BLOCK;
  __shared__ T cache[threadsPerBlock];
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

  if (n_stored_vecs > 0)
    {

    T temp =  elpaDeviceNumber<T>(0.0);
    int index_n = tid;
    while (index_n < 2*n_stored_vecs) 
      {
      temp = elpaDeviceAdd(temp, elpaDeviceMultiply(elpaDeviceComplexConjugate(vu_stored_rows_dev[(l_rows-1)+max_local_rows*index_n]), uv_stored_cols_dev[(l_cols-1)+max_local_cols*index_n]) ); // temp += vu_stored_rows_dev[(l_rows-1)+max_local_rows*index_n] * uv_stored_cols_dev[(l_cols-1)+max_local_cols*index_n]
      index_n += blockDim.x * gridDim.x;
      }

    // set the cache values
    cache[threadIdx.x] = temp;
    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    int i = blockDim.x/2;
    while (i > 0) 
      {
      if (threadIdx.x < i) cache[threadIdx.x] = elpaDeviceAdd(cache[threadIdx.x] , cache[threadIdx.x + i]); // cache[threadIdx.x] += cache[threadIdx.x + i];
      __syncthreads();
      i /= 2;
      }

    if (threadIdx.x==0) 
      {
      atomicAdd(&a_dev[(l_rows-1) + matrixRows*(l_cols-1)], cache[0]);
      if (!isSkewsymmetric) atomicAdd( &d_vec_dev[istep-1-1] , elpaDeviceRealPart(cache[0]) );
      }
    }
}

template <typename T, typename T_real>
void gpu_update_matrix_element_add (T *vu_stored_rows_dev, T *uv_stored_cols_dev, T *a_dev, T_real *d_vec_dev, 
                                    int l_rows, int l_cols, int matrixRows, int max_local_rows, int max_local_cols, int istep, int n_stored_vecs, 
                                    int isSkewsymmetric, int wantDebug, int SM_count, gpuStream_t my_stream){

  int blocks = std::min((2*n_stored_vecs+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK, SM_count);
  if (n_stored_vecs==0) blocks=1;
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_update_matrix_element_add_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev,
                                                  l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs,
                                                  isSkewsymmetric);
#else
  gpu_update_matrix_element_add_kernel<<<blocks,threadsPerBlock>>>            (vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, 
                                                  l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, 
                                                  isSkewsymmetric);
#endif
  if (wantDebug){
#ifdef WITH_GPU_STREAMS
    gpuStreamSynchronize(my_stream);
#else
    gpuDeviceSynchronize();
#endif
    gpuError_t cuerr = gpuGetLastError();
    if (cuerr != gpuSuccess){
      printf("Error in executing gpu_update_matrix_element_add_kernel: %s\n",gpuGetErrorString(cuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU, _update_matrix_element_add_FromC) (char dataType, intptr_t vu_stored_rows_dev, intptr_t uv_stored_cols_dev, intptr_t a_dev, intptr_t d_vec_dev, 
                                                            int l_rows, int l_cols, int matrixRows, int max_local_rows, int max_local_cols, int istep, int n_stored_vecs, 
                                                            int isSkewsymmetric, int wantDebug, int SM_count, gpuStream_t my_stream){
  if      (dataType=='D') gpu_update_matrix_element_add<double, double>((double *)vu_stored_rows_dev, (double *)uv_stored_cols_dev, (double *)a_dev, (double *)d_vec_dev, 
                                                        l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, isSkewsymmetric, wantDebug, SM_count, my_stream);
  else if (dataType=='S') gpu_update_matrix_element_add<float,  float> ((float  *)vu_stored_rows_dev, (float  *)uv_stored_cols_dev, (float  *)a_dev, (float  *)d_vec_dev,
                                                        l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, isSkewsymmetric, wantDebug, SM_count, my_stream);
  else if (dataType=='Z') gpu_update_matrix_element_add<gpuDoubleComplex, double>((gpuDoubleComplex *)vu_stored_rows_dev, (gpuDoubleComplex *)uv_stored_cols_dev, (gpuDoubleComplex *)a_dev, (double *)d_vec_dev, 
                                                        l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, isSkewsymmetric, wantDebug, SM_count, my_stream);
  else if (dataType=='C') gpu_update_matrix_element_add<gpuFloatComplex , float> ((gpuFloatComplex  *)vu_stored_rows_dev, (gpuFloatComplex  *)uv_stored_cols_dev, (gpuFloatComplex  *)a_dev, (float  *)d_vec_dev, 
                                                        l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, isSkewsymmetric, wantDebug, SM_count, my_stream);
  else {
    printf("Error in gpu_update_matrix_element_add_FromC: Unsupported data type\n");
  }
}

//_________________________________________________________________________________________________

template <typename T>
__global__ void gpu_hh_transform_kernel(T *alpha_dev, T *xnorm_sq_dev, T *xf_dev, T *tau_dev){

/*
#if complexcase == 1
  alphr = real( alpha, kind=rk )
  alphi = precision_imag( alpha )
#endif

#if realcase == 1
  if ( xnorm_sq==0.0_rk ) then
#endif
#if complexcase == 1
  if ( xnorm_sq==0.0_rk .and. alphi==0.0_rk ) then
#endif

#if realcase == 1
    if ( alpha>=0.0_rk ) then
#endif
#if complexcase == 1
    if ( alphr>=0.0_rk ) then
#endif
      tau = 0.0_rk
    else
      tau = 2.0_rk
      alpha = -alpha
    endif
    xf = 0.0_rk

  else

#if realcase == 1
    beta = sign( sqrt( alpha**2 + xnorm_sq ), alpha )
#endif
#if complexcase == 1
    beta = sign( sqrt( alphr**2 + alphi**2 + xnorm_sq ), alphr )
#endif
    alpha = alpha + beta
    if ( beta<0 ) then
      beta = -beta
      tau  = -alpha / beta
    else
#if realcase == 1
      alpha = xnorm_sq / alpha
#endif
#if complexcase == 1
      alphr = alphi * (alphi/real( alpha , kind=rk))
      alphr = alphr + xnorm_sq/real( alpha, kind=rk )
#endif

#if realcase == 1
      tau = alpha / beta
      alpha = -alpha
#endif
#if complexcase == 1
      tau = precision_cmplx( alphr/beta, -alphi/beta )
      alpha = precision_cmplx( -alphr, alphi )
#endif
    end if
    xf = 1.0_rk/alpha
    alpha = beta
  endif
*/

  auto alpha_r = elpaDeviceRealPart(*alpha_dev);
  auto alpha_i = elpaDeviceImagPart(*alpha_dev);

  if (elpaDeviceRealPart(*xnorm_sq_dev)==0.0 && alpha_i==0.0)
    {
    if (alpha_r >= 0.0) *tau_dev = elpaDeviceNumber<T>(0.0);
    else
      {
      *tau_dev = elpaDeviceNumber<T>(2.0);
      *alpha_dev = elpaDeviceMultiply(*alpha_dev, elpaDeviceNumber<T>(-1.0)); // (*alpha_dev) *= -1
      }
    
    *xf_dev = elpaDeviceNumber<T>(0.0);
    }

  else
    {
    T beta = elpaDeviceNumber<T> (elpaDeviceSign( elpaDeviceSqrt( alpha_r*alpha_r + alpha_i*alpha_i +
                                          elpaDeviceRealPart(*xnorm_sq_dev) ), elpaDeviceRealPart(*alpha_dev) ));

    *alpha_dev = elpaDeviceAdd(*alpha_dev, beta);
    
    if (elpaDeviceRealPart(beta)<0)
      {
      *tau_dev  = elpaDeviceDivide(*alpha_dev, beta);
      beta = elpaDeviceMultiply(beta, elpaDeviceNumber<T>(-1.0)); // beta *= -1
      }
    else
      {
      alpha_r = alpha_i * alpha_i/elpaDeviceRealPart(*alpha_dev);
      alpha_r = alpha_r + elpaDeviceRealPart(*xnorm_sq_dev)/ elpaDeviceRealPart(*alpha_dev);

      *tau_dev = elpaDeviceDivide( elpaDeviceNumberFromRealImag<T>(alpha_r, -alpha_i) , beta );
      *alpha_dev = elpaDeviceNumberFromRealImag<T>(-alpha_r, alpha_i);
      }

    *xf_dev = elpaDeviceDivide(elpaDeviceNumber<T>(1.0), *alpha_dev);
    *alpha_dev = beta;
    }

}

template <typename T>
void gpu_hh_transform(T *alpha_dev, T *xnorm_sq_dev, T *xf_dev, T *tau_dev, 
                      int wantDebug, gpuStream_t my_stream){

  dim3 blocks = dim3(1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_hh_transform_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev);
#else  
  gpu_hh_transform_kernel<<<blocks,threadsPerBlock>>>            (alpha_dev, xnorm_sq_dev, xf_dev, tau_dev);
#endif

  if (wantDebug){
#ifdef WITH_GPU_STREAMS
    gpuStreamSynchronize(my_stream);
#else
    gpuDeviceSynchronize();
#endif
    gpuError_t cuerr = gpuGetLastError();
    if (cuerr != gpuSuccess){
      printf("Error in executing gpu_hh_transform_kernel: %s\n",gpuGetErrorString(cuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU, _hh_transform_FromC) (char dataType, intptr_t alpha_dev, intptr_t xnorm_sq_dev, intptr_t xf_dev, intptr_t tau_dev, int wantDebug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_hh_transform<double>((double *)alpha_dev, (double *)xnorm_sq_dev, (double *)xf_dev, (double *)tau_dev, wantDebug, my_stream);
  else if (dataType=='S') gpu_hh_transform<float> ((float  *)alpha_dev, (float  *)xnorm_sq_dev, (float  *)xf_dev, (float  *)tau_dev, wantDebug, my_stream);
  else if (dataType=='Z') gpu_hh_transform<gpuDoubleComplex>((gpuDoubleComplex *)alpha_dev, (gpuDoubleComplex *)xnorm_sq_dev, (gpuDoubleComplex *)xf_dev, (gpuDoubleComplex *)tau_dev, wantDebug, my_stream);
  else if (dataType=='C') gpu_hh_transform<gpuFloatComplex> ((gpuFloatComplex  *)alpha_dev, (gpuFloatComplex  *)xnorm_sq_dev, (gpuFloatComplex  *)xf_dev, (gpuFloatComplex  *)tau_dev, wantDebug, my_stream);
  else {
    printf("Error in gpu_hh_transform_FromC: Unsupported data type\n");
  }
}

//_________________________________________________________________________________________________

/*
  ! direction = 1
  do lc=1,nvc
    do i = nblks_skip+n, nblks_tot-1, lcm_s_t
      k = (i - nblks_skip - n)/lcm_s_t * nblk + (lc - 1) * auxstride
      ns = (i/nps)*nblk ! local start of block i
      nl = min(nvr-i*nblk,nblk) ! length
      aux(k+1:k+nl) = vmat_s(ns+1:ns+nl,lc)
    enddo
  enddo

  ! direction = 2
  do lc=1,nvc
    do i = nblks_skip+n, nblks_tot-1, lcm_s_t
      k = (i - nblks_skip - n)/lcm_s_t * nblk + (lc - 1) * auxstride
      ns = (i/npt)*nblk ! local start of block i
      nl = min(nvr-i*nblk,nblk) ! length
#ifdef SKEW_SYMMETRIC_BUILD
      vmat_t(ns+1:ns+nl,lc) = - aux(k+1:k+nl)
#else
      vmat_t(ns+1:ns+nl,lc) = aux(k+1:k+nl)
#endif
    enddo
  enddo
*/

// this kernel has too low occupancy, small amount of work per thread
// generalize special square-grid codepath to rectangular grids?

template <typename T>
__global__ void gpu_transpose_reduceadd_vectors_copy_block_kernel(T *aux_transpose_dev, T *vmat_st_dev, 
                                              int nvc, int nvr, int n_block, int nblks_skip, int nblks_tot, 
                                              int lcm_s_t, int nblk, int auxstride, int np_st, int ld_st, int direction, int isSkewsymmetric, int isReduceadd){

  //for (int lc=1; lc <= nvc; lc += 1)
  int lc0 = blockIdx.y; // lc0 = lc-1, 0-based index
  
  //for (int i = nblks_skip+n_block; i <= nblks_tot-1; i += lcm_s_t)
  int i = nblks_skip+n_block + blockIdx.z * lcm_s_t; // 1-based index
  if (i > nblks_tot-1) return;

  int nl = MIN(nvr-i*nblk, nblk); // length
  int k = (i - nblks_skip - n_block)/lcm_s_t * nblk + lc0*auxstride;
  int ns_plus_lc0_ld_st = (i/np_st)*nblk + lc0*ld_st; // ns(local start of block i) + lc0*ld_st

  //for (int j=tid_x; j<nl; j+=blockDim.x*gridDim.x){
  int j = threadIdx.x + blockIdx.x*blockDim.x; // 0-based index
  if (j >= nl) return;
    
  if (direction==1) aux_transpose_dev[k+j] = vmat_st_dev[ns_plus_lc0_ld_st + j];
  if (direction==2 && !isReduceadd) {
    T sign = elpaDeviceNumber<T>(1.0);
    if (isSkewsymmetric) sign = elpaDeviceNumber<T>(-1.0);
    vmat_st_dev[ns_plus_lc0_ld_st + j]  = elpaDeviceMultiply(sign, aux_transpose_dev[k+j]);
  }
  if (direction==2 &&  isReduceadd) {
    vmat_st_dev[ns_plus_lc0_ld_st + j]  = elpaDeviceAdd(vmat_st_dev[ns_plus_lc0_ld_st + j] , aux_transpose_dev[k+j]);
  }
  //}
}

template <typename T>
void gpu_transpose_reduceadd_vectors_copy_block(T *aux_transpose_dev, T *vmat_st_dev, 
                                                int nvc, int nvr,  int n_block, int nblks_skip, int nblks_tot, 
                                                int lcm_s_t, int nblk, int auxstride, int np_st, int ld_st, 
                                                int direction, int isSkewsymmetric, int isReduceadd, 
                                                int wantDebug, int SM_count, gpuStream_t my_stream){
  
  int i0    = nblks_skip + n_block;
  if (nblks_tot-1-i0 < 0) return;
  
  int num_i = (nblks_tot-1 - i0) / lcm_s_t + 1;
  int threads = MIN_THREADS_PER_BLOCK;

  dim3 blocksPerGrid = dim3((nblk+threads-1)/threads, nvc, num_i);
  dim3 threadsPerBlock = dim3(threads, 1, 1);
  
#ifdef WITH_GPU_STREAMS
  gpu_transpose_reduceadd_vectors_copy_block_kernel<<<blocksPerGrid,threadsPerBlock,0,my_stream>>>(aux_transpose_dev, vmat_st_dev, 
                          nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride, np_st, ld_st, direction, isSkewsymmetric, isReduceadd);
#else
  gpu_transpose_reduceadd_vectors_copy_block_kernel<<<blocksPerGrid,threadsPerBlock>>>            (aux_transpose_dev, vmat_st_dev, 
                          nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride, np_st, ld_st, direction, isSkewsymmetric, isReduceadd);
#endif

  if(wantDebug)
    {
#ifdef WITH_GPU_STREAMS
    gpuStreamSynchronize(my_stream);
#else
    gpuDeviceSynchronize();
#endif
    gpuError_t cuerr = gpuGetLastError();
    if (cuerr != gpuSuccess) printf("Error in executing gpu_transpose_reduceadd_vectors_copy_block_kernel: %s\n",gpuGetErrorString(cuerr));
    }
}

extern "C" void CONCATENATE(ELPA_GPU, _transpose_reduceadd_vectors_copy_block_FromC)(char dataType, intptr_t aux_transpose_dev, intptr_t vmat_st_dev, 
                                                                        int nvc, int nvr,  int n_block, int nblks_skip, int nblks_tot, 
                                                                        int lcm_s_t, int nblk, int auxstride, int np_st, int ld_st, 
                                                                        int direction, int isSkewsymmetric, int isReduceadd, int wantDebug, int SM_count, gpuStream_t my_stream){
  if      (dataType=='D') gpu_transpose_reduceadd_vectors_copy_block<double>((double *)aux_transpose_dev, (double *)vmat_st_dev, nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride, np_st, ld_st, direction, isSkewsymmetric, isReduceadd, wantDebug, SM_count, my_stream);
  else if (dataType=='S') gpu_transpose_reduceadd_vectors_copy_block<float> ((float  *)aux_transpose_dev, (float  *)vmat_st_dev, nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride, np_st, ld_st, direction, isSkewsymmetric, isReduceadd, wantDebug, SM_count, my_stream);
  else if (dataType=='Z') gpu_transpose_reduceadd_vectors_copy_block<gpuDoubleComplex>((gpuDoubleComplex *)aux_transpose_dev, (gpuDoubleComplex *)vmat_st_dev, nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride, np_st, ld_st, direction, isSkewsymmetric, isReduceadd, wantDebug, SM_count, my_stream);
  else if (dataType=='C') gpu_transpose_reduceadd_vectors_copy_block<gpuFloatComplex> ((gpuFloatComplex  *)aux_transpose_dev, (gpuFloatComplex  *)vmat_st_dev, nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride, np_st, ld_st, direction, isSkewsymmetric, isReduceadd, wantDebug, SM_count, my_stream);
  else {
    printf("Error in gpu_transpose_reduceadd_vectors_copy_block_FromC: Unsupported data type\n");
  }
}

//_________________________________________________________________________________________________

