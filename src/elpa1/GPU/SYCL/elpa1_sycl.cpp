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

#include <CL/sycl.hpp>
#include "src/GPU/SYCL/syclCommon.hpp"
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <stdint.h>
#include <stdbool.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include "config-f90.h"
#include <complex>
#include <atomic>

#define MAX_THREADS_PER_BLOCK 1024

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

template <typename T> 
T sign(T a, T b) {
    if (b>=0) return fabs(a);
    else return -fabs(a);
}

// construct a generic /float/std::complex<double>/cuFloatComplex from a double
template <typename T>  T elpaDeviceNumber(double number);
template <>  double elpaDeviceNumber<double>(double number) {return number;}
template <>  float  elpaDeviceNumber<float> (double number) {return (float) number;}
template <> std::complex<double> elpaDeviceNumber<std::complex<double>>(double number) {return std::complex<double>(number, 0.0);}
template <> std::complex<float>  elpaDeviceNumber<std::complex<float>> (double number) {return std::complex<float>((float)number, 0.0f);}

// construct a generic double/float/std::complex<double>/cuFloatComplex from a real and imaginary parts
template <typename T, typename T_real>  T elpaDeviceNumberFromRealImag(T_real Re, T_real Im);
template <> double elpaDeviceNumberFromRealImag<double>(double Real, double Imag) {return Real;}
template <> float  elpaDeviceNumberFromRealImag<float> (float  Real, float  Imag) {return Real;}
template <> std::complex<double> elpaDeviceNumberFromRealImag<std::complex<double>>(double Real, double Imag) {return std::complex<double>(Real, Imag);}
template <> std::complex<float>  elpaDeviceNumberFromRealImag<std::complex<float>> (float Real,  float Imag ) {return std::complex<float>(Real, Imag);}

double elpaDeviceAdd(double a, double b) { return a + b; }
float  elpaDeviceAdd(float a, float b)   { return a + b; }
std::complex<double> elpaDeviceAdd(std::complex<double> a, std::complex<double> b) { return a + b; }
std::complex<float> elpaDeviceAdd(std::complex<float> a, std::complex<float> b) { return a + b; }

double elpaDeviceSubtract(double a, double b) { return a - b; }
float  elpaDeviceSubtract(float a, float b)   { return a - b; }
std::complex<double> elpaDeviceSubtract(std::complex<double> a, std::complex<double> b) {return a - b;}
std::complex<float> elpaDeviceSubtract(std::complex<float> a, std::complex<float> b)  {return a - b;}

double elpaDeviceMultiply(double a, double b) { return a * b; }
float  elpaDeviceMultiply(float  a, float  b) { return a * b; }
std::complex<double> elpaDeviceMultiply(std::complex<double> a, std::complex<double> b) { return a * b; }
std::complex<float> elpaDeviceMultiply(std::complex<float> a, std::complex<float> b){ return a * b; }

double elpaDeviceDivide(double a, double b) { return a / b; }
float  elpaDeviceDivide(float  a, float  b) { return a / b; }
std::complex<double> elpaDeviceDivide(std::complex<double> a, std::complex<double> b)  { return a / b; }
std::complex<float> elpaDeviceDivide(std::complex<float> a, std::complex<float> b)  { return a / b; }

double elpaDeviceSqrt(double number) { return sycl::sqrt(number); }
float elpaDeviceSqrt(float number) { return sycl::sqrt(number); }


template <typename T> void atomicAdd(T* address, T val)
  {
  sycl::atomic_ref<T, sycl::memory_order_relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_sum (*address);
  atomic_sum += val;
  }

template <>  void atomicAdd(std::complex<double>* address, std::complex<double> val)
  {
  double* real_ptr = reinterpret_cast<double*>(address); // Pointer to the real part
  double* imag_ptr = real_ptr + 1; // Pointer to the imaginary part

  sycl::atomic_ref<double, sycl::memory_order_relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_sum_real(*real_ptr);
  sycl::atomic_ref<double, sycl::memory_order_relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_sum_imag(*imag_ptr);

  atomic_sum_real += val.real();
  atomic_sum_imag += val.imag();
  }

template <>  void atomicAdd(std::complex<float>* address, std::complex<float> val)
  {
  float* real_ptr = reinterpret_cast<float*>(address); // Pointer to the real part
  float* imag_ptr = real_ptr + 1; // Pointer to the imaginary part

  sycl::atomic_ref<float, sycl::memory_order_relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_sum_real(*real_ptr);
  sycl::atomic_ref<float, sycl::memory_order_relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_sum_imag(*imag_ptr);

  atomic_sum_real += val.real();
  atomic_sum_imag += val.imag();
  }

double elpaDeviceComplexConjugate(double number) {return number;}
float elpaDeviceComplexConjugate(float  number) {return number;}
std::complex<double> elpaDeviceComplexConjugate(std::complex<double> number) {return std::conj(number);}
std::complex<float>  elpaDeviceComplexConjugate(std::complex<float> number)  {return std::conj(number);}

double elpaDeviceRealPart(double number) {return number;}
float  elpaDeviceRealPart(float  number) {return number;}
double elpaDeviceRealPart(std::complex<double> number) { return number.real(); }
float  elpaDeviceRealPart(std::complex<float>  number) { return number.real(); }

double elpaDeviceImagPart(double number) {return 0.0;}
float  elpaDeviceImagPart(float  number) {return 0.0f;}
double elpaDeviceImagPart(std::complex<double> number) { return number.imag(); }
float  elpaDeviceImagPart(std::complex<float>  number) { return number.imag(); }

// Define a helper struct to determine if a type is a pointer
template <typename T>
struct is_pointer { static const bool value = false; };

template <typename T>
struct is_pointer<T*> { static const bool value = true; };

// Device function to convert a pointer to a value
template <typename T>
T convert_to_device(T* x, std::true_type) { return *x;}
// Device function to convert a value to a value
template <typename T>
T convert_to_device(T x, std::false_type) { return x;}

//________________________________________________________________
 
template <typename T, typename T_real>
void sycl_copy_and_set_zeros (T *v_row_dev, T *a_dev, int l_rows, int l_cols, int matrixRows, int istep,
                                         T *aux1_dev, T *vav_dev, T_real *d_vec_dev, 
                                         bool isOurProcessRow, bool isOurProcessCol, bool isOurProcessCol_prev, bool isSkewsymmetric, bool useCCL,
                                         sycl::nd_item<1> item_ct1){
  int tid = item_ct1.get_local_id(0) +
            item_ct1.get_group(0) * item_ct1.get_local_range(0);

  if (isOurProcessCol_prev)
    {
    // copy v_row to a_dev
    int i_row = tid;
    while (i_row < l_rows) 
      {
      v_row_dev[i_row] = a_dev[i_row + matrixRows*l_cols];
      i_row += item_ct1.get_local_range(0) * item_ct1.get_group_range(0);
      }
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
void sycl_copy_and_set_zeros_FromC(T *v_row_dev, T *a_dev, int *l_rows_in,
                                   int *l_cols_in, int *matrixRows_in,
                                   int *istep_in, T *aux1_dev, T *vav_dev,
                                   T_real *d_vec_dev, bool *isOurProcessRow_in,
                                   bool *isOurProcessCol_in,
                                   bool *isOurProcessCol_prev_in,
                                   bool *isSkewsymmetric_in, bool *useCCL_in,
                                   bool *wantDebug_in, intptr_t my_stream) {
  int l_rows = *l_rows_in;   
  int l_cols = *l_cols_in;   
  int matrixRows = *matrixRows_in;
  int istep = *istep_in;
  bool isOurProcessRow = *isOurProcessRow_in;
  bool isOurProcessCol = *isOurProcessCol_in;
  bool isOurProcessCol_prev = *isOurProcessCol_prev_in;
  bool isSkewsymmetric = *isSkewsymmetric_in;
  bool useCCL = *useCCL_in;
  bool wantDebug = *wantDebug_in;

  int blocks = std::max((l_rows+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK, 1);
  sycl::range<1> blocksPerGrid = sycl::range<1>(blocks);
  sycl::range<1> threadsPerBlock = sycl::range<1>(MAX_THREADS_PER_BLOCK); // TODO_23_11: change to NB?

  /*
DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  queue.submit([&](sycl::handler &cgh) 
    {
    cgh.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks) * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<1> item_ct1) {
        sycl_copy_and_set_zeros(
            v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, aux1_dev,
            vav_dev, d_vec_dev, isOurProcessRow, isOurProcessCol,
            isOurProcessCol_prev, isSkewsymmetric, useCCL, item_ct1);
      });
    });
  queue.wait_and_throw();

}

extern "C" void sycl_copy_and_set_zeros_double_FromC(
    double *v_row_dev, double *a_dev, int *l_rows_in, int *l_cols_in,
    int *matrixRows_in, int *istep_in, double *aux1_dev, double *vav_dev,
    double *d_vec_dev, bool *isOurProcessRow_in, bool *isOurProcessCol_in,
    bool *isOurProcessCol_prev_in, bool *isSkewsymmetric_in, bool *useCCL_in,
    bool *wantDebug_in, intptr_t my_stream) {
  sycl_copy_and_set_zeros_FromC(v_row_dev, a_dev, l_rows_in, l_cols_in, matrixRows_in, istep_in, aux1_dev, vav_dev, d_vec_dev, isOurProcessRow_in, isOurProcessCol_in, isOurProcessCol_prev_in, isSkewsymmetric_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void sycl_copy_and_set_zeros_float_FromC(
    float *v_row_dev, float *a_dev, int *l_rows_in, int *l_cols_in,
    int *matrixRows_in, int *istep_in, float *aux1_dev, float *vav_dev,
    float *d_vec_dev, bool *isOurProcessRow_in, bool *isOurProcessCol_in,
    bool *isOurProcessCol_prev_in, bool *isSkewsymmetric_in, bool *useCCL_in,
    bool *wantDebug_in, intptr_t my_stream) {
  sycl_copy_and_set_zeros_FromC(v_row_dev, a_dev, l_rows_in, l_cols_in, matrixRows_in, istep_in, aux1_dev, vav_dev, d_vec_dev, isOurProcessRow_in, isOurProcessCol_in, isOurProcessCol_prev_in, isSkewsymmetric_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void sycl_copy_and_set_zeros_double_complex_FromC(
    std::complex<double> *v_row_dev, std::complex<double> *a_dev, int *l_rows_in,
    int *l_cols_in, int *matrixRows_in, int *istep_in, std::complex<double> *aux1_dev,
    std::complex<double> *vav_dev, double *d_vec_dev, bool *isOurProcessRow_in,
    bool *isOurProcessCol_in, bool *isOurProcessCol_prev_in,
    bool *isSkewsymmetric_in, bool *useCCL_in, bool *wantDebug_in,
    intptr_t my_stream) {
  sycl_copy_and_set_zeros_FromC(v_row_dev, a_dev, l_rows_in, l_cols_in, matrixRows_in, istep_in, aux1_dev, vav_dev, d_vec_dev, isOurProcessRow_in, isOurProcessCol_in, isOurProcessCol_prev_in, isSkewsymmetric_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void sycl_copy_and_set_zeros_float_complex_FromC(
    std::complex<float> *v_row_dev, std::complex<float> *a_dev, int *l_rows_in,
    int *l_cols_in, int *matrixRows_in, int *istep_in, std::complex<float> *aux1_dev,
    std::complex<float> *vav_dev, float *d_vec_dev, bool *isOurProcessRow_in,
    bool *isOurProcessCol_in, bool *isOurProcessCol_prev_in,
    bool *isSkewsymmetric_in, bool *useCCL_in, bool *wantDebug_in,
    intptr_t my_stream) {
  sycl_copy_and_set_zeros_FromC(v_row_dev, a_dev, l_rows_in, l_cols_in, matrixRows_in, istep_in, aux1_dev, vav_dev, d_vec_dev, isOurProcessRow_in, isOurProcessCol_in, isOurProcessCol_prev_in, isSkewsymmetric_in, useCCL_in, wantDebug_in, my_stream);
}

//________________________________________________________________
// device syncronization is needed afterwards, e.g. gpu_memcpy

template <typename T>
void sycl_dot_product_kernel(int n, T *x_dev, int incx, T *y_dev, int incy, T *result_dev,
                             sycl::nd_item<1> it, sycl::accessor<T, 1, sycl::access_mode::read_write, sycl::access::target::local>  cache){
   // extra space of fixed size is reserved for a speedup
  int tid = it.get_local_id(0) + it.get_group(0) * it.get_local_range(0);

  T temp = elpaDeviceNumber<T>(0.0);

  int i = tid;
  while (i < n) {
    temp = elpaDeviceAdd(temp, elpaDeviceMultiply(elpaDeviceComplexConjugate(x_dev[i*incx]), y_dev[i*incy])); // temp += x_dev[i*incx] * y_dev[i*incy];
    i += it.get_local_range(0) * it.get_group_range(0);
  }

  // set the cache values
  cache[it.get_local_id(0)] = temp;
  // synchronize threads in this block
  /*
DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
performance if there is no access to global memory.
*/
  it.barrier();

  // for reductions, threadsPerBlock=blockDim.x must be a power of 2
  i = it.get_local_range(0) / 2;
  while (i > 0) {
    if (it.get_local_id(0) < i) cache[it.get_local_id(0)] =
        elpaDeviceAdd(cache[it.get_local_id(0)],
                      cache[it.get_local_id(0) + i]);
    /*
DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
performance if there is no access to global memory.
*/
    it.barrier();
    i /= 2;
  }

  if (it.get_local_id(0) == 0) 
    {
    atomicAdd(&result_dev[0], cache[0]);
    }
}

template <typename T>
void sycl_dot_product_FromC(int *n_in, T *x_dev, int *incx_in, T *y_dev,
                            int *incy_in, T *result_dev, bool *wantDebug_in,
                            intptr_t my_stream) {
  int n = *n_in;   
  int incx = *incx_in;
  int incy = *incy_in;
  bool wantDebug = *wantDebug_in;

  int SM_count=32;
  //syclDeviceGetAttribute(&SM_count, syclDevAttrMultiProcessorCount, 0); // TODO_23_11 move this outside, to set_gpu, claim the number only once during GPU setup

  int blocks = SM_count;
  sycl::range<1> blocksPerGrid   = sycl::range<1>(blocks);
  sycl::range<1> threadsPerBlock = sycl::range<1>(MAX_THREADS_PER_BLOCK); // TODO_23_11: or NB?


  /*
DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  queue.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write, sycl::access::target::local> cache_acc_ct1(sycl::range<1>(1024), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(blocksPerGrid * threadsPerBlock, threadsPerBlock),
          [=](sycl::nd_item<1> item_ct1) {
          sycl_dot_product_kernel(n, x_dev, incx, y_dev, incy, result_dev,
                                  item_ct1, cache_acc_ct1);
        });
  });
  queue.wait_and_throw();
}

extern "C" void sycl_dot_product_double_FromC(int *n_in, double *x_dev,
                                              int *incx_in, double *y_dev,
                                              int *incy_in, double *result_dev,
                                              bool *wantDebug_in,
                                              intptr_t my_stream) {
  sycl_dot_product_FromC(n_in, x_dev, incx_in, y_dev, incy_in, result_dev, wantDebug_in, my_stream);
}

extern "C" void sycl_dot_product_float_FromC(int *n_in, float *x_dev,
                                             int *incx_in, float *y_dev,
                                             int *incy_in, float *result_dev,
                                             bool *wantDebug_in,
                                             intptr_t my_stream) {
  sycl_dot_product_FromC(n_in, x_dev, incx_in, y_dev, incy_in, result_dev, wantDebug_in, my_stream);
}

extern "C" void sycl_dot_product_double_complex_FromC(
    int *n_in, std::complex<double> *x_dev, int *incx_in, std::complex<double> *y_dev,
    int *incy_in, std::complex<double> *result_dev, bool *wantDebug_in,
    intptr_t my_stream) {
  sycl_dot_product_FromC(n_in, x_dev, incx_in, y_dev, incy_in, result_dev, wantDebug_in, my_stream);
}

extern "C" void sycl_dot_product_float_complex_FromC(
    int *n_in, std::complex<float> *x_dev, int *incx_in, std::complex<float> *y_dev,
    int *incy_in, std::complex<float> *result_dev, bool *wantDebug_in,
    intptr_t my_stream) {
  sycl_dot_product_FromC(n_in, x_dev, incx_in, y_dev, incy_in, result_dev, wantDebug_in, my_stream);
}

//________________________________________________________________

template <typename T>
void sycl_dot_product_and_assign_kernel(T *v_row_dev, int l_rows, int isOurProcessRow, T *aux1_dev,
                                        sycl::nd_item<1> it, sycl::accessor<T, 1, sycl::access_mode::read_write, sycl::access::target::local>  cache){
  const int threadsPerBlock = MAX_THREADS_PER_BLOCK;

  int tid = it.get_local_id(0) +
            it.get_group(0) * it.get_local_range(0);

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
    index_global += it.get_local_range(0) * it.get_group_range(0);
  }

  // set the cache values
  cache[it.get_local_id(0)] = temp;
  // synchronize threads in this block
  /*
DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
performance if there is no access to global memory.
*/
  it.barrier();

  // for reductions, threadsPerBlock must be a power of 2
  int i = it.get_local_range(0) / 2;
  while (i > 0) {
    if (it.get_local_id(0) < i) cache[it.get_local_id(0)] =
        elpaDeviceAdd(cache[it.get_local_id(0)],
                      cache[it.get_local_id(0) + i]);
    /*
DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
performance if there is no access to global memory.
*/
    it.barrier();
    i /= 2;
  }

  if (it.get_local_id(0) == 0)  atomicAdd(&aux1_dev[0], cache[0]);

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
void sycl_dot_product_and_assign_FromC(T *v_row_dev, int *l_rows_in,
                                       int *isOurProcessRow_in, T *aux1_dev,
                                       bool *wantDebug_in,
                                       intptr_t my_stream) {
  int l_rows = *l_rows_in;   
  int isOurProcessRow = *isOurProcessRow_in;
  bool wantDebug = *wantDebug_in;

  //int numSMs;
  //syclDeviceGetAttribute(&numSMs, syclDevAttrMultiProcessorCount, 0);
  
  //int blocks = (l_rows+1023)/MAX_THREADS_PER_BLOCK;
  int blocks = 32; // TODO_23_11: change blocksPerGrid to number of SM's (108 fo A100) and threadsPerBlock to max threads per block. claim the number only once during GPU setup
  
  sycl::range<1> blocksPerGrid = sycl::range<1>(blocks);
  sycl::range<1> threadsPerBlock = sycl::range<1>(MAX_THREADS_PER_BLOCK);

  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  /*
DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/

  queue.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write, sycl::access::target::local> cache_acc_ct1(sycl::range<1>(1024), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(blocksPerGrid * threadsPerBlock, threadsPerBlock),
          [=](sycl::nd_item<1> item_ct1) {
          sycl_dot_product_and_assign_kernel(v_row_dev, l_rows, isOurProcessRow, aux1_dev,
                                  item_ct1, cache_acc_ct1);
        });
  });
  queue.wait_and_throw();

}

extern "C" void sycl_dot_product_and_assign_double_FromC(
    double *v_row_dev, int *l_rows_in, int *isOurProcessRow_in,
    double *aux1_dev, bool *wantDebug_in, intptr_t my_stream) {
  sycl_dot_product_and_assign_FromC(v_row_dev, l_rows_in, isOurProcessRow_in, aux1_dev, wantDebug_in, my_stream);
}

extern "C" void sycl_dot_product_and_assign_float_FromC(
    float *v_row_dev, int *l_rows_in, int *isOurProcessRow_in, float *aux1_dev,
    bool *wantDebug_in, intptr_t my_stream) {
  sycl_dot_product_and_assign_FromC(v_row_dev, l_rows_in, isOurProcessRow_in, aux1_dev, wantDebug_in, my_stream);
}

extern "C" void sycl_dot_product_and_assign_double_complex_FromC(
    std::complex<double> *v_row_dev, int *l_rows_in, int *isOurProcessRow_in,
    std::complex<double> *aux1_dev, bool *wantDebug_in, intptr_t my_stream) {
  sycl_dot_product_and_assign_FromC(v_row_dev, l_rows_in, isOurProcessRow_in, aux1_dev, wantDebug_in, my_stream);
}

extern "C" void sycl_dot_product_and_assign_float_complex_FromC(
    std::complex<float> *v_row_dev, int *l_rows_in, int *isOurProcessRow_in,
    std::complex<float> *aux1_dev, bool *wantDebug_in, intptr_t my_stream) {
 sycl_dot_product_and_assign_FromC(v_row_dev, l_rows_in, isOurProcessRow_in, aux1_dev, wantDebug_in, my_stream);
}

//________________________________________________________________

template <typename T, typename T_real, typename T_value_or_pointer>
void sycl_set_e_vec_scale_set_one_store_v_row_kernel(T_real *e_vec_dev, T *vrl_dev, T *a_dev, T *v_row_dev, T *tau_dev, T_value_or_pointer xf_host_or_dev, 
                                                      int l_rows, int l_cols,  int matrixRows, int istep, bool isOurProcessRow, bool useCCL,
                                                      sycl::nd_item<1> it){
  int tid = it.get_local_id(0) +
            it.get_group(0) * it.get_local_range(0);

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
    ! add tau after the end of actual v_row, to be broadcasted with it
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
    index_global += it.get_local_range(0) * it.get_group_range(0);
  }

  if (isOurProcessRow && index_global - it.get_local_range(0)*it.get_group_range(0) ==  l_rows - 1) // last element
    {
    v_row_dev[l_rows-1] = elpaDeviceNumber<T>(1.0);
    }
    
  int i_row = tid;
  while (i_row < l_rows) {
    a_dev[i_row + matrixRows*l_cols] = v_row_dev[i_row];
    i_row += it.get_local_range(0) * it.get_group_range(0);
  }

}

template <typename T, typename T_real>
void sycl_set_e_vec_scale_set_one_store_v_row_FromC(
    T_real *e_vec_dev, T *vrl_dev, T *a_dev, T *v_row_dev, T *tau_dev,
    T *xf_host_or_dev, int *l_rows_in, int *l_cols_in, int *matrixRows_in,
    int *istep_in, bool *isOurProcessRow_in, bool *useCCL_in,
    bool *wantDebug_in, intptr_t my_stream) {

  int l_rows = *l_rows_in;   
  int l_cols = *l_cols_in;   
  int matrixRows = *matrixRows_in;
  int istep = *istep_in;
  bool isOurProcessRow = *isOurProcessRow_in;
  bool useCCL = *useCCL_in;
  bool wantDebug = *wantDebug_in;

  int blocks = std::max((l_rows+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK, 1);
  sycl::range<1> blocksPerGrid = sycl::range<1>(blocks);
  sycl::range<1> threadsPerBlock = sycl::range<1>(MAX_THREADS_PER_BLOCK); // TODO_23_11 change to NB
  
  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  //sycl::usm::alloc memoryType = sycl::get_pointer_type((void *)xf_host_or_dev, queue.get_context());
  sycl::usm::alloc memoryType = sycl::usm::alloc::host; // for now, CCL is not supported for Intel GPUs, so the pointer is always host

  if (memoryType == sycl::usm::alloc::host) 
    {
    T xf_host_value = *xf_host_or_dev;

    /*
DPCT1049:10: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/  
    
    queue.submit([&](sycl::handler &cgh) 
      {
      cgh.parallel_for(
          sycl::nd_range<1> ( blocksPerGrid * threadsPerBlock, threadsPerBlock),
                [=](sycl::nd_item<1> it) {
            sycl_set_e_vec_scale_set_one_store_v_row_kernel(
                e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_value,
                l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, it);
          });
      });
    queue.wait_and_throw();
  }
 else if (memoryType == sycl::usm::alloc::device) 
   {
    // CCL is not supported for Intel GPUs yet
/*
    q_ct1.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(blocks) * threadsPerBlock,
                          threadsPerBlock),
        [=](sycl::nd_item<1> it) {
          sycl_set_e_vec_scale_set_one_store_v_row_kernel(
              e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev,
              l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL,
              it);
        });
*/
   }
    
}

extern "C" void sycl_set_e_vec_scale_set_one_store_v_row_double_FromC(
    double *e_vec_dev, double *vrl_dev, double *a_dev, double *v_row_dev,
    double *tau_dev, double *xf_host_or_dev, int *l_rows_in, int *l_cols_in,
    int *matrixRows_in, int *istep_in, bool *isOurProcessRow_in,
    bool *useCCL_in, bool *wantDebug_in, intptr_t my_stream) {
  sycl_set_e_vec_scale_set_one_store_v_row_FromC(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, l_rows_in, l_cols_in, matrixRows_in, istep_in, isOurProcessRow_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void sycl_set_e_vec_scale_set_one_store_v_row_float_FromC(
    float *e_vec_dev, float *vrl_dev, float *a_dev, float *v_row_dev,
    float *tau_dev, float *xf_host_or_dev, int *l_rows_in, int *l_cols_in,
    int *matrixRows_in, int *istep_in, bool *isOurProcessRow_in,
    bool *useCCL_in, bool *wantDebug_in, intptr_t my_stream) {
  sycl_set_e_vec_scale_set_one_store_v_row_FromC(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, l_rows_in, l_cols_in, matrixRows_in, istep_in, isOurProcessRow_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void sycl_set_e_vec_scale_set_one_store_v_row_double_complex_FromC(
    double *e_vec_dev, std::complex<double> *vrl_dev, std::complex<double> *a_dev,
    std::complex<double> *v_row_dev, std::complex<double> *tau_dev,
    std::complex<double> *xf_host_or_dev, int *l_rows_in, int *l_cols_in,
    int *matrixRows_in, int *istep_in, bool *isOurProcessRow_in,
    bool *useCCL_in, bool *wantDebug_in, intptr_t my_stream) {
  sycl_set_e_vec_scale_set_one_store_v_row_FromC(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, l_rows_in, l_cols_in, matrixRows_in, istep_in, isOurProcessRow_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void sycl_set_e_vec_scale_set_one_store_v_row_float_complex_FromC(
    float *e_vec_dev, std::complex<float> *vrl_dev, std::complex<float> *a_dev,
    std::complex<float> *v_row_dev, std::complex<float> *tau_dev,
    std::complex<float> *xf_host_or_dev, int *l_rows_in, int *l_cols_in,
    int *matrixRows_in, int *istep_in, bool *isOurProcessRow_in,
    bool *useCCL_in, bool *wantDebug_in, intptr_t my_stream) {
  sycl_set_e_vec_scale_set_one_store_v_row_FromC(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, l_rows_in, l_cols_in, matrixRows_in, istep_in, isOurProcessRow_in, useCCL_in, wantDebug_in, my_stream);
}

//________________________________________________________________


template <typename T, typename T_value_or_pointer>
void sycl_store_u_v_in_uv_vu_kernel(T *vu_stored_rows_dev, T *uv_stored_cols_dev, T *v_row_dev, T *u_row_dev,
                T *v_col_dev, T *u_col_dev, T *tau_dev, T *aux_complex_dev, T_value_or_pointer vav_host_or_dev, T_value_or_pointer tau_host_or_dev,
                int l_rows, int l_cols, int n_stored_vecs, int max_local_rows, int max_local_cols, int istep, bool useCCL,
                sycl::nd_item<1> it){
  int tid = it.get_local_id(0) + it.get_group(0) * it.get_local_range(0);

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

    i_row += it.get_local_range(0) * it.get_group_range(0);
    }

  int i_col = tid;
  while (i_col < l_cols)
    {
    conjg_tau_v_col_dev = elpaDeviceMultiply(conjg_tau, v_col_dev[i_col]);
    uv_stored_cols_dev[i_col + max_local_cols*(2*n_stored_vecs+1)] = conjg_tau_v_col_dev;
    conjg_tau_v_col_dev = elpaDeviceMultiply(conjg_tau_v_col_dev, elpaDeviceNumber<T>(0.5));
    uv_stored_cols_dev[i_col + max_local_cols*(2*n_stored_vecs+0)] = elpaDeviceSubtract( elpaDeviceMultiply(conjg_tau_v_col_dev, vav) , u_col_dev[i_col] );

    i_col += it.get_local_range(0) * it.get_group_range(0);
    }


  if ((std::is_same<T, std::complex<double>>::value || std::is_same<T, std::complex<float>>::value) && l_cols>0)
    {
    int j = tid;
    while (j < 2*(n_stored_vecs+0)) // whole vector aux_complex_dev has to be copied and not two last elements only because l_cols has changed since last istep
      {
      aux_complex_dev[j] = elpaDeviceComplexConjugate(uv_stored_cols_dev[l_cols-1 + max_local_cols*j]);
      j += it.get_local_range(0) * it.get_group_range(0);
      }
    
    // two last elements should be treated by the respective threads inorder to avoid sync problems
    i_col -= it.get_local_range(0) * it.get_group_range(0);
    if (i_col == l_cols-1) 
      {
      aux_complex_dev[2*n_stored_vecs+0] = elpaDeviceComplexConjugate(uv_stored_cols_dev[l_cols-1 + max_local_cols*(2*n_stored_vecs+0)]);
      aux_complex_dev[2*n_stored_vecs+1] = elpaDeviceComplexConjugate(uv_stored_cols_dev[l_cols-1 + max_local_cols*(2*n_stored_vecs+1)]);
      }
    }

}

template <typename T>
void sycl_store_u_v_in_uv_vu_FromC(
    T *vu_stored_rows_dev, T *uv_stored_cols_dev, T *v_row_dev, T *u_row_dev,
    T *v_col_dev, T *u_col_dev, T *tau_dev, T *aux_complex_dev, T *vav_host_or_dev, T *tau_host_or_dev, 
    int *l_rows_in, int *l_cols_in, int *n_stored_vecs_in, int *max_local_rows_in, int *max_local_cols_in, int *istep_in, 
    bool *useCCL_in, bool *wantDebug_in, intptr_t my_stream) {

  int l_rows = *l_rows_in;   
  int l_cols = *l_cols_in;   
  int n_stored_vecs  = *n_stored_vecs_in;
  int max_local_rows = *max_local_rows_in;   
  int max_local_cols = *max_local_cols_in;   
  int istep = *istep_in;   
  bool useCCL = *useCCL_in;
  bool wantDebug = *wantDebug_in;
  
  int threads = MAX_THREADS_PER_BLOCK/2; // the kernel has many local variables, for which we need memory registers. So we use less threads here to save memory.
  int blocks = std::max({(l_rows+threads-1)/threads, (l_cols+threads-1)/threads, 1});

  sycl::range<1> blocksPerGrid = sycl::range<1>(blocks);
  sycl::range<1> threadsPerBlock = sycl::range<1>(threads);

  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  //sycl::usm::alloc memoryType = sycl::get_pointer_type((void *)vav_host_or_dev, queue.get_context());
  sycl::usm::alloc memoryType = sycl::usm::alloc::host; // for now, CCL is not supported for Intel GPUs, so the pointer is always host

  if (memoryType == sycl::usm::alloc::host) 
    {
    T vav_host_value = *vav_host_or_dev;
    T tau_host_value = *tau_host_or_dev;

    /*
DPCT1049:16: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
    queue.parallel_for(
        sycl::nd_range<1>(blocksPerGrid * threadsPerBlock, threadsPerBlock),
        [=](sycl::nd_item<1> it) {
          sycl_store_u_v_in_uv_vu_kernel(
              vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev,
              v_col_dev, u_col_dev, tau_dev, aux_complex_dev, vav_host_value,
              tau_host_value, l_rows, l_cols, n_stored_vecs, max_local_rows,
              max_local_cols, istep, useCCL, it);
        });
    queue.wait_and_throw();
   } 
  
  else if (memoryType == sycl::usm::alloc::device) 
    {
    // CCL is not supported for Intel GPUs yet
    /*

    q_ct1.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(blocks) * threadsPerBlock,
                          threadsPerBlock),
        [=](sycl::nd_item<1> it) {
          sycl_store_u_v_in_uv_vu_kernel(
              vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev,
              v_col_dev, u_col_dev, tau_dev, aux_complex_dev, vav_host_or_dev,
              tau_host_or_dev, l_rows, l_cols, n_stored_vecs, max_local_rows,
              max_local_cols, istep, useCCL, it);
        });
    */
    } 

}

extern "C" void sycl_store_u_v_in_uv_vu_double_FromC(
    double *vu_stored_rows_dev, double *uv_stored_cols_dev, double *v_row_dev,
    double *u_row_dev, double *v_col_dev, double *u_col_dev, double *tau_dev,
    double *aux_complex_dev, double *vav_host_or_dev, double *tau_host_or_dev,
    int *l_rows_in, int *l_cols_in, int *n_stored_vecs_in,
    int *max_local_rows_in, int *max_local_cols_in, int *istep_in,
    bool *useCCL_in, bool *wantDebug_in, intptr_t my_stream) {
  sycl_store_u_v_in_uv_vu_FromC(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, v_col_dev, u_col_dev, tau_dev, aux_complex_dev, vav_host_or_dev, tau_host_or_dev, 
                                l_rows_in, l_cols_in, n_stored_vecs_in, max_local_rows_in, max_local_cols_in, istep_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void sycl_store_u_v_in_uv_vu_float_FromC(
    float *vu_stored_rows_dev, float *uv_stored_cols_dev, float *v_row_dev,
    float *u_row_dev, float *v_col_dev, float *u_col_dev, float *tau_dev,
    float *aux_complex_dev, float *vav_host_or_dev, float *tau_host_or_dev,
    int *l_rows_in, int *l_cols_in, int *n_stored_vecs_in,
    int *max_local_rows_in, int *max_local_cols_in, int *istep_in,
    bool *useCCL_in, bool *wantDebug_in, intptr_t my_stream) {
  sycl_store_u_v_in_uv_vu_FromC(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, v_col_dev, u_col_dev, tau_dev, aux_complex_dev, vav_host_or_dev, tau_host_or_dev, l_rows_in, l_cols_in, n_stored_vecs_in, max_local_rows_in, max_local_cols_in, istep_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void sycl_store_u_v_in_uv_vu_double_complex_FromC(
    std::complex<double> *vu_stored_rows_dev, std::complex<double> *uv_stored_cols_dev,
    std::complex<double> *v_row_dev, std::complex<double> *u_row_dev,
    std::complex<double> *v_col_dev, std::complex<double> *u_col_dev, std::complex<double> *tau_dev,
    std::complex<double> *aux_complex_dev, std::complex<double> *vav_host_or_dev,
    std::complex<double> *tau_host_or_dev, int *l_rows_in, int *l_cols_in,
    int *n_stored_vecs_in, int *max_local_rows_in, int *max_local_cols_in,
    int *istep_in, bool *useCCL_in, bool *wantDebug_in,
    intptr_t my_stream) {
  sycl_store_u_v_in_uv_vu_FromC(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, v_col_dev, u_col_dev, tau_dev, aux_complex_dev, vav_host_or_dev, tau_host_or_dev, l_rows_in, l_cols_in, n_stored_vecs_in, max_local_rows_in, max_local_cols_in, istep_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void sycl_store_u_v_in_uv_vu_float_complex_FromC(
    std::complex<float> *vu_stored_rows_dev, std::complex<float> *uv_stored_cols_dev,
    std::complex<float> *v_row_dev, std::complex<float> *u_row_dev, std::complex<float> *v_col_dev,
    std::complex<float> *u_col_dev, std::complex<float> *tau_dev,
    std::complex<float> *aux_complex_dev, std::complex<float> *vav_host_or_dev,
    std::complex<float> *tau_host_or_dev, int *l_rows_in, int *l_cols_in,
    int *n_stored_vecs_in, int *max_local_rows_in, int *max_local_cols_in,
    int *istep_in, bool *useCCL_in, bool *wantDebug_in,
    intptr_t my_stream) {
  sycl_store_u_v_in_uv_vu_FromC(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, v_col_dev, u_col_dev, tau_dev, aux_complex_dev, vav_host_or_dev, tau_host_or_dev, l_rows_in, l_cols_in, n_stored_vecs_in, max_local_rows_in, max_local_cols_in, istep_in, useCCL_in, wantDebug_in, my_stream);
}

//________________________________________________________________

template <typename T, typename T_real>
void sycl_update_matrix_element_add_kernel(T *vu_stored_rows_dev, T *uv_stored_cols_dev, T *a_dev, T_real *d_vec_dev, 
                                                      int l_rows, int l_cols, int matrixRows, int max_local_rows, int max_local_cols, int istep, int n_stored_vecs, bool isSkewsymmetric,
                                                      sycl::nd_item<1> it, sycl::accessor<T, 1, sycl::access_mode::read_write, sycl::access::target::local>  cache){
  
  const int threadsPerBlock = MAX_THREADS_PER_BLOCK;

  int tid = it.get_local_id(0) + it.get_group(0) * it.get_local_range(0);

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

  // if (it.get_local_id(0) == 0)
  //   { 
  //   if (isSkewsymmetric) 
  //     d_vec_dev[istep-1-1] = 0.0;
  //   else 
  //     d_vec_dev[istep-1-1] = elpaDeviceRealPart(a_dev[(l_rows-1) + matrixRows*(l_cols-1)]); // set initial value // TODO_23_11: move this to other kernel for thread safety
  //   }
  if (n_stored_vecs > 0)
    {

    T temp =  elpaDeviceNumber<T>(0.0);
    int index_n = tid;
    while (index_n < 2*n_stored_vecs) 
      {
      temp = elpaDeviceAdd(temp, elpaDeviceMultiply(elpaDeviceComplexConjugate(vu_stored_rows_dev[(l_rows-1)+max_local_rows*index_n]), uv_stored_cols_dev[(l_cols-1)+max_local_cols*index_n]) ); // temp += vu_stored_rows_dev[(l_rows-1)+max_local_rows*index_n] * uv_stored_cols_dev[(l_cols-1)+max_local_cols*index_n]
      index_n += it.get_local_range(0) * it.get_group_range(0);
      }

    // set the cache values
    cache[it.get_local_id(0)] = temp;
    // synchronize threads in this block
    /*
DPCT1065:22: Consider replacing sycl::nd_item::barrier() with
sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
performance if there is no access to global memory.
*/
    it.barrier();

    // for reductions, threadsPerBlock must be a power of 2
    int i = it.get_local_range(0) / 2;
    while (i > 0) 
      {
      if (it.get_local_id(0) < i) cache[it.get_local_id(0)] =
          elpaDeviceAdd( cache[it.get_local_id(0)], cache[it.get_local_id(0) + i]); // cache[threadIdx.x] += cache[threadIdx.x + i];
      /*
DPCT1065:23: Consider replacing sycl::nd_item::barrier() with
sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
performance if there is no access to global memory.
*/
      it.barrier();
      i /= 2;
      }

    if (it.get_local_id(0) == 0)
      {
      atomicAdd(&a_dev[(l_rows-1) + matrixRows*(l_cols-1)], cache[0]);
      //sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_sum (a_dev[(l_rows-1) + matrixRows*(l_cols-1)]);
      //atomic_sum += cache[0];

      if (!isSkewsymmetric) 
        {
        atomicAdd( &d_vec_dev[istep-1-1] , elpaDeviceRealPart(cache[0]) );
        //sycl::atomic_ref<T_real, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_sum_real (d_vec_dev[istep-1-1]);
        //atomic_sum_real += elpaDeviceRealPart(cache[0]);
        }
      }
    }
}

template <typename T, typename T_real>
void sycl_update_matrix_element_add_FromC(
    T *vu_stored_rows_dev, T *uv_stored_cols_dev, T *a_dev, T_real *d_vec_dev,
    int *l_rows_in, int *l_cols_in, int *matrixRows_in, int *max_local_rows_in,
    int *max_local_cols_in, int *istep_in, int *n_stored_vecs_in,
    bool *isSkewsymmetric_in, bool *wantDebug_in, intptr_t my_stream) {
  int l_rows = *l_rows_in;   
  int l_cols = *l_cols_in;
  int matrixRows = *matrixRows_in;
  int max_local_rows = *max_local_rows_in;
  int max_local_cols = *max_local_cols_in;
  int istep = *istep_in;   
  int n_stored_vecs = *n_stored_vecs_in; 
  bool isSkewsymmetric = *isSkewsymmetric_in;   
  bool wantDebug = *wantDebug_in;
  
  int blocks = std::min((2*n_stored_vecs+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK, 32);
  if (n_stored_vecs==0) blocks=1;

  sycl::range<1> blocksPerGrid   = sycl::range<1>(blocks);
  sycl::range<1> threadsPerBlock = sycl::range<1>(MAX_THREADS_PER_BLOCK);

  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  /*
DPCT1049:24: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/

  queue.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write, sycl::access::target::local> cache_acc_ct1(sycl::range<1>(1024 /*1024*/), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(blocksPerGrid * threadsPerBlock, threadsPerBlock),
          [=](sycl::nd_item<1> item_ct1) {
          sycl_update_matrix_element_add_kernel(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, l_rows,
                                  l_cols, matrixRows, max_local_rows, max_local_cols, istep,
                                  n_stored_vecs, isSkewsymmetric,
                                  item_ct1, cache_acc_ct1);
        });
  });
  queue.wait_and_throw();
  
  // queue.parallel_for(
  //     sycl::nd_range<1>(blocksPerGrid * threadsPerBlock, threadsPerBlock),
  //     [=](sycl::nd_item<1> it) {
  //       sycl_update_matrix_element_add_kernel(
  //           vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, l_rows,
  //           l_cols, matrixRows, max_local_rows, max_local_cols, istep,
  //           n_stored_vecs, isSkewsymmetric, it);
  //     });
  // queue.wait_and_throw();
}

extern "C" void sycl_update_matrix_element_add_double_FromC(
    double *vu_stored_rows_dev, double *uv_stored_cols_dev, double *a_dev,
    double *d_vec_dev, int *l_rows_in, int *l_cols_in, int *matrixRows_in,
    int *max_local_rows_in, int *max_local_cols_in, int *istep_in,
    int *n_stored_vecs_in, bool *isSkewsymmetric_in, bool *wantDebug_in,
    intptr_t my_stream) {
  sycl_update_matrix_element_add_FromC(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, l_rows_in, l_cols_in, matrixRows_in, max_local_rows_in, max_local_cols_in, istep_in, n_stored_vecs_in, isSkewsymmetric_in, wantDebug_in, my_stream);
}

extern "C" void sycl_update_matrix_element_add_float_FromC(
    float *vu_stored_rows_dev, float *uv_stored_cols_dev, float *a_dev,
    float *d_vec_dev, int *l_rows_in, int *l_cols_in, int *matrixRows_in,
    int *max_local_rows_in, int *max_local_cols_in, int *istep_in,
    int *n_stored_vecs_in, bool *isSkewsymmetric_in, bool *wantDebug_in,
    intptr_t my_stream) {
  sycl_update_matrix_element_add_FromC(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, l_rows_in, l_cols_in, matrixRows_in, max_local_rows_in, max_local_cols_in, istep_in, n_stored_vecs_in, isSkewsymmetric_in, wantDebug_in, my_stream);
}

extern "C" void sycl_update_matrix_element_add_double_complex_FromC(
    std::complex<double> *vu_stored_rows_dev, std::complex<double> *uv_stored_cols_dev,
    std::complex<double> *a_dev, double *d_vec_dev, int *l_rows_in, int *l_cols_in,
    int *matrixRows_in, int *max_local_rows_in, int *max_local_cols_in,
    int *istep_in, int *n_stored_vecs_in, bool *isSkewsymmetric_in,
    bool *wantDebug_in, intptr_t my_stream) {
 sycl_update_matrix_element_add_FromC(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, l_rows_in, l_cols_in, matrixRows_in, max_local_rows_in, max_local_cols_in, istep_in, n_stored_vecs_in, isSkewsymmetric_in, wantDebug_in, my_stream);
}

extern "C" void sycl_update_matrix_element_add_float_complex_FromC(
    std::complex<float> *vu_stored_rows_dev, std::complex<float> *uv_stored_cols_dev,
    std::complex<float> *a_dev, float *d_vec_dev, int *l_rows_in, int *l_cols_in,
    int *matrixRows_in, int *max_local_rows_in, int *max_local_cols_in,
    int *istep_in, int *n_stored_vecs_in, bool *isSkewsymmetric_in,
    bool *wantDebug_in, intptr_t my_stream) {
 sycl_update_matrix_element_add_FromC(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, l_rows_in, l_cols_in, matrixRows_in, max_local_rows_in, max_local_cols_in, istep_in, n_stored_vecs_in, isSkewsymmetric_in, wantDebug_in, my_stream);
}

//________________________________________________________________

template <typename T>
void sycl_update_array_element_kernel(T *array_dev, const int index, T value){

  array_dev[index-1] = value;

}

template <typename T>
void sycl_update_array_element_FromC(T *array_dev, int *index_in, T *value_in,
                                     intptr_t my_stream) {
  int index = *index_in;   
  T value = *value_in;

  sycl::range<1> blocksPerGrid   = sycl::range<1>(1);
  sycl::range<1> threadsPerBlock = sycl::range<1>(1);

  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  /*
DPCT1049:27: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
  queue.parallel_for(
      sycl::nd_range<1>(blocksPerGrid * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<1> it) {
        sycl_update_array_element_kernel(array_dev, index, value);
      });
  queue.wait_and_throw();
}

extern "C" void sycl_update_array_element_double_FromC(double *array_dev,
                                                       int *index_in,
                                                       double *value_in,
                                                       intptr_t my_stream) {
  sycl_update_array_element_FromC(array_dev, index_in, value_in, my_stream);
}

extern "C" void sycl_update_array_element_float_FromC(float *array_dev,
                                                      int *index_in,
                                                      float *value_in,
                                                      intptr_t my_stream) {
  sycl_update_array_element_FromC(array_dev, index_in, value_in, my_stream);
}

extern "C" void sycl_update_array_element_double_complex_FromC(
    std::complex<double> *array_dev, int *index_in, std::complex<double> *value_in,
    intptr_t my_stream) {
  sycl_update_array_element_FromC(array_dev, index_in, value_in, my_stream);
}

extern "C" void sycl_update_array_element_float_complex_FromC(
    std::complex<float> *array_dev, int *index_in, std::complex<float> *value_in,
    intptr_t my_stream) {
  sycl_update_array_element_FromC(array_dev, index_in, value_in, my_stream);
}

//________________________________________________________________

template <typename T>
void sycl_hh_transform_kernel(T *alpha_dev, T *xnorm_sq_dev, T *xf_dev, T *tau_dev, bool wantDebug_in){

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
    T beta = elpaDeviceNumber<T> (sign( elpaDeviceSqrt( alpha_r*alpha_r + alpha_i*alpha_i +
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
void sycl_hh_transform_FromC(T *alpha_dev, T *xnorm_sq_dev, T *xf_dev,
                             T *tau_dev, int *index_in, bool *wantDebug_in,
                             intptr_t my_stream) {
  bool wantDebug = *wantDebug_in;

  sycl::range<1> blocksPerGrid   = sycl::range<1>(1);
  sycl::range<1> threadsPerBlock = sycl::range<1>(1);

  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  // trivial single-thread kernel, streams can't be used here
  /*
DPCT1049:30: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
  queue.parallel_for(
      sycl::nd_range<1>(blocksPerGrid * threadsPerBlock, threadsPerBlock),
        [=](sycl::nd_item<1> it) {
        sycl_hh_transform_kernel(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug);
      });
  queue.wait_and_throw();

}

extern "C" void
sycl_hh_transform_double_FromC(double *alpha_dev, double *xnorm_sq_dev,
                               double *xf_dev, double *tau_dev, int *index_in,
                               bool *wantDebug_in, intptr_t my_stream) {
  sycl_hh_transform_FromC(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, index_in, wantDebug_in, my_stream);
}

extern "C" void sycl_hh_transform_float_FromC(float *alpha_dev,
                                              float *xnorm_sq_dev,
                                              float *xf_dev, float *tau_dev,
                                              int *index_in, bool *wantDebug_in,
                                              intptr_t my_stream) {
  sycl_hh_transform_FromC(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, index_in, wantDebug_in, my_stream);
}

extern "C" void sycl_hh_transform_double_complex_FromC(
    std::complex<double> *alpha_dev, std::complex<double> *xnorm_sq_dev,
    std::complex<double> *xf_dev, std::complex<double> *tau_dev, int *index_in,
    bool *wantDebug_in, intptr_t my_stream) {
  sycl_hh_transform_FromC(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, index_in, wantDebug_in, my_stream);
}

extern "C" void sycl_hh_transform_float_complex_FromC(
    std::complex<float> *alpha_dev, std::complex<float> *xnorm_sq_dev, std::complex<float> *xf_dev,
    std::complex<float> *tau_dev, int *index_in, bool *wantDebug_in,
    intptr_t my_stream) {
  sycl_hh_transform_FromC(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, index_in, wantDebug_in, my_stream);
}

//________________________________________________________________

template <typename T>
void sycl_transpose_reduceadd_vectors_copy_block_kernel(T *aux_transpose_dev, T *vmat_st_dev, 
                                              int nvc, int nvr, int n_block, int nblks_skip, int nblks_tot, 
                                              int lcm_s_t, int nblk, int auxstride, int np_st, int ld_st, int direction, bool isSkewsymmetric, bool isReduceadd,
                                              sycl::nd_item<1> it){
  int tid_x = it.get_local_id(0) + it.get_group(0) * it.get_local_range(0);

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

  T sign = elpaDeviceNumber<T>(1.0);
  if (isSkewsymmetric) sign = elpaDeviceNumber<T>(-1.0);

  int k, ns, nl;
  for (int lc=1; lc <= nvc; lc += 1)
    {
    for (int i = nblks_skip+n_block; i <= nblks_tot-1; i += lcm_s_t)
      {
      k = (i - nblks_skip - n_block)/lcm_s_t * nblk + (lc - 1) * auxstride;
      ns = (i/np_st)*nblk; // local start of block i
      nl = MIN(nvr-i*nblk, nblk); // length
      for (int j = tid_x; j < nl;
           j += it.get_local_range(0) * it.get_group_range(0))
        {
        if (direction==1)                 aux_transpose_dev[k+1+j-1]            = vmat_st_dev[ns+1+j-1 + (lc-1)*ld_st];
        if (direction==2 && !isReduceadd) vmat_st_dev[ns+1+j-1 + (lc-1)*ld_st]  = elpaDeviceMultiply(sign, aux_transpose_dev[k+1+j-1]);
        if (direction==2 &&  isReduceadd) vmat_st_dev[ns+1+j-1 + (lc-1)*ld_st]  = elpaDeviceAdd(vmat_st_dev[ns+1+j-1 + (lc-1)*ld_st] , aux_transpose_dev[k+1+j-1]);
        }
      }
    }

}

template <typename T>
void sycl_transpose_reduceadd_vectors_copy_block_FromC(
    T *aux_transpose_dev, T *vmat_st_dev, int *nvc_in, int *nvr_in,
    int *n_block_in, int *nblks_skip_in, int *nblks_tot_in, int *lcm_s_t_in,
    int *nblk_in, int *auxstride_in, int *np_st_in, int *ld_st_in,
    int *direction_in, bool *isSkewsymmetric_in, bool *isReduceadd_in,
    bool *wantDebug_in, intptr_t my_stream) {

  int nvc = *nvc_in;   
  int nvr = *nvr_in;   
  int n_block = *n_block_in;
  int nblks_skip = *nblks_skip_in;
  int nblks_tot = *nblks_tot_in;
  int lcm_s_t = *lcm_s_t_in;
  int nblk = *nblk_in;
  int auxstride = *auxstride_in;
  int np_st = *np_st_in;
  int ld_st = *ld_st_in;
  int direction = *direction_in;
  bool isSkewsymmetric = *isSkewsymmetric_in;
  bool isReduceadd = *isReduceadd_in;
  bool wantDebug = *wantDebug_in;

  int SM_count=32; // TODO_23_11 count and move outside
  int blocks = SM_count;

  sycl::range<1> blocksPerGrid = sycl::range<1>(blocks);
  sycl::range<1> threadsPerBlock = sycl::range<1>(nblk);

  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  /*
DPCT1049:33: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
  queue.parallel_for(
    sycl::nd_range<1>(blocksPerGrid * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<1> it) {
      sycl_transpose_reduceadd_vectors_copy_block_kernel(
          aux_transpose_dev, vmat_st_dev, nvc, nvr, n_block, nblks_skip,
          nblks_tot, lcm_s_t, nblk, auxstride, np_st, ld_st, direction,
          isSkewsymmetric, isReduceadd, it);
    });
  queue.wait_and_throw();

}

extern "C" void sycl_transpose_reduceadd_vectors_copy_block_double_FromC(
    double *aux_transpose_dev, double *vmat_st_dev, int *nvc_in, int *nvr_in,
    int *n_block_in, int *nblks_skip_in, int *nblks_tot_in, int *lcm_s_t_in,
    int *nblk_in, int *auxstride_in, int *np_st_in, int *ld_st_in,
    int *direction_in, bool *isSkewsymmetric_in, bool *isReduceadd_in,
    bool *wantDebug_in, intptr_t my_stream) {
  sycl_transpose_reduceadd_vectors_copy_block_FromC(aux_transpose_dev, vmat_st_dev, nvc_in, nvr_in, n_block_in, nblks_skip_in, nblks_tot_in, lcm_s_t_in, nblk_in, auxstride_in, np_st_in, ld_st_in, direction_in, isSkewsymmetric_in, isReduceadd_in, wantDebug_in, my_stream);
}

extern "C" void sycl_transpose_reduceadd_vectors_copy_block_float_FromC(
    float *aux_transpose_dev, float *vmat_st_dev, int *nvc_in, int *nvr_in,
    int *n_block_in, int *nblks_skip_in, int *nblks_tot_in, int *lcm_s_t_in,
    int *nblk_in, int *auxstride_in, int *np_st_in, int *ld_st_in,
    int *direction_in, bool *isSkewsymmetric_in, bool *isReduceadd_in,
    bool *wantDebug_in, intptr_t my_stream) {
  sycl_transpose_reduceadd_vectors_copy_block_FromC(aux_transpose_dev, vmat_st_dev, nvc_in, nvr_in, n_block_in, nblks_skip_in, nblks_tot_in, lcm_s_t_in, nblk_in, auxstride_in, np_st_in, ld_st_in, direction_in, isSkewsymmetric_in, isReduceadd_in, wantDebug_in, my_stream);
}

extern "C" void
sycl_transpose_reduceadd_vectors_copy_block_double_complex_FromC(
    std::complex<double> *aux_transpose_dev, std::complex<double> *vmat_st_dev, int *nvc_in,
    int *nvr_in, int *n_block_in, int *nblks_skip_in, int *nblks_tot_in,
    int *lcm_s_t_in, int *nblk_in, int *auxstride_in, int *np_st_in,
    int *ld_st_in, int *direction_in, bool *isSkewsymmetric_in,
    bool *isReduceadd_in, bool *wantDebug_in, intptr_t my_stream) {
  sycl_transpose_reduceadd_vectors_copy_block_FromC(aux_transpose_dev, vmat_st_dev, nvc_in, nvr_in, n_block_in, nblks_skip_in, nblks_tot_in, lcm_s_t_in, nblk_in, auxstride_in, np_st_in, ld_st_in, direction_in, isSkewsymmetric_in, isReduceadd_in, wantDebug_in, my_stream);
}

extern "C" void sycl_transpose_reduceadd_vectors_copy_block_float_complex_FromC(
    std::complex<float> *aux_transpose_dev, std::complex<float> *vmat_st_dev, int *nvc_in,
    int *nvr_in, int *n_block_in, int *nblks_skip_in, int *nblks_tot_in,
    int *lcm_s_t_in, int *nblk_in, int *auxstride_in, int *np_st_in,
    int *ld_st_in, int *direction_in, bool *isSkewsymmetric_in,
    bool *isReduceadd_in, bool *wantDebug_in, intptr_t my_stream) {
  sycl_transpose_reduceadd_vectors_copy_block_FromC(aux_transpose_dev, vmat_st_dev, nvc_in, nvr_in, n_block_in, nblks_skip_in, nblks_tot_in, lcm_s_t_in, nblk_in, auxstride_in, np_st_in, ld_st_in, direction_in, isSkewsymmetric_in, isReduceadd_in, wantDebug_in, my_stream);
}

//________________________________________________________________