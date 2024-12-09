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


#ifdef WITH_NVIDIA_GPU_VERSION
#define INLINE_DEVICE __forceinline__ __device__
#define double_complex cuDoubleComplex
#define float_complex  cuFloatComplex
#endif

#ifdef WITH_AMD_GPU_VERSION
#define INLINE_DEVICE __forceinline__ __device__
#define double_complex hipDoubleComplex
#define float_complex  hipFloatComplex
#endif

#ifdef WITH_SYCL_GPU_VERSION
#define INLINE_DEVICE inline
#define double_complex std::complex<double>
#define float_complex  std::complex<float>
#endif

//_________________________________________________________________________________________________
// Generic math host functions

// construct a generic double/float/double_complex/float_complex from a double
template <typename T> inline T elpaHostNumberFromInt(int number);
template <> inline double elpaHostNumberFromInt<double>(int number) {return (double) number;}
template <> inline float  elpaHostNumberFromInt<float> (int number) {return (float) number;}
template <> inline double_complex elpaHostNumberFromInt<double_complex>(int number) {
#if defined(WITH_NVIDIA_GPU_VERSION)
  return make_cuDoubleComplex ((double)number, 0.0);
#elif defined(WITH_AMD_GPU_VERSION)
  return make_hipDoubleComplex((double)number, 0.0);
#else
  return std::complex<double> ((double)number, 0.0);
#endif
}
template <>  inline float_complex elpaHostNumberFromInt<float_complex> (int number) {
#if defined(WITH_NVIDIA_GPU_VERSION)
  return make_cuFloatComplex ((float) number, 0.0f);
#elif defined(WITH_AMD_GPU_VERSION)
  return make_hipFloatComplex((float) number, 0.0f);
#else
  return std::complex<float> ((float) number, 0.0f);
#endif
}

//_________________________________________________________________________________________________
// Generic math device functions

template <typename T> 
INLINE_DEVICE T elpaDeviceSign(T a, T b) {
  if (b>=0) return fabs(a);
  else return -fabs(a);
}

// construct a generic double/float/double_complex/float_complex from a double
template <typename T> INLINE_DEVICE T elpaDeviceNumber(double number);
template <>  INLINE_DEVICE double elpaDeviceNumber<double>(double number) {return number;}
template <>  INLINE_DEVICE float  elpaDeviceNumber<float> (double number) {return (float) number;}
template <>  INLINE_DEVICE double_complex elpaDeviceNumber<double_complex>(double number) {
#if defined(WITH_NVIDIA_GPU_VERSION)
  return make_cuDoubleComplex (number, 0.0);
#elif defined(WITH_AMD_GPU_VERSION)
  return make_hipDoubleComplex(number, 0.0);
#else
  return std::complex<double> (number, 0.0);
#endif
}
template <>  INLINE_DEVICE float_complex elpaDeviceNumber<float_complex> (double number) {
#if defined(WITH_NVIDIA_GPU_VERSION)
  return make_cuFloatComplex ((float) number, 0.0f);
#elif defined(WITH_AMD_GPU_VERSION)
  return make_hipFloatComplex((float) number, 0.0f);
#else
  return std::complex<float> ((float) number, 0.0f);
#endif
}

// construct a generic double/float/double_complex/float_complex from a real and imaginary parts
template <typename T, typename T_real>  INLINE_DEVICE T elpaDeviceNumberFromRealImag(T_real Re, T_real Im);
template <> INLINE_DEVICE double elpaDeviceNumberFromRealImag<double>(double Real, double Imag) {return Real;}
template <> INLINE_DEVICE float  elpaDeviceNumberFromRealImag<float> (float  Real, float  Imag) {return Real;}
template <> INLINE_DEVICE double_complex elpaDeviceNumberFromRealImag<double_complex>(double Real, double Imag) {
#if defined(WITH_NVIDIA_GPU_VERSION)
  return make_cuDoubleComplex(Real, Imag);
#elif defined(WITH_AMD_GPU_VERSION)
  return make_hipDoubleComplex(Real, Imag);
#else
  return std::complex<double>(Real, Imag);
#endif
}
template <> INLINE_DEVICE float_complex elpaDeviceNumberFromRealImag<float_complex>(float  Real, float  Imag) {
#if defined(WITH_NVIDIA_GPU_VERSION)
  return make_cuFloatComplex (Real, Imag);
#elif defined(WITH_AMD_GPU_VERSION)
  return make_hipFloatComplex(Real, Imag);
#else
  return std::complex<float>(Real, Imag);
#endif
}

INLINE_DEVICE double elpaDeviceAdd(double a, double b) { return a + b; }
INLINE_DEVICE float  elpaDeviceAdd(float a, float b)   { return a + b; }
INLINE_DEVICE double_complex elpaDeviceAdd(double_complex a, double_complex b) {
#if defined(WITH_NVIDIA_GPU_VERSION)  
  return cuCadd (a, b); 
#elif defined(WITH_AMD_GPU_VERSION)
  return hipCadd(a, b);
#else
  return a + b;
#endif  
}
INLINE_DEVICE float_complex elpaDeviceAdd(float_complex a, float_complex b) {
#if defined(WITH_NVIDIA_GPU_VERSION)
  return cuCaddf(a, b);
#elif defined(WITH_AMD_GPU_VERSION)
  return hipCaddf(a, b);
#else
  return a + b;
#endif
}

INLINE_DEVICE double elpaDeviceSubtract(double a, double b) { return a - b; }
INLINE_DEVICE float  elpaDeviceSubtract(float a, float b)   { return a - b; }
INLINE_DEVICE double_complex elpaDeviceSubtract(double_complex a, double_complex b) {
#if defined(WITH_NVIDIA_GPU_VERSION)
  return cuCsub (a, b);
#elif defined(WITH_AMD_GPU_VERSION)
  return hipCsub(a, b);
#else
  return a - b;
#endif
}
INLINE_DEVICE float_complex elpaDeviceSubtract(float_complex a, float_complex b) {
#if defined(WITH_NVIDIA_GPU_VERSION)
  return cuCsubf(a, b);
#elif defined(WITH_AMD_GPU_VERSION)
  return hipCsubf(a, b);
#else
  return a - b;
#endif
}

INLINE_DEVICE double elpaDeviceMultiply(double a, double b) { return a * b; }
INLINE_DEVICE float  elpaDeviceMultiply(float  a, float  b) { return a * b; }
INLINE_DEVICE double_complex elpaDeviceMultiply(double_complex a, double_complex b) {
#if defined(WITH_NVIDIA_GPU_VERSION)  
  return cuCmul (a, b); 
#elif defined(WITH_AMD_GPU_VERSION)
  return hipCmul(a, b);
#else
  return a * b;
#endif  
}
INLINE_DEVICE float_complex elpaDeviceMultiply(float_complex a, float_complex b) {
#if defined(WITH_NVIDIA_GPU_VERSION)
  return cuCmulf(a, b);
#elif defined(WITH_AMD_GPU_VERSION)
  return hipCmulf(a, b);
#else
  return a * b;
#endif
}

INLINE_DEVICE double elpaDeviceDivide(double a, double b) { return a / b; }
INLINE_DEVICE float  elpaDeviceDivide(float  a, float  b) { return a / b; }
INLINE_DEVICE double_complex elpaDeviceDivide(double_complex a, double_complex b) {
#if defined(WITH_NVIDIA_GPU_VERSION)  
  return cuCdiv (a, b); 
#elif defined(WITH_AMD_GPU_VERSION)
  return hipCdiv(a, b);
#else  
  return a / b;
#endif
}
INLINE_DEVICE float_complex elpaDeviceDivide(float_complex a, float_complex b) {
#if defined(WITH_NVIDIA_GPU_VERSION)
  return cuCdivf(a, b);
#elif defined(WITH_AMD_GPU_VERSION)
  return hipCdivf(a, b);
#else
  return a / b;
#endif
}

INLINE_DEVICE double elpaDeviceSqrt(double number) { return sqrt (number); }
INLINE_DEVICE float  elpaDeviceSqrt(float  number) { return sqrtf(number); }

INLINE_DEVICE double elpaDeviceComplexConjugate(double number) {return number;}
INLINE_DEVICE float elpaDeviceComplexConjugate(float  number) {return number;}
INLINE_DEVICE double_complex elpaDeviceComplexConjugate(double_complex number) {
#if defined(WITH_NVIDIA_GPU_VERSION) 
  return cuConj(number);
#elif defined(WITH_AMD_GPU_VERSION)
  return hipConj(number);
#else
  return std::conj(number);
#endif
}
INLINE_DEVICE float_complex elpaDeviceComplexConjugate(float_complex number) {
#if defined(WITH_NVIDIA_GPU_VERSION)
  return cuConjf(number);
#elif defined(WITH_AMD_GPU_VERSION)
  return hipConjf(number);
#else
  return std::conj(number);
#endif
}

INLINE_DEVICE double elpaDeviceRealPart(double number) {return number;}
INLINE_DEVICE float  elpaDeviceRealPart(float  number) {return number;}
INLINE_DEVICE double elpaDeviceRealPart(double_complex number) {
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION)  
  return number.x;
#else
  return number.real();
#endif
}
INLINE_DEVICE float  elpaDeviceRealPart(float_complex number) {
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION)    
  return number.x;
#else
  return number.real();
#endif  
}

INLINE_DEVICE double elpaDeviceImagPart(double number) {return 0.0;}
INLINE_DEVICE float  elpaDeviceImagPart(float  number) {return 0.0f;}
INLINE_DEVICE double elpaDeviceImagPart(double_complex number) {
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION)
  return number.y;
#else
  return number.imag();
#endif
}
INLINE_DEVICE float elpaDeviceImagPart(float_complex number) {
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION)
  return number.y;
#else
  return number.imag();
#endif
}

// Device function to convert a pointer to a value
template <typename T>
INLINE_DEVICE T convert_to_device(T* x, std::true_type) { return *x;}

// Device function to convert a value to a value
template <typename T>
INLINE_DEVICE T convert_to_device(T x, std::false_type) { return x;}

//_________________________________________________________________________________________________
// atomicAdd device function for real/complex numbers


#ifdef WITH_AMD_GPU_VERSION
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600 /* HIP ON NVIDIA */
static __inline__ __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
#endif

// atomicAdd for double_complex and float_complex
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION)
template<typename T>
INLINE_DEVICE void atomicAdd(T* address, T val) {
    atomicAdd(&(address->x), val.x);
    atomicAdd(&(address->y), val.y);
}
#endif

#ifdef WITH_SYCL_GPU_VERSION
template <typename T> 
INLINE_DEVICE void atomicAdd(T* address, T val)
  {
  sycl::atomic_ref<T, sycl::memory_order_relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_sum (*address);
  atomic_sum += val;
  }

template <>  
INLINE_DEVICE void atomicAdd(std::complex<double>* address, std::complex<double> val)
  {
  double* real_ptr = reinterpret_cast<double*>(address); // Pointer to the real part
  double* imag_ptr = real_ptr + 1; // Pointer to the imaginary part

  sycl::atomic_ref<double, sycl::memory_order_relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_sum_real(*real_ptr);
  sycl::atomic_ref<double, sycl::memory_order_relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_sum_imag(*imag_ptr);

  atomic_sum_real += val.real();
  atomic_sum_imag += val.imag();
  }

template <>  
INLINE_DEVICE void atomicAdd(std::complex<float>* address, std::complex<float> val)
  {
  float* real_ptr = reinterpret_cast<float*>(address); // Pointer to the real part
  float* imag_ptr = real_ptr + 1; // Pointer to the imaginary part

  sycl::atomic_ref<float, sycl::memory_order_relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_sum_real(*real_ptr);
  sycl::atomic_ref<float, sycl::memory_order_relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_sum_imag(*imag_ptr);

  atomic_sum_real += val.real();
  atomic_sum_imag += val.imag();
  }
#endif

//_________________________________________________________________________________________________
// ELPA-specific device functions
// PETERDEBUG: cleanup after testing

// INLINE_DEVICE int pcol(int I_gl, int nblk, int np_cols){
//   // C-style 0-based indexing in assumed
//   return (I_gl/nblk)%np_cols;
// }

// // same as pcol; just for naming consistency
// INLINE_DEVICE int prow(int J_gl, int nblk, int np_rows){
//   // C-style 0-based indexing in assumed
//   return (J_gl/nblk)%np_rows;
// }

// INLINE_DEVICE int local_index(int I_gl, int my_proc, int num_procs, int nblk, int iflag){

// //  local_index: returns the local index for a given global index
// //               If the global index has no local index on the
// //               processor my_proc, return next local index after that row/col
// //               C-style 0-based indexing in assumed
// //  Parameters
// //
// //  I_gl        Global index
// //  my_proc     Processor row/column for which to calculate the local index
// //  num_procs   Total number of processors along row/column
// //  nblk        Blocksize
// //
// //  iflag       Controls the behaviour if idx is not on local processor
// //              iflag< 0 : Return last local index before that row/col
// //              iflag==0 : Return 0
// //              iflag> 0 : Return next local index after that row/col
// //
// // L_block_gl = I_gl/nblk; // global ordinal number of the nblk-block among other blocks
// // l_block_loc = L_block_gl/num_procs =  I_gl/(num_procs*nblk); // local ordinal number of the nblk-block among other blocks
// // x = I_gl%nblk; // local coordinate within the block
// // local_index = l_block*nblk + x;

//   if ((I_gl/nblk)%num_procs == my_proc) // (L_block_gl%num_procs == my_proc), block is local
//     {
//     return I_gl/(num_procs*nblk)* nblk + I_gl%nblk; // local_index = l_block_loc * nblk + x
//     }
//   else if ((I_gl/nblk)%num_procs < my_proc) // block is non-local
//     {
//     if      (iflag ==  1) return I_gl/(num_procs*nblk)* nblk;
//     else if (iflag == -1) return I_gl/(num_procs*nblk)* nblk - 1;
//     else if (iflag ==  0) return 0;
//     }
//   else // ((I_gl/nblk)%num_procs > my_proc)
//     {
//     if      (iflag ==  1) return (I_gl/(num_procs*nblk) + 1)* nblk;
//     else if (iflag == -1) return (I_gl/(num_procs*nblk) + 1)* nblk - 1;
//     else if (iflag ==  0) return 0;
//     }
// }

//_________________________________________________________________________________________________
// ELPA-specific device functions

INLINE_DEVICE int pcol(int global_col, int nblk, int np_cols){
  return ((global_col-1)/nblk)%np_cols;
}

// same as pcol; just for naming consistency
INLINE_DEVICE int prow(int global_row, int nblk, int np_rows){
  return ((global_row-1)/nblk)%np_rows;
}

INLINE_DEVICE int local_index(int idx, int my_proc, int num_procs, int nblk, int iflag){
/*
!  local_index: returns the local index for a given global index
!               If the global index has no local index on the
!               processor my_proc, behaviour is defined by iflag
!
!  Parameters
!
!  idx         Global index
!
!  my_proc     Processor row/column for which to calculate the local index
!
!  num_procs   Total number of processors along row/column
!
!  nblk        Blocksize
!
!  iflag       Controls the behaviour if idx is not on local processor
!              iflag< 0 : Return last local index before that row/col
!              iflag==0 : Return 0
!              iflag> 0 : Return next local index after that row/col

// Fortran (1-based) to C (0-based) correspondence
// L_block_gl = I_gl/nblk; // global ordinal number of the nblk-block among other blocks
// l_block_loc = L_block_gl/num_procs =  I_gl/(num_procs*nblk); // local ordinal number of the nblk-block among other blocks
// x = I_gl%nblk; // local coordinate within the block
// local_index = l_block*nblk + x;
*/

  int result;

  int L_block_gl = (idx-1)/nblk; // global block number, 0 based

  if (L_block_gl%num_procs == my_proc) {
    // block is local, always return local row/col number
     result = (L_block_gl/num_procs)*nblk + ((idx-1) % nblk) + 1;
  }
  else {
    // non local block
    if (iflag == 0) result = 0; 
    else {
      result = (L_block_gl / num_procs) * nblk;

      if ((L_block_gl % num_procs) > my_proc) result += nblk; 
      if (iflag > 0) result += 1;
    }
  }

  return result;
}