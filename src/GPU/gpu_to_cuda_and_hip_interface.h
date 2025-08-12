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


#undef gpuDeviceSynchronize
#undef gpuStream_t
#undef gpuGetLastError
#undef gpuGetErrorString
#undef gpuError_t
#undef gpuSuccess
#undef gpuDoubleComplex
#undef gpuFloatComplex
#undef make_gpuDoubleComplex
#undef make_gpuFloatComplex
#undef MAX_THREADS_PER_BLOCK
#undef MIN_THREADS_PER_BLOCK
#undef ELPA_GPU

//_________________________________________________________________________________________________

#ifdef WITH_NVIDIA_GPU_VERSION
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuStream_t cudaStream_t
#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuDoubleComplex cuDoubleComplex
#define gpuFloatComplex  cuFloatComplex
#define make_gpuDoubleComplex make_cuDoubleComplex
#define make_gpuFloatComplex make_cuFloatComplex
#define MAX_THREADS_PER_BLOCK 1024
#define MIN_THREADS_PER_BLOCK 32 /* i.e. wrap size */
#define ELPA_GPU cuda
#endif

//_________________________________________________________________________________________________

#ifdef WITH_AMD_GPU_VERSION
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuStream_t hipStream_t
#define gpuGetLastError hipGetLastError
#define gpuGetErrorString hipGetErrorString
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuDoubleComplex hipDoubleComplex
#define gpuFloatComplex  hipFloatComplex
#define make_gpuDoubleComplex make_hipDoubleComplex
#define make_gpuFloatComplex make_hipFloatComplex
#define MAX_THREADS_PER_BLOCK 1024
#define MIN_THREADS_PER_BLOCK 64
#define ELPA_GPU hip
#endif

#ifdef WITH_SYCL_GPU_VERSION
#define gpuDeviceSynchronize syclDeviceSynchronize
#define gpuStream_t QueueData*
#define gpuGetLastError XXXERRORXXX
#define gpuGetErrorString XXXERRORXXX
#define gpuError_t XXXERRORXXX
#define gpuSuccess XXXERRORXXX
#define gpuDoubleComplex std::complex<double>
#define gpuFloatComplex std::complex<float>
#define make_gpuDoubleComplex std::complex<double>
#define make_gpuFloatComplex std::complex<float>
#define MAX_THREADS_PER_BLOCK 1024
#define MIN_THREADS_PER_BLOCK 16
#define ELPA_GPU sycl
#endif

#define CONCATENATE_WITHOUT_EVALUATION(prefix, name) prefix##name
#define CONCATENATE(prefix, name) CONCATENATE_WITHOUT_EVALUATION(prefix, name)
