//
//    Copyright 2022, P. Karpov, MPCDF
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

#include <stdio.h>
#include <math.h>

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <complex.h>

#include "./test_gpu_vendor_agnostic_layerFunctions.h"

#include "config-f90.h"

#ifdef WITH_NVIDIA_GPU_VERSION
#include "./CUDA/test_cudaFunctions.h"
#endif 
#ifdef WITH_AMD_GPU_VERSION
#include "./ROCm/test_rocmFunctions.h"
#endif 
#ifdef WITH_SYCL_GPU_VERSION
#include "./SYCL/test_syclFunctions.h"
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#error "openmp_offload missing"
#endif


void set_gpu_parameters(){
#ifdef WITH_NVIDIA_GPU_VERSION
   gpuMemcpyHostToDevice = cudaMemcpyHostToDeviceFromC();
   gpuMemcpyDeviceToHost = cudaMemcpyDeviceToHostFromC();
#endif 
#ifdef WITH_AMD_GPU_VERSION
   gpuMemcpyHostToDevice = hipMemcpyHostToDeviceFromC();
   gpuMemcpyDeviceToHost = hipMemcpyDeviceToHostFromC();
#endif
#ifdef WITH_SYCL_GPU_VERSION
   gpuMemcpyHostToDevice = syclMemcpyHostToDeviceFromC();
   gpuMemcpyDeviceToHost = syclMemcpyDeviceToHostFromC();
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#error "openmp_offload missing"
#endif
}

int gpuGetDeviceCount(int *count){
#ifdef WITH_NVIDIA_GPU_VERSION
   return cudaGetDeviceCountFromC(count);
#endif 
#ifdef WITH_AMD_GPU_VERSION
   return hipGetDeviceCountFromC(count);
#endif
#ifdef WITH_SYCL_GPU_VERSION
   return syclGetDeviceCountFromC(count);
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#error "openmp_offload missing"
#endif
   return -1;
}

int gpuSetDevice(int n){
#ifdef WITH_NVIDIA_GPU_VERSION
   return cudaSetDeviceFromC(n);
#endif   
#ifdef WITH_AMD_GPU_VERSION
   return hipSetDeviceFromC(n);
#endif
#ifdef WITH_SYCL_GPU_VERSION
   return syclSetDeviceFromC(n);
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#error "openmp_offload missing"
#endif
}

int gpuMalloc(intptr_t *a, size_t width_height) {
#ifdef WITH_NVIDIA_GPU_VERSION
   return cudaMallocFromC(a, width_height);
#endif   
#ifdef WITH_AMD_GPU_VERSION
   return hipMallocFromC(a, width_height);
#endif
#ifdef WITH_SYCL_GPU_VERSION
   return syclMallocFromC(a, width_height);
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#error "openmp_offload missing"
#endif   
}

int gpuFree(intptr_t *a) {
#ifdef WITH_NVIDIA_GPU_VERSION
   return cudaFreeFromC(a);
#endif   
#ifdef WITH_AMD_GPU_VERSION
   return hipFreeFromC(a);
#endif
#ifdef WITH_SYCL_GPU_VERSION
   //return syclFreeFromC(a);
   return syclFreeVoidPtr(a);
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#error "openmp_offload missing"
#endif   
}

int gpuMemcpy(intptr_t *dest, intptr_t *src, size_t count, int dir){
#ifdef WITH_NVIDIA_GPU_VERSION
   return cudaMemcpyFromC(dest, src, count, dir);
#endif  
#ifdef WITH_AMD_GPU_VERSION
   return hipMemcpyFromC(dest, src, count, dir);
#endif
#ifdef WITH_SYCL_GPU_VERSION
   return syclMemcpyFromC(dest, src, count, dir);
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#error "openmp_offload missing"
#endif  
}

#ifdef WITH_SYCL_GPU_VERSION
int syclGetCpuCount(int numberOfDevices) {
   return syclGetCpuCountFromC(&numberOfDevices);
}
#endif