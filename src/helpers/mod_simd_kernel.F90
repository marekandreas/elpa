!    Copyright 2018, A. Marek
!
!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!      Informatik,
!    - Technische Universität München, Lehrstuhl für Informatik mit
!      Schwerpunkt Wissenschaftliches Rechnen ,
!    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!
!    More information can be found here:
!    https://elpa.mpcdf.mpg.de/
!
!    ELPA is free software: you can redistribute it and/or modify
!    it under the terms of the version 3 of the license of the
!    GNU Lesser General Public License as published by the Free
!    Software Foundation.
!
!    ELPA is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public License
!    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
!
!    ELPA reflects a substantial effort on the part of the original
!    ELPA consortium, and we ask you to respect the spirit of the
!    license that we chose: i.e., please contribute any changes you
!    may have back to the original ELPA library distribution, and keep
!    any derivatives of ELPA under the same license that we chose for
!    the original distribution, the GNU Lesser General Public License.
!
! This file was written by A. Marek, MPCDF

#include "config-f90.h"
#include "elpa/elpa_simd_constants.h"

module simd_kernel
  use elpa_constants
  use, intrinsic :: iso_c_binding

  integer(kind=c_int) :: realKernels_to_simdTable(ELPA_2STAGE_NUMBER_OF_REAL_KERNELS)
  integer(kind=c_int) :: simdTable_to_realKernels(NUMBER_OF_INSTR)
  integer(kind=c_int) :: complexKernels_to_simdTable(ELPA_2STAGE_NUMBER_OF_COMPLEX_KERNELS)
  integer(kind=c_int) :: simdTable_to_complexKernels(NUMBER_OF_INSTR)

  contains

  function map_real_kernel_to_simd_instruction(kernel) result(simd_set_index)
    
    use, intrinsic :: iso_c_binding
    implicit none

    integer(kind=c_int), intent(in) :: kernel
    integer(kind=c_int)             :: simd_set_index

    realKernels_to_simdTable(ELPA_2STAGE_REAL_GENERIC)               = GENERIC_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_GENERIC_SIMPLE)        = GENERIC_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_BGP)                   = BLUEGENE_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_BGQ)                   = BLUEGENE_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_SSE_ASSEMBLY)          = SSE_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_SSE_BLOCK2)            = SSE_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_SSE_BLOCK4)            = SSE_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_SSE_BLOCK6)            = SSE_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_AVX_BLOCK2)            = AVX_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_AVX_BLOCK4)            = AVX_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_AVX_BLOCK6)            = AVX_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_AVX2_BLOCK2)           = AVX2_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_AVX2_BLOCK4)           = AVX2_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_AVX2_BLOCK6)           = AVX2_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_AVX512_BLOCK2)         = AVX512_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_AVX512_BLOCK4)         = AVX512_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_AVX512_BLOCK6)         = AVX512_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_SVE128_BLOCK2)         = SVE128_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_SVE128_BLOCK4)         = SVE128_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_SVE128_BLOCK6)         = SVE128_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_SVE256_BLOCK2)         = SVE256_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_SVE256_BLOCK4)         = SVE256_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_SVE256_BLOCK6)         = SVE256_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_SVE512_BLOCK2)         = SVE512_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_SVE512_BLOCK4)         = SVE512_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_SVE512_BLOCK6)         = SVE512_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_NVIDIA_GPU)            = NVIDIA_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_AMD_GPU)               = AMD_GPU_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_INTEL_GPU)             = INTEL_GPU_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_SPARC64_BLOCK2)        = SPARC_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_SPARC64_BLOCK4)        = SPARC_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_SPARC64_BLOCK6)        = SPARC_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_NEON_ARCH64_BLOCK2)    = ARCH64_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_NEON_ARCH64_BLOCK4)    = ARCH64_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_NEON_ARCH64_BLOCK6)    = ARCH64_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_VSX_BLOCK2)            = VSX_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_VSX_BLOCK4)            = VSX_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_VSX_BLOCK6)            = VSX_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_GENERIC_SIMPLE_BLOCK4) = GENERIC_INSTR
    realKernels_to_simdTable(ELPA_2STAGE_REAL_GENERIC_SIMPLE_BLOCK6) = GENERIC_INSTR

    simd_set_index = realKernels_to_simdTable(kernel)


  end

  function map_simd_instruction_to_real_kernel(simd_set_index) result(kernel)
    
    use, intrinsic :: iso_c_binding
    implicit none


    integer(kind=c_int)                        :: kernel
    integer(kind=c_int), intent(in)            :: simd_set_index

    simdTable_to_realKernels(GENERIC_INSTR)   = ELPA_2STAGE_REAL_GENERIC
    simdTable_to_realKernels(BLUEGENE_INSTR)  = ELPA_2STAGE_REAL_BGP
    simdTable_to_realKernels(SSE_INSTR)       = ELPA_2STAGE_REAL_SSE_BLOCK2
    simdTable_to_realKernels(AVX_INSTR)       = ELPA_2STAGE_REAL_AVX_BLOCK2
    simdTable_to_realKernels(AVX2_INSTR)      = ELPA_2STAGE_REAL_AVX2_BLOCK2
    simdTable_to_realKernels(AVX512_INSTR)    = ELPA_2STAGE_REAL_AVX512_BLOCK2
    simdTable_to_realKernels(NVIDIA_INSTR)    = ELPA_2STAGE_REAL_NVIDIA_GPU
    simdTable_to_realKernels(AMD_GPU_INSTR)   = ELPA_2STAGE_REAL_AMD_GPU
    simdTable_to_realKernels(INTEL_GPU_INSTR) = ELPA_2STAGE_REAL_INTEL_GPU
    simdTable_to_realKernels(SPARC_INSTR)     = ELPA_2STAGE_REAL_SPARC64_BLOCK2
    simdTable_to_realKernels(ARCH64_INSTR)    = ELPA_2STAGE_REAL_NEON_ARCH64_BLOCK2
    simdTable_to_realKernels(VSX_INSTR)       = ELPA_2STAGE_REAL_VSX_BLOCK2
    simdTable_to_realKernels(SVE128_INSTR)    = ELPA_2STAGE_REAL_SVE128_BLOCK2
    simdTable_to_realKernels(SVE256_INSTR)    = ELPA_2STAGE_REAL_SVE256_BLOCK2
    simdTable_to_realKernels(SVE512_INSTR)    = ELPA_2STAGE_REAL_SVE512_BLOCK2

    kernel = simdTable_to_realKernels(simd_set_index)

  end

  function map_complex_kernel_to_simd_instruction(kernel) result(simd_set_index)
    
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int), intent(in)  :: kernel
    integer(kind=c_int)              :: simd_set_index

    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_GENERIC)             = GENERIC_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_GENERIC_SIMPLE)      = GENERIC_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_BGP)                 = BLUEGENE_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_BGQ)                 = BLUEGENE_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_SSE_ASSEMBLY)        = SSE_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_SSE_BLOCK1)          = SSE_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_SSE_BLOCK2)          = SSE_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_AVX_BLOCK1)          = AVX_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_AVX_BLOCK2)          = AVX_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_AVX2_BLOCK1)         = AVX2_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_AVX2_BLOCK2)         = AVX2_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_AVX512_BLOCK1)       = AVX512_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_AVX512_BLOCK2)       = AVX512_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_SVE128_BLOCK1)       = SVE128_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_SVE128_BLOCK2)       = SVE128_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_SVE256_BLOCK1)       = SVE256_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_SVE256_BLOCK2)       = SVE256_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_SVE512_BLOCK1)       = SVE512_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_SVE512_BLOCK2)       = SVE512_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_NEON_ARCH64_BLOCK1)  = ARCH64_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_NEON_ARCH64_BLOCK2)  = ARCH64_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_NVIDIA_GPU)          = NVIDIA_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_AMD_GPU)             = AMD_GPU_INSTR
    complexKernels_to_simdTable(ELPA_2STAGE_COMPLEX_INTEL_GPU)           = INTEL_GPU_INSTR
    

    simd_set_index = complexKernels_to_simdTable(kernel)

  end

  function map_simd_instruction_to_complex_kernel(simd_set_index) result(kernel)
    
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int)              :: kernel
    integer(kind=c_int), intent(in)  :: simd_set_index

    simdTable_to_complexKernels(GENERIC_INSTR)   = ELPA_2STAGE_COMPLEX_GENERIC
    simdTable_to_complexKernels(BLUEGENE_INSTR)  = ELPA_2STAGE_COMPLEX_BGP
    simdTable_to_complexKernels(SSE_INSTR)       = ELPA_2STAGE_COMPLEX_SSE_BLOCK1
    simdTable_to_complexKernels(AVX_INSTR)       = ELPA_2STAGE_COMPLEX_AVX_BLOCK1
    simdTable_to_complexKernels(AVX2_INSTR)      = ELPA_2STAGE_COMPLEX_AVX2_BLOCK1
    simdTable_to_complexKernels(AVX512_INSTR)    = ELPA_2STAGE_COMPLEX_AVX512_BLOCK1
    simdTable_to_complexKernels(SVE128_INSTR)    = ELPA_2STAGE_COMPLEX_SVE128_BLOCK1
    simdTable_to_complexKernels(SVE256_INSTR)    = ELPA_2STAGE_COMPLEX_SVE256_BLOCK1
    simdTable_to_complexKernels(SVE512_INSTR)    = ELPA_2STAGE_COMPLEX_SVE512_BLOCK1
    simdTable_to_complexKernels(ARCH64_INSTR)    = ELPA_2STAGE_COMPLEX_NEON_ARCH64_BLOCK1
    simdTable_to_complexKernels(NVIDIA_INSTR)    = ELPA_2STAGE_COMPLEX_NVIDIA_GPU
    simdTable_to_complexKernels(AMD_GPU_INSTR)   = ELPA_2STAGE_COMPLEX_AMD_GPU
    simdTable_to_complexKernels(INTEL_GPU_INSTR) = ELPA_2STAGE_COMPLEX_INTEL_GPU

    kernel = simdTable_to_complexKernels(simd_set_index)

  end

end module

