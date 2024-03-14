#if 0
!    Copyright 2021, A. Marek, MPCDF
!
!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), formerly known as
!      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
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
!    This particular source code file contains additions, changes and
!    enhancements authored by Intel Corporation which is not part of
!    the ELPA consortium.
!
!    More information can be found here:
!    http://elpa.mpcdf.mpg.de/
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
#endif


#include "config-f90.h"
module elpa_gpu_setup
  !use precision
  use iso_c_binding

  type :: elpa_gpu_setup_t
    integer(kind=c_int)            :: use_gpu_vendor
    logical                        :: gpuIsAssigned

    ! per task information should be stored elsewhere
    integer(kind=C_intptr_T), allocatable :: gpublasHandleArray(:)
    integer(kind=C_intptr_T), allocatable :: gpusolverHandleArray(:)
    integer(kind=c_int), allocatable      :: gpuDeviceArray(:)
    integer(kind=c_intptr_t)              :: my_stream

    integer(kind=C_intptr_T), allocatable :: cublasHandleArray(:)
    integer(kind=C_intptr_T), allocatable :: cusolverHandleArray(:)
    integer(kind=c_int), allocatable      :: cudaDeviceArray(:)

    integer(kind=C_intptr_T), allocatable :: rocblasHandleArray(:)
    integer(kind=C_intptr_T), allocatable :: rocsolverHandleArray(:)
    integer(kind=c_int), allocatable      :: hipDeviceArray(:)

    integer(kind=C_intptr_T), allocatable :: syclHandleArray(:)
    integer(kind=C_intptr_T), allocatable :: syclsolverHandleArray(:)
    integer(kind=c_int), allocatable      :: syclDeviceArray(:)

    integer(kind=C_intptr_T), allocatable :: openmpOffloadHandleArray(:)
    integer(kind=C_intptr_T), allocatable :: openmpOffloadsolverHandleArray(:)
    integer(kind=c_int), allocatable      :: openmpOffloadDeviceArray(:)

    logical                               :: gpuAlreadySet
#ifdef WITH_SYCL_GPU_VERSION
    logical                               :: syclCPU
#endif
    integer(kind=c_int)        :: gpuDevAttrMaxThreadsPerBlock  = 0
    integer(kind=c_int)        :: gpuDevAttrMaxBlockDimX        = 1
    integer(kind=c_int)        :: gpuDevAttrMaxBlockDimY        = 2
    integer(kind=c_int)        :: gpuDevAttrMaxBlockDimZ        = 3
    integer(kind=c_int)        :: gpuDevAttrMaxGridDimX         = 4
    integer(kind=c_int)        :: gpuDevAttrMaxGridDimY         = 5
    integer(kind=c_int)        :: gpuDevAttrMaxGridDimZ         = 6
    integer(kind=c_int)        :: gpuDevAttrWarpSize            = 7
    integer(kind=c_int)        :: gpuDevAttrMultiProcessorCount = 8


    integer(kind=c_int)                   :: gpublasVersion
    integer(kind=c_int)                   :: rocblasVersion
    integer(kind=c_int)                   :: cublasVersion

    integer(kind=c_int)                   :: gpusPerNode

    integer(kind=c_int)                   :: nvidiaSMcount
    integer(kind=c_int)                   :: amdSMcount
    integer(kind=c_int)                   :: gpuSMcount

    integer(kind=c_int)                   :: nvidiaMaxThreadsPerBlock
    integer(kind=c_int)                   :: amdMaxThreadsPerBlock
    integer(kind=c_int)                   :: gpuMaxThreadsPerBlock

    integer(kind=c_int)                   :: nvidiaDevMaxBlockDimX
    integer(kind=c_int)                   :: amdDevMaxBlockDimX
    integer(kind=c_int)                   :: gpuDevMaxBlockDimX

    integer(kind=c_int)                   :: nvidiaDevMaxBlockDimY
    integer(kind=c_int)                   :: amdDevMaxBlockDimY
    integer(kind=c_int)                   :: gpuDevMaxBlockDimY

    integer(kind=c_int)                   :: nvidiaDevMaxBlockDimZ
    integer(kind=c_int)                   :: amdDevMaxBlockDimZ
    integer(kind=c_int)                   :: gpuDevMaxBlockDimZ

    integer(kind=c_int)                   :: nvidiaDevMaxGridDimX
    integer(kind=c_int)                   :: amdDevMaxGridDimX
    integer(kind=c_int)                   :: gpuDevMaxGridDimX

    integer(kind=c_int)                   :: nvidiaDevMaxGridDimY
    integer(kind=c_int)                   :: amdDevMaxGridDimY
    integer(kind=c_int)                   :: gpuDevMaxGridDimY

    integer(kind=c_int)                   :: nvidiaDevMaxGridDimZ
    integer(kind=c_int)                   :: amdDevMaxGridDimZ
    integer(kind=c_int)                   :: gpuDevMaxGridDimZ

    integer(kind=c_int)                   :: nvidiaDevWarpSize
    integer(kind=c_int)                   :: amdDevWarpSize
    integer(kind=c_int)                   :: gpuDevWarpSize

    integer(kind=c_intptr_t)              :: ccl_comm_rows, ccl_comm_cols, ccl_comm_all
  end type

end module

