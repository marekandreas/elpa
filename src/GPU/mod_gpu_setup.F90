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

    integer(kind=c_intptr_t)              :: ccl_comm_rows, ccl_comm_cols, ccl_comm_all
  end type

end module

