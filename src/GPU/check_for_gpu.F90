!    Copyright 2014-2023, A. Marek
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
!
! This file was written by A. Marek, MPCDF

#include "config-f90.h"

module mod_check_for_gpu
  contains
    ! TODO: proper cleanup of handles and hanldeArrays

    ! check_for_gpu could be called at several places during a run of ELPA
    ! for example in cholesky, invert_trm, multiply and of course the solvers
    ! Thus the following logic is implemented
    ! if use_gpu_id is set -> do according to the user settings
    ! if NOT the first call to check_for_gpu will set the MPI GPU relation and then
    ! _SET_ use_gpu_id such that subsequent calls abide this setting
    function check_for_gpu(obj, myid, numberOfDevices, wantDebug) result(gpuAvailable)
      use elpa_gpu, only : gpublasDefaultPointerMode, gpu_getdevicecount, gpublas_get_version
      use cuda_functions
      use hip_functions
      use openmp_offload_functions
      use sycl_functions
      use precision
      use elpa_mpi
      use elpa_omp

#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
      use elpa_ccl_gpu
#endif
      use elpa_abstract_impl
      use ELPA_utilities, only : error_unit
      implicit none

      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)            :: myid
      logical, optional, intent(in)              :: wantDebug
      logical                                    :: success, wantDebugMessage
      integer(kind=ik), intent(out)              :: numberOfDevices
      integer(kind=ik)                           :: deviceNumber, mpierr, maxNumberOfDevices
      logical                                    :: gpuAvailable
      integer(kind=ik)                           :: error, mpi_comm_all, use_gpu_id, min_use_gpu_id
      !logical, save                              :: alreadySET=.false.
      integer(kind=ik)                           :: maxThreads, thread
      integer(kind=c_int)                        :: cublas_version
      integer(kind=c_int)                        :: syclShowOnlyIntelGpus
      integer(kind=ik)                           :: syclShowAllDevices
      integer(kind=c_intptr_t)                   :: handle_tmp
      !integer(kind=c_intptr_t)                   :: stream
      !logical                                    :: gpuIsInitialized=.false.
      !character(len=1024)           :: envname
      character(len=8)                           :: fmt
      character(len=12)                          :: gpu_string
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
      TYPE(ncclUniqueId)                         :: ncclId
      integer(kind=c_int)                        :: nprocs
      integer(kind=c_intptr_t)                   :: ccl_comm_all, ccl_comm_rows, ccl_comm_cols
      integer(kind=ik)                           :: myid_rows, myid_cols, mpi_comm_rows, mpi_comm_cols, nprows, npcols
#endif
      integer(kind=ik)                           :: attribute, value
#define OBJECT obj
#define ADDITIONAL_OBJECT_CODE
#include "./check_for_gpu_template.F90"
#undef OBJECT
#undef ADDITIONAL_OBJECT_CODE
    end function
end module
