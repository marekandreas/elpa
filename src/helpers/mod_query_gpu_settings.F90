!    Copyright 2023, A. Marek
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
! Author: Andreas Marek, MPCDF

#include "config-f90.h"

module mod_query_gpu_usage
   use, intrinsic :: iso_c_binding
   use precision
   implicit none
   public

   contains
     function query_gpu_usage(obj, functionname, useGPU) result(success)
       use, intrinsic :: iso_c_binding
       use elpa_gpu
       use mod_check_for_gpu
       use elpa_abstract_impl
       use elpa_utilities, only : error_unit

       implicit none

       class(elpa_abstract_impl_t), intent(inout)                         :: obj
       integer(kind=c_int)        :: gpu_old
       integer(kind=ik)           :: error
       logical                    :: success
       character(*)               :: functionname
       integer(kind=c_int)        :: gpu
       logical, intent(out)       :: useGPU

       success = .false.
       gpu = 0

       !if (trim(functionname) .eq. "ELPA_MULITPLY_AB") then
       !  call obj%get("gpu_hermitian_multiply", gpu_functionname, error)
       !else if (trim(functionname) .eq. "ELPA_CHOLESKY") then
       !  call obj%get("gpu_cholesky", gpu_functionname, error)
       !else if (trim(functionname) .eq. "ELPA_INVERT_TRM") then
       !  call obj%get("gpu_invert_trm", gpu_functionname, error)
       !else if (trim(functionname) .eq. "ELPA_INVERT_TRM") then
       !  call obj%get("gpu_invert_trm", gpu_functionname, error)
       !else
       !  write(error_unit,*) "QUERY GPU: unknown error"
       !  success = .false.
       !  return
       !endif

       ! check for legacy GPU keyword
       if (obj%is_set("gpu") == 1) then
         write(error_unit,*) "You still use the deprecated option 'gpu', consider switching to one of &
                & 'nvidia-gpu', 'amd-gpu' or 'intel-gpu' ! This deprecated keyword might be removed from &
                & the API without further notice!"

         call obj%get("gpu", gpu_old, error)
         if (error .ne. ELPA_OK) then
           write(error_unit,*) trim(functionname),": Problem getting value for keyword 'gpu'. Aborting..."
           success = .false.
           return
         endif
         if (obj%is_set("nvidia-gpu") == 0) then
           ! set 'gpu' and 'nvidia-gpu' consistent
           call obj%set("nvidia-gpu", gpu_old, error)
           if (error .ne. ELPA_OK) then
             write(error_unit,*) trim(functionname),": Problem setting value for keyword 'nvidia-gpu'. Aborting..."
             success = .false.
             return
           endif
           write(error_unit,*) trim(functionname),": Will set the mandatory keyword 'nvidia-gpu' now and &
                   & ignore the keyword 'gpu'."
         else ! obj%is_set("nvidia-gpu") == 0
           call obj%get("nvidia-gpu", gpu, error)
           if (error .ne. ELPA_OK) then
             write(error_unit, *) trim(functionname),": Problem getting option for 'nvidia-gpu'. Aborting..."
             success = .false.
             return
           endif
           if (gpu_old .ne. gpu) then
             write(error_unit,*) "Please do not set 'gpu' but set 'nvidia-gpu' instead. You cannot set &
                     & gpu = ",gpu_old," and nvidia-gpu=",gpu,". Hence, aborting..."
             success = .false.
             return
           endif
         endif ! obj%is_set("nvidia-gpu") == 0

         if (obj%is_set("amd-gpu") == 0) then
           ! amd-gpu is not set, but gpu is set
           ! this is ok in anycase
         else ! obj%is_set("amd-gpu") == 0
           call obj%get("amd-gpu", gpu, error)
           if (error .ne. ELPA_OK) then
             write(error_unit,*) trim(functionname)," Problem getting keyword for 'amd-gpu'. Aborting..."
             success = .false.
             return
           endif
           ! this is ok, if gpu == 0 and amd-gpu == 1 or
           !             if gpu == 0 and amd-gpu == 0 or
           !             if gpu == 1 and amd-gpu == 0
           if (gpu_old .eq. 1 .and. gpu .eq. 1) then
             write(error_unit,*) trim(functionname),": You cannot set gpu = 1 and amd-gpu = 1. Aborting..."
             success = .false.
             return
           endif
         endif ! amd-gpu
         if (obj%is_set("intel-gpu") == 0) then
           ! intel-gpu is not set, but gpu is set
           ! this is ok in anycase
         else ! obj%is_set("intel-gpu") == 0
                   call obj%get("intel-gpu", gpu, error)
           if (error .ne. ELPA_OK) then
             write(error_unit,*) trim(functionname)," Problem getting option for intel-gpu. Aborting..."
             success = .false.
             return
           endif
           ! this is ok, if gpu == 0 and intel-gpu == 1 or
           !             if gpu == 0 and intel-gpu == 0 or
           !             if gpu == 1 and intel-gpu == 0
           if (gpu_old .eq. 1 .and. gpu .eq. 1) then
             write(error_unit,*) trim(functionname),": You cannot set gpu = 1 and intel-gpu = 1. Aborting..."
             success = .false.
             return
           endif
         endif ! intel-gpu
       else ! gpu not set
         ! nothing to do, since obsolete option has not been used
       endif

       if (gpu_vendor() == NVIDIA_GPU) then
         call obj%get("nvidia-gpu", gpu, error)
         if (error .ne. ELPA_OK) then
           write(error_unit,*) trim(functionname),": Problem getting value for keyword 'nvidia-gpu' Aborting..."
           success = .false.
           return
         endif
       else if (gpu_vendor() == AMD_GPU) then
         call obj%get("amd-gpu", gpu, error)
         if (error .ne. ELPA_OK) then
           write(error_unit,*) trim(functionname),": Problem getting value for keyword 'amd-gpu'. Aborting..."
           success = .false.
           return
         endif

       else if (gpu_vendor() == INTEL_GPU) then
         call obj%get("intel-gpu", gpu, error)
         if (error .ne. ELPA_OK) then
           write(error_unit,*)  trim(functionname),": Problem getting value for keyword 'intel-gpu'. Aborting..."
           success = .false.
           return
         endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
       else if (gpu_vendor() == OPENMP_OFFLOAD_GPU) then
         call obj%get("intel-gpu", gpu, error)
         if (error .ne. ELPA_OK) then
           write(error_unit,*) trim(functionname),": Problem getting value for keyword 'intel-gpu' for OpenMP offloading. Aborting..."
           success = .false.
           return
         endif
#endif
#ifdef WITH_SYCL_GPU_VERSION
       else if (gpu_vendor() == SYCL_GPU) then
         call obj%get("intel-gpu", gpu, error)
         if (error .ne. ELPA_OK) then
           write(error_unit,*) trim(functionname),": Problem getting value for keyword 'intel-gpu' for SYCL. Aborting..."
           success = .false.
           return
         endif
#endif
       else
         gpu = 0
       endif
       
       if (error .ne. ELPA_OK) then
         write(error_unit,*) trim(functionname),": Problem getting option for gpu. Aborting..."
         success = .false.
         return
       endif

       if (gpu .eq. 1) then
         useGPU = (gpu == 1)
       else
         useGPU = .false.
       endif

       if (.not.(useGPU)) then
#ifdef DEVICE_POINTER
         write(error_unit,*) trim(functionname)",: You used the interface for device pointers &
            but did not specify GPU usage!. Aborting..."
         success = .false.
       return
#endif
     endif
     success = .true.
     end function
end module
