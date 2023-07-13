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
       
       implicit none

       class(elpa_abstract_impl_t), intent(inout)                         :: obj
       integer(kind=c_int)        :: gpu
       integer(kind=ik)           :: error
       logical                    :: success
       character(*)               :: functionname
       integer(kind=c_int)        :: gpu_functionname
       logical, intent(out)       :: useGPU

       success = .false.
       gpu_functionname = 0

       ! GPU settings
       if (gpu_vendor() == NVIDIA_GPU) then
         call obj%get("gpu", gpu, error)
         if (error .ne. ELPA_OK) then
           print *, trim(functionname),": Problem getting option for GPU. Aborting..."
           success = .false.
           return
         endif
         if (gpu .eq. 1) then
           print *,"You still use the deprecated option 'gpu', consider switching to 'nvidia-gpu'. Will set the new &
                   & keyword 'nvidia-gpu'"
           call obj%set("nvidia-gpu", gpu, error)
           if (error .ne. ELPA_OK) then
             print *,trim(functionname),": Problem setting option for NVIDIA GPU. Aborting..."
             success = .false.
             return
           endif
         endif
         call obj%get("nvidia-gpu", gpu, error)
         if (error .ne. ELPA_OK) then
           print *,trim(functionname),": Problem getting option for NVIDIA GPU. Aborting..."
           success = .false.
           return
         endif

       else if (gpu_vendor() == AMD_GPU) then
         call obj%get("amd-gpu", gpu, error)
         if (error .ne. ELPA_OK) then
           print *,trim(functionname),": Problem getting option for AMD GPU. Aborting..."
           success = .false.
           return
         endif

       else if (gpu_vendor() == SYCL_GPU) then
         call obj%get("intel-gpu",gpu,error)
         if (error .ne. ELPA_OK) then
           print *, trim(functionname),": Problem getting option for SYCL GPU. Aborting..."
           success = .false.
           return
         endif
       else
         gpu = 0
       endif
       
       if (trim(functionname) .eq. "ELPA_MULITPLY_AB") then
         call obj%get("gpu_hermitian_multiply",gpu_functionname, error)
       else if (trim(functionname) .eq. "ELPA_CHOLESKY") then
         call obj%get("gpu_cholesky", gpu_functionname, error)
       else if (trim(functionname) .eq. "ELPA_INVERT_TRM") then
         call obj%get("gpu_invert_trm", gpu_functionname, error)
       else
         print *,"QUERY GPU: unknown error"
         success = .false.
         return
       endif

       if (error .ne. ELPA_OK) then
         print *,trim(functionname),": Problem getting option for gpu. Aborting..."
         success = .false.
         return
       endif

       if (gpu_functionname .eq. 1) then
         useGPU = (gpu == 1)
       else
         useGPU = .false.
       endif

       if (.not.(useGPU)) then
#ifdef DEVICE_POINTER
         print *,trim(functionname)",: You used the interface for device pointers &
            but did not specify GPU usage!. Aborting..."
         success = .false.
       return
#endif
     endif
     success = .true.
     end function
end module
