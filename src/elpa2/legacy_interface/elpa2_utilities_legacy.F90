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
!
! ELPA2 -- 2-stage solver for ELPA
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! Author: Andreas Marek, MPCDF

#include "config-f90.h"

module elpa2_utilities
  use elpa
  use precision
  implicit none
  public

  integer(kind=ik), parameter :: number_of_real_kernels = ELPA_2STAGE_NUMBER_OF_REAL_KERNELS
  integer(kind=ik), parameter :: number_of_complex_kernels = ELPA_2STAGE_NUMBER_OF_COMPLEX_KERNELS

#ifdef WITH_KERNEL_REAL_GENERIC
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_GENERIC = ELPA_2STAGE_REAL_GENERIC
#endif
#ifdef WITH_KERNEL_REAL_GENERIC_SIMPLE
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_GENERIC_SIMPLE = ELPA_2STAGE_REAL_GENERIC_SIMPLE
#endif
#ifdef WITH_KERNEL_REAL_BGP
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_BGP = ELPA_2STAGE_REAL_BGP
#endif
#ifdef WITH_KERNEL_REAL_BGQ
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_BGQ = ELPA_2STAGE_REAL_BGQ
#endif
#ifdef WITH_KERNEL_REAL_SSE
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_SSE = ELPA_2STAGE_REAL_SSE
#endif
#ifdef WITH_KERNEL_REAL_SSE_BLOCK2
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_SSE_BLOCK2 = ELPA_2STAGE_REAL_SSE_BLOCK2
#endif
#ifdef WITH_KERNEL_REAL_SSE_BLOCK4
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_SSE_BLOCK4 = ELPA_2STAGE_REAL_SSE_BLOCK4
#endif
#ifdef WITH_KERNEL_REAL_SSE_BLOCK6
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_SSE_BLOCK6 = ELPA_2STAGE_REAL_SSE_BLOCK6
#endif
#ifdef WITH_KERNEL_REAL_AVX_BLOCK2
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_AVX_BLOCK2 = ELPA_2STAGE_REAL_AVX_BLOCK2
#endif
#ifdef WITH_KERNEL_REAL_AVX_BLOCK4
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_AVX_BLOCK4 = ELPA_2STAGE_REAL_AVX_BLOCK4
#endif
#ifdef WITH_KERNEL_REAL_AVX_BLOCK6
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_AVX_BLOCK6 = ELPA_2STAGE_REAL_AVX_BLOCK6
#endif
#ifdef WITH_KERNEL_REAL_AVX2_BLOCK2
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_AVX2_BLOCK2 = ELPA_2STAGE_REAL_AVX2_BLOCK2
#endif
#ifdef WITH_KERNEL_REAL_AVX2_BLOCK4
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_AVX2_BLOCK4 = ELPA_2STAGE_REAL_AVX2_BLOCK4
#endif
#ifdef WITH_KERNEL_REAL_AVX2_BLOCK6
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_AVX2_BLOCK6 = ELPA_2STAGE_REAL_AVX2_BLOCK6
#endif
#ifdef WITH_KERNEL_REAL_AVX512_BLOCK2
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_AVX512_BLOCK2 = ELPA_2STAGE_REAL_AVX512_BLOCK2
#endif
#ifdef WITH_KERNEL_REAL_AVX512_BLOCK4
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_AVX512_BLOCK4 = ELPA_2STAGE_REAL_AVX512_BLOCK4
#endif
#ifdef WITH_KERNEL_REAL_AVX512_BLOCK6
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_AVX512_BLOCK6 = ELPA_2STAGE_REAL_AVX512_BLOCK6
#endif
#ifdef WITH_KERNEL_REAL_GPU
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_GPU = ELPA_2STAGE_REAL_GPU
#endif

  integer(kind=ik), parameter :: DEFAULT_REAL_ELPA_KERNEL = ELPA_2STAGE_REAL_DEFAULT

#ifdef WITH_KERNEL_COMPLEX_GENERIC
  integer(kind=ik), parameter :: COMPLEX_ELPA_KERNEL_GENERIC = ELPA_2STAGE_COMPLEX_GENERIC
#endif
#ifdef WITH_KERNEL_COMPLEX_GENERIC_SIMPLE
  integer(kind=ik), parameter :: COMPLEX_ELPA_KERNEL_GENERIC_SIMPLE = ELPA_2STAGE_COMPLEX_GENERIC_SIMPLE
#endif
#ifdef WITH_KERNEL_COMPLEX_BGP
  integer(kind=ik), parameter :: COMPLEX_ELPA_KERNEL_BGP = ELPA_2STAGE_COMPLEX_BGP
#endif
#ifdef WITH_KERNEL_COMPLEX_BGQ
  integer(kind=ik), parameter :: COMPLEX_ELPA_KERNEL_BGQ = ELPA_2STAGE_COMPLEX_BGQ
#endif
#ifdef WITH_KERNEL_COMPLEX_SSE
  integer(kind=ik), parameter :: COMPLEX_ELPA_KERNEL_SSE = ELPA_2STAGE_COMPLEX_SSE
#endif
#ifdef WITH_KERNEL_COMPLEX_SSE_BLOCK1
  integer(kind=ik), parameter :: COMPLEX_ELPA_KERNEL_SSE_BLOCK1 = ELPA_2STAGE_COMPLEX_SSE_BLOCK1
#endif
#ifdef WITH_KERNEL_COMPLEX_SSE_BLOCK2
  integer(kind=ik), parameter :: COMPLEX_ELPA_KERNEL_SSE_BLOCK2 = ELPA_2STAGE_COMPLEX_SSE_BLOCK2
#endif
#ifdef WITH_KERNEL_COMPLEX_AVX_BLOCK1
  integer(kind=ik), parameter :: COMPLEX_ELPA_KERNEL_AVX_BLOCK1 = ELPA_2STAGE_COMPLEX_AVX_BLOCK1
#endif
#ifdef WITH_KERNEL_COMPLEX_AVX_BLOCK2
  integer(kind=ik), parameter :: COMPLEX_ELPA_KERNEL_AVX_BLOCK2 = ELPA_2STAGE_COMPLEX_AVX_BLOCK2
#endif
#ifdef WITH_KERNEL_COMPLEX_AVX2_BLOCK1
  integer(kind=ik), parameter :: COMPLEX_ELPA_KERNEL_AVX2_BLOCK1 = ELPA_2STAGE_COMPLEX_AVX2_BLOCK1
#endif
#ifdef WITH_KERNEL_COMPLEX_AVX2_BLOCK2
  integer(kind=ik), parameter :: COMPLEX_ELPA_KERNEL_AVX2_BLOCK2 = ELPA_2STAGE_COMPLEX_AVX2_BLOCK2
#endif
#ifdef WITH_KERNEL_COMPLEX_AVX512_BLOCK1
  integer(kind=ik), parameter :: COMPLEX_ELPA_KERNEL_AVX512_BLOCK1 = ELPA_2STAGE_COMPLEX_AVX512_BLOCK1
#endif
#ifdef WITH_KERNEL_COMPLEX_AVX512_BLOCK2
  integer(kind=ik), parameter :: COMPLEX_ELPA_KERNEL_AVX512_BLOCK2 = ELPA_2STAGE_COMPLEX_AVX512_BLOCK2
#endif
#ifdef WITH_KERNEL_COMPLEX_GPU
  integer(kind=ik), parameter :: COMPLEX_ELPA_KERNEL_GPU = ELPA_2STAGE_COMPLEX_GPU
#endif

  integer(kind=ik), parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = ELPA_2STAGE_COMPLEX_DEFAULT


  contains

   function elpa_real_kernel_name(THIS_ELPA_REAL_KERNEL) result(name)
     use elpa
     implicit none
     integer, intent(in) :: THIS_ELPA_REAL_KERNEL
     character(35)        :: name

     name = elpa_value_to_string("real_kernel", THIS_ELPA_REAL_KERNEL)
   end function


   function elpa_complex_kernel_name(THIS_ELPA_COMPLEX_KERNEL) result(name)
     use elpa
     implicit none
     integer, intent(in) :: THIS_ELPA_COMPLEX_KERNEL
     character(35)        :: name

     name = elpa_value_to_string("complex_kernel", THIS_ELPA_COMPLEX_KERNEL)
    end function

end module elpa2_utilities
