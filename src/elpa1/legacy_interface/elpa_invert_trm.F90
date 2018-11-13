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
!
!
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".

#include "../../general/sanity.F90"

       use precision
       use elpa
!       use elpa1_compute
!       use elpa_utilities
       use elpa_mpi
       implicit none

       integer(kind=ik)                 :: na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
#if  REALCASE ==  1
#ifdef USE_ASSUMED_SIZE
       real(kind=REAL_DATATYPE)         :: a(lda,*)
#else
       real(kind=REAL_DATATYPE)         :: a(lda,matrixCols)
#endif
#endif

#if COMPLEXCASE == 1
#ifdef USE_ASSUMED_SIZE
       complex(kind=COMPLEX_DATATYPE)   :: a(lda,*)
#else
       complex(kind=COMPLEX_DATATYPE)   :: a(lda,matrixCols)
#endif
#endif

!      integer(kind=ik)                 :: my_prow, my_pcol, np_rows, np_cols, mpierr
!      integer(kind=ik)                 :: l_cols, l_rows, l_col1, l_row1, l_colx, l_rowx
!      integer(kind=ik)                 :: n, nc, i, info, ns, nb
!#if REALCASE ==  1
!       real(kind=REAL_DATATYPE), allocatable   :: tmp1(:), tmp2(:,:), tmat1(:,:), tmat2(:,:)
!#endif
!#if COMPLEXCASE == 1
!       complex(kind=COMPLEX_DATATYPE), allocatable    :: tmp1(:), tmp2(:,:), tmat1(:,:), tmat2(:,:)
!#endif
       logical, intent(in)          :: wantDebug
       logical                      :: success

       integer(kind=iK)            :: successInternal, error
       class(elpa_t), pointer      :: e

       !call timer%start("elpa_invert_trm_&
       !&MATH_DATATYPE&
       !&_&
       !&PRECISION&
       !&_legacy_interface")

       success = .true.

       if (elpa_init(CURRENT_API_VERSION) /= ELPA_OK) then
         print *, "ELPA API version not supported"
         success = .false.
         return
       endif

       e => elpa_allocate()

       call e%set("na", na, error)
       if (error .ne. ELPA_OK) then
          print *,"Problem setting option. Aborting..."
          stop
       endif
       call e%set("local_nrows", lda, error)
       if (error .ne. ELPA_OK) then
          print *,"Problem setting option. Aborting..."
          stop
       endif
       call e%set("local_ncols", matrixCols, error)
       if (error .ne. ELPA_OK) then
          print *,"Problem setting option. Aborting..."
          stop
       endif
       call e%set("nblk", nblk, error)
       if (error .ne. ELPA_OK) then
          print *,"Problem setting option. Aborting..."
          stop
       endif

       call e%set("mpi_comm_rows", mpi_comm_rows, error)
       if (error .ne. ELPA_OK) then
          print *,"Problem setting option. Aborting..."
          stop
       endif
       call e%set("mpi_comm_cols", mpi_comm_cols, error)
       if (error .ne. ELPA_OK) then
          print *,"Problem setting option. Aborting..."
          stop
       endif
       call e%set("legacy_api", 1, error)
       if (error .ne. ELPA_OK) then
          print *,"Problem setting option. Aborting..."
          stop
       endif


       if (e%setup() .ne. ELPA_OK) then
         print *, "Cannot setup ELPA instance"
         success = .false.
         return
       endif

       if (wantDebug) then
         call e%set("debug",1, error)
         if (error .ne. ELPA_OK) then
            print *,"Problem setting option. Aborting..."
            stop
         endif
       endif

       call e%invert_triangular(a(1:lda,1:matrixCols), successInternal)

       if (successInternal .ne. ELPA_OK) then
         print *, "Cannot run invert_trm"
         success = .false.
         return
       else
         success =.true.
       endif
       call elpa_deallocate(e)

       call elpa_uninit()

       !call timer%stop("elpa_invert_trm_&
       !&MATH_DATATYPE&
       !&_&
       !&PRECISION&
       !&_legacy_interface")

#undef REALCASE
#undef COMPLEXCASE
#undef DOUBLE_PRECISION
#undef SINGLE_PRECISION
