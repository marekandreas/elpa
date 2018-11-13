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

#include "../../general/sanity.F90"
      use elpa
      use precision
      implicit none

      integer(kind=ik)                 :: na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
#if REALCASE == 1
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
      logical, intent(in)              :: wantDebug
      logical                          :: success
      integer(kind=ik)                 :: successInternal, error

      class(elpa_t), pointer           :: e

      !call timer%start("elpa_cholesky_&
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
         stop 1
      endif

      !! the elpa object needs nev to be set (in case the EVP-solver is
      !! called later. Thus it is set by user, do nothing, otherwise,
      !! set it to na as default
      !if (e%is_set("nev")) .ne. 1) then
      !  call e%set("nev", na)
      !endif

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
      call e%cholesky(a(1:lda,1:matrixCols), successInternal)

      if (successInternal .ne. ELPA_OK) then
        print *, "Cannot run cholesky"
        success = .false.
        return
      else
        success =.true.
      endif
      call elpa_deallocate(e)

      call elpa_uninit()

      !call timer%stop("elpa_cholesky_&
      !&MATH_DATATYPE&
      !&_&
      !&PRECISION&
      !&_legacy_interface")

#undef REALCASE
#undef COMPLEXCASE
#undef DOUBLE_PRECISION
#undef SINGLE_PRECISION

! vim: syntax=fortran
