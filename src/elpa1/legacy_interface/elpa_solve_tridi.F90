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
!
! Author: A. Marek, MPCDF



#include "../../general/sanity.F90"

      use precision
      use elpa1_auxiliary_impl, only : elpa_solve_tridi_&
      &PRECISION&
      &_impl
      use elpa
      use elpa_abstract_impl
#ifdef WITH_OPENMP
      use omp_lib
#endif
      implicit none
      integer(kind=ik)            :: na, nev, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
      real(kind=REAL_DATATYPE)    :: d(na), e(na)
#ifdef USE_ASSUMED_SIZE
      real(kind=REAL_DATATYPE)    :: q(ldq,*)
#else
      real(kind=REAL_DATATYPE)    :: q(ldq,matrixCols)
#endif

      logical, intent(in)         :: wantDebug
      logical                     :: success ! the return value
      integer                     :: error
      class(elpa_t), pointer      :: obj
#ifdef WITH_OPENMP
      integer                     :: nrThreads
#endif

      !call timer%start("elpa_solve_tridi_&
      !&PRECISION&
      !&_legacy_interface")

      success = .false.

      if (elpa_init(CURRENT_API_VERSION) /= ELPA_OK) then
        print *, "ELPA API version not supported"
        success = .false.
      endif

      obj => elpa_allocate()

      call obj%set("na", na, error)
      if (error .ne. ELPA_OK) then
         print *,"Problem setting option. Aborting..."
         stop
      endif
      call obj%set("nev", nev, error)
      if (error .ne. ELPA_OK) then
         print *,"Problem setting option. Aborting..."
         stop
      endif
      call obj%set("local_nrows", ldq, error)
      if (error .ne. ELPA_OK) then
         print *,"Problem setting option. Aborting..."
         stop
      endif
      call obj%set("local_ncols", matrixCols, error)
      if (error .ne. ELPA_OK) then
         print *,"Problem setting option. Aborting..."
         stop
      endif
      call obj%set("nblk", nblk, error)
      if (error .ne. ELPA_OK) then
         print *,"Problem setting option. Aborting..."
         stop
      endif

      call obj%set("mpi_comm_rows", mpi_comm_rows, error)
      if (error .ne. ELPA_OK) then
         print *,"Problem setting option. Aborting..."
         stop
      endif
      call obj%set("mpi_comm_cols", mpi_comm_cols, error)
      if (error .ne. ELPA_OK) then
         print *,"Problem setting option. Aborting..."
         stop
      endif

      if (obj%setup() .ne. ELPA_OK) then
        print *, "Cannot setup ELPA instance"
        success = .false.
        return
      endif

#ifdef WITH_OPENMP
      nrThreads = omp_get_max_threads()
      call obj%set("omp_threads", nrThreads, error)
#else
      call obj%set("omp_threads", 1, error)
#endif


      if (wantDebug) then
        call obj%set("debug",1, error)
        if (error .ne. ELPA_OK) then
           print *,"Problem setting option. Aborting..."
           stop
        endif
      endif

      call obj%solve_tridiagonal(d, e, q(1:ldq,1:matrixCols), error)
      if (error /= ELPA_OK) then
        print *, "Cannot run solve_tridi"
        success = .false.
        return
      else
        success = .true.
      endif

      call elpa_deallocate(obj)
      call elpa_uninit()

     !call timer%stop("elpa_solve_tridi_&
     !&PRECISION&
     !&_legacy_interface")

#undef REALCASE
#undef COMPLEXCASE
#undef DOUBLE_PRECISION
#undef SINGLE_PRECISION

! vim: syntax=fortran
