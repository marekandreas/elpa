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
 function solve_evp_&
  &MATH_DATATYPE&
  &_&
  &2stage_&
  &PRECISION &
  (na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all,   &
#if REALCASE == 1
   THIS_ELPA_KERNEL_API, useQR,   &
#endif
#if COMPLEXCASE == 1
   THIS_ELPA_KERNEL_API,          &
#endif
   useGPU) result(success)

   use iso_c_binding
   use elpa
   use elpa_mpi

   implicit none

   logical, intent(in), optional             :: useGPU
#if REALCASE == 1
   logical, intent(in), optional             :: useQR
#endif
   integer(kind=c_int), intent(in), optional :: THIS_ELPA_KERNEL_API

   integer(kind=c_int), intent(in)           :: na, nev, lda, ldq, matrixCols, mpi_comm_rows, &
                                                mpi_comm_cols, mpi_comm_all
   integer(kind=c_int), intent(in)           :: nblk

#ifdef USE_ASSUMED_SIZE
   MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(inout) :: a(lda,*), q(ldq,*)
#else
   MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(inout) :: a(lda,matrixCols), q(ldq,matrixCols)
#endif
   real(kind=C_DATATYPE_KIND), intent(inout)          :: ev(na)

   integer(kind=c_int)                       :: my_prow, my_pcol, mpierr
   logical                                   :: success

   integer(kind=c_int)                       :: successInternal
   class(elpa_t), pointer                    :: e

    call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
    call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)

    success = .true.
    if (elpa_init(CURRENT_API_VERSION) /= ELPA_OK) then
      print *,  "ELPA API version not supported"
      success = .false.
      return
    endif

    e => elpa_allocate()

    call e%set("na", na)
    call e%set("nev", nev)
    call e%set("local_nrows", lda)
    call e%set("local_ncols", matrixCols)
    call e%set("nblk", nblk)

    call e%set("mpi_comm_parent", mpi_comm_all)
    call e%set("process_row", my_prow)
    call e%set("process_col", my_pcol)

    call e%set("timings",1)

    if (e%setup() .ne. ELPA_OK) then
      print *, "Cannot setup ELPA instance"
      success = .false.
      return
    endif

    call e%set("solver", ELPA_SOLVER_2STAGE, successInternal)
    if (successInternal .ne. ELPA_OK) then
      print *, "Cannot set ELPA 1stage solver"
      success = .false.
      return
    endif

    if (present(useGPU)) then
      if (useGPU) then
        call e%set("gpu", 1, successInternal)
        if (successInternal .ne. ELPA_OK) then
          print *, "Cannot set gpu"
          success = .false.
          return
        endif
      else
        call e%set("gpu", 0, successInternal)
        if (successInternal .ne. ELPA_OK) then
          print *, "Cannot set gpu"
          success = .false.
          return
        endif
      endif
    endif

#if REALCASE == 1
    if (present(useQR)) then
      if (useQR) then
        call e%set("qr", 1, successInternal)
        if (successInternal .ne. ELPA_OK) then
          print *, "Cannot set qr"
          success = .false.
          return
        endif
      else
        call e%set("qr", 0, successInternal)
        if (successInternal .ne. ELPA_OK) then
          print *, "Cannot set qr"
          success = .false.
          return
        endif
      endif
    endif
#endif

#if REALCASE == 1
    if (present(THIS_ELPA_KERNEL_API)) then
      call e%set("real_kernel",THIS_ELPA_KERNEL_API, successInternal)
      if (successInternal .ne. ELPA_OK) then
        print *, "Cannot set ELPA2 stage real_kernel"
        success = .false.
        return
      endif
    endif
#endif

#if COMPLEXCASE == 1
    if (present(THIS_ELPA_KERNEL_API)) then
      call e%set("complex_kernel",THIS_ELPA_KERNEL_API, successInternal)
      if (successInternal .ne. ELPA_OK) then
        print *, "Cannot set ELPA2 stage complex_kernel"
        success = .false.
        return
      endif
    endif
#endif

    call e%set("print_flops", 1,successInternal)
    if (successInternal .ne. ELPA_OK) then
      print *, "Cannot set print_flops"
      success = .false.
      return
    endif
    call e%set("timings", 1)

    call e%eigenvectors(a(1:lda,1:matrixCols), ev, q(1:ldq,1:matrixCols), successInternal)
    if (successInternal .ne. ELPA_OK) then
      print *, "Cannot solve with ELPA 2stage"
      success = .false.
      return
    endif

    time_evp_fwd = e%get_time("elpa_solve_evp_&
   &MATH_DATATYPE&
   &_2stage_&
   &PRECISION&
   &","bandred")

   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      write(error_unit,*) 'Time bandred_real               :', time_evp_fwd

   time_evp_fwd =  time_evp_fwd + e%get_time("elpa_solve_evp_&
   &MATH_DATATYPE&
   &_2stage_&
   &PRECISION&
   &","tridiag")

   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      write(error_unit,*) 'Time tridiag_band_real          :',e%get_time("elpa_solve_evp_&
   &MATH_DATATYPE&
   &_2stage_&
   &PRECISION&
   &","tridiag")

   time_evp_solve = e%get_time("elpa_solve_evp_&
   &MATH_DATATYPE&
   &_2stage_&
   &PRECISION&
   &","solve")

   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
     write(error_unit,*) 'Time solve_tridi                :',time_evp_solve

   time_evp_back = e%get_time("elpa_solve_evp_&
   &MATH_DATATYPE&
   &_2stage_&
   &PRECISION&
   &","trans_ev_to_band")

   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
     write(error_unit,*) 'Time trans_ev_tridi_to_band_real:',time_evp_back

   time_evp_back = time_evp_back  + &
   e%get_time("elpa_solve_evp_&
   &MATH_DATATYPE&
   &_2stage_&
   &PRECISION&
   &","trans_ev_to_full")

   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
     write(error_unit,*) 'Time trans_ev_band_to_full_real :',e%get_time("elpa_solve_evp_&
   &MATH_DATATYPE&
   &_2stage_&
   &PRECISION&
   &","trans_ev_to_full")


    call elpa_deallocate(e)

    call elpa_uninit()

   end function

! vim: syntax=fortran
