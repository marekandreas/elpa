!
!    Copyright 2017, L. Hüdepohl and A. Marek, MPCDF
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
!

! using elpa internal Hermitian multiply is faster then scalapack multiply, but we need an extra
! temporary matrix.
! using cannon algorithm should be the fastest. After this is verified, the other options should be removed
! however, we need the extra temporary matrix as well.

   subroutine elpa_transform_generalized_&
            &ELPA_IMPL_SUFFIX&
            &(self, a, b, is_already_decomposed, error)
     use precision
     implicit none
#include "general/precision_kinds.F90"
     class(elpa_impl_t)  :: self
#ifdef USE_ASSUMED_SIZE
     MATH_DATATYPE(kind=rck) :: a(self%local_nrows, *), b(self%local_nrows, *)
#else
     MATH_DATATYPE(kind=rck) :: a(self%local_nrows, self%local_ncols), b(self%local_nrows, self%local_ncols)
#endif
     integer                :: error
     logical                :: is_already_decomposed
     integer                :: sc_desc(SC_DESC_LEN)
     integer(kind=ik)       :: my_p, my_prow, my_pcol, np_rows, np_cols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
     integer(kind=MPI_KIND) :: my_pMPI, my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI
     integer(kind=ik)       :: BuffLevelInt, use_cannon
     integer(kind=MPI_KIND) :: mpierr
     logical, save          :: firstCall = .true.

     MATH_DATATYPE(kind=rck) :: tmp(self%local_nrows, self%local_ncols)

     call self%get("mpi_comm_rows",mpi_comm_rows,error)
     call self%get("mpi_comm_cols",mpi_comm_cols,error)
     call self%get("mpi_comm_parent", mpi_comm_all,error)

     call mpi_comm_rank(int(mpi_comm_all,kind=MPI_KIND), my_pMPI, mpierr)
     call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND),my_prowMPI, mpierr)
     call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND),np_rowsMPI, mpierr)
     call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND),my_pcolMPI, mpierr)
     call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND),np_colsMPI, mpierr)

     my_p = int(my_pMPI, kind=c_int)
     my_prow = int(my_prowMPI, kind=c_int)
     np_rows = int(np_rowsMPI, kind=c_int)
     my_pcol = int(my_pcolMPI, kind=c_int)
     np_cols = int(np_colsMPI, kind=c_int)

     call self%timer_start("transform_generalized()")
     call self%get("cannon_for_generalized",use_cannon,error)

#if !defined(WITH_MPI)
     if ((my_p == 0) .and. firstCall) then
       write(*,*) "Cannons algorithm can only be used with MPI"
       write(*,*) "Switching to elpa Hermitian and scalapack"
       firstCall = .false.
     end if
     use_cannon = 0
#endif

     if (mod(np_cols, np_rows) /= 0) then
       if ((my_p == 0) .and. firstCall) then
         write(*,*) "To use Cannons algorithm, np_cols must be a multiple of np_rows."
         write(*,*) "Switching to elpa Hermitian and scalapack"
         firstCall = .false.
       end if
       use_cannon = 0
     endif

     error = self%construct_scalapack_descriptor(sc_desc, .false.)
     if(error .NE. ELPA_OK) return

     if (.not. is_already_decomposed) then
       ! B = U^T*U, B<-U
       call self%elpa_cholesky_&
           &ELPA_IMPL_SUFFIX&
           &(b, error)
       if(error .NE. ELPA_OK) return
       ! B <- inv(U)
       call self%elpa_invert_trm_&
           &ELPA_IMPL_SUFFIX&
           &(b, error)
       if(error .NE. ELPA_OK) return
     end if

     if(use_cannon == 1) then
       call self%get("cannon_buffer_size",BuffLevelInt,error)
       call self%timer_start("cannons_reduction")
       ! BEWARE! even though tmp is output from the routine, it has to be zero on input!
       tmp = 0.0_rck
#ifdef WITH_MPI
       call cannons_reduction_&
         &ELPA_IMPL_SUFFIX&
         &(a, b, self%local_nrows, self%local_ncols, &
           int(sc_desc,kind=BLAS_KIND), tmp, int(BuffLevelInt,kind=MPI_KIND),                    &
           int(mpi_comm_rows,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND))
#endif
       call self%timer_stop("cannons_reduction")

       a(1:self%local_nrows, 1:self%local_ncols) = tmp(1:self%local_nrows, 1:self%local_ncols)

     else  ! do not use cannon algorithm, use elpa hermitian multiply and scalapack instead
       ! tmp <- inv(U^T) * A (we have to use temporary variable)
       call self%elpa_hermitian_multiply_&
           &ELPA_IMPL_SUFFIX&
           &('U','F', self%na, b, a, self%local_nrows, self%local_ncols, tmp, &
                                 self%local_nrows, self%local_ncols, error)
       if(error .NE. ELPA_OK) return

       ! A <- inv(U)^T * A
       a(1:self%local_nrows, 1:self%local_ncols) = tmp(1:self%local_nrows, 1:self%local_ncols)

       ! A <- inv(U)^T * A * inv(U)
       ! For this multiplication we do not have internal function in ELPA,
       ! so we have to call scalapack
       call self%timer_start("scalapack multiply A * inv(U)")
#ifdef WITH_MPI
       call p&
           &BLAS_CHAR&
           &trmm("R", "U", "N", "N", int(self%na,kind=BLAS_KIND), int(self%na,kind=BLAS_KIND), &
                 ONE, b, 1_BLAS_KIND, 1_BLAS_KIND, int(sc_desc,kind=BLAS_KIND), &
                 a, 1_BLAS_KIND, 1_BLAS_KIND, int(sc_desc,kind=BLAS_KIND))
#else
       call BLAS_CHAR&
           &trmm("R", "U", "N", "N", int(self%na,kind=BLAS_KIND), int(self%na,kind=BLAS_KIND), &
                 ONE, b, int(self%na,kind=BLAS_KIND), a, int(self%na,kind=BLAS_KIND))
#endif
       call self%timer_stop("scalapack multiply A * inv(U)")
     endif ! use_cannon

     !write(*, *) my_prow, my_pcol, "A(2,3)", a(2,3)

     call self%timer_stop("transform_generalized()")
    end subroutine


    subroutine elpa_transform_back_generalized_&
            &ELPA_IMPL_SUFFIX&
            &(self, b, q, error)
        implicit none
#include "general/precision_kinds.F90"
        class(elpa_impl_t)  :: self
#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=rck) :: b(self%local_nrows, *), q(self%local_nrows, *)
#else
      MATH_DATATYPE(kind=rck) :: b(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
     integer(kind=ik)       :: my_p, my_prow, my_pcol, np_rows, np_cols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
     integer(kind=MPI_KIND) :: mpierr, my_pMPI, my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI
     integer                :: error
     integer                :: sc_desc(SC_DESC_LEN)
     integer                :: sc_desc_ev(SC_DESC_LEN)
     integer(kind=ik)       :: use_cannon

     MATH_DATATYPE(kind=rck) :: tmp(self%local_nrows, self%local_ncols)

     call self%get("mpi_comm_rows",mpi_comm_rows,error)
     call self%get("mpi_comm_cols",mpi_comm_cols,error)
     call self%get("mpi_comm_parent", mpi_comm_all,error)

     call mpi_comm_rank(int(mpi_comm_all,kind=MPI_KIND), my_pMPI,mpierr)
     call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND),my_prowMPI,mpierr)
     call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND),np_rowsMPI,mpierr)
     call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND),my_pcolMPI,mpierr)
     call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND),np_colsMPI,mpierr)

     my_p = int(my_pMPI,kind=c_int)
     my_prow = int(my_prowMPI,kind=c_int)
     np_rows = int(np_rowsMPI,kind=c_int)
     my_pcol = int(my_pcolMPI,kind=c_int)
     np_cols = int(np_colsMPI,kind=c_int)

     call self%timer_start("transform_back_generalized()")
     call self%get("cannon_for_generalized",use_cannon,error)

#if !defined(WITH_MPI)
     use_cannon = 0
#endif

     if (mod(np_cols, np_rows) /= 0) then
       use_cannon = 0
     endif

     error = self%construct_scalapack_descriptor(sc_desc, .false.)
     error = self%construct_scalapack_descriptor(sc_desc_ev, .true.)
     if(error .NE. ELPA_OK) return

     if(use_cannon == 1) then
       call self%timer_start("cannons_triang_rectangular")
#ifdef WITH_MPI
       call cannons_triang_rectangular_&
         &ELPA_IMPL_SUFFIX&
         &(b, q, self%local_nrows, self%local_ncols, &
           int(sc_desc,kind=BLAS_KIND), int(sc_desc_ev,kind=BLAS_KIND), tmp,  &
           int(mpi_comm_rows,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND) );
#endif
       call self%timer_stop("cannons_triang_rectangular")

       q(1:self%local_nrows, 1:self%local_ncols) = tmp(1:self%local_nrows, 1:self%local_ncols)
     else
       call self%timer_start("scalapack multiply inv(U) * Q")
#ifdef WITH_MPI
       ! Q <- inv(U) * Q
       call p&
           &BLAS_CHAR&
           &trmm("L", "U", "N", "N", int(self%na,kind=BLAS_KIND), int(self%nev,kind=BLAS_KIND), &
                 ONE, b, 1_BLAS_KIND, 1_BLAS_KIND, int(sc_desc,kind=BLAS_KIND),  &
                 q, 1_BLAS_KIND, 1_BLAS_KIND, int(sc_desc,kind=BLAS_KIND))
#else
       call BLAS_CHAR&
           &trmm("L", "U", "N", "N", int(self%na,kind=BLAS_KIND), int(self%nev,kind=BLAS_KIND), &
                 ONE, b, int(self%na,kind=BLAS_KIND), q, int(self%na,kind=BLAS_KIND))
#endif
       call self%timer_stop("scalapack multiply inv(U) * Q")
     endif
     call self%timer_stop("transform_back_generalized()")

    end subroutine

