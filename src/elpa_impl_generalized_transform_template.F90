! using elpa internal Hermitian multiply is faster then scalapack multiply, but we need an extra
! temporary matrix.
! using cannon algorithm should be the fastest. After this is verified, the other options should be removed
! however, we need the extra temporary matrix as well.

   subroutine elpa_transform_generalized_&
            &ELPA_IMPL_SUFFIX&
            &(self, a, b, is_already_decomposed, error)
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
     integer(kind=ik)       :: my_p, my_prow, my_pcol, np_rows, np_cols, mpierr, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
     integer(kind=ik)       :: BuffLevelInt, use_cannon

     MATH_DATATYPE(kind=rck) :: tmp(self%local_nrows, self%local_ncols)

     call self%get("mpi_comm_rows",mpi_comm_rows,error)
     call self%get("mpi_comm_cols",mpi_comm_cols,error)
     call self%get("mpi_comm_parent", mpi_comm_all,error)

     call mpi_comm_rank(mpi_comm_all,my_p,mpierr)
     call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
     call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
     call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
     call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

     call self%timer_start("transform_generalized()")
     call self%get("cannon_for_generalized",use_cannon,error)

#if !defined(WITH_MPI)
     if(my_p == 0) then
       write(*,*) "Cannons algorithm can be used with MPI"
       write(*,*) "Switching to elpa Hermitian and scalapack"
     end if
     use_cannon = 0
#endif

     if (mod(np_cols, np_rows) /= 0) then
       if(my_p == 0) then
         write(*,*) "To use Cannons algorithm, np_cols must be a multiple of np_rows."
         write(*,*) "Switching to elpa Hermitian and scalapack"
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
       !TODO set the value properly
       !TODO tunable parameter?
       BuffLevelInt = 1

       call self%timer_start("cannons_reduction")
       ! BEWARE! even though tmp is output from the routine, it has to be zero on input!
       tmp = 0.0_rck
       call cannons_reduction_&
         &ELPA_IMPL_SUFFIX&
         &(a, b, self%local_nrows, self%local_ncols, sc_desc, tmp, BuffLevelInt, mpi_comm_rows, mpi_comm_cols)
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
           &trmm("R", "U", "N", "N", self%na, self%na, &
                 ONE, b, 1, 1, sc_desc, a, 1, 1, sc_desc)
#else
       call BLAS_CHAR&
           &trmm("R", "U", "N", "N", self%na, self%na, &
                 ONE, b, self%na, a, self%na)
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
     integer(kind=ik)       :: my_p, my_prow, my_pcol, np_rows, np_cols, mpierr, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
     integer                :: error
     integer                :: sc_desc(SC_DESC_LEN)
     integer                :: sc_desc_ev(SC_DESC_LEN)
     integer(kind=ik)       :: use_cannon

     MATH_DATATYPE(kind=rck) :: tmp(self%local_nrows, self%local_ncols)

     call self%get("mpi_comm_rows",mpi_comm_rows,error)
     call self%get("mpi_comm_cols",mpi_comm_cols,error)
     call self%get("mpi_comm_parent", mpi_comm_all,error)

     call mpi_comm_rank(mpi_comm_all,my_p,mpierr)
     call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
     call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
     call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
     call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

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
       call cannons_triang_rectangular_&
         &ELPA_IMPL_SUFFIX&
         &(b, q, self%local_nrows, self%local_ncols, sc_desc, sc_desc_ev, tmp, mpi_comm_rows, mpi_comm_cols);
       call self%timer_stop("cannons_triang_rectangular")

       q(1:self%local_nrows, 1:self%local_ncols) = tmp(1:self%local_nrows, 1:self%local_ncols)
     else
       call self%timer_start("scalapack multiply inv(U) * Q")
#ifdef WITH_MPI
       ! Q <- inv(U) * Q
       call p&
           &BLAS_CHAR&
           &trmm("L", "U", "N", "N", self%na, self%nev, &
                 ONE, b, 1, 1, sc_desc,  q, 1, 1, sc_desc)
#else
       call BLAS_CHAR&
           &trmm("L", "U", "N", "N", self%na, self%nev, &
                 ONE, b, self%na, q, self%na)
#endif
       call self%timer_stop("scalapack multiply inv(U) * Q")
     endif
     call self%timer_stop("transform_back_generalized()")

    end subroutine

