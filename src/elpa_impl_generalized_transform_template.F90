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

! using elpa internal Hermitian multiply is faster then scalapack multiply, but we need an extra
! temporary variable. Therefore both options are provided and at the moment controled by this switch
#define DO_USE_ELPA_HERMITIAN_MULTIPLY

#ifdef DO_USE_ELPA_HERMITIAN_MULTIPLY
     MATH_DATATYPE(kind=rck) :: tmp(self%local_nrows, self%local_ncols)
#endif

     call self%timer_start("transform_generalized()")

     error = self%construct_scalapack_descriptor(sc_desc)
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

#ifdef DO_USE_ELPA_HERMITIAN_MULTIPLY
     ! tmp <- inv(U^T) * A (we have to use temporary variable)
     call self%elpa_hermitian_multiply_&
         &ELPA_IMPL_SUFFIX&
         &('U','F', self%na, b, a, self%local_nrows, self%local_ncols, tmp, &
                               self%local_nrows, self%local_ncols, error)
     if(error .NE. ELPA_OK) return

     ! A <- inv(U)^T * A
     a(1:self%local_nrows, 1:self%local_ncols) = tmp(1:self%local_nrows, 1:self%local_ncols)
#else
     ! A <- inv(U)^T * A (using scalapack, we can directly update A)
     call self%timer_start("scalapack multiply inv(U)^T * A")
#ifdef WITH_MPI
     call p&
         &BLAS_CHAR&
         &trmm("L", "U", BLAS_TRANS_OR_CONJ, "N", self%na, self%na, &
               ONE, b, 1, 1, sc_desc,  a, 1, 1, sc_desc)
#else
     call BLAS_CHAR&
         &trmm("L", "U", BLAS_TRANS_OR_CONJ, "N", self%na, self%na, &
               ONE, b, self%na, a, self%na)
#endif

     call self%timer_stop("scalapack multiply inv(U)^T * A")
#endif /* DO_USE_ELPA_HERMITIAN_MULTIPLY */

     ! A <- inv(U)^T * A * inv(U)
     ! For this multiplication we do not have internal function in ELPA, 
     ! so we have to call scalapack anyway
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
     integer                :: error
     integer                :: sc_desc(SC_DESC_LEN)

     call self%timer_start("transform_back_generalized()")

     error = self%construct_scalapack_descriptor(sc_desc)
     if(error .NE. ELPA_OK) return

     !todo: part of eigenvectors only
     call self%timer_start("scalapack multiply inv(U) * Q")
#ifdef WITH_MPI
     ! Q <- inv(U) * Q
     call p&
         &BLAS_CHAR&
         &trmm("L", "U", "N", "N", self%na, self%na, &
               ONE, b, 1, 1, sc_desc,  q, 1, 1, sc_desc)
#else
     call BLAS_CHAR&
         &trmm("L", "U", "N", "N", self%na, self%na, &
               ONE, b, self%na, q, self%na)
#endif
     call self%timer_stop("scalapack multiply inv(U) * Q")

     call self%timer_stop("transform_back_generalized()")

    end subroutine

