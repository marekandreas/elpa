#if 0
    subroutine elpa_transform_generalized_&
            &ELPA_IMPL_SUFFIX&
            &(self, a, b, sc_desc, error)
        implicit none
#include "general/precision_kinds.F90"
        class(elpa_impl_t)  :: self
#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=rck) :: a(self%local_nrows, *), b(self%local_nrows, *)
#else
      MATH_DATATYPE(kind=rck) :: a(self%local_nrows, self%local_ncols), b(self%local_nrows, self%local_ncols)
#endif
     integer                :: error
     integer                :: sc_desc(9)

     ! local helper array. TODO: do we want it this way? (do we need it? )
     MATH_DATATYPE(kind=rck) :: tmp(self%local_nrows, self%local_ncols)

     ! TODO: why I cannot use self%elpa ??
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
!     ! tmp <- inv(U^T) * A
!     call self%elpa_hermitian_multiply_&
!         &ELPA_IMPL_SUFFIX&
!         &('U','F', self%na, b, a, self%local_nrows, self%local_ncols, tmp, &
!                               self%local_nrows, self%local_ncols, error)
!     if(error .NE. ELPA_OK) return
#ifdef WITH_MPI
     ! A <= inv(U)^T * A
     call p&
         &BLAS_CHAR&
         &trmm("L", "U", BLAS_TRANS_OR_CONJ, "N", self%na, self%na, &
               ONE, b, 1, 1, sc_desc,  a, 1, 1, sc_desc)
     ! A <= inv(U)^T * A * inv(U)
     call p&
         &BLAS_CHAR&
         &trmm("R", "U", "N", "N", self%na, self%na, &
               ONE, b, 1, 1, sc_desc, a, 1, 1, sc_desc)
#else
     call BLAS_CHAR&
         &trmm("L", "U", BLAS_TRANS_OR_CONJ, "N", self%na, self%na, &
               ONE, b, self%na, a, self%na)
     call BLAS_CHAR&
         &trmm("R", "U", "N", "N", self%na, self%na, &
               ONE, b, self%na, a, self%na)
#endif

    end subroutine


    subroutine elpa_transform_back_generalized_&
            &ELPA_IMPL_SUFFIX&
            &(self, b, q, sc_desc, error)
        implicit none
#include "general/precision_kinds.F90"
        class(elpa_impl_t)  :: self
#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=rck) :: b(self%local_nrows, *), q(self%local_nrows, *)
#else
      MATH_DATATYPE(kind=rck) :: b(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
     integer                :: error
     integer                :: sc_desc(9)

     ! local helper array. TODO: do we want it this way? (do we need it? )
     MATH_DATATYPE(kind=rck) :: tmp(self%local_nrows, self%local_ncols)

     !todo: part of eigenvectors only
#ifdef WITH_MPI
     ! Q <= inv(U) * Q
     call p&
         &BLAS_CHAR&
         &trmm("L", "U", "N", "N", self%na, self%na, &
               ONE, b, 1, 1, sc_desc,  q, 1, 1, sc_desc)
#else
     call BLAS_CHAR&
         &trmm("L", "U", "N", "N", self%na, self%na, &
               ONE, b, self%na, q, self%na)
#endif

    end subroutine
#endif

