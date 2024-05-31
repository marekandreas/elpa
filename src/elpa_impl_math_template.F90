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

    !_____________________________________________________________________________________________________________________
    ! hermitian_multiply

    !> \brief  elpa_hermitian_multiply_a_h_a_d: class method to perform C : = A**T * B
    !>         where   A is a square matrix (self%na,self%na) which is optionally upper or lower triangular
    !>                 B is a (self%na,ncb) matrix
    !>                 C is a (self%na,ncb) matrix where optionally only the upper or lower
    !>                   triangle may be computed
    !>
    !> the MPI commicators and the block-cyclic distribution block size are already known to the type.
    !> Thus the class method "setup" must be called BEFORE this method is used
    !>
    !> \details
    !>
    !> \param  self                 class(elpa_t), the ELPA object
    !> \param  uplo_a               'U' if A is upper triangular
    !>                              'L' if A is lower triangular
    !>                              anything else if A is a full matrix
    !>                              Please note: This pertains to the original A (as set in the calling program)
    !>                                           whereas the transpose of A is used for calculations
    !>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
    !>                              i.e. it may contain arbitrary numbers
    !> \param uplo_c                'U' if only the upper diagonal part of C is needed
    !>                              'L' if only the upper diagonal part of C is needed
    !>                              anything else if the full matrix C is needed
    !>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
    !>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
    !> \param ncb                   Number of columns  of global matrices B and C
    !> \param a                     matrix a
    !> \param local_nrows           number of rows of local (sub) matrix a, set with class method set("local_nrows",value)
    !> \param local_ncols           number of columns of local (sub) matrix a, set with class method set("local_ncols",value)
    !> \param b                     matrix b
    !> \param nrows_b               number of rows of local (sub) matrix b
    !> \param ncols_b               number of columns of local (sub) matrix b
    !> \param c                     matrix c
    !> \param nrows_c               number of rows of local (sub) matrix c
    !> \param ncols_c               number of columns of local (sub) matrix c
    !> \param error                 optional argument, error code which can be queried with elpa_strerr
    subroutine elpa_hermitian_multiply_a_h_a_&
                   &ELPA_IMPL_SUFFIX&
                   & (self, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, &
                                          c, nrows_c, ncols_c, error)
      class(elpa_impl_t)              :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=c_int), intent(in) :: nrows_b, ncols_b, nrows_c, ncols_c, ncb
#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows,*), b(nrows_b,*), c(nrows_c,*)
#else
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows,self%local_ncols), b(nrows_b,ncols_b), c(nrows_c,ncols_c)
#endif
#ifdef USE_FORTRAN2008
      integer, optional               :: error
#else
      integer                         :: error
#endif
      logical                         :: success_l
     
      success_l = .false.
#if defined(INCLUDE_ROUTINES)
#ifdef REALCASE
      success_l = elpa_mult_at_b_a_h_a_&
#endif
#ifdef COMPLEXCASE
      success_l = elpa_mult_ah_b_a_h_a_&
#endif
              &MATH_DATATYPE&
              &_&
              &PRECISION&
              &_impl(self, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, &
                                                  c, nrows_c, ncols_c)
#endif
#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in hermitian_multiply() and you did not check for errors!"
      endif
#else
      if (success_l) then
        error = ELPA_OK
      else
        error = ELPA_ERROR
      endif
#endif
    end subroutine  


    !> \brief  elpa_hermitian_multiply_d_ptr_d: class method to perform C : = A**T * B
    !>         where   A is a square matrix (self%na,self%na) which is optionally upper or lower triangular
    !>                 B is a (self%na,ncb) matrix
    !>                 C is a (self%na,ncb) matrix where optionally only the upper or lower
    !>                   triangle may be computed
    !>
    !> the MPI commicators and the block-cyclic distribution block size are already known to the type.
    !> Thus the class method "setup" must be called BEFORE this method is used
    !>
    !> \details
    !>
    !> \param  self                 class(elpa_t), the ELPA object
    !> \param  uplo_a               'U' if A is upper triangular
    !>                              'L' if A is lower triangular
    !>                              anything else if A is a full matrix
    !>                              Please note: This pertains to the original A (as set in the calling program)
    !>                                           whereas the transpose of A is used for calculations
    !>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
    !>                              i.e. it may contain arbitrary numbers
    !> \param uplo_c                'U' if only the upper-triangle part of C is needed
    !>                              'L' if only the lower-triangle part of C is needed
    !>                              anything else if the full matrix C is needed
    !>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
    !>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
    !> \param ncb                   Number of columns  of global matrices B and C
    !> \param a                     matrix a, as device pointer of type(c_ptr)
    !> \param local_nrows           number of rows of local (sub) matrix a, set with class method set("local_nrows",value)
    !> \param local_ncols           number of columns of local (sub) matrix a, set with class method set("local_ncols",value)
    !> \param b                     matrix b, as device pointer of type(c_ptr)
    !> \param nrows_b               number of rows of local (sub) matrix b
    !> \param ncols_b               number of columns of local (sub) matrix b
    !> \param c                     matrix c, as device pointer of type(c_ptr)
    !> \param nrows_c               number of rows of local (sub) matrix c
    !> \param ncols_c               number of columns of local (sub) matrix c
    !> \param error                 optional argument, error code which can be queried with elpa_strerr
    subroutine elpa_hermitian_multiply_d_ptr_&
                   &ELPA_IMPL_SUFFIX&
                   & (self, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, &
                                          c, nrows_c, ncols_c, error)
      class(elpa_impl_t)              :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=c_int), intent(in) :: nrows_b, ncols_b, nrows_c, ncols_c, ncb
      type(c_ptr)                     :: a, b, c
#ifdef USE_FORTRAN2008
      integer, optional               :: error
#else
      integer                         :: error
#endif
      logical                         :: success_l
     
      success_l = .false.
#if defined(INCLUDE_ROUTINES)
#ifdef REALCASE
      success_l = elpa_mult_at_b_d_ptr_&
#endif
#ifdef COMPLEXCASE
      success_l = elpa_mult_ah_b_d_ptr_&
#endif
              &MATH_DATATYPE&
              &_&
              &PRECISION&
              &_impl(self, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, &
                                                  c, nrows_c, ncols_c)
#endif
#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in hermitian_multiply() and you did not check for errors!"
      endif
#else
      if (success_l) then
        error = ELPA_OK
      else
        error = ELPA_ERROR
      endif
#endif
    end subroutine  

    !c> // /src/elpa_impl_math_template.F90
    
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL    
    !c> void elpa_hermitian_multiply_a_h_a_d(elpa_t handle, char uplo_a, char uplo_c, int ncb, double *a, double *b, int nrows_b, int ncols_b, double *c, int nrows_c, int ncols_c, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> void elpa_hermitian_multiply_a_h_a_f(elpa_t handle, char uplo_a, char uplo_c, int ncb, float *a, float *b, int nrows_b, int ncols_b, float *c, int nrows_c, int ncols_c, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX    
    !c> void elpa_hermitian_multiply_a_h_a_dc(elpa_t handle, char uplo_a, char uplo_c, int ncb, double_complex *a, double_complex *b, int nrows_b, int ncols_b, double_complex *c, int nrows_c, int ncols_c, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    !c> void elpa_hermitian_multiply_a_h_a_fc(elpa_t handle, char uplo_a, char uplo_c, int ncb, float_complex *a, float_complex *b, int nrows_b, int ncols_b, float_complex *c, int nrows_c, int ncols_c, int *error);
#endif
#endif
    subroutine elpa_hermitian_multiply_a_h_a_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, uplo_a, uplo_c, ncb, a_p, b_p, nrows_b, &
                                           ncols_b, c_p, nrows_c, ncols_c, error)          &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL 
                                           bind(C, name="elpa_hermitian_multiply_a_h_a_d")
#endif
#ifdef SINGLE_PRECISION_REAL 
                                           bind(C, name="elpa_hermitian_multiply_a_h_a_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
                                           bind(C, name="elpa_hermitian_multiply_a_h_a_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX 
                                           bind(C, name="elpa_hermitian_multiply_a_h_a_fc")
#endif
#endif

      type(c_ptr), intent(in), value               :: handle, a_p, b_p, c_p
      character(1,C_CHAR), value                   :: uplo_a, uplo_c
      integer(kind=c_int), value                   :: ncb, nrows_b, ncols_b, nrows_c, ncols_c
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(in)    :: error
#else
      integer(kind=c_int), intent(in)              :: error
#endif
      MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer :: a(:, :), b(:,:), c(:,:)
!#ifdef USE_ASSUMED_SIZE
!      MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer :: b(nrows_b,*), c(nrows_c,*)
!#else
!      MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer :: b(nrows_b,ncols_b), c(nrows_c,ncols_c)
!#endif
      type(elpa_impl_t), pointer                   :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(b_p, b, [nrows_b, ncols_b])
      call c_f_pointer(c_p, c, [nrows_c, ncols_c])

      call elpa_hermitian_multiply_a_h_a_&
              &ELPA_IMPL_SUFFIX&
              & (self, uplo_a, uplo_c, ncb, a, b, nrows_b, &
                                     ncols_b, c, nrows_c, ncols_c, error)
    end subroutine

#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL    
    !c> void elpa_hermitian_multiply_d_ptr_d(elpa_t handle, char uplo_a, char uplo_c, int ncb, double *a, double *b, int nrows_b, int ncols_b, double *c, int nrows_c, int ncols_c, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> void elpa_hermitian_multiply_d_ptr_f(elpa_t handle, char uplo_a, char uplo_c, int ncb, float *a, float *b, int nrows_b, int ncols_b, float *c, int nrows_c, int ncols_c, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX    
    !c> void elpa_hermitian_multiply_d_ptr_dc(elpa_t handle, char uplo_a, char uplo_c, int ncb, double_complex *a, double_complex *b, int nrows_b, int ncols_b, double_complex *c, int nrows_c, int ncols_c, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    !c> void elpa_hermitian_multiply_d_ptr_fc(elpa_t handle, char uplo_a, char uplo_c, int ncb, float_complex *a, float_complex *b, int nrows_b, int ncols_b, float_complex *c, int nrows_c, int ncols_c, int *error);
#endif
#endif
    subroutine elpa_hermitian_multiply_d_ptr_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, uplo_a, uplo_c, ncb, a_p, b_p, nrows_b, &
                                           ncols_b, c_p, nrows_c, ncols_c, error)          &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL 
                                           bind(C, name="elpa_hermitian_multiply_d_ptr_d")
#endif
#ifdef SINGLE_PRECISION_REAL 
                                           bind(C, name="elpa_hermitian_multiply_d_ptr_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
                                           bind(C, name="elpa_hermitian_multiply_d_ptr_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX 
                                           bind(C, name="elpa_hermitian_multiply_d_ptr_fc")
#endif
#endif

      type(c_ptr), intent(in), value            :: handle, a_p, b_p, c_p
      character(1,C_CHAR), value                :: uplo_a, uplo_c
      integer(kind=c_int), value                :: ncb, nrows_b, ncols_b, nrows_c, ncols_c
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(in) :: error
#else
      integer(kind=c_int), intent(in)           :: error
#endif
      type(elpa_impl_t), pointer                :: self

      call c_f_pointer(handle, self)
      !call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_hermitian_multiply_d_ptr_&
              &ELPA_IMPL_SUFFIX&
              & (self, uplo_a, uplo_c, ncb, a_p, b_p, nrows_b, &
                                     ncols_b, c_p, nrows_c, ncols_c, error)
    end subroutine

    !_____________________________________________________________________________________________________________________
    ! multiply

!     !> \brief  elpa_multiply_a_h_a_d: class method to perform C : = op(A) * B
!     !>         where   op(A) is one of: A, A**T, A**H
!     !>                 A is a square matrix (self%na,self%na) which is optionally upper or lower triangular ! PETERDEBUG: does the matrix have to be square?
!     !>                 B is a (self%na,ncb) matrix
!     !>                 C is a (self%na,ncb) matrix where optionally only the upper or lower
!     !>                   triangle may be computed
!     !>
!     !> the MPI commicators and the block-cyclic distribution block size are already known to the type.
!     !> Thus the class method "setup" must be called BEFORE this method is used
!     !>
!     !> \details
!     !>
!     !> \param  self                 class(elpa_t), the ELPA object
!     !> \param  uplo_a               'U' if A is upper triangular
!     !>                              'L' if A is lower triangular
!     !>                              anything else if A is a full matrix
!     !>                              Please note: U/L label pertains to the original A (as set in the calling program)
!     !>                                           whereas the transpose of A is might be used for calculations
!     !>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!     !>                              i.e. it may contain arbitrary numbers
!     !> \param uplo_c                'U' if only the upper-triangle part of C is needed
!     !>                              'L' if only the lower-triangle part of C is needed
!     !>                              anything else if the full matrix C is needed
!     !>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!     !>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!     !> \param trans_a
!     !> \param ncb                   Number of columns  of global matrices B and C
!     !> \param a                     matrix a
!     !> \param local_nrows           number of rows of local (sub) matrix a, set with class method set("local_nrows",value)
!     !> \param local_ncols           number of columns of local (sub) matrix a, set with class method set("local_ncols",value)
!     !> \param b                     matrix b
!     !> \param nrows_b               number of rows of local (sub) matrix b
!     !> \param ncols_b               number of columns of local (sub) matrix b
!     !> \param c                     matrix c
!     !> \param nrows_c               number of rows of local (sub) matrix c
!     !> \param ncols_c               number of columns of local (sub) matrix c
!     !> \param error                 optional argument, error code which can be queried with elpa_strerr
!     subroutine elpa_multiply_a_h_a_&
!                    &ELPA_IMPL_SUFFIX&
!                    & (self, uplo_a, uplo_c, trans_a, ncb, a, b, nrows_b, ncols_b, &
!                                           c, nrows_c, ncols_c, error)
!       class(elpa_impl_t)              :: self
!       character*1                     :: uplo_a, uplo_c, trans_a
!       integer(kind=c_int), intent(in) :: nrows_b, ncols_b, nrows_c, ncols_c, ncb
! #ifdef USE_ASSUMED_SIZE
!       MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows,*), b(nrows_b,*), c(nrows_c,*)
! #else
!       MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows,self%local_ncols), b(nrows_b,ncols_b), c(nrows_c,ncols_c)
! #endif
! #ifdef USE_FORTRAN2008
!       integer, optional               :: error
! #else
!       integer                         :: error
! #endif
!       logical                         :: success_l
     
!       success_l = .false.
! #if defined(INCLUDE_ROUTINES)
! #ifdef REALCASE
!       success_l = elpa_mult_at_b_a_h_a_&
! #endif
! #ifdef COMPLEXCASE
!       success_l = elpa_mult_ah_b_a_h_a_&
! #endif
!               &MATH_DATATYPE&
!               &_&
!               &PRECISION&
!               &_impl(self, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, &
!                                                   c, nrows_c, ncols_c)
! #endif
! #ifdef USE_FORTRAN2008
!       if (present(error)) then
!         if (success_l) then
!           error = ELPA_OK
!         else
!           error = ELPA_ERROR
!         endif
!       else if (.not. success_l) then
!         write(error_unit,'(a)') "ELPA: Error in hermitian_multiply() and you did not check for errors!"
!       endif
! #else
!       if (success_l) then
!         error = ELPA_OK
!       else
!         error = ELPA_ERROR
!       endif
! #endif
!     end subroutine  


!     !> \brief  elpa_hermitian_multiply_d_ptr_d: class method to perform C : = A**T * B
!     !>         where   A is a square matrix (self%na,self%na) which is optionally upper or lower triangular
!     !>                 B is a (self%na,ncb) matrix
!     !>                 C is a (self%na,ncb) matrix where optionally only the upper or lower
!     !>                   triangle may be computed
!     !>
!     !> the MPI commicators and the block-cyclic distribution block size are already known to the type.
!     !> Thus the class method "setup" must be called BEFORE this method is used
!     !>
!     !> \details
!     !>
!     !> \param  self                 class(elpa_t), the ELPA object
!     !> \param  uplo_a               'U' if A is upper triangular
!     !>                              'L' if A is lower triangular
!     !>                              anything else if A is a full matrix
!     !>                              Please note: This pertains to the original A (as set in the calling program)
!     !>                                           whereas the transpose of A is used for calculations
!     !>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!     !>                              i.e. it may contain arbitrary numbers
!     !> \param uplo_c                'U' if only the upper diagonal part of C is needed
!     !>                              'L' if only the upper diagonal part of C is needed
!     !>                              anything else if the full matrix C is needed
!     !>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!     !>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!     !> \param ncb                   Number of columns  of global matrices B and C
!     !> \param a                     matrix a, as device pointer of type(c_ptr)
!     !> \param local_nrows           number of rows of local (sub) matrix a, set with class method set("local_nrows",value)
!     !> \param local_ncols           number of columns of local (sub) matrix a, set with class method set("local_ncols",value)
!     !> \param b                     matrix b, as device pointer of type(c_ptr)
!     !> \param nrows_b               number of rows of local (sub) matrix b
!     !> \param ncols_b               number of columns of local (sub) matrix b
!     !> \param c                     matrix c, as device pointer of type(c_ptr)
!     !> \param nrows_c               number of rows of local (sub) matrix c
!     !> \param ncols_c               number of columns of local (sub) matrix c
!     !> \param error                 optional argument, error code which can be queried with elpa_strerr
!     subroutine elpa_hermitian_multiply_d_ptr_&
!                    &ELPA_IMPL_SUFFIX&
!                    & (self, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, &
!                                           c, nrows_c, ncols_c, error)
!       class(elpa_impl_t)              :: self
!       character*1                     :: uplo_a, uplo_c
!       integer(kind=c_int), intent(in) :: nrows_b, ncols_b, nrows_c, ncols_c, ncb
!       type(c_ptr)                     :: a, b, c
! #ifdef USE_FORTRAN2008
!       integer, optional               :: error
! #else
!       integer                         :: error
! #endif
!       logical                         :: success_l
     
!       success_l = .false.
! #if defined(INCLUDE_ROUTINES)
! #ifdef REALCASE
!       success_l = elpa_mult_at_b_d_ptr_&
! #endif
! #ifdef COMPLEXCASE
!       success_l = elpa_mult_ah_b_d_ptr_&
! #endif
!               &MATH_DATATYPE&
!               &_&
!               &PRECISION&
!               &_impl(self, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, &
!                                                   c, nrows_c, ncols_c)
! #endif
! #ifdef USE_FORTRAN2008
!       if (present(error)) then
!         if (success_l) then
!           error = ELPA_OK
!         else
!           error = ELPA_ERROR
!         endif
!       else if (.not. success_l) then
!         write(error_unit,'(a)') "ELPA: Error in hermitian_multiply() and you did not check for errors!"
!       endif
! #else
!       if (success_l) then
!         error = ELPA_OK
!       else
!         error = ELPA_ERROR
!       endif
! #endif
!     end subroutine  

!     !c> // /src/elpa_impl_math_template.F90
    
! #ifdef REALCASE
! #ifdef DOUBLE_PRECISION_REAL    
!     !c> void elpa_hermitian_multiply_a_h_a_d(elpa_t handle, char uplo_a, char uplo_c, int ncb, double *a, double *b, int nrows_b, int ncols_b, double *c, int nrows_c, int ncols_c, int *error);
! #endif
! #ifdef SINGLE_PRECISION_REAL
!     !c> void elpa_hermitian_multiply_a_h_a_f(elpa_t handle, char uplo_a, char uplo_c, int ncb, float *a, float *b, int nrows_b, int ncols_b, float *c, int nrows_c, int ncols_c, int *error);
! #endif
! #endif
! #ifdef COMPLEXCASE
! #ifdef DOUBLE_PRECISION_COMPLEX    
!     !c> void elpa_hermitian_multiply_a_h_a_dc(elpa_t handle, char uplo_a, char uplo_c, int ncb, double_complex *a, double_complex *b, int nrows_b, int ncols_b, double_complex *c, int nrows_c, int ncols_c, int *error);
! #endif
! #ifdef SINGLE_PRECISION_COMPLEX
!     !c> void elpa_hermitian_multiply_a_h_a_fc(elpa_t handle, char uplo_a, char uplo_c, int ncb, float_complex *a, float_complex *b, int nrows_b, int ncols_b, float_complex *c, int nrows_c, int ncols_c, int *error);
! #endif
! #endif
!     subroutine elpa_hermitian_multiply_a_h_a_&
!                     &ELPA_IMPL_SUFFIX&
!                     &_c(handle, uplo_a, uplo_c, ncb, a_p, b_p, nrows_b, &
!                                            ncols_b, c_p, nrows_c, ncols_c, error)          &
! #ifdef REALCASE
! #ifdef DOUBLE_PRECISION_REAL 
!                                            bind(C, name="elpa_hermitian_multiply_a_h_a_d")
! #endif
! #ifdef SINGLE_PRECISION_REAL 
!                                            bind(C, name="elpa_hermitian_multiply_a_h_a_f")
! #endif
! #endif
! #ifdef COMPLEXCASE
! #ifdef DOUBLE_PRECISION_COMPLEX
!                                            bind(C, name="elpa_hermitian_multiply_a_h_a_dc")
! #endif
! #ifdef SINGLE_PRECISION_COMPLEX 
!                                            bind(C, name="elpa_hermitian_multiply_a_h_a_fc")
! #endif
! #endif

!       type(c_ptr), intent(in), value               :: handle, a_p, b_p, c_p
!       character(1,C_CHAR), value                   :: uplo_a, uplo_c
!       integer(kind=c_int), value                   :: ncb, nrows_b, ncols_b, nrows_c, ncols_c
! #ifdef USE_FORTRAN2008
!       integer(kind=c_int), optional, intent(in)    :: error
! #else
!       integer(kind=c_int), intent(in)              :: error
! #endif
!       MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer :: a(:, :), b(:,:), c(:,:)
! !#ifdef USE_ASSUMED_SIZE
! !      MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer :: b(nrows_b,*), c(nrows_c,*)
! !#else
! !      MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer :: b(nrows_b,ncols_b), c(nrows_c,ncols_c)
! !#endif
!       type(elpa_impl_t), pointer                   :: self

!       call c_f_pointer(handle, self)
!       call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
!       call c_f_pointer(b_p, b, [nrows_b, ncols_b])
!       call c_f_pointer(c_p, c, [nrows_c, ncols_c])

!       call elpa_hermitian_multiply_a_h_a_&
!               &ELPA_IMPL_SUFFIX&
!               & (self, uplo_a, uplo_c, ncb, a, b, nrows_b, &
!                                      ncols_b, c, nrows_c, ncols_c, error)
!     end subroutine

! #ifdef REALCASE
! #ifdef DOUBLE_PRECISION_REAL    
!     !c> void elpa_hermitian_multiply_d_ptr_d(elpa_t handle, char uplo_a, char uplo_c, int ncb, double *a, double *b, int nrows_b, int ncols_b, double *c, int nrows_c, int ncols_c, int *error);
! #endif
! #ifdef SINGLE_PRECISION_REAL
!     !c> void elpa_hermitian_multiply_d_ptr_f(elpa_t handle, char uplo_a, char uplo_c, int ncb, float *a, float *b, int nrows_b, int ncols_b, float *c, int nrows_c, int ncols_c, int *error);
! #endif
! #endif
! #ifdef COMPLEXCASE
! #ifdef DOUBLE_PRECISION_COMPLEX    
!     !c> void elpa_hermitian_multiply_d_ptr_dc(elpa_t handle, char uplo_a, char uplo_c, int ncb, double_complex *a, double_complex *b, int nrows_b, int ncols_b, double_complex *c, int nrows_c, int ncols_c, int *error);
! #endif
! #ifdef SINGLE_PRECISION_COMPLEX
!     !c> void elpa_hermitian_multiply_d_ptr_fc(elpa_t handle, char uplo_a, char uplo_c, int ncb, float_complex *a, float_complex *b, int nrows_b, int ncols_b, float_complex *c, int nrows_c, int ncols_c, int *error);
! #endif
! #endif
!     subroutine elpa_hermitian_multiply_d_ptr_&
!                     &ELPA_IMPL_SUFFIX&
!                     &_c(handle, uplo_a, uplo_c, ncb, a_p, b_p, nrows_b, &
!                                            ncols_b, c_p, nrows_c, ncols_c, error)          &
! #ifdef REALCASE
! #ifdef DOUBLE_PRECISION_REAL 
!                                            bind(C, name="elpa_hermitian_multiply_d_ptr_d")
! #endif
! #ifdef SINGLE_PRECISION_REAL 
!                                            bind(C, name="elpa_hermitian_multiply_d_ptr_f")
! #endif
! #endif
! #ifdef COMPLEXCASE
! #ifdef DOUBLE_PRECISION_COMPLEX
!                                            bind(C, name="elpa_hermitian_multiply_d_ptr_dc")
! #endif
! #ifdef SINGLE_PRECISION_COMPLEX 
!                                            bind(C, name="elpa_hermitian_multiply_d_ptr_fc")
! #endif
! #endif

!       type(c_ptr), intent(in), value            :: handle, a_p, b_p, c_p
!       character(1,C_CHAR), value                :: uplo_a, uplo_c
!       integer(kind=c_int), value                :: ncb, nrows_b, ncols_b, nrows_c, ncols_c
! #ifdef USE_FORTRAN2008
!       integer(kind=c_int), optional, intent(in) :: error
! #else
!       integer(kind=c_int), intent(in)           :: error
! #endif
!       type(elpa_impl_t), pointer                :: self

!       call c_f_pointer(handle, self)
!       !call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

!       call elpa_hermitian_multiply_d_ptr_&
!               &ELPA_IMPL_SUFFIX&
!               & (self, uplo_a, uplo_c, ncb, a_p, b_p, nrows_b, &
!                                      ncols_b, c_p, nrows_c, ncols_c, error)
!     end subroutine

    !_____________________________________________________________________________________________________________________
    ! cholesky

    !>  \brief elpa_cholesky_a_h_a_d: class method to do a cholesky factorization
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
    !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_cholesky_a_h_a_&
                   &ELPA_IMPL_SUFFIX&
                   & (self, a, error)
      class(elpa_impl_t)              :: self
#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=C_DATATYPE_KIND)                  :: a(self%local_nrows,*)
#else
      MATH_DATATYPE(kind=C_DATATYPE_KIND)                  :: a(self%local_nrows,self%local_ncols)
#endif
#ifdef USE_FORTRAN2008
      integer, optional               :: error
#else
      integer                         :: error
#endif
      logical                         :: success_l

      success_l = .false.
#if defined(INCLUDE_ROUTINES)
      success_l = elpa_cholesky_a_h_a_&
              &MATH_DATATYPE&
              &_&
              &PRECISION&
              &_impl (self, a)
#endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in cholesky() and you did not check for errors!"
      endif
#else
      if (success_l) then
        error = ELPA_OK
      else
        error = ELPA_ERROR
      endif
#endif
    end subroutine    

#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
    !c> void elpa_cholesky_a_h_a_d(elpa_t handle, double *a, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> void elpa_cholesky_a_h_a_f(elpa_t handle, float *a, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void elpa_cholesky_a_h_a_dc(elpa_t handle, double_complex *a, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    !c> void elpa_cholesky_a_h_a_fc(elpa_t handle, float_complex *a, int *error);
#endif
#endif
    subroutine elpa_choleksy_a_h_a_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
                    bind(C, name="elpa_cholesky_a_h_a_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                    bind(C, name="elpa_cholesky_a_h_a_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
                    bind(C, name="elpa_cholesky_a_h_a_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX
                    bind(C, name="elpa_cholesky_a_h_a_fc")
#endif
#endif

      type(c_ptr), intent(in), value            :: handle, a_p
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(in) :: error
#else
      integer(kind=c_int), intent(in)           :: error
#endif
      MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer              :: a(:, :)
      type(elpa_impl_t), pointer                :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_cholesky_a_h_a_&
              &ELPA_IMPL_SUFFIX&
              & (self, a, error)
    end subroutine      


    !>  \brief elpa_cholesky_d_ptr_d: class method to do a cholesky factorization
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
    !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed as type(c_ptr) on a device
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_cholesky_d_ptr_&
                   &ELPA_IMPL_SUFFIX&
                   & (self, a, error)
      use iso_c_binding
      class(elpa_impl_t)              :: self
      type(c_ptr)                     :: a
#ifdef USE_FORTRAN2008
      integer, optional               :: error
#else
      integer                         :: error
#endif
      logical                         :: success_l

      success_l = .false.
#if defined(INCLUDE_ROUTINES)
      success_l = elpa_cholesky_d_ptr_&
              &MATH_DATATYPE&
              &_&
              &PRECISION&
              &_impl (self, a)
#endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in cholesky() and you did not check for errors!"
      endif
#else
      if (success_l) then
        error = ELPA_OK
      else
        error = ELPA_ERROR
      endif
#endif
    end subroutine    

#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
    !c> void elpa_cholesky_d_ptr_d(elpa_t handle, double *a, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> void elpa_cholesky_d_ptr_f(elpa_t handle, float *a, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void elpa_cholesky_d_ptr_dc(elpa_t handle, double_complex *a, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    !c> void elpa_cholesky_d_ptr_fc(elpa_t handle, float_complex *a, int *error);
#endif
#endif
    subroutine elpa_choleksy_d_ptr_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
                    bind(C, name="elpa_cholesky_d_ptr_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                    bind(C, name="elpa_cholesky_d_ptr_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
                    bind(C, name="elpa_cholesky_d_ptr_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX
                    bind(C, name="elpa_cholesky_d_ptr_fc")
#endif
#endif

      type(c_ptr), intent(in), value            :: handle, a_p
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(in) :: error
#else
      integer(kind=c_int), intent(in)           :: error
#endif
      type(elpa_impl_t), pointer                :: self

      call c_f_pointer(handle, self)
      !call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_cholesky_d_ptr_&
              &ELPA_IMPL_SUFFIX&
              & (self, a_p, error)
    end subroutine      

    !_____________________________________________________________________________________________________________________
    ! invert_trm

    !>  \brief elpa_invert_trm_a_h_a_d: class method to invert a triangular
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
    !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_invert_trm_a_h_a_&
                   &ELPA_IMPL_SUFFIX&
                  & (self, a, error)
      class(elpa_impl_t)              :: self
#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=C_DATATYPE_KIND)             :: a(self%local_nrows,*)
#else
      MATH_DATATYPE(kind=C_DATATYPE_KIND)             :: a(self%local_nrows,self%local_ncols)
#endif
#ifdef USE_FORTRAN2008
      integer, optional               :: error
#else
      integer                         :: error
#endif
      logical                         :: success_l

      success_l = .false.
#if defined(INCLUDE_ROUTINES)
      success_l = elpa_invert_trm_a_h_a_&
              &MATH_DATATYPE&
              &_&
              &PRECISION&
              &_impl (self, a)
#endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in invert_trm() and you did not check for errors!"
      endif
#else
      if (success_l) then
        error = ELPA_OK
      else
        error = ELPA_ERROR
      endif
#endif
    end subroutine   



#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL      
    !c> void elpa_invert_trm_a_h_a_d(elpa_t handle, double *a, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL      
    !c> void elpa_invert_trm_a_h_a_f(elpa_t handle, float *a, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX     
    !c> void elpa_invert_trm_a_h_a_dc(elpa_t handle, double_complex *a, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX      
    !c> void elpa_invert_trm_a_h_a_fc(elpa_t handle, float_complex *a, int *error);
#endif
#endif
    subroutine elpa_invert_trm_a_h_a_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
                    bind(C, name="elpa_invert_trm_a_h_a_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                    bind(C, name="elpa_invert_trm_a_h_a_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
                    bind(C, name="elpa_invert_trm_a_h_a_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX
                    bind(C, name="elpa_invert_trm_a_h_a_fc")
#endif
#endif

      type(c_ptr), intent(in), value            :: handle, a_p
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(in) :: error
#else
      integer(kind=c_int), intent(in)           :: error
#endif
      MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer              :: a(:, :)
      type(elpa_impl_t), pointer                :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_invert_trm_a_h_a_&
              &ELPA_IMPL_SUFFIX&
              & (self, a, error)
    end subroutine


    !>  \brief elpa_invert_trm_d_ptr_d: class method to invert a triangular
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
    !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed as device pointer
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_invert_trm_d_ptr_&
                   &ELPA_IMPL_SUFFIX&
                  & (self, a, error)
      use iso_c_binding
      class(elpa_impl_t)              :: self
      type(c_ptr)                     :: a
#ifdef USE_FORTRAN2008
      integer, optional               :: error
#else
      integer                         :: error
#endif
      logical                         :: success_l

      success_l = .false.
#if defined(INCLUDE_ROUTINES)
      success_l = elpa_invert_trm_d_ptr_&
              &MATH_DATATYPE&
              &_&
              &PRECISION&
              &_impl (self, a)
#endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in invert_trm() and you did not check for errors!"
      endif
#else
      if (success_l) then
        error = ELPA_OK
      else
        error = ELPA_ERROR
      endif
#endif
    end subroutine   



#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL      
    !c> void elpa_invert_trm_d_ptr_d(elpa_t handle, double *a, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL      
    !c> void elpa_invert_trm_d_ptr_f(elpa_t handle, float *a, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX     
    !c> void elpa_invert_trm_d_ptr_dc(elpa_t handle, double_complex *a, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX      
    !c> void elpa_invert_trm_d_ptr_fc(elpa_t handle, float_complex *a, int *error);
#endif
#endif
    subroutine elpa_invert_trm_d_ptr_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
                    bind(C, name="elpa_invert_trm_d_ptr_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                    bind(C, name="elpa_invert_trm_d_ptr_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
                    bind(C, name="elpa_invert_trm_d_ptr_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX
                    bind(C, name="elpa_invert_trm_d_ptr_fc")
#endif
#endif

      type(c_ptr), intent(in), value            :: handle, a_p
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(in) :: error
#else
      integer(kind=c_int), intent(in)           :: error
#endif
      type(elpa_impl_t), pointer                :: self

      call c_f_pointer(handle, self)

      call elpa_invert_trm_d_ptr_&
              &ELPA_IMPL_SUFFIX&
              & (self, a_p, error)
    end subroutine

    !_____________________________________________________________________________________________________________________
    ! solve_tridiagonal

    !>  \brief elpa_solve_tridiagonal_d: class method to solve the eigenvalue problem for a tridiagonal matrix a
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
    !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param d        array d  on input diagonal elements of tridiagonal matrix, on
    !>                           output the eigenvalues in ascending order
    !>  \param e        array e on input subdiagonal elements of matrix, on exit destroyed
    !>  \param q        matrix  on exit : contains the eigenvectors
    !>  \param error    integer, optional: returns an error code, which can be queried with elpa_strerr 
    subroutine elpa_solve_tridiagonal_&
                   &ELPA_IMPL_SUFFIX&
                   & (self, d, e, q, error)
      use solve_tridi
      implicit none
      class(elpa_impl_t)              :: self
      real(kind=C_REAL_DATATYPE)                  :: d(self%na), e(self%na)
#ifdef USE_ASSUMED_SIZE
      real(kind=C_REAL_DATATYPE)                  :: q(self%local_nrows,*)
#else
      real(kind=C_REAL_DATATYPE)                  :: q(self%local_nrows,self%local_ncols)
#endif
#ifdef USE_FORTRAN2008
      integer, optional               :: error
#else
      integer                         :: error
#endif
      logical                         :: success_l

#if defined(INCLUDE_ROUTINES)
      success_l = elpa_solve_tridi_&
              &PRECISION&
              &_impl(self, d, e, q)
#else
     write(error_unit,*) "ELPA is not compiled with single-precision support"
#ifdef USE_FORTRAN2008
     if (present(error)) then
       error = ELPA_ERROR
       return
     else
       return
     endif
#else
     error = ELPA_ERROR
     return
#endif
#endif
#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve_tridiagonal() and you did not check for errors!"
      endif
#else
      if (success_l) then
        error = ELPA_OK
      else
        error = ELPA_ERROR
      endif
#endif
    end subroutine   


#ifdef DOUBLE_PRECISION_REAL      
    !c> void elpa_solve_tridiagonal_d(elpa_t handle, double *d, double *e, double *q, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL      
    !c> void elpa_solve_tridiagonal_f(elpa_t handle, float *d, float *e, float *q, int *error);
#endif

    subroutine elpa_solve_tridiagonal_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, d_p, e_p, q_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
                    bind(C, name="elpa_solve_tridiagonal_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                    bind(C, name="elpa_solve_tridiagonal_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
                    & !bind(C, name="elpa_solve_tridiagonal_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX
                    & !bind(C, name="elpa_solve_tridiagonal_fc")
#endif
#endif

      type(c_ptr), intent(in), value            :: handle, d_p, e_p, q_p
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(in) :: error
#else
      integer(kind=c_int), intent(in)           :: error
#endif
      real(kind=C_REAL_DATATYPE), pointer       :: d(:), e(:), q(:, :)
      type(elpa_impl_t), pointer                :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(d_p, d, [self%na])
      call c_f_pointer(e_p, e, [self%na])
      call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])

      call elpa_solve_tridiagonal_&
              &ELPA_IMPL_SUFFIX&
              & (self, d, e, q, error)
    end subroutine
    