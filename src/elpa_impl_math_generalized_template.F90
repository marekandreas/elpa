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
    !>  \brief elpa_generalized_eigenvectors_d: class method to solve the eigenvalue problem
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
    !>  blocksize, the number of eigenvectors
    !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
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
    !>  \param b                                    Distributed matrix, part of the generalized eigenvector problem, or the
    !>                                              product of a previous call to this function (see is_already_decomposed).
    !>                                              Distribution is like in Scalapack.
    !>                                              If is_already_decomposed is false, on exit replaced by the decomposition
    !>
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param q                                    On output: Eigenvectors of a
    !>                                              Distribution is like in Scalapack.
    !>                                              Must be always dimensioned to the full size (corresponding to (na,na))
    !>                                              even if only a part of the eigenvalues is needed.
    !>
    !>  \param is_already_decomposed                has to be set to .false. for the first call with a given b and .true. for
    !>                                              each subsequent call with the same b, since b then already contains
    !>                                              decomposition and thus the decomposing step is skipped
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr 
    subroutine elpa_generalized_eigenvectors_&
                    &ELPA_IMPL_SUFFIX&
                    & (self, a, b, ev, q, is_already_decomposed, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit
      use, intrinsic :: iso_c_binding
      class(elpa_impl_t)  :: self

#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, *), b(self%local_nrows, *), q(self%local_nrows, *)
#else
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, self%local_ncols), b(self%local_nrows, self%local_ncols), &
                             q(self%local_nrows, self%local_ncols)
#endif
      real(kind=C_REAL_DATATYPE) :: ev(self%na)
      logical             :: is_already_decomposed

      integer, optional   :: error
      integer             :: error_l
      integer(kind=c_int) :: solver
      logical             :: success_l

      error_l   = -10
      success_l = .false.
#if defined(INCLUDE_ROUTINES)
      call self%elpa_transform_generalized_&
              &ELPA_IMPL_SUFFIX&
              & (a, b, is_already_decomposed, error_l)
#endif
      if (present(error)) then
          error = error_l
      else if (error_l .ne. ELPA_OK) then
        write(error_unit,'(a)') "ELPA: Error in transform_generalized() and you did not check for errors!"
      endif

      call self%get("solver", solver,error_l)
      if (solver .eq. ELPA_SOLVER_1STAGE) then
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                &_1stage_a_h_a_&
                &PRECISION&
                &_impl(self, a, ev, q)
#endif
      else if (solver .eq. ELPA_SOLVER_2STAGE) then
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                &_2stage_a_h_a_&
                &PRECISION&
                &_impl(self, a, ev, q)
#endif
      else
        write(error_unit,'(a)') "Unknown solver: Aborting!"
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
      endif

      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve() and you did not check for errors!"
      endif

#if defined(INCLUDE_ROUTINES)
      call self%elpa_transform_back_generalized_&
              &ELPA_IMPL_SUFFIX&
              & (b, q, error_l)
#endif
      if (present(error)) then
          error = error_l
      else if (error_l .ne. ELPA_OK) then
        write(error_unit,'(a)') "ELPA: Error in transform_back_generalized() and you did not check for errors!"
      endif
    end subroutine

    !c> // /src/elpa_impl_math_generalized_template.F90

#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL  
    !c> void elpa_generalized_eigenvectors_d(elpa_t handle, double *a, double *b, double *ev, double *q,
    !c> int is_already_decomposed, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL  
    !c> void elpa_generalized_eigenvectors_f(elpa_t handle, float *a, float *b, float *ev, float *q,
    !c> int is_already_decomposed, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void elpa_generalized_eigenvectors_dc(elpa_t handle, double_complex *a, double_complex *b, double *ev, double_complex *q,
    !c> int is_already_decomposed, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    !c> void elpa_generalized_eigenvectors_fc(elpa_t handle, float_complex *a, float_complex *b, float *ev, float_complex *q,
    !c> int is_already_decomposed, int *error);
#endif
#endif
    subroutine elpa_generalized_eigenvectors_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, b_p, ev_p, q_p, is_already_decomposed, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL 
                                                            bind(C, name="elpa_generalized_eigenvectors_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                                                            bind(C, name="elpa_generalized_eigenvectors_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
                                                            bind(C, name="elpa_generalized_eigenvectors_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX
                                                            bind(C, name="elpa_generalized_eigenvectors_fc")
#endif
#endif
      type(c_ptr), intent(in), value :: handle, a_p, b_p, ev_p, q_p
      integer(kind=c_int), intent(in), value :: is_already_decomposed
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(in) :: error
#else
      integer(kind=c_int), intent(in) :: error
#endif
      MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer :: a(:, :), b(:, :), q(:, :)
      real(kind=C_REAL_DATATYPE), pointer :: ev(:)
      logical :: is_already_decomposed_fortran
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(b_p, b, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])
      call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])
      if(is_already_decomposed .eq. 0) then
        is_already_decomposed_fortran = .false.
      else
        is_already_decomposed_fortran = .true.
      end if

      call elpa_generalized_eigenvectors_&
              &ELPA_IMPL_SUFFIX&
              & (self, a, b, ev, q, is_already_decomposed_fortran, error)
    end subroutine

    

    !>  \brief elpa_generalized_eigenvalues_d: class method to solve the eigenvalue problem
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
    !>  blocksize, the number of eigenvectors
    !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
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
    !>  \param b                                    Distributed matrix, part of the generalized eigenvector problem, or the
    !>                                              product of a previous call to this function (see is_already_decomposed).
    !>                                              Distribution is like in Scalapack.
    !>                                              If is_already_decomposed is false, on exit replaced by the decomposition
    !>
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param is_already_decomposed                has to be set to .false. for the first call with a given b and .true. for
    !>                                              each subsequent call with the same b, since b then already contains
    !>                                              decomposition and thus the decomposing step is skipped
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_generalized_eigenvalues_&
                    &ELPA_IMPL_SUFFIX&
                    & (self, a, b, ev, is_already_decomposed, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit
      use, intrinsic :: iso_c_binding
      class(elpa_impl_t)  :: self

#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, *), b(self%local_nrows, *)
#else
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, self%local_ncols), b(self%local_nrows, self%local_ncols)
#endif
      real(kind=C_REAL_DATATYPE) :: ev(self%na)
      logical             :: is_already_decomposed

      integer, optional   :: error
      integer             :: error_l
      integer(kind=c_int) :: solver
      logical             :: success_l

      error_l = -10
      success_l = .false.
#if defined(INCLUDE_ROUTINES)
      call self%elpa_transform_generalized_&
              &ELPA_IMPL_SUFFIX&
              & (a, b, is_already_decomposed, error_l)
#endif
      if (present(error)) then
          error = error_l
      else if (error_l .ne. ELPA_OK) then
        write(error_unit,'(a)') "ELPA: Error in transform_generalized() and you did not check for errors!"
      endif

      call self%get("solver", solver,error_l)
      if (solver .eq. ELPA_SOLVER_1STAGE) then
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                &_1stage_a_h_a_&
                &PRECISION&
                &_impl(self, a, ev)
#endif
      else if (solver .eq. ELPA_SOLVER_2STAGE) then
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                &_2stage_a_h_a_&
                &PRECISION&
                &_impl(self, a, ev)
#endif
      else
        write(error_unit,'(a)') "Unknown solver: Aborting!"
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
      endif

      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve() and you did not check for errors!"
      endif

    end subroutine

#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL  
    !c> void elpa_generalized_eigenvalues_d(elpa_t handle, double *a, double *b, double *ev,
    !c> int is_already_decomposed, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL  
    !c> void elpa_generalized_eigenvalues_f(elpa_t handle, float *a, float *b, float *ev,
    !c> int is_already_decomposed, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void elpa_generalized_eigenvalues_dc(elpa_t handle, double_complex *a, double_complex *b, double *ev,
    !c> int is_already_decomposed, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    !c> void elpa_generalized_eigenvalues_fc(elpa_t handle, float_complex *a, float_complex *b, float *ev,
    !c> int is_already_decomposed, int *error);
#endif
#endif
    subroutine elpa_generalized_eigenvalues_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, b_p, ev_p, is_already_decomposed, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL 
                                                            bind(C, name="elpa_generalized_eigenvalues_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                                                            bind(C, name="elpa_generalized_eigenvalues_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
                                                            bind(C, name="elpa_generalized_eigenvalues_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX
                                                            bind(C, name="elpa_generalized_eigenvalues_fc")
#endif
#endif
      type(c_ptr), intent(in), value :: handle, a_p, b_p, ev_p
      integer(kind=c_int), intent(in), value :: is_already_decomposed
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(in) :: error
#else
      integer(kind=c_int), intent(in) :: error
#endif

      MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer :: a(:, :), b(:, :)
      real(kind=C_REAL_DATATYPE), pointer :: ev(:)
      logical :: is_already_decomposed_fortran
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(b_p, b, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])
      if(is_already_decomposed .eq. 0) then
        is_already_decomposed_fortran = .false.
      else
        is_already_decomposed_fortran = .true.
      end if

      call elpa_generalized_eigenvalues_&
              &ELPA_IMPL_SUFFIX&
              & (self, a, b, ev, is_already_decomposed_fortran, error)
    end subroutine

