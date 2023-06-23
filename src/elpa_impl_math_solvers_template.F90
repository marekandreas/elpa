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
    !>  \brief elpa_eigenvectors_d: class method to solve the eigenvalue problem
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
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param q                                    On output: Eigenvectors of a
    !>                                              Distribution is like in Scalapack.
    !>                                              Must be always dimensioned to the full size (corresponding to (na,na))
    !>                                              even if only a part of the eigenvalues is needed.
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr   

    subroutine elpa_eigenvectors_a_h_a_&
                    &ELPA_IMPL_SUFFIX&
                    & (self, a, ev, q, error)
      class(elpa_impl_t)  :: self

#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=C_REAL_DATATYPE) :: ev(self%na)

#ifdef USE_FORTRAN2008
      integer, optional   :: error
#else
      integer             :: error
#endif
      integer             :: error2
      integer(kind=c_int) :: solver
      logical             :: success_l

      success_l = .false.
      call self%get("solver", solver,error2)
      if (error2 .ne. ELPA_OK) then
        print *,"Problem setting solver. Aborting..."
#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = error2
        endif
#else
        error = error2
#endif
        return
      endif
      if (solver .eq. ELPA_SOLVER_1STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                &_1stage_a_h_a_&
                &PRECISION&
                &_impl(self, a, ev, q)
#endif
        call self%autotune_timer%stop("accumulator")

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                &_2stage_a_h_a_&
                &PRECISION&
                &_impl(self, a, ev, q)
#endif
        call self%autotune_timer%stop("accumulator")

      else ! solver
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

#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR_DURING_COMPUTATION
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in eigenvectors() and you did not check for errors!"
      endif
#else
      if (success_l) then
        error = ELPA_OK
      else
        error = ELPA_ERROR_DURING_COMPUTATION
      endif
#endif
    end subroutine 

    !>  \brief elpa_eigenvectors_d_ptr_d: class method to solve the eigenvalue problem
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
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param q                                    On output: Eigenvectors of a
    !>                                              Distribution is like in Scalapack.
    !>                                              Must be always dimensioned to the full size (corresponding to (na,na))
    !>                                              even if only a part of the eigenvalues is needed.
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr   

    subroutine elpa_eigenvectors_d_ptr_&
                    &ELPA_IMPL_SUFFIX&
                    & (self, a, ev, q, error)
      use iso_c_binding

      implicit none
      class(elpa_impl_t)  :: self

      type(c_ptr)         :: a, q, ev

#ifdef USE_FORTRAN2008
      integer, optional   :: error
#else
      integer             :: error
#endif
      integer             :: error2
      integer(kind=c_int) :: solver
      logical             :: success_l

      success_l = .false.
      call self%get("solver", solver,error2)
      if (error2 .ne. ELPA_OK) then
        print *,"Problem setting solver. Aborting..."
#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = error2
        endif
#else
        error = error2
#endif
        return
      endif
      if (solver .eq. ELPA_SOLVER_1STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                &_1stage_d_ptr_&
                &PRECISION&
                &_impl(self, a, ev, q)
#endif
        call self%autotune_timer%stop("accumulator")

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                &_2stage_d_ptr_&
                &PRECISION&
                &_impl(self, a, ev, q)
#endif
        call self%autotune_timer%stop("accumulator")

      else ! solver
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

#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR_DURING_COMPUTATION
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in eigenvectors() and you did not check for errors!"
      endif
#else
      if (success_l) then
        error = ELPA_OK
      else
        error = ELPA_ERROR_DURING_COMPUTATION
      endif
#endif
    end subroutine 
    
    !c> // /src/elpa_impl_math_solvers_template.F90 
    
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
    !c> void elpa_eigenvectors_a_h_a_d(elpa_t handle, double *a, double *ev, double *q, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> void elpa_eigenvectors_a_h_a_f(elpa_t handle, float *a, float *ev, float *q, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void elpa_eigenvectors_a_h_a_dc(elpa_t handle, double_complex *a, double *ev, double_complex *q, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    !c> void elpa_eigenvectors_a_h_a_fc(elpa_t handle, float_complex *a, float *ev, float_complex *q, int *error);
#endif
#endif
    subroutine elpa_eigenvectors_a_h_a_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, ev_p, q_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL  
                    bind(C, name="elpa_eigenvectors_a_h_a_d")
#endif
#ifdef SINGLE_PRECISION_REAL  
                    bind(C, name="elpa_eigenvectors_a_h_a_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
                    bind(C, name="elpa_eigenvectors_a_h_a_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX
                    bind(C, name="elpa_eigenvectors_a_h_a_fc")
#endif
#endif
      type(c_ptr), intent(in), value            :: handle, a_p, ev_p, q_p
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(in) :: error
#else
      integer(kind=c_int), intent(in)           :: error
#endif

      MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer :: a(:, :), q(:, :)
      real(kind=C_REAL_DATATYPE), pointer          :: ev(:)
      type(elpa_impl_t), pointer                   :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])
      call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])

      call elpa_eigenvectors_a_h_a_&
              &ELPA_IMPL_SUFFIX&
              & (self, a, ev, q, error)
    end subroutine    

#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
    !c> void elpa_eigenvectors_d_ptr_d(elpa_t handle, double *a, double *ev, double *q, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> void elpa_eigenvectors_d_ptr_f(elpa_t handle, float *a, float *ev, float *q, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void elpa_eigenvectors_d_ptr_dc(elpa_t handle, double_complex *a, double *ev, double_complex *q, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    !c> void elpa_eigenvectors_d_ptr_fc(elpa_t handle, float_complex *a, float *ev, float_complex *q, int *error);
#endif
#endif
    subroutine elpa_eigenvectors_d_ptr_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, ev_p, q_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL  
                    bind(C, name="elpa_eigenvectors_d_ptr_d")
#endif
#ifdef SINGLE_PRECISION_REAL  
                    bind(C, name="elpa_eigenvectors_d_ptr_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
                    bind(C, name="elpa_eigenvectors_d_ptr_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX
                    bind(C, name="elpa_eigenvectors_d_ptr_fc")
#endif
#endif
      type(c_ptr), intent(in), value            :: handle, a_p, ev_p, q_p
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(in) :: error
#else
      integer(kind=c_int), intent(in)           :: error
#endif

      !MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer :: a(:, :), q(:, :)
      !real(kind=C_REAL_DATATYPE), pointer          :: ev(:)
      type(elpa_impl_t), pointer                   :: self

      call c_f_pointer(handle, self)
      !call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      !call c_f_pointer(ev_p, ev, [self%na])
      !call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])

      call elpa_eigenvectors_d_ptr_&
              &ELPA_IMPL_SUFFIX&
              & (self, a_p, ev_p, q_p, error)
    end subroutine

#ifdef HAVE_SKEWSYMMETRIC
#ifdef REALCASE 
    !>  \brief elpa_skew_eigenvectors_d: class method to solve the real valued skew-symmetric eigenvalue problem
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
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param q                                    On output: Eigenvectors of a
    !>                                              Distribution is like in Scalapack.
    !>                                              Must be always dimensioned to the full size (corresponding to (na,na))
    !>                                              even if only a part of the eigenvalues is needed.
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr   

    subroutine elpa_skew_eigenvectors_a_h_a_&
                    &ELPA_IMPL_SUFFIX&
                    & (self, a, ev, q, error)
      class(elpa_impl_t)  :: self

#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, self%local_ncols)
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: q(self%local_nrows, 2*self%local_ncols)
#endif
      real(kind=C_REAL_DATATYPE)          :: ev(self%na)

#ifdef USE_FORTRAN2008
      integer, optional                   :: error
#else
      integer                             :: error
#endif
      integer                             :: error2
      integer(kind=c_int)                 :: solver
      logical                             :: success_l

      success_l = .false.
      call self%get("solver", solver,error2)
      !call self%set("is_skewsymmetric",1,error2)
      if (error2 .ne. ELPA_OK) then
        print *,"Problem setting is_skewsymmetric. Aborting..."
#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = error2
        endif
#else
        error = error2
#endif
        return
      endif
      if (solver .eq. ELPA_SOLVER_1STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_skew_evp_&
                &MATH_DATATYPE&
                &_1stage_a_h_a_&
                &PRECISION&
                &_impl(self, a, ev, q)
#endif
        call self%autotune_timer%stop("accumulator")

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_skew_evp_&
                &MATH_DATATYPE&
                &_2stage_a_h_a_&
                &PRECISION&
                &_impl(self, a, ev, q)
#endif
        call self%autotune_timer%stop("accumulator")

      else ! solver
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

#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR_DURING_COMPUTATION
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in skew_eigenvectors() and you did not check for errors!"
      endif
#else
      if (success_l) then
        error = ELPA_OK
      else
        error = ELPA_ERROR_DURING_COMPUTATION
      endif
#endif
    end subroutine

#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
    !c> #ifdef HAVE_SKEWSYMMETRIC
    !c> void elpa_skew_eigenvectors_a_h_a_d(elpa_t handle, double *a, double *ev, double *q, int *error);
    !c> #endif
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> #ifdef HAVE_SKEWSYMMETRIC
    !c> void elpa_skew_eigenvectors_a_h_a_f(elpa_t handle, float *a, float *ev, float *q, int *error);
    !c> #endif
#endif
#endif
    subroutine elpa_skew_eigenvectors_a_h_a_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, ev_p, q_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
                    bind(C, name="elpa_skew_eigenvectors_a_h_a_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                    bind(C, name="elpa_skew_eigenvectors_a_h_a_f")
#endif
#endif

      type(c_ptr), intent(in), value            :: handle, a_p, ev_p, q_p
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(in) :: error
#else
      integer(kind=c_int), intent(in)           :: error
#endif

      MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer :: a(:, :), q(:, :)
      real(kind=C_REAL_DATATYPE), pointer          :: ev(:)
      type(elpa_impl_t), pointer                   :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])
      call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])

      call elpa_skew_eigenvectors_a_h_a_&
              &ELPA_IMPL_SUFFIX&
              & (self, a, ev, q, error)
    end subroutine

    !>  \brief elpa_skew_eigenvectors_d: class method to solve the real valued skew-symmetric eigenvalue problem
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
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param q                                    On output: Eigenvectors of a
    !>                                              Distribution is like in Scalapack.
    !>                                              Must be always dimensioned to the full size (corresponding to (na,na))
    !>                                              even if only a part of the eigenvalues is needed.
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr   

    subroutine elpa_skew_eigenvectors_d_ptr_&
                    &ELPA_IMPL_SUFFIX&
                    & (self, a, ev, q, error)
      use iso_c_binding
      implicit none
      class(elpa_impl_t)  :: self

      type(c_ptr)         :: a, q, ev

#ifdef USE_FORTRAN2008
      integer, optional                   :: error
#else
      integer                             :: error
#endif
      integer                             :: error2
      integer(kind=c_int)                 :: solver
      logical                             :: success_l

      success_l = .false.
      call self%get("solver", solver,error2)
      !call self%set("is_skewsymmetric",1,error2)
      if (error2 .ne. ELPA_OK) then
        print *,"Problem setting is_skewsymmetric. Aborting..."
#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = error2
        endif
#else
        error = error2
#endif
        return
      endif
      if (solver .eq. ELPA_SOLVER_1STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_skew_evp_&
                &MATH_DATATYPE&
                &_1stage_d_ptr_&
                &PRECISION&
                &_impl(self, a, ev, q)
#endif
        call self%autotune_timer%stop("accumulator")

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_skew_evp_&
                &MATH_DATATYPE&
                &_2stage_d_ptr_&
                &PRECISION&
                &_impl(self, a, ev, q)
#endif
        call self%autotune_timer%stop("accumulator")

      else ! solver
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

#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR_DURING_COMPUTATION
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in skew_eigenvectors() and you did not check for errors!"
      endif
#else
      if (success_l) then
        error = ELPA_OK
      else
        error = ELPA_ERROR_DURING_COMPUTATION
      endif
#endif
    end subroutine

#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
    !c> #ifdef HAVE_SKEWSYMMETRIC
    !c> void elpa_skew_eigenvectors_d_ptr_d(elpa_t handle, double *a, double *ev, double *q, int *error);
    !c> #endif
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> #ifdef HAVE_SKEWSYMMETRIC
    !c> void elpa_skew_eigenvectors_d_ptr_f(elpa_t handle, float *a, float *ev, float *q, int *error);
    !c> #endif
#endif
#endif
    subroutine elpa_skew_eigenvectors_d_ptr_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, ev_p, q_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
                    bind(C, name="elpa_skew_eigenvectors_d_ptr_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                    bind(C, name="elpa_skew_eigenvectors_d_ptr_f")
#endif
#endif

      type(c_ptr), intent(in), value            :: handle, a_p, ev_p, q_p
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(in) :: error
#else
      integer(kind=c_int), intent(in)           :: error
#endif

      !MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer :: a(:, :), q(:, :)
      !real(kind=C_REAL_DATATYPE), pointer          :: ev(:)
      type(elpa_impl_t), pointer                   :: self

      call c_f_pointer(handle, self)
      !call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      !call c_f_pointer(ev_p, ev, [self%na])
      !call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])

      call elpa_skew_eigenvectors_d_ptr_&
              &ELPA_IMPL_SUFFIX&
              & (self, a_p, ev_p, q_p, error)
    end subroutine

#endif /* REALCASE */
#endif /* HAVE_SKEWSYMMETRIC */

    !>  \brief elpa_eigenvalues_d: class method to solve the eigenvalue problem
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
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_eigenvalues_a_h_a_&
                    &ELPA_IMPL_SUFFIX&
                    & (self, a, ev, error)
      class(elpa_impl_t)  :: self
#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, *)
#else
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, self%local_ncols)
#endif
      real(kind=C_REAL_DATATYPE) :: ev(self%na)
#ifdef USE_FORTRAN2008
      integer, optional   :: error
#else
      integer             :: error
#endif
      integer             :: error2
      integer(kind=c_int) :: solver
      logical             :: success_l

      success_l = .false.
      call self%get("solver", solver,error2)
      if (error2 .ne. ELPA_OK) then
         print *,"Problem getting solver option. Aborting..."
#ifdef USE_FORTRAN2008
         if (present(error)) then
           error = error2
         endif
#else
         error = error2
#endif
         return
      endif

      if (solver .eq. ELPA_SOLVER_1STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                          &_1stage_a_h_a_&
                          &PRECISION&
                          &_impl(self, a, ev)
#endif
        call self%autotune_timer%stop("accumulator")

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                                   &_2stage_a_h_a_&
                                   &PRECISION&
                                   &_impl(self, a, ev)
#endif
        call self%autotune_timer%stop("accumulator")

      else ! solver
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
#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR_DURING_COMPUTATION
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in eigenvalues() and you did not check for errors!"
      endif
#else
      if (success_l) then
        error = ELPA_OK
      else
        error = ELPA_ERROR_DURING_COMPUTATION
      endif
#endif
    end subroutine


    !>  \brief elpa_eigenvalues_d: class method to solve the eigenvalue problem
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
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_eigenvalues_d_ptr_&
                    &ELPA_IMPL_SUFFIX&
                    & (self, a, ev, error)
      use iso_c_binding
      implicit none

      class(elpa_impl_t)  :: self
      type(c_ptr)         :: a, ev
#ifdef USE_FORTRAN2008
      integer, optional   :: error
#else
      integer             :: error
#endif
      integer             :: error2
      integer(kind=c_int) :: solver
      logical             :: success_l

      success_l = .false.
      call self%get("solver", solver,error2)
      if (error2 .ne. ELPA_OK) then
         print *,"Problem getting solver option. Aborting..."
#ifdef USE_FORTRAN2008
         if (present(error)) then
           error = error2
         endif
#else
         error = error2
#endif
         return
      endif

      if (solver .eq. ELPA_SOLVER_1STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                          &_1stage_d_ptr_&
                          &PRECISION&
                          &_impl(self, a, ev)
#endif
        call self%autotune_timer%stop("accumulator")

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                                   &_2stage_d_ptr_&
                                   &PRECISION&
                                   &_impl(self, a, ev)
#endif
        call self%autotune_timer%stop("accumulator")

      else ! solver
        write(error_unit,*) "Unkown solver. Aborting!"
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
#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR_DURING_COMPUTATION
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in eigenvalues() and you did not check for errors!"
      endif
#else
      if (success_l) then
        error = ELPA_OK
      else
        error = ELPA_ERROR_DURING_COMPUTATION
      endif
#endif
    end subroutine

#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
    !c> void elpa_eigenvalues_a_h_a_d(elpa_t handle, double *a, double *ev, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> void elpa_eigenvalues_a_h_a_f(elpa_t handle, float *a, float *ev, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void elpa_eigenvalues_a_h_a_dc(elpa_t handle, double_complex *a, double *ev, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    !c> void elpa_eigenvalues_a_h_a_fc(elpa_t handle, float_complex *a, float *ev, int *error);
#endif
#endif
    subroutine elpa_eigenvalues_a_h_a_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, ev_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL    
                    bind(C, name="elpa_eigenvalues_a_h_a_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                    bind(C, name="elpa_eigenvalues_a_h_a_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX 
                    bind(C, name="elpa_eigenvalues_a_h_a_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX
                    bind(C, name="elpa_eigenvalues_a_h_a_fc")
#endif
#endif

      type(c_ptr), intent(in), value :: handle, a_p, ev_p
      integer(kind=c_int), intent(in) :: error

      MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer :: a(:, :)
      real(kind=C_REAL_DATATYPE), pointer :: ev(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])

      call elpa_eigenvalues_a_h_a_&
              &ELPA_IMPL_SUFFIX&
              & (self, a, ev, error)
    end subroutine    

#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
    !c> void elpa_eigenvalues_d_ptr_d(elpa_t handle, double *a, double *ev, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> void elpa_eigenvalues_d_ptr_f(elpa_t handle, float *a, float *ev, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void elpa_eigenvalues_d_ptr_dc(elpa_t handle, double_complex *a, double *ev, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    !c> void elpa_eigenvalues_d_ptr_fc(elpa_t handle, float_complex *a, float *ev, int *error);
#endif
#endif
    subroutine elpa_eigenvalues_d_ptr_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, ev_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL    
                    bind(C, name="elpa_eigenvalues_d_ptr_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                    bind(C, name="elpa_eigenvalues_d_ptr_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX 
                    bind(C, name="elpa_eigenvalues_d_ptr_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX
                    bind(C, name="elpa_eigenvalues_d_ptr_fc")
#endif
#endif

      type(c_ptr), intent(in), value :: handle, a_p, ev_p
      integer(kind=c_int), intent(in) :: error

      !MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer :: a(:, :)
      !real(kind=C_REAL_DATATYPE), pointer :: ev(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      !call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      !call c_f_pointer(ev_p, ev, [self%na])

      call elpa_eigenvalues_d_ptr_&
              &ELPA_IMPL_SUFFIX&
              & (self, a_p, ev_p, error)
    end subroutine    

#ifdef HAVE_SKEWSYMMETRIC
#ifdef REALCASE
    !>  \brief elpa_skew_eigenvalues_d: class method to solve the real valued skew-symmetric eigenvalue problem
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
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_skew_eigenvalues_a_h_a_&
                    &ELPA_IMPL_SUFFIX&
                    & (self, a, ev, error)
      class(elpa_impl_t)  :: self
#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, *)
#else
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, self%local_ncols)
#endif
      real(kind=C_REAL_DATATYPE)          :: ev(self%na)
#ifdef USE_FORTRAN2008
      integer, optional                   :: error
#else
      integer                             :: error
#endif
      integer                             :: error2
      integer(kind=c_int)                 :: solver
      logical                             :: success_l

      success_l = .false.
      call self%get("solver", solver,error2)
      !call self%set("is_skewsymmetric",1,error2)
      if (error2 .ne. ELPA_OK) then
         print *,"Problem getting solver option. Aborting..."
#ifdef USE_FORTRAN2008
         if (present(error)) then
           error = error2
         endif
#else
         error = error2
#endif
         return
      endif

      if (solver .eq. ELPA_SOLVER_1STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_skew_evp_&
                &MATH_DATATYPE&
                          &_1stage_a_h_a_&
                          &PRECISION&
                          &_impl(self, a, ev)
#endif
        call self%autotune_timer%stop("accumulator")

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_skew_evp_&
                &MATH_DATATYPE&
                                   &_2stage_a_h_a_&
                                   &PRECISION&
                                   &_impl(self, a, ev)
#endif
        call self%autotune_timer%stop("accumulator")

      else ! solver
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
#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR_DURING_COMPUTATION
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in skew_eigenvalues() and you did not check for errors!"
      endif
#else
      if (success_l) then
        error = ELPA_OK
      else
        error = ELPA_ERROR_DURING_COMPUTATION
      endif
#endif
    end subroutine

    !>  \brief elpa_skew_eigenvalues_d: class method to solve the real valued skew-symmetric eigenvalue problem
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
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_skew_eigenvalues_d_ptr_&
                    &ELPA_IMPL_SUFFIX&
                    & (self, a, ev, error)
      use iso_c_binding
      implicit none
      class(elpa_impl_t)  :: self
      type(c_ptr)         :: a, ev
#ifdef USE_FORTRAN2008
      integer, optional                   :: error
#else
      integer                             :: error
#endif
      integer                             :: error2
      integer(kind=c_int)                 :: solver
      logical                             :: success_l

      success_l = .false.
      call self%get("solver", solver,error2)
      !call self%set("is_skewsymmetric",1,error2)
      if (error2 .ne. ELPA_OK) then
         print *,"Problem getting solver option. Aborting..."
#ifdef USE_FORTRAN2008
         if (present(error)) then
           error = error2
         endif
#else
         error = error2
#endif
         return
      endif

      if (solver .eq. ELPA_SOLVER_1STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_skew_evp_&
                &MATH_DATATYPE&
                          &_1stage_d_ptr_&
                          &PRECISION&
                          &_impl(self, a, ev)
#endif
        call self%autotune_timer%stop("accumulator")

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_skew_evp_&
                &MATH_DATATYPE&
                                   &_2stage_d_ptr_&
                                   &PRECISION&
                                   &_impl(self, a, ev)
#endif
        call self%autotune_timer%stop("accumulator")

      else ! solver
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
#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR_DURING_COMPUTATION
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in skew_eigenvalues() and you did not check for errors!"
      endif
#else
      if (success_l) then
        error = ELPA_OK
      else
        error = ELPA_ERROR_DURING_COMPUTATION
      endif
#endif
    end subroutine


#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
    !c> #ifdef HAVE_SKEWSYMMETRIC
    !c> void elpa_skew_eigenvalues_a_h_a_d(elpa_t handle, double *a, double *ev, int *error);
    !c> #endif
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> #ifdef HAVE_SKEWSYMMETRIC
    !c> void elpa_skew_eigenvalues_a_h_a_f(elpa_t handle, float *a, float *ev, int *error);
    !c> #endif
#endif
#endif
    subroutine elpa_skew_eigenvalues_a_h_a_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, ev_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
                    bind(C, name="elpa_skew_eigenvalues_a_h_a_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                    bind(C, name="elpa_skew_eigenvalues_a_h_a_f")
#endif
#endif
      type(c_ptr), intent(in), value :: handle, a_p, ev_p
      integer(kind=c_int), intent(in) :: error

      MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer :: a(:, :)
      real(kind=C_REAL_DATATYPE), pointer :: ev(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])

      call elpa_skew_eigenvalues_a_h_a_&
              &ELPA_IMPL_SUFFIX&
              & (self, a, ev, error)
    end subroutine

#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
    !c> #ifdef HAVE_SKEWSYMMETRIC
    !c> void elpa_skew_eigenvalues_d_ptr_d(elpa_t handle, double *a, double *ev, int *error);
    !c> #endif
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> #ifdef HAVE_SKEWSYMMETRIC
    !c> void elpa_skew_eigenvalues_d_ptr_f(elpa_t handle, float *a, float *ev, int *error);
    !c> #endif
#endif
#endif
    subroutine elpa_skew_eigenvalues_d_ptr_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, ev_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
                    bind(C, name="elpa_skew_eigenvalues_d_ptr_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                    bind(C, name="elpa_skew_eigenvalues_d_ptr_f")
#endif
#endif
      type(c_ptr), intent(in), value :: handle, a_p, ev_p
      integer(kind=c_int), intent(in) :: error

      !MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer :: a(:, :)
      !real(kind=C_REAL_DATATYPE), pointer :: ev(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      !call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      !call c_f_pointer(ev_p, ev, [self%na])

      call elpa_skew_eigenvalues_d_ptr_&
              &ELPA_IMPL_SUFFIX&
              & (self, a_p, ev_p, error)
    end subroutine
#endif /* REALCASE */
#endif /* HAVE_SKEWSYMMETRIC */

