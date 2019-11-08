
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

    subroutine elpa_eigenvectors_&
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


      call self%get("solver", solver,error2)
      if (error2 .ne. ELPA_OK) then
        print *,"Problem setting option. Aborting..."
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
                &_1stage_&
                &PRECISION&
                &_impl(self, a, ev, q)
#endif
        call self%autotune_timer%stop("accumulator")

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                &_2stage_&
                &PRECISION&
                &_impl(self, a, ev, q)
#endif
        call self%autotune_timer%stop("accumulator")

      else
        print *,"unknown solver"
        stop
      endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in eigenvectors() and you did not check for errors!"
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
    !c> void elpa_eigenvectors_d(elpa_t handle, double *a, double *ev, double *q, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> void elpa_eigenvectors_f(elpa_t handle, float *a, float *ev, float *q, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void elpa_eigenvectors_dc(elpa_t handle, double complex *a, double *ev, double complex *q, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    !c> void elpa_eigenvectors_fc(elpa_t handle, float complex *a, float *ev, float complex *q, int *error);
#endif
#endif
    subroutine elpa_eigenvectors_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, ev_p, q_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL  
                    bind(C, name="elpa_eigenvectors_d")
#endif
#ifdef SINGLE_PRECISION_REAL  
                    bind(C, name="elpa_eigenvectors_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
                    bind(C, name="elpa_eigenvectors_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX
                    bind(C, name="elpa_eigenvectors_fc")
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

      call elpa_eigenvectors_&
              &ELPA_IMPL_SUFFIX&
              & (self, a, ev, q, error)
    end subroutine    

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

    subroutine elpa_skew_eigenvectors_&
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


      call self%get("solver", solver,error2)
      call self%set("is_skewsymmetric",1)
      if (error2 .ne. ELPA_OK) then
        print *,"Problem setting option. Aborting..."
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
                &_1stage_&
                &PRECISION&
                &_impl(self, a, ev, q)
#endif
        call self%autotune_timer%stop("accumulator")

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                &_2stage_&
                &PRECISION&
                &_impl(self, a, ev, q)
#endif
        call self%autotune_timer%stop("accumulator")

      else
        print *,"unknown solver"
        stop
      endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in skew_eigenvectors() and you did not check for errors!"
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
    !c> void elpa_skew_eigenvectors_d(elpa_t handle, double *a, double *ev, double *q, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> void elpa_skew_eigenvectors_f(elpa_t handle, float *a, float *ev, float *q, int *error);
#endif
#endif
    subroutine elpa_skew_eigenvectors_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, ev_p, q_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
                    bind(C, name="elpa_skew_eigenvectors_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                    bind(C, name="elpa_skew_eigenvectors_f")
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

      call elpa_skew_eigenvectors_&
              &ELPA_IMPL_SUFFIX&
              & (self, a, ev, q, error)
    end subroutine
#endif /* REALCASE */

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
    subroutine elpa_eigenvalues_&
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


      call self%get("solver", solver,error2)
      if (error2 .ne. ELPA_OK) then
         print *,"Problem getting option. Aborting..."
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
                          &_1stage_&
                          &PRECISION&
                          &_impl(self, a, ev)
#endif
        call self%autotune_timer%stop("accumulator")

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                                   &_2stage_&
                                   &PRECISION&
                                   &_impl(self, a, ev)
#endif
        call self%autotune_timer%stop("accumulator")

      else
        print *,"unknown solver"
        stop
      endif
#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in eigenvalues() and you did not check for errors!"
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
    !c> void elpa_eigenvalues_d(elpa_t handle, double *a, double *ev, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> void elpa_eigenvalues_f(elpa_t handle, float *a, float *ev, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void elpa_eigenvalues_dc(elpa_t handle, double complex *a, double *ev, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    !c> void elpa_eigenvalues_fc(elpa_t handle, float complex *a, float *ev, int *error);
#endif
#endif
    subroutine elpa_eigenvalues_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, ev_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL    
                    bind(C, name="elpa_eigenvalues_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                    bind(C, name="elpa_eigenvalues_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX 
                    bind(C, name="elpa_eigenvalues_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX
                    bind(C, name="elpa_eigenvalues_fc")
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

      call elpa_eigenvalues_&
              &ELPA_IMPL_SUFFIX&
              & (self, a, ev, error)
    end subroutine    

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
    subroutine elpa_skew_eigenvalues_&
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

      call self%get("solver", solver,error2)
      call self%set("is_skewsymmetric",1)
      if (error2 .ne. ELPA_OK) then
         print *,"Problem getting option. Aborting..."
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
                          &_1stage_&
                          &PRECISION&
                          &_impl(self, a, ev)
#endif
        call self%autotune_timer%stop("accumulator")

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
        call self%autotune_timer%start("accumulator")
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                                   &_2stage_&
                                   &PRECISION&
                                   &_impl(self, a, ev)
#endif
        call self%autotune_timer%stop("accumulator")

      else
        print *,"unknown solver"
        stop
      endif
#ifdef USE_FORTRAN2008
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in skew_eigenvalues() and you did not check for errors!"
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
    !c> void elpa_skew_eigenvalues_d(elpa_t handle, double *a, double *ev, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> void elpa_skew_eigenvalues_f(elpa_t handle, float *a, float *ev, int *error);
#endif
#endif
    subroutine elpa_skew_eigenvalues_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, ev_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
                    bind(C, name="elpa_skew_eigenvalues_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                    bind(C, name="elpa_skew_eigenvalues_f")
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

      call elpa_skew_eigenvalues_&
              &ELPA_IMPL_SUFFIX&
              & (self, a, ev, error)
    end subroutine
#endif /* REALCASE */

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
      use iso_c_binding
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
                &_1stage_&
                &PRECISION&
                &_impl(self, a, ev, q)
#endif
      else if (solver .eq. ELPA_SOLVER_2STAGE) then
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                &_2stage_&
                &PRECISION&
                &_impl(self, a, ev, q)
#endif
      else
        print *,"unknown solver"
        stop
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
    !c> void elpa_generalized_eigenvectors_dc(elpa_t handle, double complex *a, double complex *b, double *ev, double complex *q,
    !c> int is_already_decomposed, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    !c> void elpa_generalized_eigenvectors_fc(elpa_t handle, float complex *a, float complex *b, float *ev, float complex *q,
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
      use iso_c_binding
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
                &_1stage_&
                &PRECISION&
                &_impl(self, a, ev)
#endif
      else if (solver .eq. ELPA_SOLVER_2STAGE) then
#if defined(INCLUDE_ROUTINES)
        success_l = elpa_solve_evp_&
                &MATH_DATATYPE&
                &_2stage_&
                &PRECISION&
                &_impl(self, a, ev)
#endif
      else
        print *,"unknown solver"
        stop
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
    !c> void elpa_generalized_eigenvalues_dc(elpa_t handle, double complex *a, double complex *b, double *ev,
    !c> int is_already_decomposed, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    !c> void elpa_generalized_eigenvalues_fc(elpa_t handle, float complex *a, float complex *b, float *ev,
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


    !> \brief  elpa_hermitian_multiply_d: class method to perform C : = A**T * B
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
    subroutine elpa_hermitian_multiply_&
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

#if defined(INCLUDE_ROUTINES)
#ifdef REALCASE
      success_l = elpa_mult_at_b_&
#endif
#ifdef COMPLEXCASE
      success_l = elpa_mult_ah_b_&
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

#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL    
    !c> void elpa_hermitian_multiply_d(elpa_t handle, char uplo_a, char uplo_c, int ncb, double *a, double *b, int nrows_b, int ncols_b, double *c, int nrows_c, int ncols_c, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> void elpa_hermitian_multiply_df(elpa_t handle, char uplo_a, char uplo_c, int ncb, float *a, float *b, int nrows_b, int ncols_b, float *c, int nrows_c, int ncols_c, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX    
    !c> void elpa_hermitian_multiply_dc(elpa_t handle, char uplo_a, char uplo_c, int ncb, double complex *a, double complex *b, int nrows_b, int ncols_b, double complex *c, int nrows_c, int ncols_c, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    !c> void elpa_hermitian_multiply_fc(elpa_t handle, char uplo_a, char uplo_c, int ncb, float complex *a, float complex *b, int nrows_b, int ncols_b, float complex *c, int nrows_c, int ncols_c, int *error);
#endif
#endif
    subroutine elpa_hermitian_multiply_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, uplo_a, uplo_c, ncb, a_p, b, nrows_b, &
                                           ncols_b, c, nrows_c, ncols_c, error)          &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL 
                                           bind(C, name="elpa_hermitian_multiply_d")
#endif
#ifdef SINGLE_PRECISION_REAL 
                                           bind(C, name="elpa_hermitian_multiply_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
                                           bind(C, name="elpa_hermitian_multiply_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX 
                                           bind(C, name="elpa_hermitian_multiply_fc")
#endif
#endif

      type(c_ptr), intent(in), value            :: handle, a_p
      character(1,C_CHAR), value                :: uplo_a, uplo_c
      integer(kind=c_int), value                :: ncb, nrows_b, ncols_b, nrows_c, ncols_c
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(in) :: error
#else
      integer(kind=c_int), intent(in)           :: error
#endif
      MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer              :: a(:, :)
#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=C_DATATYPE_KIND)                       :: b(nrows_b,*), c(nrows_c,*)
#else
      MATH_DATATYPE(kind=C_DATATYPE_KIND)                       :: b(nrows_b,ncols_b), c(nrows_c,ncols_c)
#endif
      type(elpa_impl_t), pointer                :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_hermitian_multiply_&
              &ELPA_IMPL_SUFFIX&
              & (self, uplo_a, uplo_c, ncb, a, b, nrows_b, &
                                     ncols_b, c, nrows_c, ncols_c, error)
    end subroutine


    !>  \brief elpa_choleksy_d: class method to do a cholesky factorization
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
    subroutine elpa_cholesky_&
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

#if defined(INCLUDE_ROUTINES)
      success_l = elpa_cholesky_&
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
    !c> void elpa_cholesky_d(elpa_t handle, double *a, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL
    !c> void elpa_cholesky_f(elpa_t handle, float *a, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void elpa_cholesky_dc(elpa_t handle, double complex *a, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
    !c> void elpa_cholesky_fc(elpa_t handle, float complex *a, int *error);
#endif
#endif
    subroutine elpa_choleksy_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
                    bind(C, name="elpa_cholesky_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                    bind(C, name="elpa_cholesky_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
                    bind(C, name="elpa_cholesky_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX
                    bind(C, name="elpa_cholesky_fc")
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

      call elpa_cholesky_&
              &ELPA_IMPL_SUFFIX&
              & (self, a, error)
    end subroutine      


    !>  \brief elpa_invert_trm_d: class method to invert a triangular
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
    subroutine elpa_invert_trm_&
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

#if defined(INCLUDE_ROUTINES)
      success_l = elpa_invert_trm_&
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
    !c> void elpa_invert_trm_d(elpa_t handle, double *a, int *error);
#endif
#ifdef SINGLE_PRECISION_REAL      
    !c> void elpa_invert_trm_f(elpa_t handle, float *a, int *error);
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX     
    !c> void elpa_invert_trm_dc(elpa_t handle, double complex *a, int *error);
#endif
#ifdef SINGLE_PRECISION_COMPLEX      
    !c> void elpa_invert_trm_fc(elpa_t handle, float complex *a, int *error);
#endif
#endif
    subroutine elpa_invert_trm_&
                    &ELPA_IMPL_SUFFIX&
                    &_c(handle, a_p, error) &
#ifdef REALCASE
#ifdef DOUBLE_PRECISION_REAL
                    bind(C, name="elpa_invert_trm_d")
#endif
#ifdef SINGLE_PRECISION_REAL
                    bind(C, name="elpa_invert_trm_f")
#endif
#endif
#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION_COMPLEX
                    bind(C, name="elpa_invert_trm_dc")
#endif
#ifdef SINGLE_PRECISION_COMPLEX
                    bind(C, name="elpa_invert_trm_fc")
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

      call elpa_invert_trm_&
              &ELPA_IMPL_SUFFIX&
              & (self, a, error)
    end subroutine


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
     print *,"ELPA is not compiled with single-precision support"
     stop
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

