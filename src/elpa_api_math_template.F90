
  !> \brief abstract definition of interface to solve double real eigenvalue problem
  !>
  !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
  !>  blocksize, the number of eigenvectors
  !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
  !>  with the class method "setup"
  !>
  !>  It is possible to change the behaviour of the method by setting tunable parameters with the
  !>  class method "set"
  !> Parameters
  !> \details
  !> \param   self        class(elpa_t), the ELPA object
#if ELPA_IMPL_SUFFIX == d
  !> \param   a           double real matrix a: defines the problem to solve
  !> \param   ev          double real: on output stores the eigenvalues
  !> \param   q           double real matrix q: on output stores the eigenvectors
#endif
#if ELPA_IMPL_SUFFIX == f
  !> \param   a           single real matrix a: defines the problem to solve
  !> \param   ev          single real: on output stores the eigenvalues
  !> \param   q           single real matrix q: on output stores the eigenvectors
#endif  
#if ELPA_IMPL_SUFFIX == dc
  !> \param   a           double complex matrix a: defines the problem to solve
  !> \param   ev          double real: on output stores the eigenvalues
  !> \param   q           double complex matrix q: on output stores the eigenvectors
#endif  
#if ELPA_IMPL_SUFFIX == fc
  !> \param   a           single complex matrix a: defines the problem to solve
  !> \param   ev          single real: on output stores the eigenvalues
  !> \param   q           single complex matrix q: on output stores the eigenvectors
#endif
  !> \result  error       integer, optional : error code, which can be queried with elpa_strerr
  abstract interface
    subroutine elpa_eigenvectors_&
           &ELPA_IMPL_SUFFIX&
           &_i(self, a, ev, q, error)
      use iso_c_binding
      import elpa_t
      implicit none
      class(elpa_t)       :: self

#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, *), q(self%local_nrows,*)
#else
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=C_REAL_DATATYPE) :: ev(self%na)

#ifdef USE_FORTRAN2008
      integer, optional   :: error
#else
      integer             :: error
#endif
    end subroutine
  end interface

  !> \brief abstract definition of interface to solve double real skew-symmetric eigenvalue problem
  !>
  !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
  !>  blocksize, the number of eigenvectors
  !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
  !>  with the class method "setup"
  !>
  !>  It is possible to change the behaviour of the method by setting tunable parameters with the
  !>  class method "set"
  !> Parameters
  !> \details
  !> \param   self        class(elpa_t), the ELPA object
#if ELPA_IMPL_SUFFIX == d
  !> \param   a           double real matrix a: defines the problem to solve
  !> \param   ev          double real: on output stores the eigenvalues
  !> \param   q           double real matrix q: on output stores the eigenvectors
#endif
#if ELPA_IMPL_SUFFIX == f
  !> \param   a           single real matrix a: defines the problem to solve
  !> \param   ev          single real: on output stores the eigenvalues
  !> \param   q           single real matrix q: on output stores the eigenvectors
#endif  
  !> \result  error       integer, optional : error code, which can be queried with elpa_strerr
  abstract interface
    subroutine elpa_skew_eigenvectors_&
           &ELPA_IMPL_SUFFIX&
           &_i(self, a, ev, q, error)
      use iso_c_binding
      import elpa_t
      implicit none
      class(elpa_t)       :: self

#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, *), q(self%local_nrows,*)
#else
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, 2*self%local_ncols)
#endif
      real(kind=C_REAL_DATATYPE) :: ev(self%na)

#ifdef USE_FORTRAN2008
      integer, optional   :: error
#else
      integer             :: error
#endif
    end subroutine
  end interface

  !> \brief abstract definition of interface to solve a eigenvalue problem
  !>
  !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
  !>  blocksize, the number of eigenvectors
  !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
  !>  with the class method "setup"
  !>
  !>  It is possible to change the behaviour of the method by setting tunable parameters with the
  !>  class method "set"
  !> Parameters
  !> \details
  !> \param   self        class(elpa_t), the ELPA object
#if ELPA_IMPL_SUFFIX == d
  !> \param   a           double real matrix a: defines the problem to solve
  !> \param   ev          double real: on output stores the eigenvalues
#endif
#if ELPA_IMPL_SUFFIX == f
  !> \param   a           single real matrix a: defines the problem to solve
  !> \param   ev          single real: on output stores the eigenvalues
#endif
#if ELPA_IMPL_SUFFIX == dc
  !> \param   a           double complex matrix a: defines the problem to solve
  !> \param   ev          double real: on output stores the eigenvalues
#endif  
#if ELPA_IMPL_SUFFIX ==fc
  !> \param   a           single complex matrix a: defines the problem to solve
  !> \param   ev          single real: on output stores the eigenvalues
#endif  
  !> \result  error       integer, optional : error code, which can be queried with elpa_strerr
  abstract interface
    subroutine elpa_eigenvalues_&
        &ELPA_IMPL_SUFFIX&
        &_i(self, a, ev, error)
      use iso_c_binding
      import elpa_t
      implicit none
      class(elpa_t)       :: self
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
    end subroutine
  end interface       


  !> \brief abstract definition of interface to solve a skew-symmetric eigenvalue problem
  !>
  !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
  !>  blocksize, the number of eigenvectors
  !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
  !>  with the class method "setup"
  !>
  !>  It is possible to change the behaviour of the method by setting tunable parameters with the
  !>  class method "set"
  !> Parameters
  !> \details
  !> \param   self        class(elpa_t), the ELPA object
#if ELPA_IMPL_SUFFIX == d
  !> \param   a           double real matrix a: defines the problem to solve
  !> \param   ev          double real: on output stores the eigenvalues
#endif
#if ELPA_IMPL_SUFFIX == f
  !> \param   a           single real matrix a: defines the problem to solve
  !> \param   ev          single real: on output stores the eigenvalues
#endif
  !> \result  error       integer, optional : error code, which can be queried with elpa_strerr
  abstract interface
    subroutine elpa_skew_eigenvalues_&
        &ELPA_IMPL_SUFFIX&
        &_i(self, a, ev, error)
      use iso_c_binding
      import elpa_t
      implicit none
      class(elpa_t)       :: self
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
    end subroutine
  end interface       


  !> \brief abstract definition of interface to solve a generalized eigenvalue problem
  !>
  !>  The dimensions of the matrix a and b (locally ditributed and global), the block-cyclic distribution
  !>  blocksize, the number of eigenvectors
  !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
  !>  with the class method "setup"
  !>
  !>  It is possible to change the behaviour of the method by setting tunable parameters with the
  !>  class method "set"
  !> Parameters
  !> \details
  !> \param   self        class(elpa_t), the ELPA object
#if ELPA_IMPL_SUFFIX == d   
  !> \param   a           double real matrix a: defines the problem to solve
  !> \param   b           double real matrix b: defines the problem to solve
  !> \param   ev          double real: on output stores the eigenvalues
  !> \param   q           double real matrix q: on output stores the eigenvalues
#endif
#if ELPA_IMPL_SUFFIX == f  
  !> \param   a           single real matrix a: defines the problem to solve
  !> \param   b           single real matrix b: defines the problem to solve
  !> \param   ev          single real: on output stores the eigenvalues
  !> \param   q           single real matrix q: on output stores the eigenvalues
#endif
#if ELPA_IMPL_SUFFIX == dc  
  !> \param   a           double complex matrix a: defines the problem to solve
  !> \param   b           double complex matrix b: defines the problem to solve
  !> \param   ev          double real: on output stores the eigenvalues
  !> \param   q           double complex matrix q: on output stores the eigenvalues
#endif
#if ELPA_IMPL_SUFFIX == fc
  !> \param   a           single complex matrix a: defines the problem to solve
  !> \param   b           single complex matrix b: defines the problem to solve
  !> \param   ev          single real: on output stores the eigenvalues
  !> \param   q           single complex matrix q: on output stores the eigenvalues
#endif

  !> \param   is_already_decomposed   logical, input: is it repeated call with the same b (decomposed in the fist call)?
  !> \result  error       integer, optional : error code, which can be queried with elpa_strerr
  abstract interface
    subroutine elpa_generalized_eigenvectors_&
           &ELPA_IMPL_SUFFIX&
           &_i(self, a, b, ev, q, is_already_decomposed, error)
      use iso_c_binding
      use elpa_constants
      import elpa_t
      implicit none
      class(elpa_t)       :: self
#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, *), b(self%local_nrows, *), q(self%local_nrows, *)
#else
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, self%local_ncols), b(self%local_nrows, self%local_ncols), &
                             q(self%local_nrows, self%local_ncols)
#endif
      real(kind=C_REAL_DATATYPE) :: ev(self%na)

      logical             :: is_already_decomposed
      integer, optional   :: error
    end subroutine
  end interface

  !> \brief abstract definition of interface to solve a generalized eigenvalue problem
  !>
  !>  The dimensions of the matrix a and b (locally ditributed and global), the block-cyclic distribution
  !>  blocksize, the number of eigenvectors
  !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
  !>  with the class method "setup"
  !>
  !>  It is possible to change the behaviour of the method by setting tunable parameters with the
  !>  class method "set"
  !> Parameters
  !> \details
  !> \param   self        class(elpa_t), the ELPA object
#if ELPA_IMPL_SUFFIX == d   
  !> \param   a           double real matrix a: defines the problem to solve
  !> \param   b           double real matrix b: defines the problem to solve
  !> \param   ev          double real: on output stores the eigenvalues
#endif
#if ELPA_IMPL_SUFFIX == f  
  !> \param   a           single real matrix a: defines the problem to solve
  !> \param   b           single real matrix b: defines the problem to solve
  !> \param   ev          single real: on output stores the eigenvalues
#endif
#if ELPA_IMPL_SUFFIX == dc  
  !> \param   a           double complex matrix a: defines the problem to solve
  !> \param   b           double complex matrix b: defines the problem to solve
  !> \param   ev          double real: on output stores the eigenvalues
#endif
#if ELPA_IMPL_SUFFIX == fc
  !> \param   a           single complex matrix a: defines the problem to solve
  !> \param   b           single complex matrix b: defines the problem to solve
  !> \param   ev          single real: on output stores the eigenvalues
#endif

  !> \param   is_already_decomposed   logical, input: is it repeated call with the same b (decomposed in the fist call)?
  !> \result  error       integer, optional : error code, which can be queried with elpa_strerr
  abstract interface
    subroutine elpa_generalized_eigenvalues_&
           &ELPA_IMPL_SUFFIX&
           &_i(self, a, b, ev, is_already_decomposed, error)
      use iso_c_binding
      use elpa_constants
      import elpa_t
      implicit none
      class(elpa_t)       :: self
#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, *), b(self%local_nrows, *)
#else
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows, self%local_ncols), b(self%local_nrows, self%local_ncols)
#endif
      real(kind=C_REAL_DATATYPE) :: ev(self%na)

      logical             :: is_already_decomposed
      integer, optional   :: error
    end subroutine
  end interface


  !> \brief abstract definition of interface to compute C : = A**T * B
  !>         where   A is a square matrix (self%a,self%na) which is optionally upper or lower triangular
  !>                 B is a (self%na,ncb) matrix
  !>                 C is a (self%na,ncb) matrix where optionally only the upper or lower
  !>                   triangle may be computed
  !>
  !> the MPI commicators are already known to the type. Thus the class method "setup" must be called
  !> BEFORE this method is used
  !> \details
  !>
  !> \param   self                class(elpa_t), the ELPA object
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
  !> \param self%local_nrows      number of rows of local (sub) matrix a, set with method set("local_nrows,value")
  !> \param self%local_ncols      number of columns of local (sub) matrix a, set with method set("local_ncols,value")
  !> \param b                     matrix b
  !> \param nrows_b               number of rows of local (sub) matrix b
  !> \param ncols_b               number of columns of local (sub) matrix b
  !> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
  !> \param c                     matrix c
  !> \param nrows_c               number of rows of local (sub) matrix c
  !> \param ncols_c               number of columns of local (sub) matrix c
  !> \param error                 optional argument, error code which can be queried with elpa_strerr
  abstract interface
    subroutine elpa_hermitian_multiply_&
        &ELPA_IMPL_SUFFIX&
        &_i (self,uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, &
                                          c, nrows_c, ncols_c, error)
      use iso_c_binding
      import elpa_t
      implicit none
      class(elpa_t)                   :: self
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
    end subroutine
  end interface


  !> \brief abstract definition of interface to do a cholesky decomposition of a matrix
  !>
  !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
  !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
  !>  with the class method "setup"
  !>
  !> Parameters
  !> \param   self        class(elpa_t), the ELPA object
#if ELPA_IMPL_SUFFIX == d
  !> \param   a           double real matrix: the matrix to be decomposed
#endif
#if ELPA_IMPL_SUFFIX == f
  !> \param   a           single real matrix: the matrix to be decomposed
#endif
#if ELPA_IMPL_SUFFIX == dc
  !> \param   a           double complex matrix: the matrix to be decomposed
#endif
#if ELPA_IMPL_SUFFIX == fc
  !> \param   a           single complex matrix: the matrix to be decomposed
#endif
  !> \param   error       integer, optional : error code, which can be queried with elpa_strerr
  abstract interface
    subroutine elpa_cholesky_&
          &ELPA_IMPL_SUFFIX&
          &_i (self, a, error)
      use iso_c_binding
      import elpa_t
      implicit none
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows,*)
#else
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows,self%local_ncols)
#endif

#ifdef USE_FORTRAN2008
      integer, optional               :: error
#else
      integer                         :: error
#endif
    end subroutine
  end interface



  !> \brief abstract definition of interface to invert a triangular matrix
  !>
  !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
  !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
  !>  with the class method "setup"
  !>
  !> Parameters
  !> \param   self        class(elpa_t), the ELPA object
#if ELPA_IMPL_SUFFIX == d
  !> \param   a           double real matrix: the matrix to be inverted
#endif
#if ELPA_IMPL_SUFFIX == f
  !> \param   a           single real matrix: the matrix to be inverted
#endif
#if ELPA_IMPL_SUFFIX == dc
  !> \param   a           double complex matrix: the matrix to be inverted
#endif
#if ELPA_IMPL_SUFFIX == fc
  !> \param   a           single complex matrix: the matrix to be inverted
#endif

  !> \param   error       integer, optional : error code, which can be queried with elpa_strerr
  abstract interface
    subroutine elpa_invert_trm_&
        &ELPA_IMPL_SUFFIX&
        &_i (self, a, error)
      use iso_c_binding
      import elpa_t
      implicit none
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows,*)
#else
      MATH_DATATYPE(kind=C_DATATYPE_KIND) :: a(self%local_nrows,self%local_ncols)
#endif

#ifdef USE_FORTRAN2008
      integer, optional               :: error
#else
      integer                         :: error
#endif
    end subroutine
  end interface



  !> \brief abstract definition of interface to solve the eigenvalue problem for a valued tridiangular matrix
  !>
  !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
  !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
  !>  with the class method "setup"
  !>
  !> Parameters
  !> \param   self        class(elpa_t), the ELPA object
#if ELPA_IMPL_SUFFIX == d
  !> \param   d           double real 1d array: the diagonal elements of a matrix defined in setup, on output the eigenvalues
  !>                      in ascending order
  !> \param   e           double real 1d array: the subdiagonal elements of a matrix defined in setup
  !> \param   q           double real matrix: on output contains the eigenvectors
#endif
#if ELPA_IMPL_SUFFIX == f
  !> \param   d           single real 1d array: the diagonal elements of a matrix defined in setup, on output the eigenvalues
  !>                      in ascending order
  !> \param   e           single real 1d array: the subdiagonal elements of a matrix defined in setup
  !> \param   q           single real matrix: on output contains the eigenvectors
#endif
  !> \param   error       integer, optional : error code, which can be queried with elpa_strerr
  abstract interface
    subroutine elpa_solve_tridiagonal_&
          &ELPA_IMPL_SUFFIX&
          &_i (self, d, e, q, error)
      use iso_c_binding
      import elpa_t
      implicit none
      class(elpa_t)                   :: self
      real(kind=C_REAL_DATATYPE)        :: d(self%na), e(self%na)
#ifdef USE_ASSUMED_SIZE
      real(kind=C_REAL_DATATYPE)        :: q(self%local_nrows,*)
#else
      real(kind=C_REAL_DATATYPE)        :: q(self%local_nrows,self%local_ncols)
#endif

#ifdef USE_FORTRAN2008
      integer, optional               :: error
#else
      integer                         :: error
#endif
    end subroutine
  end interface

