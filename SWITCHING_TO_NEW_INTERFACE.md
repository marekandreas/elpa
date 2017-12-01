## Documentation how to switch from the legay API to the new API of the *ELPA* library ##


### Using *ELPA* from a Fortran code ###

Up to now, if you have been using the (legacy API of the) *ELPA* library you had to do the following
steps: (we assume that MPI and a distributed matrix in block-cyclic scalapack layout is already created in
the user application)

1. including the *ELPA* modules

   use elpa1
   use elpa2   ! this step was only needed if you wanted to use the ELPA 2stage solver

2. call the "elpa_get_communicators" routine, in order to obtain the row/column MPI communicators needed by *ELPA*

   mpierr = elpa_get_communicators(mpi_comm_world, my_prow, my_pcol, &
                                   mpi_comm_rows, mpi_comm_cols)

3. do the desired task with the *ELPA* library, which could be
  a) elpa_solve_[real|complex]_1stage_[double|single]     ! solve EV problem with ELPA 1stage solver
  b) elpa_solve_[real|complex]_2stage_[double|single]     ! solve EV problem with ELPA 2stage solver
  c) elpa_solve_tridi_[double|single]                     ! solve a the problem for a tri-diagonal matrix
  d) elpa_cholesky_[real|complex]_[double|single]         ! Cholesky decomposition
  e) elpa_invert_trm_[real|complex]_[double|single]       ! invert triangular matrix
  f) elpa_mult_at_b_real_[double|single]                  ! multiply a**T * b
  g) elpa_mult_ah_b_complex_[double|single]               ! multiply a**H * b

For each of the function calls you had to set some parameters (see man pages) to control the execution like
useGPU=[.false.|.true.], choice of ELPA 2stage kernel .... New parameters were likely added with a new release of
the *ELPA* library to reflect the growing functionality.


The new interface of *ELPA* is more generic, which ,however, requires ONCE the adaption of the user code if the new
interface should be used.

This are the new steps to do (again it is assumed that MPI and a distributed matrix in block-cyclic scalapack layout is already created in
the user application):

1. include the correct *ELPA* module and define a name for the ELPA instance

   use elpa   ! this is the only module needed for ELPA

   class(elpa_t), pointer :: e   ! name the ELPA instance "e"

2. initialize ELPA and create the instance

   if (elpa_init(20170403) /= ELPA_OK) then
     error stop "ELPA API version not supported"
   endif

   e => elpa_allocate()

!>   call e%set("solver", ELPA_SOLVER_2STAGE, success)
!> \endcode
!>   ... set and get all other options that are desired
!> \code{.f90}
!>
!>   ! use method solve to solve the eigenvalue problem
!>   ! other possible methods are desribed in \ref elpa_api::elpa_t derived type
!>   call e%eigenvectors(a, ev, z, success)
!>
!>   ! cleanup
!>   call elpa_deallocate(e)
!>
!>   call elpa_uninit()



3. set the parameters which describe the matrix setup and the MPI

   call e%set("na", na,success)                          ! size of matrix
   call e%set("local_nrows", na_rows,success)            ! MPI process local rows of the distributed matrixdo the
                                                         ! desired task with the *ELPA* library, which could be

   call e%set("local_ncols", na_cols,success)            ! MPI process local cols of the distributed matrix
   call e%set("nblk", nblk, success)                     ! size of block-cylic distribution

   call e%set("mpi_comm_parent", MPI_COMM_WORLD,succes)  ! global communicator for all processes which have parts of
                                                         ! the distributed matrix
   call e%set("process_row", my_prow, success)           ! row coordinate of MPI task
   call e%set("process_col", my_pcol, success)           ! column coordinate of MPI task

4. setup the ELPA instance

   success = e%setup()

5. set/get any possible option (see man pages)

   call e%get("qr", qr, success)                        ! querry whether QR-decomposition is set
   print *, "qr =", qr
   if (success .ne. ELPA_OK) stop

   call e%set("solver", ELPA_SOLVER_2STAGE, success)    ! set solver to 2stage
   if (success .ne. ELPA_OK) stop

   call e%set("real_kernel", ELPA_2STAGE_REAL_GENERIC, success) ! set kernel of ELPA 2stage solver for
                                                                !real case to the generic kernel

   ....

   At the moment, the following configurable runtime options are supported:

   "solver"       can be one of {ELPA_SOLVER_1STAGE | ELPA_SOLVER_2STAGE }
   "real_kernel"  can be one of { [real,complex]_generic | [real,complex]_generic_simple |
                                  complex_sse_block1 | [real,complex]_sse_block2 |
				  real_sse_block4 | real_sse_block6 | [real,complex]_sse_assembly |
				  complex_avx_block1 | [real,complex]_avx_block2 |
				  real_avx_block4 | real_avx_block6 |
  				  complex_avx2_block1 | [real,complex]_avx2_block2 |
				  real_avx2_block4 | real_avx2_block6 |
				  complex_avx512_block1 | [real,complex]_avx512_block2 |
				  real_avx512_block4 | real_avx512_block6 |
				  [real,complex]_bgp | [real,complex]_bgq }
		 depending on your system and the installed kernels. This can be queried with the
		 helper binary "elpa2_print_kernels"

   "qr"       can be one of { 0 | 1 }, depending whether you want to use QR decomposition in the REAL
              ELPA_SOLVER_2STAGE
   "gpu"      can be one of { 0 | 1 }, depending whether you want to use GPU acceleration (assuming your
              ELPA installation has ben build with GPU support

   "timings"  can be one of { 0 | 1 }, depending whether you want to measure times within the library calls

   "debug"    can be one of { 0 | 1 }, will give more information case of an error if set to 1


6. do the desired task with the *ELPA* library, which could be
   a) e%eigenvectors                  ! solve EV problem with solver as set by "set" method; computes eigenvalues AND eigenvectors
                                      ! (replaces a) and b) from legacy API)
   b) e%eigenvalues                   ! solve EV problem with solver as set by "set" method; computes eigenvalues only
   c) e%choleksy                      ! do a cholesky decomposition (replaces  d) from legacy API)
   d) e%invert_triangular             ! invert triangular matrix (replaces  e) from legacy API)
   e) e%hermitian_multiply            ! multiply a**T *b or a**H *b (replaces f) and g) from legacy API)
   f) e%solve_tridiagonal             ! solves the eigenvalue problem for a tridiagonal matrix (replaces c) from legacy
                                      ! API)

7. when not needed anymore, destroy the instance
   call elpa_deallocate()

8. when *ELPA* is not needed anymore, unitialize the *ELPA* library
   call elpa_uninit()


### Online and local documentation ###

Local documentation (via man pages) should be available (if *ELPA* has been installed with the documentation):

For example "man elpa2_print_kernels" should provide the documentation for the *ELPA* program which prints all
the available kernels.

Also a [online doxygen documentation] (http://elpa.mpcdf.mpg.de/html/Documentation/ELPA-2017.11.001.rc1/html/index.html)
for each *ELPA* release is available.


