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

   use elpa_type   ! this is the only module needed for ELPA

   type(elpa_t) :: e   ! name the ELPA instance "e"

2. initialize ELPA and create the instance

   if (elpa_init(20170403) /= ELPA_OK) then           
     error stop "ELPA API version not supported"
   endif

   e = elpa_allocate()

3. set the parameters which describe the matrix setup and the MPI

   e%set("na", na)                     ! size of matrix
   e%set("local_nrows", na_rows)       ! MPI process local rows of the distributed matrixdo the desired task with the *ELPA* library, which could be

   e%set("local_ncols", na_cols)       ! MPI process local cols of the distributed matrix
   e%set("nblk", nblk)                 ! size of block-cylic distribution

   e%set("mpi_comm_parent", MPI_COMM_WORLD)  ! global communicator for all processes which have parts of the distributed matrix
   e%set("process_row", my_prow)             ! row coordinate of MPI task
   e%set("process_col", my_pcol)             ! column coordinate of MPI task

4. setup the ELPA instance

   call e%setup()

5. set/get any possible option (see man pages)

   qr = e%get("qr", success)        ! querry whether QR-decomposition is set 
   print *, "qr =", qr
   assert(success == ELPA_OK)

   call e%set("solver", ELPA_SOLVER_2STAGE, success)    ! set solver to 2stage
   assert(success == ELPA_OK)

   call e%set("real_kernel", ELPA_2STAGE_REAL_GENERIC, success) ! set kernel of ELPA 2stage solver for real case to the generic kernel
   assert(success == ELPA_OK)

   ....

6. do the desired task with the *ELPA* library, which could be 
   a) e%solve                         ! solve EV problem with solver as set by "set" method; (replaces a) and b) from legacy API)
   b) e%solve_tridi                   ! solve problem with tridiagonal matrix  (replacement for c) from legacy API)
   c) e%choleksy                      ! do a cholesky decomposition (replaces  d) from legacy API)
   d) e%invert_tridiagonal            ! invert triangular matrix (replaces  e) from legacy API)
   e) e%hermitian_multiply            ! multiply a**T *b or a**H *b (replaces f) and g) from legacy API)

7. when not needed anymore, destroy the instance
   call e%destroy()

8. when *ELPA* is not needed anymore, unitialize the *ELPA* library
   call elpa_uninit()


### Online and local documentation ###

Local documentation (via man pages) should be available (if *ELPA* has been installed with the documentation):

For example "man elpa2_print_kernels" should provide the documentation for the *ELPA* program which prints all
the available kernels.

Also a [online doxygen documentation] (http://elpa.mpcdf.mpg.de/html/Documentation/ELPA-2017.05.001/html/index.html)
for each *ELPA* release is available.


## API of the *ELPA* library ##

With release 2017.05.001 of the *ELPA* library the interface has been rewritten substantially, in order to have a more
generic interface and avoid future interface changes.

For compatibility reasons the interface defined in the previous release 2016.11.001 is also still available
IF AND ONLY IF *ELPA* has been build with support of this legacy interface. If you want to use the legacy
interface, please look to section "B) Using the legacy API of the *ELPA* library.

The legacy API defines all the functionallity as has been defined in *ELPA* release 2016.11.011. Note, however,
that all future features of *ELPA* will only be accessible via the new API defined in release 2017.05.001.

## A) Using the final API definition of the *ELPA* library ##

Using *ELPA* with the latest API is done in the following steps

- include elpa headers (C-Case) or use the Fortran module
- define a instance of the elpa type
- call elpa_init
- call elpa_allocate to allocate an instance of *ELPA*
- use ELPA-type function "set" to set matrix parameters
- call ELPA-type function "setup"
- set or get all possible ELPA options with ELPA-type functions get/set
- call ELPA-type function solve or others
- call ELPA-type function destroy
- call elpa_uninit



## B) Using the legacy API of the *ELPA* library ##

The following description describes the usage of the *ELPA* library with the legacy interface.

### General concept of the *ELPA* library ###

The *ELPA* library consists of two main parts:
- *ELPA 1stage* solver
- *ELPA 2stage* solver

Both variants of the *ELPA* solvers are available for real or complex singe and double precision valued matrices.

Thus *ELPA* provides the following user functions (see man pages or [online] (http://elpa.mpcdf.mpg.de/html/Documentation/ELPA-2016.11.001/html/index.html) for details):

- elpa_get_communicators                        : set the row / column communicators for *ELPA*
- elpa_solve_evp_complex_1stage_{single|double} : solve a {single|double} precision complex eigenvalue proplem with the *ELPA 1stage* solver
- elpa_solve_evp_real_1stage_{single|double}    : solve a {single|double} precision real eigenvalue proplem with the *ELPA 1stage* solver
- elpa_solve_evp_complex_2stage_{single|double} : solve a {single|double} precision complex eigenvalue proplem with the *ELPA 2stage* solver
- elpa_solve_evp_real_2stage_{single|double}    : solve a {single|double} precision real eigenvalue proplem with the *ELPA 2stage* solver
- elpa_solve_evp_real_{single|double}           : driver for the {single|double} precision real *ELPA 1stage* or *ELPA 2stage* solver
- elpa_solve_evp_complex_{single|double}        : driver for the {single|double} precision complex *ELPA 1stage* or *ELPA 2stage* solver



Furthermore *ELPA* provides the utility binary "elpa2_print_available_kernels": it tells the user
which *ELPA 2stage* compute kernels have been installed and which default kernels are set

If you want to solve an eigenvalue problem with *ELPA*, you have to decide whether you
want to use *ELPA 1stage* or *ELPA 2stage* solver. Normally, *ELPA 2stage* is the better
choice since it is faster, but there a matrix dimensions where *ELPA 1stage* is supperior.

Independent of the choice of the solver, the concept of calling *ELPA* is always the same:

#### MPI version of *ELPA* ####

In this case, *ELPA* relies on a BLACS distributed matrix.
To solve a Eigenvalue problem of this matrix with *ELPA*, one has

1. to include the *ELPA* header (C case) or module (Fortran)
2. to create row and column MPI communicators for ELPA (with "elpa_get_communicators")
3. to call to the *ELPA driver* or directly call *ELPA 1stage* or *ELPA 2stage* for the matrix.

Here is a very simple MPI code snippet for using *ELPA 1stage*: For the definition of all variables
please have a look at the man pages and/or the online documentation (see above). A full version
of a simple example program can be found in ./test_project/src.


   ! All ELPA routines need MPI communicators for communicating within
   ! rows or columns of processes, these are set in get_elpa_communicators

   success = elpa_get_communicators(mpi_comm_world, my_prow, my_pcol, &
                                    mpi_comm_rows, mpi_comm_cols)

   if (myid==0) then
     print '(a)','| Past split communicator setup for rows and columns.'
   end if

   ! Determine the necessary size of the distributed matrices,
   ! we use the Scalapack tools routine NUMROC for that.

   na_rows = numroc(na, nblk, my_prow, 0, np_rows)
   na_cols = numroc(na, nblk, my_pcol, 0, np_cols)

   !-------------------------------------------------------------------------------
   ! Calculate eigenvalues/eigenvectors

   if (myid==0) then
     print '(a)','| Entering one-step ELPA solver ... '
     print *
   end if

   success = elpa_solve_evp_real_1stage_{single|double} (na, nev, a, na_rows, ev, z, na_rows, nblk, &
                                   matrixCols, mpi_comm_rows, mpi_comm_cols)

   if (myid==0) then
     print '(a)','| One-step ELPA solver complete.'
     print *
   end if


#### Shared-memory version of *ELPA* ####

If the *ELPA* library has been compiled with the configure option "--with-mpi=0",
no MPI will be used.

Still the **same** call sequence as in the MPI case can be used (see above).

#### Setting the row and column communicators ####

SYNOPSIS
   FORTRAN INTERFACE
       use elpa1

       success = elpa_get_communicators (mpi_comm_global, my_prow, my_pcol, mpi_comm_rows, mpi_comm_cols)

       integer, intent(in)   mpi_comm_global:  global communicator for the calculation
       integer, intent(in)   my_prow:          row coordinate of the calling process in the process grid
       integer, intent(in)   my_pcol:          column coordinate of the calling process in the process grid
       integer, intent(out)  mpi_comm_row:     communicator for communication within rows of processes
       integer, intent(out)  mpi_comm_row:     communicator for communication within columns of processes

       integer               success:          return value indicating success or failure of the underlying MPI_COMM_SPLIT function

   C INTERFACE
       #include "elpa_generated.h"

       success = elpa_get_communicators (int mpi_comm_world, int my_prow, my_pcol, int *mpi_comm_rows, int *Pmpi_comm_cols);

       int mpi_comm_global:  global communicator for the calculation
       int my_prow:          row coordinate of the calling process in the process grid
       int my_pcol:          column coordinate of the calling process in the process grid
       int *mpi_comm_row:    pointer to the communicator for communication within rows of processes
       int *mpi_comm_row:    pointer to the communicator for communication within columns of processes

       int  success:         return value indicating success or failure of the underlying MPI_COMM_SPLIT function


#### Using *ELPA 1stage* ####

After setting up the *ELPA* row and column communicators (by calling elpa_get_communicators),
only the real or complex valued solver has to be called:

SYNOPSIS
   FORTRAN INTERFACE
       use elpa1
       success = elpa_solve_evp_real_1stage_{single|double} (na, nev, a(lda,matrixCols), ev(nev), q(ldq, matrixCols), ldq, nblk, matrixCols, mpi_comm_rows,
       mpi_comm_cols)

       With the definintions of the input and output variables:

       integer, intent(in)    na:            global dimension of quadratic matrix a to solve
       integer, intent(in)    nev:           number of eigenvalues to be computed; the first nev eigenvalules are calculated
       real*{4|8},  intent(inout) a:         locally distributed part of the matrix a. The local dimensions are lda x matrixCols
       integer, intent(in)    lda:           leading dimension of locally distributed matrix a
       real*{4|8},  intent(inout) ev:        on output the first nev computed eigenvalues
       real*{4|8},  intent(inout) q:         on output the first nev computed eigenvectors
       integer, intent(in)    ldq:           leading dimension of matrix q which stores the eigenvectors
       integer, intent(in)    nblk:          blocksize of block cyclic distributin, must be the same in both directions
       integer, intent(in)    matrixCols:    number of columns of locally distributed matrices a and q
       integer, intent(in)    mpi_comm_rows: communicator for communication in rows. Constructed with get_elpa_communicators(3)
       integer, intent(in)    mpi_comm_cols: communicator for communication in colums. Constructed with get_elpa_communicators(3)

       logical                success:       return value indicating success or failure

   C INTERFACE
       #include "elpa.h"

       success = elpa_solve_evp_real_1stage_{single|double} (int na, int nev,  double *a, int lda,  double *ev, double *q, int ldq, int nblk, int matrixCols, int
       mpi_comm_rows, int mpi_comm_cols);

       With the definintions of the input and output variables:

       int     na:            global dimension of quadratic matrix a to solve
       int     nev:           number of eigenvalues to be computed; the first nev eigenvalules are calculated
       {float|double} *a:     pointer to locally distributed part of the matrix a. The local dimensions are lda x matrixCols
       int     lda:           leading dimension of locally distributed matrix a
       {float|double} *ev:    pointer to memory containing on output the first nev computed eigenvalues
       {float|double} *q:     pointer to memory containing on output the first nev computed eigenvectors
       int     ldq:           leading dimension of matrix q which stores the eigenvectors
       int     nblk:          blocksize of block cyclic distributin, must be the same in both directions
       int     matrixCols:    number of columns of locally distributed matrices a and q
       int     mpi_comm_rows: communicator for communication in rows. Constructed with get_elpa_communicators(3)
       int     mpi_comm_cols: communicator for communication in colums. Constructed with get_elpa_communicators(3)

       int     success:       return value indicating success (1) or failure (0)

DESCRIPTION
       Solve the real eigenvalue problem with the 1-stage solver. The ELPA communicators mpi_comm_rows and mpi_comm_cols are obtained with the
       get_elpa_communicators(3) function. The distributed quadratic marix a has global dimensions na x na, and a local size lda x matrixCols.
       The solver will compute the first nev eigenvalues, which will be stored on exit in ev. The eigenvectors corresponding to the eigenvalues
       will be stored in q. All memory of the arguments must be allocated outside the call to the solver.

   FORTRAN INTERFACE
       use elpa1
       success = elpa_solve_evp_complex_1stage_{single|double} (na, nev, a(lda,matrixCols), ev(nev), q(ldq, matrixCols), ldq, nblk, matrixCols, mpi_comm_rows,
       mpi_comm_cols)

       With the definintions of the input and output variables:

       integer,     intent(in)    na:            global dimension of quadratic matrix a to solve
       integer,     intent(in)    nev:           number of eigenvalues to be computed; the first nev eigenvalules are calculated
       complex*{8|16},  intent(inout) a:         locally distributed part of the matrix a. The local dimensions are lda x matrixCols
       integer,     intent(in)    lda:           leading dimension of locally distributed matrix a
       real*{4|8},      intent(inout) ev:        on output the first nev computed eigenvalues
       complex*{8|16},  intent(inout) q:         on output the first nev computed eigenvectors
       integer,     intent(in)    ldq:           leading dimension of matrix q which stores the eigenvectors
       integer,     intent(in)    nblk:          blocksize of block cyclic distributin, must be the same in both directions
       integer,     intent(in)    matrixCols:    number of columns of locally distributed matrices a and q
       integer,     intent(in)    mpi_comm_rows: communicator for communication in rows. Constructed with get_elpa_communicators(3)
       integer, intent(in)        mpi_comm_cols: communicator for communication in colums. Constructed with get_elpa_communicators(3)

       logical                    success:       return value indicating success or failure

   C INTERFACE
       #include "elpa.h"
       #include <complex.h>

       success = elpa_solve_evp_complex_1stage_{single|double} (int na, int nev,  double complex *a, int lda,  double *ev, double complex*q, int ldq, int nblk, int
       matrixCols, int mpi_comm_rows, int mpi_comm_cols);

       With the definintions of the input and output variables:

       int             na:            global dimension of quadratic matrix a to solve
       int             nev:           number of eigenvalues to be computed; the first nev eigenvalules are calculated
       {float|double} complex *a:     pointer to locally distributed part of the matrix a. The local dimensions are lda x matrixCols
       int             lda:           leading dimension of locally distributed matrix a
       {float|double}         *ev:    pointer to memory containing on output the first nev computed eigenvalues
       {float|double} complex *q:     pointer to memory containing on output the first nev computed eigenvectors
       int             ldq:           leading dimension of matrix q which stores the eigenvectors
       int             nblk:          blocksize of block cyclic distributin, must be the same in both directions
       int             matrixCols:    number of columns of locally distributed matrices a and q
       int             mpi_comm_rows: communicator for communication in rows. Constructed with get_elpa_communicators(3)
       int             mpi_comm_cols: communicator for communication in colums. Constructed with get_elpa_communicators(3)

       int             success:       return value indicating success (1) or failure (0)

DESCRIPTION
       Solve the complex eigenvalue problem with the 1-stage solver. The ELPA communicators mpi_comm_rows and mpi_comm_cols are obtained with the
       get_elpa_communicators(3) function. The distributed quadratic marix a has global dimensions na x na, and a local size lda x matrixCols.
       The solver will compute the first nev eigenvalues, which will be stored on exit in ev. The eigenvectors corresponding to the eigenvalues
       will be stored in q. All memory of the arguments must be allocated outside the call to the solver.


The *ELPA 1stage* solver, does not need or accept any other parameters than in the above
specification.

#### Using *ELPA 2stage* ####

The *ELPA 2stage* solver can be used in the same manner, as the *ELPA 1stage* solver.
However, the 2 stage solver, can be used with different compute kernels, which offers
more possibilities for configuration.

It is recommended to first call the utillity program

elpa2_print_kernels

which will tell all the compute kernels that can be used with *ELPA 2stage*". It will
also give information, whether a kernel can be set via environment variables.

##### Using the default kernels #####

If no kernel is set either via an environment variable or the *ELPA 2stage API* then
the default kernels will be set.

##### Setting the *ELPA 2stage* compute kernels #####

##### Setting the *ELPA 2stage* compute kernels with environment variables#####

If the *ELPA* installation allows setting ther compute kernels with enviroment variables,
setting the variables "REAL_ELPA_KERNEL" and "COMPLEX_ELPA_KERNEL" will set the compute
kernels. The environment variable setting will take precedence over all other settings!

The utility program "elpa2_print_kernels" can list which kernels are available and which
would be choosen. This reflects, as well the setting of the default kernel or the settings
with the environment variables

##### Setting the *ELPA 2stage* compute kernels with API calls#####

It is also possible to set the *ELPA 2stage* compute kernels via the API.

As an example the API for ELPA real double-precision 2stage is shown:

SYNOPSIS
   FORTRAN INTERFACE
       use elpa1
       use elpa2
       success = elpa_solve_evp_real_2stage_double (na, nev, a(lda,matrixCols), ev(nev), q(ldq, matrixCols), ldq, nblk, matrixCols, mpi_comm_rows,
       mpi_comm_cols, mpi_comm_all, THIS_REAL_ELPA_KERNEL, useQR, useGPU)

       With the definintions of the input and output variables:

       integer, intent(in)            na:            global dimension of quadratic matrix a to solve
       integer, intent(in)            nev:           number of eigenvalues to be computed; the first nev eigenvalules are calculated
       real*{4|8},  intent(inout)         a:         locally distributed part of the matrix a. The local dimensions are lda x matrixCols
       integer, intent(in)            lda:           leading dimension of locally distributed matrix a
       real*{4|8},  intent(inout)         ev:        on output the first nev computed eigenvalues
       real*{4|8},  intent(inout)         q:         on output the first nev computed eigenvectors
       integer, intent(in)            ldq:           leading dimension of matrix q which stores the eigenvectors
       integer, intent(in)            nblk:          blocksize of block cyclic distributin, must be the same in both directions
       integer, intent(in)            matrixCols:    number of columns of locally distributed matrices a and q
       integer, intent(in)            mpi_comm_rows: communicator for communication in rows. Constructed with get_elpa_communicators(3)
       integer, intent(in)            mpi_comm_cols: communicator for communication in colums. Constructed with get_elpa_communicators(3)
       integer, intent(in)            mpi_comm_all:  communicator for all processes in the processor set involved in ELPA
       logical, intent(in), optional: useQR:         optional argument; switches to QR-decomposition if set to .true.
       logical, intent(in), optional: useGPU:        decide whether GPUs should be used ore not

      logical                        success:       return value indicating success or failure

   C INTERFACE
       #include "elpa.h"

       success = elpa_solve_evp_real_2stage_double (int na, int nev,  double *a, int lda,  double *ev, double *q, int ldq, int nblk, int matrixCols, int
       mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_ELPA_REAL_KERNEL, int useQR, int useGPU);

       With the definintions of the input and output variables:

       int     na:            global dimension of quadratic matrix a to solve
       int     nev:           number of eigenvalues to be computed; the first nev eigenvalules are calculated
       double *a:             pointer to locally distributed part of the matrix a. The local dimensions are lda x matrixCols
       int     lda:           leading dimension of locally distributed matrix a
       double *ev:            pointer to memory containing on output the first nev computed eigenvalues
       double *q:             pointer to memory containing on output the first nev computed eigenvectors
       int     ldq:           leading dimension of matrix q which stores the eigenvectors
       int     nblk:          blocksize of block cyclic distributin, must be the same in both directions
       int     matrixCols:    number of columns of locally distributed matrices a and q
       int     mpi_comm_rows: communicator for communication in rows. Constructed with get_elpa_communicators(3)
       int     mpi_comm_cols: communicator for communication in colums. Constructed with get_elpa_communicators(3)
       int     mpi_comm_all:  communicator for all processes in the processor set involved in ELPA
       int     useQR:         if set to 1 switch to QR-decomposition
       int     useGPU:        decide whether the GPU version should be used or not

       int     success:       return value indicating success (1) or failure (0)


DESCRIPTION
       Solve the real eigenvalue problem with the 2-stage solver. The ELPA communicators mpi_comm_rows and mpi_comm_cols are obtained with the
       get_elpa_communicators(3) function. The distributed quadratic marix a has global dimensions na x na, and a local size lda x matrixCols.
       The solver will compute the first nev eigenvalues, which will be stored on exit in ev. The eigenvectors corresponding to the eigenvalues
       will be stored in q. All memory of the arguments must be allocated outside the call to the solver.

##### Setting up *ELPA 1stage* or *ELPA 2stage* with the *ELPA driver interface* #####

Since release ELPA 2016.005.004 a driver routine allows to choose more easily which solver (1stage or 2stage) will be used.

As an exmple the real double-precision case is explained:

 SYNOPSIS

 FORTRAN INTERFACE

  use elpa_driver

  success = elpa_solve_evp_real_double (na, nev, a(lda,matrixCols), ev(nev), q(ldq, matrixCols), ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all, THIS_REAL_ELPA_KERNEL=THIS_REAL_ELPA_KERNEL, useQR, useGPU, method=method)

  Generalized interface to the ELPA 1stage and 2stage solver for real-valued problems

  With the definintions of the input and output variables:


  integer, intent(in)            na:                    global dimension of quadratic matrix a to solve

  integer, intent(in)            nev:                   number of eigenvalues to be computed; the first nev eigenvalules are calculated

  real*8,  intent(inout)         a:                     locally distributed part of the matrix a. The local dimensions are lda x matrixCols

  integer, intent(in)            lda:                   leading dimension of locally distributed matrix a

  real*8,  intent(inout)         ev:                    on output the first nev computed eigenvalues"

  real*8,  intent(inout)         q:                     on output the first nev computed eigenvectors"

  integer, intent(in)            ldq:                   leading dimension of matrix q which stores the eigenvectors

  integer, intent(in)            nblk:                  blocksize of block cyclic distributin, must be the same in both directions

  integer, intent(in)            matrixCols:            number of columns of locally distributed matrices a and q

  integer, intent(in)            mpi_comm_rows:         communicator for communication in rows. Constructed with elpa_get_communicators

  integer, intent(in)            mpi_comm_cols:         communicator for communication in colums. Constructed with elpa_get_communicators

  integer, intent(in)            mpi_comm_all:          communicator for all processes in the processor set involved in ELPA

  integer, intent(in), optional: THIS_REAL_ELPA_KERNEL: optional argument, choose the compute kernel for 2-stage solver

  logical, intent(in), optional: useQR:                 optional argument; switches to QR-decomposition if set to .true.

  logical, intent(in), optional: useQPU:                decide whether the GPU version should be used or not

  character(*), optional         method:                use 1stage solver if "1stage", use 2stage solver if "2stage", (at the moment) use 2stage solver if "auto"

  logical                        success:               return value indicating success or failure


 C INTERFACE

 #include "elpa.h"

 success = elpa_solve_evp_real_double (int na, int nev, double *a, int lda, double *ev, double *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_ELPA_REAL_KERNEL, int useQR, int useGPU, char *method);"


 With the definintions of the input and output variables:"


 int     na:                    global dimension of quadratic matrix a to solve

 int     nev:                   number of eigenvalues to be computed; the first nev eigenvalules are calculated

 double *a:                     pointer to locally distributed part of the matrix a. The local dimensions are lda x matrixCols

 int     lda:                   leading dimension of locally distributed matrix a

 double *ev:                    pointer to memory containing on output the first nev computed eigenvalues

 double *q:                     pointer to memory containing on output the first nev computed eigenvectors

 int     ldq:                   leading dimension of matrix q which stores the eigenvectors

 int     nblk:                  blocksize of block cyclic distributin, must be the same in both directions

 int     matrixCols:            number of columns of locally distributed matrices a and q

 int     mpi_comm_rows:         communicator for communication in rows. Constructed with elpa_get_communicators

 int     mpi_comm_cols:         communicator for communication in colums. Constructed with elpa_get_communicators

 int     mpi_comm_all:          communicator for all processes in the processor set involved in ELPA

 int     THIS_ELPA_REAL_KERNEL: choose the compute kernel for 2-stage solver

 int     useQR:                 if set to 1 switch to QR-decomposition

 int     useGPU:                decide whether the GPU version should be used or not

 char   *method:                use 1stage solver if "1stage", use 2stage solver if "2stage", (at the moment) use 2stage solver if "auto"

 int     success:               return value indicating success (1) or failure (0)

 DESCRIPTION
 Solve the real eigenvalue problem. The value of method desides whether the 1stage or 2stage solver is used. The ELPA communicators mpi_comm_rows and mpi_comm_cols are obtained with the elpa_get_communicators function. The distributed quadratic marix a has global dimensions na x na, and a local size lda x matrixCols. The solver will compute the first nev eigenvalues, which will be stored on exit in ev. The eigenvectors corresponding to the eigenvalues will be stored in q. All memory of the arguments must be allocated outside the call to the solver.

##### Setting up the GPU version of *ELPA* 1 and 2 stage #####

Since release ELPA 2016.011.001.pre *ELPA* offers GPU support, IF *ELPA* has been build with the configure option "--enabble-gpu-support".

At run-time the GPU version can be used by setting the environment variable "ELPA_USE_GPU" to "yes", or by calling the *ELPA* functions
(elpa_solve_evp_real_{double|single}, elpa_solve_evp_real_1stage_{double|single}, elpa_solve_evp_real_2stage_{double|single}) with the
argument "useGPU = .true." or "useGPU = 1" for the Fortran and C case, respectively. Please, not that similiar to the choice of the
*ELPA* 2stage compute kernels, the enviroment variable takes precendence over the setting in the API call.

Further note that it is NOT allowed to define the usage of GPUs AND to EXPLICITLY set an ELPA 2stage compute kernel other than
"REAL_ELPA_KERNEL_GPU" or "COMPLEX_ELPA_KERNEL_GPU".



