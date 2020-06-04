## Users guide for the *ELPA* library with the legacy interface ##

**DISCLAIMER**
This document provides some guidelines for using the legacy interface of the *ELPA* library with user applications.
The legacy interface is deprecated and will be disabled at some point without any special announcement.
The following guidelines will not be updated or corrected anymore.
**We strongly recommend all users to use the long-term supported new API of ELPA, which has been published with the
release of 2017.05.001.**

## A) Using the legacy API of the *ELPA* library ##

The following description describes the usage of the *ELPA* library with the legacy interface.
This legacy API is deprecated and will be disabled at some point. We strongly recommend all users
to switch to the new API!. Nevertheless, for historic reasons we give some hints on how to use the legacy
API.

### A.1) General concept of the *ELPA* library ###

The *ELPA* library consists of two main parts:
- *ELPA 1stage* solver
- *ELPA 2stage* solver

Both variants of the *ELPA* solvers are available for real or complex singe and double precision valued matrices.

Thus *ELPA* provides the following user functions (see man pages or [online](http://elpa.mpcdf.mpg.de/html/Documentation/ELPA-2020.05.001/html/index.html) for details):

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
choice since it is faster, but there are matrix dimensions where *ELPA 1stage* is superior.

Independent of the choice of the solver, the concept of calling *ELPA* is always the same:

### A.2) MPI version of *ELPA* ###

In this case, *ELPA* relies on a BLACS distributed matrix.
To solve a Eigenvalue problem of this matrix with *ELPA*, one has

1. to include the *ELPA* header (C case) or module (Fortran)
2. to create row and column MPI communicators for ELPA (with "elpa_get_communicators")
3. to call to the *ELPA driver* or directly call *ELPA 1stage* or *ELPA 2stage* for the matrix.

Here is a very simple MPI code snippet for using *ELPA 1stage*: For the definition of all variables
please have a look at the man pages and/or the online documentation (see above). A full version
of a simple example program can be found in ./test_project_1stage_legacy_api/src.


```fortran
   ! All ELPA routines need MPI communicators for communicating within
   ! rows or columns of processes, these are set in elpa_get_communicators

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
```fortran


#### Shared-memory version of *ELPA* ####

If the *ELPA* library has been compiled with the configure option `--with-mpi=0`,
no MPI will be used.

Still the **same** call sequence as in the MPI case can be used (see above).



#### Setting the row and column communicators ####

SYNOPSIS
   FORTRAN INTERFACE
```fortran
       use elpa1

       success = elpa_get_communicators (mpi_comm_global, my_prow, my_pcol, mpi_comm_rows, mpi_comm_cols)

       integer, intent(in)   mpi_comm_global:  global communicator for the calculation
       integer, intent(in)   my_prow:          row coordinate of the calling process in the process grid
       integer, intent(in)   my_pcol:          column coordinate of the calling process in the process grid
       integer, intent(out)  mpi_comm_row:     communicator for communication within rows of processes
       integer, intent(out)  mpi_comm_row:     communicator for communication within columns of processes

       integer               success:          return value indicating success or failure of the underlying MPI_COMM_SPLIT function
```

   C INTERFACE
```c
       #include "elpa_generated.h"

       success = elpa_get_communicators (int mpi_comm_world, int my_prow, my_pcol, int *mpi_comm_rows, int *Pmpi_comm_cols);

       int mpi_comm_global:  global communicator for the calculation
       int my_prow:          row coordinate of the calling process in the process grid
       int my_pcol:          column coordinate of the calling process in the process grid
       int *mpi_comm_row:    pointer to the communicator for communication within rows of processes
       int *mpi_comm_row:    pointer to the communicator for communication within columns of processes

       int  success:         return value indicating success or failure of the underlying MPI_COMM_SPLIT function
```

#### Using *ELPA 1stage* ####

After setting up the *ELPA* row and column communicators (by calling elpa_get_communicators),
only the real or complex valued solver has to be called:

SYNOPSIS
   FORTRAN INTERFACE
```fortran
       use elpa1
       success = elpa_solve_evp_real_1stage_{single|double} (na, nev, a(lda,matrixCols), ev(nev), q(ldq, matrixCols), ldq, nblk, matrixCols, mpi_comm_rows,
       mpi_comm_cols)
```

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
       integer, intent(in)    mpi_comm_rows: communicator for communication in rows. Constructed with elpa_get_communicators(3)
       integer, intent(in)    mpi_comm_cols: communicator for communication in colums. Constructed with elpa_get_communicators(3)

       logical                success:       return value indicating success or failure

   C INTERFACE
```c
       #include "elpa.h"

       success = elpa_solve_evp_real_1stage_{single|double} (int na, int nev,  double *a, int lda,  double *ev, double *q, int ldq, int nblk, int matrixCols, int
       mpi_comm_rows, int mpi_comm_cols);
```
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
       int     mpi_comm_rows: communicator for communication in rows. Constructed with elpa_get_communicators(3)
       int     mpi_comm_cols: communicator for communication in colums. Constructed with elpa_get_communicators(3)

       int     success:       return value indicating success (1) or failure (0)

DESCRIPTION
       Solve the real eigenvalue problem with the 1-stage solver. The ELPA communicators mpi_comm_rows and mpi_comm_cols are obtained with the
       elpa_get_communicators(3) function. The distributed quadratic marix a has global dimensions na x na, and a local size lda x matrixCols.
       The solver will compute the first nev eigenvalues, which will be stored on exit in ev. The eigenvectors corresponding to the eigenvalues
       will be stored in q. All memory of the arguments must be allocated outside the call to the solver.

   FORTRAN INTERFACE
```fortran
       use elpa1
       success = elpa_solve_evp_complex_1stage_{single|double} (na, nev, a(lda,matrixCols), ev(nev), q(ldq, matrixCols), ldq, nblk, matrixCols, mpi_comm_rows,
       mpi_comm_cols)
```
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
       integer,     intent(in)    mpi_comm_rows: communicator for communication in rows. Constructed with elpa_get_communicators(3)
       integer, intent(in)        mpi_comm_cols: communicator for communication in colums. Constructed with elpa_get_communicators(3)

       logical                    success:       return value indicating success or failure

   C INTERFACE
```c
       #include "elpa.h"
       #include <complex.h>

       success = elpa_solve_evp_complex_1stage_{single|double} (int na, int nev,  double complex *a, int lda,  double *ev, double complex*q, int ldq, int nblk, int
       matrixCols, int mpi_comm_rows, int mpi_comm_cols);
```

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
       int             mpi_comm_rows: communicator for communication in rows. Constructed with elpa_get_communicators(3)
       int             mpi_comm_cols: communicator for communication in colums. Constructed with elpa_get_communicators(3)

       int             success:       return value indicating success (1) or failure (0)

DESCRIPTION
       Solve the complex eigenvalue problem with the 1-stage solver. The ELPA communicators mpi_comm_rows and mpi_comm_cols are obtained with the
       elpa_get_communicators(3) function. The distributed quadratic marix a has global dimensions na x na, and a local size lda x matrixCols.
       The solver will compute the first nev eigenvalues, which will be stored on exit in ev. The eigenvectors corresponding to the eigenvalues
       will be stored in q. All memory of the arguments must be allocated outside the call to the solver.


The *ELPA 1stage* solver, does not need or accept any other parameters than in the above
specification.

#### Using *ELPA 2stage* ####

The *ELPA 2stage* solver can be used in the same manner, as the *ELPA 1stage* solver.
However, the 2 stage solver, can be used with different compute kernels, which offers
more possibilities for configuration.

It is recommended to first call the utility program

elpa2_print_kernels

which will tell all the compute kernels that can be used with *ELPA 2stage*

##### Using the default kernels #####

If no kernel is set via the *ELPA 2stage API* then the default kernels will be set.

##### Setting the *ELPA 2stage* compute kernels #####

##### Setting the *ELPA 2stage* compute kernels with environment variables #####


The utility program "elpa2_print_kernels" can list which kernels are available and which
would be chosen. This reflects the setting of the default kernel.

##### Setting the *ELPA 2stage* compute kernels with API calls #####

It is also possible to set the *ELPA 2stage* compute kernels via the API.

As an example the API for ELPA real double-precision 2stage is shown:

SYNOPSIS
   FORTRAN INTERFACE
```fortran
       use elpa1
       use elpa2
       success = elpa_solve_evp_real_2stage_double (na, nev, a(lda,matrixCols), ev(nev), q(ldq, matrixCols), ldq, nblk, matrixCols, mpi_comm_rows,
       mpi_comm_cols, mpi_comm_all, THIS_REAL_ELPA_KERNEL, useQR, useGPU)
```

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
       integer, intent(in)            mpi_comm_rows: communicator for communication in rows. Constructed with elpa_get_communicators(3)
       integer, intent(in)            mpi_comm_cols: communicator for communication in colums. Constructed with elpa_get_communicators(3)
       integer, intent(in)            mpi_comm_all:  communicator for all processes in the processor set involved in ELPA
       logical, intent(in), optional: useQR:         optional argument; switches to QR-decomposition if set to .true.
       logical, intent(in), optional: useGPU:        decide whether GPUs should be used ore not

      logical                        success:       return value indicating success or failure

   C INTERFACE
```c
       #include "elpa.h"

       success = elpa_solve_evp_real_2stage_double (int na, int nev,  double *a, int lda,  double *ev, double *q, int ldq, int nblk, int matrixCols, int
       mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_ELPA_REAL_KERNEL, int useQR, int useGPU);
```

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
       int     mpi_comm_rows: communicator for communication in rows. Constructed with elpa_get_communicators(3)
       int     mpi_comm_cols: communicator for communication in colums. Constructed with elpa_get_communicators(3)
       int     mpi_comm_all:  communicator for all processes in the processor set involved in ELPA
       int     useQR:         if set to 1 switch to QR-decomposition
       int     useGPU:        decide whether the GPU version should be used or not

       int     success:       return value indicating success (1) or failure (0)


DESCRIPTION
       Solve the real eigenvalue problem with the 2-stage solver. The ELPA communicators mpi_comm_rows and mpi_comm_cols are obtained with the
       elpa_get_communicators(3) function. The distributed quadratic marix a has global dimensions na x na, and a local size lda x matrixCols.
       The solver will compute the first nev eigenvalues, which will be stored on exit in ev. The eigenvectors corresponding to the eigenvalues
       will be stored in q. All memory of the arguments must be allocated outside the call to the solver.

##### Setting up *ELPA 1stage* or *ELPA 2stage* with the *ELPA driver interface* #####

Since release ELPA 2016.005.004 a driver routine allows to choose more easily which solver (1stage or 2stage) will be used.

As an exmple the real double-precision case is explained:

 SYNOPSIS

 FORTRAN INTERFACE
```fortran
  use elpa_driver

  success = elpa_solve_evp_real_double (na, nev, a(lda,matrixCols), ev(nev), q(ldq, matrixCols), ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all, THIS_REAL_ELPA_KERNEL=THIS_REAL_ELPA_KERNEL, useQR, useGPU, method=method)
```

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
```c
 #include "elpa.h"

 success = elpa_solve_evp_real_double (int na, int nev, double *a, int lda, double *ev, double *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_ELPA_REAL_KERNEL, int useQR, int useGPU, char *method);"
```


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
`REAL_ELPA_KERNEL_GPU` or `COMPLEX_ELPA_KERNEL_GPU`.



