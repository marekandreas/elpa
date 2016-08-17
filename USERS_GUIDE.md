## Users guide for the ELPA library ##

This document provides the guide for using the *ELPA* library in user applications.

### Online and local documentation ###

Local documentation (via man pages) should be available (if *ELPA* has been installed with the documentation):

For example "man get_elpa_communicators" should provide the documentation for the *ELPA* function which sets
the necessary communicators.

Also a [online doxygen documentation] (http://elpa.mpcdf.mpg.de/html/Documentation/ELPA-2016.05.003/html/index.html)
for each *ELPA* release is available.

### General concept of the  *ELPA* library ###

The *ELPA* library consists of two main parts:
- *ELPA 1stage* solver
- *ELPA 2stage* solver

Both variants of the *ELPA* solvers are available for real or complex valued matrices.

Thus *ELPA* provides the following user functions (see man pages or [online] (http://elpa.mpcdf.mpg.de/html/Documentation/ELPA-2016.05.003/html/index.html) for details):

- get_elpa_communicators   : set the row / column communicators for *ELPA*
- solve_evp_complex_1stage : solve a complex valued eigenvale proplem with the *ELPA 1stage* solver
- solve_evp_real_1stage    : solve a real valued eigenvale proplem with the *ELPA 1stage* solver
- solve_evp_complex_2stage : solve a complex valued eigenvale proplem with the *ELPA 2stage* solver
- solve_evp_real_2stage    : solve a real valued eigenvale proplem with the *ELPA 2stage* solver

Furthermore *ELPA* provides the utility binary "print_available_elpa2_kernels": it tells the user
which *ELPA 2stage* compute kernels have been installed and which default kernels are set

If you want to solve an eigenvalue problem with *ELPA*, you have to decide whether you
want to use *ELPA 1stage* or *ELPA 2stage* solver. Normally, *ELPA 2stage* is the better
choice since it is faster, but there a matrix dimensions where *ELPA 1stage* is supperior.

Independent of the choice of the solver, the concept of calling *ELPA* is always the same:

#### MPI version of *ELPA* ####

In this case, *ELPA* relies on a BLACS distributed matrix.
To solve a Eigenvalue problem of this matrix with *ELPA*, one has

1. to include the *ELPA* header (C case) or module (Fortran)
2. to create row and column MPI communicators for ELPA (with "get_elpa_communicators")
3. to call *ELPA 1stage* or *ELPA 2stage* for the matrix.

Here is a very simple MPI code snippet for using *ELPA 1stage*: For the definition of all variables
please have a look at the man pages and/or the online documentation (see above). A full version
of a simple example program can be found in ./test_project/src.


   ! All ELPA routines need MPI communicators for communicating within
   ! rows or columns of processes, these are set in get_elpa_communicators

   success = get_elpa_communicators(mpi_comm_world, my_prow, my_pcol, &
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

   success = solve_evp_real_1stage(na, nev, a, na_rows, ev, z, na_rows, nblk, &
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

       success = get_elpa_communicators (mpi_comm_global, my_prow, my_pcol, mpi_comm_rows, mpi_comm_cols)

       integer, intent(in)   mpi_comm_global:  global communicator for the calculation
       integer, intent(in)   my_prow:          row coordinate of the calling process in the process grid
       integer, intent(in)   my_pcol:          column coordinate of the calling process in the process grid
       integer, intent(out)  mpi_comm_row:     communicator for communication within rows of processes
       integer, intent(out)  mpi_comm_row:     communicator for communication within columns of processes

       integer               success:          return value indicating success or failure of the underlying MPI_COMM_SPLIT function

   C INTERFACE
       #include "elpa_generated.h"

       success = get_elpa_communicators (int mpi_comm_world, int my_prow, my_pcol, int *mpi_comm_rows, int *Pmpi_comm_cols);

       int mpi_comm_global:  global communicator for the calculation
       int my_prow:          row coordinate of the calling process in the process grid
       int my_pcol:          column coordinate of the calling process in the process grid
       int *mpi_comm_row:    pointer to the communicator for communication within rows of processes
       int *mpi_comm_row:    pointer to the communicator for communication within columns of processes

       int  success:         return value indicating success or failure of the underlying MPI_COMM_SPLIT function


#### Using *ELPA 1stage* ####

After setting up the *ELPA* row and column communicators (by calling get_elpa_communicators),
only the real or complex valued solver has to be called:

SYNOPSIS
   FORTRAN INTERFACE
       use elpa1
       success = solve_evp_real_1stage (na, nev, a(lda,matrixCols), ev(nev), q(ldq, matrixCols), ldq, nblk, matrixCols, mpi_comm_rows,
       mpi_comm_cols)

       With the definintions of the input and output variables:

       integer, intent(in)    na:            global dimension of quadratic matrix a to solve
       integer, intent(in)    nev:           number of eigenvalues to be computed; the first nev eigenvalules are calculated
       real*8,  intent(inout) a:             locally distributed part of the matrix a. The local dimensions are lda x matrixCols
       integer, intent(in)    lda:           leading dimension of locally distributed matrix a
       real*8,  intent(inout) ev:            on output the first nev computed eigenvalues
       real*8,  intent(inout) q:             on output the first nev computed eigenvectors
       integer, intent(in)    ldq:           leading dimension of matrix q which stores the eigenvectors
       integer, intent(in)    nblk:          blocksize of block cyclic distributin, must be the same in both directions
       integer, intent(in)    matrixCols:    number of columns of locally distributed matrices a and q
       integer, intent(in)    mpi_comm_rows: communicator for communication in rows. Constructed with get_elpa_communicators(3)
       integer, intent(in)    mpi_comm_cols: communicator for communication in colums. Constructed with get_elpa_communicators(3)

       logical                success:       return value indicating success or failure

   C INTERFACE
       #include "elpa.h"

       success = solve_evp_real_1stage (int na, int nev,  double *a, int lda,  double *ev, double *q, int ldq, int nblk, int matrixCols, int
       mpi_comm_rows, int mpi_comm_cols);

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

       int     success:       return value indicating success (1) or failure (0)

DESCRIPTION
       Solve the real eigenvalue problem with the 1-stage solver. The ELPA communicators mpi_comm_rows and mpi_comm_cols are obtained with the
       get_elpa_communicators(3) function. The distributed quadratic marix a has global dimensions na x na, and a local size lda x matrixCols.
       The solver will compute the first nev eigenvalues, which will be stored on exit in ev. The eigenvectors corresponding to the eigenvalues
       will be stored in q. All memory of the arguments must be allocated outside the call to the solver.

   FORTRAN INTERFACE
       use elpa1
       success = solve_evp_complex_1stage (na, nev, a(lda,matrixCols), ev(nev), q(ldq, matrixCols), ldq, nblk, matrixCols, mpi_comm_rows,
       mpi_comm_cols)

       With the definintions of the input and output variables:

       integer,     intent(in)    na:            global dimension of quadratic matrix a to solve
       integer,     intent(in)    nev:           number of eigenvalues to be computed; the first nev eigenvalules are calculated
       complex*16,  intent(inout) a:             locally distributed part of the matrix a. The local dimensions are lda x matrixCols
       integer,     intent(in)    lda:           leading dimension of locally distributed matrix a
       real*8,      intent(inout) ev:            on output the first nev computed eigenvalues
       complex*16,  intent(inout) q:             on output the first nev computed eigenvectors
       integer,     intent(in)    ldq:           leading dimension of matrix q which stores the eigenvectors
       integer,     intent(in)    nblk:          blocksize of block cyclic distributin, must be the same in both directions
       integer,     intent(in)    matrixCols:    number of columns of locally distributed matrices a and q
       integer,     intent(in)    mpi_comm_rows: communicator for communication in rows. Constructed with get_elpa_communicators(3)
       integer, intent(in)        mpi_comm_cols: communicator for communication in colums. Constructed with get_elpa_communicators(3)

       logical                    success:       return value indicating success or failure

   C INTERFACE
       #include "elpa.h"
       #include <complex.h>

       success = solve_evp_complex_1stage (int na, int nev,  double complex *a, int lda,  double *ev, double complex*q, int ldq, int nblk, int
       matrixCols, int mpi_comm_rows, int mpi_comm_cols);

       With the definintions of the input and output variables:

       int             na:            global dimension of quadratic matrix a to solve
       int             nev:           number of eigenvalues to be computed; the first nev eigenvalules are calculated
       double complex *a:             pointer to locally distributed part of the matrix a. The local dimensions are lda x matrixCols
       int             lda:           leading dimension of locally distributed matrix a
       double         *ev:            pointer to memory containing on output the first nev computed eigenvalues
       double complex *q:             pointer to memory containing on output the first nev computed eigenvectors
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

If the *ELPA* installation allows setting ther compute kernels with enviroment variables,
setting the variables "REAL_ELPA_KERNEL" and "COMPLEX_ELPA_KERNEL" will set the compute
kernels. The environment variable setting will take precedence over all other settings!

It is also possible to set the *ELPA 2stage* compute kernels via the API.

SYNOPSIS
   FORTRAN INTERFACE
       use elpa1 use elpa2
       success = solve_evp_real_2stage (na, nev, a(lda,matrixCols), ev(nev), q(ldq, matrixCols), ldq, nblk, matrixCols, mpi_comm_rows,
       mpi_comm_cols, mpi_comm_all, THIS_REAL_ELPA_KERNEL, useQr=useQR)

       With the definintions of the input and output variables:

       integer, intent(in)            na:            global dimension of quadratic matrix a to solve
       integer, intent(in)            nev:           number of eigenvalues to be computed; the first nev eigenvalules are calculated
       real*8,  intent(inout)         a:             locally distributed part of the matrix a. The local dimensions are lda x matrixCols
       integer, intent(in)            lda:           leading dimension of locally distributed matrix a
       real*8,  intent(inout)         ev:            on output the first nev computed eigenvalues
       real*8,  intent(inout)         q:             on output the first nev computed eigenvectors
       integer, intent(in)            ldq:           leading dimension of matrix q which stores the eigenvectors
       integer, intent(in)            nblk:          blocksize of block cyclic distributin, must be the same in both directions
       integer, intent(in)            matrixCols:    number of columns of locally distributed matrices a and q
       integer, intent(in)            mpi_comm_rows: communicator for communication in rows. Constructed with get_elpa_communicators(3)
       integer, intent(in)            mpi_comm_cols: communicator for communication in colums. Constructed with get_elpa_communicators(3)
       integer, intent(in)            mpi_comm_all:  communicator for all processes in the processor set involved in ELPA
       logical, intent(in), optional: useQR:         optional argument; switches to QR-decomposition if set to .true.

      logical                        success:       return value indicating success or failure

   C INTERFACE
       #include "elpa.h"

       success = solve_evp_real_2stage (int na, int nev,  double *a, int lda,  double *ev, double *q, int ldq, int nblk, int matrixCols, int
       mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_ELPA_REAL_KERNEL, int useQr);

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

       int     success:       return value indicating success (1) or failure (0)


DESCRIPTION
       Solve the real eigenvalue problem with the 2-stage solver. The ELPA communicators mpi_comm_rows and mpi_comm_cols are obtained with the
       get_elpa_communicators(3) function. The distributed quadratic marix a has global dimensions na x na, and a local size lda x matrixCols.
       The solver will compute the first nev eigenvalues, which will be stored on exit in ev. The eigenvectors corresponding to the eigenvalues
       will be stored in q. All memory of the arguments must be allocated outside the call to the solver.

SYNOPSIS
   FORTRAN INTERFACE
       use elpa1 use elpa2
       success = solve_evp_real_2stage (na, nev, a(lda,matrixCols), ev(nev), q(ldq, matrixCols), ldq, nblk, matrixCols, mpi_comm_rows,
       mpi_comm_cols, mpi_comm_all, THIS_REAL_ELPA_KERNEL)

       With the definintions of the input and output variables:

       integer,     intent(in)    na:            global dimension of quadratic matrix a to solve
       integer,     intent(in)    nev:           number of eigenvalues to be computed; the first nev eigenvalules are calculated
       complex*16,  intent(inout) a:             locally distributed part of the matrix a. The local dimensions are lda x matrixCols
       integer,     intent(in)    lda:           leading dimension of locally distributed matrix a
       real*8,      intent(inout) ev:            on output the first nev computed eigenvalues
       complex*16,  intent(inout) q:             on output the first nev computed eigenvectors
       integer,     intent(in)    ldq:           leading dimension of matrix q which stores the eigenvectors
       integer,     intent(in)    nblk:          blocksize of block cyclic distributin, must be the same in both directions
       integer,     intent(in)    matrixCols:    number of columns of locally distributed matrices a and q
       integer,     intent(in)    mpi_comm_rows: communicator for communication in rows. Constructed with get_elpa_communicators(3)
       integer,     intent(in)    mpi_comm_cols: communicator for communication in colums. Constructed with get_elpa_communicators(3)
       integer,     intent(in)    mpi_comm_all:  communicator for all processes in the processor set involved in ELPA
       logical                    success:       return value indicating success or failure

   C INTERFACE
       #include "elpa.h"
       #include <complex.h>

       success = solve_evp_complex_2stage (int na, int nev,  double complex *a, int lda,  double *ev, double complex *q, int ldq, int nblk, int
       matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_ELPA_REAL_KERNEL);

       With the definintions of the input and output variables:

       int             na:            global dimension of quadratic matrix a to solve
       int             nev:           number of eigenvalues to be computed; the first nev eigenvalules are calculated
       double complex *a:             pointer to locally distributed part of the matrix a. The local dimensions are lda x matrixCols
       int             lda:           leading dimension of locally distributed matrix a
       double         *ev:            pointer to memory containing on output the first nev computed eigenvalues
       double complex *q:             pointer to memory containing on output the first nev computed eigenvectors
       int             ldq:           leading dimension of matrix q which stores the eigenvectors
       int             nblk:          blocksize of block cyclic distributin, must be the same in both directions
       int             matrixCols:    number of columns of locally distributed matrices a and q
       int             mpi_comm_rows: communicator for communication in rows. Constructed with get_elpa_communicators(3)
       int             mpi_comm_cols: communicator for communication in colums. Constructed with get_elpa_communicators(3)
       int             mpi_comm_all:  communicator for all processes in the processor set involved in ELPA
       int             success:       return value indicating success (1) or failure (0)

DESCRIPTION
       Solve the complex eigenvalue problem with the 2-stage solver. The ELPA communicators mpi_comm_rows and mpi_comm_cols are obtained with the
       get_elpa_communicators(3) function. The distributed quadratic marix a has global dimensions na x na, and a local size lda x matrixCols.
       The solver will compute the first nev eigenvalues, which will be stored on exit in ev. The eigenvectors corresponding to the eigenvalues
       will be stored in q. All memory of the arguments must be allocated outside the call to the solver.


