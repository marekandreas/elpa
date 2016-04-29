## Users guide for the ELPA library ##

This document provides the guide for using the *ELPA* library in user applications.

### Online and local documentation ###

Local documentation (via man pages) should be available (if *ELPA* has been installed with the documentation):

For example "man get_elpa_communicators" should provide the documentation for the *ELPA* function which sets
the necessary communicators.

Also a [online doxygen documentation] (http://elpa.mpcdf.mpg.de/html/Documentation/ELPA-2015.11.001/html/index.html)
for each *ELPA* release is available.

### General concept to use ELPA ###

#### MPI version of ELPA ####

In this case, *ELPA* relies on a BLACS distributed matrix.
To solve a Eigenvalue problem of this matrix with *ELPA*, one has

1. to include the *ELPA* header (C case) or module (Fortran)
2. to create row and column MPI communicators for ELPA (with "get_elpa_communicators")
3. to call *ELPA 1stage* or *ELPA 2stage* for the matrix.

Here is a very simple MPI code snippet for using *ELPA*: For the definition of all variables
please have a look at the man pages and/or the online documentation (see above)


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

   success solve_evp_real_1stage(na, nev, a, na_rows, ev, z, na_rows, nblk, &
                                 matrixCols, mpi_comm_rows, mpi_comm_cols)

   if (myid==0) then
     print '(a)','| One-step ELPA solver complete.'
     print *
   end if


#### Shared-memory version of ELPA ####

If the *ELPA* library has been compiled with the configure option "--enable-shared-memory-only",
no MPI will be used.

Still the **same** call sequence as in the MPI case can be used (see above).





