## Users guide for the *ELPA* library ##

This document provides the guide for using the *ELPA* library with the new API (API version 20170403 or higher).

If you need instructions on how to build *ELPA*, please look at [INSTALL document](INSTALL.md) .

### Online and local documentation ###

Local documentation (via man pages) should be available (if *ELPA* has been installed with the documentation):

For example `man elpa2_print_kernels` should provide the documentation for the *ELPA* program, which prints all
the available kernels.

Also a [online doxygen documentation](https://elpa.mpcdf.mpg.de/documentation/doxygen/ELPA_DOXYGEN_PAGES/ELPA-2024.05.001/html/index.html)
for each *ELPA* release is available.


### API of the *ELPA* library ###

With release 2017.05.001 of the *ELPA* library the interface has been rewritten substantially, in order to have a more generic 
interface and to avoid future interface changes.


### Table of Contents: ###

- I)   General concept of the *ELPA* API
- II)  List of supported tunable parameters
- III) List of computational routines
- IV)  Using OpenMP threading
- V)   Influencing default values with environment variables
- VI)  Autotuning
- VII) A simple example how to use ELPA in an MPI application

## I) General concept of the *ELPA* API ##

Using *ELPA* just requires a few steps:

- include elpa headers `elpa/elpa.h` (C-Case) or use the Fortran module `use elpa`

- define a instance of the elpa type

- call elpa_init

- call elpa_allocate to allocate an instance of *ELPA*
  note that you can define (and configure individually) as many different instances
  for ELPA as you want, e.g. one for CPU only computations and for larger matrices on GPUs

- use ELPA-type function "set" to set matrix and MPI parameters

- call the ELPA-type function "setup"

- set or get all possible ELPA tunable options with ELPA-type functions get/set

- call ELPA-type function solve or others

- if the ELPA object is not needed any more call ELPA-type function destroy

- call elpa_uninit at the end of the program

To be more precise a basic call sequence for Fortran and C looks as follows:

Fortran synopsis

```fortran
 use elpa
 class(elpa_t), pointer :: elpa
 integer :: success

 if (elpa_init(20171201) /= ELPA_OK) then        ! put here the API version that you are using
    print *, "ELPA API version not supported"
    stop 1
  endif
  elpa => elpa_allocate(success)
  if (success != ELPA_OK) then
    ! react on the error
    ! we urge every user to always check the error codes
    ! of all ELPA functions
  endif

  ! set parameters decribing the matrix and it's MPI distribution
  call elpa%set("na", na, success)                          ! size of the na x na matrix
  call elpa%set("nev", nev, success)                        ! number of eigenvectors that should be computed ( 1<= nev <= na)
  call elpa%set("local_nrows", na_rows, success)            ! number of local rows of the distributed matrix on this MPI task 
  call elpa%set("local_ncols", na_cols, success)            ! number of local columns of the distributed matrix on this MPI task
  call elpa%set("nblk", nblk, success)                      ! size of the BLACS block cyclic distribution
  call elpa%set("mpi_comm_parent", MPI_COMM_WORLD, success) ! the global MPI communicator
  call elpa%set("process_row", my_prow, success)            ! row coordinate of MPI process
  call elpa%set("process_col", my_pcol, success)            ! column coordinate of MPI process

  success = elpa%setup()

  ! if desired, set any number of tunable run-time options
  ! look at the list of possible options as detailed later in
  ! USERS_GUIDE.md
  call e%set("solver", ELPA_SOLVER_2STAGE, success)

  ! set the AVX BLOCK2 kernel, otherwise ELPA_2STAGE_REAL_DEFAULT will
  ! be used
  call e%set("real_kernel", ELPA_2STAGE_REAL_AVX_BLOCK2, success)

  ! use method solve to solve the eigenvalue problem to obtain eigenvalues
  ! and eigenvectors
  ! other possible methods are desribed in USERS_GUIDE.md
  call e%eigenvectors(a, ev, z, success)

  ! cleanup
  call elpa_deallocate(e)

  call elpa_uninit()
```

C Synopsis:
```x
   #include <elpa/elpa.h>

   elpa_t handle;
   int error;

   if (elpa_init(20171201) != ELPA_OK) {                          // put here the API version that you are using
     fprintf(stderr, "Error: ELPA API version not supported");
     exit(1);
   }

   handle = elpa_allocate(&error);
   if (error != ELPA_OK) {
     /* react on the error code */
     /* we urge the user to always check the error codes of all ELPA functions */
   }


   /* Set parameters the matrix and it's MPI distribution */
   elpa_set(handle, "na", na, &error);                                           // size of the na x na matrix
   elpa_set(handle, "nev", nev, &error);                                         // number of eigenvectors that should be computed ( 1<= nev <= na)
   elpa_set(handle, "local_nrows", na_rows, &error);                             // number of local rows of the distributed matrix on this MPI task 
   elpa_set(handle, "local_ncols", na_cols, &error);                             // number of local columns of the distributed matrix on this MPI task
   elpa_set(handle, "nblk", nblk, &error);                                       // size of the BLACS block cyclic distribution
   elpa_set(handle, "mpi_comm_parent", MPI_Comm_c2f(MPI_COMM_WORLD), &error);    // the global MPI communicator
   elpa_set(handle, "process_row", my_prow, &error);                             // row coordinate of MPI process
   elpa_set(handle, "process_col", my_pcol, &error);                             // column coordinate of MPI process

   /* Setup */
   error = elpa_setup(handle);

   /* if desired, set any number of tunable run-time options */
   /* look at the list of possible options as detailed later in
      USERS_GUIDE.md */

   elpa_set(handle, "solver", ELPA_SOLVER_2STAGE, &error);
  
   // set the AVX BLOCK2 kernel, otherwise ELPA_2STAGE_REAL_DEFAULT will
   // be used
   elpa_set(handle, "real_kernel", ELPA_2STAGE_REAL_AVX_BLOCK2, &error)

   /* use method solve to solve the eigenvalue problem */
   /* other possible methods are desribed in USERS_GUIDE.md */
   elpa_eigenvectors(handle, a, ev, z, &error);

   /* cleanup */
   elpa_deallocate(handle);
   elpa_uninit();
```

## II) List of supported tunable parameters ##

The following table gives a list of all supported parameters which can be used to tune (influence) the runtime behaviour of *ELPA* ([see here if you cannot read it in your editor](https://gitlab.mpcdf.mpg.de/elpa/elpa/wikis/USERS_GUIDE))

| Parameter name | Short description     | default value               | possible values         | since API version | 
| :------------- |:--------------------- | :-------------------------- | :---------------------- | :---------------- | 
| solver         | use ELPA 1 stage <br>  or 2 stage solver | ELPA_SOLVER_1STAGE          | ELPA_SOLVER_1STAGE <br> ELPA_SOLVER_2STAGE      | 20170403          |
| gpu            | use GPU (if build <br> with GPU support)| 0                           | 0 or 1             | 20170403          | 
| real_kernel    | real kernel to be <br> used in ELPA 2 | ELPA_2STAGE_REAL_DEFAULT    | see output of <br> elpa2_print_kernels    | 20170403          |
| complex kernel | complex kernel to <br>  be used in ELPA 2 | ELPA_2STAGE_COMPLEX_DEFAULT | see output of <br>  elpa2_print_kernels     | 20170403          |
| omp_threads    | OpenMP threads used <br> (if build with OpenMP <br> support) | 1 | >1 | 20180525 |
| qr | Use QR decomposition in <br> ELPA 2 real | 0 | 0 or 1 |  20170403  |
| timings | Enable time <br> measurement | 1 | 0 or 1 |  20170403  |
| debug | give debug information | 0 | 0 or 1 | 20170403  |
       

## III) List of computational routines ##

The following compute routines are available in *ELPA*: Please have a look at the man pages or 
[online doxygen documentation](https://elpa.mpcdf.mpg.de/documentation/doxygen/ELPA_DOXYGEN_PAGES/ELPA-2024.05.001/html/index.html)
for details.


| Name         | Purpose                                                                 | since API version |
| :----------- | :---------------------------------------------------------------------- | :---------------- |
| eigenvectors | solve std. eigenvalue problem <br> compute eigenvalues and eigenvectors | 20170403  |
| eigenvalues  | solve std. eigenvalue problem <br> compute eigenvalues only             | 20170403  |
| generalized_eigenvectors | solve generalized eigenvalule problem <br> compute eigenvalues and eigenvectors | 20180525 |
| generalized_eigenvalues  | solve generalized eigenvalule problem <br> compute eigenvalues only             | 20180525 |
| hermitian_multiply       | do (real) a^T x b <br> (complex) a^H x b                                        | 20170403 |
| cholesky                 | do cholesky factorisation                                                       | 20170403 |
| invert_triangular        | invert a upper triangular matrix                                                | 20170403 |
| solve_tridiagonal        | solve EVP for a tridiagonal matrix                                              | 20170403 |


## IV) Using OpenMP threading ##

IMPORTANT: In case of hybrid MPI and OpenMP builds it is **mandatory** that your MPI library supports the threading levels "MPI_THREAD_SERIALIZED" or
"MPI_THREAD_MULTIPLE" (you can check this for example by building ELPA with MPI and OpenMP and run one of the test programs, they will warn you
if this prerequiste is not met). If your MPI library does **not** provide these threading levels, then ELPA will internally (independent of what you
set) use only **one** OpenMP thread and inform you at runtime with a warning. The number of threads used in a threaded implementation of your BLAS library
are not affected by this, as long as these threads can be controlled with another method than specifying OMP_NUM_THREADS (for instance with Intel's MKL
libray you can specify MKL_NUM_THREADS).

If *ELPA* has been build with OpenMP threading support you can specify the number of OpenMP threads that *ELPA* will use internally.
Please note that it is **mandatory**  to set the number of threads to be used with the OMP_NUM_THREADS environment variable **and**
with the **set method** 

```fortran
call e%set("omp_threads", 4, error)
```

**or the *ELPA* environment variable**

```
export ELPA_DEFAULT_omp_threads=4 (see Section V for an explanation of this variable).
```

Just setting the environment variable OMP_NUM_THREADS is **not** sufficient.

This is necessary to make the threading an autotunable option.

## V) Influencing default values with environment variables ##

For each tunable parameter mentioned in Section II, there exists a default value. This means, that if this parameter is **not explicitly** set by the user by the
*ELPA* set method, *ELPA* takes the default value for the parameter. E.g. if the user does not set a solver method, than *ELPA* will take the default 1`ELPA_SOLVER_1STAGE`.

The user can change this default value by setting an environment variable to the desired value.

The name of this variable is always constructed in the following way:
```
ELPA_DEFAULT_tunable_parameter_name=value
```

, e.g. in case of the solver the user can

```
export ELPA_DEFAULT_solver=ELPA_SOLVER_2STAGE
```

in order to define the 2stage solver as the default.

Speciall care has to be taken for keywords of the ELPA library, which contain a dash in the variable name, especially
the variables "nivida-gpu", "amd-gpu", and "intel-gpu". Since environment variables containing a dash are not allowed, for
these variables a work-around must be taken, for example
```
env 'ELPA_DEFAULT_nvidia-gpu=1' ./test_elpa.x ...
```
We will at a later release the alternative names "nvidia_gpu", "amd_gpu", and "intel_gpu", where the usual setting of
environment variables will work.


**Important note**
The default valule is completly ignored, if the user has manually set a parameter-value pair with the *ELPA* set method!
Thus the above environemnt variable will **not** have an effect, if the user code contains a line
```fortran
call e%set("solver",ELPA_SOLVER_1STAGE,error)
```
.

## VI) Using autotuning ##

Since API version 20171201 *ELPA* supports the autotuning of some "tunable" parameters (see Section II). The idea is that if *ELPA* is called multiple times (like typical in
self-consistent-iterations) some parameters can be tuned to an optimal value, which is hard to set for the user. Note, that not every parameter mentioned in Section II can actually be tuned with the autotuning. At the moment, only the parameters mentioned in the table below are affected by autotuning.

There are two ways, how the user can influence the autotuning steps:

1.) the user can set one of the following autotuning levels
- ELPA_AUTOTUNE_FAST
- ELPA_AUTOTUNE_MEDIUM

Each level defines a different set of tunable parameter. The autouning option will be extended by future releases of the *ELPA* library, at the moment the following
sets are supported: 

| AUTOTUNE LEVEL          | Parameters                                              |
| :---------------------- | :------------------------------------------------------ |
| ELPA_AUTOTUNE_FAST      | { solver, real_kernel, complex_kernel, omp_threads }    |
| ELPA_AUTOTUNE_MEDIUM    | all of abvoe + { gpu, partly gpu }                      |
| ELPA_AUTOTUNE_EXTENSIVE | all of above + { various blocking factors, stripewidth, intermediate_bandwidth } |

2.) the user can **remove** tunable parameters from the list of autotuning possibilites by explicetly setting this parameter,
e.g. if the user sets in his code 

```fortran
call e%set("solver", ELPA_SOLVER_2STAGE, error)
```
**before** invoking the autotuning, then the solver is fixed and not considered anymore for autotuning. Thus the ELPA_SOLVER_1STAGE would be skipped and, consequently, all possible autotuning parameters, which depend on ELPA_SOLVER_1STAGE.

The user can invoke autotuning in the following way:


Fortran synopsis

```fortran
 ! prepare elpa as you are used to (see Section I)
 ! only steps for autotuning are commentd
 use elpa
 class(elpa_t), pointer :: elpa
 class(elpa_autotune_t), pointer :: tune_state   ! create an autotuning pointer
 integer :: success

 if (elpa_init(20171201) /= ELPA_OK) then
    print *, "ELPA API version not supported"
    stop 1
  endif
  elpa => elpa_allocate(success)

  ! set parameters decribing the matrix and it's MPI distribution
  call elpa%set("na", na, success)
  call elpa%set("nev", nev, success))
  call elpa%set("local_nrows", na_rows, success)
  call elpa%set("local_ncols", na_cols, success)
  call elpa%set("nblk", nblk, success)
  call elpa%set("mpi_comm_parent", MPI_COMM_WORLD, success)
  call elpa%set("process_row", my_prow, success)
  call elpa%set("process_col", my_pcol, success)

  success = elpa%setup()

  tune_state => e%autotune_setup(ELPA_AUTOTUNE_MEDIUM, ELPA_AUTOTUNE_DOMAIN_REAL, success)   ! prepare autotuning, set AUTOTUNE_LEVEL and the domain (real or complex)

  ! do the loop of subsequent ELPA calls which will be used to do the autotuning
  do i=1, scf_cycles
    unfinished = e%autotune_step(tune_state, success)   ! check whether autotuning is finished; If not do next step

    if (.not.(unfinished)) then
      print *,"autotuning finished at step ",i
    endif

    call e%eigenvectors(a, ev, z, success)       ! do the normal computation

  enddo

  call e%autotune_set_best(tune_state, success)         ! from now use the values found by autotuning

  call elpa_autotune_deallocate(tune_state)    ! cleanup autotuning object 
```

C Synopsis
```c
   /* prepare ELPA the usual way; only steps for autotuning are commented */
   #include <elpa/elpa.h>

   elpa_t handle;
   elpa_autotune_t autotune_handle;                               // handle for autotuning
   int error;

   if (elpa_init(20171201) != ELPA_OK) { 
     fprintf(stderr, "Error: ELPA API version not supported");
     exit(1);
   }

   handle = elpa_allocate(&error);

   /* Set parameters the matrix and it's MPI distribution */
   elpa_set(handle, "na", na, &error);
   elpa_set(handle, "nev", nev, &error);
   elpa_set(handle, "local_nrows", na_rows, &error);
   elpa_set(handle, "local_ncols", na_cols, &error);
   elpa_set(handle, "nblk", nblk, &error);
   elpa_set(handle, "mpi_comm_parent", MPI_Comm_c2f(MPI_COMM_WORLD), &error);
   elpa_set(handle, "process_row", my_prow, &error);
   elpa_set(handle, "process_col", my_pcol, &error);
   /* Setup */
   elpa_setup(handle);

   autotune_handle = elpa_autotune_setup(handle, ELPA_AUTOTUNE_FAST, ELPA_AUTOTUNE_DOMAIN_REAL, &error);   // create autotune object

   // repeatedl call ELPA, e.g. in an scf iteration
   for (i=0; i < scf_cycles; i++) {

     unfinished = elpa_autotune_step(handle, autotune_handle, &error);      // check whether autotuning finished. If not do next step

     if (unfinished == 0) {
       printf("ELPA autotuning finished in the %d th scf step \n",i);
      }


      /* do the normal computation */
      elpa_eigenvectors(handle, a, ev, z, &error);
   }
   elpa_autotune_set_best(handle, autotune_handle &error);  // from now on use values used by autotuning
   elpa_autotune_deallocate(autotune_handle);        // cleanup autotuning
```


## VII) A simple example how to use ELPA in an MPI application ##

The following is a skeleton code of an basic example on how to use ELPA. The purpose is to show the steps that have 
to be done in the application using MPI which wants to call ELPA, namely

- Initializing the MPI
- creating a blacs distributed matrix
- IMPORTANT: it is very, very important that you check the return value of "descinit" of your blacs distribution!
  ELPA relies that the distribution it should work on is _valid_. If this is not the case the behavior is undefined!
- using this matrix within ELPA

The skeleton is not ment to be copied and pasted, since the details will always be dependent on the application which should 
call ELPA.

For simplicity only a Fortran example is shown


```fortran

use mpi

implicit none

integer :: mpierr, myid, nprocs
integer :: np_cols, np_rows, npcol, nprow
integer :: my_blacs_ctxt, sc_desc(9), info
integer :: na = [some value] ! global dimension of the matrix to be solved
integer :: nblk = [some value ] ! the block size of the scalapack block cyclic distribution
real*8, allocatable :: a(:,:), ev(:)

!-------------------------------------------------------------------------------
!  MPI Initialization

call mpi_init(mpierr)
call mpi_comm_rank(mpi_comm_world,myid,mpierr)
call mpi_comm_size(mpi_comm_world,nprocs,mpierr)  

!-------------------------------------------------------------------------------
! Selection of number of processor rows/columns
! the application has to decide how the matrix should be distributed
np_cols = [ some value ]
np_rows = [ some value ]


!-------------------------------------------------------------------------------
! Set up BLACS context and MPI communicators
!
! The BLACS context is only necessary for using Scalapack.
!
! For ELPA, the MPI communicators along rows/cols are sufficient,
! and the grid setup may be done in an arbitrary way as long as it is
! consistent (i.e. 0<=my_prow<np_rows, 0<=my_pcol<np_cols and every
! process has a unique (my_prow,my_pcol) pair).
! For details look at the documentation of  BLACS_Gridinit and
! BLACS_Gridinfo of your BLACS installation

my_blacs_ctxt = mpi_comm_world
call BLACS_Gridinit( my_blacs_ctxt, 'C', np_rows, np_cols )
call BLACS_Gridinfo( my_blacs_ctxt, nprow, npcol, my_prow, my_pcol )

! compute for your distributed matrix the number of local rows and columns 
! per MPI task, e.g. with
! the Scalapack tools routine NUMROC 

! Set up a scalapack descriptor for the checks below.
! For ELPA the following restrictions hold:
! - block sizes in both directions must be identical (args 4+5)
! - first row and column of the distributed matrix must be on row/col 0/0 (args 6+7)

call descinit( sc_desc, na, na, nblk, nblk, 0, 0, my_blacs_ctxt, na_rows, info )

! check the return code
if (info .ne. 0) then
  print *,"Invalid blacs-distribution. Abort!"
  stop 1
endif

! Allocate matrices 

allocate(a (na_rows,na_cols))
allocate(ev(na))

! fill the matrix with resonable values

a(i,j) = [ your problem to be solved]

! UP to this point this where all the prerequisites which have to be done in the
! application if you have a distributed eigenvalue problem to be solved, independent of
! whether you want to use ELPA, Scalapack, EigenEXA or alike

! Now you can start using ELPA

if (elpa_init(20171201) /= ELPA_OK) then        ! put here the API version that you are using
   print *, "ELPA API version not supported"
   stop 1
 endif
 elpa => elpa_allocate(success)
 if (success != ELPA_OK) then
   ! react on the error
   ! we urge every user to always check the error codes
   ! of all ELPA functions
 endif

 ! set parameters decribing the matrix and it's MPI distribution
 call elpa%set("na", na, success)                          ! size of the na x na matrix
 call elpa%set("nev", nev, success)                        ! number of eigenvectors that should be computed ( 1<= nev <= na)
 call elpa%set("local_nrows", na_rows, success)            ! number of local rows of the distributed matrix on this MPI task 
 call elpa%set("local_ncols", na_cols, success)            ! number of local columns of the distributed matrix on this MPI task
 call elpa%set("nblk", nblk, success)                      ! size of the BLACS block cyclic distribution
 call elpa%set("mpi_comm_parent", MPI_COMM_WORLD, success) ! the global MPI communicator
 call elpa%set("process_row", my_prow, success)            ! row coordinate of MPI process
 call elpa%set("process_col", my_pcol, success)            ! column coordinate of MPI process

 success = elpa%setup()

 ! if desired, set any number of tunable run-time options
 ! look at the list of possible options as detailed later in
 ! USERS_GUIDE.md
 call e%set("solver", ELPA_SOLVER_2STAGE, success)

 ! set the AVX BLOCK2 kernel, otherwise ELPA_2STAGE_REAL_DEFAULT will
 ! be used
 call e%set("real_kernel", ELPA_2STAGE_REAL_AVX_BLOCK2, success)

 ! use method solve to solve the eigenvalue problem to obtain eigenvalues
 ! and eigenvectors
 ! other possible methods are desribed in USERS_GUIDE.md
 call e%eigenvectors(a, ev, z, success)

 ! cleanup
 call elpa_deallocate(e)

 call elpa_uninit()
```
