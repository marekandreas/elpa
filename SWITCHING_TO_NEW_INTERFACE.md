## Documentation how to switch from the legacy API to the new API of the *ELPA* library ##

This document gives some hints how one can switch from the **deprecated** legacy API to new, long-term supported API of the *ELPA* library.
**At latest with the release ELPA 2019.11.001 you have to use the new API, since the old API has been removed**

### Using *ELPA* from a Fortran code ###

Up to now, if you have been using the (legacy API of the) *ELPA* library you had to do the following
steps: (we assume that MPI and a distributed matrix in block-cyclic scalapack layout is already created in
the user application)

1. including the *ELPA* modules

```fortran
   use elpa1
   use elpa2   ! this step was only needed if you wanted to use the ELPA 2stage solver
```

2. invoke the `elpa_get_communicators` routine, in order to obtain the row/column MPI communicators needed by *ELPA*

```fortran
   mpierr = elpa_get_communicators(mpi_comm_world, my_prow, my_pcol, &
                                   mpi_comm_rows, mpi_comm_cols)
```

3. do the desired task with the *ELPA* library, which could be
  - a) elpa_solve_[real|complex]_1stage_[double|single]     ! solve EV problem with ELPA 1stage solver
  - b) elpa_solve_[real|complex]_2stage_[double|single]     ! solve EV problem with ELPA 2stage solver
  - c) elpa_solve_tridi_[double|single]                     ! solve a the problem for a tri-diagonal matrix
  - d) elpa_cholesky_[real|complex]_[double|single]         ! Cholesky decomposition
  - e) elpa_invert_trm_[real|complex]_[double|single]       ! invert triangular matrix
  - f) elpa_mult_at_b_real_[double|single]                  ! multiply a**T * b
  - g) elpa_mult_ah_b_complex_[double|single]               ! multiply a**H * b

For each of the function calls you had to set some parameters (see man pages) to control the execution like
useGPU=[.false.|.true.], choice of ELPA 2stage kernel .... New parameters were likely added with a new release of
the *ELPA* library to reflect the growing functionality.


**The new interface of *ELPA* is more generic, which, however, requires ONCE the adaption of the user code if the new
interface should be used.**

This are the new steps to do (again it is assumed that MPI and a distributed matrix in block-cyclic scalapack layout is already created in
the user application):

1. include the correct *ELPA* module and define a name for the ELPA instance

```fortran
   use elpa   ! this is the only module needed for ELPA

   class(elpa_t), pointer :: e   ! name the ELPA instance "e"
```

2. initialize ELPA and create the instance

```fortran
   if (elpa_init(20170403) /= ELPA_OK) then       ! put here the version number of the API
     error stop "ELPA API version not supported"  ! which you are using
   endif

   e => elpa_allocate()
```

3. set the parameters which describe the matrix setup and the MPI

```fortran
   call e%set("na", na,success)                          ! size of matrix
   call e%set("local_nrows", na_rows,success)            ! MPI process local rows of the distributed matrixdo the
                                                         ! desired task with the *ELPA* library, which could be

   call e%set("local_ncols", na_cols,success)            ! MPI process local cols of the distributed matrix
   call e%set("nblk", nblk, success)                     ! size of block-cylic distribution

   call e%set("mpi_comm_parent", MPI_COMM_WORLD,succes)  ! global communicator for all processes which have parts of
                                                         ! the distributed matrix
   call e%set("process_row", my_prow, success)           ! row coordinate of MPI task
   call e%set("process_col", my_pcol, success)           ! column coordinate of MPI task
```

4. setup the ELPA instance

```fortran
   success = e%setup()
```

5. set/get any possible option (see man pages, or the document [USERS_GUIDE.md](USERS_GUIDE.md))

```Fortran
   call e%get("qr", qr, success)                        ! query whether QR-decomposition is set
   print *, "qr =", qr
   if (success .ne. ELPA_OK) stop

   call e%set("solver", ELPA_SOLVER_2STAGE, success)    ! set solver to 2stage
   if (success .ne. ELPA_OK) stop

   call e%set("real_kernel", ELPA_2STAGE_REAL_GENERIC, success) ! set kernel of ELPA 2stage solver for
                                                                !real case to the generic kernel

```

   At the moment, the following configurable runtime options are supported ([see here if you cannot read it in your editor](https://gitlab.mpcdf.mpg.de/elpa/elpa/wikis/USERS_GUIDE)):


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


6. do the desired task with the *ELPA* library, which could be


| Name         | Purpose                                                                 | since API version |
| :----------- | :---------------------------------------------------------------------- | :---------------- |
| eigenvectors | solve std. eigenvalue problem <br> compute eigenvalues and eigenvectors | 20170403  |
| eigenvalues  | solve std. eigenvalue problem <br> compute eigenvalues only             | 20170403  |
| generalized_eigenvectors | solve generalized eigenvalule problem <br> compute eigenvalues and eigenvectors | 20180525 |
| generalized_eigenvalues  | solve generalized eigenvalule problem <br> compute eigenvalues only             | 20180525 |
| hermitian_multiply       | do (real) a^T x b <br> (complex) a^H x b                                        | 20170403 |
| cholesky                 | do cholesky factorisation                                                       | 20170403 |
| invert_triangular        | invert a upper triangular matrix                                                | 20170403 |


7. when not needed anymore, destroy the instance

```fortran
   call elpa_deallocate()
```

8. when *ELPA* is not needed anymore, unitialize the *ELPA* library

```fortran
   call elpa_uninit()
```

### Online and local documentation ###

Local documentation (via man pages) should be available (if *ELPA* has been installed with the documentation):

For example `man elpa2_print_kernels` should provide the documentation for the *ELPA* program which prints all
the available kernels.

Also a [online doxygen documentation](http://elpa.mpcdf.mpg.de/html/Documentation/ELPA-2020.05.001/html/index.html)
for each *ELPA* release is available.


