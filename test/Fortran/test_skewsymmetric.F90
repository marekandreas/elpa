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
!
#include "config-f90.h"

! Define one of TEST_REAL or TEST_COMPLEX
! Define one of TEST_SINGLE or TEST_DOUBLE
! Define one of TEST_SOLVER_1STAGE or TEST_SOLVER_2STAGE
! Define TEST_NVIDIA_GPU \in [0, 1]
! Define TEST_INTEL_GPU \in [0, 1]
! Define TEST_AMD_GPU \in [0, 1]
! Define either TEST_ALL_KERNELS or a TEST_KERNEL \in [any valid kernel]

#if !(defined(TEST_REAL) ^ defined(TEST_COMPLEX))
error: define exactly one of TEST_REAL or TEST_COMPLEX
#endif

#if !(defined(TEST_SINGLE) ^ defined(TEST_DOUBLE))
error: define exactly one of TEST_SINGLE or TEST_DOUBLE
#endif

#ifdef TEST_SINGLE
#  define EV_TYPE real(kind=C_FLOAT)
#  define EV_TYPE_COMPLEX complex(kind=C_FLOAT_COMPLEX)
#  define MATRIX_TYPE_COMPLEX complex(kind=C_FLOAT_COMPLEX)
#  ifdef TEST_REAL
#    define MATRIX_TYPE real(kind=C_FLOAT)
#  else
#    define MATRIX_TYPE complex(kind=C_FLOAT_COMPLEX)
#  endif
#else
#  define MATRIX_TYPE_COMPLEX complex(kind=C_DOUBLE_COMPLEX)
#  define EV_TYPE_COMPLEX complex(kind=C_DOUBLE_COMPLEX)
#  define EV_TYPE real(kind=C_DOUBLE)
#  ifdef TEST_REAL
#    define MATRIX_TYPE real(kind=C_DOUBLE)
#  else
#    define MATRIX_TYPE complex(kind=C_DOUBLE_COMPLEX)
#  endif
#endif

#ifdef TEST_REAL
#  define AUTOTUNE_DOMAIN ELPA_AUTOTUNE_DOMAIN_REAL
#else
#  define AUTOTUNE_DOMAIN ELPA_AUTOTUNE_DOMAIN_COMPLEX
#endif
#ifdef HAVE_64BIT_INTEGER_MATH_SUPPORT
#define TEST_INT_TYPE integer(kind=c_int64_t)
#define INT_TYPE c_int64_t
#else
#define TEST_INT_TYPE integer(kind=c_int32_t)
#define INT_TYPE c_int32_t
#endif
#ifdef HAVE_64BIT_INTEGER_MPI_SUPPORT
#define TEST_INT_MPI_TYPE integer(kind=c_int64_t)
#define INT_MPI_TYPE c_int64_t
#else
#define TEST_INT_MPI_TYPE integer(kind=c_int32_t)
#define INT_MPI_TYPE c_int32_t
#endif

#define TEST_GPU 0
#if (TEST_NVIDIA_GPU == 1) || (TEST_AMD_GPU == 1) || (TEST_INTEL_GPU == 1)
#undef TEST_GPU
#define TEST_GPU 1
#endif

#include "assert.h"

program test
   use elpa

   !use test_util
   use test_setup_mpi
   use test_prepare_matrix
   use test_read_input_parameters
   use test_blacs_infrastructure
   use test_check_correctness
   use precision_for_tests
   use iso_fortran_env

#ifdef HAVE_REDIRECT
   use test_redirect
#endif
   implicit none

   ! matrix dimensions
   TEST_INT_TYPE                          :: na, nev, nblk

   ! mpi
   TEST_INT_TYPE                          :: myid, nprocs
   TEST_INT_TYPE                          :: na_cols, na_rows  ! local matrix size
   TEST_INT_TYPE                          :: np_cols, np_rows  ! number of MPI processes per column/row
   TEST_INT_TYPE                          :: my_prow, my_pcol  ! local MPI task position (my_prow, my_pcol) in the grid (0..np_cols -1, 0..np_rows -1)
   TEST_INT_MPI_TYPE                      :: mpierr

   ! blacs
   character(len=1)                 :: layout
   TEST_INT_TYPE                          :: my_blacs_ctxt, sc_desc(9), info, nprow, npcol

   ! The Matrix
   MATRIX_TYPE, allocatable         :: a_skewsymmetric(:,:), as_skewsymmetric(:,:)
   MATRIX_TYPE_COMPLEX, allocatable :: a_complex(:,:), as_complex(:,:)
   ! eigenvectors
   MATRIX_TYPE, allocatable         :: z_skewsymmetric(:,:)
   MATRIX_TYPE_COMPLEX, allocatable :: z_complex(:,:)
   ! eigenvalues
   EV_TYPE, allocatable             :: ev_skewsymmetric(:), ev_complex(:)

   TEST_INT_TYPE                    :: status, i, j
   integer(kind=c_int)              :: error_elpa

   type(output_t)                   :: write_to_file
   class(elpa_t), pointer           :: e_complex, e_skewsymmetric
           
   call read_input_parameters(na, nev, nblk, write_to_file)
   call setup_mpi(myid, nprocs)
#ifdef HAVE_REDIRECT
#ifdef WITH_MPI
   call MPI_BARRIER(MPI_COMM_WORLD, mpierr)
   call redirect_stdout(myid)
#endif
#endif

   if (elpa_init(CURRENT_API_VERSION) /= ELPA_OK) then
     print *, "ELPA API version not supported"
     stop 1
   endif
! 
   layout = 'C'
   do np_cols = NINT(SQRT(REAL(nprocs))),2,-1
      if(mod(nprocs,np_cols) == 0 ) exit
   enddo
   np_rows = nprocs/np_cols
   assert(nprocs == np_rows * np_cols)

   if (myid == 0) then
     print '((a,i0))', 'Matrix size: ', na
     print '((a,i0))', 'Num eigenvectors: ', nev
     print '((a,i0))', 'Blocksize: ', nblk
#ifdef WITH_MPI
     print '((a,i0))', 'Num MPI proc: ', nprocs
     print '(3(a,i0))','Number of processor rows=',np_rows,', cols=',np_cols,', total=',nprocs
     print '(a)',      'Process layout: ' // layout
#endif
     print *,''
   endif

   call set_up_blacsgrid(int(mpi_comm_world,kind=BLAS_KIND), np_rows, &
                             np_cols, layout, &
                             my_blacs_ctxt, my_prow, my_pcol)

   call set_up_blacs_descriptor(na, nblk, my_prow, my_pcol, np_rows, np_cols, &
                                na_rows, na_cols, sc_desc, my_blacs_ctxt, info)

   allocate(a_skewsymmetric (na_rows,na_cols))
   allocate(as_skewsymmetric(na_rows,na_cols))
   allocate(z_skewsymmetric (na_rows,2*na_cols))
   allocate(ev_skewsymmetric(na))

   a_skewsymmetric(:,:) = 0.0
   z_skewsymmetric(:,:) = 0.0
   ev_skewsymmetric(:) = 0.0

   call prepare_matrix_random(na, myid, sc_desc, a_skewsymmetric, &
   z_skewsymmetric(:,1:na_cols), as_skewsymmetric, is_skewsymmetric=1)
   
   !call MPI_BARRIER(MPI_COMM_WORLD, mpierr)  
   as_skewsymmetric(:,:) = a_skewsymmetric(:,:)
   

   ! prepare the complex matrix for the "brute force" case
   allocate(a_complex (na_rows,na_cols))
   allocate(as_complex(na_rows,na_cols))
   allocate(z_complex (na_rows,na_cols))
   allocate(ev_complex(na))

   a_complex(1:na_rows,1:na_cols) = 0.0
   z_complex(1:na_rows,1:na_cols) = 0.0
   as_complex(1:na_rows,1:na_cols) = 0.0
   

      do j=1, na_cols
        do i=1,na_rows
#ifdef TEST_DOUBLE
          a_complex(i,j) = dcmplx(0.0, a_skewsymmetric(i,j))
#endif
#ifdef TEST_SINGLE
          a_complex(i,j) = cmplx(0.0, a_skewsymmetric(i,j))
#endif
        enddo
      enddo
   


   z_complex(1:na_rows,1:na_cols)  = a_complex(1:na_rows,1:na_cols)
   as_complex(1:na_rows,1:na_cols) = a_complex(1:na_rows,1:na_cols)

   ! first set up and solve the brute force problem
   e_complex => elpa_allocate(error_elpa)
   call set_basic_params(e_complex, na, nev, na_rows, na_cols, my_prow, my_pcol)

   call e_complex%set("timings",1, error_elpa)

   call e_complex%set("debug",1,error_elpa)

#if TEST_NVIDIA_GPU == 1 || (TEST_NVIDIA_GPU == 0) && (TEST_AMD_GPU == 0) && (TEST_INTEL_GPU == 0)  
   call e_complex%set("nvidia-gpu", TEST_GPU,error_elpa)
#endif
#if TEST_AMD_GPU == 1
   call e_complex%set("amd-gpu", TEST_GPU,error_elpa)
#endif
#if TEST_INTEL_GPU == 1
   call e_complex%set("intel-gpu", TEST_GPU,error_elpa)
#endif

   call e_complex%set("omp_threads", 8, error_elpa)

   assert_elpa_ok(e_complex%setup())
   call e_complex%set("solver", elpa_solver_2stage, error_elpa)

   call e_complex%timer_start("eigenvectors: brute force as complex matrix")
   call e_complex%eigenvectors(a_complex, ev_complex, z_complex, error_elpa)
   call e_complex%timer_stop("eigenvectors: brute force as complex matrix")

   if (myid .eq. 0) then
     print *, ""
     call e_complex%print_times("eigenvectors: brute force as complex matrix")
   endif 
#ifdef WITH_MPI
     call MPI_BARRIER(MPI_COMM_WORLD, mpierr)
#endif     
!      as_complex(:,:) = z_complex(:,:)
#ifdef TEST_SINGLE
     status = check_correctness_evp_numeric_residuals_complex_single(na, nev, as_complex, z_complex, ev_complex, sc_desc, &
                                                    nblk, myid, np_rows,np_cols, my_prow, my_pcol)
#else
     status = check_correctness_evp_numeric_residuals_complex_double(na, nev, as_complex, z_complex, ev_complex, sc_desc, &
                                                    nblk, myid, np_rows,np_cols, my_prow, my_pcol)
#endif
    status = 0
    call check_status(status, myid)

#ifdef WITH_MPI
     call MPI_BARRIER(MPI_COMM_WORLD, mpierr)
#endif
   ! now run the skewsymmetric case
   e_skewsymmetric => elpa_allocate(error_elpa)
   call set_basic_params(e_skewsymmetric, na, nev, na_rows, na_cols, my_prow, my_pcol)

   call e_skewsymmetric%set("timings",1, error_elpa)

   call e_skewsymmetric%set("debug",1,error_elpa)

#if TEST_NVIDIA_GPU == 1 || (TEST_NVIDIA_GPU == 0) && (TEST_AMD_GPU == 0) && (TEST_INTEL_GPU == 0)
   call e_skewsymmetric%set("nvidia-gpu", TEST_GPU,error_elpa)
#endif
#if TEST_AMD_GPU == 1
   call e_skewsymmetric%set("amd-gpu", TEST_GPU,error_elpa)
#endif
#if TEST_INTEL_GPU == 1
   call e_skewsymmetric%set("intel-gpu", TEST_GPU,error_elpa)
#endif
   call e_skewsymmetric%set("omp_threads",8, error_elpa)

   assert_elpa_ok(e_skewsymmetric%setup())
   
   call e_skewsymmetric%set("solver", elpa_solver_2stage, error_elpa)

   call e_skewsymmetric%timer_start("eigenvectors: skewsymmetric ")
   call e_skewsymmetric%skew_eigenvectors(a_skewsymmetric, ev_skewsymmetric, z_skewsymmetric, error_elpa)
   call e_skewsymmetric%timer_stop("eigenvectors: skewsymmetric ")

   if (myid .eq. 0) then
     print *, ""
     call e_skewsymmetric%print_times("eigenvectors: skewsymmetric")
   endif
   
   ! check eigenvalues
   do i=1, na
     if (myid == 0) then
#ifdef TEST_DOUBLE
       if (abs(ev_complex(i)-ev_skewsymmetric(i))/abs(ev_complex(i)) .gt. 1e-10) then
#endif
#ifdef TEST_SINGLE
       if (abs(ev_complex(i)-ev_skewsymmetric(i))/abs(ev_complex(i)) .gt. 1e-4) then
#endif
         print *,"ev: i=",i,ev_complex(i),ev_skewsymmetric(i)
         status = 1
     endif
     endif
   enddo


!    call check_status(status, myid)
   
   z_complex(:,:) = 0
   do j=1, na_cols
     do i=1,na_rows
#ifdef TEST_DOUBLE
       z_complex(i,j) = dcmplx(z_skewsymmetric(i,j), z_skewsymmetric(i,na_cols+j))
#endif
#ifdef TEST_SINGLE
       z_complex(i,j) = cmplx(z_skewsymmetric(i,j), z_skewsymmetric(i,na_cols+j))
#endif
     enddo
   enddo
#ifdef WITH_MPI
   call MPI_BARRIER(MPI_COMM_WORLD, mpierr)
#endif

#ifdef TEST_SINGLE
   status = check_correctness_evp_numeric_residuals_ss_real_single(na, nev, as_skewsymmetric, z_complex, ev_skewsymmetric, &
                              sc_desc, nblk, myid, np_rows,np_cols, my_prow, my_pcol)
#else
   status = check_correctness_evp_numeric_residuals_ss_real_double(na, nev, as_skewsymmetric, z_complex, ev_skewsymmetric, &
                              sc_desc, nblk, myid, np_rows,np_cols, my_prow, my_pcol)
#endif
   
#ifdef WITH_MPI
    call MPI_BARRIER(MPI_COMM_WORLD, mpierr)
#endif
   call elpa_deallocate(e_complex,error_elpa)
   call elpa_deallocate(e_skewsymmetric,error_elpa)


   !to do 
   ! - check whether brute-force check_correctness_evp_numeric_residuals worsk (complex ev)
   ! - invent a test for skewsymmetric residuals

   deallocate(a_complex)
   deallocate(as_complex)
   deallocate(z_complex)
   deallocate(ev_complex)

   deallocate(a_skewsymmetric)
   deallocate(as_skewsymmetric)
   deallocate(z_skewsymmetric)
   deallocate(ev_skewsymmetric)
   call elpa_uninit(error_elpa)



#ifdef WITH_MPI
   call blacs_gridexit(my_blacs_ctxt)
   call mpi_finalize(mpierr)
#endif

   call exit(status)

contains
   subroutine set_basic_params(elpa, na, nev, na_rows, na_cols, my_prow, my_pcol)
     implicit none
     class(elpa_t), pointer      :: elpa
     TEST_INT_TYPE, intent(in)         :: na, nev, na_rows, na_cols, my_prow, my_pcol

     call elpa%set("na", int(na,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
     call elpa%set("nev", int(nev,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
     call elpa%set("local_nrows", int(na_rows,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
     call elpa%set("local_ncols", int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
     call elpa%set("nblk", int(nblk,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)

#ifdef WITH_MPI
     call elpa%set("mpi_comm_parent", int(MPI_COMM_WORLD,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
     call elpa%set("process_row", int(my_prow,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
     call elpa%set("process_col", int(my_pcol,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif
   end subroutine
   subroutine check_status(status, myid)
     implicit none
     TEST_INT_TYPE, intent(in) :: status, myid
     TEST_INT_MPI_TYPE         :: mpierr
     if (status /= 0) then
       if (myid == 0) print *, "Result incorrect!"
#ifdef WITH_MPI
       call mpi_finalize(mpierr)
#endif
       call exit(status)
     endif
   end subroutine
end program
