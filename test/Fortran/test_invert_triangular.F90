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
! Define TEST_NVIDIA_GPU \in [0, 1]
! Define TEST_INTEL_GPU \in [0, 1]
! Define TEST_AMD_GPU \in [0, 1]

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
#if (TEST_NVIDIA_GPU == 1) || (TEST_AMD_GPU == 1) || (TEST_INTEL_GPU == 1) || (TEST_INTEL_GPU_OPENMP == 1) || (TEST_INTEL_GPU_SYCL == 1)
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
   use test_analytic ! for print_matrix
#ifdef WITH_OPENMP_TRADITIONAL
   use omp_lib
#endif

#ifdef HAVE_REDIRECT
   use test_redirect
#endif

#if TEST_GPU == 1
   use test_gpu
   use mod_check_for_gpu
#if TEST_NVIDIA_GPU == 1
   use test_cuda_functions
#endif
#if TEST_AMD_GPU == 1
   use test_hip_functions
#endif
#if TEST_INTEL_GPU_SYCL == 1
   use test_sycl_functions
#endif

#endif /* TEST_GPU */

   implicit none

   ! matrix dimensions
   TEST_INT_TYPE                          :: na, nev, nblk

   ! mpi
   TEST_INT_TYPE                          :: myid, nprocs
   TEST_INT_TYPE                          :: na_cols, na_rows  ! local matrix size
   TEST_INT_TYPE                          :: np_cols, np_rows  ! number of MPI processes per column/row
   TEST_INT_TYPE                          :: my_prow, my_pcol  ! local MPI task position (my_prow, my_pcol) in the grid (0..np_cols -1, 0..np_rows -1)
   TEST_INT_MPI_TYPE                      :: mpierr, blacs_ok_mpi

   ! blacs
   character(len=1)                       :: layout
   TEST_INT_TYPE                          :: my_blacs_ctxt, sc_desc(9), info, blacs_ok

   ! The Matrix
   MATRIX_TYPE, allocatable, target       :: a(:,:), as(:,:)
   MATRIX_TYPE, allocatable               :: c(:,:) ! = a^{-1}*as - should be a unit matrix
   

   TEST_INT_TYPE                          :: status, i, j
   integer(kind=c_int)                    :: error_elpa
#ifdef WITH_OPENMP_TRADITIONAL
   TEST_INT_TYPE                          :: max_threads
#endif

   type(output_t)                         :: write_to_file
   class(elpa_t), pointer                 :: e

   logical                                :: successGPU

#if TEST_GPU_DEVICE_POINTER_API == 1
   type(c_ptr)                            :: a_dev
#endif
#if TEST_GPU_SET_ID == 1
   TEST_INT_TYPE                          :: numberOfDevices
   TEST_INT_TYPE                          :: gpuID
#endif
   
   logical                                :: skip_check_correctness

! for gpu_malloc
#if TEST_GPU == 1
#if TEST_REAL == 1
#if TEST_DOUBLE
   integer(kind=c_intptr_t), parameter :: size_of_datatype      = size_of_double_real
#endif
#if TEST_SINGLE
   integer(kind=c_intptr_t), parameter :: size_of_datatype      = size_of_single_real
#endif
#endif /* TEST_REAL == 1 */

#if TEST_COMPLEX == 1
#if TEST_DOUBLE
   integer(kind=c_intptr_t), parameter :: size_of_datatype      = size_of_double_complex
#endif
#if TEST_SINGLE
   integer(kind=c_intptr_t), parameter :: size_of_datatype      = size_of_single_complex
#endif
#endif
#endif /* TEST_GPU == 1 */


   call read_input_parameters_traditional(na, nev, nblk, write_to_file, skip_check_correctness)
   call setup_mpi(myid, nprocs)
#ifdef HAVE_REDIRECT
#ifdef WITH_MPI
   call MPI_BARRIER(MPI_COMM_WORLD, mpierr)
   call redirect_stdout(myid)
#endif
#endif

#ifdef WITH_CUDA_AWARE_MPI
#if TEST_NVIDIA_GPU != 1
#ifdef WITH_MPI
     call mpi_finalize(mpierr)
#endif
     stop 77
#endif
#ifdef TEST_COMPLEX
#ifdef WITH_MPI
     call mpi_finalize(mpierr)
#endif
     stop 77
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
                                na_rows, na_cols, sc_desc, my_blacs_ctxt, info, blacs_ok)
#ifdef WITH_MPI
   blacs_ok_mpi = int(blacs_ok, kind=INT_MPI_TYPE)
   call mpi_allreduce(MPI_IN_PLACE, blacs_ok_mpi, 1_MPI_KIND, MPI_INTEGER, MPI_MIN, int(MPI_COMM_WORLD,kind=MPI_KIND), mpierr)
   blacs_ok = int(blacs_ok_mpi, kind=INT_TYPE)
#endif

   if (blacs_ok .eq. 0) then
     if (myid .eq. 0) then
       print *," Encountered critical error when setting up blacs. Aborting..."
     endif
#ifdef WITH_MPI
     call mpi_finalize(mpierr)
#endif
     stop 1
   endif

	! Allocate the matrices needed for elpa 
   
   allocate(a (na_rows,na_cols))
   allocate(as(na_rows,na_cols))
   allocate(c (na_rows,na_cols))
   
   a(:,:) = 0.0
   c(:,:) = 0.0
   
   call prepare_matrix_random_triangular (na, a, nblk, myid, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol)
   !call print_matrix(myid, na_rows, a, "a") ! DEBUG; print_matrix prints only for myid=0 and works only for square local matrices
   as(:,:) = a(:,:)
   
   call prepare_matrix_unit (na, c, nblk, myid, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol)
   !call print_matrix(myid, na_rows, c, "c")
   
   
   e => elpa_allocate(error_elpa)
   assert_elpa_ok(error_elpa)

	! Set parameters
	
   call e%set("na", int(na,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("nev", int(nev,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("local_nrows", int(na_rows,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("local_ncols", int(na_cols,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("nblk", int(nblk,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   
#ifdef WITH_MPI
     call e%set("mpi_comm_parent", int(MPI_COMM_WORLD,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
     call e%set("process_row", int(my_prow,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
     call e%set("process_col", int(my_pcol,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif

   call e%set("timings",1, error_elpa)
   assert_elpa_ok(error_elpa)

   call e%set("debug",1,error_elpa)
   assert_elpa_ok(error_elpa)

   assert_elpa_ok(e%setup())

#if TEST_NVIDIA_GPU == 1
   call e%set("nvidia-gpu", TEST_GPU,error_elpa)
   assert_elpa_ok(error_elpa)
#endif
#if TEST_AMD_GPU == 1
   call e%set("amd-gpu", TEST_GPU,error_elpa)
   assert_elpa_ok(error_elpa)
#endif
#if TEST_INTEL_GPU == 1 || (TEST_INTEL_GPU_OPENMP == 1) || (TEST_INTEL_GPU_SYCL == 1)
   call e%set("intel-gpu", TEST_GPU,error_elpa)
   assert_elpa_ok(error_elpa)
#endif

#if defined(TEST_NVIDIA_GPU) || defined(TEST_AMD_GPU) || defined(TEST_INTEL_GPU) || defined(TEST_INTEL_GPU_OPENMP) || defined(TEST_INTEL_GPU_SYCL)
   assert_elpa_ok(e%setup_gpu())
#endif 

#if (TEST_GPU_SET_ID == 1) && (TEST_INTEL_GPU == 0) && (TEST_INTEL_GPU_OPENMP == 0) && (TEST_INTEL_GPU_SYCL == 0)
   if (gpu_vendor() /= no_gpu) then
      call set_gpu_parameters()
   else 
      print *,"Cannot set gpu vendor!"
      stop 1
   endif

   successGPU = gpu_GetDeviceCount(numberOfDevices)
   if (.not.(successGPU)) then
      print *,"Error in gpu_GetDeviceCount. Aborting..."
      stop 1
   endif
   gpuID = mod(myid, numberOfDevices)

   call e%set("use_gpu_id", int(gpuID,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
#endif

#if TEST_GPU_DEVICE_POINTER_API == 1
   ! create device pointers for a,q, ev; copy a to device
     
   if (gpu_vendor() /= no_gpu) then
     call set_gpu_parameters()
   else 
      print *,"Cannot set gpu vendor!"
      stop 1
   endif

   ! Set device 
   successGPU = .true.        
#if TEST_INTEL_GPU_SYCL == 1
   successGPU = sycl_getcpucount(numberOfDevices) ! temporary fix for SYCL on CPU
   if (.not.(successGPU)) then
      print *,"Error in sycl_getcpucount. Aborting..."
      stop 1
    endif
#endif

   successGPU = gpu_setdevice(gpuID)
   if (.not.(successGPU)) then
     print *,"Cannot set GPU device. Aborting..."
     stop 1
   endif

   successGPU = gpu_malloc(a_dev, na_rows*na_cols*size_of_datatype)
   if (.not.(successGPU)) then
     print *,"Cannot allocate matrix a on GPU! Aborting..."
     stop 1
   endif
   
   successGPU = gpu_memcpy(a_dev, c_loc(a), na_rows*na_cols*size_of_datatype, &
                           gpuMemcpyHostToDevice)
   if (.not.(successGPU)) then
     print *,"Cannot copy matrix a to GPU! Aborting..."
     stop 1
   endif
   
#endif /* TEST_GPU_DEVICE_POINTER_API */

   !-----------------------------------------------------------------------------------------------------------------------------
   ! The actual solve step
   call e%timer_start("e%triangular")

#ifdef TEST_EXPLICIT_NAME
   if (myid == 0) then
     print *,"Inverting with TEST_EXPLICIT_NAME"
   endif
#if defined(TEST_REAL)
#if defined(TEST_DOUBLE)
#if (TEST_GPU_DEVICE_POINTER_API == 1)
   if (myid == 0) then
     print *, "Inverting with device API"
   endif
   call e%invert_triangular_double(a_dev, error_elpa)
   assert_elpa_ok(error_elpa)
#else
   call e%invert_triangular_double(a, error_elpa)
   assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_DOUBLE */
#if defined(TEST_SINGLE)
#if (TEST_GPU_DEVICE_POINTER_API == 1)
   if (myid == 0) then
     print *, "Inverting with device API"
   endif
   call e%invert_triangular_float(a_dev, error_elpa)
   assert_elpa_ok(error_elpa)
#else
   call e%invert_triangular_float(a, error_elpa)
   assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_SINGLE */
#endif /* TEST_REAL */

#if defined(TEST_COMPLEX)
#if defined(TEST_DOUBLE)
#if (TEST_GPU_DEVICE_POINTER_API == 1)
   if (myid == 0) then
     print *, "Inverting with device API"
   endif
   call e%invert_triangular_double_complex(a_dev, error_elpa)
   assert_elpa_ok(error_elpa)
#else
   call e%invert_triangular_double_complex(a, error_elpa)
   assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_DOUBLE */
#if defined(TEST_SINGLE)
#if (TEST_GPU_DEVICE_POINTER_API == 1)
   if (myid == 0) then
     print *, "Inverting with device API"
   endif
   call e%invert_triangular_float_complex(a_dev, error_elpa)
   assert_elpa_ok(error_elpa)
#else
   call e%invert_triangular_float_complex(a, error_elpa)
   assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_SINGLE */
#endif /* TEST_COMPLEX */

#else /* TEST_EXPLICIT_NAME */
   if (myid == 0) then
     print *, "Inverting without TEST_EXPLICIT_NAME"
   endif
   call e%invert_triangular (a, error_elpa)
   assert_elpa_ok(error_elpa)
#endif /* TEST_EXPLICIT_NAME */

   call e%timer_stop("e%triangular")


   !call print_matrix(myid, na_rows, a, "a_inverted")

   !-----------------------------------------------------------------------------------------------------------------------------     
   ! TEST_GPU == 1: copy for testing from device to host, deallocate device pointers
#if TEST_GPU_DEVICE_POINTER_API == 1

   ! copy for testing
   successGPU = gpu_memcpy(c_loc(a), a_dev, na_rows*na_cols*size_of_datatype, &
                           gpuMemcpyDeviceToHost)
   if (.not.(successGPU)) then
     print *,"cannot copy matrix of eigenvectors from GPU to host! Aborting..."
     stop 1
   endif

   ! and deallocate device pointer
   successGPU = gpu_free(a_dev)
   if (.not.(successGPU)) then
     print *,"cannot free memory of a_dev on GPU. Aborting..."
     stop 1
   endif

#endif /* TEST_GPU_DEVICE_POINTER_API */

   if (myid .eq. 0) then
     call e%print_times("e%triangular")
   endif
   !-----------------------------------------------------------------------------------------------------------------------------
   ! Check the results
   
   if(.not. skip_check_correctness) then
     status = check_correctness_hermitian_multiply("N", na, a, as, c, na_rows, sc_desc, myid )
     call check_status(status, myid)
   endif

   !-----------------------------------------------------------------------------------------------------------------------------
   ! Deallocate
   
   call elpa_deallocate(e, error_elpa)
   assert_elpa_ok(error_elpa)
   
   deallocate(a)
   deallocate(as)
   deallocate(c)
   

   call elpa_uninit(error_elpa)


   if (myid == 0) then
     print *, "Done!"
   endif

#ifdef WITH_MPI
   call blacs_gridexit(my_blacs_ctxt)
   call mpi_finalize(mpierr)
#endif

   call exit(status)

   !-----------------------------------------------------------------------------------------------------------------------------

contains
   
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
