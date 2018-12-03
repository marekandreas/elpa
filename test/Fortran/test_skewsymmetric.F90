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
! Define TEST_GPU \in [0, 1]
! Define either TEST_ALL_KERNELS or a TEST_KERNEL \in [any valid kernel]

#if !(defined(TEST_REAL) ^ defined(TEST_COMPLEX))
error: define exactly one of TEST_REAL or TEST_COMPLEX
#endif

#if !(defined(TEST_SINGLE) ^ defined(TEST_DOUBLE))
error: define exactly one of TEST_SINGLE or TEST_DOUBLE
#endif

#ifdef TEST_SINGLE
#  define EV_TYPE real(kind=C_FLOAT)
#  ifdef TEST_REAL
#    define MATRIX_TYPE real(kind=C_FLOAT)
#  else
#    define MATRIX_TYPE complex(kind=C_FLOAT_COMPLEX)
#  endif
#else
#  define EV_TYPE real(kind=C_DOUBLE)
#  ifdef TEST_REAL
#    define MATRIX_TYPE real(kind=C_DOUBLE)
#  else
#    define MATRIX_TYPE complex(kind=C_DOUBLE_COMPLEX)
#  endif
#endif

#ifdef TEST_SINGLE
#define MATRIX_TYPE_COMPLEX complex(kind=C_FLOAT_COMPLEX)
#define EV_TYPE_COMPLEX complex(kind=C_FLOAT_COMPLEX)
#else
#define MATRIX_TYPE_COMPLEX complex(kind=C_DOUBLE_COMPLEX)
#define EV_TYPE_COMPLEX complex(kind=C_DOUBLE_COMPLEX)
#endif

#ifdef TEST_REAL
#  define AUTOTUNE_DOMAIN ELPA_AUTOTUNE_DOMAIN_REAL
#else
#  define AUTOTUNE_DOMAIN ELPA_AUTOTUNE_DOMAIN_COMPLEX
#endif

#include "assert.h"

program test
   use elpa

   use test_util
   use test_setup_mpi
   use test_prepare_matrix
   use test_read_input_parameters
   use test_blacs_infrastructure
   use test_check_correctness
   use iso_fortran_env

#ifdef HAVE_REDIRECT
   use test_redirect
#endif
   implicit none

   ! matrix dimensions
   integer                     :: na, nev, nblk

   ! mpi
   integer                     :: myid, nprocs
   integer                     :: na_cols, na_rows  ! local matrix size
   integer                     :: np_cols, np_rows  ! number of MPI processes per column/row
   integer                     :: my_prow, my_pcol  ! local MPI task position (my_prow, my_pcol) in the grid (0..np_cols -1, 0..np_rows -1)
   integer                     :: mpierr, ierr

   ! blacs
   character(len=1)            :: layout
   integer                     :: my_blacs_ctxt, sc_desc(9), info, nprow, npcol

   ! The Matrix
   MATRIX_TYPE, allocatable    :: a_skewsymmetric(:,:), as_skewsymmetric(:,:)
   MATRIX_TYPE_COMPLEX, allocatable    :: a_complex(:,:), as_complex(:,:)
   ! eigenvectors
   MATRIX_TYPE, allocatable    :: z_skewsymmetric(:,:)
   MATRIX_TYPE_COMPLEX, allocatable    :: z_complex(:,:)
   ! eigenvalues
   EV_TYPE, allocatable:: ev_skewsymmetric(:), ev_complex(:)

   integer                     :: error, status, i, j

   type(output_t)              :: write_to_file
   class(elpa_t), pointer      :: e_complex, e_skewsymmetric
           
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

   call set_up_blacsgrid(mpi_comm_world, np_rows, np_cols, layout, &
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
   
   call MPI_BARRIER(MPI_COMM_WORLD, ierr)  
   as_skewsymmetric(:,:) = a_skewsymmetric(:,:)
   

   ! prepare the complex matrix for the "brute force" case
   allocate(a_complex (na_rows,na_cols))
   allocate(as_complex(na_rows,na_cols))
   allocate(z_complex (na_rows,na_cols))
   allocate(ev_complex(na))

   a_complex(:,:) = 0.0
   z_complex(:,:) = 0.0
   as_complex(:,:) = 0.0
   

      do j=1, na_cols
         do i=1,na_rows
               a_complex(i,j) = dcmplx(0.0, a_skewsymmetric(i,j))
         enddo
      enddo
   
   z_complex(:,:)  = a_complex(:,:)
   as_complex(:,:) = a_complex(:,:)

   ! first set up and solve the brute force problem
   e_complex => elpa_allocate()
   call set_basic_params(e_complex, na, nev, na_rows, na_cols, my_prow, my_pcol)

   call e_complex%set("timings",1, error)

   call e_complex%set("debug",1)
   call e_complex%set("gpu", 0)

   assert_elpa_ok(e_complex%setup())
   call e_complex%set("solver", elpa_solver_2stage, error)

   call e_complex%timer_start("eigenvectors: brute force ")
   call e_complex%eigenvectors(a_complex, ev_complex, z_complex, error)
   call e_complex%timer_stop("eigenvectors: brute force ")

   if (myid .eq. 0) then
     print *, ""
!      call e_complex%print_times("eigenvectors: brute force")
   endif 

   status = check_correctness_evp_numeric_residuals(na, nev, as_complex, z_complex, ev_complex, sc_desc, &
                                                    nblk, myid, np_rows,np_cols, my_prow, my_pcol)
   call check_status(status, myid)

#ifdef WITH_MPI
     call MPI_BARRIER(MPI_COMM_WORLD, ierr)
#endif
   ! now run the skewsymmetric case
   e_skewsymmetric => elpa_allocate()
   call set_basic_params(e_skewsymmetric, na, nev, na_rows, na_cols, my_prow, my_pcol)

   call e_skewsymmetric%set("timings",1, error)

   call e_skewsymmetric%set("debug",1)
   call e_skewsymmetric%set("gpu", 0)

   call e_skewsymmetric%set("is_skewsymmetric",1)
   assert_elpa_ok(e_skewsymmetric%setup())
   
   call e_skewsymmetric%set("solver", elpa_solver_2stage, error)

   call e_skewsymmetric%get("is_skewsymmetric", i,error)
   
   call e_skewsymmetric%timer_start("eigenvectors: skewsymmetric ")
   call e_skewsymmetric%eigenvectors(a_skewsymmetric, ev_skewsymmetric, z_skewsymmetric, error)
   call e_skewsymmetric%timer_stop("eigenvectors: skewsymmetric ")

   if (myid .eq. 0) then
     print *, ""
!      call e_skewsymmetric%print_times("eigenvectors: skewsymmetric")
   endif
   
   
   ! check eigenvalues
   do i=1, na
     if (myid == 0) then
!          print *,"ev(", i,")=",ev_skewsymmetric(i)
       if (abs(ev_complex(i)-ev_skewsymmetric(i))/abs(ev_complex(i)) .gt. 1e-10) then
         print *,"ev: i=",i,ev_complex(i),ev_skewsymmetric(i)
         status = 1
     endif
     endif
   enddo
   call check_status(status, myid)
   
   z_complex(:,:) = 0
   do j=1, na_cols
     do i=1,na_rows
       z_complex(i,j) = dcmplx(z_skewsymmetric(i,j), z_skewsymmetric(i,na_cols+j))
     enddo
   enddo
   call MPI_BARRIER(MPI_COMM_WORLD, ierr)
   
   status = check_correctness_evp_numeric_residuals_ss(na, nev, as_skewsymmetric, z_complex, ev_skewsymmetric, &
                              sc_desc, nblk, myid, np_rows,np_cols, my_prow, my_pcol)

   
#ifdef WITH_MPI
!    call MPI_BARRIER(MPI_COMM_WORLD, ierr)
#endif
   call elpa_deallocate(e_complex)
   call elpa_deallocate(e_skewsymmetric)


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
   call elpa_uninit()



#ifdef WITH_MPI
   call blacs_gridexit(my_blacs_ctxt)
   call mpi_finalize(mpierr)
#endif

   call exit(status)

contains
   subroutine set_basic_params(elpa, na, nev, na_rows, na_cols, my_prow, my_pcol)
     implicit none
     class(elpa_t), pointer      :: elpa
     integer, intent(in)         :: na, nev, na_rows, na_cols, my_prow, my_pcol

     call elpa%set("na", na, error)
     assert_elpa_ok(error)
     call elpa%set("nev", nev, error)
     assert_elpa_ok(error)
     call elpa%set("local_nrows", na_rows, error)
     assert_elpa_ok(error)
     call elpa%set("local_ncols", na_cols, error)
     assert_elpa_ok(error)
     call elpa%set("nblk", nblk, error)
     assert_elpa_ok(error)

#ifdef WITH_MPI
     call elpa%set("mpi_comm_parent", MPI_COMM_WORLD, error)
     assert_elpa_ok(error)
     call elpa%set("process_row", my_prow, error)
     assert_elpa_ok(error)
     call elpa%set("process_col", my_pcol, error)
     assert_elpa_ok(error)
#endif
   end subroutine
   subroutine check_status(status, myid)
     implicit none
     integer, intent(in) :: status, myid
     integer :: mpierr
     if (status /= 0) then
       if (myid == 0) print *, "Result incorrect!"
#ifdef WITH_MPI
       call mpi_finalize(mpierr)
#endif
       call exit(status)
     endif
   end subroutine
end program
