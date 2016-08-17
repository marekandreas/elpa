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
!>
!> Fortran test programm to demonstrates the use of
!> the elpa_solve_tridi library function
!>
program test_solve_tridi

!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
!-------------------------------------------------------------------------------
   use precision
   use ELPA1
   use elpa_utilities
#ifdef WITH_OPENMP
   use test_util
#endif

   use mod_read_input_parameters
   use mod_check_correctness
   use mod_setup_mpi
   use mod_blacs_infrastructure
   use mod_prepare_matrix

   use elpa_mpi
#ifdef HAVE_REDIRECT
   use redirect
#endif
#ifdef HAVE_DETAILED_TIMINGS
  use timings
#endif
  use output_types

   implicit none

   !-------------------------------------------------------------------------------
   ! Please set system size parameters below!
   ! na:   System size
   ! nev:  Number of eigenvectors to be calculated
   ! nblk: Blocking factor in block cyclic distribution
   !-------------------------------------------------------------------------------
   integer(kind=ik)           :: nblk
   integer(kind=ik)           :: na, nev

   integer(kind=ik)           :: np_rows, np_cols, na_rows, na_cols

   integer(kind=ik)           :: myid, nprocs, my_prow, my_pcol, mpi_comm_rows, mpi_comm_cols
   integer(kind=ik)           :: i, mpierr, my_blacs_ctxt, sc_desc(9), info, nprow, npcol

   integer, external          :: numroc

   real(kind=rk8), allocatable :: a(:,:), d(:), e(:), ev_analytic(:), ev(:)
   real(kind=rk8)              :: diagonalELement, subdiagonalElement

   real(kind=rk8), allocatable :: tmp1(:,:), tmp2(:,:), as(:,:)
   real(kind=rk8)              :: tmp
   integer(kind=ik)           :: loctmp ,rowLocal, colLocal


   real(kind=rk8)              :: maxerr

   logical                    :: wantDebug

   real(kind=rk8), parameter   :: pi = 3.141592653589793238462643383279_rk8

   integer(kind=ik)           :: iseed(4096) ! Random seed, size should be sufficient for every generator

   integer(kind=ik)           :: STATUS
#ifdef WITH_OPENMP
   integer(kind=ik)           :: omp_get_max_threads,  required_mpi_thread_level, &
                                 provided_mpi_thread_level
#endif
   type(output_t)             :: write_to_file
   logical                    :: success
   character(len=8)           :: task_suffix
   integer(kind=ik)           :: j
   !-------------------------------------------------------------------------------

   success = .true.

   call read_input_parameters(na, nev, nblk, write_to_file)

   !-------------------------------------------------------------------------------
   !  MPI Initialization
   call setup_mpi(myid, nprocs)

   STATUS = 0

!#define DATATYPE REAL
!#define ELPA1
!#include "elpa_print_headers.X90"

#ifdef HAVE_DETAILED_TIMINGS

   ! initialise the timing functionality

#ifdef HAVE_LIBPAPI
   call timer%measure_flops(.true.)
#endif

   call timer%measure_allocated_memory(.true.)
   call timer%measure_virtual_memory(.true.)
   call timer%measure_max_allocated_memory(.true.)

   call timer%set_print_options(&
#ifdef HAVE_LIBPAPI
                print_flop_count=.true., &
                print_flop_rate=.true., &
#endif
                print_allocated_memory = .true. , &
                print_virtual_memory=.true., &
                print_max_allocated_memory=.true.)


  call timer%enable()

  call timer%start("program")
#endif

   do np_cols = NINT(SQRT(REAL(nprocs))),2,-1
      if(mod(nprocs,np_cols) == 0 ) exit
   enddo

   ! at the end of the above loop, nprocs is always divisible by np_cols

   np_rows = nprocs/np_cols

   if(myid==0) then
      print *
      print '(a)','Test program for elpa_solve_tridi with a Toeplitz matrix'
      print *
      print '(3(a,i0))','Matrix size=',na,', Number of eigenvectors=',nev,', Block size=',nblk
      print '(3(a,i0))','Number of processor rows=',np_rows,', cols=',np_cols,', total=',nprocs
      print *
   endif

   !-------------------------------------------------------------------------------
   ! Set up BLACS context and MPI communicators
   !
   ! The BLACS context is only necessary for using Scalapack.
   !
   ! For ELPA, the MPI communicators along rows/cols are sufficient,
   ! and the grid setup may be done in an arbitrary way as long as it is
   ! consistent (i.e. 0<=my_prow<np_rows, 0<=my_pcol<np_cols and every
   ! process has a unique (my_prow,my_pcol) pair).

   call set_up_blacsgrid(mpi_comm_world, my_blacs_ctxt, np_rows, np_cols, &
                         nprow, npcol, my_prow, my_pcol)

   if (myid==0) then
     print '(a)','| Past BLACS_Gridinfo.'
   end if

   ! All ELPA routines need MPI communicators for communicating within
   ! rows or columns of processes, these are set in get_elpa_communicators.

   mpierr = get_elpa_communicators(mpi_comm_world, my_prow, my_pcol, &
                                   mpi_comm_rows, mpi_comm_cols)

   if (myid==0) then
     print '(a)','| Past split communicator setup for rows and columns.'
   end if

   call set_up_blacs_descriptor(na ,nblk, my_prow, my_pcol, np_rows, np_cols, &
                                na_rows, na_cols, sc_desc, my_blacs_ctxt, info)

   if (myid==0) then
     print '(a)','| Past scalapack descriptor setup.'
   end if

   !-------------------------------------------------------------------------------
   ! Allocate matrices and set up a test toeplitz matrix for solve_tridi

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("set up matrix")
#endif
   allocate(a (na_rows,na_cols))
   allocate(as(na_rows,na_cols))

   allocate(d (na))
   allocate(e (na))
   allocate(ev_analytic(na))
   allocate(ev(na))

   a(:,:) = 0.0_rk8


   ! changeable numbers here would be nice
   diagonalElement = 0.45_rk8
   subdiagonalElement =  0.78_rk8

   d(:) = diagonalElement
   e(:) = subdiagonalElement


   ! set up the diagonal and subdiagonals (for general solver test)
   do i=1, na ! for diagonal elements
     if (map_global_array_index_to_local_index(i, i, rowLocal, colLocal, nblk, np_rows, np_cols, my_prow, my_pcol)) then
       a(rowLocal,colLocal) = diagonalElement
     endif
   enddo

   do i=1, na-1
     if (map_global_array_index_to_local_index(i, i+1, rowLocal, colLocal, nblk, np_rows, np_cols, my_prow, my_pcol)) then
       a(rowLocal,colLocal) = subdiagonalElement
     endif
   enddo

   do i=2, na
     if (map_global_array_index_to_local_index(i, i-1, rowLocal, colLocal, nblk, np_rows, np_cols, my_prow, my_pcol)) then
       a(rowLocal,colLocal) = subdiagonalElement
     endif
   enddo

   as = a

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("set up matrix")
#endif


   !-------------------------------------------------------------------------------
   ! Calculate eigenvalues/eigenvectors

   if (myid==0) then
     print '(a)','| Entering elpa_solve_tridi ... '
     print *
   end if

#ifdef WITH_MPI
   call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif

   success = elpa_solve_tridi_double(na, nev, d, e, a, na_rows, nblk, na_cols, mpi_comm_rows, &
                              mpi_comm_cols, wantDebug)

   if (.not.(success)) then
      write(error_unit,*) "elpa_solve_tridi produced an error! Aborting..."
#ifdef WITH_MPI
      call MPI_ABORT(mpi_comm_world, 1, mpierr)
#endif
   endif

   if (myid==0) then
     print '(a)','| elpa_solve_tridi complete.'
     print *
   end if


   ev = d

!
!   if(myid == 0) print *,'Time tridiag_real     :',time_evp_fwd
!   if(myid == 0) print *,'Time solve_tridi      :',time_evp_solve
!   if(myid == 0) print *,'Time trans_ev_real    :',time_evp_back
!   if(myid == 0) print *,'Total time (sum above):',time_evp_back+time_evp_solve+time_evp_fwd
!
!   if(write_to_file%eigenvectors) then
!     write(unit = task_suffix, fmt = '(i8.8)') myid
!     open(17,file="EVs_real_out_task_"//task_suffix(1:8)//".txt",form='formatted',status='new')
!     write(17,*) "Part of eigenvectors: na_rows=",na_rows,"of na=",na," na_cols=",na_cols," of na=",na
!
!     do i=1,na_rows
!       do j=1,na_cols
!         write(17,*) "row=",i," col=",j," element of eigenvector=",z(i,j)
!       enddo
!     enddo
!     close(17)
!   endif
!
!   if(write_to_file%eigenvalues) then
!      if (myid == 0) then
!         open(17,file="Eigenvalues_real_out.txt",form='formatted',status='new')
!         do i=1,na
!            write(17,*) i,ev(i)
!         enddo
!         close(17)
!      endif
!   endif
!
!
   !-------------------------------------------------------------------------------

   ! analytic solution
   do i=1, na
     ev_analytic(i) = diagonalElement + 2.0_rk8 * subdiagonalElement *cos( pi*real(i,kind=rk8)/ real(na+1,kind=rk8) )
   enddo

   ! sort analytic solution:

   ! this hack is neihter elegant, nor optimized: for huge matrixes it might be expensive
   ! a proper sorting algorithmus might be implemented here

   tmp    = minval(ev_analytic)
   loctmp = minloc(ev_analytic, 1)

   ev_analytic(loctmp) = ev_analytic(1)
   ev_analytic(1) = tmp

   do i=2, na
     tmp = ev_analytic(i)
     do j= i, na
       if (ev_analytic(j) .lt. tmp) then
         tmp    = ev_analytic(j)
         loctmp = j
       endif
     enddo
     ev_analytic(loctmp) = ev_analytic(i)
     ev_analytic(i) = tmp
   enddo

   ! compute a simple error max of eigenvalues
   maxerr = 0.0_rk8
   maxerr = maxval( (d(:) - ev_analytic(:))/ev_analytic(:) , 1)

   if (maxerr .gt. 8.e-13) then
     if (myid .eq. 0) then
       print *,"Eigenvalues differ from analytic solution: maxerr = ",maxerr
     endif

   status = 1

   endif

   ! Test correctness of result (using plain scalapack routines)
   allocate(tmp1(na_rows,na_cols))
   allocate(tmp2(na_rows,na_cols))

   status = check_correctness(na, nev, as, a, ev, sc_desc, myid, tmp1, tmp2)

   deallocate(a)

   deallocate(as)
   deallocate(d)

   deallocate(tmp1)
   deallocate(tmp2)
   deallocate(e)
   deallocate(ev)
   deallocate(ev_analytic)

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("program")
   print *," "
   print *,"Timings program:"
   print *," "
   call timer%print("program")
   print *," "
   print *,"End timings program"
   print *," "
#endif

#ifdef WITH_MPI
   call blacs_gridexit(my_blacs_ctxt)
   call mpi_finalize(mpierr)
#endif

   call EXIT(STATUS)


end

!-------------------------------------------------------------------------------
