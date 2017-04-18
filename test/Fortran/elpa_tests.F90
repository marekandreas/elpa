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

program test_all_real

!-------------------------------------------------------------------------------
! Standard eigenvalueGproblem - REAL version
!
! This program demonstrates the use of the ELPA module
! together with standard scalapack routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
!-------------------------------------------------------------------------------
   use precision
   use elpa1_legacy
   use elpa2_legacy
   use elpa_utilities, only : error_unit, map_global_array_index_to_local_index
   use elpa2_utilities
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
   integer(kind=ik)                       :: nblk
   integer(kind=ik)                       :: na, nev

   integer(kind=ik)                       :: np_rows, np_cols, na_rows, na_cols

   integer(kind=ik)                       :: myid, nprocs, my_prow, my_pcol, mpi_comm_rows, mpi_comm_cols
   integer(kind=ik)                       :: i, mpierr, my_blacs_ctxt, sc_desc(9), info, nprow, npcol

   integer, external                      :: numroc

   logical                                :: wantDebug
   real(kind=rk8), allocatable            :: ev(:)
   real(kind=rk8), allocatable, target    :: a_real(:,:), z_real(:,:), tmp1_real(:,:), tmp2_real(:,:), as_real(:,:)
   complex(kind=ck8), allocatable, target :: a_complex(:,:), z_complex(:,:), tmp1_complex(:,:), tmp2_complex(:,:), as_complex(:,:)

   real(kind=rk8), allocatable, target    :: b_real(:,:), bs_real(:,:), c_real(:,:)
   complex(kind=ck8), allocatable, target :: b_complex(:,:), bs_complex(:,:), c_complex(:,:)

   complex(kind=ck8), parameter           :: CZERO = (0.0_rk8,0.0_rk8), CONE = (1.0_rk8,0.0_rk8)
   real(kind=rk8), allocatable, target    :: d_real(:), e_real(:), ev_analytic_real(:)
   real(kind=rk8),  target                :: diagonalELement_real, subdiagonalElement_real
   complex(kind=ck8)                      :: diagonalElement_complex
   real(kind=rk8), target                 :: tmp
   real(kind=rk8)                         :: norm, normmax

   real(kind=rk8), parameter              :: pi = 3.141592653589793238462643383279_rk8
   integer(kind=ik)                       :: STATUS
#ifdef WITH_OPENMP
   integer(kind=ik)                       :: omp_get_max_threads,  required_mpi_thread_level, &
                                             provided_mpi_thread_level
#endif
   type(input_options_t)                  :: input_options
   logical                                :: success
   character(len=8)                       :: task_suffix
   integer(kind=ik)                       :: j, this_kernel
   logical                                :: this_gpu
   logical                                :: this_qr
   integer(kind=ik)                       :: loctmp ,rowLocal, colLocal
   real(kind=rk8)                         :: tStart, tEnd
   real(kind=rk8)                         :: maxerr
   logical                                :: gpuAvailable
#ifdef WITH_MPI
   real(kind=rk8)                         :: pzlange, pdlange
#else
   real(kind=rk8)                         :: zlange, dlange
#endif

   !-------------------------------------------------------------------------------


   success = .true.

   call read_input_parameters(input_options)

   na =  input_options%na
   nev = input_options%nev
   nblk = input_options%nblk

   if (input_options%justHelpMessage) then
     call EXIT(0)
   endif


   !-------------------------------------------------------------------------------
   !  MPI Initialization
   call setup_mpi(myid, nprocs)


   STATUS = 0

!#define DATATYPE REAL !!! check here
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
      if (input_options%datatype .eq. 1) then
        print '(a)','Standard eigenvalue problem - REAL version'
      else if (input_options%datatype .eq. 2) then
        print '(a)','Standard eigenvalue problem - COMPLEX version'
      endif
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

   mpierr = elpa_get_communicators(mpi_comm_world, my_prow, my_pcol, &
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
   ! Allocate matrices and set up a test matrix for the eigenvalue problem
#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("allocate arrays")
#endif

   if (input_options%datatype .eq. 1) then
     allocate(a_real (na_rows,na_cols))
     allocate(z_real (na_rows,na_cols))
     allocate(as_real(na_rows,na_cols))

     if (input_options%doInvertTrm .or. input_options%doTransposeMultiply) then
       allocate(b_real(na_rows,na_cols))
       allocate(bs_real(na_rows,na_cols))
     endif

     if (input_options%doTransposeMultiply) then
       allocate(c_real(na_rows,na_cols))
     endif
   endif
   if (input_options%datatype .eq. 2) then
     allocate(a_complex (na_rows,na_cols))
     allocate(z_complex (na_rows,na_cols))
     allocate(as_complex(na_rows,na_cols))

     if (input_options%doInvertTrm .or. input_options%doTransposeMultiply) then
       allocate(b_complex(na_rows,na_cols))
       allocate(bs_complex(na_rows,na_cols))
     endif

     if (input_options%doTransposeMultiply) then
       allocate(c_complex(na_rows,na_cols))
     endif
   endif

   if (input_options%datatype .eq. 1) then
     allocate(d_real (na))
     allocate(e_real (na))
     allocate(ev_analytic_real(na))
   endif

   allocate(ev(na))

   if (input_options%datatype .eq. 1) then
     allocate(tmp1_real(na_rows,na_cols))
     allocate(tmp2_real(na_rows,na_cols))
   endif
   if (input_options%datatype .eq. 2) then
     allocate(tmp1_complex(na_rows,na_cols))
     allocate(tmp2_complex(na_rows,na_cols))
   endif

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("allocate arrays")
#endif


#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("set up matrix")
#endif

   if (input_options%datatype .eq. 1) then
     call prepare_matrix_double(na, myid, sc_desc, a_real, z_real, as_real)

     if (input_options%doInvertTrm) then
       b_real(:,:) = a_real(:,:)
       bs_real(:,:) = a_real(:,:)
     endif
   endif
   if (input_options%datatype .eq. 2) then
     call prepare_matrix_double(na, myid, sc_desc, a_complex, z_complex, as_complex)
     if (input_options%doInvertTrm) then
       b_complex(:,:) = a_complex(:,:)
       bs_complex(:,:) = a_complex(:,:)
     endif

   endif


#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("set up matrix")
#endif

   if (input_options%doSolveTridi) then
     ! first the toeplitz test only in real case
     ! changeable numbers here would be nice
     if (input_options%datatype .eq. 1) then
#ifdef HAVE_DETAILED_TIMINGS
       call timer%start("set up matrix")
#endif

       diagonalElement_real = 0.45_rk8
       subdiagonalElement_real =  0.78_rk8

       d_real(:) = diagonalElement_real
       e_real(:) = subdiagonalElement_real


       ! set up the diagonal and subdiagonals (for general solver test)
       do i=1, na ! for diagonal elements
         if (map_global_array_index_to_local_index(i, i, rowLocal, colLocal, nblk, np_rows, np_cols, my_prow, my_pcol)) then
           a_real(rowLocal,colLocal) = diagonalElement_real
         endif
       enddo

       do i=1, na-1
         if (map_global_array_index_to_local_index(i, i+1, rowLocal, colLocal, nblk, np_rows, np_cols, my_prow, my_pcol)) then
           a_real(rowLocal,colLocal) = subdiagonalElement_real
         endif
       enddo

       do i=2, na
         if (map_global_array_index_to_local_index(i, i-1, rowLocal, colLocal, nblk, np_rows, np_cols, my_prow, my_pcol)) then
           a_real(rowLocal,colLocal) = subdiagonalElement_real
         endif
       enddo

       as_real = a_real

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

       success = elpa_solve_tridi_double(na, nev, d_real, e_real, a_real, na_rows, nblk, na_cols, mpi_comm_rows, &
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


       ev = d_real

       ! analytic solution
       do i=1, na
         ev_analytic_real(i) = diagonalElement_real + 2.0_rk8 * subdiagonalElement_real * &
                                cos( pi*real(i,kind=rk8)/ real(na+1,kind=rk8) )
       enddo

       ! sort analytic solution:

       ! this hack is neither elegant, nor optimized: for huge matrixes it might be expensive
       ! a proper sorting algorithmus might be implemented here

       tmp    = minval(ev_analytic_real)
       loctmp = minloc(ev_analytic_real, 1)

       ev_analytic_real(loctmp) = ev_analytic_real(1)
       ev_analytic_real(1) = tmp

       do i=2, na
         tmp = ev_analytic_real(i)
         do j= i, na
           if (ev_analytic_real(j) .lt. tmp) then
             tmp    = ev_analytic_real(j)
             loctmp = j
           endif
         enddo
         ev_analytic_real(loctmp) = ev_analytic_real(i)
         ev_analytic_real(i) = tmp
       enddo

       ! compute a simple error max of eigenvalues
       maxerr = 0.0_rk8
       maxerr = maxval( (d_real(:) - ev_analytic_real(:))/ev_analytic_real(:) , 1)

       if (maxerr .gt. 8.e-13_rk8) then
         if (myid .eq. 0) then
           print *,"Eigenvalues differ from analytic solution: maxerr = ",maxerr
         endif

         status = 1

       endif
     endif ! datatype == 1 for toeplitz
    endif ! doSolve-tridi

    if (input_options%doCholesky) then
      if (input_options%datatype .eq. 1) then
        a_real(:,:) = 0.0_rk8
        diagonalElement_real = 2.546_rk8

        do i = 1, na
          if (map_global_array_index_to_local_index(i, i, rowLocal, colLocal, nblk, np_rows, np_cols, my_prow, my_pcol)) then
            a_real(rowLocal,colLocal) = diagonalElement_real * abs(cos( pi*real(i,kind=rk8)/ real(na+1,kind=rk8) ))
          endif
        enddo

        as_real(:,:) = a_real(:,:)

        if (myid==0) then
          print '(a)','| Compute real cholesky decomposition ... '
          print *
        end if
#ifdef WITH_MPI
        call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif

        success = elpa_cholesky_real_double(na, a_real, na_rows, nblk, na_cols, mpi_comm_rows, mpi_comm_cols, .true.)

        if (.not.(success)) then
          write(error_unit,*) "elpa_cholseky_real produced an error! Aborting..."
#ifdef WITH_MPI
          call MPI_ABORT(mpi_comm_world, 1, mpierr)
#endif
        endif


        if (myid==0) then
          print '(a)','| Real cholesky decomposition complete.'
          print *
        end if

        tmp1_real(:,:) = 0.0_rk8
#ifdef WITH_MPI
        call pdtran(na, na, 1.0_rk8, a_real, 1, 1, sc_desc, 0.0_rk8, tmp1_real, 1, 1, sc_desc)
#else
        tmp1_real = transpose(a_real)
#endif
        ! tmp2 = a * a**T
#ifdef WITH_MPI
        call pdgemm("N","N", na, na, na, 1.0_rk8, a_real, 1, 1, sc_desc, tmp1_real, 1, 1, &
                    sc_desc, 0.0_rk8, tmp2_real, 1, 1, sc_desc)
#else
        call dgemm("N","N", na, na, na, 1.0_rk8, a_real, na, tmp1_real, na, 0.0_rk8, tmp2_real, na)
#endif

        ! compare tmp2 with original matrix
        tmp2_real(:,:) = tmp2_real(:,:) - as_real(:,:)

#ifdef WITH_MPI
        norm = pdlange("M",na, na, tmp2_real, 1, 1, sc_desc, tmp1_real)
#else
        norm = dlange("M", na, na, tmp2_real, na_rows, tmp1_real)
#endif

#ifdef WITH_MPI
        call mpi_allreduce(norm,normmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
#else
        normmax = norm
#endif
        if (myid .eq. 0) then
          print *," Maximum error of result: ", normmax
        endif

        if (normmax .gt. 5e-12_rk8) then
             status = 1
        endif

      endif ! real case

      if (input_options%datatype .eq. 2) then
        a_complex(:,:) = CONE - CONE
        diagonalElement_complex = (2.546_rk8, 0.0_rk8)
        do i = 1, na
          if (map_global_array_index_to_local_index(i, i, rowLocal, colLocal, nblk, np_rows, np_cols, my_prow, my_pcol)) then
            a_complex(rowLocal,colLocal) = diagonalElement_complex * abs(cos( pi*real(i,kind=rk8)/ real(na+1,kind=rk8) ))
          endif
        enddo
        as_complex(:,:) = a_complex(:,:)

        if (myid==0) then
          print '(a)','| Compute complex cholesky decomposition ... '
          print *
        end if
#ifdef WITH_MPI
        call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif

        success = elpa_cholesky_complex_double(na, a_complex, na_rows, nblk, na_cols, mpi_comm_rows, mpi_comm_cols, .true.)

        if (.not.(success)) then
          write(error_unit,*) " elpa_cholesky_complex produced an error! Aborting..."
#ifdef WITH_MPI
          call MPI_ABORT(mpi_comm_world, 1, mpierr)
#endif
        endif


        if (myid==0) then
          print '(a)','| Solve cholesky decomposition complete.'
          print *
        end if

        tmp1_complex(:,:) = 0.0_ck8

        ! tmp1 = a**H
#ifdef WITH_MPI
        call pztranc(na, na, CONE, a_complex, 1, 1, sc_desc, CZERO, tmp1_complex, 1, 1, sc_desc)
#else
        tmp1_complex = transpose(conjg(a_complex))
#endif
        ! tmp2 = a * a**H
#ifdef WITH_MPI
        call pzgemm("N","N", na, na, na, CONE, a_complex, 1, 1, sc_desc, tmp1_complex, 1, 1, &
                     sc_desc, CZERO, tmp2_complex, 1, 1, sc_desc)
#else
        call zgemm("N","N", na, na, na, CONE, a_complex, na, tmp1_complex, na, CZERO, tmp2_complex, na)
#endif

        ! compare tmp2 with c
        tmp2_complex(:,:) = tmp2_complex(:,:) - as_complex(:,:)

#ifdef WITH_MPI
        norm = pzlange("M",na, na, tmp2_complex, 1, 1, sc_desc,tmp1_complex)
#else
        norm = zlange("M",na, na, tmp2_complex, na_rows, tmp1_complex)
#endif
#ifdef WITH_MPI
        call mpi_allreduce(norm,normmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
#else
        normmax = norm
#endif
        if (myid .eq. 0) then
          print *," Maximum error of result: ", normmax
        endif

        if (normmax .gt. 5e-11_rk8) then
             status = 1
        endif

      endif ! complex case


    endif ! doCholesky


    if (input_options%doInvertTrm) then
      if (input_options%datatype .eq. 1) then

        a_real(:,:) = 0.0_rk8
        diagonalElement_real = 2.546_rk8

        do i = 1, na
          if (map_global_array_index_to_local_index(i, i, rowLocal, colLocal, nblk, np_rows, np_cols, my_prow, my_pcol)) then
            a_real(rowLocal,colLocal) = diagonalElement_real * abs(cos( pi*real(i,kind=rk8)/ real(na+1,kind=rk8) ))
          endif
        enddo

        as_real(:,:) = a_real(:,:)

        if (myid==0) then
          print '(a)','| Setup an upper triangular matrix ... '
          print *
        end if
#ifdef WITH_MPI
        call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif

        success = elpa_cholesky_real_double(na, a_real, na_rows, nblk, na_cols, mpi_comm_rows, mpi_comm_cols, .true.)

        if (.not.(success)) then
           write(error_unit,*) "elpa_cholseky_real produced an error! Aborting..."
#ifdef WITH_MPI
           call MPI_ABORT(mpi_comm_world, 1, mpierr)
#endif
        endif


        if (myid==0) then
          print '(a)','| Upper triangular matrix created.'
          print *
        end if

        as_real(:,:) = a_real(:,:)

        if (myid==0) then
          print '(a)','| Invert the upper triangular matrix ... '
          print *
        end if
#ifdef WITH_MPI
        call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif

        success = elpa_invert_trm_real_double(na, a_real, na_rows, nblk, na_cols, mpi_comm_rows, mpi_comm_cols, .true.)

        if (.not.(success)) then
          write(error_unit,*) "elpa_cholseky_real produced an error! Aborting..."
#ifdef WITH_MPI
          call MPI_ABORT(mpi_comm_world, 1, mpierr)
#endif
        endif


        if (myid==0) then
          print '(a)','| Upper triangular matrix inverted.'
          print *
        end if


        tmp1_real(:,:) = 0.0_rk8

        ! tmp1 = as * a^-1 ! should give unity matrix

#ifdef WITH_MPI
        call pdgemm("N","N", na, na, na, 1.0_rk8, as_real, 1, 1, sc_desc, a_real, 1, 1, &
                    sc_desc, 0.0_rk8, tmp1_real, 1, 1, sc_desc)
#else
        call dgemm("N","N", na, na, na, 1.0_rk8, as_real, na, a_real, na, 0.0_rk8, tmp1_real, na)
#endif

        ! check the quality of unity matrix

        ! tmp2 = b * tmp1
#ifdef WITH_MPI
        call pdgemm("N","N", na, na, na, 1.0_rk8, b_real, 1, 1, sc_desc, tmp1_real, 1, 1, &
                    sc_desc, 0.0_rk8, tmp2_real, 1, 1, sc_desc)
#else
        call dgemm("N","N", na, na, na, 1.0_rk8, b_real, na, tmp1_real, na, 0.0_rk8, tmp2_real, na)
#endif

        ! compare tmp2 with original matrix b
        tmp2_real(:,:) = tmp2_real(:,:) - bs_real(:,:)

#ifdef WITH_MPI
        norm = pdlange("M",na, na, tmp2_real, 1, 1, sc_desc, bs_real)
#else
        norm = dlange("M", na, na, tmp2_real, na_rows, bs_real)
#endif

#ifdef WITH_MPI
        call mpi_allreduce(norm,normmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
#else
        normmax = norm
#endif
        if (myid .eq. 0) then
          print *," Maximum error of result: ", normmax
        endif

        if (normmax .gt. 5e-12_rk8) then
          status = 1
        endif

      endif ! realcase

      if (input_options%datatype .eq. 2) then
        a_complex(:,:) = CONE - CONE
        diagonalElement_complex = (2.546_rk8, 0.0_rk8)
        do i = 1, na
          if (map_global_array_index_to_local_index(i, i, rowLocal, colLocal, nblk, np_rows, np_cols, my_prow, my_pcol)) then
            a_complex(rowLocal,colLocal) = diagonalElement_complex * abs(cos( pi*real(i,kind=rk8)/ real(na+1,kind=rk8) ))
          endif
        enddo
        as_complex(:,:) = a_complex(:,:)

        if (myid==0) then
          print '(a)','| Setting up tridiagonal matrix ... '
          print *
        end if
#ifdef WITH_MPI
        call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif

        success = elpa_cholesky_complex_double(na, a_complex, na_rows, nblk, na_cols, mpi_comm_rows, mpi_comm_cols, .true.)

        if (.not.(success)) then
          write(error_unit,*) " elpa_cholesky_complex produced an error! Aborting..."
#ifdef WITH_MPI
          call MPI_ABORT(mpi_comm_world, 1, mpierr)
#endif
        endif

        as_complex(:,:) = a_complex(:,:)

        if (myid==0) then
          print '(a)','| Setting up tridiagonal matrix complete.'
          print *
        end if

        if (myid==0) then
          print '(a)','| Inverting tridiagonal matrix ... '
          print *
        end if
#ifdef WITH_MPI
        call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif

        success = elpa_invert_trm_complex_double(na, a_complex, na_rows, nblk, na_cols, mpi_comm_rows, mpi_comm_cols, .true.)

        if (.not.(success)) then
           write(error_unit,*) " elpa_invert_trm_complex produced an error! Aborting..."
#ifdef WITH_MPI
           call MPI_ABORT(mpi_comm_world, 1, mpierr)
#endif
        endif


        if (myid==0) then
          print '(a)','| Inversion of tridiagonal matrix complete.'
          print *
        end if


        tmp1_complex(:,:) = 0.0_ck8

        ! tmp1 = a * a^-1 ! should be unity matrix
#ifdef WITH_MPI
        call pzgemm("N","N", na, na, na, CONE, as_complex, 1, 1, sc_desc, a_complex, 1, 1, &
                    sc_desc, CZERO, tmp1_complex, 1, 1, sc_desc)
#else
        call zgemm("N","N", na, na, na, CONE, as_complex, na, a_complex, na, CZERO, tmp1_complex, na)
#endif

        ! tmp2 = b * tmp1
        tmp2_complex(:,:) = 0.0_ck8
#ifdef WITH_MPI
        call pzgemm("N","N", na, na, na, CONE, b_complex, 1, 1, sc_desc, tmp1_complex, 1, 1, &
                    sc_desc, CZERO, tmp2_complex, 1, 1, sc_desc)
#else
        call zgemm("N","N", na, na, na, CONE, b_complex, na, tmp1_complex, na, CZERO, tmp2_complex, na)
#endif

        ! compare tmp2 with c
        tmp2_complex(:,:) = tmp2_complex(:,:) - bs_complex(:,:)
        tmp1_complex(:,:) = 0.0_ck8
#ifdef WITH_MPI
        norm = pzlange("M",na, na, tmp2_complex, 1, 1, sc_desc,tmp1_complex)
#else
        norm = zlange("M",na, na, tmp2_complex, na_rows, tmp1_complex)
#endif
#ifdef WITH_MPI
        call mpi_allreduce(norm,normmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
#else
        normmax = norm
#endif
        if (myid .eq. 0) then
          print *," Maximum error of result: ", normmax
        endif

        if (normmax .gt. 5e-11_rk8) then
          status = 1
        endif

      endif ! complexcase

    endif ! input_options%doInvertTrm

    if (input_options%doTransposeMultiply) then
      if (input_options%datatype .eq. 1) then
        b_real(:,:) = 2.0_rk8 * a_real(:,:)
        c_real(:,:) = 0.0_rk8

        if (myid==0) then
          print '(a)','| Compute c= a**T * b ... '
          print *
        end if
#ifdef WITH_MPI
        call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif

        success = elpa_mult_at_b_real_double("F","F", na, na, a_real, na_rows, na_cols, b_real, na_rows, &
                                             na_cols, nblk, mpi_comm_rows, mpi_comm_cols, c_real,   &
                                             na_rows, na_cols)

        if (.not.(success)) then
          write(error_unit,*) "elpa_mult_at_b_real produced an error! Aborting..."
#ifdef WITH_MPI
          call MPI_ABORT(mpi_comm_world, 1, mpierr)
#endif
        endif


        if (myid==0) then
          print '(a)','| Solve c = a**T * b complete.'
          print *
        end if

        tmp1_real(:,:) = 0.0_rk8

        ! tmp1 = a**T
#ifdef WITH_MPI
        call pdtran(na, na, 1.0_rk8, a_real, 1, 1, sc_desc, 0.0_rk8, tmp1_real, 1, 1, sc_desc)
#else
        tmp1_real = transpose(a_real)
#endif
        ! tmp2 = tmp1 * b
#ifdef WITH_MPI
        call pdgemm("N","N", na, na, na, 1.0_rk8, tmp1_real, 1, 1, sc_desc, b_real, 1, 1, &
                    sc_desc, 0.0_rk8, tmp2_real, 1, 1, sc_desc)
#else
        call dgemm("N","N", na, na, na, 1.0_rk8, tmp1_real, na, b_real, na, 0.0_rk8, tmp2_real, na)
#endif

        ! compare tmp2 with c
        tmp2_real(:,:) = tmp2_real(:,:) - c_real(:,:)

#ifdef WITH_MPI
        norm = pdlange("M", na, na, tmp2_real, 1, 1, sc_desc, tmp1_real)
#else
        norm = dlange("M", na, na, tmp2_real, na_rows, tmp1_real)
#endif

#ifdef WITH_MPI
        call mpi_allreduce(norm,normmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
#else
        normmax = norm
#endif
        if (myid .eq. 0) then
          print *," Maximum error of result: ", normmax
        endif

        if (normmax .gt. 5e-11_rk8) then
             status = 1
        endif

      endif ! realcase

      if (input_options%datatype .eq. 2) then
        b_complex(:,:) = 2.0_ck8 * a_complex(:,:)
        c_complex(:,:) = 0.0_ck8

        if (myid==0) then
          print '(a)','| Compute c= a**T * b ... '
          print *
        end if
#ifdef WITH_MPI
        call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif

        success = elpa_mult_ah_b_complex_double("F","F", na, na, a_complex, na_rows, na_cols, b_complex, na_rows, na_cols, &
                                                nblk, mpi_comm_rows, mpi_comm_cols, c_complex, na_rows, na_cols)

        if (.not.(success)) then
          write(error_unit,*) " elpa_mult_at_b_complex produced an error! Aborting..."
#ifdef WITH_MPI
          call MPI_ABORT(mpi_comm_world, 1, mpierr)
#endif
        endif

        if (myid==0) then
          print '(a)','| Solve c = a**T * b complete.'
          print *
        end if

        tmp1_complex(:,:) = 0.0_ck8

        ! tmp1 = a**T
#ifdef WITH_MPI
        call pztranc(na, na, CONE, a_complex, 1, 1, sc_desc, CZERO, tmp1_complex, 1, 1, sc_desc)
#else
        tmp1_complex = transpose(conjg(a_complex))
#endif
        ! tmp2 = tmp1 * b
#ifdef WITH_MPI
        call pzgemm("N","N", na, na, na, CONE, tmp1_complex, 1, 1, sc_desc, b_complex, 1, 1, &
                    sc_desc, CZERO, tmp2_complex, 1, 1, sc_desc)
#else
        call zgemm("N","N", na, na, na, CONE, tmp1_complex, na, b_complex, na, CZERO, tmp2_complex, na)
#endif

        ! compare tmp2 with c
        tmp2_complex(:,:) = tmp2_complex(:,:) - c_complex(:,:)

#ifdef WITH_MPI
        norm = pzlange("M",na, na, tmp2_complex, 1, 1, sc_desc,tmp1_complex)
#else
        norm = zlange("M",na, na, tmp2_complex, na_rows, tmp1_complex)
#endif
#ifdef WITH_MPI
        call mpi_allreduce(norm,normmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
#else
        normmax = norm
#endif
        if (myid .eq. 0) then
          print *," Maximum error of result: ", normmax
        endif

        if (normmax .gt. 5e-11_rk8) then
             status = 1
        endif

      endif ! complexcase


    endif ! input_options%doTransposeMultiply

!#ifdef HAVE_DETAILED_TIMINGS
!   call timer%start("set up matrix")
!#endif
!
!   if (input_options%datatype .eq. 0) then
!     call prepare_matrix_double(na, myid, sc_desc, a_real, z_real, as_real)
!   endif
!   if (input_options%datatype .eq. 1) then
!     call prepare_matrix_double(na, myid, sc_desc, a_complex, z_complex, as_complex)
!   endif
!
!
!#ifdef HAVE_DETAILED_TIMINGS
!   call timer%stop("set up matrix")
!#endif


   if (input_options%do1stage) then
     !-------------------------------------------------------------------------------
     ! Calculate eigenvalues/eigenvectors

     if (myid==0) then
       print '(a)','| Entering one-step ELPA solver ... '
       print *
     end if

     if (input_options%useGPUIsSet .eq. 1) this_gpu = .true.
     if (input_options%useGPUIsSet .eq. 0) this_gpu = .false.

#ifdef WITH_MPI
     call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif
     tStart = mpi_wtime()

     if (input_options%datatype .eq. 1) then
       success = elpa_solve_evp_real_1stage_double(na, nev, a_real, na_rows, ev, z_real, na_rows, nblk, &
                                                   na_cols, mpi_comm_rows, mpi_comm_cols, mpi_comm_world, useGPU=this_gpu)
     endif
     if (input_options%datatype .eq. 2) then
       success = elpa_solve_evp_complex_1stage_double(na, nev, a_complex, na_rows, ev, z_complex, na_rows, nblk, &
                                                      na_cols, mpi_comm_rows, mpi_comm_cols, mpi_comm_world, useGPU=this_gpu)
     endif

     if (.not.(success)) then
       if (input_options%datatype .eq. 1) then
         write(error_unit,*) "elpa_solve_evp_real_1stage_double produced an error! Aborting..."
       endif
       if (input_options%datatype .eq. 2) then
         write(error_unit,*) "elpa_solve_evp_complex_1stage_double produced an error! Aborting..."
       endif

#ifdef WITH_MPI
       call MPI_ABORT(mpi_comm_world, 1, mpierr)
#endif
     endif


     if (myid==0) then
       print '(a)','| One-step ELPA solver complete.'
       print *
     end if

     if (input_options%datatype .eq. 1) then
       if (myid == 0) print *,'Time tridiag_real     :',time_evp_fwd
       if (myid == 0) print *,'Time solve_tridi      :',time_evp_solve
       if (myid == 0) print *,'Time trans_ev_real    :',time_evp_back
       if (myid == 0) print *,'Total time (sum above):',time_evp_back+time_evp_solve+time_evp_fwd
       if (myid == 0) print *," "
     endif
     if (input_options%datatype .eq. 2) then
       if (myid == 0) print *,'Time tridiag_complex     :',time_evp_fwd
       if (myid == 0) print *,'Time solve_tridi         :',time_evp_solve
       if (myid == 0) print *,'Time trans_ev_complex    :',time_evp_back
       if (myid == 0) print *,'Total time (sum above)   :',time_evp_back+time_evp_solve+time_evp_fwd
       if (myid == 0) print *," "
     endif
#ifdef WITH_MPI
     call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif
     tEnd = mpi_wtime()
     if(myid == 0) print *," "
     if (input_options%datatype .eq. 1) then
       if(myid == 0) print *,'Total time for solve_evp_real_1stage:',tEnd - tStart
     endif
     if (input_options%datatype .eq. 2) then
       if(myid == 0) print *,'Total time for solve_evp_complex_1stage:',tEnd - tStart
     endif

     if(myid == 0) print *," "

! TODO write of output

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


     !-------------------------------------------------------------------------------
     ! Test correctness of result (using plain scalapack routines)


     if (input_options%datatype .eq. 1) then
       status = check_correctness_double(na, nev, as_real, z_real, ev, sc_desc, myid)
     endif
     if (input_options%datatype .eq. 2) then
       status = check_correctness_double(na, nev, as_complex, z_complex, ev, sc_desc, myid)
     endif

     if (status .eq. 1) then
#ifdef WITH_MPI
       call blacs_gridexit(my_blacs_ctxt)
       call mpi_finalize(mpierr)
#endif

       call EXIT(STATUS)
     endif
   endif !do1Stage

   if (input_options%do2Stage) then
     if (input_options%datatype .eq. 1) then
       ! real cases

       if (.not.(input_options%realKernelIsSet)) then
         ! start again with ELPA2 generic and so forth

         ! first default kernel

         if (myid .eq. 0) print *," "
         if (myid .eq. 0) print *,"Testing 2stage solver with default kernel: ", trim(elpa_get_actual_real_kernel_name())
         if (myid .eq. 0) print *," "

         a_real = as_real
         z_real = a_real

        if (input_options%useGPUIsSet .eq. 1) this_gpu = .true.
        if (input_options%useGPUIsSet .eq. 0) this_gpu = .false.

        if (input_options%useQrIsSet .eq. 1) this_qr = .true.
        if (input_options%useQrIsSet .eq. 0) this_qr = .false.

         if (myid==0) then
           print *," "
           print '(a)','| Entering two-stage ELPA solver ... '
           print *
         end if

#ifdef WITH_MPI
         call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif

         tStart = mpi_wtime()

         success = elpa_solve_evp_real_2stage_double(na, nev, a_real, na_rows, ev, z_real, na_rows,  nblk, na_cols, &
                                                     mpi_comm_rows, mpi_comm_cols, mpi_comm_world, useQr=this_qr, useGPU=this_gpu)

         if (.not.(success)) then
           write(error_unit,*) "solve_evp_real_2stage with default kernel ",trim(elpa_get_actual_real_kernel_name()), &
                               " produced an error! Aborting..."
#ifdef WITH_MPI
           call MPI_ABORT(mpi_comm_world, 1, mpierr)
#endif
         endif

         if (myid==0) then
           print '(a)','| Two-step ELPA solver complete.'
           print *
         end if

         if (myid == 0) print *,'Time transform to tridi :',time_evp_fwd
         if (myid == 0) print *,'Time solve tridi        :',time_evp_solve
         if (myid == 0) print *,'Time transform back EVs :',time_evp_back
         if (myid == 0) print *,'Total time (sum above)  :',time_evp_back+time_evp_solve+time_evp_fwd
         if (myid == 0) print *," "

#ifdef WITH_MPI
         call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif
         tEnd = mpi_wtime()

         if (myid == 0) print *," "
         if (myid == 0) print *,'Total time for solve_evp_real2_stage with ', &
                        trim(elpa_get_actual_real_kernel_name()),' default kernel:',tEnd - tStart
         if (myid == 0) print *," "

         status = check_correctness_double(na, nev, as_real, z_real, ev, sc_desc, myid)
         if (myid == 0) print *," "

         if (status .eq. 1) then
           if (myid == 0) print *," ERROR in solve_evp_real2_stage with ",trim(elpa_get_actual_real_kernel_name()), &
             ' kernel!'
#ifdef WITH_MPI
           call blacs_gridexit(my_blacs_ctxt)
           call mpi_finalize(mpierr)
#endif

           call EXIT(STATUS)
         endif
         if (myid .eq. 0) print *," "


         if (myid .eq. 0) print *," "
         if (myid .eq. 0) print *,"Iterating over all available ELPA2 real kernels ..."
         if (myid .eq. 0) print *," "

         do this_kernel = 1 , elpa_number_of_real_kernels()
           if (input_options%useGPUIsSet .eq. 1) this_gpu = .true.
           if (input_options%useGPUIsSet .eq. 0) this_gpu = .false.

           if (input_options%useQrIsSet .eq. 1) this_qr = .true.
           if (input_options%useQrIsSet .eq. 0) this_qr = .false.


           a_real = as_real
           z_real = a_real
           if (elpa_real_kernel_is_available(this_kernel)) then
             if (input_options%useQrIsSet .eq. 0) then
               if (myid == 0) print *,"ELPA2 kernel ",trim(elpa_real_kernel_name(this_kernel)),":"
             else
               if (myid == 0) print *,"ELPA2 kernel ",trim(elpa_real_kernel_name(this_kernel))," with qr decompostion:"
             endif
             if (myid==0) then
               print *," "
               print '(a)','| Entering two-stage ELPA solver ... '
               print *
             end if
#ifdef WITH_MPI
             call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif

             tStart = mpi_wtime()

             success = elpa_solve_evp_real_2stage_double(na, nev, a_real, na_rows, ev, z_real, na_rows,  nblk, na_cols, &
                                                         mpi_comm_rows, mpi_comm_cols, mpi_comm_world,        &
                                                         THIS_ELPA_KERNEL_API = this_kernel, useQR=this_qr, useGPU=this_gpu)

             if (.not.(success)) then
               if (input_options%useQrIsSet .eq. 0) then
                 write(error_unit,*) "solve_evp_real_2stage with kernel ",trim(elpa_real_kernel_name(this_kernel)), &
                                     " produced an error! Aborting..."
               else
                 write(error_unit,*) "solve_evp_real_2stage with kernel ",trim(elpa_real_kernel_name(this_kernel)), &
                                     " and qr-decompostion produced an error! Aborting..."

               endif
#ifdef WITH_MPI
                 call MPI_ABORT(mpi_comm_world, 1, mpierr)
#endif
             endif

             if (myid==0) then
               print '(a)','| Two-step ELPA solver complete.'
               print *
             end if

             if (myid == 0) print *,'Time transform to tridi :',time_evp_fwd
             if (myid == 0) print *,'Time solve tridi        :',time_evp_solve
             if (myid == 0) print *,'Time transform back EVs :',time_evp_back
             if (myid == 0) print *,'Total time (sum above)  :',time_evp_back+time_evp_solve+time_evp_fwd
             if (myid == 0) print *," "
#ifdef WITH_MPI
             call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif
             tEnd = mpi_wtime()

             if (myid == 0) print *," "
             if (myid == 0) print *,'Total time for solve_evp_real2_stage with ', &
                            trim(elpa_real_kernel_name(this_kernel)),' kernel:',tEnd - tStart
             if (myid == 0) print *," "

             status = check_correctness_double(na, nev, as_real, z_real, ev, sc_desc, myid)
             if (myid == 0) print *," "

             if (status .eq. 1) then
               if (input_options%useQrIsSet.eq. 0) then
                 if (myid == 0) print *," ERROR in solve_evp_real2_stage with ",trim(elpa_real_kernel_name(this_kernel)), &
                 ' kernel!'
               else
                 if (myid == 0) print *," ERROR in solve_evp_real2_stage with ",trim(elpa_real_kernel_name(this_kernel)), &
                 ' kernel and qr-decompostion!'
               endif
#ifdef WITH_MPI
               call blacs_gridexit(my_blacs_ctxt)
               call mpi_finalize(mpierr)
#endif

               call EXIT(STATUS)
             endif ! status == 1

           endif ! elpa_real_kernel
         enddo  ! kernel loop

       else ! realKernelSet

         if (input_options%useGPUIsSet .eq. 1) this_gpu=.true.
         if (input_options%useGPUIsSet .eq. 0) this_gpu=.false.

         if (input_options%useQrIsSet .eq. 1) this_qr = .true.
         if (input_options%useQrIsSet .eq. 0) this_qr = .false.


         a_real = as_real
         z_real = a_real
         if (elpa_real_kernel_is_available(input_options%this_real_kernel)) then
           if (input_options%useQrIsSet  .eq. 0) then
             if (myid == 0) print *,"ELPA2 kernel ",trim(elpa_real_kernel_name(input_options%this_real_kernel)),":"
           else
             if (myid == 0) print *,"ELPA2 kernel ",trim(elpa_real_kernel_name(input_options%this_real_kernel)), &
                 " with qr decompostion:"
           endif
           if (myid==0) then
             print *," "
             print '(a)','| Entering two-stage ELPA solver ... '
             print *
           end if
#ifdef WITH_MPI
           call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif

           tStart = mpi_wtime()

           success = elpa_solve_evp_real_2stage_double(na, nev, a_real, na_rows, ev, z_real, na_rows,  nblk, na_cols, &
                                                       mpi_comm_rows, mpi_comm_cols, mpi_comm_world,        &
                                                       THIS_ELPA_KERNEL_API = input_options%this_real_kernel, useQR=this_qr, &
                                                       useGPU=this_gpu)

           if (.not.(success)) then
             if (input_options%useQrIsSet  .eq. 0) then
               write(error_unit,*) "solve_evp_real_2stage with kernel ", &
                    trim(elpa_real_kernel_name(input_options%this_real_kernel)), &
                                   " produced an error! Aborting..."
             else
               write(error_unit,*) "solve_evp_real_2stage with kernel ", &
                   trim(elpa_real_kernel_name(input_options%this_real_kernel)), &
                                   " and qr-decompostion produced an error! Aborting..."

             endif
#ifdef WITH_MPI
             call MPI_ABORT(mpi_comm_world, 1, mpierr)
#endif
           endif

           if (myid==0) then
             print '(a)','| Two-step ELPA solver complete.'
             print *
           end if

           if (myid == 0) print *,'Time transform to tridi :',time_evp_fwd
           if (myid == 0) print *,'Time solve tridi        :',time_evp_solve
           if (myid == 0) print *,'Time transform back EVs :',time_evp_back
           if (myid == 0) print *,'Total time (sum above)  :',time_evp_back+time_evp_solve+time_evp_fwd
           if (myid == 0) print *," "
#ifdef WITH_MPI
           call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif
           tEnd = mpi_wtime()

           if (myid == 0) print *," "
           if (myid == 0) print *,'Total time for solve_evp_real2_stage with ', &
                          trim(elpa_real_kernel_name(input_options%this_real_kernel)),' kernel:',tEnd - tStart
           if (myid == 0) print *," "

           status = check_correctness_double(na, nev, as_real, z_real, ev, sc_desc, myid)
           if (myid == 0) print *," "

           if (status .eq. 1) then
             if (input_options%useQrIsSet  .eq. 0) then
               if (myid == 0) print *," ERROR in solve_evp_real2_stage with ", &
                   trim(elpa_real_kernel_name(input_options%this_real_kernel)), &
               ' kernel!'
             else
               if (myid == 0) print *," ERROR in solve_evp_real2_stage with ", &
                   trim(elpa_real_kernel_name(input_options%this_real_kernel)), &
               ' kernel and qr-decompostion!'
             endif
#ifdef WITH_MPI
             call blacs_gridexit(my_blacs_ctxt)
             call mpi_finalize(mpierr)
#endif

             call EXIT(STATUS)
           endif

         endif
       endif ! realKernelSet
     endif ! datatype == 1
   endif ! do2Stage

   if (input_options%do2stage) then
     if (input_options%datatype .eq. 2) then
       ! complex cases

       if (.not.(input_options%complexKernelIsSet)) then
         ! start again with ELPA2 generic and so forth

         ! first default kernel

         if (myid .eq. 0) print *," "
         if (myid .eq. 0) print *,"Testing 2stage solver with default kernel: ", trim(elpa_get_actual_complex_kernel_name())
         if (myid .eq. 0) print *," "

         a_complex = as_complex
         z_complex = a_complex

         if (myid==0) then
           print *," "
           print '(a)','| Entering two-stage ELPA solver ... '
           print *
         end if

         if (input_options%useGPUIsSet .eq. 1) this_gpu = .true.
         if (input_options%useGPUIsSet .eq. 0) this_gpu = .false.

         if (input_options%useQrIsSet .eq. 1) then
           print *,"error: nr QR decomposition for complex case available!"
           stop 1
         endif
#ifdef WITH_MPI
         call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif

         tStart = mpi_wtime()

         success = elpa_solve_evp_complex_2stage_double(na, nev, a_complex, na_rows, ev, z_complex, na_rows,  nblk, na_cols, &
                                                        mpi_comm_rows, mpi_comm_cols, mpi_comm_world, useGPU=this_gpu)

         if (.not.(success)) then
           write(error_unit,*) "solve_evp_complex_2stage with default kernel ",trim(elpa_get_actual_complex_kernel_name()), &
                               " produced an error! Aborting..."
#ifdef WITH_MPI
           call MPI_ABORT(mpi_comm_world, 1, mpierr)
#endif
         endif

         if (myid==0) then
           print '(a)','| Two-step ELPA solver complete.'
           print *
         end if

         if (myid == 0) print *,'Time transform to tridi :',time_evp_fwd
         if (myid == 0) print *,'Time solve tridi        :',time_evp_solve
         if (myid == 0) print *,'Time transform back EVs :',time_evp_back
         if (myid == 0) print *,'Total time (sum above)  :',time_evp_back+time_evp_solve+time_evp_fwd
         if (myid == 0) print *," "

#ifdef WITH_MPI
         call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif
         tEnd = mpi_wtime()

         if (myid == 0) print *," "
         if (myid == 0) print *,'Total time for solve_evp_complex_2stage with ', &
                        trim(elpa_get_actual_complex_kernel_name()),' default kernel:',tEnd - tStart
         if (myid == 0) print *," "

         status = check_correctness_double(na, nev, as_complex, z_complex, ev, sc_desc, myid)
         if (myid == 0) print *," "

         if (status .eq. 1) then
           if (myid == 0) print *," ERROR in solve_evp_complex_2stage with ",trim(elpa_get_actual_complex_kernel_name()), &
             ' kernel!'
#ifdef WITH_MPI
           call blacs_gridexit(my_blacs_ctxt)
           call mpi_finalize(mpierr)
#endif

           call EXIT(STATUS)
         endif
         if (myid .eq. 0) print *," "


         if (myid .eq. 0) print *," "
         if (myid .eq. 0) print *,"Iterating over all available ELPA2 complex kernels ..."
         if (myid .eq. 0) print *," "

         do this_kernel = 1 , elpa_number_of_complex_kernels()
           if (input_options%useGPUIsSet .eq. 1) this_gpu=.true.
           if (input_options%useGPUIsSet .eq. 0) this_gpu=.false.

           if (input_options%useQrIsSet .eq. 1) then
             print *,"error: no QR-decomposition for complex case available!"
             stop 1
           endif

           a_complex = as_complex
           z_complex = a_complex
           if (elpa_complex_kernel_is_available(this_kernel)) then
             if (input_options%useQrIsSet  .eq. 0) then
               if (myid == 0) print *,"ELPA2 kernel ",trim(elpa_complex_kernel_name(this_kernel)),":"
             else
               if (myid == 0) print *,"ELPA2 kernel ",trim(elpa_complex_kernel_name(this_kernel))," with qr decompostion:"
             endif
             if (myid==0) then
               print *," "
               print '(a)','| Entering two-stage ELPA solver ... '
               print *
             end if
#ifdef WITH_MPI
             call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif

             tStart = mpi_wtime()

             success = elpa_solve_evp_complex_2stage_double(na, nev, a_complex, na_rows, ev, z_complex, na_rows,  nblk, na_cols, &
                                                            mpi_comm_rows, mpi_comm_cols, mpi_comm_world,        &
                                                            THIS_ELPA_KERNEL_API = this_kernel, useGPU=this_gpu)

             if (.not.(success)) then
               if (input_options%useQrIsSet  .eq. 0) then
                 write(error_unit,*) "solve_evp_complex_2stage with kernel ",trim(elpa_complex_kernel_name(this_kernel)), &
                                     " produced an error! Aborting..."
               else
                 write(error_unit,*) "solve_evp_complex_2stage with kernel ",trim(elpa_complex_kernel_name(this_kernel)), &
                                     " and qr-decompostion produced an error! Aborting..."

               endif
#ifdef WITH_MPI
               call MPI_ABORT(mpi_comm_world, 1, mpierr)
#endif
             endif

             if (myid==0) then
               print '(a)','| Two-step ELPA solver complete.'
               print *
             end if

             if (myid == 0) print *,'Time transform to tridi :',time_evp_fwd
             if (myid == 0) print *,'Time solve tridi        :',time_evp_solve
             if (myid == 0) print *,'Time transform back EVs :',time_evp_back
             if (myid == 0) print *,'Total time (sum above)  :',time_evp_back+time_evp_solve+time_evp_fwd
             if (myid == 0) print *," "
#ifdef WITH_MPI
             call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif
             tEnd = mpi_wtime()

             if (myid == 0) print *," "
             if (myid == 0) print *,'Total time for solve_evp_complex_2stage with ', &
                              trim(elpa_complex_kernel_name(this_kernel)),' kernel:',tEnd - tStart
             if (myid == 0) print *," "

             status = check_correctness_double(na, nev, as_complex, z_complex, ev, sc_desc, myid)
             if (myid == 0) print *," "

             if (status .eq. 1) then
               if (input_options%useQrIsSet  .eq. 0) then
                 if (myid == 0) print *," ERROR in solve_evp_complex_2stage with ",trim(elpa_complex_kernel_name(this_kernel)), &
                 ' kernel!'
               else
                 if (myid == 0) print *," ERROR in solve_evp_complex_2stage with ",trim(elpa_complex_kernel_name(this_kernel)), &
                 ' kernel and qr-decompostion!'
               endif
#ifdef WITH_MPI
               call blacs_gridexit(my_blacs_ctxt)
               call mpi_finalize(mpierr)
#endif

               call EXIT(STATUS)
             endif

           endif
         enddo

       else ! complexKernelSet

         a_complex = as_complex
         z_complex = a_complex
         if (elpa_complex_kernel_is_available(input_options%this_complex_kernel)) then
           if (input_options%useQrIsSet  .eq. 0) then
             if (myid == 0) print *,"ELPA2 kernel ",trim(elpa_complex_kernel_name(input_options%this_complex_kernel)),":"
           else
             if (myid == 0) print *,"ELPA2 kernel ",trim(elpa_complex_kernel_name(input_options%this_complex_kernel)), &
                 " with qr decompostion:"
           endif
           if (myid==0) then
             print *," "
             print '(a)','| Entering two-stage ELPA solver ... '
             print *
           end if

           if (input_options%useGPUIsSet .eq. 1) this_gpu=.true.
           if (input_options%useGPUIsSet .eq. 0) this_gpu=.false.

           if (input_options%useQrIsSet .eq. 1) then
             print *,"error: no QR-decomposition for complex case available!"
             stop 1
           endif


#ifdef WITH_MPI
           call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif

           tStart = mpi_wtime()

           success = elpa_solve_evp_complex_2stage_double(na, nev, a_complex, na_rows, ev, z_complex, na_rows,  nblk, na_cols, &
                                                          mpi_comm_rows, mpi_comm_cols, mpi_comm_world,        &
                                                          THIS_ELPA_KERNEL_API = input_options%this_complex_kernel, &
                                                          useGPU=this_gpu)

           if (.not.(success)) then
             if (input_options%useQrIsSet  .eq. 0) then
               write(error_unit,*) "solve_evp_complex_2stage with kernel ", &
                   trim(elpa_complex_kernel_name(input_options%this_complex_kernel)), &
                                   " produced an error! Aborting..."
             else
               write(error_unit,*) "solve_evp_complex_2stage with kernel ", &
                   trim(elpa_complex_kernel_name(input_options%this_complex_kernel)), &
                                   " and qr-decompostion produced an error! Aborting..."

             endif
#ifdef WITH_MPI
             call MPI_ABORT(mpi_comm_world, 1, mpierr)
#endif
           endif

           if (myid==0) then
             print '(a)','| Two-step ELPA solver complete.'
             print *
           end if

           if (myid == 0) print *,'Time transform to tridi :',time_evp_fwd
           if (myid == 0) print *,'Time solve tridi        :',time_evp_solve
           if (myid == 0) print *,'Time transform back EVs :',time_evp_back
           if (myid == 0) print *,'Total time (sum above)  :',time_evp_back+time_evp_solve+time_evp_fwd
           if (myid == 0) print *," "
#ifdef WITH_MPI
           call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
#endif
           tEnd = mpi_wtime()

           if (myid == 0) print *," "
           if (myid == 0) print *,'Total time for solve_evp_complex_2stage with ', &
                          trim(elpa_complex_kernel_name(input_options%this_complex_kernel)),' kernel:',tEnd - tStart
           if (myid == 0) print *," "

           status = check_correctness_double(na, nev, as_complex, z_complex, ev, sc_desc, myid)
           if (myid == 0) print *," "

           if (status .eq. 1) then
             if (input_options%useQrIsSet  .eq. 0) then
               if (myid == 0) print *," ERROR in solve_evp_complex_2stage with ", &
                 trim(elpa_complex_kernel_name(input_options%this_complex_kernel)), ' kernel!'
             else
               if (myid == 0) print *," ERROR in solve_evp_complex_2stage with ", &
                 trim(elpa_complex_kernel_name(input_options%this_complex_kernel)), ' kernel and qr-decompostion!'
             endif
#ifdef WITH_MPI
             call blacs_gridexit(my_blacs_ctxt)
             call mpi_finalize(mpierr)
#endif

             call EXIT(STATUS)
           endif

         endif
       endif ! realKernelSet
     endif ! datatype == 1
   endif ! do2Stage

   if (input_options%datatype .eq. 1) then
     deallocate(a_real)
     deallocate(as_real)

     deallocate(z_real)
     deallocate(tmp1_real)
     deallocate(tmp2_real)

     if (input_options%doInvertTrm .or. input_options%doTransposeMultiply) then
       deallocate(b_real)
       deallocate(bs_real)
     endif

     if (input_options%doTransposeMultiply) then
       deallocate(c_real)
     endif
   endif

   if (input_options%datatype .eq. 2) then
     deallocate(a_complex)
     deallocate(as_complex)

     deallocate(z_complex)
     deallocate(tmp1_complex)
     deallocate(tmp2_complex)

     if (input_options%doInvertTrm .or. input_options%doTransposeMultiply) then
       deallocate(b_complex)
       deallocate(bs_complex)
     endif

     if (input_options%doTransposeMultiply) then
       deallocate(c_complex)
     endif
   endif

   deallocate(ev)

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
