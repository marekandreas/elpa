#if 0
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
!    This particular source code file contains additions, changes and
!    enhancements authored by Intel Corporation which is not part of
!    the ELPA consortium.
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
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
#endif

#include "../general/sanity.F90"

function elpa_solve_evp_&
         &MATH_DATATYPE&
   &_1stage_&
   &PRECISION&
   &_impl (obj, a, ev, q) result(success)
   use precision
   use cuda_functions
   use mod_check_for_gpu
   use iso_c_binding
   use elpa_abstract_impl
   use elpa_mpi
   use elpa1_compute
   use elpa_omp

   implicit none
#include "../general/precision_kinds.F90"
   class(elpa_abstract_impl_t), intent(inout) :: obj
   real(kind=REAL_DATATYPE), intent(out)           :: ev(obj%na)

#ifdef USE_ASSUMED_SIZE
   MATH_DATATYPE(kind=rck), intent(inout)       :: a(obj%local_nrows,*)
   MATH_DATATYPE(kind=rck), optional,target,intent(out)  :: q(obj%local_nrows,*)
#else
   MATH_DATATYPE(kind=rck), intent(inout)       :: a(obj%local_nrows,obj%local_ncols)
#ifdef HAVE_SKEWSYMMETRIC
   MATH_DATATYPE(kind=C_DATATYPE_KIND), optional, target, intent(out) :: q(obj%local_nrows,2*obj%local_ncols)
#else
   MATH_DATATYPE(kind=C_DATATYPE_KIND), optional, target, intent(out) :: q(obj%local_nrows,obj%local_ncols)
#endif
#endif

#if REALCASE == 1
   real(kind=C_DATATYPE_KIND), allocatable         :: tau(:)
   real(kind=C_DATATYPE_KIND), allocatable, target         :: q_dummy(:,:)
   real(kind=C_DATATYPE_KIND), pointer             :: q_actual(:,:)
#endif /* REALCASE */

#if COMPLEXCASE == 1
   real(kind=REAL_DATATYPE), allocatable           :: q_real(:,:)
   complex(kind=C_DATATYPE_KIND), allocatable      :: tau(:)
   complex(kind=C_DATATYPE_KIND), allocatable,target :: q_dummy(:,:)
   complex(kind=C_DATATYPE_KIND), pointer          :: q_actual(:,:)
#endif /* COMPLEXCASE */


   integer(kind=c_int)                             :: l_cols, l_rows, l_cols_nev, np_rows, np_cols
   integer(kind=MPI_KIND)                          :: np_rowsMPI, np_colsMPI

   logical                                         :: useGPU
   integer(kind=c_int)                             :: skewsymmetric
   logical                                         :: isSkewsymmetric
   logical                                         :: success

   logical                                         :: do_useGPU, do_useGPU_tridiag, &
                                                      do_useGPU_solve_tridi, do_useGPU_trans_ev
   integer(kind=ik)                                :: numberOfGPUDevices

   integer(kind=c_int)                             :: my_pe, n_pes, my_prow, my_pcol
   integer(kind=MPI_KIND)                          :: mpierr, my_peMPI, n_pesMPI, my_prowMPI, my_pcolMPI
   real(kind=C_DATATYPE_KIND), allocatable         :: e(:)
   logical                                         :: wantDebug
   integer(kind=c_int)                             :: istat, debug, gpu
   character(200)                                  :: errorMessage
   integer(kind=ik)                                :: na, nev, lda, ldq, nblk, matrixCols, &
                                                      mpi_comm_rows, mpi_comm_cols,        &
                                                      mpi_comm_all, check_pd, i, error

   logical                                         :: do_tridiag, do_solve, do_trans_ev
   integer(kind=ik)                                :: nrThreads
   integer(kind=ik)                                :: global_index
   
   call obj%timer%start("elpa_solve_evp_&
   &MATH_DATATYPE&
   &_1stage_&
   &PRECISION&
   &")

#ifdef WITH_OPENMP
   ! store the number of OpenMP threads used in the calling function
   ! restore this at the end of ELPA 2
   omp_threads_caller = omp_get_max_threads()

   ! check the number of threads that ELPA should use internally
   call obj%get("omp_threads",nrThreads,error)
   call omp_set_num_threads(nrThreads)
#else
   nrThreads = 1
#endif
#ifdef WITH_NVTX
   call nvtxRangePush("elpa1")
#endif


   success = .true.

   if (present(q)) then
     obj%eigenvalues_only = .false.
   else
     obj%eigenvalues_only = .true.
   endif

   na         = obj%na
   nev        = obj%nev
   lda        = obj%local_nrows
   ldq        = obj%local_nrows
   nblk       = obj%nblk
   matrixCols = obj%local_ncols

   ! special case na = 1
   if (na .eq. 1) then
#if REALCASE == 1
     ev(1) = a(1,1)
#endif
#if COMPLEXCASE == 1
     ev(1) = real(a(1,1))
#endif
     if (.not.(obj%eigenvalues_only)) then
       q(1,1) = ONE
     endif

     ! restore original OpenMP settings
#ifdef WITH_OPENMP
     ! store the number of OpenMP threads used in the calling function
     ! restore this at the end of ELPA 2
     call omp_set_num_threads(omp_threads_caller)
#endif
     call obj%timer%stop("elpa_solve_evp_&
     &MATH_DATATYPE&
     &_1stage_&
     &PRECISION&
     &")
     return
   endif

   if (nev == 0) then
     nev = 1
     obj%eigenvalues_only = .true.
   endif


   call obj%get("mpi_comm_rows",mpi_comm_rows,error)
   if (error .ne. ELPA_OK) then
     print *,"Problem getting option. Aborting..."
     stop
   endif
   call obj%get("mpi_comm_cols",mpi_comm_cols,error)
   if (error .ne. ELPA_OK) then
     print *,"Problem getting option. Aborting..."
     stop
   endif

   call obj%get("gpu",gpu,error)
   if (error .ne. ELPA_OK) then
     print *,"Problem getting option. Aborting..."
     stop
   endif
   if (gpu .eq. 1) then
     useGPU =.true.
   else
     useGPU = .false.
   endif
   
   call obj%get("is_skewsymmetric",skewsymmetric,error)
   if (error .ne. ELPA_OK) then
     print *,"Problem getting option. Aborting..."
     stop
   endif
    
   isSkewsymmetric = (skewsymmetric == 1)
   
   call obj%timer%start("mpi_communication")

   call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND), my_prowMPI, mpierr)
   call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND), my_pcolMPI, mpierr)

   my_prow = int(my_prowMPI,kind=c_int)
   my_pcol = int(my_pcolMPI,kind=c_int)

   call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
   call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)

   np_rows = int(np_rowsMPI,kind=c_int)
   np_cols = int(np_colsMPI,kind=c_int)

   call obj%timer%stop("mpi_communication")

   call obj%get("debug", debug,error)
   if (error .ne. ELPA_OK) then
     print *,"Problem setting option. Aborting..."
     stop
   endif
   wantDebug = debug == 1
   do_useGPU = .false.

   
   if (useGPU) then
     call obj%timer%start("check_for_gpu")
     call obj%get("mpi_comm_parent", mpi_comm_all,error)
     if (error .ne. ELPA_OK) then
       print *,"Problem getting option. Aborting..."
       stop
     endif
     call mpi_comm_rank(int(mpi_comm_all,kind=MPI_KIND), my_peMPI, mpierr)
     my_pe = int(my_peMPI,kind=c_int)

     if (check_for_gpu(my_pe,numberOfGPUDevices, wantDebug=wantDebug)) then
       do_useGPU = .true.
       ! set the neccessary parameters
       cudaMemcpyHostToDevice   = cuda_memcpyHostToDevice()
       cudaMemcpyDeviceToHost   = cuda_memcpyDeviceToHost()
       cudaMemcpyDeviceToDevice = cuda_memcpyDeviceToDevice()
       cudaHostRegisterPortable = cuda_hostRegisterPortable()
       cudaHostRegisterMapped   = cuda_hostRegisterMapped()
     else
       print *,"GPUs are requested but not detected! Aborting..."
       success = .false.
       return
     endif
     call obj%timer%stop("check_for_gpu")
   endif


   do_useGPU_tridiag = do_useGPU
   do_useGPU_solve_tridi = do_useGPU
   do_useGPU_trans_ev = do_useGPU
   ! only if we want (and can) use GPU in general, look what are the
   ! requirements for individual routines. Implicitly they are all set to 1, so
   ! unles specified otherwise by the user, GPU versions of all individual
   ! routines should be used
   if(do_useGPU) then
     call obj%get("gpu_tridiag", gpu, error)
     if (error .ne. ELPA_OK) then
       print *,"Problem getting option. Aborting..."
       stop
     endif
     do_useGPU_tridiag = (gpu == 1)

     call obj%get("gpu_solve_tridi", gpu, error)
     if (error .ne. ELPA_OK) then
       print *,"Problem getting option. Aborting..."
       stop
     endif
     do_useGPU_solve_tridi = (gpu == 1)

     call obj%get("gpu_trans_ev", gpu, error)
     if (error .ne. ELPA_OK) then
       print *,"Problem getting option. Aborting..."
       stop
     endif
     do_useGPU_trans_ev = (gpu == 1)
   endif
   ! for elpa1 the easy thing is, that the individual phases of the algorithm
   ! do not share any data on the GPU. 


   ! allocate a dummy q_intern, if eigenvectors should not be commputed and thus q is NOT present
   if (.not.(obj%eigenvalues_only)) then
     q_actual => q(1:obj%local_nrows,1:obj%local_ncols)
   else
     allocate(q_dummy(obj%local_nrows,obj%local_ncols))
     q_actual => q_dummy
   endif

#if COMPLEXCASE == 1
   l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a and q
   l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of q

   l_cols_nev = local_index(nev, my_pcol, np_cols, nblk, -1) ! Local columns corresponding to nev

   allocate(q_real(l_rows,l_cols), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"solve_evp_&
     &MATH_DATATYPE&
     &_1stage_&
     &PRECISION&
     &" // ": error when allocating q_real "//errorMessage
     stop 1
   endif
#endif
   allocate(e(na), tau(na), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"solve_evp_&
     &MATH_DATATYPE&
     &_1stage_&
     &PRECISION&
     &" // ": error when allocating e, tau "//errorMessage
     stop 1
   endif


   ! start the computations
   ! as default do all three steps (this might change at some point)
   do_tridiag  = .true.
   do_solve    = .true.
   do_trans_ev = .true.

   if (do_tridiag) then
     call obj%timer%start("forward")
#ifdef HAVE_LIKWID
     call likwid_markerStartRegion("tridi")
#endif
#ifdef WITH_NVTX
     call nvtxRangePush("tridi")
#endif

     call tridiag_&
     &MATH_DATATYPE&
     &_&
     &PRECISION&
     & (obj, na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, ev, e, tau, do_useGPU_tridiag, wantDebug, nrThreads)

#ifdef WITH_NVTX
     call nvtxRangePop()
#endif
#ifdef HAVE_LIKWID
     call likwid_markerStopRegion("tridi")
#endif
     call obj%timer%stop("forward")
    endif  !do_tridiag

    if (do_solve) then
     call obj%timer%start("solve")
#ifdef HAVE_LIKWID
     call likwid_markerStartRegion("solve")
#endif
#ifdef WITH_NVTX
     call nvtxRangePush("solve")
#endif

     call solve_tridi_&
     &PRECISION&
     & (obj, na, nev, ev, e,  &
#if REALCASE == 1
        q_actual, ldq,          &
#endif
#if COMPLEXCASE == 1
        q_real, l_rows,  &
#endif
        nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, do_useGPU_solve_tridi, wantDebug, success, nrThreads)

#ifdef WITH_NVTX
     call nvtxRangePop()
#endif
#ifdef HAVE_LIKWID
     call likwid_markerStopRegion("solve")
#endif
     call obj%timer%stop("solve")
     if (.not.(success)) return
   endif !do_solve

   if (obj%eigenvalues_only) then
     do_trans_ev = .false.
   else
     call obj%get("check_pd",check_pd,error)
     if (error .ne. ELPA_OK) then
       print *,"Problem setting option. Aborting..."
       stop
     endif
     if (check_pd .eq. 1) then
       check_pd = 0
       do i = 1, na
         if (ev(i) .gt. THRESHOLD) then
           check_pd = check_pd + 1
         endif
       enddo
       if (check_pd .lt. na) then
         ! not positiv definite => eigenvectors needed
         do_trans_ev = .true.
       else
         do_trans_ev = .false.
       endif
     endif ! check_pd
   endif ! eigenvalues_only

   if (do_trans_ev) then
    ! q must be given thats why from here on we can use q and not q_actual
#if COMPLEXCASE == 1
     q(1:l_rows,1:l_cols_nev) = q_real(1:l_rows,1:l_cols_nev)
#endif
     if (isSkewsymmetric) then
     ! Extra transformation step for skew-symmetric matrix. Multiplication with diagonal complex matrix D.
     ! This makes the eigenvectors complex. 
     ! For now real part of eigenvectors is generated in first half of q, imaginary part in second part.
       q(1:obj%local_nrows, obj%local_ncols+1:2*obj%local_ncols) = 0.0
       do i = 1, obj%local_nrows
!        global_index = indxl2g(i, nblk, my_prow, 0, np_rows)
         global_index = np_rows*nblk*((i-1)/nblk) + MOD(i-1,nblk) + MOD(np_rows+my_prow-0, np_rows)*nblk + 1
         if (mod(global_index-1,4) .eq. 0) then
            ! do nothing
         end if
         if (mod(global_index-1,4) .eq. 1) then
            q(i,obj%local_ncols+1:2*obj%local_ncols) = q(i,1:obj%local_ncols)
            q(i,1:obj%local_ncols) = 0
         end if
         if (mod(global_index-1,4) .eq. 2) then
            q(i,1:obj%local_ncols) = -q(i,1:obj%local_ncols)
         end if
         if (mod(global_index-1,4) .eq. 3) then
            q(i,obj%local_ncols+1:2*obj%local_ncols) = -q(i,1:obj%local_ncols)
            q(i,1:obj%local_ncols) = 0
         end if
       end do       
     endif

     call obj%timer%start("back")
#ifdef HAVE_LIKWID
     call likwid_markerStartRegion("trans_ev")
#endif
#ifdef WITH_NVTX
     call nvtxRangePush("trans_ev")
#endif

     ! In the skew-symmetric case this transforms the real part
     call trans_ev_&
     &MATH_DATATYPE&
     &_&
     &PRECISION&
     & (obj, na, nev, a, lda, tau, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, do_useGPU_trans_ev)
     if (isSkewsymmetric) then
       ! Transform imaginary part
       ! Transformation of real and imaginary part could also be one call of trans_ev_tridi acting on the n x 2n matrix.
       call trans_ev_&
             &MATH_DATATYPE&
             &_&
             &PRECISION&
             & (obj, na, nev, a, lda, tau, q(1:obj%local_nrows, obj%local_ncols+1:2*obj%local_ncols), ldq, nblk, matrixCols, &
                mpi_comm_rows, mpi_comm_cols, do_useGPU_trans_ev)
       endif

#ifdef WITH_NVTX
     call nvtxRangePop()
#endif
#ifdef HAVE_LIKWID
     call likwid_markerStopRegion("trans_ev")
#endif
     call obj%timer%stop("back")
   endif ! do_trans_ev

#if COMPLEXCASE == 1
    deallocate(q_real, stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"solve_evp_&
      &MATH_DATATYPE&
      &_1stage_&
      &PRECISION&
      &" // ": error when deallocating q_real "//errorMessage
      stop 1
    endif
#endif

   deallocate(e, tau, stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"solve_evp_&
     &MATH_DATATYPE&
     &_1stage_&
     &PRECISION&
     &" // ": error when deallocating e, tau "//errorMessage
     stop 1
   endif

   if (obj%eigenvalues_only) then
     deallocate(q_dummy, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_evp_&
       &MATH_DATATYPE&
       &_1stage_&
       &PRECISION&
       &" // ": error when deallocating q_dummy "//errorMessage
       stop 1
     endif
   endif

#ifdef WITH_NVTX
   call nvtxRangePop()
#endif
   ! restore original OpenMP settings
#ifdef WITH_OPENMP
   ! store the number of OpenMP threads used in the calling function
   ! restore this at the end of ELPA 2
   call omp_set_num_threads(omp_threads_caller)
#endif

   call obj%timer%stop("elpa_solve_evp_&
   &MATH_DATATYPE&
   &_1stage_&
   &PRECISION&
   &")
end function


