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
 function elpa_solve_evp_&
  &MATH_DATATYPE&
  &_&
  &2stage_&
  &PRECISION&
  &_impl (obj, a, ev, q) result(success)

   use elpa_abstract_impl
   use elpa_utilities
   use elpa1_compute
   use elpa2_compute
   use elpa_mpi
   use cuda_functions
   use mod_check_for_gpu
   use iso_c_binding
   implicit none
   class(elpa_abstract_impl_t), intent(inout)                         :: obj
   logical                                                            :: useGPU
#if REALCASE == 1
   logical                                                            :: useQR
#endif
   logical                                                            :: useQRActual

   integer(kind=c_int)                                                :: bandwidth

   integer(kind=c_int)                                                :: kernel

#ifdef USE_ASSUMED_SIZE
   MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(inout)                 :: a(obj%local_nrows,*)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), optional, target, intent(out) :: q(obj%local_nrows,*)
#else
   MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(inout)                 :: a(obj%local_nrows,obj%local_ncols)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), optional, target, intent(out) :: q(obj%local_nrows,obj%local_ncols)
#endif
   real(kind=C_DATATYPE_KIND), intent(inout)                          :: ev(obj%na)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable                   :: hh_trans(:,:)

   integer(kind=c_int)                                                :: my_pe, n_pes, my_prow, my_pcol, np_rows, np_cols, mpierr
   integer(kind=c_int)                                                :: l_cols, l_rows, l_cols_nev, nbw, num_blocks
   MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable                   :: tmat(:,:,:)
   real(kind=C_DATATYPE_KIND), allocatable                            :: e(:)
#if COMPLEXCASE == 1
   real(kind=C_DATATYPE_KIND), allocatable                            :: q_real(:,:)
#endif
   MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable, target           :: q_dummy(:,:)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer                       :: q_actual(:,:)


   integer(kind=c_intptr_t)                                           :: tmat_dev, q_dev, a_dev

   integer(kind=c_int)                                                :: i
   logical                                                            :: success, successCUDA
   logical                                                            :: wantDebug
   integer(kind=c_int)                                                :: istat, gpu, debug, qr
   character(200)                                                     :: errorMessage
   logical                                                            :: do_useGPU, do_useGPU_trans_ev_tridi
   integer(kind=c_int)                                                :: numberOfGPUDevices
   integer(kind=c_intptr_t), parameter                                :: size_of_datatype = size_of_&
                                                                                            &PRECISION&
                                                                                            &_&
                                                                                            &MATH_DATATYPE
    integer(kind=ik)                                                  :: na, nev, lda, ldq, nblk, matrixCols, &
                                                                         mpi_comm_rows, mpi_comm_cols,        &
					                                 mpi_comm_all, check_pd

    logical                                                           :: do_bandred, do_tridiag, do_solve_tridi,  &
                                                                         do_trans_to_band, do_trans_to_full

    call obj%timer%start("elpa_solve_evp_&
    &MATH_DATATYPE&
    &_2stage_&
    &PRECISION&
    &")

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

#if REALCASE == 1
    call obj%get("real_kernel",kernel)
    ! check consistency between request for GPUs and defined kernel
    call obj%get("gpu", gpu)
    if (gpu == 1) then
      if (kernel .ne. ELPA_2STAGE_REAL_GPU) then
        write(error_unit,*) "ELPA: Warning, GPU usage has been requested but compute kernel is defined as non-GPU!"
	write(error_unit,*) "The compute kernel will be executed on CPUs!"
      else if (nblk .ne. 128) then
        kernel = ELPA_2STAGE_REAL_GENERIC
      endif
    endif
    if (kernel .eq. ELPA_2STAGE_REAL_GPU) then
      if (gpu .ne. 1) then
        write(error_unit,*) "ELPA: Warning, GPU usage has been requested but compute kernel is defined as non-GPU!"
      endif
    endif
#endif

#if COMPLEXCASE == 1
    call obj%get("complex_kernel",kernel)
    ! check consistency between request for GPUs and defined kernel
    call obj%get("gpu", gpu)
    if (gpu == 1) then
      if (kernel .ne. ELPA_2STAGE_COMPLEX_GPU) then
        write(error_unit,*) "ELPA: Warning, GPU usage has been requested but compute kernel is defined as non-GPU!"
	write(error_unit,*) "The compute kernel will be executed on CPUs!"
      else if (nblk .ne. 128) then
        kernel = ELPA_2STAGE_COMPLEX_GENERIC
      endif
    endif
    if (kernel .eq. ELPA_2STAGE_COMPLEX_GPU) then
      if (gpu .ne. 1) then
        write(error_unit,*) "ELPA: Warning, GPU usage has been requested but compute kernel is defined as non-GPU!"
      endif
    endif

#endif
    call obj%get("mpi_comm_rows",mpi_comm_rows)
    call obj%get("mpi_comm_cols",mpi_comm_cols)
    call obj%get("mpi_comm_parent",mpi_comm_all)

    if (gpu .eq. 1) then
      useGPU = .true.
    else
      useGPU = .false.
    endif

#if REALCASE == 1
    call obj%get("qr",qr)
    if (qr .eq. 1) then
      useQR = .true.
    else
      useQR = .false.
    endif

#endif
    call obj%timer%start("mpi_communication")
    call mpi_comm_rank(mpi_comm_all,my_pe,mpierr)
    call mpi_comm_size(mpi_comm_all,n_pes,mpierr)

    call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
    call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
    call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
    call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
    call obj%timer%stop("mpi_communication")

    call obj%get("debug",debug)
    wantDebug = debug == 1
    success = .true.

    do_useGPU      = .false.
    do_useGPU_trans_ev_tridi =.false.


#if REALCASE == 1
    useQRActual = .false.
    ! set usage of qr decomposition via API call
    if (useQR) useQRActual = .true.
    if (.not.(useQR)) useQRACtual = .false.

    if (useQRActual) then
      if (mod(na,2) .ne. 0) then
        if (wantDebug) then
          write(error_unit,*) "solve_evp_real_2stage: QR-decomposition: blocksize does not fit with matrixsize"
        endif
        print *, "Do not use QR-decomposition for this matrix and blocksize."
        success = .false.
        return
      endif
    endif
#endif /* REALCASE */

    if (useGPU) then
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
    else
      ! check whether set by environment variable
      call obj%get("gpu",gpu)
      do_useGPU = gpu == 1
      if (do_useGPU) then
        if (check_for_gpu(my_pe,numberOfGPUDevices, wantDebug=wantDebug)) then

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
      endif
    endif

    ! check consistency between request for GPUs and defined kernel
    if (do_useGPU) then
      if (nblk .ne. 128) then
        ! cannot run on GPU with this blocksize
        ! disable GPU usage for trans_ev_tridi
        do_useGPU_trans_ev_tridi = .false.
      else
#if REALCASE == 1
        if (kernel .eq. ELPA_2STAGE_REAL_GPU) then
#endif
#if COMPLEXCASE == 1
        if (kernel .eq. ELPA_2STAGE_COMPLEX_GPU) then
#endif
          do_useGPU_trans_ev_tridi = .true.
	else
          do_useGPU_trans_ev_tridi = .false.
	endif
      endif
    endif



    if (.not. obj%eigenvalues_only) then
      q_actual => q(1:obj%local_nrows,1:obj%local_ncols)
    else
     allocate(q_dummy(1:obj%local_nrows,1:obj%local_ncols))
     q_actual => q_dummy(1:obj%local_nrows,1:obj%local_ncols)
    endif


    ! set the default values for each of the 5 compute steps
    do_bandred        = .true.
    do_tridiag        = .true.
    do_solve_tridi    = .true.
    do_trans_to_band  = .true.
    do_trans_to_full  = .true.

    if (obj%eigenvalues_only) then
      do_trans_to_band  = .false.
      do_trans_to_full  = .false.
    endif

    if (obj%is_set("bandwidth") == 1) then
      call obj%get("bandwidth",nbw)
      if (nbw == 0) then
        if (wantDebug) then
          write(error_unit,*) "Specified bandwidth = 0; ELPA refuses to solve the eigenvalue problem ", &
	                      "for a diagonal matrix! This is too simple"
	  endif
        print *, "Specified bandwidth = 0; ELPA refuses to solve the eigenvalue problem ", &
	         "for a diagonal matrix! This is too simple"
        success = .false.
        return
      endif
      if (mod(nbw, nblk) .ne. 0) then
        ! treat matrix with an effective bandwidth slightly bigger than specified bandwidth
	! such that effective bandwidth is a multiply of nblk. which is a prerequiste for ELPA
        nbw = nblk * ceiling(real(nbw,kind=c_double)/real(nblk,kind=c_double))

        ! just check that effective bandwidth is NOT larger than matrix size
	if (nbw .gt. na) then
          if (wantDebug) then
            write(error_unit,*) "Specified bandwidth ",nbw," leads internaly to a computed bandwidth ", &
	                        "which is larger than the matrix size ",na," ! ELPA will abort! Try to", &
				"solve your problem by not specifing a bandwidth"
	  endif
          print *, "Specified bandwidth ",nbw," leads internaly to a computed bandwidth ", &
	                        "which is larger than the matrix size ",na," ! ELPA will abort! Try to", &
				"solve your problem by not specifing a bandwidth"
          success = .false.
          return
	endif
      endif
      do_bandred       = .false. ! we already have a banded matrix
      do_solve_tridi   = .true.  ! we also have to solve something :-)
      do_trans_to_band = .true.  ! and still we have to backsub to banded
      do_trans_to_full = .false. ! but not to full since we have a banded matrix
    else ! bandwidth is not set

      ! Choose bandwidth, must be a multiple of nblk, set to a value >= 32
      ! On older systems (IBM Bluegene/P, Intel Nehalem) a value of 32 was optimal.
      ! For Intel(R) Xeon(R) E5 v2 and v3, better use 64 instead of 32!
      ! For IBM Bluegene/Q this is not clear at the moment. We have to keep an eye
      ! on this and maybe allow a run-time optimization here
      if (do_useGPU) then
        nbw = nblk
      else
#if REALCASE == 1
        nbw = (63/nblk+1)*nblk
#elif COMPLEXCASE == 1
        nbw = (31/nblk+1)*nblk
#endif
      endif

      num_blocks = (na-1)/nbw + 1

      allocate(tmat(nbw,nbw,num_blocks), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_evp_&
        &MATH_DATATYPE&
        &_2stage_&
        &PRECISION&
        &" // ": error when allocating tmat "//errorMessage
        stop 1
      endif

      do_bandred       = .true.
      do_solve_tridi   = .true.
      do_trans_to_band = .true.
      do_trans_to_full = .true.
    end if  ! matrix not already banded on input

    ! start the computations in 5 steps

    if (do_bandred) then
      call obj%timer%start("bandred")
      ! Reduction full -> band
      call bandred_&
      &MATH_DATATYPE&
      &_&
      &PRECISION &
      (obj, na, a, &
      a_dev, lda, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, tmat, &
      tmat_dev,  wantDebug, do_useGPU, success &
#if REALCASE == 1
      , useQRActual &
#endif
      )
      call obj%timer%stop("bandred")
      if (.not.(success)) return
    endif


     ! Reduction band -> tridiagonal
     if (do_tridiag) then
       allocate(e(na), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"solve_evp_&
         &MATH_DATATYPE&
         &_2stage_&
         &PRECISION " // ": error when allocating e "//errorMessage
         stop 1
       endif

       call obj%timer%start("tridiag")
       call tridiag_band_&
       &MATH_DATATYPE&
       &_&
       &PRECISION&
       (obj, na, nbw, nblk, a, a_dev, lda, ev, e, matrixCols, hh_trans, mpi_comm_rows, mpi_comm_cols, mpi_comm_all, &
        do_useGPU, wantDebug)

#ifdef WITH_MPI
       call obj%timer%start("mpi_communication")
       call mpi_bcast(ev, na, MPI_REAL_PRECISION, 0, mpi_comm_all, mpierr)
       call mpi_bcast(e, na, MPI_REAL_PRECISION, 0, mpi_comm_all, mpierr)
       call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
       call obj%timer%stop("tridiag")
     endif ! do_tridiag

#if COMPLEXCASE == 1
     l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a and q
     l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of q
     l_cols_nev = local_index(nev, my_pcol, np_cols, nblk, -1) ! Local columns corresponding to nev

     allocate(q_real(l_rows,l_cols), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_evp_&
       &MATH_DATATYPE&
       &_2stage: error when allocating q_real"//errorMessage
       stop 1
     endif
#endif

     ! Solve tridiagonal system
     if (do_solve_tridi) then
       call obj%timer%start("solve")
       call solve_tridi_&
       &PRECISION &
       (obj, na, nev, ev, e, &
#if REALCASE == 1
       q_actual, ldq,   &
#endif
#if COMPLEXCASE == 1
       q_real, ubound(q_real,dim=1), &
#endif
       nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug, success)
       call obj%timer%stop("solve")
       if (.not.(success)) return
     endif ! do_solve_tridi

     deallocate(e, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_evp_&
       &MATH_DATATYPE&
       &_2stage: error when deallocating e "//errorMessage
       stop 1
     endif

     if (obj%eigenvalues_only) then
       do_trans_to_band = .false.
       do_trans_to_full = .false.
     else

       call obj%get("check_pd",check_pd)
       if (check_pd .eq. 1) then
         check_pd = 0
         do i = 1, na
           if (ev(i) .gt. THRESHOLD) then
             check_pd = check_pd + 1
           endif
         enddo
         if (check_pd .lt. na) then
           ! not positiv definite => eigenvectors needed
           do_trans_to_band = .true.
           do_trans_to_full = .true.
	 else
           do_trans_to_band = .false.
           do_trans_to_full = .false.
         endif
       endif
     endif ! eigenvalues only

     if (do_trans_to_band) then
#if COMPLEXCASE == 1
       ! q must be given thats why from here on we can use q and not q_actual

       q(1:l_rows,1:l_cols_nev) = q_real(1:l_rows,1:l_cols_nev)

       deallocate(q_real, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"solve_evp_&
         &MATH_DATATYPE&
         &_2stage: error when deallocating q_real"//errorMessage
         stop 1
       endif
#endif

       ! Backtransform stage 1
       call obj%timer%start("trans_ev_to_band")

       call trans_ev_tridi_to_band_&
       &MATH_DATATYPE&
       &_&
       &PRECISION &
       (obj, na, nev, nblk, nbw, q, &
       q_dev, &
       ldq, matrixCols, hh_trans, mpi_comm_rows, mpi_comm_cols, wantDebug, do_useGPU_trans_ev_tridi, &
       success=success, kernel=kernel)
       call obj%timer%stop("trans_ev_to_band")

       if (.not.(success)) return

       ! We can now deallocate the stored householder vectors
       deallocate(hh_trans, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *, "solve_evp_&
         &MATH_DATATYPE&
         &_2stage_&
         &PRECISION " // ": error when deallocating hh_trans "//errorMessage
         stop 1
       endif
     endif ! do_trans_to_band

     if (do_trans_to_full) then
       call obj%timer%start("trans_ev_to_full")
       if ( (do_useGPU) .and. .not.(do_useGPU_trans_ev_tridi) ) then
         ! copy to device if we want to continue on GPU
         successCUDA = cuda_malloc(q_dev, ldq*matrixCols*size_of_datatype)

         successCUDA = cuda_memcpy(q_dev, loc(q), ldq*matrixCols* size_of_datatype, cudaMemcpyHostToDevice)
       endif

       ! Backtransform stage 2

       call trans_ev_band_to_full_&
       &MATH_DATATYPE&
       &_&
       &PRECISION &
       (obj, na, nev, nblk, nbw, a, &
       a_dev, lda, tmat, tmat_dev,  q,  &
       q_dev, &
       ldq, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, do_useGPU &
#if REALCASE == 1
       , useQRActual  &
#endif
       )


       deallocate(tmat, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"solve_evp_&
         &MATH_DATATYPE&
         &_2stage_&
         &PRECISION " // ": error when deallocating tmat"//errorMessage
         stop 1
       endif
       call obj%timer%stop("trans_ev_to_full")
     endif ! do_trans_to_full

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

     call obj%timer%stop("elpa_solve_evp_&
     &MATH_DATATYPE&
     &_2stage_&
    &PRECISION&
    &")
1    format(a,f10.3)

   end function elpa_solve_evp_&
   &MATH_DATATYPE&
   &_2stage_&
   &PRECISION&
   &_impl

! vim: syntax=fortran
