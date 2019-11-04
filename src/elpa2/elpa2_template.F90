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

#include "elpa/elpa_simd_constants.h"

 function elpa_solve_evp_&
  &MATH_DATATYPE&
  &_&
  &2stage_&
  &PRECISION&
  &_impl (obj, a, ev, q) result(success)
   use matrix_plot
   use elpa_abstract_impl
   use elpa_utilities
   use elpa1_compute
   use elpa2_compute
   use elpa_mpi
   use cuda_functions
   use mod_check_for_gpu
   use elpa_omp
#ifdef HAVE_HETEROGENOUS_CLUSTER_SUPPORT
   use simd_kernel
#endif
   use iso_c_binding
   implicit none
#include "../general/precision_kinds.F90"
   class(elpa_abstract_impl_t), intent(inout)                         :: obj
   logical                                                            :: useGPU
   logical                                                            :: isSkewsymmetric
#if REALCASE == 1
   logical                                                            :: useQR
   logical                                                            :: useQRActual
#endif
   integer(kind=c_int)                                                :: kernel, kernelByUser

#ifdef USE_ASSUMED_SIZE
   MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(inout)                 :: a(obj%local_nrows,*)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), optional, intent(out), target :: q(obj%local_nrows,*)
#else
   MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(inout)                 :: a(obj%local_nrows,obj%local_ncols)
#ifdef HAVE_SKEWSYMMETRIC
   MATH_DATATYPE(kind=C_DATATYPE_KIND), optional, target, intent(out) :: q(obj%local_nrows,2*obj%local_ncols)
#else
   MATH_DATATYPE(kind=C_DATATYPE_KIND), optional, target, intent(out) :: q(obj%local_nrows,obj%local_ncols)
#endif
#endif
   real(kind=C_DATATYPE_KIND), intent(inout)                          :: ev(obj%na)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable                   :: hh_trans(:,:)

   integer(kind=c_int)                                                :: my_pe, n_pes, my_prow, my_pcol, np_rows, np_cols
   integer(kind=MPI_KIND)                                             :: my_peMPI, n_pesMPI, my_prowMPI, my_pcolMPI, &
                                                                         np_rowsMPI, np_colsMPI, mpierr
   integer(kind=c_int)                                                :: nbw, num_blocks
#if COMPLEXCASE == 1
   integer(kind=c_int)                                                :: l_cols_nev, l_rows, l_cols
#endif
   MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable                   :: tmat(:,:,:)
   real(kind=C_DATATYPE_KIND), allocatable                            :: e(:)
#if COMPLEXCASE == 1
   real(kind=C_DATATYPE_KIND), allocatable                            :: q_real(:,:)
#endif
   MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable, target           :: q_dummy(:,:)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer                       :: q_actual(:,:)


   integer(kind=c_intptr_t)                                           :: tmat_dev, q_dev, a_dev

   integer(kind=c_int)                                                :: i, j
   logical                                                            :: success, successCUDA
   logical                                                            :: wantDebug
   integer(kind=c_int)                                                :: istat, gpu, skewsymmetric, debug, qr
   character(200)                                                     :: errorMessage
   logical                                                            :: do_useGPU, do_useGPU_bandred, &
                                                                         do_useGPU_tridiag_band, do_useGPU_solve_tridi, &
                                                                         do_useGPU_trans_ev_tridi_to_band, &
                                                                         do_useGPU_trans_ev_band_to_full
   integer(kind=c_int)                                                :: numberOfGPUDevices
   integer(kind=c_intptr_t), parameter                                :: size_of_datatype = size_of_&
                                                                                            &PRECISION&
                                                                                            &_&
                                                                                            &MATH_DATATYPE
    integer(kind=ik)                                                  :: na, nev, lda, ldq, nblk, matrixCols, &
                                                                         mpi_comm_rows, mpi_comm_cols,        &
                                                                         mpi_comm_all, check_pd, error

    logical                                                           :: do_bandred, do_tridiag, do_solve_tridi,  &
                                                                         do_trans_to_band, do_trans_to_full

    integer(kind=ik)                                                  :: nrThreads
#ifdef HAVE_HETEROGENOUS_CLUSTER_SUPPORT
    integer(kind=c_int)                                               :: simdSetAvailable(NUMBER_OF_INSTR)
#endif
    integer(kind=ik)                                                  :: global_index

#if REALCASE == 1
#undef GPU_KERNEL
#undef GENERIC_KERNEL
#undef KERNEL_STRING
#define GPU_KERNEL ELPA_2STAGE_REAL_GPU
#define GENERIC_KERNEL ELPA_2STAGE_REAL_GENERIC
#define KERNEL_STRING "real_kernel"
#endif
#if COMPLEXCASE == 1
#undef GPU_KERNEL
#undef GENERIC_KERNEL
#undef KERNEL_STRING
#define GPU_KERNEL ELPA_2STAGE_COMPLEX_GPU
#define GENERIC_KERNEL ELPA_2STAGE_COMPLEX_GENERIC
#define KERNEL_STRING "complex_kernel"
#endif

    call obj%timer%start("elpa_solve_evp_&
    &MATH_DATATYPE&
    &_2stage_&
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
    call obj%get("mpi_comm_parent",mpi_comm_all,error)
    if (error .ne. ELPA_OK) then
      print *,"Problem getting option. Aborting..."
      stop
    endif

    call obj%timer%start("mpi_communication")
    call mpi_comm_rank(int(mpi_comm_all,kind=MPI_KIND) ,my_peMPI ,mpierr)
    call mpi_comm_size(int(mpi_comm_all,kind=MPI_KIND) ,n_pesMPI ,mpierr)

    call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND) ,my_prowMPI ,mpierr)
    call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND) ,np_rowsMPI ,mpierr)
    call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND) ,my_pcolMPI ,mpierr)
    call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND) ,np_colsMPI ,mpierr)

    my_pe = int(my_peMPI, kind=c_int)
    n_pes = int(n_pesMPI, kind=c_int)
    my_prow = int(my_prowMPI, kind=c_int)
    np_rows = int(np_rowsMPI, kind=c_int)
    my_pcol = int(my_pcolMPI, kind=c_int)
    np_cols = int(np_colsMPI, kind=c_int)

    call obj%timer%stop("mpi_communication")

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
     &_2stage_&
     &PRECISION&
     &")
     return
   endif

   if (nev == 0) then
     nev = 1
     obj%eigenvalues_only = .true.
   endif

    call obj%get(KERNEL_STRING,kernel,error)
    if (error .ne. ELPA_OK) then
      print *,"Problem getting option. Aborting..."
      stop
    endif
 
    call obj%get("is_skewsymmetric",skewsymmetric,error)
    if (error .ne. ELPA_OK) then
      print *,"Problem getting option. Aborting..."
      stop
    endif

    isSkewsymmetric = (skewsymmetric == 1)

    ! GPU settings
    call obj%get("gpu", gpu,error)
    if (error .ne. ELPA_OK) then
      print *,"Problem getting option. Aborting..."
      stop
    endif

    useGPU = (gpu == 1)

    do_useGPU = .false.
    if (useGPU) then
      call obj%timer%start("check_for_gpu")
      if (check_for_gpu(my_pe,numberOfGPUDevices, wantDebug=wantDebug)) then

         do_useGPU = .true.
         a_dev = 0

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

    do_useGPU_bandred = do_useGPU
    ! tridiag-band not ported to GPU yet
    do_useGPU_tridiag_band = .false.
    do_useGPU_solve_tridi = do_useGPU
    ! trans tridi to band GPU implementation does not work properly
    do_useGPU_trans_ev_tridi_to_band = .false.
    do_useGPU_trans_ev_band_to_full = do_useGPU

    ! only if we want (and can) use GPU in general, look what are the
    ! requirements for individual routines. Implicitly they are all set to 1, so
    ! unles specified otherwise by the user, GPU versions of all individual
    ! routines should be used
    if(do_useGPU) then
      call obj%get("gpu_bandred", gpu, error)
      if (error .ne. ELPA_OK) then
        print *,"Problem getting option. Aborting..."
        stop
      endif
      do_useGPU_bandred = (gpu == 1)

!      call obj%get("gpu_tridiag_band", gpu, error)
!      if (error .ne. ELPA_OK) then
!        print *,"Problem getting option. Aborting..."
!        stop
!      endif
!      do_useGPU_tridiag_band = (gpu == 1)
    ! tridiag-band not ported to GPU yet
      do_useGPU_tridiag_band = .false.

      call obj%get("gpu_solve_tridi", gpu, error)
      if (error .ne. ELPA_OK) then
        print *,"Problem getting option. Aborting..."
        stop
      endif
      do_useGPU_solve_tridi = (gpu == 1)

!      call obj%get("gpu_trans_ev_tridi_to_band", gpu, error)
!      if (error .ne. ELPA_OK) then
!        print *,"Problem getting option. Aborting..."
!        stop
!      endif
!      do_useGPU_trans_ev_tridi_to_band = (gpu == 1)
      do_useGPU_trans_ev_tridi_to_band = .false.

      call obj%get("gpu_trans_ev_band_to_full", gpu, error)
      if (error .ne. ELPA_OK) then
        print *,"Problem getting option. Aborting..."
        stop
      endif
      do_useGPU_trans_ev_band_to_full = (gpu == 1)
    endif

    ! check consistency between request for GPUs and defined kernel
    if (do_useGPU_trans_ev_tridi_to_band) then
    !!! this currently cannot happen,  GPU_trans_ev_tridi_to_band is always false
        write(error_unit,*) "ELPA: internal error!"
        stop
!      if (kernel .ne. GPU_KERNEL) then
!        write(error_unit,*) "ELPA: Warning, GPU usage has been requested but compute kernel is defined as non-GPU!"
!        write(error_unit,*) "The compute kernel will be executed on CPUs!"
!        do_useGPU_trans_ev_tridi_to_band = .false.
!      else if (nblk .ne. 128) then
!        write(error_unit,*) "ELPA: Warning, GPU kernel can run only with scalapack block size 128!"
!        write(error_unit,*) "The compute kernel will be executed on CPUs!"
!        do_useGPU_trans_ev_tridi_to_band = .false.
!        kernel = GENERIC_KERNEL
!      endif
    else
      if (kernel .eq. GPU_KERNEL) then
        ! We have currently forbidden to use GPU version of trans ev tridi to band, but we did not forbid the possibility
        ! to select the GPU kernel. If done such, give warning and swicht to the  generic kernel
        ! TODO it would be better to forbid the possibility to set the GPU kernel completely
        write(error_unit,*) "ELPA: ERROR, GPU kernel currently not implemented.&
                             & Use optimized CPU kernel even for GPU runs! &
                           Switching to the non-optimized generic kernel"
        kernel = GENERIC_KERNEL
      endif
    endif

    ! check again, now kernel and do_useGPU_trans_ev_tridi_to_band sould be
    ! finally consistent
    if (do_useGPU_trans_ev_tridi_to_band) then
    !!! this currently cannot happen,  GPU_trans_ev_tridi_to_band is always false
      write(error_unit,*) "ELPA: internal error!"
      stop
!      if (kernel .ne. GPU_KERNEL) then
!        ! this should never happen, checking as an assert
!        write(error_unit,*) "ELPA: INTERNAL ERROR setting GPU kernel!  Aborting..."
!        stop
!      endif
!      if (nblk .ne. 128) then
!        ! this should never happen, checking as an assert
!        write(error_unit,*) "ELPA: INTERNAL ERROR setting GPU kernel and blocksize!  Aborting..."
!        stop
!      endif
    else
      if (kernel .eq. GPU_KERNEL) then
        ! combination not allowed
        write(error_unit,*) "ELPA: Warning, GPU usage has NOT been requested but compute kernel &
                            &is defined as the GPU kernel!  Aborting..."
        stop
        !TODO do error handling properly
      endif
    endif


#if REALCASE == 1
#ifdef SINGLE_PRECISION_REAL
    ! special case at the moment NO single precision kernels on POWER 8 -> set GENERIC for now
    if (kernel .eq. ELPA_2STAGE_REAL_VSX_BLOCK2 .or. &
        kernel .eq. ELPA_2STAGE_REAL_VSX_BLOCK4 .or. &
        kernel .eq. ELPA_2STAGE_REAL_VSX_BLOCK6        ) then
        write(error_unit,*) "ELPA: At the moment there exist no specific SINGLE precision kernels for POWER8"
        write(error_unit,*) "The GENERIC kernel will be used at the moment"
        kernel = ELPA_2STAGE_REAL_GENERIC
    endif
    ! special case at the moment NO single precision kernels on SPARC64 -> set GENERIC for now
    if (kernel .eq. ELPA_2STAGE_REAL_SPARC64_BLOCK2 .or. &
        kernel .eq. ELPA_2STAGE_REAL_SPARC64_BLOCK4 .or. &
        kernel .eq. ELPA_2STAGE_REAL_SPARC64_BLOCK6        ) then
        write(error_unit,*) "ELPA: At the moment there exist no specific SINGLE precision kernels for SPARC64"
        write(error_unit,*) "The GENERIC kernel will be used at the moment"
        kernel = ELPA_2STAGE_REAL_GENERIC
    endif
#endif

#endif

     ! consistency check: is user set kernel still identical with "kernel" or did
     ! we change it above? This is a mess and should be cleaned up
     call obj%get(KERNEL_STRING,kernelByUser,error)
     if (error .ne. ELPA_OK) then
       print *,"Problem getting option. Aborting..."
       stop
     endif

     if (kernelByUser .ne. kernel) then
       call obj%set(KERNEL_STRING, kernel, error)
       if (error .ne. ELPA_OK) then
         print *,"Problem setting option. Aborting..."
         stop
       endif
     endif

#ifdef HAVE_HETEROGENOUS_CLUSTER_SUPPORT
     ! find a kernel which is supported on all used CPUs
     ! at the moment this works only on Intel CPUs
     simdSetAvailable(:) = 0
     call get_cpuid_set(simdSetAvailable, NUMBER_OF_INSTR)
#ifdef WITH_MPI
     call MPI_ALLREDUCE(mpi_in_place, simdSetAvailable, NUMBER_OF_INSTR, MPI_INTEGER, MPI_BAND, int(mpi_comm_all,kind=MPI_KIND), mpierr)
#endif

     ! compare user chosen kernel with possible kernels
     call obj%get(KERNEL_STRING,kernelByUser,error)
     if (error .ne. ELPA_OK) then
       print *,"Problem getting option. Aborting..."
       stop
     endif

     ! map kernel to SIMD Set, and check whether this is set is available on all cores

#if REALCASE == 1
    if (simdSetAvailable(map_real_kernel_to_simd_instruction(kernelByUser)) /= 1) then
#endif
#if COMPLEXCASE == 1
    if (simdSetAvailable(map_complex_kernel_to_simd_instruction(kernelByUser)) /=1) then
#endif

      ! if we are not purely running on Intel CPUs, this feature does not work at the moment
      ! this restriction should be lifted step by step
      if (simdSetAvailable(CPU_MANUFACTURER) /= 1) then
         if (my_pe == 0 ) then
         write(error_unit,*) "You enabled the experimental feature of an heterogenous cluster support."
         write(error_unit,*) "However, this works at the moment only if ELPA is run on (different) Intel CPUs!"
         write(error_unit,*) "ELPA detected also non Intel-CPUs, and will this abort now"
         stop
        endif
      else
        if (my_pe == 0 ) then
          write(error_unit,*) "The ELPA 2stage kernel of your choice, cannot be run on all CPUs"
          write(error_unit,*) "ELPA will use another kernel..."
        endif

        ! find best kernel available for supported instruction sets
        do i = NUMBER_OF_INSTR, 2, -1
          if (simdSetAvailable(i) == 1) then
            ! map to "best" kernel with this instruction set
            ! this can be only done for kernels that ELPA has been configured to use
#if REALCASE == 1
            kernel = map_simd_instruction_to_real_kernel(i)
#endif
#if COMPLEXCASE == 1
            kernel = map_simd_instruction_to_complex_kernel(i)
#endif
            if (obj%can_set(KERNEL_STRING, kernel) == ELPA_OK) then
              call obj%set(KERNEL_STRING, kernel, error)
              if (error .ne. ELPA_OK) then
                print *,"Problem setting option. Aborting..."
                stop
              endif
              if (my_pe == 0 ) write(error_unit,*) "ELPA decided to use ",elpa_int_value_to_string(KERNEL_STRING, kernel)
              exit 
            endif
          endif
        enddo
      endif

    endif
#endif /* HAVE_HETEROGENOUS_CLUSTER_SUPPORT */

#if REALCASE == 1
    call obj%get("qr",qr,error)
    if (error .ne. ELPA_OK) then
      print *,"Problem getting option. Aborting..."
      stop
    endif
    if (qr .eq. 1) then
      useQR = .true.
    else
      useQR = .false.
    endif

#endif

    call obj%get("debug",debug,error)
    if (error .ne. ELPA_OK) then
      print *,"Problem getting option. Aborting..."
      stop
    endif
    wantDebug = debug == 1



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
      ! bandwidth is set. That means, that the inputed matrix is actually banded and thus the 
      ! first step of ELPA2 should be skipped
      call obj%get("bandwidth",nbw,error)
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
    else ! matrix is not banded, determine the intermediate bandwidth for full->banded->tridi
      !first check if the intermediate bandwidth was set by the user
      call obj%get("intermediate_bandwidth", nbw, error)
      if (error .ne. ELPA_OK) then
        print *,"Problem getting option. Aborting..."
        stop
      endif

      if(nbw == 0) then
        ! intermediate bandwidth was not specified, select one of the defaults

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

      else
        ! intermediate bandwidth has been specified by the user, check, whether correctly
        if (mod(nbw, nblk) .ne. 0) then
          print *, "Specified bandwidth ",nbw," has to be mutiple of the blocksize ", nblk, ". Aborting..."
          success = .false.
          return
        endif
      endif !nbw == 0

      num_blocks = (na-1)/nbw + 1

      ! tmat is needed only in full->band and band->full steps, so alocate here
      ! (not allocated for banded matrix on input)
      allocate(tmat(nbw,nbw,num_blocks), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_evp_&
        &MATH_DATATYPE&
        &_2stage_&
        &PRECISION&
        &" // ": error when allocating tmat "//errorMessage
        stop 1
      endif

      ! if either of full->band or band->full steps are to be done on GPU,
      ! allocate also corresponding array on GPU.
      if (do_useGPU_bandred .or.  do_useGPU_trans_ev_band_to_full) then
        successCUDA = cuda_malloc(tmat_dev, nbw*nbw* size_of_datatype)
        if (.not.(successCUDA)) then
          print *,"bandred_&
                  &MATH_DATATYPE&
                  &: error in cudaMalloc tmat_dev 1"
          stop 1
        endif
      endif

      do_bandred       = .true.
      do_solve_tridi   = .true.
      do_trans_to_band = .true.
      do_trans_to_full = .true.
    endif  ! matrix not already banded on input

    ! start the computations in 5 steps
!     print *
!     print *, 'do_useGPU_bandred', do_useGPU_bandred
!     print *, 'do_useGPU_tridiag_band', do_useGPU_tridiag_band
!     print *, 'do_useGPU_solve_tridi', do_useGPU_solve_tridi
!     print *, 'do_useGPU_trans_ev_tridi_to_band', do_useGPU_trans_ev_tridi_to_band
!     print *, 'do_useGPU_trans_ev_band_to_full', do_useGPU_trans_ev_band_to_full
!     print *
    if (do_bandred) then
!       print *, 'do_useGPU_bandred=', do_useGPU_bandred
      call obj%timer%start("bandred")
#ifdef HAVE_LIKWID
      call likwid_markerStartRegion("bandred")
#endif
      ! Reduction full -> band
      call bandred_&
      &MATH_DATATYPE&
      &_&
      &PRECISION &
      (obj, na, a, &
      a_dev, lda, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, tmat, &
      tmat_dev,  wantDebug, do_useGPU_bandred, success, &
#if REALCASE == 1
      useQRActual, &
#endif
       nrThreads)
#ifdef HAVE_LIKWID
      call likwid_markerStopRegion("bandred")
#endif
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
#ifdef HAVE_LIKWID
       call likwid_markerStartRegion("tridiag")
#endif
       call tridiag_band_&
       &MATH_DATATYPE&
       &_&
       &PRECISION&
       (obj, na, nbw, nblk, a, a_dev, lda, ev, e, matrixCols, hh_trans, mpi_comm_rows, mpi_comm_cols, mpi_comm_all, &
       do_useGPU_tridiag_band, wantDebug, nrThreads)

#ifdef WITH_MPI
       call obj%timer%start("mpi_communication")
       call mpi_bcast(ev, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, 0_MPI_KIND, int(mpi_comm_all,kind=MPI_KIND), mpierr)
       call mpi_bcast(e, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, 0_MPI_KIND, int(mpi_comm_all,kind=MPI_KIND), mpierr)
       call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
#ifdef HAVE_LIKWID
       call likwid_markerStopRegion("tridiag")
#endif
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
!        print *, 'do_useGPU_solve_tridi=', do_useGPU_solve_tridi
       call obj%timer%start("solve")
#ifdef HAVE_LIKWID
       call likwid_markerStartRegion("solve")
#endif
       call solve_tridi_&
       &PRECISION &
       (obj, na, nev, ev, e, &
#if REALCASE == 1
       q_actual, ldq,   &
#endif
#if COMPLEXCASE == 1
       q_real, ubound(q_real,dim=1), &
#endif
       nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, do_useGPU_solve_tridi, wantDebug, success, nrThreads)
#ifdef HAVE_LIKWID
       call likwid_markerStopRegion("solve")
#endif
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

       call obj%get("check_pd",check_pd,error)
       if (error .ne. ELPA_OK) then
         print *,"Problem getting option. Aborting..."
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
     endif
 
       if (isSkewsymmetric) then
       ! Extra transformation step for skew-symmetric matrix. Multiplication with diagonal complex matrix D.
       ! This makes the eigenvectors complex.
       ! For now real part of eigenvectors is generated in first half of q, imaginary part in second part.
         q(1:obj%local_nrows, obj%local_ncols+1:2*obj%local_ncols) = 0.0
         do i = 1, obj%local_nrows
!          global_index = indxl2g(i, nblk, my_prow, 0, np_rows)
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
!        print * , "q="
!        do i=1,na
!          write(*,"(100g15.5)") ( q(i,j), j=1,na )
!        enddo
       ! Backtransform stage 1
     if (do_trans_to_band) then
       call obj%timer%start("trans_ev_to_band")
#ifdef HAVE_LIKWID
       call likwid_markerStartRegion("trans_ev_to_band")
#endif

       ! In the skew-symmetric case this transforms the real part
       call trans_ev_tridi_to_band_&
       &MATH_DATATYPE&
       &_&
       &PRECISION &
       (obj, na, nev, nblk, nbw, q, &
       q_dev, &
       ldq, matrixCols, hh_trans, mpi_comm_rows, mpi_comm_cols, wantDebug, do_useGPU_trans_ev_tridi_to_band, &
       nrThreads, success=success, kernel=kernel)
!        if (isSkewsymmetric) then
!        ! Transform imaginary part
!        ! Transformation of real and imaginary part could also be one call of trans_ev_tridi acting on the n x 2n matrix.
!          call trans_ev_tridi_to_band_&
!          &MATH_DATATYPE&
!          &_&
!          &PRECISION &
!          (obj, na, nev, nblk, nbw, q(1:obj%local_nrows, obj%local_ncols+1:2*obj%local_ncols), &
!          q_dev, &
!          ldq, matrixCols, hh_trans, mpi_comm_rows, mpi_comm_cols, wantDebug, do_useGPU_trans_ev_tridi_to_band, &
!          nrThreads, success=success, kernel=kernel)
!        endif
!        print * , "After trans_ev_tridi_to_band: real part of q="
!        do i=1,na
!          write(*,"(100g15.5)") ( q(i,j), j=1,na )
!        enddo
! #ifdef DOUBLE_PRECISION_REAL
!        call prmat(na,useGPU,q(1:obj%local_nrows, obj%local_ncols+1:2*obj%local_ncols),q_dev,lda,matrixCols,nblk,my_prow,my_pcol,np_rows,np_cols,'R',0)
! #endif
#ifdef HAVE_LIKWID
       call likwid_markerStopRegion("trans_ev_to_band")
#endif
       call obj%timer%stop("trans_ev_to_band")

       if (.not.(success)) return
 
!        ! We can now deallocate the stored householder vectors
!        deallocate(hh_trans, stat=istat, errmsg=errorMessage)
!        if (istat .ne. 0) then
!          print *, "solve_evp_&
!          &MATH_DATATYPE&
!          &_2stage_&
!          &PRECISION " // ": error when deallocating hh_trans "//errorMessage
!          stop 1
!        endif
     endif ! do_trans_to_band
!      print *, 'after do_useGPU_trans_ev_tridi_to_band', do_useGPU_trans_ev_tridi_to_band
!      print*, 'do_useGPU_trans_ev_band_to_full=', do_useGPU_trans_ev_band_to_full
     ! the array q might reside on device or host, depending on whether GPU is
     ! used or not. We thus have to transfer he data manually, if one of the
     ! routines is run on GPU and the other not.

     ! first deal with the situation that first backward step was on GPU
     if(do_useGPU_trans_ev_tridi_to_band) then
       ! if the second backward step is to be performed, but not on GPU, we have
       ! to transfer q to the host
       if(do_trans_to_full .and. (.not. do_useGPU_trans_ev_band_to_full)) then
         successCUDA = cuda_memcpy(int(loc(q),kind=c_intptr_t), q_dev, ldq*matrixCols* size_of_datatype, cudaMemcpyDeviceToHost)
         if (.not.(successCUDA)) then
           print *,"elpa2_template, error in copy to host"
           stop 1
         endif
       endif

       ! if the last step is not required at all, or will be performed on CPU,
       ! release the memmory allocated on the device
       if((.not. do_trans_to_full) .or. (.not. do_useGPU_trans_ev_band_to_full)) then
         successCUDA = cuda_free(q_dev)
         print *, 'q_dev is freed'
       endif
     endif

     !TODO check that the memory is properly deallocated on the host in case that
     !the last step is not required

     if (do_trans_to_full) then
       call obj%timer%start("trans_ev_to_full")
#ifdef HAVE_LIKWID
       call likwid_markerStartRegion("trans_ev_to_full")
#endif
       if ( (do_useGPU_trans_ev_band_to_full) .and. .not.(do_useGPU_trans_ev_tridi_to_band) ) then
         ! copy to device if we want to continue on GPU
 
         successCUDA = cuda_malloc(q_dev, ldq*matrixCols*size_of_datatype)
!          if (.not.(successCUDA)) then
!            print *,"elpa2_template, error in cuda_malloc"
!            stop 1
!          endif         
!          print *, 'q_dev=', q_dev, 'loc(q)=', loc(q)&
!          , 'ldq*matrixCols* size_of_datatype=', ldq*matrixCols* size_of_datatype, ', q(1,1)=', q(1,1)

         successCUDA = cuda_memcpy(q_dev, int(loc(q),kind=c_intptr_t), ldq*matrixCols* size_of_datatype, cudaMemcpyHostToDevice)
         if (.not.(successCUDA)) then
           print *,"elpa2_template, error in copy to device", successCUDA
           stop 1
         endif
       endif
         
       ! Backtransform stage 2
       ! In the skew-symemtric case this transforms the real part
       
       call trans_ev_band_to_full_&
       &MATH_DATATYPE&
       &_&
       &PRECISION &
       (obj, na, nev, nblk, nbw, a, &
       a_dev, lda, tmat, tmat_dev,  q,  &
       q_dev, &
       ldq, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, do_useGPU_trans_ev_band_to_full &
#if REALCASE == 1
       , useQRActual  &
#endif
       )
!        print * , "After trans_ev_band_to_full: real part of q="
!        do i=1,na
!          write(*,"(100g15.5)") ( q(i,j), j=1,na )
!        enddo
       call obj%timer%stop("trans_ev_to_full")
     endif ! do_trans_to_full
! #ifdef DOUBLE_PRECISION_REAL
!        call prmat(na,useGPU,q(1:obj%local_nrows, 1:obj%local_ncols),q_dev,lda,matrixCols,nblk,my_prow,my_pcol,np_rows,np_cols,'R',1)
! #endif
!        New position:
     if (do_trans_to_band) then
       if (isSkewsymmetric) then
         ! Transform imaginary part
         ! Transformation of real and imaginary part could also be one call of trans_ev_tridi acting on the n x 2n matrix.
           call trans_ev_tridi_to_band_&
           &MATH_DATATYPE&
           &_&
           &PRECISION &
           (obj, na, nev, nblk, nbw, q(1:obj%local_nrows, obj%local_ncols+1:2*obj%local_ncols), &
           q_dev, &
           ldq, matrixCols, hh_trans, mpi_comm_rows, mpi_comm_cols, wantDebug, do_useGPU_trans_ev_tridi_to_band, &
           nrThreads, success=success, kernel=kernel)
         endif
!          print * , "After trans_ev_tridi_to_band: imaginary part of q="
!          do i=1,na
!            write(*,"(100g15.5)") ( q(i,j+na), j=1,na )
!          enddo
! #ifdef DOUBLE_PRECISION_REAL
!        call prmat(na,useGPU,q(1:obj%local_nrows, obj%local_ncols+1:2*obj%local_ncols),q_dev,lda,matrixCols,nblk,my_prow,my_pcol,np_rows,np_cols,'R',1)
! #endif
              ! We can now deallocate the stored householder vectors
       deallocate(hh_trans, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *, "solve_evp_&
         &MATH_DATATYPE&
         &_2stage_&
         &PRECISION " // ": error when deallocating hh_trans "//errorMessage
         stop 1
       endif
     endif
     if (isSkewsymmetric) then
       ! first deal with the situation that first backward step was on GPU
       if(do_useGPU_trans_ev_tridi_to_band) then
          ! if the second backward step is to be performed, but not on GPU, we have
          ! to transfer q to the host
          if(do_trans_to_full .and. (.not. do_useGPU_trans_ev_band_to_full)) then
            successCUDA = cuda_memcpy(loc(q(1,obj%local_ncols+1)), q_dev, ldq*matrixCols* size_of_datatype, cudaMemcpyDeviceToHost)
            if (.not.(successCUDA)) then
              print *,"elpa2_template, error in copy to host"
              stop 1
            endif
          endif

         ! if the last step is not required at all, or will be performed on CPU,
         ! release the memmory allocated on the device
         if((.not. do_trans_to_full) .or. (.not. do_useGPU_trans_ev_band_to_full)) then
           successCUDA = cuda_free(q_dev)
         endif
       endif
     endif
     
     if (do_trans_to_full) then
       call obj%timer%start("trans_ev_to_full")  
       if (isSkewsymmetric) then
         if ( (do_useGPU_trans_ev_band_to_full) .and. .not.(do_useGPU_trans_ev_tridi_to_band) ) then
           ! copy to device if we want to continue on GPU
           successCUDA = cuda_malloc(q_dev, ldq*matrixCols*size_of_datatype)
!            if (.not.(successCUDA)) then
!              print *,"elpa2_template, error in cuda_malloc"
!              stop 1
!            endif
           successCUDA = cuda_memcpy(q_dev, loc(q(1,obj%local_ncols+1)), ldq*matrixCols* size_of_datatype, cudaMemcpyHostToDevice)
           if (.not.(successCUDA)) then
             print *,"elpa2_template, error in copy to device"
             stop 1
           endif
         endif
! #ifdef DOUBLE_PRECISION_REAL
!          call prmat(na,useGPU,q(1:obj%local_nrows, obj%local_ncols+1:2*obj%local_ncols),q_dev,lda,matrixCols,nblk,my_prow,my_pcol,np_rows,np_cols,'I',0)
! #endif
         ! Transform imaginary part
         ! Transformation of real and imaginary part could also be one call of trans_ev_band_to_full_ acting on the n x 2n matrix.

         call trans_ev_band_to_full_&
         &MATH_DATATYPE&
         &_&
         &PRECISION &
         (obj, na, nev, nblk, nbw, a, &
         a_dev, lda, tmat, tmat_dev,  q(1:obj%local_nrows, obj%local_ncols+1:2*obj%local_ncols),  &
         q_dev, &
         ldq, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, do_useGPU_trans_ev_band_to_full &
#if REALCASE == 1
         , useQRActual  &
#endif
         )
!          print * , "After trans_ev_band_to_full: imaginary part of q="
!          do i=1,na
!            write(*,"(100g15.5)") ( q(i,j+na), j=1,na )
!          enddo
! #ifdef DOUBLE_PRECISION_REAL
!          call prmat(na,useGPU,q(1:obj%local_nrows, obj%local_ncols+1:2*obj%local_ncols),q_dev,lda,matrixCols,nblk,my_prow,my_pcol,np_rows,np_cols,'I',1)
! #endif
       endif

       deallocate(tmat, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"solve_evp_&
         &MATH_DATATYPE&
         &_2stage_&
         &PRECISION " // ": error when deallocating tmat"//errorMessage
         stop 1
       endif
#ifdef HAVE_LIKWID
       call likwid_markerStopRegion("trans_ev_to_full")
#endif
       call obj%timer%stop("trans_ev_to_full")
     endif ! do_trans_to_full

     if(do_bandred .or. do_trans_to_full) then
       if (do_useGPU_bandred .or. do_useGPU_trans_ev_band_to_full) then
         successCUDA = cuda_free(tmat_dev)
         if (.not.(successCUDA)) then
           print *,"elpa2_template: error in cudaFree, tmat_dev"
           stop 1
         endif
       endif
     endif

     if(do_useGPU .and. (a_dev .ne. 0)) then
       successCUDA = cuda_free(a_dev)
       if (.not.(successCUDA)) then
         print *,"elpa2_template: error in cudaFree, a_dev"
         stop 1
       endif
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

     ! restore original OpenMP settings
#ifdef WITH_OPENMP
    ! store the number of OpenMP threads used in the calling function
    ! restore this at the end of ELPA 2
    call omp_set_num_threads(omp_threads_caller)
#endif

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
