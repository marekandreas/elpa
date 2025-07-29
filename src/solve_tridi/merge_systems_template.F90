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
#include "../general/error_checking.inc"

#ifdef SOLVE_TRIDI_GPU_BUILD
    subroutine merge_systems_gpu_&
    &PRECISION &
                         (obj, na, nm, d, e, q_dev, &
                          matrixRows, nqoff, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols_self, &
                          l_col, p_col, l_col_out, p_col_out, npc_0, npc_n, useGPU, wantDebug, success, max_threads)
#else
    subroutine merge_systems_cpu_&
    &PRECISION &
                         (obj, na, nm, d, e, q, &
                          matrixRows, nqoff, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols_self, &
                          l_col, p_col, l_col_out, p_col_out, npc_0, npc_n, useGPU, wantDebug, success, max_threads)
#endif


      use elpa_gpu
      use, intrinsic :: iso_c_binding
      use precision
      use elpa_abstract_impl
      use elpa_blas_interfaces
      use global_product
      use global_gather
      use resort_ev
      use transform_columns
      use check_monotony
      use add_tmp
      use v_add_s
      use ELPA_utilities
      use elpa_mpi
      use solve_secular_equation
      use elpa_ccl_gpu
      use merge_systems_gpu_new
#if defined(WITH_NVIDIA_GPU_VERSION) && defined(WITH_NVTX)
      use cuda_functions ! for NVTX labels
#elif defined(WITH_AMD_GPU_VERSION) && defined(WITH_ROCTX)
      use hip_functions  ! for ROCTX labels
#endif

#ifdef WITH_OPENMP_TRADITIONAL
      use omp_lib
#endif
      implicit none
#include "../general/precision_kinds.F90"
      class(elpa_abstract_impl_t), intent(inout)  :: obj
      integer(kind=ik), intent(in)                :: na ! this is rather na_local, not global na
      integer(kind=ik), intent(in)                :: nm, matrixRows, nqoff, nblk, matrixCols, mpi_comm_rows, &
                                                     mpi_comm_cols_self, npc_0, npc_n
      integer(kind=ik), intent(in)                :: l_col(na), p_col(na), l_col_out(na), p_col_out(na)

#ifdef SOLVE_TRIDI_GPU_BUILD
      integer(kind=c_intptr_t)                    :: d_dev, e_dev ! shifted; for e_dev only one element is used
      integer(kind=c_intptr_t)                    :: q_dev
#else
      integer(kind=c_intptr_t)                    :: d_dev, e_dev ! dummy variables
      integer(kind=c_intptr_t)                    :: q_dev
#endif

      real(kind=REAL_DATATYPE), intent(inout)     :: d(na), e
#if defined(USE_ASSUMED_SIZE) && !defined(SOLVE_TRIDI_GPU_BUILD)
      real(kind=REAL_DATATYPE)                    :: q(matrixRows,*)
#else
      real(kind=REAL_DATATYPE)                    :: q(matrixRows,matrixCols)
#endif
      logical, intent(in)                         :: useGPU, wantDebug
      integer(kind=c_int)                         :: SM_count

      logical, intent(out)                        :: success

      ! TODO: play with max_strip. If it was larger, matrices being multiplied
      ! might be larger as well! 
      ! Peter: two out of three GEMM dimensions are already "big" and filling the GPU computation rate. 
      ! Increasing max_strip doesn't improve the performance.
      integer(kind=ik)                            :: max_strip

      
      real(kind=REAL_DATATYPE)                    :: beta, sig, s, c, t, tau, rho, eps, tol, &
                                                     qtrans(2,2), dmax, zmax, d1new, d2new
      real(kind=REAL_DATATYPE)                    :: z(na), d1(na), d2(na), z1(na), delta(na),  &
                                                     dbase(na), ddiff(na), ev_scale(na), tmp(na)
      real(kind=REAL_DATATYPE)                    :: d1u(na), zu(na), d1l(na), zl(na)
      real(kind=REAL_DATATYPE), allocatable       :: qtmp1(:,:), qtmp2(:,:), ev(:,:)
#ifdef WITH_OPENMP_TRADITIONAL
      real(kind=REAL_DATATYPE), allocatable       :: z_p(:,:)
      integer(kind=ik)                            :: my_thread
#endif

      integer(kind=ik)                            :: i, j, k, na1, na2, l_rows, l_cols, l_rqs, l_rqe, &
                                                     l_rqm, ns, lc1, lc2, info
      integer(kind=BLAS_KIND)                     :: infoBLAS
      integer(kind=ik)                            :: sig_int
      integer(kind=ik)                            :: l_rnm, nnzu, nnzl, ndef, ncnt, max_local_cols, &
                                                     l_cols_qreorg, np, l_idx, nqcols1 !, nqcols2
      integer(kind=ik)                            :: nnzu_save, nnzl_save
      integer(kind=ik)                            :: my_proc, n_procs, my_prow, my_pcol, np_rows, &
                                                     np_cols
      integer(kind=MPI_KIND)                      :: mpierr
      integer(kind=MPI_KIND)                      :: my_prowMPI, np_rowsMPI, my_pcolMPI, np_colsMPI
      integer(kind=ik)                            :: np_next, np_prev, np_rem
      integer(kind=ik)                            :: idx(na), idx1(na), idx2(na)
      integer(kind=BLAS_KIND)                     :: idxBLAS(NA)
      integer(kind=ik)                            :: coltyp(na), idxq1(na) !, idxq2(na)

      integer(kind=ik)                            :: istat, debug
      character(200)                              :: errorMessage
      integer(kind=ik)                            :: gemm_dim_k, gemm_dim_l, gemm_dim_m

      integer(kind=c_intptr_t)                    :: num
      integer(kind=C_intptr_t)                    :: qtmp1_dev, qtmp1_tmp_dev, qtmp2_dev, ev_dev
      integer(kind=c_intptr_t)                    :: z1_dev, delta_dev, rho_dev
      integer(kind=c_intptr_t)                    :: d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev
      integer(kind=c_intptr_t)                    :: d1l_dev, zl_dev, z_dev, d1_dev, ztmp_extended_dev
      integer(kind=c_intptr_t)                    :: idx1_dev, p_col_dev, coltyp_dev, p_col_out_dev, ndef_c_dev
      integer(kind=c_intptr_t)                    :: idxq1_dev, l_col_out_dev, idx_dev, idx2_dev, l_col_dev
      integer(kind=c_intptr_t)                    :: nnzul_dev
      integer(kind=c_intptr_t)                    :: tmp_dev, zero_dev, one_dev, qtrans_dev ! for transform_columns_gpu

      integer(kind=c_intptr_t)                    :: nnzu_val_dev, nnzl_val_dev
      logical                                     :: successGPU
      integer(kind=c_intptr_t), parameter         :: size_of_datatype = size_of_&
                                                                      &PRECISION&
                                                                      &_real
      integer(kind=c_intptr_t)                    :: gpuHandle
      integer(kind=ik), intent(in)                :: max_threads
      integer(kind=c_intptr_t)                    :: my_stream
      integer(kind=ik)                            :: l_col_out_tmp
      integer(kind=ik), allocatable               :: nnzu_val(:,:), nnzl_val(:,:)
      integer(kind=ik)                            :: nnzul(2)

      integer(kind=ik)                            :: nnzu_start, nnzl_start

      integer(kind=ik), allocatable               :: ndef_c(:)

      integer(kind=ik) :: ii,jj, indx, ind_ex, ind_ex2, p_col_tmp, index2, counter1, counter2

      logical                                     :: useCCL
      integer(kind=c_intptr_t)                    :: ccl_comm_rows, ccl_comm_cols
      integer(kind=c_int)                         :: cclDataType

      call obj%timer%start("merge_systems" // PRECISION_SUFFIX)
      success = .true.

      call obj%timer%start("mpi_communication")
      call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND) ,my_prowMPI, mpierr)
      call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND) ,np_rowsMPI, mpierr)
      call mpi_comm_rank(int(mpi_comm_cols_self,kind=MPI_KIND) ,my_pcolMPI, mpierr)
      call mpi_comm_size(int(mpi_comm_cols_self,kind=MPI_KIND) ,np_colsMPI, mpierr)

      my_prow = int(my_prowMPI,kind=c_int)
      np_rows = int(np_rowsMPI,kind=c_int)
      my_pcol = int(my_pcolMPI,kind=c_int)
      np_cols = int(np_colsMPI,kind=c_int)

      call obj%timer%stop("mpi_communication")

      if (wantDebug) then
        debug = 1
      else
        debug = 0
      endif

      if (useGPU) then
        max_strip=128
      else
        max_strip=128
      endif
      if (wantDebug .and. my_prow==0 .and. my_pcol==0) print *, "max_strip = ", max_strip

      useCCL = obj%gpu_setup%useCCL

      if (useGPU) then
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
#endif
        SM_count = obj%gpu_setup%gpuSMcount
        
        if (useCCL) then
          my_stream = obj%gpu_setup%my_stream
          ccl_comm_rows = obj%gpu_setup%ccl_comm_rows
          ccl_comm_cols = obj%gpu_setup%ccl_comm_cols
#if defined(DOUBLE_PRECISION)
          cclDataType = cclDouble
#endif      
#if defined(SINGLE_PRECISION)
          cclDataType = cclFloat
#endif
        endif ! useCCL
      endif ! useGPU

! #ifdef WITH_OPENMP_TRADITIONAL
!       allocate(z_p(na,0:max_threads-1), stat=istat, errmsg=errorMessage)
!       check_allocate("merge_systems: z_p",istat, errorMessage)
! #endif

      ! If my processor column isn't in the requested set, do nothing
      if (my_pcol<npc_0 .or. my_pcol>=npc_0+npc_n) then
        call obj%timer%stop("merge_systems" // PRECISION_SUFFIX)
        return
      endif

      ! Determine number of "next" and "prev" column for ring sends
      if (my_pcol == npc_0+npc_n-1) then
        np_next = npc_0
      else
        np_next = my_pcol + 1
      endif

      if (my_pcol == npc_0) then
        np_prev = npc_0+npc_n-1
      else
        np_prev = my_pcol - 1
      endif

      call check_monotony_&
      &PRECISION&
      &(obj, nm,d,'Input1',wantDebug, success)
      if (.not.(success)) then
        call obj%timer%stop("merge_systems" // PRECISION_SUFFIX)
        return
      endif

      call check_monotony_&
      &PRECISION&
      &(obj,na-nm,d(nm+1),'Input2',wantDebug, success)
      if (.not.(success)) then
        call obj%timer%stop("merge_systems" // PRECISION_SUFFIX)
        return
      endif
      ! Get global number of processors and my processor number.
      ! Please note that my_proc does not need to match any real processor number,
      ! it is just used for load balancing some loops.

      n_procs = np_rows*npc_n
      my_proc = my_prow*npc_n + (my_pcol-npc_0) ! Row major


      ! Local limits of the rows of Q

      l_rqs = local_index(nqoff+1 , my_prow, np_rows, nblk, +1) ! First row of Q
      l_rqm = local_index(nqoff+nm, my_prow, np_rows, nblk, -1) ! Last row <= nm
      l_rqe = local_index(nqoff+na, my_prow, np_rows, nblk, -1) ! Last row of Q

      l_rnm  = l_rqm-l_rqs+1 ! Number of local rows <= nm
      l_rows = l_rqe-l_rqs+1 ! Total number of local rows


      ! My number of local columns

      l_cols = COUNT(p_col(1:na)==my_pcol)

      ! Get max number of local columns

      max_local_cols = 0
      do np = npc_0, npc_0+npc_n-1
        max_local_cols = MAX(max_local_cols,COUNT(p_col(1:na)==np))
      enddo


      if (useGPU) then
        num = na * size_of_int   
        successGPU = gpu_malloc(ndef_c_dev, num)
        check_alloc_gpu("merge_systems: ndef_c_dev", successGPU)

        num = na * size_of_int     
        successGPU = gpu_malloc(idx1_dev, num)
        check_alloc_gpu("merge_systems: idx1_dev", successGPU)

        num = na * size_of_int     
        successGPU = gpu_malloc(p_col_dev, num)
        check_alloc_gpu("merge_systems: p_col_dev", successGPU)

        num = na * size_of_int     
        successGPU = gpu_malloc(p_col_out_dev, num)
        check_alloc_gpu("merge_systems: p_col_out_dev", successGPU)

        num = na * size_of_int     
        successGPU = gpu_malloc(coltyp_dev, num)
        check_alloc_gpu("merge_systems: coltyp_dev", successGPU)

        num = na * size_of_int     
        successGPU = gpu_malloc(idx2_dev, num)
        check_alloc_gpu("merge_systems: idx2_dev", successGPU)
        
        num = na * size_of_int
        successGPU = gpu_malloc(l_col_dev, num)
        check_alloc_gpu("merge_systems: l_col_dev", successGPU)

        num = na * size_of_int     
        successGPU = gpu_malloc(l_col_out_dev, num)
        check_alloc_gpu("merge_systems: l_col_out_dev", successGPU)

        num = (na) * size_of_datatype
        successGPU = gpu_malloc(z_dev, num)
        check_alloc_gpu("merge_systems: z_dev", successGPU)

        num = (na) * size_of_datatype
        successGPU = gpu_malloc(z1_dev, num)
        check_alloc_gpu("merge_systems: z1_dev", successGPU)

        num = (na) * size_of_datatype
        successGPU = gpu_malloc(d1_dev, num)
        check_alloc_gpu("merge_systems: d1_dev", successGPU)

        num = 1 * size_of_datatype
        successGPU = gpu_malloc(rho_dev, num)
        check_alloc_gpu("merge_systems: rho_dev", successGPU)

        num = (na) * size_of_datatype
        successGPU = gpu_malloc(d1u_dev, num)
        check_alloc_gpu("merge_systems: d1u_dev", successGPU)

        num = (na) * size_of_datatype
        successGPU = gpu_malloc(dbase_dev, num)
        check_alloc_gpu("merge_systems: dbase_dev", successGPU)

        num = (na) * size_of_datatype
        successGPU = gpu_malloc(ddiff_dev, num)
        check_alloc_gpu("merge_systems: ddiff_dev", successGPU)

        num = (na) * size_of_datatype
        successGPU = gpu_malloc(zu_dev, num)
        check_alloc_gpu("merge_systems: zu_dev", successGPU)

        num = (na) * size_of_datatype
        successGPU = gpu_malloc(ev_scale_dev, num)
        check_alloc_gpu("merge_systems: ev_scale_dev", successGPU)

        num = (na) * size_of_datatype
        successGPU = gpu_malloc(d1l_dev, num)
        check_alloc_gpu("merge_systems: d1l_dev", successGPU)

        num = (na) * size_of_datatype
        successGPU = gpu_malloc(zl_dev, num)
        check_alloc_gpu("merge_systems: zl_dev", successGPU)

        num = (l_rows) * size_of_datatype
        successGPU = gpu_malloc(tmp_dev, num)
        check_alloc_gpu("merge_systems: tmp_dev", successGPU)

        num = 1 * size_of_datatype
        successGPU = gpu_malloc(zero_dev, num)
        check_alloc_gpu("merge_systems: zero_dev", successGPU)

        num = 1 * size_of_datatype
        successGPU = gpu_malloc(one_dev, num)
        check_alloc_gpu("merge_systems: one_dev", successGPU)

        num = 4 * size_of_datatype
        successGPU = gpu_malloc(qtrans_dev, num)
        check_alloc_gpu("merge_systems: qtrans_dev", successGPU)

        num = na * size_of_int
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(p_col_dev, int(loc(p_col(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
#else
        successGPU = gpu_memcpy(p_col_dev, int(loc(p_col(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
#endif
        check_memcpy_gpu("merge_systems: p_col_dev", successGPU)

        num = na * size_of_int
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(l_col_dev, int(loc(l_col(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
#else
        successGPU = gpu_memcpy(l_col_dev, int(loc(l_col(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
#endif
        check_memcpy_gpu("merge_systems: l_col_dev", successGPU)

        num = 1 * size_of_datatype
        beta = 0.0_rk
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(zero_dev, int(loc(beta),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
#else
        successGPU = gpu_memcpy(zero_dev, int(loc(beta),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
#endif
        check_memcpy_gpu("merge_systems: zero_dev", successGPU)

        num = 1 * size_of_datatype
        beta = 1.0_rk
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(one_dev, int(loc(beta),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
#else
        successGPU = gpu_memcpy(one_dev, int(loc(beta),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
#endif
        check_memcpy_gpu("merge_systems: one_dev", successGPU)

      endif ! useGPU

      ! Calculations start here

      beta = abs(e)
      sig  = sign(1.0_rk,e)

      ! Calculate rank-1 modifier z
      if (useGPU) then
        num = na * size_of_datatype
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memset_async(z_dev, 0, num, my_stream)
#else
        successGPU = gpu_memset      (z_dev, 0, num)
#endif
        check_memcpy_gpu("merge_systems: memset z_dev", successGPU)      
      else
        z(:) = 0
      endif


      if (MOD((nqoff+nm-1)/nblk,np_rows)==my_prow) then
        ! nm is local on my row
        if (useGPU) then
          sig_int = 1
          NVTX_RANGE_PUSH("gpu_fill_z_kernel")
          call gpu_fill_z(PRECISION_CHAR, z_dev, q_dev, p_col_dev, l_col_dev, &
                          sig_int, na, my_pcol, l_rqm, matrixRows, SM_count, debug, my_stream)
          NVTX_RANGE_POP("gpu_fill_z_kernel")
        else
          do i = 1, na
            if (p_col(i)==my_pcol) z(i) = q(l_rqm,l_col(i))
          enddo
        endif
      endif

      if (MOD((nqoff+nm)/nblk,np_rows)==my_prow) then
        ! nm+1 is local on my row

        if (useGPU) then
          if (sig>0) then
            sig_int = 1
          else
            sig_int = -1
          endif

          NVTX_RANGE_PUSH("gpu_fill_z_kernel")
          call gpu_fill_z(PRECISION_CHAR, z_dev, q_dev, p_col_dev, l_col_dev, &
                          sig_int, na, my_pcol, l_rqm+1, matrixRows, SM_count, debug, my_stream)
          NVTX_RANGE_POP("gpu_fill_z_kernel")
        else
          do i = 1, na
            if (p_col(i)==my_pcol) z(i) = z(i) + sig*q(l_rqm+1,l_col(i))
          enddo
        endif
      endif

      if (useGPU) then
        num = na * size_of_datatype
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(int(loc(z(1)),kind=c_intptr_t), z_dev, num, gpuMemcpyDeviceToHost, my_stream)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("merge_systems: z_dev -> z", successGPU)
#else
        successGPU = gpu_memcpy      (int(loc(z(1)),kind=c_intptr_t), z_dev, num, gpuMemcpyDeviceToHost)
#endif
        check_memcpy_gpu("merge_systems: z_dev", successGPU)
      endif

      
      call global_gather_&
      &PRECISION&
      &(obj, z, na, mpi_comm_rows, mpi_comm_cols_self, npc_n, np_prev, np_next, success)
      if (.not.(success)) then
        write(error_unit,*) "Error in global_gather. Aborting"
        success = .false.
        return
      endif

      ! Normalize z so that norm(z) = 1.  Since z is the concatenation of
      ! two normalized vectors, norm2(z) = sqrt(2).
      z = z/sqrt(2.0_rk)
      rho = 2.0_rk*beta
      ! Calculate index for merging both systems by ascending eigenvalues
      call obj%timer%start("lapack_lamrg")
      NVTX_RANGE_PUSH("lapack_lamrg_1")
      call PRECISION_LAMRG( int(nm,kind=BLAS_KIND), int(na-nm,kind=BLAS_KIND), d, &
                            1_BLAS_KIND, 1_BLAS_KIND, idxBLAS )
      NVTX_RANGE_POP("lapack_lamrg_1")
      idx(:) = int(idxBLAS(:),kind=ik)
      call obj%timer%stop("lapack_lamrg")

      ! Calculate the allowable deflation tolerance

      zmax = maxval(abs(z))
      dmax = maxval(abs(d))
      EPS = PRECISION_LAMCH( 'E' ) ! return epsilon
      TOL = 8.0_rk*EPS*MAX(dmax,zmax)

      ! If the rank-1 modifier is small enough, no more needs to be done
      ! except to reorganize D and Q

      IF ( RHO*zmax <= TOL ) THEN
        ! Rearrange eigenvalues
        tmp = d
        do i=1,na
          d(i) = tmp(idx(i))
        enddo
        
        ! Rearrange eigenvectors
        if (useGPU) then
          NVTX_RANGE_PUSH("resort_ev_gpu")
          call obj%timer%start("resort_ev_gpu")
          call resort_ev_gpu_&
                              &PRECISION&
                              (obj, idx, na, na, p_col_out, q_dev, matrixRows, matrixCols, l_rows, l_rqe, &
                              l_rqs, mpi_comm_cols_self, p_col, l_col, l_col_out)
          call obj%timer%stop("resort_ev_gpu")
          NVTX_RANGE_POP("resort_ev_gpu")
        else ! useGPU
          call resort_ev_cpu_&
                             &PRECISION&
                             (obj, idx, na, na, p_col_out, q    , matrixRows, matrixCols, l_rows, l_rqe, &
                              l_rqs, mpi_comm_cols_self, p_col, l_col, l_col_out)
        endif ! useGPU

        call obj%timer%stop("merge_systems" // PRECISION_SUFFIX)

        if (wantDebug) write(error_unit,*) "Returing early from merge_systems (RHO*zmax <= TOL): matrix is block-diagonal"
        ! tested by validate_real_double_solve_tridiagonal_1stage_blocktridi

        return
      ENDIF

      ! Merge and deflate system

      na1 = 0
      na2 = 0

      ! COLTYP:
      ! 1 : non-zero in the upper half only;
      ! 2 : dense;
      ! 3 : non-zero in the lower half only;
      ! 4 : deflated.

      coltyp(1:nm) = 1
      coltyp(nm+1:na) = 3

      NVTX_RANGE_PUSH("deflation_loop")
      do i=1,na

        if (rho*abs(z(idx(i))) <= tol) then

          ! Deflate due to small z component.

          na2 = na2+1
          d2(na2)   = d(idx(i))
          idx2(na2) = idx(i)
          coltyp(idx(i)) = 4

        else if (na1>0) then

          ! Check if eigenvalues are close enough to allow deflation.

          S = Z(idx(i))
          C = Z1(na1)

          ! Find TAU = sqrt(a**2+b**2) without overflow or
          ! destructive underflow.
          TAU = PRECISION_LAPY2( C, S )
          T = D1(na1) - D(idx(i))
          C = C / TAU
          S = -S / TAU
          IF ( ABS( T*C*S ) <= TOL ) THEN

            ! Deflation is possible.

            na2 = na2+1

            Z1(na1) = TAU

            d2new = D(idx(i))*C**2 + D1(na1)*S**2
            d1new = D(idx(i))*S**2 + D1(na1)*C**2

            ! D(idx(i)) >= D1(na1) and C**2 + S**2 == 1.0
            ! This means that after the above transformation it must be
            !    D1(na1) <= d1new <= D(idx(i))
            !    D1(na1) <= d2new <= D(idx(i))
            !
            ! D1(na1) may get bigger but it is still smaller than the next D(idx(i+1))
            ! so there is no problem with sorting here.
            ! d2new <= D(idx(i)) which means that it might be smaller than D2(na2-1)
            ! which makes a check (and possibly a resort) necessary.
            !
            ! The above relations may not hold exactly due to numeric differences
            ! so they have to be enforced in order not to get troubles with sorting.


            if (d1new<D1(na1)  ) d1new = D1(na1)
            if (d1new>D(idx(i))) d1new = D(idx(i))

            if (d2new<D1(na1)  ) d2new = D1(na1)
            if (d2new>D(idx(i))) d2new = D(idx(i))

            D1(na1) = d1new

            do j=na2-1,1,-1
              if (d2new<d2(j)) then
                d2(j+1)   = d2(j)
                idx2(j+1) = idx2(j)
              else
                exit ! Loop
              endif
            enddo

            d2(j+1)   = d2new
            idx2(j+1) = idx(i)

            qtrans(1,1) = C; qtrans(1,2) =-S
            qtrans(2,1) = S; qtrans(2,2) = C

            NVTX_RANGE_PUSH("transform_columns")
            if (useGPU) then
#ifdef WITH_GPU_STREAMS
              successGPU = gpu_memcpy_async(qtrans_dev, int(loc(qtrans(1,1)),kind=c_intptr_t), &
                                            4*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
              if (wantDebug) successGPU = gpu_DeviceSynchronize()
#else
              successGPU = gpu_memcpy(qtrans_dev, int(loc(qtrans(1,1)),kind=c_intptr_t), &
                                      4*size_of_datatype, gpuMemcpyHostToDevice)
#endif
              check_memcpy_gpu("transform_columns: q_dev", successGPU)

              call transform_columns_gpu_&
                                         &PRECISION &
                                        (obj, idx(i), idx1(na1), na, tmp, l_rqs, l_rqe, &
                                          q_dev, matrixRows, matrixCols, l_rows, mpi_comm_cols_self, &
                                          p_col, l_col, qtrans_dev, &
                                          tmp_dev, zero_dev, one_dev, debug, my_stream)
            else
              call transform_columns_cpu_&
                                        &PRECISION &
                                        (obj, idx(i), idx1(na1), na, tmp, l_rqs, l_rqe, &
                                          q    , matrixRows, matrixCols, l_rows, mpi_comm_cols_self, &
                                          p_col, l_col, qtrans)
            endif
            NVTX_RANGE_POP("transform_columns")

            if (coltyp(idx(i))==1 .and. coltyp(idx1(na1))/=1) coltyp(idx1(na1)) = 2
            if (coltyp(idx(i))==3 .and. coltyp(idx1(na1))/=3) coltyp(idx1(na1)) = 2

            coltyp(idx(i)) = 4

          else
            na1 = na1+1
            d1(na1) = d(idx(i))
            z1(na1) = z(idx(i))
            idx1(na1) = idx(i)
          endif
        else
          na1 = na1+1
          d1(na1) = d(idx(i))
          z1(na1) = z(idx(i))
          idx1(na1) = idx(i)
        endif

      enddo ! do i=1,na
      NVTX_RANGE_POP("deflation_loop")

      call check_monotony_&
      &PRECISION&
      &(obj, na1,d1,'Sorted1', wantDebug, success)
      if (.not.(success)) then
        call obj%timer%stop("merge_systems" // PRECISION_SUFFIX)
        return
      endif
      call check_monotony_&
      &PRECISION&
      &(obj, na2,d2,'Sorted2', wantDebug, success)
      if (.not.(success)) then
        call obj%timer%stop("merge_systems" // PRECISION_SUFFIX)
        return
      endif

      if (na1==1 .or. na1==2) then
        ! if(my_proc==0) print *,'--- Remark solve_tridi: na1==',na1,' proc==',myid

        if (na1==1) then
          d(1) = d1(1) + rho*z1(1)**2 ! solve secular equation
        else ! na1==2
          call obj%timer%start("lapack_laed5_x2")
          NVTX_RANGE_PUSH("lapack_laed5_x2")
          call PRECISION_LAED5(1_BLAS_KIND, d1, z1, qtrans(1,1), rho, d(1))
          call PRECISION_LAED5(2_BLAS_KIND, d1, z1, qtrans(1,2), rho, d(2))
          NVTX_RANGE_POP("lapack_laed5_x2")
          call obj%timer%stop("lapack_laed5_x2")

          if (useGPU) then
#ifdef WITH_GPU_STREAMS
            successGPU = gpu_memcpy_async(qtrans_dev, int(loc(qtrans(1,1)),kind=c_intptr_t), &
                                          4*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
            if (wantDebug) successGPU = gpu_DeviceSynchronize()
#else
            successGPU = gpu_memcpy(qtrans_dev, int(loc(qtrans(1,1)),kind=c_intptr_t), &
                                    4*size_of_datatype, gpuMemcpyHostToDevice)
#endif
            check_memcpy_gpu("transform_columns: q_dev", successGPU)

            call transform_columns_gpu_&
                                        &PRECISION &
                                      (obj, idx1(1), idx1(2), na, tmp, l_rqs, l_rqe, &
                                        q_dev, matrixRows, matrixCols, l_rows, mpi_comm_cols_self, &
                                        p_col, l_col, qtrans_dev, &
                                        tmp_dev, zero_dev, one_dev, debug, my_stream)
          else
            call transform_columns_cpu_&
                                       &PRECISION&
                                       & (obj, idx1(1), idx1(2), na, tmp, l_rqs, l_rqe, q, &
                                          matrixRows, matrixCols, l_rows, mpi_comm_cols_self, &
                                          p_col, l_col, qtrans)
          endif
        endif ! na1==2

        ! Add the deflated eigenvalues
        d(na1+1:na) = d2(1:na2)

        ! Calculate arrangement of all eigenvalues  in output
        call obj%timer%start("lapack_lamrg")
        NVTX_RANGE_PUSH("lapack_lamrg_2")
        call PRECISION_LAMRG( int(na1,kind=BLAS_KIND), int(na-na1,kind=BLAS_KIND), d, &
                              1_BLAS_KIND, 1_BLAS_KIND, idxBLAS )
        NVTX_RANGE_POP("lapack_lamrg_2")
        idx(:) = int(idxBLAS(:),kind=ik)
        call obj%timer%stop("lapack_lamrg")
        ! Rearrange eigenvalues

        tmp = d
        do i=1,na
          d(i) = tmp(idx(i))
        enddo

        ! Rearrange eigenvectors

        do i=1,na
          if (idx(i)<=na1) then
            idxq1(i) = idx1(idx(i))
          else
            idxq1(i) = idx2(idx(i)-na1)
          endif
        enddo

        if (useGPU) then
          call resort_ev_gpu_&
                         &PRECISION&
                         &(obj, idxq1, na, na, p_col_out, q_dev, matrixRows, matrixCols, l_rows, l_rqe, &
                           l_rqs, mpi_comm_cols_self, p_col, l_col, l_col_out)
        else
          call resort_ev_cpu_&
                         &PRECISION&
                         &(obj, idxq1, na, na, p_col_out, q    , matrixRows, matrixCols, l_rows, l_rqe, &
                           l_rqs, mpi_comm_cols_self, p_col, l_col, l_col_out)
        endif

        write(error_unit,*) "Returing early from merge_systems (na1==1 .or. na1==2)"
        ! na=1 can be tested with "mpirun -n 4 ./validate_real_double_solve_tridiagonal_1stage_gpu_blocktridi 3 3 1"
        ! na=2 can be tested with "mpirun -n 4 ./validate_real_double_solve_tridiagonal_1stage_gpu_toeplitz 4 4 2"
      else if (na1>2) then

        ! Solve secular equation
       
        if (useGPU) then
          num = (na1*SM_count) * size_of_datatype
          successGPU = gpu_malloc(ztmp_extended_dev, num)
          check_alloc_gpu("merge_systems: ztmp_extended_dev", successGPU)

          call gpu_fill_array(PRECISION_CHAR, ztmp_extended_dev, one_dev, na1*SM_count, SM_count, debug, my_stream)
          
          call gpu_fill_array(PRECISION_CHAR, z_dev, one_dev, na1, SM_count, debug, my_stream)
          

          num = na1 * size_of_datatype
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memset_async(dbase_dev, 0, num, my_stream)
#else
          successGPU = gpu_memset      (dbase_dev, 0, num)
#endif
          check_memcpy_gpu("merge_systems: memset dbase_dev", successGPU)

          num = na1 * size_of_datatype
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memset_async(ddiff_dev, 0, num, my_stream)
#else
          successGPU = gpu_memset      (ddiff_dev, 0, num)
#endif
          check_memcpy_gpu("merge_systems: memset ddiff_dev", successGPU)
        else
          z(1:na1) = 1
! #ifdef WITH_OPENMP_TRADITIONAL
!           z_p(1:na1,:) = 1
! #endif
          dbase(1:na1) = 0
          ddiff(1:na1) = 0
        endif


        NVTX_RANGE_PUSH("lapack_laed4_loop")
        if (useGPU) then
          ! data transfer to GPU
#ifdef WITH_GPU_STREAMS
          num = na * size_of_datatype
          successGPU = gpu_memcpy_async(d1_dev, int(loc(d1(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: d1_dev", successGPU)

          successGPU = gpu_memcpy_async(z1_dev, int(loc(z1(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: z1_dev", successGPU)

          num = 1 * size_of_datatype
          successGPU = gpu_memcpy_async(rho_dev, int(loc(rho),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: rho_dev", successGPU)
#else
          num = na * size_of_datatype
          successGPU = gpu_memcpy(d1_dev, int(loc(d1(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: d1_dev", successGPU)

          successGPU = gpu_memcpy(z1_dev, int(loc(z1(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: z1_dev", successGPU)

          num = 1 * size_of_datatype
          successGPU = gpu_memcpy(rho_dev, int(loc(rho),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: rho_dev", successGPU)
#endif

          ! delta_dev is a temporary buffer, not used afterwards
          num = (na1*SM_count) * size_of_datatype
          successGPU = gpu_malloc(delta_dev, num)
          check_alloc_gpu("merge_systems: delta_dev", successGPU)

          call gpu_solve_secular_equation_loop (PRECISION_CHAR, d1_dev, z1_dev, delta_dev, rho_dev, &
                  ztmp_extended_dev, dbase_dev, ddiff_dev, my_proc, na1, n_procs, SM_count, debug, my_stream)
          
          call gpu_local_product(PRECISION_CHAR, z_dev, ztmp_extended_dev, na1, SM_count, debug, my_stream)
          
          successGPU = gpu_free(delta_dev)
          check_dealloc_gpu("merge_systems: delta_dev", successGPU)

          ! data transfer back to CPU
#ifdef WITH_GPU_STREAMS
          num = na * size_of_datatype
          successGPU = gpu_memcpy_async(int(loc(d1(1)),kind=c_intptr_t), d1_dev, num, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("merge_systems: d1_dev", successGPU)

          successGPU = gpu_memcpy_async(int(loc(z1(1)),kind=c_intptr_t), z1_dev, num, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("merge_systems: z1_dev", successGPU)

          successGPU = gpu_memcpy_async(int(loc(dbase(1)),kind=c_intptr_t), dbase_dev, num, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("merge_systems: dbase_dev", successGPU)

          successGPU = gpu_memcpy_async(int(loc(ddiff(1)),kind=c_intptr_t), ddiff_dev, num, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("merge_systems: ddiff_dev", successGPU)

          successGPU = gpu_memcpy_async(int(loc(z(1)),kind=c_intptr_t), z_dev, num, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("merge_systems: z_dev", successGPU)

          num = 1 * size_of_datatype
          successGPU = gpu_memcpy_async(int(loc(rho),kind=c_intptr_t), rho_dev, num, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("merge_systems: rho_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("solve_tridi_single: rho_dev -> rho", successGPU)
#else
          num = na * size_of_datatype
          successGPU = gpu_memcpy(int(loc(d1(1)),kind=c_intptr_t), d1_dev, num, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("merge_systems: d1_dev", successGPU)

          successGPU = gpu_memcpy(int(loc(z1(1)),kind=c_intptr_t), z1_dev, num, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("merge_systems: z1_dev", successGPU)

          successGPU = gpu_memcpy(int(loc(dbase(1)),kind=c_intptr_t), dbase_dev, num, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("merge_systems: dbase_dev", successGPU)

          successGPU = gpu_memcpy(int(loc(ddiff(1)),kind=c_intptr_t), ddiff_dev, num, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("merge_systems: ddiff_dev", successGPU)

          successGPU = gpu_memcpy(int(loc(z(1)),kind=c_intptr_t), z_dev, num, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("merge_systems: z_dev", successGPU)

          num = 1 * size_of_datatype
          successGPU = gpu_memcpy(int(loc(rho),kind=c_intptr_t), rho_dev, num, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("merge_systems: rho_dev", successGPU)
#endif
        else
!        info = 0
!        infoBLAS = int(info,kind=BLAS_KIND)
!#ifdef WITH_OPENMP_TRADITIONAL
!
!        call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)
!!$OMP PARALLEL PRIVATE(i,my_thread,delta,s,info,infoBLAS,j)
!        my_thread = omp_get_thread_num()
!!$OMP DO
!#endif          
          do i = my_proc+1, na1, n_procs ! work distributed over all processors
            call obj%timer%start("lapack_laed4")
            NVTX_RANGE_PUSH("lapack_laed4")
            call PRECISION_LAED4(int(na1,kind=BLAS_KIND), int(i,kind=BLAS_KIND), d1, z1, delta, &
                                rho, s, infoBLAS) ! s is not used!
            info = int(infoBLAS,kind=ik)
            NVTX_RANGE_POP("lapack_laed4")
            call obj%timer%stop("lapack_laed4")
            if (info/=0) then
              ! If DLAED4 fails (may happen especially for LAPACK versions before 3.2)
              ! use the more stable bisection algorithm in solve_secular_equation
              ! print *,'ERROR DLAED4 n=',na1,'i=',i,' Using Bisection'
              call solve_secular_equation_&
                                &PRECISION&
                                &(obj, na1, i, d1, z1, delta, rho, s) ! s is not used!
            endif

            ! Compute updated z

  !#ifdef WITH_OPENMP_TRADITIONAL
  !          do j=1,na1
  !            if (i/=j)  z_p(j,my_thread) = z_p(j,my_thread)*( delta(j) / (d1(j)-d1(i)) )
  !          enddo
  !          z_p(i,my_thread) = z_p(i,my_thread)*delta(i)
  !#else
            do j=1,na1
              if (i/=j)  z(j) = z(j)*( delta(j) / (d1(j)-d1(i)) )
            enddo
            z(i) = z(i)*delta(i)
  !#endif
            ! Store dbase/ddiff

            if (i<na1) then
              if (abs(delta(i+1)) < abs(delta(i))) then
                dbase(i) = d1(i+1)
                ddiff(i) = delta(i+1)
              else
                dbase(i) = d1(i)
                ddiff(i) = delta(i)
              endif
            else
              dbase(i) = d1(i)
              ddiff(i) = delta(i)
            endif
          enddo ! i = my_proc+1, na1, n_procs
        endif ! useGPU
        NVTX_RANGE_POP("lapack_laed4_loop")
        
!#ifdef WITH_OPENMP_TRADITIONAL
!!$OMP END PARALLEL
!
!        call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)
!
!        do i = 0, max_threads-1
!          z(1:na1) = z(1:na1)*z_p(1:na1,i)
!        enddo
!#endif

        NVTX_RANGE_PUSH("global_product")
        call global_product_&
                  &PRECISION&
                  (obj, z, na1, mpi_comm_rows, mpi_comm_cols_self, npc_0, npc_n, success)
        if (.not.(success)) then
          write(error_unit,*) "Error in global_product. Aborting..."
          return
        endif
        NVTX_RANGE_POP("global_product")
        
        z(1:na1) = SIGN( SQRT( ABS( z(1:na1) ) ), z1(1:na1) )

        NVTX_RANGE_PUSH("global_gather_x2")
        call global_gather_&
        &PRECISION&
        &(obj, dbase, na1, mpi_comm_rows, mpi_comm_cols_self, npc_n, np_prev, np_next, success)
        if (.not.(success)) then
          write(error_unit,*) "Error in global_gather. Aborting..."
          return
        endif
        call global_gather_&
        &PRECISION&
        &(obj, ddiff, na1, mpi_comm_rows, mpi_comm_cols_self, npc_n, np_prev, np_next, success)
        if (.not.(success)) then
          write(error_unit,*) "Error in global_gather. Aborting..."
          return
        endif
        NVTX_RANGE_POP("global_gather_x2")

        d(1:na1) = dbase(1:na1) - ddiff(1:na1)

        ! Calculate scale factors for eigenvectors
        if (useGPU) then
          num = na * size_of_datatype
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memset_async(ev_scale_dev, 0, num, my_stream)
#else
          successGPU = gpu_memset      (ev_scale_dev, 0, num)
#endif
          check_memcpy_gpu("merge_systems: memset ev_scale_dev", successGPU)
        else  ! useGPU
          ev_scale(:) = 0.0_rk
        endif ! useGPU

    
        NVTX_RANGE_PUSH("add_tmp_loop")
        if (wantDebug) call obj%timer%start("add_tmp_loop")

        if (useGPU) then
          ! data transfer to GPU
          num = na * size_of_datatype
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memcpy_async(d1_dev, int(loc(d1(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: d1_dev", successGPU)

          successGPU = gpu_memcpy_async(dbase_dev, int(loc(dbase(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: dbase_dev", successGPU)

          successGPU = gpu_memcpy_async(ddiff_dev, int(loc(ddiff(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: ddiff_dev", successGPU)

          successGPU = gpu_memcpy_async(z_dev, int(loc(z(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: z_dev", successGPU)
#else
          successGPU = gpu_memcpy(d1_dev, int(loc(d1(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: d1_dev", successGPU)

          successGPU = gpu_memcpy(dbase_dev, int(loc(dbase(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: dbase_dev", successGPU)

          successGPU = gpu_memcpy(ddiff_dev, int(loc(ddiff(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: ddiff_dev", successGPU)

          successGPU = gpu_memcpy(z_dev, int(loc(z(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: z_dev", successGPU)
#endif
          call gpu_add_tmp_loop(PRECISION_CHAR, d1_dev, dbase_dev, ddiff_dev, z_dev, ev_scale_dev, ztmp_extended_dev, &
                                na1, my_proc, n_procs, SM_count, debug, my_stream)
          

          successGPU = gpu_free(ztmp_extended_dev)
          check_dealloc_gpu("merge_systems: ztmp_extended_dev", successGPU)

          ! data transfer back to CPU
          num = na * size_of_datatype
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memcpy_async(int(loc(d1(1)),kind=c_intptr_t), d1_dev, num, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("merge_systems: d1_dev", successGPU)

          successGPU = gpu_memcpy_async(int(loc(dbase(1)),kind=c_intptr_t), dbase_dev, num, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("merge_systems: dbase_dev", successGPU)

          successGPU = gpu_memcpy_async(int(loc(ddiff(1)),kind=c_intptr_t), ddiff_dev, num, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("merge_systems: ddiff_dev", successGPU)

          successGPU = gpu_memcpy_async(int(loc(z1(1)),kind=c_intptr_t), z1_dev, num, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("merge_systems: z1_dev", successGPU)

          successGPU = gpu_memcpy_async(int(loc(ev_scale(1)),kind=c_intptr_t), ev_scale_dev, num, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("merge_systems: ev_scale_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("solve_tridi_single: ev_scale_dev -> ev_scale", successGPU)
#else
          successGPU = gpu_memcpy(int(loc(d1(1)),kind=c_intptr_t), d1_dev, num, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("merge_systems: d1_dev", successGPU)

          successGPU = gpu_memcpy(int(loc(dbase(1)),kind=c_intptr_t), dbase_dev, num, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("merge_systems: dbase_dev", successGPU)

          successGPU = gpu_memcpy(int(loc(ddiff(1)),kind=c_intptr_t), ddiff_dev, num, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("merge_systems: ddiff_dev", successGPU)

          successGPU = gpu_memcpy(int(loc(z1(1)),kind=c_intptr_t), z1_dev, num, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("merge_systems: z1_dev", successGPU)

          successGPU = gpu_memcpy(int(loc(ev_scale(1)),kind=c_intptr_t), ev_scale_dev, num, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("merge_systems: ev_scale_dev", successGPU)
#endif
        else
#ifdef WITH_OPENMP_TRADITIONAL
          call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)

!$omp PARALLEL DO &
!$omp default(none) &
!$omp private(i) &
!$omp SHARED(na1, my_proc, n_procs,  &
!$OMP d1, dbase, ddiff, z, ev_scale, obj)
#endif
          do i = my_proc+1, na1, n_procs ! work distributed over all processors

            ! tmp(1:na1) = z(1:na1) / delta(1:na1,i)  ! original code
            ! tmp(1:na1) = z(1:na1) / (d1(1:na1)-d(i))! bad results

            ! All we want to calculate is tmp = (d1(1:na1)-dbase(i))+ddiff(i)
            ! in exactly this order, but we want to prevent compiler optimization
  !         ev_scale_val = ev_scale(i)
            call add_tmp_&
            &PRECISION&
            &(obj, d1, dbase, ddiff, z, ev_scale(i), na1, i)
  !         ev_scale(i) = ev_scale_val
          enddo
#ifdef WITH_OPENMP_TRADITIONAL
!$OMP END PARALLEL DO

          call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)
#endif
        endif ! useGPU

        if (wantDebug) call obj%timer%stop("add_tmp_loop")
        NVTX_RANGE_POP("add_tmp_loop")

        
        NVTX_RANGE_PUSH("global_gather")
        call global_gather_&
                  &PRECISION&
                  &(obj, ev_scale, na1, mpi_comm_rows, mpi_comm_cols_self, npc_n, np_prev, np_next, success)
        if (.not.(success)) then
          write(error_unit,*) "Error in global_gather. Aborting..."
          return
        endif
        NVTX_RANGE_POP("global_gather")

        ! Add the deflated eigenvalues
        d(na1+1:na) = d2(1:na2)

        call obj%timer%start("lapack_lamrg")
        NVTX_RANGE_PUSH("lapack_lamrg_3")
        ! Calculate arrangement of all eigenvalues  in output
        call PRECISION_LAMRG(int(na1,kind=BLAS_KIND), int(na-na1,kind=BLAS_KIND), d, &
                             1_BLAS_KIND, 1_BLAS_KIND, idxBLAS )
        NVTX_RANGE_POP("lapack_lamrg_3")
        idx(:) = int(idxBLAS(:),kind=ik)
        call obj%timer%stop("lapack_lamrg")

        ! Rearrange eigenvalues
        tmp = d
        do i=1,na
          d(i) = tmp(idx(i))
        enddo
        call check_monotony_&
        &PRECISION&
        &(obj, na,d,'Output', wantDebug, success)

        if (.not.(success)) then
          call obj%timer%stop("merge_systems" // PRECISION_SUFFIX)
          write(error_unit,*) "Error in check_monotony. Aborting..."
          return
        endif
        ! Eigenvector calculations
        if (useGPU) then
          num = 2 * size_of_int     
          successGPU = gpu_malloc(nnzul_dev, num) ! packs together nnzu and nnzl
          check_alloc_gpu("merge_systems: ", successGPU)

          num = na * size_of_int     
          successGPU = gpu_malloc(idxq1_dev, num)
          check_alloc_gpu("merge_systems: ", successGPU)

          num = na * size_of_int     
          successGPU = gpu_malloc(idx_dev, num)
          check_alloc_gpu("merge_systems: idx_dev", successGPU)

          num = na * size_of_int
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memcpy_async(idx_dev, int(loc(idx(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: ", successGPU)
#else
          successGPU = gpu_memcpy      (idx_dev, int(loc(idx(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: idx_dev", successGPU)
#endif
        endif

        ! Calculate the number of columns in the new local matrix Q
        ! which are updated from non-deflated/deflated eigenvectors.
        ! idxq1/2 stores the global column numbers.

        !if (useGPU) then



        !  !nqcols1 is needed later on host !!
        !  ! memcopy back needed!!
        !else
          nqcols1 = 0 ! number of non-deflated eigenvectors
          !nqcols2 = 0 ! number of deflated eigenvectors
          NVTX_RANGE_PUSH("loop_idxq1")
          DO i = 1, na
            if (p_col_out(i)==my_pcol) then
              if (idx(i)<=na1) then
                nqcols1 = nqcols1+1
                idxq1(nqcols1) = i
              !else
                !nqcols2 = nqcols2+1
                !idxq2(nqcols2) = i
              endif
            endif
          enddo
          NVTX_RANGE_POP("loop_idxq1")
        !endif

        if (useGPU) then
          num = na * size_of_int
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memcpy_async(idxq1_dev, int(loc(idxq1(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: ", successGPU)
#else
          successGPU = gpu_memcpy      (idxq1_dev, int(loc(idxq1(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: idxq1_dev", successGPU)
#endif
        endif


        if (useGPU) then
          allocate(ndef_c(na), stat=istat, errmsg=errorMessage)
          check_allocate("merge_systems: ndef_c",istat, errorMessage)
        endif


        gemm_dim_k = MAX(1,l_rows)
        gemm_dim_l = max_local_cols
        gemm_dim_m = MIN(max_strip, MAX(1,nqcols1))

        if (.not. useCCL) then
          allocate(qtmp1(gemm_dim_k, gemm_dim_l), stat=istat, errmsg=errorMessage)
          check_allocate("merge_systems: qtmp1",istat, errorMessage)

          allocate(ev(gemm_dim_l,gemm_dim_m), stat=istat, errmsg=errorMessage)
          check_allocate("merge_systems: ev",istat, errorMessage)

          allocate(qtmp2(gemm_dim_k, gemm_dim_m), stat=istat, errmsg=errorMessage)
          check_allocate("merge_systems: qtmp2",istat, errorMessage)
        endif

        if (useGPU) then
          num = (gemm_dim_k * gemm_dim_l) * size_of_datatype
          successGPU = gpu_malloc(qtmp1_dev, num)
          check_alloc_gpu("merge_systems: qtmp1_dev", successGPU)

          num = (gemm_dim_k * gemm_dim_l) * size_of_datatype
          successGPU = gpu_malloc(qtmp1_tmp_dev, num)
          check_alloc_gpu("merge_systems: qtmp1_tmp_dev", successGPU)

          num = (gemm_dim_l * gemm_dim_m) * size_of_datatype
          successGPU = gpu_malloc(ev_dev, num)
          check_alloc_gpu("merge_systems: ev_dev", successGPU)

          num = (gemm_dim_k * gemm_dim_m) * size_of_datatype
          successGPU = gpu_malloc(qtmp2_dev, num)
          check_alloc_gpu("merge_systems: qtmp2_dev", successGPU)

          if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
            if (wantDebug) call obj%timer%start("gpu_host_register")
            
            if (.not. useCCL) then
              num = (gemm_dim_k * gemm_dim_l) * size_of_datatype
              NVTX_RANGE_PUSH("gpu_host_register_qtmp1")
              successGPU = gpu_host_register(int(loc(qtmp1),kind=c_intptr_t), num, gpuHostRegisterDefault)
              check_host_register_gpu("merge_systems: qtmp1", successGPU)
              NVTX_RANGE_POP("gpu_host_register_qtmp1")

              num = (gemm_dim_l * gemm_dim_m) * size_of_datatype
              successGPU = gpu_host_register(int(loc(ev),kind=c_intptr_t), num, gpuHostRegisterDefault)
              check_host_register_gpu("merge_systems: ev", successGPU)
              
              num = (gemm_dim_k * gemm_dim_m) * size_of_datatype
              successGPU = gpu_host_register(int(loc(qtmp2),kind=c_intptr_t), num, gpuHostRegisterDefault)
              check_host_register_gpu("merge_systems: qtmp2", successGPU)
            endif
  
            if (wantDebug) then
              successGPU = gpu_DeviceSynchronize()
              call obj%timer%stop("gpu_host_register")
            endif
          endif
        endif ! useGPU

        if (useGPU) then
          num = gemm_dim_k * gemm_dim_l * size_of_datatype
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memset_async(qtmp1_dev, 0, num, my_stream)
#else
          successGPU = gpu_memset      (qtmp1_dev, 0, num)
#endif
          check_memcpy_gpu("merge_systems: memset qtmp1_dev", successGPU)
        else
          NVTX_RANGE_PUSH("set_qtmp1_qtmp2_0")
          call obj%timer%start("set_qtmp1_qtmp2_0")
          qtmp1 = 0 ! May contain empty (unset) parts
          qtmp2 = 0 ! Not really needed
          call obj%timer%stop("set_qtmp1_qtmp2_0")
          NVTX_RANGE_POP("set_qtmp1_qtmp2_0")
        endif

        ! Gather nonzero upper/lower components of old matrix Q
        ! which are needed for multiplication with new eigenvectors

        ! kernel compute nnzu on device
        if (useGPU) then
          ! data transfer to GPU
          num = na * size_of_int
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memcpy_async(idx1_dev, int(loc(idx1(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: idx1_dev", successGPU)

          successGPU = gpu_memcpy_async(coltyp_dev, int(loc(coltyp(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: coltyp_dev", successGPU)
#else
          successGPU = gpu_memcpy(idx1_dev, int(loc(idx1(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: idx1_dev", successGPU)

          successGPU = gpu_memcpy(coltyp_dev, int(loc(coltyp(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: coltyp_dev", successGPU)
#endif

          NVTX_RANGE_PUSH("gpu_copy_qtmp1_q_compute_nnzu_nnzl_kernel")
          if (wantDebug) call obj%timer%start("gpu_copy_qtmp1_q_compute_nnzu_nnzl_kernel")

          call gpu_copy_qtmp1_q_compute_nnzu_nnzl(PRECISION_CHAR, qtmp1_dev, q_dev, &
                                                  p_col_dev, l_col_dev, idx1_dev, coltyp_dev, nnzul_dev, &
                                                  na1, l_rnm, l_rqs, l_rqm, l_rows, my_pcol, gemm_dim_k, matrixRows, &
                                                  SM_count, debug, my_stream)

          if (wantDebug) call obj%timer%stop("gpu_copy_qtmp1_q_compute_nnzu_nnzl_kernel")
          NVTX_RANGE_POP("gpu_copy_qtmp1_q_compute_nnzu_nnzl_kernel")

          ! num = 2 * size_of_int
          ! successGPU = gpu_memcpy(int(loc(nnzul(1)),kind=c_intptr_t), nnzul_dev, num, gpuMemcpyDeviceToHost)
          ! check_memcpy_gpu("merge_systems: nnzul_dev", successGPU)

          num = 2 * size_of_int
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memcpy_async(int(loc(nnzul(1)),kind=c_intptr_t), nnzul_dev, num, gpuMemcpyDeviceToHost, my_stream)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("solve_tridi_single: nnzul_dev -> nnzul", successGPU)
#else
          successGPU = gpu_memcpy      (int(loc(nnzul(1)),kind=c_intptr_t), nnzul_dev, num, gpuMemcpyDeviceToHost)
#endif
          check_memcpy_gpu("merge_systems: nnzul_dev", successGPU)

          nnzu = nnzul(1)
          nnzl = nnzul(2)
        else ! useGPU
          NVTX_RANGE_PUSH("loop_compute_nnzu")
          if (wantDebug) call obj%timer%start("loop_compute_nnzu")
          nnzu = 0
          nnzl = 0
          do i = 1, na1
            if (p_col(idx1(i))==my_pcol) then
              l_idx = l_col(idx1(i))
            
              if (coltyp(idx1(i))==1 .or. coltyp(idx1(i))==2) then
                nnzu = nnzu+1
                qtmp1(1:l_rnm,nnzu) = q(l_rqs:l_rqm,l_idx)
              endif

              if (coltyp(idx1(i))==3 .or. coltyp(idx1(i))==2) then
                nnzl = nnzl+1
                qtmp1(l_rnm+1:l_rows,nnzl) = q(l_rqm+1:l_rqe,l_idx)
              endif
            endif
          enddo
          if (wantDebug) call obj%timer%stop("loop_compute_nnzu")
          NVTX_RANGE_POP("loop_compute_nnzu")
        endif ! useGPU

        if (useGPU) then
          call obj%timer%start("gpu_memcpy")
          num = na * size_of_int
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memcpy_async(l_col_dev, int(loc(l_col(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: ", successGPU)
#else
          successGPU = gpu_memcpy      (l_col_dev, int(loc(l_col(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: l_col_dev", successGPU)
#endif
          call obj%timer%stop("gpu_memcpy")
        endif

        ! Gather deflated eigenvalues behind nonzero components

        ! compute ndef on device
        ndef = max(nnzu,nnzl)

        if (useGPU) then
          num = na * size_of_int
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memcpy_async(idx2_dev, int(loc(idx2(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: ", successGPU)

          successGPU = gpu_memcpy_async(p_col_dev, int(loc(p_col(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: ", successGPU)
#else
          successGPU = gpu_memcpy      (idx2_dev, int(loc(idx2(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: idx2_dev", successGPU)

          successGPU = gpu_memcpy      (p_col_dev, int(loc(p_col(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: p_col_dev", successGPU)
#endif
        endif


        if (useGPU) then
          ndef_c(:) = ndef

          num = na * size_of_int     
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memcpy_async(ndef_c_dev, int(loc(ndef_c(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: ndef_c_dev 4", successGPU) 
#else
          successGPU = gpu_memcpy      (ndef_c_dev, int(loc(ndef_c(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: ndef_c_dev", successGPU)
#endif

          call gpu_copy_q_slice_to_qtmp1 (PRECISION_CHAR, qtmp1_dev, q_dev, ndef_c_dev, l_col_dev, idx2_dev, p_col_dev, &
                                          na2, na, my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k, debug, my_stream)
        else
          do i = 1, na2
            l_idx = l_col(idx2(i))
            if (p_col(idx2(i))==my_pcol) then
              ndef = ndef+1
              qtmp1(1:l_rows,ndef) = q(l_rqs:l_rqe,l_idx)
            endif
          enddo
        endif

        l_cols_qreorg = ndef ! Number of columns in reorganized matrix
        if (useGPU) then
          num = na * size_of_int
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memcpy_async(p_col_out_dev, int(loc(p_col_out(1)),kind=c_intptr_t), &
                                        num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: ", successGPU)

          successGPU = gpu_memcpy_async(l_col_out_dev, int(loc(l_col_out(1)),kind=c_intptr_t), &
                                        num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: l_col_out_dev", successGPU)
#else
          successGPU = gpu_memcpy(p_col_out_dev, int(loc(p_col_out(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: p_col_out_dev", successGPU)

          successGPU = gpu_memcpy(l_col_out_dev, int(loc(l_col_out(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: l_col_out_dev", successGPU)
#endif      
        endif


        ! Set (output) Q to 0, it will sum up new Q


        if (useGPU) then
          call gpu_zero_q(PRECISION_CHAR, q_dev, p_col_out_dev, l_col_out_dev, &
                          na, my_pcol, l_rqs, l_rqe, matrixRows, debug, my_stream)
        else
          DO i = 1, na
            if(p_col_out(i)==my_pcol) q(l_rqs:l_rqe,l_col_out(i)) = 0
          enddo
        endif

       ! check memory copies

        if (useGPU) then
#ifdef WITH_GPU_STREAMS
          num = na * size_of_int
          successGPU = gpu_memcpy_async(idx1_dev, int(loc(idx1(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: idx1_dev", successGPU)

          successGPU = gpu_memcpy_async(p_col_dev, int(loc(p_col(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: p_col_dev", successGPU)

          successGPU = gpu_memcpy_async(coltyp_dev, int(loc(coltyp(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: coltyp_dev", successGPU)

          num = na * size_of_datatype
          successGPU = gpu_memcpy_async(ev_scale_dev, int(loc(ev_scale(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: ev_scale_dev", successGPU)

          successGPU = gpu_memcpy_async(z_dev, int(loc(z(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: z_dev", successGPU)

          successGPU = gpu_memcpy_async(d1_dev, int(loc(d1(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: d1_dev", successGPU)

          successGPU = gpu_memcpy_async(dbase_dev, int(loc(dbase(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: dbase_dev", successGPU)

          successGPU = gpu_memcpy_async(ddiff_dev, int(loc(ddiff(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("merge_systems: ddiff_dev", successGPU)
#else
          num = na * size_of_int
          successGPU = gpu_memcpy(idx1_dev, int(loc(idx1(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: idx1_dev", successGPU)

          successGPU = gpu_memcpy(p_col_dev, int(loc(p_col(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: p_col_dev", successGPU)

          successGPU = gpu_memcpy(coltyp_dev, int(loc(coltyp(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: coltyp_dev", successGPU)

          num = na * size_of_datatype
          successGPU = gpu_memcpy(ev_scale_dev, int(loc(ev_scale(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: ev_scale_dev", successGPU)

          successGPU = gpu_memcpy(z_dev, int(loc(z(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: z_dev", successGPU)

          successGPU = gpu_memcpy(d1_dev, int(loc(d1(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: d1_dev", successGPU)

          successGPU = gpu_memcpy(dbase_dev, int(loc(dbase(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: dbase_dev", successGPU)

          successGPU = gpu_memcpy(ddiff_dev, int(loc(ddiff(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("merge_systems: ddiff_dev", successGPU)
#endif      
        endif

        allocate(nnzu_val(na1,npc_n))
        allocate(nnzl_val(na1,npc_n))

        nnzu_val(:,:) = 0
        nnzl_val(:,:) = 0

        if (useGPU) then
          num = na1 * npc_n* size_of_int     
          successGPU = gpu_malloc(nnzu_val_dev, num)
          check_alloc_gpu("merge_systems: nnzu_val_dev", successGPU)

          num = na1 * npc_n* size_of_int     
          successGPU = gpu_malloc(nnzl_val_dev, num)
          check_alloc_gpu("merge_systems: nnzl_val_dev", successGPU)
        endif


        np_rem = my_pcol
        if (useGPU) then
          do np = 1, npc_n
            if (np > 1) then
              if (np_rem == npc_0) then
                np_rem = npc_0+npc_n-1
              else
                np_rem = np_rem-1
              endif
            endif
            nnzu = 0
            nnzl = 0

            call gpu_compute_nnzl_nnzu_val_part1 (p_col_dev, idx1_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, &
                                                  na, na1, np_rem, npc_n, nnzu_save, nnzl_save, np, debug, my_stream)

          enddo ! np = 1, npc_n

          nnzu_start = 0
          nnzl_start = 0

          call gpu_compute_nnzl_nnzu_val_part2 (nnzu_val_dev, nnzl_val_dev, na, na1, nnzu_start, nnzl_start, npc_n, &
                                                debug, my_stream)
        else
          ! precompute nnzu_val, nnzl_val
          do np = 1, npc_n
            if (np > 1) then
              if (np_rem == npc_0) then
                np_rem = npc_0+npc_n-1
              else
                np_rem = np_rem-1
              endif
            endif
            nnzu = 0
            nnzl = 0
            do i=1,na1
              if (p_col(idx1(i)) == np_rem) then
                if (coltyp(idx1(i)) == 1 .or. coltyp(idx1(i)) == 2) then
                  nnzu = nnzu+1
                  nnzu_val(i,np) =  nnzu
                endif
                if (coltyp(idx1(i)) == 3 .or. coltyp(idx1(i)) == 2) then
                  nnzl = nnzl+1
                  nnzl_val(i,np) =  nnzl
                endif
              endif
            enddo
          enddo ! np = 1, npc_n
        endif

        np_rem = my_pcol

        ! is nnzu updated in main loop
        
        ! main loop
        NVTX_RANGE_PUSH("main_loop")
        do np = 1, npc_n
          NVTX_RANGE_PUSH("np=1,npc_n")
          ! Do a ring send of qtmp1
          if (np > 1) then

            if (np_rem == npc_0) then
              np_rem = npc_0+npc_n-1
            else
              np_rem = np_rem-1
            endif

            if (useGPU) then
              if (useCCL) then
                call gpu_copy_qtmp1_to_qtmp1_tmp (PRECISION_CHAR, qtmp1_dev, qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l, &
                                                  debug, my_stream)

                call obj%timer%start("ccl_send_recv")
                successGPU = ccl_group_start()
                if (.not.successGPU) then
                  print *,"Error in setting up ccl_group_start!"
                  stop 1
                endif

                successGPU = ccl_send(qtmp1_tmp_dev, int(l_rows*max_local_cols,kind=c_size_t), &
                                      cclDataType, np_next, ccl_comm_cols, my_stream)

                if (.not.successGPU) then
                  print *,"Error in ccl_send"
                  stop 1
                endif

                successGPU = ccl_recv(qtmp1_dev, int(l_rows*max_local_cols,kind=c_size_t), &
                                      cclDataType, np_prev, ccl_comm_cols, my_stream)


                if (.not.successGPU) then
                  print *,"Error in ccl_recv"
                  stop 1
                endif

                successGPU = ccl_group_end()

                if (.not.successGPU) then
                  print *,"Error in setting up ccl_group_end!"
                  stop 1
                endif
                successGPU = gpu_stream_synchronize(my_stream)
                check_stream_synchronize_gpu("trans_ev", successGPU)
                call obj%timer%stop("ccl_send_recv")
              else ! useCCL        
#ifdef WITH_MPI
                call obj%timer%start("mpi_communication")
#ifdef WITH_GPU_STREAMS
                successGPU = gpu_stream_synchronize(my_stream)
                check_stream_synchronize_gpu("merge_systems qtmp1_dev", successGPU)

                successGPU = gpu_memcpy_async(int(loc(qtmp1(1,1)),kind=c_intptr_t), qtmp1_dev, &
                     gemm_dim_k * gemm_dim_l  * size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
                check_memcpy_gpu("merge_systems: qtmp1_dev", successGPU)

                successGPU = gpu_stream_synchronize(my_stream)
                check_stream_synchronize_gpu("merge_systems: qtmp1_dev", successGPU)
                ! synchronize streamsPerThread; maybe not neccessary
                successGPU = gpu_stream_synchronize()
                check_stream_synchronize_gpu("merge_systems: qtmp1_dev", successGPU)
              
#else
                successGPU = gpu_memcpy(int(loc(qtmp1(1,1)),kind=c_intptr_t), qtmp1_dev, &
                     gemm_dim_k * gemm_dim_l  * size_of_datatype, gpuMemcpyDeviceToHost)
                check_memcpy_gpu("merge_systems: qtmp1_dev", successGPU)
#endif


                call MPI_Sendrecv_replace(qtmp1, int(l_rows*max_local_cols,kind=MPI_KIND), MPI_REAL_PRECISION,     &
                                          int(np_next,kind=MPI_KIND), 1111_MPI_KIND, int(np_prev,kind=MPI_KIND), &
                                          1111_MPI_KIND, int(mpi_comm_cols_self,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
#ifdef WITH_GPU_STREAMS
                successGPU = gpu_stream_synchronize(my_stream)
                check_stream_synchronize_gpu("merge_systems qtmp1_dev", successGPU)
      
                successGPU = gpu_memcpy_async(qtmp1_dev, int(loc(qtmp1(1,1)),kind=c_intptr_t), &
                     gemm_dim_k * gemm_dim_l  * size_of_datatype, gpuMemcpyHostToDevice, my_stream)
                check_memcpy_gpu("merge_systems: qtmp1_dev", successGPU)
      
                successGPU = gpu_stream_synchronize(my_stream)
                check_stream_synchronize_gpu("merge_systems: qtmp1_dev", successGPU)
                ! synchronize streamsPerThread; maybe not neccessary
                successGPU = gpu_stream_synchronize()
                check_stream_synchronize_gpu("merge_systems: qtmp1_dev", successGPU)
            
#else 
                successGPU = gpu_memcpy(qtmp1_dev, int(loc(qtmp1(1,1)),kind=c_intptr_t), &
                     gemm_dim_k * gemm_dim_l  * size_of_datatype, gpuMemcpyHostToDevice)
                check_memcpy_gpu("merge_systems: qtmp1_dev", successGPU)
#endif
                call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */

              endif ! useCCL
            else ! useGPU

#ifdef WITH_MPI
              call obj%timer%start("mpi_communication")
              call MPI_Sendrecv_replace(qtmp1, int(l_rows*max_local_cols,kind=MPI_KIND), MPI_REAL_PRECISION,     &
                                          int(np_next,kind=MPI_KIND), 1111_MPI_KIND, int(np_prev,kind=MPI_KIND), &
                                          1111_MPI_KIND, int(mpi_comm_cols_self,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
              call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */

            endif ! useGPU

          endif ! (np > 1) then

          ! Gather the parts in d1 and z which are fitting to qtmp1.
          ! This also delivers nnzu/nnzl for proc np_rem
          nnzu = 0
          nnzl = 0
          if (useGPU) then

            NVTX_RANGE_PUSH("gpu_fill_tmp_arrays") 
            call gpu_fill_tmp_arrays (PRECISION_CHAR, d1u_dev, d1_dev, zu_dev, z_dev, d1l_dev, zl_dev, &
                                      idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev, &
                                      na, np, na1, np_rem, debug, my_stream)
            if (wantDebug) successGPU = gpu_DeviceSynchronize()
            NVTX_RANGE_POP("gpu_fill_tmp_arrays")

            num = 2* size_of_int
#ifdef WITH_GPU_STREAMS
            successGPU = gpu_memcpy_async(int(loc(nnzul(1)),kind=c_intptr_t), nnzul_dev, num, gpuMemcpyDeviceToHost, my_stream)
            check_memcpy_gpu("merge_systems: nnzul_dev", successGPU)
#else
            successGPU = gpu_memcpy(int(loc(nnzul(1)),kind=c_intptr_t), nnzul_dev, num, gpuMemcpyDeviceToHost)
            check_memcpy_gpu("merge_systems: nnzl_val", successGPU)
#endif
            nnzu = nnzul(1)
            nnzl = nnzul(2)

          else ! useGPU
            do i=1,na1
              if (p_col(idx1(i)) == np_rem) then
                if (coltyp(idx1(i)) == 1 .or. coltyp(idx1(i)) == 2) then
                  nnzu = nnzu+1
                  d1u(nnzu) = d1(i)
                  zu (nnzu) = z (i)
                endif
                if (coltyp(idx1(i)) == 3 .or. coltyp(idx1(i)) == 2) then
                  nnzl = nnzl+1
                  d1l(nnzl) = d1(i)
                  zl (nnzl) = z (i)
                endif
              endif
            enddo
          endif ! useGPU

          ! Set the deflated eigenvectors in Q (comming from proc np_rem)


          ndef = MAX(nnzu,nnzl) ! Remote counter in input matrix
          if (useGPU) then
            call gpu_update_ndef_c(ndef_c_dev, idx_dev, p_col_dev, idx2_dev, na, na1, np_rem, ndef, debug, my_stream)

          endif ! useGPU

          ndef = MAX(nnzu,nnzl) ! Remote counter in input matrix
          if (useGPU) then
            call gpu_copy_qtmp1_slice_to_q (PRECISION_CHAR, q_dev, qtmp1_dev, &
                                            l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, idx2_dev, idx_dev, &
                                            l_rqs, l_rqe, l_rows, matrixRows, gemm_dim_k,  my_pcol, na1, np_rem,  na, &
                                            debug, my_stream)
          else ! ! useGPU
            ndef = MAX(nnzu,nnzl) ! Remote counter in input matrix
            do i = 1, na
              j = idx(i)
              if (j>na1) then
                if (p_col(idx2(j-na1)) == np_rem) then
                  ndef = ndef+1
                  if (p_col_out(i) == my_pcol) then
                    q(l_rqs:l_rqe,l_col_out(i)) = qtmp1(1:l_rows,ndef)
                  endif
                endif
              endif
            enddo

          endif ! useGPU


          do ns = 0, nqcols1-1, max_strip ! "strimining" (strip mining) loop
            NVTX_RANGE_PUSH("ns=0,nqcols1-1,max_strip")
            ncnt = MIN(max_strip,nqcols1-ns) ! number of columns in this strip

            ! Get partial result from (output) Q
            if (useGPU) then
              call gpu_copy_q_slice_to_qtmp2 (PRECISION_CHAR, q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, & 
                                              l_rows, l_rqs, l_rqe, matrixRows, matrixCols, & 
                                              gemm_dim_k, gemm_dim_m, ns, ncnt, ind_ex, ind_ex2, na, debug, my_stream)
            else ! useGPU
!$omp PARALLEL DO &
!$omp default(none) &
!$omp private(i, j, k) &
!$omp SHARED(ns, q, l_rqs, l_rqe, l_col_out, idxq1, qtmp2, l_rows, ncnt)
              do i = 1, ncnt
                j = idxq1(i+ns)
                k = l_col_out(j)
                qtmp2(1:l_rows,i) = q(l_rqs:l_rqe, k)
              enddo
!$OMP END PARALLEL DO
            endif ! useGPU

            ! Compute eigenvectors of the rank-1 modified matrix.
            ! Parts for multiplying with upper half of Q:
            if (useGPU) then
              if (nnzu .ge. 1) then
                ! Calculate the j-th eigenvector of the deflated system
                ! See above why we are doing it this way!
                call gpu_fill_ev (PRECISION_CHAR, ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev, idxq1_dev, idx_dev,&
                                  na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt, debug, my_stream) 
              endif ! nnzu

            else ! useGPU
!$omp PARALLEL DO &
!$omp default(none) &
!$omp private(i, j, k, tmp) &
!$omp shared(ncnt, nnzu, idx, idxq1, ns, d1u, dbase, ddiff, zu, ev_scale, ev)
              do i = 1, ncnt
                do k = 1, nnzu
                  j = idx(idxq1(i+ns))

                  ! Calculate the j-th eigenvector of the deflated system
                  ! See above why we are doing it this way!

                  ! kernel here
                  tmp(k) = d1u(k) - dbase(j)
                  tmp(k) = tmp(k) + ddiff(j)
                  ev(k,i) = zu(k) / tmp(k) * ev_scale(j)
                enddo
              enddo
!$OMP END PARALLEL DO
            endif ! useGPU

            ! Multiply old Q with eigenvectors (upper half)

            if (l_rnm>0 .and. ncnt>0 .and. nnzu>0) then
              if (useGPU) then
                call obj%timer%start("gpublas_gemm")
                NVTX_RANGE_PUSH("gpublas_gemm_upper")
                gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
                call gpublas_PRECISION_GEMM('N', 'N', l_rnm, ncnt, nnzu, 1.0_rk, &
                                            qtmp1_dev, gemm_dim_k,    &
                                            ev_dev,    gemm_dim_l, 1.0_rk, &
                                            qtmp2_dev, gemm_dim_k, gpuHandle)
                if (wantDebug) successGPU = gpu_DeviceSynchronize()
                NVTX_RANGE_POP("gpublas_gemm_upper")
                call obj%timer%stop("gpublas_gemm")
              else ! useGPU
                call obj%timer%start("blas_gemm")
                call PRECISION_GEMM('N', 'N', int(l_rnm,kind=BLAS_KIND), int(ncnt,kind=BLAS_KIND), &
                                    int(nnzu,kind=BLAS_KIND), 1.0_rk, &
                                    qtmp1,      int(gemm_dim_k,kind=BLAS_KIND), &
                                    ev,         int(gemm_dim_l,kind=BLAS_KIND), 1.0_rk, &
                                    qtmp2(1,1), int(gemm_dim_k,kind=BLAS_KIND))
                call obj%timer%stop("blas_gemm")
              endif ! useGPU
            endif ! (l_rnm>0 .and. ncnt>0 .and. nnzu>0) then


            ! Compute eigenvectors of the rank-1 modified matrix.
            ! Parts for multiplying with lower half of Q:

            if (useGPU) then
              if (nnzl .ge. 1) then
                call gpu_fill_ev (PRECISION_CHAR, ev_dev, d1l_dev, dbase_dev, ddiff_dev, zl_dev, ev_scale_dev, idxq1_dev, idx_dev, &
                                            na, gemm_dim_l, gemm_dim_m, nnzl, ns, ncnt, debug, my_stream)
              endif
            else ! useGPU
!$omp PARALLEL DO &
!$omp private(i, j, k, tmp)
              do i = 1, ncnt
                do k = 1, nnzl
                  j = idx(idxq1(i+ns))
                  ! Calculate the j-th eigenvector of the deflated system
                  ! See above why we are doing it this way!
                  tmp(k) = d1l(k) - dbase(j)
                  tmp(k) = tmp(k) + ddiff(j)
                  ev(k,i) = zl(k) / tmp(k) * ev_scale(j)
                enddo
              enddo
!$OMP END PARALLEL DO
            endif ! useGPU


            ! Multiply old Q with eigenvectors (lower half)

            if (l_rows-l_rnm>0 .and. ncnt>0 .and. nnzl>0) then
              if (useGPU) then
                call obj%timer%start("gpublas_gemm")
                NVTX_RANGE_PUSH("gpublas_gemm_lower")
                gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
                call gpublas_PRECISION_GEMM('N', 'N', l_rows-l_rnm, ncnt, nnzl, 1.0_rk, &
                                            qtmp1_dev + l_rnm*size_of_datatype, gemm_dim_k,   &
                                            ev_dev,                             gemm_dim_l, 1.0_rk, &
                                            qtmp2_dev + l_rnm*size_of_datatype, gemm_dim_k, gpuHandle)
                if (wantDebug) successGPU = gpu_DeviceSynchronize()
                NVTX_RANGE_POP("gpublas_gemm_lower")
                call obj%timer%stop("gpublas_gemm")
              else ! useGPU
                call obj%timer%start("blas_gemm")
                call PRECISION_GEMM('N', 'N', int(l_rows-l_rnm,kind=BLAS_KIND), int(ncnt,kind=BLAS_KIND),  &
                                    int(nnzl,kind=BLAS_KIND), 1.0_rk, &
                                    qtmp1(l_rnm+1,1), int(gemm_dim_k,kind=BLAS_KIND), &
                                    ev,               int(gemm_dim_l,kind=BLAS_KIND), 1.0_rk, &
                                    qtmp2(l_rnm+1,1), int(gemm_dim_k,kind=BLAS_KIND))
                call obj%timer%stop("blas_gemm")
              endif ! useGPU
            endif

            ! Put partial result into (output) Q
            if (useGPU) then
              call gpu_copy_qtmp2_slice_to_q (PRECISION_CHAR, q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, &
                                              l_rqs, l_rqe, l_rows, ncnt, gemm_dim_k, matrixRows, ns, debug, my_stream)

            else ! useGPU
!$omp PARALLEL DO &
!$omp default(none) &
!$omp private(i) &
!$omp SHARED(q, ns, l_rqs, l_rqe, l_col_out, idxq1, qtmp2, l_rows, ncnt)
              do i = 1, ncnt
                q(l_rqs:l_rqe,l_col_out(idxq1(i+ns))) = qtmp2(1:l_rows,i)
              enddo
!$OMP END PARALLEL DO
            endif ! useGPU

            NVTX_RANGE_POP("ns=0,nqcols1-1,max_strip")
          enddo   ! ns = 0, nqcols1-1, max_strip ! strimining loop
        
          NVTX_RANGE_POP("np=1,npc_n")
        enddo    ! do np = 1, npc_n
        NVTX_RANGE_POP("main_loop")
        

        deallocate(nnzu_val, nnzl_val)


        if (useGPU) then
          deallocate(ndef_c, stat=istat, errmsg=errorMessage)
          check_deallocate("merge_systems: ndef_c",istat, errorMessage)
        endif

        if (useGPU) then
          successGPU = gpu_free(nnzul_dev)
          check_dealloc_gpu("merge_systems: nnzul_dev", successGPU)

          successGPU = gpu_free(l_col_dev)
          check_dealloc_gpu("merge_systems: l_col_dev", successGPU)

          successGPU = gpu_free(ndef_c_dev)
          check_dealloc_gpu("merge_systems: ndef_c_dev", successGPU)

          successGPU = gpu_free(nnzu_val_dev)
          check_dealloc_gpu("merge_systems: nnzu_val_dev", successGPU)

          successGPU = gpu_free(nnzl_val_dev)
          check_dealloc_gpu("merge_systems: nnzl_val_dev", successGPU)

          successGPU = gpu_free(idx1_dev)
          check_dealloc_gpu("merge_systems: idx1_dev", successGPU)

          successGPU = gpu_free(idx2_dev)
          check_dealloc_gpu("merge_systems: idx2_dev", successGPU)

          successGPU = gpu_free(p_col_dev)
          check_dealloc_gpu("merge_systems: p_col_dev", successGPU)

          successGPU = gpu_free(p_col_out_dev)
          check_dealloc_gpu("merge_systems: p_col_out_dev", successGPU)

          successGPU = gpu_free(coltyp_dev)
          check_dealloc_gpu("merge_systems: coltyp_dev", successGPU)

          successGPU = gpu_free(idx_dev)
          check_dealloc_gpu("merge_systems: idx_dev", successGPU)

          successGPU = gpu_free(l_col_out_dev)
          check_dealloc_gpu("merge_systems: l_col_out_dev", successGPU)

          successGPU = gpu_free(idxq1_dev)
          check_dealloc_gpu("merge_systems: ", successGPU)

          successGPU = gpu_free(d1_dev)
          check_dealloc_gpu("merge_systems: d1_dev", successGPU)

          successGPU = gpu_free(z_dev)
          check_dealloc_gpu("merge_systems: z_dev", successGPU)

          successGPU = gpu_free(z1_dev)
          check_dealloc_gpu("merge_systems: z1_dev", successGPU)

          successGPU = gpu_free(rho_dev)
          check_dealloc_gpu("merge_systems: rho_dev", successGPU)

          successGPU = gpu_free(d1u_dev)
          check_dealloc_gpu("merge_systems: d1u_dev", successGPU)

          successGPU = gpu_free(dbase_dev)
          check_dealloc_gpu("merge_systems: dbase_dev", successGPU)

          successGPU = gpu_free(ddiff_dev)
          check_dealloc_gpu("merge_systems: ddiff_dev", successGPU)

          successGPU = gpu_free(zu_dev)
          check_dealloc_gpu("merge_systems: zu_dev", successGPU)

          successGPU = gpu_free(ev_scale_dev)
          check_dealloc_gpu("merge_systems: ev_scale_dev", successGPU)

          successGPU = gpu_free(d1l_dev)
          check_dealloc_gpu("merge_systems: d1l_dev", successGPU)

          successGPU = gpu_free(zl_dev)
          check_dealloc_gpu("merge_systems: zl_dev", successGPU)
        
          successGPU = gpu_free(qtmp1_dev)
          check_dealloc_gpu("merge_systems: qtmp1_dev", successGPU)
          
          successGPU = gpu_free(qtmp1_tmp_dev)
          check_dealloc_gpu("merge_systems: qtmp1_tmp_dev", successGPU)

          successGPU = gpu_free(qtmp2_dev)
          check_dealloc_gpu("merge_systems: qtmp2_dev", successGPU)

          successGPU = gpu_free(ev_dev)
          check_dealloc_gpu("merge_systems: ev_dev", successGPU)

          successGPU = gpu_free(tmp_dev)
          check_dealloc_gpu("merge_systems: tmp_dev", successGPU)

          successGPU = gpu_free(zero_dev)
          check_dealloc_gpu("merge_systems: zero_dev", successGPU)

          successGPU = gpu_free(one_dev)
          check_dealloc_gpu("merge_systems: one_dev", successGPU)

          successGPU = gpu_free(qtrans_dev)
          check_dealloc_gpu("merge_systems: qtrans_dev", successGPU)

          if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
            if (wantDebug) call obj%timer%start("gpu_host_register")
            if (.not. useCCL) then
              successGPU = gpu_host_unregister(int(loc(qtmp1),kind=c_intptr_t))
              check_host_unregister_gpu("merge_systems: qtmp1", successGPU)

              successGPU = gpu_host_unregister(int(loc(qtmp2),kind=c_intptr_t))
              check_host_unregister_gpu("merge_systems: qtmp2", successGPU)
  
              successGPU = gpu_host_unregister(int(loc(ev),kind=c_intptr_t))
              check_host_unregister_gpu("merge_systems: ev", successGPU)
            endif

            if (wantDebug) successGPU = gpu_DeviceSynchronize()
            if (wantDebug) call obj%timer%stop("gpu_host_register")
          endif
        endif ! useGPU

        if (.not. useCCL) then
          deallocate(ev, qtmp1, qtmp2, stat=istat, errmsg=errorMessage)
          check_deallocate("merge_systems: ev, qtmp1, qtmp2",istat, errorMessage)
        endif
      endif !very outer test if (na1==1 .or. na1==2) else (na1>2)

! #ifdef WITH_OPENMP_TRADITIONAL
!       deallocate(z_p, stat=istat, errmsg=errorMessage)
!       check_deallocate("merge_systems: z_p",istat, errorMessage)
! #endif

      call obj%timer%stop("merge_systems" // PRECISION_SUFFIX)

      return

    end 
