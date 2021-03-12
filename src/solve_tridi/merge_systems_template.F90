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

    subroutine merge_systems_&
    &PRECISION &
                         (obj, na, nm, d, e, q, ldq, nqoff, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, &
                          l_col, p_col, l_col_out, p_col_out, npc_0, npc_n, useGPU, wantDebug, success, max_threads)
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
#ifdef WITH_OPENMP_TRADITIONAL
      use omp_lib
#endif
      implicit none
#include "../general/precision_kinds.F90"
      class(elpa_abstract_impl_t), intent(inout)  :: obj
      integer(kind=ik), intent(in)                :: na, nm, ldq, nqoff, nblk, matrixCols, mpi_comm_rows, &
                                                     mpi_comm_cols, npc_0, npc_n
      integer(kind=ik), intent(in)                :: l_col(na), p_col(na), l_col_out(na), p_col_out(na)
      real(kind=REAL_DATATYPE), intent(inout)     :: d(na), e
#ifdef USE_ASSUMED_SIZE
      real(kind=REAL_DATATYPE), intent(inout)     :: q(ldq,*)
#else
      real(kind=REAL_DATATYPE), intent(inout)     :: q(ldq,matrixCols)
#endif
      logical, intent(in)                         :: useGPU, wantDebug
      logical                                     :: useIntelGPU

      logical, intent(out)                        :: success

      ! TODO: play with max_strip. If it was larger, matrices being multiplied
      ! might be larger as well!
      integer(kind=ik), parameter                 :: max_strip=128

      
      real(kind=REAL_DATATYPE)                    :: beta, sig, s, c, t, tau, rho, eps, tol, &
                                                     qtrans(2,2), dmax, zmax, d1new, d2new
      real(kind=REAL_DATATYPE)                    :: z(na), d1(na), d2(na), z1(na), delta(na),  &
                                                     dbase(na), ddiff(na), ev_scale(na), tmp(na)
      real(kind=REAL_DATATYPE)                    :: d1u(na), zu(na), d1l(na), zl(na)
      real(kind=REAL_DATATYPE), allocatable       :: qtmp1(:,:), qtmp2(:,:), ev(:,:)
#ifdef WITH_OPENMP_TRADITIONAL
      real(kind=REAL_DATATYPE), allocatable       :: z_p(:,:)
#endif

      integer(kind=ik)                            :: i, j, na1, na2, l_rows, l_cols, l_rqs, l_rqe, &
                                                     l_rqm, ns, info
      integer(kind=BLAS_KIND)                     :: infoBLAS
      integer(kind=ik)                            :: l_rnm, nnzu, nnzl, ndef, ncnt, max_local_cols, &
                                                     l_cols_qreorg, np, l_idx, nqcols1, nqcols2
      integer(kind=ik)                            :: my_proc, n_procs, my_prow, my_pcol, np_rows, &
                                                     np_cols
      integer(kind=MPI_KIND)                      :: mpierr
      integer(kind=MPI_KIND)                      :: my_prowMPI, np_rowsMPI, my_pcolMPI, np_colsMPI
      integer(kind=ik)                            :: np_next, np_prev, np_rem
      integer(kind=ik)                            :: idx(na), idx1(na), idx2(na)
      integer(kind=BLAS_KIND)                     :: idxBLAS(NA)
      integer(kind=ik)                            :: coltyp(na), idxq1(na), idxq2(na)

      integer(kind=ik)                            :: istat
      character(200)                              :: errorMessage
      integer(kind=ik)                            :: gemm_dim_k, gemm_dim_l, gemm_dim_m

      integer(kind=c_intptr_t)                    :: num
      integer(kind=C_intptr_T)                    :: qtmp1_dev, qtmp2_dev, ev_dev
      logical                                     :: successGPU
      integer(kind=c_intptr_t), parameter         :: size_of_datatype = size_of_&
                                                                      &PRECISION&
                                                                      &_real
      integer(kind=ik), intent(in)                :: max_threads
#ifdef WITH_OPENMP_TRADITIONAL
      integer(kind=ik)                            :: my_thread

      allocate(z_p(na,0:max_threads-1), stat=istat, errmsg=errorMessage)
      check_allocate("merge_systems: z_p",istat, errorMessage)
#endif
      useIntelGPU = .false.
      if (useGPU) then
        if (gpu_vendor() == INTEL_GPU) then
          useIntelGPU = .true.
        endif
      endif

      call obj%timer%start("merge_systems" // PRECISION_SUFFIX)
      success = .true.
      call obj%timer%start("mpi_communication")
      call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND) ,my_prowMPI, mpierr)
      call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND) ,np_rowsMPI, mpierr)
      call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND) ,my_pcolMPI, mpierr)
      call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND) ,np_colsMPI, mpierr)

      my_prow = int(my_prowMPI,kind=c_int)
      np_rows = int(np_rowsMPI,kind=c_int)
      my_pcol = int(my_pcolMPI,kind=c_int)
      np_cols = int(np_colsMPI,kind=c_int)

      call obj%timer%stop("mpi_communication")

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

      ! Calculations start here

      beta = abs(e)
      sig  = sign(1.0_rk,e)

      ! Calculate rank-1 modifier z

      z(:) = 0

      if (MOD((nqoff+nm-1)/nblk,np_rows)==my_prow) then
        ! nm is local on my row
        do i = 1, na
          if (p_col(i)==my_pcol) z(i) = q(l_rqm,l_col(i))
         enddo
      endif

      if (MOD((nqoff+nm)/nblk,np_rows)==my_prow) then
        ! nm+1 is local on my row
        do i = 1, na
          if (p_col(i)==my_pcol) z(i) = z(i) + sig*q(l_rqm+1,l_col(i))
        enddo
      endif

      call global_gather_&
      &PRECISION&
      &(obj, z, na, mpi_comm_rows, mpi_comm_cols, npc_n, np_prev, np_next)
      ! Normalize z so that norm(z) = 1.  Since z is the concatenation of
      ! two normalized vectors, norm2(z) = sqrt(2).
      z = z/sqrt(2.0_rk)
      rho = 2.0_rk*beta
      ! Calculate index for merging both systems by ascending eigenvalues
      call obj%timer%start("blas")
      call PRECISION_LAMRG( int(nm,kind=BLAS_KIND), int(na-nm,kind=BLAS_KIND), d, &
                            1_BLAS_KIND, 1_BLAS_KIND, idxBLAS )
      idx(:) = int(idxBLAS(:),kind=ik)
      call obj%timer%stop("blas")

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
        call resort_ev_&
        &PRECISION &
                       (obj, idx, na, na, p_col_out, q, ldq, matrixCols, l_rows, l_rqe, &
                        l_rqs, mpi_comm_cols, p_col, l_col, l_col_out)

        call obj%timer%stop("merge_systems" // PRECISION_SUFFIX)

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

          ! Find sqrt(a**2+b**2) without overflow or
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
            call transform_columns_&
            &PRECISION &
                        (obj, idx(i), idx1(na1), na, tmp, l_rqs, l_rqe, &
                         q, ldq, matrixCols, l_rows, mpi_comm_cols, &
                          p_col, l_col, qtrans)
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

      enddo
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
          call obj%timer%start("blas")
          call PRECISION_LAED5(1_BLAS_KIND, d1, z1, qtrans(1,1), rho, d(1))
          call PRECISION_LAED5(2_BLAS_KIND, d1, z1, qtrans(1,2), rho, d(2))
          call obj%timer%stop("blas")
          call transform_columns_&
          &PRECISION&
          &(obj, idx1(1), idx1(2), na, tmp, l_rqs, l_rqe, q, &
            ldq, matrixCols, l_rows, mpi_comm_cols, &
             p_col, l_col, qtrans)

        endif

        ! Add the deflated eigenvalues
        d(na1+1:na) = d2(1:na2)

        ! Calculate arrangement of all eigenvalues  in output
        call obj%timer%start("blas")
        call PRECISION_LAMRG( int(na1,kind=BLAS_KIND), int(na-na1,kind=BLAS_KIND), d, &
                              1_BLAS_KIND, 1_BLAS_KIND, idxBLAS )
        idx(:) = int(idxBLAS(:),kind=ik)
        call obj%timer%stop("blas")
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
        call resort_ev_&
        &PRECISION&
        &(obj, idxq1, na, na, p_col_out, q, ldq, matrixCols, l_rows, l_rqe, &
          l_rqs, mpi_comm_cols, p_col, l_col, l_col_out)

      else if (na1>2) then

        ! Solve secular equation

        z(1:na1) = 1
#ifdef WITH_OPENMP_TRADITIONAL
        z_p(1:na1,:) = 1
#endif
        dbase(1:na1) = 0
        ddiff(1:na1) = 0

        info = 0
        infoBLAS = int(info,kind=BLAS_KIND)
!#ifdef WITH_OPENMP_TRADITIONAL
!
!        call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)
!!$OMP PARALLEL PRIVATE(i,my_thread,delta,s,info,infoBLAS,j)
!        my_thread = omp_get_thread_num()
!!$OMP DO
!#endif
        DO i = my_proc+1, na1, n_procs ! work distributed over all processors
          call obj%timer%start("blas")
          call PRECISION_LAED4(int(na1,kind=BLAS_KIND), int(i,kind=BLAS_KIND), d1, z1, delta, &
                               rho, s, infoBLAS) ! s is not used!
          info = int(infoBLAS,kind=ik)
          call obj%timer%stop("blas")
          if (info/=0) then
            ! If DLAED4 fails (may happen especially for LAPACK versions before 3.2)
            ! use the more stable bisection algorithm in solve_secular_equation
            ! print *,'ERROR DLAED4 n=',na1,'i=',i,' Using Bisection'
            call solve_secular_equation_&
            &PRECISION&
            &(obj, na1, i, d1, z1, delta, rho, s)
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
          ! store dbase/ddiff

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
        enddo
!#ifdef WITH_OPENMP_TRADITIONAL
!!$OMP END PARALLEL
!
!        call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)
!
!        do i = 0, max_threads-1
!          z(1:na1) = z(1:na1)*z_p(1:na1,i)
!        enddo
!#endif

        call global_product_&
        &PRECISION&
        (obj, z, na1, mpi_comm_rows, mpi_comm_cols, npc_0, npc_n)
        z(1:na1) = SIGN( SQRT( -z(1:na1) ), z1(1:na1) )

        call global_gather_&
        &PRECISION&
        &(obj, dbase, na1, mpi_comm_rows, mpi_comm_cols, npc_n, np_prev, np_next)
        call global_gather_&
        &PRECISION&
        &(obj, ddiff, na1, mpi_comm_rows, mpi_comm_cols, npc_n, np_prev, np_next)
        d(1:na1) = dbase(1:na1) - ddiff(1:na1)

        ! Calculate scale factors for eigenvectors
        ev_scale(:) = 0.0_rk

#ifdef WITH_OPENMP_TRADITIONAL

        call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)

!$omp PARALLEL DO &
!$omp default(none) &
!$omp private(i) &
!$omp SHARED(na1, my_proc, n_procs,  &
!$OMP d1,dbase, ddiff, z, ev_scale, obj)

#endif
        DO i = my_proc+1, na1, n_procs ! work distributed over all processors

          ! tmp(1:na1) = z(1:na1) / delta(1:na1,i)  ! original code
          ! tmp(1:na1) = z(1:na1) / (d1(1:na1)-d(i))! bad results

          ! All we want to calculate is tmp = (d1(1:na1)-dbase(i))+ddiff(i)
          ! in exactly this order, but we want to prevent compiler optimization
!         ev_scale_val = ev_scale(i)
          call add_tmp_&
          &PRECISION&
          &(obj, d1, dbase, ddiff, z, ev_scale(i), na1,i)
!         ev_scale(i) = ev_scale_val
        enddo
#ifdef WITH_OPENMP_TRADITIONAL
!$OMP END PARALLEL DO

        call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)

#endif

        call global_gather_&
        &PRECISION&
        &(obj, ev_scale, na1, mpi_comm_rows, mpi_comm_cols, npc_n, np_prev, np_next)
        ! Add the deflated eigenvalues
        d(na1+1:na) = d2(1:na2)

        call obj%timer%start("blas")
        ! Calculate arrangement of all eigenvalues  in output
        call PRECISION_LAMRG(int(na1,kind=BLAS_KIND), int(na-na1,kind=BLAS_KIND), d, &
                             1_BLAS_KIND, 1_BLAS_KIND, idxBLAS )
        idx(:) = int(idxBLAS(:),kind=ik)
        call obj%timer%stop("blas")
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
          return
        endif
        ! Eigenvector calculations


        ! Calculate the number of columns in the new local matrix Q
        ! which are updated from non-deflated/deflated eigenvectors.
        ! idxq1/2 stores the global column numbers.

        nqcols1 = 0 ! number of non-deflated eigenvectors
        nqcols2 = 0 ! number of deflated eigenvectors
        DO i = 1, na
          if (p_col_out(i)==my_pcol) then
            if (idx(i)<=na1) then
              nqcols1 = nqcols1+1
              idxq1(nqcols1) = i
            else
              nqcols2 = nqcols2+1
              idxq2(nqcols2) = i
            endif
          endif
        enddo

        gemm_dim_k = MAX(1,l_rows)
        gemm_dim_l = max_local_cols
        gemm_dim_m = MIN(max_strip,MAX(1,nqcols1))

        allocate(qtmp1(gemm_dim_k, gemm_dim_l), stat=istat, errmsg=errorMessage)
        check_allocate("merge_systems: qtmp1",istat, errorMessage)

        allocate(ev(gemm_dim_l,gemm_dim_m), stat=istat, errmsg=errorMessage)
        check_allocate("merge_systems: ev",istat, errorMessage)

        allocate(qtmp2(gemm_dim_k, gemm_dim_m), stat=istat, errmsg=errorMessage)
        check_allocate("merge_systems: qtmp2",istat, errorMessage)

        qtmp1 = 0 ! May contain empty (unset) parts
        qtmp2 = 0 ! Not really needed

        if (useGPU .and. .not.(useIntelGPU) ) then
          num = (gemm_dim_k * gemm_dim_l) * size_of_datatype
          successGPU = gpu_host_register(int(loc(qtmp1),kind=c_intptr_t),num,&
                        gpuHostRegisterDefault)
          check_host_register_gpu("merge_systems: qtmp1", successGPU)

          successGPU = gpu_malloc(qtmp1_dev, num)
          check_alloc_gpu("merge_systems: qtmp1_dev", successGPU)

          num = (gemm_dim_l * gemm_dim_m) * size_of_datatype
          successGPU = gpu_host_register(int(loc(ev),kind=c_intptr_t),num,&
                        gpuHostRegisterDefault)
          check_host_register_gpu("merge_systems: ev", successGPU)

          successGPU = gpu_malloc(ev_dev, num)
          check_alloc_gpu("merge_systems: ev_dev", successGPU)


          num = (gemm_dim_k * gemm_dim_m) * size_of_datatype
          successGPU = gpu_host_register(int(loc(qtmp2),kind=c_intptr_t),num,&
                        gpuHostRegisterDefault)
          check_host_register_gpu("merge_systems: qtmp2", successGPU)

          successGPU = gpu_malloc(qtmp2_dev, num)
          check_alloc_gpu("merge_systems: qtmp2_dev", successGPU)
        endif

        !if (useIntelGPU) then
        !  ! needed later
        !endif

        ! Gather nonzero upper/lower components of old matrix Q
        ! which are needed for multiplication with new eigenvectors

        nnzu = 0
        nnzl = 0
        do i = 1, na1
          l_idx = l_col(idx1(i))
          if (p_col(idx1(i))==my_pcol) then
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

        ! Gather deflated eigenvalues behind nonzero components

        ndef = max(nnzu,nnzl)
        do i = 1, na2
          l_idx = l_col(idx2(i))
          if (p_col(idx2(i))==my_pcol) then
            ndef = ndef+1
            qtmp1(1:l_rows,ndef) = q(l_rqs:l_rqe,l_idx)
          endif
        enddo

        l_cols_qreorg = ndef ! Number of columns in reorganized matrix

        ! Set (output) Q to 0, it will sum up new Q

        DO i = 1, na
          if(p_col_out(i)==my_pcol) q(l_rqs:l_rqe,l_col_out(i)) = 0
        enddo

        np_rem = my_pcol

        do np = 1, npc_n
          ! Do a ring send of qtmp1

          if (np>1) then

            if (np_rem==npc_0) then
              np_rem = npc_0+npc_n-1
            else
              np_rem = np_rem-1
            endif
#ifdef WITH_MPI
            call obj%timer%start("mpi_communication")
            call MPI_Sendrecv_replace(qtmp1, int(l_rows*max_local_cols,kind=MPI_KIND), MPI_REAL_PRECISION,     &
                                        int(np_next,kind=MPI_KIND), 1111_MPI_KIND, int(np_prev,kind=MPI_KIND), &
                                        1111_MPI_KIND, int(mpi_comm_cols,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
            call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
          endif

          if (useGPU .and. .not.(useIntelGPU)) then
            successGPU = gpu_memcpy(qtmp1_dev, int(loc(qtmp1(1,1)),kind=c_intptr_t), &
                 gemm_dim_k * gemm_dim_l  * size_of_datatype, gpuMemcpyHostToDevice)
            check_memcpy_gpu("merge_systems: qtmp1_dev", successGPU)
          endif

          !if (useIntelGPU) then
          !  ! needed later
          !endif


          ! Gather the parts in d1 and z which are fitting to qtmp1.
          ! This also delivers nnzu/nnzl for proc np_rem

          nnzu = 0
          nnzl = 0
          do i=1,na1
            if (p_col(idx1(i))==np_rem) then
              if (coltyp(idx1(i))==1 .or. coltyp(idx1(i))==2) then
                nnzu = nnzu+1
                d1u(nnzu) = d1(i)
                zu (nnzu) = z (i)
              endif
              if (coltyp(idx1(i))==3 .or. coltyp(idx1(i))==2) then
                nnzl = nnzl+1
                d1l(nnzl) = d1(i)
                zl (nnzl) = z (i)
              endif
            endif
          enddo

          ! Set the deflated eigenvectors in Q (comming from proc np_rem)

          ndef = MAX(nnzu,nnzl) ! Remote counter in input matrix
          do i = 1, na
            j = idx(i)
            if (j>na1) then
              if (p_col(idx2(j-na1))==np_rem) then
                ndef = ndef+1
                if (p_col_out(i)==my_pcol) &
                      q(l_rqs:l_rqe,l_col_out(i)) = qtmp1(1:l_rows,ndef)
              endif
            endif
          enddo

          do ns = 0, nqcols1-1, max_strip ! strimining loop

            ncnt = MIN(max_strip,nqcols1-ns) ! number of columns in this strip

            ! Get partial result from (output) Q

            do i = 1, ncnt
              qtmp2(1:l_rows,i) = q(l_rqs:l_rqe,l_col_out(idxq1(i+ns)))
            enddo

            ! Compute eigenvectors of the rank-1 modified matrix.
            ! Parts for multiplying with upper half of Q:

            do i = 1, ncnt
              j = idx(idxq1(i+ns))
              ! Calculate the j-th eigenvector of the deflated system
              ! See above why we are doing it this way!
              tmp(1:nnzu) = d1u(1:nnzu)-dbase(j)
              call v_add_s_&
              &PRECISION&
              &(obj,tmp,nnzu,ddiff(j))
              ev(1:nnzu,i) = zu(1:nnzu) / tmp(1:nnzu) * ev_scale(j)
            enddo

            if(useGPU .and. .not.(useIntelGPU) ) then
              !TODO: it should be enough to copy l_rows x ncnt
              successGPU = gpu_memcpy(qtmp2_dev, int(loc(qtmp2(1,1)),kind=c_intptr_t), &
                                 gemm_dim_k * gemm_dim_m * size_of_datatype, gpuMemcpyHostToDevice)
              check_memcpy_gpu("merge_systems: qtmp2_dev", successGPU)

              !TODO the previous loop could be possible to do on device and thus
              !copy less
              successGPU = gpu_memcpy(ev_dev, int(loc(ev(1,1)),kind=c_intptr_t), &
                                 gemm_dim_l * gemm_dim_m * size_of_datatype, gpuMemcpyHostToDevice)
              check_memcpy_gpu("merge_systems: ev_dev", successGPU)
            endif

            !if (useIntelGPU) then
            !  ! needed later
            !endif

            ! Multiply old Q with eigenvectors (upper half)

            if (l_rnm>0 .and. ncnt>0 .and. nnzu>0) then
              if (useGPU) then
                if (useIntelGPU) then
                  call obj%timer%start("mkl_offload")
#ifdef WITH_INTEL_GPU_VERSION
                  call mkl_offload_PRECISION_GEMM('N', 'N', int(l_rnm,kind=BLAS_KIND), int(ncnt,kind=BLAS_KIND), &
                                    int(nnzu,kind=BLAS_KIND),   &
                                    1.0_rk, qtmp1, int(ubound(qtmp1,dim=1),kind=BLAS_KIND),    &
                                    ev, int(ubound(ev,dim=1),kind=BLAS_KIND), &
                                    1.0_rk, qtmp2(1,1), int(ubound(qtmp2,dim=1),kind=BLAS_KIND))
#endif
                  call obj%timer%stop("mkl_offload")
                else
                  call obj%timer%start("gpublas")
                  call gpublas_PRECISION_GEMM('N', 'N', l_rnm, ncnt, nnzu,   &
                                      1.0_rk, qtmp1_dev, ubound(qtmp1,dim=1),    &
                                      ev_dev, ubound(ev,dim=1), &
                                      1.0_rk, qtmp2_dev, ubound(qtmp2,dim=1))
                  call obj%timer%stop("gpublas")
                endif
              else
                call obj%timer%start("blas")
                call obj%timer%start("gemm")
                call PRECISION_GEMM('N', 'N', int(l_rnm,kind=BLAS_KIND), int(ncnt,kind=BLAS_KIND), &
                                    int(nnzu,kind=BLAS_KIND),   &
                                    1.0_rk, qtmp1, int(ubound(qtmp1,dim=1),kind=BLAS_KIND),    &
                                    ev, int(ubound(ev,dim=1),kind=BLAS_KIND), &
                                    1.0_rk, qtmp2(1,1), int(ubound(qtmp2,dim=1),kind=BLAS_KIND))
                call obj%timer%stop("gemm")
                call obj%timer%stop("blas")
              endif ! useGPU
            endif

            ! Compute eigenvectors of the rank-1 modified matrix.
            ! Parts for multiplying with lower half of Q:

            do i = 1, ncnt
              j = idx(idxq1(i+ns))
              ! Calculate the j-th eigenvector of the deflated system
              ! See above why we are doing it this way!
              tmp(1:nnzl) = d1l(1:nnzl)-dbase(j)
              call v_add_s_&
              &PRECISION&
              &(obj,tmp,nnzl,ddiff(j))
              ev(1:nnzl,i) = zl(1:nnzl) / tmp(1:nnzl) * ev_scale(j)
            enddo

            if (useGPU .and. .not.(useIntelGPU) ) then
              !TODO the previous loop could be possible to do on device and thus
              !copy less
              successGPU = gpu_memcpy(ev_dev, int(loc(ev(1,1)),kind=c_intptr_t), &
                                 gemm_dim_l * gemm_dim_m * size_of_datatype, gpuMemcpyHostToDevice)
              check_memcpy_gpu("merge_systems: ev_dev", successGPU)
            endif

            !if (useIntelGPU) then
            !  ! needed later      
            !endif

            ! Multiply old Q with eigenvectors (lower half)

            if (l_rows-l_rnm>0 .and. ncnt>0 .and. nnzl>0) then
              if (useGPU) then
                if (useIntelGPU) then
                  call obj%timer%start("mkl_offload")
#ifdef WITH_INTEL_GPU_VERSION
                  call mkl_offload_PRECISION_GEMM('N', 'N', int(l_rows-l_rnm,kind=BLAS_KIND), int(ncnt,kind=BLAS_KIND),  &
                                     int(nnzl,kind=BLAS_KIND),   &
                                     1.0_rk, qtmp1(l_rnm+1,1), int(ubound(qtmp1,dim=1),kind=BLAS_KIND),    &
                                     ev,  int(ubound(ev,dim=1),kind=BLAS_KIND),   &
                                     1.0_rk, qtmp2(l_rnm+1,1), int(ubound(qtmp2,dim=1),kind=BLAS_KIND))
#endif
                  call obj%timer%stop("mkl_offload")

                else
                  call obj%timer%start("gpublas")
                  call gpublas_PRECISION_GEMM('N', 'N', l_rows-l_rnm, ncnt, nnzl,   &
                                      1.0_rk, qtmp1_dev + l_rnm * size_of_datatype, ubound(qtmp1,dim=1),    &
                                      ev_dev, ubound(ev,dim=1), &
                                      1.0_rk, qtmp2_dev + l_rnm * size_of_datatype, ubound(qtmp2,dim=1))
                  call obj%timer%stop("gpublas")
                endif
              else
                call obj%timer%start("blas")
                call obj%timer%start("gemm")
                call PRECISION_GEMM('N', 'N', int(l_rows-l_rnm,kind=BLAS_KIND), int(ncnt,kind=BLAS_KIND),  &
                                     int(nnzl,kind=BLAS_KIND),   &
                                     1.0_rk, qtmp1(l_rnm+1,1), int(ubound(qtmp1,dim=1),kind=BLAS_KIND),    &
                                     ev,  int(ubound(ev,dim=1),kind=BLAS_KIND),   &
                                     1.0_rk, qtmp2(l_rnm+1,1), int(ubound(qtmp2,dim=1),kind=BLAS_KIND))
                call obj%timer%stop("gemm")
                call obj%timer%stop("blas")
              endif ! useGPU
            endif

            if (useGPU .and. .not.(useIntelGPU) ) then
              !TODO either copy only half of the matrix here, and get rid of the
              !previous copy or copy whole array here
              successGPU = gpu_memcpy(int(loc(qtmp2(1,1)),kind=c_intptr_t), qtmp2_dev, &
                                 gemm_dim_k * gemm_dim_m * size_of_datatype, gpuMemcpyDeviceToHost)
              check_memcpy_gpu("merge_systems: qtmp2_dev", successGPU)
            endif

            !if (useIntelGPU) then
            !  ! needed at a later time
            !endif


             ! Put partial result into (output) Q

            do i = 1, ncnt
              q(l_rqs:l_rqe,l_col_out(idxq1(i+ns))) = qtmp2(1:l_rows,i)
            enddo

          enddo   !ns = 0, nqcols1-1, max_strip ! strimining loop
        enddo    !do np = 1, npc_n

        if (useGPU .and. .not.(useIntelGPU) ) then
          successGPU = gpu_host_unregister(int(loc(qtmp1),kind=c_intptr_t))
          check_host_unregister_gpu("merge_systems: qtmp1", successGPU)

          successGPU = gpu_free(qtmp1_dev)
          check_dealloc_gpu("merge_systems: qtmp1_dev", successGPU)
          
          successGPU = gpu_host_unregister(int(loc(qtmp2),kind=c_intptr_t))
          check_host_unregister_gpu("merge_systems: qtmp2", successGPU)

          successGPU = gpu_free(qtmp2_dev)
          check_dealloc_gpu("merge_systems: qtmp2_dev", successGPU)

          successGPU = gpu_host_unregister(int(loc(ev),kind=c_intptr_t))
          check_host_unregister_gpu("merge_systems: ev", successGPU)

          successGPU = gpu_free(ev_dev)
          check_dealloc_gpu("merge_systems: ev_dev", successGPU)
        endif
        !if (useIntelGPU) then
        !  ! needed later
        !endif

        deallocate(ev, qtmp1, qtmp2, stat=istat, errmsg=errorMessage)
        check_deallocate("merge_systems: ev, qtmp1, qtmp2",istat, errorMessage)
      endif !very outer test (na1==1 .or. na1==2)
#ifdef WITH_OPENMP_TRADITIONAL
      deallocate(z_p, stat=istat, errmsg=errorMessage)
      check_deallocate("merge_systems: z_p",istat, errorMessage)
#endif

      call obj%timer%stop("merge_systems" // PRECISION_SUFFIX)

      return

    end subroutine merge_systems_&
    &PRECISION
