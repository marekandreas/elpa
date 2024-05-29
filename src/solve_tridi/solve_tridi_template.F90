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
subroutine solve_tridi_gpu_&
&PRECISION_AND_SUFFIX &
#else
subroutine solve_tridi_cpu_&
&PRECISION_AND_SUFFIX &
#endif
    ( obj, na, nev, &
#ifdef SOLVE_TRIDI_GPU_BUILD
      d_dev, e_dev, q_dev, &
#else
      d, e, q, &
#endif
      ldq, nblk, matrixCols, mpi_comm_all, mpi_comm_rows, &
                                           mpi_comm_cols, wantDebug, success, max_threads )

      use precision
      use elpa_abstract_impl
      use merge_recursive
      use merge_systems
      use elpa_mpi
      use ELPA_utilities
      use distribute_global_column
      use elpa_mpi
      use elpa_gpu
      use elpa_gpu_util
      implicit none
#include "../../src/general/precision_kinds.F90"
      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=ik), intent(in)               :: na, nev, ldq, nblk, matrixCols, &
                                                    mpi_comm_all, mpi_comm_rows, mpi_comm_cols

      integer(kind=c_intptr_t)                   :: d_dev, e_dev, q_dev
#ifndef SOLVE_TRIDI_GPU_BUILD
      real(kind=REAL_DATATYPE), intent(inout)    :: d(na), e(na)
#ifdef USE_ASSUMED_SIZE
      real(kind=REAL_DATATYPE), intent(inout)    :: q(ldq,*)
#else
      real(kind=REAL_DATATYPE), intent(inout)    :: q(ldq,matrixCols)
#endif
#else /* SOLVE_TRIDI_GPU_BUILD */
      real(kind=REAL_DATATYPE)                   :: d(na), e(na)
      real(kind=REAL_DATATYPE)                   :: q(ldq,matrixCols)
#endif /* SOLVE_TRIDI_GPU_BUILD */

      logical, intent(in)                        :: wantDebug
      logical, intent(out)                       :: success

      integer(kind=ik)                           :: i, j, n, np, nc, nev1, l_cols, l_rows
      integer(kind=ik)                           :: my_prow, my_pcol, np_rows, np_cols
      integer(kind=MPI_KIND)                     :: mpierr, my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI
      integer(kind=ik), allocatable              :: limits(:), l_col(:), p_col(:), l_col_bc(:), p_col_bc(:)

      integer(kind=ik)                           :: istat
      character(200)                             :: errorMessage
      character(20)                              :: gpuString
      integer(kind=ik), intent(in)               :: max_threads
      logical                                    :: useGPU
      integer(kind=c_intptr_t)                   :: num
      integer(kind=c_intptr_t), parameter        :: size_of_datatype = size_of_&
                                                                      &PRECISION&
                                                                      &_&
                                                                      &MATH_DATATYPE
      integer(kind=c_intptr_t), parameter        :: size_of_datatype_real = size_of_&
                                                                      &PRECISION&
                                                                      &_real
      integer(kind=c_intptr_t)                   :: gpuHandle, my_stream
      logical                                    :: successGPU

      useGPU = .false.
#ifdef SOLVE_TRIDI_GPU_BUILD
      useGPU = .true.
#endif

      if(useGPU) then
        gpuString = "_gpu"
      else
        gpuString = ""
      endif

      call obj%timer%start("solve_tridi" // PRECISION_SUFFIX // gpuString)

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
      if (useGPU) then
        ! dirty hack
        num = na * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        call gpu_memcpy_async_and_stream_synchronize &
            ("solve_tridi d_dev -> d", d_dev, 0_c_intptr_t, &
                                                 d(1:na), &
                                  1, num, gpuMemcpyDeviceToHost, my_stream, .false., .false., .false.)
#else
        successGPU = gpu_memcpy(int(loc(d(1)),kind=c_intptr_t),  d_dev, &
                              num, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("solve_tridi: 1: d_dev", successGPU)
#endif
        num = na * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        call gpu_memcpy_async_and_stream_synchronize &
            ("solve_tridi e_dev -> e", e_dev, 0_c_intptr_t, &
                                                 e(1:na), &
                                 1, num, gpuMemcpyDeviceToHost, my_stream, .false., .false., .false.)
#else
       successGPU = gpu_memcpy(int(loc(e(1)),kind=c_intptr_t),  e_dev, &
                              num, gpuMemcpyDeviceToHost)
       check_memcpy_gpu("solve_tridi: e_dev", successGPU)
#endif
        if (.not.(obj%eigenvalues_only)) then
          num = ldq*matrixCols * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
            ("solve_tridi q_dev -> q_vec", q_dev, 0_c_intptr_t, &
                                                 q(1:ldq,1:matrixCols), &
                                 1, 1, num, gpuMemcpyDeviceToHost, my_stream, .false., .false., .false.)
#else
          successGPU = gpu_memcpy(int(loc(q(1,1)),kind=c_intptr_t),  q_dev, &
                              num, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("solve_tridi: q_dev", successGPU)
#endif
        endif ! eigenvalues_only
      endif



      success = .true.

      l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a and q
      l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of q

      !if (.not.(obj%eigenvalues_only)) then
        ! Set Q to 0
        q(1:l_rows, 1:l_cols) = 0.0_rk
      !endif

      ! Get the limits of the subdivisons, each subdivison has as many cols
      ! as fit on the respective processor column

      allocate(limits(0:np_cols), stat=istat, errmsg=errorMessage)
      check_allocate("solve_tridi: limits", istat, errorMessage)

      limits(0) = 0
      do np=0,np_cols-1
        nc = local_index(na, np, np_cols, nblk, -1) ! number of columns on proc column np

        ! Check for the case that a column has have zero width.
        ! This is not supported!
        ! Scalapack supports it but delivers no results for these columns,
        ! which is rather annoying
        if (nc==0) then
          call obj%timer%stop("solve_tridi" // PRECISION_SUFFIX)
          if (wantDebug) write(error_unit,*) 'ELPA1_solve_tridi: ERROR: Problem contains processor column with zero width'
          success = .false.
          return
        endif
        limits(np+1) = limits(np) + nc
      enddo

      ! Subdivide matrix by subtracting rank 1 modifications

      do i=1,np_cols-1
        n = limits(i)
        d(n) = d(n)-abs(e(n))
        d(n+1) = d(n+1)-abs(e(n))
      enddo

      ! Solve sub problems on processsor columns

      nc = limits(my_pcol) ! column after which my problem starts

      if (np_cols>1) then
        nev1 = l_cols ! all eigenvectors are needed
      else
        nev1 = MIN(nev,l_cols)
      endif
      call solve_tridi_col_&
           &PRECISION_AND_SUFFIX &
             (obj, l_cols, nev1, nc, d(nc+1), e(nc+1), q, ldq, nblk,  &
                        matrixCols, mpi_comm_rows, useGPU, wantDebug, success, max_threads)
      if (.not.(success)) then
        call obj%timer%stop("solve_tridi" // PRECISION_SUFFIX // gpuString)
        return
      endif
      ! If there is only 1 processor column, we are done

      if (np_cols==1) then
        deallocate(limits, stat=istat, errmsg=errorMessage)
        check_deallocate("solve_tridi: limits", istat, errorMessage)
        if (useGPU) then
          ! dirty hack
          num = na * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
            ("solve_trid d -> d_dev", d_dev, 0_c_intptr_t, &
                                                 d(1:na), &
                                  1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
          successGPU = gpu_memcpy(d_dev, int(loc(d(1)),kind=c_intptr_t),  &
                              num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("solve_tridi: d_dev", successGPU)
#endif
          num = na * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
            ("solve_tridi e_dev -> e", e_dev, 0_c_intptr_t, &
                                                 e(1:na), &
                                 1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
         successGPU = gpu_memcpy(e_dev, int(loc(e(1)),kind=c_intptr_t),  &
                              num, gpuMemcpyHostToDevice)
         check_memcpy_gpu("solve_tridi: e_dev", successGPU)
#endif
         if (.not.(obj%eigenvalues_only)) then
           num = ldq*matrixCols * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
           my_stream = obj%gpu_setup%my_stream
           call gpu_memcpy_async_and_stream_synchronize &
            ("solve_tride q_dev -> q_vec", q_dev, 0_c_intptr_t, &
                                                 q(1:ldq,1:matrixCols), &
                                 1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
           successGPU = gpu_memcpy(q_dev, int(loc(q(1,1)),kind=c_intptr_t),  &
                              num, gpuMemcpyHostToDevice)
           check_memcpy_gpu("solve_tridi: q_dev", successGPU)
#endif
          endif ! eigenvalues_only
        endif

        call obj%timer%stop("solve_tridi" // PRECISION_SUFFIX // gpuString)
        return
      endif

      ! Set index arrays for Q columns

      ! Dense distribution scheme:

      allocate(l_col(na), stat=istat, errmsg=errorMessage)
      check_allocate("solve_tridi: l_col", istat, errorMessage)

      allocate(p_col(na), stat=istat, errmsg=errorMessage)
      check_allocate("solve_tridi: p_col", istat, errorMessage)

      n = 0
      do np=0,np_cols-1
        nc = local_index(na, np, np_cols, nblk, -1)
        do i=1,nc
          n = n+1
          l_col(n) = i
          p_col(n) = np
        enddo
      enddo

      ! Block cyclic distribution scheme, only nev columns are set:

      allocate(l_col_bc(na), stat=istat, errmsg=errorMessage)
      check_allocate("solve_tridi: l_col_bc", istat, errorMessage)

      allocate(p_col_bc(na), stat=istat, errmsg=errorMessage)
      check_allocate("solve_tridi: p_col_bc", istat, errorMessage)

      p_col_bc(:) = -1
      l_col_bc(:) = -1

      do i = 0, na-1, nblk*np_cols
        do j = 0, np_cols-1
          do n = 1, nblk
            if (i+j*nblk+n <= MIN(nev,na)) then
              p_col_bc(i+j*nblk+n) = j
              l_col_bc(i+j*nblk+n) = i/np_cols + n
             endif
           enddo
         enddo
      enddo

      ! Recursively merge sub problems
      call merge_recursive_&
           &PRECISION &
           (obj, 0, np_cols, ldq, matrixCols, nblk, &
           l_col, p_col, l_col_bc, p_col_bc, limits, &
           np_cols, na, q, d, e, &
           mpi_comm_all, mpi_comm_rows, mpi_comm_cols,&
           useGPU, wantDebug, success, max_threads)

      if (.not.(success)) then
        call obj%timer%stop("solve_tridi" // PRECISION_SUFFIX // gpuString)
        return
      endif

      deallocate(limits,l_col,p_col,l_col_bc,p_col_bc, stat=istat, errmsg=errorMessage)
      check_deallocate("solve_tridi: limits, l_col, p_col, l_col_bc, p_col_bc", istat, errorMessage)


      if (useGPU) then
        ! dirty hack
        num = na * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        call gpu_memcpy_async_and_stream_synchronize &
            ("solve_trid d -> d_dev", d_dev, 0_c_intptr_t, &
                                                 d(1:na), &
                                  1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
        successGPU = gpu_memcpy(d_dev, int(loc(d(1)),kind=c_intptr_t),  &
                              num, gpuMemcpyHostToDevice)
        check_memcpy_gpu("solve_tridi: d_dev", successGPU)
#endif
        num = na * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        call gpu_memcpy_async_and_stream_synchronize &
            ("solve_tridi e_dev -> e", e_dev, 0_c_intptr_t, &
                                                 e(1:na), &
                                 1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
       successGPU = gpu_memcpy(e_dev, int(loc(e(1)),kind=c_intptr_t),  &
                              num, gpuMemcpyHostToDevice)
       check_memcpy_gpu("solve_tridi: e_dev", successGPU)
#endif
        if (.not.(obj%eigenvalues_only)) then
          num = ldq*matrixCols * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
            ("solve_tride q_dev -> q_vec", q_dev, 0_c_intptr_t, &
                                                 q(1:ldq,1:matrixCols), &
                                 1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
         successGPU = gpu_memcpy(q_dev, int(loc(q(1,1)),kind=c_intptr_t),  &
                              num, gpuMemcpyHostToDevice)
         check_memcpy_gpu("solve_tridi: q_dev", successGPU)
#endif
        endif ! eigenvalues_only
      endif

      call obj%timer%stop("solve_tridi" // PRECISION_SUFFIX // gpuString)
      return

    end 
