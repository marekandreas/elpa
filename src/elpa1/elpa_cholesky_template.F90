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

#include "../general/sanity.F90"
#include "../general/error_checking.inc"
     use elpa1_compute
     use elpa_utilities
     use elpa_mpi
     use precision
     use elpa_abstract_impl
     use elpa_omp
     use elpa_blas_interfaces

     implicit none
#include "../general/precision_kinds.F90"
      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=ik)              :: na, matrixRows, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=rck)      :: a(obj%local_nrows,*)
#else
      MATH_DATATYPE(kind=rck)      :: a(obj%local_nrows,obj%local_ncols)
#endif
      integer(kind=ik)              :: my_prow, my_pcol, np_rows, np_cols
      integer(kind=MPI_KIND)        :: mpierr, my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI
      integer(kind=ik)              :: l_cols, l_rows, l_col1, l_row1, l_colx, l_rowx
      integer(kind=ik)              :: n, nc, i, info
      integer(kind=BLAS_KIND)       :: infoBLAS
      integer(kind=ik)              :: lcs, lce, lrs, lre
      integer(kind=ik)              :: tile_size, l_rows_tile, l_cols_tile

      MATH_DATATYPE(kind=rck), allocatable    :: tmp1(:), tmp2(:,:), tmatr(:,:), tmatc(:,:)
      logical                       :: wantDebug
      logical                       :: success
      integer(kind=ik)              :: istat, debug, error
      character(200)                :: errorMessage
      integer(kind=ik)              :: nrThreads

      call obj%timer%start("elpa_cholesky_&
      &MATH_DATATYPE&
      &_&
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
      nrThreads=1
#endif

      na         = obj%na
      matrixRows = obj%local_nrows
      nblk       = obj%nblk
      matrixCols = obj%local_ncols

      call obj%get("mpi_comm_rows",mpi_comm_rows,error )
      if (error .ne. ELPA_OK) then
        print *,"Problem getting option for mpi_comm_rows. Aborting..."
        stop
      endif
      call obj%get("mpi_comm_cols",mpi_comm_cols,error)
      if (error .ne. ELPA_OK) then
        print *,"Problem getting option for mpi_comm_cols. Aborting..."
        stop
      endif

      call obj%get("debug",debug,error)
      if (error .ne. ELPA_OK) then
        print *,"Problem getting option for debug settings. Aborting..."
        stop
      endif
      if (debug == 1) then
        wantDebug = .true.
      else
        wantDebug = .false.
      endif

      call obj%timer%start("mpi_communication")
      call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND), my_prowMPI, mpierr)
      call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
      call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND), my_pcolMPI, mpierr)
      call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)

      my_prow = int(my_prowMPI, kind=c_int)
      np_rows = int(np_rowsMPI, kind=c_int)
      my_pcol = int(my_pcolMPI, kind=c_int)
      np_cols = int(np_colsMPI, kind=c_int)
      call obj%timer%stop("mpi_communication")
      success = .true.

      ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

      tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size
      tile_size = ((128*max(np_rows,np_cols)-1)/tile_size+1)*tile_size ! make local tiles at least 128 wide

      l_rows_tile = tile_size/np_rows ! local rows of a tile
      l_cols_tile = tile_size/np_cols ! local cols of a tile

      l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a
      l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local cols of a

      allocate(tmp1(nblk*nblk), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_cholesky: tmp1", istat, errorMessage)

      allocate(tmp2(nblk,nblk), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_cholesky: tmp2", istat, errorMessage)

      tmp1 = 0
      tmp2 = 0

      allocate(tmatr(l_rows,nblk), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_cholesky: tmatr", istat, errorMessage)

      allocate(tmatc(l_cols,nblk), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_cholesky: tmatc", istat, errorMessage)

      tmatr = 0
      tmatc = 0

      do n = 1, na, nblk
        ! Calculate first local row and column of the still remaining matrix
        ! on the local processor

        l_row1 = local_index(n, my_prow, np_rows, nblk, +1)
        l_col1 = local_index(n, my_pcol, np_cols, nblk, +1)

        l_rowx = local_index(n+nblk, my_prow, np_rows, nblk, +1)
        l_colx = local_index(n+nblk, my_pcol, np_cols, nblk, +1)

        if (n+nblk > na) then

          ! This is the last step, just do a Cholesky-Factorization
          ! of the remaining block

          if (my_prow==prow(n, nblk, np_rows) .and. my_pcol==pcol(n, nblk, np_cols)) then
            call obj%timer%start("blas")

            call PRECISION_POTRF('U', int(na-n+1,kind=BLAS_KIND), a(l_row1,l_col1), &
                                 int(matrixRows,kind=BLAS_KIND), infoBLAS )
            info = int(infoBLAS,kind=ik)
            call obj%timer%stop("blas")

            if (info/=0) then
              if (wantDebug) write(error_unit,*) "elpa_cholesky_&
        &MATH_DATATYPE&

#if REALCASE == 1
        &: Error in dpotrf: ",info
#endif
#if COMPLEXCASE == 1
              &: Error in zpotrf: ",info
#endif
              success = .false.
              return
            endif

          endif

          exit ! Loop

        endif

        if (my_prow==prow(n, nblk, np_rows)) then

          if (my_pcol==pcol(n, nblk, np_cols)) then

            ! The process owning the upper left remaining block does the
            ! Cholesky-Factorization of this block
            call obj%timer%start("blas")

            call PRECISION_POTRF('U', int(nblk,kind=BLAS_KIND), a(l_row1,l_col1), &
                                 int(matrixRows,kind=BLAS_KIND) , infoBLAS )
            info = int(infoBLAS,kind=ik)
            call obj%timer%stop("blas")

            if (info/=0) then
              if (wantDebug) write(error_unit,*) "elpa_cholesky_&
        &MATH_DATATYPE&

#if REALCASE == 1
        &: Error in dpotrf 2: ",info
#endif
#if COMPLEXCASE == 1
        &: Error in zpotrf 2: ",info

#endif
              success = .false.
              return
            endif

            nc = 0
            do i=1,nblk
              tmp1(nc+1:nc+i) = a(l_row1:l_row1+i-1,l_col1+i-1)
              nc = nc+i
            enddo
          endif
#ifdef WITH_MPI
          call obj%timer%start("mpi_communication")

          call MPI_Bcast(tmp1, int(nblk*(nblk+1)/2,kind=MPI_KIND),      &
#if REALCASE == 1
                         MPI_REAL_PRECISION,         &
#endif
#if COMPLEXCASE == 1
                         MPI_COMPLEX_PRECISION,      &
#endif
                         int(pcol(n, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)

          call obj%timer%stop("mpi_communication")

#endif /* WITH_MPI */
          nc = 0
          do i=1,nblk
            tmp2(1:i,i) = tmp1(nc+1:nc+i)
            nc = nc+i
          enddo

          call obj%timer%start("blas")
          if (l_cols-l_colx+1>0) &
              call PRECISION_TRSM('L', 'U', BLAS_TRANS_OR_CONJ, 'N', int(nblk,kind=BLAS_KIND),  &
                                  int(l_cols-l_colx+1,kind=BLAS_KIND), ONE, tmp2, &
                                  int(ubound(tmp2,dim=1),kind=BLAS_KIND), a(l_row1,l_colx), int(matrixRows,kind=BLAS_KIND) )
          call obj%timer%stop("blas")
        endif

        do i=1,nblk

#if REALCASE == 1
          if (my_prow==prow(n, nblk, np_rows)) tmatc(l_colx:l_cols,i) = a(l_row1+i-1,l_colx:l_cols)
#endif
#if COMPLEXCASE == 1
          if (my_prow==prow(n, nblk, np_rows)) tmatc(l_colx:l_cols,i) = conjg(a(l_row1+i-1,l_colx:l_cols))
#endif

#ifdef WITH_MPI

          call obj%timer%start("mpi_communication")
          if (l_cols-l_colx+1>0) &
            call MPI_Bcast(tmatc(l_colx,i), int(l_cols-l_colx+1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                           int(prow(n, nblk, np_rows),kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)

          call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
        enddo
        ! this has to be checked since it was changed substantially when doing type safe
        call elpa_transpose_vectors_&
  &MATH_DATATYPE&
  &_&
  &PRECISION &
                 (obj, tmatc, ubound(tmatc,dim=1), mpi_comm_cols, &
                                      tmatr, ubound(tmatr,dim=1), mpi_comm_rows, &
                                      n, na, nblk, nblk, nrThreads)

        do i=0,(na-1)/tile_size
          lcs = max(l_colx,i*l_cols_tile+1)
          lce = min(l_cols,(i+1)*l_cols_tile)
          lrs = l_rowx
          lre = min(l_rows,(i+1)*l_rows_tile)
          if (lce<lcs .or. lre<lrs) cycle
          call obj%timer%start("blas")
          call PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, int(lre-lrs+1,kind=BLAS_KIND), int(lce-lcs+1,kind=BLAS_KIND), &
                              int(nblk,kind=BLAS_KIND), -ONE,  &
                              tmatr(lrs,1), int(ubound(tmatr,dim=1),kind=BLAS_KIND), tmatc(lcs,1), &
                              int(ubound(tmatc,dim=1),kind=BLAS_KIND), &
                              ONE, a(lrs,lcs), int(matrixRows,kind=BLAS_KIND))
          call obj%timer%stop("blas")

        enddo

      enddo

      deallocate(tmp1, tmp2, tmatr, tmatc, stat=istat, errmsg=errorMessage)
      check_deallocate("elpa_cholesky: tmp1, tmp2, tmatr, tmatc", istat, errorMessage)

      ! Set the lower triangle to 0, it contains garbage (form the above matrix multiplications)

      do i=1,na
        if (my_pcol==pcol(i, nblk, np_cols)) then
          ! column i is on local processor
          l_col1 = local_index(i  , my_pcol, np_cols, nblk, +1) ! local column number
          l_row1 = local_index(i+1, my_prow, np_rows, nblk, +1) ! first row below diagonal
          a(l_row1:l_rows,l_col1) = 0
        endif
      enddo

      ! restore original OpenMP settings
#ifdef WITH_OPENMP
      ! store the number of OpenMP threads used in the calling function
      ! restore this at the end of ELPA 2
      call omp_set_num_threads(omp_threads_caller)
#endif
      
      call obj%timer%stop("elpa_cholesky_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      &")
