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
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
#endif

#include "../general/sanity.F90"

    subroutine trans_ev_band_to_full_&
    &MATH_DATATYPE&
    &_&
    &PRECISION &
    (obj, na, nqc, nblk, nbw, a_mat, a_dev, lda, tmat, tmat_dev, q_mat, &
     q_dev, ldq, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols, useGPU &
#if REALCASE == 1
     ,useQr)
#endif
#if COMPLEXCASE == 1
     )
#endif

    !-------------------------------------------------------------------------------
    !  trans_ev_band_to_full_real/complex:
    !  Transforms the eigenvectors of a band matrix back to the eigenvectors of the original matrix
    !
    !  Parameters
    !
    !  na          Order of matrix a_mat, number of rows of matrix q_mat
    !
    !  nqc         Number of columns of matrix q_mat
    !
    !  nblk        blocksize of cyclic distribution, must be the same in both directions!
    !
    !  nbw         semi bandwith
    !
    !  a_mat(lda,matrixCols)    Matrix containing the Householder vectors (i.e. matrix a_mat after bandred_real/complex)
    !              Distribution is like in Scalapack.
    !
    !  lda         Leading dimension of a_mat
    !  matrixCols  local columns of matrix a_mat and q_mat
    !
    !  tmat(nbw,nbw,numBlocks) Factors returned by bandred_real/complex
    !
    !  q_mat           On input: Eigenvectors of band matrix
    !              On output: Transformed eigenvectors
    !              Distribution is like in Scalapack.
    !
    !  ldq         Leading dimension of q_mat
    !
    !  mpi_comm_rows
    !  mpi_comm_cols
    !              MPI-Communicators for rows/columns
    !
    !-------------------------------------------------------------------------------
      use precision
      use cuda_functions
      use iso_c_binding
      use elpa_abstract_impl
      use elpa_blas_interfaces

      implicit none
#include "../general/precision_kinds.F90"
      class(elpa_abstract_impl_t), intent(inout) :: obj
      logical, intent(in)                    :: useGPU
#if REALCASE == 1
     logical, intent(in)                     :: useQR
#endif
      integer(kind=ik)                       :: na, nqc, lda, ldq, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols
#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=rck)               :: a_mat(lda,*)
      MATH_DATATYPE(kind=rck)               :: q_mat(ldq,*), tmat(nbw,nbw,*)
#else
      MATH_DATATYPE(kind=rck)               :: a_mat(lda,matrixCols)
      MATH_DATATYPE(kind=rck)               :: q_mat(ldq,matrixCols), tmat(nbw, nbw, numBlocks)
#endif
      integer(kind=C_intptr_T)               :: a_dev ! passed from bandred_real at the moment not used since copied in bandred_real

      integer(kind=ik)                       :: my_prow, my_pcol, np_rows, np_cols
      integer(kind=MPI_KIND)                 :: my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI, mpierr
      integer(kind=ik)                       :: max_blocks_row, max_blocks_col, max_local_rows, &
                                                max_local_cols
      integer(kind=ik)                       :: l_cols, l_rows, l_colh, n_cols
      integer(kind=ik)                       :: istep, lc, ncol, nrow, nb, ns

      MATH_DATATYPE(kind=rck), allocatable   :: hvb(:)
      MATH_DATATYPE(kind=rck), allocatable   ::  tmp1(:), tmp2(:), hvm(:,:)
      ! hvm_dev is fist used and set in this routine
      ! q_mat is changed in trans_ev_tridi on the host, copied to device and passed here. this can be adapted
      ! tmp_dev is first used in this routine
      ! tmat_dev is passed along from bandred_real
      integer(kind=C_intptr_T)               :: hvm_dev, q_dev, tmp_dev, tmat_dev

      integer(kind=ik)                       :: i

#ifdef BAND_TO_FULL_BLOCKING
      MATH_DATATYPE(kind=rck), allocatable   :: tmat_complete(:,:), t_tmp(:,:), t_tmp2(:,:)
      integer(kind=ik)                       :: cwy_blocking, t_blocking, t_cols, t_rows
#endif

      integer(kind=ik)                       :: istat
      character(200)                         :: errorMessage
      character(20)                          :: gpuString
      logical                                :: successCUDA
      integer(kind=c_intptr_t), parameter    :: size_of_datatype = size_of_&
                                                                   &PRECISION&
                                                                   &_&
                                                                   &MATH_DATATYPE
      integer                                :: blocking_factor, error

      if(useGPU) then
        gpuString = "_gpu"
      else
        gpuString = ""
      endif

      call obj%timer%start("trans_ev_band_to_full_&
      &MATH_DATATYPE&
      &" // &
      &PRECISION_SUFFIX //&
      gpuString)

#ifdef BAND_TO_FULL_BLOCKING
      call obj%get("blocking_in_band_to_full",blocking_factor,error)
      if (error .ne. ELPA_OK) then
        print *,"Problem getting option. Aborting..."
        stop
      endif
#endif
      call obj%timer%start("mpi_communication")

      call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND) ,my_prowMPI ,mpierr)
      call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND) ,np_rowsMPI ,mpierr)
      call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND) ,my_pcolMPI ,mpierr)
      call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND) ,np_colsMPI ,mpierr)

      my_prow = int(my_prowMPI,kind=c_int)
      my_pcol = int(my_pcolMPI,kind=c_int)
      np_rows = int(np_rowsMPI,kind=c_int)
      np_cols = int(np_colsMPI,kind=c_int)
      call obj%timer%stop("mpi_communication")

      max_blocks_row = ((na -1)/nblk)/np_rows + 1  ! Rows of a_mat
      max_blocks_col = ((nqc-1)/nblk)/np_cols + 1  ! Columns of q_mat!

      max_local_rows = max_blocks_row*nblk
      max_local_cols = max_blocks_col*nblk

      if (useGPU) then

#if REALCASE == 1
        ! here the GPU and CPU version diverged: the CPU version now always uses the useQR path which
        ! is not implemented in the GPU version
#endif

        ! the GPU version does not (yet) support blocking
        ! but the handling is the same for real/complex case

        allocate(tmp1(max_local_cols*nbw), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating tmp1 "//errorMessage
          stop 1
        endif

        allocate(tmp2(max_local_cols*nbw), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                   &MATH_DATATYPE&
                   &: error when allocating tmp2 "//errorMessage
          stop 1
        endif

        allocate(hvb(max_local_rows*nbw), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating hvb "//errorMessage
          stop 1
        endif

        allocate(hvm(max_local_rows,nbw), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating hvm "//errorMessage
          stop 1
        endif

        successCUDA = cuda_malloc(hvm_dev, (max_local_rows)*nbw* size_of_datatype)
        if (.not.(successCUDA)) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error in cudaMalloc"
          stop 1
        endif

        successCUDA = cuda_malloc(tmp_dev, (max_local_cols)*nbw* size_of_datatype)
        if (.not.(successCUDA)) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error in cudaMalloc"
          stop 1
        endif

!#ifdef WITH_MPI
!! it should be possible to keep tmat dev on the device and not copy it around
!! already existent on GPU
!        successCUDA = cuda_malloc(tmat_dev, nbw*nbw* &
!#if REALCASE == 1
!  size_of_PRECISION_real)
!#endif
!#if COMPLEXCASE == 1
!        size_of_PRECISION_complex)
!#endif
!
!        if (.not.(successCUDA)) then
!          print *,"trans_ev_band_to_full_&
!    &MATH_DATATYPE&
!    &: error in cudaMalloc"
!          stop 1
!        endif
!#endif

#if REALCASE == 1
! q_dev already living on device
!        successCUDA = cuda_malloc(q_dev, ldq*matrixCols*size_of_datatype)
!        if (.not.(successCUDA)) then
!          print *,"trans_ev_band_to_full_real: error in cudaMalloc"
!          stop 1
!        endif
  !      q_temp(:,:) = 0.0
  !      q_temp(1:ldq,1:na_cols) = q_mat(1:ldq,1:na_cols)

!        ! copy q_dev to device, maybe this can be avoided if q_dev can be kept on device in trans_ev_tridi_to_band
!        successCUDA = cuda_memcpy(q_dev, c_loc(q_mat), (ldq)*(matrixCols)*size_of_PRECISION_real, cudaMemcpyHostToDevice)
!        if (.not.(successCUDA)) then
!          print *,"trans_ev_band_to_full_real: error in cudaMalloc"
!          stop 1
!        endif
#endif
#if COMPLEXCASE == 1
!         successCUDA = cuda_malloc(q_dev, ldq*matrixCols*size_of_PRECISION_complex)
!         if (.not.(successCUDA)) then
!           print *,"trans_ev_band_to_full_complex: error in cudaMalloc"
!           stop 1
!         endif
!
!         successCUDA = cuda_memcpy(q_dev, c_loc(q_mat),ldq*matrixCols*size_of_PRECISION_complex, cudaMemcpyHostToDevice)
!          if (.not.(successCUDA)) then
!            print *,"trans_ev_band_to_full_complex: error in cudaMemcpy"
!            stop 1
!          endif
#endif

        ! if MPI is NOT used the following steps could be done on the GPU and memory transfers could be avoided
        successCUDA = cuda_memset(hvm_dev, 0, (max_local_rows)*(nbw)* size_of_datatype)
        if (.not.(successCUDA)) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error in cudaMalloc"
          stop 1
        endif

        hvm = 0.0_rck   ! Must be set to 0 !!!
        hvb = 0.0_rck   ! Safety only
        l_cols = local_index(nqc, my_pcol, np_cols, nblk, -1) ! Local columns of q_mat

        do istep=1,(na-1)/nbw

          n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step

          ! Broadcast all Householder vectors for current step compressed in hvb

          nb = 0
          ns = 0

          do lc = 1, n_cols
            ncol = istep*nbw + lc ! absolute column number of householder Vector
            nrow = ncol - nbw ! absolute number of pivot row

            l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast
            l_colh = local_index(ncol  , my_pcol, np_cols, nblk, -1) ! HV local column number

            if (my_pcol==pcol(ncol, nblk, np_cols)) hvb(nb+1:nb+l_rows) = a_mat(1:l_rows,l_colh)

            nb = nb+l_rows

            if (lc==n_cols .or. mod(ncol,nblk)==0) then
#ifdef WITH_MPI
              call obj%timer%start("mpi_communication")
              call MPI_Bcast(hvb(ns+1), int(nb-ns,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,&
                             int(pcol(ncol, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)

              call obj%timer%stop("mpi_communication")

#endif /* WITH_MPI */
              ns = nb
            endif
          enddo

          ! Expand compressed Householder vectors into matrix hvm

          nb = 0
          do lc = 1, n_cols
            nrow = (istep-1)*nbw+lc ! absolute number of pivot row
            l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast

            hvm(1:l_rows,lc) = hvb(nb+1:nb+l_rows)
            if (my_prow==prow(nrow, nblk, np_rows)) hvm(l_rows+1,lc) = 1.0_rck
            nb = nb+l_rows
          enddo

          successCUDA = cuda_memcpy(hvm_dev, int(loc(hvm),kind=c_intptr_t), &
                        max_local_rows*nbw* size_of_datatype, cudaMemcpyHostToDevice)

          if (.not.(successCUDA)) then
            print *,"trans_ev_band_to_full_real: error in cudaMemcpy, hvm"
            stop 1

          endif

          l_rows = local_index(MIN(na,(istep+1)*nbw), my_prow, np_rows, nblk, -1)

          ! Q = Q - V * T**T * V**T * Q

          if (l_rows>0) then
            call obj%timer%start("cublas")
            call cublas_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',        &
                                 n_cols, l_cols, l_rows, ONE, hvm_dev, max_local_rows, &
                                       q_dev, ldq , ZERO, tmp_dev, n_cols)
            call obj%timer%stop("cublas")

#ifdef WITH_MPI

            ! copy data from device to host for a later MPI_ALLREDUCE
            ! copy to host maybe this can be avoided this is needed if MPI is used (allreduce)
            successCUDA = cuda_memcpy(int(loc(tmp1),kind=c_intptr_t), &
                          tmp_dev, l_cols*n_cols*size_of_datatype, cudaMemcpyDeviceToHost)
            if (.not.(successCUDA)) then
              print *,"trans_ev_band_to_full_real: error in cudaMemcpy, tmp1 to host"
              stop 1
            endif


#else /* WITH_MPI */
            ! in real case no copy needed. Don't do it in complex case neither
#endif /* WITH_MPI */

          else ! l_rows>0
            tmp1(1:l_cols*n_cols) = 0.0_rck
          endif ! l_rows>0

#ifdef WITH_MPI
          call obj%timer%start("mpi_communication")
          call mpi_allreduce(tmp1, tmp2, int(n_cols*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                             MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
          call obj%timer%stop("mpi_communication")

#else /* WITH_MPI */
!          tmp2(1:n_cols*l_cols) = tmp1(1:n_cols*l_cols)
#endif /* WITH_MPI */

          if (l_rows>0) then
#ifdef WITH_MPI
            ! after the mpi_allreduce we have to copy back to the device
            ! copy back to device
            successCUDA = cuda_memcpy(tmp_dev, int(loc(tmp2),kind=c_intptr_t), &
                          n_cols*l_cols* size_of_datatype, &
              cudaMemcpyHostToDevice)
            if (.not.(successCUDA)) then
              print *,"trans_ev_band_to_full_&
                      &MATH_DATATYPE&
                      &: error in cudaMemcpy, tmp2"
              stop 1
            endif
#else /* WITH_MPI */
            ! in real case no memcopy needed. Don't do it in complex case neither
#endif /* WITH_MPI */

!#ifdef WITH_MPI
           ! IMPORTANT: even though tmat_dev is transfered from the previous rutine, we have to copy from tmat again
           ! tmat is 3-dimensional array, while tmat_dev contains only one 2-dimensional slice of it - and here we
           ! need to upload another slice
           successCUDA = cuda_memcpy(tmat_dev, int(loc(tmat(1,1,istep)),kind=c_intptr_t), &
                         nbw*nbw*size_of_datatype, cudaMemcpyHostToDevice)

           if (.not.(successCUDA)) then
             print *,"trans_ev_band_to_full_&
                     &MATH_DATATYPE&
                     &: error in cudaMemcpy, tmat"
             stop 1
           endif
!#endif /* WITH_MPI */

            call obj%timer%start("cublas")
            call cublas_PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N',       &
                                        n_cols, l_cols, ONE, tmat_dev, nbw, tmp_dev, n_cols)

            call cublas_PRECISION_GEMM('N', 'N', l_rows, l_cols, n_cols, -ONE, hvm_dev, max_local_rows, &
                                       tmp_dev, n_cols, one, q_dev, ldq)
            call obj%timer%stop("cublas")

            ! copy to host maybe this can be avoided
            ! this is not necessary hvm is not used anymore
            successCUDA = cuda_memcpy(int(loc(hvm),kind=c_intptr_t), &
                          hvm_dev, ((max_local_rows)*nbw*size_of_datatype),cudaMemcpyDeviceToHost)
            if (.not.(successCUDA)) then
              print *,"trans_ev_band_to_full_real: error in cudaMemcpy hvm to host"
              stop 1
            endif
          endif ! l_rows > 0

        enddo ! istep



      else ! do not useGPU

#ifdef BAND_TO_FULL_BLOCKING
        ! t_blocking was formerly 2; 3 is a better choice
        t_blocking = blocking_factor ! number of matrices T (tmat) which are aggregated into a new (larger) T matrix (tmat_complete) and applied at once

        ! we only use the t_blocking if we could call it fully, this is might be better but needs to benchmarked.
!       if ( na >= ((t_blocking+1)*nbw) ) then
        cwy_blocking = t_blocking * nbw

        allocate(tmp1(max_local_cols*cwy_blocking), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating tmp1 "//errorMessage
          stop 1
        endif

        allocate(tmp2(max_local_cols*cwy_blocking), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating tmp2 "//errorMessage
          stop 1
        endif

        allocate(hvb(max_local_rows*cwy_blocking), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating hvb "//errorMessage
          stop 1
        endif

        allocate(hvm(max_local_rows,cwy_blocking), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating hvm "//errorMessage
          stop 1
        endif

#else /* BAND_TO_FULL_BLOCKING */

        allocate(tmp1(max_local_cols*nbw), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating tmp1 "//errorMessage
          stop 1
        endif

        allocate(tmp2(max_local_cols*nbw), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&: error when allocating tmp2 "//errorMessage
          stop 1
        endif

        allocate(hvb(max_local_rows*nbw), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating hvb "//errorMessage
          stop 1
        endif

        allocate(hvm(max_local_rows,nbw), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating hvm "//errorMessage
          stop 1
        endif
#endif /* BAND_TO_FULL_BLOCKING */

#ifdef BAND_TO_FULL_BLOCKING
        allocate(tmat_complete(cwy_blocking,cwy_blocking), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                   &MATH_DATATYPE&
                   &: error when allocating tmat_complete "//errorMessage
          stop 1
        endif
        allocate(t_tmp(cwy_blocking,nbw), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating t_tmp "//errorMessage
          stop 1
        endif
        allocate(t_tmp2(cwy_blocking,nbw), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating t_tmp2 "//errorMessage
          stop 1
        endif
#endif
!        else
!          allocate(tmp1(max_local_cols*nbw))
!          allocate(tmp2(max_local_cols*nbw))
!          allocate(hvb(max_local_rows*nbw))
!          allocate(hvm(max_local_rows,nbw))
!        endif

        hvm = 0.0_rck   ! Must be set to 0 !!!
        hvb = 0.0_rck   ! Safety only
        l_cols = local_index(nqc, my_pcol, np_cols, nblk, -1) ! Local columns of q_mat

!       if ( na >= ((t_blocking+1)*nbw) ) then

#ifdef BAND_TO_FULL_BLOCKING
        do istep=1,((na-1)/nbw-1)/t_blocking + 1
#else
        do istep=1,(na-1)/nbw
#endif

#ifdef BAND_TO_FULL_BLOCKING
          ! This the call when using  na >= ((t_blocking+1)*nbw)
          !      n_cols = MIN(na,istep*cwy_blocking+nbw) - (istep-1)*cwy_blocking - nbw
          ! Number of columns in current step
          ! As an alternative we add some special case handling if na < cwy_blocking
          IF (na < cwy_blocking) THEN
            n_cols = MAX(0, na-nbw)
            IF ( n_cols .eq. 0 ) THEN
              EXIT
            END IF
          ELSE
            n_cols = MIN(na,istep*cwy_blocking+nbw) - (istep-1)*cwy_blocking - nbw ! Number of columns in current step
          END IF
#else /* BAND_TO_FULL_BLOCKING */
          n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step
#endif /* BAND_TO_FULL_BLOCKING */
          ! Broadcast all Householder vectors for current step compressed in hvb

          nb = 0
          ns = 0

          do lc = 1, n_cols
#ifdef BAND_TO_FULL_BLOCKING
            ncol = (istep-1)*cwy_blocking + nbw + lc ! absolute column number of householder Vector
#else
            ncol = istep*nbw + lc ! absolute column number of householder Vector
#endif
            nrow = ncol - nbw ! absolute number of pivot row

            l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast
            l_colh = local_index(ncol  , my_pcol, np_cols, nblk, -1) ! HV local column number

            if (my_pcol==pcol(ncol, nblk, np_cols)) hvb(nb+1:nb+l_rows) = a_mat(1:l_rows,l_colh)

            nb = nb+l_rows

            if (lc==n_cols .or. mod(ncol,nblk)==0) then
#ifdef WITH_MPI
              call obj%timer%start("mpi_communication")
              call MPI_Bcast(hvb(ns+1), int(nb-ns,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,    &
                             int(pcol(ncol, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)

              call obj%timer%stop("mpi_communication")

#endif /* WITH_MPI */
              ns = nb
            endif
          enddo ! lc

          ! Expand compressed Householder vectors into matrix hvm

          nb = 0
          do lc = 1, n_cols
#ifdef BAND_TO_FULL_BLOCKING
            nrow = (istep-1)*cwy_blocking + lc ! absolute number of pivot row
#else
            nrow = (istep-1)*nbw+lc ! absolute number of pivot row
#endif
            l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast

            hvm(1:l_rows,lc) = hvb(nb+1:nb+l_rows)
            if (my_prow==prow(nrow, nblk, np_rows)) hvm(l_rows+1,lc) = 1.0_rck
            nb = nb+l_rows
          enddo

#ifdef BAND_TO_FULL_BLOCKING
          l_rows = local_index(MIN(na,(istep+1)*cwy_blocking), my_prow, np_rows, nblk, -1)

          ! compute tmat2 out of tmat(:,:,)
          tmat_complete = 0
          do i = 1, t_blocking
            t_cols = MIN(nbw, n_cols - (i-1)*nbw)
            if (t_cols <= 0) exit
            t_rows = (i - 1) * nbw
            tmat_complete(t_rows+1:t_rows+t_cols,t_rows+1:t_rows+t_cols) = tmat(1:t_cols,1:t_cols,(istep-1)*t_blocking + i)

            if (i > 1) then
              call obj%timer%start("blas")
              call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',      &
                                  int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), &
                                  ONE, hvm(1,1), int(max_local_rows,kind=BLAS_KIND), hvm(1,(i-1)*nbw+1), &
                                  int(max_local_rows,kind=BLAS_KIND), ZERO, t_tmp, int(cwy_blocking,kind=BLAS_KIND) )

              call obj%timer%stop("blas")
#ifdef WITH_MPI
              call obj%timer%start("mpi_communication")

              call mpi_allreduce(t_tmp, t_tmp2, int(cwy_blocking*nbw,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,    &
                                 MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
              call obj%timer%stop("mpi_communication")
              call obj%timer%start("blas")
              call PRECISION_TRMM('L', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), &
                                  ONE, tmat_complete, int(cwy_blocking,kind=BLAS_KIND), t_tmp2, &
                                  int(cwy_blocking,kind=BLAS_KIND) )
              call PRECISION_TRMM('R', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), &
                                  -ONE, tmat_complete(t_rows+1,t_rows+1), int(cwy_blocking,kind=BLAS_KIND), &
                                  t_tmp2, int(cwy_blocking,kind=BLAS_KIND))
              call obj%timer%stop("blas")

              tmat_complete(1:t_rows,t_rows+1:t_rows+t_cols) = t_tmp2(1:t_rows,1:t_cols)

#else /* WITH_MPI */
!              t_tmp2(1:cwy_blocking,1:nbw) = t_tmp(1:cwy_blocking,1:nbw)
              call obj%timer%start("blas")
              call PRECISION_TRMM('L', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), &
                                   ONE, tmat_complete, int(cwy_blocking,kind=BLAS_KIND), t_tmp, &
                                   int(cwy_blocking,kind=BLAS_KIND))
              call PRECISION_TRMM('R', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), &
                                  -ONE, tmat_complete(t_rows+1,t_rows+1), int(cwy_blocking,kind=BLAS_KIND), &
                                  t_tmp, int(cwy_blocking,kind=BLAS_KIND))
              call obj%timer%stop("blas")

              tmat_complete(1:t_rows,t_rows+1:t_rows+t_cols) = t_tmp(1:t_rows,1:t_cols)

#endif /* WITH_MPI */

!              call PRECISION_TRMM('L', 'U', 'N', 'N', t_rows, t_cols, ONE, tmat_complete, cwy_blocking, t_tmp2, cwy_blocking)
!              call PRECISION_TRMM('R', 'U', 'N', 'N', t_rows, t_cols, -ONE, tmat_complete(t_rows+1,t_rows+1), cwy_blocking, &
!                         t_tmp2, cwy_blocking)

!              tmat_complete(1:t_rows,t_rows+1:t_rows+t_cols) = t_tmp2(1:t_rows,1:t_cols)
             endif
          enddo
#else /* BAND_TO_FULL_BLOCKING */
          l_rows = local_index(MIN(na,(istep+1)*nbw), my_prow, np_rows, nblk, -1)
#endif

          ! Q = Q - V * T**T * V**T * Q

          if (l_rows>0) then
            call obj%timer%start("blas")

            call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',         &
                                int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), &
                                ONE, hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), &
                                q_mat, int(ldq,kind=BLAS_KIND), ZERO, tmp1, int(n_cols,kind=BLAS_KIND))
            call obj%timer%stop("blas")

          else ! l_rows>0

            tmp1(1:l_cols*n_cols) = 0.0_rck
          endif ! l_rows>0

#ifdef WITH_MPI
          call obj%timer%start("mpi_communication")
          call mpi_allreduce(tmp1, tmp2, int(n_cols*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                             int(mpi_comm_rows,kind=MPI_KIND) ,mpierr)
          call obj%timer%stop("mpi_communication")

          call obj%timer%start("blas")

          if (l_rows>0) then
#ifdef BAND_TO_FULL_BLOCKING

            call PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N',        &
                                int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), ONE, tmat_complete, &
                                int(cwy_blocking,kind=BLAS_KIND), tmp2, int(n_cols,kind=BLAS_KIND))
            call PRECISION_GEMM('N', 'N', int(l_rows,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), &
                                -ONE, hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), tmp2, int(n_cols,kind=BLAS_KIND), &
                                 ONE, q_mat, int(ldq,kind=BLAS_KIND))

#else /* BAND_TO_FULL_BLOCKING */

            call PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N',        &
                                int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), ONE, tmat(1,1,istep), &
                                int(ubound(tmat,dim=1),kind=BLAS_KIND), tmp2, int(n_cols,kind=BLAS_KIND))
            call PRECISION_GEMM('N', 'N', int(l_rows,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), &
                                int(n_cols,kind=BLAS_KIND), -ONE, hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), &
                                tmp2, int(n_cols,kind=BLAS_KIND), ONE, q_mat, int(ldq,kind=BLAS_KIND))

#endif /* BAND_TO_FULL_BLOCKING */

          endif
          call obj%timer%stop("blas")
#else /* WITH_MPI */
!          tmp2 = tmp1
          call obj%timer%start("blas")
          if (l_rows>0) then
#ifdef BAND_TO_FULL_BLOCKING
            call PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N',        &
                                int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), ONE, tmat_complete, &
                                int(cwy_blocking,kind=BLAS_KIND), tmp1, int(n_cols,kind=BLAS_KIND))
            call PRECISION_GEMM('N', 'N', int(l_rows,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), &
                                int(n_cols,kind=BLAS_KIND), -ONE, hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), &
                                tmp1, int(n_cols,kind=BLAS_KIND), ONE, q_mat, int(ldq,kind=BLAS_KIND))
#else /* BAND_TO_FULL_BLOCKING */

            call PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N',        &
                                int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), ONE, tmat(1,1,istep), &
                                int(ubound(tmat,dim=1),kind=BLAS_KIND), tmp1, int(n_cols,kind=BLAS_KIND))
            call PRECISION_GEMM('N', 'N', int(l_rows,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), &
                                int(n_cols,kind=BLAS_KIND), -ONE, hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), &
                                tmp1, int(n_cols,kind=BLAS_KIND), ONE, q_mat, int(ldq,kind=BLAS_KIND))

#endif  /* BAND_TO_FULL_BLOCKING */
          endif
          call obj%timer%stop("blas")
#endif /* WITH_MPI */

!          if (l_rows>0) then
!            call PRECISION_TRMM('L', 'U', 'T', 'N', n_cols, l_cols, ONE, tmat_complete, cwy_blocking, tmp2, n_cols)
!            call PRECISION_GEMM('N', 'N', l_rows, l_cols, n_cols, -ONE, hvm, ubound(hvm,dim=1), tmp2, n_cols, ONE, q_mat, ldq)
!          endif

        enddo ! istep

      endif ! useGPU

      deallocate(tmp1, tmp2, hvb, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_band_to_full_&
                 &MATH_DATATYPE&
                 &: error when deallocating tmp1 tmp2 hvb "//errorMessage
        stop 1
      endif

      if (useGPU) then
        successCUDA = cuda_free(hvm_dev)
        if (.not.(successCUDA)) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error in cudaFree"
          stop 1
        endif

        successCUDA = cuda_free(tmp_dev)
        if (.not.(successCUDA)) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error in cudaFree"
          stop 1
        endif


         ! final transfer of q_dev
         successCUDA = cuda_memcpy(int(loc(q_mat),kind=c_intptr_t), q_dev, ldq*matrixCols* size_of_datatype, &
                       cudaMemcpyDeviceToHost)

         if (.not.(successCUDA)) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error in cudamemcpu q_dev"
          stop 1
         endif

         !   q_mat(1:ldq,1:na_cols) = q_temp(1:ldq,1:na_cols)

         successCUDA = cuda_free(q_dev)
         if (.not.(successCUDA)) then
           print *,"trans_ev_band_to_full_&
                   &MATH_DATATYPE&
                   &: error in cudaFree"
           stop 1
         endif

         !   deallocate(q_temp, stat=istat, errmsg=errorMessage)
         !   if (istat .ne. 0) then
         !     print *,"error when deallocating q_temp "//errorMessage
         !     stop 1
         !   endif
         !   deallocate(tmat_temp, stat=istat, errmsg=errorMessage)
         !   if (istat .ne. 0) then
         !     print *,"trans_ev_band_to_full_real: error when deallocating tmat_temp "//errorMessage
         !     stop 1
         !   endif

      endif ! useGPU

      deallocate(hvm, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_band_to_full_&
                &MATH_DATATYPE&
                &: error when deallocating hvm "//errorMessage
        stop 1
      endif

#if BAND_TO_FULL_BLOCKING
      if (.not.(useGPU)) then
        deallocate(tmat_complete, t_tmp, t_tmp2, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when deallocating tmat_complete, t_tmp, t_tmp2 "//errorMessage
          stop 1
        endif
      endif
#endif

      call obj%timer%stop("trans_ev_band_to_full_&
      &MATH_DATATYPE&
      &" // &
      &PRECISION_SUFFIX //&
      gpuString)

    end subroutine trans_ev_band_to_full_&
    &MATH_DATATYPE&
    &_&
    &PRECISION


