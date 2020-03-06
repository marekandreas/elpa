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
#ifndef ALREADY_DEFINED
  subroutine local_size_offset_1d(n,nb,baseidx,idx,rev,rank,nprocs, &
                                lsize,baseoffset,offset)

    use precision
    use ELPA1_compute

    implicit none

    ! input
    integer(kind=ik) :: n,nb,baseidx,idx,rev,rank,nprocs

    ! output
    integer(kind=ik) :: lsize,baseoffset,offset

    ! local scalars
    integer(kind=ik) :: rank_idx

    rank_idx = MOD((idx-1)/nb,nprocs)

    ! calculate local size and offsets
    if (rev .eq. 1) then
        if (idx > 0) then
            lsize = local_index(idx,rank,nprocs,nb,-1)
        else
            lsize = 0
        end if

        baseoffset = 1
        offset = 1
    else
        offset = local_index(idx,rank,nprocs,nb,1)
        baseoffset = local_index(baseidx,rank,nprocs,nb,1)

        lsize = local_index(n,rank,nprocs,nb,-1)
        !print *,'baseidx,idx',baseidx,idx,lsize,n

        lsize = lsize - offset + 1

        baseoffset = offset - baseoffset + 1
    end if

end subroutine local_size_offset_1d
#endif

subroutine reverse_vector_local_&
           &PRECISION &
     (n,x,incx,work,lwork)
    use precision
    implicit none
#include "../../general/precision_kinds.F90"

    ! input
    integer(kind=ik)              :: incx, n, lwork
    real(kind=C_DATATYPE_KIND)    :: x(*), work(*)

    ! local scalars
    real(kind=C_DATATYPE_KIND)    :: temp
    integer(kind=ik)              :: srcoffset, destoffset, ientry

    if (lwork .eq. -1) then
        work(1) = 0.0_rk
        return
    end if

    do ientry=1,n/2
        srcoffset=1+(ientry-1)*incx
        destoffset=1+(n-ientry)*incx

        temp = x(srcoffset)
        x(srcoffset) = x(destoffset)
        x(destoffset) = temp
    end do

end subroutine

subroutine reverse_matrix_local_&
           &PRECISION &
     (trans, m, n, a, lda, work, lwork)
    use precision
    implicit none

    ! input
    integer(kind=ik)              :: lda,m,n,lwork,trans
    real(kind=C_DATATYPE_KIND)    :: a(lda,*),work(*)

    ! local scalars
    real(kind=C_DATATYPE_KIND)    :: dworksize(1)
    integer(kind=ik)              :: incx
    integer(kind=ik)              :: dimsize
    integer(kind=ik)              :: i

    if (trans .eq. 1) then
        incx = lda
        dimsize = n
    else
        incx = 1
        dimsize = m
    end if

    if (lwork .eq. -1) then
        call reverse_vector_local_&
       &PRECISION &
       (dimsize, a, incx, dworksize, -1)
        work(1) = dworksize(1)
        return
    end if

    if (trans .eq. 1) then
        do i=1,m
            call reverse_vector_local_&
      &PRECISION &
      (dimsize, a(i,1), incx, work, lwork)
        end do
    else
        do i=1,n
            call reverse_vector_local_&
      &PRECISION &
      (dimsize, a(1,i), incx, work, lwork)
        end do
    end if

end subroutine

subroutine reverse_matrix_2dcomm_ref_&
           &PRECISION &
     (m, n, mb, nb, a, lda, work, lwork, mpicomm_cols, mpicomm_rows)
    use precision
    implicit none

    ! input
    integer(kind=ik)              :: m, n, lda, lwork, mpicomm_cols, mpicomm_rows, mb, nb
    real(kind=C_DATATYPE_KIND)    :: a(lda,*),work(*)

    ! local scalars
    real(kind=C_DATATYPE_KIND)    :: reverse_column_size(1)
    real(kind=C_DATATYPE_KIND)    :: reverse_row_size(1)

    integer(kind=ik)              :: mpirank_cols, mpirank_rows
    integer(kind=ik)              :: mpiprocs_cols, mpiprocs_rows
    integer(kind=MPI_KIND)        :: mpirank_colsMPI, mpirank_rowsMPI
    integer(kind=MPI_KIND)        :: mpiprocs_colsMPI, mpiprocs_rowsMPI
    integer(kind=MPI_KIND)        :: mpierr
    integer(kind=ik)              :: lrows, lcols, offset, baseoffset

    call MPI_Comm_rank(int(mpicomm_cols,kind=MPI_KIND) ,mpirank_colsMPI,  mpierr)
    call MPI_Comm_rank(int(mpicomm_rows,kind=MPI_KIND) ,mpirank_rowsMPI,  mpierr)
    call MPI_Comm_size(int(mpicomm_cols,kind=MPI_KIND) ,mpiprocs_colsMPI, mpierr)
    call MPI_Comm_size(int(mpicomm_rows,kind=MPI_KIND) ,mpiprocs_rowsMPI, mpierr)

    mpirank_cols = int(mpirank_colsMPI,kind=c_int)
    mpirank_rows = int(mpirank_rowsMPI,kind=c_int)
    mpiprocs_cols = int(mpiprocs_colsMPI,kind=c_int)
    mpiprocs_rows = int(mpiprocs_rowsMPI,kind=c_int)

    call local_size_offset_1d(m,mb,1,1,0,mpirank_cols,mpiprocs_cols, &
                                  lrows,baseoffset,offset)

    call local_size_offset_1d(n,nb,1,1,0,mpirank_rows,mpiprocs_rows, &
                                  lcols,baseoffset,offset)

    if (lwork .eq. -1) then
        call reverse_matrix_1dcomm_&
  &PRECISION &
  (0,m,lcols,mb,a,lda,reverse_column_size,-1,mpicomm_cols)
        call reverse_matrix_1dcomm_&
  &PRECISION &
  (1,lrows,n,nb,a,lda,reverse_row_size,-1,mpicomm_rows)
        work(1) = max(reverse_column_size(1),reverse_row_size(1))
        return
    end if

    call reverse_matrix_1dcomm_&
    &PRECISION &
    (0,m,lcols,mb,a,lda,work,lwork,mpicomm_cols)
    call reverse_matrix_1dcomm_&
    &PRECISION &
    (1,lrows,n,nb,a,lda,work,lwork,mpicomm_rows)
end subroutine

! b: if trans = 'N': b is size of block distribution between rows
! b: if trans = 'T': b is size of block distribution between columns
subroutine reverse_matrix_1dcomm_&
           &PRECISION &
     (trans, m, n, b, a, lda, work, lwork, mpicomm)
    use precision
    use elpa_mpi

    implicit none

    ! input
    integer(kind=ik)              :: trans
    integer(kind=ik)              :: m, n, b, lda, lwork, mpicomm
    real(kind=C_DATATYPE_KIND)    :: a(lda,*), work(*)

    ! local scalars
    integer(kind=ik)              :: mpirank, mpiprocs
    integer(kind=MPI_KIND)        :: mpirankMPI, mpiprocsMPI, mpierr
#ifdef WITH_MPI
    integer(kind=ik)              :: my_mpistatus(MPI_STATUS_SIZE)
#endif
    integer(kind=ik)              :: nr_blocks,dest_process,src_process,step
    integer(kind=ik)              :: lsize,baseoffset,offset
    integer(kind=ik)              :: current_index,destblk,srcblk,icol,next_index
    integer(kind=ik)              :: sendcount,recvcount
    integer(kind=ik)              :: sendoffset,recvoffset
    integer(kind=ik)              :: newmatrix_offset,work_offset
    integer(kind=ik)              :: lcols,lrows,lroffset,lcoffset,dimsize,fixedsize
    real(kind=C_DATATYPE_KIND)    :: dworksize(1)

    call MPI_Comm_rank(int(mpicomm,kind=MPI_KIND), mpirankMPI, mpierr)
    call MPI_Comm_size(int(mpicomm,kind=MPI_KIND), mpiprocsMPI, mpierr)

    mpirank = int(mpirankMPI,kind=c_int)
    mpiprocs = int(mpiprocsMPI,kind=c_int)

    if (trans .eq. 1) then
        call local_size_offset_1d(n,b,1,1,0,mpirank,mpiprocs, &
                                  lcols,baseoffset,lcoffset)
        lrows = m
    else
        call local_size_offset_1d(m,b,1,1,0,mpirank,mpiprocs, &
                                  lrows,baseoffset,lroffset)
        lcols = n
    end if

    if (lwork .eq. -1) then
        call reverse_matrix_local_&
  &PRECISION &
  (trans,lrows,lcols,a,max(lrows,lcols),dworksize,-1)
        work(1) = real(3*lrows*lcols,kind=REAL_DATATYPE) + dworksize(1)
        return
    end if

    sendoffset = 1
    recvoffset = sendoffset + lrows*lcols
    newmatrix_offset = recvoffset + lrows*lcols
    work_offset = newmatrix_offset + lrows*lcols

    if (trans .eq. 1) then
        dimsize = n
        fixedsize = m
    else
        dimsize = m
        fixedsize = n
    end if

    if (dimsize .le. 1) then
        return ! nothing to do
    end if

    ! 1. adjust step size to remainder size
    nr_blocks = dimsize / b
    nr_blocks = nr_blocks * b
    step = dimsize - nr_blocks
    if (step .eq. 0) step = b

    ! 2. iterate over destination blocks starting with process 0
    current_index = 1
    do while (current_index .le. dimsize)
        destblk = (current_index-1) / b
        dest_process = mod(destblk,mpiprocs)
        srcblk = (dimsize-current_index) / b
        src_process = mod(srcblk,mpiprocs)

        next_index = current_index+step

        ! block for dest_process is located on mpirank if lsize > 0
        call local_size_offset_1d(dimsize-current_index+1,b,dimsize-next_index+2,dimsize-next_index+2,0, &
                                  src_process,mpiprocs,lsize,baseoffset,offset)

        sendcount = lsize*fixedsize
        recvcount = sendcount

        ! TODO: this send/recv stuff seems to blow up on BlueGene/P
        ! TODO: is there actually room for the requested matrix part? the target
        ! process might not have any parts at all (thus no room)
        if ((src_process .eq. mpirank) .and. (dest_process .eq. src_process)) then
                ! 5. pack data
                if (trans .eq. 1) then
                    do icol=offset,offset+lsize-1
                        work(sendoffset+(icol-offset)*lrows:sendoffset+(icol-offset+1)*lrows-1) = &
                            a(1:lrows,icol)
                    end do
                else
                    do icol=1,lcols
                        work(sendoffset+(icol-1)*lsize:sendoffset+icol*lsize-1) = &
                            a(offset:offset+lsize-1,icol)
                    end do
                end if

                ! 7. reverse data
                if (trans .eq. 1) then
                    call reverse_matrix_local_&
        &PRECISION &
        (1,lrows,lsize,work(sendoffset),lrows,work(work_offset),lwork)
                else
                    call reverse_matrix_local_&
        &PRECISION &
        (0,lsize,lcols,work(sendoffset),lsize,work(work_offset),lwork)
                end if

                ! 8. store in temp matrix
                if (trans .eq. 1) then
                    do icol=1,lsize
                        work(newmatrix_offset+(icol-1)*lrows:newmatrix_offset+icol*lrows-1) = &
                            work(sendoffset+(icol-1)*lrows:sendoffset+icol*lrows-1)
                    end do

                    newmatrix_offset = newmatrix_offset + lsize*lrows
                else
                    do icol=1,lcols
                        work(newmatrix_offset+(icol-1)*lrows:newmatrix_offset+(icol-1)*lrows+lsize-1) = &
                            work(sendoffset+(icol-1)*lsize:sendoffset+icol*lsize-1)
                    end do

                    newmatrix_offset = newmatrix_offset + lsize
                end if
        else

            if (dest_process .eq. mpirank) then
                ! 6b. call MPI_Recv

#ifdef WITH_MPI
                call MPI_Recv(work(recvoffset), int(recvcount,kind=MPI_KIND), MPI_REAL_PRECISION, &
                              int(src_process,kind=MPI_KIND), int(current_index,kind=MPI_KIND),   &
                              int(mpicomm,kind=MPI_KIND), my_mpistatus, mpierr)

#else /* WITH_MPI */
                work(recvoffset:recvoffset+recvcount-1) = work(sendoffset:sendoffset+sendcount-1)
#endif /* WITH_MPI */

                ! 7. reverse data
                if (trans .eq. 1) then
                    call reverse_matrix_local_&
        &PRECISION &
        (1,lrows,lsize,work(recvoffset),lrows,work(work_offset),lwork)
                else
                    call reverse_matrix_local_&
        &PRECISION &
        (0,lsize,lcols,work(recvoffset),lsize,work(work_offset),lwork)
                end if

                ! 8. store in temp matrix
                if (trans .eq. 1) then
                    do icol=1,lsize
                        work(newmatrix_offset+(icol-1)*lrows:newmatrix_offset+icol*lrows-1) = &
                            work(recvoffset+(icol-1)*lrows:recvoffset+icol*lrows-1)
                    end do

                    newmatrix_offset = newmatrix_offset + lsize*lrows
                else
                    do icol=1,lcols
                        work(newmatrix_offset+(icol-1)*lrows:newmatrix_offset+(icol-1)*lrows+lsize-1) = &
                            work(recvoffset+(icol-1)*lsize:recvoffset+icol*lsize-1)
                    end do

                    newmatrix_offset = newmatrix_offset + lsize
                end if
            end if

            if (src_process .eq. mpirank) then
                ! 5. pack data
                if (trans .eq. 1) then
                    do icol=offset,offset+lsize-1
                        work(sendoffset+(icol-offset)*lrows:sendoffset+(icol-offset+1)*lrows-1) = &
                            a(1:lrows,icol)
                    end do
                else
                    do icol=1,lcols
                        work(sendoffset+(icol-1)*lsize:sendoffset+icol*lsize-1) = &
                            a(offset:offset+lsize-1,icol)
                    end do
                end if

                ! 6a. call MPI_Send
#ifdef WITH_MPI
                call MPI_Send(work(sendoffset), int(sendcount,kind=MPI_KIND), MPI_REAL_PRECISION, &
                              int(dest_process,kind=MPI_KIND), int(current_index,kind=MPI_KIND),  &
                              int(mpicomm,kind=MPI_KIND), mpierr)
#endif /* WITH_MPI */
            end if
        end if

        current_index = next_index
    end do

   ! 9. copy temp matrix to real matrix
   newmatrix_offset = recvoffset + lrows*lcols
   do icol=1,lcols
        a(1:lrows,icol) = &
            work(newmatrix_offset+(icol-1)*lrows:newmatrix_offset+icol*lrows-1)
   end do
end subroutine
