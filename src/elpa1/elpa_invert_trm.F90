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

#include "../general/sanity.F90"
#include "../general/error_checking.inc"

       use precision
       use elpa1_compute
       use elpa_utilities
       use elpa_mpi
       use elpa_abstract_impl
       use elpa_blas_interfaces

       implicit none
#include "../general/precision_kinds.F90"
       class(elpa_abstract_impl_t), intent(inout) :: obj
       integer(kind=ik)             :: na, matrixRows, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
#ifdef USE_ASSUMED_SIZE
       MATH_DATATYPE(kind=rck)     :: a(obj%local_nrows,*)
#else
       MATH_DATATYPE(kind=rck)     :: a(obj%local_nrows,obj%local_ncols)
#endif

       integer(kind=ik)             :: my_prow, my_pcol, np_rows, np_cols
       integer(kind=MPI_KIND)       :: mpierr, my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI
       integer(kind=ik)             :: l_cols, l_rows, l_col1, l_row1, l_colx, l_rowx
       integer(kind=ik)             :: n, nc, i, info, ns, nb
       integer(kind=BLAS_KIND)      :: infoBLAS
       MATH_DATATYPE(kind=rck), allocatable   :: tmp1(:), tmp2(:,:), tmat1(:,:), tmat2(:,:)
       logical                      :: wantDebug
       logical                      :: success
       integer(kind=ik)             :: istat, debug, error
       character(200)               :: errorMessage

      call obj%timer%start("elpa_invert_trm_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      &")

       na         = obj%na
       matrixRows = obj%local_nrows
       nblk       = obj%nblk
       matrixCols = obj%local_ncols

       call obj%get("mpi_comm_rows",mpi_comm_rows,error)
       if (error .ne. ELPA_OK) then
         print *,"Error getting option for mpi_comm_rows. Aborting..."
         stop
       endif
       call obj%get("mpi_comm_cols",mpi_comm_cols,error)
       if (error .ne. ELPA_OK) then
         print *,"Error getting option for mpi_comm_cols. Aborting..."
         stop
       endif

       call obj%get("debug", debug,error)
       if (error .ne. ELPA_OK) then
         print *,"Error getting option for debug. Aborting..."
         stop
       endif
       if (debug == 1) then
         wantDebug = .true.
       else
         wantDebug = .true.
       endif
       call obj%timer%start("mpi_communication")
       call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND), my_prowMPI, mpierr)
       call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
       call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND), my_pcolMPI, mpierr)
       call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)

       my_prow = int(my_prowMPI,kind=c_int)
       np_rows = int(np_rowsMPI,kind=c_int)
       my_pcol = int(my_pcolMPI,kind=c_int)
       np_cols = int(np_colsMPI,kind=c_int)
       call obj%timer%stop("mpi_communication")
       success = .true.

       l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a
       l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local cols of a

       allocate(tmp1(nblk*nblk), stat=istat, errmsg=errorMessage)
       check_allocate("elpa_invert_trm: tmp1", istat, errorMessage)

       allocate(tmp2(nblk,nblk), stat=istat, errmsg=errorMessage)
       check_allocate("elpa_invert_trm: tmp2", istat, errorMessage)

       tmp1 = 0
       tmp2 = 0

       allocate(tmat1(l_rows,nblk), stat=istat, errmsg=errorMessage)
       check_allocate("elpa_invert_trm: tmat1", istat, errorMessage)

       allocate(tmat2(nblk,l_cols), stat=istat, errmsg=errorMessage)
       check_allocate("elpa_invert_trm: tmat2", istat, errorMessage)

       tmat1 = 0
       tmat2 = 0


       ns = ((na-1)/nblk)*nblk + 1

       do n = ns,1,-nblk

         l_row1 = local_index(n, my_prow, np_rows, nblk, +1)
         l_col1 = local_index(n, my_pcol, np_cols, nblk, +1)

         nb = nblk
         if (na-n+1 < nblk) nb = na-n+1

         l_rowx = local_index(n+nb, my_prow, np_rows, nblk, +1)
         l_colx = local_index(n+nb, my_pcol, np_cols, nblk, +1)

         if (my_prow==prow(n, nblk, np_rows)) then

           if (my_pcol==pcol(n, nblk, np_cols)) then
             call obj%timer%start("blas")

             call PRECISION_TRTRI('U', 'N', int(nb,kind=BLAS_KIND), a(l_row1,l_col1), int(matrixRows,kind=BLAS_KIND), &
                                  infoBLAS)
             info = int(infoBLAS,kind=ik)
             call obj%timer%stop("blas")

             if (info/=0) then
               if (wantDebug) write(error_unit,*) "elpa_invert_trm_&
         &MATH_DATATYPE&

#if REALCASE == 1
         &: Error in DTRTRI"
#endif
#if COMPLEXCASE == 1
         &: Error in ZTRTRI"
#endif

               success = .false.
               call obj%timer%stop("elpa_invert_trm_&
               &MATH_DATATYPE&
               &_&
               &PRECISION&
               &")
               return
             endif

             nc = 0
             do i=1,nb
               tmp1(nc+1:nc+i) = a(l_row1:l_row1+i-1,l_col1+i-1)
               nc = nc+i
             enddo
           endif
#ifdef WITH_MPI
           call obj%timer%start("mpi_communication")
           call MPI_Bcast(tmp1, int(nb*(nb+1)/2,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,       &
                          int(pcol(n, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
           call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
           nc = 0
           do i=1,nb
             tmp2(1:i,i) = tmp1(nc+1:nc+i)
             nc = nc+i
           enddo

           call obj%timer%start("blas")
           if (l_cols-l_colx+1>0) &
           call PRECISION_TRMM('L', 'U', 'N', 'N', int(nb,kind=BLAS_KIND), int(l_cols-l_colx+1,kind=BLAS_KIND), ONE, &
                                   tmp2, int(ubound(tmp2,dim=1),kind=BLAS_KIND), a(l_row1,l_colx), int(matrixRows,kind=BLAS_KIND))
           call obj%timer%stop("blas")
           if (l_colx<=l_cols)   tmat2(1:nb,l_colx:l_cols) = a(l_row1:l_row1+nb-1,l_colx:l_cols)
           if (my_pcol==pcol(n, nblk, np_cols)) tmat2(1:nb,l_col1:l_col1+nb-1) = tmp2(1:nb,1:nb) ! tmp2 has the lower left triangle 0

         endif

         if (l_row1>1) then
           if (my_pcol==pcol(n, nblk, np_cols)) then
             tmat1(1:l_row1-1,1:nb) = a(1:l_row1-1,l_col1:l_col1+nb-1)
             a(1:l_row1-1,l_col1:l_col1+nb-1) = 0
           endif

           do i=1,nb
#ifdef WITH_MPI
             call obj%timer%start("mpi_communication")
             call MPI_Bcast(tmat1(1,i), int(l_row1-1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                            int(pcol(n, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)

             call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
           enddo
         endif
#ifdef WITH_MPI
         call obj%timer%start("mpi_communication")
         if (l_cols-l_col1+1>0) &
      call MPI_Bcast(tmat2(1,l_col1), int((l_cols-l_col1+1)*nblk,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                     int(prow(n, nblk, np_rows),kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)

          call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */

         call obj%timer%start("blas")
         if (l_row1>1 .and. l_cols-l_col1+1>0) &
           call PRECISION_GEMM('N', 'N', int(l_row1-1,kind=BLAS_KIND), int(l_cols-l_col1+1,kind=BLAS_KIND), &
                               int(nb,kind=BLAS_KIND), -ONE, &
                                tmat1, int(ubound(tmat1,dim=1),kind=BLAS_KIND), tmat2(1,l_col1), &
                                int(ubound(tmat2,dim=1),kind=BLAS_KIND), ONE, &
                                 a(1,l_col1), int(matrixRows,kind=BLAS_KIND) )

         call obj%timer%stop("blas")

       enddo

       deallocate(tmp1, tmp2, tmat1, tmat2, stat=istat, errmsg=errorMessage)
       check_deallocate("elpa_invert_trm: tmp1, tmp2, tmat1, tmat2", istat, errorMessage)

       call obj%timer%stop("elpa_invert_trm_&
       &MATH_DATATYPE&
       &_&
       &PRECISION&
       &")
