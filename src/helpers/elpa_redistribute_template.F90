!
!    Copyright 2019, A. Marek, MPCDF
!
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

#ifdef REDISTRIBUTE_MATRIX
   doRedistributeMatrix = .false.
   !call obj%set("internal_nblk",32)
   !call obj%set("matrix_order",1)

   if (obj%is_set("internal_nblk") == 1) then
     if (obj%is_set("matrix_order") == 1) then
       reDistributeMatrix = .true.
     else
       reDistributeMatrix = .false.
       if (my_pe == 0) then
         write(error_unit,*) 'Warning: Matrix re-distribution is not used, since you did not set the matrix_order'
         write(error_unit,*) '         Only task 0 prints this warning'
       endif
     endif
   endif
   if (reDistributeMatrix) then
     ! get the block-size to be used
     call obj%get("internal_nblk", nblkInternal, error)
     if (error .ne. ELPA_OK) then
       print *,"Problem getting option. Aborting..."
       stop 1
     endif

     call obj%get("matrix_order", matrixOrder, error)
     if (error .ne. ELPA_OK) then
       print *,"Problem getting option. Aborting..."
       stop 1
     endif
     layoutExternal=matrixLayouts(matrixOrder)

     if (nblkInternal == nblk) then
       doRedistributeMatrix = .false.
     else
       doRedistributeMatrix = .true.
     endif
   endif

   if (doRedistributeMatrix) then
     call obj%timer%start("redistribute")

     ! collect some necessary values
     ! and np_rows and np_cols
     call obj%get("mpi_comm_rows",mpi_comm_rows,error)
     call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
     call obj%get("mpi_comm_cols",mpi_comm_cols,error)
     call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)

     np_rows = int(np_rowsMPI,kind=c_int)
     np_cols = int(np_colsMPI,kind=c_int)


     ! store external variables
     np_rowsExt  = np_rows
     np_colsExt  = np_cols
     my_prowExt  = my_prow
     my_pcolExt  = my_prow

     na_rowsExt = matrixRows
     na_colsExt = matrixCols

     ! create a new internal blacs discriptor
     ! matrix will still be distributed over the same process grid
     ! as input matrices A and Q where
     !external_blacs_ctxt = int(mpi_comm_all,kind=BLAS_KIND)
     ! construct current descriptor
     if (obj%is_set("blacs_context") == 1) then
       call obj%get("blacs_context",external_blacs_ctxt, error)
       external_blacs_ctxtBLAS = int(external_blacs_ctxt,kind=BLAS_KIND)
     else
       ! we have to re-create the blacs context
       external_blacs_ctxtBLAS = int(mpi_comm_all,kind=BLAS_KIND)
       call BLACS_Gridinit(external_blacs_ctxtBLAS, layoutExternal, int(np_rows,kind=BLAS_KIND), int(np_cols,kind=BLAS_KIND))
     endif


     sc_desc(1) = 1
     sc_desc(2) = external_blacs_ctxtBLAS
     sc_desc(3) = int(obj%na,kind=BLAS_KIND)
     sc_desc(4) = int(obj%na,kind=BLAS_KIND)
     sc_desc(5) = int(obj%nblk,kind=BLAS_KIND)
     sc_desc(6) = int(obj%nblk,kind=BLAS_KIND)
     sc_desc(7) = 0
     sc_desc(8) = 0
     sc_desc(9) = int(obj%local_nrows,kind=BLAS_KIND)


     sc_descExt(1:9) = sc_desc(1:9)

     layoutInternal = 'C'

     blacs_ctxtInternal = int(mpi_comm_all,kind=BLAS_KIND)

     !if (layoutInternal /= layoutExternal) then
       ! we get new blacs context ; we want to keep the np_rows, and np_cols
       np_rowsInternal = int(np_rows,kind=BLAS_KIND)
       np_colsInternal = int(np_cols,kind=BLAS_KIND)

       ! blacs_ctxtInternal inout
       ! np_rows,cols in
       call BLACS_Gridinit(blacs_ctxtInternal, layoutInternal, int(np_rows,kind=BLAS_KIND), int(np_cols,kind=BLAS_KIND))


       call BLACS_Gridinfo(blacs_ctxtInternal, np_rowsInternal, np_colsInternal, my_prowInternal, my_pcolInternal)
       if (np_rows /= np_rowsInternal) then
         print *, "BLACS_Gridinfo returned different values for np_rows as set by BLACS_Gridinit"
         print *,np_rows,np_rowsInternal
         stop 1
       endif
       if (np_cols /= np_colsInternal) then
         print *, "BLACS_Gridinfo returned different values for np_cols as set by BLACS_Gridinit"
         print *,np_cols,np_colsInternal
         stop 1
       endif

       call mpi_comm_split(int(mpi_comm_all,kind=MPI_KIND), int(my_pcolInternal,kind=MPI_KIND), &
                           int(my_prowInternal,kind=MPI_KIND), mpi_comm_rowsMPIInternal, mpierr)
       mpi_comm_rowsInternal = int(mpi_comm_rowsMPIInternal,kind=c_int)

       call mpi_comm_split(int(mpi_comm_all,kind=MPI_KIND), int(my_prowInternal,kind=MPI_KIND), &
                           int(my_pcolInternal,kind=MPI_KIND), mpi_comm_colsMPIInternal, mpierr)
       mpi_comm_colsInternal = int(mpi_comm_colsMPIInternal,kind=c_int)

       if (int(np_rowsInternal,kind=c_int) /= np_rows) then
         print *, "BLACS_Gridinfo returned different values for np_rows as set by BLACS_Gridinit"
         stop 1
       endif
       if (int(np_colsInternal,kind=c_int) /= np_cols) then
         print *, "BLACS_Gridinfo returned different values for np_cols as set by BLACS_Gridinit"
         stop 1
       endif
     !else
     !  na_rowsInternal = int(na_rows,kind=BLAS_KIND)
     !  na_colsInternal = int(na_cols,kind=BLAS_KIND)
     !  sc_desc_ 
     !endif
     
     ! now we can set up the the blacs descriptor

     !sc_desc_(:) = 0
     na_rowsInternal = numroc(int(na,kind=BLAS_KIND), int(nblkInternal,kind=BLAS_KIND), my_prowInternal, &
                              0_BLAS_KIND, np_rowsInternal)
     na_colsInternal = numroc(int(na,kind=BLAS_KIND), int(nblkInternal,kind=BLAS_KIND), my_pcolInternal, &
                              0_BLAS_KIND, np_colsInternal)

     info_ = 0
     call descinit(sc_descInternal, int(na,kind=BLAS_KIND), int(na,kind=BLAS_KIND), int(nblkInternal,kind=BLAS_KIND), &
                   int(nblkInternal,kind=BLAS_KIND), 0_BLAS_KIND, 0_BLAS_KIND, &
                   blacs_ctxtInternal, na_rowsInternal, info_)

     if (info_ .ne. 0) then
       write(error_unit,*) 'Error in BLACS descinit! info=',info_
       write(error_unit,*) 'the internal re-distribution of the matrices failed!'
       call MPI_ABORT(int(mpi_comm_all,kind=MPI_KIND), 1_MPI_KIND, mpierr)
     endif

     if (useGPU) then
       ! allocate internal GPU device arrays
       num = (na_rowsInternal * na_colsInternal ) * size_of_datatype
       successGPU = gpu_malloc(a_devIntern, num)
       check_alloc_gpu("redistribute a_devIntern", successGPU)

       if (isSkewsymmetric) then
         num = (na_rowsInternal * 2*na_colsInternal ) * size_of_datatype
       else
         num = (na_rowsInternal * na_colsInternal ) * size_of_datatype
       endif
       successGPU = gpu_malloc(q_devIntern, num)
       check_alloc_gpu("redistribute q_devIntern", successGPU)

#ifdef DEVICE_POINTER
       ! data is provided via {a|q|ev}DevExtern

       ! allocate dummy HOST arrays and copy
       allocate(aExtern(1:matrixRows,1:matrixCols), stat=istat, errmsg=errorMessage)
       check_allocate("redistribute: aExtern", istat, errorMessage)

       if (present(qDevExtern)) then
         if (isSkewsymmetric) then
           allocate(qExtern(1:matrixRows,1:2*matrixCols), stat=istat, errmsg=errorMessage)
         else
           allocate(qExtern(1:matrixRows,1:matrixCols), stat=istat, errmsg=errorMessage)
         endif
         check_allocate("redistribute: qExtern", istat, errorMessage)
       endif

       num = (matrixRows*matrixCols) * size_of_datatype
       successGPU = gpu_memcpy(int(loc(aExtern(1,1)),kind=c_intptr_t), aDevExtern, &
                               num, gpuMemcpyDeviceToHost)
       check_memcpy_gpu("redistribute aDevExtern -> aExtern", successGPU)

       if (present(qDevExtern)) then
         if (isSkewsymmetric) then
           num = (matrixRows*2*matrixCols) * size_of_datatype
         else
           num = (matrixRows*matrixCols) * size_of_datatype
         endif
         successGPU = gpu_memcpy(int(loc(qExtern(1,1)),kind=c_intptr_t), qDevExtern, &
                                 num, gpuMemcpyDeviceToHost)
         check_memcpy_gpu("redistribute qDevExtern -> qExtern", successGPU)
       endif
#else
       ! data is provided via host arrays {a|q|ev|}Extern
#endif
     endif ! useGPU
             

     ! allocate the memory for the redistributed matrices
     allocate(aIntern(na_rowsInternal,na_colsInternal))

     if (isSkewsymmetric) then
       allocate(qIntern(na_rowsInternal,2*na_colsInternal))
     else
       allocate(qIntern(na_rowsInternal,na_colsInternal))
     endif

     call obj%timer%start("GEMR2D")
     call scal_PRECISION_GEMR2D &
     (int(na,kind=BLAS_KIND), int(na,kind=BLAS_KIND), aExtern, 1_BLAS_KIND, 1_BLAS_KIND, sc_descExt, aIntern, &
     1_BLAS_KIND, 1_BLAS_KIND, sc_descInternal, blacs_ctxtInternal)
     call obj%timer%stop("GEMR2D")

     !map all important new variables to be used from here
     nblk                    = nblkInternal
     matrixCols              = na_colsInternal
     matrixRows              = na_rowsInternal
     mpi_comm_cols           = mpi_comm_colsInternal
     mpi_comm_rows           = mpi_comm_rowsInternal

!#ifndef DEVICE_POINTER
!     a => aIntern(1:matrixRows,1:matrixCols)
!     q => qIntern(1:matrixRows,1:matrixCols)
!#endif

     if (useGPU) then
       num = (matrixRows*matrixCols) * size_of_datatype
       successGPU = gpu_memcpy(a_devIntern, int(loc(aIntern(1,1)),kind=c_intptr_t), &
                               num, gpuMemcpyHostToDevice)
       check_memcpy_gpu("redistribute aIntern -> a_devIntern", successGPU)

       if (isSkewsymmetric) then
         num = (matrixRows*2*matrixCols) * size_of_datatype
       else
         num = (matrixRows*matrixCols) * size_of_datatype
       endif
       successGPU = gpu_memcpy(q_devIntern, int(loc(qIntern(1,1)),kind=c_intptr_t), &
                               num, gpuMemcpyHostToDevice)
       check_memcpy_gpu("redistribute qIntern -> q_devIntern", successGPU)

#ifdef DEVICE_POINTER
       if (present(qDevExtern)) then
         deallocate(qExtern)
       endif
       deallocate(aExtern)
       deallocate(aIntern, qIntern)
#else
       deallocate(aIntern, qIntern)
#endif
     endif ! useGPU


     call obj%timer%stop("redistribute")
   !else
   !  print *,"not redistributing"
   !  a => aExtern(1:matrixRows,1:matrixCols)
   !  q => qExtern(1:matrixRows,1:matrixCols)
   !  print *,"in redistribute", associated(a), associated(q)
   endif

#endif /* REDISTRIBUTE_MATRIX */

