!
!    Copyright 2024, A. Marek, MPCDF
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


! redistribute back if necessary
   ! works for CPU and GPU
   if (doRedistributeMatrix) then

     if (layoutInternal /= layoutExternal) then
       ! maybe this can be skiped I now the process grid
       ! and np_rows and np_cols

       call obj%get("mpi_comm_rows",mpi_comm_rows,error)
       call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
       call obj%get("mpi_comm_cols",mpi_comm_cols,error)
       call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)

       np_rowsExt = int(np_rowsMPI,kind=c_int)
       np_colsExt = int(np_colsMPI,kind=c_int)

       ! we get new blacs context and the local process grid coordinates
       call BLACS_Gridinit(external_blacs_ctxt, layoutInternal, int(np_rows,kind=BLAS_KIND), int(np_cols,kind=BLAS_KIND))
       call BLACS_Gridinfo(int(external_blacs_ctxt,KIND=BLAS_KIND), np_rowsExt, &
                           np_colsExt, my_prowExt, my_pcolExt)

     endif
#ifdef DEVICE_POINTER
       if (present(qDevExtern)) then
         ! data is provided via {a|q|ev}DevExtern
     
         ! allocate dummy HOST arrays and copy
         allocate(qIntern(1:matrixRows,1:matrixCols), stat=istat, errmsg=errorMessage)
         check_allocate("redistribute: qIntern", istat, errorMessage)
   
         allocate(qExtern(1:na_rowsExt,1:na_colsExt), stat=istat, errmsg=errorMessage)
         check_allocate("redistribute: qExtern", istat, errorMessage)

         ! copy back
         num = (matrixRows* matrixCols) * size_of_datatype
         successGPU = gpu_memcpy(int(loc(qIntern(1,1)),kind=c_intptr_t), q_dev, &
                      num, gpuMemcpyDeviceToHost)
         check_memcpy_gpu("elpa1_template q_dev -> qIntern", successGPU)
       endif

#else
       ! data is provided via host arrays {a|q|ev|}Extern
#endif


#ifdef DEVICE_POINTER
       if (present(qDevExtern)) then
#else
       if (present(qExtern)) then
#endif
       call obj%timer%start("GEMR2D")
       call scal_PRECISION_GEMR2D &
       (int(na,kind=BLAS_KIND), int(na,kind=BLAS_KIND), qIntern, 1_BLAS_KIND, 1_BLAS_KIND, sc_descInternal, qExtern, &
       1_BLAS_KIND, 1_BLAS_KIND, sc_descExt, external_blacs_ctxtBLAS)
       call obj%timer%stop("GEMR2D")
     endif

#ifdef DEVICE_POINTER
     if (present(qDevExtern)) then
       num = (na_rowsExt*na_colsExt) * size_of_datatype
       successGPU = gpu_memcpy(qDevExtern,int(loc(qExtern(1,1)),kind=c_intptr_t), &
                                 num, gpuMemcpyHostToDevice)
       check_memcpy_gpu("redistribute qExtern -> qDevExtern", successGPU)

       deallocate(qExtern)
     endif

     deallocate(qIntern)
#endif
     !clean MPI communicators and blacs grid
     !of the internal re-distributed matrix
     call mpi_comm_free(mpi_comm_rowsMPIInternal, mpierr)
     call mpi_comm_free(mpi_comm_colsMPIInternal, mpierr)
     call blacs_gridexit(blacs_ctxtInternal)
   endif

