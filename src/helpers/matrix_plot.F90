!    Copyright 2016, P. Kus
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

#define  REAL_DATATYPE rk8

! module for producing matrix traces, to be plotted by provided python plotter
! currently the module is very simple and non-flexible
! it is only usable for printing the matrix A and possibly its counterpart A_DEV
! both are assumed to be in block-cyclic distribution
! At the moment, the output works for double real only
! To simplify things, a convenience macro (as follows) can be placed in a template file:

! #undef SAVE_MATR
! #ifdef DOUBLE_PRECISION_REAL
! #define SAVE_MATR(name, iteration) \
! call prmat(na,useGpu,a_mat,a_dev,lda,matrixCols,nblk,my_prow,my_pcol,np_rows,np_cols,name,iteration)
! #else
! #define SAVE_MATR(name, iteration)
! #endif

! traces are stored into directory "matrices", that has to be created

module matrix_plot

  contains

    subroutine prmat(na, useGpu, a_mat, a_dev, lda, matrixCols, nblk, my_prow, my_pcol, np_rows, np_cols, name, iteration)
      use cuda_functions
      use hip_functions
      use elpa_gpu
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer, parameter :: out_unit=20
      character(len = 1025) :: directory = "matrices"
      character(len = 1024) :: filename

      character(len = *), intent(in)             :: name
      integer(kind=ik), intent(in)                  :: na, lda, nblk, matrixCols, my_prow, my_pcol, np_rows, np_cols, iteration
      real(kind=REAL_DATATYPE), intent(in)          :: a_mat(lda,matrixCols)
      integer(kind=C_intptr_T), intent(in)          :: a_dev
      logical, intent(in)                           :: useGPU

      integer(kind=ik)                              :: row, col, mpi_rank
      integer(kind=ik), save                        :: counter = 0
      real(kind=REAL_DATATYPE)                      :: a_dev_helper(lda,matrixCols)
      logical                                       :: successGPU
      integer(kind=c_size_t), parameter             :: size_of_datatype = size_of_double_real

      mpi_rank = np_rows * my_pcol + my_prow

      ! print a_mat
      write(filename, "(A,A,I0.4,A,I0.2,A)") trim(directory), "/a_mat-", counter, "-", mpi_rank, ".txt"
      write(*,*) trim(filename)
      open(unit=out_unit, file=trim(filename), action="write",status="replace")

      write(out_unit, "(9I5)") na, nblk, lda, matrixCols, my_prow, my_pcol, np_rows, np_cols, iteration
      write(out_unit, "(A)") name
      do row = 1, lda
          write(out_unit, *) a_mat(row, :)
      end do
      close(out_unit)

      ! print a_dev

      if(useGpu) then
#ifdef HAVE_GPU_VERSION
        successGPU = cuda_memcpy(int(loc(a_dev_helper(1,1)),kind=c_intptr_t), &
                      a_dev, lda * matrixCols * size_of_datatype, cudaMemcpyDeviceToHost)
#endif
        write(filename, "(A,A,I0.4,A,I0.2,A)") trim(directory), "/a_dev-", counter, "-", mpi_rank, ".txt"
        write(*,*) trim(filename)
        open(unit=out_unit, file=trim(filename), action="write",status="replace")

        write(out_unit, "(9I5)") na, nblk, lda, matrixCols, my_prow, my_pcol, np_rows, np_cols, iteration
        write(out_unit, "(A)") name
        do row = 1, lda
            write(out_unit, *) a_dev_helper(row, :)
        end do
        close(out_unit)
      end if

      counter = counter + 1

    end subroutine

end module
