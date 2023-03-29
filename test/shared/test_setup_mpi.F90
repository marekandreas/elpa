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

#ifdef HAVE_64BIT_INTEGER_MATH_SUPPORT
#define TEST_INT_TYPE integer(kind=c_int64_t)
#define INT_TYPE c_int64_t
#else
#define TEST_INT_TYPE integer(kind=c_int32_t)
#define INT_TYPE c_int32_t
#endif
#ifdef HAVE_64BIT_INTEGER_MPI_SUPPORT
#define TEST_INT_MPI_TYPE integer(kind=c_int64_t)
#define INT_MPI_TYPE c_int64_t
#else
#define TEST_INT_MPI_TYPE integer(kind=c_int32_t)
#define INT_MPI_TYPE c_int32_t
#endif

module test_setup_mpi

  contains

    subroutine setup_mpi(myid, nprocs)
      use test_util
      use ELPA_utilities
      use precision_for_tests
      implicit none

      TEST_INT_MPI_TYPE              :: mpierr

      TEST_INT_TYPE, intent(out)     :: myid, nprocs
      TEST_INT_MPI_TYPE              :: myidMPI, nprocsMPI
#ifdef WITH_OPENMP_TRADITIONAL
      TEST_INT_MPI_TYPE              :: required_mpi_thread_level, &
                                        provided_mpi_thread_level
#endif


#ifdef WITH_MPI

#ifndef WITH_OPENMP_TRADITIONAL
      call mpi_init(mpierr)
#else
      required_mpi_thread_level = MPI_THREAD_MULTIPLE

      call mpi_init_thread(required_mpi_thread_level,     &
                           provided_mpi_thread_level, mpierr)

      if (required_mpi_thread_level .ne. provided_mpi_thread_level) then
        write(error_unit,*) "MPI ERROR: MPI_THREAD_MULTIPLE is not provided on this system"
        write(error_unit,*) "           only ", mpi_thread_level_name(provided_mpi_thread_level), " is available"
        call MPI_FINALIZE(mpierr)
        call exit(77)
      endif
#endif
      call mpi_comm_rank(mpi_comm_world, myidMPI,  mpierr)
      call mpi_comm_size(mpi_comm_world, nprocsMPI,mpierr)

      myid   = int(myidMPI,kind=BLAS_KIND)
      nprocs = int(nprocsMPI,kind=BLAS_KIND)

      !if (nprocs <= 1) then
      !  print *, "The test programs must be run with more than 1 task to ensure that usage with MPI is actually tested"
      !  stop 1
      !endif
#else
      myid = 0
      nprocs = 1
#endif

    end subroutine


end module
