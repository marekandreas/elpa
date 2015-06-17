!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
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
!    http://elpa.rzg.mpg.de/
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
module mod_setup_mpi

  contains

    subroutine setup_mpi(myid, nprocs)
      use test_util
      use ELPA_utilities
      implicit none
      include 'mpif.h'

      integer              :: mpierr

      integer, intent(out) :: myid, nprocs
#ifdef WITH_OPENMP
      integer    :: required_mpi_thread_level, &
                    provided_mpi_thread_level
#endif
#ifndef WITH_OPENMP
      call mpi_init(mpierr)
#else
      required_mpi_thread_level = MPI_THREAD_MULTIPLE

      call mpi_init_thread(required_mpi_thread_level,     &
                           provided_mpi_thread_level, mpierr)

      if (required_mpi_thread_level .ne. provided_mpi_thread_level) then
        write(error_unit,*) "MPI ERROR: MPI_THREAD_MULTIPLE is not provided on this system"
        write(error_unit,*) "           only ", mpi_thread_level_name(provided_mpi_thread_level), " is available"
        call exit(77)
      endif

#endif
      call mpi_comm_rank(mpi_comm_world,myid,mpierr)
      call mpi_comm_size(mpi_comm_world,nprocs,mpierr)



    end subroutine


end module mod_setup_mpi
