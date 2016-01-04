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
module test_util
  implicit none
  private
  public mpi_thread_level_name
  include 'mpif.h'

  contains
!>
!> This function translates, if ELPA was build with OpenMP support,
!> the found evel of "thread safetiness" from the internal number
!> of the MPI library into a human understandable value
!>
!> \param level thread-saftiness of the MPI library
!> \return str human understandable value of thread saftiness
  pure function mpi_thread_level_name(level) result(str)
    use precision
    implicit none
    integer(kind=ik), intent(in) :: level
    character(len=21)            :: str
    select case(level)
      case (MPI_THREAD_SINGLE)
        str = "MPI_THREAD_SINGLE"
      case (MPI_THREAD_FUNNELED)
        str = "MPI_THREAD_FUNNELED"
      case (MPI_THREAD_SERIALIZED)
        str = "MPI_THREAD_SERIALIZED"
      case (MPI_THREAD_MULTIPLE)
        str = "MPI_THREAD_MULTIPLE"
      case default
        write(str,'(i0,1x,a)') level, "(Unknown level)"
    end select
  end function

end module
