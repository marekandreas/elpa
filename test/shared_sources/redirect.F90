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

module redirect
  use, intrinsic :: iso_c_binding

  implicit none
  public

  logical :: use_redirect_stdout

  interface
    integer(kind=C_INT) function create_directories_c() bind(C, name="create_directories")
      use, intrinsic :: iso_c_binding
      implicit none
    end function
  end interface

  interface
    subroutine redirect_stdout_c(myproc) bind(C, name="redirect_stdout")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT), intent(in) :: myproc
    end subroutine
  end interface

  contains
!>
!> This function is the Fortran driver for the
!> C program to create the redirect output
!> directory
!>
!> \param none
!> \result res integer indicates success or failure
    function create_directories() result(res)
      use precision
      implicit none
      integer(kind=ik) :: res
      res = int(create_directories_c())
    end function
!>
!> This subroutine is the Fortran driver for the
!> redirection of stdout and stderr of each MPI
!> task
!>
!> \param myproc MPI task id
    subroutine redirect_stdout(myproc)
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer(kind=ik), intent(in) :: myproc
      call redirect_stdout_c(int(myproc, kind=C_INT))
    end subroutine
!>
!> This function checks, whether the environment variable
!> "REDIRECT_ELPA_TEST_OUTPUT" is set to "true".
!> Returns ".true." if variable is set, otherwise ".false."
!> This function only works if the during the build process
!> "HAVE_ENVIRONMENT_CHECKING" was tested successfully
!>
!> \param none
!> \return logical
    function check_redirect_environment_variable() result(redirect)
      implicit none
      logical            :: redirect
      character(len=255) :: REDIRECT_VARIABLE

      redirect = .false.

#if defined(HAVE_ENVIRONMENT_CHECKING)
      call get_environment_variable("REDIRECT_ELPA_TEST_OUTPUT",REDIRECT_VARIABLE)
#endif
      if (trim(REDIRECT_VARIABLE) .eq. "true") redirect = .true.

    end function

end module redirect
