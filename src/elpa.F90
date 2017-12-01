!
!    Copyright 2017, L. Hüdepohl and A. Marek, MPCDF
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

! The ELPA public API


!> \mainpage
!> Eigenvalue SoLvers for Petaflop-Applications (ELPA)
!> \par
!> http://elpa.mpcdf.mpg.de
!>
!> \par
!>    The ELPA library was originally created by the ELPA consortium,
!>    consisting of the following organizations:
!>
!>    - Max Planck Computing and Data Facility (MPCDF) formerly known as
!>      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!>    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!>      Informatik,
!>    - Technische Universität München, Lehrstuhl für Informatik mit
!>      Schwerpunkt Wissenschaftliches Rechnen ,
!>    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
!>    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
!>      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!>      and
!>    - IBM Deutschland GmbH
!>
!>   Some parts and enhancements of ELPA have been contributed and authored
!>   by the Intel Corporation and Nvidia Corporation, which are not part of
!>   the ELPA consortium.
!>
!>   Maintainance and development of the ELPA library is done by the
!>   Max Planck Computing and Data Facility (MPCDF)
!>
!>
!>   Futher support of the ELPA library is done by the ELPA-AEO consortium,
!>   consisting of the following organizations:
!>
!>    - Max Planck Computing and Data Facility (MPCDF) formerly known as
!>      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!>    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!>      Informatik,
!>    - Technische Universität München, Lehrstuhl für Informatik mit
!>      Schwerpunkt Wissenschaftliches Rechnen ,
!>    - Technische Universität München, Lehrstuhl für theoretische Chemie,
!>    - Fritz-Haber-Institut, Berlin, Abt. Theorie
!>
!>
!>   Contributions to the ELPA source have been authored by (in alphabetical order):
!>
!> \author T. Auckenthaler, Volker Blum, A. Heinecke, L. Huedepohl, R. Johanni, Werner Jürgens, Pavel Kus, and A. Marek
!>
!> All the important information is in the \ref elpa_api::elpa_t derived type
!>
!> \brief Abstract definition of the elpa_t type
!>
!>
!> A typical usage of ELPA might look like this:
!>
!> Fortran synopsis
!>
!> \code{.f90}
!>  use elpa
!>  class(elpa_t), pointer :: elpa
!>  integer :: success
!>
!>  if (elpa_init(20171201) /= ELPA_OK) then
!>     print *, "ELPA API version not supported"
!>     stop
!>   endif
!>   elpa => elpa_allocate()
!>
!>   ! set parameters decribing the matrix and it's MPI distribution
!>   call elpa%set("na", na, success)
!>   call elpa%set("nev", nev, success)
!>   call elpa%set("local_nrows", na_rows, success)
!>   call elpa%set("local_ncols", na_cols, success)
!>   call elpa%set("nblk", nblk, success)
!>   call elpa%set("mpi_comm_parent", MPI_COMM_WORLD, success)
!>   call elpa%set("process_row", my_prow, success)
!>   call elpa%set("process_col", my_pcol, success)
!>
!>   succes = elpa%setup()
!>
!>   ! if desired, set tunable run-time options
!>   call e%set("solver", ELPA_SOLVER_2STAGE, success)
!> \endcode
!>   ... set and get all other options that are desired
!> \code{.f90}
!>
!>   ! use method solve to solve the eigenvalue problem to obtain eigenvalues
!>   ! and eigenvectors
!>   ! other possible methods are desribed in \ref elpa_api::elpa_t derived type
!>   call e%eigenvectors(a, ev, z, success)
!>
!>   ! cleanup
!>   call elpa_deallocate(e)
!>
!>   call elpa_uninit()
!> \endcode
!>
!>
!> C synopsis
!>
!>  \code{.c}
!>   #include <elpa/elpa.h>
!>
!>   elpa_t handle;
!>   int error;
!>
!>   if (elpa_init(20171201) != ELPA_OK) {
!>     fprintf(stderr, "Error: ELPA API version not supported");
!>     exit(1);
!>   }
!>
!>   handle = elpa_allocate(&error);
!>
!>   /* Set parameters the matrix and it's MPI distribution */
!>   elpa_set(handle, "na", na, &error);
!>   elpa_set(handle, "nev", nev, &error);
!>   elpa_set(handle, "local_nrows", na_rows, &error);
!>   elpa_set(handle, "local_ncols", na_cols, &error);
!>   elpa_set(handle, "nblk", nblk, &error);
!>   elpa_set(handle, "mpi_comm_parent", MPI_Comm_c2f(MPI_COMM_WORLD), &error);
!>   elpa_set(handle, "process_row", my_prow, &error);
!>   elpa_set(handle, "process_col", my_pcol, &error);
!>
!>   /* Setup */
!>   elpa_setup(handle);
!>
!>   /* if desired, set tunable run-time options */
!>   elpa_set(handle, "solver", ELPA_SOLVER_2STAGE, &error);
!>  \endcode
!>   ... set and get all other options that are desired
!>  \code{.c}
!>
!>   /* use method solve to solve the eigenvalue problem */
!>   /* other possible methods are desribed in \ref elpa_api::elpa_t derived type */
!>   elpa_eigenvectors(handle, a, ev, z, &error);
!>
!>   /* cleanup */
!>   elpa_deallocate(handle);
!>   elpa_uninit();
!> \endcode
!>
!> \brief Fortran module to use the ELPA library. No other module shoule be used
module elpa
  use elpa_constants
  use elpa_api

  implicit none
  public

  contains

    !> \brief function to allocate an ELPA instance
    !> Parameters
    !> \details
    !> \result  obj        class(elpa_t), pointer : pointer to allocated object
    function elpa_allocate() result(obj)
      use elpa_impl
      class(elpa_t), pointer :: obj
      obj => elpa_impl_allocate()
    end function


    !> \brief function to deallocate an ELPA instance
    !> Parameters
    !> \details
    !> \param  obj        class(elpa_t), pointer : pointer to the ELPA object to be destroyed and deallocated
    subroutine elpa_deallocate(obj)
      class(elpa_t), pointer :: obj
      call obj%destroy()
      deallocate(obj)
    end subroutine


    !> \brief function to deallocate an ELPA autotune instance
    !> Parameters
    !> \details
    !> \param  obj        class(elpa_autotune_t), pointer : pointer to the autotune object to be destroyed and deallocated   
    subroutine elpa_autotune_deallocate(obj)
      class(elpa_autotune_t), pointer :: obj
      call obj%destroy()
      deallocate(obj)
    end subroutine

end module
