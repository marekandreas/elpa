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
!>  class(elpa_t), pointer :: elpaInstance
!>  integer :: success
!>
!>  ! We urge the user to always check the error code of all ELPA functions
!>
!>  if (elpa_init(20200417) /= ELPA_OK) then
!>     print *, "ELPA API version not supported"
!>     stop
!>   endif
!>   elpa => elpa_allocate(success)
!>   if (success /= ELPA_OK) then
!>     print *,"Could not allocate ELPA"
!>   endif
!>
!>   ! set parameters decribing the matrix and it's MPI distribution
!>   call elpaIstance%set("na", na, success, success)
!>   if (success /= ELPA_OK) then
!>     print *,"Could not set entry"
!>   endif
!>   call elpaInstance%set("nev", nev, success, success)
!>   ! check success code ...
!>
!>   call elpaInstance%set("local_nrows", na_rows, success)
!>   ! check success code ...
!>
!>   call elpaInstance%set("local_ncols", na_cols, success)
!>   call elpaInstance%set("nblk", nblk, success)
!>   call elpaInstance%set("mpi_comm_parent", MPI_COMM_WORLD, success)
!>   call elpaInstance%set("process_row", my_prow, success)
!>   call elpaInstance%set("process_col", my_pcol, success)
!>
!>   ! set up the elpa object
!>   success = elpaInstance%setup()
!>   if (succes /= ELPA_OK) then
!>     print *,"Could not setup ELPA object"
!>   endif
!>
!>   ! settings for GPU
!>   call elpaInstance%set("gpu", 1, success) ! 1=on, 2=off
!>   ! in case of GPU usage you have the choice whether ELPA
!>   ! should automatically assign each MPI task to a certain GPU
!>   ! (this is default) or whether you want to set this assignment
!>   ! for _each_ task yourself
!>   ! set assignment your self (only using one task here and assigning it 
!>   ! to GPU id 1)
!>   if (my_rank .eq. 0) call elpaInstance%set("use_gpu_id", 1, success)
!>
!>   ! if desired, set tunable run-time options
!>   ! here we want to use the 2-stage solver
!>   call elpaInstance%set("solver", ELPA_SOLVER_2STAGE, success)
!>
!>   ! and set a specific kernel (must be supported on the machine)
!>   call elpaInstance%set("real_kernel", ELPA_2STAGE_REAL_AVX_BLOCK2)
!> \endcode
!>   ... set and get all other options that are desired
!> \code{.f90}
!>
!>   ! if wanted you can store the settings and load them in another program
!>   call elpa%store_settings("save_to_disk.txt", success)
!>
!>   ! use method solve to solve the eigenvalue problem to obtain eigenvalues
!>   ! and eigenvectors
!>   ! other possible methods are desribed in \ref elpa_api::elpa_t derived type
!>   call elpaInstance%eigenvectors(a, ev, z, success)
!>
!>   ! cleanup
!>   call elpa_deallocate(e, success)
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
!>   /*  We urge the user to always check the error code of all ELPA functions */
!>
!>   if (elpa_init(20200417) != ELPA_OK) {
!>     fprintf(stderr, "Error: ELPA API version not supported");
!>     exit(1);
!>   }
!>
!>   
!>   handle = elpa_allocate(&error);
!>   if (error != ELPA_OK) {
!>   /* do sth. */
!>   }
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
!>   error = elpa_setup(handle);
!>
!>   /* if desired, set tunable run-time options */
!>   /* here we want to use the 2-stage solver */
!>   elpa_set(handle, "solver", ELPA_SOLVER_2STAGE, &error);
!>
!>   /* settings for GPU */
!>   elpa_set(handle, "gpu", 1, &error);  /* 1=on, 2=off */
!>   /* in case of GPU usage you have the choice whether ELPA
!>      should automatically assign each MPI task to a certain GPU
!>      (this is default) or whether you want to set this assignment
!>      for _each_ task yourself
!>      set assignment your self (only using one task here and assigning it 
!>      to GPU id 1) */
!>   if (my_rank == 0) elpa_set(handle, "use_gpu_id", 1, &error);
!>
!>   elpa_set(handle,"real_kernel", ELPA_2STAGE_REAL_AVX_BLOCK2, &error);
!>  \endcode
!>   ... set and get all other options that are desired
!>  \code{.c}
!>
!>   /* if you want you can store the settings and load them in another program */
!>   elpa_store_settings(handle, "save_to_disk.txt");
!>
!>   /* use method solve to solve the eigenvalue problem */
!>   /* other possible methods are desribed in \ref elpa_api::elpa_t derived type */
!>   elpa_eigenvectors(handle, a, ev, z, &error);
!>
!>   /* cleanup */
!>   elpa_deallocate(handle, &error);
!>   elpa_uninit();
!> \endcode
!>
!> the autotuning could be used like this:
!>
!> Fortran synopsis
!>
!> \code{.f90}
!>  use elpa
!>  class(elpa_t), pointer :: elpa
!>  class(elpa_autotune_t), pointer :: tune_state
!>  integer :: success
!>
!>  if (elpa_init(20200417) /= ELPA_OK) then
!>     print *, "ELPA API version not supported"
!>     stop
!>   endif
!>   elpa => elpa_allocate(success)
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
!>   ! set up the elpa object
!>   success = elpa%setup()
!>
!>   ! create autotune object
!>   tune_state => elpa%autotune_setup(ELPA_AUTOTUNE_FAST, ELPA_AUTOTUNE_DOMAIN_REAL, success)
!>
!>   ! you can set some options, these will be then FIXED for the autotuning step
!>   ! if desired, set tunable run-time options
!>   ! here we want to use the 2-stage solver
!>   call e%set("solver", ELPA_SOLVER_2STAGE, success)
!>
!>   ! and set a specific kernel (must be supported on the machine)
!>   call e%set("real_kernel", ELPA_2STAGE_REAL_AVX_BLOCK2, success)
!> \endcode
!>   ... set and get all other options that are desired
!> \code{.f90}
!>
!>   iter = 0
!>   do while (elpa%autotune_step(tune_state, success))
!>     iter = iter + 1
!>     call e%eigenvectors(a, ev, z, success)
!>
!>     ! if needed you can save the autotune state at any point
!>     ! and resume it
!>     if (iter > MAX_ITER) then
!>       call elpa%autotune_save_state(tune_state,"autotune_checkpoint.txt", success)
!>       exit
!>     endif
!>   enddo
!>
!>   !set and print the finished autotuning
!>   call elpa%autotune_set_best(tune_state, success)
!>   
!>   ! store _TUNED_ ELPA object, if needed
!>   call elpa%store("autotuned_object.txt", success)
!>
!>   !deallocate autotune object
!>   call elpa_autotune_deallocate(tune_state, success)
!>
!>   ! cleanup
!>   call elpa_deallocate(e, success)
!>
!>   call elpa_uninit()
!> \endcode
!>
!> More examples can be found in the folder "test", where Fortran and C example programs
!> are stored

!> \brief Fortran module to use the ELPA library. No other module shoule be used

#include "config-f90.h"
module elpa
  use elpa_constants
  use elpa_api

  implicit none
  public

  contains

    !> \brief function to allocate an ELPA instance
    !> Parameters
    !> \details
    !> \params  error      integer, optional : error code
    !> \result  obj        class(elpa_t), pointer : pointer to allocated object
    function elpa_allocate(error) result(obj)
      use elpa_impl
      class(elpa_t), pointer         :: obj
#ifdef USE_FORTRAN2008
      integer, optional, intent(out) :: error
#else
      integer, intent(out)           :: error
#endif
      integer                        :: error2

      obj => elpa_impl_allocate(error2)

#ifdef USE_FORTRAN2008
      if (present(error)) then
#endif
        error = error2
        if (error .ne. ELPA_OK) then
          write(*,*) "Cannot allocate the ELPA object!"
          write(*,*) "This is a critical error!"
          write(*,*) "ELPA not usable with this error"
        endif
#ifdef USE_FORTRAN2008
      else
        if (error2 .ne. ELPA_OK) then
          write(*,*) "Cannot allocate the ELPA object!"
          write(*,*) "This is a critical error, but you do not check the error codes!"
          write(*,*) "ELPA not usable with this error"
          stop
        endif
      endif
#endif

    end function


    !> \brief function to deallocate an ELPA instance
    !> Parameters
    !> \details
    !> \param  obj        class(elpa_t), pointer : pointer to the ELPA object to be destroyed and deallocated
    !> \param  error      integer, optional : error code
    subroutine elpa_deallocate(obj, error)
      class(elpa_t), pointer         :: obj
#ifdef USE_FORTRAN2008
      integer, optional, intent(out) :: error
#else
      integer, intent(out)           :: error
#endif
      integer                        :: error2
        
      call obj%destroy(error2)
#ifdef USE_FORTRAN2008
      if (present(error)) then
#endif
        error = error2
        if (error .ne. ELPA_OK) then
          write(*,*) "Cannot destroy the ELPA object!"  
          write(*,*) "This is a critical error!"  
          write(*,*) "This might lead to a memory leak in your application!"
          error = ELPA_ERROR_CRITICAL
          return
        endif
#ifdef USE_FORTRAN2008
      else
        if (error2 .ne. ELPA_OK) then
          write(*,*) "Cannot destroy the ELPA object!"
          write(*,*) "This is a critical error!"
          write(*,*) "This might lead to a memory leak in your application!"
          write(*,*) "But you do not check the error codes!"
          return
        endif
      endif
#endif
      deallocate(obj, stat=error2)
      if (error2 .ne. 0) then
        write(*,*) "Cannot deallocate the ELPA object!"  
        write(*,*) "This is a critical error!"  
        write(*,*) "This might lead to a memory leak in your application!"
#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = ELPA_ERROR_CRITICAL
          return
        endif
#else
        error = ELPA_ERROR_CRITICAL
        return
#endif
      endif
    end subroutine

#ifdef ENABLE_AUTOTUNING
    !> \brief function to deallocate an ELPA autotune instance
    !> Parameters
    !> \details
    !> \param  obj        class(elpa_autotune_t), pointer : pointer to the autotune object to be destroyed and deallocated
    !> \param  error      integer, optional : error code
    subroutine elpa_autotune_deallocate(obj, error)
      class(elpa_autotune_t), pointer :: obj
#ifdef USE_FORTRAN2008
      integer, optional, intent(out)  :: error
#else
      integer, intent(out)            :: error
#endif
      integer                         :: error2
      call obj%destroy(error2)
#ifdef USE_FORTRAN2008
      if (present(error)) then
#endif
        error = error2
        if (error2 .ne. ELPA_OK) then
          write(*,*) "Cannot destroy the ELPA autotuning object!"
          write(*,*) "This is a critical error!"
          write(*,*) "This might lead to a memory leak in your application!"
          error = ELPA_ERROR_CRITICAL
          return
        endif
#ifdef USE_FORTRAN2008
      else
        if (error2 .ne. ELPA_OK) then
          write(*,*) "Cannot destroy the ELPA autotuning object!"
          write(*,*) "This is a critical error!"
          write(*,*) "This might lead to a memory leak in your application!"
          write(*,*) "But you do not check the error codes"
          return
        endif
      endif
#endif
      deallocate(obj, stat=error2)
      if (error2 .ne. 0) then
        write(*,*) "Cannot deallocate the ELPA autotuning object!"  
        write(*,*) "This is a critical error!"  
        write(*,*) "This might lead to a memory leak in your application!"
#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = ELPA_ERROR_CRITICAL
          return
        endif
#else
        error = ELPA_ERROR_CRITICAL
        return
#endif
      endif

    end subroutine
#endif

end module
