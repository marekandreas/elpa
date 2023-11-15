!    Copyright 2021, A. Marek
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
!    along with ELPA. If not, see <http://www.gnu.org/licenses/>
!
!    ELPA reflects a substantial effort on the part of the original
!    ELPA consortium, and we ask you to respect the spirit of the
!    license that we chose: i.e., please contribute any changes you
!    may have back to the original ELPA library distribution, and keep
!    any derivatives of ELPA under the same license that we chose for
!    the original distribution, the GNU Lesser General Public License.
!
! Author: Andreas Marek, MPCDF

#include "config-f90.h"
module thread_affinity
  use precision

  implicit none

  public :: check_thread_affinity, &
            init_thread_affinity, cleanup_thread_affinity, print_thread_affinity
  private
! integer(kind=ik) :: thread_num
  integer(kind=ik) :: thread_max
  integer(kind=ik) :: process_cpu_id
  integer(kind=ik), allocatable :: cpu_ids(:)

#ifdef HAVE_AFFINITY_CHECKING
  interface
    subroutine get_process_id_c(process_id, pprocess_id) bind(C, name="get_process_id")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT), intent(out) :: process_id, pprocess_id
    end subroutine
  end interface

  interface
    subroutine get_thread_affinity_c(cpu_id) bind(C, name="get_thread_affinity")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT) :: cpu_id
    end subroutine
  end interface
  interface
    subroutine get_process_affinity_c(cpu_id) bind(C, name="get_process_affinity")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT), value :: cpu_id
    end subroutine
  end interface
#endif

contains
  subroutine get_thread_affinity(cpu_id)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=ik), intent(out) :: cpu_id
    integer(kind=C_INT) :: cpu_id_c
#ifdef HAVE_AFFINITY_CHECKING
    call get_thread_affinity_c(cpu_id_c)
    cpu_id = int(cpu_id_c, kind=ik)
#endif
  end subroutine
  subroutine get_process_affinity(cpu_id)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=ik), intent(out) :: cpu_id
    integer(kind=C_INT) :: cpu_id_c
#ifdef HAVE_AFFINITY_CHECKING
    call get_process_affinity_c(cpu_id_c)
    cpu_id = int(cpu_id_c, kind=ik)
#endif
  end subroutine
  subroutine get_process_id(process_id, pprocess_id)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=ik), intent(out) :: process_id, pprocess_id
    integer(kind=C_INT) :: process_id_c, pprocess_id_c
#ifdef HAVE_AFFINITY_CHECKING
    call get_process_id_c(process_id_c, pprocess_id_c)
#endif
    process_id  = int(process_id_c,  kind=ik)
    pprocess_id = int(pprocess_id_c, kind=ik)
  end subroutine


  subroutine init_thread_affinity(nrThreads)
    use precision
    use omp_lib

    implicit none
    integer(kind=ik)             :: istat
    integer(kind=ik), intent(in) :: nrThreads

    thread_max = nrThreads
#ifdef WITH_OPENMP_TRADITIONAL
    if(.not.(allocated(cpu_ids))) then
       allocate(cpu_ids(0:thread_max-1), stat=istat)
       if (istat .ne. 0) then
         print *,"Error when allocating init_thread_affinity"
       endif
    endif
#endif
  end subroutine init_thread_affinity

  subroutine cleanup_thread_affinity
    use precision
    implicit none
    integer(kind=ik) :: istat

    if((allocated(cpu_ids))) then
       deallocate(cpu_ids, stat=istat)
       if (istat .ne. 0) then
         print *,"Error when deallocating init_thread_affinity"
       endif
    endif

  end subroutine cleanup_thread_affinity

  subroutine check_thread_affinity()
    use precision
    use omp_lib
    implicit none
    integer(kind=ik)             :: thread_cpu_id
    integer(kind=ik)             :: i, actuall_num

    call get_process_affinity(process_cpu_id)

#ifdef WITH_OPENMP_TRADITIONAL

!$OMP  PARALLEL DO &
!$OMP  DEFAULT(NONE) &
!$OMP  PRIVATE(i,thread_cpu_id,actuall_num) &
!$OMP  SHARED(thread_max,cpu_ids) &
!$OMP  SCHEDULE(STATIC)

    do i=0,thread_max-1
       call get_thread_affinity(thread_cpu_id)
       actuall_num=omp_get_thread_num()
       cpu_ids(actuall_num)=thread_cpu_id
    enddo
#endif

  end subroutine check_thread_affinity

  subroutine print_thread_affinity(mype)

    use precision
    implicit none
    integer(kind=ik) :: i
    integer(kind=ik), intent(in) :: mype
    integer(kind=ik) :: pid, ppid

    call get_process_id(pid, ppid)
    write(*,'("Task ",i4," runs on process id: ",i4," with pid ",i4," and ppid ",i4)') mype, process_cpu_id,pid,ppid
#ifdef WITH_OPENMP_TRADITIONAL
    write(*,'("Each task uses ",i4," threads")') thread_max
       do i=0,thread_max-1
          write(*,'("Thread ",i4," is running on logical CPU-ID ",i4)') i,cpu_ids(i)
          print *,i,cpu_ids(i)
       enddo
#endif
  end subroutine print_thread_affinity

end module thread_affinity
