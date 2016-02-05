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

module mod_check_for_gpu

  contains

    function check_for_gpu(myid, numberOfDevices, wantDebug) result(gpuAvailable)
      use cuda_functions
      use precision
      implicit none
      include 'mpif.h'
      integer(kind=ik), intent(in)  :: myid
      logical, optional, intent(in) :: wantDebug
      logical                       :: success, wantDebugMessage
      integer(kind=ik), intent(out) :: numberOfDevices
      integer(kind=ik)              :: deviceNumber, mpierr, maxNumberOfDevices
      logical                       :: gpuAvailable
      character(len=1024)           :: envname

      gpuAvailable = .false.

      if (.not.(present(wantDebug))) then
        wantDebugMessage = .false.
      else
        if (wantDebug) then
          wantDebugMessage=.true.
        else
          wantDebugMessage=.false.
        endif
      endif

      ! call getenv("CUDA_PROXY_PIPE_DIRECTORY", envname)
      success = cuda_getdevicecount(numberOfDevices)

      if (.not.(success)) then
        print *,"error in cuda_getdevicecount"
        stop
      endif

      ! make sure that all nodes have the same number of GPU's, otherwise
      ! we run into loadbalancing trouble
      call mpi_allreduce(numberOfDevices, maxNumberOfDevices, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD, mpierr)

      if (maxNumberOfDevices .ne. numberOfDevices) then
        print *,"Different number of GPU devices on MPI tasks!"
        print *,"GPUs will NOT be used!"
        gpuAvailable = .false.
        return
      endif

      if (numberOfDevices .ne. 0) then
        gpuAvailable = .true.
        ! Usage of GPU is possible since devices have been detected

        if (myid==0) then
          if (wantDebugMessage) then
            print *
            print '(3(a,i0))','Found ', numberOfDevices, ' GPUs'
          endif
        endif

        deviceNumber = mod(myid, numberOfDevices)
        success = cuda_setdevice(deviceNumber)

        if (.not.(success)) then
          print *,"Cannot set CudaDevice"
          stop
        endif
        if (wantDebugMessage) then
          print '(3(a,i0))', 'MPI rank ', myid, ' uses GPU #', deviceNumber
        endif
      endif

    end function
end module
