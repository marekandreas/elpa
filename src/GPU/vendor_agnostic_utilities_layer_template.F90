#if 0
!    Copyright 2021, A. Marek, MPCDF
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
! This file is the generated version. Do NOT edit
#endif
  interface gpu_memcpy_async_and_stream_synchronize
    module procedure gpu_memcpy_async_and_stream_synchronize_double_scalar
    module procedure gpu_memcpy_async_and_stream_synchronize_single_scalar
    module procedure gpu_memcpy_async_and_stream_synchronize_complex_scalar
    module procedure gpu_memcpy_async_and_stream_synchronize_complex_single_scalar
    module procedure gpu_memcpy_async_and_stream_synchronize_double_1d
    module procedure gpu_memcpy_async_and_stream_synchronize_single_1d
    module procedure gpu_memcpy_async_and_stream_synchronize_complex_1d
    module procedure gpu_memcpy_async_and_stream_synchronize_complex_single_1d
    module procedure gpu_memcpy_async_and_stream_synchronize_double_2d
    module procedure gpu_memcpy_async_and_stream_synchronize_single_2d
    module procedure gpu_memcpy_async_and_stream_synchronize_complex_2d
    module procedure gpu_memcpy_async_and_stream_synchronize_complex_single_2d
    module procedure gpu_memcpy_async_and_stream_synchronize_double_3d
    module procedure gpu_memcpy_async_and_stream_synchronize_single_3d
    module procedure gpu_memcpy_async_and_stream_synchronize_complex_3d
    module procedure gpu_memcpy_async_and_stream_synchronize_complex_single_3d
  end interface

  contains


    subroutine gpu_memcpy_async_and_stream_synchronize_double_scalar &
                (errormessage, devPtr, devPtrOffset, &
                 hostarray, &
                num, direction, my_stream, doSyncBefore, doSyncAfter, doSyncDefault)

      use iso_c_binding
      use elpa_gpu
      use elpa_utilities

      implicit none
      integer(kind=c_intptr_t), intent(in)          :: my_stream
      logical                                       :: successGPU
      character(len=*)                              :: errormessage
      integer(kind=c_intptr_t)                      :: devPtr
      integer(kind=c_intptr_t)                      :: devPtrOffset

      logical                                       :: doSyncBefore
      logical                                       :: doSyncAfter
      logical                                       :: doSyncDefault

      real(kind=c_double)                           :: &
                                             hostArray
      integer(kind=c_intptr_t)                      :: num
      integer(kind=c_int), intent(in)               :: direction

      if (doSyncBefore) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif

      if (direction .eq. gpuMemcpyHostToDevice) then
        successGPU = gpu_memcpy_async(devPtr+devPtrOffset, &
                    int(loc(hostarray),kind=c_intptr_t), &
                                      num, direction, my_stream)
      else if (direction .eq. gpuMemcpyDeviceToHost) then
        successGPU = gpu_memcpy_async(&
                    int(loc(hostarray),kind=c_intptr_t), &
                                      devPtr+devPtrOffset, num, direction, my_stream)
      else
        print *,"gpu_memcpy_async_and_stream_synchronize: unknown error"
        stop 1
      endif

      if (doSyncAfter) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
      if (doSyncDefault) then
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
    end


    subroutine gpu_memcpy_async_and_stream_synchronize_single_scalar &
                (errormessage, devPtr, devPtrOffset, &
                 hostarray, &
                num, direction, my_stream, doSyncBefore, doSyncAfter, doSyncDefault)

      use iso_c_binding
      use elpa_gpu
      use elpa_utilities

      implicit none
      integer(kind=c_intptr_t), intent(in)          :: my_stream
      logical                                       :: successGPU
      character(len=*)                              :: errormessage
      integer(kind=c_intptr_t)                      :: devPtr
      integer(kind=c_intptr_t)                      :: devPtrOffset

      logical                                       :: doSyncBefore
      logical                                       :: doSyncAfter
      logical                                       :: doSyncDefault

      real(kind=c_float)                            :: &
                                             hostArray
      integer(kind=c_intptr_t)                      :: num
      integer(kind=c_int), intent(in)               :: direction

      if (doSyncBefore) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif

      if (direction .eq. gpuMemcpyHostToDevice) then
        successGPU = gpu_memcpy_async(devPtr+devPtrOffset, &
                    int(loc(hostarray),kind=c_intptr_t), &
                                      num, direction, my_stream)
      else if (direction .eq. gpuMemcpyDeviceToHost) then
        successGPU = gpu_memcpy_async(&
                    int(loc(hostarray),kind=c_intptr_t), &
                                      devPtr+devPtrOffset, num, direction, my_stream)
      else
        print *,"gpu_memcpy_async_and_stream_synchronize: unknown error"
        stop 1
      endif

      if (doSyncAfter) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
      if (doSyncDefault) then
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
    end


    subroutine gpu_memcpy_async_and_stream_synchronize_complex_scalar &
                (errormessage, devPtr, devPtrOffset, &
                 hostarray, &
                num, direction, my_stream, doSyncBefore, doSyncAfter, doSyncDefault)

      use iso_c_binding
      use elpa_gpu
      use elpa_utilities

      implicit none
      integer(kind=c_intptr_t), intent(in)          :: my_stream
      logical                                       :: successGPU
      character(len=*)                              :: errormessage
      integer(kind=c_intptr_t)                      :: devPtr
      integer(kind=c_intptr_t)                      :: devPtrOffset

      logical                                       :: doSyncBefore
      logical                                       :: doSyncAfter
      logical                                       :: doSyncDefault

      complex(kind=c_double_complex)                :: &
                                             hostArray
      integer(kind=c_intptr_t)                      :: num
      integer(kind=c_int), intent(in)               :: direction

      if (doSyncBefore) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif

      if (direction .eq. gpuMemcpyHostToDevice) then
        successGPU = gpu_memcpy_async(devPtr+devPtrOffset, &
                    int(loc(hostarray),kind=c_intptr_t), &
                                      num, direction, my_stream)
      else if (direction .eq. gpuMemcpyDeviceToHost) then
        successGPU = gpu_memcpy_async(&
                    int(loc(hostarray),kind=c_intptr_t), &
                                      devPtr+devPtrOffset, num, direction, my_stream)
      else
        print *,"gpu_memcpy_async_and_stream_synchronize: unknown error"
        stop 1
      endif

      if (doSyncAfter) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
      if (doSyncDefault) then
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
    end


    subroutine gpu_memcpy_async_and_stream_synchronize_complex_single_scalar &
                (errormessage, devPtr, devPtrOffset, &
                 hostarray, &
                num, direction, my_stream, doSyncBefore, doSyncAfter, doSyncDefault)

      use iso_c_binding
      use elpa_gpu
      use elpa_utilities

      implicit none
      integer(kind=c_intptr_t), intent(in)          :: my_stream
      logical                                       :: successGPU
      character(len=*)                              :: errormessage
      integer(kind=c_intptr_t)                      :: devPtr
      integer(kind=c_intptr_t)                      :: devPtrOffset

      logical                                       :: doSyncBefore
      logical                                       :: doSyncAfter
      logical                                       :: doSyncDefault

      complex(kind=c_float_complex)                 :: &
                                             hostArray
      integer(kind=c_intptr_t)                      :: num
      integer(kind=c_int), intent(in)               :: direction

      if (doSyncBefore) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif

      if (direction .eq. gpuMemcpyHostToDevice) then
        successGPU = gpu_memcpy_async(devPtr+devPtrOffset, &
                    int(loc(hostarray),kind=c_intptr_t), &
                                      num, direction, my_stream)
      else if (direction .eq. gpuMemcpyDeviceToHost) then
        successGPU = gpu_memcpy_async(&
                    int(loc(hostarray),kind=c_intptr_t), &
                                      devPtr+devPtrOffset, num, direction, my_stream)
      else
        print *,"gpu_memcpy_async_and_stream_synchronize: unknown error"
        stop 1
      endif

      if (doSyncAfter) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
      if (doSyncDefault) then
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
    end


    subroutine gpu_memcpy_async_and_stream_synchronize_double_1d &
                (errormessage, devPtr, devPtrOffset, &
                 hostarray, &
                 off1, &
                num, direction, my_stream, doSyncBefore, doSyncAfter, doSyncDefault)

      use iso_c_binding
      use elpa_gpu
      use elpa_utilities

      implicit none
      integer(kind=c_intptr_t), intent(in)          :: my_stream
      logical                                       :: successGPU
      character(len=*)                              :: errormessage
      integer(kind=c_intptr_t)                      :: devPtr
      integer(kind=c_intptr_t)                      :: devPtrOffset

      integer(kind=ik), intent(in)                  :: off1
      logical                                       :: doSyncBefore
      logical                                       :: doSyncAfter
      logical                                       :: doSyncDefault

      real(kind=c_double)                           :: &
                                         hostArray(:)
      integer(kind=c_intptr_t)                      :: num
      integer(kind=c_int), intent(in)               :: direction

      if (doSyncBefore) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif

      if (direction .eq. gpuMemcpyHostToDevice) then
        successGPU = gpu_memcpy_async(devPtr+devPtrOffset, &
                    int(loc(hostarray(off1)),kind=c_intptr_t), &
                                      num, direction, my_stream)
      else if (direction .eq. gpuMemcpyDeviceToHost) then
        successGPU = gpu_memcpy_async(&
                    int(loc(hostarray(off1)),kind=c_intptr_t), &
                                      devPtr+devPtrOffset, num, direction, my_stream)
      else
        print *,"gpu_memcpy_async_and_stream_synchronize: unknown error"
        stop 1
      endif

      if (doSyncAfter) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
      if (doSyncDefault) then
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
    end


    subroutine gpu_memcpy_async_and_stream_synchronize_single_1d &
                (errormessage, devPtr, devPtrOffset, &
                 hostarray, &
                 off1, &
                num, direction, my_stream, doSyncBefore, doSyncAfter, doSyncDefault)

      use iso_c_binding
      use elpa_gpu
      use elpa_utilities

      implicit none
      integer(kind=c_intptr_t), intent(in)          :: my_stream
      logical                                       :: successGPU
      character(len=*)                              :: errormessage
      integer(kind=c_intptr_t)                      :: devPtr
      integer(kind=c_intptr_t)                      :: devPtrOffset

      integer(kind=ik), intent(in)                  :: off1
      logical                                       :: doSyncBefore
      logical                                       :: doSyncAfter
      logical                                       :: doSyncDefault

      real(kind=c_float)                            :: &
                                         hostArray(:)
      integer(kind=c_intptr_t)                      :: num
      integer(kind=c_int), intent(in)               :: direction

      if (doSyncBefore) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif

      if (direction .eq. gpuMemcpyHostToDevice) then
        successGPU = gpu_memcpy_async(devPtr+devPtrOffset, &
                    int(loc(hostarray(off1)),kind=c_intptr_t), &
                                      num, direction, my_stream)
      else if (direction .eq. gpuMemcpyDeviceToHost) then
        successGPU = gpu_memcpy_async(&
                    int(loc(hostarray(off1)),kind=c_intptr_t), &
                                      devPtr+devPtrOffset, num, direction, my_stream)
      else
        print *,"gpu_memcpy_async_and_stream_synchronize: unknown error"
        stop 1
      endif

      if (doSyncAfter) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
      if (doSyncDefault) then
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
    end


    subroutine gpu_memcpy_async_and_stream_synchronize_complex_1d &
                (errormessage, devPtr, devPtrOffset, &
                 hostarray, &
                 off1, &
                num, direction, my_stream, doSyncBefore, doSyncAfter, doSyncDefault)

      use iso_c_binding
      use elpa_gpu
      use elpa_utilities

      implicit none
      integer(kind=c_intptr_t), intent(in)          :: my_stream
      logical                                       :: successGPU
      character(len=*)                              :: errormessage
      integer(kind=c_intptr_t)                      :: devPtr
      integer(kind=c_intptr_t)                      :: devPtrOffset

      integer(kind=ik), intent(in)                  :: off1
      logical                                       :: doSyncBefore
      logical                                       :: doSyncAfter
      logical                                       :: doSyncDefault

      complex(kind=c_double_complex)                :: &
                                         hostArray(:)
      integer(kind=c_intptr_t)                      :: num
      integer(kind=c_int), intent(in)               :: direction

      if (doSyncBefore) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif

      if (direction .eq. gpuMemcpyHostToDevice) then
        successGPU = gpu_memcpy_async(devPtr+devPtrOffset, &
                    int(loc(hostarray(off1)),kind=c_intptr_t), &
                                      num, direction, my_stream)
      else if (direction .eq. gpuMemcpyDeviceToHost) then
        successGPU = gpu_memcpy_async(&
                    int(loc(hostarray(off1)),kind=c_intptr_t), &
                                      devPtr+devPtrOffset, num, direction, my_stream)
      else
        print *,"gpu_memcpy_async_and_stream_synchronize: unknown error"
        stop 1
      endif

      if (doSyncAfter) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
      if (doSyncDefault) then
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
    end


    subroutine gpu_memcpy_async_and_stream_synchronize_complex_single_1d &
                (errormessage, devPtr, devPtrOffset, &
                 hostarray, &
                 off1, &
                num, direction, my_stream, doSyncBefore, doSyncAfter, doSyncDefault)

      use iso_c_binding
      use elpa_gpu
      use elpa_utilities

      implicit none
      integer(kind=c_intptr_t), intent(in)          :: my_stream
      logical                                       :: successGPU
      character(len=*)                              :: errormessage
      integer(kind=c_intptr_t)                      :: devPtr
      integer(kind=c_intptr_t)                      :: devPtrOffset

      integer(kind=ik), intent(in)                  :: off1
      logical                                       :: doSyncBefore
      logical                                       :: doSyncAfter
      logical                                       :: doSyncDefault

      complex(kind=c_float_complex)                 :: &
                                         hostArray(:)
      integer(kind=c_intptr_t)                      :: num
      integer(kind=c_int), intent(in)               :: direction

      if (doSyncBefore) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif

      if (direction .eq. gpuMemcpyHostToDevice) then
        successGPU = gpu_memcpy_async(devPtr+devPtrOffset, &
                    int(loc(hostarray(off1)),kind=c_intptr_t), &
                                      num, direction, my_stream)
      else if (direction .eq. gpuMemcpyDeviceToHost) then
        successGPU = gpu_memcpy_async(&
                    int(loc(hostarray(off1)),kind=c_intptr_t), &
                                      devPtr+devPtrOffset, num, direction, my_stream)
      else
        print *,"gpu_memcpy_async_and_stream_synchronize: unknown error"
        stop 1
      endif

      if (doSyncAfter) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
      if (doSyncDefault) then
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
    end


    subroutine gpu_memcpy_async_and_stream_synchronize_double_2d &
                (errormessage, devPtr, devPtrOffset, &
                 hostarray, &
                 off1, off2, &
                num, direction, my_stream, doSyncBefore, doSyncAfter, doSyncDefault)

      use iso_c_binding
      use elpa_gpu
      use elpa_utilities

      implicit none
      integer(kind=c_intptr_t), intent(in)          :: my_stream
      logical                                       :: successGPU
      character(len=*)                              :: errormessage
      integer(kind=c_intptr_t)                      :: devPtr
      integer(kind=c_intptr_t)                      :: devPtrOffset

      integer(kind=ik), intent(in)                  :: off1
      integer(kind=ik), intent(in)                  :: off2
      logical                                       :: doSyncBefore
      logical                                       :: doSyncAfter
      logical                                       :: doSyncDefault

      real(kind=c_double)                           :: &
                                         hostArray(:,:)
      integer(kind=c_intptr_t)                      :: num
      integer(kind=c_int), intent(in)               :: direction

      if (doSyncBefore) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif

      if (direction .eq. gpuMemcpyHostToDevice) then
        successGPU = gpu_memcpy_async(devPtr+devPtrOffset, &
                    int(loc(hostarray(off1,off2)),kind=c_intptr_t), &
                                      num, direction, my_stream)
      else if (direction .eq. gpuMemcpyDeviceToHost) then
        successGPU = gpu_memcpy_async(&
                    int(loc(hostarray(off1,off2)),kind=c_intptr_t), &
                                      devPtr+devPtrOffset, num, direction, my_stream)
      else
        print *,"gpu_memcpy_async_and_stream_synchronize: unknown error"
        stop 1
      endif

      if (doSyncAfter) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
      if (doSyncDefault) then
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
    end


    subroutine gpu_memcpy_async_and_stream_synchronize_single_2d &
                (errormessage, devPtr, devPtrOffset, &
                 hostarray, &
                 off1, off2, &
                num, direction, my_stream, doSyncBefore, doSyncAfter, doSyncDefault)

      use iso_c_binding
      use elpa_gpu
      use elpa_utilities

      implicit none
      integer(kind=c_intptr_t), intent(in)          :: my_stream
      logical                                       :: successGPU
      character(len=*)                              :: errormessage
      integer(kind=c_intptr_t)                      :: devPtr
      integer(kind=c_intptr_t)                      :: devPtrOffset

      integer(kind=ik), intent(in)                  :: off1
      integer(kind=ik), intent(in)                  :: off2
      logical                                       :: doSyncBefore
      logical                                       :: doSyncAfter
      logical                                       :: doSyncDefault

      real(kind=c_float)                            :: &
                                         hostArray(:,:)
      integer(kind=c_intptr_t)                      :: num
      integer(kind=c_int), intent(in)               :: direction

      if (doSyncBefore) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif

      if (direction .eq. gpuMemcpyHostToDevice) then
        successGPU = gpu_memcpy_async(devPtr+devPtrOffset, &
                    int(loc(hostarray(off1,off2)),kind=c_intptr_t), &
                                      num, direction, my_stream)
      else if (direction .eq. gpuMemcpyDeviceToHost) then
        successGPU = gpu_memcpy_async(&
                    int(loc(hostarray(off1,off2)),kind=c_intptr_t), &
                                      devPtr+devPtrOffset, num, direction, my_stream)
      else
        print *,"gpu_memcpy_async_and_stream_synchronize: unknown error"
        stop 1
      endif

      if (doSyncAfter) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
      if (doSyncDefault) then
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
    end


    subroutine gpu_memcpy_async_and_stream_synchronize_complex_2d &
                (errormessage, devPtr, devPtrOffset, &
                 hostarray, &
                 off1, off2, &
                num, direction, my_stream, doSyncBefore, doSyncAfter, doSyncDefault)

      use iso_c_binding
      use elpa_gpu
      use elpa_utilities

      implicit none
      integer(kind=c_intptr_t), intent(in)          :: my_stream
      logical                                       :: successGPU
      character(len=*)                              :: errormessage
      integer(kind=c_intptr_t)                      :: devPtr
      integer(kind=c_intptr_t)                      :: devPtrOffset

      integer(kind=ik), intent(in)                  :: off1
      integer(kind=ik), intent(in)                  :: off2
      logical                                       :: doSyncBefore
      logical                                       :: doSyncAfter
      logical                                       :: doSyncDefault

      complex(kind=c_double_complex)                :: &
                                         hostArray(:,:)
      integer(kind=c_intptr_t)                      :: num
      integer(kind=c_int), intent(in)               :: direction

      if (doSyncBefore) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif

      if (direction .eq. gpuMemcpyHostToDevice) then
        successGPU = gpu_memcpy_async(devPtr+devPtrOffset, &
                    int(loc(hostarray(off1,off2)),kind=c_intptr_t), &
                                      num, direction, my_stream)
      else if (direction .eq. gpuMemcpyDeviceToHost) then
        successGPU = gpu_memcpy_async(&
                    int(loc(hostarray(off1,off2)),kind=c_intptr_t), &
                                      devPtr+devPtrOffset, num, direction, my_stream)
      else
        print *,"gpu_memcpy_async_and_stream_synchronize: unknown error"
        stop 1
      endif

      if (doSyncAfter) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
      if (doSyncDefault) then
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
    end


    subroutine gpu_memcpy_async_and_stream_synchronize_complex_single_2d &
                (errormessage, devPtr, devPtrOffset, &
                 hostarray, &
                 off1, off2, &
                num, direction, my_stream, doSyncBefore, doSyncAfter, doSyncDefault)

      use iso_c_binding
      use elpa_gpu
      use elpa_utilities

      implicit none
      integer(kind=c_intptr_t), intent(in)          :: my_stream
      logical                                       :: successGPU
      character(len=*)                              :: errormessage
      integer(kind=c_intptr_t)                      :: devPtr
      integer(kind=c_intptr_t)                      :: devPtrOffset

      integer(kind=ik), intent(in)                  :: off1
      integer(kind=ik), intent(in)                  :: off2
      logical                                       :: doSyncBefore
      logical                                       :: doSyncAfter
      logical                                       :: doSyncDefault

      complex(kind=c_float_complex)                 :: &
                                         hostArray(:,:)
      integer(kind=c_intptr_t)                      :: num
      integer(kind=c_int), intent(in)               :: direction

      if (doSyncBefore) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif

      if (direction .eq. gpuMemcpyHostToDevice) then
        successGPU = gpu_memcpy_async(devPtr+devPtrOffset, &
                    int(loc(hostarray(off1,off2)),kind=c_intptr_t), &
                                      num, direction, my_stream)
      else if (direction .eq. gpuMemcpyDeviceToHost) then
        successGPU = gpu_memcpy_async(&
                    int(loc(hostarray(off1,off2)),kind=c_intptr_t), &
                                      devPtr+devPtrOffset, num, direction, my_stream)
      else
        print *,"gpu_memcpy_async_and_stream_synchronize: unknown error"
        stop 1
      endif

      if (doSyncAfter) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
      if (doSyncDefault) then
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
    end


    subroutine gpu_memcpy_async_and_stream_synchronize_double_3d &
                (errormessage, devPtr, devPtrOffset, &
                 hostarray, &
                 off1, off2, off3, &
                num, direction, my_stream, doSyncBefore, doSyncAfter, doSyncDefault)

      use iso_c_binding
      use elpa_gpu
      use elpa_utilities

      implicit none
      integer(kind=c_intptr_t), intent(in)          :: my_stream
      logical                                       :: successGPU
      character(len=*)                              :: errormessage
      integer(kind=c_intptr_t)                      :: devPtr
      integer(kind=c_intptr_t)                      :: devPtrOffset

      integer(kind=ik), intent(in)                  :: off1
      integer(kind=ik), intent(in)                  :: off2
      integer(kind=ik), intent(in)                  :: off3
      logical                                       :: doSyncBefore
      logical                                       :: doSyncAfter
      logical                                       :: doSyncDefault

      real(kind=c_double)                           :: &
                                         hostArray(:,:,:)
      integer(kind=c_intptr_t)                      :: num
      integer(kind=c_int), intent(in)               :: direction

      if (doSyncBefore) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif

      if (direction .eq. gpuMemcpyHostToDevice) then
        successGPU = gpu_memcpy_async(devPtr+devPtrOffset, &
                    int(loc(hostarray(off1,off2,off3)),kind=c_intptr_t), &
                                      num, direction, my_stream)
      else if (direction .eq. gpuMemcpyDeviceToHost) then
        successGPU = gpu_memcpy_async(&
                    int(loc(hostarray(off1,off2,off3)),kind=c_intptr_t), &
                                      devPtr+devPtrOffset, num, direction, my_stream)
      else
        print *,"gpu_memcpy_async_and_stream_synchronize: unknown error"
        stop 1
      endif

      if (doSyncAfter) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
      if (doSyncDefault) then
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
    end


    subroutine gpu_memcpy_async_and_stream_synchronize_single_3d &
                (errormessage, devPtr, devPtrOffset, &
                 hostarray, &
                 off1, off2, off3, &
                num, direction, my_stream, doSyncBefore, doSyncAfter, doSyncDefault)

      use iso_c_binding
      use elpa_gpu
      use elpa_utilities

      implicit none
      integer(kind=c_intptr_t), intent(in)          :: my_stream
      logical                                       :: successGPU
      character(len=*)                              :: errormessage
      integer(kind=c_intptr_t)                      :: devPtr
      integer(kind=c_intptr_t)                      :: devPtrOffset

      integer(kind=ik), intent(in)                  :: off1
      integer(kind=ik), intent(in)                  :: off2
      integer(kind=ik), intent(in)                  :: off3
      logical                                       :: doSyncBefore
      logical                                       :: doSyncAfter
      logical                                       :: doSyncDefault

      real(kind=c_float)                            :: &
                                         hostArray(:,:,:)
      integer(kind=c_intptr_t)                      :: num
      integer(kind=c_int), intent(in)               :: direction

      if (doSyncBefore) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif

      if (direction .eq. gpuMemcpyHostToDevice) then
        successGPU = gpu_memcpy_async(devPtr+devPtrOffset, &
                    int(loc(hostarray(off1,off2,off3)),kind=c_intptr_t), &
                                      num, direction, my_stream)
      else if (direction .eq. gpuMemcpyDeviceToHost) then
        successGPU = gpu_memcpy_async(&
                    int(loc(hostarray(off1,off2,off3)),kind=c_intptr_t), &
                                      devPtr+devPtrOffset, num, direction, my_stream)
      else
        print *,"gpu_memcpy_async_and_stream_synchronize: unknown error"
        stop 1
      endif

      if (doSyncAfter) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
      if (doSyncDefault) then
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
    end


    subroutine gpu_memcpy_async_and_stream_synchronize_complex_3d &
                (errormessage, devPtr, devPtrOffset, &
                 hostarray, &
                 off1, off2, off3, &
                num, direction, my_stream, doSyncBefore, doSyncAfter, doSyncDefault)

      use iso_c_binding
      use elpa_gpu
      use elpa_utilities

      implicit none
      integer(kind=c_intptr_t), intent(in)          :: my_stream
      logical                                       :: successGPU
      character(len=*)                              :: errormessage
      integer(kind=c_intptr_t)                      :: devPtr
      integer(kind=c_intptr_t)                      :: devPtrOffset

      integer(kind=ik), intent(in)                  :: off1
      integer(kind=ik), intent(in)                  :: off2
      integer(kind=ik), intent(in)                  :: off3
      logical                                       :: doSyncBefore
      logical                                       :: doSyncAfter
      logical                                       :: doSyncDefault

      complex(kind=c_double_complex)                :: &
                                         hostArray(:,:,:)
      integer(kind=c_intptr_t)                      :: num
      integer(kind=c_int), intent(in)               :: direction

      if (doSyncBefore) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif

      if (direction .eq. gpuMemcpyHostToDevice) then
        successGPU = gpu_memcpy_async(devPtr+devPtrOffset, &
                    int(loc(hostarray(off1,off2,off3)),kind=c_intptr_t), &
                                      num, direction, my_stream)
      else if (direction .eq. gpuMemcpyDeviceToHost) then
        successGPU = gpu_memcpy_async(&
                    int(loc(hostarray(off1,off2,off3)),kind=c_intptr_t), &
                                      devPtr+devPtrOffset, num, direction, my_stream)
      else
        print *,"gpu_memcpy_async_and_stream_synchronize: unknown error"
        stop 1
      endif

      if (doSyncAfter) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
      if (doSyncDefault) then
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
    end


    subroutine gpu_memcpy_async_and_stream_synchronize_complex_single_3d &
                (errormessage, devPtr, devPtrOffset, &
                 hostarray, &
                 off1, off2, off3, &
                num, direction, my_stream, doSyncBefore, doSyncAfter, doSyncDefault)

      use iso_c_binding
      use elpa_gpu
      use elpa_utilities

      implicit none
      integer(kind=c_intptr_t), intent(in)          :: my_stream
      logical                                       :: successGPU
      character(len=*)                              :: errormessage
      integer(kind=c_intptr_t)                      :: devPtr
      integer(kind=c_intptr_t)                      :: devPtrOffset

      integer(kind=ik), intent(in)                  :: off1
      integer(kind=ik), intent(in)                  :: off2
      integer(kind=ik), intent(in)                  :: off3
      logical                                       :: doSyncBefore
      logical                                       :: doSyncAfter
      logical                                       :: doSyncDefault

      complex(kind=c_float_complex)                 :: &
                                         hostArray(:,:,:)
      integer(kind=c_intptr_t)                      :: num
      integer(kind=c_int), intent(in)               :: direction

      if (doSyncBefore) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif

      if (direction .eq. gpuMemcpyHostToDevice) then
        successGPU = gpu_memcpy_async(devPtr+devPtrOffset, &
                    int(loc(hostarray(off1,off2,off3)),kind=c_intptr_t), &
                                      num, direction, my_stream)
      else if (direction .eq. gpuMemcpyDeviceToHost) then
        successGPU = gpu_memcpy_async(&
                    int(loc(hostarray(off1,off2,off3)),kind=c_intptr_t), &
                                      devPtr+devPtrOffset, num, direction, my_stream)
      else
        print *,"gpu_memcpy_async_and_stream_synchronize: unknown error"
        stop 1
      endif

      if (doSyncAfter) then
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
      if (doSyncDefault) then
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu(trim(errormessage), successGPU)
      endif
    end

