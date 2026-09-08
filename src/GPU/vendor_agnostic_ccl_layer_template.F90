#if 0
!    Copyright 2024, A. Marek, MPCDF
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
#endif


  integer(kind=c_int) :: cclSum
  integer(kind=c_int) :: cclMax
  integer(kind=c_int) :: cclMin
  integer(kind=c_int) :: cclAvg
  integer(kind=c_int) :: cclProd

  integer(kind=c_int) :: cclInt
  integer(kind=c_int) :: cclInt32
  integer(kind=c_int) :: cclInt64
  integer(kind=c_int) :: cclFloat
  integer(kind=c_int) :: cclFloat32
  integer(kind=c_int) :: cclFloat64
  integer(kind=c_int) :: cclDouble

!  !type, BIND(C,name="ncclUniqueId") :: uniqueId_c
!  type, BIND(C) ::ncclUniqueId
!  !type :: uniqueId_c
!    CHARACTER(KIND=C_CHAR) :: str(128)
!   end type

  interface ccl_Allreduce
    module procedure ccl_allreduce_intptr
    module procedure ccl_allreduce_cptr
  end interface

  interface ccl_Reduce
    module procedure ccl_reduce_intptr
    module procedure ccl_reduce_cptr
  end interface

  interface ccl_Bcast
    module procedure ccl_Bcast_intptr
    module procedure ccl_Bcast_cptr
  end interface

 interface ccl_Send
    module procedure ccl_Send_intptr
    module procedure ccl_Send_cptr
  end interface

  interface ccl_Recv
    module procedure ccl_Recv_intptr
    module procedure ccl_Recv_cptr
  end interface

  interface ccl_Sendrecv
    module procedure ccl_Sendrecv_intptr
    !module procedure ccl_Sendrecv_cptr
  end interface

  interface ccl_Isend
    module procedure ccl_Isend_intptr
    !module procedure ccl_Isend_cptr
  end interface

  interface ccl_Irecv
    module procedure ccl_Irecv_intptr
    !module procedure ccl_Irecv_cptr
  end interface

  contains

    function ccl_redOp_cclSum() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = nccl_redOp_ncclSum()
#endif

#ifdef WITH_AMD_RCCL
      flag = rccl_redOp_ncclSum()
#endif

#ifdef WITH_ONEAPI_ONECCL
      flag = oneccl_redOp_onecclSum()
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      flag = mpiccl_redOp_mpicclSum()
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif

    end function

    function ccl_redOp_cclMax() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = nccl_redOp_ncclMax()
#endif

#ifdef WITH_AMD_RCCL
      flag = rccl_redOp_ncclMax()
#endif

#ifdef WITH_ONEAPI_ONECCL
      flag = oneccl_redOp_onecclMax()
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      flag = mpiccl_redOp_mpicclMax()
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif

    end function

    function ccl_redOp_cclMin() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = nccl_redOp_ncclMin()
#endif

#ifdef WITH_AMD_RCCL
      flag = rccl_redOp_ncclMin()
#endif

#ifdef WITH_ONEAPI_ONECCL
      flag = oneccl_redOp_onecclMin()
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      flag = mpiccl_redOp_mpicclMin()
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif

    end function

    function ccl_redOp_cclAvg() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = nccl_redOp_ncclAvg()
#endif

#ifdef WITH_AMD_RCCL
      flag = rccl_redOp_ncclAvg()
#endif

#ifdef WITH_ONEAPI_ONECCL
      flag = oneccl_redOp_onecclAvg()
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      flag = mpiccl_redOp_mpicclAvg()
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif

    end function

    function ccl_redOp_cclProd() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = nccl_redOp_ncclProd()
#endif

#ifdef WITH_AMD_RCCL
      flag = rccl_redOp_ncclProd()
#endif

#ifdef WITH_ONEAPI_ONECCL
      flag = oneccl_redOp_onecclProd()
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      flag = mpiccl_redOp_mpicclProd()
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif

    end function

    function ccl_dataType_cclInt() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = nccl_dataType_ncclInt()
#endif

#ifdef WITH_AMD_RCCL
      flag = rccl_dataType_ncclInt()
#endif

#ifdef WITH_ONEAPI_ONECCL
      flag = oneccl_dataType_onecclInt()
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      flag = mpiccl_dataType_mpicclInt()
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif

    end function

    function ccl_dataType_cclInt32() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = nccl_dataType_ncclInt32()
#endif

#ifdef WITH_AMD_RCCL
      flag = rccl_dataType_ncclInt32()
#endif

#ifdef WITH_ONEAPI_ONECCL
      flag = oneccl_dataType_onecclInt32()
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      flag = mpiccl_dataType_mpicclInt32()
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif

    end function

    function ccl_dataType_cclInt64() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = nccl_dataType_ncclInt64()
#endif

#ifdef WITH_AMD_RCCL
      flag = rccl_dataType_ncclInt64()
#endif

#ifdef WITH_ONEAPI_ONECCL
      flag = oneccl_dataType_onecclInt64()
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      flag = mpiccl_dataType_mpicclInt64()
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif

    end function

    function ccl_dataType_cclFloat() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = nccl_dataType_ncclFloat()
#endif

#ifdef WITH_AMD_RCCL
      flag = rccl_dataType_ncclFloat()
#endif

#ifdef WITH_ONEAPI_ONECCL
      flag = oneccl_dataType_onecclFloat()
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      flag = mpiccl_dataType_mpicclFloat()
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif

    end function

    function ccl_dataType_cclFloat32() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = nccl_dataType_ncclFloat32()
#endif

#ifdef WITH_AMD_RCCL
      flag = rccl_dataType_ncclFloat32()
#endif

#ifdef WITH_ONEAPI_ONECCL
      flag = oneccl_dataType_onecclFloat32()
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      flag = mpiccl_dataType_mpicclFloat32()
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif

    end function

    function ccl_dataType_cclFloat64() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = nccl_dataType_ncclFloat64()
#endif

#ifdef WITH_AMD_RCCL
      flag = rccl_dataType_ncclFloat64()
#endif

#ifdef WITH_ONEAPI_ONECCL
      flag = oneccl_dataType_onecclFloat64()
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      flag = mpiccl_dataType_mpicclFloat64()
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif

    end function

    function ccl_dataType_cclDouble() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = nccl_dataType_ncclDouble()
#endif

#ifdef WITH_AMD_RCCL
      flag = rccl_dataType_ncclDouble()
#endif

#ifdef WITH_ONEAPI_ONECCL
      flag = oneccl_dataType_onecclDouble()
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      flag = mpiccl_dataType_mpicclDouble()
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif

    end function

    function ccl_group_start() result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      logical :: success
#ifdef WITH_NVIDIA_NCCL
      success = nccl_group_start()
#endif

#ifdef WITH_AMD_RCCL
      success = rccl_group_start()
#endif

#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_group_start()
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_group_start()
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif
    end function

    function ccl_group_end() result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      logical :: success
#ifdef WITH_NVIDIA_NCCL
      success = nccl_group_end()
#endif

#ifdef WITH_AMD_RCCL
      success = rccl_group_end()
#endif

#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_group_end()
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_group_end()
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif
    end function

    function ccl_get_unique_id(cclId) result(success)
      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_NCCL
      use nccl_functions, only : ncclUniqueId
#endif
#ifdef WITH_AMD_RCCL
      use rccl_functions, only : ncclUniqueId
#endif
#ifdef WITH_ONEAPI_ONECCL
      use oneccl_functions, only : ncclUniqueId
#endif
#ifdef WITH_GPU_AWARE_MPICCL
      use mpiccl_functions, only : ncclUniqueId
#endif
      implicit none
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL) || defined(WITH_ONEAPI_ONECCL) || defined(WITH_GPU_AWARE_MPICCL)
      type(ncclUniqueId)  :: cclId
#else
      ! dummy argument
      integer(kind=c_intptr_t) :: cclId
#endif
      logical             :: success
      integer :: i
#ifdef WITH_NVIDIA_NCCL
      success = nccl_get_unique_id(cclId)
#endif
#ifdef WITH_AMD_RCCL
      success = rccl_get_unique_id(cclId)
#endif

#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_get_unique_id(cclId)
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_get_unique_id(cclId)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif
    end function

    function ccl_comm_init_rank(cclComm, nRanks, cclId, myRank) result(success)
      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_NCCL
      use nccl_functions, only : ncclUniqueId
#endif
#ifdef WITH_AMD_RCCL
      use rccl_functions, only : ncclUniqueId
#endif
#ifdef WITH_ONEAPI_ONECCL
      use oneccl_functions, only : ncclUniqueId
#endif
#ifdef WITH_GPU_AWARE_MPICCL
      use mpiccl_functions, only : ncclUniqueId
#endif
      implicit none
      integer(kind=C_intptr_T)                  :: cclComm
      integer(kind=c_int)                       :: nRanks
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL) || defined(WITH_ONEAPI_ONECCL) || defined(WITH_GPU_AWARE_MPICCL)
      type(ncclUniqueId)                        :: cclId
#else
      ! dummy argument
      integer(kind=c_intptr_t) :: cclId
#endif
      integer(kind=c_int)                       :: myRank
      logical                                   :: success

#ifdef WITH_NVIDIA_NCCL
      success = nccl_comm_init_rank(cclComm, nRanks, cclId, myRank)
#endif
#ifdef WITH_AMD_RCCL
      success = rccl_comm_init_rank(cclComm, nRanks, cclId, myRank)
#endif
#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_comm_init_rank(cclComm, nRanks, cclId, myRank)
#endif
#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_comm_init_rank(cclComm, nRanks, cclId, myRank)
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif
    end function

 ! only for version >= 2.13
 !    function nccl_comm_finalize(ncclComm) result(success)
 !      use, intrinsic :: iso_c_binding
 !      implicit none
 !      integer(kind=C_intptr_t)                  :: ncclComm
 !      logical                                   :: success
 !#ifdef WITH_NVIDIA_NCCL
 !      success = nccl_comm_finalize_c(ncclComm) /= 0
 !#else
 !      success = .true.
 !#endif
 !    end function

    function ccl_comm_destroy(cclComm) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: cclComm
      logical                                   :: success
#ifdef WITH_NVIDIA_NCCL
      success = nccl_comm_destroy(cclComm)
#endif

#ifdef WITH_AMD_RCCL
      success = rccl_comm_destroy(cclComm)
#endif

#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_comm_destroy(cclComm)
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_comm_destroy(cclComm)
#endif
    end function

    function ccl_allreduce_intptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, cclComm, gpuStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: sendbuff
      integer(kind=C_intptr_t)                  :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: cclDatatype
      integer(kind=c_int)                       :: cclOp
      integer(kind=C_intptr_t)                  :: cclComm
      integer(kind=C_intptr_t)                  :: gpuStream
      logical                                   :: success

#ifdef WITH_NVIDIA_NCCL
      success = nccl_allreduce_intptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, cclComm, gpuStream)
#endif

#ifdef WITH_AMD_RCCL
      success = rccl_allreduce_intptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, cclComm, gpuStream)
#endif

#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_allreduce_intptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, cclComm, gpuStream)
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_allreduce_intptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, cclComm, gpuStream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif

    end function

    function ccl_allreduce_cptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, cclComm, gpuStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                               :: sendbuff
      type(c_ptr)                               :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: cclDatatype
      integer(kind=c_int)                       :: cclOp
      integer(kind=C_intptr_t)                  :: cclComm
      integer(kind=C_intptr_t)                  :: gpuStream
      logical                                   :: success

#ifdef WITH_NVIDIA_NCCL
      success = nccl_allreduce_cptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, cclComm, gpuStream)
#endif

#ifdef WITH_AMD_RCCL
      success = rccl_allreduce_cptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, cclComm, gpuStream)
#endif

#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_allreduce_cptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, cclComm, gpuStream)
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_allreduce_cptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, cclComm, gpuStream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif
    end function

    function ccl_reduce_intptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, root, cclComm, gpuStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: sendbuff
      integer(kind=C_intptr_t)                  :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: cclDatatype
      integer(kind=c_int)                       :: cclOp
      integer(kind=c_int)                       :: root
      integer(kind=C_intptr_t)                  :: cclComm
      integer(kind=C_intptr_t)                  :: gpuStream
      logical                                   :: success

#ifdef WITH_NVIDIA_NCCL
      success = nccl_reduce_intptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, root, cclComm, gpuStream)
#endif

#ifdef WITH_AMD_RCCL
      success = rccl_reduce_intptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, root, cclComm, gpuStream)
#endif

#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_reduce_intptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, root, cclComm, gpuStream)
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_reduce_intptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, root, cclComm, gpuStream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif
    end function

    function ccl_reduce_cptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, root, cclComm, gpuStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)               :: sendbuff
      type(c_ptr)               :: recvbuff
      integer(kind=c_size_t)    :: nrElements
      integer(kind=c_int)       :: cclDatatype
      integer(kind=c_int)       :: cclOp
      integer(kind=c_int)       :: root
      integer(kind=C_intptr_t)  :: cclComm
      integer(kind=C_intptr_t)  :: gpuStream
      logical                   :: success

#ifdef WITH_NVIDIA_NCCL
      success = nccl_reduce_cptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, root, cclComm, gpuStream)
#endif

#ifdef WITH_AMD_RCCL
      success = rccl_reduce_cptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, root, cclComm, gpuStream)
#endif

#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_reduce_cptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, root, cclComm, gpuStream)
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_reduce_cptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, root, cclComm, gpuStream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif
    end function

    function ccl_bcast_intptr(sendbuff, recvbuff, nrElements, cclDatatype, root, cclComm, gpuStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: sendbuff
      integer(kind=C_intptr_t)                  :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: cclDatatype
      integer(kind=c_int)                       :: root
      integer(kind=C_intptr_t)                  :: cclComm
      integer(kind=C_intptr_t)                  :: gpuStream
      logical                                   :: success

#ifdef WITH_NVIDIA_NCCL
      success = nccl_bcast_intptr(sendbuff, recvbuff, nrElements, cclDatatype, root, cclComm, gpuStream)
#endif

#ifdef WITH_AMD_RCCL
      success = rccl_bcast_intptr(sendbuff, recvbuff, nrElements, cclDatatype, root, cclComm, gpuStream)
#endif

#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_bcast_intptr(sendbuff, recvbuff, nrElements, cclDatatype, root, cclComm, gpuStream)
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_bcast_intptr(sendbuff, recvbuff, nrElements, cclDatatype, root, cclComm, gpuStream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif
    end function

    function ccl_bcast_cptr(sendbuff, recvbuff, nrElements, cclDatatype, root, cclComm, gpuStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                  :: sendbuff
      type(c_ptr)                  :: recvbuff
      integer(kind=c_size_t)       :: nrElements
      integer(kind=c_int)          :: cclDatatype
      integer(kind=c_int)          :: root
      integer(kind=C_intptr_t)     :: cclComm
      integer(kind=C_intptr_t)     :: gpuStream
      logical                      :: success

#ifdef WITH_NVIDIA_NCCL
      success = nccl_bcast_cptr(sendbuff, recvbuff, nrElements, cclDatatype, root, cclComm, gpuStream)
#endif

#ifdef WITH_AMD_RCCL
      success = rccl_bcast_cptr(sendbuff, recvbuff, nrElements, cclDatatype, root, cclComm, gpuStream)
#endif

#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_bcast_cptr(sendbuff, recvbuff, nrElements, cclDatatype, root, cclComm, gpuStream)
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_bcast_cptr(sendbuff, recvbuff, nrElements, cclDatatype, root, cclComm, gpuStream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif
    end function

    function ccl_send_intptr(sendbuff, nrElements, cclDatatype, peer, cclComm, gpuStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)       :: sendbuff
      integer(kind=c_size_t)         :: nrElements
      integer(kind=c_int)            :: cclDatatype
      integer(kind=c_int)            :: peer
      integer(kind=C_intptr_t)       :: cclComm
      integer(kind=C_intptr_t)       :: gpuStream
      logical                        :: success

#ifdef WITH_NVIDIA_NCCL
      success = nccl_send_intptr(sendbuff, nrElements, cclDatatype, peer, cclComm, gpuStream)
#endif

#ifdef WITH_AMD_RCCL
      success = rccl_send_intptr(sendbuff, nrElements, cclDatatype, peer, cclComm, gpuStream)
#endif

#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_send_intptr(sendbuff, nrElements, cclDatatype, peer, cclComm, gpuStream)
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_send_intptr(sendbuff, nrElements, cclDatatype, peer, cclComm, gpuStream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif
    end function

    function ccl_send_cptr(sendbuff, nrElements, cclDatatype, peer, cclComm, gpuStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                    :: sendbuff
      integer(kind=c_size_t)         :: nrElements
      integer(kind=c_int)            :: cclDatatype
      integer(kind=c_int)            :: peer
      integer(kind=C_intptr_t)       :: cclComm
      integer(kind=C_intptr_t)       :: gpuStream
      logical                        :: success

#ifdef WITH_NVIDIA_NCCL
      success = nccl_send_cptr(sendbuff, nrElements, cclDatatype, peer, cclComm, gpuStream)
#endif

#ifdef WITH_AMD_RCCL
      success = rccl_send_cptr(sendbuff, nrElements, cclDatatype, peer, cclComm, gpuStream)
#endif

#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_send_cptr(sendbuff, nrElements, cclDatatype, peer, cclComm, gpuStream)
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_send_cptr(sendbuff, nrElements, cclDatatype, peer, cclComm, gpuStream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif
    end function

    function ccl_recv_intptr(recvbuff, nrElements, cclDatatype, peer, cclComm, gpuStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)      :: recvbuff
      integer(kind=c_size_t)        :: nrElements
      integer(kind=c_int)           :: cclDatatype
      integer(kind=c_int)           :: peer
      integer(kind=C_intptr_t)      :: cclComm
      integer(kind=C_intptr_t)      :: gpuStream
      logical                       :: success

#ifdef WITH_NVIDIA_NCCL
      success = nccl_recv_intptr(recvbuff, nrElements, cclDatatype, peer, cclComm, gpuStream)
#endif

#ifdef WITH_AMD_RCCL
      success = rccl_recv_intptr(recvbuff, nrElements, cclDatatype, peer, cclComm, gpuStream)
#endif

#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_recv_intptr(recvbuff, nrElements, cclDatatype, peer, cclComm, gpuStream)
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_recv_intptr(recvbuff, nrElements, cclDatatype, peer, cclComm, gpuStream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif
    end function

    function ccl_recv_cptr(recvbuff, nrElements, cclDatatype, peer, cclComm, gpuStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                   :: recvbuff
      integer(kind=c_size_t)        :: nrElements
      integer(kind=c_int)           :: cclDatatype
      integer(kind=c_int)           :: peer
      integer(kind=C_intptr_t)      :: cclComm
      integer(kind=C_intptr_t)      :: gpuStream
      logical                       :: success

#ifdef WITH_NVIDIA_NCCL
      success = nccl_recv_cptr(recvbuff, nrElements, cclDatatype, peer, cclComm, gpuStream)
#endif

#ifdef WITH_AMD_RCCL
      success = rccl_recv_cptr(recvbuff, nrElements, cclDatatype, peer, cclComm, gpuStream)
#endif

#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_recv_cptr(recvbuff, nrElements, cclDatatype, peer, cclComm, gpuStream)
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_recv_cptr(recvbuff, nrElements, cclDatatype, peer, cclComm, gpuStream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop
#endif
    end function

    function ccl_sendrecv_intptr(sendbuff , sendNrElements, sendCclDatatype, dest, &
                                 recvbuff, recvNrElements, recvCclDatatype, source, cclComm, gpuStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=c_intptr_t)      :: sendbuff, recvbuff
      integer(kind=c_size_t)        :: sendNrElements, recvNrElements
      integer(kind=c_int)           :: sendCclDatatype, recvCclDatatype
      integer(kind=c_int)           :: dest, source
      integer(kind=c_intptr_t)      :: cclComm
      integer(kind=c_intptr_t)      :: gpuStream
      logical                       :: success

#ifdef WITH_NVIDIA_NCCL
      print *,"Not yet implemented"
      stop 1
#endif

#ifdef WITH_AMD_RCCL
      print *,"Not yet implemented"
      stop 1
#endif

#ifdef WITH_ONEAPI_ONECCL
      print *,"Not yet implemented"
      stop 1
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_sendrecv_intptr(sendbuff, sendNrElements, sendCclDatatype, dest, &
                                       recvbuff, recvNrElements, recvCclDatatype, source, cclComm, gpuStream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop 1
#endif
    end function


    function ccl_isend_intptr(sendbuff, nrElements, cclDatatype, peer, cclComm, request, gpuStream) result(success)
      use, intrinsic :: iso_c_binding
      use precision, only : MPI_KIND
      implicit none

      integer(kind=c_intptr_t)       :: sendbuff
      integer(kind=c_size_t)         :: nrElements
      integer(kind=c_int)            :: cclDatatype
      integer(kind=c_int)            :: peer
      integer(kind=c_intptr_t)       :: cclComm
      integer(kind=MPI_KIND), intent(out) :: request
      integer(kind=c_intptr_t)       :: gpuStream
      logical                        :: success

#ifdef WITH_NVIDIA_NCCL
      print *,"Not yet implemented"
      stop 1
#endif

#ifdef WITH_AMD_RCCL
      print *,"Not yet implemented"
      stop 1
#endif

#ifdef WITH_ONEAPI_ONECCL
      print *,"Not yet implemented"
      stop 1
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_isend_intptr(sendbuff, nrElements, cclDatatype, peer, cclComm, request, gpuStream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop 1
#endif
    end function


    function ccl_irecv_intptr(recvbuff, nrElements, cclDatatype, peer, cclComm, request, gpuStream) result(success)
      use, intrinsic :: iso_c_binding
      use precision, only : MPI_KIND
      implicit none

      integer(kind=c_intptr_t)      :: recvbuff
      integer(kind=c_size_t)        :: nrElements
      integer(kind=c_int)           :: cclDatatype
      integer(kind=c_int)           :: peer
      integer(kind=c_intptr_t)      :: cclComm
      integer(kind=MPI_KIND), intent(out) :: request
      integer(kind=c_intptr_t)      :: gpuStream
      logical                       :: success

#ifdef WITH_NVIDIA_NCCL
      print *,"Not yet implemented"
      stop 1
#endif

#ifdef WITH_AMD_RCCL
      print *,"Not yet implemented"
      stop 1
#endif

#ifdef WITH_ONEAPI_ONECCL
      print *,"Not yet implemented"
      stop 1
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_irecv_intptr(recvbuff, nrElements, cclDatatype, peer, cclComm, request, gpuStream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop 1
#endif
    end function


    function ccl_waitall(count, requests, gpuStream) result(success)
      use, intrinsic :: iso_c_binding
      use precision, only : MPI_KIND
      implicit none

      integer(kind=c_int), intent(in)       :: count
      integer(kind=MPI_KIND), intent(inout) :: requests(count)
      integer(kind=c_intptr_t)      :: gpuStream
      logical                       :: success

#ifdef WITH_NVIDIA_NCCL
      print *,"Not yet implemented"
      stop 1
#endif

#ifdef WITH_AMD_RCCL
      print *,"Not yet implemented"
      stop 1
#endif

#ifdef WITH_ONEAPI_ONECCL
      print *,"Not yet implemented"
      stop 1
#endif

#ifdef WITH_GPU_AWARE_MPICCL
      success = mpiccl_waitall(count, requests, gpuStream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"Not yet implemented"
      stop 1
#endif
    end function
