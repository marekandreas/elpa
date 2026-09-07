!    Copyright 2025, P. Karpov
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
! Author: Peter Karpov, MPCDF

    
             
  !integer(kind=c_int) :: mpicclSum
  !integer(kind=c_int) :: mpicclMax
  !integer(kind=c_int) :: mpicclMin
  !integer(kind=c_int) :: mpicclAvg
  !integer(kind=c_int) :: mpicclProd

  !integer(kind=c_int) :: mpicclInt
  !integer(kind=c_int) :: mpicclInt32
  !integer(kind=c_int) :: mpicclInt64
  !integer(kind=c_int) :: mpicclFloat
  !integer(kind=c_int) :: mpicclFloat32
  !integer(kind=c_int) :: mpicclFloat64
  !integer(kind=c_int) :: mpicclDouble

type, BIND(C) :: ncclUniqueId
  CHARACTER(KIND=C_CHAR) :: str(128)
end type


interface
  function mpiccl_redOp_mpicclSum_c() result(flag) &
            bind(C, name="mpicclRedOpSumFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int) :: flag
  end function
end interface

interface
  function mpiccl_redOp_mpicclMax_c() result(flag) &
            bind(C, name="mpicclRedOpMaxFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int) :: flag
  end function
end interface

interface
  function mpiccl_redOp_mpicclMin_c() result(flag) &
            bind(C, name="mpicclRedOpMinFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int) :: flag
  end function
end interface

interface
  function mpiccl_redOp_mpicclAvg_c() result(flag) &
            bind(C, name="mpicclRedOpAvgFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int) :: flag
  end function
end interface

interface
  function mpiccl_redOp_mpicclProd_c() result(flag) &
            bind(C, name="mpicclRedOpProdFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int) :: flag
  end function
end interface

interface
  function mpiccl_dataType_mpicclInt_c() result(flag) &
            bind(C, name="mpicclDataTypeMpicclIntFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int) :: flag
  end function
end interface

interface
  function mpiccl_dataType_mpicclInt32_c() result(flag) &
            bind(C, name="mpicclDataTypeMpicclInt32FromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int) :: flag
  end function
end interface

interface
  function mpiccl_dataType_mpicclInt64_c() result(flag) &
            bind(C, name="mpicclDataTypeMpicclInt64FromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int) :: flag
  end function
end interface

interface
  function mpiccl_dataType_mpicclFloat_c() result(flag) &
            bind(C, name="mpicclDataTypeMpicclFloatFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int) :: flag
  end function
end interface

interface
  function mpiccl_dataType_mpicclFloat32_c() result(flag) &
            bind(C, name="mpicclDataTypeMpicclFloat32FromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int) :: flag
  end function
end interface

interface
  function mpiccl_dataType_mpicclFloat64_c() result(flag) &
            bind(C, name="mpicclDataTypeMpicclFloat64FromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int) :: flag
  end function
end interface

interface
  function mpiccl_dataType_mpicclDouble_c() result(flag) &
            bind(C, name="mpicclDataTypeMpicclDoubleFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int) :: flag
  end function
end interface

interface  
  function mpiccl_group_start_c() result(istat) &
            bind(C, name="mpicclGroupStartFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_INT)      :: istat
  end function
end interface


interface  
  function mpiccl_group_end_c() result(istat) &
            bind(C, name="mpicclGroupEndFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_INT)      :: istat
  end function
end interface

interface
  function mpiccl_get_unique_id_c(mpicclId) result(istat) &
            bind(C, name="mpicclGetUniqueIdFromC")
    use, intrinsic :: iso_c_binding
    import :: ncclUniqueId
    implicit none
    !integer(kind=C_intptr_T) :: mpicclId(16)
    !integer(kind=C_intptr_T) :: mpicclId
    type(ncclUniqueId)            :: mpicclId
    !character(len=128)        :: mpicclId
    integer(kind=C_INT)      :: istat
  end function
end interface

interface
  function mpiccl_request_size_c() result(request_size) &
            bind(C, name="mpicclRequestSizeFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_size_t) :: request_size
  end function
end interface

interface
  function mpiccl_comm_init_rank_c(mpicclComm, nRanks, mpicclId, myRank) result(istat) &
            bind(C, name="mpicclCommInitRankFromC")
    use, intrinsic :: iso_c_binding
    import :: ncclUniqueId
    implicit none
    integer(kind=C_intptr_T)        :: mpicclComm
    integer(kind=c_int), value      :: nRanks
    !should be value, not possible since dimension trick
    !integer(kind=c_intptr_t), value :: mpicclId(16)
    !integer(kind=c_intptr_t)        :: mpicclId(16)
    !integer(kind=c_intptr_t), value :: mpicclId
    type(ncclUniqueId)            :: mpicclId
    integer(kind=c_int), value      :: myRank
    integer(kind=C_INT)             :: istat
  end function
end interface
  
! only for version >=2.13  
!interface
!  function mpiccl_comm_finalize_c(mpicclComm) result(istat) &
!           bind(C, name="mpicclCommFinalizeFromC")
!    use, intrinsic :: iso_c_binding
!    implicit none
!    integer(kind=C_intptr_T), value :: mpicclComm
!    integer(kind=C_INT)             :: istat  
!  end function                                
!end interface
    
interface
  function mpiccl_comm_destroy_c(mpicclComm) result(istat) &
            bind(C, name="mpicclCommDestroyFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_intptr_T), value :: mpicclComm
    integer(kind=C_INT)             :: istat  
  end function                                
end interface

interface mpiccl_Allreduce
  module procedure mpiccl_allreduce_intptr
  module procedure mpiccl_allreduce_cptr
end interface


interface
  function mpiccl_allreduce_intptr_c(sendbuff, recvbuff, nrElements, mpicclDatatype, mpicclOp, mpicclComm, &
                                     gpuStream) result(istat) &
                                     bind(C, name="mpicclAllReduceFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_intptr_t), value              :: sendbuff
    integer(kind=C_intptr_t), value              :: recvbuff
    integer(kind=c_size_t), intent(in), value    :: nrElements
    integer(kind=C_INT), intent(in), value       :: mpicclDatatype
    integer(kind=C_INT), intent(in), value       :: mpicclOp
    integer(kind=C_intptr_t), value              :: mpicclComm
    integer(kind=C_intptr_t), value              :: gpuStream
    integer(kind=C_INT)                          :: istat
  end function
end interface

interface
  function mpiccl_allreduce_cptr_c(sendbuff, recvbuff, nrElements, mpicclDatatype, mpicclOp, mpicclComm, &
                                   gpuStream) result(istat) &
                                   bind(C, name="mpicclAllReduceFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    type(c_ptr), value                           :: sendbuff
    type(c_ptr), value                           :: recvbuff
    integer(kind=c_size_t), intent(in), value    :: nrElements
    integer(kind=C_INT), intent(in), value       :: mpicclDatatype
    integer(kind=C_INT), intent(in), value       :: mpicclOp
    integer(kind=C_intptr_t), value              :: mpicclComm
    integer(kind=C_intptr_t), value              :: gpuStream
    integer(kind=C_INT)                          :: istat
  end function
end interface

interface mpiccl_reduce
  module procedure mpiccl_reduce_intptr
  module procedure mpiccl_reduce_cptr
end interface


interface
  function mpiccl_reduce_intptr_c(sendbuff, recvbuff, nrElements, mpicclDatatype, mpicclOp, root, mpicclComm, &
                                  gpuStream) result(istat) &
                                  bind(C, name="mpicclReduceFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_intptr_t), value              :: sendbuff
    integer(kind=C_intptr_t), value              :: recvbuff
    integer(kind=c_size_t), intent(in), value    :: nrElements
    integer(kind=C_INT), intent(in), value       :: mpicclDatatype
    integer(kind=C_INT), intent(in), value       :: mpicclOp
    integer(kind=C_INT), intent(in), value       :: root
    integer(kind=C_intptr_t), value              :: mpicclComm
    integer(kind=C_intptr_t), value              :: gpuStream
    integer(kind=C_INT)                          :: istat
  end function
end interface

interface
  function mpiccl_reduce_cptr_c(sendbuff, recvbuff, nrElements, mpicclDatatype, mpicclOp, root, mpicclComm, &
                                gpuStream) result(istat) &
                                bind(C, name="mpicclReduceFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    type(c_ptr), value                           :: sendbuff
    type(c_ptr), value                           :: recvbuff
    integer(kind=c_size_t), intent(in), value    :: nrElements
    integer(kind=C_INT), intent(in), value       :: mpicclDatatype
    integer(kind=C_INT), intent(in), value       :: mpicclOp
    integer(kind=C_INT), intent(in), value       :: root
    integer(kind=C_intptr_t), value              :: mpicclComm
    integer(kind=C_intptr_t), value              :: gpuStream
    integer(kind=C_INT)                          :: istat
  end function
end interface

interface mpiccl_Bcast
  module procedure mpiccl_Bcast_intptr
  module procedure mpiccl_Bcast_cptr
end interface


interface
  function mpiccl_bcast_intptr_c(sendbuff, recvbuff, nrElements, mpicclDatatype, root, mpicclComm, gpuStream) result(istat) &
            bind(C, name="mpicclBroadcastFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_intptr_t), value              :: sendbuff
    integer(kind=C_intptr_t), value              :: recvbuff
    integer(kind=c_size_t), intent(in), value    :: nrElements
    integer(kind=C_INT), intent(in), value       :: mpicclDatatype
    integer(kind=C_INT), intent(in), value       :: root
    integer(kind=C_intptr_t), value              :: mpicclComm
    integer(kind=C_intptr_t), value              :: gpuStream
    integer(kind=C_INT)                          :: istat
  end function
end interface

interface
  function mpiccl_bcast_cptr_c(sendbuff, recvbuff, nrElements, mpicclDatatype, root, mpicclComm, gpuStream) result(istat) &
            bind(C, name="mpicclBroadcastFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    type(c_ptr), value                           :: sendbuff
    type(c_ptr), value                           :: recvbuff
    integer(kind=c_size_t), intent(in), value    :: nrElements
    integer(kind=C_INT), intent(in), value       :: mpicclDatatype
    integer(kind=C_INT), intent(in), value       :: root
    integer(kind=C_intptr_t), value              :: mpicclComm
    integer(kind=C_intptr_t), value              :: gpuStream
    integer(kind=C_INT)                          :: istat
  end function
end interface

interface mpiccl_Send
  module procedure mpiccl_Send_intptr
  module procedure mpiccl_Send_cptr
end interface

interface
  function mpiccl_send_intptr_c(sendbuff, nrElements, mpicclDatatype, peer, mpicclComm, gpuStream) result(istat) &
            bind(C, name="mpicclSendFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_intptr_t), value              :: sendbuff
    integer(kind=c_size_t), intent(in), value    :: nrElements
    integer(kind=C_INT), intent(in), value       :: mpicclDatatype
    integer(kind=C_INT), intent(in), value       :: peer
    integer(kind=C_intptr_t), value              :: mpicclComm
    integer(kind=C_intptr_t), value              :: gpuStream
    integer(kind=C_INT)                          :: istat
  end function
end interface

interface
  function mpiccl_send_cptr_c(sendbuff, nrElements, mpicclDatatype, peer, mpicclComm, gpuStream) result(istat) &
            bind(C, name="mpicclSendFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    type(c_ptr), value                           :: sendbuff
    integer(kind=c_size_t), intent(in), value    :: nrElements
    integer(kind=C_INT), intent(in), value       :: mpicclDatatype
    integer(kind=C_INT), intent(in), value       :: peer
    integer(kind=C_intptr_t), value              :: mpicclComm
    integer(kind=C_intptr_t), value              :: gpuStream
    integer(kind=C_INT)                          :: istat
  end function
end interface

interface mpiccl_Recv
  module procedure mpiccl_Recv_intptr
  module procedure mpiccl_Recv_cptr
end interface

interface
  function mpiccl_recv_intptr_c(recvbuff, nrElements, mpicclDatatype, peer, mpicclComm, gpuStream) result(istat) &
            bind(C, name="mpicclRecvFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_intptr_t), value              :: recvbuff
    integer(kind=c_size_t), intent(in), value    :: nrElements
    integer(kind=C_INT), intent(in), value       :: mpicclDatatype
    integer(kind=C_INT), intent(in), value       :: peer
    integer(kind=C_intptr_t), value              :: mpicclComm
    integer(kind=C_intptr_t), value              :: gpuStream
    integer(kind=C_INT)                          :: istat
  end function
end interface

interface
  function mpiccl_recv_cptr_c(recvbuff, nrElements, mpicclDatatype, peer, mpicclComm, gpuStream) result(istat) &
            bind(C, name="mpicclRecvFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    type(c_ptr), value                           :: recvbuff
    integer(kind=c_size_t), intent(in), value    :: nrElements
    integer(kind=C_INT), intent(in), value       :: mpicclDatatype
    integer(kind=C_INT), intent(in), value       :: peer
    integer(kind=C_intptr_t), value              :: mpicclComm
    integer(kind=C_intptr_t), value              :: gpuStream
    integer(kind=C_INT)                          :: istat
  end function
end interface

interface mpiccl_Sendrecv
  module procedure mpiccl_Sendrecv_intptr
  !module procedure mpiccl_Send_cptr
end interface

interface
  function mpiccl_sendrecv_intptr_c(sendbuff, sendNrElements, sendMpicclDatatype, dest, &
                                    recvbuff, recvNrElements, recvMpicclDatatype, source, mpicclComm, gpuStream) result(istat) &
            bind(C, name="mpicclSendrecvFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_intptr_t), value              :: sendbuff, recvbuff
    integer(kind=c_size_t), intent(in), value    :: sendNrElements, recvNrElements
    integer(kind=c_int), intent(in), value       :: sendMpicclDatatype, recvMpicclDatatype
    integer(kind=c_int), intent(in), value       :: dest, source
    integer(kind=c_intptr_t), value              :: mpicclComm
    integer(kind=c_intptr_t), value              :: gpuStream
    integer(kind=c_int)                          :: istat
  end function
end interface

interface mpiccl_Isend
  module procedure mpiccl_Isend_intptr
  !module procedure mpiccl_Isend_cptr
end interface

interface
  function mpiccl_isend_intptr_c(sendbuff, nrElements, mpicclDatatype, peer, mpicclComm, request, gpuStream) result(istat) &
            bind(C, name="mpicclIsendFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_intptr_t), value              :: sendbuff
    integer(kind=c_size_t), intent(in), value    :: nrElements
    integer(kind=c_int), intent(in), value       :: mpicclDatatype
    integer(kind=c_int), intent(in), value       :: peer
    integer(kind=c_intptr_t), value              :: mpicclComm
    integer(kind=c_intptr_t), value              :: request
    integer(kind=c_intptr_t), value              :: gpuStream
    integer(kind=c_int)                          :: istat
  end function
end interface

interface mpiccl_Irecv
  module procedure mpiccl_Irecv_intptr
  !module procedure mpiccl_Irecv_cptr
end interface

interface
  function mpiccl_irecv_intptr_c(recvbuff, nrElements, mpicclDatatype, peer, mpicclComm, request, gpuStream) result(istat) &
            bind(C, name="mpicclIrecvFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_intptr_t), value              :: recvbuff
    integer(kind=c_size_t), intent(in), value    :: nrElements
    integer(kind=c_int), intent(in), value       :: mpicclDatatype
    integer(kind=c_int), intent(in), value       :: peer
    integer(kind=c_intptr_t), value              :: mpicclComm
    integer(kind=c_intptr_t), value              :: request
    integer(kind=c_intptr_t), value              :: gpuStream
    integer(kind=c_int)                          :: istat
  end function
end interface


interface
  function mpiccl_waitall_c(count, requests, gpuStream) result(istat) &
            bind(C, name="mpicclWaitallFromC")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int), intent(in), value       :: count
    integer(kind=c_intptr_t), value              :: requests
    integer(kind=c_intptr_t), value              :: gpuStream
    integer(kind=c_int)                          :: istat
  end function
end interface


contains


  function mpiccl_redOp_mpicclSum() result(flag)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=ik) :: flag
#ifdef WITH_GPU_AWARE_MPICCL
    flag = int(mpiccl_redOp_mpicclSum_c())
#else
    flag = 0
#endif
  end function

  function mpiccl_redOp_mpicclMax() result(flag)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=ik) :: flag
#ifdef WITH_GPU_AWARE_MPICCL
    flag = int(mpiccl_redOp_mpicclMax_c())
#else
    flag = 0
#endif
  end function

  function mpiccl_redOp_mpicclMin() result(flag)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=ik) :: flag
#ifdef WITH_GPU_AWARE_MPICCL
    flag = int(mpiccl_redOp_mpicclMin_c())
#else
    flag = 0
#endif
  end function

  function mpiccl_redOp_mpicclAvg() result(flag)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=ik) :: flag
#ifdef WITH_GPU_AWARE_MPICCL
    flag = int(mpiccl_redOp_mpicclAvg_c())
#else
    flag = 0
#endif
  end function

  function mpiccl_redOp_mpicclProd() result(flag)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=ik) :: flag
#ifdef WITH_GPU_AWARE_MPICCL
    flag = int(mpiccl_redOp_mpicclProd_c())
#else
    flag = 0
#endif
  end function

  function mpiccl_dataType_mpicclInt() result(flag)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=ik) :: flag
#ifdef WITH_GPU_AWARE_MPICCL
    flag = int(mpiccl_dataType_mpicclInt_c())
#else
    flag = 0
#endif
  end function

  function mpiccl_dataType_mpicclInt32() result(flag)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=ik) :: flag
#ifdef WITH_GPU_AWARE_MPICCL
    flag = int(mpiccl_dataType_mpicclInt32_c())
#else
    flag = 0
#endif
  end function

  function mpiccl_dataType_mpicclInt64() result(flag)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=ik) :: flag
#ifdef WITH_GPU_AWARE_MPICCL
    flag = int(mpiccl_dataType_mpicclInt64_c())
#else
    flag = 0
#endif
  end function

  function mpiccl_dataType_mpicclFloat() result(flag)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=ik) :: flag
#ifdef WITH_GPU_AWARE_MPICCL
    flag = int(mpiccl_dataType_mpicclFloat_c())
#else
    flag = 0
#endif
  end function

  function mpiccl_dataType_mpicclFloat32() result(flag)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=ik) :: flag
#ifdef WITH_GPU_AWARE_MPICCL
    flag = int(mpiccl_dataType_mpicclFloat32_c())
#else
    flag = 0
#endif
  end function

  function mpiccl_dataType_mpicclFloat64() result(flag)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=ik) :: flag
#ifdef WITH_GPU_AWARE_MPICCL
    flag = int(mpiccl_dataType_mpicclFloat64_c())
#else
    flag = 0
#endif
  end function

  function mpiccl_dataType_mpicclDouble() result(flag)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=ik) :: flag
#ifdef WITH_GPU_AWARE_MPICCL
    flag = int(mpiccl_dataType_mpicclDouble_c())
#else
    flag = 0
#endif
  end function


  function mpiccl_group_start() result(success)
    use, intrinsic :: iso_c_binding
    implicit none
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_group_start_c() /= 0
#else
    success = .true.
#endif
  end function


  function mpiccl_group_end() result(success)
    use, intrinsic :: iso_c_binding
    implicit none
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_group_end_c() /= 0
#else
    success = .true.
#endif
  end function
  
  function mpiccl_get_unique_id(mpicclId) result(success)
    use, intrinsic :: iso_c_binding
    implicit none
    !integer(kind=C_intptr_t)                  :: mpicclId(16)
    type(ncclUniqueId)                        :: mpicclId
    !integer(kind=C_intptr_t)                  :: mpicclId
    !character(len=128)                        :: mpicclId
    logical                                   :: success
    integer :: i
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_get_unique_id_c(mpicclId) /= 0
#else 
    success = .true.
#endif
  end function

  function mpiccl_request_size() result(request_size)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_size_t) :: request_size
#ifdef WITH_GPU_AWARE_MPICCL
    request_size = mpiccl_request_size_c()
#else
    request_size = 0_c_size_t
#endif
  end function

  
  function mpiccl_comm_init_rank(mpicclComm, nRanks, mpicclId, myRank) result(success)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_intptr_T)                  :: mpicclComm
    integer(kind=c_int)                       :: nRanks
    !integer(kind=C_intptr_t)                  :: mpicclId(16)
    type(ncclUniqueId)                        :: mpicclId
    !integer(kind=C_intptr_t)                  :: mpicclId
    !character(len=128)                        :: mpicclID
    integer(kind=c_int)                       :: myRank
    logical                                   :: success

#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_comm_init_rank_c(mpicclComm, nRanks, mpicclId, myRank) /= 0
#else 
    success = .true.
#endif
  end function

! only for version >= 2.13    
!    function mpiccl_comm_finalize(mpicclComm) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: mpicclComm
!      logical                                   :: success
!#ifdef WITH_GPU_AWARE_MPICCL
!      success = mpiccl_comm_finalize_c(mpicclComm) /= 0
!#else
!      success = .true.
!#endif
!    end function

  function mpiccl_comm_destroy(mpicclComm) result(success)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_intptr_t)                  :: mpicclComm
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_comm_destroy_c(mpicclComm) /= 0
#else
    success = .true.
#endif
  end function

  function mpiccl_allreduce_intptr(sendbuff, recvbuff, nrElements, mpicclDatatype, mpicclOp, mpicclComm, &
                                   gpuStream) result(success)
    use, intrinsic :: iso_c_binding
    implicit none

    integer(kind=C_intptr_t)                  :: sendbuff
    integer(kind=C_intptr_t)                  :: recvbuff
    integer(kind=c_size_t)                    :: nrElements
    integer(kind=c_int)                       :: mpicclDatatype
    integer(kind=c_int)                       :: mpicclOp
    integer(kind=C_intptr_t)                  :: mpicclComm
    integer(kind=C_intptr_t)                  :: gpuStream
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_allreduce_intptr_c(sendbuff, recvbuff, nrElements, mpicclDatatype, mpicclOp, mpicclComm, gpuStream) /= 0
#else
    success = .true.
#endif
  end function

  function mpiccl_allreduce_cptr(sendbuff, recvbuff, nrElements, mpicclDatatype, mpicclOp, mpicclComm, gpuStream) result(success)
    use, intrinsic :: iso_c_binding
    implicit none

    type(c_ptr)                               :: sendbuff
    type(c_ptr)                               :: recvbuff
    integer(kind=c_size_t)                    :: nrElements
    integer(kind=c_int)                       :: mpicclDatatype
    integer(kind=c_int)                       :: mpicclOp
    integer(kind=C_intptr_t)                  :: mpicclComm
    integer(kind=C_intptr_t)                  :: gpuStream
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_allreduce_cptr_c(sendbuff, recvbuff, nrElements, mpicclDatatype, mpicclOp, mpicclComm, gpuStream) /= 0
#else
    success = .true.
#endif
  end function

  function mpiccl_reduce_intptr(sendbuff, recvbuff, nrElements, mpicclDatatype, mpicclOp, root, mpicclComm, &
                                gpuStream) result(success)
    use, intrinsic :: iso_c_binding
    implicit none

    integer(kind=C_intptr_t)                  :: sendbuff
    integer(kind=C_intptr_t)                  :: recvbuff
    integer(kind=c_size_t)                    :: nrElements
    integer(kind=c_int)                       :: mpicclDatatype
    integer(kind=c_int)                       :: mpicclOp
    integer(kind=c_int)                       :: root
    integer(kind=C_intptr_t)                  :: mpicclComm
    integer(kind=C_intptr_t)                  :: gpuStream
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_reduce_intptr_c(sendbuff, recvbuff, nrElements, mpicclDatatype, mpicclOp, root, mpicclComm, gpuStream) /= 0
#else
    success = .true.
#endif
  end function

  function mpiccl_reduce_cptr(sendbuff, recvbuff, nrElements, mpicclDatatype, mpicclOp, root, mpicclComm, &
                              gpuStream) result(success)
    use, intrinsic :: iso_c_binding
    implicit none

    type(c_ptr)                               :: sendbuff
    type(c_ptr)                               :: recvbuff
    integer(kind=c_size_t)                    :: nrElements
    integer(kind=c_int)                       :: mpicclDatatype
    integer(kind=c_int)                       :: mpicclOp
    integer(kind=c_int)                       :: root
    integer(kind=C_intptr_t)                  :: mpicclComm
    integer(kind=C_intptr_t)                  :: gpuStream
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_reduce_cptr_c(sendbuff, recvbuff, nrElements, mpicclDatatype, mpicclOp, root, mpicclComm, gpuStream) /= 0
#else
    success = .true.
#endif
  end function

  function mpiccl_bcast_intptr(sendbuff, recvbuff, nrElements, mpicclDatatype, root, mpicclComm, gpuStream) result(success)
    use, intrinsic :: iso_c_binding
    implicit none

    integer(kind=C_intptr_t)                  :: sendbuff
    integer(kind=C_intptr_t)                  :: recvbuff
    integer(kind=c_size_t)                    :: nrElements
    integer(kind=c_int)                       :: mpicclDatatype
    integer(kind=c_int)                       :: root
    integer(kind=C_intptr_t)                  :: mpicclComm
    integer(kind=C_intptr_t)                  :: gpuStream
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_bcast_intptr_c(sendbuff, recvbuff, nrElements, mpicclDatatype, root, mpicclComm, gpuStream) /= 0
#else
    success = .true.
#endif
  end function

  function mpiccl_bcast_cptr(sendbuff, recvbuff, nrElements, mpicclDatatype, root, mpicclComm, gpuStream) result(success)
    use, intrinsic :: iso_c_binding
    implicit none

    type(c_ptr)                               :: sendbuff
    type(c_ptr)                               :: recvbuff
    integer(kind=c_size_t)                    :: nrElements
    integer(kind=c_int)                       :: mpicclDatatype
    integer(kind=c_int)                       :: root
    integer(kind=C_intptr_t)                  :: mpicclComm
    integer(kind=C_intptr_t)                  :: gpuStream
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_bcast_cptr_c(sendbuff, recvbuff, nrElements, mpicclDatatype, root, mpicclComm, gpuStream) /= 0
#else
    success = .true.
#endif
  end function

  function mpiccl_send_intptr(sendbuff, nrElements, mpicclDatatype, peer, mpicclComm, gpuStream) result(success)
    use, intrinsic :: iso_c_binding
    implicit none

    integer(kind=C_intptr_t)                  :: sendbuff
    integer(kind=c_size_t)                    :: nrElements
    integer(kind=c_int)                       :: mpicclDatatype
    integer(kind=c_int)                       :: peer
    integer(kind=C_intptr_t)                  :: mpicclComm
    integer(kind=C_intptr_t)                  :: gpuStream
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_send_intptr_c(sendbuff, nrElements, mpicclDatatype, peer, mpicclComm, gpuStream) /= 0
#else
    success = .true.
#endif
  end function

  function mpiccl_send_cptr(sendbuff, nrElements, mpicclDatatype, peer, mpicclComm, gpuStream) result(success)
    use, intrinsic :: iso_c_binding
    implicit none

    type(c_ptr)                               :: sendbuff
    integer(kind=c_size_t)                    :: nrElements
    integer(kind=c_int)                       :: mpicclDatatype
    integer(kind=c_int)                       :: peer
    integer(kind=C_intptr_t)                  :: mpicclComm
    integer(kind=C_intptr_t)                  :: gpuStream
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_send_cptr_c(sendbuff, nrElements, mpicclDatatype, peer, mpicclComm, gpuStream) /= 0
#else
    success = .true.
#endif
  end function

  function mpiccl_recv_intptr(recvbuff, nrElements, mpicclDatatype, peer, mpicclComm, gpuStream) result(success)
    use, intrinsic :: iso_c_binding
    implicit none

    integer(kind=C_intptr_t)                  :: recvbuff
    integer(kind=c_size_t)                    :: nrElements
    integer(kind=c_int)                       :: mpicclDatatype
    integer(kind=c_int)                       :: peer
    integer(kind=C_intptr_t)                  :: mpicclComm
    integer(kind=C_intptr_t)                  :: gpuStream
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_recv_intptr_c(recvbuff, nrElements, mpicclDatatype, peer, mpicclComm, gpuStream) /= 0
#else
    success = .true.
#endif
  end function

  function mpiccl_recv_cptr(recvbuff, nrElements, mpicclDatatype, peer, mpicclComm, gpuStream) result(success)
    use, intrinsic :: iso_c_binding
    implicit none

    type(c_ptr)                               :: recvbuff
    integer(kind=c_size_t)                    :: nrElements
    integer(kind=c_int)                       :: mpicclDatatype
    integer(kind=c_int)                       :: peer
    integer(kind=C_intptr_t)                  :: mpicclComm
    integer(kind=C_intptr_t)                  :: gpuStream
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_recv_cptr_c(recvbuff, nrElements, mpicclDatatype, peer, mpicclComm, gpuStream) /= 0
#else
    success = .true.
#endif
  end function


  function mpiccl_sendrecv_intptr(sendbuff, sendNrElements, sendMpicclDatatype, dest, &
                                  recvbuff, recvNrElements, recvMpicclDatatype, source, mpicclComm, gpuStream) result(success)
    use, intrinsic :: iso_c_binding
    implicit none

    integer(kind=c_intptr_t)                  :: sendbuff, recvbuff
    integer(kind=c_size_t)                    :: sendNrElements, recvNrElements
    integer(kind=c_int)                       :: sendMpicclDatatype, recvMpicclDatatype
    integer(kind=c_int)                       :: dest, source
    integer(kind=c_intptr_t)                  :: mpicclComm
    integer(kind=c_intptr_t)                  :: gpuStream
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_sendrecv_intptr_c(sendbuff, sendNrElements, sendMpicclDatatype, dest, &
                                       recvbuff, recvNrElements, recvMpicclDatatype, source, mpicclComm, gpuStream) /= 0
#else
    success = .true.
#endif
  end function

  function mpiccl_isend_intptr(sendbuff, nrElements, mpicclDatatype, peer, mpicclComm, request, gpuStream) result(success)
    use, intrinsic :: iso_c_binding
    implicit none

    integer(kind=c_intptr_t)                  :: sendbuff
    integer(kind=c_size_t)                    :: nrElements
    integer(kind=c_int)                       :: mpicclDatatype
    integer(kind=c_int)                       :: peer
    integer(kind=c_intptr_t)                  :: mpicclComm
    integer(kind=c_intptr_t)                  :: request
    integer(kind=c_intptr_t)                  :: gpuStream
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_isend_intptr_c(sendbuff, nrElements, mpicclDatatype, peer, mpicclComm, request, gpuStream) /= 0
#else
    success = .true.
#endif
  end function

  function mpiccl_irecv_intptr(recvbuff, nrElements, mpicclDatatype, peer, mpicclComm, request, gpuStream) result(success)
    use, intrinsic :: iso_c_binding
    implicit none

    integer(kind=c_intptr_t)                  :: recvbuff
    integer(kind=c_size_t)                    :: nrElements
    integer(kind=c_int)                       :: mpicclDatatype
    integer(kind=c_int)                       :: peer
    integer(kind=c_intptr_t)                  :: mpicclComm
    integer(kind=c_intptr_t)                  :: request
    integer(kind=c_intptr_t)                  :: gpuStream
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_irecv_intptr_c(recvbuff, nrElements, mpicclDatatype, peer, mpicclComm, request, gpuStream) /= 0
#else
    success = .true.
#endif
  end function

  function mpiccl_waitall(count, requests, gpuStream) result(success)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int)                       :: count
    integer(kind=c_intptr_t)                  :: requests
    integer(kind=c_intptr_t)                  :: gpuStream
    logical                                   :: success
#ifdef WITH_GPU_AWARE_MPICCL
    success = mpiccl_waitall_c(count, requests, gpuStream) /= 0
#else
    success = .true.
#endif
  end function