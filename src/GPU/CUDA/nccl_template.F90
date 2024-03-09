!    Copyright 2024, A. Marek
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
! Author: Andreas Marek, MPCDF
! This file is the generated version. Do NOT edit
    
             
  !integer(kind=c_int) :: ncclSum
  !integer(kind=c_int) :: ncclMax
  !integer(kind=c_int) :: ncclMin
  !integer(kind=c_int) :: ncclAvg
  !integer(kind=c_int) :: ncclProd

  !integer(kind=c_int) :: ncclInt
  !integer(kind=c_int) :: ncclInt32
  !integer(kind=c_int) :: ncclInt64
  !integer(kind=c_int) :: ncclFloat
  !integer(kind=c_int) :: ncclFloat32
  !integer(kind=c_int) :: ncclFloat64
  !integer(kind=c_int) :: ncclDouble

  type, BIND(C) ::ncclUniqueId
    CHARACTER(KIND=C_CHAR) :: str(128)
  end type


  interface
    function nccl_redOp_ncclSum_c() result(flag) &
             bind(C, name="ncclRedOpSumFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function nccl_redOp_ncclMax_c() result(flag) &
             bind(C, name="ncclRedOpMaxFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function nccl_redOp_ncclMin_c() result(flag) &
             bind(C, name="ncclRedOpMinFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function nccl_redOp_ncclAvg_c() result(flag) &
             bind(C, name="ncclRedOpAvgFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function nccl_redOp_ncclProd_c() result(flag) &
             bind(C, name="ncclRedOpProdFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function nccl_dataType_ncclInt_c() result(flag) &
             bind(C, name="ncclDataTypeNcclIntFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function nccl_dataType_ncclInt32_c() result(flag) &
             bind(C, name="ncclDataTypeNcclInt32FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function nccl_dataType_ncclInt64_c() result(flag) &
             bind(C, name="ncclDataTypeNcclInt64FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function nccl_dataType_ncclFloat_c() result(flag) &
             bind(C, name="ncclDataTypeNcclFloatFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function nccl_dataType_ncclFloat32_c() result(flag) &
             bind(C, name="ncclDataTypeNcclFloat32FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function nccl_dataType_ncclFloat64_c() result(flag) &
             bind(C, name="ncclDataTypeNcclFloat64FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function nccl_dataType_ncclDouble_c() result(flag) &
             bind(C, name="ncclDataTypeNcclDoubleFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface  
    function nccl_group_start_c() result(istat) &
             bind(C, name="ncclGroupStartFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)      :: istat
    end function
  end interface


  interface  
    function nccl_group_end_c() result(istat) &
             bind(C, name="ncclGroupEndFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)      :: istat
    end function
  end interface
  
  interface
    function nccl_get_unique_id_c(ncclId) result(istat) &
             bind(C, name="ncclGetUniqueIdFromC")
      use, intrinsic :: iso_c_binding
      import :: ncclUniqueId
      implicit none
      !integer(kind=C_intptr_T) :: ncclId(16)
      !integer(kind=C_intptr_T) :: ncclId
      type(ncclUniqueId)            :: ncclId
      !character(len=128)        :: ncclId
      integer(kind=C_INT)      :: istat
    end function
  end interface

  interface
    function nccl_comm_init_rank_c(ncclComm, nRanks, ncclId, myRank) result(istat) &
             bind(C, name="ncclCommInitRankFromC")
      use, intrinsic :: iso_c_binding
      import :: ncclUniqueId
      implicit none
      integer(kind=C_intptr_T)        :: ncclComm
      integer(kind=c_int), value      :: nRanks
      !should be value, not possible since dimension trick
      !integer(kind=c_intptr_t), value :: ncclId(16)
      !integer(kind=c_intptr_t)        :: ncclId(16)
      !integer(kind=c_intptr_t), value :: ncclId
      type(ncclUniqueId)            :: ncclId
      integer(kind=c_int), value      :: myRank
      integer(kind=C_INT)             :: istat
    end function
  end interface
   
  ! only for version >=2.13  
  !interface
  !  function nccl_comm_finalize_c(ncclComm) result(istat) &
  !           bind(C, name="ncclCommFinalizeFromC")
  !    use, intrinsic :: iso_c_binding
  !    implicit none
  !    integer(kind=C_intptr_T), value :: ncclComm
  !    integer(kind=C_INT)             :: istat  
  !  end function                                
  !end interface
      
  interface
    function nccl_comm_destroy_c(ncclComm) result(istat) &
             bind(C, name="ncclCommDestroyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value :: ncclComm
      integer(kind=C_INT)             :: istat  
    end function                                
  end interface

  interface nccl_Allreduce
    module procedure nccl_allreduce_intptr
    module procedure nccl_allreduce_cptr
  end interface


  interface
    function nccl_allreduce_intptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, ncclComm, cudaStream) result(istat) &
             bind(C, name="ncclAllReduceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), value              :: sendbuff
      integer(kind=C_intptr_t), value              :: recvbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: ncclDatatype
      integer(kind=C_INT), intent(in), value       :: ncclOp
      integer(kind=C_intptr_t), value              :: ncclComm
      integer(kind=C_intptr_t), value              :: cudaStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function nccl_allreduce_cptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, ncclComm, cudaStream) result(istat) &
             bind(C, name="ncclAllReduceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: sendbuff
      type(c_ptr), value                           :: recvbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: ncclDatatype
      integer(kind=C_INT), intent(in), value       :: ncclOp
      integer(kind=C_intptr_t), value              :: ncclComm
      integer(kind=C_intptr_t), value              :: cudaStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface nccl_reduce
    module procedure nccl_reduce_intptr
    module procedure nccl_reduce_cptr
  end interface


  interface
    function nccl_reduce_intptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, root, ncclComm, cudaStream) result(istat) &
             bind(C, name="ncclReduceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), value              :: sendbuff
      integer(kind=C_intptr_t), value              :: recvbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: ncclDatatype
      integer(kind=C_INT), intent(in), value       :: ncclOp
      integer(kind=C_INT), intent(in), value       :: root
      integer(kind=C_intptr_t), value              :: ncclComm
      integer(kind=C_intptr_t), value              :: cudaStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function nccl_reduce_cptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, root, ncclComm, cudaStream) result(istat) &
             bind(C, name="ncclReduceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: sendbuff
      type(c_ptr), value                           :: recvbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: ncclDatatype
      integer(kind=C_INT), intent(in), value       :: ncclOp
      integer(kind=C_INT), intent(in), value       :: root
      integer(kind=C_intptr_t), value              :: ncclComm
      integer(kind=C_intptr_t), value              :: cudaStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface nccl_Bcast
    module procedure nccl_Bcast_intptr
    module procedure nccl_Bcast_cptr
  end interface


  interface
    function nccl_bcast_intptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, root, ncclComm, cudaStream) result(istat) &
             bind(C, name="ncclBroadcastFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), value              :: sendbuff
      integer(kind=C_intptr_t), value              :: recvbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: ncclDatatype
      integer(kind=C_INT), intent(in), value       :: root
      integer(kind=C_intptr_t), value              :: ncclComm
      integer(kind=C_intptr_t), value              :: cudaStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function nccl_bcast_cptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, root, ncclComm, cudaStream) result(istat) &
             bind(C, name="ncclBroadcastFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: sendbuff
      type(c_ptr), value                           :: recvbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: ncclDatatype
      integer(kind=C_INT), intent(in), value       :: root
      integer(kind=C_intptr_t), value              :: ncclComm
      integer(kind=C_intptr_t), value              :: cudaStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface nccl_Send
    module procedure nccl_Send_intptr
    module procedure nccl_Send_cptr
  end interface

  interface
    function nccl_send_intptr_c(sendbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) result(istat) &
             bind(C, name="ncclSendFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), value              :: sendbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: ncclDatatype
      integer(kind=C_INT), intent(in), value       :: peer
      integer(kind=C_intptr_t), value              :: ncclComm
      integer(kind=C_intptr_t), value              :: cudaStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function nccl_send_cptr_c(sendbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) result(istat) &
             bind(C, name="ncclSendFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: sendbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: ncclDatatype
      integer(kind=C_INT), intent(in), value       :: peer
      integer(kind=C_intptr_t), value              :: ncclComm
      integer(kind=C_intptr_t), value              :: cudaStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface nccl_Recv
    module procedure nccl_Recv_intptr
    module procedure nccl_Recv_cptr
  end interface

  interface
    function nccl_recv_intptr_c(recvbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) result(istat) &
             bind(C, name="ncclRecvFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), value              :: recvbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: ncclDatatype
      integer(kind=C_INT), intent(in), value       :: peer
      integer(kind=C_intptr_t), value              :: ncclComm
      integer(kind=C_intptr_t), value              :: cudaStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function nccl_recv_cptr_c(recvbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) result(istat) &
             bind(C, name="ncclRecvFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: recvbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: ncclDatatype
      integer(kind=C_INT), intent(in), value       :: peer
      integer(kind=C_intptr_t), value              :: ncclComm
      integer(kind=C_intptr_t), value              :: cudaStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  contains


    function nccl_redOp_ncclSum() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = int(nccl_redOp_ncclSum_c())
#else
      flag = 0
#endif
    end function

    function nccl_redOp_ncclMax() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = int(nccl_redOp_ncclMax_c())
#else
      flag = 0
#endif
    end function

    function nccl_redOp_ncclMin() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = int(nccl_redOp_ncclMin_c())
#else
      flag = 0
#endif
    end function

    function nccl_redOp_ncclAvg() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = int(nccl_redOp_ncclAvg_c())
#else
      flag = 0
#endif
    end function

    function nccl_redOp_ncclProd() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = int(nccl_redOp_ncclProd_c())
#else
      flag = 0
#endif
    end function

    function nccl_dataType_ncclInt() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = int(nccl_dataType_ncclInt_c())
#else
      flag = 0
#endif
    end function

    function nccl_dataType_ncclInt32() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = int(nccl_dataType_ncclInt32_c())
#else
      flag = 0
#endif
    end function

    function nccl_dataType_ncclInt64() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = int(nccl_dataType_ncclInt64_c())
#else
      flag = 0
#endif
    end function

    function nccl_dataType_ncclFloat() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = int(nccl_dataType_ncclFloat_c())
#else
      flag = 0
#endif
    end function

    function nccl_dataType_ncclFloat32() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = int(nccl_dataType_ncclFloat32_c())
#else
      flag = 0
#endif
    end function

    function nccl_dataType_ncclFloat64() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = int(nccl_dataType_ncclFloat64_c())
#else
      flag = 0
#endif
    end function

    function nccl_dataType_ncclDouble() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_NCCL
      flag = int(nccl_dataType_ncclDouble_c())
#else
      flag = 0
#endif
    end function


    function nccl_group_start() result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      logical                                   :: success
#ifdef WITH_NVIDIA_NCCL
      success = nccl_group_start_c() /= 0
#else
      success = .true.
#endif
    end function


    function nccl_group_end() result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      logical                                   :: success
#ifdef WITH_NVIDIA_NCCL
      success = nccl_group_end_c() /= 0
#else
      success = .true.
#endif
    end function
    
    function nccl_get_unique_id(ncclId) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      !integer(kind=C_intptr_t)                  :: ncclId(16)
      type(ncclUniqueId)                        :: ncclId
      !integer(kind=C_intptr_t)                  :: ncclId
      !character(len=128)                        :: ncclId
      logical                                   :: success
      integer :: i
#ifdef WITH_NVIDIA_NCCL
      success = nccl_get_unique_id_c(ncclId) /= 0
#else 
      success = .true.
#endif
    end function
  
    
    function nccl_comm_init_rank(ncclComm, nRanks, ncclId, myRank) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T)                  :: ncclComm
      integer(kind=c_int)                       :: nRanks
      !integer(kind=C_intptr_t)                  :: ncclId(16)
      type(ncclUniqueId)                        :: ncclId
      !integer(kind=C_intptr_t)                  :: ncclId
      !character(len=128)                        :: ncclID
      integer(kind=c_int)                       :: myRank
      logical                                   :: success

#ifdef WITH_NVIDIA_NCCL
      success = nccl_comm_init_rank_c(ncclComm, nRanks, ncclId, myRank) /= 0
#else 
      success = .true.
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
  
    function nccl_comm_destroy(ncclComm) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: ncclComm
      logical                                   :: success
#ifdef WITH_NVIDIA_NCCL
      success = nccl_comm_destroy_c(ncclComm) /= 0
#else
      success = .true.
#endif
    end function
  
    function nccl_allreduce_intptr(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, ncclComm, cudaStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: sendbuff
      integer(kind=C_intptr_t)                  :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: ncclDatatype
      integer(kind=c_int)                       :: ncclOp
      integer(kind=C_intptr_t)                  :: ncclComm
      integer(kind=C_intptr_t)                  :: cudaStream
      logical                                   :: success
#ifdef WITH_NVIDIA_NCCL
      success = nccl_allreduce_intptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function nccl_allreduce_cptr(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, ncclComm, cudaStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                               :: sendbuff
      type(c_ptr)                               :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: ncclDatatype
      integer(kind=c_int)                       :: ncclOp
      integer(kind=C_intptr_t)                  :: ncclComm
      integer(kind=C_intptr_t)                  :: cudaStream
      logical                                   :: success
#ifdef WITH_NVIDIA_NCCL
      success = nccl_allreduce_cptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function

    function nccl_reduce_intptr(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, root, ncclComm, cudaStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: sendbuff
      integer(kind=C_intptr_t)                  :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: ncclDatatype
      integer(kind=c_int)                       :: ncclOp
      integer(kind=c_int)                       :: root
      integer(kind=C_intptr_t)                  :: ncclComm
      integer(kind=C_intptr_t)                  :: cudaStream
      logical                                   :: success
#ifdef WITH_NVIDIA_NCCL
      success = nccl_reduce_intptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, root, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function nccl_reduce_cptr(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, root, ncclComm, cudaStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                               :: sendbuff
      type(c_ptr)                               :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: ncclDatatype
      integer(kind=c_int)                       :: ncclOp
      integer(kind=c_int)                       :: root
      integer(kind=C_intptr_t)                  :: ncclComm
      integer(kind=C_intptr_t)                  :: cudaStream
      logical                                   :: success
#ifdef WITH_NVIDIA_NCCL
      success = nccl_reduce_cptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, root, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function nccl_bcast_intptr(sendbuff, recvbuff, nrElements, ncclDatatype, root, ncclComm, cudaStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: sendbuff
      integer(kind=C_intptr_t)                  :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: ncclDatatype
      integer(kind=c_int)                       :: root
      integer(kind=C_intptr_t)                  :: ncclComm
      integer(kind=C_intptr_t)                  :: cudaStream
      logical                                   :: success
#ifdef WITH_NVIDIA_NCCL
      success = nccl_bcast_intptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, root, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function nccl_bcast_cptr(sendbuff, recvbuff, nrElements, ncclDatatype, root, ncclComm, cudaStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                               :: sendbuff
      type(c_ptr)                               :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: ncclDatatype
      integer(kind=c_int)                       :: root
      integer(kind=C_intptr_t)                  :: ncclComm
      integer(kind=C_intptr_t)                  :: cudaStream
      logical                                   :: success
#ifdef WITH_NVIDIA_NCCL
      success = nccl_bcast_cptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, root, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function nccl_send_intptr(sendbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: sendbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: ncclDatatype
      integer(kind=c_int)                       :: peer
      integer(kind=C_intptr_t)                  :: ncclComm
      integer(kind=C_intptr_t)                  :: cudaStream
      logical                                   :: success
#ifdef WITH_NVIDIA_NCCL
      success = nccl_send_intptr_c(sendbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function nccl_send_cptr(sendbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                               :: sendbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: ncclDatatype
      integer(kind=c_int)                       :: peer
      integer(kind=C_intptr_t)                  :: ncclComm
      integer(kind=C_intptr_t)                  :: cudaStream
      logical                                   :: success
#ifdef WITH_NVIDIA_NCCL
      success = nccl_send_cptr_c(sendbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function nccl_recv_intptr(recvbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: ncclDatatype
      integer(kind=c_int)                       :: peer
      integer(kind=C_intptr_t)                  :: ncclComm
      integer(kind=C_intptr_t)                  :: cudaStream
      logical                                   :: success
#ifdef WITH_NVIDIA_NCCL
      success = nccl_recv_intptr_c(recvbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function nccl_recv_cptr(recvbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                               :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: ncclDatatype
      integer(kind=c_int)                       :: peer
      integer(kind=C_intptr_t)                  :: ncclComm
      integer(kind=C_intptr_t)                  :: cudaStream
      logical                                   :: success
#ifdef WITH_NVIDIA_NCCL
      success = nccl_recv_cptr_c(recvbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
