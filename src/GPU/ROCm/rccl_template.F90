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
    function rccl_redOp_ncclSum_c() result(flag) &
             bind(C, name="rcclRedOpSumFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function rccl_redOp_ncclMax_c() result(flag) &
             bind(C, name="rcclRedOpMaxFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function rccl_redOp_ncclMin_c() result(flag) &
             bind(C, name="rcclRedOpMinFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function rccl_redOp_ncclAvg_c() result(flag) &
             bind(C, name="rcclRedOpAvgFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function rccl_redOp_ncclProd_c() result(flag) &
             bind(C, name="rcclRedOpProdFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function rccl_dataType_ncclInt_c() result(flag) &
             bind(C, name="rcclDataTypeNcclIntFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function rccl_dataType_ncclInt32_c() result(flag) &
             bind(C, name="rcclDataTypeNcclInt32FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function rccl_dataType_ncclInt64_c() result(flag) &
             bind(C, name="rcclDataTypeNcclInt64FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function rccl_dataType_ncclFloat_c() result(flag) &
             bind(C, name="rcclDataTypeNcclFloatFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function rccl_dataType_ncclFloat32_c() result(flag) &
             bind(C, name="rcclDataTypeNcclFloat32FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function rccl_dataType_ncclFloat64_c() result(flag) &
             bind(C, name="rcclDataTypeNcclFloat64FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function rccl_dataType_ncclDouble_c() result(flag) &
             bind(C, name="rcclDataTypeNcclDoubleFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface  
    function rccl_group_start_c() result(istat) &
             bind(C, name="rcclGroupStartFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)      :: istat
    end function
  end interface


  interface  
    function rccl_group_end_c() result(istat) &
             bind(C, name="rcclGroupEndFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)      :: istat
    end function
  end interface
  
  interface
    function rccl_get_unique_id_c(ncclId) result(istat) &
             bind(C, name="rcclGetUniqueIdFromC")
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
    function rccl_comm_init_rank_c(ncclComm, nRanks, ncclId, myRank) result(istat) &
             bind(C, name="rcclCommInitRankFromC")
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
  !  function rccl_comm_finalize_c(ncclComm) result(istat) &
  !           bind(C, name="rcclCommFinalizeFromC")
  !    use, intrinsic :: iso_c_binding
  !    implicit none
  !    integer(kind=C_intptr_T), value :: ncclComm
  !    integer(kind=C_INT)             :: istat  
  !  end function                                
  !end interface
      
  interface
    function rccl_comm_destroy_c(ncclComm) result(istat) &
             bind(C, name="rcclCommDestroyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value :: ncclComm
      integer(kind=C_INT)             :: istat  
    end function                                
  end interface

  interface rccl_Allreduce
    module procedure rccl_allreduce_intptr
    module procedure rccl_allreduce_cptr
  end interface


  interface
    function rccl_allreduce_intptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, ncclComm, cudaStream) result(istat) &
             bind(C, name="rcclAllReduceFromC")
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
    function rccl_allreduce_cptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, ncclComm, cudaStream) result(istat) &
             bind(C, name="rcclAllReduceFromC")
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

  interface rccl_reduce
    module procedure rccl_reduce_intptr
    module procedure rccl_reduce_cptr
  end interface


  interface
    function rccl_reduce_intptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, root, ncclComm, cudaStream) result(istat) &
             bind(C, name="rcclReduceFromC")
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
    function rccl_reduce_cptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, root, ncclComm, cudaStream) result(istat) &
             bind(C, name="rcclReduceFromC")
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

  interface rccl_Bcast
    module procedure rccl_Bcast_intptr
    module procedure rccl_Bcast_cptr
  end interface


  interface
    function rccl_bcast_intptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, root, ncclComm, cudaStream) result(istat) &
             bind(C, name="rcclBroadcastFromC")
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
    function rccl_bcast_cptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, root, ncclComm, cudaStream) result(istat) &
             bind(C, name="rcclBroadcastFromC")
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

  interface rccl_Send
    module procedure rccl_Send_intptr
    module procedure rccl_Send_cptr
  end interface

  interface
    function rccl_send_intptr_c(sendbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) result(istat) &
             bind(C, name="rcclSendFromC")
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
    function rccl_send_cptr_c(sendbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) result(istat) &
             bind(C, name="rcclSendFromC")
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

  interface rccl_Recv
    module procedure rccl_Recv_intptr
    module procedure rccl_Recv_cptr
  end interface

  interface
    function rccl_recv_intptr_c(recvbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) result(istat) &
             bind(C, name="rcclRecvFromC")
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
    function rccl_recv_cptr_c(recvbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) result(istat) &
             bind(C, name="rcclRecvFromC")
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


    function rccl_redOp_ncclSum() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_RCCL
      flag = int(rccl_redOp_ncclSum_c())
#else
      flag = 0
#endif
    end function

    function rccl_redOp_ncclMax() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_RCCL
      flag = int(rccl_redOp_ncclMax_c())
#else
      flag = 0
#endif
    end function

    function rccl_redOp_ncclMin() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_RCCL
      flag = int(rccl_redOp_ncclMin_c())
#else
      flag = 0
#endif
    end function

    function rccl_redOp_ncclAvg() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_RCCL
      flag = int(rccl_redOp_ncclAvg_c())
#else
      flag = 0
#endif
    end function

    function rccl_redOp_ncclProd() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_RCCL
      flag = int(rccl_redOp_ncclProd_c())
#else
      flag = 0
#endif
    end function

    function rccl_dataType_ncclInt() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_RCCL
      flag = int(rccl_dataType_ncclInt_c())
#else
      flag = 0
#endif
    end function

    function rccl_dataType_ncclInt32() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_RCCL
      flag = int(rccl_dataType_ncclInt32_c())
#else
      flag = 0
#endif
    end function

    function rccl_dataType_ncclInt64() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_RCCL
      flag = int(rccl_dataType_ncclInt64_c())
#else
      flag = 0
#endif
    end function

    function rccl_dataType_ncclFloat() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_RCCL
      flag = int(rccl_dataType_ncclFloat_c())
#else
      flag = 0
#endif
    end function

    function rccl_dataType_ncclFloat32() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_RCCL
      flag = int(rccl_dataType_ncclFloat32_c())
#else
      flag = 0
#endif
    end function

    function rccl_dataType_ncclFloat64() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_RCCL
      flag = int(rccl_dataType_ncclFloat64_c())
#else
      flag = 0
#endif
    end function

    function rccl_dataType_ncclDouble() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_RCCL
      flag = int(rccl_dataType_ncclDouble_c())
#else
      flag = 0
#endif
    end function


    function rccl_group_start() result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      logical                                   :: success
#ifdef WITH_AMD_RCCL
      success = rccl_group_start_c() /= 0
#else
      success = .true.
#endif
    end function


    function rccl_group_end() result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      logical                                   :: success
#ifdef WITH_AMD_RCCL
      success = rccl_group_end_c() /= 0
#else
      success = .true.
#endif
    end function
    
    function rccl_get_unique_id(ncclId) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      !integer(kind=C_intptr_t)                  :: ncclId(16)
      type(ncclUniqueId)                        :: ncclId
      !integer(kind=C_intptr_t)                  :: ncclId
      !character(len=128)                        :: ncclId
      logical                                   :: success
      integer :: i
#ifdef WITH_AMD_RCCL
      success = rccl_get_unique_id_c(ncclId) /= 0
#else 
      success = .true.
#endif
    end function
  
    
    function rccl_comm_init_rank(ncclComm, nRanks, ncclId, myRank) result(success)
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

#ifdef WITH_AMD_RCCL
      success = rccl_comm_init_rank_c(ncclComm, nRanks, ncclId, myRank) /= 0
#else 
      success = .true.
#endif
    end function
 
! only for version >= 2.13    
!    function rccl_comm_finalize(ncclComm) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: ncclComm
!      logical                                   :: success
!#ifdef WITH_AMD_RCCL
!      success = rccl_comm_finalize_c(ncclComm) /= 0
!#else
!      success = .true.
!#endif
!    end function
  
    function rccl_comm_destroy(ncclComm) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: ncclComm
      logical                                   :: success
#ifdef WITH_AMD_RCCL
      success = rccl_comm_destroy_c(ncclComm) /= 0
#else
      success = .true.
#endif
    end function
  
    function rccl_allreduce_intptr(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, ncclComm, cudaStream) result(success)
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
#ifdef WITH_AMD_RCCL
      success = rccl_allreduce_intptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function rccl_allreduce_cptr(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, ncclComm, cudaStream) result(success)
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
#ifdef WITH_AMD_RCCL
      success = rccl_allreduce_cptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function

    function rccl_reduce_intptr(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, root, ncclComm, cudaStream) result(success)
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
#ifdef WITH_AMD_RCCL
      success = rccl_reduce_intptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, root, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function rccl_reduce_cptr(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, root, ncclComm, cudaStream) result(success)
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
#ifdef WITH_AMD_RCCL
      success = rccl_reduce_cptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, ncclOp, root, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function rccl_bcast_intptr(sendbuff, recvbuff, nrElements, ncclDatatype, root, ncclComm, cudaStream) result(success)
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
#ifdef WITH_AMD_RCCL
      success = rccl_bcast_intptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, root, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function rccl_bcast_cptr(sendbuff, recvbuff, nrElements, ncclDatatype, root, ncclComm, cudaStream) result(success)
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
#ifdef WITH_AMD_RCCL
      success = rccl_bcast_cptr_c(sendbuff, recvbuff, nrElements, ncclDatatype, root, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function rccl_send_intptr(sendbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: sendbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: ncclDatatype
      integer(kind=c_int)                       :: peer
      integer(kind=C_intptr_t)                  :: ncclComm
      integer(kind=C_intptr_t)                  :: cudaStream
      logical                                   :: success
#ifdef WITH_AMD_RCCL
      success = rccl_send_intptr_c(sendbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function rccl_send_cptr(sendbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                               :: sendbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: ncclDatatype
      integer(kind=c_int)                       :: peer
      integer(kind=C_intptr_t)                  :: ncclComm
      integer(kind=C_intptr_t)                  :: cudaStream
      logical                                   :: success
#ifdef WITH_AMD_RCCL
      success = rccl_send_cptr_c(sendbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function rccl_recv_intptr(recvbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: ncclDatatype
      integer(kind=c_int)                       :: peer
      integer(kind=C_intptr_t)                  :: ncclComm
      integer(kind=C_intptr_t)                  :: cudaStream
      logical                                   :: success
#ifdef WITH_AMD_RCCL
      success = rccl_recv_intptr_c(recvbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function rccl_recv_cptr(recvbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                               :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: ncclDatatype
      integer(kind=c_int)                       :: peer
      integer(kind=C_intptr_t)                  :: ncclComm
      integer(kind=C_intptr_t)                  :: cudaStream
      logical                                   :: success
#ifdef WITH_AMD_RCCL
      success = rccl_recv_cptr_c(recvbuff, nrElements, ncclDatatype, peer, ncclComm, cudaStream) /= 0
#else
      success = .true.
#endif
    end function
