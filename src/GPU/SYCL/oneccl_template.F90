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
! Authors: Alexander Pöppl, Intel Corporation,
!          Andreas Marek, MPCDF
! This file is the generated version. Do NOT edit
    
             
  !integer(kind=c_int) :: onecclSum
  !integer(kind=c_int) :: onecclMax
  !integer(kind=c_int) :: onecclMin
  !integer(kind=c_int) :: onecclAvg
  !integer(kind=c_int) :: onecclProd

  !integer(kind=c_int) :: onecclInt
  !integer(kind=c_int) :: onecclInt32
  !integer(kind=c_int) :: onecclInt64
  !integer(kind=c_int) :: onecclFloat
  !integer(kind=c_int) :: onecclFloat32
  !integer(kind=c_int) :: onecclFloat64
  !integer(kind=c_int) :: onecclDouble

  type, BIND(C) ::ncclUniqueId
    CHARACTER(KIND=C_CHAR) :: str(256)
  end type


  interface
    function oneccl_redOp_onecclSum_c() result(flag) &
             bind(C, name="onecclRedOpSumFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function oneccl_redOp_onecclMax_c() result(flag) &
             bind(C, name="onecclRedOpMaxFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function oneccl_redOp_onecclMin_c() result(flag) &
             bind(C, name="onecclRedOpMinFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function oneccl_redOp_onecclAvg_c() result(flag) &
             bind(C, name="onecclRedOpAvgFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function oneccl_redOp_onecclProd_c() result(flag) &
             bind(C, name="onecclRedOpProdFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function oneccl_dataType_onecclInt_c() result(flag) &
             bind(C, name="onecclDataTypeOnecclIntFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function oneccl_dataType_onecclInt32_c() result(flag) &
             bind(C, name="onecclDataTypeOnecclInt32FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function oneccl_dataType_onecclInt64_c() result(flag) &
             bind(C, name="onecclDataTypeOnecclInt64FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function oneccl_dataType_onecclFloat_c() result(flag) &
             bind(C, name="onecclDataTypeOnecclFloatFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function oneccl_dataType_onecclFloat32_c() result(flag) &
             bind(C, name="onecclDataTypeOnecclFloat32FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function oneccl_dataType_onecclFloat64_c() result(flag) &
             bind(C, name="onecclDataTypeOnecclFloat64FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function oneccl_dataType_onecclDouble_c() result(flag) &
             bind(C, name="onecclDataTypeOnecclDoubleFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface  
    function oneccl_group_start_c() result(istat) &
             bind(C, name="onecclGroupStartFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)      :: istat
    end function
  end interface


  interface  
    function oneccl_group_end_c() result(istat) &
             bind(C, name="onecclGroupEndFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)      :: istat
    end function
  end interface
  
  interface
    function oneccl_get_unique_id_c(onecclId) result(istat) &
             bind(C, name="onecclGetUniqueIdFromC")
      use, intrinsic :: iso_c_binding
      import :: ncclUniqueId
      implicit none
      !integer(kind=C_intptr_T) :: onecclId(16)
      !integer(kind=C_intptr_T) :: onecclId
      type(ncclUniqueId)            :: onecclId
      !character(len=128)        :: onecclId
      integer(kind=C_INT)      :: istat
    end function
  end interface

  interface
    function oneccl_comm_init_rank_c(onecclComm, nRanks, onecclId, myRank) result(istat) &
             bind(C, name="onecclCommInitRankFromC")
      use, intrinsic :: iso_c_binding
      import :: ncclUniqueId
      implicit none
      integer(kind=C_intptr_T)        :: onecclComm
      integer(kind=c_int), value      :: nRanks
      !should be value, not possible since dimension trick
      !integer(kind=c_intptr_t), value :: onecclId(16)
      !integer(kind=c_intptr_t)        :: onecclId(16)
      !integer(kind=c_intptr_t), value :: onecclId
      type(ncclUniqueId)            :: onecclId
      integer(kind=c_int), value      :: myRank
      integer(kind=C_INT)             :: istat
    end function
  end interface
   
  ! only for version >=2.13  
  !interface
  !  function oneccl_comm_finalize_c(onecclComm) result(istat) &
  !           bind(C, name="onecclCommFinalizeFromC")
  !    use, intrinsic :: iso_c_binding
  !    implicit none
  !    integer(kind=C_intptr_T), value :: onecclComm
  !    integer(kind=C_INT)             :: istat  
  !  end function                                
  !end interface
      
  interface
    function oneccl_comm_destroy_c(onecclComm) result(istat) &
             bind(C, name="onecclCommDestroyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value :: onecclComm
      integer(kind=C_INT)             :: istat  
    end function                                
  end interface
  
  interface
    function oneccl_stream_synchronize_c(syclStream) result(istat) &
             bind(C, name="onecclStreamSynchronizeFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), value :: syclStream
      integer(kind=C_INT)             :: istat
    end function
  end interface

  interface oneccl_Allreduce
    module procedure oneccl_allreduce_intptr
    module procedure oneccl_allreduce_cptr
  end interface


  interface
    function oneccl_allreduce_intptr_c(sendbuff, recvbuff, nrElements, onecclDatatype, onecclOp, onecclComm, syclStream) result(istat) &
             bind(C, name="onecclAllReduceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), value              :: sendbuff
      integer(kind=C_intptr_t), value              :: recvbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: onecclDatatype
      integer(kind=C_INT), intent(in), value       :: onecclOp
      integer(kind=C_intptr_t), value              :: onecclComm
      integer(kind=C_intptr_t), value              :: syclStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function oneccl_allreduce_cptr_c(sendbuff, recvbuff, nrElements, onecclDatatype, onecclOp, onecclComm, syclStream) result(istat) &
             bind(C, name="onecclAllReduceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: sendbuff
      type(c_ptr), value                           :: recvbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: onecclDatatype
      integer(kind=C_INT), intent(in), value       :: onecclOp
      integer(kind=C_intptr_t), value              :: onecclComm
      integer(kind=C_intptr_t), value              :: syclStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface oneccl_reduce
    module procedure oneccl_reduce_intptr
    module procedure oneccl_reduce_cptr
  end interface


  interface
    function oneccl_reduce_intptr_c(sendbuff, recvbuff, nrElements, onecclDatatype, onecclOp, root, onecclComm, syclStream) result(istat) &
             bind(C, name="onecclReduceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), value              :: sendbuff
      integer(kind=C_intptr_t), value              :: recvbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: onecclDatatype
      integer(kind=C_INT), intent(in), value       :: onecclOp
      integer(kind=C_INT), intent(in), value       :: root
      integer(kind=C_intptr_t), value              :: onecclComm
      integer(kind=C_intptr_t), value              :: syclStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function oneccl_reduce_cptr_c(sendbuff, recvbuff, nrElements, onecclDatatype, onecclOp, root, onecclComm, syclStream) result(istat) &
             bind(C, name="onecclReduceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: sendbuff
      type(c_ptr), value                           :: recvbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: onecclDatatype
      integer(kind=C_INT), intent(in), value       :: onecclOp
      integer(kind=C_INT), intent(in), value       :: root
      integer(kind=C_intptr_t), value              :: onecclComm
      integer(kind=C_intptr_t), value              :: syclStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface


  interface oneccl_Bcast
    module procedure oneccl_Bcast_intptr
    module procedure oneccl_Bcast_cptr
  end interface


  interface
    function oneccl_bcast_intptr_c(sendbuff, recvbuff, nrElements, onecclDatatype, root, onecclComm, syclStream) result(istat) &
             bind(C, name="onecclBroadcastFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), value              :: sendbuff
      integer(kind=C_intptr_t), value              :: recvbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: onecclDatatype
      integer(kind=C_INT), intent(in), value       :: root
      integer(kind=C_intptr_t), value              :: onecclComm
      integer(kind=C_intptr_t), value              :: syclStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function oneccl_bcast_cptr_c(sendbuff, recvbuff, nrElements, onecclDatatype, root, onecclComm, syclStream) result(istat) &
             bind(C, name="onecclBroadcastFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: sendbuff
      type(c_ptr), value                           :: recvbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: onecclDatatype
      integer(kind=C_INT), intent(in), value       :: root
      integer(kind=C_intptr_t), value              :: onecclComm
      integer(kind=C_intptr_t), value              :: syclStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface oneccl_Send
    module procedure oneccl_Send_intptr
    module procedure oneccl_Send_cptr
  end interface

  interface
    function oneccl_send_intptr_c(sendbuff, nrElements, onecclDatatype, peer, onecclComm, syclStream) result(istat) &
             bind(C, name="onecclSendFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), value              :: sendbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: onecclDatatype
      integer(kind=C_INT), intent(in), value       :: peer
      integer(kind=C_intptr_t), value              :: onecclComm
      integer(kind=C_intptr_t), value              :: syclStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function oneccl_send_cptr_c(sendbuff, nrElements, onecclDatatype, peer, onecclComm, syclStream) result(istat) &
             bind(C, name="onecclSendFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: sendbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: onecclDatatype
      integer(kind=C_INT), intent(in), value       :: peer
      integer(kind=C_intptr_t), value              :: onecclComm
      integer(kind=C_intptr_t), value              :: syclStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface oneccl_Recv
    module procedure oneccl_Recv_intptr
    module procedure oneccl_Recv_cptr
  end interface

  interface
    function oneccl_recv_intptr_c(recvbuff, nrElements, onecclDatatype, peer, onecclComm, syclStream) result(istat) &
             bind(C, name="onecclRecvFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), value              :: recvbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: onecclDatatype
      integer(kind=C_INT), intent(in), value       :: peer
      integer(kind=C_intptr_t), value              :: onecclComm
      integer(kind=C_intptr_t), value              :: syclStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function oneccl_recv_cptr_c(recvbuff, nrElements, onecclDatatype, peer, onecclComm, syclStream) result(istat) &
             bind(C, name="onecclRecvFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: recvbuff
      integer(kind=c_size_t), intent(in), value    :: nrElements
      integer(kind=C_INT), intent(in), value       :: onecclDatatype
      integer(kind=C_INT), intent(in), value       :: peer
      integer(kind=C_intptr_t), value              :: onecclComm
      integer(kind=C_intptr_t), value              :: syclStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  contains


    function oneccl_redOp_onecclSum() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_ONEAPI_ONECCL
      flag = int(oneccl_redOp_onecclSum_c())
#else
      flag = 0
#endif
    end function

    function oneccl_redOp_onecclMax() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_ONEAPI_ONECCL
      flag = int(oneccl_redOp_onecclMax_c())
#else
      flag = 0
#endif
    end function

    function oneccl_redOp_onecclMin() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_ONEAPI_ONECCL
      flag = int(oneccl_redOp_onecclMin_c())
#else
      flag = 0
#endif
    end function

    function oneccl_redOp_onecclAvg() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_ONEAPI_ONECCL
      flag = int(oneccl_redOp_onecclAvg_c())
#else
      flag = 0
#endif
    end function

    function oneccl_redOp_onecclProd() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_ONEAPI_ONECCL
      flag = int(oneccl_redOp_onecclProd_c())
#else
      flag = 0
#endif
    end function

    function oneccl_dataType_onecclInt() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_ONEAPI_ONECCL
      flag = int(oneccl_dataType_onecclInt_c())
#else
      flag = 0
#endif
    end function

    function oneccl_dataType_onecclInt32() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_ONEAPI_ONECCL
      flag = int(oneccl_dataType_onecclInt32_c())
#else
      flag = 0
#endif
    end function

    function oneccl_dataType_onecclInt64() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_ONEAPI_ONECCL
      flag = int(oneccl_dataType_onecclInt64_c())
#else
      flag = 0
#endif
    end function

    function oneccl_dataType_onecclFloat() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_ONEAPI_ONECCL
      flag = int(oneccl_dataType_onecclFloat_c())
#else
      flag = 0
#endif
    end function

    function oneccl_dataType_onecclFloat32() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_ONEAPI_ONECCL
      flag = int(oneccl_dataType_onecclFloat32_c())
#else
      flag = 0
#endif
    end function

    function oneccl_dataType_onecclFloat64() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_ONEAPI_ONECCL
      flag = int(oneccl_dataType_onecclFloat64_c())
#else
      flag = 0
#endif
    end function

    function oneccl_dataType_onecclDouble() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_ONEAPI_ONECCL
      flag = int(oneccl_dataType_onecclDouble_c())
#else
      flag = 0
#endif
    end function


    function oneccl_group_start() result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      logical                                   :: success
#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_group_start_c() /= 0
#else
      success = .true.
#endif
    end function


    function oneccl_group_end() result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      logical                                   :: success
#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_group_end_c() /= 0
#else
      success = .true.
#endif
    end function
    
    function oneccl_get_unique_id(onecclId) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(ncclUniqueId)                        :: onecclId
      logical                                   :: success
      integer :: i
#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_get_unique_id_c(onecclId) /= 0
#else 
      success = .true.
#endif
    end function
  
    
    function oneccl_comm_init_rank(onecclComm, nRanks, onecclId, myRank) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T)                  :: onecclComm
      integer(kind=c_int)                       :: nRanks
      type(ncclUniqueId)                      :: onecclId
      integer(kind=c_int)                       :: myRank
      logical                                   :: success

#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_comm_init_rank_c(onecclComm, nRanks, onecclId, myRank) /= 0
#else 
      success = .true.
#endif
    end function
 
! only for version >= 2.13    
!    function oneccl_comm_finalize(onecclComm) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: onecclComm
!      logical                                   :: success
!#ifdef WITH_ONEAPI_ONECCL
!      success = oneccl_comm_finalize_c(onecclComm) /= 0
!#else
!      success = .true.
!#endif
!    end function
  
    function oneccl_comm_destroy(onecclComm) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: onecclComm
      logical                                   :: success
#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_comm_destroy_c(onecclComm) /= 0
#else
      success = .true.
#endif
    end function
  
    function oneccl_stream_synchronize(syclStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), value :: syclStream
      logical                         :: success
      success = oneccl_stream_synchronize_c(syclStream) /= 0
    end function

    function oneccl_allreduce_intptr(sendbuff, recvbuff, nrElements, onecclDatatype, onecclOp, onecclComm, syclStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: sendbuff
      integer(kind=C_intptr_t)                  :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: onecclDatatype
      integer(kind=c_int)                       :: onecclOp
      integer(kind=C_intptr_t)                  :: onecclComm
      integer(kind=C_intptr_t)                  :: syclStream
      logical                                   :: success
#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_allreduce_intptr_c(sendbuff, recvbuff, nrElements, onecclDatatype, onecclOp, onecclComm, syclStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function oneccl_allreduce_cptr(sendbuff, recvbuff, nrElements, onecclDatatype, onecclOp, onecclComm, syclStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                               :: sendbuff
      type(c_ptr)                               :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: onecclDatatype
      integer(kind=c_int)                       :: onecclOp
      integer(kind=C_intptr_t)                  :: onecclComm
      integer(kind=C_intptr_t)                  :: syclStream
      logical                                   :: success
#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_allreduce_cptr_c(sendbuff, recvbuff, nrElements, onecclDatatype, onecclOp, onecclComm, syclStream) /= 0
#else
      success = .true.
#endif
    end function

    function oneccl_reduce_intptr(sendbuff, recvbuff, nrElements, onecclDatatype, onecclOp, root, onecclComm, syclStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: sendbuff
      integer(kind=C_intptr_t)                  :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: onecclDatatype
      integer(kind=c_int)                       :: onecclOp
      integer(kind=c_int)                       :: root
      integer(kind=C_intptr_t)                  :: onecclComm
      integer(kind=C_intptr_t)                  :: syclStream
      logical                                   :: success
#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_reduce_intptr_c(sendbuff, recvbuff, nrElements, onecclDatatype, onecclOp, root, onecclComm, syclStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function oneccl_reduce_cptr(sendbuff, recvbuff, nrElements, onecclDatatype, onecclOp, root, onecclComm, syclStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                               :: sendbuff
      type(c_ptr)                               :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: onecclDatatype
      integer(kind=c_int)                       :: onecclOp
      integer(kind=c_int)                       :: root
      integer(kind=C_intptr_t)                  :: onecclComm
      integer(kind=C_intptr_t)                  :: syclStream
      logical                                   :: success
#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_reduce_cptr_c(sendbuff, recvbuff, nrElements, onecclDatatype, onecclOp, root, onecclComm, syclStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function oneccl_bcast_intptr(sendbuff, recvbuff, nrElements, onecclDatatype, root, onecclComm, syclStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: sendbuff
      integer(kind=C_intptr_t)                  :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: onecclDatatype
      integer(kind=c_int)                       :: root
      integer(kind=C_intptr_t)                  :: onecclComm
      integer(kind=C_intptr_t)                  :: syclStream
      logical                                   :: success
#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_bcast_intptr_c(sendbuff, recvbuff, nrElements, onecclDatatype, root, onecclComm, syclStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function oneccl_bcast_cptr(sendbuff, recvbuff, nrElements, onecclDatatype, root, onecclComm, syclStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                               :: sendbuff
      type(c_ptr)                               :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: onecclDatatype
      integer(kind=c_int)                       :: root
      integer(kind=C_intptr_t)                  :: onecclComm
      integer(kind=C_intptr_t)                  :: syclStream
      logical                                   :: success
#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_bcast_cptr_c(sendbuff, recvbuff, nrElements, onecclDatatype, root, onecclComm, syclStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function oneccl_send_intptr(sendbuff, nrElements, onecclDatatype, peer, onecclComm, syclStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: sendbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: onecclDatatype
      integer(kind=c_int)                       :: peer
      integer(kind=C_intptr_t)                  :: onecclComm
      integer(kind=C_intptr_t)                  :: syclStream
      logical                                   :: success
#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_send_intptr_c(sendbuff, nrElements, onecclDatatype, peer, onecclComm, syclStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function oneccl_send_cptr(sendbuff, nrElements, onecclDatatype, peer, onecclComm, syclStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                               :: sendbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: onecclDatatype
      integer(kind=c_int)                       :: peer
      integer(kind=C_intptr_t)                  :: onecclComm
      integer(kind=C_intptr_t)                  :: syclStream
      logical                                   :: success
#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_send_cptr_c(sendbuff, nrElements, onecclDatatype, peer, onecclComm, syclStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function oneccl_recv_intptr(recvbuff, nrElements, onecclDatatype, peer, onecclComm, syclStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: onecclDatatype
      integer(kind=c_int)                       :: peer
      integer(kind=C_intptr_t)                  :: onecclComm
      integer(kind=C_intptr_t)                  :: syclStream
      logical                                   :: success
#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_recv_intptr_c(recvbuff, nrElements, onecclDatatype, peer, onecclComm, syclStream) /= 0
#else
      success = .true.
#endif
    end function
  
    function oneccl_recv_cptr(recvbuff, nrElements, onecclDatatype, peer, onecclComm, syclStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                               :: recvbuff
      integer(kind=c_size_t)                    :: nrElements
      integer(kind=c_int)                       :: onecclDatatype
      integer(kind=c_int)                       :: peer
      integer(kind=C_intptr_t)                  :: onecclComm
      integer(kind=C_intptr_t)                  :: syclStream
      logical                                   :: success
#ifdef WITH_ONEAPI_ONECCL
      success = oneccl_recv_cptr_c(recvbuff, nrElements, onecclDatatype, peer, onecclComm, syclStream) /= 0
#else
      success = .true.
#endif
    end function
