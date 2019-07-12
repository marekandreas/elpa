#if 0
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
!
! --------------------------------------------------------------------------------------------------
!
! This file contains the compute intensive kernels for the Householder transformations.
!
! This is the small and simple version (no hand unrolling of loops etc.) but for some
! compilers this performs better than a sophisticated version with transformed and unrolled loops.
!
! It should be compiled with the highest possible optimization level.
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! --------------------------------------------------------------------------------------------------
#endif


#if REALCASE==1
  subroutine quad_hh_trafo_&
  &MATH_DATATYPE&
  &_blas_4hv_&
  &PRECISION&
  & (useGPU, q, hh, nb, nq, ldq, ldh, h_dev, s_dev, q_dev, w_dev)

    use precision
    use iso_c_binding
    use cuda_functions
    implicit none
#include "../../general/precision_kinds.F90"

    logical                         :: useGPU
    integer(kind=ik), intent(in)    :: nb, nq, ldq, ldh

#ifdef USE_ASSUMED_SIZE
    real(kind=C_DATATYPE_KIND), intent(inout) :: q(ldq,*)
    real(kind=C_DATATYPE_KIND), intent(in)    :: hh(ldh,*)
#else
    real(kind=C_DATATYPE_KIND), intent(inout) :: q(1:ldq,1:nb+3)
    real(kind=C_DATATYPE_KIND), intent(in)    :: hh(1:ldh,1:6)
#endif

    real(kind=C_DATATYPE_KIND)                :: w_comb(ldq, 4)
    real(kind=C_DATATYPE_KIND)                :: h_mat(4, nb+3)
    real(kind=C_DATATYPE_KIND)                :: s_mat(4, 4)


    !TODO remove
    !real(kind=C_DATATYPE_KIND)                :: q_extra(1:ldq,1:nb+3)


    integer(kind=c_intptr_t)                  :: h_dev, s_dev, q_dev, w_dev
    logical                                   :: successCUDA
    integer(kind=c_intptr_t), parameter       :: size_of_datatype = size_of_&
                                                                            &PRECISION&
                                                                            &_&
                                                                            &MATH_DATATYPE

    integer(kind=ik)                          :: i, j, k

    integer(kind=ik), parameter               :: max_block_blas = 4


    ! Calculate dot product of the two Householder vectors

   h_mat(:,:) = 0.0_rk

   h_mat(1,4) = -1.0_rk
   h_mat(2,3) = -1.0_rk
   h_mat(3,2) = -1.0_rk
   h_mat(4,1) = -1.0_rk

   h_mat(1,5:nb+3) = -hh(2:nb, 1)
   h_mat(2,4:nb+2) = -hh(2:nb, 2)
   h_mat(3,3:nb+1) = -hh(2:nb, 3)
   h_mat(4,2:nb)   = -hh(2:nb, 4)

   if(useGPU) then
      !    nb == nbw
      successCUDA =  cuda_memcpy(h_dev, loc(h_mat(1,1)),  &
                            max_block_blas * (nb+3) * size_of_datatype, &
                            cudaMemcpyHostToDevice)
      if (.not.(successCUDA)) then
        print *,"blas_block4_kernel: error in cudaMemcpy, h_dev host to device"
        stop 1
      endif
      !    nq == stripe_width
      successCUDA =  cuda_memcpy(q_dev, loc(q(1,1)),  &
                            ldq * (nb+3) * size_of_datatype, &
                            cudaMemcpyHostToDevice)
      if (.not.(successCUDA)) then
        print *,"blas_block4_kernel: error in cudaMemcpy, q_dev host to device"
        stop 1
      endif
   endif


   ! TODO we do not need the diagonal, but how to do it with BLAS?
   !s_mat = - matmul(h_mat, transpose(h_mat))
   if(useGPU) then
     call cublas_PRECISION_SYRK('L', 'N', 4, nb+3, &
                         -ONE, h_dev, 4, &
                         ZERO, s_dev, 4)
   else
     call PRECISION_SYRK('L', 'N', 4, nb+3, &
                         -ONE, h_mat, 4, &
                         ZERO, s_mat, 4)
   endif

   !w_comb = - matmul(q(1:nq, 1:nb+3), transpose(h_mat))
   if(useGPU) then
     call cublas_PRECISION_GEMM('N', 'T', nq, 4, nb+3, &
                         -ONE, q_dev, ldq, &
                         h_dev, 4, &
                         ZERO, w_dev, ldq)
   else
     call PRECISION_GEMM('N', 'T', nq, 4, nb+3, &
                         -ONE, q, ldq, &
                         h_mat, 4, &
                         ZERO, w_comb, ldq)
   endif

   ! Rank-1 update
   !w_comb(1:nq,1) = hh(1,1) * w_comb(1:nq, 1)
   if(useGPU) then
     call cublas_PRECISION_SCAL(nq, hh(1,1), w_dev, 1)
   else
     call PRECISION_SCAL(nq, hh(1,1), w_comb(1, 1), 1)
   endif
   do i = 2, 4
!     w_comb(1:nq,i) = matmul(w_comb(1:nq,1:i-1), hh(1,i) * s_mat(i,1:i-1)) + hh(1,i) * w_comb(1:nq, i)
     if(useGPU) then
       call cublas_PRECISION_GEMV('N', nq, i-1, &
                           hh(1,i), w_dev, ldq, &
                           s_dev + (i - 1) * size_of_datatype, 4, &
                           hh(1,i), w_dev + (i-1) * ldq * size_of_datatype, 1)
     else
       call PRECISION_GEMV('N', nq, i-1, &
                           hh(1,i), w_comb(1, 1), ldq, &
                           s_mat(i,1), 4, &
                           hh(1,i), w_comb(1,i), 1)
     endif
   enddo

   !  ---------------------
   if(useGPU) then
!      successCUDA =  cuda_memcpy(loc(s_mat(1,1)), s_dev,  &
!                            4 * 4 * size_of_datatype, &
!                            cudaMemcpyDeviceToHost)
!      if (.not.(successCUDA)) then
!        print *,"blas_block4_kernel: error in cudaMemcpy, q_dev device to host"
!        stop 1
!      endif



      successCUDA =  cuda_memcpy(loc(w_comb(1,1)), w_dev,  &
                            nq * 4 * size_of_datatype, &
                            cudaMemcpyDeviceToHost)
      if (.not.(successCUDA)) then
        print *,"blas_block4_kernel: error in cudaMemcpy, w_dev device to host"
        stop 1
      endif

      successCUDA =  cuda_memcpy(loc(h_mat(1,1)), h_dev,  &
                              max_block_blas * (nb+3) * size_of_datatype, &
                              cudaMemcpyDeviceToHost)
        if (.not.(successCUDA)) then
          print *,"blas_block4_kernel: error in cudaMemcpy, w_dev device to host"
          stop 1
        endif
     endif


   useGPU = .false.
   !  ---------------------




   !q(1:nq, 1:nb+3) = matmul(w_comb, h_mat) + q(1:nq, 1:nb+3)
   if(useGPU) then
     call cublas_PRECISION_GEMM('N', 'N', nq, nb+3, 4, &
                         ONE, w_dev, ldq, &
                         h_dev, 4, &
                         ONE, q_dev, ldq)
   else
     call PRECISION_GEMM('N', 'N', nq, nb+3, 4, &
                         ONE, w_comb, ldq, &
                         h_mat, 4, &
                         ONE, q, ldq)
   endif


   if(useGPU) then
      !successCUDA =  cuda_memcpy(loc(q_extra(1,1)), q_dev,  &
      successCUDA =  cuda_memcpy(loc(q(1,1)), q_dev,  &
                            ldq * (nb+3) * size_of_datatype, &
                            cudaMemcpyDeviceToHost)
      if (.not.(successCUDA)) then
        print *,"blas_block4_kernel: error in cudaMemcpy, q_dev device to host"
        stop 1
      endif
   endif

!   print *, "difference ", norm2(q(1:ldq,1:nb+3)-q_extra(1:ldq,1:nb+3)), ", ldq ", ldq, ", nq ", nq, ", nb ", nb

!   print *, q(1:ldq,1:nb+3)
!   stop 1

  end subroutine

#endif
