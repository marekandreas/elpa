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
  &_generic_blas_4hv_&
  &PRECISION&
  & (q, hh, nb, nq, ldq, ldh)

    use precision
    use elpa_abstract_impl
    implicit none

    !class(elpa_abstract_impl_t), intent(inout) :: obj
    integer(kind=ik), intent(in)    :: nb, nq, ldq, ldh

#ifdef USE_ASSUMED_SIZE
    real(kind=C_DATATYPE_KIND), intent(inout) :: q(ldq,*)
    real(kind=C_DATATYPE_KIND), intent(in)    :: hh(ldh,*)
#else
    real(kind=C_DATATYPE_KIND), intent(inout) :: q(1:ldq,1:nb+3)
    real(kind=C_DATATYPE_KIND), intent(in)    :: hh(1:ldh,1:6)
#endif

    real(kind=C_DATATYPE_KIND)                :: w_comb(nq, 4)
    real(kind=C_DATATYPE_KIND)                :: h_mat(4, nb+3)
    real(kind=C_DATATYPE_KIND)                :: s_mat(4, 4)

    integer(kind=ik)                             :: i


    ! Calculate dot product of the two Householder vectors

   h_mat(:,:) = 0.0

   h_mat(1,4) = -1.0
   h_mat(2,3) = -1.0
   h_mat(3,2) = -1.0
   h_mat(4,1) = -1.0

   h_mat(1,5:nb+3) = -hh(2:nb, 1)
   h_mat(2,4:nb+2) = -hh(2:nb, 2)
   h_mat(3,3:nb+1) = -hh(2:nb, 3)
   h_mat(4,2:nb)   = -hh(2:nb, 4)

   ! TODO we actually need just the strictly upper triangle of s_mat
   ! TODO take care when changing to BLAS
   ! TODO we do not even need diagonal, which might not be achievable by blas.
   ! TODO lets see how much does it matter
   s_mat = - matmul(h_mat, transpose(h_mat))
   s_mat(1,1) = 1
   s_mat(2,2) = 1
   s_mat(3,3) = 1
   s_mat(4,4) = 1

   w_comb = matmul(q(1:ldq, 1:nb+3), -transpose(h_mat))

   ! Rank-1 update
   w_comb(1:nq,1) = w_comb(1:nq,1) * hh(1,1) * s_mat(1,1)
   w_comb(1:nq,2) = matmul(w_comb(1:nq,1:2), hh(1,2) * s_mat(2,1:2))
   w_comb(1:nq,3) = matmul(w_comb(1:nq,1:3), hh(1,3) * s_mat(3,1:3))
   w_comb(1:nq,4) = matmul(w_comb(1:nq,1:4), hh(1,4) * s_mat(4,1:4))

   q(1:nq, 1:nb+3) = matmul(w_comb, h_mat) + q(1:nq, 1:nb+3)

  end subroutine

#endif
