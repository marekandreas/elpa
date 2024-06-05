#if 0
!    Copyright 2024, A. Marek
!
!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), fomerly known as
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
! This file was written by A. Marek, MPCDF
#endif

subroutine get_hh_vec_&
&MATH_DATATYPE&
&_&
&PRECISION &
(obj, my_prow, nrow, nblk, np_rows, mpi_comm_rows, lr, allreduce_request1, &
 useNonBlockingCollectivesRows, wantDebug, vec_in, vr, tau, vrl)
  use precision
  use elpa1_compute

  use elpa_abstract_impl
  use elpa_mpi
  use ELPA_utilities

  implicit none
#include "../general/precision_kinds.F90"

  class(elpa_abstract_impl_t), intent(inout)  :: obj
  integer(kind=ik), intent(in)                :: nrow, my_prow, nblk, np_rows, lr
  integer(kind=ik), intent(in)                :: mpi_comm_rows
  logical, intent(in)                         :: useNonBlockingCollectivesRows, wantDebug
  integer(kind=MPI_KIND)                      :: mpierr
  integer(kind=MPI_KIND), intent(inout)       :: allreduce_request1
  MATH_DATATYPE(kind=rck):: vr(:), vec_in(:), tau, vrl
  MATH_DATATYPE(kind=rck):: aux1(2), xf
  real(kind=rk):: vnorm2
  ! Get Vector to be transformed; distribute last element and norm of
  ! remaining elements to all procs in current column

  if (my_prow==prow(nrow, nblk, np_rows)) then
     aux1(1) = dot_product(vec_in(1:lr-1),vec_in(1:lr-1))
     aux1(2) = vec_in(lr)
  else
     aux1(1) = dot_product(vec_in(1:lr),vec_in(1:lr))
     aux1(2) = 0.0_rck
  endif

#ifdef WITH_MPI
  if (useNonBlockingCollectivesRows) then
     if (wantDebug) call obj%timer%start("mpi_nbc_communication")
     call mpi_iallreduce(MPI_IN_PLACE, aux1, 2_MPI_KIND, MPI_MATH_DATATYPE_PRECISION, &
          MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), &
          allreduce_request1, mpierr)

     call mpi_wait(allreduce_request1, MPI_STATUS_IGNORE, mpierr)

     if (wantDebug) call obj%timer%stop("mpi_nbc_communication")
  else
     !             if (wantDebug)             call obj%timer%start("mpi_comm")
     call mpi_allreduce(MPI_IN_PLACE, aux1, 2_MPI_KIND, MPI_MATH_DATATYPE_PRECISION, &
          MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), &
          mpierr)

     !            if (wantDebug)            call obj%timer%stop("mpi_comm")
  endif

#endif

#if REALCASE == 1
  vnorm2 = aux1(1)
#endif
#if COMPLEXCASE == 1
  vnorm2 = real(aux1(1),kind=rk)
#endif
  vrl    = aux1(2)

  ! Householder transformation
  call hh_transform_&
       &MATH_DATATYPE&
       &_&
       &PRECISION &
       (obj, vrl, vnorm2, xf, tau, wantDebug)
  ! Scale vr and store Householder Vector for back transformation

  vr(1:lr) = vec_in(1:lr) * xf
  if (my_prow==prow(nrow, nblk, np_rows)) vr(lr) = 1.0_rck

end


subroutine apply_ht_&
&MATH_DATATYPE&
&_&
&PRECISION &
  (obj, max_threads, lr, nbw, mpi_comm_rows, useNonBlockingCollectivesRows, allreduce_request3, wantDebug, tau, vr, ex_buff2d)
  use precision
  use elpa_abstract_impl
  use elpa_mpi
  use ELPA_utilities

  implicit none
  class(elpa_abstract_impl_t), intent(inout)  :: obj
  integer(kind=MPI_KIND)                      :: mpierr
  integer(kind=ik),intent(in)                 :: max_threads, lr, nbw
  integer(kind=ik), intent(in)                :: mpi_comm_rows
  logical, intent(in)                         :: useNonBlockingCollectivesRows, wantDebug
  integer(kind=MPI_KIND), intent(inout)       :: allreduce_request3

#include "../general/precision_kinds.F90"
  MATH_DATATYPE(kind=rck)                     :: tau, vr(:), ex_buff2d(:,:)
  MATH_DATATYPE(kind=rck)                     :: tauc
  MATH_DATATYPE(kind=rck)                     :: aux1(nbw)
  integer                                     :: nlc, imax
  logical                                     :: use_blas

  imax=ubound(ex_buff2d,2)

  if((imax.lt.3).or.(max_threads.gt.1)) then
     !don't use BLAS for very small imax because overhead is too high
     !don't use BLAS with OpenMP because measurements showed that threading is not effective for these routines
     use_blas=.false.
  else
     use_blas=.true.
  end if

  !we need to transform the remaining ex_buff
  if (lr>0) then
     if(use_blas) then !note that aux1 is conjg between > and < thresh_blas!!
        call PRECISION_GEMV(BLAS_TRANS_OR_CONJ,int(lr,kind=BLAS_KIND),int(imax,kind=BLAS_KIND), &
             ONE, ex_buff2d, size(ex_buff2d,1,kind=BLAS_KIND), vr, 1_BLAS_KIND, ZERO, aux1, &
             1_BLAS_KIND)
     else
#ifdef WITH_OPENMP_TRADITIONAL
        !$omp  parallel do private(nlc)
#endif
        do nlc=1,imax
           aux1(nlc) = dot_product(vr(1:lr),ex_buff2d(1:lr,nlc))
        end do
#ifdef WITH_OPENMP_TRADITIONAL
        !$omp end parallel do
#endif
     end if
  else
     aux1(1:imax) = 0.
  end if

  ! Get global dot products
#ifdef WITH_MPI
  if (useNonBlockingCollectivesRows) then
     if (wantDebug) call obj%timer%start("mpi_nbc_communication")
     if (imax > 0) then
        call mpi_iallreduce(MPI_IN_PLACE, aux1, int(imax,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
             MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), &
             allreduce_request3, mpierr)
        call mpi_wait(allreduce_request3, MPI_STATUS_IGNORE, mpierr)
     endif
     if (wantDebug) call obj%timer%stop("mpi_nbc_communication")
  else
     if (wantDebug) call obj%timer%start("mpi_communication")
     if (imax>0) then
        call mpi_allreduce(MPI_IN_PLACE, aux1, int(imax,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
             MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), &
             mpierr)
     endif
     if (wantDebug) call obj%timer%stop("mpi_communication")
  endif
#endif /* WITH_MPI */

  if(lr.le.0) return !no data on this processor

  ! Transform
#if REALCASE == 1
  tauc=-tau
#else
  tauc=-conjg(tau)
#endif
  if(use_blas) then
     call PRECISION_GERC(int(lr,kind=BLAS_KIND),int(imax,kind=BLAS_KIND),tauc,vr,1_BLAS_KIND,&
          aux1,1_BLAS_KIND,ex_buff2d,ubound(ex_buff2d,1,kind=BLAS_KIND))
  else
#ifdef WITH_OPENMP_TRADITIONAL
     !$omp  parallel do private(nlc)
#endif
     do nlc=1,imax
        ex_buff2d(1:lr,nlc) = ex_buff2d(1:lr,nlc) + tauc*aux1(nlc)*vr(1:lr)
     end do
#ifdef WITH_OPENMP_TRADITIONAL
     !$omp end parallel do
#endif
  end if

end

