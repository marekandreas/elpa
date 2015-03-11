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
module mod_prepare_matrix

  interface prepare_matrix
    module procedure prepare_matrix_complex
        module procedure prepare_matrix_real
  end interface

  contains

    subroutine prepare_matrix_complex(na, myid, sc_desc, iseed, xr, a, z, as)

      implicit none

      integer, intent(in)       :: myid, na, sc_desc(:)
      integer, intent(inout)    :: iseed(:)
      real*8, intent(inout)     :: xr(:,:)
      complex*16, intent(inout) :: z(:,:), a(:,:), as(:,:)

      complex*16, parameter     :: CZERO = (0.d0, 0.d0), CONE = (1.d0, 0.d0)

      ! for getting a hermitian test matrix A we get a random matrix Z
      ! and calculate A = Z + Z**H

      ! we want different random numbers on every process
      ! (otherwise A might get rank deficient):

      iseed(:) = myid
      call RANDOM_SEED(put=iseed)
      call RANDOM_NUMBER(xr)
      z(:,:) = xr(:,:)
      call RANDOM_NUMBER(xr)
      z(:,:) = z(:,:) + (0.d0,1.d0)*xr(:,:)

      a(:,:) = z(:,:)

      if (myid == 0) then
        print '(a)','| Random matrix block has been set up. (only processor 0 confirms this step)'
      endif

      call pztranc(na, na, CONE, z, 1, 1, sc_desc, CONE, a, 1, 1, sc_desc) ! A = A + Z**H

      if (myid == 0) then
        print '(a)','| Random matrix block has been symmetrized'
      endif

      ! save original matrix A for later accuracy checks

      as = a

    end subroutine

    subroutine prepare_matrix_real(na, myid, sc_desc, iseed, a, z, as)

      implicit none

      integer, intent(in)       :: myid, na, sc_desc(:)
      integer, intent(inout)    :: iseed(:)
      real*8, intent(inout)     :: z(:,:), a(:,:), as(:,:)

      ! for getting a hermitian test matrix A we get a random matrix Z
      ! and calculate A = Z + Z**H

      ! we want different random numbers on every process
      ! (otherwise A might get rank deficient):

      iseed(:) = myid
      call RANDOM_SEED(put=iseed)
      call RANDOM_NUMBER(z)

      a(:,:) = z(:,:)

      if (myid == 0) then
        print '(a)','| Random matrix block has been set up. (only processor 0 confirms this step)'
      endif

      call pdtran(na, na, 1.d0, z, 1, 1, sc_desc, 1.d0, a, 1, 1, sc_desc) ! A = A + Z**T

      if (myid == 0) then
        print '(a)','| Random matrix block has been symmetrized'
      endif

      ! save original matrix A for later accuracy checks

      as = a

    end subroutine

    subroutine prepare_matrix_real_wrapper(na, myid, na_rows, na_cols, sc_desc, iseed, a, z, as) &
                                          bind(C, name="prepare_matrix_real_from_fortran")
      use iso_c_binding

      implicit none

      integer(kind=c_int) , value   :: myid, na, na_rows, na_cols
      integer(kind=c_int)           :: sc_desc(1:9)
      integer(kind=c_int)           :: iseed(1:4096)
      real(kind=c_double)           :: z(1:na_rows,1:na_cols), a(1:na_rows,1:na_cols),  &
                                       as(1:na_rows,1:na_cols)

      print *,"in prepare wrapper"
      call prepare_matrix_real(na, myid, sc_desc, iseed, a, z, as)
    end subroutine

end module
