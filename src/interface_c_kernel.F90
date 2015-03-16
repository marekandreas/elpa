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
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaftrn,
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

!This is a module contains all CUDA C Calls
! it was provided by NVIDIA with their ELPA GPU port and
! adapted for an ELPA release by A.Marek, RZG

#include "config-f90.h"

#ifdef WITH_GPU_VERSION
module cuda_c_kernel
  implicit none

  interface

    subroutine launch_dot_product_kernel(hs_dev, hv_new_dev, tau_new, x_dev, h_dev,hv_dev, nr) bind(c)

      use iso_c_binding

      implicit none
      integer, value :: nr
      integer(C_SIZE_T), value :: hs_dev ,hv_new_dev,x_dev,h_dev, hv_dev
      complex*16,value         :: tau_new

    end subroutine

    subroutine launch_dot_product_kernel_1(ab_dev, hs_dev, hv_new_dev, x_dev,h_dev,hv_dev,nb, nr, ns) bind(c)

      use iso_c_binding

      implicit none
      integer, value           ::  nb, nr, ns
      integer(C_SIZE_T), value :: x_dev,h_dev, hv_dev, ab_dev, hs_dev,hv_new_dev

    end subroutine

    subroutine launch_dot_product_kernel_2(ab_dev, hs_dev, hv_dev,hd_dev,nb, nr, ne) bind(c)

      use iso_c_binding

      implicit none
      integer, value           ::  nb, nr, ne
      integer(C_SIZE_T), value :: hd_dev,hv_dev, hs_dev, ab_dev

    end subroutine

    subroutine launch_double_hh_transform_1(ab_dev, hs_dev,hv_dev,nb,ns) bind(c)

      use iso_c_binding

      implicit none
      integer, value           ::  nb, ns
      integer(C_SIZE_T), value :: hv_dev, ab_dev,hs_dev

    end subroutine

   subroutine launch_double_hh_transform_2(ab_dev, hd_dev,hv_dev,nc,ns, nb) bind(c)

     use iso_c_binding

     implicit none
     integer, value           ::  nc, ns, nb
     integer(C_SIZE_T), value :: hv_dev, ab_dev,hd_dev
   end subroutine

   subroutine launch_compute_kernel_reduce(a_dev, lda, n, nbw, h1_dev) bind(c)

     use iso_c_binding

     implicit none
     integer, value           :: n,lda,nbw
     integer(C_SIZE_T), value :: h1_dev ,a_dev
   end subroutine

   subroutine launch_compute_kernel_reduce_1(a_dev, lda, n, h1_dev) bind(c)

     use iso_c_binding

     implicit none
     integer, value           :: n,lda
     integer(C_SIZE_T), value :: h1_dev ,a_dev

   end subroutine

   subroutine launch_compute_hh_trafo_c_kernel(q, hh, hh_dot, hh_tau, nev, nb, ldq, off, ncols) bind(c)

     use iso_c_binding

     implicit none
     integer, value           :: nev, nb, ldq, off, ncols
     integer*8, value         :: q
     integer*8, value         :: hh_dot
     integer(C_SIZE_T), value :: hh_tau ,hh
   end subroutine

   subroutine launch_compute_hh_trafo_c_kernel_complex(q, hh, hh_tau, nev, nb,ldq,off, ncols) bind(c)

     use iso_c_binding

     implicit none
     integer, value   :: nev, nb, ldq, off, ncols
     integer*8, value :: q
     integer*8, value :: hh_tau ,hh
   end subroutine

   subroutine launch_compute_hh_trafo_c_kernel_complex_1(q, hh, hh_dot, hh_tau, nev, nb, ldq, off, ncols) bind(c)

     use iso_c_binding

     implicit none
     integer, value   :: nev, nb, ldq, off, ncols
     integer*8, value :: q
     integer*8, value :: hh_tau ,hh, hh_dot

   end subroutine

   subroutine launch_my_unpack_c_kernel(row_count, n_offset, max_idx,stripe_width, a_dim2, stripe_count, l_nev,row_group_dev, &
                                        a_dev) bind(c)

     use iso_c_binding

     implicit none
     integer, value    :: row_count
     integer, value    :: n_offset, max_idx,stripe_width, a_dim2, stripe_count, l_nev
     integer*8, value  :: a_dev, row_group_dev

   end subroutine

   subroutine launch_my_pack_c_kernel(row_count, n_offset, max_idx,stripe_width, a_dim2, stripe_count, l_nev, a_dev, &
                                      row_group_dev) bind(c)

     use iso_c_binding

     implicit none
     integer, value   :: row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev
     integer*8, value :: a_dev
     integer*8, value :: row_group_dev

   end subroutine

   subroutine launch_compute_hh_dotp_c_kernel(bcast_buffer_dev, hh_dot_dev, nbw, n) bind(c)

     use iso_c_binding

     implicit none
     integer*8, value :: bcast_buffer_dev
     integer*8, value :: hh_dot_dev
     integer, value   :: nbw, n

   end subroutine

   subroutine launch_extract_hh_tau_c_kernel(hh, hh_tau, nb, n, is_zero) bind(c)

     use iso_c_binding

     implicit none
     integer*8, value :: hh
     integer*8, value :: hh_tau
     integer, value   :: nb, n
     integer, value   :: is_zero

   end subroutine

   subroutine launch_my_unpack_c_kernel_complex(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, &
                                                row_group_dev, a_dev) bind(c)

     use iso_c_binding

     implicit none

     integer, value    :: row_count
     integer, value    :: n_offset, max_idx,stripe_width, a_dim2, stripe_count,l_nev
     integer*8, value  :: a_dev, row_group_dev

   end subroutine

   subroutine launch_my_pack_c_kernel_complex(row_count, n_offset, max_idx,stripe_width,a_dim2, stripe_count, l_nev, a_dev, &
                                              row_group_dev) bind(c)

     use iso_c_binding

     implicit none
     integer, value   :: row_count, n_offset, max_idx, stripe_width, a_dim2,stripe_count, l_nev
     integer*8, value :: a_dev
     integer*8, value :: row_group_dev

   end subroutine

   subroutine launch_compute_hh_dotp_c_kernel_complex(bcast_buffer_dev, hh_dot_dev, nbw,n) bind(c)

     use iso_c_binding

     implicit none
     integer*8, value :: bcast_buffer_dev
     integer*8, value :: hh_dot_dev
     integer, value   :: nbw, n
   end subroutine

   subroutine launch_extract_hh_tau_c_kernel_complex(hh, hh_tau, nb, n, is_zero) bind(c)

     use iso_c_binding

     implicit none
     integer*8, value :: hh
     integer*8, value :: hh_tau
     integer, value   :: nb, n
     integer, value   :: is_zero

   end subroutine

end interface

end module cuda_c_kernel

#endif /* WITH_GPU_VERSION */

