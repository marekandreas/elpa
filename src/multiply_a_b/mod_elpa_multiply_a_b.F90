!    Copyright 2021, A. Marek
!
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
#include "config-f90.h"

module elpa_multiply_a_b
  use, intrinsic :: iso_c_binding
  use precision
  implicit none

  public

  public :: elpa_mult_at_b_a_h_a_real_double_impl      !< Multiply double-precision real matrices A**T * B
  public :: elpa_mult_at_b_d_ptr_real_double_impl      !< Multiply double-precision real matrices A**T * B (device pointer)

  public :: elpa_mult_ah_b_a_h_a_complex_double_impl   !< Multiply double-precision complex matrices A**H * B
  public :: elpa_mult_ah_b_d_ptr_complex_double_impl   !< Multiply double-precision complex matrices A**H * B (device pointer)

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: elpa_mult_at_b_a_h_a_real_single_impl      !< Multiply single-precision real matrices A**T * B
  public :: elpa_mult_at_b_d_ptr_real_single_impl      !< Multiply single-precision real matrices A**T * B (device pointer)
#endif

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: elpa_mult_ah_b_a_h_a_complex_single_impl   !< Multiply single-precision complex matrices A**H * B
  public :: elpa_mult_ah_b_d_ptr_complex_single_impl   !< Multiply single-precision complex matrices A**H * B (device pointer)
#endif

  contains
#define REALCASE 1
#define DOUBLE_PRECISION
#undef DEVICE_POINTER
#include "../general/precision_macros.h"

!> \brief  elpa_mult_at_b_a_h_a_real_double_impl: Performs C : = A**T * B
!>         where   A is a square matrix (obj%na,obj%na) which is optionally upper or lower triangular
!>                 B is a (obj%na,ncb) matrix
!>                 C is a (obj%na,ncb) matrix where optionally only the upper or lower
!>                   triangle may be computed
!> \details
!>
!> \param  uplo_a               'U' if A is upper triangular
!>                              'L' if A is lower triangular
!>                              anything else if A is a full matrix
!>                              Please note: This pertains to the original A (as set in the calling program)
!>                                           whereas the transpose of A is used for calculations
!>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!>                              i.e. it may contain arbitrary numbers
!> \param uplo_c                'U' if only the upper diagonal part of C is needed
!>                              'L' if only the upper diagonal part of C is needed
!>                              anything else if the full matrix C is needed
!>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!> \param na                    Number of rows/columns of A, number of rows of B and C
!> \param ncb                   Number of columns  of B and C
!> \param a                     matrix a
!> \param obj%local_nrows       leading dimension of matrix a, set with class method obj%set("local_nrows",value)
!> \param b                     matrix b
!> \param ldb                   leading dimension of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c
!> \param ldc                   leading dimension of matrix c
!> \result success
    function elpa_mult_at_b_a_h_a_real_double_impl(obj, uplo_a, uplo_c, ncb, a, b, ldb, ldbCols, &
                                             c, ldc, ldcCols) result(success)
#include "elpa_multiply_a_b_template.F90"
    end function elpa_mult_at_b_a_h_a_real_double_impl
#undef DOUBLE_PRECISION
#undef REALCASE

#define REALCASE 1
#define DOUBLE_PRECISION
#define DEVICE_POINTER
#include "../general/precision_macros.h"

!> \brief  elpa_mult_at_b_d_ptr_real_double_impl: Performs C : = A**T * B
!>         where   A is a square matrix (obj%na,obj%na) which is optionally upper or lower triangular
!>                 B is a (obj%na,ncb) matrix
!>                 C is a (obj%na,ncb) matrix where optionally only the upper or lower
!>                   triangle may be computed
!> \details
!>
!> \param  uplo_a               'U' if A is upper triangular
!>                              'L' if A is lower triangular
!>                              anything else if A is a full matrix
!>                              Please note: This pertains to the original A (as set in the calling program)
!>                                           whereas the transpose of A is used for calculations
!>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!>                              i.e. it may contain arbitrary numbers
!> \param uplo_c                'U' if only the upper diagonal part of C is needed
!>                              'L' if only the upper diagonal part of C is needed
!>                              anything else if the full matrix C is needed
!>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!> \param na                    Number of rows/columns of A, number of rows of B and C
!> \param ncb                   Number of columns  of B and C
!> \param a                     matrix a, as a device pointer of type(c_ptr)
!> \param obj%local_nrows       leading dimension of matrix a, set with class method obj%set("local_nrows",value)
!> \param b                     matrix b, as a device pointer of type(c_ptr)
!> \param ldb                   leading dimension of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c, as a device pointer of type(c_ptr)
!> \param ldc                   leading dimension of matrix c
!> \result success
    function elpa_mult_at_b_d_ptr_real_double_impl(obj, uplo_a, uplo_c, ncb, aDev, bDev, ldb, ldbCols, &
                                             cDev, ldc, ldcCols) result(success)
#include "elpa_multiply_a_b_template.F90"
    end function elpa_mult_at_b_d_ptr_real_double_impl
#undef DOUBLE_PRECISION
#undef REALCASE
#undef DEVICE_POINTER


#ifdef WANT_SINGLE_PRECISION_REAL
#define REALCASE 1
#define SINGLE_PRECISION
#undef DEVICE_POINTER
#include "../general/precision_macros.h"
!> \brief  elpa_mult_at_b_real_a_h_a_single_impl: Performs C : = A**T * B
!>         where   A is a square matrix (obj%na,obj%na) which is optionally upper or lower triangular
!>                 B is a (obj%na,ncb) matrix
!>                 C is a (obj%na,ncb) matrix where optionally only the upper or lower
!>                   triangle may be computed
!> \details
!>
!> \param  uplo_a               'U' if A is upper triangular
!>                              'L' if A is lower triangular
!>                              anything else if A is a full matrix
!>                              Please note: This pertains to the original A (as set in the calling program)
!>                                           whereas the transpose of A is used for calculations
!>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!>                              i.e. it may contain arbitrary numbers
!> \param uplo_c                'U' if only the upper diagonal part of C is needed
!>                              'L' if only the upper diagonal part of C is needed
!>                              anything else if the full matrix C is needed
!>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!> \param na                    Number of rows/columns of A, number of rows of B and C
!> \param ncb                   Number of columns  of B and C
!> \param a                     matrix a
!> \param obj%local_nrows       leading dimension of matrix a, set with class method obj%set("local_nrows",value)
!> \param b                     matrix b
!> \param ldb                   leading dimension of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c
!> \param ldc                   leading dimension of matrix c
!> \result success
    function elpa_mult_at_b_a_h_a_real_single_impl(obj, uplo_a, uplo_c, ncb, a, b, ldb, ldbCols, &
                                             c, ldc, ldcCols) result(success)

#include "elpa_multiply_a_b_template.F90"

    end function elpa_mult_at_b_a_h_a_real_single_impl
#undef SINGLE_PRECISION
#undef REALCASE

#define REALCASE 1
#define SINGLE_PRECISION
#define DEVICE_POINTER
#include "../general/precision_macros.h"
!> \brief  elpa_mult_at_b_real_d_ptr_single_impl: Performs C : = A**T * B
!>         where   A is a square matrix (obj%na,obj%na) which is optionally upper or lower triangular
!>                 B is a (obj%na,ncb) matrix
!>                 C is a (obj%na,ncb) matrix where optionally only the upper or lower
!>                   triangle may be computed
!> \details
!>
!> \param  uplo_a               'U' if A is upper triangular
!>                              'L' if A is lower triangular
!>                              anything else if A is a full matrix
!>                              Please note: This pertains to the original A (as set in the calling program)
!>                                           whereas the transpose of A is used for calculations
!>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!>                              i.e. it may contain arbitrary numbers
!> \param uplo_c                'U' if only the upper diagonal part of C is needed
!>                              'L' if only the upper diagonal part of C is needed
!>                              anything else if the full matrix C is needed
!>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!> \param na                    Number of rows/columns of A, number of rows of B and C
!> \param ncb                   Number of columns  of B and C
!> \param a                     matrix a, as a device pointer of type(c_ptr)
!> \param obj%local_nrows       leading dimension of matrix a, set with class method obj%set("local_nrows",value)
!> \param b                     matrix b, as a device pointer of type(c_ptr)
!> \param ldb                   leading dimension of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c, as a device pointer of type(c_ptr)
!> \param ldc                   leading dimension of matrix c
!> \result success
    function elpa_mult_at_b_d_ptr_real_single_impl(obj, uplo_a, uplo_c, ncb, aDev, bDev, ldb, ldbCols, &
                                             cDev, ldc, ldcCols) result(success)

#include "elpa_multiply_a_b_template.F90"

    end function elpa_mult_at_b_d_ptr_real_single_impl
#undef SINGLE_PRECISION
#undef REALCASE
#undef DEVICE_POINTER


#endif /* WANT_SINGLE_PRECSION_REAL */

#define COMPLEXCASE 1
#define DOUBLE_PRECISION
#undef DEVICE_POINTER
#include "../general/precision_macros.h"
!> \brief  elpa_mult_ah_b_a_h_a_complex_double_impl: Performs C : = A**H * B
!>         where   A is a square matrix (obj%na,obj%na) which is optionally upper or lower triangular
!>                 B is a (obj%na,ncb) matrix
!>                 C is a (obj%na,ncb) matrix where optionally only the upper or lower
!>                   triangle may be computed
!> \details
!>
!> \param  uplo_a               'U' if A is upper triangular
!>                              'L' if A is lower triangular
!>                              anything else if A is a full matrix
!>                              Please note: This pertains to the original A (as set in the calling program)
!>                                           whereas the transpose of A is used for calculations
!>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!>                              i.e. it may contain arbitrary numbers
!> \param uplo_c                'U' if only the upper diagonal part of C is needed
!>                              'L' if only the upper diagonal part of C is needed
!>                              anything else if the full matrix C is needed
!>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!> \param na                    Number of rows/columns of A, number of rows of B and C
!> \param ncb                   Number of columns  of B and C
!> \param a                     matrix a
!> \param obj%local_ncols       leading dimension of matrix a, set with class method obj%set("local_nrows",value)
!> \param ldaCols               columns of matrix a
!> \param b                     matrix b
!> \param ldb                   leading dimension of matrix b
!> \param ldbCols               columns of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c
!> \param ldc                   leading dimension of matrix c
!> \result success
    function elpa_mult_ah_b_a_h_a_complex_double_impl(obj, uplo_a, uplo_c, ncb, a, b, ldb, ldbCols, &
                                                c, ldc, ldcCols) result(success)
#include "elpa_multiply_a_b_template.F90"

    end function elpa_mult_ah_b_a_h_a_complex_double_impl

#define COMPLEXCASE 1
#define DOUBLE_PRECISION
#define DEVICE_POINTER
#include "../general/precision_macros.h"
!> \brief  elpa_mult_ah_b_a_h_a_complex_double_impl: Performs C : = A**H * B
!>         where   A is a square matrix (obj%na,obj%na) which is optionally upper or lower triangular
!>                 B is a (obj%na,ncb) matrix
!>                 C is a (obj%na,ncb) matrix where optionally only the upper or lower
!>                   triangle may be computed
!> \details
!>
!> \param  uplo_a               'U' if A is upper triangular
!>                              'L' if A is lower triangular
!>                              anything else if A is a full matrix
!>                              Please note: This pertains to the original A (as set in the calling program)
!>                                           whereas the transpose of A is used for calculations
!>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!>                              i.e. it may contain arbitrary numbers
!> \param uplo_c                'U' if only the upper diagonal part of C is needed
!>                              'L' if only the upper diagonal part of C is needed
!>                              anything else if the full matrix C is needed
!>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!> \param na                    Number of rows/columns of A, number of rows of B and C
!> \param ncb                   Number of columns  of B and C
!> \param a                     matrix a, as a device pointer of type(c_ptr)
!> \param obj%local_ncols       leading dimension of matrix a, set with class method obj%set("local_nrows",value)
!> \param ldaCols               columns of matrix a
!> \param b                     matrix b, as a device pointer of type(c_ptr)
!> \param ldb                   leading dimension of matrix b
!> \param ldbCols               columns of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c, as a device_pointer of type(c_ptr)
!> \param ldc                   leading dimension of matrix c
!> \result success
    function elpa_mult_ah_b_d_ptr_complex_double_impl(obj, uplo_a, uplo_c, ncb, aDev, bDev, ldb, ldbCols, &
                                                cDev, ldc, ldcCols) result(success)
#include "elpa_multiply_a_b_template.F90"

    end function elpa_mult_ah_b_d_ptr_complex_double_impl

#undef DOUBLE_PRECISION
#undef COMPLEXCASE
#undef DEVICE_POINTER

#ifdef WANT_SINGLE_PRECISION_COMPLEX
#define COMPLEXCASE 1
#define SINGLE_PRECISION
#undef DEVICE_POINTER
#include "../general/precision_macros.h"
!> \brief  elpa_mult_ah_b_a_h_a_complex_single_impl: Performs C : = A**H * B
!>         where   A is a square matrix (obj%na,obj%na) which is optionally upper or lower triangular
!>                 B is a (obj%na,ncb) matrix
!>                 C is a (obj%na,ncb) matrix where optionally only the upper or lower
!>                   triangle may be computed
!> \details
!>
!> \param  uplo_a               'U' if A is upper triangular
!>                              'L' if A is lower triangular
!>                              anything else if A is a full matrix
!>                              Please note: This pertains to the original A (as set in the calling program)
!>                                           whereas the transpose of A is used for calculations
!>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!>                              i.e. it may contain arbitrary numbers
!> \param uplo_c                'U' if only the upper diagonal part of C is needed
!>                              'L' if only the upper diagonal part of C is needed
!>                              anything else if the full matrix C is needed
!>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!> \param na                    Number of rows/columns of A, number of rows of B and C
!> \param ncb                   Number of columns  of B and C
!> \param a                     matrix a
!> \param obj%local_ncols       leading dimension of matrix a, set with class method obj%set("local_nrows",value)
!> \param ldaCols               columns of matrix a
!> \param b                     matrix b
!> \param ldb                   leading dimension of matrix b
!> \param ldbCols               columns of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c
!> \param ldc                   leading dimension of matrix c
!> \result success
    function elpa_mult_ah_b_a_h_a_complex_single_impl(obj, uplo_a, uplo_c, ncb, a, b, ldb, ldbCols, &
                                                c, ldc, ldcCols) result(success)

#include "elpa_multiply_a_b_template.F90"

    end function elpa_mult_ah_b_a_h_a_complex_single_impl
#undef SINGLE_PRECISION
#undef COMPLEXCASE

#define COMPLEXCASE 1
#define SINGLE_PRECISION
#define DEVICE_POINTER
#include "../general/precision_macros.h"
!> \brief  elpa_mult_ah_b_d_ptr_complex_single_impl: Performs C : = A**H * B
!>         where   A is a square matrix (obj%na,obj%na) which is optionally upper or lower triangular
!>                 B is a (obj%na,ncb) matrix
!>                 C is a (obj%na,ncb) matrix where optionally only the upper or lower
!>                   triangle may be computed
!> \details
!>
!> \param  uplo_a               'U' if A is upper triangular
!>                              'L' if A is lower triangular
!>                              anything else if A is a full matrix
!>                              Please note: This pertains to the original A (as set in the calling program)
!>                                           whereas the transpose of A is used for calculations
!>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!>                              i.e. it may contain arbitrary numbers
!> \param uplo_c                'U' if only the upper diagonal part of C is needed
!>                              'L' if only the upper diagonal part of C is needed
!>                              anything else if the full matrix C is needed
!>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!> \param na                    Number of rows/columns of A, number of rows of B and C
!> \param ncb                   Number of columns  of B and C
!> \param a                     matrix a, as a device pointer of type(c_ptr)
!> \param obj%local_ncols       leading dimension of matrix a, set with class method obj%set("local_nrows",value)
!> \param ldaCols               columns of matrix a
!> \param b                     matrix b, as a device pointer of type(c_ptr)
!> \param ldb                   leading dimension of matrix b
!> \param ldbCols               columns of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c, as a device pointer of type(c_ptr)
!> \param ldc                   leading dimension of matrix c
!> \result success
    function elpa_mult_ah_b_d_ptr_complex_single_impl(obj, uplo_a, uplo_c, ncb, aDev, bDev, ldb, ldbCols, &
                                                cDev, ldc, ldcCols) result(success)

#include "elpa_multiply_a_b_template.F90"

    end function elpa_mult_ah_b_d_ptr_complex_single_impl
#undef SINGLE_PRECISION
#undef COMPLEXCASE

#endif /* WANT_SINGLE_PRECISION_COMPLEX */


end module

