#if 0
!
!    Copyright 2017, L. Hüdepohl and A. Marek, MPCDF
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
#endif

#include <elpa/elpa_constants.h>

#define FORTRAN_CONSTANT(name, value, ...) \
        integer(kind=C_INT), parameter :: name = value !ELPA_C_DEFINE

! General constants
 ELPA_FOR_ALL_ERRORS(FORTRAN_CONSTANT)

! matrix layout constants
 ELPA_FOR_ALL_MATRIX_LAYOUTS(FORTRAN_CONSTANT)

! Solver constants
 ELPA_FOR_ALL_SOLVERS(FORTRAN_CONSTANT)
#undef ELPA_NUMBER_OF_SOLVERS
 FORTRAN_CONSTANT(ELPA_NUMBER_OF_SOLVERS, (0 ELPA_FOR_ALL_SOLVERS(ELPA_ENUM_SUM)))


! Real kernels
 ELPA_FOR_ALL_2STAGE_REAL_KERNELS_AND_DEFAULT(FORTRAN_CONSTANT)
#undef ELPA_2STAGE_NUMBER_OF_REAL_KERNELS
 FORTRAN_CONSTANT(ELPA_2STAGE_NUMBER_OF_REAL_KERNELS, & NEWLINE (0 ELPA_FOR_ALL_2STAGE_REAL_KERNELS(ELPA_ENUM_SUM)))


! Complex kernels
 ELPA_FOR_ALL_2STAGE_COMPLEX_KERNELS_AND_DEFAULT(FORTRAN_CONSTANT)
#undef ELPA_2STAGE_NUMBER_OF_COMPLEX_KERNELS
 FORTRAN_CONSTANT(ELPA_2STAGE_NUMBER_OF_COMPLEX_KERNELS, & NEWLINE (0 ELPA_FOR_ALL_2STAGE_COMPLEX_KERNELS(ELPA_ENUM_SUM)))


! Autotune
 ELPA_FOR_ALL_AUTOTUNE_LEVELS(FORTRAN_CONSTANT)
 ELPA_FOR_ALL_AUTOTUNE_DOMAINS(FORTRAN_CONSTANT)
 ELPA_FOR_ALL_AUTOTUNE_PARTS(FORTRAN_CONSTANT)
#undef ELPA_NUMBER_OF_AUTOTUNE_LEVELS
 FORTRAN_CONSTANT(ELPA_NUMBER_OF_AUTOTUNE_LEVELS, & NEWLINE (0 ELPA_FOR_ALL_AUTOTUNE_LEVELS(ELPA_ENUM_SUM)))

