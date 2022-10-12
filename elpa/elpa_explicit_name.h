//    Copyright 2022, P. Karpov, MPCDF
//
//    This file is part of ELPA.
//
//    The ELPA library was originally created by the ELPA consortium,
//    consisting of the following organizations:
//
//    - Max Planck Computing and Data Facility (MPCDF), formerly known as
//      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
//    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
//      Informatik,
//    - Technische Universität München, Lehrstuhl für Informatik mit
//      Schwerpunkt Wissenschaftliches Rechnen ,
//    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
//    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
//      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
//      and
//    - IBM Deutschland GmbH
//
//
//    This particular source code file contains additions, changes and
//    enhancements authored by Intel Corporation which is not part of
//    the ELPA consortium.
//
//    More information can be found here:
//    http://elpa.mpcdf.mpg.de/
//
//    ELPA is free software: you can redistribute it and/or modify
//    it under the terms of the version 3 of the license of the
//    GNU Lesser General Public License as published by the Free
//    Software Foundation.
//
//    ELPA is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU Lesser General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with ELPA. If not, see <http://www.gnu.org/licenses/>
//
//    ELPA reflects a substantial effort on the part of the original
//    ELPA consortium, and we ask you to respect the spirit of the
//    license that we chose: i.e., please contribute any changes you
//    may have back to the original ELPA library distribution, and keep
//    any derivatives of ELPA under the same license that we chose for
//    the original distribution, the GNU Lesser General Public License.
//
#include <elpa/elpa.h>

void elpa_eigenvectors_double(elpa_t handle, double *a, double *ev, double *q, int *error);
void elpa_eigenvectors_float(elpa_t handle, float *a, float *ev, float *q, int *error);
void elpa_eigenvectors_double_complex(elpa_t handle, double complex *a, double *ev, double complex *q, int *error);
void elpa_eigenvectors_float_complex(elpa_t handle, float complex *a, float *ev, float complex *q, int *error);

void elpa_eigenvalues_double(elpa_t handle, double *a, double *ev, int *error);
void elpa_eigenvalues_float(elpa_t handle, float *a, float *ev, int *error);
void elpa_eigenvalues_double_complex(elpa_t handle, double complex *a, double *ev, int *error);
void elpa_eigenvalues_float_complex(elpa_t handle, float complex *a, float *ev, int *error);

void elpa_cholesky_double(elpa_t handle, double *a, int *error);
void elpa_cholesky_float(elpa_t handle, float *a, int *error);
void elpa_cholesky_double_complex(elpa_t handle, double complex *a, int *error);
void elpa_cholesky_float_complex(elpa_t handle, float complex *a, int *error);

void elpa_hermitian_multiply_double(elpa_t handle, char uplo_a, char uplo_c, int ncb, double *a, double *b, int nrows_b, int ncols_b, double *c, int nrows_c, int ncols_c, int *error);
void elpa_hermitian_multiply_float(elpa_t handle, char uplo_a, char uplo_c, int ncb, float *a, float *b, int nrows_b, int ncols_b, float *c, int nrows_c, int ncols_c, int *error);
void elpa_hermitian_multiply_double_complex(elpa_t handle, char uplo_a, char uplo_c, int ncb, double complex *a, double complex *b, int nrows_b, int ncols_b, double complex *c, int nrows_c, int ncols_c, int *error);
void elpa_hermitian_multiply_float_complex(elpa_t handle, char uplo_a, char uplo_c, int ncb, float complex *a, float complex *b, int nrows_b, int ncols_b, float complex *c, int nrows_c, int ncols_c, int *error);
 
void elpa_invert_triangular_double(elpa_t handle, double *a, int *error);
void elpa_invert_triangular_float(elpa_t handle, float *a, int *error);
void elpa_invert_triangular_double_complex(elpa_t handle, double complex *a, int *error);
void elpa_invert_triangular_float_complex(elpa_t handle, float complex *a, int *error);