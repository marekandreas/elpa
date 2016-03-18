/*     This file is part of ELPA. */
/*  */
/*     The ELPA library was originally created by the ELPA consortium, */
/*     consisting of the following organizations: */
/*  */
/*     - Max Planck Computing and Data Facility (MPCDF), formerly known as */
/*       Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG), */
/*     - Bergische Universität Wuppertal, Lehrstuhl für angewandte */
/*       Informatik, */
/*     - Technische Universität München, Lehrstuhl für Informatik mit */
/*       Schwerpunkt Wissenschaftliches Rechnen , */
/*     - Fritz-Haber-Institut, Berlin, Abt. Theorie, */
/*     - Max-Plack-Institut für Mathematik in den Naturwissenschaften, */
/*       Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition, */
/*       and */
/*     - IBM Deutschland GmbH */
/*  */
/*  */
/*     More information can be found here: */
/*     http://elpa.mpcdf.mpg.de/ */
/*  */
/*     ELPA is free software: you can redistribute it and/or modify */
/*     it under the terms of the version 3 of the license of the */
/*     GNU Lesser General Public License as published by the Free */
/*     Software Foundation. */
/*  */
/*     ELPA is distributed in the hope that it will be useful, */
/*     but WITHOUT ANY WARRANTY; without even the implied warranty of */
/*     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/*     GNU Lesser General Public License for more details. */
/*  */
/*     You should have received a copy of the GNU Lesser General Public License */
/*     along with ELPA.  If not, see <http://www.gnu.org/licenses/> */
/*  */
/*     ELPA reflects a substantial effort on the part of the original */
/*     ELPA consortium, and we ask you to respect the spirit of the */
/*     license that we chose: i.e., please contribute any changes you */
/*     may have back to the original ELPA library distribution, and keep */
/*     any derivatives of ELPA under the same license that we chose for */
/*     the original distribution, the GNU Lesser General Public License. */
/*  */
/*  */
#include "config-f90.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <elpa/elpa.h>
#include <complex.h>

int call_elpa1_real_solver_from_c_double(int na, int nev, double *a, int lda, double *ev, double *q, int ldq, int nblk, int ncols, int mpi_comm_rows, int mpi_comm_cols) {
  return elpa_solve_evp_real_1stage_double_precision(na, nev, a, lda, ev, q, ldq, nblk, ncols, mpi_comm_rows, mpi_comm_cols);
}
#ifdef WANT_SINGLE_PRECISION_REAL
int call_elpa1_real_solver_from_c_single(int na, int nev, float *a, int lda, float *ev, float *q, int ldq, int nblk, int ncols, int mpi_comm_rows, int mpi_comm_cols) {
  return elpa_solve_evp_real_1stage_single_precision(na, nev, a, lda, ev, q, ldq, nblk, ncols, mpi_comm_rows, mpi_comm_cols);
}
#endif

int call_elpa1_complex_solver_from_c_double(int na, int nev, complex double *a, int lda, double *ev, complex double *q, int ldq, int nblk, int ncols, int mpi_comm_rows, int mpi_comm_cols) {
  return elpa_solve_evp_complex_1stage_double_precision(na, nev, a, lda, ev, q, ldq, nblk, ncols, mpi_comm_rows, mpi_comm_cols);
}
#ifdef WANT_SINGLE_PRECISION_COMPLEX
int call_elpa1_complex_solver_from_c_single(int na, int nev, complex  *a, int lda, float *ev, complex  *q, int ldq, int nblk, int ncols, int mpi_comm_rows, int mpi_comm_cols) {
  return elpa_solve_evp_complex_1stage_single_precision(na, nev, a, lda, ev, q, ldq, nblk, ncols, mpi_comm_rows, mpi_comm_cols);
}
#endif

int call_elpa_get_comm_from_c(int mpi_comm_world, int my_prow, int my_pcol, int *mpi_comm_rows, int *mpi_comm_cols) {
  return elpa_get_communicators(mpi_comm_world, my_prow, my_pcol, mpi_comm_rows, mpi_comm_cols);
}
