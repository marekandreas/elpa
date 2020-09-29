#if 0
!
!    Copyright 2019, A. Marek, MPCDF
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

#ifdef HAVE_64BIT_INTEGER_MATH_SUPPORT
#define C_INT_TYPE_PTR long int*
#define C_INT_TYPE long int
#else
#define C_INT_TYPE_PTR int*
#define C_INT_TYPE int
#endif

void dlacpy_(char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, double*, C_INT_TYPE_PTR, double*, C_INT_TYPE_PTR);
void dgemm_(char*, char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, double*, double*, C_INT_TYPE_PTR, double*, C_INT_TYPE_PTR, double*, double*, C_INT_TYPE_PTR); 


void slacpy_(char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, float*, C_INT_TYPE_PTR, float*, C_INT_TYPE_PTR);
void sgemm_(char*, char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, float*, float*, C_INT_TYPE_PTR, float*, C_INT_TYPE_PTR, float*, float*, C_INT_TYPE_PTR); 




void zlacpy_(char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, double complex*, C_INT_TYPE_PTR, double complex*, C_INT_TYPE_PTR);
void zgemm_(char*, char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, double complex*, double complex*, C_INT_TYPE_PTR, double complex*, C_INT_TYPE_PTR, double complex*, double complex*, C_INT_TYPE_PTR); 


void clacpy_(char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, float complex*, C_INT_TYPE_PTR, float complex*, C_INT_TYPE_PTR);
void cgemm_(char*, char*, C_INT_TYPE_PTR, C_INT_TYPE_PTR, C_INT_TYPE_PTR, float complex*, float complex*, C_INT_TYPE_PTR, float complex*, C_INT_TYPE_PTR, float complex*, float complex*, C_INT_TYPE_PTR); 


