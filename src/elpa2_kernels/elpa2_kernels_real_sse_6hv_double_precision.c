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
//    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
//
//    ELPA reflects a substantial effort on the part of the original
//    ELPA consortium, and we ask you to respect the spirit of the
//    license that we chose: i.e., please contribute any changes you
//    may have back to the original ELPA library distribution, and keep
//    any derivatives of ELPA under the same license that we chose for
//    the original distribution, the GNU Lesser General Public License.
//
//
// --------------------------------------------------------------------------------------------------
//
// This file contains the compute intensive kernels for the Householder transformations.
// It should be compiled with the highest possible optimization level.
//
// On Intel Nehalem or Intel Westmere or AMD Magny Cours use -O3 -msse3
// On Intel Sandy Bridge use -O3 -mavx
//
// Copyright of the original code rests with the authors inside the ELPA
// consortium. The copyright of any additional modifications shall rest
// with their original authors, but shall adhere to the licensing terms
// distributed along with the original code in the file "COPYING".
//
// Author: Alexander Heinecke (alexander.heinecke@mytum.de)
// Adapted for building a shared-library by Andreas Marek, MPCDF (andreas.marek@mpcdf.mpg.de)
// --------------------------------------------------------------------------------------------------

#include "config-f90.h"

#include <x86intrin.h>

#define __forceinline __attribute__((always_inline)) static

#ifdef HAVE_SSE_INTRINSICS
#undef __AVX__
#endif

//Forward declaration
static void hh_trafo_kernel_2_SSE_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);
static void hh_trafo_kernel_4_SSE_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);
void hexa_hh_trafo_real_sse_6hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);

/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine hexa_hh_trafo_real_sse_6hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="hexa_hh_trafo_real_sse_6hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_double)     :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

void hexa_hh_trafo_real_sse_6hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;

	// calculating scalar products to compute
	// 6 householder vectors simultaneously
	double scalarprods[15];

//	scalarprods[0] = s_1_2;
//	scalarprods[1] = s_1_3;
//	scalarprods[2] = s_2_3;
//	scalarprods[3] = s_1_4;
//	scalarprods[4] = s_2_4;
//	scalarprods[5] = s_3_4;
//	scalarprods[6] = s_1_5;
//	scalarprods[7] = s_2_5;
//	scalarprods[8] = s_3_5;
//	scalarprods[9] = s_4_5;
//	scalarprods[10] = s_1_6;
//	scalarprods[11] = s_2_6;
//	scalarprods[12] = s_3_6;
//	scalarprods[13] = s_4_6;
//	scalarprods[14] = s_5_6;

	scalarprods[0] = hh[(ldh+1)];
	scalarprods[1] = hh[(ldh*2)+2];
	scalarprods[2] = hh[(ldh*2)+1];
	scalarprods[3] = hh[(ldh*3)+3];
	scalarprods[4] = hh[(ldh*3)+2];
	scalarprods[5] = hh[(ldh*3)+1];
	scalarprods[6] = hh[(ldh*4)+4];
	scalarprods[7] = hh[(ldh*4)+3];
	scalarprods[8] = hh[(ldh*4)+2];
	scalarprods[9] = hh[(ldh*4)+1];
	scalarprods[10] = hh[(ldh*5)+5];
	scalarprods[11] = hh[(ldh*5)+4];
	scalarprods[12] = hh[(ldh*5)+3];
	scalarprods[13] = hh[(ldh*5)+2];
	scalarprods[14] = hh[(ldh*5)+1];

	// calculate scalar product of first and fourth householder vector
	// loop counter = 2
	scalarprods[0] += hh[1] * hh[(2+ldh)];
	scalarprods[2] += hh[(ldh)+1] * hh[2+(ldh*2)];
	scalarprods[5] += hh[(ldh*2)+1] * hh[2+(ldh*3)];
	scalarprods[9] += hh[(ldh*3)+1] * hh[2+(ldh*4)];
	scalarprods[14] += hh[(ldh*4)+1] * hh[2+(ldh*5)];

	// loop counter = 3
	scalarprods[0] += hh[2] * hh[(3+ldh)];
	scalarprods[2] += hh[(ldh)+2] * hh[3+(ldh*2)];
	scalarprods[5] += hh[(ldh*2)+2] * hh[3+(ldh*3)];
	scalarprods[9] += hh[(ldh*3)+2] * hh[3+(ldh*4)];
	scalarprods[14] += hh[(ldh*4)+2] * hh[3+(ldh*5)];

	scalarprods[1] += hh[1] * hh[3+(ldh*2)];
	scalarprods[4] += hh[(ldh*1)+1] * hh[3+(ldh*3)];
	scalarprods[8] += hh[(ldh*2)+1] * hh[3+(ldh*4)];
	scalarprods[13] += hh[(ldh*3)+1] * hh[3+(ldh*5)];

	// loop counter = 4
	scalarprods[0] += hh[3] * hh[(4+ldh)];
	scalarprods[2] += hh[(ldh)+3] * hh[4+(ldh*2)];
	scalarprods[5] += hh[(ldh*2)+3] * hh[4+(ldh*3)];
	scalarprods[9] += hh[(ldh*3)+3] * hh[4+(ldh*4)];
	scalarprods[14] += hh[(ldh*4)+3] * hh[4+(ldh*5)];

	scalarprods[1] += hh[2] * hh[4+(ldh*2)];
	scalarprods[4] += hh[(ldh*1)+2] * hh[4+(ldh*3)];
	scalarprods[8] += hh[(ldh*2)+2] * hh[4+(ldh*4)];
	scalarprods[13] += hh[(ldh*3)+2] * hh[4+(ldh*5)];

	scalarprods[3] += hh[1] * hh[4+(ldh*3)];
	scalarprods[7] += hh[(ldh)+1] * hh[4+(ldh*4)];
	scalarprods[12] += hh[(ldh*2)+1] * hh[4+(ldh*5)];

	// loop counter = 5
	scalarprods[0] += hh[4] * hh[(5+ldh)];
	scalarprods[2] += hh[(ldh)+4] * hh[5+(ldh*2)];
	scalarprods[5] += hh[(ldh*2)+4] * hh[5+(ldh*3)];
	scalarprods[9] += hh[(ldh*3)+4] * hh[5+(ldh*4)];
	scalarprods[14] += hh[(ldh*4)+4] * hh[5+(ldh*5)];

	scalarprods[1] += hh[3] * hh[5+(ldh*2)];
	scalarprods[4] += hh[(ldh*1)+3] * hh[5+(ldh*3)];
	scalarprods[8] += hh[(ldh*2)+3] * hh[5+(ldh*4)];
	scalarprods[13] += hh[(ldh*3)+3] * hh[5+(ldh*5)];

	scalarprods[3] += hh[2] * hh[5+(ldh*3)];
	scalarprods[7] += hh[(ldh)+2] * hh[5+(ldh*4)];
	scalarprods[12] += hh[(ldh*2)+2] * hh[5+(ldh*5)];

	scalarprods[6] += hh[1] * hh[5+(ldh*4)];
	scalarprods[11] += hh[(ldh)+1] * hh[5+(ldh*5)];

	#pragma ivdep
	for (i = 6; i < nb; i++)
	{
		scalarprods[0] += hh[i-1] * hh[(i+ldh)];
		scalarprods[2] += hh[(ldh)+i-1] * hh[i+(ldh*2)];
		scalarprods[5] += hh[(ldh*2)+i-1] * hh[i+(ldh*3)];
		scalarprods[9] += hh[(ldh*3)+i-1] * hh[i+(ldh*4)];
		scalarprods[14] += hh[(ldh*4)+i-1] * hh[i+(ldh*5)];

		scalarprods[1] += hh[i-2] * hh[i+(ldh*2)];
		scalarprods[4] += hh[(ldh*1)+i-2] * hh[i+(ldh*3)];
		scalarprods[8] += hh[(ldh*2)+i-2] * hh[i+(ldh*4)];
		scalarprods[13] += hh[(ldh*3)+i-2] * hh[i+(ldh*5)];

		scalarprods[3] += hh[i-3] * hh[i+(ldh*3)];
		scalarprods[7] += hh[(ldh)+i-3] * hh[i+(ldh*4)];
		scalarprods[12] += hh[(ldh*2)+i-3] * hh[i+(ldh*5)];

		scalarprods[6] += hh[i-4] * hh[i+(ldh*4)];
		scalarprods[11] += hh[(ldh)+i-4] * hh[i+(ldh*5)];

		scalarprods[10] += hh[i-5] * hh[i+(ldh*5)];
	}

//	printf("s_1_2: %f\n", scalarprods[0]);
//	printf("s_1_3: %f\n", scalarprods[1]);
//	printf("s_2_3: %f\n", scalarprods[2]);
//	printf("s_1_4: %f\n", scalarprods[3]);
//	printf("s_2_4: %f\n", scalarprods[4]);
//	printf("s_3_4: %f\n", scalarprods[5]);
//	printf("s_1_5: %f\n", scalarprods[6]);
//	printf("s_2_5: %f\n", scalarprods[7]);
//	printf("s_3_5: %f\n", scalarprods[8]);
//	printf("s_4_5: %f\n", scalarprods[9]);
//	printf("s_1_6: %f\n", scalarprods[10]);
//	printf("s_2_6: %f\n", scalarprods[11]);
//	printf("s_3_6: %f\n", scalarprods[12]);
//	printf("s_4_6: %f\n", scalarprods[13]);
//	printf("s_5_6: %f\n", scalarprods[14]);

	// Production level kernel calls with padding
	for (i = 0; i < nq-2; i+=4)
	{
		hh_trafo_kernel_4_SSE_6hv_double(&q[i], hh, nb, ldq, ldh, scalarprods);
	}
	if (nq == i)
	{
		return;
	}
	else
	{
		hh_trafo_kernel_2_SSE_6hv_double(&q[i], hh, nb, ldq, ldh, scalarprods);
	}
}

#if 0
void hexa_hh_trafo_fast_(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;

	// calculating scalar products to compute
	// 6 householder vectors simultaneously
	double scalarprods[15];

//	scalarprods[0] = s_1_2;
//	scalarprods[1] = s_1_3;
//	scalarprods[2] = s_2_3;
//	scalarprods[3] = s_1_4;
//	scalarprods[4] = s_2_4;
//	scalarprods[5] = s_3_4;
//	scalarprods[6] = s_1_5;
//	scalarprods[7] = s_2_5;
//	scalarprods[8] = s_3_5;
//	scalarprods[9] = s_4_5;
//	scalarprods[10] = s_1_6;
//	scalarprods[11] = s_2_6;
//	scalarprods[12] = s_3_6;
//	scalarprods[13] = s_4_6;
//	scalarprods[14] = s_5_6;

	scalarprods[0] = hh[(ldh+1)];
	scalarprods[1] = hh[(ldh*2)+2];
	scalarprods[2] = hh[(ldh*2)+1];
	scalarprods[3] = hh[(ldh*3)+3];
	scalarprods[4] = hh[(ldh*3)+2];
	scalarprods[5] = hh[(ldh*3)+1];
	scalarprods[6] = hh[(ldh*4)+4];
	scalarprods[7] = hh[(ldh*4)+3];
	scalarprods[8] = hh[(ldh*4)+2];
	scalarprods[9] = hh[(ldh*4)+1];
	scalarprods[10] = hh[(ldh*5)+5];
	scalarprods[11] = hh[(ldh*5)+4];
	scalarprods[12] = hh[(ldh*5)+3];
	scalarprods[13] = hh[(ldh*5)+2];
	scalarprods[14] = hh[(ldh*5)+1];

	// calculate scalar product of first and fourth householder vector
	// loop counter = 2
	scalarprods[0] += hh[1] * hh[(2+ldh)];
	scalarprods[2] += hh[(ldh)+1] * hh[2+(ldh*2)];
	scalarprods[5] += hh[(ldh*2)+1] * hh[2+(ldh*3)];
	scalarprods[9] += hh[(ldh*3)+1] * hh[2+(ldh*4)];
	scalarprods[14] += hh[(ldh*4)+1] * hh[2+(ldh*5)];

	// loop counter = 3
	scalarprods[0] += hh[2] * hh[(3+ldh)];
	scalarprods[2] += hh[(ldh)+2] * hh[3+(ldh*2)];
	scalarprods[5] += hh[(ldh*2)+2] * hh[3+(ldh*3)];
	scalarprods[9] += hh[(ldh*3)+2] * hh[3+(ldh*4)];
	scalarprods[14] += hh[(ldh*4)+2] * hh[3+(ldh*5)];

	scalarprods[1] += hh[1] * hh[3+(ldh*2)];
	scalarprods[4] += hh[(ldh*1)+1] * hh[3+(ldh*3)];
	scalarprods[8] += hh[(ldh*2)+1] * hh[3+(ldh*4)];
	scalarprods[13] += hh[(ldh*3)+1] * hh[3+(ldh*5)];

	// loop counter = 4
	scalarprods[0] += hh[3] * hh[(4+ldh)];
	scalarprods[2] += hh[(ldh)+3] * hh[4+(ldh*2)];
	scalarprods[5] += hh[(ldh*2)+3] * hh[4+(ldh*3)];
	scalarprods[9] += hh[(ldh*3)+3] * hh[4+(ldh*4)];
	scalarprods[14] += hh[(ldh*4)+3] * hh[4+(ldh*5)];

	scalarprods[1] += hh[2] * hh[4+(ldh*2)];
	scalarprods[4] += hh[(ldh*1)+2] * hh[4+(ldh*3)];
	scalarprods[8] += hh[(ldh*2)+2] * hh[4+(ldh*4)];
	scalarprods[13] += hh[(ldh*3)+2] * hh[4+(ldh*5)];

	scalarprods[3] += hh[1] * hh[4+(ldh*3)];
	scalarprods[7] += hh[(ldh)+1] * hh[4+(ldh*4)];
	scalarprods[12] += hh[(ldh*2)+1] * hh[4+(ldh*5)];

	// loop counter = 5
	scalarprods[0] += hh[4] * hh[(5+ldh)];
	scalarprods[2] += hh[(ldh)+4] * hh[5+(ldh*2)];
	scalarprods[5] += hh[(ldh*2)+4] * hh[5+(ldh*3)];
	scalarprods[9] += hh[(ldh*3)+4] * hh[5+(ldh*4)];
	scalarprods[14] += hh[(ldh*4)+4] * hh[5+(ldh*5)];

	scalarprods[1] += hh[3] * hh[5+(ldh*2)];
	scalarprods[4] += hh[(ldh*1)+3] * hh[5+(ldh*3)];
	scalarprods[8] += hh[(ldh*2)+3] * hh[5+(ldh*4)];
	scalarprods[13] += hh[(ldh*3)+3] * hh[5+(ldh*5)];

	scalarprods[3] += hh[2] * hh[5+(ldh*3)];
	scalarprods[7] += hh[(ldh)+2] * hh[5+(ldh*4)];
	scalarprods[12] += hh[(ldh*2)+2] * hh[5+(ldh*5)];

	scalarprods[6] += hh[1] * hh[5+(ldh*4)];
	scalarprods[11] += hh[(ldh)+1] * hh[5+(ldh*5)];

	#pragma ivdep
	for (i = 6; i < nb; i++)
	{
		scalarprods[0] += hh[i-1] * hh[(i+ldh)];
		scalarprods[2] += hh[(ldh)+i-1] * hh[i+(ldh*2)];
		scalarprods[5] += hh[(ldh*2)+i-1] * hh[i+(ldh*3)];
		scalarprods[9] += hh[(ldh*3)+i-1] * hh[i+(ldh*4)];
		scalarprods[14] += hh[(ldh*4)+i-1] * hh[i+(ldh*5)];

		scalarprods[1] += hh[i-2] * hh[i+(ldh*2)];
		scalarprods[4] += hh[(ldh*1)+i-2] * hh[i+(ldh*3)];
		scalarprods[8] += hh[(ldh*2)+i-2] * hh[i+(ldh*4)];
		scalarprods[13] += hh[(ldh*3)+i-2] * hh[i+(ldh*5)];

		scalarprods[3] += hh[i-3] * hh[i+(ldh*3)];
		scalarprods[7] += hh[(ldh)+i-3] * hh[i+(ldh*4)];
		scalarprods[12] += hh[(ldh*2)+i-3] * hh[i+(ldh*5)];

		scalarprods[6] += hh[i-4] * hh[i+(ldh*4)];
		scalarprods[11] += hh[(ldh)+i-4] * hh[i+(ldh*5)];

		scalarprods[10] += hh[i-5] * hh[i+(ldh*5)];
	}

//	printf("s_1_2: %f\n", scalarprods[0]);
//	printf("s_1_3: %f\n", scalarprods[1]);
//	printf("s_2_3: %f\n", scalarprods[2]);
//	printf("s_1_4: %f\n", scalarprods[3]);
//	printf("s_2_4: %f\n", scalarprods[4]);
//	printf("s_3_4: %f\n", scalarprods[5]);
//	printf("s_1_5: %f\n", scalarprods[6]);
//	printf("s_2_5: %f\n", scalarprods[7]);
//	printf("s_3_5: %f\n", scalarprods[8]);
//	printf("s_4_5: %f\n", scalarprods[9]);
//	printf("s_1_6: %f\n", scalarprods[10]);
//	printf("s_2_6: %f\n", scalarprods[11]);
//	printf("s_3_6: %f\n", scalarprods[12]);
//	printf("s_4_6: %f\n", scalarprods[13]);
//	printf("s_5_6: %f\n", scalarprods[14]);

	// Production level kernel calls with padding
#ifdef __AVX__
	for (i = 0; i < nq; i+=8)
	{
		hh_trafo_kernel_8_AVX_6hv_double(&q[i], hh, nb, ldq, ldh, scalarprods);
	}
#else
	for (i = 0; i < nq; i+=4)
	{
		hh_trafo_kernel_4_SSE_6hv_double(&q[i], hh, nb, ldq, ldh, scalarprods);
	}
#endif
}
#endif

/**
 * Unrolled kernel that computes
 * 4 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_4_SSE_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m128d a1_1 = _mm_load_pd(&q[ldq*5]);
	__m128d a2_1 = _mm_load_pd(&q[ldq*4]);
	__m128d a3_1 = _mm_load_pd(&q[ldq*3]);
	__m128d a4_1 = _mm_load_pd(&q[ldq*2]);
	__m128d a5_1 = _mm_load_pd(&q[ldq]);
	__m128d a6_1 = _mm_load_pd(&q[0]);

	__m128d h_6_5 = _mm_loaddup_pd(&hh[(ldh*5)+1]);
	__m128d h_6_4 = _mm_loaddup_pd(&hh[(ldh*5)+2]);
	__m128d h_6_3 = _mm_loaddup_pd(&hh[(ldh*5)+3]);
	__m128d h_6_2 = _mm_loaddup_pd(&hh[(ldh*5)+4]);
	__m128d h_6_1 = _mm_loaddup_pd(&hh[(ldh*5)+5]);

	register __m128d t1 = _mm_add_pd(a6_1, _mm_mul_pd(a5_1, h_6_5));
	t1 = _mm_add_pd(t1, _mm_mul_pd(a4_1, h_6_4));
	t1 = _mm_add_pd(t1, _mm_mul_pd(a3_1, h_6_3));
	t1 = _mm_add_pd(t1, _mm_mul_pd(a2_1, h_6_2));
	t1 = _mm_add_pd(t1, _mm_mul_pd(a1_1, h_6_1));

	__m128d h_5_4 = _mm_loaddup_pd(&hh[(ldh*4)+1]);
	__m128d h_5_3 = _mm_loaddup_pd(&hh[(ldh*4)+2]);
	__m128d h_5_2 = _mm_loaddup_pd(&hh[(ldh*4)+3]);
	__m128d h_5_1 = _mm_loaddup_pd(&hh[(ldh*4)+4]);

	register __m128d v1 = _mm_add_pd(a5_1, _mm_mul_pd(a4_1, h_5_4));
	v1 = _mm_add_pd(v1, _mm_mul_pd(a3_1, h_5_3));
	v1 = _mm_add_pd(v1, _mm_mul_pd(a2_1, h_5_2));
	v1 = _mm_add_pd(v1, _mm_mul_pd(a1_1, h_5_1));

	__m128d h_4_3 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
	__m128d h_4_2 = _mm_loaddup_pd(&hh[(ldh*3)+2]);
	__m128d h_4_1 = _mm_loaddup_pd(&hh[(ldh*3)+3]);

	register __m128d w1 = _mm_add_pd(a4_1, _mm_mul_pd(a3_1, h_4_3));
	w1 = _mm_add_pd(w1, _mm_mul_pd(a2_1, h_4_2));
	w1 = _mm_add_pd(w1, _mm_mul_pd(a1_1, h_4_1));

	__m128d h_2_1 = _mm_loaddup_pd(&hh[ldh+1]);
	__m128d h_3_2 = _mm_loaddup_pd(&hh[(ldh*2)+1]);
	__m128d h_3_1 = _mm_loaddup_pd(&hh[(ldh*2)+2]);

	register __m128d z1 = _mm_add_pd(a3_1, _mm_mul_pd(a2_1, h_3_2));
	z1 = _mm_add_pd(z1, _mm_mul_pd(a1_1, h_3_1));
	register __m128d y1 = _mm_add_pd(a2_1, _mm_mul_pd(a1_1, h_2_1));

	register __m128d x1 = a1_1;

	__m128d a1_2 = _mm_load_pd(&q[(ldq*5)+2]);
	__m128d a2_2 = _mm_load_pd(&q[(ldq*4)+2]);
	__m128d a3_2 = _mm_load_pd(&q[(ldq*3)+2]);
	__m128d a4_2 = _mm_load_pd(&q[(ldq*2)+2]);
	__m128d a5_2 = _mm_load_pd(&q[(ldq)+2]);
	__m128d a6_2 = _mm_load_pd(&q[2]);

	register __m128d t2 = _mm_add_pd(a6_2, _mm_mul_pd(a5_2, h_6_5));
	t2 = _mm_add_pd(t2, _mm_mul_pd(a4_2, h_6_4));
	t2 = _mm_add_pd(t2, _mm_mul_pd(a3_2, h_6_3));
	t2 = _mm_add_pd(t2, _mm_mul_pd(a2_2, h_6_2));
	t2 = _mm_add_pd(t2, _mm_mul_pd(a1_2, h_6_1));
	register __m128d v2 = _mm_add_pd(a5_2, _mm_mul_pd(a4_2, h_5_4));
	v2 = _mm_add_pd(v2, _mm_mul_pd(a3_2, h_5_3));
	v2 = _mm_add_pd(v2, _mm_mul_pd(a2_2, h_5_2));
	v2 = _mm_add_pd(v2, _mm_mul_pd(a1_2, h_5_1));
	register __m128d w2 = _mm_add_pd(a4_2, _mm_mul_pd(a3_2, h_4_3));
	w2 = _mm_add_pd(w2, _mm_mul_pd(a2_2, h_4_2));
	w2 = _mm_add_pd(w2, _mm_mul_pd(a1_2, h_4_1));
	register __m128d z2 = _mm_add_pd(a3_2, _mm_mul_pd(a2_2, h_3_2));
	z2 = _mm_add_pd(z2, _mm_mul_pd(a1_2, h_3_1));
	register __m128d y2 = _mm_add_pd(a2_2, _mm_mul_pd(a1_2, h_2_1));

	register __m128d x2 = a1_2;

	__m128d q1;
	__m128d q2;

	__m128d h1;
	__m128d h2;
	__m128d h3;
	__m128d h4;
	__m128d h5;
	__m128d h6;

	for(i = 6; i < nb; i++)
	{
		h1 = _mm_loaddup_pd(&hh[i-5]);
		q1 = _mm_load_pd(&q[i*ldq]);
		q2 = _mm_load_pd(&q[(i*ldq)+2]);

		x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
		x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));

		h2 = _mm_loaddup_pd(&hh[ldh+i-4]);

		y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
		y2 = _mm_add_pd(y2, _mm_mul_pd(q2,h2));

		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-3]);

		z1 = _mm_add_pd(z1, _mm_mul_pd(q1,h3));
		z2 = _mm_add_pd(z2, _mm_mul_pd(q2,h3));

		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i-2]);

		w1 = _mm_add_pd(w1, _mm_mul_pd(q1,h4));
		w2 = _mm_add_pd(w2, _mm_mul_pd(q2,h4));

		h5 = _mm_loaddup_pd(&hh[(ldh*4)+i-1]);

		v1 = _mm_add_pd(v1, _mm_mul_pd(q1,h5));
		v2 = _mm_add_pd(v2, _mm_mul_pd(q2,h5));

		h6 = _mm_loaddup_pd(&hh[(ldh*5)+i]);

		t1 = _mm_add_pd(t1, _mm_mul_pd(q1,h6));
		t2 = _mm_add_pd(t2, _mm_mul_pd(q2,h6));
	}

	h1 = _mm_loaddup_pd(&hh[nb-5]);
	q1 = _mm_load_pd(&q[nb*ldq]);
	q2 = _mm_load_pd(&q[(nb*ldq)+2]);

	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
	x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));

	h2 = _mm_loaddup_pd(&hh[ldh+nb-4]);

	y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
	y2 = _mm_add_pd(y2, _mm_mul_pd(q2,h2));

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-3]);

	z1 = _mm_add_pd(z1, _mm_mul_pd(q1,h3));
	z2 = _mm_add_pd(z2, _mm_mul_pd(q2,h3));

	h4 = _mm_loaddup_pd(&hh[(ldh*3)+nb-2]);

	w1 = _mm_add_pd(w1, _mm_mul_pd(q1,h4));
	w2 = _mm_add_pd(w2, _mm_mul_pd(q2,h4));

	h5 = _mm_loaddup_pd(&hh[(ldh*4)+nb-1]);

	v1 = _mm_add_pd(v1, _mm_mul_pd(q1,h5));
	v2 = _mm_add_pd(v2, _mm_mul_pd(q2,h5));

	h1 = _mm_loaddup_pd(&hh[nb-4]);
	q1 = _mm_load_pd(&q[(nb+1)*ldq]);
	q2 = _mm_load_pd(&q[((nb+1)*ldq)+2]);

	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
	x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));

	h2 = _mm_loaddup_pd(&hh[ldh+nb-3]);

	y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
	y2 = _mm_add_pd(y2, _mm_mul_pd(q2,h2));

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-2]);

	z1 = _mm_add_pd(z1, _mm_mul_pd(q1,h3));
	z2 = _mm_add_pd(z2, _mm_mul_pd(q2,h3));

	h4 = _mm_loaddup_pd(&hh[(ldh*3)+nb-1]);

	w1 = _mm_add_pd(w1, _mm_mul_pd(q1,h4));
	w2 = _mm_add_pd(w2, _mm_mul_pd(q2,h4));

	h1 = _mm_loaddup_pd(&hh[nb-3]);
	q1 = _mm_load_pd(&q[(nb+2)*ldq]);
	q2 = _mm_load_pd(&q[((nb+2)*ldq)+2]);

	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
	x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));

	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);

	y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
	y2 = _mm_add_pd(y2, _mm_mul_pd(q2,h2));

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);

	z1 = _mm_add_pd(z1, _mm_mul_pd(q1,h3));
	z2 = _mm_add_pd(z2, _mm_mul_pd(q2,h3));

	h1 = _mm_loaddup_pd(&hh[nb-2]);
	q1 = _mm_load_pd(&q[(nb+3)*ldq]);
	q2 = _mm_load_pd(&q[((nb+3)*ldq)+2]);

	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
	x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));

	h2 = _mm_loaddup_pd(&hh[ldh+nb-1]);

	y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
	y2 = _mm_add_pd(y2, _mm_mul_pd(q2,h2));

	h1 = _mm_loaddup_pd(&hh[nb-1]);
	q1 = _mm_load_pd(&q[(nb+4)*ldq]);
	q2 = _mm_load_pd(&q[((nb+4)*ldq)+2]);

	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
	x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));

	/////////////////////////////////////////////////////
	// Apply tau, correct wrong calculation using pre-calculated scalar products
	/////////////////////////////////////////////////////

	__m128d tau1 = _mm_loaddup_pd(&hh[0]);
	x1 = _mm_mul_pd(x1, tau1);
	x2 = _mm_mul_pd(x2, tau1);

	__m128d tau2 = _mm_loaddup_pd(&hh[ldh]);
	__m128d vs_1_2 = _mm_loaddup_pd(&scalarprods[0]);
	h2 = _mm_mul_pd(tau2, vs_1_2);

	y1 = _mm_sub_pd(_mm_mul_pd(y1,tau2), _mm_mul_pd(x1,h2));
	y2 = _mm_sub_pd(_mm_mul_pd(y2,tau2), _mm_mul_pd(x2,h2));

	__m128d tau3 = _mm_loaddup_pd(&hh[ldh*2]);
	__m128d vs_1_3 = _mm_loaddup_pd(&scalarprods[1]);
	__m128d vs_2_3 = _mm_loaddup_pd(&scalarprods[2]);
	h2 = _mm_mul_pd(tau3, vs_1_3);
	h3 = _mm_mul_pd(tau3, vs_2_3);

	z1 = _mm_sub_pd(_mm_mul_pd(z1,tau3), _mm_add_pd(_mm_mul_pd(y1,h3), _mm_mul_pd(x1,h2)));
	z2 = _mm_sub_pd(_mm_mul_pd(z2,tau3), _mm_add_pd(_mm_mul_pd(y2,h3), _mm_mul_pd(x2,h2)));

	__m128d tau4 = _mm_loaddup_pd(&hh[ldh*3]);
	__m128d vs_1_4 = _mm_loaddup_pd(&scalarprods[3]);
	__m128d vs_2_4 = _mm_loaddup_pd(&scalarprods[4]);
	h2 = _mm_mul_pd(tau4, vs_1_4);
	h3 = _mm_mul_pd(tau4, vs_2_4);
	__m128d vs_3_4 = _mm_loaddup_pd(&scalarprods[5]);
	h4 = _mm_mul_pd(tau4, vs_3_4);

	w1 = _mm_sub_pd(_mm_mul_pd(w1,tau4), _mm_add_pd(_mm_mul_pd(z1,h4), _mm_add_pd(_mm_mul_pd(y1,h3), _mm_mul_pd(x1,h2))));
	w2 = _mm_sub_pd(_mm_mul_pd(w2,tau4), _mm_add_pd(_mm_mul_pd(z2,h4), _mm_add_pd(_mm_mul_pd(y2,h3), _mm_mul_pd(x2,h2))));

	__m128d tau5 = _mm_loaddup_pd(&hh[ldh*4]);
	__m128d vs_1_5 = _mm_loaddup_pd(&scalarprods[6]);
	__m128d vs_2_5 = _mm_loaddup_pd(&scalarprods[7]);
	h2 = _mm_mul_pd(tau5, vs_1_5);
	h3 = _mm_mul_pd(tau5, vs_2_5);
	__m128d vs_3_5 = _mm_loaddup_pd(&scalarprods[8]);
	__m128d vs_4_5 = _mm_loaddup_pd(&scalarprods[9]);
	h4 = _mm_mul_pd(tau5, vs_3_5);
	h5 = _mm_mul_pd(tau5, vs_4_5);

	v1 = _mm_sub_pd(_mm_mul_pd(v1,tau5), _mm_add_pd(_mm_add_pd(_mm_mul_pd(w1,h5), _mm_mul_pd(z1,h4)), _mm_add_pd(_mm_mul_pd(y1,h3), _mm_mul_pd(x1,h2))));
	v2 = _mm_sub_pd(_mm_mul_pd(v2,tau5), _mm_add_pd(_mm_add_pd(_mm_mul_pd(w2,h5), _mm_mul_pd(z2,h4)), _mm_add_pd(_mm_mul_pd(y2,h3), _mm_mul_pd(x2,h2))));

	__m128d tau6 = _mm_loaddup_pd(&hh[ldh*5]);
	__m128d vs_1_6 = _mm_loaddup_pd(&scalarprods[10]);
	__m128d vs_2_6 = _mm_loaddup_pd(&scalarprods[11]);
	h2 = _mm_mul_pd(tau6, vs_1_6);
	h3 = _mm_mul_pd(tau6, vs_2_6);
	__m128d vs_3_6 = _mm_loaddup_pd(&scalarprods[12]);
	__m128d vs_4_6 = _mm_loaddup_pd(&scalarprods[13]);
	__m128d vs_5_6 = _mm_loaddup_pd(&scalarprods[14]);
	h4 = _mm_mul_pd(tau6, vs_3_6);
	h5 = _mm_mul_pd(tau6, vs_4_6);
	h6 = _mm_mul_pd(tau6, vs_5_6);

	t1 = _mm_sub_pd(_mm_mul_pd(t1,tau6), _mm_add_pd( _mm_mul_pd(v1,h6), _mm_add_pd(_mm_add_pd(_mm_mul_pd(w1,h5), _mm_mul_pd(z1,h4)), _mm_add_pd(_mm_mul_pd(y1,h3), _mm_mul_pd(x1,h2)))));
	t2 = _mm_sub_pd(_mm_mul_pd(t2,tau6), _mm_add_pd( _mm_mul_pd(v2,h6), _mm_add_pd(_mm_add_pd(_mm_mul_pd(w2,h5), _mm_mul_pd(z2,h4)), _mm_add_pd(_mm_mul_pd(y2,h3), _mm_mul_pd(x2,h2)))));

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [4 x nb+3]
	/////////////////////////////////////////////////////

	q1 = _mm_load_pd(&q[0]);
	q2 = _mm_load_pd(&q[2]);
	q1 = _mm_sub_pd(q1, t1);
	q2 = _mm_sub_pd(q2, t2);
	_mm_store_pd(&q[0],q1);
	_mm_store_pd(&q[2],q2);

	h6 = _mm_loaddup_pd(&hh[(ldh*5)+1]);
	q1 = _mm_load_pd(&q[ldq]);
	q2 = _mm_load_pd(&q[(ldq+2)]);
	q1 = _mm_sub_pd(q1, v1);
	q2 = _mm_sub_pd(q2, v2);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(t2, h6));

	_mm_store_pd(&q[ldq],q1);
	_mm_store_pd(&q[(ldq+2)],q2);

	h5 = _mm_loaddup_pd(&hh[(ldh*4)+1]);
	q1 = _mm_load_pd(&q[ldq*2]);
	q2 = _mm_load_pd(&q[(ldq*2)+2]);
	q1 = _mm_sub_pd(q1, w1);
	q2 = _mm_sub_pd(q2, w2);
	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(v2, h5));

	h6 = _mm_loaddup_pd(&hh[(ldh*5)+2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(t2, h6));

	_mm_store_pd(&q[ldq*2],q1);
	_mm_store_pd(&q[(ldq*2)+2],q2);

	h4 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
	q1 = _mm_load_pd(&q[ldq*3]);
	q2 = _mm_load_pd(&q[(ldq*3)+2]);
	q1 = _mm_sub_pd(q1, z1);
	q2 = _mm_sub_pd(q2, z2);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(w2, h4));

	h5 = _mm_loaddup_pd(&hh[(ldh*4)+2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(v2, h5));

	h6 = _mm_loaddup_pd(&hh[(ldh*5)+3]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(t2, h6));

	_mm_store_pd(&q[ldq*3],q1);
	_mm_store_pd(&q[(ldq*3)+2],q2);

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+1]);
	q1 = _mm_load_pd(&q[ldq*4]);
	q2 = _mm_load_pd(&q[(ldq*4)+2]);
	q1 = _mm_sub_pd(q1, y1);
	q2 = _mm_sub_pd(q2, y2);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(z2, h3));

	h4 = _mm_loaddup_pd(&hh[(ldh*3)+2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(w2, h4));

	h5 = _mm_loaddup_pd(&hh[(ldh*4)+3]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(v2, h5));

	h6 = _mm_loaddup_pd(&hh[(ldh*5)+4]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(t2, h6));

	_mm_store_pd(&q[ldq*4],q1);
	_mm_store_pd(&q[(ldq*4)+2],q2);

	h2 = _mm_loaddup_pd(&hh[(ldh)+1]);
	q1 = _mm_load_pd(&q[ldq*5]);
	q2 = _mm_load_pd(&q[(ldq*5)+2]);
	q1 = _mm_sub_pd(q1, x1);
	q2 = _mm_sub_pd(q2, x2);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(y2, h2));

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(z2, h3));

	h4 = _mm_loaddup_pd(&hh[(ldh*3)+3]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(w2, h4));

	h5 = _mm_loaddup_pd(&hh[(ldh*4)+4]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(v2, h5));

	h6 = _mm_loaddup_pd(&hh[(ldh*5)+5]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(t2, h6));

	_mm_store_pd(&q[ldq*5],q1);
	_mm_store_pd(&q[(ldq*5)+2],q2);

	for (i = 6; i < nb; i++)
	{
		q1 = _mm_load_pd(&q[i*ldq]);
		q2 = _mm_load_pd(&q[(i*ldq)+2]);
		h1 = _mm_loaddup_pd(&hh[i-5]);

		q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
		q2 = _mm_sub_pd(q2, _mm_mul_pd(x2, h1));

		h2 = _mm_loaddup_pd(&hh[ldh+i-4]);

		q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
		q2 = _mm_sub_pd(q2, _mm_mul_pd(y2, h2));

		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-3]);

		q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
		q2 = _mm_sub_pd(q2, _mm_mul_pd(z2, h3));

		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i-2]);

		q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
		q2 = _mm_sub_pd(q2, _mm_mul_pd(w2, h4));

		h5 = _mm_loaddup_pd(&hh[(ldh*4)+i-1]);

		q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
		q2 = _mm_sub_pd(q2, _mm_mul_pd(v2, h5));

		h6 = _mm_loaddup_pd(&hh[(ldh*5)+i]);

		q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
		q2 = _mm_sub_pd(q2, _mm_mul_pd(t2, h6));

		_mm_store_pd(&q[i*ldq],q1);
		_mm_store_pd(&q[(i*ldq)+2],q2);
	}

	h1 = _mm_loaddup_pd(&hh[nb-5]);
	q1 = _mm_load_pd(&q[nb*ldq]);
	q2 = _mm_load_pd(&q[(nb*ldq)+2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(x2, h1));

	h2 = _mm_loaddup_pd(&hh[ldh+nb-4]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(y2, h2));

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-3]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(z2, h3));

	h4 = _mm_loaddup_pd(&hh[(ldh*3)+nb-2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(w2, h4));

	h5 = _mm_loaddup_pd(&hh[(ldh*4)+nb-1]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(v2, h5));

	_mm_store_pd(&q[nb*ldq],q1);
	_mm_store_pd(&q[(nb*ldq)+2],q2);

	h1 = _mm_loaddup_pd(&hh[nb-4]);
	q1 = _mm_load_pd(&q[(nb+1)*ldq]);
	q2 = _mm_load_pd(&q[((nb+1)*ldq)+2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(x2, h1));

	h2 = _mm_loaddup_pd(&hh[ldh+nb-3]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(y2, h2));

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(z2, h3));

	h4 = _mm_loaddup_pd(&hh[(ldh*3)+nb-1]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(w2, h4));

	_mm_store_pd(&q[(nb+1)*ldq],q1);
	_mm_store_pd(&q[((nb+1)*ldq)+2],q2);

	h1 = _mm_loaddup_pd(&hh[nb-3]);
	q1 = _mm_load_pd(&q[(nb+2)*ldq]);
	q2 = _mm_load_pd(&q[((nb+2)*ldq)+2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(x2, h1));

	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(y2, h2));

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(z2, h3));

	_mm_store_pd(&q[(nb+2)*ldq],q1);
	_mm_store_pd(&q[((nb+2)*ldq)+2],q2);

	h1 = _mm_loaddup_pd(&hh[nb-2]);
	q1 = _mm_load_pd(&q[(nb+3)*ldq]);
	q2 = _mm_load_pd(&q[((nb+3)*ldq)+2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(x2, h1));

	h2 = _mm_loaddup_pd(&hh[ldh+nb-1]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(y2, h2));

	_mm_store_pd(&q[(nb+3)*ldq],q1);
	_mm_store_pd(&q[((nb+3)*ldq)+2],q2);

	h1 = _mm_loaddup_pd(&hh[nb-1]);
	q1 = _mm_load_pd(&q[(nb+4)*ldq]);
	q2 = _mm_load_pd(&q[((nb+4)*ldq)+2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(x2, h1));

	_mm_store_pd(&q[(nb+4)*ldq],q1);
	_mm_store_pd(&q[((nb+4)*ldq)+2],q2);
}

/**
 * Unrolled kernel that computes
 * 2 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_2_SSE_6hv_double(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [2 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m128d a1_1 = _mm_load_pd(&q[ldq*5]);
	__m128d a2_1 = _mm_load_pd(&q[ldq*4]);
	__m128d a3_1 = _mm_load_pd(&q[ldq*3]);
	__m128d a4_1 = _mm_load_pd(&q[ldq*2]);
	__m128d a5_1 = _mm_load_pd(&q[ldq]);
	__m128d a6_1 = _mm_load_pd(&q[0]);

	__m128d h_6_5 = _mm_loaddup_pd(&hh[(ldh*5)+1]);
	__m128d h_6_4 = _mm_loaddup_pd(&hh[(ldh*5)+2]);
	__m128d h_6_3 = _mm_loaddup_pd(&hh[(ldh*5)+3]);
	__m128d h_6_2 = _mm_loaddup_pd(&hh[(ldh*5)+4]);
	__m128d h_6_1 = _mm_loaddup_pd(&hh[(ldh*5)+5]);

	register __m128d t1 = _mm_add_pd(a6_1, _mm_mul_pd(a5_1, h_6_5));
	t1 = _mm_add_pd(t1, _mm_mul_pd(a4_1, h_6_4));
	t1 = _mm_add_pd(t1, _mm_mul_pd(a3_1, h_6_3));
	t1 = _mm_add_pd(t1, _mm_mul_pd(a2_1, h_6_2));
	t1 = _mm_add_pd(t1, _mm_mul_pd(a1_1, h_6_1));

	__m128d h_5_4 = _mm_loaddup_pd(&hh[(ldh*4)+1]);
	__m128d h_5_3 = _mm_loaddup_pd(&hh[(ldh*4)+2]);
	__m128d h_5_2 = _mm_loaddup_pd(&hh[(ldh*4)+3]);
	__m128d h_5_1 = _mm_loaddup_pd(&hh[(ldh*4)+4]);

	register __m128d v1 = _mm_add_pd(a5_1, _mm_mul_pd(a4_1, h_5_4));
	v1 = _mm_add_pd(v1, _mm_mul_pd(a3_1, h_5_3));
	v1 = _mm_add_pd(v1, _mm_mul_pd(a2_1, h_5_2));
	v1 = _mm_add_pd(v1, _mm_mul_pd(a1_1, h_5_1));

	__m128d h_4_3 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
	__m128d h_4_2 = _mm_loaddup_pd(&hh[(ldh*3)+2]);
	__m128d h_4_1 = _mm_loaddup_pd(&hh[(ldh*3)+3]);

	register __m128d w1 = _mm_add_pd(a4_1, _mm_mul_pd(a3_1, h_4_3));
	w1 = _mm_add_pd(w1, _mm_mul_pd(a2_1, h_4_2));
	w1 = _mm_add_pd(w1, _mm_mul_pd(a1_1, h_4_1));

	__m128d h_2_1 = _mm_loaddup_pd(&hh[ldh+1]);
	__m128d h_3_2 = _mm_loaddup_pd(&hh[(ldh*2)+1]);
	__m128d h_3_1 = _mm_loaddup_pd(&hh[(ldh*2)+2]);

	register __m128d z1 = _mm_add_pd(a3_1, _mm_mul_pd(a2_1, h_3_2));
	z1 = _mm_add_pd(z1, _mm_mul_pd(a1_1, h_3_1));
	register __m128d y1 = _mm_add_pd(a2_1, _mm_mul_pd(a1_1, h_2_1));

	register __m128d x1 = a1_1;

	__m128d q1;

	__m128d h1;
	__m128d h2;
	__m128d h3;
	__m128d h4;
	__m128d h5;
	__m128d h6;

	for(i = 6; i < nb; i++)
	{
		h1 = _mm_loaddup_pd(&hh[i-5]);
		q1 = _mm_load_pd(&q[i*ldq]);

		x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));

		h2 = _mm_loaddup_pd(&hh[ldh+i-4]);

		y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));

		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-3]);

		z1 = _mm_add_pd(z1, _mm_mul_pd(q1,h3));

		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i-2]);

		w1 = _mm_add_pd(w1, _mm_mul_pd(q1,h4));

		h5 = _mm_loaddup_pd(&hh[(ldh*4)+i-1]);

		v1 = _mm_add_pd(v1, _mm_mul_pd(q1,h5));

		h6 = _mm_loaddup_pd(&hh[(ldh*5)+i]);

		t1 = _mm_add_pd(t1, _mm_mul_pd(q1,h6));

	}

	h1 = _mm_loaddup_pd(&hh[nb-5]);
	q1 = _mm_load_pd(&q[nb*ldq]);

	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));

	h2 = _mm_loaddup_pd(&hh[ldh+nb-4]);

	y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-3]);

	z1 = _mm_add_pd(z1, _mm_mul_pd(q1,h3));

	h4 = _mm_loaddup_pd(&hh[(ldh*3)+nb-2]);

	w1 = _mm_add_pd(w1, _mm_mul_pd(q1,h4));

	h5 = _mm_loaddup_pd(&hh[(ldh*4)+nb-1]);

	v1 = _mm_add_pd(v1, _mm_mul_pd(q1,h5));


	h1 = _mm_loaddup_pd(&hh[nb-4]);
	q1 = _mm_load_pd(&q[(nb+1)*ldq]);

	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));

	h2 = _mm_loaddup_pd(&hh[ldh+nb-3]);

	y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-2]);

	z1 = _mm_add_pd(z1, _mm_mul_pd(q1,h3));

	h4 = _mm_loaddup_pd(&hh[(ldh*3)+nb-1]);

	w1 = _mm_add_pd(w1, _mm_mul_pd(q1,h4));

	h1 = _mm_loaddup_pd(&hh[nb-3]);
	q1 = _mm_load_pd(&q[(nb+2)*ldq]);

	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));

	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);

	y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);

	z1 = _mm_add_pd(z1, _mm_mul_pd(q1,h3));

	h1 = _mm_loaddup_pd(&hh[nb-2]);
	q1 = _mm_load_pd(&q[(nb+3)*ldq]);

	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));

	h2 = _mm_loaddup_pd(&hh[ldh+nb-1]);

	y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));

	h1 = _mm_loaddup_pd(&hh[nb-1]);
	q1 = _mm_load_pd(&q[(nb+4)*ldq]);

	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));

	/////////////////////////////////////////////////////
	// Apply tau, correct wrong calculation using pre-calculated scalar products
	/////////////////////////////////////////////////////

	__m128d tau1 = _mm_loaddup_pd(&hh[0]);
	x1 = _mm_mul_pd(x1, tau1);

	__m128d tau2 = _mm_loaddup_pd(&hh[ldh]);
	__m128d vs_1_2 = _mm_loaddup_pd(&scalarprods[0]);
	h2 = _mm_mul_pd(tau2, vs_1_2);

	y1 = _mm_sub_pd(_mm_mul_pd(y1,tau2), _mm_mul_pd(x1,h2));

	__m128d tau3 = _mm_loaddup_pd(&hh[ldh*2]);
	__m128d vs_1_3 = _mm_loaddup_pd(&scalarprods[1]);
	__m128d vs_2_3 = _mm_loaddup_pd(&scalarprods[2]);
	h2 = _mm_mul_pd(tau3, vs_1_3);
	h3 = _mm_mul_pd(tau3, vs_2_3);

	z1 = _mm_sub_pd(_mm_mul_pd(z1,tau3), _mm_add_pd(_mm_mul_pd(y1,h3), _mm_mul_pd(x1,h2)));

	__m128d tau4 = _mm_loaddup_pd(&hh[ldh*3]);
	__m128d vs_1_4 = _mm_loaddup_pd(&scalarprods[3]);
	__m128d vs_2_4 = _mm_loaddup_pd(&scalarprods[4]);
	h2 = _mm_mul_pd(tau4, vs_1_4);
	h3 = _mm_mul_pd(tau4, vs_2_4);
	__m128d vs_3_4 = _mm_loaddup_pd(&scalarprods[5]);
	h4 = _mm_mul_pd(tau4, vs_3_4);

	w1 = _mm_sub_pd(_mm_mul_pd(w1,tau4), _mm_add_pd(_mm_mul_pd(z1,h4), _mm_add_pd(_mm_mul_pd(y1,h3), _mm_mul_pd(x1,h2))));

	__m128d tau5 = _mm_loaddup_pd(&hh[ldh*4]);
	__m128d vs_1_5 = _mm_loaddup_pd(&scalarprods[6]);
	__m128d vs_2_5 = _mm_loaddup_pd(&scalarprods[7]);
	h2 = _mm_mul_pd(tau5, vs_1_5);
	h3 = _mm_mul_pd(tau5, vs_2_5);
	__m128d vs_3_5 = _mm_loaddup_pd(&scalarprods[8]);
	__m128d vs_4_5 = _mm_loaddup_pd(&scalarprods[9]);
	h4 = _mm_mul_pd(tau5, vs_3_5);
	h5 = _mm_mul_pd(tau5, vs_4_5);

	v1 = _mm_sub_pd(_mm_mul_pd(v1,tau5), _mm_add_pd(_mm_add_pd(_mm_mul_pd(w1,h5), _mm_mul_pd(z1,h4)), _mm_add_pd(_mm_mul_pd(y1,h3), _mm_mul_pd(x1,h2))));

	__m128d tau6 = _mm_loaddup_pd(&hh[ldh*5]);
	__m128d vs_1_6 = _mm_loaddup_pd(&scalarprods[10]);
	__m128d vs_2_6 = _mm_loaddup_pd(&scalarprods[11]);
	h2 = _mm_mul_pd(tau6, vs_1_6);
	h3 = _mm_mul_pd(tau6, vs_2_6);
	__m128d vs_3_6 = _mm_loaddup_pd(&scalarprods[12]);
	__m128d vs_4_6 = _mm_loaddup_pd(&scalarprods[13]);
	__m128d vs_5_6 = _mm_loaddup_pd(&scalarprods[14]);
	h4 = _mm_mul_pd(tau6, vs_3_6);
	h5 = _mm_mul_pd(tau6, vs_4_6);
	h6 = _mm_mul_pd(tau6, vs_5_6);

	t1 = _mm_sub_pd(_mm_mul_pd(t1,tau6), _mm_add_pd( _mm_mul_pd(v1,h6), _mm_add_pd(_mm_add_pd(_mm_mul_pd(w1,h5), _mm_mul_pd(z1,h4)), _mm_add_pd(_mm_mul_pd(y1,h3), _mm_mul_pd(x1,h2)))));

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [2 x nb+3]
	/////////////////////////////////////////////////////

	q1 = _mm_load_pd(&q[0]);
	q1 = _mm_sub_pd(q1, t1);
	_mm_store_pd(&q[0],q1);

	h6 = _mm_loaddup_pd(&hh[(ldh*5)+1]);
	q1 = _mm_load_pd(&q[ldq]);
	q1 = _mm_sub_pd(q1, v1);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));

	_mm_store_pd(&q[ldq],q1);

	h5 = _mm_loaddup_pd(&hh[(ldh*4)+1]);
	q1 = _mm_load_pd(&q[ldq*2]);
	q1 = _mm_sub_pd(q1, w1);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));

	h6 = _mm_loaddup_pd(&hh[(ldh*5)+2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));

	_mm_store_pd(&q[ldq*2],q1);

	h4 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
	q1 = _mm_load_pd(&q[ldq*3]);
	q1 = _mm_sub_pd(q1, z1);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));

	h5 = _mm_loaddup_pd(&hh[(ldh*4)+2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));

	h6 = _mm_loaddup_pd(&hh[(ldh*5)+3]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));

	_mm_store_pd(&q[ldq*3],q1);

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+1]);
	q1 = _mm_load_pd(&q[ldq*4]);
	q1 = _mm_sub_pd(q1, y1);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));

	h4 = _mm_loaddup_pd(&hh[(ldh*3)+2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));

	h5 = _mm_loaddup_pd(&hh[(ldh*4)+3]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));

	h6 = _mm_loaddup_pd(&hh[(ldh*5)+4]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));

	_mm_store_pd(&q[ldq*4],q1);

	h2 = _mm_loaddup_pd(&hh[(ldh)+1]);
	q1 = _mm_load_pd(&q[ldq*5]);
	q1 = _mm_sub_pd(q1, x1);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));

	h4 = _mm_loaddup_pd(&hh[(ldh*3)+3]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));

	h5 = _mm_loaddup_pd(&hh[(ldh*4)+4]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));

	h6 = _mm_loaddup_pd(&hh[(ldh*5)+5]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));

	_mm_store_pd(&q[ldq*5],q1);

	for (i = 6; i < nb; i++)
	{
		q1 = _mm_load_pd(&q[i*ldq]);
		h1 = _mm_loaddup_pd(&hh[i-5]);

		q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));

		h2 = _mm_loaddup_pd(&hh[ldh+i-4]);

		q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));

		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-3]);

		q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));

		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i-2]);

		q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));

		h5 = _mm_loaddup_pd(&hh[(ldh*4)+i-1]);

		q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));

		h6 = _mm_loaddup_pd(&hh[(ldh*5)+i]);

		q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));

		_mm_store_pd(&q[i*ldq],q1);
	}

	h1 = _mm_loaddup_pd(&hh[nb-5]);
	q1 = _mm_load_pd(&q[nb*ldq]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));

	h2 = _mm_loaddup_pd(&hh[ldh+nb-4]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-3]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));

	h4 = _mm_loaddup_pd(&hh[(ldh*3)+nb-2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));

	h5 = _mm_loaddup_pd(&hh[(ldh*4)+nb-1]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));

	_mm_store_pd(&q[nb*ldq],q1);

	h1 = _mm_loaddup_pd(&hh[nb-4]);
	q1 = _mm_load_pd(&q[(nb+1)*ldq]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));

	h2 = _mm_loaddup_pd(&hh[ldh+nb-3]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));

	h4 = _mm_loaddup_pd(&hh[(ldh*3)+nb-1]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));

	_mm_store_pd(&q[(nb+1)*ldq],q1);

	h1 = _mm_loaddup_pd(&hh[nb-3]);
	q1 = _mm_load_pd(&q[(nb+2)*ldq]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));

	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));

	_mm_store_pd(&q[(nb+2)*ldq],q1);

	h1 = _mm_loaddup_pd(&hh[nb-2]);
	q1 = _mm_load_pd(&q[(nb+3)*ldq]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));

	h2 = _mm_loaddup_pd(&hh[ldh+nb-1]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));

	_mm_store_pd(&q[(nb+3)*ldq],q1);

	h1 = _mm_loaddup_pd(&hh[nb-1]);
	q1 = _mm_load_pd(&q[(nb+4)*ldq]);

	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));

	_mm_store_pd(&q[(nb+4)*ldq],q1);
}
