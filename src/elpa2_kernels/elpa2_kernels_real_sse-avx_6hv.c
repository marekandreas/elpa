//    This file is part of ELPA.
//
//    The ELPA library was originally created by the ELPA consortium,
//    consisting of the following organizations:
//
//    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
//    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
//      Informatik,
//    - Technische Universität München, Lehrstuhl für Informatik mit
//      Schwerpunkt Wissenschaftliches Rechnen ,
//    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
//    - Max-Plack-Institut für Mathematik in den Naturwissenschaftrn,
//      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
//      and
//    - IBM Deutschland GmbH
//
//    This particular source code file contains additions, changes and
//    enhancements authored by Intel Corporation which is not part of
//    the ELPA consortium.
//
//    More information can be found here:
//    http://elpa.rzg.mpg.de/
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
// Adapted for building a shared-library by Andreas Marek, RZG (andreas.marek@rzg.mpg.de)
// --------------------------------------------------------------------------------------------------

#include <x86intrin.h>

#define __forceinline __attribute__((always_inline)) static

#ifdef __USE_AVX128__
#undef __AVX__
#endif

#ifdef __FMA4__
#define __ELPA_USE_FMA__
#define _mm256_FMA_pd(a,b,c) _mm256_macc_pd(a,b,c)
#define _mm256_NFMA_pd(a,b,c) _mm256_nmacc_pd(a,b,c)
#define _mm256_FMSUB_pd(a,b,c) _mm256_msub(a,b,c)
#endif

#ifdef __AVX2__
#define __ELPA_USE_FMA__
#define _mm256_FMA_pd(a,b,c) _mm256_fmadd_pd(a,b,c)
#define _mm256_NFMA_pd(a,b,c) _mm256_fnmadd_pd(a,b,c)
#define _mm256_FMSUB_pd(a,b,c) _mm256_fmsub_pd(a,b,c)
#endif

//Forward declaration
#ifdef __AVX__
static void hh_trafo_kernel_4_AVX_6hv(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);
static void hh_trafo_kernel_8_AVX_6hv(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);
#else
static void hh_trafo_kernel_2_SSE_6hv(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);
static void hh_trafo_kernel_4_SSE_6hv(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods);
#endif

void hexa_hh_trafo_real_sse_avx_6hv_(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#if 0
void hexa_hh_trafo_fast_(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif

void hexa_hh_trafo_real_sse_avx_6hv_(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
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
	for (i = 0; i < nq-4; i+=8)
	{
		hh_trafo_kernel_8_AVX_6hv(&q[i], hh, nb, ldq, ldh, scalarprods);
	}
	if (nq == i)
	{
		return;
	}
	else
	{
		hh_trafo_kernel_4_AVX_6hv(&q[i], hh, nb, ldq, ldh, scalarprods);
	}
#else
	for (i = 0; i < nq-2; i+=4)
	{
		hh_trafo_kernel_4_SSE_6hv(&q[i], hh, nb, ldq, ldh, scalarprods);
	}
	if (nq == i)
	{
		return;
	}
	else
	{
		hh_trafo_kernel_2_SSE_6hv(&q[i], hh, nb, ldq, ldh, scalarprods);
	}
#endif
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
		hh_trafo_kernel_8_AVX_6hv(&q[i], hh, nb, ldq, ldh, scalarprods);
	}
#else
	for (i = 0; i < nq; i+=4)
	{
		hh_trafo_kernel_4_SSE_6hv(&q[i], hh, nb, ldq, ldh, scalarprods);
	}
#endif
}
#endif

#ifdef __AVX__
/**
 * Unrolled kernel that computes
 * 8 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_8_AVX_6hv(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [8 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m256d a1_1 = _mm256_load_pd(&q[ldq*5]);
	__m256d a2_1 = _mm256_load_pd(&q[ldq*4]);
	__m256d a3_1 = _mm256_load_pd(&q[ldq*3]);
	__m256d a4_1 = _mm256_load_pd(&q[ldq*2]);
	__m256d a5_1 = _mm256_load_pd(&q[ldq]);
	__m256d a6_1 = _mm256_load_pd(&q[0]);

	__m256d h_6_5 = _mm256_broadcast_sd(&hh[(ldh*5)+1]);
	__m256d h_6_4 = _mm256_broadcast_sd(&hh[(ldh*5)+2]);
	__m256d h_6_3 = _mm256_broadcast_sd(&hh[(ldh*5)+3]);
	__m256d h_6_2 = _mm256_broadcast_sd(&hh[(ldh*5)+4]);
	__m256d h_6_1 = _mm256_broadcast_sd(&hh[(ldh*5)+5]);
#ifdef __ELPA_USE_FMA__
	register __m256d t1 = _mm256_FMA_pd(a5_1, h_6_5, a6_1);
	t1 = _mm256_FMA_pd(a4_1, h_6_4, t1);
	t1 = _mm256_FMA_pd(a3_1, h_6_3, t1);
	t1 = _mm256_FMA_pd(a2_1, h_6_2, t1);
	t1 = _mm256_FMA_pd(a1_1, h_6_1, t1);
#else
	register __m256d t1 = _mm256_add_pd(a6_1, _mm256_mul_pd(a5_1, h_6_5));
	t1 = _mm256_add_pd(t1, _mm256_mul_pd(a4_1, h_6_4));
	t1 = _mm256_add_pd(t1, _mm256_mul_pd(a3_1, h_6_3));
	t1 = _mm256_add_pd(t1, _mm256_mul_pd(a2_1, h_6_2));
	t1 = _mm256_add_pd(t1, _mm256_mul_pd(a1_1, h_6_1));
#endif
	__m256d h_5_4 = _mm256_broadcast_sd(&hh[(ldh*4)+1]);
	__m256d h_5_3 = _mm256_broadcast_sd(&hh[(ldh*4)+2]);
	__m256d h_5_2 = _mm256_broadcast_sd(&hh[(ldh*4)+3]);
	__m256d h_5_1 = _mm256_broadcast_sd(&hh[(ldh*4)+4]);
#ifdef __ELPA_USE_FMA__
	register __m256d v1 = _mm256_FMA_pd(a4_1, h_5_4, a5_1);
	v1 = _mm256_FMA_pd(a3_1, h_5_3, v1);
	v1 = _mm256_FMA_pd(a2_1, h_5_2, v1);
	v1 = _mm256_FMA_pd(a1_1, h_5_1, v1);
#else
	register __m256d v1 = _mm256_add_pd(a5_1, _mm256_mul_pd(a4_1, h_5_4));
	v1 = _mm256_add_pd(v1, _mm256_mul_pd(a3_1, h_5_3));
	v1 = _mm256_add_pd(v1, _mm256_mul_pd(a2_1, h_5_2));
	v1 = _mm256_add_pd(v1, _mm256_mul_pd(a1_1, h_5_1));
#endif
	__m256d h_4_3 = _mm256_broadcast_sd(&hh[(ldh*3)+1]);
	__m256d h_4_2 = _mm256_broadcast_sd(&hh[(ldh*3)+2]);
	__m256d h_4_1 = _mm256_broadcast_sd(&hh[(ldh*3)+3]);
#ifdef __ELPA_USE_FMA__
	register __m256d w1 = _mm256_FMA_pd(a3_1, h_4_3, a4_1);
	w1 = _mm256_FMA_pd(a2_1, h_4_2, w1);
	w1 = _mm256_FMA_pd(a1_1, h_4_1, w1);
#else
	register __m256d w1 = _mm256_add_pd(a4_1, _mm256_mul_pd(a3_1, h_4_3));
	w1 = _mm256_add_pd(w1, _mm256_mul_pd(a2_1, h_4_2));
	w1 = _mm256_add_pd(w1, _mm256_mul_pd(a1_1, h_4_1));
#endif
	__m256d h_2_1 = _mm256_broadcast_sd(&hh[ldh+1]);
	__m256d h_3_2 = _mm256_broadcast_sd(&hh[(ldh*2)+1]);
	__m256d h_3_1 = _mm256_broadcast_sd(&hh[(ldh*2)+2]);
#ifdef __ELPA_USE_FMA__
	register __m256d z1 = _mm256_FMA_pd(a2_1, h_3_2, a3_1);
	z1 = _mm256_FMA_pd(a1_1, h_3_1, z1);
	register __m256d y1 = _mm256_FMA_pd(a1_1, h_2_1, a2_1);
#else
	register __m256d z1 = _mm256_add_pd(a3_1, _mm256_mul_pd(a2_1, h_3_2));
	z1 = _mm256_add_pd(z1, _mm256_mul_pd(a1_1, h_3_1));
	register __m256d y1 = _mm256_add_pd(a2_1, _mm256_mul_pd(a1_1, h_2_1));
#endif
	register __m256d x1 = a1_1;


	__m256d a1_2 = _mm256_load_pd(&q[(ldq*5)+4]);
	__m256d a2_2 = _mm256_load_pd(&q[(ldq*4)+4]);
	__m256d a3_2 = _mm256_load_pd(&q[(ldq*3)+4]);
	__m256d a4_2 = _mm256_load_pd(&q[(ldq*2)+4]);
	__m256d a5_2 = _mm256_load_pd(&q[(ldq)+4]);
	__m256d a6_2 = _mm256_load_pd(&q[4]);

#ifdef __ELPA_USE_FMA__
	register __m256d t2 = _mm256_FMA_pd(a5_2, h_6_5, a6_2);
	t2 = _mm256_FMA_pd(a4_2, h_6_4, t2);
	t2 = _mm256_FMA_pd(a3_2, h_6_3, t2);
	t2 = _mm256_FMA_pd(a2_2, h_6_2, t2);
	t2 = _mm256_FMA_pd(a1_2, h_6_1, t2);
	register __m256d v2 = _mm256_FMA_pd(a4_2, h_5_4, a5_2);
	v2 = _mm256_FMA_pd(a3_2, h_5_3, v2);
	v2 = _mm256_FMA_pd(a2_2, h_5_2, v2);
	v2 = _mm256_FMA_pd(a1_2, h_5_1, v2);
	register __m256d w2 = _mm256_FMA_pd(a3_2, h_4_3, a4_2);
	w2 = _mm256_FMA_pd(a2_2, h_4_2, w2);
	w2 = _mm256_FMA_pd(a1_2, h_4_1, w2);
	register __m256d z2 = _mm256_FMA_pd(a2_2, h_3_2, a3_2);
	z2 = _mm256_FMA_pd(a1_2, h_3_1, z2);
	register __m256d y2 = _mm256_FMA_pd(a1_2, h_2_1, a2_2);
#else
	register __m256d t2 = _mm256_add_pd(a6_2, _mm256_mul_pd(a5_2, h_6_5));
	t2 = _mm256_add_pd(t2, _mm256_mul_pd(a4_2, h_6_4));
	t2 = _mm256_add_pd(t2, _mm256_mul_pd(a3_2, h_6_3));
	t2 = _mm256_add_pd(t2, _mm256_mul_pd(a2_2, h_6_2));
	t2 = _mm256_add_pd(t2, _mm256_mul_pd(a1_2, h_6_1));
	register __m256d v2 = _mm256_add_pd(a5_2, _mm256_mul_pd(a4_2, h_5_4));
	v2 = _mm256_add_pd(v2, _mm256_mul_pd(a3_2, h_5_3));
	v2 = _mm256_add_pd(v2, _mm256_mul_pd(a2_2, h_5_2));
	v2 = _mm256_add_pd(v2, _mm256_mul_pd(a1_2, h_5_1));
	register __m256d w2 = _mm256_add_pd(a4_2, _mm256_mul_pd(a3_2, h_4_3));
	w2 = _mm256_add_pd(w2, _mm256_mul_pd(a2_2, h_4_2));
	w2 = _mm256_add_pd(w2, _mm256_mul_pd(a1_2, h_4_1));
	register __m256d z2 = _mm256_add_pd(a3_2, _mm256_mul_pd(a2_2, h_3_2));
	z2 = _mm256_add_pd(z2, _mm256_mul_pd(a1_2, h_3_1));
	register __m256d y2 = _mm256_add_pd(a2_2, _mm256_mul_pd(a1_2, h_2_1));
#endif
	register __m256d x2 = a1_2;

	__m256d q1;
	__m256d q2;

	__m256d h1;
	__m256d h2;
	__m256d h3;
	__m256d h4;
	__m256d h5;
	__m256d h6;

	for(i = 6; i < nb; i++)
	{
		h1 = _mm256_broadcast_sd(&hh[i-5]);
		q1 = _mm256_load_pd(&q[i*ldq]);
		q2 = _mm256_load_pd(&q[(i*ldq)+4]);
#ifdef __ELPA_USE_FMA__
		x1 = _mm256_FMA_pd(q1, h1, x1);
		x2 = _mm256_FMA_pd(q2, h1, x2);
#else
		x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
		x2 = _mm256_add_pd(x2, _mm256_mul_pd(q2,h1));
#endif
		h2 = _mm256_broadcast_sd(&hh[ldh+i-4]);
#ifdef __ELPA_USE_FMA__
		y1 = _mm256_FMA_pd(q1, h2, y1);
		y2 = _mm256_FMA_pd(q2, h2, y2);
#else
		y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
		y2 = _mm256_add_pd(y2, _mm256_mul_pd(q2,h2));
#endif
		h3 = _mm256_broadcast_sd(&hh[(ldh*2)+i-3]);
#ifdef __ELPA_USE_FMA__
		z1 = _mm256_FMA_pd(q1, h3, z1);
		z2 = _mm256_FMA_pd(q2, h3, z2);
#else
		z1 = _mm256_add_pd(z1, _mm256_mul_pd(q1,h3));
		z2 = _mm256_add_pd(z2, _mm256_mul_pd(q2,h3));
#endif
		h4 = _mm256_broadcast_sd(&hh[(ldh*3)+i-2]);
#ifdef __ELPA_USE_FMA__
		w1 = _mm256_FMA_pd(q1, h4, w1);
		w2 = _mm256_FMA_pd(q2, h4, w2);
#else
		w1 = _mm256_add_pd(w1, _mm256_mul_pd(q1,h4));
		w2 = _mm256_add_pd(w2, _mm256_mul_pd(q2,h4));
#endif
		h5 = _mm256_broadcast_sd(&hh[(ldh*4)+i-1]);
#ifdef __ELPA_USE_FMA__
		v1 = _mm256_FMA_pd(q1, h5, v1);
		v2 = _mm256_FMA_pd(q2, h5, v2);
#else
		v1 = _mm256_add_pd(v1, _mm256_mul_pd(q1,h5));
		v2 = _mm256_add_pd(v2, _mm256_mul_pd(q2,h5));
#endif
		h6 = _mm256_broadcast_sd(&hh[(ldh*5)+i]);
#ifdef __ELPA_USE_FMA__
		t1 = _mm256_FMA_pd(q1, h6, t1);
		t2 = _mm256_FMA_pd(q2, h6, t2);
#else
		t1 = _mm256_add_pd(t1, _mm256_mul_pd(q1,h6));
		t2 = _mm256_add_pd(t2, _mm256_mul_pd(q2,h6));
#endif
	}

	h1 = _mm256_broadcast_sd(&hh[nb-5]);
	q1 = _mm256_load_pd(&q[nb*ldq]);
	q2 = _mm256_load_pd(&q[(nb*ldq)+4]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm256_FMA_pd(q1, h1, x1);
	x2 = _mm256_FMA_pd(q2, h1, x2);
#else
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
	x2 = _mm256_add_pd(x2, _mm256_mul_pd(q2,h1));
#endif
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-4]);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMA_pd(q1, h2, y1);
	y2 = _mm256_FMA_pd(q2, h2, y2);
#else
	y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
	y2 = _mm256_add_pd(y2, _mm256_mul_pd(q2,h2));
#endif
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+nb-3]);
#ifdef __ELPA_USE_FMA__
	z1 = _mm256_FMA_pd(q1, h3, z1);
	z2 = _mm256_FMA_pd(q2, h3, z2);
#else
	z1 = _mm256_add_pd(z1, _mm256_mul_pd(q1,h3));
	z2 = _mm256_add_pd(z2, _mm256_mul_pd(q2,h3));
#endif
	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+nb-2]);
#ifdef __ELPA_USE_FMA__
	w1 = _mm256_FMA_pd(q1, h4, w1);
	w2 = _mm256_FMA_pd(q2, h4, w2);
#else
	w1 = _mm256_add_pd(w1, _mm256_mul_pd(q1,h4));
	w2 = _mm256_add_pd(w2, _mm256_mul_pd(q2,h4));
#endif
	h5 = _mm256_broadcast_sd(&hh[(ldh*4)+nb-1]);
#ifdef __ELPA_USE_FMA__
	v1 = _mm256_FMA_pd(q1, h5, v1);
	v2 = _mm256_FMA_pd(q2, h5, v2);
#else
	v1 = _mm256_add_pd(v1, _mm256_mul_pd(q1,h5));
	v2 = _mm256_add_pd(v2, _mm256_mul_pd(q2,h5));
#endif

	h1 = _mm256_broadcast_sd(&hh[nb-4]);
	q1 = _mm256_load_pd(&q[(nb+1)*ldq]);
	q2 = _mm256_load_pd(&q[((nb+1)*ldq)+4]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm256_FMA_pd(q1, h1, x1);
	x2 = _mm256_FMA_pd(q2, h1, x2);
#else
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
	x2 = _mm256_add_pd(x2, _mm256_mul_pd(q2,h1));
#endif
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-3]);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMA_pd(q1, h2, y1);
	y2 = _mm256_FMA_pd(q2, h2, y2);
#else
	y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
	y2 = _mm256_add_pd(y2, _mm256_mul_pd(q2,h2));
#endif
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+nb-2]);
#ifdef __ELPA_USE_FMA__
	z1 = _mm256_FMA_pd(q1, h3, z1);
	z2 = _mm256_FMA_pd(q2, h3, z2);
#else
	z1 = _mm256_add_pd(z1, _mm256_mul_pd(q1,h3));
	z2 = _mm256_add_pd(z2, _mm256_mul_pd(q2,h3));
#endif
	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+nb-1]);
#ifdef __ELPA_USE_FMA__
	w1 = _mm256_FMA_pd(q1, h4, w1);
	w2 = _mm256_FMA_pd(q2, h4, w2);
#else
	w1 = _mm256_add_pd(w1, _mm256_mul_pd(q1,h4));
	w2 = _mm256_add_pd(w2, _mm256_mul_pd(q2,h4));
#endif

	h1 = _mm256_broadcast_sd(&hh[nb-3]);
	q1 = _mm256_load_pd(&q[(nb+2)*ldq]);
	q2 = _mm256_load_pd(&q[((nb+2)*ldq)+4]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm256_FMA_pd(q1, h1, x1);
	x2 = _mm256_FMA_pd(q2, h1, x2);
#else
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
	x2 = _mm256_add_pd(x2, _mm256_mul_pd(q2,h1));
#endif
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-2]);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMA_pd(q1, h2, y1);
	y2 = _mm256_FMA_pd(q2, h2, y2);
#else
	y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
	y2 = _mm256_add_pd(y2, _mm256_mul_pd(q2,h2));
#endif
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+nb-1]);
#ifdef __ELPA_USE_FMA__
	z1 = _mm256_FMA_pd(q1, h3, z1);
	z2 = _mm256_FMA_pd(q2, h3, z2);
#else
	z1 = _mm256_add_pd(z1, _mm256_mul_pd(q1,h3));
	z2 = _mm256_add_pd(z2, _mm256_mul_pd(q2,h3));
#endif

	h1 = _mm256_broadcast_sd(&hh[nb-2]);
	q1 = _mm256_load_pd(&q[(nb+3)*ldq]);
	q2 = _mm256_load_pd(&q[((nb+3)*ldq)+4]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm256_FMA_pd(q1, h1, x1);
	x2 = _mm256_FMA_pd(q2, h1, x2);
#else
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
	x2 = _mm256_add_pd(x2, _mm256_mul_pd(q2,h1));
#endif
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-1]);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMA_pd(q1, h2, y1);
	y2 = _mm256_FMA_pd(q2, h2, y2);
#else
	y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
	y2 = _mm256_add_pd(y2, _mm256_mul_pd(q2,h2));
#endif

	h1 = _mm256_broadcast_sd(&hh[nb-1]);
	q1 = _mm256_load_pd(&q[(nb+4)*ldq]);
	q2 = _mm256_load_pd(&q[((nb+4)*ldq)+4]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm256_FMA_pd(q1, h1, x1);
	x2 = _mm256_FMA_pd(q2, h1, x2);
#else
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
	x2 = _mm256_add_pd(x2, _mm256_mul_pd(q2,h1));
#endif

	/////////////////////////////////////////////////////
	// Apply tau, correct wrong calculation using pre-calculated scalar products
	/////////////////////////////////////////////////////

	__m256d tau1 = _mm256_broadcast_sd(&hh[0]);
	x1 = _mm256_mul_pd(x1, tau1);
	x2 = _mm256_mul_pd(x2, tau1);

	__m256d tau2 = _mm256_broadcast_sd(&hh[ldh]);
	__m256d vs_1_2 = _mm256_broadcast_sd(&scalarprods[0]);
	h2 = _mm256_mul_pd(tau2, vs_1_2);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMSUB_pd(y1, tau2, _mm256_mul_pd(x1,h2));
	y2 = _mm256_FMSUB_pd(y2, tau2, _mm256_mul_pd(x2,h2));
#else
	y1 = _mm256_sub_pd(_mm256_mul_pd(y1,tau2), _mm256_mul_pd(x1,h2));
	y2 = _mm256_sub_pd(_mm256_mul_pd(y2,tau2), _mm256_mul_pd(x2,h2));
#endif

	__m256d tau3 = _mm256_broadcast_sd(&hh[ldh*2]);
	__m256d vs_1_3 = _mm256_broadcast_sd(&scalarprods[1]);
	__m256d vs_2_3 = _mm256_broadcast_sd(&scalarprods[2]);
	h2 = _mm256_mul_pd(tau3, vs_1_3);
	h3 = _mm256_mul_pd(tau3, vs_2_3);
#ifdef __ELPA_USE_FMA__
	z1 = _mm256_FMSUB_pd(z1, tau3, _mm256_FMA_pd(y1, h3, _mm256_mul_pd(x1,h2)));
	z2 = _mm256_FMSUB_pd(z2, tau3, _mm256_FMA_pd(y2, h3, _mm256_mul_pd(x2,h2)));
#else
	z1 = _mm256_sub_pd(_mm256_mul_pd(z1,tau3), _mm256_add_pd(_mm256_mul_pd(y1,h3), _mm256_mul_pd(x1,h2)));
	z2 = _mm256_sub_pd(_mm256_mul_pd(z2,tau3), _mm256_add_pd(_mm256_mul_pd(y2,h3), _mm256_mul_pd(x2,h2)));
#endif

	__m256d tau4 = _mm256_broadcast_sd(&hh[ldh*3]);
	__m256d vs_1_4 = _mm256_broadcast_sd(&scalarprods[3]);
	__m256d vs_2_4 = _mm256_broadcast_sd(&scalarprods[4]);
	h2 = _mm256_mul_pd(tau4, vs_1_4);
	h3 = _mm256_mul_pd(tau4, vs_2_4);
	__m256d vs_3_4 = _mm256_broadcast_sd(&scalarprods[5]);
	h4 = _mm256_mul_pd(tau4, vs_3_4);
#ifdef __ELPA_USE_FMA__
	w1 = _mm256_FMSUB_pd(w1, tau4, _mm256_FMA_pd(z1, h4, _mm256_FMA_pd(y1, h3, _mm256_mul_pd(x1,h2))));
	w2 = _mm256_FMSUB_pd(w2, tau4, _mm256_FMA_pd(z2, h4, _mm256_FMA_pd(y2, h3, _mm256_mul_pd(x2,h2))));
#else
	w1 = _mm256_sub_pd(_mm256_mul_pd(w1,tau4), _mm256_add_pd(_mm256_mul_pd(z1,h4), _mm256_add_pd(_mm256_mul_pd(y1,h3), _mm256_mul_pd(x1,h2))));
	w2 = _mm256_sub_pd(_mm256_mul_pd(w2,tau4), _mm256_add_pd(_mm256_mul_pd(z2,h4), _mm256_add_pd(_mm256_mul_pd(y2,h3), _mm256_mul_pd(x2,h2))));
#endif

	__m256d tau5 = _mm256_broadcast_sd(&hh[ldh*4]);
	__m256d vs_1_5 = _mm256_broadcast_sd(&scalarprods[6]);
	__m256d vs_2_5 = _mm256_broadcast_sd(&scalarprods[7]);
	h2 = _mm256_mul_pd(tau5, vs_1_5);
	h3 = _mm256_mul_pd(tau5, vs_2_5);
	__m256d vs_3_5 = _mm256_broadcast_sd(&scalarprods[8]);
	__m256d vs_4_5 = _mm256_broadcast_sd(&scalarprods[9]);
	h4 = _mm256_mul_pd(tau5, vs_3_5);
	h5 = _mm256_mul_pd(tau5, vs_4_5);
#ifdef __ELPA_USE_FMA__
	v1 = _mm256_FMSUB_pd(v1, tau5, _mm256_add_pd(_mm256_FMA_pd(w1, h5, _mm256_mul_pd(z1,h4)), _mm256_FMA_pd(y1, h3, _mm256_mul_pd(x1,h2))));
	v2 = _mm256_FMSUB_pd(v2, tau5, _mm256_add_pd(_mm256_FMA_pd(w2, h5, _mm256_mul_pd(z2,h4)), _mm256_FMA_pd(y2, h3, _mm256_mul_pd(x2,h2))));
#else
	v1 = _mm256_sub_pd(_mm256_mul_pd(v1,tau5), _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(w1,h5), _mm256_mul_pd(z1,h4)), _mm256_add_pd(_mm256_mul_pd(y1,h3), _mm256_mul_pd(x1,h2))));
	v2 = _mm256_sub_pd(_mm256_mul_pd(v2,tau5), _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(w2,h5), _mm256_mul_pd(z2,h4)), _mm256_add_pd(_mm256_mul_pd(y2,h3), _mm256_mul_pd(x2,h2))));
#endif

	__m256d tau6 = _mm256_broadcast_sd(&hh[ldh*5]);
	__m256d vs_1_6 = _mm256_broadcast_sd(&scalarprods[10]);
	__m256d vs_2_6 = _mm256_broadcast_sd(&scalarprods[11]);
	h2 = _mm256_mul_pd(tau6, vs_1_6);
	h3 = _mm256_mul_pd(tau6, vs_2_6);
	__m256d vs_3_6 = _mm256_broadcast_sd(&scalarprods[12]);
	__m256d vs_4_6 = _mm256_broadcast_sd(&scalarprods[13]);
	__m256d vs_5_6 = _mm256_broadcast_sd(&scalarprods[14]);
	h4 = _mm256_mul_pd(tau6, vs_3_6);
	h5 = _mm256_mul_pd(tau6, vs_4_6);
	h6 = _mm256_mul_pd(tau6, vs_5_6);
#ifdef __ELPA_USE_FMA__
	t1 = _mm256_FMSUB_pd(t1, tau6, _mm256_FMA_pd(v1, h6, _mm256_add_pd(_mm256_FMA_pd(w1, h5, _mm256_mul_pd(z1,h4)), _mm256_FMA_pd(y1, h3, _mm256_mul_pd(x1,h2)))));
	t2 = _mm256_FMSUB_pd(t2, tau6, _mm256_FMA_pd(v2, h6, _mm256_add_pd(_mm256_FMA_pd(w2, h5, _mm256_mul_pd(z2,h4)), _mm256_FMA_pd(y2, h3, _mm256_mul_pd(x2,h2)))));
#else
	t1 = _mm256_sub_pd(_mm256_mul_pd(t1,tau6), _mm256_add_pd( _mm256_mul_pd(v1,h6), _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(w1,h5), _mm256_mul_pd(z1,h4)), _mm256_add_pd(_mm256_mul_pd(y1,h3), _mm256_mul_pd(x1,h2)))));
	t2 = _mm256_sub_pd(_mm256_mul_pd(t2,tau6), _mm256_add_pd( _mm256_mul_pd(v2,h6), _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(w2,h5), _mm256_mul_pd(z2,h4)), _mm256_add_pd(_mm256_mul_pd(y2,h3), _mm256_mul_pd(x2,h2)))));
#endif

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	q1 = _mm256_load_pd(&q[0]);
	q2 = _mm256_load_pd(&q[4]);
	q1 = _mm256_sub_pd(q1, t1);
	q2 = _mm256_sub_pd(q2, t2);
	_mm256_store_pd(&q[0],q1);
	_mm256_store_pd(&q[4],q2);

	h6 = _mm256_broadcast_sd(&hh[(ldh*5)+1]);
	q1 = _mm256_load_pd(&q[ldq]);
	q2 = _mm256_load_pd(&q[(ldq+4)]);
	q1 = _mm256_sub_pd(q1, v1);
	q2 = _mm256_sub_pd(q2, v2);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(t1, h6, q1);
	q2 = _mm256_NFMA_pd(t2, h6, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(t1, h6));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(t2, h6));
#endif
	_mm256_store_pd(&q[ldq],q1);
	_mm256_store_pd(&q[(ldq+4)],q2);

	h5 = _mm256_broadcast_sd(&hh[(ldh*4)+1]);
	q1 = _mm256_load_pd(&q[ldq*2]);
	q2 = _mm256_load_pd(&q[(ldq*2)+4]);
	q1 = _mm256_sub_pd(q1, w1);
	q2 = _mm256_sub_pd(q2, w2);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(v1, h5, q1);
	q2 = _mm256_NFMA_pd(v2, h5, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(v1, h5));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(v2, h5));
#endif
	h6 = _mm256_broadcast_sd(&hh[(ldh*5)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(t1, h6, q1);
	q2 = _mm256_NFMA_pd(t2, h6, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(t1, h6));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(t2, h6));
#endif
	_mm256_store_pd(&q[ldq*2],q1);
	_mm256_store_pd(&q[(ldq*2)+4],q2);

	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+1]);
	q1 = _mm256_load_pd(&q[ldq*3]);
	q2 = _mm256_load_pd(&q[(ldq*3)+4]);
	q1 = _mm256_sub_pd(q1, z1);
	q2 = _mm256_sub_pd(q2, z2);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(w1, h4, q1);
	q2 = _mm256_NFMA_pd(w2, h4, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(w1, h4));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(w2, h4));
#endif
	h5 = _mm256_broadcast_sd(&hh[(ldh*4)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(v1, h5, q1);
	q2 = _mm256_NFMA_pd(v2, h5, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(v1, h5));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(v2, h5));
#endif
	h6 = _mm256_broadcast_sd(&hh[(ldh*5)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(t1, h6, q1);
	q2 = _mm256_NFMA_pd(t2, h6, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(t1, h6));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(t2, h6));
#endif
	_mm256_store_pd(&q[ldq*3],q1);
	_mm256_store_pd(&q[(ldq*3)+4],q2);

	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+1]);
	q1 = _mm256_load_pd(&q[ldq*4]);
	q2 = _mm256_load_pd(&q[(ldq*4)+4]);
	q1 = _mm256_sub_pd(q1, y1);
	q2 = _mm256_sub_pd(q2, y2);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(z1, h3, q1);
	q2 = _mm256_NFMA_pd(z2, h3, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(z1, h3));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(z2, h3));
#endif
	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(w1, h4, q1);
	q2 = _mm256_NFMA_pd(w2, h4, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(w1, h4));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(w2, h4));
#endif
	h5 = _mm256_broadcast_sd(&hh[(ldh*4)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(v1, h5, q1);
	q2 = _mm256_NFMA_pd(v2, h5, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(v1, h5));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(v2, h5));
#endif
	h6 = _mm256_broadcast_sd(&hh[(ldh*5)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(t1, h6, q1);
	q2 = _mm256_NFMA_pd(t2, h6, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(t1, h6));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(t2, h6));
#endif
	_mm256_store_pd(&q[ldq*4],q1);
	_mm256_store_pd(&q[(ldq*4)+4],q2);

	h2 = _mm256_broadcast_sd(&hh[(ldh)+1]);
	q1 = _mm256_load_pd(&q[ldq*5]);
	q2 = _mm256_load_pd(&q[(ldq*5)+4]);
	q1 = _mm256_sub_pd(q1, x1);
	q2 = _mm256_sub_pd(q2, x2);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(y1, h2, q1);
	q2 = _mm256_NFMA_pd(y2, h2, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(y1, h2));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(y2, h2));
#endif
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(z1, h3, q1);
	q2 = _mm256_NFMA_pd(z2, h3, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(z1, h3));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(z2, h3));
#endif
	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(w1, h4, q1);
	q2 = _mm256_NFMA_pd(w2, h4, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(w1, h4));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(w2, h4));
#endif
	h5 = _mm256_broadcast_sd(&hh[(ldh*4)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(v1, h5, q1);
	q2 = _mm256_NFMA_pd(v2, h5, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(v1, h5));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(v2, h5));
#endif
	h6 = _mm256_broadcast_sd(&hh[(ldh*5)+5]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(t1, h6, q1);
	q2 = _mm256_NFMA_pd(t2, h6, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(t1, h6));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(t2, h6));
#endif
	_mm256_store_pd(&q[ldq*5],q1);
	_mm256_store_pd(&q[(ldq*5)+4],q2);

	for (i = 6; i < nb; i++)
	{
		q1 = _mm256_load_pd(&q[i*ldq]);
		q2 = _mm256_load_pd(&q[(i*ldq)+4]);
		h1 = _mm256_broadcast_sd(&hh[i-5]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_NFMA_pd(x1, h1, q1);
		q2 = _mm256_NFMA_pd(x2, h1, q2);
#else
		q1 = _mm256_sub_pd(q1, _mm256_mul_pd(x1, h1));
		q2 = _mm256_sub_pd(q2, _mm256_mul_pd(x2, h1));
#endif
		h2 = _mm256_broadcast_sd(&hh[ldh+i-4]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_NFMA_pd(y1, h2, q1);
		q2 = _mm256_NFMA_pd(y2, h2, q2);
#else
		q1 = _mm256_sub_pd(q1, _mm256_mul_pd(y1, h2));
		q2 = _mm256_sub_pd(q2, _mm256_mul_pd(y2, h2));
#endif
		h3 = _mm256_broadcast_sd(&hh[(ldh*2)+i-3]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_NFMA_pd(z1, h3, q1);
		q2 = _mm256_NFMA_pd(z2, h3, q2);
#else
		q1 = _mm256_sub_pd(q1, _mm256_mul_pd(z1, h3));
		q2 = _mm256_sub_pd(q2, _mm256_mul_pd(z2, h3));
#endif
		h4 = _mm256_broadcast_sd(&hh[(ldh*3)+i-2]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_NFMA_pd(w1, h4, q1);
		q2 = _mm256_NFMA_pd(w2, h4, q2);
#else
		q1 = _mm256_sub_pd(q1, _mm256_mul_pd(w1, h4));
		q2 = _mm256_sub_pd(q2, _mm256_mul_pd(w2, h4));
#endif
		h5 = _mm256_broadcast_sd(&hh[(ldh*4)+i-1]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_NFMA_pd(v1, h5, q1);
		q2 = _mm256_NFMA_pd(v2, h5, q2);
#else
		q1 = _mm256_sub_pd(q1, _mm256_mul_pd(v1, h5));
		q2 = _mm256_sub_pd(q2, _mm256_mul_pd(v2, h5));
#endif
		h6 = _mm256_broadcast_sd(&hh[(ldh*5)+i]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_NFMA_pd(t1, h6, q1);
		q2 = _mm256_NFMA_pd(t2, h6, q2);
#else
		q1 = _mm256_sub_pd(q1, _mm256_mul_pd(t1, h6));
		q2 = _mm256_sub_pd(q2, _mm256_mul_pd(t2, h6));
#endif
		_mm256_store_pd(&q[i*ldq],q1);
		_mm256_store_pd(&q[(i*ldq)+4],q2);
	}

	h1 = _mm256_broadcast_sd(&hh[nb-5]);
	q1 = _mm256_load_pd(&q[nb*ldq]);
	q2 = _mm256_load_pd(&q[(nb*ldq)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(x1, h1, q1);
	q2 = _mm256_NFMA_pd(x2, h1, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(x1, h1));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(x2, h1));
#endif
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(y1, h2, q1);
	q2 = _mm256_NFMA_pd(y2, h2, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(y1, h2));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(y2, h2));
#endif
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+nb-3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(z1, h3, q1);
	q2 = _mm256_NFMA_pd(z2, h3, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(z1, h3));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(z2, h3));
#endif
	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(w1, h4, q1);
	q2 = _mm256_NFMA_pd(w2, h4, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(w1, h4));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(w2, h4));
#endif
	h5 = _mm256_broadcast_sd(&hh[(ldh*4)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(v1, h5, q1);
	q2 = _mm256_NFMA_pd(v2, h5, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(v1, h5));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(v2, h5));
#endif
	_mm256_store_pd(&q[nb*ldq],q1);
	_mm256_store_pd(&q[(nb*ldq)+4],q2);

	h1 = _mm256_broadcast_sd(&hh[nb-4]);
	q1 = _mm256_load_pd(&q[(nb+1)*ldq]);
	q2 = _mm256_load_pd(&q[((nb+1)*ldq)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(x1, h1, q1);
	q2 = _mm256_NFMA_pd(x2, h1, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(x1, h1));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(x2, h1));
#endif
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(y1, h2, q1);
	q2 = _mm256_NFMA_pd(y2, h2, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(y1, h2));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(y2, h2));
#endif
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(z1, h3, q1);
	q2 = _mm256_NFMA_pd(z2, h3, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(z1, h3));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(z2, h3));
#endif
	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(w1, h4, q1);
	q2 = _mm256_NFMA_pd(w2, h4, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(w1, h4));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(w2, h4));
#endif
	_mm256_store_pd(&q[(nb+1)*ldq],q1);
	_mm256_store_pd(&q[((nb+1)*ldq)+4],q2);

	h1 = _mm256_broadcast_sd(&hh[nb-3]);
	q1 = _mm256_load_pd(&q[(nb+2)*ldq]);
	q2 = _mm256_load_pd(&q[((nb+2)*ldq)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(x1, h1, q1);
	q2 = _mm256_NFMA_pd(x2, h1, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(x1, h1));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(x2, h1));
#endif
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(y1, h2, q1);
	q2 = _mm256_NFMA_pd(y2, h2, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(y1, h2));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(y2, h2));
#endif
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(z1, h3, q1);
	q2 = _mm256_NFMA_pd(z2, h3, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(z1, h3));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(z2, h3));
#endif
	_mm256_store_pd(&q[(nb+2)*ldq],q1);
	_mm256_store_pd(&q[((nb+2)*ldq)+4],q2);

	h1 = _mm256_broadcast_sd(&hh[nb-2]);
	q1 = _mm256_load_pd(&q[(nb+3)*ldq]);
	q2 = _mm256_load_pd(&q[((nb+3)*ldq)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(x1, h1, q1);
	q2 = _mm256_NFMA_pd(x2, h1, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(x1, h1));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(x2, h1));
#endif
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(y1, h2, q1);
	q2 = _mm256_NFMA_pd(y2, h2, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(y1, h2));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(y2, h2));
#endif
	_mm256_store_pd(&q[(nb+3)*ldq],q1);
	_mm256_store_pd(&q[((nb+3)*ldq)+4],q2);

	h1 = _mm256_broadcast_sd(&hh[nb-1]);
	q1 = _mm256_load_pd(&q[(nb+4)*ldq]);
	q2 = _mm256_load_pd(&q[((nb+4)*ldq)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(x1, h1, q1);
	q2 = _mm256_NFMA_pd(x2, h1, q2);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(x1, h1));
	q2 = _mm256_sub_pd(q2, _mm256_mul_pd(x2, h1));
#endif
	_mm256_store_pd(&q[(nb+4)*ldq],q1);
	_mm256_store_pd(&q[((nb+4)*ldq)+4],q2);
}

/**
 * Unrolled kernel that computes
 * 4 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_4_AVX_6hv(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [8 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m256d a1_1 = _mm256_load_pd(&q[ldq*5]);
	__m256d a2_1 = _mm256_load_pd(&q[ldq*4]);
	__m256d a3_1 = _mm256_load_pd(&q[ldq*3]);
	__m256d a4_1 = _mm256_load_pd(&q[ldq*2]);
	__m256d a5_1 = _mm256_load_pd(&q[ldq]);
	__m256d a6_1 = _mm256_load_pd(&q[0]);

	__m256d h_6_5 = _mm256_broadcast_sd(&hh[(ldh*5)+1]);
	__m256d h_6_4 = _mm256_broadcast_sd(&hh[(ldh*5)+2]);
	__m256d h_6_3 = _mm256_broadcast_sd(&hh[(ldh*5)+3]);
	__m256d h_6_2 = _mm256_broadcast_sd(&hh[(ldh*5)+4]);
	__m256d h_6_1 = _mm256_broadcast_sd(&hh[(ldh*5)+5]);
#ifdef __ELPA_USE_FMA__
	register __m256d t1 = _mm256_FMA_pd(a5_1, h_6_5, a6_1);
	t1 = _mm256_FMA_pd(a4_1, h_6_4, t1);
	t1 = _mm256_FMA_pd(a3_1, h_6_3, t1);
	t1 = _mm256_FMA_pd(a2_1, h_6_2, t1);
	t1 = _mm256_FMA_pd(a1_1, h_6_1, t1);
#else
	register __m256d t1 = _mm256_add_pd(a6_1, _mm256_mul_pd(a5_1, h_6_5));
	t1 = _mm256_add_pd(t1, _mm256_mul_pd(a4_1, h_6_4));
	t1 = _mm256_add_pd(t1, _mm256_mul_pd(a3_1, h_6_3));
	t1 = _mm256_add_pd(t1, _mm256_mul_pd(a2_1, h_6_2));
	t1 = _mm256_add_pd(t1, _mm256_mul_pd(a1_1, h_6_1));
#endif
	__m256d h_5_4 = _mm256_broadcast_sd(&hh[(ldh*4)+1]);
	__m256d h_5_3 = _mm256_broadcast_sd(&hh[(ldh*4)+2]);
	__m256d h_5_2 = _mm256_broadcast_sd(&hh[(ldh*4)+3]);
	__m256d h_5_1 = _mm256_broadcast_sd(&hh[(ldh*4)+4]);
#ifdef __ELPA_USE_FMA__
	register __m256d v1 = _mm256_FMA_pd(a4_1, h_5_4, a5_1);
	v1 = _mm256_FMA_pd(a3_1, h_5_3, v1);
	v1 = _mm256_FMA_pd(a2_1, h_5_2, v1);
	v1 = _mm256_FMA_pd(a1_1, h_5_1, v1);
#else
	register __m256d v1 = _mm256_add_pd(a5_1, _mm256_mul_pd(a4_1, h_5_4));
	v1 = _mm256_add_pd(v1, _mm256_mul_pd(a3_1, h_5_3));
	v1 = _mm256_add_pd(v1, _mm256_mul_pd(a2_1, h_5_2));
	v1 = _mm256_add_pd(v1, _mm256_mul_pd(a1_1, h_5_1));
#endif
	__m256d h_4_3 = _mm256_broadcast_sd(&hh[(ldh*3)+1]);
	__m256d h_4_2 = _mm256_broadcast_sd(&hh[(ldh*3)+2]);
	__m256d h_4_1 = _mm256_broadcast_sd(&hh[(ldh*3)+3]);
#ifdef __ELPA_USE_FMA__
	register __m256d w1 = _mm256_FMA_pd(a3_1, h_4_3, a4_1);
	w1 = _mm256_FMA_pd(a2_1, h_4_2, w1);
	w1 = _mm256_FMA_pd(a1_1, h_4_1, w1);
#else
	register __m256d w1 = _mm256_add_pd(a4_1, _mm256_mul_pd(a3_1, h_4_3));
	w1 = _mm256_add_pd(w1, _mm256_mul_pd(a2_1, h_4_2));
	w1 = _mm256_add_pd(w1, _mm256_mul_pd(a1_1, h_4_1));
#endif
	__m256d h_2_1 = _mm256_broadcast_sd(&hh[ldh+1]);
	__m256d h_3_2 = _mm256_broadcast_sd(&hh[(ldh*2)+1]);
	__m256d h_3_1 = _mm256_broadcast_sd(&hh[(ldh*2)+2]);
#ifdef __ELPA_USE_FMA__
	register __m256d z1 = _mm256_FMA_pd(a2_1, h_3_2, a3_1);
	z1 = _mm256_FMA_pd(a1_1, h_3_1, z1);
	register __m256d y1 = _mm256_FMA_pd(a1_1, h_2_1, a2_1);
#else
	register __m256d z1 = _mm256_add_pd(a3_1, _mm256_mul_pd(a2_1, h_3_2));
	z1 = _mm256_add_pd(z1, _mm256_mul_pd(a1_1, h_3_1));
	register __m256d y1 = _mm256_add_pd(a2_1, _mm256_mul_pd(a1_1, h_2_1));
#endif
	register __m256d x1 = a1_1;

	__m256d q1;

	__m256d h1;
	__m256d h2;
	__m256d h3;
	__m256d h4;
	__m256d h5;
	__m256d h6;

	for(i = 6; i < nb; i++)
	{
		h1 = _mm256_broadcast_sd(&hh[i-5]);
		q1 = _mm256_load_pd(&q[i*ldq]);
#ifdef __ELPA_USE_FMA__
		x1 = _mm256_FMA_pd(q1, h1, x1);
#else
		x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
#endif
		h2 = _mm256_broadcast_sd(&hh[ldh+i-4]);
#ifdef __ELPA_USE_FMA__
		y1 = _mm256_FMA_pd(q1, h2, y1);
#else
		y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
#endif
		h3 = _mm256_broadcast_sd(&hh[(ldh*2)+i-3]);
#ifdef __ELPA_USE_FMA__
		z1 = _mm256_FMA_pd(q1, h3, z1);
#else
		z1 = _mm256_add_pd(z1, _mm256_mul_pd(q1,h3));
#endif
		h4 = _mm256_broadcast_sd(&hh[(ldh*3)+i-2]);
#ifdef __ELPA_USE_FMA__
		w1 = _mm256_FMA_pd(q1, h4, w1);
#else
		w1 = _mm256_add_pd(w1, _mm256_mul_pd(q1,h4));
#endif
		h5 = _mm256_broadcast_sd(&hh[(ldh*4)+i-1]);
#ifdef __ELPA_USE_FMA__
		v1 = _mm256_FMA_pd(q1, h5, v1);
#else
		v1 = _mm256_add_pd(v1, _mm256_mul_pd(q1,h5));
#endif
		h6 = _mm256_broadcast_sd(&hh[(ldh*5)+i]);
#ifdef __ELPA_USE_FMA__
		t1 = _mm256_FMA_pd(q1, h6, t1);
#else
		t1 = _mm256_add_pd(t1, _mm256_mul_pd(q1,h6));
#endif
	}

	h1 = _mm256_broadcast_sd(&hh[nb-5]);
	q1 = _mm256_load_pd(&q[nb*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm256_FMA_pd(q1, h1, x1);
#else
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
#endif
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-4]);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMA_pd(q1, h2, y1);
#else
	y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
#endif
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+nb-3]);
#ifdef __ELPA_USE_FMA__
	z1 = _mm256_FMA_pd(q1, h3, z1);
#else
	z1 = _mm256_add_pd(z1, _mm256_mul_pd(q1,h3));
#endif
	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+nb-2]);
#ifdef __ELPA_USE_FMA__
	w1 = _mm256_FMA_pd(q1, h4, w1);
#else
	w1 = _mm256_add_pd(w1, _mm256_mul_pd(q1,h4));
#endif
	h5 = _mm256_broadcast_sd(&hh[(ldh*4)+nb-1]);
#ifdef __ELPA_USE_FMA__
	v1 = _mm256_FMA_pd(q1, h5, v1);
#else
	v1 = _mm256_add_pd(v1, _mm256_mul_pd(q1,h5));
#endif

	h1 = _mm256_broadcast_sd(&hh[nb-4]);
	q1 = _mm256_load_pd(&q[(nb+1)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm256_FMA_pd(q1, h1, x1);
#else
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
#endif
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-3]);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMA_pd(q1, h2, y1);
#else
	y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
#endif
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+nb-2]);
#ifdef __ELPA_USE_FMA__
	z1 = _mm256_FMA_pd(q1, h3, z1);
#else
	z1 = _mm256_add_pd(z1, _mm256_mul_pd(q1,h3));
#endif
	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+nb-1]);
#ifdef __ELPA_USE_FMA__
	w1 = _mm256_FMA_pd(q1, h4, w1);
#else
	w1 = _mm256_add_pd(w1, _mm256_mul_pd(q1,h4));
#endif

	h1 = _mm256_broadcast_sd(&hh[nb-3]);
	q1 = _mm256_load_pd(&q[(nb+2)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm256_FMA_pd(q1, h1, x1);
#else
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
#endif
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-2]);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMA_pd(q1, h2, y1);
#else
	y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
#endif
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+nb-1]);
#ifdef __ELPA_USE_FMA__
	z1 = _mm256_FMA_pd(q1, h3, z1);
#else
	z1 = _mm256_add_pd(z1, _mm256_mul_pd(q1,h3));
#endif

	h1 = _mm256_broadcast_sd(&hh[nb-2]);
	q1 = _mm256_load_pd(&q[(nb+3)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm256_FMA_pd(q1, h1, x1);
#else
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
#endif
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-1]);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMA_pd(q1, h2, y1);
#else
	y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
#endif

	h1 = _mm256_broadcast_sd(&hh[nb-1]);
	q1 = _mm256_load_pd(&q[(nb+4)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm256_FMA_pd(q1, h1, x1);
#else
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
#endif

	/////////////////////////////////////////////////////
	// Apply tau, correct wrong calculation using pre-calculated scalar products
	/////////////////////////////////////////////////////

	__m256d tau1 = _mm256_broadcast_sd(&hh[0]);
	x1 = _mm256_mul_pd(x1, tau1);

	__m256d tau2 = _mm256_broadcast_sd(&hh[ldh]);
	__m256d vs_1_2 = _mm256_broadcast_sd(&scalarprods[0]);
	h2 = _mm256_mul_pd(tau2, vs_1_2);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMSUB_pd(y1, tau2, _mm256_mul_pd(x1,h2));
#else
	y1 = _mm256_sub_pd(_mm256_mul_pd(y1,tau2), _mm256_mul_pd(x1,h2));
#endif

	__m256d tau3 = _mm256_broadcast_sd(&hh[ldh*2]);
	__m256d vs_1_3 = _mm256_broadcast_sd(&scalarprods[1]);
	__m256d vs_2_3 = _mm256_broadcast_sd(&scalarprods[2]);
	h2 = _mm256_mul_pd(tau3, vs_1_3);
	h3 = _mm256_mul_pd(tau3, vs_2_3);
#ifdef __ELPA_USE_FMA__
	z1 = _mm256_FMSUB_pd(z1, tau3, _mm256_FMA_pd(y1, h3, _mm256_mul_pd(x1,h2)));
#else
	z1 = _mm256_sub_pd(_mm256_mul_pd(z1,tau3), _mm256_add_pd(_mm256_mul_pd(y1,h3), _mm256_mul_pd(x1,h2)));
#endif

	__m256d tau4 = _mm256_broadcast_sd(&hh[ldh*3]);
	__m256d vs_1_4 = _mm256_broadcast_sd(&scalarprods[3]);
	__m256d vs_2_4 = _mm256_broadcast_sd(&scalarprods[4]);
	h2 = _mm256_mul_pd(tau4, vs_1_4);
	h3 = _mm256_mul_pd(tau4, vs_2_4);
	__m256d vs_3_4 = _mm256_broadcast_sd(&scalarprods[5]);
	h4 = _mm256_mul_pd(tau4, vs_3_4);
#ifdef __ELPA_USE_FMA__
	w1 = _mm256_FMSUB_pd(w1, tau4, _mm256_FMA_pd(z1, h4, _mm256_FMA_pd(y1, h3, _mm256_mul_pd(x1,h2))));
#else
	w1 = _mm256_sub_pd(_mm256_mul_pd(w1,tau4), _mm256_add_pd(_mm256_mul_pd(z1,h4), _mm256_add_pd(_mm256_mul_pd(y1,h3), _mm256_mul_pd(x1,h2))));
#endif

	__m256d tau5 = _mm256_broadcast_sd(&hh[ldh*4]);
	__m256d vs_1_5 = _mm256_broadcast_sd(&scalarprods[6]);
	__m256d vs_2_5 = _mm256_broadcast_sd(&scalarprods[7]);
	h2 = _mm256_mul_pd(tau5, vs_1_5);
	h3 = _mm256_mul_pd(tau5, vs_2_5);
	__m256d vs_3_5 = _mm256_broadcast_sd(&scalarprods[8]);
	__m256d vs_4_5 = _mm256_broadcast_sd(&scalarprods[9]);
	h4 = _mm256_mul_pd(tau5, vs_3_5);
	h5 = _mm256_mul_pd(tau5, vs_4_5);
#ifdef __ELPA_USE_FMA__
	v1 = _mm256_FMSUB_pd(v1, tau5, _mm256_add_pd(_mm256_FMA_pd(w1, h5, _mm256_mul_pd(z1,h4)), _mm256_FMA_pd(y1, h3, _mm256_mul_pd(x1,h2))));
#else
	v1 = _mm256_sub_pd(_mm256_mul_pd(v1,tau5), _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(w1,h5), _mm256_mul_pd(z1,h4)), _mm256_add_pd(_mm256_mul_pd(y1,h3), _mm256_mul_pd(x1,h2))));
#endif

	__m256d tau6 = _mm256_broadcast_sd(&hh[ldh*5]);
	__m256d vs_1_6 = _mm256_broadcast_sd(&scalarprods[10]);
	__m256d vs_2_6 = _mm256_broadcast_sd(&scalarprods[11]);
	h2 = _mm256_mul_pd(tau6, vs_1_6);
	h3 = _mm256_mul_pd(tau6, vs_2_6);
	__m256d vs_3_6 = _mm256_broadcast_sd(&scalarprods[12]);
	__m256d vs_4_6 = _mm256_broadcast_sd(&scalarprods[13]);
	__m256d vs_5_6 = _mm256_broadcast_sd(&scalarprods[14]);
	h4 = _mm256_mul_pd(tau6, vs_3_6);
	h5 = _mm256_mul_pd(tau6, vs_4_6);
	h6 = _mm256_mul_pd(tau6, vs_5_6);
#ifdef __ELPA_USE_FMA__
	t1 = _mm256_FMSUB_pd(t1, tau6, _mm256_FMA_pd(v1, h6, _mm256_add_pd(_mm256_FMA_pd(w1, h5, _mm256_mul_pd(z1,h4)), _mm256_FMA_pd(y1, h3, _mm256_mul_pd(x1,h2)))));
#else
	t1 = _mm256_sub_pd(_mm256_mul_pd(t1,tau6), _mm256_add_pd( _mm256_mul_pd(v1,h6), _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(w1,h5), _mm256_mul_pd(z1,h4)), _mm256_add_pd(_mm256_mul_pd(y1,h3), _mm256_mul_pd(x1,h2)))));
#endif

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [4 x nb+3]
	/////////////////////////////////////////////////////

	q1 = _mm256_load_pd(&q[0]);
	q1 = _mm256_sub_pd(q1, t1);
	_mm256_store_pd(&q[0],q1);

	h6 = _mm256_broadcast_sd(&hh[(ldh*5)+1]);
	q1 = _mm256_load_pd(&q[ldq]);
	q1 = _mm256_sub_pd(q1, v1);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(t1, h6, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(t1, h6));
#endif
	_mm256_store_pd(&q[ldq],q1);

	h5 = _mm256_broadcast_sd(&hh[(ldh*4)+1]);
	q1 = _mm256_load_pd(&q[ldq*2]);
	q1 = _mm256_sub_pd(q1, w1);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(v1, h5, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(v1, h5));
#endif
	h6 = _mm256_broadcast_sd(&hh[(ldh*5)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(t1, h6, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(t1, h6));
#endif
	_mm256_store_pd(&q[ldq*2],q1);

	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+1]);
	q1 = _mm256_load_pd(&q[ldq*3]);
	q1 = _mm256_sub_pd(q1, z1);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(w1, h4, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(w1, h4));
#endif
	h5 = _mm256_broadcast_sd(&hh[(ldh*4)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(v1, h5, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(v1, h5));
#endif
	h6 = _mm256_broadcast_sd(&hh[(ldh*5)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(t1, h6, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(t1, h6));
#endif
	_mm256_store_pd(&q[ldq*3],q1);

	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+1]);
	q1 = _mm256_load_pd(&q[ldq*4]);
	q1 = _mm256_sub_pd(q1, y1);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(z1, h3, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(z1, h3));
#endif
	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(w1, h4, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(w1, h4));
#endif
	h5 = _mm256_broadcast_sd(&hh[(ldh*4)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(v1, h5, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(v1, h5));
#endif
	h6 = _mm256_broadcast_sd(&hh[(ldh*5)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(t1, h6, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(t1, h6));
#endif
	_mm256_store_pd(&q[ldq*4],q1);

	h2 = _mm256_broadcast_sd(&hh[(ldh)+1]);
	q1 = _mm256_load_pd(&q[ldq*5]);
	q1 = _mm256_sub_pd(q1, x1);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(y1, h2, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(y1, h2));
#endif
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(z1, h3, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(z1, h3));
#endif
	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(w1, h4, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(w1, h4));
#endif
	h5 = _mm256_broadcast_sd(&hh[(ldh*4)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(v1, h5, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(v1, h5));
#endif
	h6 = _mm256_broadcast_sd(&hh[(ldh*5)+5]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(t1, h6, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(t1, h6));
#endif
	_mm256_store_pd(&q[ldq*5],q1);

	for (i = 6; i < nb; i++)
	{
		q1 = _mm256_load_pd(&q[i*ldq]);
		h1 = _mm256_broadcast_sd(&hh[i-5]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_NFMA_pd(x1, h1, q1);
#else
		q1 = _mm256_sub_pd(q1, _mm256_mul_pd(x1, h1));
#endif
		h2 = _mm256_broadcast_sd(&hh[ldh+i-4]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_NFMA_pd(y1, h2, q1);
#else
		q1 = _mm256_sub_pd(q1, _mm256_mul_pd(y1, h2));
#endif
		h3 = _mm256_broadcast_sd(&hh[(ldh*2)+i-3]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_NFMA_pd(z1, h3, q1);
#else
		q1 = _mm256_sub_pd(q1, _mm256_mul_pd(z1, h3));
#endif
		h4 = _mm256_broadcast_sd(&hh[(ldh*3)+i-2]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_NFMA_pd(w1, h4, q1);
#else
		q1 = _mm256_sub_pd(q1, _mm256_mul_pd(w1, h4));
#endif
		h5 = _mm256_broadcast_sd(&hh[(ldh*4)+i-1]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_NFMA_pd(v1, h5, q1);
#else
		q1 = _mm256_sub_pd(q1, _mm256_mul_pd(v1, h5));
#endif
		h6 = _mm256_broadcast_sd(&hh[(ldh*5)+i]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_NFMA_pd(t1, h6, q1);
#else
		q1 = _mm256_sub_pd(q1, _mm256_mul_pd(t1, h6));
#endif
		_mm256_store_pd(&q[i*ldq],q1);
	}

	h1 = _mm256_broadcast_sd(&hh[nb-5]);
	q1 = _mm256_load_pd(&q[nb*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(x1, h1, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(x1, h1));
#endif
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(y1, h2, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(y1, h2));
#endif
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+nb-3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(z1, h3, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(z1, h3));
#endif
	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(w1, h4, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(w1, h4));
#endif
	h5 = _mm256_broadcast_sd(&hh[(ldh*4)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(v1, h5, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(v1, h5));
#endif
	_mm256_store_pd(&q[nb*ldq],q1);

	h1 = _mm256_broadcast_sd(&hh[nb-4]);
	q1 = _mm256_load_pd(&q[(nb+1)*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(x1, h1, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(x1, h1));
#endif
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(y1, h2, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(y1, h2));
#endif
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(z1, h3, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(z1, h3));
#endif
	h4 = _mm256_broadcast_sd(&hh[(ldh*3)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(w1, h4, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(w1, h4));
#endif
	_mm256_store_pd(&q[(nb+1)*ldq],q1);

	h1 = _mm256_broadcast_sd(&hh[nb-3]);
	q1 = _mm256_load_pd(&q[(nb+2)*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(x1, h1, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(x1, h1));
#endif
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(y1, h2, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(y1, h2));
#endif
	h3 = _mm256_broadcast_sd(&hh[(ldh*2)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(z1, h3, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(z1, h3));
#endif
	_mm256_store_pd(&q[(nb+2)*ldq],q1);

	h1 = _mm256_broadcast_sd(&hh[nb-2]);
	q1 = _mm256_load_pd(&q[(nb+3)*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(x1, h1, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(x1, h1));
#endif
	h2 = _mm256_broadcast_sd(&hh[ldh+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(y1, h2, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(y1, h2));
#endif
	_mm256_store_pd(&q[(nb+3)*ldq],q1);

	h1 = _mm256_broadcast_sd(&hh[nb-1]);
	q1 = _mm256_load_pd(&q[(nb+4)*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_NFMA_pd(x1, h1, q1);
#else
	q1 = _mm256_sub_pd(q1, _mm256_mul_pd(x1, h1));
#endif
	_mm256_store_pd(&q[(nb+4)*ldq],q1);
}
#else
/**
 * Unrolled kernel that computes
 * 4 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_4_SSE_6hv(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
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
#ifdef __ELPA_USE_FMA__
	register __m128d t1 = _mm_macc_pd(a5_1, h_6_5, a6_1);
	t1 = _mm_macc_pd(a4_1, h_6_4, t1);
	t1 = _mm_macc_pd(a3_1, h_6_3, t1);
	t1 = _mm_macc_pd(a2_1, h_6_2, t1);
	t1 = _mm_macc_pd(a1_1, h_6_1, t1);
#else
	register __m128d t1 = _mm_add_pd(a6_1, _mm_mul_pd(a5_1, h_6_5));
	t1 = _mm_add_pd(t1, _mm_mul_pd(a4_1, h_6_4));
	t1 = _mm_add_pd(t1, _mm_mul_pd(a3_1, h_6_3));
	t1 = _mm_add_pd(t1, _mm_mul_pd(a2_1, h_6_2));
	t1 = _mm_add_pd(t1, _mm_mul_pd(a1_1, h_6_1));
#endif
	__m128d h_5_4 = _mm_loaddup_pd(&hh[(ldh*4)+1]);
	__m128d h_5_3 = _mm_loaddup_pd(&hh[(ldh*4)+2]);
	__m128d h_5_2 = _mm_loaddup_pd(&hh[(ldh*4)+3]);
	__m128d h_5_1 = _mm_loaddup_pd(&hh[(ldh*4)+4]);
#ifdef __ELPA_USE_FMA__
	register __m128d v1 = _mm_macc_pd(a4_1, h_5_4, a5_1);
	v1 = _mm_macc_pd(a3_1, h_5_3, v1);
	v1 = _mm_macc_pd(a2_1, h_5_2, v1);
	v1 = _mm_macc_pd(a1_1, h_5_1, v1);
#else
	register __m128d v1 = _mm_add_pd(a5_1, _mm_mul_pd(a4_1, h_5_4));
	v1 = _mm_add_pd(v1, _mm_mul_pd(a3_1, h_5_3));
	v1 = _mm_add_pd(v1, _mm_mul_pd(a2_1, h_5_2));
	v1 = _mm_add_pd(v1, _mm_mul_pd(a1_1, h_5_1));
#endif
	__m128d h_4_3 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
	__m128d h_4_2 = _mm_loaddup_pd(&hh[(ldh*3)+2]);
	__m128d h_4_1 = _mm_loaddup_pd(&hh[(ldh*3)+3]);
#ifdef __ELPA_USE_FMA__
	register __m128d w1 = _mm_macc_pd(a3_1, h_4_3, a4_1);
	w1 = _mm_macc_pd(a2_1, h_4_2, w1);
	w1 = _mm_macc_pd(a1_1, h_4_1, w1);
#else
	register __m128d w1 = _mm_add_pd(a4_1, _mm_mul_pd(a3_1, h_4_3));
	w1 = _mm_add_pd(w1, _mm_mul_pd(a2_1, h_4_2));
	w1 = _mm_add_pd(w1, _mm_mul_pd(a1_1, h_4_1));
#endif
	__m128d h_2_1 = _mm_loaddup_pd(&hh[ldh+1]);
	__m128d h_3_2 = _mm_loaddup_pd(&hh[(ldh*2)+1]);
	__m128d h_3_1 = _mm_loaddup_pd(&hh[(ldh*2)+2]);
#ifdef __ELPA_USE_FMA__
	register __m128d z1 = _mm_macc_pd(a2_1, h_3_2, a3_1);
	z1 = _mm_macc_pd(a1_1, h_3_1, z1);
	register __m128d y1 = _mm_macc_pd(a1_1, h_2_1, a2_1);
#else
	register __m128d z1 = _mm_add_pd(a3_1, _mm_mul_pd(a2_1, h_3_2));
	z1 = _mm_add_pd(z1, _mm_mul_pd(a1_1, h_3_1));
	register __m128d y1 = _mm_add_pd(a2_1, _mm_mul_pd(a1_1, h_2_1));
#endif
	register __m128d x1 = a1_1;

	__m128d a1_2 = _mm_load_pd(&q[(ldq*5)+2]);
	__m128d a2_2 = _mm_load_pd(&q[(ldq*4)+2]);
	__m128d a3_2 = _mm_load_pd(&q[(ldq*3)+2]);
	__m128d a4_2 = _mm_load_pd(&q[(ldq*2)+2]);
	__m128d a5_2 = _mm_load_pd(&q[(ldq)+2]);
	__m128d a6_2 = _mm_load_pd(&q[2]);

#ifdef __ELPA_USE_FMA__
	register __m128d t2 = _mm_macc_pd(a5_2, h_6_5, a6_2);
	t2 = _mm_macc_pd(a4_2, h_6_4, t2);
	t2 = _mm_macc_pd(a3_2, h_6_3, t2);
	t2 = _mm_macc_pd(a2_2, h_6_2, t2);
	t2 = _mm_macc_pd(a1_2, h_6_1, t2);
	register __m128d v2 = _mm_macc_pd(a4_2, h_5_4, a5_2);
	v2 = _mm_macc_pd(a3_2, h_5_3, v2);
	v2 = _mm_macc_pd(a2_2, h_5_2, v2);
	v2 = _mm_macc_pd(a1_2, h_5_1, v2);
	register __m128d w2 = _mm_macc_pd(a3_2, h_4_3, a4_2);
	w2 = _mm_macc_pd(a2_2, h_4_2, w2);
	w2 = _mm_macc_pd(a1_2, h_4_1, w2);
	register __m128d z2 = _mm_macc_pd(a2_2, h_3_2, a3_2);
	z2 = _mm_macc_pd(a1_2, h_3_1, z2);
	register __m128d y2 = _mm_macc_pd(a1_2, h_2_1, a2_2);
#else
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
#endif
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
#ifdef __ELPA_USE_FMA__
		x1 = _mm_macc_pd(q1, h1, x1);
		x2 = _mm_macc_pd(q2, h1, x2);
#else
		x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
		x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));
#endif
		h2 = _mm_loaddup_pd(&hh[ldh+i-4]);
#ifdef __ELPA_USE_FMA__
		y1 = _mm_macc_pd(q1, h2, y1);
		y2 = _mm_macc_pd(q2, h2, y2);
#else
		y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
		y2 = _mm_add_pd(y2, _mm_mul_pd(q2,h2));
#endif
		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-3]);
#ifdef __ELPA_USE_FMA__
		z1 = _mm_macc_pd(q1, h3, z1);
		z2 = _mm_macc_pd(q2, h3, z2);
#else
		z1 = _mm_add_pd(z1, _mm_mul_pd(q1,h3));
		z2 = _mm_add_pd(z2, _mm_mul_pd(q2,h3));
#endif
		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i-2]);
#ifdef __ELPA_USE_FMA__
		w1 = _mm_macc_pd(q1, h4, w1);
		w2 = _mm_macc_pd(q2, h4, w2);
#else
		w1 = _mm_add_pd(w1, _mm_mul_pd(q1,h4));
		w2 = _mm_add_pd(w2, _mm_mul_pd(q2,h4));
#endif
		h5 = _mm_loaddup_pd(&hh[(ldh*4)+i-1]);
#ifdef __ELPA_USE_FMA__
		v1 = _mm_macc_pd(q1, h5, v1);
		v2 = _mm_macc_pd(q2, h5, v2);
#else
		v1 = _mm_add_pd(v1, _mm_mul_pd(q1,h5));
		v2 = _mm_add_pd(v2, _mm_mul_pd(q2,h5));
#endif
		h6 = _mm_loaddup_pd(&hh[(ldh*5)+i]);
#ifdef __ELPA_USE_FMA__
		t1 = _mm_macc_pd(q1, h6, t1);
		t2 = _mm_macc_pd(q2, h6, t2);
#else
		t1 = _mm_add_pd(t1, _mm_mul_pd(q1,h6));
		t2 = _mm_add_pd(t2, _mm_mul_pd(q2,h6));
#endif
	}

	h1 = _mm_loaddup_pd(&hh[nb-5]);
	q1 = _mm_load_pd(&q[nb*ldq]);
	q2 = _mm_load_pd(&q[(nb*ldq)+2]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm_macc_pd(q1, h1, x1);
	x2 = _mm_macc_pd(q2, h1, x2);
#else
	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
	x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));
#endif
	h2 = _mm_loaddup_pd(&hh[ldh+nb-4]);
#ifdef __ELPA_USE_FMA__
	y1 = _mm_macc_pd(q1, h2, y1);
	y2 = _mm_macc_pd(q2, h2, y2);
#else
	y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
	y2 = _mm_add_pd(y2, _mm_mul_pd(q2,h2));
#endif
	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-3]);
#ifdef __ELPA_USE_FMA__
	z1 = _mm_macc_pd(q1, h3, z1);
	z2 = _mm_macc_pd(q2, h3, z2);
#else
	z1 = _mm_add_pd(z1, _mm_mul_pd(q1,h3));
	z2 = _mm_add_pd(z2, _mm_mul_pd(q2,h3));
#endif
	h4 = _mm_loaddup_pd(&hh[(ldh*3)+nb-2]);
#ifdef __ELPA_USE_FMA__
	w1 = _mm_macc_pd(q1, h4, w1);
	w2 = _mm_macc_pd(q2, h4, w2);
#else
	w1 = _mm_add_pd(w1, _mm_mul_pd(q1,h4));
	w2 = _mm_add_pd(w2, _mm_mul_pd(q2,h4));
#endif
	h5 = _mm_loaddup_pd(&hh[(ldh*4)+nb-1]);
#ifdef __ELPA_USE_FMA__
	v1 = _mm_macc_pd(q1, h5, v1);
	v2 = _mm_macc_pd(q2, h5, v2);
#else
	v1 = _mm_add_pd(v1, _mm_mul_pd(q1,h5));
	v2 = _mm_add_pd(v2, _mm_mul_pd(q2,h5));
#endif

	h1 = _mm_loaddup_pd(&hh[nb-4]);
	q1 = _mm_load_pd(&q[(nb+1)*ldq]);
	q2 = _mm_load_pd(&q[((nb+1)*ldq)+2]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm_macc_pd(q1, h1, x1);
	x2 = _mm_macc_pd(q2, h1, x2);
#else
	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
	x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));
#endif
	h2 = _mm_loaddup_pd(&hh[ldh+nb-3]);
#ifdef __ELPA_USE_FMA__
	y1 = _mm_macc_pd(q1, h2, y1);
	y2 = _mm_macc_pd(q2, h2, y2);
#else
	y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
	y2 = _mm_add_pd(y2, _mm_mul_pd(q2,h2));
#endif
	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-2]);
#ifdef __ELPA_USE_FMA__
	z1 = _mm_macc_pd(q1, h3, z1);
	z2 = _mm_macc_pd(q2, h3, z2);
#else
	z1 = _mm_add_pd(z1, _mm_mul_pd(q1,h3));
	z2 = _mm_add_pd(z2, _mm_mul_pd(q2,h3));
#endif
	h4 = _mm_loaddup_pd(&hh[(ldh*3)+nb-1]);
#ifdef __ELPA_USE_FMA__
	w1 = _mm_macc_pd(q1, h4, w1);
	w2 = _mm_macc_pd(q2, h4, w2);
#else
	w1 = _mm_add_pd(w1, _mm_mul_pd(q1,h4));
	w2 = _mm_add_pd(w2, _mm_mul_pd(q2,h4));
#endif

	h1 = _mm_loaddup_pd(&hh[nb-3]);
	q1 = _mm_load_pd(&q[(nb+2)*ldq]);
	q2 = _mm_load_pd(&q[((nb+2)*ldq)+2]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm_macc_pd(q1, h1, x1);
	x2 = _mm_macc_pd(q2, h1, x2);
#else
	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
	x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));
#endif
	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);
#ifdef __ELPA_USE_FMA__
	y1 = _mm_macc_pd(q1, h2, y1);
	y2 = _mm_macc_pd(q2, h2, y2);
#else
	y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
	y2 = _mm_add_pd(y2, _mm_mul_pd(q2,h2));
#endif
	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);
#ifdef __ELPA_USE_FMA__
	z1 = _mm_macc_pd(q1, h3, z1);
	z2 = _mm_macc_pd(q2, h3, z2);
#else
	z1 = _mm_add_pd(z1, _mm_mul_pd(q1,h3));
	z2 = _mm_add_pd(z2, _mm_mul_pd(q2,h3));
#endif

	h1 = _mm_loaddup_pd(&hh[nb-2]);
	q1 = _mm_load_pd(&q[(nb+3)*ldq]);
	q2 = _mm_load_pd(&q[((nb+3)*ldq)+2]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm_macc_pd(q1, h1, x1);
	x2 = _mm_macc_pd(q2, h1, x2);
#else
	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
	x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));
#endif
	h2 = _mm_loaddup_pd(&hh[ldh+nb-1]);
#ifdef __ELPA_USE_FMA__
	y1 = _mm_macc_pd(q1, h2, y1);
	y2 = _mm_macc_pd(q2, h2, y2);
#else
	y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
	y2 = _mm_add_pd(y2, _mm_mul_pd(q2,h2));
#endif

	h1 = _mm_loaddup_pd(&hh[nb-1]);
	q1 = _mm_load_pd(&q[(nb+4)*ldq]);
	q2 = _mm_load_pd(&q[((nb+4)*ldq)+2]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm_macc_pd(q1, h1, x1);
	x2 = _mm_macc_pd(q2, h1, x2);
#else
	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
	x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));
#endif

	/////////////////////////////////////////////////////
	// Apply tau, correct wrong calculation using pre-calculated scalar products
	/////////////////////////////////////////////////////

	__m128d tau1 = _mm_loaddup_pd(&hh[0]);
	x1 = _mm_mul_pd(x1, tau1);
	x2 = _mm_mul_pd(x2, tau1);

	__m128d tau2 = _mm_loaddup_pd(&hh[ldh]);
	__m128d vs_1_2 = _mm_loaddup_pd(&scalarprods[0]);
	h2 = _mm_mul_pd(tau2, vs_1_2);
#ifdef __ELPA_USE_FMA__
	y1 = _mm_msub_pd(y1, tau2, _mm_mul_pd(x1,h2));
	y2 = _mm_msub_pd(y2, tau2, _mm_mul_pd(x2,h2));
#else
	y1 = _mm_sub_pd(_mm_mul_pd(y1,tau2), _mm_mul_pd(x1,h2));
	y2 = _mm_sub_pd(_mm_mul_pd(y2,tau2), _mm_mul_pd(x2,h2));
#endif

	__m128d tau3 = _mm_loaddup_pd(&hh[ldh*2]);
	__m128d vs_1_3 = _mm_loaddup_pd(&scalarprods[1]);
	__m128d vs_2_3 = _mm_loaddup_pd(&scalarprods[2]);
	h2 = _mm_mul_pd(tau3, vs_1_3);
	h3 = _mm_mul_pd(tau3, vs_2_3);
#ifdef __ELPA_USE_FMA__
	z1 = _mm_msub_pd(z1, tau3, _mm_macc_pd(y1, h3, _mm_mul_pd(x1,h2)));
	z2 = _mm_msub_pd(z2, tau3, _mm_macc_pd(y2, h3, _mm_mul_pd(x2,h2)));
#else
	z1 = _mm_sub_pd(_mm_mul_pd(z1,tau3), _mm_add_pd(_mm_mul_pd(y1,h3), _mm_mul_pd(x1,h2)));
	z2 = _mm_sub_pd(_mm_mul_pd(z2,tau3), _mm_add_pd(_mm_mul_pd(y2,h3), _mm_mul_pd(x2,h2)));
#endif

	__m128d tau4 = _mm_loaddup_pd(&hh[ldh*3]);
	__m128d vs_1_4 = _mm_loaddup_pd(&scalarprods[3]);
	__m128d vs_2_4 = _mm_loaddup_pd(&scalarprods[4]);
	h2 = _mm_mul_pd(tau4, vs_1_4);
	h3 = _mm_mul_pd(tau4, vs_2_4);
	__m128d vs_3_4 = _mm_loaddup_pd(&scalarprods[5]);
	h4 = _mm_mul_pd(tau4, vs_3_4);
#ifdef __ELPA_USE_FMA__
	w1 = _mm_msub_pd(w1, tau4, _mm_macc_pd(z1, h4, _mm_macc_pd(y1, h3, _mm_mul_pd(x1,h2))));
	w2 = _mm_msub_pd(w2, tau4, _mm_macc_pd(z2, h4, _mm_macc_pd(y2, h3, _mm_mul_pd(x2,h2))));
#else
	w1 = _mm_sub_pd(_mm_mul_pd(w1,tau4), _mm_add_pd(_mm_mul_pd(z1,h4), _mm_add_pd(_mm_mul_pd(y1,h3), _mm_mul_pd(x1,h2))));
	w2 = _mm_sub_pd(_mm_mul_pd(w2,tau4), _mm_add_pd(_mm_mul_pd(z2,h4), _mm_add_pd(_mm_mul_pd(y2,h3), _mm_mul_pd(x2,h2))));
#endif

	__m128d tau5 = _mm_loaddup_pd(&hh[ldh*4]);
	__m128d vs_1_5 = _mm_loaddup_pd(&scalarprods[6]);
	__m128d vs_2_5 = _mm_loaddup_pd(&scalarprods[7]);
	h2 = _mm_mul_pd(tau5, vs_1_5);
	h3 = _mm_mul_pd(tau5, vs_2_5);
	__m128d vs_3_5 = _mm_loaddup_pd(&scalarprods[8]);
	__m128d vs_4_5 = _mm_loaddup_pd(&scalarprods[9]);
	h4 = _mm_mul_pd(tau5, vs_3_5);
	h5 = _mm_mul_pd(tau5, vs_4_5);
#ifdef __ELPA_USE_FMA__
	v1 = _mm_msub_pd(v1, tau5, _mm_add_pd(_mm_macc_pd(w1, h5, _mm_mul_pd(z1,h4)), _mm_macc_pd(y1, h3, _mm_mul_pd(x1,h2))));
	v2 = _mm_msub_pd(v2, tau5, _mm_add_pd(_mm_macc_pd(w2, h5, _mm_mul_pd(z2,h4)), _mm_macc_pd(y2, h3, _mm_mul_pd(x2,h2))));
#else
	v1 = _mm_sub_pd(_mm_mul_pd(v1,tau5), _mm_add_pd(_mm_add_pd(_mm_mul_pd(w1,h5), _mm_mul_pd(z1,h4)), _mm_add_pd(_mm_mul_pd(y1,h3), _mm_mul_pd(x1,h2))));
	v2 = _mm_sub_pd(_mm_mul_pd(v2,tau5), _mm_add_pd(_mm_add_pd(_mm_mul_pd(w2,h5), _mm_mul_pd(z2,h4)), _mm_add_pd(_mm_mul_pd(y2,h3), _mm_mul_pd(x2,h2))));
#endif

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
#ifdef __ELPA_USE_FMA__
	t1 = _mm_msub_pd(t1, tau6, _mm_macc_pd(v1, h6, _mm_add_pd(_mm_macc_pd(w1, h5, _mm_mul_pd(z1,h4)), _mm_macc_pd(y1, h3, _mm_mul_pd(x1,h2)))));
	t2 = _mm_msub_pd(t2, tau6, _mm_macc_pd(v2, h6, _mm_add_pd(_mm_macc_pd(w2, h5, _mm_mul_pd(z2,h4)), _mm_macc_pd(y2, h3, _mm_mul_pd(x2,h2)))));
#else
	t1 = _mm_sub_pd(_mm_mul_pd(t1,tau6), _mm_add_pd( _mm_mul_pd(v1,h6), _mm_add_pd(_mm_add_pd(_mm_mul_pd(w1,h5), _mm_mul_pd(z1,h4)), _mm_add_pd(_mm_mul_pd(y1,h3), _mm_mul_pd(x1,h2)))));
	t2 = _mm_sub_pd(_mm_mul_pd(t2,tau6), _mm_add_pd( _mm_mul_pd(v2,h6), _mm_add_pd(_mm_add_pd(_mm_mul_pd(w2,h5), _mm_mul_pd(z2,h4)), _mm_add_pd(_mm_mul_pd(y2,h3), _mm_mul_pd(x2,h2)))));
#endif

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
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(t1, h6, q1);
	q2 = _mm_nmacc_pd(t2, h6, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(t2, h6));
#endif
	_mm_store_pd(&q[ldq],q1);
	_mm_store_pd(&q[(ldq+2)],q2);

	h5 = _mm_loaddup_pd(&hh[(ldh*4)+1]);
	q1 = _mm_load_pd(&q[ldq*2]);
	q2 = _mm_load_pd(&q[(ldq*2)+2]);
	q1 = _mm_sub_pd(q1, w1);
	q2 = _mm_sub_pd(q2, w2);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(v1, h5, q1);
	q2 = _mm_nmacc_pd(v2, h5, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(v2, h5));
#endif
	h6 = _mm_loaddup_pd(&hh[(ldh*5)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(t1, h6, q1);
	q2 = _mm_nmacc_pd(t2, h6, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(t2, h6));
#endif
	_mm_store_pd(&q[ldq*2],q1);
	_mm_store_pd(&q[(ldq*2)+2],q2);

	h4 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
	q1 = _mm_load_pd(&q[ldq*3]);
	q2 = _mm_load_pd(&q[(ldq*3)+2]);
	q1 = _mm_sub_pd(q1, z1);
	q2 = _mm_sub_pd(q2, z2);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(w1, h4, q1);
	q2 = _mm_nmacc_pd(w2, h4, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(w2, h4));
#endif
	h5 = _mm_loaddup_pd(&hh[(ldh*4)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(v1, h5, q1);
	q2 = _mm_nmacc_pd(v2, h5, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(v2, h5));
#endif
	h6 = _mm_loaddup_pd(&hh[(ldh*5)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(t1, h6, q1);
	q2 = _mm_nmacc_pd(t2, h6, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(t2, h6));
#endif
	_mm_store_pd(&q[ldq*3],q1);
	_mm_store_pd(&q[(ldq*3)+2],q2);

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+1]);
	q1 = _mm_load_pd(&q[ldq*4]);
	q2 = _mm_load_pd(&q[(ldq*4)+2]);
	q1 = _mm_sub_pd(q1, y1);
	q2 = _mm_sub_pd(q2, y2);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(z1, h3, q1);
	q2 = _mm_nmacc_pd(z2, h3, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(z2, h3));
#endif
	h4 = _mm_loaddup_pd(&hh[(ldh*3)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(w1, h4, q1);
	q2 = _mm_nmacc_pd(w2, h4, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(w2, h4));
#endif
	h5 = _mm_loaddup_pd(&hh[(ldh*4)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(v1, h5, q1);
	q2 = _mm_nmacc_pd(v2, h5, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(v2, h5));
#endif
	h6 = _mm_loaddup_pd(&hh[(ldh*5)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(t1, h6, q1);
	q2 = _mm_nmacc_pd(t2, h6, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(t2, h6));
#endif
	_mm_store_pd(&q[ldq*4],q1);
	_mm_store_pd(&q[(ldq*4)+2],q2);

	h2 = _mm_loaddup_pd(&hh[(ldh)+1]);
	q1 = _mm_load_pd(&q[ldq*5]);
	q2 = _mm_load_pd(&q[(ldq*5)+2]);
	q1 = _mm_sub_pd(q1, x1);
	q2 = _mm_sub_pd(q2, x2);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(y1, h2, q1);
	q2 = _mm_nmacc_pd(y2, h2, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(y2, h2));
#endif
	h3 = _mm_loaddup_pd(&hh[(ldh*2)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(z1, h3, q1);
	q2 = _mm_nmacc_pd(z2, h3, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(z2, h3));
#endif
	h4 = _mm_loaddup_pd(&hh[(ldh*3)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(w1, h4, q1);
	q2 = _mm_nmacc_pd(w2, h4, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(w2, h4));
#endif
	h5 = _mm_loaddup_pd(&hh[(ldh*4)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(v1, h5, q1);
	q2 = _mm_nmacc_pd(v2, h5, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(v2, h5));
#endif
	h6 = _mm_loaddup_pd(&hh[(ldh*5)+5]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(t1, h6, q1);
	q2 = _mm_nmacc_pd(t2, h6, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(t2, h6));
#endif
	_mm_store_pd(&q[ldq*5],q1);
	_mm_store_pd(&q[(ldq*5)+2],q2);

	for (i = 6; i < nb; i++)
	{
		q1 = _mm_load_pd(&q[i*ldq]);
		q2 = _mm_load_pd(&q[(i*ldq)+2]);
		h1 = _mm_loaddup_pd(&hh[i-5]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm_nmacc_pd(x1, h1, q1);
		q2 = _mm_nmacc_pd(x2, h1, q2);
#else
		q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
		q2 = _mm_sub_pd(q2, _mm_mul_pd(x2, h1));
#endif
		h2 = _mm_loaddup_pd(&hh[ldh+i-4]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm_nmacc_pd(y1, h2, q1);
		q2 = _mm_nmacc_pd(y2, h2, q2);
#else
		q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
		q2 = _mm_sub_pd(q2, _mm_mul_pd(y2, h2));
#endif
		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-3]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm_nmacc_pd(z1, h3, q1);
		q2 = _mm_nmacc_pd(z2, h3, q2);
#else
		q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
		q2 = _mm_sub_pd(q2, _mm_mul_pd(z2, h3));
#endif
		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i-2]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm_nmacc_pd(w1, h4, q1);
		q2 = _mm_nmacc_pd(w2, h4, q2);
#else
		q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
		q2 = _mm_sub_pd(q2, _mm_mul_pd(w2, h4));
#endif
		h5 = _mm_loaddup_pd(&hh[(ldh*4)+i-1]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm_nmacc_pd(v1, h5, q1);
		q2 = _mm_nmacc_pd(v2, h5, q2);
#else
		q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
		q2 = _mm_sub_pd(q2, _mm_mul_pd(v2, h5));
#endif
		h6 = _mm_loaddup_pd(&hh[(ldh*5)+i]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm_nmacc_pd(t1, h6, q1);
		q2 = _mm_nmacc_pd(t2, h6, q2);
#else
		q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
		q2 = _mm_sub_pd(q2, _mm_mul_pd(t2, h6));
#endif
		_mm_store_pd(&q[i*ldq],q1);
		_mm_store_pd(&q[(i*ldq)+2],q2);
	}

	h1 = _mm_loaddup_pd(&hh[nb-5]);
	q1 = _mm_load_pd(&q[nb*ldq]);
	q2 = _mm_load_pd(&q[(nb*ldq)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(x1, h1, q1);
	q2 = _mm_nmacc_pd(x2, h1, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(x2, h1));
#endif
	h2 = _mm_loaddup_pd(&hh[ldh+nb-4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(y1, h2, q1);
	q2 = _mm_nmacc_pd(y2, h2, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(y2, h2));
#endif
	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(z1, h3, q1);
	q2 = _mm_nmacc_pd(z2, h3, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(z2, h3));
#endif
	h4 = _mm_loaddup_pd(&hh[(ldh*3)+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(w1, h4, q1);
	q2 = _mm_nmacc_pd(w2, h4, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(w2, h4));
#endif
	h5 = _mm_loaddup_pd(&hh[(ldh*4)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(v1, h5, q1);
	q2 = _mm_nmacc_pd(v2, h5, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(v2, h5));
#endif
	_mm_store_pd(&q[nb*ldq],q1);
	_mm_store_pd(&q[(nb*ldq)+2],q2);

	h1 = _mm_loaddup_pd(&hh[nb-4]);
	q1 = _mm_load_pd(&q[(nb+1)*ldq]);
	q2 = _mm_load_pd(&q[((nb+1)*ldq)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(x1, h1, q1);
	q2 = _mm_nmacc_pd(x2, h1, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(x2, h1));
#endif
	h2 = _mm_loaddup_pd(&hh[ldh+nb-3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(y1, h2, q1);
	q2 = _mm_nmacc_pd(y2, h2, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(y2, h2));
#endif
	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(z1, h3, q1);
	q2 = _mm_nmacc_pd(z2, h3, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(z2, h3));
#endif
	h4 = _mm_loaddup_pd(&hh[(ldh*3)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(w1, h4, q1);
	q2 = _mm_nmacc_pd(w2, h4, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(w2, h4));
#endif
	_mm_store_pd(&q[(nb+1)*ldq],q1);
	_mm_store_pd(&q[((nb+1)*ldq)+2],q2);

	h1 = _mm_loaddup_pd(&hh[nb-3]);
	q1 = _mm_load_pd(&q[(nb+2)*ldq]);
	q2 = _mm_load_pd(&q[((nb+2)*ldq)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(x1, h1, q1);
	q2 = _mm_nmacc_pd(x2, h1, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(x2, h1));
#endif
	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(y1, h2, q1);
	q2 = _mm_nmacc_pd(y2, h2, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(y2, h2));
#endif
	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(z1, h3, q1);
	q2 = _mm_nmacc_pd(z2, h3, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(z2, h3));
#endif
	_mm_store_pd(&q[(nb+2)*ldq],q1);
	_mm_store_pd(&q[((nb+2)*ldq)+2],q2);

	h1 = _mm_loaddup_pd(&hh[nb-2]);
	q1 = _mm_load_pd(&q[(nb+3)*ldq]);
	q2 = _mm_load_pd(&q[((nb+3)*ldq)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(x1, h1, q1);
	q2 = _mm_nmacc_pd(x2, h1, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(x2, h1));
#endif
	h2 = _mm_loaddup_pd(&hh[ldh+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(y1, h2, q1);
	q2 = _mm_nmacc_pd(y2, h2, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(y2, h2));
#endif
	_mm_store_pd(&q[(nb+3)*ldq],q1);
	_mm_store_pd(&q[((nb+3)*ldq)+2],q2);

	h1 = _mm_loaddup_pd(&hh[nb-1]);
	q1 = _mm_load_pd(&q[(nb+4)*ldq]);
	q2 = _mm_load_pd(&q[((nb+4)*ldq)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(x1, h1, q1);
	q2 = _mm_nmacc_pd(x2, h1, q2);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
	q2 = _mm_sub_pd(q2, _mm_mul_pd(x2, h1));
#endif
	_mm_store_pd(&q[(nb+4)*ldq],q1);
	_mm_store_pd(&q[((nb+4)*ldq)+2],q2);
}

/**
 * Unrolled kernel that computes
 * 2 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_2_SSE_6hv(double* q, double* hh, int nb, int ldq, int ldh, double* scalarprods)
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
#ifdef __ELPA_USE_FMA__
	register __m128d t1 = _mm_macc_pd(a5_1, h_6_5, a6_1);
	t1 = _mm_macc_pd(a4_1, h_6_4, t1);
	t1 = _mm_macc_pd(a3_1, h_6_3, t1);
	t1 = _mm_macc_pd(a2_1, h_6_2, t1);
	t1 = _mm_macc_pd(a1_1, h_6_1, t1);
#else
	register __m128d t1 = _mm_add_pd(a6_1, _mm_mul_pd(a5_1, h_6_5));
	t1 = _mm_add_pd(t1, _mm_mul_pd(a4_1, h_6_4));
	t1 = _mm_add_pd(t1, _mm_mul_pd(a3_1, h_6_3));
	t1 = _mm_add_pd(t1, _mm_mul_pd(a2_1, h_6_2));
	t1 = _mm_add_pd(t1, _mm_mul_pd(a1_1, h_6_1));
#endif
	__m128d h_5_4 = _mm_loaddup_pd(&hh[(ldh*4)+1]);
	__m128d h_5_3 = _mm_loaddup_pd(&hh[(ldh*4)+2]);
	__m128d h_5_2 = _mm_loaddup_pd(&hh[(ldh*4)+3]);
	__m128d h_5_1 = _mm_loaddup_pd(&hh[(ldh*4)+4]);
#ifdef __ELPA_USE_FMA__
	register __m128d v1 = _mm_macc_pd(a4_1, h_5_4, a5_1);
	v1 = _mm_macc_pd(a3_1, h_5_3, v1);
	v1 = _mm_macc_pd(a2_1, h_5_2, v1);
	v1 = _mm_macc_pd(a1_1, h_5_1, v1);
#else
	register __m128d v1 = _mm_add_pd(a5_1, _mm_mul_pd(a4_1, h_5_4));
	v1 = _mm_add_pd(v1, _mm_mul_pd(a3_1, h_5_3));
	v1 = _mm_add_pd(v1, _mm_mul_pd(a2_1, h_5_2));
	v1 = _mm_add_pd(v1, _mm_mul_pd(a1_1, h_5_1));
#endif
	__m128d h_4_3 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
	__m128d h_4_2 = _mm_loaddup_pd(&hh[(ldh*3)+2]);
	__m128d h_4_1 = _mm_loaddup_pd(&hh[(ldh*3)+3]);
#ifdef __ELPA_USE_FMA__
	register __m128d w1 = _mm_macc_pd(a3_1, h_4_3, a4_1);
	w1 = _mm_macc_pd(a2_1, h_4_2, w1);
	w1 = _mm_macc_pd(a1_1, h_4_1, w1);
#else
	register __m128d w1 = _mm_add_pd(a4_1, _mm_mul_pd(a3_1, h_4_3));
	w1 = _mm_add_pd(w1, _mm_mul_pd(a2_1, h_4_2));
	w1 = _mm_add_pd(w1, _mm_mul_pd(a1_1, h_4_1));
#endif
	__m128d h_2_1 = _mm_loaddup_pd(&hh[ldh+1]);
	__m128d h_3_2 = _mm_loaddup_pd(&hh[(ldh*2)+1]);
	__m128d h_3_1 = _mm_loaddup_pd(&hh[(ldh*2)+2]);
#ifdef __ELPA_USE_FMA__
	register __m128d z1 = _mm_macc_pd(a2_1, h_3_2, a3_1);
	z1 = _mm_macc_pd(a1_1, h_3_1, z1);
	register __m128d y1 = _mm_macc_pd(a1_1, h_2_1, a2_1);
#else
	register __m128d z1 = _mm_add_pd(a3_1, _mm_mul_pd(a2_1, h_3_2));
	z1 = _mm_add_pd(z1, _mm_mul_pd(a1_1, h_3_1));
	register __m128d y1 = _mm_add_pd(a2_1, _mm_mul_pd(a1_1, h_2_1));
#endif
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
#ifdef __ELPA_USE_FMA__
		x1 = _mm_macc_pd(q1, h1, x1);
#else
		x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
#endif
		h2 = _mm_loaddup_pd(&hh[ldh+i-4]);
#ifdef __ELPA_USE_FMA__
		y1 = _mm_macc_pd(q1, h2, y1);
#else
		y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
#endif
		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-3]);
#ifdef __ELPA_USE_FMA__
		z1 = _mm_macc_pd(q1, h3, z1);
#else
		z1 = _mm_add_pd(z1, _mm_mul_pd(q1,h3));
#endif
		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i-2]);
#ifdef __ELPA_USE_FMA__
		w1 = _mm_macc_pd(q1, h4, w1);
#else
		w1 = _mm_add_pd(w1, _mm_mul_pd(q1,h4));
#endif
		h5 = _mm_loaddup_pd(&hh[(ldh*4)+i-1]);
#ifdef __ELPA_USE_FMA__
		v1 = _mm_macc_pd(q1, h5, v1);
#else
		v1 = _mm_add_pd(v1, _mm_mul_pd(q1,h5));
#endif
		h6 = _mm_loaddup_pd(&hh[(ldh*5)+i]);
#ifdef __ELPA_USE_FMA__
		t1 = _mm_macc_pd(q1, h6, t1);
#else
		t1 = _mm_add_pd(t1, _mm_mul_pd(q1,h6));
#endif
	}

	h1 = _mm_loaddup_pd(&hh[nb-5]);
	q1 = _mm_load_pd(&q[nb*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm_macc_pd(q1, h1, x1);
#else
	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
#endif
	h2 = _mm_loaddup_pd(&hh[ldh+nb-4]);
#ifdef __ELPA_USE_FMA__
	y1 = _mm_macc_pd(q1, h2, y1);
#else
	y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
#endif
	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-3]);
#ifdef __ELPA_USE_FMA__
	z1 = _mm_macc_pd(q1, h3, z1);
#else
	z1 = _mm_add_pd(z1, _mm_mul_pd(q1,h3));
#endif
	h4 = _mm_loaddup_pd(&hh[(ldh*3)+nb-2]);
#ifdef __ELPA_USE_FMA__
	w1 = _mm_macc_pd(q1, h4, w1);
#else
	w1 = _mm_add_pd(w1, _mm_mul_pd(q1,h4));
#endif
	h5 = _mm_loaddup_pd(&hh[(ldh*4)+nb-1]);
#ifdef __ELPA_USE_FMA__
	v1 = _mm_macc_pd(q1, h5, v1);
#else
	v1 = _mm_add_pd(v1, _mm_mul_pd(q1,h5));
#endif

	h1 = _mm_loaddup_pd(&hh[nb-4]);
	q1 = _mm_load_pd(&q[(nb+1)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm_macc_pd(q1, h1, x1);
#else
	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
#endif
	h2 = _mm_loaddup_pd(&hh[ldh+nb-3]);
#ifdef __ELPA_USE_FMA__
	y1 = _mm_macc_pd(q1, h2, y1);
#else
	y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
#endif
	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-2]);
#ifdef __ELPA_USE_FMA__
	z1 = _mm_macc_pd(q1, h3, z1);
#else
	z1 = _mm_add_pd(z1, _mm_mul_pd(q1,h3));
#endif
	h4 = _mm_loaddup_pd(&hh[(ldh*3)+nb-1]);
#ifdef __ELPA_USE_FMA__
	w1 = _mm_macc_pd(q1, h4, w1);
#else
	w1 = _mm_add_pd(w1, _mm_mul_pd(q1,h4));
#endif

	h1 = _mm_loaddup_pd(&hh[nb-3]);
	q1 = _mm_load_pd(&q[(nb+2)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm_macc_pd(q1, h1, x1);
#else
	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
#endif
	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);
#ifdef __ELPA_USE_FMA__
	y1 = _mm_macc_pd(q1, h2, y1);
#else
	y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
#endif
	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);
#ifdef __ELPA_USE_FMA__
	z1 = _mm_macc_pd(q1, h3, z1);
#else
	z1 = _mm_add_pd(z1, _mm_mul_pd(q1,h3));
#endif

	h1 = _mm_loaddup_pd(&hh[nb-2]);
	q1 = _mm_load_pd(&q[(nb+3)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm_macc_pd(q1, h1, x1);
#else
	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
#endif
	h2 = _mm_loaddup_pd(&hh[ldh+nb-1]);
#ifdef __ELPA_USE_FMA__
	y1 = _mm_macc_pd(q1, h2, y1);
#else
	y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
#endif

	h1 = _mm_loaddup_pd(&hh[nb-1]);
	q1 = _mm_load_pd(&q[(nb+4)*ldq]);
#ifdef __ELPA_USE_FMA__
	x1 = _mm_macc_pd(q1, h1, x1);
#else
	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
#endif

	/////////////////////////////////////////////////////
	// Apply tau, correct wrong calculation using pre-calculated scalar products
	/////////////////////////////////////////////////////

	__m128d tau1 = _mm_loaddup_pd(&hh[0]);
	x1 = _mm_mul_pd(x1, tau1);

	__m128d tau2 = _mm_loaddup_pd(&hh[ldh]);
	__m128d vs_1_2 = _mm_loaddup_pd(&scalarprods[0]);
	h2 = _mm_mul_pd(tau2, vs_1_2);
#ifdef __ELPA_USE_FMA__
	y1 = _mm_msub_pd(y1, tau2, _mm_mul_pd(x1,h2));
#else
	y1 = _mm_sub_pd(_mm_mul_pd(y1,tau2), _mm_mul_pd(x1,h2));
#endif

	__m128d tau3 = _mm_loaddup_pd(&hh[ldh*2]);
	__m128d vs_1_3 = _mm_loaddup_pd(&scalarprods[1]);
	__m128d vs_2_3 = _mm_loaddup_pd(&scalarprods[2]);
	h2 = _mm_mul_pd(tau3, vs_1_3);
	h3 = _mm_mul_pd(tau3, vs_2_3);
#ifdef __ELPA_USE_FMA__
	z1 = _mm_msub_pd(z1, tau3, _mm_macc_pd(y1, h3, _mm_mul_pd(x1,h2)));
#else
	z1 = _mm_sub_pd(_mm_mul_pd(z1,tau3), _mm_add_pd(_mm_mul_pd(y1,h3), _mm_mul_pd(x1,h2)));
#endif

	__m128d tau4 = _mm_loaddup_pd(&hh[ldh*3]);
	__m128d vs_1_4 = _mm_loaddup_pd(&scalarprods[3]);
	__m128d vs_2_4 = _mm_loaddup_pd(&scalarprods[4]);
	h2 = _mm_mul_pd(tau4, vs_1_4);
	h3 = _mm_mul_pd(tau4, vs_2_4);
	__m128d vs_3_4 = _mm_loaddup_pd(&scalarprods[5]);
	h4 = _mm_mul_pd(tau4, vs_3_4);
#ifdef __ELPA_USE_FMA__
	w1 = _mm_msub_pd(w1, tau4, _mm_macc_pd(z1, h4, _mm_macc_pd(y1, h3, _mm_mul_pd(x1,h2))));
#else
	w1 = _mm_sub_pd(_mm_mul_pd(w1,tau4), _mm_add_pd(_mm_mul_pd(z1,h4), _mm_add_pd(_mm_mul_pd(y1,h3), _mm_mul_pd(x1,h2))));
#endif

	__m128d tau5 = _mm_loaddup_pd(&hh[ldh*4]);
	__m128d vs_1_5 = _mm_loaddup_pd(&scalarprods[6]);
	__m128d vs_2_5 = _mm_loaddup_pd(&scalarprods[7]);
	h2 = _mm_mul_pd(tau5, vs_1_5);
	h3 = _mm_mul_pd(tau5, vs_2_5);
	__m128d vs_3_5 = _mm_loaddup_pd(&scalarprods[8]);
	__m128d vs_4_5 = _mm_loaddup_pd(&scalarprods[9]);
	h4 = _mm_mul_pd(tau5, vs_3_5);
	h5 = _mm_mul_pd(tau5, vs_4_5);
#ifdef __ELPA_USE_FMA__
	v1 = _mm_msub_pd(v1, tau5, _mm_add_pd(_mm_macc_pd(w1, h5, _mm_mul_pd(z1,h4)), _mm_macc_pd(y1, h3, _mm_mul_pd(x1,h2))));
#else
	v1 = _mm_sub_pd(_mm_mul_pd(v1,tau5), _mm_add_pd(_mm_add_pd(_mm_mul_pd(w1,h5), _mm_mul_pd(z1,h4)), _mm_add_pd(_mm_mul_pd(y1,h3), _mm_mul_pd(x1,h2))));
#endif

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
#ifdef __ELPA_USE_FMA__
	t1 = _mm_msub_pd(t1, tau6, _mm_macc_pd(v1, h6, _mm_add_pd(_mm_macc_pd(w1, h5, _mm_mul_pd(z1,h4)), _mm_macc_pd(y1, h3, _mm_mul_pd(x1,h2)))));
#else
	t1 = _mm_sub_pd(_mm_mul_pd(t1,tau6), _mm_add_pd( _mm_mul_pd(v1,h6), _mm_add_pd(_mm_add_pd(_mm_mul_pd(w1,h5), _mm_mul_pd(z1,h4)), _mm_add_pd(_mm_mul_pd(y1,h3), _mm_mul_pd(x1,h2)))));
#endif

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [2 x nb+3]
	/////////////////////////////////////////////////////

	q1 = _mm_load_pd(&q[0]);
	q1 = _mm_sub_pd(q1, t1);
	_mm_store_pd(&q[0],q1);

	h6 = _mm_loaddup_pd(&hh[(ldh*5)+1]);
	q1 = _mm_load_pd(&q[ldq]);
	q1 = _mm_sub_pd(q1, v1);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(t1, h6, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
#endif
	_mm_store_pd(&q[ldq],q1);

	h5 = _mm_loaddup_pd(&hh[(ldh*4)+1]);
	q1 = _mm_load_pd(&q[ldq*2]);
	q1 = _mm_sub_pd(q1, w1);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(v1, h5, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
#endif
	h6 = _mm_loaddup_pd(&hh[(ldh*5)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(t1, h6, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
#endif
	_mm_store_pd(&q[ldq*2],q1);

	h4 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
	q1 = _mm_load_pd(&q[ldq*3]);
	q1 = _mm_sub_pd(q1, z1);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(w1, h4, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
#endif
	h5 = _mm_loaddup_pd(&hh[(ldh*4)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(v1, h5, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
#endif
	h6 = _mm_loaddup_pd(&hh[(ldh*5)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(t1, h6, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
#endif
	_mm_store_pd(&q[ldq*3],q1);

	h3 = _mm_loaddup_pd(&hh[(ldh*2)+1]);
	q1 = _mm_load_pd(&q[ldq*4]);
	q1 = _mm_sub_pd(q1, y1);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(z1, h3, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
#endif
	h4 = _mm_loaddup_pd(&hh[(ldh*3)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(w1, h4, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
#endif
	h5 = _mm_loaddup_pd(&hh[(ldh*4)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(v1, h5, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
#endif
	h6 = _mm_loaddup_pd(&hh[(ldh*5)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(t1, h6, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
#endif
	_mm_store_pd(&q[ldq*4],q1);

	h2 = _mm_loaddup_pd(&hh[(ldh)+1]);
	q1 = _mm_load_pd(&q[ldq*5]);
	q1 = _mm_sub_pd(q1, x1);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(y1, h2, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
#endif
	h3 = _mm_loaddup_pd(&hh[(ldh*2)+2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(z1, h3, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
#endif
	h4 = _mm_loaddup_pd(&hh[(ldh*3)+3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(w1, h4, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
#endif
	h5 = _mm_loaddup_pd(&hh[(ldh*4)+4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(v1, h5, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
#endif
	h6 = _mm_loaddup_pd(&hh[(ldh*5)+5]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(t1, h6, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
#endif
	_mm_store_pd(&q[ldq*5],q1);

	for (i = 6; i < nb; i++)
	{
		q1 = _mm_load_pd(&q[i*ldq]);
		h1 = _mm_loaddup_pd(&hh[i-5]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm_nmacc_pd(x1, h1, q1);
#else
		q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
#endif
		h2 = _mm_loaddup_pd(&hh[ldh+i-4]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm_nmacc_pd(y1, h2, q1);
#else
		q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
#endif
		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-3]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm_nmacc_pd(z1, h3, q1);
#else
		q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
#endif
		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i-2]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm_nmacc_pd(w1, h4, q1);
#else
		q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
#endif
		h5 = _mm_loaddup_pd(&hh[(ldh*4)+i-1]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm_nmacc_pd(v1, h5, q1);
#else
		q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
#endif
		h6 = _mm_loaddup_pd(&hh[(ldh*5)+i]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm_nmacc_pd(t1, h6, q1);
#else
		q1 = _mm_sub_pd(q1, _mm_mul_pd(t1, h6));
#endif
		_mm_store_pd(&q[i*ldq],q1);
	}

	h1 = _mm_loaddup_pd(&hh[nb-5]);
	q1 = _mm_load_pd(&q[nb*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(x1, h1, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
#endif
	h2 = _mm_loaddup_pd(&hh[ldh+nb-4]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(y1, h2, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
#endif
	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(z1, h3, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
#endif
	h4 = _mm_loaddup_pd(&hh[(ldh*3)+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(w1, h4, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
#endif
	h5 = _mm_loaddup_pd(&hh[(ldh*4)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(v1, h5, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(v1, h5));
#endif
	_mm_store_pd(&q[nb*ldq],q1);

	h1 = _mm_loaddup_pd(&hh[nb-4]);
	q1 = _mm_load_pd(&q[(nb+1)*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(x1, h1, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
#endif
	h2 = _mm_loaddup_pd(&hh[ldh+nb-3]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(y1, h2, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
#endif
	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(z1, h3, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
#endif
	h4 = _mm_loaddup_pd(&hh[(ldh*3)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(w1, h4, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(w1, h4));
#endif
	_mm_store_pd(&q[(nb+1)*ldq],q1);

	h1 = _mm_loaddup_pd(&hh[nb-3]);
	q1 = _mm_load_pd(&q[(nb+2)*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(x1, h1, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
#endif
	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(y1, h2, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
#endif
	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(z1, h3, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(z1, h3));
#endif
	_mm_store_pd(&q[(nb+2)*ldq],q1);

	h1 = _mm_loaddup_pd(&hh[nb-2]);
	q1 = _mm_load_pd(&q[(nb+3)*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(x1, h1, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
#endif
	h2 = _mm_loaddup_pd(&hh[ldh+nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(y1, h2, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(y1, h2));
#endif
	_mm_store_pd(&q[(nb+3)*ldq],q1);

	h1 = _mm_loaddup_pd(&hh[nb-1]);
	q1 = _mm_load_pd(&q[(nb+4)*ldq]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm_nmacc_pd(x1, h1, q1);
#else
	q1 = _mm_sub_pd(q1, _mm_mul_pd(x1, h1));
#endif
	_mm_store_pd(&q[(nb+4)*ldq],q1);
}
#endif
