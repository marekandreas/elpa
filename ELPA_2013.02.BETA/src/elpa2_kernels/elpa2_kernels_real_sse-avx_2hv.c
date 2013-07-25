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

//Forward declaration
#ifdef __AVX__
__forceinline void hh_trafo_kernel_4_AVX_2hv(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_8_AVX_2hv(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_16_AVX_2hv(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_24_AVX_2hv(double* q, double* hh, int nb, int ldq, int ldh, double s);
#else
__forceinline void hh_trafo_kernel_4_SSE_2hv(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_8_SSE_2hv(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_12_SSE_2hv(double* q, double* hh, int nb, int ldq, int ldh, double s);
#endif

void double_hh_trafo_(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#if 0
void double_hh_trafo_fast_(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif

void double_hh_trafo_(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;

	// calculating scalar product to compute
	// 2 householder vectors simultaneously
	double s = hh[(ldh)+1]*1.0;

	#pragma ivdep
	for (i = 2; i < nb; i++)
	{
		s += hh[i-1] * hh[(i+ldh)];
	}

	// Production level kernel calls with padding
#ifdef __AVX__
	for (i = 0; i < nq-20; i+=24)
	{
		hh_trafo_kernel_24_AVX_2hv(&q[i], hh, nb, ldq, ldh, s);
	}

	if (nq == i)
	{
		return;
	}

	if (nq-i == 20)
	{
		hh_trafo_kernel_16_AVX_2hv(&q[i], hh, nb, ldq, ldh, s);
		hh_trafo_kernel_4_AVX_2hv(&q[i+16], hh, nb, ldq, ldh, s);
	}
	else if (nq-i == 16)
	{
		hh_trafo_kernel_16_AVX_2hv(&q[i], hh, nb, ldq, ldh, s);
	}
	else if (nq-i == 12)
	{
		hh_trafo_kernel_8_AVX_2hv(&q[i], hh, nb, ldq, ldh, s);
		hh_trafo_kernel_4_AVX_2hv(&q[i+8], hh, nb, ldq, ldh, s);
	}
	else if (nq-i == 8)
	{
		hh_trafo_kernel_8_AVX_2hv(&q[i], hh, nb, ldq, ldh, s);
	}
	else
	{
		hh_trafo_kernel_4_AVX_2hv(&q[i], hh, nb, ldq, ldh, s);
	}
#else
	for (i = 0; i < nq-8; i+=12)
	{
		hh_trafo_kernel_12_SSE_2hv(&q[i], hh, nb, ldq, ldh, s);
	}
	if (nq == i)
	{
		return;
	}
	else
	{
		if (nq-i > 4)
		{
			hh_trafo_kernel_8_SSE_2hv(&q[i], hh, nb, ldq, ldh, s);
		}
		else if (nq-i > 0)
		{
			hh_trafo_kernel_4_SSE_2hv(&q[i], hh, nb, ldq, ldh, s);
		}
	}
#endif
}

#if 0
void double_hh_trafo_fast_(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;

	// calculating scalar product to compute
	// 2 householder vectors simultaneously
	double s = hh[(ldh)+1]*1.0;

	#pragma ivdep
	for (i = 2; i < nb; i++)
	{
		s += hh[i-1] * hh[(i+ldh)];
	}

	// Production level kernel calls with padding
#ifdef __AVX__
	for (i = 0; i < nq; i+=24)
	{
		hh_trafo_kernel_24_AVX_2hv(&q[i], hh, nb, ldq, ldh, s);
	}
#else
	for (i = 0; i < nq; i+=12)
	{
		hh_trafo_kernel_12_SSE_2hv(&q[i], hh, nb, ldq, ldh, s);
	}
#endif
}
#endif

#ifdef __AVX__
/**
 * Unrolled kernel that computes
 * 24 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_24_AVX_2hv(double* q, double* hh, int nb, int ldq, int ldh, double s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [24 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	__m256d sign = (__m256d)_mm256_set1_epi64x(0x8000000000000000);

	__m256d x1 = _mm256_load_pd(&q[ldq]);
	__m256d x2 = _mm256_load_pd(&q[ldq+4]);
	__m256d x3 = _mm256_load_pd(&q[ldq+8]);
	__m256d x4 = _mm256_load_pd(&q[ldq+12]);
	__m256d x5 = _mm256_load_pd(&q[ldq+16]);
	__m256d x6 = _mm256_load_pd(&q[ldq+20]);

	__m256d h1 = _mm256_broadcast_sd(&hh[ldh+1]);
	__m256d h2;

#ifdef __FMA4__
	__m256d q1 = _mm256_load_pd(q);
	__m256d y1 = _mm256_macc_pd(x1, h1, q1);
	__m256d q2 = _mm256_load_pd(&q[4]);
	__m256d y2 = _mm256_macc_pd(x2, h1, q2);
	__m256d q3 = _mm256_load_pd(&q[8]);
	__m256d y3 = _mm256_macc_pd(x3, h1, q3);
	__m256d q4 = _mm256_load_pd(&q[12]);
	__m256d y4 = _mm256_macc_pd(x4, h1, q4);
	__m256d q5 = _mm256_load_pd(&q[16]);
	__m256d y5 = _mm256_macc_pd(x5, h1, q5);
	__m256d q6 = _mm256_load_pd(&q[20]);
	__m256d y6 = _mm256_macc_pd(x6, h1, q6);
#else
	__m256d q1 = _mm256_load_pd(q);
	__m256d y1 = _mm256_add_pd(q1, _mm256_mul_pd(x1, h1));
	__m256d q2 = _mm256_load_pd(&q[4]);
	__m256d y2 = _mm256_add_pd(q2, _mm256_mul_pd(x2, h1));
	__m256d q3 = _mm256_load_pd(&q[8]);
	__m256d y3 = _mm256_add_pd(q3, _mm256_mul_pd(x3, h1));
	__m256d q4 = _mm256_load_pd(&q[12]);
	__m256d y4 = _mm256_add_pd(q4, _mm256_mul_pd(x4, h1));
	__m256d q5 = _mm256_load_pd(&q[16]);
	__m256d y5 = _mm256_add_pd(q5, _mm256_mul_pd(x5, h1));
	__m256d q6 = _mm256_load_pd(&q[20]);
	__m256d y6 = _mm256_add_pd(q6, _mm256_mul_pd(x6, h1));
#endif

	for(i = 2; i < nb; i++)
	{
		h1 = _mm256_broadcast_sd(&hh[i-1]);
		h2 = _mm256_broadcast_sd(&hh[ldh+i]);
#ifdef __FMA4__
		q1 = _mm256_load_pd(&q[i*ldq]);
		x1 = _mm256_macc_pd(q1, h1, x1);
		y1 = _mm256_macc_pd(q1, h2, y1);
		q2 = _mm256_load_pd(&q[(i*ldq)+4]);
		x2 = _mm256_macc_pd(q2, h1, x2);
		y2 = _mm256_macc_pd(q2, h2, y2);
		q3 = _mm256_load_pd(&q[(i*ldq)+8]);
		x3 = _mm256_macc_pd(q3, h1, x3);
		y3 = _mm256_macc_pd(q3, h2, y3);
		q4 = _mm256_load_pd(&q[(i*ldq)+12]);
		x4 = _mm256_macc_pd(q4, h1, x4);
		y4 = _mm256_macc_pd(q4, h2, y4);
		q5 = _mm256_load_pd(&q[(i*ldq)+16]);
		x5 = _mm256_macc_pd(q5, h1, x5);
		y5 = _mm256_macc_pd(q5, h2, y5);
		q6 = _mm256_load_pd(&q[(i*ldq)+20]);
		x6 = _mm256_macc_pd(q6, h1, x6);
		y6 = _mm256_macc_pd(q6, h2, y6);
#else
		q1 = _mm256_load_pd(&q[i*ldq]);
		x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
		y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
		q2 = _mm256_load_pd(&q[(i*ldq)+4]);
		x2 = _mm256_add_pd(x2, _mm256_mul_pd(q2,h1));
		y2 = _mm256_add_pd(y2, _mm256_mul_pd(q2,h2));
		q3 = _mm256_load_pd(&q[(i*ldq)+8]);
		x3 = _mm256_add_pd(x3, _mm256_mul_pd(q3,h1));
		y3 = _mm256_add_pd(y3, _mm256_mul_pd(q3,h2));
		q4 = _mm256_load_pd(&q[(i*ldq)+12]);
		x4 = _mm256_add_pd(x4, _mm256_mul_pd(q4,h1));
		y4 = _mm256_add_pd(y4, _mm256_mul_pd(q4,h2));
		q5 = _mm256_load_pd(&q[(i*ldq)+16]);
		x5 = _mm256_add_pd(x5, _mm256_mul_pd(q5,h1));
		y5 = _mm256_add_pd(y5, _mm256_mul_pd(q5,h2));
		q6 = _mm256_load_pd(&q[(i*ldq)+20]);
		x6 = _mm256_add_pd(x6, _mm256_mul_pd(q6,h1));
		y6 = _mm256_add_pd(y6, _mm256_mul_pd(q6,h2));
#endif
	}

	h1 = _mm256_broadcast_sd(&hh[nb-1]);
#ifdef __FMA4__
	q1 = _mm256_load_pd(&q[nb*ldq]);
	x1 = _mm256_macc_pd(q1, h1, x1);
	q2 = _mm256_load_pd(&q[(nb*ldq)+4]);
	x2 = _mm256_macc_pd(q2, h1, x2);
	q3 = _mm256_load_pd(&q[(nb*ldq)+8]);
	x3 = _mm256_macc_pd(q3, h1, x3);
	q4 = _mm256_load_pd(&q[(nb*ldq)+12]);
	x4 = _mm256_macc_pd(q4, h1, x4);
	q5 = _mm256_load_pd(&q[(nb*ldq)+16]);
	x5 = _mm256_macc_pd(q5, h1, x5);
	q6 = _mm256_load_pd(&q[(nb*ldq)+20]);
	x6 = _mm256_macc_pd(q6, h1, x6);
#else
	q1 = _mm256_load_pd(&q[nb*ldq]);
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
	q2 = _mm256_load_pd(&q[(nb*ldq)+4]);
	x2 = _mm256_add_pd(x2, _mm256_mul_pd(q2,h1));
	q3 = _mm256_load_pd(&q[(nb*ldq)+8]);
	x3 = _mm256_add_pd(x3, _mm256_mul_pd(q3,h1));
	q4 = _mm256_load_pd(&q[(nb*ldq)+12]);
	x4 = _mm256_add_pd(x4, _mm256_mul_pd(q4,h1));
	q5 = _mm256_load_pd(&q[(nb*ldq)+16]);
	x5 = _mm256_add_pd(x5, _mm256_mul_pd(q5,h1));
	q6 = _mm256_load_pd(&q[(nb*ldq)+20]);
	x6 = _mm256_add_pd(x6, _mm256_mul_pd(q6,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [24 x nb+1]
	/////////////////////////////////////////////////////

	__m256d tau1 = _mm256_broadcast_sd(hh);
	__m256d tau2 = _mm256_broadcast_sd(&hh[ldh]);
	__m256d vs = _mm256_broadcast_sd(&s);

	h1 = _mm256_xor_pd(tau1, sign);
	x1 = _mm256_mul_pd(x1, h1);
	x2 = _mm256_mul_pd(x2, h1);
	x3 = _mm256_mul_pd(x3, h1);
	x4 = _mm256_mul_pd(x4, h1);
	x5 = _mm256_mul_pd(x5, h1);
	x6 = _mm256_mul_pd(x6, h1);
	h1 = _mm256_xor_pd(tau2, sign);
	h2 = _mm256_mul_pd(h1, vs);
#ifdef __FMA4__
	y1 = _mm256_macc_pd(y1, h1, _mm256_mul_pd(x1,h2));
	y2 = _mm256_macc_pd(y2, h1, _mm256_mul_pd(x2,h2));
	y3 = _mm256_macc_pd(y3, h1, _mm256_mul_pd(x3,h2));
	y4 = _mm256_macc_pd(y4, h1, _mm256_mul_pd(x4,h2));
	y5 = _mm256_macc_pd(y5, h1, _mm256_mul_pd(x5,h2));
	y6 = _mm256_macc_pd(y6, h1, _mm256_mul_pd(x6,h2));
#else
	y1 = _mm256_add_pd(_mm256_mul_pd(y1,h1), _mm256_mul_pd(x1,h2));
	y2 = _mm256_add_pd(_mm256_mul_pd(y2,h1), _mm256_mul_pd(x2,h2));
	y3 = _mm256_add_pd(_mm256_mul_pd(y3,h1), _mm256_mul_pd(x3,h2));
	y4 = _mm256_add_pd(_mm256_mul_pd(y4,h1), _mm256_mul_pd(x4,h2));
	y5 = _mm256_add_pd(_mm256_mul_pd(y5,h1), _mm256_mul_pd(x5,h2));
	y6 = _mm256_add_pd(_mm256_mul_pd(y6,h1), _mm256_mul_pd(x6,h2));
#endif

	q1 = _mm256_load_pd(q);
	q1 = _mm256_add_pd(q1, y1);
	_mm256_store_pd(q,q1);
	q2 = _mm256_load_pd(&q[4]);
	q2 = _mm256_add_pd(q2, y2);
	_mm256_store_pd(&q[4],q2);
	q3 = _mm256_load_pd(&q[8]);
	q3 = _mm256_add_pd(q3, y3);
	_mm256_store_pd(&q[8],q3);
	q4 = _mm256_load_pd(&q[12]);
	q4 = _mm256_add_pd(q4, y4);
	_mm256_store_pd(&q[12],q4);
	q5 = _mm256_load_pd(&q[16]);
	q5 = _mm256_add_pd(q5, y5);
	_mm256_store_pd(&q[16],q5);
	q6 = _mm256_load_pd(&q[20]);
	q6 = _mm256_add_pd(q6, y6);
	_mm256_store_pd(&q[20],q6);

	h2 = _mm256_broadcast_sd(&hh[ldh+1]);
#ifdef __FMA4__
	q1 = _mm256_load_pd(&q[ldq]);
	q1 = _mm256_add_pd(q1, _mm256_macc_pd(y1, h2, x1));
	_mm256_store_pd(&q[ldq],q1);
	q2 = _mm256_load_pd(&q[ldq+4]);
	q2 = _mm256_add_pd(q2, _mm256_macc_pd(y2, h2, x2));
	_mm256_store_pd(&q[ldq+4],q2);
	q3 = _mm256_load_pd(&q[ldq+8]);
	q3 = _mm256_add_pd(q3, _mm256_macc_pd(y3, h2, x3));
	_mm256_store_pd(&q[ldq+8],q3);
	q4 = _mm256_load_pd(&q[ldq+12]);
	q4 = _mm256_add_pd(q4, _mm256_macc_pd(y4, h2, x4));
	_mm256_store_pd(&q[ldq+12],q4);
	q5 = _mm256_load_pd(&q[ldq+16]);
	q5 = _mm256_add_pd(q5, _mm256_macc_pd(y5, h2, x5));
	_mm256_store_pd(&q[ldq+16],q5);
	q6 = _mm256_load_pd(&q[ldq+20]);
	q6 = _mm256_add_pd(q6, _mm256_macc_pd(y6, h2, x6));
	_mm256_store_pd(&q[ldq+20],q6);
#else
	q1 = _mm256_load_pd(&q[ldq]);
	q1 = _mm256_add_pd(q1, _mm256_add_pd(x1, _mm256_mul_pd(y1, h2)));
	_mm256_store_pd(&q[ldq],q1);
	q2 = _mm256_load_pd(&q[ldq+4]);
	q2 = _mm256_add_pd(q2, _mm256_add_pd(x2, _mm256_mul_pd(y2, h2)));
	_mm256_store_pd(&q[ldq+4],q2);
	q3 = _mm256_load_pd(&q[ldq+8]);
	q3 = _mm256_add_pd(q3, _mm256_add_pd(x3, _mm256_mul_pd(y3, h2)));
	_mm256_store_pd(&q[ldq+8],q3);
	q4 = _mm256_load_pd(&q[ldq+12]);
	q4 = _mm256_add_pd(q4, _mm256_add_pd(x4, _mm256_mul_pd(y4, h2)));
	_mm256_store_pd(&q[ldq+12],q4);
	q5 = _mm256_load_pd(&q[ldq+16]);
	q5 = _mm256_add_pd(q5, _mm256_add_pd(x5, _mm256_mul_pd(y5, h2)));
	_mm256_store_pd(&q[ldq+16],q5);
	q6 = _mm256_load_pd(&q[ldq+20]);
	q6 = _mm256_add_pd(q6, _mm256_add_pd(x6, _mm256_mul_pd(y6, h2)));
	_mm256_store_pd(&q[ldq+20],q6);
#endif

	for (i = 2; i < nb; i++)
	{
		h1 = _mm256_broadcast_sd(&hh[i-1]);
		h2 = _mm256_broadcast_sd(&hh[ldh+i]);
#ifdef __FMA4__
		q1 = _mm256_load_pd(&q[i*ldq]);
		q1 = _mm256_add_pd(q1, _mm256_macc_pd(x1, h1, _mm256_mul_pd(y1, h2)));
		_mm256_store_pd(&q[i*ldq],q1);
		q2 = _mm256_load_pd(&q[(i*ldq)+4]);
		q2 = _mm256_add_pd(q2, _mm256_macc_pd(x2, h1, _mm256_mul_pd(y2, h2)));
		_mm256_store_pd(&q[(i*ldq)+4],q2);
		q3 = _mm256_load_pd(&q[(i*ldq)+8]);
		q3 = _mm256_add_pd(q3, _mm256_macc_pd(x3, h1, _mm256_mul_pd(y3, h2)));
		_mm256_store_pd(&q[(i*ldq)+8],q3);
		q4 = _mm256_load_pd(&q[(i*ldq)+12]);
		q4 = _mm256_add_pd(q4, _mm256_macc_pd(x4, h1, _mm256_mul_pd(y4, h2)));
		_mm256_store_pd(&q[(i*ldq)+12],q4);
		q5 = _mm256_load_pd(&q[(i*ldq)+16]);
		q5 = _mm256_add_pd(q5, _mm256_macc_pd(x5, h1, _mm256_mul_pd(y5, h2)));
		_mm256_store_pd(&q[(i*ldq)+16],q5);
		q6 = _mm256_load_pd(&q[(i*ldq)+20]);
		q6 = _mm256_add_pd(q6, _mm256_macc_pd(x6, h1, _mm256_mul_pd(y6, h2)));
		_mm256_store_pd(&q[(i*ldq)+20],q6);
#else
		q1 = _mm256_load_pd(&q[i*ldq]);
		q1 = _mm256_add_pd(q1, _mm256_add_pd(_mm256_mul_pd(x1,h1), _mm256_mul_pd(y1, h2)));
		_mm256_store_pd(&q[i*ldq],q1);
		q2 = _mm256_load_pd(&q[(i*ldq)+4]);
		q2 = _mm256_add_pd(q2, _mm256_add_pd(_mm256_mul_pd(x2,h1), _mm256_mul_pd(y2, h2)));
		_mm256_store_pd(&q[(i*ldq)+4],q2);
		q3 = _mm256_load_pd(&q[(i*ldq)+8]);
		q3 = _mm256_add_pd(q3, _mm256_add_pd(_mm256_mul_pd(x3,h1), _mm256_mul_pd(y3, h2)));
		_mm256_store_pd(&q[(i*ldq)+8],q3);
		q4 = _mm256_load_pd(&q[(i*ldq)+12]);
		q4 = _mm256_add_pd(q4, _mm256_add_pd(_mm256_mul_pd(x4,h1), _mm256_mul_pd(y4, h2)));
		_mm256_store_pd(&q[(i*ldq)+12],q4);
		q5 = _mm256_load_pd(&q[(i*ldq)+16]);
		q5 = _mm256_add_pd(q5, _mm256_add_pd(_mm256_mul_pd(x5,h1), _mm256_mul_pd(y5, h2)));
		_mm256_store_pd(&q[(i*ldq)+16],q5);
		q6 = _mm256_load_pd(&q[(i*ldq)+20]);
		q6 = _mm256_add_pd(q6, _mm256_add_pd(_mm256_mul_pd(x6,h1), _mm256_mul_pd(y6, h2)));
		_mm256_store_pd(&q[(i*ldq)+20],q6);
#endif
	}

	h1 = _mm256_broadcast_sd(&hh[nb-1]);
#ifdef __FMA4__
	q1 = _mm256_load_pd(&q[nb*ldq]);
	q1 = _mm256_macc_pd(x1, h1, q1);
	_mm256_store_pd(&q[nb*ldq],q1);
	q2 = _mm256_load_pd(&q[(nb*ldq)+4]);
	q2 = _mm256_macc_pd(x2, h1, q2);
	_mm256_store_pd(&q[(nb*ldq)+4],q2);
	q3 = _mm256_load_pd(&q[(nb*ldq)+8]);
	q3 = _mm256_macc_pd(x3, h1, q3);
	_mm256_store_pd(&q[(nb*ldq)+8],q3);
	q4 = _mm256_load_pd(&q[(nb*ldq)+12]);
	q4 = _mm256_macc_pd(x4, h1, q4);
	_mm256_store_pd(&q[(nb*ldq)+12],q4);
	q5 = _mm256_load_pd(&q[(nb*ldq)+16]);
	q5 = _mm256_macc_pd(x5, h1, q5);
	_mm256_store_pd(&q[(nb*ldq)+16],q5);
	q6 = _mm256_load_pd(&q[(nb*ldq)+20]);
	q6 = _mm256_macc_pd(x6, h1, q6);
	_mm256_store_pd(&q[(nb*ldq)+20],q6);
#else
	q1 = _mm256_load_pd(&q[nb*ldq]);
	q1 = _mm256_add_pd(q1, _mm256_mul_pd(x1, h1));
	_mm256_store_pd(&q[nb*ldq],q1);
	q2 = _mm256_load_pd(&q[(nb*ldq)+4]);
	q2 = _mm256_add_pd(q2, _mm256_mul_pd(x2, h1));
	_mm256_store_pd(&q[(nb*ldq)+4],q2);
	q3 = _mm256_load_pd(&q[(nb*ldq)+8]);
	q3 = _mm256_add_pd(q3, _mm256_mul_pd(x3, h1));
	_mm256_store_pd(&q[(nb*ldq)+8],q3);
	q4 = _mm256_load_pd(&q[(nb*ldq)+12]);
	q4 = _mm256_add_pd(q4, _mm256_mul_pd(x4, h1));
	_mm256_store_pd(&q[(nb*ldq)+12],q4);
	q5 = _mm256_load_pd(&q[(nb*ldq)+16]);
	q5 = _mm256_add_pd(q5, _mm256_mul_pd(x5, h1));
	_mm256_store_pd(&q[(nb*ldq)+16],q5);
	q6 = _mm256_load_pd(&q[(nb*ldq)+20]);
	q6 = _mm256_add_pd(q6, _mm256_mul_pd(x6, h1));
	_mm256_store_pd(&q[(nb*ldq)+20],q6);
#endif
}

/**
 * Unrolled kernel that computes
 * 16 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_16_AVX_2hv(double* q, double* hh, int nb, int ldq, int ldh, double s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [16 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	__m256d sign = (__m256d)_mm256_set1_epi64x(0x8000000000000000);

	__m256d x1 = _mm256_load_pd(&q[ldq]);
	__m256d x2 = _mm256_load_pd(&q[ldq+4]);
	__m256d x3 = _mm256_load_pd(&q[ldq+8]);
	__m256d x4 = _mm256_load_pd(&q[ldq+12]);

	__m256d h1 = _mm256_broadcast_sd(&hh[ldh+1]);
	__m256d h2;

#ifdef __FMA4__
	__m256d q1 = _mm256_load_pd(q);
	__m256d y1 = _mm256_macc_pd(x1, h1, q1);
	__m256d q2 = _mm256_load_pd(&q[4]);
	__m256d y2 = _mm256_macc_pd(x2, h1, q2);
	__m256d q3 = _mm256_load_pd(&q[8]);
	__m256d y3 = _mm256_macc_pd(x3, h1, q3);
	__m256d q4 = _mm256_load_pd(&q[12]);
	__m256d y4 = _mm256_macc_pd(x4, h1, q4);
#else
	__m256d q1 = _mm256_load_pd(q);
	__m256d y1 = _mm256_add_pd(q1, _mm256_mul_pd(x1, h1));
	__m256d q2 = _mm256_load_pd(&q[4]);
	__m256d y2 = _mm256_add_pd(q2, _mm256_mul_pd(x2, h1));
	__m256d q3 = _mm256_load_pd(&q[8]);
	__m256d y3 = _mm256_add_pd(q3, _mm256_mul_pd(x3, h1));
	__m256d q4 = _mm256_load_pd(&q[12]);
	__m256d y4 = _mm256_add_pd(q4, _mm256_mul_pd(x4, h1));
#endif

	for(i = 2; i < nb; i++)
	{
		h1 = _mm256_broadcast_sd(&hh[i-1]);
		h2 = _mm256_broadcast_sd(&hh[ldh+i]);
#ifdef __FMA4__
		q1 = _mm256_load_pd(&q[i*ldq]);
		x1 = _mm256_macc_pd(q1, h1, x1);
		y1 = _mm256_macc_pd(q1, h2, y1);
		q2 = _mm256_load_pd(&q[(i*ldq)+4]);
		x2 = _mm256_macc_pd(q2, h1, x2);
		y2 = _mm256_macc_pd(q2, h2, y2);
		q3 = _mm256_load_pd(&q[(i*ldq)+8]);
		x3 = _mm256_macc_pd(q3, h1, x3);
		y3 = _mm256_macc_pd(q3, h2, y3);
		q4 = _mm256_load_pd(&q[(i*ldq)+12]);
		x4 = _mm256_macc_pd(q4, h1, x4);
		y4 = _mm256_macc_pd(q4, h2, y4);
#else
		q1 = _mm256_load_pd(&q[i*ldq]);
		x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
		y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
		q2 = _mm256_load_pd(&q[(i*ldq)+4]);
		x2 = _mm256_add_pd(x2, _mm256_mul_pd(q2,h1));
		y2 = _mm256_add_pd(y2, _mm256_mul_pd(q2,h2));
		q3 = _mm256_load_pd(&q[(i*ldq)+8]);
		x3 = _mm256_add_pd(x3, _mm256_mul_pd(q3,h1));
		y3 = _mm256_add_pd(y3, _mm256_mul_pd(q3,h2));
		q4 = _mm256_load_pd(&q[(i*ldq)+12]);
		x4 = _mm256_add_pd(x4, _mm256_mul_pd(q4,h1));
		y4 = _mm256_add_pd(y4, _mm256_mul_pd(q4,h2));
#endif
	}

	h1 = _mm256_broadcast_sd(&hh[nb-1]);
#ifdef __FMA4__
	q1 = _mm256_load_pd(&q[nb*ldq]);
	x1 = _mm256_macc_pd(q1, h1, x1);
	q2 = _mm256_load_pd(&q[(nb*ldq)+4]);
	x2 = _mm256_macc_pd(q2, h1, x2);
	q3 = _mm256_load_pd(&q[(nb*ldq)+8]);
	x3 = _mm256_macc_pd(q3, h1, x3);
	q4 = _mm256_load_pd(&q[(nb*ldq)+12]);
	x4 = _mm256_macc_pd(q4, h1, x4);
#else
	q1 = _mm256_load_pd(&q[nb*ldq]);
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
	q2 = _mm256_load_pd(&q[(nb*ldq)+4]);
	x2 = _mm256_add_pd(x2, _mm256_mul_pd(q2,h1));
	q3 = _mm256_load_pd(&q[(nb*ldq)+8]);
	x3 = _mm256_add_pd(x3, _mm256_mul_pd(q3,h1));
	q4 = _mm256_load_pd(&q[(nb*ldq)+12]);
	x4 = _mm256_add_pd(x4, _mm256_mul_pd(q4,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [16 x nb+1]
	/////////////////////////////////////////////////////

	__m256d tau1 = _mm256_broadcast_sd(hh);
	__m256d tau2 = _mm256_broadcast_sd(&hh[ldh]);
	__m256d vs = _mm256_broadcast_sd(&s);

	h1 = _mm256_xor_pd(tau1, sign);
	x1 = _mm256_mul_pd(x1, h1);
	x2 = _mm256_mul_pd(x2, h1);
	x3 = _mm256_mul_pd(x3, h1);
	x4 = _mm256_mul_pd(x4, h1);
	h1 = _mm256_xor_pd(tau2, sign);
	h2 = _mm256_mul_pd(h1, vs);
#ifdef __FMA4__
	y1 = _mm256_macc_pd(y1, h1, _mm256_mul_pd(x1,h2));
	y2 = _mm256_macc_pd(y2, h1, _mm256_mul_pd(x2,h2));
	y3 = _mm256_macc_pd(y3, h1, _mm256_mul_pd(x3,h2));
	y4 = _mm256_macc_pd(y4, h1, _mm256_mul_pd(x4,h2));
#else
	y1 = _mm256_add_pd(_mm256_mul_pd(y1,h1), _mm256_mul_pd(x1,h2));
	y2 = _mm256_add_pd(_mm256_mul_pd(y2,h1), _mm256_mul_pd(x2,h2));
	y3 = _mm256_add_pd(_mm256_mul_pd(y3,h1), _mm256_mul_pd(x3,h2));
	y4 = _mm256_add_pd(_mm256_mul_pd(y4,h1), _mm256_mul_pd(x4,h2));
#endif

	q1 = _mm256_load_pd(q);
	q1 = _mm256_add_pd(q1, y1);
	_mm256_store_pd(q,q1);
	q2 = _mm256_load_pd(&q[4]);
	q2 = _mm256_add_pd(q2, y2);
	_mm256_store_pd(&q[4],q2);
	q3 = _mm256_load_pd(&q[8]);
	q3 = _mm256_add_pd(q3, y3);
	_mm256_store_pd(&q[8],q3);
	q4 = _mm256_load_pd(&q[12]);
	q4 = _mm256_add_pd(q4, y4);
	_mm256_store_pd(&q[12],q4);

	h2 = _mm256_broadcast_sd(&hh[ldh+1]);
#ifdef __FMA4__
	q1 = _mm256_load_pd(&q[ldq]);
	q1 = _mm256_add_pd(q1, _mm256_macc_pd(y1, h2, x1));
	_mm256_store_pd(&q[ldq],q1);
	q2 = _mm256_load_pd(&q[ldq+4]);
	q2 = _mm256_add_pd(q2, _mm256_macc_pd(y2, h2, x2));
	_mm256_store_pd(&q[ldq+4],q2);
	q3 = _mm256_load_pd(&q[ldq+8]);
	q3 = _mm256_add_pd(q3, _mm256_macc_pd(y3, h2, x3));
	_mm256_store_pd(&q[ldq+8],q3);
	q4 = _mm256_load_pd(&q[ldq+12]);
	q4 = _mm256_add_pd(q4, _mm256_macc_pd(y4, h2, x4));
	_mm256_store_pd(&q[ldq+12],q4);
#else
	q1 = _mm256_load_pd(&q[ldq]);
	q1 = _mm256_add_pd(q1, _mm256_add_pd(x1, _mm256_mul_pd(y1, h2)));
	_mm256_store_pd(&q[ldq],q1);
	q2 = _mm256_load_pd(&q[ldq+4]);
	q2 = _mm256_add_pd(q2, _mm256_add_pd(x2, _mm256_mul_pd(y2, h2)));
	_mm256_store_pd(&q[ldq+4],q2);
	q3 = _mm256_load_pd(&q[ldq+8]);
	q3 = _mm256_add_pd(q3, _mm256_add_pd(x3, _mm256_mul_pd(y3, h2)));
	_mm256_store_pd(&q[ldq+8],q3);
	q4 = _mm256_load_pd(&q[ldq+12]);
	q4 = _mm256_add_pd(q4, _mm256_add_pd(x4, _mm256_mul_pd(y4, h2)));
	_mm256_store_pd(&q[ldq+12],q4);
#endif

	for (i = 2; i < nb; i++)
	{
		h1 = _mm256_broadcast_sd(&hh[i-1]);
		h2 = _mm256_broadcast_sd(&hh[ldh+i]);
#ifdef __FMA4__
		q1 = _mm256_load_pd(&q[i*ldq]);
		q1 = _mm256_add_pd(q1, _mm256_macc_pd(x1, h1, _mm256_mul_pd(y1, h2)));
		_mm256_store_pd(&q[i*ldq],q1);
		q2 = _mm256_load_pd(&q[(i*ldq)+4]);
		q2 = _mm256_add_pd(q2, _mm256_macc_pd(x2, h1, _mm256_mul_pd(y2, h2)));
		_mm256_store_pd(&q[(i*ldq)+4],q2);
		q3 = _mm256_load_pd(&q[(i*ldq)+8]);
		q3 = _mm256_add_pd(q3, _mm256_macc_pd(x3, h1, _mm256_mul_pd(y3, h2)));
		_mm256_store_pd(&q[(i*ldq)+8],q3);
		q4 = _mm256_load_pd(&q[(i*ldq)+12]);
		q4 = _mm256_add_pd(q4, _mm256_macc_pd(x4, h1, _mm256_mul_pd(y4, h2)));
		_mm256_store_pd(&q[(i*ldq)+12],q4);
#else
		q1 = _mm256_load_pd(&q[i*ldq]);
		q1 = _mm256_add_pd(q1, _mm256_add_pd(_mm256_mul_pd(x1,h1), _mm256_mul_pd(y1, h2)));
		_mm256_store_pd(&q[i*ldq],q1);
		q2 = _mm256_load_pd(&q[(i*ldq)+4]);
		q2 = _mm256_add_pd(q2, _mm256_add_pd(_mm256_mul_pd(x2,h1), _mm256_mul_pd(y2, h2)));
		_mm256_store_pd(&q[(i*ldq)+4],q2);
		q3 = _mm256_load_pd(&q[(i*ldq)+8]);
		q3 = _mm256_add_pd(q3, _mm256_add_pd(_mm256_mul_pd(x3,h1), _mm256_mul_pd(y3, h2)));
		_mm256_store_pd(&q[(i*ldq)+8],q3);
		q4 = _mm256_load_pd(&q[(i*ldq)+12]);
		q4 = _mm256_add_pd(q4, _mm256_add_pd(_mm256_mul_pd(x4,h1), _mm256_mul_pd(y4, h2)));
		_mm256_store_pd(&q[(i*ldq)+12],q4);
#endif
	}

	h1 = _mm256_broadcast_sd(&hh[nb-1]);
#ifdef __FMA4__
	q1 = _mm256_load_pd(&q[nb*ldq]);
	q1 = _mm256_macc_pd(x1, h1, q1);
	_mm256_store_pd(&q[nb*ldq],q1);
	q2 = _mm256_load_pd(&q[(nb*ldq)+4]);
	q2 = _mm256_macc_pd(x2, h1, q2);
	_mm256_store_pd(&q[(nb*ldq)+4],q2);
	q3 = _mm256_load_pd(&q[(nb*ldq)+8]);
	q3 = _mm256_macc_pd(x3, h1, q3);
	_mm256_store_pd(&q[(nb*ldq)+8],q3);
	q4 = _mm256_load_pd(&q[(nb*ldq)+12]);
	q4 = _mm256_macc_pd(x4, h1, q4);
	_mm256_store_pd(&q[(nb*ldq)+12],q4);
#else
	q1 = _mm256_load_pd(&q[nb*ldq]);
	q1 = _mm256_add_pd(q1, _mm256_mul_pd(x1, h1));
	_mm256_store_pd(&q[nb*ldq],q1);
	q2 = _mm256_load_pd(&q[(nb*ldq)+4]);
	q2 = _mm256_add_pd(q2, _mm256_mul_pd(x2, h1));
	_mm256_store_pd(&q[(nb*ldq)+4],q2);
	q3 = _mm256_load_pd(&q[(nb*ldq)+8]);
	q3 = _mm256_add_pd(q3, _mm256_mul_pd(x3, h1));
	_mm256_store_pd(&q[(nb*ldq)+8],q3);
	q4 = _mm256_load_pd(&q[(nb*ldq)+12]);
	q4 = _mm256_add_pd(q4, _mm256_mul_pd(x4, h1));
	_mm256_store_pd(&q[(nb*ldq)+12],q4);
#endif
}

/**
 * Unrolled kernel that computes
 * 8 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_8_AVX_2hv(double* q, double* hh, int nb, int ldq, int ldh, double s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [8 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	__m256d sign = (__m256d)_mm256_set1_epi64x(0x8000000000000000);

	__m256d x1 = _mm256_load_pd(&q[ldq]);
	__m256d x2 = _mm256_load_pd(&q[ldq+4]);

	__m256d h1 = _mm256_broadcast_sd(&hh[ldh+1]);
	__m256d h2;

#ifdef __FMA4__
	__m256d q1 = _mm256_load_pd(q);
	__m256d y1 = _mm256_macc_pd(x1, h1, q1);
	__m256d q2 = _mm256_load_pd(&q[4]);
	__m256d y2 = _mm256_macc_pd(x2, h1, q2);
#else
	__m256d q1 = _mm256_load_pd(q);
	__m256d y1 = _mm256_add_pd(q1, _mm256_mul_pd(x1, h1));
	__m256d q2 = _mm256_load_pd(&q[4]);
	__m256d y2 = _mm256_add_pd(q2, _mm256_mul_pd(x2, h1));
#endif

	for(i = 2; i < nb; i++)
	{
		h1 = _mm256_broadcast_sd(&hh[i-1]);
		h2 = _mm256_broadcast_sd(&hh[ldh+i]);
#ifdef __FMA4__
		q1 = _mm256_load_pd(&q[i*ldq]);
		x1 = _mm256_macc_pd(q1, h1, x1);
		y1 = _mm256_macc_pd(q1, h2, y1);
		q2 = _mm256_load_pd(&q[(i*ldq)+4]);
		x2 = _mm256_macc_pd(q2, h1, x2);
		y2 = _mm256_macc_pd(q2, h2, y2);
#else
		q1 = _mm256_load_pd(&q[i*ldq]);
		x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
		y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
		q2 = _mm256_load_pd(&q[(i*ldq)+4]);
		x2 = _mm256_add_pd(x2, _mm256_mul_pd(q2,h1));
		y2 = _mm256_add_pd(y2, _mm256_mul_pd(q2,h2));
#endif
	}

	h1 = _mm256_broadcast_sd(&hh[nb-1]);
#ifdef __FMA4__
	q1 = _mm256_load_pd(&q[nb*ldq]);
	x1 = _mm256_macc_pd(q1, h1, x1);
	q2 = _mm256_load_pd(&q[(nb*ldq)+4]);
	x2 = _mm256_macc_pd(q2, h1, x2);
#else
	q1 = _mm256_load_pd(&q[nb*ldq]);
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
	q2 = _mm256_load_pd(&q[(nb*ldq)+4]);
	x2 = _mm256_add_pd(x2, _mm256_mul_pd(q2,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [8 x nb+1]
	/////////////////////////////////////////////////////

	__m256d tau1 = _mm256_broadcast_sd(hh);
	__m256d tau2 = _mm256_broadcast_sd(&hh[ldh]);
	__m256d vs = _mm256_broadcast_sd(&s);

	h1 = _mm256_xor_pd(tau1, sign);
	x1 = _mm256_mul_pd(x1, h1);
	x2 = _mm256_mul_pd(x2, h1);
	h1 = _mm256_xor_pd(tau2, sign);
	h2 = _mm256_mul_pd(h1, vs);
#ifdef __FMA4__
	y1 = _mm256_macc_pd(y1, h1, _mm256_mul_pd(x1,h2));
	y2 = _mm256_macc_pd(y2, h1, _mm256_mul_pd(x2,h2));
#else
	y1 = _mm256_add_pd(_mm256_mul_pd(y1,h1), _mm256_mul_pd(x1,h2));
	y2 = _mm256_add_pd(_mm256_mul_pd(y2,h1), _mm256_mul_pd(x2,h2));
#endif

	q1 = _mm256_load_pd(q);
	q1 = _mm256_add_pd(q1, y1);
	_mm256_store_pd(q,q1);
	q2 = _mm256_load_pd(&q[4]);
	q2 = _mm256_add_pd(q2, y2);
	_mm256_store_pd(&q[4],q2);

	h2 = _mm256_broadcast_sd(&hh[ldh+1]);
#ifdef __FMA4__
	q1 = _mm256_load_pd(&q[ldq]);
	q1 = _mm256_add_pd(q1, _mm256_macc_pd(y1, h2, x1));
	_mm256_store_pd(&q[ldq],q1);
	q2 = _mm256_load_pd(&q[ldq+4]);
	q2 = _mm256_add_pd(q2, _mm256_macc_pd(y2, h2, x2));
	_mm256_store_pd(&q[ldq+4],q2);
#else
	q1 = _mm256_load_pd(&q[ldq]);
	q1 = _mm256_add_pd(q1, _mm256_add_pd(x1, _mm256_mul_pd(y1, h2)));
	_mm256_store_pd(&q[ldq],q1);
	q2 = _mm256_load_pd(&q[ldq+4]);
	q2 = _mm256_add_pd(q2, _mm256_add_pd(x2, _mm256_mul_pd(y2, h2)));
	_mm256_store_pd(&q[ldq+4],q2);
#endif

	for (i = 2; i < nb; i++)
	{
		h1 = _mm256_broadcast_sd(&hh[i-1]);
		h2 = _mm256_broadcast_sd(&hh[ldh+i]);
#ifdef __FMA4__
		q1 = _mm256_load_pd(&q[i*ldq]);
		q1 = _mm256_add_pd(q1, _mm256_macc_pd(x1, h1, _mm256_mul_pd(y1, h2)));
		_mm256_store_pd(&q[i*ldq],q1);
		q2 = _mm256_load_pd(&q[(i*ldq)+4]);
		q2 = _mm256_add_pd(q2, _mm256_macc_pd(x2, h1, _mm256_mul_pd(y2, h2)));
		_mm256_store_pd(&q[(i*ldq)+4],q2);
#else
		q1 = _mm256_load_pd(&q[i*ldq]);
		q1 = _mm256_add_pd(q1, _mm256_add_pd(_mm256_mul_pd(x1,h1), _mm256_mul_pd(y1, h2)));
		_mm256_store_pd(&q[i*ldq],q1);
		q2 = _mm256_load_pd(&q[(i*ldq)+4]);
		q2 = _mm256_add_pd(q2, _mm256_add_pd(_mm256_mul_pd(x2,h1), _mm256_mul_pd(y2, h2)));
		_mm256_store_pd(&q[(i*ldq)+4],q2);
#endif
	}

	h1 = _mm256_broadcast_sd(&hh[nb-1]);
#ifdef __FMA4__
	q1 = _mm256_load_pd(&q[nb*ldq]);
	q1 = _mm256_macc_pd(x1, h1, q1);
	_mm256_store_pd(&q[nb*ldq],q1);
	q2 = _mm256_load_pd(&q[(nb*ldq)+4]);
	q2 = _mm256_macc_pd(x2, h1, q2);
	_mm256_store_pd(&q[(nb*ldq)+4],q2);
#else
	q1 = _mm256_load_pd(&q[nb*ldq]);
	q1 = _mm256_add_pd(q1, _mm256_mul_pd(x1, h1));
	_mm256_store_pd(&q[nb*ldq],q1);
	q2 = _mm256_load_pd(&q[(nb*ldq)+4]);
	q2 = _mm256_add_pd(q2, _mm256_mul_pd(x2, h1));
	_mm256_store_pd(&q[(nb*ldq)+4],q2);
#endif
}

/**
 * Unrolled kernel that computes
 * 4 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_4_AVX_2hv(double* q, double* hh, int nb, int ldq, int ldh, double s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	__m256d sign = (__m256d)_mm256_set1_epi64x(0x8000000000000000);

	__m256d x1 = _mm256_load_pd(&q[ldq]);

	__m256d h1 = _mm256_broadcast_sd(&hh[ldh+1]);
	__m256d h2;

#ifdef __FMA4__
	__m256d q1 = _mm256_load_pd(q);
	__m256d y1 = _mm256_macc_pd(x1, h1, q1);
#else
	__m256d q1 = _mm256_load_pd(q);
	__m256d y1 = _mm256_add_pd(q1, _mm256_mul_pd(x1, h1));
#endif

	for(i = 2; i < nb; i++)
	{
		h1 = _mm256_broadcast_sd(&hh[i-1]);
		h2 = _mm256_broadcast_sd(&hh[ldh+i]);
#ifdef __FMA4__
		q1 = _mm256_load_pd(&q[i*ldq]);
		x1 = _mm256_macc_pd(q1, h1, x1);
		y1 = _mm256_macc_pd(q1, h2, y1);
#else
		q1 = _mm256_load_pd(&q[i*ldq]);
		x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
		y1 = _mm256_add_pd(y1, _mm256_mul_pd(q1,h2));
#endif
	}

	h1 = _mm256_broadcast_sd(&hh[nb-1]);
#ifdef __FMA4__
	q1 = _mm256_load_pd(&q[nb*ldq]);
	x1 = _mm256_macc_pd(q1, h1, x1);
#else
	q1 = _mm256_load_pd(&q[nb*ldq]);
	x1 = _mm256_add_pd(x1, _mm256_mul_pd(q1,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [4 x nb+1]
	/////////////////////////////////////////////////////

	__m256d tau1 = _mm256_broadcast_sd(hh);
	__m256d tau2 = _mm256_broadcast_sd(&hh[ldh]);
	__m256d vs = _mm256_broadcast_sd(&s);

	h1 = _mm256_xor_pd(tau1, sign);
	x1 = _mm256_mul_pd(x1, h1);
	h1 = _mm256_xor_pd(tau2, sign);
	h2 = _mm256_mul_pd(h1, vs);
#ifdef __FMA4__
	y1 = _mm256_macc_pd(y1, h1, _mm256_mul_pd(x1,h2));
#else
	y1 = _mm256_add_pd(_mm256_mul_pd(y1,h1), _mm256_mul_pd(x1,h2));
#endif

	q1 = _mm256_load_pd(q);
	q1 = _mm256_add_pd(q1, y1);
	_mm256_store_pd(q,q1);

	h2 = _mm256_broadcast_sd(&hh[ldh+1]);
#ifdef __FMA4__
	q1 = _mm256_load_pd(&q[ldq]);
	q1 = _mm256_add_pd(q1, _mm256_macc_pd(y1, h2, x1));
	_mm256_store_pd(&q[ldq],q1);
#else
	q1 = _mm256_load_pd(&q[ldq]);
	q1 = _mm256_add_pd(q1, _mm256_add_pd(x1, _mm256_mul_pd(y1, h2)));
	_mm256_store_pd(&q[ldq],q1);
#endif

	for (i = 2; i < nb; i++)
	{
		h1 = _mm256_broadcast_sd(&hh[i-1]);
		h2 = _mm256_broadcast_sd(&hh[ldh+i]);
#ifdef __FMA4__
		q1 = _mm256_load_pd(&q[i*ldq]);
		q1 = _mm256_add_pd(q1, _mm256_macc_pd(x1, h1, _mm256_mul_pd(y1, h2)));
		_mm256_store_pd(&q[i*ldq],q1);
#else
		q1 = _mm256_load_pd(&q[i*ldq]);
		q1 = _mm256_add_pd(q1, _mm256_add_pd(_mm256_mul_pd(x1,h1), _mm256_mul_pd(y1, h2)));
		_mm256_store_pd(&q[i*ldq],q1);
#endif
	}

	h1 = _mm256_broadcast_sd(&hh[nb-1]);
#ifdef __FMA4__
	q1 = _mm256_load_pd(&q[nb*ldq]);
	q1 = _mm256_macc_pd(x1, h1, q1);
	_mm256_store_pd(&q[nb*ldq],q1);
#else
	q1 = _mm256_load_pd(&q[nb*ldq]);
	q1 = _mm256_add_pd(q1, _mm256_mul_pd(x1, h1));
	_mm256_store_pd(&q[nb*ldq],q1);
#endif
}
#else
/**
 * Unrolled kernel that computes
 * 12 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_12_SSE_2hv(double* q, double* hh, int nb, int ldq, int ldh, double s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [12 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	__m64 smallsign = _mm_set_pi32(0x80000000, 0x00000000);
	__m128d sign = (__m128d)_mm_set1_epi64(smallsign);

	__m128d x1 = _mm_load_pd(&q[ldq]);
	__m128d x2 = _mm_load_pd(&q[ldq+2]);
	__m128d x3 = _mm_load_pd(&q[ldq+4]);
	__m128d x4 = _mm_load_pd(&q[ldq+6]);
	__m128d x5 = _mm_load_pd(&q[ldq+8]);
	__m128d x6 = _mm_load_pd(&q[ldq+10]);

	__m128d h1 = _mm_loaddup_pd(&hh[ldh+1]);
	__m128d h2;

#ifdef __FMA4__
	__m128d q1 = _mm_load_pd(q);
	__m128d y1 = _mm_macc_pd(x1, h1, q1);
	__m128d q2 = _mm_load_pd(&q[2]);
	__m128d y2 = _mm_macc_pd(x2, h1, q2);
	__m128d q3 = _mm_load_pd(&q[4]);
	__m128d y3 = _mm_macc_pd(x3, h1, q3);
	__m128d q4 = _mm_load_pd(&q[6]);
	__m128d y4 = _mm_macc_pd(x4, h1, q4);
	__m128d q5 = _mm_load_pd(&q[8]);
	__m128d y5 = _mm_macc_pd(x5, h1, q5);
	__m128d q6 = _mm_load_pd(&q[10]);
	__m128d y6 = _mm_macc_pd(x6, h1, q6);
#else
	__m128d q1 = _mm_load_pd(q);
	__m128d y1 = _mm_add_pd(q1, _mm_mul_pd(x1, h1));
	__m128d q2 = _mm_load_pd(&q[2]);
	__m128d y2 = _mm_add_pd(q2, _mm_mul_pd(x2, h1));
	__m128d q3 = _mm_load_pd(&q[4]);
	__m128d y3 = _mm_add_pd(q3, _mm_mul_pd(x3, h1));
	__m128d q4 = _mm_load_pd(&q[6]);
	__m128d y4 = _mm_add_pd(q4, _mm_mul_pd(x4, h1));
	__m128d q5 = _mm_load_pd(&q[8]);
	__m128d y5 = _mm_add_pd(q5, _mm_mul_pd(x5, h1));
	__m128d q6 = _mm_load_pd(&q[10]);
	__m128d y6 = _mm_add_pd(q6, _mm_mul_pd(x6, h1));
#endif
	for(i = 2; i < nb; i++)
	{
		h1 = _mm_loaddup_pd(&hh[i-1]);
		h2 = _mm_loaddup_pd(&hh[ldh+i]);
#ifdef __FMA4__
		q1 = _mm_load_pd(&q[i*ldq]);
		x1 = _mm_macc_pd(q1, h1, x1);
		y1 = _mm_macc_pd(q1, h2, y1);
		q2 = _mm_load_pd(&q[(i*ldq)+2]);
		x2 = _mm_macc_pd(q2, h1, x2);
		y2 = _mm_macc_pd(q2, h2, y2);
		q3 = _mm_load_pd(&q[(i*ldq)+4]);
		x3 = _mm_macc_pd(q3, h1, x3);
		y3 = _mm_macc_pd(q3, h2, y3);
		q4 = _mm_load_pd(&q[(i*ldq)+6]);
		x4 = _mm_macc_pd(q4, h1, x4);
		y4 = _mm_macc_pd(q4, h2, y4);
		q5 = _mm_load_pd(&q[(i*ldq)+8]);
		x5 = _mm_macc_pd(q5, h1, x5);
		y5 = _mm_macc_pd(q5, h2, y5);
		q6 = _mm_load_pd(&q[(i*ldq)+10]);
		x6 = _mm_macc_pd(q6, h1, x6);
		y6 = _mm_macc_pd(q6, h2, y6);
#else
		q1 = _mm_load_pd(&q[i*ldq]);
		x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
		y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
		q2 = _mm_load_pd(&q[(i*ldq)+2]);
		x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));
		y2 = _mm_add_pd(y2, _mm_mul_pd(q2,h2));
		q3 = _mm_load_pd(&q[(i*ldq)+4]);
		x3 = _mm_add_pd(x3, _mm_mul_pd(q3,h1));
		y3 = _mm_add_pd(y3, _mm_mul_pd(q3,h2));
		q4 = _mm_load_pd(&q[(i*ldq)+6]);
		x4 = _mm_add_pd(x4, _mm_mul_pd(q4,h1));
		y4 = _mm_add_pd(y4, _mm_mul_pd(q4,h2));
		q5 = _mm_load_pd(&q[(i*ldq)+8]);
		x5 = _mm_add_pd(x5, _mm_mul_pd(q5,h1));
		y5 = _mm_add_pd(y5, _mm_mul_pd(q5,h2));
		q6 = _mm_load_pd(&q[(i*ldq)+10]);
		x6 = _mm_add_pd(x6, _mm_mul_pd(q6,h1));
		y6 = _mm_add_pd(y6, _mm_mul_pd(q6,h2));
#endif
	}

	h1 = _mm_loaddup_pd(&hh[nb-1]);
#ifdef __FMA4__
	q1 = _mm_load_pd(&q[nb*ldq]);
	x1 = _mm_macc_pd(q1, h1, x1);
	q2 = _mm_load_pd(&q[(nb*ldq)+2]);
	x2 = _mm_macc_pd(q2, h1, x2);
	q3 = _mm_load_pd(&q[(nb*ldq)+4]);
	x3 = _mm_macc_pd(q3, h1, x3);
	q4 = _mm_load_pd(&q[(nb*ldq)+6]);
	x4 = _mm_macc_pd(q4, h1, x4);
	q5 = _mm_load_pd(&q[(nb*ldq)+8]);
	x5 = _mm_macc_pd(q5, h1, x5);
	q6 = _mm_load_pd(&q[(nb*ldq)+10]);
	x6 = _mm_macc_pd(q6, h1, x6);
#else
	q1 = _mm_load_pd(&q[nb*ldq]);
	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
	q2 = _mm_load_pd(&q[(nb*ldq)+2]);
	x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));
	q3 = _mm_load_pd(&q[(nb*ldq)+4]);
	x3 = _mm_add_pd(x3, _mm_mul_pd(q3,h1));
	q4 = _mm_load_pd(&q[(nb*ldq)+6]);
	x4 = _mm_add_pd(x4, _mm_mul_pd(q4,h1));
	q5 = _mm_load_pd(&q[(nb*ldq)+8]);
	x5 = _mm_add_pd(x5, _mm_mul_pd(q5,h1));
	q6 = _mm_load_pd(&q[(nb*ldq)+10]);
	x6 = _mm_add_pd(x6, _mm_mul_pd(q6,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [12 x nb+1]
	/////////////////////////////////////////////////////

	__m128d tau1 = _mm_loaddup_pd(hh);
	__m128d tau2 = _mm_loaddup_pd(&hh[ldh]);
	__m128d vs = _mm_loaddup_pd(&s);

	h1 = _mm_xor_pd(tau1, sign);
	x1 = _mm_mul_pd(x1, h1);
	x2 = _mm_mul_pd(x2, h1);
	x3 = _mm_mul_pd(x3, h1);
	x4 = _mm_mul_pd(x4, h1);
	x5 = _mm_mul_pd(x5, h1);
	x6 = _mm_mul_pd(x6, h1);
	h1 = _mm_xor_pd(tau2, sign);
	h2 = _mm_mul_pd(h1, vs);
#ifdef __FMA4__
	y1 = _mm_macc_pd(y1, h1, _mm_mul_pd(x1,h2));
	y2 = _mm_macc_pd(y2, h1, _mm_mul_pd(x2,h2));
	y3 = _mm_macc_pd(y3, h1, _mm_mul_pd(x3,h2));
	y4 = _mm_macc_pd(y4, h1, _mm_mul_pd(x4,h2));
	y5 = _mm_macc_pd(y5, h1, _mm_mul_pd(x5,h2));
	y6 = _mm_macc_pd(y6, h1, _mm_mul_pd(x6,h2));
#else
	y1 = _mm_add_pd(_mm_mul_pd(y1,h1), _mm_mul_pd(x1,h2));
	y2 = _mm_add_pd(_mm_mul_pd(y2,h1), _mm_mul_pd(x2,h2));
	y3 = _mm_add_pd(_mm_mul_pd(y3,h1), _mm_mul_pd(x3,h2));
	y4 = _mm_add_pd(_mm_mul_pd(y4,h1), _mm_mul_pd(x4,h2));
	y5 = _mm_add_pd(_mm_mul_pd(y5,h1), _mm_mul_pd(x5,h2));
	y6 = _mm_add_pd(_mm_mul_pd(y6,h1), _mm_mul_pd(x6,h2));
#endif

	q1 = _mm_load_pd(q);
	q1 = _mm_add_pd(q1, y1);
	_mm_store_pd(q,q1);
	q2 = _mm_load_pd(&q[2]);
	q2 = _mm_add_pd(q2, y2);
	_mm_store_pd(&q[2],q2);
	q3 = _mm_load_pd(&q[4]);
	q3 = _mm_add_pd(q3, y3);
	_mm_store_pd(&q[4],q3);
	q4 = _mm_load_pd(&q[6]);
	q4 = _mm_add_pd(q4, y4);
	_mm_store_pd(&q[6],q4);
	q5 = _mm_load_pd(&q[8]);
	q5 = _mm_add_pd(q5, y5);
	_mm_store_pd(&q[8],q5);
	q6 = _mm_load_pd(&q[10]);
	q6 = _mm_add_pd(q6, y6);
	_mm_store_pd(&q[10],q6);

	h2 = _mm_loaddup_pd(&hh[ldh+1]);
#ifdef __FMA4__
	q1 = _mm_load_pd(&q[ldq]);
	q1 = _mm_add_pd(q1, _mm_macc_pd(y1, h2, x1));
	_mm_store_pd(&q[ldq],q1);
	q2 = _mm_load_pd(&q[ldq+2]);
	q2 = _mm_add_pd(q2, _mm_macc_pd(y2, h2, x2));
	_mm_store_pd(&q[ldq+2],q2);
	q3 = _mm_load_pd(&q[ldq+4]);
	q3 = _mm_add_pd(q3, _mm_macc_pd(y3, h2, x3));
	_mm_store_pd(&q[ldq+4],q3);
	q4 = _mm_load_pd(&q[ldq+6]);
	q4 = _mm_add_pd(q4, _mm_macc_pd(y4, h2, x4));
	_mm_store_pd(&q[ldq+6],q4);
	q5 = _mm_load_pd(&q[ldq+8]);
	q5 = _mm_add_pd(q5, _mm_macc_pd(y5, h2, x5));
	_mm_store_pd(&q[ldq+8],q5);
	q6 = _mm_load_pd(&q[ldq+10]);
	q6 = _mm_add_pd(q6, _mm_macc_pd(y6, h2, x6));
	_mm_store_pd(&q[ldq+10],q6);
#else
	q1 = _mm_load_pd(&q[ldq]);
	q1 = _mm_add_pd(q1, _mm_add_pd(x1, _mm_mul_pd(y1, h2)));
	_mm_store_pd(&q[ldq],q1);
	q2 = _mm_load_pd(&q[ldq+2]);
	q2 = _mm_add_pd(q2, _mm_add_pd(x2, _mm_mul_pd(y2, h2)));
	_mm_store_pd(&q[ldq+2],q2);
	q3 = _mm_load_pd(&q[ldq+4]);
	q3 = _mm_add_pd(q3, _mm_add_pd(x3, _mm_mul_pd(y3, h2)));
	_mm_store_pd(&q[ldq+4],q3);
	q4 = _mm_load_pd(&q[ldq+6]);
	q4 = _mm_add_pd(q4, _mm_add_pd(x4, _mm_mul_pd(y4, h2)));
	_mm_store_pd(&q[ldq+6],q4);
	q5 = _mm_load_pd(&q[ldq+8]);
	q5 = _mm_add_pd(q5, _mm_add_pd(x5, _mm_mul_pd(y5, h2)));
	_mm_store_pd(&q[ldq+8],q5);
	q6 = _mm_load_pd(&q[ldq+10]);
	q6 = _mm_add_pd(q6, _mm_add_pd(x6, _mm_mul_pd(y6, h2)));
	_mm_store_pd(&q[ldq+10],q6);
#endif
	for (i = 2; i < nb; i++)
	{
		h1 = _mm_loaddup_pd(&hh[i-1]);
		h2 = _mm_loaddup_pd(&hh[ldh+i]);

#ifdef __FMA4__
		q1 = _mm_load_pd(&q[i*ldq]);
		q1 = _mm_add_pd(q1, _mm_macc_pd(x1, h1, _mm_mul_pd(y1, h2)));
		_mm_store_pd(&q[i*ldq],q1);
		q2 = _mm_load_pd(&q[(i*ldq)+2]);
		q2 = _mm_add_pd(q2, _mm_macc_pd(x2, h1, _mm_mul_pd(y2, h2)));
		_mm_store_pd(&q[(i*ldq)+2],q2);
		q3 = _mm_load_pd(&q[(i*ldq)+4]);
		q3 = _mm_add_pd(q3, _mm_macc_pd(x3, h1, _mm_mul_pd(y3, h2)));
		_mm_store_pd(&q[(i*ldq)+4],q3);
		q4 = _mm_load_pd(&q[(i*ldq)+6]);
		q4 = _mm_add_pd(q4, _mm_macc_pd(x4, h1, _mm_mul_pd(y4, h2)));
		_mm_store_pd(&q[(i*ldq)+6],q4);
		q5 = _mm_load_pd(&q[(i*ldq)+8]);
		q5 = _mm_add_pd(q5, _mm_macc_pd(x5, h1, _mm_mul_pd(y5, h2)));
		_mm_store_pd(&q[(i*ldq)+8],q5);
		q6 = _mm_load_pd(&q[(i*ldq)+10]);
		q6 = _mm_add_pd(q6, _mm_macc_pd(x6, h1, _mm_mul_pd(y6, h2)));
		_mm_store_pd(&q[(i*ldq)+10],q6);
#else
		q1 = _mm_load_pd(&q[i*ldq]);
		q1 = _mm_add_pd(q1, _mm_add_pd(_mm_mul_pd(x1,h1), _mm_mul_pd(y1, h2)));
		_mm_store_pd(&q[i*ldq],q1);
		q2 = _mm_load_pd(&q[(i*ldq)+2]);
		q2 = _mm_add_pd(q2, _mm_add_pd(_mm_mul_pd(x2,h1), _mm_mul_pd(y2, h2)));
		_mm_store_pd(&q[(i*ldq)+2],q2);
		q3 = _mm_load_pd(&q[(i*ldq)+4]);
		q3 = _mm_add_pd(q3, _mm_add_pd(_mm_mul_pd(x3,h1), _mm_mul_pd(y3, h2)));
		_mm_store_pd(&q[(i*ldq)+4],q3);
		q4 = _mm_load_pd(&q[(i*ldq)+6]);
		q4 = _mm_add_pd(q4, _mm_add_pd(_mm_mul_pd(x4,h1), _mm_mul_pd(y4, h2)));
		_mm_store_pd(&q[(i*ldq)+6],q4);
		q5 = _mm_load_pd(&q[(i*ldq)+8]);
		q5 = _mm_add_pd(q5, _mm_add_pd(_mm_mul_pd(x5,h1), _mm_mul_pd(y5, h2)));
		_mm_store_pd(&q[(i*ldq)+8],q5);
		q6 = _mm_load_pd(&q[(i*ldq)+10]);
		q6 = _mm_add_pd(q6, _mm_add_pd(_mm_mul_pd(x6,h1), _mm_mul_pd(y6, h2)));
		_mm_store_pd(&q[(i*ldq)+10],q6);
#endif
	}

	h1 = _mm_loaddup_pd(&hh[nb-1]);
#ifdef __FMA4__
	q1 = _mm_load_pd(&q[nb*ldq]);
	q1 = _mm_macc_pd(x1, h1, q1);
	_mm_store_pd(&q[nb*ldq],q1);
	q2 = _mm_load_pd(&q[(nb*ldq)+2]);
	q2 = _mm_macc_pd(x2, h1, q2);
	_mm_store_pd(&q[(nb*ldq)+2],q2);
	q3 = _mm_load_pd(&q[(nb*ldq)+4]);
	q3 = _mm_macc_pd(x3, h1, q3);
	_mm_store_pd(&q[(nb*ldq)+4],q3);
	q4 = _mm_load_pd(&q[(nb*ldq)+6]);
	q4 = _mm_macc_pd(x4, h1, q4);
	_mm_store_pd(&q[(nb*ldq)+6],q4);
	q5 = _mm_load_pd(&q[(nb*ldq)+8]);
	q5 = _mm_macc_pd(x5, h1, q5);
	_mm_store_pd(&q[(nb*ldq)+8],q5);
	q6 = _mm_load_pd(&q[(nb*ldq)+10]);
	q6 = _mm_macc_pd(x6, h1, q6);
	_mm_store_pd(&q[(nb*ldq)+10],q6);
#else
	q1 = _mm_load_pd(&q[nb*ldq]);
	q1 = _mm_add_pd(q1, _mm_mul_pd(x1, h1));
	_mm_store_pd(&q[nb*ldq],q1);
	q2 = _mm_load_pd(&q[(nb*ldq)+2]);
	q2 = _mm_add_pd(q2, _mm_mul_pd(x2, h1));
	_mm_store_pd(&q[(nb*ldq)+2],q2);
	q3 = _mm_load_pd(&q[(nb*ldq)+4]);
	q3 = _mm_add_pd(q3, _mm_mul_pd(x3, h1));
	_mm_store_pd(&q[(nb*ldq)+4],q3);
	q4 = _mm_load_pd(&q[(nb*ldq)+6]);
	q4 = _mm_add_pd(q4, _mm_mul_pd(x4, h1));
	_mm_store_pd(&q[(nb*ldq)+6],q4);
	q5 = _mm_load_pd(&q[(nb*ldq)+8]);
	q5 = _mm_add_pd(q5, _mm_mul_pd(x5, h1));
	_mm_store_pd(&q[(nb*ldq)+8],q5);
	q6 = _mm_load_pd(&q[(nb*ldq)+10]);
	q6 = _mm_add_pd(q6, _mm_mul_pd(x6, h1));
	_mm_store_pd(&q[(nb*ldq)+10],q6);
#endif
}

/**
 * Unrolled kernel that computes
 * 8 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
__forceinline void hh_trafo_kernel_8_SSE_2hv(double* q, double* hh, int nb, int ldq, int ldh, double s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [8 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	__m64 smallsign = _mm_set_pi32(0x80000000, 0x00000000);
	__m128d sign = (__m128d)_mm_set1_epi64(smallsign);

	__m128d x1 = _mm_load_pd(&q[ldq]);
	__m128d x2 = _mm_load_pd(&q[ldq+2]);
	__m128d x3 = _mm_load_pd(&q[ldq+4]);
	__m128d x4 = _mm_load_pd(&q[ldq+6]);

	__m128d h1 = _mm_loaddup_pd(&hh[ldh+1]);
	__m128d h2;

#ifdef __FMA4__
	__m128d q1 = _mm_load_pd(q);
	__m128d y1 = _mm_macc_pd(x1, h1, q1);
	__m128d q2 = _mm_load_pd(&q[2]);
	__m128d y2 = _mm_macc_pd(x2, h1, q2);
	__m128d q3 = _mm_load_pd(&q[4]);
	__m128d y3 = _mm_macc_pd(x3, h1, q3);
	__m128d q4 = _mm_load_pd(&q[6]);
	__m128d y4 = _mm_macc_pd(x4, h1, q4);
#else
	__m128d q1 = _mm_load_pd(q);
	__m128d y1 = _mm_add_pd(q1, _mm_mul_pd(x1, h1));
	__m128d q2 = _mm_load_pd(&q[2]);
	__m128d y2 = _mm_add_pd(q2, _mm_mul_pd(x2, h1));
	__m128d q3 = _mm_load_pd(&q[4]);
	__m128d y3 = _mm_add_pd(q3, _mm_mul_pd(x3, h1));
	__m128d q4 = _mm_load_pd(&q[6]);
	__m128d y4 = _mm_add_pd(q4, _mm_mul_pd(x4, h1));
#endif

	for(i = 2; i < nb; i++)
	{
		h1 = _mm_loaddup_pd(&hh[i-1]);
		h2 = _mm_loaddup_pd(&hh[ldh+i]);
#ifdef __FMA4__
		q1 = _mm_load_pd(&q[i*ldq]);
		x1 = _mm_macc_pd(q1, h1, x1);
		y1 = _mm_macc_pd(q1, h2, y1);
		q2 = _mm_load_pd(&q[(i*ldq)+2]);
		x2 = _mm_macc_pd(q2, h1, x2);
		y2 = _mm_macc_pd(q2, h2, y2);
		q3 = _mm_load_pd(&q[(i*ldq)+4]);
		x3 = _mm_macc_pd(q3, h1, x3);
		y3 = _mm_macc_pd(q3, h2, y3);
		q4 = _mm_load_pd(&q[(i*ldq)+6]);
		x4 = _mm_macc_pd(q4, h1, x4);
		y4 = _mm_macc_pd(q4, h2, y4);
#else
		q1 = _mm_load_pd(&q[i*ldq]);
		x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
		y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
		q2 = _mm_load_pd(&q[(i*ldq)+2]);
		x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));
		y2 = _mm_add_pd(y2, _mm_mul_pd(q2,h2));
		q3 = _mm_load_pd(&q[(i*ldq)+4]);
		x3 = _mm_add_pd(x3, _mm_mul_pd(q3,h1));
		y3 = _mm_add_pd(y3, _mm_mul_pd(q3,h2));
		q4 = _mm_load_pd(&q[(i*ldq)+6]);
		x4 = _mm_add_pd(x4, _mm_mul_pd(q4,h1));
		y4 = _mm_add_pd(y4, _mm_mul_pd(q4,h2));
#endif
	}

	h1 = _mm_loaddup_pd(&hh[nb-1]);
#ifdef __FMA4__
	q1 = _mm_load_pd(&q[nb*ldq]);
	x1 = _mm_macc_pd(q1, h1, x1);
	q2 = _mm_load_pd(&q[(nb*ldq)+2]);
	x2 = _mm_macc_pd(q2, h1, x2);
	q3 = _mm_load_pd(&q[(nb*ldq)+4]);
	x3 = _mm_macc_pd(q3, h1, x3);
	q4 = _mm_load_pd(&q[(nb*ldq)+6]);
	x4 = _mm_macc_pd(q4, h1, x4);
#else
	q1 = _mm_load_pd(&q[nb*ldq]);
	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
	q2 = _mm_load_pd(&q[(nb*ldq)+2]);
	x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));
	q3 = _mm_load_pd(&q[(nb*ldq)+4]);
	x3 = _mm_add_pd(x3, _mm_mul_pd(q3,h1));
	q4 = _mm_load_pd(&q[(nb*ldq)+6]);
	x4 = _mm_add_pd(x4, _mm_mul_pd(q4,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [8 x nb+1]
	/////////////////////////////////////////////////////

	__m128d tau1 = _mm_loaddup_pd(hh);
	__m128d tau2 = _mm_loaddup_pd(&hh[ldh]);
	__m128d vs = _mm_loaddup_pd(&s);

	h1 = _mm_xor_pd(tau1, sign);
	x1 = _mm_mul_pd(x1, h1);
	x2 = _mm_mul_pd(x2, h1);
	x3 = _mm_mul_pd(x3, h1);
	x4 = _mm_mul_pd(x4, h1);
	h1 = _mm_xor_pd(tau2, sign);
	h2 = _mm_mul_pd(h1, vs);
#ifdef __FMA4__
	y1 = _mm_macc_pd(y1, h1, _mm_mul_pd(x1,h2));
	y2 = _mm_macc_pd(y2, h1, _mm_mul_pd(x2,h2));
	y3 = _mm_macc_pd(y3, h1, _mm_mul_pd(x3,h2));
	y4 = _mm_macc_pd(y4, h1, _mm_mul_pd(x4,h2));
#else
	y1 = _mm_add_pd(_mm_mul_pd(y1,h1), _mm_mul_pd(x1,h2));
	y2 = _mm_add_pd(_mm_mul_pd(y2,h1), _mm_mul_pd(x2,h2));
	y3 = _mm_add_pd(_mm_mul_pd(y3,h1), _mm_mul_pd(x3,h2));
	y4 = _mm_add_pd(_mm_mul_pd(y4,h1), _mm_mul_pd(x4,h2));
#endif

	q1 = _mm_load_pd(q);
	q1 = _mm_add_pd(q1, y1);
	_mm_store_pd(q,q1);
	q2 = _mm_load_pd(&q[2]);
	q2 = _mm_add_pd(q2, y2);
	_mm_store_pd(&q[2],q2);
	q3 = _mm_load_pd(&q[4]);
	q3 = _mm_add_pd(q3, y3);
	_mm_store_pd(&q[4],q3);
	q4 = _mm_load_pd(&q[6]);
	q4 = _mm_add_pd(q4, y4);
	_mm_store_pd(&q[6],q4);

	h2 = _mm_loaddup_pd(&hh[ldh+1]);
#ifdef __FMA4__
	q1 = _mm_load_pd(&q[ldq]);
	q1 = _mm_add_pd(q1, _mm_macc_pd(y1, h2, x1));
	_mm_store_pd(&q[ldq],q1);
	q2 = _mm_load_pd(&q[ldq+2]);
	q2 = _mm_add_pd(q2, _mm_macc_pd(y2, h2, x2));
	_mm_store_pd(&q[ldq+2],q2);
	q3 = _mm_load_pd(&q[ldq+4]);
	q3 = _mm_add_pd(q3, _mm_macc_pd(y3, h2, x3));
	_mm_store_pd(&q[ldq+4],q3);
	q4 = _mm_load_pd(&q[ldq+6]);
	q4 = _mm_add_pd(q4, _mm_macc_pd(y4, h2, x4));
	_mm_store_pd(&q[ldq+6],q4);
#else
	q1 = _mm_load_pd(&q[ldq]);
	q1 = _mm_add_pd(q1, _mm_add_pd(x1, _mm_mul_pd(y1, h2)));
	_mm_store_pd(&q[ldq],q1);
	q2 = _mm_load_pd(&q[ldq+2]);
	q2 = _mm_add_pd(q2, _mm_add_pd(x2, _mm_mul_pd(y2, h2)));
	_mm_store_pd(&q[ldq+2],q2);
	q3 = _mm_load_pd(&q[ldq+4]);
	q3 = _mm_add_pd(q3, _mm_add_pd(x3, _mm_mul_pd(y3, h2)));
	_mm_store_pd(&q[ldq+4],q3);
	q4 = _mm_load_pd(&q[ldq+6]);
	q4 = _mm_add_pd(q4, _mm_add_pd(x4, _mm_mul_pd(y4, h2)));
	_mm_store_pd(&q[ldq+6],q4);
#endif

	for (i = 2; i < nb; i++)
	{
		h1 = _mm_loaddup_pd(&hh[i-1]);
		h2 = _mm_loaddup_pd(&hh[ldh+i]);

#ifdef __FMA4__
		q1 = _mm_load_pd(&q[i*ldq]);
		q1 = _mm_add_pd(q1, _mm_macc_pd(x1, h1, _mm_mul_pd(y1, h2)));
		_mm_store_pd(&q[i*ldq],q1);
		q2 = _mm_load_pd(&q[(i*ldq)+2]);
		q2 = _mm_add_pd(q2, _mm_macc_pd(x2, h1, _mm_mul_pd(y2, h2)));
		_mm_store_pd(&q[(i*ldq)+2],q2);
		q3 = _mm_load_pd(&q[(i*ldq)+4]);
		q3 = _mm_add_pd(q3, _mm_macc_pd(x3, h1, _mm_mul_pd(y3, h2)));
		_mm_store_pd(&q[(i*ldq)+4],q3);
		q4 = _mm_load_pd(&q[(i*ldq)+6]);
		q4 = _mm_add_pd(q4, _mm_macc_pd(x4, h1, _mm_mul_pd(y4, h2)));
		_mm_store_pd(&q[(i*ldq)+6],q4);
#else
		q1 = _mm_load_pd(&q[i*ldq]);
		q1 = _mm_add_pd(q1, _mm_add_pd(_mm_mul_pd(x1,h1), _mm_mul_pd(y1, h2)));
		_mm_store_pd(&q[i*ldq],q1);
		q2 = _mm_load_pd(&q[(i*ldq)+2]);
		q2 = _mm_add_pd(q2, _mm_add_pd(_mm_mul_pd(x2,h1), _mm_mul_pd(y2, h2)));
		_mm_store_pd(&q[(i*ldq)+2],q2);
		q3 = _mm_load_pd(&q[(i*ldq)+4]);
		q3 = _mm_add_pd(q3, _mm_add_pd(_mm_mul_pd(x3,h1), _mm_mul_pd(y3, h2)));
		_mm_store_pd(&q[(i*ldq)+4],q3);
		q4 = _mm_load_pd(&q[(i*ldq)+6]);
		q4 = _mm_add_pd(q4, _mm_add_pd(_mm_mul_pd(x4,h1), _mm_mul_pd(y4, h2)));
		_mm_store_pd(&q[(i*ldq)+6],q4);
#endif
	}

	h1 = _mm_loaddup_pd(&hh[nb-1]);
#ifdef __FMA4__
	q1 = _mm_load_pd(&q[nb*ldq]);
	q1 = _mm_macc_pd(x1, h1, q1);
	_mm_store_pd(&q[nb*ldq],q1);
	q2 = _mm_load_pd(&q[(nb*ldq)+2]);
	q2 = _mm_macc_pd(x2, h1, q2);
	_mm_store_pd(&q[(nb*ldq)+2],q2);
	q3 = _mm_load_pd(&q[(nb*ldq)+4]);
	q3 = _mm_macc_pd(x3, h1, q3);
	_mm_store_pd(&q[(nb*ldq)+4],q3);
	q4 = _mm_load_pd(&q[(nb*ldq)+6]);
	q4 = _mm_macc_pd(x4, h1, q4);
	_mm_store_pd(&q[(nb*ldq)+6],q4);
#else
	q1 = _mm_load_pd(&q[nb*ldq]);
	q1 = _mm_add_pd(q1, _mm_mul_pd(x1, h1));
	_mm_store_pd(&q[nb*ldq],q1);
	q2 = _mm_load_pd(&q[(nb*ldq)+2]);
	q2 = _mm_add_pd(q2, _mm_mul_pd(x2, h1));
	_mm_store_pd(&q[(nb*ldq)+2],q2);
	q3 = _mm_load_pd(&q[(nb*ldq)+4]);
	q3 = _mm_add_pd(q3, _mm_mul_pd(x3, h1));
	_mm_store_pd(&q[(nb*ldq)+4],q3);
	q4 = _mm_load_pd(&q[(nb*ldq)+6]);
	q4 = _mm_add_pd(q4, _mm_mul_pd(x4, h1));
	_mm_store_pd(&q[(nb*ldq)+6],q4);
#endif
}

/**
 * Unrolled kernel that computes
 * 4 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
__forceinline void hh_trafo_kernel_4_SSE_2hv(double* q, double* hh, int nb, int ldq, int ldh, double s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	__m64 smallsign = _mm_set_pi32(0x80000000, 0x00000000);
	__m128d sign = (__m128d)_mm_set1_epi64(smallsign);

	__m128d x1 = _mm_load_pd(&q[ldq]);
	__m128d x2 = _mm_load_pd(&q[ldq+2]);

	__m128d h1 = _mm_loaddup_pd(&hh[ldh+1]);
	__m128d h2;

#ifdef __FMA4__
	__m128d q1 = _mm_load_pd(q);
	__m128d y1 = _mm_macc_pd(x1, h1, q1);
	__m128d q2 = _mm_load_pd(&q[2]);
	__m128d y2 = _mm_macc_pd(x2, h1, q2);
#else
	__m128d q1 = _mm_load_pd(q);
	__m128d y1 = _mm_add_pd(q1, _mm_mul_pd(x1, h1));
	__m128d q2 = _mm_load_pd(&q[2]);
	__m128d y2 = _mm_add_pd(q2, _mm_mul_pd(x2, h1));
#endif

	for(i = 2; i < nb; i++)
	{
		h1 = _mm_loaddup_pd(&hh[i-1]);
		h2 = _mm_loaddup_pd(&hh[ldh+i]);
#ifdef __FMA4__
		q1 = _mm_load_pd(&q[i*ldq]);
		x1 = _mm_macc_pd(q1, h1, x1);
		y1 = _mm_macc_pd(q1, h2, y1);
		q2 = _mm_load_pd(&q[(i*ldq)+2]);
		x2 = _mm_macc_pd(q2, h1, x2);
		y2 = _mm_macc_pd(q2, h2, y2);
#else
		q1 = _mm_load_pd(&q[i*ldq]);
		x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
		y1 = _mm_add_pd(y1, _mm_mul_pd(q1,h2));
		q2 = _mm_load_pd(&q[(i*ldq)+2]);
		x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));
		y2 = _mm_add_pd(y2, _mm_mul_pd(q2,h2));
#endif
	}

	h1 = _mm_loaddup_pd(&hh[nb-1]);
#ifdef __FMA4__
	q1 = _mm_load_pd(&q[nb*ldq]);
	x1 = _mm_macc_pd(q1, h1, x1);
	q2 = _mm_load_pd(&q[(nb*ldq)+2]);
	x2 = _mm_macc_pd(q2, h1, x2);
#else
	q1 = _mm_load_pd(&q[nb*ldq]);
	x1 = _mm_add_pd(x1, _mm_mul_pd(q1,h1));
	q2 = _mm_load_pd(&q[(nb*ldq)+2]);
	x2 = _mm_add_pd(x2, _mm_mul_pd(q2,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [12 x nb+1]
	/////////////////////////////////////////////////////

	__m128d tau1 = _mm_loaddup_pd(hh);
	__m128d tau2 = _mm_loaddup_pd(&hh[ldh]);
	__m128d vs = _mm_loaddup_pd(&s);

	h1 = _mm_xor_pd(tau1, sign);
	x1 = _mm_mul_pd(x1, h1);
	x2 = _mm_mul_pd(x2, h1);
	h1 = _mm_xor_pd(tau2, sign);
	h2 = _mm_mul_pd(h1, vs);
#ifdef __FMA4__
	y1 = _mm_macc_pd(y1, h1, _mm_mul_pd(x1,h2));
	y2 = _mm_macc_pd(y2, h1, _mm_mul_pd(x2,h2));
#else
	y1 = _mm_add_pd(_mm_mul_pd(y1,h1), _mm_mul_pd(x1,h2));
	y2 = _mm_add_pd(_mm_mul_pd(y2,h1), _mm_mul_pd(x2,h2));
#endif

	q1 = _mm_load_pd(q);
	q1 = _mm_add_pd(q1, y1);
	_mm_store_pd(q,q1);
	q2 = _mm_load_pd(&q[2]);
	q2 = _mm_add_pd(q2, y2);
	_mm_store_pd(&q[2],q2);

	h2 = _mm_loaddup_pd(&hh[ldh+1]);
#ifdef __FMA4__
	q1 = _mm_load_pd(&q[ldq]);
	q1 = _mm_add_pd(q1, _mm_macc_pd(y1, h2, x1));
	_mm_store_pd(&q[ldq],q1);
	q2 = _mm_load_pd(&q[ldq+2]);
	q2 = _mm_add_pd(q2, _mm_macc_pd(y2, h2, x2));
	_mm_store_pd(&q[ldq+2],q2);
#else
	q1 = _mm_load_pd(&q[ldq]);
	q1 = _mm_add_pd(q1, _mm_add_pd(x1, _mm_mul_pd(y1, h2)));
	_mm_store_pd(&q[ldq],q1);
	q2 = _mm_load_pd(&q[ldq+2]);
	q2 = _mm_add_pd(q2, _mm_add_pd(x2, _mm_mul_pd(y2, h2)));
	_mm_store_pd(&q[ldq+2],q2);
#endif

	for (i = 2; i < nb; i++)
	{
		h1 = _mm_loaddup_pd(&hh[i-1]);
		h2 = _mm_loaddup_pd(&hh[ldh+i]);

#ifdef __FMA4__
		q1 = _mm_load_pd(&q[i*ldq]);
		q1 = _mm_add_pd(q1, _mm_macc_pd(x1, h1, _mm_mul_pd(y1, h2)));
		_mm_store_pd(&q[i*ldq],q1);
		q2 = _mm_load_pd(&q[(i*ldq)+2]);
		q2 = _mm_add_pd(q2, _mm_macc_pd(x2, h1, _mm_mul_pd(y2, h2)));
		_mm_store_pd(&q[(i*ldq)+2],q2);
#else
		q1 = _mm_load_pd(&q[i*ldq]);
		q1 = _mm_add_pd(q1, _mm_add_pd(_mm_mul_pd(x1,h1), _mm_mul_pd(y1, h2)));
		_mm_store_pd(&q[i*ldq],q1);
		q2 = _mm_load_pd(&q[(i*ldq)+2]);
		q2 = _mm_add_pd(q2, _mm_add_pd(_mm_mul_pd(x2,h1), _mm_mul_pd(y2, h2)));
		_mm_store_pd(&q[(i*ldq)+2],q2);
#endif
	}

	h1 = _mm_loaddup_pd(&hh[nb-1]);
#ifdef __FMA4__
	q1 = _mm_load_pd(&q[nb*ldq]);
	q1 = _mm_macc_pd(x1, h1, q1);
	_mm_store_pd(&q[nb*ldq],q1);
	q2 = _mm_load_pd(&q[(nb*ldq)+2]);
	q2 = _mm_macc_pd(x2, h1, q2);
	_mm_store_pd(&q[(nb*ldq)+2],q2);
#else
	q1 = _mm_load_pd(&q[nb*ldq]);
	q1 = _mm_add_pd(q1, _mm_mul_pd(x1, h1));
	_mm_store_pd(&q[nb*ldq],q1);
	q2 = _mm_load_pd(&q[(nb*ldq)+2]);
	q2 = _mm_add_pd(q2, _mm_mul_pd(x2, h1));
	_mm_store_pd(&q[(nb*ldq)+2],q2);
#endif
}
#endif
