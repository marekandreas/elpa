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
// Author: Andreas Marek (andreas.marek@mpcdf.mpg.de)
// --------------------------------------------------------------------------------------------------

#include "config-f90.h"

#include <x86intrin.h>

#define __forceinline __attribute__((always_inline)) static

#ifdef HAVE_AVX512
#define __ELPA_USE_FMA__
#define _mm512_FMA_ps(a,b,c) _mm512_fmadd_ps(a,b,c)
#endif


//Forward declaration
//__forceinline void hh_trafo_kernel_8_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_16_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
//__forceinline void hh_trafo_kernel_24_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_32_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_48_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_64_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);

void double_hh_trafo_real_avx512_2hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);
/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine double_hh_trafo_real_avx512_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_real_avx512_2hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

void double_hh_trafo_real_avx512_2hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;

	// calculating scalar product to compute
	// 2 householder vectors simultaneously
	float s = hh[(ldh)+1]*1.0;

	#pragma ivdep
	for (i = 2; i < nb; i++)
	{
		s += hh[i-1] * hh[(i+ldh)];
	}

	// Production level kernel calls with padding
	for (i = 0; i < nq-48; i+=64)
	{
		hh_trafo_kernel_64_AVX512_2hv_single(&q[i], hh, nb, ldq, ldh, s);
	}

	if (nq == i)
	{
		return;
	}

	if (nq-i == 48)
	{
		hh_trafo_kernel_48_AVX512_2hv_single(&q[i], hh, nb, ldq, ldh, s);
	}
	else if (nq-i == 32)
	{
		hh_trafo_kernel_32_AVX512_2hv_single(&q[i], hh, nb, ldq, ldh, s);
	}

	else
	{
		hh_trafo_kernel_16_AVX512_2hv_single(&q[i], hh, nb, ldq, ldh, s);
	}
}
/**
 * Unrolled kernel that computes
 * 64 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_64_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [24 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	// carefull here
        __m512 sign = (__m512)_mm512_set1_epi32(0x80000000);

	__m512 x1 = _mm512_load_ps(&q[ldq]);
	__m512 x2 = _mm512_load_ps(&q[ldq+16]);
	__m512 x3 = _mm512_load_ps(&q[ldq+32]);
	__m512 x4 = _mm512_load_ps(&q[ldq+48]);


	__m512 h1 = _mm512_set1_ps(hh[ldh+1]);
	__m512 h2;

	__m512 q1 = _mm512_load_ps(q);
	__m512 y1 = _mm512_FMA_ps(x1, h1, q1);
	__m512 q2 = _mm512_load_ps(&q[16]);
	__m512 y2 = _mm512_FMA_ps(x2, h1, q2);
	__m512 q3 = _mm512_load_ps(&q[32]);
	__m512 y3 = _mm512_FMA_ps(x3, h1, q3);
	__m512 q4 = _mm512_load_ps(&q[48]);
	__m512 y4 = _mm512_FMA_ps(x4, h1, q4);

	for(i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-1]);
		h2 = _mm512_set1_ps(hh[ldh+i]);

		q1 = _mm512_load_ps(&q[i*ldq]);
		x1 = _mm512_FMA_ps(q1, h1, x1);
		y1 = _mm512_FMA_ps(q1, h2, y1);
		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
		x2 = _mm512_FMA_ps(q2, h1, x2);
		y2 = _mm512_FMA_ps(q2, h2, y2);
		q3 = _mm512_load_ps(&q[(i*ldq)+32]);
		x3 = _mm512_FMA_ps(q3, h1, x3);
		y3 = _mm512_FMA_ps(q3, h2, y3);
		q4 = _mm512_load_ps(&q[(i*ldq)+48]);
		x4 = _mm512_FMA_ps(q4, h1, x4);
		y4 = _mm512_FMA_ps(q4, h2, y4);

	}

	h1 = _mm512_set1_ps(hh[nb-1]);

	q1 = _mm512_load_ps(&q[nb*ldq]);
	x1 = _mm512_FMA_ps(q1, h1, x1);
	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);
	x3 = _mm512_FMA_ps(q3, h1, x3);
	q4 = _mm512_load_ps(&q[(nb*ldq)+48]);
	x4 = _mm512_FMA_ps(q4, h1, x4);


	/////////////////////////////////////////////////////
	// Rank-2 update of Q [24 x nb+1]
	/////////////////////////////////////////////////////

	__m512 tau1 = _mm512_set1_ps(hh[0]);
	__m512 tau2 = _mm512_set1_ps(hh[ldh]);
	__m512 vs = _mm512_set1_ps(s);

	h1 = (__m512) _mm512_xor_epi32((__m512i) tau1, (__m512i) sign);
	x1 = _mm512_mul_ps(x1, h1);
	x2 = _mm512_mul_ps(x2, h1);
	x3 = _mm512_mul_ps(x3, h1);
	x4 = _mm512_mul_ps(x4, h1);

	h1 = (__m512) _mm512_xor_epi32((__m512i) tau2, (__m512i) sign);
	h2 = _mm512_mul_ps(h1, vs);
	y1 = _mm512_FMA_ps(y1, h1, _mm512_mul_ps(x1,h2));
	y2 = _mm512_FMA_ps(y2, h1, _mm512_mul_ps(x2,h2));
	y3 = _mm512_FMA_ps(y3, h1, _mm512_mul_ps(x3,h2));
	y4 = _mm512_FMA_ps(y4, h1, _mm512_mul_ps(x4,h2));

	q1 = _mm512_load_ps(q);
	q1 = _mm512_add_ps(q1, y1);
	_mm512_store_ps(q,q1);
	q2 = _mm512_load_ps(&q[16]);
	q2 = _mm512_add_ps(q2, y2);
	_mm512_store_ps(&q[16],q2);
	q3 = _mm512_load_ps(&q[32]);
	q3 = _mm512_add_ps(q3, y3);
	_mm512_store_ps(&q[32],q3);
	q4 = _mm512_load_ps(&q[48]);
	q4 = _mm512_add_ps(q4, y4);
	_mm512_store_ps(&q[48],q4);

	h2 = _mm512_set1_ps(hh[ldh+1]);

	q1 = _mm512_load_ps(&q[ldq]);
	q1 = _mm512_add_ps(q1, _mm512_FMA_ps(y1, h2, x1));
	_mm512_store_ps(&q[ldq],q1);
	q2 = _mm512_load_ps(&q[ldq+16]);
	q2 = _mm512_add_ps(q2, _mm512_FMA_ps(y2, h2, x2));
	_mm512_store_ps(&q[ldq+16],q2);
	q3 = _mm512_load_ps(&q[ldq+32]);
	q3 = _mm512_add_ps(q3, _mm512_FMA_ps(y3, h2, x3));
	_mm512_store_ps(&q[ldq+32],q3);
	q4 = _mm512_load_ps(&q[ldq+48]);
	q4 = _mm512_add_ps(q4, _mm512_FMA_ps(y4, h2, x4));
	_mm512_store_ps(&q[ldq+48],q4);

	for (i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-1]);
		h2 = _mm512_set1_ps(hh[ldh+i]);

		q1 = _mm512_load_ps(&q[i*ldq]);
		q1 = _mm512_FMA_ps(x1, h1, q1);
		q1 = _mm512_FMA_ps(y1, h2, q1);
		_mm512_store_ps(&q[i*ldq],q1);
		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
		q2 = _mm512_FMA_ps(x2, h1, q2);
		q2 = _mm512_FMA_ps(y2, h2, q2);
		_mm512_store_ps(&q[(i*ldq)+16],q2);
		q3 = _mm512_load_ps(&q[(i*ldq)+32]);
		q3 = _mm512_FMA_ps(x3, h1, q3);
		q3 = _mm512_FMA_ps(y3, h2, q3);
		_mm512_store_ps(&q[(i*ldq)+32],q3);
		q4 = _mm512_load_ps(&q[(i*ldq)+48]);
		q4 = _mm512_FMA_ps(x4, h1, q4);
		q4 = _mm512_FMA_ps(y4, h2, q4);
		_mm512_store_ps(&q[(i*ldq)+48],q4);

	}

	h1 = _mm512_set1_ps(hh[nb-1]);

	q1 = _mm512_load_ps(&q[nb*ldq]);
	q1 = _mm512_FMA_ps(x1, h1, q1);
	_mm512_store_ps(&q[nb*ldq],q1);
	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
	q2 = _mm512_FMA_ps(x2, h1, q2);
	_mm512_store_ps(&q[(nb*ldq)+16],q2);
	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);
	q3 = _mm512_FMA_ps(x3, h1, q3);
	_mm512_store_ps(&q[(nb*ldq)+32],q3);
	q4 = _mm512_load_ps(&q[(nb*ldq)+48]);
	q4 = _mm512_FMA_ps(x4, h1, q4);
	_mm512_store_ps(&q[(nb*ldq)+48],q4);

}

/**
 * Unrolled kernel that computes
 * 48 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_48_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [24 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	// carefull here
        __m512 sign = (__m512)_mm512_set1_epi32(0x80000000);

	__m512 x1 = _mm512_load_ps(&q[ldq]);
	__m512 x2 = _mm512_load_ps(&q[ldq+16]);
	__m512 x3 = _mm512_load_ps(&q[ldq+32]);
//	__m512 x4 = _mm512_load_ps(&q[ldq+64]);


	__m512 h1 = _mm512_set1_ps(hh[ldh+1]);
	__m512 h2;

	__m512 q1 = _mm512_load_ps(q);
	__m512 y1 = _mm512_FMA_ps(x1, h1, q1);
	__m512 q2 = _mm512_load_ps(&q[16]);
	__m512 y2 = _mm512_FMA_ps(x2, h1, q2);
	__m512 q3 = _mm512_load_ps(&q[32]);
	__m512 y3 = _mm512_FMA_ps(x3, h1, q3);
//	__m512 q4 = _mm512_load_ps(&q[48]);
//	__m512 y4 = _mm512_FMA_ps(x4, h1, q4);

	for(i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-1]);
		h2 = _mm512_set1_ps(hh[ldh+i]);

		q1 = _mm512_load_ps(&q[i*ldq]);
		x1 = _mm512_FMA_ps(q1, h1, x1);
		y1 = _mm512_FMA_ps(q1, h2, y1);
		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
		x2 = _mm512_FMA_ps(q2, h1, x2);
		y2 = _mm512_FMA_ps(q2, h2, y2);
		q3 = _mm512_load_ps(&q[(i*ldq)+32]);
		x3 = _mm512_FMA_ps(q3, h1, x3);
		y3 = _mm512_FMA_ps(q3, h2, y3);
//		q4 = _mm512_load_ps(&q[(i*ldq)+48]);
//		x4 = _mm512_FMA_ps(q4, h1, x4);
//		y4 = _mm512_FMA_ps(q4, h2, y4);

	}

	h1 = _mm512_set1_ps(hh[nb-1]);

	q1 = _mm512_load_ps(&q[nb*ldq]);
	x1 = _mm512_FMA_ps(q1, h1, x1);
	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);
	x3 = _mm512_FMA_ps(q3, h1, x3);
//	q4 = _mm512_load_ps(&q[(nb*ldq)+48]);
//	x4 = _mm512_FMA_ps(q4, h1, x4);


	/////////////////////////////////////////////////////
	// Rank-2 update of Q [24 x nb+1]
	/////////////////////////////////////////////////////

	__m512 tau1 = _mm512_set1_ps(hh[0]);
	__m512 tau2 = _mm512_set1_ps(hh[ldh]);
	__m512 vs = _mm512_set1_ps(s);

	h1 = (__m512) _mm512_xor_epi32((__m512i) tau1, (__m512i) sign);
	x1 = _mm512_mul_ps(x1, h1);
	x2 = _mm512_mul_ps(x2, h1);
	x3 = _mm512_mul_ps(x3, h1);
//	x4 = _mm512_mul_ps(x4, h1);

	h1 = (__m512) _mm512_xor_epi32((__m512i) tau2, (__m512i) sign);
	h2 = _mm512_mul_ps(h1, vs);
	y1 = _mm512_FMA_ps(y1, h1, _mm512_mul_ps(x1,h2));
	y2 = _mm512_FMA_ps(y2, h1, _mm512_mul_ps(x2,h2));
	y3 = _mm512_FMA_ps(y3, h1, _mm512_mul_ps(x3,h2));
//	y4 = _mm512_FMA_ps(y4, h1, _mm512_mul_ps(x4,h2));

	q1 = _mm512_load_ps(q);
	q1 = _mm512_add_ps(q1, y1);
	_mm512_store_ps(q,q1);
	q2 = _mm512_load_ps(&q[16]);
	q2 = _mm512_add_ps(q2, y2);
	_mm512_store_ps(&q[16],q2);
	q3 = _mm512_load_ps(&q[32]);
	q3 = _mm512_add_ps(q3, y3);
	_mm512_store_ps(&q[32],q3);
//	q4 = _mm512_load_ps(&q[48]);
//	q4 = _mm512_add_ps(q4, y4);
//	_mm512_store_ps(&q[48],q4);

	h2 = _mm512_set1_ps(hh[ldh+1]);

	q1 = _mm512_load_ps(&q[ldq]);
	q1 = _mm512_add_ps(q1, _mm512_FMA_ps(y1, h2, x1));
	_mm512_store_ps(&q[ldq],q1);
	q2 = _mm512_load_ps(&q[ldq+16]);
	q2 = _mm512_add_ps(q2, _mm512_FMA_ps(y2, h2, x2));
	_mm512_store_ps(&q[ldq+16],q2);
	q3 = _mm512_load_ps(&q[ldq+32]);
	q3 = _mm512_add_ps(q3, _mm512_FMA_ps(y3, h2, x3));
	_mm512_store_ps(&q[ldq+32],q3);
//	q4 = _mm512_load_ps(&q[ldq+48]);
//	q4 = _mm512_add_ps(q4, _mm512_FMA_ps(y4, h2, x4));
//	_mm512_store_ps(&q[ldq+48],q4);

	for (i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-1]);
		h2 = _mm512_set1_ps(hh[ldh+i]);

		q1 = _mm512_load_ps(&q[i*ldq]);
		q1 = _mm512_FMA_ps(x1, h1, q1);
		q1 = _mm512_FMA_ps(y1, h2, q1);
		_mm512_store_ps(&q[i*ldq],q1);
		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
		q2 = _mm512_FMA_ps(x2, h1, q2);
		q2 = _mm512_FMA_ps(y2, h2, q2);
		_mm512_store_ps(&q[(i*ldq)+16],q2);
		q3 = _mm512_load_ps(&q[(i*ldq)+32]);
		q3 = _mm512_FMA_ps(x3, h1, q3);
		q3 = _mm512_FMA_ps(y3, h2, q3);
		_mm512_store_ps(&q[(i*ldq)+32],q3);
//		q4 = _mm512_load_ps(&q[(i*ldq)+48]);
//		q4 = _mm512_FMA_ps(x4, h1, q4);
//		q4 = _mm512_FMA_ps(y4, h2, q4);
//		_mm512_store_ps(&q[(i*ldq)+48],q4);

	}

	h1 = _mm512_set1_ps(hh[nb-1]);

	q1 = _mm512_load_ps(&q[nb*ldq]);
	q1 = _mm512_FMA_ps(x1, h1, q1);
	_mm512_store_ps(&q[nb*ldq],q1);
	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
	q2 = _mm512_FMA_ps(x2, h1, q2);
	_mm512_store_ps(&q[(nb*ldq)+16],q2);
	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);
	q3 = _mm512_FMA_ps(x3, h1, q3);
	_mm512_store_ps(&q[(nb*ldq)+32],q3);
//	q4 = _mm512_load_ps(&q[(nb*ldq)+48]);
//	q4 = _mm512_FMA_ps(x4, h1, q4);
//	_mm512_store_ps(&q[(nb*ldq)+48],q4);

}


/**
 * Unrolled kernel that computes
 * 32 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_32_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [24 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	// carefull here
        __m512 sign = (__m512)_mm512_set1_epi32(0x80000000);

	__m512 x1 = _mm512_load_ps(&q[ldq]);
	__m512 x2 = _mm512_load_ps(&q[ldq+16]);
//	__m512 x3 = _mm512_load_ps(&q[ldq+48]);
//	__m512 x4 = _mm512_load_ps(&q[ldq+64]);


	__m512 h1 = _mm512_set1_ps(hh[ldh+1]);
	__m512 h2;

	__m512 q1 = _mm512_load_ps(q);
	__m512 y1 = _mm512_FMA_ps(x1, h1, q1);
	__m512 q2 = _mm512_load_ps(&q[16]);
	__m512 y2 = _mm512_FMA_ps(x2, h1, q2);
//	__m512 q3 = _mm512_load_ps(&q[32]);
//	__m512 y3 = _mm512_FMA_ps(x3, h1, q3);
//	__m512 q4 = _mm512_load_ps(&q[48]);
//	__m512 y4 = _mm512_FMA_ps(x4, h1, q4);

	for(i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-1]);
		h2 = _mm512_set1_ps(hh[ldh+i]);

		q1 = _mm512_load_ps(&q[i*ldq]);
		x1 = _mm512_FMA_ps(q1, h1, x1);
		y1 = _mm512_FMA_ps(q1, h2, y1);
		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
		x2 = _mm512_FMA_ps(q2, h1, x2);
		y2 = _mm512_FMA_ps(q2, h2, y2);
//		q3 = _mm512_load_ps(&q[(i*ldq)+32]);
//		x3 = _mm512_FMA_ps(q3, h1, x3);
//		y3 = _mm512_FMA_ps(q3, h2, y3);
//		q4 = _mm512_load_ps(&q[(i*ldq)+48]);
//		x4 = _mm512_FMA_ps(q4, h1, x4);
//		y4 = _mm512_FMA_ps(q4, h2, y4);

	}

	h1 = _mm512_set1_ps(hh[nb-1]);

	q1 = _mm512_load_ps(&q[nb*ldq]);
	x1 = _mm512_FMA_ps(q1, h1, x1);
	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
	x2 = _mm512_FMA_ps(q2, h1, x2);
//	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);
//	x3 = _mm512_FMA_ps(q3, h1, x3);
//	q4 = _mm512_load_ps(&q[(nb*ldq)+48]);
//	x4 = _mm512_FMA_ps(q4, h1, x4);


	/////////////////////////////////////////////////////
	// Rank-2 update of Q [24 x nb+1]
	/////////////////////////////////////////////////////

	__m512 tau1 = _mm512_set1_ps(hh[0]);
	__m512 tau2 = _mm512_set1_ps(hh[ldh]);
	__m512 vs = _mm512_set1_ps(s);

	h1 = (__m512) _mm512_xor_epi32((__m512i) tau1, (__m512i) sign);
	x1 = _mm512_mul_ps(x1, h1);
	x2 = _mm512_mul_ps(x2, h1);
//	x3 = _mm512_mul_ps(x3, h1);
//	x4 = _mm512_mul_ps(x4, h1);

	h1 = (__m512) _mm512_xor_epi32((__m512i) tau2, (__m512i) sign);
	h2 = _mm512_mul_ps(h1, vs);
	y1 = _mm512_FMA_ps(y1, h1, _mm512_mul_ps(x1,h2));
	y2 = _mm512_FMA_ps(y2, h1, _mm512_mul_ps(x2,h2));
//	y3 = _mm512_FMA_ps(y3, h1, _mm512_mul_ps(x3,h2));
//	y4 = _mm512_FMA_ps(y4, h1, _mm512_mul_ps(x4,h2));

	q1 = _mm512_load_ps(q);
	q1 = _mm512_add_ps(q1, y1);
	_mm512_store_ps(q,q1);
	q2 = _mm512_load_ps(&q[16]);
	q2 = _mm512_add_ps(q2, y2);
	_mm512_store_ps(&q[16],q2);
//	q3 = _mm512_load_ps(&q[32]);
//	q3 = _mm512_add_ps(q3, y3);
//	_mm512_store_ps(&q[32],q3);
//	q4 = _mm512_load_ps(&q[48]);
//	q4 = _mm512_add_ps(q4, y4);
//	_mm512_store_ps(&q[48],q4);

	h2 = _mm512_set1_ps(hh[ldh+1]);

	q1 = _mm512_load_ps(&q[ldq]);
	q1 = _mm512_add_ps(q1, _mm512_FMA_ps(y1, h2, x1));
	_mm512_store_ps(&q[ldq],q1);
	q2 = _mm512_load_ps(&q[ldq+16]);
	q2 = _mm512_add_ps(q2, _mm512_FMA_ps(y2, h2, x2));
	_mm512_store_ps(&q[ldq+16],q2);
//	q3 = _mm512_load_ps(&q[ldq+32]);
//	q3 = _mm512_add_ps(q3, _mm512_FMA_ps(y3, h2, x3));
//	_mm512_store_ps(&q[ldq+32],q3);
//	q4 = _mm512_load_ps(&q[ldq+48]);
//	q4 = _mm512_add_ps(q4, _mm512_FMA_ps(y4, h2, x4));
//	_mm512_store_ps(&q[ldq+48],q4);

	for (i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-1]);
		h2 = _mm512_set1_ps(hh[ldh+i]);

		q1 = _mm512_load_ps(&q[i*ldq]);
		q1 = _mm512_FMA_ps(x1, h1, q1);
		q1 = _mm512_FMA_ps(y1, h2, q1);
		_mm512_store_ps(&q[i*ldq],q1);
		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
		q2 = _mm512_FMA_ps(x2, h1, q2);
		q2 = _mm512_FMA_ps(y2, h2, q2);
		_mm512_store_ps(&q[(i*ldq)+16],q2);
//		q3 = _mm512_load_ps(&q[(i*ldq)+32]);
//		q3 = _mm512_FMA_ps(x3, h1, q3);
//		q3 = _mm512_FMA_ps(y3, h2, q3);
//		_mm512_store_ps(&q[(i*ldq)+32],q3);
//		q4 = _mm512_load_ps(&q[(i*ldq)+48]);
//		q4 = _mm512_FMA_ps(x4, h1, q4);
//		q4 = _mm512_FMA_ps(y4, h2, q4);
//		_mm512_store_ps(&q[(i*ldq)+48],q4);

	}

	h1 = _mm512_set1_ps(hh[nb-1]);

	q1 = _mm512_load_ps(&q[nb*ldq]);
	q1 = _mm512_FMA_ps(x1, h1, q1);
	_mm512_store_ps(&q[nb*ldq],q1);
	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
	q2 = _mm512_FMA_ps(x2, h1, q2);
	_mm512_store_ps(&q[(nb*ldq)+16],q2);
//	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);
//	q3 = _mm512_FMA_ps(x3, h1, q3);
//	_mm512_store_ps(&q[(nb*ldq)+32],q3);
//	q4 = _mm512_load_ps(&q[(nb*ldq)+48]);
//	q4 = _mm512_FMA_ps(x4, h1, q4);
//	_mm512_store_ps(&q[(nb*ldq)+48],q4);

}


/**
 * Unrolled kernel that computes
 * 16 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_16_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [24 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	// carefull here
        __m512 sign = (__m512)_mm512_set1_epi32(0x80000000);

	__m512 x1 = _mm512_load_ps(&q[ldq]);
//	__m512 x2 = _mm512_load_ps(&q[ldq+32]);
//	__m512 x3 = _mm512_load_ps(&q[ldq+48]);
//	__m512 x4 = _mm512_load_ps(&q[ldq+64]);


	__m512 h1 = _mm512_set1_ps(hh[ldh+1]);
	__m512 h2;

	__m512 q1 = _mm512_load_ps(q);
	__m512 y1 = _mm512_FMA_ps(x1, h1, q1);
//	__m512 q2 = _mm512_load_ps(&q[16]);
//	__m512 y2 = _mm512_FMA_ps(x2, h1, q2);
//	__m512 q3 = _mm512_load_ps(&q[32]);
//	__m512 y3 = _mm512_FMA_ps(x3, h1, q3);
//	__m512 q4 = _mm512_load_ps(&q[48]);
//	__m512 y4 = _mm512_FMA_ps(x4, h1, q4);

	for(i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-1]);
		h2 = _mm512_set1_ps(hh[ldh+i]);

		q1 = _mm512_load_ps(&q[i*ldq]);
		x1 = _mm512_FMA_ps(q1, h1, x1);
		y1 = _mm512_FMA_ps(q1, h2, y1);
//		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
//		x2 = _mm512_FMA_ps(q2, h1, x2);
//		y2 = _mm512_FMA_ps(q2, h2, y2);
//		q3 = _mm512_load_ps(&q[(i*ldq)+32]);
//		x3 = _mm512_FMA_ps(q3, h1, x3);
//		y3 = _mm512_FMA_ps(q3, h2, y3);
//		q4 = _mm512_load_ps(&q[(i*ldq)+48]);
//		x4 = _mm512_FMA_ps(q4, h1, x4);
//		y4 = _mm512_FMA_ps(q4, h2, y4);

	}

	h1 = _mm512_set1_ps(hh[nb-1]);

	q1 = _mm512_load_ps(&q[nb*ldq]);
	x1 = _mm512_FMA_ps(q1, h1, x1);
//	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
//	x2 = _mm512_FMA_ps(q2, h1, x2);
//	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);
//	x3 = _mm512_FMA_ps(q3, h1, x3);
//	q4 = _mm512_load_ps(&q[(nb*ldq)+48]);
//	x4 = _mm512_FMA_ps(q4, h1, x4);


	/////////////////////////////////////////////////////
	// Rank-2 update of Q [24 x nb+1]
	/////////////////////////////////////////////////////

	__m512 tau1 = _mm512_set1_ps(hh[0]);
	__m512 tau2 = _mm512_set1_ps(hh[ldh]);
	__m512 vs = _mm512_set1_ps(s);

	h1 = (__m512) _mm512_xor_epi32((__m512i) tau1, (__m512i) sign);
	x1 = _mm512_mul_ps(x1, h1);
//	x2 = _mm512_mul_ps(x2, h1);
//	x3 = _mm512_mul_ps(x3, h1);
//	x4 = _mm512_mul_ps(x4, h1);

	h1 = (__m512) _mm512_xor_epi32((__m512i) tau2, (__m512i) sign);
	h2 = _mm512_mul_ps(h1, vs);
	y1 = _mm512_FMA_ps(y1, h1, _mm512_mul_ps(x1,h2));
//	y2 = _mm512_FMA_ps(y2, h1, _mm512_mul_ps(x2,h2));
//	y3 = _mm512_FMA_ps(y3, h1, _mm512_mul_ps(x3,h2));
//	y4 = _mm512_FMA_ps(y4, h1, _mm512_mul_ps(x4,h2));

	q1 = _mm512_load_ps(q);
	q1 = _mm512_add_ps(q1, y1);
	_mm512_store_ps(q,q1);
//	q2 = _mm512_load_ps(&q[16]);
//	q2 = _mm512_add_ps(q2, y2);
//	_mm512_store_ps(&q[16],q2);
//	q3 = _mm512_load_ps(&q[32]);
//	q3 = _mm512_add_ps(q3, y3);
//	_mm512_store_ps(&q[32],q3);
//	q4 = _mm512_load_ps(&q[48]);
//	q4 = _mm512_add_ps(q4, y4);
//	_mm512_store_ps(&q[48],q4);

	h2 = _mm512_set1_ps(hh[ldh+1]);

	q1 = _mm512_load_ps(&q[ldq]);
	q1 = _mm512_add_ps(q1, _mm512_FMA_ps(y1, h2, x1));
	_mm512_store_ps(&q[ldq],q1);
//	q2 = _mm512_load_ps(&q[ldq+16]);
//	q2 = _mm512_add_ps(q2, _mm512_FMA_ps(y2, h2, x2));
//	_mm512_store_ps(&q[ldq+16],q2);
//	q3 = _mm512_load_ps(&q[ldq+32]);
//	q3 = _mm512_add_ps(q3, _mm512_FMA_ps(y3, h2, x3));
//	_mm512_store_ps(&q[ldq+32],q3);
//	q4 = _mm512_load_ps(&q[ldq+48]);
//	q4 = _mm512_add_ps(q4, _mm512_FMA_ps(y4, h2, x4));
//	_mm512_store_ps(&q[ldq+48],q4);

	for (i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-1]);
		h2 = _mm512_set1_ps(hh[ldh+i]);

		q1 = _mm512_load_ps(&q[i*ldq]);
		q1 = _mm512_FMA_ps(x1, h1, q1);
		q1 = _mm512_FMA_ps(y1, h2, q1);
		_mm512_store_ps(&q[i*ldq],q1);
//		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
//		q2 = _mm512_FMA_ps(x2, h1, q2);
//		q2 = _mm512_FMA_ps(y2, h2, q2);
//		_mm512_store_ps(&q[(i*ldq)+16],q2);
//		q3 = _mm512_load_ps(&q[(i*ldq)+32]);
//		q3 = _mm512_FMA_ps(x3, h1, q3);
//		q3 = _mm512_FMA_ps(y3, h2, q3);
//		_mm512_store_ps(&q[(i*ldq)+32],q3);
//		q4 = _mm512_load_ps(&q[(i*ldq)+48]);
//		q4 = _mm512_FMA_ps(x4, h1, q4);
//		q4 = _mm512_FMA_ps(y4, h2, q4);
//		_mm512_store_ps(&q[(i*ldq)+48],q4);

	}

	h1 = _mm512_set1_ps(hh[nb-1]);

	q1 = _mm512_load_ps(&q[nb*ldq]);
	q1 = _mm512_FMA_ps(x1, h1, q1);
	_mm512_store_ps(&q[nb*ldq],q1);
//	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
//	q2 = _mm512_FMA_ps(x2, h1, q2);
//	_mm512_store_ps(&q[(nb*ldq)+16],q2);
//	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);
//	q3 = _mm512_FMA_ps(x3, h1, q3);
//	_mm512_store_ps(&q[(nb*ldq)+32],q3);
//	q4 = _mm512_load_ps(&q[(nb*ldq)+48]);
//	q4 = _mm512_FMA_ps(x4, h1, q4);
//	_mm512_store_ps(&q[(nb*ldq)+48],q4);

}

