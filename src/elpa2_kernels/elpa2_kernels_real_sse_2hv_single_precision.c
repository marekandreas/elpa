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
// Author: Andreas Marek, MPCDF, based on the double precision case of A. Heinecke
//
#include "config-f90.h"

#include <x86intrin.h>

#define __forceinline __attribute__((always_inline)) static


#ifdef HAVE_SSE_INTRINSICS
#undef __AVX__
#endif

//Forward declaration
__forceinline void hh_trafo_kernel_4_SSE_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_8_SSE_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_12_SSE_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);

void double_hh_trafo_real_sse_2hv_single_(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);

/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine double_hh_trafo_real_sse_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_real_sse_2hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

void double_hh_trafo_real_sse_2hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
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
	for (i = 0; i < nq-8; i+=12)
	{
	hh_trafo_kernel_12_SSE_2hv_single(&q[i], hh, nb, ldq, ldh, s);

	}
	if (nq == i)
	{
		return;
	}
	if (nq-i == 8)
	{
		hh_trafo_kernel_8_SSE_2hv_single(&q[i], hh, nb, ldq, ldh, s);
	}
	else
	{
		hh_trafo_kernel_4_SSE_2hv_single(&q[i], hh, nb, ldq, ldh, s);
	}

}

/**
 * Unrolled kernel that computes
 * 12 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_12_SSE_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [12 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	//
	// carefull here
	__m128 sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));

	__m128 x1 = _mm_load_ps(&q[ldq]);       // load | q(1,2) | q(2,2) | q(3,2) | q(4,2)
	__m128 x2 = _mm_load_ps(&q[ldq+4]);     // load | q(5,2) ..... | q(8,2)
	__m128 x3 = _mm_load_ps(&q[ldq+8]);     // load | q(9,2) ... | q(12,2)

	__m128 h1 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+1])));     // x4 contains 4 times hh(2,2)
	__m128 h2;

	__m128 q1 = _mm_load_ps(q);
	__m128 y1 = _mm_add_ps(q1, _mm_mul_ps(x1, h1));
	__m128 q2 = _mm_load_ps(&q[4]);
	__m128 y2 = _mm_add_ps(q2, _mm_mul_ps(x2, h1));
	__m128 q3 = _mm_load_ps(&q[8]);
	__m128 y3 = _mm_add_ps(q3, _mm_mul_ps(x3, h1));

	for(i = 2; i < nb; i++)
	{
		h1 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[i-1])));
		h2 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+i])));

		q1 = _mm_load_ps(&q[i*ldq]);
		x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
		y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
		q2 = _mm_load_ps(&q[(i*ldq)+4]);
		x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
		y2 = _mm_add_ps(y2, _mm_mul_ps(q2,h2));
		q3 = _mm_load_ps(&q[(i*ldq)+8]);
		x3 = _mm_add_ps(x3, _mm_mul_ps(q3,h1));
		y3 = _mm_add_ps(y3, _mm_mul_ps(q3,h2));
	}

	h1 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[nb-1])));

	q1 = _mm_load_ps(&q[nb*ldq]);
	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
	q2 = _mm_load_ps(&q[(nb*ldq)+4]);
	x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
	q3 = _mm_load_ps(&q[(nb*ldq)+8]);
	x3 = _mm_add_ps(x3, _mm_mul_ps(q3,h1));

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [12 x nb+1]
	/////////////////////////////////////////////////////

	__m128 tau1 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) hh)));
        __m128 tau2 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh])));

	__m128 vs = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd((double*) &s)));

	h1 = _mm_xor_ps(tau1, sign);
	x1 = _mm_mul_ps(x1, h1);
	x2 = _mm_mul_ps(x2, h1);
	x3 = _mm_mul_ps(x3, h1);

	h1 = _mm_xor_ps(tau2, sign);
	h2 = _mm_mul_ps(h1, vs);

	y1 = _mm_add_ps(_mm_mul_ps(y1,h1), _mm_mul_ps(x1,h2));
	y2 = _mm_add_ps(_mm_mul_ps(y2,h1), _mm_mul_ps(x2,h2));
	y3 = _mm_add_ps(_mm_mul_ps(y3,h1), _mm_mul_ps(x3,h2));

	q1 = _mm_load_ps(q);
	q1 = _mm_add_ps(q1, y1);
	_mm_store_ps(q,q1);
	q2 = _mm_load_ps(&q[4]);
	q2 = _mm_add_ps(q2, y2);
	_mm_store_ps(&q[4],q2);
	q3 = _mm_load_ps(&q[8]);
	q3 = _mm_add_ps(q3, y3);
	_mm_store_ps(&q[8],q3);

        h2 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+1])));
//	h2 = _mm_castpd_ps(_mm_loaddup_pd(&hh[ldh+1]));

	q1 = _mm_load_ps(&q[ldq]);
	q1 = _mm_add_ps(q1, _mm_add_ps(x1, _mm_mul_ps(y1, h2)));
	_mm_store_ps(&q[ldq],q1);
	q2 = _mm_load_ps(&q[ldq+4]);
	q2 = _mm_add_ps(q2, _mm_add_ps(x2, _mm_mul_ps(y2, h2)));
	_mm_store_ps(&q[ldq+4],q2);
	q3 = _mm_load_ps(&q[ldq+8]);
	q3 = _mm_add_ps(q3, _mm_add_ps(x3, _mm_mul_ps(y3, h2)));
	_mm_store_ps(&q[ldq+8],q3);
//	q4 = _mm_load_pd(&q[ldq+6]);
//	q4 = _mm_add_pd(q4, _mm_add_pd(x4, _mm_mul_pd(y4, h2)));
//	_mm_store_pd(&q[ldq+6],q4);
//	q5 = _mm_load_pd(&q[ldq+8]);
//	q5 = _mm_add_pd(q5, _mm_add_pd(x5, _mm_mul_pd(y5, h2)));
//	_mm_store_pd(&q[ldq+8],q5);
//	q6 = _mm_load_pd(&q[ldq+10]);
//	q6 = _mm_add_pd(q6, _mm_add_pd(x6, _mm_mul_pd(y6, h2)));
//	_mm_store_pd(&q[ldq+10],q6);

	for (i = 2; i < nb; i++)
	{
		h1 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[i-1])));
//		h1 = _mm_castpd_ps(_mm_loaddup_pd(&hh[i-1]));

		h2 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+i])));
//		h2 = _mm_castpd_ps(_mm_loaddup_pd(&hh[ldh+i]));

		q1 = _mm_load_ps(&q[i*ldq]);
		q1 = _mm_add_ps(q1, _mm_add_ps(_mm_mul_ps(x1,h1), _mm_mul_ps(y1, h2)));
		_mm_store_ps(&q[i*ldq],q1);
		q2 = _mm_load_ps(&q[(i*ldq)+4]);
		q2 = _mm_add_ps(q2, _mm_add_ps(_mm_mul_ps(x2,h1), _mm_mul_ps(y2, h2)));
		_mm_store_ps(&q[(i*ldq)+4],q2);
		q3 = _mm_load_ps(&q[(i*ldq)+8]);
		q3 = _mm_add_ps(q3, _mm_add_ps(_mm_mul_ps(x3,h1), _mm_mul_ps(y3, h2)));
		_mm_store_ps(&q[(i*ldq)+8],q3);
//		q4 = _mm_load_pd(&q[(i*ldq)+6]);
//		q4 = _mm_add_pd(q4, _mm_add_pd(_mm_mul_pd(x4,h1), _mm_mul_pd(y4, h2)));
//		_mm_store_pd(&q[(i*ldq)+6],q4);
//		q5 = _mm_load_pd(&q[(i*ldq)+8]);
//		q5 = _mm_add_pd(q5, _mm_add_pd(_mm_mul_pd(x5,h1), _mm_mul_pd(y5, h2)));
//		_mm_store_pd(&q[(i*ldq)+8],q5);
//		q6 = _mm_load_pd(&q[(i*ldq)+10]);
//		q6 = _mm_add_pd(q6, _mm_add_pd(_mm_mul_pd(x6,h1), _mm_mul_pd(y6, h2)));
//		_mm_store_pd(&q[(i*ldq)+10],q6);
	}

	h1 = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[nb-1])));
//	h1 = _mm_castpd_ps(_mm_loaddup_pd(&hh[nb-1]));

	q1 = _mm_load_ps(&q[nb*ldq]);
	q1 = _mm_add_ps(q1, _mm_mul_ps(x1, h1));
	_mm_store_ps(&q[nb*ldq],q1);
	q2 = _mm_load_ps(&q[(nb*ldq)+4]);
	q2 = _mm_add_ps(q2, _mm_mul_ps(x2, h1));
	_mm_store_ps(&q[(nb*ldq)+4],q2);
	q3 = _mm_load_ps(&q[(nb*ldq)+8]);
	q3 = _mm_add_ps(q3, _mm_mul_ps(x3, h1));
	_mm_store_ps(&q[(nb*ldq)+8],q3);
//	q4 = _mm_load_pd(&q[(nb*ldq)+6]);
//	q4 = _mm_add_pd(q4, _mm_mul_pd(x4, h1));
//	_mm_store_pd(&q[(nb*ldq)+6],q4);
//	q5 = _mm_load_pd(&q[(nb*ldq)+8]);
//	q5 = _mm_add_pd(q5, _mm_mul_pd(x5, h1));
//	_mm_store_pd(&q[(nb*ldq)+8],q5);
//	q6 = _mm_load_pd(&q[(nb*ldq)+10]);
//	q6 = _mm_add_pd(q6, _mm_mul_pd(x6, h1));
//	_mm_store_pd(&q[(nb*ldq)+10],q6);
}

/**
 * Unrolled kernel that computes
 * 8 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
__forceinline void hh_trafo_kernel_8_SSE_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [8 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	// carefull here
	__m128 sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));

	__m128 x1 = _mm_load_ps(&q[ldq]);
	__m128 x2 = _mm_load_ps(&q[ldq+4]);

	__m128 x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+1]));
	__m128 x4 ;
        __m128 h1 = _mm_moveldup_ps(x3);
	__m128 h2;

	__m128 q1 = _mm_load_ps(q);
	__m128 y1 = _mm_add_ps(q1, _mm_mul_ps(x1, h1));
	__m128 q2 = _mm_load_ps(&q[4]);
	__m128 y2 = _mm_add_ps(q2, _mm_mul_ps(x2, h1));

	for(i = 2; i < nb; i++)
	{
		x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[i-1]));
		h1 = _mm_moveldup_ps(x3);

		x4 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+i]));
                h2 = _mm_moveldup_ps(x4);

		q1 = _mm_load_ps(&q[i*ldq]);
		x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
		y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
		q2 = _mm_load_ps(&q[(i*ldq)+4]);
		x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
		y2 = _mm_add_ps(y2, _mm_mul_ps(q2,h2));
	}

	x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[nb-1]));
	h1 = _mm_moveldup_ps(x3);

	q1 = _mm_load_ps(&q[nb*ldq]);
	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
	q2 = _mm_load_ps(&q[(nb*ldq)+4]);
	x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [8 x nb+1]
	/////////////////////////////////////////////////////

	x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) hh));
        __m128 tau1 = _mm_moveldup_ps(x3);

	x4 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh]));
  	__m128 tau2 = _mm_moveldup_ps(x4);
        __m128 x5 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &s));
        __m128 vs = _mm_moveldup_ps(x5);

	h1 = _mm_xor_ps(tau1, sign);
	x1 = _mm_mul_ps(x1, h1);
	x2 = _mm_mul_ps(x2, h1);
	h1 = _mm_xor_ps(tau2, sign);
	h2 = _mm_mul_ps(h1, vs);

	y1 = _mm_add_ps(_mm_mul_ps(y1,h1), _mm_mul_ps(x1,h2));
	y2 = _mm_add_ps(_mm_mul_ps(y2,h1), _mm_mul_ps(x2,h2));

	q1 = _mm_load_ps(q);
	q1 = _mm_add_ps(q1, y1);
	_mm_store_ps(q,q1);
	q2 = _mm_load_ps(&q[4]);
	q2 = _mm_add_ps(q2, y2);
	_mm_store_ps(&q[4],q2);

        x4 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+1]));
        h2 = _mm_moveldup_ps(x4);

	q1 = _mm_load_ps(&q[ldq]);
	q1 = _mm_add_ps(q1, _mm_add_ps(x1, _mm_mul_ps(y1, h2)));
	_mm_store_ps(&q[ldq],q1);
	q2 = _mm_load_ps(&q[ldq+4]);
	q2 = _mm_add_ps(q2, _mm_add_ps(x2, _mm_mul_ps(y2, h2)));
	_mm_store_ps(&q[ldq+4],q2);

	for (i = 2; i < nb; i++)
	{
                x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[i-1]));
                h1 = _mm_moveldup_ps(x3);

		x4 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+i]));
                h2 = _mm_moveldup_ps(x4);

		q1 = _mm_load_ps(&q[i*ldq]);
		q1 = _mm_add_ps(q1, _mm_add_ps(_mm_mul_ps(x1,h1), _mm_mul_ps(y1, h2)));
		_mm_store_ps(&q[i*ldq],q1);
		q2 = _mm_load_ps(&q[(i*ldq)+4]);
		q2 = _mm_add_ps(q2, _mm_add_ps(_mm_mul_ps(x2,h1), _mm_mul_ps(y2, h2)));
		_mm_store_ps(&q[(i*ldq)+4],q2);
	}

	x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[nb-1]));
        h1 = _mm_moveldup_ps(x3);

	q1 = _mm_load_ps(&q[nb*ldq]);
	q1 = _mm_add_ps(q1, _mm_mul_ps(x1, h1));
	_mm_store_ps(&q[nb*ldq],q1);
	q2 = _mm_load_ps(&q[(nb*ldq)+4]);
	q2 = _mm_add_ps(q2, _mm_mul_ps(x2, h1));
	_mm_store_ps(&q[(nb*ldq)+4],q2);
}

/**
 * Unrolled kernel that computes
 * 4 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
__forceinline void hh_trafo_kernel_4_SSE_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	// carefull here
	__m128 sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));

	__m128 x1 = _mm_load_ps(&q[ldq]);

        __m128 x2 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+1]));
	__m128 x3 ;
        __m128 h1 = _mm_moveldup_ps(x2);
	__m128 h2;

	__m128 q1 = _mm_load_ps(q);
	__m128 y1 = _mm_add_ps(q1, _mm_mul_ps(x1, h1));

	for(i = 2; i < nb; i++)
	{

		x2 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[i-1]));
                h1 = _mm_moveldup_ps(x2);

		x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+i]));
                h2 = _mm_moveldup_ps(x3);

		q1 = _mm_load_ps(&q[i*ldq]);
		x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
		y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
	}

	x2 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[nb-1]));
        h1 = _mm_moveldup_ps(x2);

	q1 = _mm_load_ps(&q[nb*ldq]);
	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [12 x nb+1]
	/////////////////////////////////////////////////////

	x2 = _mm_castpd_ps(_mm_loaddup_pd( (double *) hh));
        __m128 tau1 = _mm_moveldup_ps(x2);

	x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh]));
        __m128 tau2 = _mm_moveldup_ps(x3);

	__m128 x4 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &s));
        __m128 vs = _mm_moveldup_ps(x4);

	h1 = _mm_xor_ps(tau1, sign);
	x1 = _mm_mul_ps(x1, h1);
	h1 = _mm_xor_ps(tau2, sign);
	h2 = _mm_mul_ps(h1, vs);

	y1 = _mm_add_ps(_mm_mul_ps(y1,h1), _mm_mul_ps(x1,h2));

	q1 = _mm_load_ps(q);
	q1 = _mm_add_ps(q1, y1);
	_mm_store_ps(q,q1);

	x2 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+1]));
        h2 = _mm_moveldup_ps(x2);

	q1 = _mm_load_ps(&q[ldq]);
	q1 = _mm_add_ps(q1, _mm_add_ps(x1, _mm_mul_ps(y1, h2)));
	_mm_store_ps(&q[ldq],q1);

	for (i = 2; i < nb; i++)
	{
                x2 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[i-1]));
                h1 = _mm_moveldup_ps(x2);

		x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[ldh+i]));
		h2 = _mm_moveldup_ps(x3);

		q1 = _mm_load_ps(&q[i*ldq]);
		q1 = _mm_add_ps(q1, _mm_add_ps(_mm_mul_ps(x1,h1), _mm_mul_ps(y1, h2)));
		_mm_store_ps(&q[i*ldq],q1);
	}

        x2 = _mm_castpd_ps(_mm_loaddup_pd( (double *) &hh[nb-1]));
        h1 = _mm_moveldup_ps(x2);

	q1 = _mm_load_ps(&q[nb*ldq]);
	q1 = _mm_add_ps(q1, _mm_mul_ps(x1, h1));
	_mm_store_ps(&q[nb*ldq],q1);
}

