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

#ifdef HAVE_AVX2

#ifdef __FMA4__
#define __ELPA_USE_FMA__
#define _mm256_FMA_ps(a,b,c) _mm256_macc_ps(a,b,c)
#endif

#ifdef __AVX2__
#define __ELPA_USE_FMA__
#define _mm256_FMA_ps(a,b,c) _mm256_fmadd_ps(a,b,c)
#endif

#endif

//Forward declaration
// 4 rows single presision does not work in AVX since it cannot be 32 aligned use sse instead
__forceinline void hh_trafo_kernel_4_AVX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
//__forceinline void hh_trafo_kernel_4_sse_instead_of_avx_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_8_AVX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_16_AVX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_24_AVX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);

void double_hh_trafo_real_avx_avx2_2hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);

/*
!f>#if defined(HAVE_AVX) || defined(HAVE_AVX2)
!f> interface
!f>   subroutine double_hh_trafo_real_avx_avx2_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_real_avx_avx2_2hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

void double_hh_trafo_real_avx_avx2_2hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;

	// calculating scalar product to compute
	// 2 householder vectors simultaneously
	//
	// Fortran:
	// s = hh(2,2)*1
	float s = hh[(ldh)+1]*1.0;

	// FORTRAN:
	// do = 3, nb
	// s =s + hh(i,2)*hh(i-1,1)
	#pragma ivdep
	for (i = 2; i < nb; i++)
	{
		s += hh[i-1] * hh[(i+ldh)];
	}

	// Production level kernel calls with padding
	for (i = 0; i < nq-20; i+=24)
	{
		hh_trafo_kernel_24_AVX_2hv_single(&q[i], hh, nb, ldq, ldh, s);
	}
	if (nq == i)
	{
		return;
	}
	if (nq-i == 20)
	{
		hh_trafo_kernel_16_AVX_2hv_single(&q[i], hh, nb, ldq, ldh, s);
		hh_trafo_kernel_4_AVX_2hv_single(&q[i+16], hh, nb, ldq, ldh, s);
// 		hh_trafo_kernel_4_sse_instead_of_avx_2hv_single(&q[i+8], hh, nb, ldq, ldh, s);

	}
	else if (nq-i == 16)
	{
		hh_trafo_kernel_16_AVX_2hv_single(&q[i], hh, nb, ldq, ldh, s);
	}
	else if (nq-i == 12)
	{
		hh_trafo_kernel_8_AVX_2hv_single(&q[i], hh, nb, ldq, ldh, s);
		hh_trafo_kernel_4_AVX_2hv_single(&q[i+8], hh, nb, ldq, ldh, s);
// 		hh_trafo_kernel_4_sse_instead_of_avx_2hv_single(&q[i+8], hh, nb, ldq, ldh, s);
	}
	else if (nq-i == 8)
	{
		hh_trafo_kernel_8_AVX_2hv_single(&q[i], hh, nb, ldq, ldh, s);
	}
	else
	{
		hh_trafo_kernel_4_AVX_2hv_single(&q[i], hh, nb, ldq, ldh, s);
//		hh_trafo_kernel_4_sse_instead_of_avx_2hv_single(&q[i], hh, nb, ldq, ldh, s);

	}
}


/**
 * Unrolled kernel that computes
 * 24 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_24_AVX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [24 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	__m256 sign = (__m256)_mm256_set1_epi32(0x80000000);

	__m256 x1 = _mm256_load_ps(&q[ldq]);      //load q(1,2), q(2,2), q(3,2),q(4,2), q(5,2), q(6,2), q(7,2), q(8,2)

	__m256 x2 = _mm256_load_ps(&q[ldq+8]);   // load q(9,2) ... q(16,2)

	__m256 x3 = _mm256_load_ps(&q[ldq+16]);  // load q(17,2) .. q(24,2)
//	__m256 x4 = _mm256_load_ps(&q[ldq+12]);
//	__m256 x5 = _mm256_load_ps(&q[ldq+16]);
//	__m256 x6 = _mm256_load_ps(&q[ldq+20]);

	__m256 h1 = _mm256_broadcast_ss(&hh[ldh+1]);  // h1 = hh(2,2) | hh(2,2) | hh(2,2) | hh(2,2) | hh(2,2) | hh(2,2) | hh(2,2) | hh(2,2)
	__m256 h2;

#ifdef __ELPA_USE_FMA__
	__m256 q1 = _mm256_load_ps(q);             // q1 = q(1,1), q(2,1), q(3,1), q(4,1), q(5,1), q(6,1), q(7,1), q(8,1)
	__m256 y1 = _mm256_FMA_ps(x1, h1, q1);     // y1 = q(1,2) * h(2,2) + q(1,1) | q(2,2) * h(2,2) + q(2,1) | .... | q(8,2) * h(2,2) + q(8,1)
	__m256 q2 = _mm256_load_ps(&q[8]);         // q2 = q(9,1) | .... | q(16,1)
	__m256 y2 = _mm256_FMA_ps(x2, h1, q2);     // y2 = q(9,2) * hh(2,2) + q(9,1) | ... | q(16,2) * h(2,2) + q(16,1)
	__m256 q3 = _mm256_load_ps(&q[16]);        // q3 = q(17,1) | ... | q(24,1)
	__m256 y3 = _mm256_FMA_ps(x3, h1, q3);     // y3 = q(17,2) * hh(2,2) + q(17,1) ... | q(24,2) * hh(2,2) + q(24,1)
//	__m256 q4 = _mm256_load_ps(&q[12]);
//	__m256 y4 = _mm256_FMA_ps(x4, h1, q4);
//	__m256 q5 = _mm256_load_ps(&q[16]);
//	__m256 y5 = _mm256_FMA_ps(x5, h1, q5);
//	__m256 q6 = _mm256_load_ps(&q[20]);
//	__m256 y6 = _mm256_FMA_ps(x6, h1, q6);
#else
	__m256 q1 = _mm256_load_ps(q);                               // q1 = q(1,1), q(2,1), q(3,1), q(4,1), q(5,1), q(6,1), q(7,1), q(8,1)
	__m256 y1 = _mm256_add_ps(q1, _mm256_mul_ps(x1, h1));        // y1 = q(1,2) * h(2,2) + q(1,1) | q(2,2) * h(2,2) + q(2,1) | .... | q(8,2) * h(2,2) + q(8,1)
	__m256 q2 = _mm256_load_ps(&q[8]);                           // q2 = q(9,1) | .... | q(16,1)
	__m256 y2 = _mm256_add_ps(q2, _mm256_mul_ps(x2, h1));        // y2 = q(9,2) * hh(2,2) + q(9,1) | ... | q(16,2) * h(2,2) + q(16,1)
	__m256 q3 = _mm256_load_ps(&q[16]);                          // q3 = q(17,1) | ... | q(24,1)
	__m256 y3 = _mm256_add_ps(q3, _mm256_mul_ps(x3, h1));        // y3 = q(17,2) * hh(2,2) + q(17,1) ... | q(24,2) * hh(2,2) + q(24,1)
//	__m256 q4 = _mm256_load_ps(&q[12]);
//	__m256 y4 = _mm256_add_ps(q4, _mm256_mul_ps(x4, h1));
//	__m256 q5 = _mm256_load_ps(&q[16]);
//	__m256 y5 = _mm256_add_ps(q5, _mm256_mul_ps(x5, h1));
//	__m256 q6 = _mm256_load_ps(&q[20]);
//	__m256 y6 = _mm256_add_ps(q6, _mm256_mul_ps(x6, h1));
#endif
	for(i = 2; i < nb; i++)
	{
		h1 = _mm256_broadcast_ss(&hh[i-1]);     //  h1 = hh(i-1,1) | ... | hh(i-1,1)
		h2 = _mm256_broadcast_ss(&hh[ldh+i]);   //  h2 = hh(i,2) | ... | hh(i,2)
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_load_ps(&q[i*ldq]);                    // q1 = q(1,i) | q(2,i) | q(3,i) | ... | q(8,i)
		x1 = _mm256_FMA_ps(q1, h1, x1);
		y1 = _mm256_FMA_ps(q1, h2, y1);
		q2 = _mm256_load_ps(&q[(i*ldq)+8]);
		x2 = _mm256_FMA_ps(q2, h1, x2);
		y2 = _mm256_FMA_ps(q2, h2, y2);
		q3 = _mm256_load_ps(&q[(i*ldq)+16]);
		x3 = _mm256_FMA_ps(q3, h1, x3);
		y3 = _mm256_FMA_ps(q3, h2, y3);
//		q4 = _mm256_load_ps(&q[(i*ldq)+12]);
//		x4 = _mm256_FMA_ps(q4, h1, x4);
//		y4 = _mm256_FMA_ps(q4, h2, y4);
//		q5 = _mm256_load_ps(&q[(i*ldq)+16]);
//		x5 = _mm256_FMA_ps(q5, h1, x5);
//		y5 = _mm256_FMA_ps(q5, h2, y5);
//		q6 = _mm256_load_ps(&q[(i*ldq)+20]);
//		x6 = _mm256_FMA_ps(q6, h1, x6);
//		y6 = _mm256_FMA_ps(q6, h2, y6);
#else
		q1 = _mm256_load_ps(&q[i*ldq]);                    // q1 = q(1,i) | q(2,i) | q(3,i) | ... | q(8,i)
		x1 = _mm256_add_ps(x1, _mm256_mul_ps(q1,h1));      // x1 = q(1,i) * hh(i-1,1) + x1 | ... | q(8,i) ** hh(i-1,1) * x1
		y1 = _mm256_add_ps(y1, _mm256_mul_ps(q1,h2));      // y1 = q(1,i) * hh(i,2) + y1 | ...
		q2 = _mm256_load_ps(&q[(i*ldq)+8]);
		x2 = _mm256_add_ps(x2, _mm256_mul_ps(q2,h1));
		y2 = _mm256_add_ps(y2, _mm256_mul_ps(q2,h2));
		q3 = _mm256_load_ps(&q[(i*ldq)+16]);
		x3 = _mm256_add_ps(x3, _mm256_mul_ps(q3,h1));
		y3 = _mm256_add_ps(y3, _mm256_mul_ps(q3,h2));
//		q4 = _mm256_load_ps(&q[(i*ldq)+12]);
//		x4 = _mm256_add_ps(x4, _mm256_mul_ps(q4,h1));
//		y4 = _mm256_add_ps(y4, _mm256_mul_ps(q4,h2));
//		q5 = _mm256_load_ps(&q[(i*ldq)+16]);
//		x5 = _mm256_add_ps(x5, _mm256_mul_ps(q5,h1));
//		y5 = _mm256_add_ps(y5, _mm256_mul_ps(q5,h2));
//		q6 = _mm256_load_ps(&q[(i*ldq)+20]);
//		x6 = _mm256_add_ps(x6, _mm256_mul_ps(q6,h1));
//		y6 = _mm256_add_ps(y6, _mm256_mul_ps(q6,h2));
#endif
	}
	h1 = _mm256_broadcast_ss(&hh[nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_load_ps(&q[nb*ldq]);
	x1 = _mm256_FMA_ps(q1, h1, x1);
	q2 = _mm256_load_ps(&q[(nb*ldq)+8]);
	x2 = _mm256_FMA_ps(q2, h1, x2);
	q3 = _mm256_load_ps(&q[(nb*ldq)+16]);
	x3 = _mm256_FMA_ps(q3, h1, x3);
//	q4 = _mm256_load_ps(&q[(nb*ldq)+12]);
//	x4 = _mm256_FMA_ps(q4, h1, x4);
//	q5 = _mm256_load_ps(&q[(nb*ldq)+16]);
//	x5 = _mm256_FMA_ps(q5, h1, x5);
//	q6 = _mm256_load_ps(&q[(nb*ldq)+20]);
//	x6 = _mm256_FMA_ps(q6, h1, x6);
#else
	q1 = _mm256_load_ps(&q[nb*ldq]);
	x1 = _mm256_add_ps(x1, _mm256_mul_ps(q1,h1));
	q2 = _mm256_load_ps(&q[(nb*ldq)+8]);
	x2 = _mm256_add_ps(x2, _mm256_mul_ps(q2,h1));
	q3 = _mm256_load_ps(&q[(nb*ldq)+16]);
	x3 = _mm256_add_ps(x3, _mm256_mul_ps(q3,h1));
//	q4 = _mm256_load_ps(&q[(nb*ldq)+12]);
//	x4 = _mm256_add_ps(x4, _mm256_mul_ps(q4,h1));
//	q5 = _mm256_load_ps(&q[(nb*ldq)+16]);
//	x5 = _mm256_add_ps(x5, _mm256_mul_ps(q5,h1));
//	q6 = _mm256_load_ps(&q[(nb*ldq)+20]);
//	x6 = _mm256_add_ps(x6, _mm256_mul_ps(q6,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [24 x nb+1]
	/////////////////////////////////////////////////////

	__m256 tau1 = _mm256_broadcast_ss(hh);
	__m256 tau2 = _mm256_broadcast_ss(&hh[ldh]);
	__m256 vs = _mm256_broadcast_ss(&s);


// carefull here

	h1 = _mm256_xor_ps(tau1, sign);
	x1 = _mm256_mul_ps(x1, h1);
	x2 = _mm256_mul_ps(x2, h1);
	x3 = _mm256_mul_ps(x3, h1);
//	x4 = _mm256_mul_ps(x4, h1);
//	x5 = _mm256_mul_ps(x5, h1);
//	x6 = _mm256_mul_ps(x6, h1);
	h1 = _mm256_xor_ps(tau2, sign);
	h2 = _mm256_mul_ps(h1, vs);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMA_ps(y1, h1, _mm256_mul_ps(x1,h2));
	y2 = _mm256_FMA_ps(y2, h1, _mm256_mul_ps(x2,h2));
	y3 = _mm256_FMA_ps(y3, h1, _mm256_mul_ps(x3,h2));
//	y4 = _mm256_FMA_ps(y4, h1, _mm256_mul_ps(x4,h2));
//	y5 = _mm256_FMA_ps(y5, h1, _mm256_mul_ps(x5,h2));
//	y6 = _mm256_FMA_ps(y6, h1, _mm256_mul_ps(x6,h2));
#else
	y1 = _mm256_add_ps(_mm256_mul_ps(y1,h1), _mm256_mul_ps(x1,h2));
	y2 = _mm256_add_ps(_mm256_mul_ps(y2,h1), _mm256_mul_ps(x2,h2));
	y3 = _mm256_add_ps(_mm256_mul_ps(y3,h1), _mm256_mul_ps(x3,h2));
//	y4 = _mm256_add_ps(_mm256_mul_ps(y4,h1), _mm256_mul_ps(x4,h2));
//	y5 = _mm256_add_ps(_mm256_mul_ps(y5,h1), _mm256_mul_ps(x5,h2));
//	y6 = _mm256_add_ps(_mm256_mul_ps(y6,h1), _mm256_mul_ps(x6,h2));
#endif

	q1 = _mm256_load_ps(q);
	q1 = _mm256_add_ps(q1, y1);
	_mm256_store_ps(q,q1);
	q2 = _mm256_load_ps(&q[8]);
	q2 = _mm256_add_ps(q2, y2);
	_mm256_store_ps(&q[8],q2);
	q3 = _mm256_load_ps(&q[16]);
	q3 = _mm256_add_ps(q3, y3);
	_mm256_store_ps(&q[16],q3);
//	q4 = _mm256_load_ps(&q[12]);
//	q4 = _mm256_add_ps(q4, y4);
//	_mm256_store_ps(&q[12],q4);
//	q5 = _mm256_load_ps(&q[16]);
//	q5 = _mm256_add_ps(q5, y5);
//	_mm256_store_ps(&q[16],q5);
//	q6 = _mm256_load_ps(&q[20]);
//	q6 = _mm256_add_ps(q6, y6);
//	_mm256_store_ps(&q[20],q6);

	h2 = _mm256_broadcast_ss(&hh[ldh+1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_load_ps(&q[ldq]);
	q1 = _mm256_add_ps(q1, _mm256_FMA_ps(y1, h2, x1));
	_mm256_store_ps(&q[ldq],q1);
	q2 = _mm256_load_ps(&q[ldq+8]);
	q2 = _mm256_add_ps(q2, _mm256_FMA_ps(y2, h2, x2));
	_mm256_store_ps(&q[ldq+8],q2);
	q3 = _mm256_load_ps(&q[ldq+16]);
	q3 = _mm256_add_ps(q3, _mm256_FMA_ps(y3, h2, x3));
	_mm256_store_ps(&q[ldq+16],q3);
//	q4 = _mm256_load_ps(&q[ldq+12]);
//	q4 = _mm256_add_ps(q4, _mm256_FMA_ps(y4, h2, x4));
//	_mm256_store_ps(&q[ldq+12],q4);
//	q5 = _mm256_load_ps(&q[ldq+16]);
//	q5 = _mm256_add_ps(q5, _mm256_FMA_ps(y5, h2, x5));
//	_mm256_store_ps(&q[ldq+16],q5);
//	q6 = _mm256_load_ps(&q[ldq+20]);
//	q6 = _mm256_add_ps(q6, _mm256_FMA_ps(y6, h2, x6));
//	_mm256_store_ps(&q[ldq+20],q6);
#else
	q1 = _mm256_load_ps(&q[ldq]);
	q1 = _mm256_add_ps(q1, _mm256_add_ps(x1, _mm256_mul_ps(y1, h2)));
	_mm256_store_ps(&q[ldq],q1);
	q2 = _mm256_load_ps(&q[ldq+8]);
	q2 = _mm256_add_ps(q2, _mm256_add_ps(x2, _mm256_mul_ps(y2, h2)));
	_mm256_store_ps(&q[ldq+8],q2);
	q3 = _mm256_load_ps(&q[ldq+16]);
	q3 = _mm256_add_ps(q3, _mm256_add_ps(x3, _mm256_mul_ps(y3, h2)));
	_mm256_store_ps(&q[ldq+16],q3);
//	q4 = _mm256_load_ps(&q[ldq+12]);
//	q4 = _mm256_add_ps(q4, _mm256_add_ps(x4, _mm256_mul_ps(y4, h2)));
//	_mm256_store_ps(&q[ldq+12],q4);
//	q5 = _mm256_load_ps(&q[ldq+16]);
//	q5 = _mm256_add_ps(q5, _mm256_add_ps(x5, _mm256_mul_ps(y5, h2)));
//	_mm256_store_ps(&q[ldq+16],q5);
//	q6 = _mm256_load_ps(&q[ldq+20]);
//	q6 = _mm256_add_ps(q6, _mm256_add_ps(x6, _mm256_mul_ps(y6, h2)));
//	_mm256_store_ps(&q[ldq+20],q6);
#endif

	for (i = 2; i < nb; i++)
	{
		h1 = _mm256_broadcast_ss(&hh[i-1]);
		h2 = _mm256_broadcast_ss(&hh[ldh+i]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_load_ps(&q[i*ldq]);
		q1 = _mm256_FMA_ps(x1, h1, q1);
		q1 = _mm256_FMA_ps(y1, h2, q1);
		_mm256_store_ps(&q[i*ldq],q1);
		q2 = _mm256_load_ps(&q[(i*ldq)+8]);
		q2 = _mm256_FMA_ps(x2, h1, q2);
		q2 = _mm256_FMA_ps(y2, h2, q2);
		_mm256_store_ps(&q[(i*ldq)+8],q2);
		q3 = _mm256_load_ps(&q[(i*ldq)+16]);
		q3 = _mm256_FMA_ps(x3, h1, q3);
		q3 = _mm256_FMA_ps(y3, h2, q3);
		_mm256_store_ps(&q[(i*ldq)+16],q3);
//		q4 = _mm256_load_ps(&q[(i*ldq)+12]);
//		q4 = _mm256_FMA_ps(x4, h1, q4);
//		q4 = _mm256_FMA_ps(y4, h2, q4);
//		_mm256_store_ps(&q[(i*ldq)+12],q4);
//		q5 = _mm256_load_ps(&q[(i*ldq)+16]);
///		q5 = _mm256_FMA_ps(x5, h1, q5);
//		q5 = _mm256_FMA_ps(y5, h2, q5);
//		_mm256_store_ps(&q[(i*ldq)+16],q5);
//		q6 = _mm256_load_ps(&q[(i*ldq)+20]);
//		q6 = _mm256_FMA_ps(x6, h1, q6);
//		q6 = _mm256_FMA_ps(y6, h2, q6);
//		_mm256_store_ps(&q[(i*ldq)+20],q6);
#else
		q1 = _mm256_load_ps(&q[i*ldq]);
		q1 = _mm256_add_ps(q1, _mm256_add_ps(_mm256_mul_ps(x1,h1), _mm256_mul_ps(y1, h2)));
		_mm256_store_ps(&q[i*ldq],q1);
		q2 = _mm256_load_ps(&q[(i*ldq)+8]);
		q2 = _mm256_add_ps(q2, _mm256_add_ps(_mm256_mul_ps(x2,h1), _mm256_mul_ps(y2, h2)));
		_mm256_store_ps(&q[(i*ldq)+8],q2);
		q3 = _mm256_load_ps(&q[(i*ldq)+16]);
		q3 = _mm256_add_ps(q3, _mm256_add_ps(_mm256_mul_ps(x3,h1), _mm256_mul_ps(y3, h2)));
		_mm256_store_ps(&q[(i*ldq)+16],q3);
//		q4 = _mm256_load_ps(&q[(i*ldq)+12]);
//		q4 = _mm256_add_ps(q4, _mm256_add_ps(_mm256_mul_ps(x4,h1), _mm256_mul_ps(y4, h2)));
//		_mm256_store_ps(&q[(i*ldq)+12],q4);
//		q5 = _mm256_load_ps(&q[(i*ldq)+16]);
//		q5 = _mm256_add_ps(q5, _mm256_add_ps(_mm256_mul_ps(x5,h1), _mm256_mul_ps(y5, h2)));
//		_mm256_store_ps(&q[(i*ldq)+16],q5);
//		q6 = _mm256_load_ps(&q[(i*ldq)+20]);
//		q6 = _mm256_add_ps(q6, _mm256_add_ps(_mm256_mul_ps(x6,h1), _mm256_mul_ps(y6, h2)));
//		_mm256_store_ps(&q[(i*ldq)+20],q6);
#endif
	}

	h1 = _mm256_broadcast_ss(&hh[nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_load_ps(&q[nb*ldq]);
	q1 = _mm256_FMA_ps(x1, h1, q1);
	_mm256_store_ps(&q[nb*ldq],q1);
	q2 = _mm256_load_ps(&q[(nb*ldq)+8]);
	q2 = _mm256_FMA_ps(x2, h1, q2);
	_mm256_store_ps(&q[(nb*ldq)+8],q2);
	q3 = _mm256_load_ps(&q[(nb*ldq)+16]);
	q3 = _mm256_FMA_ps(x3, h1, q3);
	_mm256_store_ps(&q[(nb*ldq)+16],q3);
//	q4 = _mm256_load_ps(&q[(nb*ldq)+12]);
///	q4 = _mm256_FMA_ps(x4, h1, q4);
//	_mm256_store_ps(&q[(nb*ldq)+12],q4);
//	q5 = _mm256_load_ps(&q[(nb*ldq)+16]);
//	q5 = _mm256_FMA_ps(x5, h1, q5);
//	_mm256_store_ps(&q[(nb*ldq)+16],q5);
//	q6 = _mm256_load_ps(&q[(nb*ldq)+20]);
//	q6 = _mm256_FMA_ps(x6, h1, q6);
//	_mm256_store_ps(&q[(nb*ldq)+20],q6);
#else
	q1 = _mm256_load_ps(&q[nb*ldq]);
	q1 = _mm256_add_ps(q1, _mm256_mul_ps(x1, h1));
	_mm256_store_ps(&q[nb*ldq],q1);
	q2 = _mm256_load_ps(&q[(nb*ldq)+8]);
	q2 = _mm256_add_ps(q2, _mm256_mul_ps(x2, h1));
	_mm256_store_ps(&q[(nb*ldq)+8],q2);
	q3 = _mm256_load_ps(&q[(nb*ldq)+16]);
	q3 = _mm256_add_ps(q3, _mm256_mul_ps(x3, h1));
	_mm256_store_ps(&q[(nb*ldq)+16],q3);
//	q4 = _mm256_load_ps(&q[(nb*ldq)+12]);
//	q4 = _mm256_add_ps(q4, _mm256_mul_ps(x4, h1));
//	_mm256_store_ps(&q[(nb*ldq)+12],q4);
//	q5 = _mm256_load_ps(&q[(nb*ldq)+16]);
//	q5 = _mm256_add_ps(q5, _mm256_mul_ps(x5, h1));
//	_mm256_store_ps(&q[(nb*ldq)+16],q5);
//	q6 = _mm256_load_ps(&q[(nb*ldq)+20]);
//	q6 = _mm256_add_ps(q6, _mm256_mul_ps(x6, h1));
//	_mm256_store_ps(&q[(nb*ldq)+20],q6);
#endif
}

/**
 * Unrolled kernel that computes
 * 16 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_16_AVX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [16 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	__m256 sign = (__m256)_mm256_set1_epi32(0x80000000);

	__m256 x1 = _mm256_load_ps(&q[ldq]);
	__m256 x2 = _mm256_load_ps(&q[ldq+8]);
//	__m256 x3 = _mm256_load_ps(&q[ldq+16]);
//	__m256 x4 = _mm256_load_ps(&q[ldq+12]);

	__m256 h1 = _mm256_broadcast_ss(&hh[ldh+1]);
	__m256 h2;

#ifdef __ELPA_USE_FMA__
	__m256 q1 = _mm256_load_ps(q);
	__m256 y1 = _mm256_FMA_ps(x1, h1, q1);
	__m256 q2 = _mm256_load_ps(&q[8]);
	__m256 y2 = _mm256_FMA_ps(x2, h1, q2);
//	__m256 q3 = _mm256_load_ps(&q[16]);
//	__m256 y3 = _mm256_FMA_ps(x3, h1, q3);
//	__m256 q4 = _mm256_load_ps(&q[12]);
//	__m256 y4 = _mm256_FMA_ps(x4, h1, q4);
#else
	__m256 q1 = _mm256_load_ps(q);
	__m256 y1 = _mm256_add_ps(q1, _mm256_mul_ps(x1, h1));
	__m256 q2 = _mm256_load_ps(&q[8]);
	__m256 y2 = _mm256_add_ps(q2, _mm256_mul_ps(x2, h1));
//	__m256 q3 = _mm256_load_ps(&q[16]);
//	__m256 y3 = _mm256_add_ps(q3, _mm256_mul_ps(x3, h1));
//	__m256 q4 = _mm256_load_ps(&q[12]);
//	__m256 y4 = _mm256_add_ps(q4, _mm256_mul_ps(x4, h1));
#endif

	for(i = 2; i < nb; i++)
	{
		h1 = _mm256_broadcast_ss(&hh[i-1]);
		h2 = _mm256_broadcast_ss(&hh[ldh+i]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_load_ps(&q[i*ldq]);
		x1 = _mm256_FMA_ps(q1, h1, x1);
		y1 = _mm256_FMA_ps(q1, h2, y1);
		q2 = _mm256_load_ps(&q[(i*ldq)+8]);
		x2 = _mm256_FMA_ps(q2, h1, x2);
		y2 = _mm256_FMA_ps(q2, h2, y2);
//		q3 = _mm256_load_ps(&q[(i*ldq)+8]);
//		x3 = _mm256_FMA_ps(q3, h1, x3);
//		y3 = _mm256_FMA_ps(q3, h2, y3);
//		q4 = _mm256_load_ps(&q[(i*ldq)+12]);
//		x4 = _mm256_FMA_ps(q4, h1, x4);
//		y4 = _mm256_FMA_ps(q4, h2, y4);
#else
		q1 = _mm256_load_ps(&q[i*ldq]);
		x1 = _mm256_add_ps(x1, _mm256_mul_ps(q1,h1));
		y1 = _mm256_add_ps(y1, _mm256_mul_ps(q1,h2));
		q2 = _mm256_load_ps(&q[(i*ldq)+8]);
		x2 = _mm256_add_ps(x2, _mm256_mul_ps(q2,h1));
		y2 = _mm256_add_ps(y2, _mm256_mul_ps(q2,h2));
//		q3 = _mm256_load_ps(&q[(i*ldq)+8]);
//		x3 = _mm256_add_ps(x3, _mm256_mul_ps(q3,h1));
//		y3 = _mm256_add_ps(y3, _mm256_mul_ps(q3,h2));
//		q4 = _mm256_load_ps(&q[(i*ldq)+12]);
//		x4 = _mm256_add_ps(x4, _mm256_mul_ps(q4,h1));
//		y4 = _mm256_add_ps(y4, _mm256_mul_ps(q4,h2));
#endif
	}

	h1 = _mm256_broadcast_ss(&hh[nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_load_ps(&q[nb*ldq]);
	x1 = _mm256_FMA_ps(q1, h1, x1);
	q2 = _mm256_load_ps(&q[(nb*ldq)+8]);
	x2 = _mm256_FMA_ps(q2, h1, x2);
//	q3 = _mm256_load_ps(&q[(nb*ldq)+8]);
//	x3 = _mm256_FMA_ps(q3, h1, x3);
//	q4 = _mm256_load_ps(&q[(nb*ldq)+12]);
//	x4 = _mm256_FMA_ps(q4, h1, x4);
#else
	q1 = _mm256_load_ps(&q[nb*ldq]);
	x1 = _mm256_add_ps(x1, _mm256_mul_ps(q1,h1));
	q2 = _mm256_load_ps(&q[(nb*ldq)+8]);
	x2 = _mm256_add_ps(x2, _mm256_mul_ps(q2,h1));
//	q3 = _mm256_load_ps(&q[(nb*ldq)+8]);
//	x3 = _mm256_add_ps(x3, _mm256_mul_ps(q3,h1));
//	q4 = _mm256_load_ps(&q[(nb*ldq)+12]);
//	x4 = _mm256_add_ps(x4, _mm256_mul_ps(q4,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [16 x nb+1]
	/////////////////////////////////////////////////////

	__m256 tau1 = _mm256_broadcast_ss(hh);
	__m256 tau2 = _mm256_broadcast_ss(&hh[ldh]);
	__m256 vs = _mm256_broadcast_ss(&s);


// carefulle

	h1 = _mm256_xor_ps(tau1, sign);
	x1 = _mm256_mul_ps(x1, h1);
	x2 = _mm256_mul_ps(x2, h1);
//	x3 = _mm256_mul_ps(x3, h1);
//	x4 = _mm256_mul_ps(x4, h1);
	h1 = _mm256_xor_ps(tau2, sign);
	h2 = _mm256_mul_ps(h1, vs);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMA_ps(y1, h1, _mm256_mul_ps(x1,h2));
	y2 = _mm256_FMA_ps(y2, h1, _mm256_mul_ps(x2,h2));
//	y3 = _mm256_FMA_ps(y3, h1, _mm256_mul_ps(x3,h2));
//	y4 = _mm256_FMA_ps(y4, h1, _mm256_mul_ps(x4,h2));
#else
	y1 = _mm256_add_ps(_mm256_mul_ps(y1,h1), _mm256_mul_ps(x1,h2));
	y2 = _mm256_add_ps(_mm256_mul_ps(y2,h1), _mm256_mul_ps(x2,h2));
//	y3 = _mm256_add_ps(_mm256_mul_ps(y3,h1), _mm256_mul_ps(x3,h2));
//	y4 = _mm256_add_ps(_mm256_mul_ps(y4,h1), _mm256_mul_ps(x4,h2));
#endif

	q1 = _mm256_load_ps(q);
	q1 = _mm256_add_ps(q1, y1);
	_mm256_store_ps(q,q1);
	q2 = _mm256_load_ps(&q[8]);
	q2 = _mm256_add_ps(q2, y2);
	_mm256_store_ps(&q[8],q2);
//	q3 = _mm256_load_psa(&q[8]);
//	q3 = _mm256_add_ps(q3, y3);
//	_mm256_store_ps(&q[8],q3);
//	q4 = _mm256_load_ps(&q[12]);
//	q4 = _mm256_add_ps(q4, y4);
//	_mm256_store_ps(&q[12],q4);

	h2 = _mm256_broadcast_ss(&hh[ldh+1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_load_ps(&q[ldq]);
	q1 = _mm256_add_ps(q1, _mm256_FMA_ps(y1, h2, x1));
	_mm256_store_ps(&q[ldq],q1);
	q2 = _mm256_load_ps(&q[ldq+8]);
	q2 = _mm256_add_ps(q2, _mm256_FMA_ps(y2, h2, x2));
	_mm256_store_ps(&q[ldq+8],q2);
//	q3 = _mm256_load_ps(&q[ldq+8]);
//	q3 = _mm256_add_ps(q3, _mm256_FMA_ps(y3, h2, x3));
//	_mm256_store_ps(&q[ldq+8],q3);
//	q4 = _mm256_load_ps(&q[ldq+12]);
//	q4 = _mm256_add_ps(q4, _mm256_FMA_ps(y4, h2, x4));
//	_mm256_store_ps(&q[ldq+12],q4);
#else
	q1 = _mm256_load_ps(&q[ldq]);
	q1 = _mm256_add_ps(q1, _mm256_add_ps(x1, _mm256_mul_ps(y1, h2)));
	_mm256_store_ps(&q[ldq],q1);
	q2 = _mm256_load_ps(&q[ldq+8]);
	q2 = _mm256_add_ps(q2, _mm256_add_ps(x2, _mm256_mul_ps(y2, h2)));
	_mm256_store_ps(&q[ldq+8],q2);
//	q3 = _mm256_load_ps(&q[ldq+8]);
//	q3 = _mm256_add_ps(q3, _mm256_add_ps(x3, _mm256_mul_ps(y3, h2)));
//	_mm256_store_ps(&q[ldq+8],q3);
//	q4 = _mm256_load_ps(&q[ldq+12]);
//	q4 = _mm256_add_ps(q4, _mm256_add_ps(x4, _mm256_mul_ps(y4, h2)));
//	_mm256_store_ps(&q[ldq+12],q4);
#endif

	for (i = 2; i < nb; i++)
	{
		h1 = _mm256_broadcast_ss(&hh[i-1]);
		h2 = _mm256_broadcast_ss(&hh[ldh+i]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_load_ps(&q[i*ldq]);
		q1 = _mm256_FMA_ps(x1, h1, q1);
		q1 = _mm256_FMA_ps(y1, h2, q1);
		_mm256_store_ps(&q[i*ldq],q1);
		q2 = _mm256_load_ps(&q[(i*ldq)+8]);
		q2 = _mm256_FMA_ps(x2, h1, q2);
		q2 = _mm256_FMA_ps(y2, h2, q2);
		_mm256_store_ps(&q[(i*ldq)+8],q2);
//		q3 = _mm256_load_ps(&q[(i*ldq)+8]);
//		q3 = _mm256_FMA_ps(x3, h1, q3);
//		q3 = _mm256_FMA_ps(y3, h2, q3);
//		_mm256_store_ps(&q[(i*ldq)+8],q3);
//		q4 = _mm256_load_ps(&q[(i*ldq)+12]);
//		q4 = _mm256_FMA_ps(x4, h1, q4);
//		q4 = _mm256_FMA_ps(y4, h2, q4);
//		_mm256_store_ps(&q[(i*ldq)+12],q4);
#else
		q1 = _mm256_load_ps(&q[i*ldq]);
		q1 = _mm256_add_ps(q1, _mm256_add_ps(_mm256_mul_ps(x1,h1), _mm256_mul_ps(y1, h2)));
		_mm256_store_ps(&q[i*ldq],q1);
		q2 = _mm256_load_ps(&q[(i*ldq)+8]);
		q2 = _mm256_add_ps(q2, _mm256_add_ps(_mm256_mul_ps(x2,h1), _mm256_mul_ps(y2, h2)));
		_mm256_store_ps(&q[(i*ldq)+8],q2);
//		q3 = _mm256_load_ps(&q[(i*ldq)+8]);
//		q3 = _mm256_add_ps(q3, _mm256_add_ps(_mm256_mul_ps(x3,h1), _mm256_mul_ps(y3, h2)));
//		_mm256_store_ps(&q[(i*ldq)+8],q3);
//		q4 = _mm256_load_ps(&q[(i*ldq)+12]);
//		q4 = _mm256_add_ps(q4, _mm256_add_ps(_mm256_mul_ps(x4,h1), _mm256_mul_ps(y4, h2)));
//		_mm256_store_ps(&q[(i*ldq)+12],q4);
#endif
	}

	h1 = _mm256_broadcast_ss(&hh[nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_load_ps(&q[nb*ldq]);
	q1 = _mm256_FMA_ps(x1, h1, q1);
	_mm256_store_ps(&q[nb*ldq],q1);
	q2 = _mm256_load_ps(&q[(nb*ldq)+8]);
	q2 = _mm256_FMA_ps(x2, h1, q2);
	_mm256_store_ps(&q[(nb*ldq)+8],q2);
//	q3 = _mm256_load_ps(&q[(nb*ldq)+8]);
//	q3 = _mm256_FMA_ps(x3, h1, q3);
//	_mm256_store_ps(&q[(nb*ldq)+8],q3);
//	q4 = _mm256_load_ps(&q[(nb*ldq)+12]);
//	q4 = _mm256_FMA_ps(x4, h1, q4);
//	_mm256_store_ps(&q[(nb*ldq)+12],q4);
#else
	q1 = _mm256_load_ps(&q[nb*ldq]);
	q1 = _mm256_add_ps(q1, _mm256_mul_ps(x1, h1));
	_mm256_store_ps(&q[nb*ldq],q1);
	q2 = _mm256_load_ps(&q[(nb*ldq)+8]);
	q2 = _mm256_add_ps(q2, _mm256_mul_ps(x2, h1));
	_mm256_store_ps(&q[(nb*ldq)+8],q2);
//	q3 = _mm256_load_ps(&q[(nb*ldq)+8]);
//	q3 = _mm256_add_ps(q3, _mm256_mul_ps(x3, h1));
//	_mm256_store_ps(&q[(nb*ldq)+8],q3);
//	q4 = _mm256_load_ps(&q[(nb*ldq)+12]);
//	q4 = _mm256_add_ps(q4, _mm256_mul_ps(x4, h1));
//	_mm256_store_ps(&q[(nb*ldq)+12],q4);
#endif
}

/**
 * Unrolled kernel that computes
 * 8 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_8_AVX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [8 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	__m256 sign = (__m256)_mm256_set1_epi32(0x80000000);

	__m256 x1 = _mm256_load_ps(&q[ldq]);
//	__m256 x2 = _mm256_load_ps(&q[ldq+8]);

	__m256 h1 = _mm256_broadcast_ss(&hh[ldh+1]);
	__m256 h2;

#ifdef __ELPA_USE_FMA__
	__m256 q1 = _mm256_load_ps(q);
	__m256 y1 = _mm256_FMA_ps(x1, h1, q1);
//	__m256 q2 = _mm256_load_ps(&q[4]);
//	__m256 y2 = _mm256_FMA_ps(x2, h1, q2);
#else
	__m256 q1 = _mm256_load_ps(q);
	__m256 y1 = _mm256_add_ps(q1, _mm256_mul_ps(x1, h1));
//	__m256 q2 = _mm256_load_ps(&q[4]);
//	__m256 y2 = _mm256_add_ps(q2, _mm256_mul_ps(x2, h1));
#endif

	for(i = 2; i < nb; i++)
	{
		h1 = _mm256_broadcast_ss(&hh[i-1]);
		h2 = _mm256_broadcast_ss(&hh[ldh+i]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_load_ps(&q[i*ldq]);
		x1 = _mm256_FMA_ps(q1, h1, x1);
		y1 = _mm256_FMA_ps(q1, h2, y1);
//		q2 = _mm256_load_ps(&q[(i*ldq)+4]);
//		x2 = _mm256_FMA_ps(q2, h1, x2);
//		y2 = _mm256_FMA_ps(q2, h2, y2);
#else
		q1 = _mm256_load_ps(&q[i*ldq]);
		x1 = _mm256_add_ps(x1, _mm256_mul_ps(q1,h1));
		y1 = _mm256_add_ps(y1, _mm256_mul_ps(q1,h2));
//		q2 = _mm256_load_ps(&q[(i*ldq)+4]);
//		x2 = _mm256_add_ps(x2, _mm256_mul_ps(q2,h1));
//		y2 = _mm256_add_ps(y2, _mm256_mul_ps(q2,h2));
#endif
	}

	h1 = _mm256_broadcast_ss(&hh[nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_load_ps(&q[nb*ldq]);
	x1 = _mm256_FMA_ps(q1, h1, x1);
//	q2 = _mm256_load_ps(&q[(nb*ldq)+4]);
//	x2 = _mm256_FMA_ps(q2, h1, x2);
#else
	q1 = _mm256_load_ps(&q[nb*ldq]);
	x1 = _mm256_add_ps(x1, _mm256_mul_ps(q1,h1));
//	q2 = _mm256_load_ps(&q[(nb*ldq)+4]);
//	x2 = _mm256_add_ps(x2, _mm256_mul_ps(q2,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [8 x nb+1]
	/////////////////////////////////////////////////////

	__m256 tau1 = _mm256_broadcast_ss(hh);
	__m256 tau2 = _mm256_broadcast_ss(&hh[ldh]);
	__m256 vs = _mm256_broadcast_ss(&s);

// carefulle

	h1 = _mm256_xor_ps(tau1, sign);
	x1 = _mm256_mul_ps(x1, h1);
//	x2 = _mm256_mul_ps(x2, h1);
	h1 = _mm256_xor_ps(tau2, sign);
	h2 = _mm256_mul_ps(h1, vs);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMA_ps(y1, h1, _mm256_mul_ps(x1,h2));
//	y2 = _mm256_FMA_ps(y2, h1, _mm256_mul_ps(x2,h2));
#else
	y1 = _mm256_add_ps(_mm256_mul_ps(y1,h1), _mm256_mul_ps(x1,h2));
//	y2 = _mm256_add_ps(_mm256_mul_ps(y2,h1), _mm256_mul_ps(x2,h2));
#endif

	q1 = _mm256_load_ps(q);
	q1 = _mm256_add_ps(q1, y1);
	_mm256_store_ps(q,q1);
//	q2 = _mm256_load_ps(&q[4]);
//	q2 = _mm256_add_ps(q2, y2);
//	_mm256_store_ps(&q[4],q2);

	h2 = _mm256_broadcast_ss(&hh[ldh+1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_load_ps(&q[ldq]);
	q1 = _mm256_add_ps(q1, _mm256_FMA_ps(y1, h2, x1));
	_mm256_store_ps(&q[ldq],q1);
//	q2 = _mm256_load_ps(&q[ldq+4]);
//	q2 = _mm256_add_ps(q2, _mm256_FMA_ps(y2, h2, x2));
//	_mm256_store_ps(&q[ldq+4],q2);
#else
	q1 = _mm256_load_ps(&q[ldq]);
	q1 = _mm256_add_ps(q1, _mm256_add_ps(x1, _mm256_mul_ps(y1, h2)));
	_mm256_store_ps(&q[ldq],q1);
//	q2 = _mm256_load_ps(&q[ldq+4]);
//	q2 = _mm256_add_ps(q2, _mm256_add_ps(x2, _mm256_mul_ps(y2, h2)));
//	_mm256_store_ps(&q[ldq+4],q2);
#endif

	for (i = 2; i < nb; i++)
	{
		h1 = _mm256_broadcast_ss(&hh[i-1]);
		h2 = _mm256_broadcast_ss(&hh[ldh+i]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_load_ps(&q[i*ldq]);
		q1 = _mm256_FMA_ps(x1, h1, q1);
		q1 = _mm256_FMA_ps(y1, h2, q1);
		_mm256_store_ps(&q[i*ldq],q1);
//		q2 = _mm256_load_ps(&q[(i*ldq)+4]);
//		q2 = _mm256_FMA_ps(x2, h1, q2);
//		q2 = _mm256_FMA_ps(y2, h2, q2);
//		_mm256_store_ps(&q[(i*ldq)+4],q2);
#else
		q1 = _mm256_load_ps(&q[i*ldq]);
		q1 = _mm256_add_ps(q1, _mm256_add_ps(_mm256_mul_ps(x1,h1), _mm256_mul_ps(y1, h2)));
		_mm256_store_ps(&q[i*ldq],q1);
//		q2 = _mm256_load_ps(&q[(i*ldq)+4]);
//		q2 = _mm256_add_ps(q2, _mm256_add_ps(_mm256_mul_ps(x2,h1), _mm256_mul_ps(y2, h2)));
//		_mm256_store_ps(&q[(i*ldq)+4],q2);
#endif
	}

	h1 = _mm256_broadcast_ss(&hh[nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_load_ps(&q[nb*ldq]);
	q1 = _mm256_FMA_ps(x1, h1, q1);
	_mm256_store_ps(&q[nb*ldq],q1);
//	q2 = _mm256_load_ps(&q[(nb*ldq)+4]);
//	q2 = _mm256_FMA_ps(x2, h1, q2);
//	_mm256_store_ps(&q[(nb*ldq)+4],q2);
#else
	q1 = _mm256_load_ps(&q[nb*ldq]);
	q1 = _mm256_add_ps(q1, _mm256_mul_ps(x1, h1));
	_mm256_store_ps(&q[nb*ldq],q1);
//	q2 = _mm256_load_ps(&q[(nb*ldq)+4]);
//	q2 = _mm256_add_ps(q2, _mm256_mul_ps(x2, h1));
//	_mm256_store_ps(&q[(nb*ldq)+4],q2);
#endif
}

/**
 * Unrolled kernel that computes
 * 4 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
__forceinline void hh_trafo_kernel_4_sse_instead_of_avx_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
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

/**
 * Unrolled kernel that computes
 * 4 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_4_AVX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	__m256 sign = (__m256)_mm256_set1_epi32(0x80000000);

	__m256 x1 = _mm256_castps128_ps256(_mm_load_ps(&q[ldq]));

	__m256 h1 = _mm256_broadcast_ss(&hh[ldh+1]);
	__m256 h2;

#ifdef __ELPA_USE_FMA__
	__m256 q1 = _mm256_castps128_ps256(_mm_load_ps(q));
	__m256 y1 = _mm256_FMA_ps(x1, h1, q1);
#else
	__m256 q1 = _mm256_castps128_ps256(_mm_load_ps(q));
	__m256 y1 = _mm256_add_ps(q1, _mm256_mul_ps(x1, h1));
#endif

	for(i = 2; i < nb; i++)
	{
		h1 = _mm256_broadcast_ss(&hh[i-1]);
		h2 = _mm256_broadcast_ss(&hh[ldh+i]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_castps128_ps256(_mm_load_ps(&q[i*ldq]));
		x1 = _mm256_FMA_ps(q1, h1, x1);
		y1 = _mm256_FMA_ps(q1, h2, y1);
#else
		q1 = _mm256_castps128_ps256(_mm_load_ps(&q[i*ldq]));
		x1 = _mm256_add_ps(x1, _mm256_mul_ps(q1,h1));
		y1 = _mm256_add_ps(y1, _mm256_mul_ps(q1,h2));
#endif
	}

	h1 = _mm256_broadcast_ss(&hh[nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[nb*ldq]));
	x1 = _mm256_FMA_ps(q1, h1, x1);
#else
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[nb*ldq]));
	x1 = _mm256_add_ps(x1, _mm256_mul_ps(q1,h1));
#endif

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [4 x nb+1]
	/////////////////////////////////////////////////////

	__m256 tau1 = _mm256_broadcast_ss(hh);
	__m256 tau2 = _mm256_broadcast_ss(&hh[ldh]);
	__m256 vs = _mm256_broadcast_ss(&s);

	h1 = _mm256_xor_ps(tau1, sign);
	x1 = _mm256_mul_ps(x1, h1);
	h1 = _mm256_xor_ps(tau2, sign);
	h2 = _mm256_mul_ps(h1, vs);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMA_ps(y1, h1, _mm256_mul_ps(x1,h2));
#else
	y1 = _mm256_add_ps(_mm256_mul_ps(y1,h1), _mm256_mul_ps(x1,h2));
#endif

	q1 = _mm256_castps128_ps256(_mm_load_ps(q));
	q1 = _mm256_add_ps(q1, y1);
	_mm_store_ps(q, _mm256_castps256_ps128(q1));
//	_mm256_store_ps(q,q1);

	h2 = _mm256_broadcast_ss(&hh[ldh+1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[ldq]));
	q1 = _mm256_add_ps(q1, _mm256_FMA_ps(y1, h2, x1));
	_mm_store_ps(&q[ldq], _mm256_castps256_ps128(q1));
//	_mm256_store_ps(&q[ldq],q1);
#else
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[ldq]));
	q1 = _mm256_add_ps(q1, _mm256_add_ps(x1, _mm256_mul_ps(y1, h2)));
	_mm_store_ps(&q[ldq], _mm256_castps256_ps128(q1));

//	_mm256_store_ps(&q[ldq],q1);
#endif

	for (i = 2; i < nb; i++)
	{
		h1 = _mm256_broadcast_ss(&hh[i-1]);
		h2 = _mm256_broadcast_ss(&hh[ldh+i]);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_castps128_ps256(_mm_load_ps(&q[i*ldq]));
		q1 = _mm256_FMA_ps(x1, h1, q1);
		q1 = _mm256_FMA_ps(y1, h2, q1);
		_mm_store_ps(&q[i*ldq], _mm256_castps256_ps128(q1));
//		_mm256_store_ps(&q[i*ldq],q1);
#else
		q1 = _mm256_castps128_ps256(_mm_load_ps(&q[i*ldq]));
		q1 = _mm256_add_ps(q1, _mm256_add_ps(_mm256_mul_ps(x1,h1), _mm256_mul_ps(y1, h2)));
		_mm_store_ps(&q[i*ldq], _mm256_castps256_ps128(q1));
//		_mm256_store_ps(&q[i*ldq],q1);
#endif
	}

	h1 = _mm256_broadcast_ss(&hh[nb-1]);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[nb*ldq]));
	q1 = _mm256_FMA_ps(x1, h1, q1);
	_mm_store_ps(&q[nb*ldq], _mm256_castps256_ps128(q1));
//	_mm256_store_ps(&q[nb*ldq],q1);
#else
	q1 = _mm256_castps128_ps256(_mm_load_ps(&q[nb*ldq]));
	q1 = _mm256_add_ps(q1, _mm256_mul_ps(x1, h1));
	_mm_store_ps(&q[nb*ldq], _mm256_castps256_ps128(q1));
//	_mm256_store_ps(&q[nb*ldq],q1);
#endif
}

