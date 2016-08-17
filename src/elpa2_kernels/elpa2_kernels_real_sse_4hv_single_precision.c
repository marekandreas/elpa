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
__forceinline void hh_trafo_kernel_4_SSE_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);
__forceinline void hh_trafo_kernel_8_SSE_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);
__forceinline void hh_trafo_kernel_12_SSE_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);

void quad_hh_trafo_real_sse_4hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);

/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine quad_hh_trafo_real_sse_4hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="quad_hh_trafo_real_sse_4hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

void quad_hh_trafo_real_sse_4hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;

	// calculating scalar products to compute
	// 4 householder vectors simultaneously
	float s_1_2 = hh[(ldh)+1];
	float s_1_3 = hh[(ldh*2)+2];
	float s_2_3 = hh[(ldh*2)+1];
	float s_1_4 = hh[(ldh*3)+3];
	float s_2_4 = hh[(ldh*3)+2];
	float s_3_4 = hh[(ldh*3)+1];

	// calculate scalar product of first and fourth householder vector
	// loop counter = 2
	s_1_2 += hh[2-1] * hh[(2+ldh)];
	s_2_3 += hh[(ldh)+2-1] * hh[2+(ldh*2)];
	s_3_4 += hh[(ldh*2)+2-1] * hh[2+(ldh*3)];

	// loop counter = 3
	s_1_2 += hh[3-1] * hh[(3+ldh)];
	s_2_3 += hh[(ldh)+3-1] * hh[3+(ldh*2)];
	s_3_4 += hh[(ldh*2)+3-1] * hh[3+(ldh*3)];

	s_1_3 += hh[3-2] * hh[3+(ldh*2)];
	s_2_4 += hh[(ldh*1)+3-2] * hh[3+(ldh*3)];

	#pragma ivdep
	for (i = 4; i < nb; i++)
	{
		s_1_2 += hh[i-1] * hh[(i+ldh)];
		s_2_3 += hh[(ldh)+i-1] * hh[i+(ldh*2)];
		s_3_4 += hh[(ldh*2)+i-1] * hh[i+(ldh*3)];

		s_1_3 += hh[i-2] * hh[i+(ldh*2)];
		s_2_4 += hh[(ldh*1)+i-2] * hh[i+(ldh*3)];

		s_1_4 += hh[i-3] * hh[i+(ldh*3)];
	}


	// Production level kernel calls with padding
	for (i = 0; i < nq-8; i+=12)
	{
		hh_trafo_kernel_12_SSE_4hv_single(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);

	}
	if (nq == i)
	{
		return;
	}
	if (nq-i ==8)
	{
		hh_trafo_kernel_8_SSE_4hv_single(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
	}
	else
	{
		hh_trafo_kernel_4_SSE_4hv_single(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
	}
}
/**
 * Unrolled kernel that computes
 * 12 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_12_SSE_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [6 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m128 a1_1 = _mm_load_ps(&q[ldq*3]);
	__m128 a2_1 = _mm_load_ps(&q[ldq*2]);
	__m128 a3_1 = _mm_load_ps(&q[ldq]);
	__m128 a4_1 = _mm_load_ps(&q[0]);      // q(1,1) | q(2,1) | q(3,1) q(4,1)

	//careful here
//	__m128 h_2_1 = _mm_loaddup_pd(&hh[ldh+1]);      // hh(2,2) and duplicate
//        __m128 x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+1]));        // load hh(2,2) , hh(3,2) | hh(2,2) , hh(3,2) in double precision representations
        __m128 h_2_1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+1])) ); // h_2_1 contains four times hh[ldh+1]
	__m128 h_3_2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+1])));
	__m128 h_3_1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+2])));
	__m128 h_4_3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+1])));
	__m128 h_4_2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+2])));
	__m128 h_4_1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+3])));

//	__m128 h_3_2 = _mm_loaddup_pd(&hh[(ldh*2)+1]);
//	__m128 h_3_1 = _mm_loaddup_pd(&hh[(ldh*2)+2]);
//	__m128 h_4_3 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
//	__m128 h_4_2 = _mm_loaddup_pd(&hh[(ldh*3)+2]);
//	__m128 h_4_1 = _mm_loaddup_pd(&hh[(ldh*3)+3]);

	register __m128 w1 = _mm_add_ps(a4_1, _mm_mul_ps(a3_1, h_4_3));
	w1 = _mm_add_ps(w1, _mm_mul_ps(a2_1, h_4_2));
	w1 = _mm_add_ps(w1, _mm_mul_ps(a1_1, h_4_1));
	register __m128 z1 = _mm_add_ps(a3_1, _mm_mul_ps(a2_1, h_3_2));
	z1 = _mm_add_ps(z1, _mm_mul_ps(a1_1, h_3_1));
	register __m128 y1 = _mm_add_ps(a2_1, _mm_mul_ps(a1_1, h_2_1));
	register __m128 x1 = a1_1;

	__m128 a1_2 = _mm_load_ps(&q[(ldq*3)+4]);
	__m128 a2_2 = _mm_load_ps(&q[(ldq*2)+4]);
	__m128 a3_2 = _mm_load_ps(&q[ldq+4]);
	__m128 a4_2 = _mm_load_ps(&q[0+4]);       // q(5,1) | ... q(8,1)

	register __m128 w2 = _mm_add_ps(a4_2, _mm_mul_ps(a3_2, h_4_3));
	w2 = _mm_add_ps(w2, _mm_mul_ps(a2_2, h_4_2));
	w2 = _mm_add_ps(w2, _mm_mul_ps(a1_2, h_4_1));
	register __m128 z2 = _mm_add_ps(a3_2, _mm_mul_ps(a2_2, h_3_2));
	z2 = _mm_add_ps(z2, _mm_mul_ps(a1_2, h_3_1));
	register __m128 y2 = _mm_add_ps(a2_2, _mm_mul_ps(a1_2, h_2_1));
	register __m128 x2 = a1_2;

	__m128 a1_3 = _mm_load_ps(&q[(ldq*3)+8]);
	__m128 a2_3 = _mm_load_ps(&q[(ldq*2)+8]);
	__m128 a3_3 = _mm_load_ps(&q[ldq+8]);
	__m128 a4_3 = _mm_load_ps(&q[0+8]);    // q(9,1) | .. | q(12,1)

	register __m128 w3 = _mm_add_ps(a4_3, _mm_mul_ps(a3_3, h_4_3));
	w3 = _mm_add_ps(w3, _mm_mul_ps(a2_3, h_4_2));
	w3 = _mm_add_ps(w3, _mm_mul_ps(a1_3, h_4_1));
	register __m128 z3 = _mm_add_ps(a3_3, _mm_mul_ps(a2_3, h_3_2));
	z3 = _mm_add_ps(z3, _mm_mul_ps(a1_3, h_3_1));
	register __m128 y3 = _mm_add_ps(a2_3, _mm_mul_ps(a1_3, h_2_1));
	register __m128 x3 = a1_3;

	__m128 q1;
	__m128 q2;
	__m128 q3;

	__m128 h1;
	__m128 h2;
	__m128 h3;
	__m128 h4;

	for(i = 4; i < nb; i++)
	{
	        h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[i-3])));
//		h1 = _mm_loaddup_pd(&hh[i-3]);
		q1 = _mm_load_ps(&q[i*ldq]);
		q2 = _mm_load_ps(&q[(i*ldq)+4]);
		q3 = _mm_load_ps(&q[(i*ldq)+8]);

		x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
		x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
		x3 = _mm_add_ps(x3, _mm_mul_ps(q3,h1));
	        h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+i-2])));
//		h2 = _mm_loaddup_pd(&hh[ldh+i-2]);

		y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
		y2 = _mm_add_ps(y2, _mm_mul_ps(q2,h2));
		y3 = _mm_add_ps(y3, _mm_mul_ps(q3,h2));
	        h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+i-1])));
//		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-1]);

		z1 = _mm_add_ps(z1, _mm_mul_ps(q1,h3));
		z2 = _mm_add_ps(z2, _mm_mul_ps(q2,h3));
		z3 = _mm_add_ps(z3, _mm_mul_ps(q3,h3));
	        h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+i])));
//		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i]);

		w1 = _mm_add_ps(w1, _mm_mul_ps(q1,h4));
		w2 = _mm_add_ps(w2, _mm_mul_ps(q2,h4));
		w3 = _mm_add_ps(w3, _mm_mul_ps(q3,h4));
	}
	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-3])));
//	h1 = _mm_loaddup_pd(&hh[nb-3]);

	q1 = _mm_load_ps(&q[nb*ldq]);
	q2 = _mm_load_ps(&q[(nb*ldq)+4]);
	q3 = _mm_load_ps(&q[(nb*ldq)+8]);

	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
	x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
	x3 = _mm_add_ps(x3, _mm_mul_ps(q3,h1));
	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+nb-2])));
//	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);

	y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
	y2 = _mm_add_ps(y2, _mm_mul_ps(q2,h2));
	y3 = _mm_add_ps(y3, _mm_mul_ps(q3,h2));
	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+nb-1])));
//	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);

	z1 = _mm_add_ps(z1, _mm_mul_ps(q1,h3));
	z2 = _mm_add_ps(z2, _mm_mul_ps(q2,h3));
	z3 = _mm_add_ps(z3, _mm_mul_ps(q3,h3));
	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-2])));
	//	h1 = _mm_loaddup_pd(&hh[nb-2]);

	q1 = _mm_load_ps(&q[(nb+1)*ldq]);
	q2 = _mm_load_ps(&q[((nb+1)*ldq)+4]);
	q3 = _mm_load_ps(&q[((nb+1)*ldq)+8]);

	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
	x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
	x3 = _mm_add_ps(x3, _mm_mul_ps(q3,h1));
        h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*1)+nb-1])));
//	h2 = _mm_loaddup_pd(&hh[(ldh*1)+nb-1]);

	y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
	y2 = _mm_add_ps(y2, _mm_mul_ps(q2,h2));
	y3 = _mm_add_ps(y3, _mm_mul_ps(q3,h2));
        h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-1])));
	//	h1 = _mm_loaddup_pd(&hh[nb-1]);

	q1 = _mm_load_ps(&q[(nb+2)*ldq]);
	q2 = _mm_load_ps(&q[((nb+2)*ldq)+4]);
	q3 = _mm_load_ps(&q[((nb+2)*ldq)+8]);

	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
	x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
	x3 = _mm_add_ps(x3, _mm_mul_ps(q3,h1));

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [6 x nb+3]
	/////////////////////////////////////////////////////
        __m128 tau1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[0])));
//	__m128 tau1 = _mm_loaddup_pd(&hh[0]);

	h1 = tau1;
	x1 = _mm_mul_ps(x1, h1);
	x2 = _mm_mul_ps(x2, h1);
	x3 = _mm_mul_ps(x3, h1);
        __m128 tau2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh])));
        __m128 vs_1_2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_1_2)));

//	__m128 tau2 = _mm_loaddup_pd(&hh[ldh]);
//	__m128 vs_1_2 = _mm_loaddup_pd(&s_1_2);

	h1 = tau2;
	h2 = _mm_mul_ps(h1, vs_1_2);

	y1 = _mm_sub_ps(_mm_mul_ps(y1,h1), _mm_mul_ps(x1,h2));
	y2 = _mm_sub_ps(_mm_mul_ps(y2,h1), _mm_mul_ps(x2,h2));
	y3 = _mm_sub_ps(_mm_mul_ps(y3,h1), _mm_mul_ps(x3,h2));
        __m128 tau3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh*2])));
	//	__m128 tau3 = _mm_loaddup_pd(&hh[ldh*2]);
        __m128 vs_1_3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_1_3)));
//	__m128 vs_1_3 = _mm_loaddup_pd(&s_1_3);
        __m128 vs_2_3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_2_3)));
//	__m128 vs_2_3 = _mm_loaddup_pd(&s_2_3);

	h1 = tau3;
	h2 = _mm_mul_ps(h1, vs_1_3);
	h3 = _mm_mul_ps(h1, vs_2_3);

	z1 = _mm_sub_ps(_mm_mul_ps(z1,h1), _mm_add_ps(_mm_mul_ps(y1,h3), _mm_mul_ps(x1,h2)));
	z2 = _mm_sub_ps(_mm_mul_ps(z2,h1), _mm_add_ps(_mm_mul_ps(y2,h3), _mm_mul_ps(x2,h2)));
	z3 = _mm_sub_ps(_mm_mul_ps(z3,h1), _mm_add_ps(_mm_mul_ps(y3,h3), _mm_mul_ps(x3,h2)));
        __m128 tau4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh*3])));
//	__m128 tau4 = _mm_loaddup_pd(&hh[ldh*3]);
        __m128 vs_1_4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_1_4)));
//	__m128 vs_1_4 = _mm_loaddup_pd(&s_1_4);
        __m128 vs_2_4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_2_4)));
//	__m128 vs_2_4 = _mm_loaddup_pd(&s_2_4);
        __m128 vs_3_4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_3_4)));
//	__m128 vs_3_4 = _mm_loaddup_pd(&s_3_4);

	h1 = tau4;
	h2 = _mm_mul_ps(h1, vs_1_4);
	h3 = _mm_mul_ps(h1, vs_2_4);
	h4 = _mm_mul_ps(h1, vs_3_4);

	w1 = _mm_sub_ps(_mm_mul_ps(w1,h1), _mm_add_ps(_mm_mul_ps(z1,h4), _mm_add_ps(_mm_mul_ps(y1,h3), _mm_mul_ps(x1,h2))));
	w2 = _mm_sub_ps(_mm_mul_ps(w2,h1), _mm_add_ps(_mm_mul_ps(z2,h4), _mm_add_ps(_mm_mul_ps(y2,h3), _mm_mul_ps(x2,h2))));
	w3 = _mm_sub_ps(_mm_mul_ps(w3,h1), _mm_add_ps(_mm_mul_ps(z3,h4), _mm_add_ps(_mm_mul_ps(y3,h3), _mm_mul_ps(x3,h2))));

	q1 = _mm_load_ps(&q[0]);
	q2 = _mm_load_ps(&q[4]);
	q3 = _mm_load_ps(&q[8]);
	q1 = _mm_sub_ps(q1, w1);
	q2 = _mm_sub_ps(q2, w2);
	q3 = _mm_sub_ps(q3, w3);
	_mm_store_ps(&q[0],q1);
	_mm_store_ps(&q[4],q2);
	_mm_store_ps(&q[8],q3);

	h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+1])));
//	h4 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
	q1 = _mm_load_ps(&q[ldq]);
	q2 = _mm_load_ps(&q[ldq+4]);
	q3 = _mm_load_ps(&q[ldq+8]);

	q1 = _mm_sub_ps(q1, _mm_add_ps(z1, _mm_mul_ps(w1, h4)));
	q2 = _mm_sub_ps(q2, _mm_add_ps(z2, _mm_mul_ps(w2, h4)));
	q3 = _mm_sub_ps(q3, _mm_add_ps(z3, _mm_mul_ps(w3, h4)));

	_mm_store_ps(&q[ldq],q1);
	_mm_store_ps(&q[ldq+4],q2);
	_mm_store_ps(&q[ldq+8],q3);

	h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+2])));
//	h4 = _mm_loaddup_pd(&hh[(ldh*3)+2]);
	q1 = _mm_load_ps(&q[ldq*2]);
	q2 = _mm_load_ps(&q[(ldq*2)+4]);
	q3 = _mm_load_ps(&q[(ldq*2)+8]);
	q1 = _mm_sub_ps(q1, y1);
	q2 = _mm_sub_ps(q2, y2);
	q3 = _mm_sub_ps(q3, y3);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(w1, h4));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(w2, h4));
	q3 = _mm_sub_ps(q3, _mm_mul_ps(w3, h4));

	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+1])));
//	h3 = _mm_loaddup_pd(&hh[(ldh*2)+1]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(z1, h3));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(z2, h3));
	q3 = _mm_sub_ps(q3, _mm_mul_ps(z3, h3));

	_mm_store_ps(&q[ldq*2],q1);
	_mm_store_ps(&q[(ldq*2)+4],q2);
	_mm_store_ps(&q[(ldq*2)+8],q3);

	h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+3])));
//	h4 = _mm_loaddup_pd(&hh[(ldh*3)+3]);
	q1 = _mm_load_ps(&q[ldq*3]);
	q2 = _mm_load_ps(&q[(ldq*3)+4]);
	q3 = _mm_load_ps(&q[(ldq*3)+8]);
	q1 = _mm_sub_ps(q1, x1);
	q2 = _mm_sub_ps(q2, x2);
	q3 = _mm_sub_ps(q3, x3);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(w1, h4));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(w2, h4));
	q3 = _mm_sub_ps(q3, _mm_mul_ps(w3, h4));

	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+1])));
//	h2 = _mm_loaddup_pd(&hh[ldh+1]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(y1, h2));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(y2, h2));
	q3 = _mm_sub_ps(q3, _mm_mul_ps(y3, h2));

	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+2])));
	//	h3 = _mm_loaddup_pd(&hh[(ldh*2)+2]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(z1, h3));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(z2, h3));
	q3 = _mm_sub_ps(q3, _mm_mul_ps(z3, h3));
	_mm_store_ps(&q[ldq*3], q1);
	_mm_store_ps(&q[(ldq*3)+4], q2);
	_mm_store_ps(&q[(ldq*3)+8], q3);

	for (i = 4; i < nb; i++)
	{
	        h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[i-3])));
//		h1 = _mm_loaddup_pd(&hh[i-3]);

		q1 = _mm_load_ps(&q[i*ldq]);
		q2 = _mm_load_ps(&q[(i*ldq)+4]);
		q3 = _mm_load_ps(&q[(i*ldq)+8]);

		q1 = _mm_sub_ps(q1, _mm_mul_ps(x1,h1));
		q2 = _mm_sub_ps(q2, _mm_mul_ps(x2,h1));
		q3 = _mm_sub_ps(q3, _mm_mul_ps(x3,h1));

		h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+i-2])));
//		h2 = _mm_loaddup_pd(&hh[ldh+i-2]);

		q1 = _mm_sub_ps(q1, _mm_mul_ps(y1,h2));
		q2 = _mm_sub_ps(q2, _mm_mul_ps(y2,h2));
		q3 = _mm_sub_ps(q3, _mm_mul_ps(y3,h2));

		h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+i-1])));
//		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-1]);

		q1 = _mm_sub_ps(q1, _mm_mul_ps(z1,h3));
		q2 = _mm_sub_ps(q2, _mm_mul_ps(z2,h3));
		q3 = _mm_sub_ps(q3, _mm_mul_ps(z3,h3));

		h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+i])));
//		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i]);

		q1 = _mm_sub_ps(q1, _mm_mul_ps(w1,h4));
		q2 = _mm_sub_ps(q2, _mm_mul_ps(w2,h4));
		q3 = _mm_sub_ps(q3, _mm_mul_ps(w3,h4));

		_mm_store_ps(&q[i*ldq],q1);
		_mm_store_ps(&q[(i*ldq)+4],q2);
		_mm_store_ps(&q[(i*ldq)+8],q3);
	}

	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-3])));
//	h1 = _mm_loaddup_pd(&hh[nb-3]);
	q1 = _mm_load_ps(&q[nb*ldq]);
	q2 = _mm_load_ps(&q[(nb*ldq)+4]);
	q3 = _mm_load_ps(&q[(nb*ldq)+8]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(x1, h1));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(x2, h1));
	q3 = _mm_sub_ps(q3, _mm_mul_ps(x3, h1));
	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+nb-2])));
//	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(y1, h2));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(y2, h2));
	q3 = _mm_sub_ps(q3, _mm_mul_ps(y3, h2));
	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+nb-1])));
//	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(z1, h3));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(z2, h3));
	q3 = _mm_sub_ps(q3, _mm_mul_ps(z3, h3));

	_mm_store_ps(&q[nb*ldq],q1);
	_mm_store_ps(&q[(nb*ldq)+4],q2);
	_mm_store_ps(&q[(nb*ldq)+8],q3);
	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-2])));
//	h1 = _mm_loaddup_pd(&hh[nb-2]);
	q1 = _mm_load_ps(&q[(nb+1)*ldq]);
	q2 = _mm_load_ps(&q[((nb+1)*ldq)+4]);
	q3 = _mm_load_ps(&q[((nb+1)*ldq)+8]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(x1, h1));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(x2, h1));
	q3 = _mm_sub_ps(q3, _mm_mul_ps(x3, h1));
	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+nb-1])));
//	h2 = _mm_loaddup_pd(&hh[ldh+nb-1]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(y1, h2));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(y2, h2));
	q3 = _mm_sub_ps(q3, _mm_mul_ps(y3, h2));

	_mm_store_ps(&q[(nb+1)*ldq],q1);
	_mm_store_ps(&q[((nb+1)*ldq)+4],q2);
	_mm_store_ps(&q[((nb+1)*ldq)+8],q3);

	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-1])));
//	h1 = _mm_loaddup_ps(&hh[nb-1]);
	q1 = _mm_load_ps(&q[(nb+2)*ldq]);
	q2 = _mm_load_ps(&q[((nb+2)*ldq)+4]);
	q3 = _mm_load_ps(&q[((nb+2)*ldq)+8]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(x1, h1));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(x2, h1));
	q3 = _mm_sub_ps(q3, _mm_mul_ps(x3, h1));

	_mm_store_ps(&q[(nb+2)*ldq],q1);
	_mm_store_ps(&q[((nb+2)*ldq)+4],q2);
	_mm_store_ps(&q[((nb+2)*ldq)+8],q3);
}
/**
 * Unrolled kernel that computes
 * 8 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_8_SSE_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [6 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m128 a1_1 = _mm_load_ps(&q[ldq*3]);
	__m128 a2_1 = _mm_load_ps(&q[ldq*2]);
	__m128 a3_1 = _mm_load_ps(&q[ldq]);
	__m128 a4_1 = _mm_load_ps(&q[0]);      // q(1,1) | q(2,1) | q(3,1) q(4,1)

//	__m128 h_2_1 = _mm_loaddup_pd(&hh[ldh+1]);      // hh(2,2) and duplicate
//        __m128 x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+1]));        // load hh(2,2) , hh(3,2) | hh(2,2) , hh(3,2) in double precision representations
        __m128 h_2_1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+1])) ); // h_2_1 contains four times hh[ldh+1]
	__m128 h_3_2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+1])));
	__m128 h_3_1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+2])));
	__m128 h_4_3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+1])));
	__m128 h_4_2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+2])));
	__m128 h_4_1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+3])));

//	__m128 h_3_2 = _mm_loaddup_pd(&hh[(ldh*2)+1]);
//	__m128 h_3_1 = _mm_loaddup_pd(&hh[(ldh*2)+2]);
//	__m128 h_4_3 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
//	__m128 h_4_2 = _mm_loaddup_pd(&hh[(ldh*3)+2]);
//	__m128 h_4_1 = _mm_loaddup_pd(&hh[(ldh*3)+3]);

	register __m128 w1 = _mm_add_ps(a4_1, _mm_mul_ps(a3_1, h_4_3));
	w1 = _mm_add_ps(w1, _mm_mul_ps(a2_1, h_4_2));
	w1 = _mm_add_ps(w1, _mm_mul_ps(a1_1, h_4_1));
	register __m128 z1 = _mm_add_ps(a3_1, _mm_mul_ps(a2_1, h_3_2));
	z1 = _mm_add_ps(z1, _mm_mul_ps(a1_1, h_3_1));
	register __m128 y1 = _mm_add_ps(a2_1, _mm_mul_ps(a1_1, h_2_1));
	register __m128 x1 = a1_1;

	__m128 a1_2 = _mm_load_ps(&q[(ldq*3)+4]);
	__m128 a2_2 = _mm_load_ps(&q[(ldq*2)+4]);
	__m128 a3_2 = _mm_load_ps(&q[ldq+4]);
	__m128 a4_2 = _mm_load_ps(&q[0+4]);       // q(5,1) | ... q(8,1)

	register __m128 w2 = _mm_add_ps(a4_2, _mm_mul_ps(a3_2, h_4_3));
	w2 = _mm_add_ps(w2, _mm_mul_ps(a2_2, h_4_2));
	w2 = _mm_add_ps(w2, _mm_mul_ps(a1_2, h_4_1));
	register __m128 z2 = _mm_add_ps(a3_2, _mm_mul_ps(a2_2, h_3_2));
	z2 = _mm_add_ps(z2, _mm_mul_ps(a1_2, h_3_1));
	register __m128 y2 = _mm_add_ps(a2_2, _mm_mul_ps(a1_2, h_2_1));
	register __m128 x2 = a1_2;
//
//	__m128 a1_3 = _mm_load_ps(&q[(ldq*3)+8]);
//	__m128 a2_3 = _mm_load_ps(&q[(ldq*2)+8]);
//	__m128 a3_3 = _mm_load_ps(&q[ldq+8]);
//	__m128 a4_3 = _mm_load_ps(&q[0+8]);    // q(9,1) | .. | q(12,1)
//
//	register __m128 w3 = _mm_add_ps(a4_3, _mm_mul_ps(a3_3, h_4_3));
//	w3 = _mm_add_ps(w3, _mm_mul_ps(a2_3, h_4_2));
//	w3 = _mm_add_ps(w3, _mm_mul_ps(a1_3, h_4_1));
//	register __m128 z3 = _mm_add_ps(a3_3, _mm_mul_ps(a2_3, h_3_2));
//	z3 = _mm_add_ps(z3, _mm_mul_ps(a1_3, h_3_1));
//	register __m128 y3 = _mm_add_ps(a2_3, _mm_mul_ps(a1_3, h_2_1));
//	register __m128 x3 = a1_3;

	__m128 q1;
	__m128 q2;
//	__m128 q3;

	__m128 h1;
	__m128 h2;
	__m128 h3;
	__m128 h4;

	for(i = 4; i < nb; i++)
	{
	        h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[i-3])));
//		h1 = _mm_loaddup_pd(&hh[i-3]);
		q1 = _mm_load_ps(&q[i*ldq]);
		q2 = _mm_load_ps(&q[(i*ldq)+4]);
//		q3 = _mm_load_ps(&q[(i*ldq)+8]);

		x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
		x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
//		x3 = _mm_add_ps(x3, _mm_mul_ps(q3,h1));
	        h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+i-2])));
//		h2 = _mm_loaddup_pd(&hh[ldh+i-2]);

		y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
		y2 = _mm_add_ps(y2, _mm_mul_ps(q2,h2));
//		y3 = _mm_add_ps(y3, _mm_mul_ps(q3,h2));
	        h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+i-1])));
//		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-1]);

		z1 = _mm_add_ps(z1, _mm_mul_ps(q1,h3));
		z2 = _mm_add_ps(z2, _mm_mul_ps(q2,h3));
//		z3 = _mm_add_ps(z3, _mm_mul_ps(q3,h3));
	        h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+i])));
//		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i]);

		w1 = _mm_add_ps(w1, _mm_mul_ps(q1,h4));
		w2 = _mm_add_ps(w2, _mm_mul_ps(q2,h4));
//		w3 = _mm_add_ps(w3, _mm_mul_ps(q3,h4));
	}
	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-3])));
//	h1 = _mm_loaddup_pd(&hh[nb-3]);

	q1 = _mm_load_ps(&q[nb*ldq]);
	q2 = _mm_load_ps(&q[(nb*ldq)+4]);
//	q3 = _mm_load_ps(&q[(nb*ldq)+8]);

	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
	x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
//	x3 = _mm_add_ps(x3, _mm_mul_ps(q3,h1));
	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+nb-2])));
//	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);

	y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
	y2 = _mm_add_ps(y2, _mm_mul_ps(q2,h2));
//	y3 = _mm_add_ps(y3, _mm_mul_ps(q3,h2));
	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+nb-1])));
//	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);

	z1 = _mm_add_ps(z1, _mm_mul_ps(q1,h3));
	z2 = _mm_add_ps(z2, _mm_mul_ps(q2,h3));
//	z3 = _mm_add_ps(z3, _mm_mul_ps(q3,h3));
	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-2])));
	//	h1 = _mm_loaddup_pd(&hh[nb-2]);

	q1 = _mm_load_ps(&q[(nb+1)*ldq]);
	q2 = _mm_load_ps(&q[((nb+1)*ldq)+4]);
//	q3 = _mm_load_ps(&q[((nb+1)*ldq)+8]);

	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
	x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
//	x3 = _mm_add_ps(x3, _mm_mul_ps(q3,h1));
        h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*1)+nb-1])));
//	h2 = _mm_loaddup_pd(&hh[(ldh*1)+nb-1]);

	y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
	y2 = _mm_add_ps(y2, _mm_mul_ps(q2,h2));
//	y3 = _mm_add_ps(y3, _mm_mul_ps(q3,h2));
        h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-1])));
	//	h1 = _mm_loaddup_pd(&hh[nb-1]);

	q1 = _mm_load_ps(&q[(nb+2)*ldq]);
	q2 = _mm_load_ps(&q[((nb+2)*ldq)+4]);
//	q3 = _mm_load_ps(&q[((nb+2)*ldq)+8]);

	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
	x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
//	x3 = _mm_add_ps(x3, _mm_mul_ps(q3,h1));

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [6 x nb+3]
	/////////////////////////////////////////////////////
        __m128 tau1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[0])));
//	__m128 tau1 = _mm_loaddup_pd(&hh[0]);

	h1 = tau1;
	x1 = _mm_mul_ps(x1, h1);
	x2 = _mm_mul_ps(x2, h1);
//	x3 = _mm_mul_ps(x3, h1);
        __m128 tau2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh])));
        __m128 vs_1_2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_1_2)));

//	__m128 tau2 = _mm_loaddup_pd(&hh[ldh]);
//	__m128 vs_1_2 = _mm_loaddup_pd(&s_1_2);

	h1 = tau2;
	h2 = _mm_mul_ps(h1, vs_1_2);

	y1 = _mm_sub_ps(_mm_mul_ps(y1,h1), _mm_mul_ps(x1,h2));
	y2 = _mm_sub_ps(_mm_mul_ps(y2,h1), _mm_mul_ps(x2,h2));
//	y3 = _mm_sub_ps(_mm_mul_ps(y3,h1), _mm_mul_ps(x3,h2));
        __m128 tau3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh*2])));
	//	__m128 tau3 = _mm_loaddup_pd(&hh[ldh*2]);
        __m128 vs_1_3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_1_3)));
//	__m128 vs_1_3 = _mm_loaddup_pd(&s_1_3);
        __m128 vs_2_3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_2_3)));
//	__m128 vs_2_3 = _mm_loaddup_pd(&s_2_3);

	h1 = tau3;
	h2 = _mm_mul_ps(h1, vs_1_3);
	h3 = _mm_mul_ps(h1, vs_2_3);

	z1 = _mm_sub_ps(_mm_mul_ps(z1,h1), _mm_add_ps(_mm_mul_ps(y1,h3), _mm_mul_ps(x1,h2)));
	z2 = _mm_sub_ps(_mm_mul_ps(z2,h1), _mm_add_ps(_mm_mul_ps(y2,h3), _mm_mul_ps(x2,h2)));
//	z3 = _mm_sub_ps(_mm_mul_ps(z3,h1), _mm_add_ps(_mm_mul_ps(y3,h3), _mm_mul_ps(x3,h2)));
        __m128 tau4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh*3])));
//	__m128 tau4 = _mm_loaddup_pd(&hh[ldh*3]);
        __m128 vs_1_4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_1_4)));
//	__m128 vs_1_4 = _mm_loaddup_pd(&s_1_4);
        __m128 vs_2_4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_2_4)));
//	__m128 vs_2_4 = _mm_loaddup_pd(&s_2_4);
        __m128 vs_3_4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_3_4)));
//	__m128 vs_3_4 = _mm_loaddup_pd(&s_3_4);

	h1 = tau4;
	h2 = _mm_mul_ps(h1, vs_1_4);
	h3 = _mm_mul_ps(h1, vs_2_4);
	h4 = _mm_mul_ps(h1, vs_3_4);

	w1 = _mm_sub_ps(_mm_mul_ps(w1,h1), _mm_add_ps(_mm_mul_ps(z1,h4), _mm_add_ps(_mm_mul_ps(y1,h3), _mm_mul_ps(x1,h2))));
	w2 = _mm_sub_ps(_mm_mul_ps(w2,h1), _mm_add_ps(_mm_mul_ps(z2,h4), _mm_add_ps(_mm_mul_ps(y2,h3), _mm_mul_ps(x2,h2))));
//	w3 = _mm_sub_ps(_mm_mul_ps(w3,h1), _mm_add_ps(_mm_mul_ps(z3,h4), _mm_add_ps(_mm_mul_ps(y3,h3), _mm_mul_ps(x3,h2))));

	q1 = _mm_load_ps(&q[0]);
	q2 = _mm_load_ps(&q[4]);
//	q3 = _mm_load_ps(&q[8]);
	q1 = _mm_sub_ps(q1, w1);
	q2 = _mm_sub_ps(q2, w2);
//	q3 = _mm_sub_ps(q3, w3);
	_mm_store_ps(&q[0],q1);
	_mm_store_ps(&q[4],q2);
//	_mm_store_ps(&q[8],q3);

	h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+1])));
//	h4 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
	q1 = _mm_load_ps(&q[ldq]);
	q2 = _mm_load_ps(&q[ldq+4]);
//	q3 = _mm_load_ps(&q[ldq+8]);

	q1 = _mm_sub_ps(q1, _mm_add_ps(z1, _mm_mul_ps(w1, h4)));
	q2 = _mm_sub_ps(q2, _mm_add_ps(z2, _mm_mul_ps(w2, h4)));
//	q3 = _mm_sub_ps(q3, _mm_add_ps(z3, _mm_mul_ps(w3, h4)));

	_mm_store_ps(&q[ldq],q1);
	_mm_store_ps(&q[ldq+4],q2);
//	_mm_store_ps(&q[ldq+8],q3);

	h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+2])));
//	h4 = _mm_loaddup_pd(&hh[(ldh*3)+2]);
	q1 = _mm_load_ps(&q[ldq*2]);
	q2 = _mm_load_ps(&q[(ldq*2)+4]);
//	q3 = _mm_load_ps(&q[(ldq*2)+8]);
	q1 = _mm_sub_ps(q1, y1);
	q2 = _mm_sub_ps(q2, y2);
//	q3 = _mm_sub_ps(q3, y3);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(w1, h4));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(w2, h4));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(w3, h4));

	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+1])));
//	h3 = _mm_loaddup_pd(&hh[(ldh*2)+1]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(z1, h3));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(z2, h3));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(z3, h3));

	_mm_store_ps(&q[ldq*2],q1);
	_mm_store_ps(&q[(ldq*2)+4],q2);
//	_mm_store_ps(&q[(ldq*2)+8],q3);

	h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+3])));
//	h4 = _mm_loaddup_pd(&hh[(ldh*3)+3]);
	q1 = _mm_load_ps(&q[ldq*3]);
	q2 = _mm_load_ps(&q[(ldq*3)+4]);
//	q3 = _mm_load_ps(&q[(ldq*3)+8]);
	q1 = _mm_sub_ps(q1, x1);
	q2 = _mm_sub_ps(q2, x2);
//	q3 = _mm_sub_ps(q3, x3);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(w1, h4));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(w2, h4));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(w3, h4));

	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+1])));
//	h2 = _mm_loaddup_pd(&hh[ldh+1]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(y1, h2));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(y2, h2));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(y3, h2));

	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+2])));
	//	h3 = _mm_loaddup_pd(&hh[(ldh*2)+2]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(z1, h3));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(z2, h3));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(z3, h3));
	_mm_store_ps(&q[ldq*3], q1);
	_mm_store_ps(&q[(ldq*3)+4], q2);
//	_mm_store_ps(&q[(ldq*3)+8], q3);

	for (i = 4; i < nb; i++)
	{
	        h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[i-3])));
//		h1 = _mm_loaddup_pd(&hh[i-3]);

		q1 = _mm_load_ps(&q[i*ldq]);
		q2 = _mm_load_ps(&q[(i*ldq)+4]);
//		q3 = _mm_load_ps(&q[(i*ldq)+8]);

		q1 = _mm_sub_ps(q1, _mm_mul_ps(x1,h1));
		q2 = _mm_sub_ps(q2, _mm_mul_ps(x2,h1));
//		q3 = _mm_sub_ps(q3, _mm_mul_ps(x3,h1));

		h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+i-2])));
//		h2 = _mm_loaddup_pd(&hh[ldh+i-2]);

		q1 = _mm_sub_ps(q1, _mm_mul_ps(y1,h2));
		q2 = _mm_sub_ps(q2, _mm_mul_ps(y2,h2));
//		q3 = _mm_sub_ps(q3, _mm_mul_ps(y3,h2));

		h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+i-1])));
//		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-1]);

		q1 = _mm_sub_ps(q1, _mm_mul_ps(z1,h3));
		q2 = _mm_sub_ps(q2, _mm_mul_ps(z2,h3));
//		q3 = _mm_sub_ps(q3, _mm_mul_ps(z3,h3));

		h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+i])));
//		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i]);

		q1 = _mm_sub_ps(q1, _mm_mul_ps(w1,h4));
		q2 = _mm_sub_ps(q2, _mm_mul_ps(w2,h4));
//		q3 = _mm_sub_ps(q3, _mm_mul_ps(w3,h4));

		_mm_store_ps(&q[i*ldq],q1);
		_mm_store_ps(&q[(i*ldq)+4],q2);
//		_mm_store_ps(&q[(i*ldq)+8],q3);
	}

	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-3])));
//	h1 = _mm_loaddup_pd(&hh[nb-3]);
	q1 = _mm_load_ps(&q[nb*ldq]);
	q2 = _mm_load_ps(&q[(nb*ldq)+4]);
//	q3 = _mm_load_ps(&q[(nb*ldq)+8]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(x1, h1));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(x2, h1));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(x3, h1));
	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+nb-2])));
//	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(y1, h2));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(y2, h2));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(y3, h2));
	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+nb-1])));
//	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(z1, h3));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(z2, h3));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(z3, h3));

	_mm_store_ps(&q[nb*ldq],q1);
	_mm_store_ps(&q[(nb*ldq)+4],q2);
//	_mm_store_ps(&q[(nb*ldq)+8],q3);
	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-2])));
//	h1 = _mm_loaddup_pd(&hh[nb-2]);
	q1 = _mm_load_ps(&q[(nb+1)*ldq]);
	q2 = _mm_load_ps(&q[((nb+1)*ldq)+4]);
//	q3 = _mm_load_ps(&q[((nb+1)*ldq)+8]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(x1, h1));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(x2, h1));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(x3, h1));
	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+nb-1])));
//	h2 = _mm_loaddup_pd(&hh[ldh+nb-1]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(y1, h2));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(y2, h2));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(y3, h2));

	_mm_store_ps(&q[(nb+1)*ldq],q1);
	_mm_store_ps(&q[((nb+1)*ldq)+4],q2);
//	_mm_store_ps(&q[((nb+1)*ldq)+8],q3);

	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-1])));
//	h1 = _mm_loaddup_ps(&hh[nb-1]);
	q1 = _mm_load_ps(&q[(nb+2)*ldq]);
	q2 = _mm_load_ps(&q[((nb+2)*ldq)+4]);
//	q3 = _mm_load_ps(&q[((nb+2)*ldq)+8]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(x1, h1));
	q2 = _mm_sub_ps(q2, _mm_mul_ps(x2, h1));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(x3, h1));

	_mm_store_ps(&q[(nb+2)*ldq],q1);
	_mm_store_ps(&q[((nb+2)*ldq)+4],q2);
//	_mm_store_ps(&q[((nb+2)*ldq)+8],q3);
}


///**
// * Unrolled kernel that computes
// * 8 rows of Q simultaneously, a
// * matrix vector product with two householder
// * vectors + a rank 1 update is performed
// */
//__forceinline void hh_trafo_kernel_8_SSE_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
//{
//	/////////////////////////////////////////////////////
//	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
//	// hh contains four householder vectors
//	/////////////////////////////////////////////////////
//	int i;
//
//	__m128 a1_1 = _mm_load_ps(&q[ldq*3]);
//	__m128 a2_1 = _mm_load_ps(&q[ldq*2]);
//	__m128 a3_1 = _mm_load_ps(&q[ldq]);
//	__m128 a4_1 = _mm_load_ps(&q[0]);      // q(1,1) | q(2,1) | q(3,1) | q(4,1)
//
//	// carefull
//        __m128 h_2_1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+1])));
////	__m128 h_2_1 = _mm_loaddup_pd(&hh[ldh+1]);
//        __m128 h_3_2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+1])));
////	__m128 h_3_2 = _mm_loaddup_pd(&hh[(ldh*2)+1]);
//        __m128 h_3_1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+2])));
////	__m128 h_3_1 = _mm_loaddup_pd(&hh[(ldh*2)+2]);
//        __m128 h_4_3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+1])));
////	__m128 h_4_3 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
//        __m128 h_4_2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+2])));
////	__m128 h_4_2 = _mm_loaddup_pd(&hh[(ldh*3)+2]);
//        __m128 h_4_1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+3])));
////	__m128 h_4_1 = _mm_loaddup_pd(&hh[(ldh*3)+3]);
//
//	__m128 w1 = _mm_add_ps(a4_1, _mm_mul_ps(a3_1, h_4_3));
//	w1 = _mm_add_ps(w1, _mm_mul_ps(a2_1, h_4_2));
//	w1 = _mm_add_ps(w1, _mm_mul_ps(a1_1, h_4_1));
//	__m128 z1 = _mm_add_ps(a3_1, _mm_mul_ps(a2_1, h_3_2));
//	z1 = _mm_add_ps(z1, _mm_mul_ps(a1_1, h_3_1));
//	__m128 y1 = _mm_add_ps(a2_1, _mm_mul_ps(a1_1, h_2_1));
//	__m128 x1 = a1_1;
//
//	__m128 a1_2 = _mm_load_ps(&q[(ldq*3)+4]);
//	__m128 a2_2 = _mm_load_ps(&q[(ldq*2)+4]);
//	__m128 a3_2 = _mm_load_ps(&q[ldq+4]);
//	__m128 a4_2 = _mm_load_ps(&q[0+4]);
//
//	__m128 w2 = _mm_add_ps(a4_2, _mm_mul_ps(a3_2, h_4_3));
//	w2 = _mm_add_ps(w2, _mm_mul_ps(a2_2, h_4_2));
//	w2 = _mm_add_ps(w2, _mm_mul_ps(a1_2, h_4_1));
//	__m128 z2 = _mm_add_ps(a3_2, _mm_mul_ps(a2_2, h_3_2));
//	z2 = _mm_add_ps(z2, _mm_mul_ps(a1_2, h_3_1));
//	__m128 y2 = _mm_add_ps(a2_2, _mm_mul_ps(a1_2, h_2_1));
//	__m128 x2 = a1_2;
//
//	__m128 q1;
//	__m128 q2;
//
//	__m128 h1;
//	__m128 h2;
//	__m128 h3;
//	__m128 h4;
//
//	for(i = 4; i < nb; i++)
//	{
//		h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[i-3])));
////		h1 = _mm_loaddup_pd(&hh[i-3]);
//		h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+i-2])));
////		h2 = _mm_loaddup_pd(&hh[ldh+i-2]);
//		h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+i-1])));
////		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-1]);
//		h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+i])));
////		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i]);
//
//		q1 = _mm_load_ps(&q[i*ldq]);
//
//		x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
//		y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
//		z1 = _mm_add_ps(z1, _mm_mul_ps(q1,h3));
//		w1 = _mm_add_ps(w1, _mm_mul_ps(q1,h4));
//
//		q2 = _mm_load_ps(&q[(i*ldq)+4]);
//
//		x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
//		y2 = _mm_add_ps(y2, _mm_mul_ps(q2,h2));
//		z2 = _mm_add_ps(z2, _mm_mul_ps(q2,h3));
//		w2 = _mm_add_ps(w2, _mm_mul_ps(q2,h4));
//	}
//	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-3])));
////	h1 = _mm_loaddup_pd(&hh[nb-3]);
//	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+nb-2])));
////	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);
//	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+nb-1])));
////	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);
//
//	q1 = _mm_load_ps(&q[nb*ldq]);
//	q2 = _mm_load_ps(&q[(nb*ldq)+4]);
//
//	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
//	x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
//	y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
//	y2 = _mm_add_ps(y2, _mm_mul_ps(q2,h2));
//	z1 = _mm_add_ps(z1, _mm_mul_ps(q1,h3));
//	z2 = _mm_add_ps(z2, _mm_mul_ps(q2,h3));
//
//	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-2])));
//	//	h1 = _mm_loaddup_pd(&hh[nb-2]);
//	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*1)+nb-1])));
//	//	h2 = _mm_loaddup_pd(&hh[(ldh*1)+nb-1]);
//
//	q1 = _mm_load_ps(&q[(nb+1)*ldq]);
//	q2 = _mm_load_ps(&q[((nb+1)*ldq)+2]);
//
//	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
//	x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
//	y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
//	y2 = _mm_add_ps(y2, _mm_mul_ps(q2,h2));
//
//	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-1])));
////	h1 = _mm_loaddup_pd(&hh[nb-1]);
//
//	q1 = _mm_load_ps(&q[(nb+2)*ldq]);
//	q2 = _mm_load_ps(&q[((nb+2)*ldq)+4]);
//
//	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
//	x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
//
//	/////////////////////////////////////////////////////
//	// Rank-1 update of Q [4 x nb+3]
//	/////////////////////////////////////////////////////
//
//	__m128 tau1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[0])));
////	__m128 tau1 = _mm_loaddup_pd(&hh[0]);
//	__m128 tau2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh])));
////	__m128 tau2 = _mm_loaddup_pd(&hh[ldh]);
//	__m128 tau3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh*2])));
////	__m128 tau3 = _mm_loaddup_pd(&hh[ldh*2]);
//	__m128 tau4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh*3])));
////	__m128 tau4 = _mm_loaddup_pd(&hh[ldh*3]);
//
//	__m128 vs_1_2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_1_2)));
////	__m128 vs_1_2 = _mm_loaddup_pd(&s_1_2);
//	__m128 vs_1_3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_1_3)));
////	__m128 vs_1_3 = _mm_loaddup_pd(&s_1_3);
//	__m128 vs_2_3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_2_3)));
////	__m128 vs_2_3 = _mm_loaddup_pd(&s_2_3);
//	__m128 vs_1_4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_1_4)));
////	__m128 vs_1_4 = _mm_loaddup_pd(&s_1_4);
//	__m128 vs_2_4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_2_4)));
////	__m128 vs_2_4 = _mm_loaddup_pd(&s_2_4);
//	__m128 vs_3_4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_3_4)));
////	__m128 vs_3_4 = _mm_loaddup_pd(&s_3_4);
//
//	h1 = tau1;
//	x1 = _mm_mul_ps(x1, h1);
//	x2 = _mm_mul_ps(x2, h1);
//
//	h1 = tau2;
//	h2 = _mm_mul_ps(h1, vs_1_2);
//
//	y1 = _mm_sub_ps(_mm_mul_ps(y1,h1), _mm_mul_ps(x1,h2));
//	y2 = _mm_sub_ps(_mm_mul_ps(y2,h1), _mm_mul_ps(x2,h2));
//
//	h1 = tau3;
//	h2 = _mm_mul_ps(h1, vs_1_3);
//	h3 = _mm_mul_ps(h1, vs_2_3);
//
//	z1 = _mm_sub_ps(_mm_mul_ps(z1,h1), _mm_add_ps(_mm_mul_ps(y1,h3), _mm_mul_ps(x1,h2)));
//	z2 = _mm_sub_ps(_mm_mul_ps(z2,h1), _mm_add_ps(_mm_mul_ps(y2,h3), _mm_mul_ps(x2,h2)));
//
//	h1 = tau4;
//	h2 = _mm_mul_ps(h1, vs_1_4);
//	h3 = _mm_mul_ps(h1, vs_2_4);
//	h4 = _mm_mul_ps(h1, vs_3_4);
//
//	w1 = _mm_sub_ps(_mm_mul_ps(w1,h1), _mm_add_ps(_mm_mul_ps(z1,h4), _mm_add_ps(_mm_mul_ps(y1,h3), _mm_mul_ps(x1,h2))));
//	w2 = _mm_sub_ps(_mm_mul_ps(w2,h1), _mm_add_ps(_mm_mul_ps(z2,h4), _mm_add_ps(_mm_mul_ps(y2,h3), _mm_mul_ps(x2,h2))));
//
//	q1 = _mm_load_ps(&q[0]);
//	q2 = _mm_load_ps(&q[4]);
//	q1 = _mm_sub_ps(q1, w1);
//	q2 = _mm_sub_ps(q2, w2);
//	_mm_store_ps(&q[0],q1);
//	_mm_store_ps(&q[4],q2);
//
//	h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+1])));
//	//	h4 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
//	q1 = _mm_load_ps(&q[ldq]);
//	q2 = _mm_load_ps(&q[ldq+4]);
//
//	q1 = _mm_sub_ps(q1, _mm_add_ps(z1, _mm_mul_ps(w1, h4)));
//	q2 = _mm_sub_ps(q2, _mm_add_ps(z2, _mm_mul_ps(w2, h4)));
//
//	_mm_store_ps(&q[ldq],q1);
//	_mm_store_ps(&q[ldq+4],q2);
//
//	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+1])));
//	//	h3 = _mm_loaddup_ps(&hh[(ldh*2)+1]);
//
//	h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+2])));
////	h4 = _mm_loaddup_ps(&hh[(ldh*3)+2]);
//	q1 = _mm_load_ps(&q[ldq*4]);
//	q2 = _mm_load_ps(&q[(ldq*2)+4]);
//
//	q1 = _mm_sub_ps(q1, _mm_add_ps(y1, _mm_add_ps(_mm_mul_ps(z1, h3), _mm_mul_ps(w1, h4))));
//	q2 = _mm_sub_ps(q2, _mm_add_ps(y2, _mm_add_ps(_mm_mul_ps(z2, h3), _mm_mul_ps(w2, h4))));
//	_mm_store_ps(&q[ldq*2],q1);
//	_mm_store_ps(&q[(ldq*2)+4],q2);
//
//	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+1])));
//	//	h2 = _mm_loaddup_pd(&hh[ldh+1]);
//	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+2])));
//	//	h3 = _mm_loaddup_pd(&hh[(ldh*2)+2]);
//	h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+3])));
//	//	h4 = _mm_loaddup_pd(&hh[(ldh*3)+3]);
//	q1 = _mm_load_ps(&q[ldq*3]);
//	q2 = _mm_load_ps(&q[(ldq*3)+4]);
//
//	q1 = _mm_sub_ps(q1, _mm_add_ps(x1, _mm_add_ps(_mm_mul_ps(y1, h2), _mm_add_ps(_mm_mul_ps(z1, h3), _mm_mul_ps(w1, h4)))));
//	q2 = _mm_sub_ps(q2, _mm_add_ps(x2, _mm_add_ps(_mm_mul_ps(y2, h2), _mm_add_ps(_mm_mul_ps(z2, h3), _mm_mul_ps(w2, h4)))));
//
//	_mm_store_ps(&q[ldq*3], q1);
//	_mm_store_ps(&q[(ldq*3)+4], q2);
//
//	for (i = 4; i < nb; i++)
//	{
//         	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[i-3])));
////		h1 = _mm_loaddup_pd(&hh[i-3]);
//        	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+i-2])));
////		h2 = _mm_loaddup_pd(&hh[ldh+i-2]);
//        	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+i-1])));
////		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-1]);
//        	h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+i])));
////		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i]);
//
//		q1 = _mm_load_ps(&q[i*ldq]);
//
//		q1 = _mm_sub_ps(q1, _mm_add_ps(_mm_add_ps(_mm_mul_ps(w1, h4), _mm_mul_ps(z1, h3)), _mm_add_ps(_mm_mul_ps(x1,h1), _mm_mul_ps(y1, h2))));
//
//		_mm_store_ps(&q[i*ldq],q1);
//
//		q2 = _mm_load_ps(&q[(i*ldq)+4]);
//
//		q2 = _mm_sub_ps(q2, _mm_add_ps(_mm_add_ps(_mm_mul_ps(w2, h4), _mm_mul_ps(z2, h3)), _mm_add_ps(_mm_mul_ps(x2,h1), _mm_mul_ps(y2, h2))));
//
//		_mm_store_ps(&q[(i*ldq)+4],q2);
//	}
//
//       	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-3])));
////	h1 = _mm_loaddup_pd(&hh[nb-3]);
//       	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+nb-2])));
////	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);
//       	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+nb-1])));
////	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);
//	q1 = _mm_load_ps(&q[nb*ldq]);
//	q2 = _mm_load_ps(&q[(nb*ldq)+4]);
//
//	q1 = _mm_sub_ps(q1, _mm_add_ps(_mm_add_ps(_mm_mul_ps(z1, h3), _mm_mul_ps(y1, h2)) , _mm_mul_ps(x1, h1)));
//	q2 = _mm_sub_ps(q2, _mm_add_ps(_mm_add_ps(_mm_mul_ps(z2, h3), _mm_mul_ps(y2, h2)) , _mm_mul_ps(x2, h1)));
//
//	_mm_store_ps(&q[nb*ldq],q1);
//	_mm_store_ps(&q[(nb*ldq)+4],q2);
//
//       	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-2])));
//	//	h1 = _mm_loaddup_pd(&hh[nb-2]);
//       	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+nb-1])));
//	//	h2 = _mm_loaddup_pd(&hh[ldh+nb-1]);
//	q1 = _mm_load_ps(&q[(nb+1)*ldq]);
//	q2 = _mm_load_ps(&q[((nb+1)*ldq)+4]);
//
//	q1 = _mm_sub_ps(q1, _mm_add_ps( _mm_mul_ps(y1, h2) , _mm_mul_ps(x1, h1)));
//	q2 = _mm_sub_ps(q2, _mm_add_ps( _mm_mul_ps(y2, h2) , _mm_mul_ps(x2, h1)));
//
//	_mm_store_ps(&q[(nb+1)*ldq],q1);
//	_mm_store_ps(&q[((nb+1)*ldq)+4],q2);
//
//  	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-1])));
//	//	h1 = _mm_loaddup_pd(&hh[nb-1]);
//	q1 = _mm_load_ps(&q[(nb+2)*ldq]);
//	q2 = _mm_load_ps(&q[((nb+2)*ldq)+4]);
//
//	q1 = _mm_sub_ps(q1, _mm_mul_ps(x1, h1));
//	q2 = _mm_sub_ps(q2, _mm_mul_ps(x2, h1));
//
//	_mm_store_ps(&q[(nb+2)*ldq],q1);
//	_mm_store_ps(&q[((nb+2)*ldq)+4],q2);
//}



///**
// * Unrolled kernel that computes
// * 2 rows of Q simultaneously, a
// * matrix vector product with two householder
// * vectors + a rank 1 update is performed
// */
//__forceinline void hh_trafo_kernel_4_SSE_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
//{
//	/////////////////////////////////////////////////////
//	// Matrix Vector Multiplication, Q [2 x nb+3] * hh
//	// hh contains four householder vectors
//	/////////////////////////////////////////////////////
//	int i;
//
//	__m128 a1_1 = _mm_load_ps(&q[ldq*3]);
//	__m128 a2_1 = _mm_load_ps(&q[ldq*2]);
//	__m128 a3_1 = _mm_load_ps(&q[ldq]);
//	__m128 a4_1 = _mm_load_ps(&q[0]);      // q(1,1) || .. | q(4,1)
//
//
//	__m128 h_2_1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+1])));
////	__m128 h_2_1 = _mm_loaddup_pd(&hh[ldh+1]);
//	__m128 h_3_2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+1])));
////	__m128 h_3_2 = _mm_loaddup_pd(&hh[(ldh*2)+1]);
//	__m128 h_3_1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+2])));
////	__m128 h_3_1 = _mm_loaddup_pd(&hh[(ldh*2)+2]);
//	__m128 h_4_3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+1])));
////	__m128 h_4_3 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
//	__m128 h_4_2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+2])));
////	__m128 h_4_2 = _mm_loaddup_pd(&hh[(ldh*3)+2]);
//	__m128 h_4_1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+3])));
////	__m128 h_4_1 = _mm_loaddup_pd(&hh[(ldh*3)+3]);
//
//	__m128 w1 = _mm_add_ps(a4_1, _mm_mul_ps(a3_1, h_4_3));
//	w1 = _mm_add_ps(w1, _mm_mul_ps(a2_1, h_4_2));
//	w1 = _mm_add_ps(w1, _mm_mul_ps(a1_1, h_4_1));
//	__m128 z1 = _mm_add_ps(a3_1, _mm_mul_ps(a2_1, h_3_2));
//	z1 = _mm_add_ps(z1, _mm_mul_ps(a1_1, h_3_1));
//	__m128 y1 = _mm_add_ps(a2_1, _mm_mul_ps(a1_1, h_2_1));
//	__m128 x1 = a1_1;
//
//	__m128 q1;
//
//	__m128 h1;
//	__m128 h2;
//	__m128 h3;
//	__m128 h4;
//
//	for(i = 4; i < nb; i++)
//	{
//	        h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[i-3])));
//		//		h1 = _mm_loaddup_pd(&hh[i-3]);
//	        h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+i-2])));
//		//		h2 = _mm_loaddup_pd(&hh[ldh+i-2]);
//	        h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+i-1])));
//		//		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-1]);
//	        h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+i])));
//		//		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i]);
//
//		q1 = _mm_load_ps(&q[i*ldq]);
//
//		x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
//		y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
//		z1 = _mm_add_ps(z1, _mm_mul_ps(q1,h3));
//		w1 = _mm_add_ps(w1, _mm_mul_ps(q1,h4));
//	}
//
//	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-3])));
////	h1 = _mm_loaddup_pd(&hh[nb-3]);
//	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+nb-2])));
////	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);
//	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+nb-1])));
////	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);
//	q1 = _mm_load_ps(&q[nb*ldq]);
//
//	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
//	y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
//	z1 = _mm_add_ps(z1, _mm_mul_ps(q1,h3));
//
//	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-2])));
//	//	h1 = _mm_loaddup_pd(&hh[nb-2]);
//	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*1)-1])));
//	//	h2 = _mm_loaddup_pd(&hh[(ldh*1)+nb-1]);
//	q1 = _mm_load_ps(&q[(nb+1)*ldq]);
//
//	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
//	y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
//
//	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-1])));
//	//	h1 = _mm_loaddup_pd(&hh[nb-1]);
//	q1 = _mm_load_ps(&q[(nb+2)*ldq]);
//
//	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
//	/////////////////////////////////////////////////////
//	// Rank-1 update of Q [2 x nb+3]
//	/////////////////////////////////////////////////////
//
//
//	__m128 tau1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[0])));
//	//	__m128d tau1 = _mm_loaddup_pd(&hh[0]);
//	__m128 tau2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh])));
//	//	__m128 tau2 = _mm_loaddup_pd(&hh[ldh]);
//	__m128 tau3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh*2])));
//	//	__m128 tau3 = _mm_loaddup_pd(&hh[ldh*2]);
//	__m128 tau4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh*3])));
//	//__m128d tau4 = _mm_loaddup_pd(&hh[ldh*3]);
//
//	__m128 vs_1_2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_1_2)));
////	__m128d vs_1_2 = _mm_loaddup_pd(&s_1_2);
//	__m128 vs_1_3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_1_3)));
////	__m128d vs_1_3 = _mm_loaddup_pd(&s_1_3);
//	__m128 vs_2_3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_2_3)));
////	__m128d vs_2_3 = _mm_loaddup_pd(&s_2_3);
//	__m128 vs_1_4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_1_4)));
////	__m128d vs_1_4 = _mm_loaddup_pd(&s_1_4);
//	__m128 vs_2_4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_2_4)));
////	__m128d vs_2_4 = _mm_loaddup_pd(&s_2_4);
//	__m128 vs_3_4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_3_4)));
////	__m128d vs_3_4 = _mm_loaddup_pd(&s_3_4);
//
//	h1 = tau1;
//	x1 = _mm_mul_ps(x1, h1);
//
//	h1 = tau2;
//	h2 = _mm_mul_ps(h1, vs_1_2);
//
//	y1 = _mm_sub_ps(_mm_mul_ps(y1,h1), _mm_mul_ps(x1,h2));
//
//	h1 = tau3;
//	h2 = _mm_mul_ps(h1, vs_1_3);
//	h3 = _mm_mul_ps(h1, vs_2_3);
//
//	z1 = _mm_sub_ps(_mm_mul_ps(z1,h1), _mm_add_ps(_mm_mul_ps(y1,h3), _mm_mul_ps(x1,h2)));
//
//	h1 = tau4;
//	h2 = _mm_mul_ps(h1, vs_1_4);
//	h3 = _mm_mul_ps(h1, vs_2_4);
//	h4 = _mm_mul_ps(h1, vs_3_4);
//
//	w1 = _mm_sub_ps(_mm_mul_ps(w1,h1), _mm_add_ps(_mm_mul_ps(z1,h4), _mm_add_ps(_mm_mul_ps(y1,h3), _mm_mul_ps(x1,h2))));
//
//	q1 = _mm_load_ps(&q[0]);
//	q1 = _mm_sub_ps(q1, w1);
//	_mm_store_ps(&q[0],q1);
//
//	h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+1])));
//	//	h4 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
//	q1 = _mm_load_ps(&q[ldq]);
//
//	q1 = _mm_sub_ps(q1, _mm_add_ps(z1, _mm_mul_ps(w1, h4)));
//
//	_mm_store_ps(&q[ldq],q1);
//
//	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+1])));
//	//	h3 = _mm_loaddup_pd(&hh[(ldh*2)+1]);
//	h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+2])));
//	//	h4 = _mm_loaddup_pd(&hh[(ldh*3)+2]);
//	q1 = _mm_load_ps(&q[ldq*2]);
//
//	q1 = _mm_sub_ps(q1, _mm_add_ps(y1, _mm_add_ps(_mm_mul_ps(z1, h3), _mm_mul_ps(w1, h4))));
//
//	_mm_store_ps(&q[ldq*2],q1);
//
//	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+1])));
//	//	h2 = _mm_loaddup_pd(&hh[ldh+1]);
//	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+2])));
//	//	h3 = _mm_loaddup_pd(&hh[(ldh*2)+2]);
//	h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+3])));
//	//	h4 = _mm_loaddup_pd(&hh[(ldh*3)+3]);
//	q1 = _mm_load_ps(&q[ldq*3]);
//
//	q1 = _mm_sub_ps(q1, _mm_add_ps(x1, _mm_add_ps(_mm_mul_ps(y1, h2), _mm_add_ps(_mm_mul_ps(z1, h3), _mm_mul_ps(w1, h4)))));
//
//	_mm_store_ps(&q[ldq*3], q1);
//
//	for (i = 4; i < nb; i++)
//	{
//
//		h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[i-3])));
//		//		h1 = _mm_loaddup_pd(&hh[i-3]);
//		h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+i-2])));
//		//		h2 = _mm_loaddup_pd(&hh[ldh+i-2]);
//		h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+i-1])));
//		//		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-1]);
//		h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+i])));
//		//		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i]);
//
//		q1 = _mm_load_ps(&q[i*ldq]);
//
//		q1 = _mm_sub_ps(q1, _mm_add_ps(_mm_add_ps(_mm_mul_ps(w1, h4), _mm_mul_ps(z1, h3)), _mm_add_ps(_mm_mul_ps(x1,h1), _mm_mul_ps(y1, h2))));
//
//		_mm_store_ps(&q[i*ldq],q1);
//	}
//
//	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-3])));
//	//	h1 = _mm_loaddup_pd(&hh[nb-3]);
//	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+nb-2])));
//	//	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);
//	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+nb-1])));
//	//	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);
//	q1 = _mm_load_ps(&q[nb*ldq]);
//
//	q1 = _mm_sub_ps(q1, _mm_add_ps(_mm_add_ps(_mm_mul_ps(z1, h3), _mm_mul_ps(y1, h2)) , _mm_mul_ps(x1, h1)));
//
//	_mm_store_ps(&q[nb*ldq],q1);
//
//	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-2])));
//	//	h1 = _mm_loaddup_pd(&hh[nb-2]);
//	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+nb-1])));
//	//	h2 = _mm_loaddup_pd(&hh[ldh+nb-1]);
//	q1 = _mm_load_ps(&q[(nb+1)*ldq]);
//
//	q1 = _mm_sub_ps(q1, _mm_add_ps( _mm_mul_ps(y1, h2) , _mm_mul_ps(x1, h1)));
//
//	_mm_store_ps(&q[(nb+1)*ldq],q1);
//
//	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-1])));
//	//	h1 = _mm_loaddup_pd(&hh[nb-1]);
//	q1 = _mm_load_ps(&q[(nb+2)*ldq]);
//
//	q1 = _mm_sub_ps(q1, _mm_mul_ps(x1, h1));
//
//	_mm_store_ps(&q[(nb+2)*ldq],q1);
//}

/**
 * Unrolled kernel that computes
 * 4 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_4_SSE_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [6 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m128 a1_1 = _mm_load_ps(&q[ldq*3]);
	__m128 a2_1 = _mm_load_ps(&q[ldq*2]);
	__m128 a3_1 = _mm_load_ps(&q[ldq]);
	__m128 a4_1 = _mm_load_ps(&q[0]);      // q(1,1) | q(2,1) | q(3,1) q(4,1)

	//careful here
//	__m128 h_2_1 = _mm_loaddup_pd(&hh[ldh+1]);      // hh(2,2) and duplicate
//        __m128 x3 = _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+1]));        // load hh(2,2) , hh(3,2) | hh(2,2) , hh(3,2) in double precision representations
        __m128 h_2_1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+1])) ); // h_2_1 contains four times hh[ldh+1]
	__m128 h_3_2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+1])));
	__m128 h_3_1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+2])));
	__m128 h_4_3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+1])));
	__m128 h_4_2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+2])));
	__m128 h_4_1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+3])));

//	__m128 h_3_2 = _mm_loaddup_pd(&hh[(ldh*2)+1]);
//	__m128 h_3_1 = _mm_loaddup_pd(&hh[(ldh*2)+2]);
//	__m128 h_4_3 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
//	__m128 h_4_2 = _mm_loaddup_pd(&hh[(ldh*3)+2]);
//	__m128 h_4_1 = _mm_loaddup_pd(&hh[(ldh*3)+3]);

	register __m128 w1 = _mm_add_ps(a4_1, _mm_mul_ps(a3_1, h_4_3));
	w1 = _mm_add_ps(w1, _mm_mul_ps(a2_1, h_4_2));
	w1 = _mm_add_ps(w1, _mm_mul_ps(a1_1, h_4_1));
	register __m128 z1 = _mm_add_ps(a3_1, _mm_mul_ps(a2_1, h_3_2));
	z1 = _mm_add_ps(z1, _mm_mul_ps(a1_1, h_3_1));
	register __m128 y1 = _mm_add_ps(a2_1, _mm_mul_ps(a1_1, h_2_1));
	register __m128 x1 = a1_1;

//	__m128 a1_2 = _mm_load_ps(&q[(ldq*3)+4]);
//	__m128 a2_2 = _mm_load_ps(&q[(ldq*2)+4]);
//	__m128 a3_2 = _mm_load_ps(&q[ldq+4]);
//	__m128 a4_2 = _mm_load_ps(&q[0+4]);       // q(5,1) | ... q(8,1)
//
//	register __m128 w2 = _mm_add_ps(a4_2, _mm_mul_ps(a3_2, h_4_3));
//	w2 = _mm_add_ps(w2, _mm_mul_ps(a2_2, h_4_2));
//	w2 = _mm_add_ps(w2, _mm_mul_ps(a1_2, h_4_1));
//	register __m128 z2 = _mm_add_ps(a3_2, _mm_mul_ps(a2_2, h_3_2));
//	z2 = _mm_add_ps(z2, _mm_mul_ps(a1_2, h_3_1));
//	register __m128 y2 = _mm_add_ps(a2_2, _mm_mul_ps(a1_2, h_2_1));
//	register __m128 x2 = a1_2;
//
//	__m128 a1_3 = _mm_load_ps(&q[(ldq*3)+8]);
//	__m128 a2_3 = _mm_load_ps(&q[(ldq*2)+8]);
//	__m128 a3_3 = _mm_load_ps(&q[ldq+8]);
//	__m128 a4_3 = _mm_load_ps(&q[0+8]);    // q(9,1) | .. | q(12,1)
//
//	register __m128 w3 = _mm_add_ps(a4_3, _mm_mul_ps(a3_3, h_4_3));
//	w3 = _mm_add_ps(w3, _mm_mul_ps(a2_3, h_4_2));
//	w3 = _mm_add_ps(w3, _mm_mul_ps(a1_3, h_4_1));
//	register __m128 z3 = _mm_add_ps(a3_3, _mm_mul_ps(a2_3, h_3_2));
//	z3 = _mm_add_ps(z3, _mm_mul_ps(a1_3, h_3_1));
//	register __m128 y3 = _mm_add_ps(a2_3, _mm_mul_ps(a1_3, h_2_1));
//	register __m128 x3 = a1_3;

	__m128 q1;
//	__m128 q2;
//	__m128 q3;

	__m128 h1;
	__m128 h2;
	__m128 h3;
	__m128 h4;

	for(i = 4; i < nb; i++)
	{
	        h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[i-3])));
//		h1 = _mm_loaddup_pd(&hh[i-3]);
		q1 = _mm_load_ps(&q[i*ldq]);
//		q2 = _mm_load_ps(&q[(i*ldq)+4]);
//		q3 = _mm_load_ps(&q[(i*ldq)+8]);

		x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
//		x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
//		x3 = _mm_add_ps(x3, _mm_mul_ps(q3,h1));
	        h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+i-2])));
//		h2 = _mm_loaddup_pd(&hh[ldh+i-2]);

		y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
//		y2 = _mm_add_ps(y2, _mm_mul_ps(q2,h2));
//		y3 = _mm_add_ps(y3, _mm_mul_ps(q3,h2));
	        h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+i-1])));
//		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-1]);

		z1 = _mm_add_ps(z1, _mm_mul_ps(q1,h3));
//		z2 = _mm_add_ps(z2, _mm_mul_ps(q2,h3));
//		z3 = _mm_add_ps(z3, _mm_mul_ps(q3,h3));
	        h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+i])));
//		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i]);

		w1 = _mm_add_ps(w1, _mm_mul_ps(q1,h4));
//		w2 = _mm_add_ps(w2, _mm_mul_ps(q2,h4));
//		w3 = _mm_add_ps(w3, _mm_mul_ps(q3,h4));
	}
	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-3])));
//	h1 = _mm_loaddup_pd(&hh[nb-3]);

	q1 = _mm_load_ps(&q[nb*ldq]);
//	q2 = _mm_load_ps(&q[(nb*ldq)+4]);
//	q3 = _mm_load_ps(&q[(nb*ldq)+8]);

	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
//	x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
//	x3 = _mm_add_ps(x3, _mm_mul_ps(q3,h1));
	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+nb-2])));
//	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);

	y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
//	y2 = _mm_add_ps(y2, _mm_mul_ps(q2,h2));
//	y3 = _mm_add_ps(y3, _mm_mul_ps(q3,h2));
	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+nb-1])));
//	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);

	z1 = _mm_add_ps(z1, _mm_mul_ps(q1,h3));
//	z2 = _mm_add_ps(z2, _mm_mul_ps(q2,h3));
//	z3 = _mm_add_ps(z3, _mm_mul_ps(q3,h3));
	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-2])));
	//	h1 = _mm_loaddup_pd(&hh[nb-2]);

	q1 = _mm_load_ps(&q[(nb+1)*ldq]);
//	q2 = _mm_load_ps(&q[((nb+1)*ldq)+4]);
//	q3 = _mm_load_ps(&q[((nb+1)*ldq)+8]);

	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
//	x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
//	x3 = _mm_add_ps(x3, _mm_mul_ps(q3,h1));
        h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*1)+nb-1])));
//	h2 = _mm_loaddup_pd(&hh[(ldh*1)+nb-1]);

	y1 = _mm_add_ps(y1, _mm_mul_ps(q1,h2));
//	y2 = _mm_add_ps(y2, _mm_mul_ps(q2,h2));
//	y3 = _mm_add_ps(y3, _mm_mul_ps(q3,h2));
        h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-1])));
	//	h1 = _mm_loaddup_pd(&hh[nb-1]);

	q1 = _mm_load_ps(&q[(nb+2)*ldq]);
//	q2 = _mm_load_ps(&q[((nb+2)*ldq)+4]);
//	q3 = _mm_load_ps(&q[((nb+2)*ldq)+8]);

	x1 = _mm_add_ps(x1, _mm_mul_ps(q1,h1));
//	x2 = _mm_add_ps(x2, _mm_mul_ps(q2,h1));
//	x3 = _mm_add_ps(x3, _mm_mul_ps(q3,h1));

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [6 x nb+3]
	/////////////////////////////////////////////////////
        __m128 tau1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[0])));
//	__m128 tau1 = _mm_loaddup_pd(&hh[0]);

	h1 = tau1;
	x1 = _mm_mul_ps(x1, h1);
//	x2 = _mm_mul_ps(x2, h1);
//	x3 = _mm_mul_ps(x3, h1);
        __m128 tau2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh])));
        __m128 vs_1_2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_1_2)));

//	__m128 tau2 = _mm_loaddup_pd(&hh[ldh]);
//	__m128 vs_1_2 = _mm_loaddup_pd(&s_1_2);

	h1 = tau2;
	h2 = _mm_mul_ps(h1, vs_1_2);

	y1 = _mm_sub_ps(_mm_mul_ps(y1,h1), _mm_mul_ps(x1,h2));
//	y2 = _mm_sub_ps(_mm_mul_ps(y2,h1), _mm_mul_ps(x2,h2));
//	y3 = _mm_sub_ps(_mm_mul_ps(y3,h1), _mm_mul_ps(x3,h2));
        __m128 tau3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh*2])));
	//	__m128 tau3 = _mm_loaddup_pd(&hh[ldh*2]);
        __m128 vs_1_3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_1_3)));
//	__m128 vs_1_3 = _mm_loaddup_pd(&s_1_3);
        __m128 vs_2_3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_2_3)));
//	__m128 vs_2_3 = _mm_loaddup_pd(&s_2_3);

	h1 = tau3;
	h2 = _mm_mul_ps(h1, vs_1_3);
	h3 = _mm_mul_ps(h1, vs_2_3);

	z1 = _mm_sub_ps(_mm_mul_ps(z1,h1), _mm_add_ps(_mm_mul_ps(y1,h3), _mm_mul_ps(x1,h2)));
//	z2 = _mm_sub_ps(_mm_mul_ps(z2,h1), _mm_add_ps(_mm_mul_ps(y2,h3), _mm_mul_ps(x2,h2)));
//	z3 = _mm_sub_ps(_mm_mul_ps(z3,h1), _mm_add_ps(_mm_mul_ps(y3,h3), _mm_mul_ps(x3,h2)));
        __m128 tau4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh*3])));
//	__m128 tau4 = _mm_loaddup_pd(&hh[ldh*3]);
        __m128 vs_1_4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_1_4)));
//	__m128 vs_1_4 = _mm_loaddup_pd(&s_1_4);
        __m128 vs_2_4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_2_4)));
//	__m128 vs_2_4 = _mm_loaddup_pd(&s_2_4);
        __m128 vs_3_4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&s_3_4)));
//	__m128 vs_3_4 = _mm_loaddup_pd(&s_3_4);

	h1 = tau4;
	h2 = _mm_mul_ps(h1, vs_1_4);
	h3 = _mm_mul_ps(h1, vs_2_4);
	h4 = _mm_mul_ps(h1, vs_3_4);

	w1 = _mm_sub_ps(_mm_mul_ps(w1,h1), _mm_add_ps(_mm_mul_ps(z1,h4), _mm_add_ps(_mm_mul_ps(y1,h3), _mm_mul_ps(x1,h2))));
//	w2 = _mm_sub_ps(_mm_mul_ps(w2,h1), _mm_add_ps(_mm_mul_ps(z2,h4), _mm_add_ps(_mm_mul_ps(y2,h3), _mm_mul_ps(x2,h2))));
//	w3 = _mm_sub_ps(_mm_mul_ps(w3,h1), _mm_add_ps(_mm_mul_ps(z3,h4), _mm_add_ps(_mm_mul_ps(y3,h3), _mm_mul_ps(x3,h2))));

	q1 = _mm_load_ps(&q[0]);
//	q2 = _mm_load_ps(&q[4]);
//	q3 = _mm_load_ps(&q[8]);
	q1 = _mm_sub_ps(q1, w1);
//	q2 = _mm_sub_ps(q2, w2);
//	q3 = _mm_sub_ps(q3, w3);
	_mm_store_ps(&q[0],q1);
//	_mm_store_ps(&q[4],q2);
//	_mm_store_ps(&q[8],q3);

	h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+1])));
//	h4 = _mm_loaddup_pd(&hh[(ldh*3)+1]);
	q1 = _mm_load_ps(&q[ldq]);
//	q2 = _mm_load_ps(&q[ldq+4]);
//	q3 = _mm_load_ps(&q[ldq+8]);

	q1 = _mm_sub_ps(q1, _mm_add_ps(z1, _mm_mul_ps(w1, h4)));
//	q2 = _mm_sub_ps(q2, _mm_add_ps(z2, _mm_mul_ps(w2, h4)));
//	q3 = _mm_sub_ps(q3, _mm_add_ps(z3, _mm_mul_ps(w3, h4)));

	_mm_store_ps(&q[ldq],q1);
//	_mm_store_ps(&q[ldq+4],q2);
//	_mm_store_ps(&q[ldq+8],q3);

	h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+2])));
//	h4 = _mm_loaddup_pd(&hh[(ldh*3)+2]);
	q1 = _mm_load_ps(&q[ldq*2]);
//	q2 = _mm_load_ps(&q[(ldq*2)+4]);
//	q3 = _mm_load_ps(&q[(ldq*2)+8]);
	q1 = _mm_sub_ps(q1, y1);
//	q2 = _mm_sub_ps(q2, y2);
//	q3 = _mm_sub_ps(q3, y3);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(w1, h4));
//	q2 = _mm_sub_ps(q2, _mm_mul_ps(w2, h4));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(w3, h4));

	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+1])));
//	h3 = _mm_loaddup_pd(&hh[(ldh*2)+1]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(z1, h3));
//	q2 = _mm_sub_ps(q2, _mm_mul_ps(z2, h3));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(z3, h3));

	_mm_store_ps(&q[ldq*2],q1);
//	_mm_store_ps(&q[(ldq*2)+4],q2);
//	_mm_store_ps(&q[(ldq*2)+8],q3);

	h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+3])));
//	h4 = _mm_loaddup_pd(&hh[(ldh*3)+3]);
	q1 = _mm_load_ps(&q[ldq*3]);
//	q2 = _mm_load_ps(&q[(ldq*3)+4]);
//	q3 = _mm_load_ps(&q[(ldq*3)+8]);
	q1 = _mm_sub_ps(q1, x1);
//	q2 = _mm_sub_ps(q2, x2);
//	q3 = _mm_sub_ps(q3, x3);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(w1, h4));
//	q2 = _mm_sub_ps(q2, _mm_mul_ps(w2, h4));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(w3, h4));

	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+1])));
//	h2 = _mm_loaddup_pd(&hh[ldh+1]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(y1, h2));
//	q2 = _mm_sub_ps(q2, _mm_mul_ps(y2, h2));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(y3, h2));

	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+2])));
	//	h3 = _mm_loaddup_pd(&hh[(ldh*2)+2]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(z1, h3));
//	q2 = _mm_sub_ps(q2, _mm_mul_ps(z2, h3));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(z3, h3));
	_mm_store_ps(&q[ldq*3], q1);
//	_mm_store_ps(&q[(ldq*3)+4], q2);
//	_mm_store_ps(&q[(ldq*3)+8], q3);

	for (i = 4; i < nb; i++)
	{
	        h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[i-3])));
//		h1 = _mm_loaddup_pd(&hh[i-3]);

		q1 = _mm_load_ps(&q[i*ldq]);
//		q2 = _mm_load_ps(&q[(i*ldq)+4]);
//		q3 = _mm_load_ps(&q[(i*ldq)+8]);

		q1 = _mm_sub_ps(q1, _mm_mul_ps(x1,h1));
//		q2 = _mm_sub_ps(q2, _mm_mul_ps(x2,h1));
//		q3 = _mm_sub_ps(q3, _mm_mul_ps(x3,h1));

		h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+i-2])));
//		h2 = _mm_loaddup_pd(&hh[ldh+i-2]);

		q1 = _mm_sub_ps(q1, _mm_mul_ps(y1,h2));
//		q2 = _mm_sub_ps(q2, _mm_mul_ps(y2,h2));
//		q3 = _mm_sub_ps(q3, _mm_mul_ps(y3,h2));

		h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+i-1])));
//		h3 = _mm_loaddup_pd(&hh[(ldh*2)+i-1]);

		q1 = _mm_sub_ps(q1, _mm_mul_ps(z1,h3));
//		q2 = _mm_sub_ps(q2, _mm_mul_ps(z2,h3));
//		q3 = _mm_sub_ps(q3, _mm_mul_ps(z3,h3));

		h4 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*3)+i])));
//		h4 = _mm_loaddup_pd(&hh[(ldh*3)+i]);

		q1 = _mm_sub_ps(q1, _mm_mul_ps(w1,h4));
//		q2 = _mm_sub_ps(q2, _mm_mul_ps(w2,h4));
//		q3 = _mm_sub_ps(q3, _mm_mul_ps(w3,h4));

		_mm_store_ps(&q[i*ldq],q1);
//		_mm_store_ps(&q[(i*ldq)+4],q2);
//		_mm_store_ps(&q[(i*ldq)+8],q3);
	}

	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-3])));
//	h1 = _mm_loaddup_pd(&hh[nb-3]);
	q1 = _mm_load_ps(&q[nb*ldq]);
//	q2 = _mm_load_ps(&q[(nb*ldq)+4]);
//	q3 = _mm_load_ps(&q[(nb*ldq)+8]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(x1, h1));
//	q2 = _mm_sub_ps(q2, _mm_mul_ps(x2, h1));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(x3, h1));
	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+nb-2])));
//	h2 = _mm_loaddup_pd(&hh[ldh+nb-2]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(y1, h2));
//	q2 = _mm_sub_ps(q2, _mm_mul_ps(y2, h2));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(y3, h2));
	h3 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[(ldh*2)+nb-1])));
//	h3 = _mm_loaddup_pd(&hh[(ldh*2)+nb-1]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(z1, h3));
//	q2 = _mm_sub_ps(q2, _mm_mul_ps(z2, h3));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(z3, h3));

	_mm_store_ps(&q[nb*ldq],q1);
//	_mm_store_ps(&q[(nb*ldq)+4],q2);
//	_mm_store_ps(&q[(nb*ldq)+8],q3);
	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-2])));
//	h1 = _mm_loaddup_pd(&hh[nb-2]);
	q1 = _mm_load_ps(&q[(nb+1)*ldq]);
//	q2 = _mm_load_ps(&q[((nb+1)*ldq)+4]);
//	q3 = _mm_load_ps(&q[((nb+1)*ldq)+8]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(x1, h1));
//	q2 = _mm_sub_ps(q2, _mm_mul_ps(x2, h1));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(x3, h1));
	h2 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[ldh+nb-1])));
//	h2 = _mm_loaddup_pd(&hh[ldh+nb-1]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(y1, h2));
//	q2 = _mm_sub_ps(q2, _mm_mul_ps(y2, h2));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(y3, h2));

	_mm_store_ps(&q[(nb+1)*ldq],q1);
//	_mm_store_ps(&q[((nb+1)*ldq)+4],q2);
//	_mm_store_ps(&q[((nb+1)*ldq)+8],q3);

	h1 = _mm_moveldup_ps( _mm_castpd_ps(_mm_loaddup_pd( (double *)&hh[nb-1])));
//	h1 = _mm_loaddup_ps(&hh[nb-1]);
	q1 = _mm_load_ps(&q[(nb+2)*ldq]);
//	q2 = _mm_load_ps(&q[((nb+2)*ldq)+4]);
//	q3 = _mm_load_ps(&q[((nb+2)*ldq)+8]);

	q1 = _mm_sub_ps(q1, _mm_mul_ps(x1, h1));
//	q2 = _mm_sub_ps(q2, _mm_mul_ps(x2, h1));
//	q3 = _mm_sub_ps(q3, _mm_mul_ps(x3, h1));

	_mm_store_ps(&q[(nb+2)*ldq],q1);
//	_mm_store_ps(&q[((nb+2)*ldq)+4],q2);
//	_mm_store_ps(&q[((nb+2)*ldq)+8],q3);
}

