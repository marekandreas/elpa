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

#ifdef HAVE_AVX512
#define __ELPA_USE_FMA__
#define _mm512_FMA_ps(a,b,c) _mm512_fmadd_ps(a,b,c)
#define _mm512_NFMA_ps(a,b,c) _mm512_fnmadd_ps(a,b,c)
#define _mm512_FMSUB_ps(a,b,c) _mm512_fmsub_ps(a,b,c)
#endif


//Forward declaration
__forceinline void hh_trafo_kernel_16_AVX_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);
__forceinline void hh_trafo_kernel_32_AVX_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);
__forceinline void hh_trafo_kernel_48_AVX_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4);

void quad_hh_trafo_real_avx_avx2_4hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);

/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine quad_hh_trafo_real_avx512_4hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="quad_hh_trafo_real_avx512_4hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_float)      :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/

void quad_hh_trafo_real_avx512_4hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
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

	for (i = 0; i < nq-32; i+=48)
	{
		hh_trafo_kernel_48_AVX_4hv_single(&q[i], hh, nb, ldq, ldh, s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
	}
	if (nq == i)
	{
		return;
	}
	if (nq-i == 32)
	{
		hh_trafo_kernel_32_AVX_4hv_single(&q[i], hh, nb, ldq, ldh,  s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
	}
        else
	{
		hh_trafo_kernel_16_AVX_4hv_single(&q[i], hh, nb, ldq, ldh,  s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4);
	}


}


/**
 * Unrolled kernel that computes
 * 24 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_48_AVX_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [12 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m512 a1_1 = _mm512_load_ps(&q[ldq*3]); // q(1,4) |                                                  .. | q(8,4)
	__m512 a2_1 = _mm512_load_ps(&q[ldq*2]); // q(1,3) | q(2,3) | q(3,3) | q(4,3) | q(5,3) | q(6,3) | q(7,3) | q(8,3)
	__m512 a3_1 = _mm512_load_ps(&q[ldq]);   // q(1,2) | q(2,2) | q(3,2) | q(4,2) | q(5,2) | q(6,2) | q(7,2) | q(8,2)
	__m512 a4_1 = _mm512_load_ps(&q[0]);     // q(1,1) | q(2,1) | q(3,1) | q(4,1) | q(5,1) | q(6,1) | q(7,1) | q(8,1)

	__m512 h_2_1 = _mm512_set1_ps(hh[ldh+1]);
	__m512 h_3_2 = _mm512_set1_ps(hh[(ldh*2)+1]);
	__m512 h_3_1 = _mm512_set1_ps(hh[(ldh*2)+2]);
	__m512 h_4_3 = _mm512_set1_ps(hh[(ldh*3)+1]);
	__m512 h_4_2 = _mm512_set1_ps(hh[(ldh*3)+2]);
	__m512 h_4_1 = _mm512_set1_ps(hh[(ldh*3)+3]);

	register __m512 w1 = _mm512_FMA_ps(a3_1, h_4_3, a4_1);
	w1 = _mm512_FMA_ps(a2_1, h_4_2, w1);
	w1 = _mm512_FMA_ps(a1_1, h_4_1, w1);
	register __m512 z1 = _mm512_FMA_ps(a2_1, h_3_2, a3_1);
	z1 = _mm512_FMA_ps(a1_1, h_3_1, z1);
	register __m512 y1 = _mm512_FMA_ps(a1_1, h_2_1, a2_1);
	register __m512 x1 = a1_1;

	__m512 a1_2 = _mm512_load_ps(&q[(ldq*3)+16]); // q(9,4) | ...          | q(16,4)
	__m512 a2_2 = _mm512_load_ps(&q[(ldq*2)+16]);
	__m512 a3_2 = _mm512_load_ps(&q[ldq+16]);     // q(9,2) | ...          | q(16,2)
	__m512 a4_2 = _mm512_load_ps(&q[0+16]);       // q(9,1) | q(10,1) .... | q(16,1)

	register __m512 w2 = _mm512_FMA_ps(a3_2, h_4_3, a4_2);
	w2 = _mm512_FMA_ps(a2_2, h_4_2, w2);
	w2 = _mm512_FMA_ps(a1_2, h_4_1, w2);
	register __m512 z2 = _mm512_FMA_ps(a2_2, h_3_2, a3_2);
	z2 = _mm512_FMA_ps(a1_2, h_3_1, z2);
	register __m512 y2 = _mm512_FMA_ps(a1_2, h_2_1, a2_2);
	register __m512 x2 = a1_2;

	__m512 a1_3 = _mm512_load_ps(&q[(ldq*3)+32]);
	__m512 a2_3 = _mm512_load_ps(&q[(ldq*2)+32]);
	__m512 a3_3 = _mm512_load_ps(&q[ldq+32]);
	__m512 a4_3 = _mm512_load_ps(&q[0+32]);

	register __m512 w3 = _mm512_FMA_ps(a3_3, h_4_3, a4_3);
	w3 = _mm512_FMA_ps(a2_3, h_4_2, w3);
	w3 = _mm512_FMA_ps(a1_3, h_4_1, w3);
	register __m512 z3 = _mm512_FMA_ps(a2_3, h_3_2, a3_3);
	z3 = _mm512_FMA_ps(a1_3, h_3_1, z3);
	register __m512 y3 = _mm512_FMA_ps(a1_3, h_2_1, a2_3);
	register __m512 x3 = a1_3;

	__m512 q1;
	__m512 q2;
	__m512 q3;

	__m512 h1;
	__m512 h2;
	__m512 h3;
	__m512 h4;

	for(i = 4; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-3]);
		q1 = _mm512_load_ps(&q[i*ldq]);       // | q(i,2) | q(i+1,2) | q(i+2,2) | q(i+3,2) | q(i+4,2) | q(i+5,2) | q(i+5,2) | q(i+7,2)
		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
		q3 = _mm512_load_ps(&q[(i*ldq)+32]);

		x1 = _mm512_FMA_ps(q1, h1, x1);
		x2 = _mm512_FMA_ps(q2, h1, x2);
		x3 = _mm512_FMA_ps(q3, h1, x3);

		h2 = _mm512_set1_ps(hh[ldh+i-2]);

		y1 = _mm512_FMA_ps(q1, h2, y1);
		y2 = _mm512_FMA_ps(q2, h2, y2);
		y3 = _mm512_FMA_ps(q3, h2, y3);

		h3 = _mm512_set1_ps(hh[(ldh*2)+i-1]);

		z1 = _mm512_FMA_ps(q1, h3, z1);
		z2 = _mm512_FMA_ps(q2, h3, z2);
		z3 = _mm512_FMA_ps(q3, h3, z3);

		h4 = _mm512_set1_ps(hh[(ldh*3)+i]);

		w1 = _mm512_FMA_ps(q1, h4, w1);
		w2 = _mm512_FMA_ps(q2, h4, w2);
		w3 = _mm512_FMA_ps(q3, h4, w3);
	}

	h1 = _mm512_set1_ps(hh[nb-3]);

	q1 = _mm512_load_ps(&q[nb*ldq]);
	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	x3 = _mm512_FMA_ps(q3, h1, x3);

	h2 = _mm512_set1_ps(hh[ldh+nb-2]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
	y2 = _mm512_FMA_ps(q2, h2, y2);
	y3 = _mm512_FMA_ps(q3, h2, y3);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-1]);

	z1 = _mm512_FMA_ps(q1, h3, z1);
	z2 = _mm512_FMA_ps(q2, h3, z2);
	z3 = _mm512_FMA_ps(q3, h3, z3);

	h1 = _mm512_set1_ps(hh[nb-2]);

	q1 = _mm512_load_ps(&q[(nb+1)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+1)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+1)*ldq)+32]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	x3 = _mm512_FMA_ps(q3, h1, x3);

	h2 = _mm512_set1_ps(hh[(ldh*1)+nb-1]);

	y1 = _mm512_FMA_ps(q1, h2, y1);
	y2 = _mm512_FMA_ps(q2, h2, y2);
	y3 = _mm512_FMA_ps(q3, h2, y3);

	h1 = _mm512_set1_ps(hh[nb-1]);

	q1 = _mm512_load_ps(&q[(nb+2)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+2)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+2)*ldq)+32]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	x3 = _mm512_FMA_ps(q3, h1, x3);

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [12 x nb+3]
	/////////////////////////////////////////////////////

	__m512 tau1 = _mm512_set1_ps(hh[0]);

	h1 = tau1;
	x1 = _mm512_mul_ps(x1, h1);
	x2 = _mm512_mul_ps(x2, h1);
	x3 = _mm512_mul_ps(x3, h1);

	__m512 tau2 = _mm512_set1_ps(hh[ldh]);
	__m512 vs_1_2 = _mm512_set1_ps(s_1_2);

	h1 = tau2;
	h2 = _mm512_mul_ps(h1, vs_1_2);

	y1 = _mm512_FMSUB_ps(y1, h1, _mm512_mul_ps(x1,h2));
	y2 = _mm512_FMSUB_ps(y2, h1, _mm512_mul_ps(x2,h2));
	y3 = _mm512_FMSUB_ps(y3, h1, _mm512_mul_ps(x3,h2));

	__m512 tau3 = _mm512_set1_ps(hh[ldh*2]);
	__m512 vs_1_3 = _mm512_set1_ps(s_1_3);
	__m512 vs_2_3 = _mm512_set1_ps(s_2_3);

	h1 = tau3;
	h2 = _mm512_mul_ps(h1, vs_1_3);
	h3 = _mm512_mul_ps(h1, vs_2_3);

	z1 = _mm512_FMSUB_ps(z1, h1, _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2)));
	z2 = _mm512_FMSUB_ps(z2, h1, _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2)));
	z3 = _mm512_FMSUB_ps(z3, h1, _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2)));

	__m512 tau4 = _mm512_set1_ps(hh[ldh*3]);
	__m512 vs_1_4 = _mm512_set1_ps(s_1_4);
	__m512 vs_2_4 = _mm512_set1_ps(s_2_4);
	__m512 vs_3_4 = _mm512_set1_ps(s_3_4);

	h1 = tau4;
	h2 = _mm512_mul_ps(h1, vs_1_4);
	h3 = _mm512_mul_ps(h1, vs_2_4);
	h4 = _mm512_mul_ps(h1, vs_3_4);

	w1 = _mm512_FMSUB_ps(w1, h1, _mm512_FMA_ps(z1, h4, _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2))));
	w2 = _mm512_FMSUB_ps(w2, h1, _mm512_FMA_ps(z2, h4, _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2))));
	w3 = _mm512_FMSUB_ps(w3, h1, _mm512_FMA_ps(z3, h4, _mm512_FMA_ps(y3, h3, _mm512_mul_ps(x3,h2))));

	q1 = _mm512_load_ps(&q[0]);
	q2 = _mm512_load_ps(&q[16]);
	q3 = _mm512_load_ps(&q[32]);
	q1 = _mm512_sub_ps(q1, w1);
	q2 = _mm512_sub_ps(q2, w2);
	q3 = _mm512_sub_ps(q3, w3);
	_mm512_store_ps(&q[0],q1);
	_mm512_store_ps(&q[16],q2);
	_mm512_store_ps(&q[32],q3);

	h4 = _mm512_set1_ps(hh[(ldh*3)+1]);
	q1 = _mm512_load_ps(&q[ldq]);
	q2 = _mm512_load_ps(&q[ldq+16]);
	q3 = _mm512_load_ps(&q[ldq+32]);

	q1 = _mm512_sub_ps(q1, _mm512_FMA_ps(w1, h4, z1));
	q2 = _mm512_sub_ps(q2, _mm512_FMA_ps(w2, h4, z2));
	q3 = _mm512_sub_ps(q3, _mm512_FMA_ps(w3, h4, z3));

	_mm512_store_ps(&q[ldq],q1);
	_mm512_store_ps(&q[ldq+16],q2);
	_mm512_store_ps(&q[ldq+32],q3);

	h4 = _mm512_set1_ps(hh[(ldh*3)+2]);
	q1 = _mm512_load_ps(&q[ldq*2]);
	q2 = _mm512_load_ps(&q[(ldq*2)+16]);
	q3 = _mm512_load_ps(&q[(ldq*2)+32]);
	q1 = _mm512_sub_ps(q1, y1);
	q2 = _mm512_sub_ps(q2, y2);
	q3 = _mm512_sub_ps(q3, y3);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
	q3 = _mm512_NFMA_ps(w3, h4, q3);

	h3 = _mm512_set1_ps(hh[(ldh*2)+1]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
	q3 = _mm512_NFMA_ps(z3, h3, q3);

	_mm512_store_ps(&q[ldq*2],q1);
	_mm512_store_ps(&q[(ldq*2)+16],q2);
	_mm512_store_ps(&q[(ldq*2)+32],q3);

	h4 = _mm512_set1_ps(hh[(ldh*3)+3]);
	q1 = _mm512_load_ps(&q[ldq*3]);
	q2 = _mm512_load_ps(&q[(ldq*3)+16]);
	q3 = _mm512_load_ps(&q[(ldq*3)+32]);

	q1 = _mm512_sub_ps(q1, x1);
	q2 = _mm512_sub_ps(q2, x2);
	q3 = _mm512_sub_ps(q3, x3);

	q1 = _mm512_NFMA_ps(w1, h4, q1);
	q2 = _mm512_NFMA_ps(w2, h4, q2);
	q3 = _mm512_NFMA_ps(w3, h4, q3);

	h2 = _mm512_set1_ps(hh[ldh+1]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
	q3 = _mm512_NFMA_ps(y3, h2, q3);

	h3 = _mm512_set1_ps(hh[(ldh*2)+2]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
	q3 = _mm512_NFMA_ps(z3, h3, q3);

	_mm512_store_ps(&q[ldq*3], q1);
	_mm512_store_ps(&q[(ldq*3)+16], q2);
	_mm512_store_ps(&q[(ldq*3)+32], q3);

	for (i = 4; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-3]);

		q1 = _mm512_load_ps(&q[i*ldq]);
		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
		q3 = _mm512_load_ps(&q[(i*ldq)+32]);

		q1 = _mm512_NFMA_ps(x1, h1, q1);
		q2 = _mm512_NFMA_ps(x2, h1, q2);
		q3 = _mm512_NFMA_ps(x3, h1, q3);

		h2 = _mm512_set1_ps(hh[ldh+i-2]);

		q1 = _mm512_NFMA_ps(y1, h2, q1);
		q2 = _mm512_NFMA_ps(y2, h2, q2);
		q3 = _mm512_NFMA_ps(y3, h2, q3);

		h3 = _mm512_set1_ps(hh[(ldh*2)+i-1]);

		q1 = _mm512_NFMA_ps(z1, h3, q1);
		q2 = _mm512_NFMA_ps(z2, h3, q2);
		q3 = _mm512_NFMA_ps(z3, h3, q3);

		h4 = _mm512_set1_ps(hh[(ldh*3)+i]);

		q1 = _mm512_NFMA_ps(w1, h4, q1);
		q2 = _mm512_NFMA_ps(w2, h4, q2);
		q3 = _mm512_NFMA_ps(w3, h4, q3);

		_mm512_store_ps(&q[i*ldq],q1);
		_mm512_store_ps(&q[(i*ldq)+16],q2);
		_mm512_store_ps(&q[(i*ldq)+32],q3);
	}

	h1 = _mm512_set1_ps(hh[nb-3]);
	q1 = _mm512_load_ps(&q[nb*ldq]);
	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);
	q3 = _mm512_load_ps(&q[(nb*ldq)+32]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
	q3 = _mm512_NFMA_ps(x3, h1, q3);

	h2 = _mm512_set1_ps(hh[ldh+nb-2]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
	q3 = _mm512_NFMA_ps(y3, h2, q3);

	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-1]);

	q1 = _mm512_NFMA_ps(z1, h3, q1);
	q2 = _mm512_NFMA_ps(z2, h3, q2);
	q3 = _mm512_NFMA_ps(z3, h3, q3);

	_mm512_store_ps(&q[nb*ldq],q1);
	_mm512_store_ps(&q[(nb*ldq)+16],q2);
	_mm512_store_ps(&q[(nb*ldq)+32],q3);

	h1 = _mm512_set1_ps(hh[nb-2]);
	q1 = _mm512_load_ps(&q[(nb+1)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+1)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+1)*ldq)+32]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
	q3 = _mm512_NFMA_ps(x3, h1, q3);

	h2 = _mm512_set1_ps(hh[ldh+nb-1]);

	q1 = _mm512_NFMA_ps(y1, h2, q1);
	q2 = _mm512_NFMA_ps(y2, h2, q2);
	q3 = _mm512_NFMA_ps(y3, h2, q3);

	_mm512_store_ps(&q[(nb+1)*ldq],q1);
	_mm512_store_ps(&q[((nb+1)*ldq)+16],q2);
	_mm512_store_ps(&q[((nb+1)*ldq)+32],q3);

	h1 = _mm512_set1_ps(hh[nb-1]);
	q1 = _mm512_load_ps(&q[(nb+2)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+2)*ldq)+16]);
	q3 = _mm512_load_ps(&q[((nb+2)*ldq)+32]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);
	q3 = _mm512_NFMA_ps(x3, h1, q3);

	_mm512_store_ps(&q[(nb+2)*ldq],q1);
	_mm512_store_ps(&q[((nb+2)*ldq)+16],q2);
	_mm512_store_ps(&q[((nb+2)*ldq)+32],q3);
}

/**
 * Unrolled kernel that computes
 * 32 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_32_AVX_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m512 a1_1 = _mm512_load_ps(&q[ldq*3]);  // q(1,4) | ...                                                 | q(8,4)
	__m512 a2_1 = _mm512_load_ps(&q[ldq*2]);  // q(1,3) | ...                                                 | q(8,3)
	__m512 a3_1 = _mm512_load_ps(&q[ldq]);    // q(1,2) | ...                                                 | q(8,2)
	__m512 a4_1 = _mm512_load_ps(&q[0]);      // q(1,1) | q(2,1) | q(3,1) | q(4,1) | q(5,1) | q(6,1) | q(7,1) | q(8,1)

	__m512 h_2_1 = _mm512_set1_ps(hh[ldh+1]);
	__m512 h_3_2 = _mm512_set1_ps(hh[(ldh*2)+1]);
	__m512 h_3_1 = _mm512_set1_ps(hh[(ldh*2)+2]);
	__m512 h_4_3 = _mm512_set1_ps(hh[(ldh*3)+1]);
	__m512 h_4_2 = _mm512_set1_ps(hh[(ldh*3)+2]);
	__m512 h_4_1 = _mm512_set1_ps(hh[(ldh*3)+3]);

	__m512 w1 = _mm512_FMA_ps(a3_1, h_4_3, a4_1);
	w1 = _mm512_FMA_ps(a2_1, h_4_2, w1);
	w1 = _mm512_FMA_ps(a1_1, h_4_1, w1);
	__m512 z1 = _mm512_FMA_ps(a2_1, h_3_2, a3_1);
	z1 = _mm512_FMA_ps(a1_1, h_3_1, z1);
	__m512 y1 = _mm512_FMA_ps(a1_1, h_2_1, a2_1);
	__m512 x1 = a1_1;

	__m512 a1_2 = _mm512_load_ps(&q[(ldq*3)+16]);
	__m512 a2_2 = _mm512_load_ps(&q[(ldq*2)+16]);
	__m512 a3_2 = _mm512_load_ps(&q[ldq+16]);
	__m512 a4_2 = _mm512_load_ps(&q[0+16]);       // q(9,1) | q(10,1) | q(11,1) | q(12,1) | q(13,1) | q(14,1) | q(15,1) | q(16,1)

	__m512 w2 = _mm512_FMA_ps(a3_2, h_4_3, a4_2);
	w2 = _mm512_FMA_ps(a2_2, h_4_2, w2);
	w2 = _mm512_FMA_ps(a1_2, h_4_1, w2);
	__m512 z2 = _mm512_FMA_ps(a2_2, h_3_2, a3_2);
	z2 = _mm512_FMA_ps(a1_2, h_3_1, z2);
	__m512 y2 = _mm512_FMA_ps(a1_2, h_2_1, a2_2);
	__m512 x2 = a1_2;

	__m512 q1;
	__m512 q2;

	__m512 h1;
	__m512 h2;
	__m512 h3;
	__m512 h4;

	for(i = 4; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-3]);
		h2 = _mm512_set1_ps(hh[ldh+i-2]);
		h3 = _mm512_set1_ps(hh[(ldh*2)+i-1]);
		h4 = _mm512_set1_ps(hh[(ldh*3)+i]);

		q1 = _mm512_load_ps(&q[i*ldq]);

		x1 = _mm512_FMA_ps(q1, h1, x1);
		y1 = _mm512_FMA_ps(q1, h2, y1);
		z1 = _mm512_FMA_ps(q1, h3, z1);
		w1 = _mm512_FMA_ps(q1, h4, w1);

		q2 = _mm512_load_ps(&q[(i*ldq)+16]);

		x2 = _mm512_FMA_ps(q2, h1, x2);
		y2 = _mm512_FMA_ps(q2, h2, y2);
		z2 = _mm512_FMA_ps(q2, h3, z2);
		w2 = _mm512_FMA_ps(q2, h4, w2);
	}

	h1 = _mm512_set1_ps(hh[nb-3]);
	h2 = _mm512_set1_ps(hh[ldh+nb-2]);
	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-1]);

	q1 = _mm512_load_ps(&q[nb*ldq]);
	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	y1 = _mm512_FMA_ps(q1, h2, y1);
	y2 = _mm512_FMA_ps(q2, h2, y2);
	z1 = _mm512_FMA_ps(q1, h3, z1);
	z2 = _mm512_FMA_ps(q2, h3, z2);

	h1 = _mm512_set1_ps(hh[nb-2]);
	h2 = _mm512_set1_ps(hh[(ldh*1)+nb-1]);

	q1 = _mm512_load_ps(&q[(nb+1)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+1)*ldq)+16]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);
	y1 = _mm512_FMA_ps(q1, h2, y1);
	y2 = _mm512_FMA_ps(q2, h2, y2);

	h1 = _mm512_set1_ps(hh[nb-1]);

	q1 = _mm512_load_ps(&q[(nb+2)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+2)*ldq)+16]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	x2 = _mm512_FMA_ps(q2, h1, x2);

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	__m512 tau1 = _mm512_set1_ps(hh[0]);
	__m512 tau2 = _mm512_set1_ps(hh[ldh]);
	__m512 tau3 = _mm512_set1_ps(hh[ldh*2]);
	__m512 tau4 = _mm512_set1_ps(hh[ldh*3]);

	__m512 vs_1_2 = _mm512_set1_ps(s_1_2);
	__m512 vs_1_3 = _mm512_set1_ps(s_1_3);
	__m512 vs_2_3 = _mm512_set1_ps(s_2_3);
	__m512 vs_1_4 = _mm512_set1_ps(s_1_4);
	__m512 vs_2_4 = _mm512_set1_ps(s_2_4);
	__m512 vs_3_4 = _mm512_set1_ps(s_3_4);

	h1 = tau1;
	x1 = _mm512_mul_ps(x1, h1);
	x2 = _mm512_mul_ps(x2, h1);

	h1 = tau2;
	h2 = _mm512_mul_ps(h1, vs_1_2);

	y1 = _mm512_FMSUB_ps(y1, h1, _mm512_mul_ps(x1,h2));
	y2 = _mm512_FMSUB_ps(y2, h1, _mm512_mul_ps(x2,h2));

	h1 = tau3;
	h2 = _mm512_mul_ps(h1, vs_1_3);
	h3 = _mm512_mul_ps(h1, vs_2_3);

	z1 = _mm512_FMSUB_ps(z1, h1, _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2)));
	z2 = _mm512_FMSUB_ps(z2, h1, _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2)));

	h1 = tau4;
	h2 = _mm512_mul_ps(h1, vs_1_4);
	h3 = _mm512_mul_ps(h1, vs_2_4);
	h4 = _mm512_mul_ps(h1, vs_3_4);

	w1 = _mm512_FMSUB_ps(w1, h1, _mm512_FMA_ps(z1, h4, _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2))));
	w2 = _mm512_FMSUB_ps(w2, h1, _mm512_FMA_ps(z2, h4, _mm512_FMA_ps(y2, h3, _mm512_mul_ps(x2,h2))));

	q1 = _mm512_load_ps(&q[0]);
	q2 = _mm512_load_ps(&q[16]);
	q1 = _mm512_sub_ps(q1, w1);
	q2 = _mm512_sub_ps(q2, w2);
	_mm512_store_ps(&q[0],q1);
	_mm512_store_ps(&q[16],q2);

	h4 = _mm512_set1_ps(hh[(ldh*3)+1]);
	q1 = _mm512_load_ps(&q[ldq]);
	q2 = _mm512_load_ps(&q[ldq+16]);

	q1 = _mm512_sub_ps(q1, _mm512_FMA_ps(w1, h4, z1));
	q2 = _mm512_sub_ps(q2, _mm512_FMA_ps(w2, h4, z2));

	_mm512_store_ps(&q[ldq],q1);
	_mm512_store_ps(&q[ldq+16],q2);

	h3 = _mm512_set1_ps(hh[(ldh*2)+1]);
	h4 = _mm512_set1_ps(hh[(ldh*3)+2]);
	q1 = _mm512_load_ps(&q[ldq*2]);
	q2 = _mm512_load_ps(&q[(ldq*2)+16]);

        q1 = _mm512_sub_ps(q1, y1);
        q1 = _mm512_NFMA_ps(z1, h3, q1);
        q1 = _mm512_NFMA_ps(w1, h4, q1);
        q2 = _mm512_sub_ps(q2, y2);
        q2 = _mm512_NFMA_ps(z2, h3, q2);
        q2 = _mm512_NFMA_ps(w2, h4, q2);

	_mm512_store_ps(&q[ldq*2],q1);
	_mm512_store_ps(&q[(ldq*2)+16],q2);

	h2 = _mm512_set1_ps(hh[ldh+1]);
	h3 = _mm512_set1_ps(hh[(ldh*2)+2]);
	h4 = _mm512_set1_ps(hh[(ldh*3)+3]);
	q1 = _mm512_load_ps(&q[ldq*3]);
	q2 = _mm512_load_ps(&q[(ldq*3)+16]);

        q1 = _mm512_sub_ps(q1, x1);
        q1 = _mm512_NFMA_ps(y1, h2, q1);
        q1 = _mm512_NFMA_ps(z1, h3, q1);
        q1 = _mm512_NFMA_ps(w1, h4, q1);
        q2 = _mm512_sub_ps(q2, x2);
        q2 = _mm512_NFMA_ps(y2, h2, q2);
        q2 = _mm512_NFMA_ps(z2, h3, q2);
        q2 = _mm512_NFMA_ps(w2, h4, q2);

	_mm512_store_ps(&q[ldq*3], q1);
	_mm512_store_ps(&q[(ldq*3)+16], q2);

	for (i = 4; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-3]);
		h2 = _mm512_set1_ps(hh[ldh+i-2]);
		h3 = _mm512_set1_ps(hh[(ldh*2)+i-1]);
		h4 = _mm512_set1_ps(hh[(ldh*3)+i]);

		q1 = _mm512_load_ps(&q[i*ldq]);
		q2 = _mm512_load_ps(&q[(i*ldq)+16]);
                q1 = _mm512_NFMA_ps(x1, h1, q1);
                q1 = _mm512_NFMA_ps(y1, h2, q1);
                q1 = _mm512_NFMA_ps(z1, h3, q1);
                q1 = _mm512_NFMA_ps(w1, h4, q1);
                q2 = _mm512_NFMA_ps(x2, h1, q2);
                q2 = _mm512_NFMA_ps(y2, h2, q2);
                q2 = _mm512_NFMA_ps(z2, h3, q2);
                q2 = _mm512_NFMA_ps(w2, h4, q2);
		_mm512_store_ps(&q[i*ldq],q1);
		_mm512_store_ps(&q[(i*ldq)+16],q2);
	}

	h1 = _mm512_set1_ps(hh[nb-3]);
	h2 = _mm512_set1_ps(hh[ldh+nb-2]);
	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-1]);
	q1 = _mm512_load_ps(&q[nb*ldq]);
	q2 = _mm512_load_ps(&q[(nb*ldq)+16]);

        q1 = _mm512_NFMA_ps(x1, h1, q1);
        q1 = _mm512_NFMA_ps(y1, h2, q1);
        q1 = _mm512_NFMA_ps(z1, h3, q1);
        q2 = _mm512_NFMA_ps(x2, h1, q2);
        q2 = _mm512_NFMA_ps(y2, h2, q2);
        q2 = _mm512_NFMA_ps(z2, h3, q2);

	_mm512_store_ps(&q[nb*ldq],q1);
	_mm512_store_ps(&q[(nb*ldq)+16],q2);

	h1 = _mm512_set1_ps(hh[nb-2]);
	h2 = _mm512_set1_ps(hh[ldh+nb-1]);
	q1 = _mm512_load_ps(&q[(nb+1)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+1)*ldq)+16]);

        q1 = _mm512_NFMA_ps(x1, h1, q1);
        q1 = _mm512_NFMA_ps(y1, h2, q1);
        q2 = _mm512_NFMA_ps(x2, h1, q2);
        q2 = _mm512_NFMA_ps(y2, h2, q2);

	_mm512_store_ps(&q[(nb+1)*ldq],q1);
	_mm512_store_ps(&q[((nb+1)*ldq)+16],q2);

	h1 = _mm512_set1_ps(hh[nb-1]);
	q1 = _mm512_load_ps(&q[(nb+2)*ldq]);
	q2 = _mm512_load_ps(&q[((nb+2)*ldq)+16]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
	q2 = _mm512_NFMA_ps(x2, h1, q2);

	_mm512_store_ps(&q[(nb+2)*ldq],q1);
	_mm512_store_ps(&q[((nb+2)*ldq)+16],q2);
}

/**
 * Unrolled kernel that computes
 * 16 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 1 update is performed
 */
__forceinline void hh_trafo_kernel_16_AVX_4hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s_1_2, float s_1_3, float s_2_3, float s_1_4, float s_2_4, float s_3_4)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+3] * hh
	// hh contains four householder vectors
	/////////////////////////////////////////////////////
	int i;

	__m512 a1_1 = _mm512_load_ps(&q[ldq*3]);  // q(1,4) | ...                                                 | q(8,4)
	__m512 a2_1 = _mm512_load_ps(&q[ldq*2]);  // q(1,3) | ...                                                 | q(8,3)
	__m512 a3_1 = _mm512_load_ps(&q[ldq]);    // q(1,2) | ...                                                 | q(8,2)
	__m512 a4_1 = _mm512_load_ps(&q[0]);      // q(1,1) | q(2,1) | q(3,1) | q(4,1) | q(5,1) | q(6,1) | q(7,1) | q(8,1)

	__m512 h_2_1 = _mm512_set1_ps(hh[ldh+1]);
	__m512 h_3_2 = _mm512_set1_ps(hh[(ldh*2)+1]);
	__m512 h_3_1 = _mm512_set1_ps(hh[(ldh*2)+2]);
	__m512 h_4_3 = _mm512_set1_ps(hh[(ldh*3)+1]);
	__m512 h_4_2 = _mm512_set1_ps(hh[(ldh*3)+2]);
	__m512 h_4_1 = _mm512_set1_ps(hh[(ldh*3)+3]);

	__m512 w1 = _mm512_FMA_ps(a3_1, h_4_3, a4_1);
	w1 = _mm512_FMA_ps(a2_1, h_4_2, w1);
	w1 = _mm512_FMA_ps(a1_1, h_4_1, w1);
	__m512 z1 = _mm512_FMA_ps(a2_1, h_3_2, a3_1);
	z1 = _mm512_FMA_ps(a1_1, h_3_1, z1);
	__m512 y1 = _mm512_FMA_ps(a1_1, h_2_1, a2_1);
	__m512 x1 = a1_1;

	__m512 q1;

	__m512 h1;
	__m512 h2;
	__m512 h3;
	__m512 h4;

	for(i = 4; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-3]);
		h2 = _mm512_set1_ps(hh[ldh+i-2]);
		h3 = _mm512_set1_ps(hh[(ldh*2)+i-1]);
		h4 = _mm512_set1_ps(hh[(ldh*3)+i]);

		q1 = _mm512_load_ps(&q[i*ldq]);

		x1 = _mm512_FMA_ps(q1, h1, x1);
		y1 = _mm512_FMA_ps(q1, h2, y1);
		z1 = _mm512_FMA_ps(q1, h3, z1);
		w1 = _mm512_FMA_ps(q1, h4, w1);

	}

	h1 = _mm512_set1_ps(hh[nb-3]);
	h2 = _mm512_set1_ps(hh[ldh+nb-2]);
	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-1]);

	q1 = _mm512_load_ps(&q[nb*ldq]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	y1 = _mm512_FMA_ps(q1, h2, y1);
	z1 = _mm512_FMA_ps(q1, h3, z1);

	h1 = _mm512_set1_ps(hh[nb-2]);
	h2 = _mm512_set1_ps(hh[(ldh*1)+nb-1]);

	q1 = _mm512_load_ps(&q[(nb+1)*ldq]);

	x1 = _mm512_FMA_ps(q1, h1, x1);
	y1 = _mm512_FMA_ps(q1, h2, y1);

	h1 = _mm512_set1_ps(hh[nb-1]);

	q1 = _mm512_load_ps(&q[(nb+2)*ldq]);

	x1 = _mm512_FMA_ps(q1, h1, x1);

	/////////////////////////////////////////////////////
	// Rank-1 update of Q [8 x nb+3]
	/////////////////////////////////////////////////////

	__m512 tau1 = _mm512_set1_ps(hh[0]);
	__m512 tau2 = _mm512_set1_ps(hh[ldh]);
	__m512 tau3 = _mm512_set1_ps(hh[ldh*2]);
	__m512 tau4 = _mm512_set1_ps(hh[ldh*3]);

	__m512 vs_1_2 = _mm512_set1_ps(s_1_2);
	__m512 vs_1_3 = _mm512_set1_ps(s_1_3);
	__m512 vs_2_3 = _mm512_set1_ps(s_2_3);
	__m512 vs_1_4 = _mm512_set1_ps(s_1_4);
	__m512 vs_2_4 = _mm512_set1_ps(s_2_4);
	__m512 vs_3_4 = _mm512_set1_ps(s_3_4);

	h1 = tau1;
	x1 = _mm512_mul_ps(x1, h1);

	h1 = tau2;
	h2 = _mm512_mul_ps(h1, vs_1_2);
	y1 = _mm512_FMSUB_ps(y1, h1, _mm512_mul_ps(x1,h2));

	h1 = tau3;
	h2 = _mm512_mul_ps(h1, vs_1_3);
	h3 = _mm512_mul_ps(h1, vs_2_3);

	z1 = _mm512_FMSUB_ps(z1, h1, _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2)));

	h1 = tau4;
	h2 = _mm512_mul_ps(h1, vs_1_4);
	h3 = _mm512_mul_ps(h1, vs_2_4);
	h4 = _mm512_mul_ps(h1, vs_3_4);

	w1 = _mm512_FMSUB_ps(w1, h1, _mm512_FMA_ps(z1, h4, _mm512_FMA_ps(y1, h3, _mm512_mul_ps(x1,h2))));

	q1 = _mm512_load_ps(&q[0]);
	q1 = _mm512_sub_ps(q1, w1);
	_mm512_store_ps(&q[0],q1);

	h4 = _mm512_set1_ps(hh[(ldh*3)+1]);
	q1 = _mm512_load_ps(&q[ldq]);

	q1 = _mm512_sub_ps(q1, _mm512_FMA_ps(w1, h4, z1));

	_mm512_store_ps(&q[ldq],q1);

	h3 = _mm512_set1_ps(hh[(ldh*2)+1]);
	h4 = _mm512_set1_ps(hh[(ldh*3)+2]);
	q1 = _mm512_load_ps(&q[ldq*2]);

        q1 = _mm512_sub_ps(q1, y1);
        q1 = _mm512_NFMA_ps(z1, h3, q1);
        q1 = _mm512_NFMA_ps(w1, h4, q1);

	_mm512_store_ps(&q[ldq*2],q1);

	h2 = _mm512_set1_ps(hh[ldh+1]);
	h3 = _mm512_set1_ps(hh[(ldh*2)+2]);
	h4 = _mm512_set1_ps(hh[(ldh*3)+3]);
	q1 = _mm512_load_ps(&q[ldq*3]);

	q1 = _mm512_sub_ps(q1, x1);
        q1 = _mm512_NFMA_ps(y1, h2, q1);
        q1 = _mm512_NFMA_ps(z1, h3, q1);
        q1 = _mm512_NFMA_ps(w1, h4, q1);

	_mm512_store_ps(&q[ldq*3], q1);

	for (i = 4; i < nb; i++)
	{
		h1 = _mm512_set1_ps(hh[i-3]);
		h2 = _mm512_set1_ps(hh[ldh+i-2]);
		h3 = _mm512_set1_ps(hh[(ldh*2)+i-1]);
		h4 = _mm512_set1_ps(hh[(ldh*3)+i]);

		q1 = _mm512_load_ps(&q[i*ldq]);
                q1 = _mm512_NFMA_ps(x1, h1, q1);
                q1 = _mm512_NFMA_ps(y1, h2, q1);
                q1 = _mm512_NFMA_ps(z1, h3, q1);
                q1 = _mm512_NFMA_ps(w1, h4, q1);
		_mm512_store_ps(&q[i*ldq],q1);
	}

	h1 = _mm512_set1_ps(hh[nb-3]);
	h2 = _mm512_set1_ps(hh[ldh+nb-2]);
	h3 = _mm512_set1_ps(hh[(ldh*2)+nb-1]);
	q1 = _mm512_load_ps(&q[nb*ldq]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
        q1 = _mm512_NFMA_ps(y1, h2, q1);
        q1 = _mm512_NFMA_ps(z1, h3, q1);

	_mm512_store_ps(&q[nb*ldq],q1);

	h1 = _mm512_set1_ps(hh[nb-2]);
	h2 = _mm512_set1_ps(hh[ldh+nb-1]);
	q1 = _mm512_load_ps(&q[(nb+1)*ldq]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);
        q1 = _mm512_NFMA_ps(y1, h2, q1);

	_mm512_store_ps(&q[(nb+1)*ldq],q1);

	h1 = _mm512_set1_ps(hh[nb-1]);
	q1 = _mm512_load_ps(&q[(nb+2)*ldq]);

	q1 = _mm512_NFMA_ps(x1, h1, q1);

	_mm512_store_ps(&q[(nb+2)*ldq],q1);
}


