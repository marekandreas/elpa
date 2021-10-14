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
// Author: Andreas Marek, MPCDF

#include "config-f90.h"

#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../../general/precision_macros.h"
#ifdef HAVE_VSX_SSE
#include <altivec.h>
#endif
#include <stdio.h>
#include <stdlib.h>


#define __forceinline __attribute__((always_inline)) static
#ifdef DOUBLE_PRECISION_REAL
#define __SSE_DATATYPE __vector double
#define _SSE_LOAD vec_ld
#define _SSE_ADD vec_add
#define _SSE_MUL vec_mul
#define _SSE_NEG vec_neg
#define _SSE_STORE vec_st
#define offset 2
#endif

#ifdef SINGLE_PRECISION_REAL
#define __SSE_DATATYPE __vector float
#define _SSE_LOAD vec_ld
#define _SSE_ADD vec_add
#define _SSE_MUL vec_mul
#define _SSE_NEG vec_neg
#define _SSE_STORE vec_st
#define offset 4
#endif


#ifdef HAVE_SSE_INTRINSICS
#undef __AVX__
#endif

//Forward declaration
#ifdef DOUBLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_2_VSX_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_4_VSX_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_6_VSX_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_8_VSX_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_10_VSX_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_12_VSX_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);
#endif
#ifdef SINGLE_PRECISION_REAL
__forceinline void hh_trafo_kernel_4_VSX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_8_VSX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_12_VSX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_16_VSX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_20_VSX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_24_VSX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
#endif


#ifdef DOUBLE_PRECISION_REAL
void double_hh_trafo_real_VSX_2hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif
#ifdef SINGLE_PRECISION_REAL
void double_hh_trafo_real_VSX_2hv_single_(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif

#ifdef DOUBLE_PRECISION_REAL
void double_hh_trafo_real_vsx_2hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
#ifdef SINGLE_PRECISION_REAL
void double_hh_trafo_real_vsx_2hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;
	int worked_on;

	worked_on = 0;

	// calculating scalar product to compute
	// 2 householder vectors simultaneously
#ifdef DOUBLE_PRECISION_REAL
	double s = hh[(ldh)+1]*1.0;
#endif
#ifdef SINGLE_PRECISION_REAL
	float s = hh[(ldh)+1]*1.0;
#endif

#ifdef HAVE_SSE_INTRINSICS
	#pragma ivdep
#endif
	for (i = 2; i < nb; i++)
	{
		s += hh[i-1] * hh[(i+ldh)];
	}

	// Production level kernel calls with padding
#ifdef DOUBLE_PRECISION_REAL
	for (i = 0; i < nq-10; i+=12)
	{
		hh_trafo_kernel_12_VSX_2hv_double(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 12;
	}
#endif
#ifdef SINGLE_PRECISION_REAL
	for (i = 0; i < nq-20; i+=24)
	{
		hh_trafo_kernel_24_VSX_2hv_single(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 24;
	}
#endif

	if (nq == i)
	{
		return;
	}

#ifdef DOUBLE_PRECISION_REAL
	if (nq-i == 10)
	{
		hh_trafo_kernel_10_VSX_2hv_double(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 10;
	}
#endif

#ifdef SINGLE_PRECISION_REAL
	if (nq-i == 20)
	{
		hh_trafo_kernel_20_VSX_2hv_single(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 20;
	}
#endif

#ifdef DOUBLE_PRECISION_REAL
	if (nq-i == 8)
	{
		hh_trafo_kernel_8_VSX_2hv_double(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 8;
	}
#endif

#ifdef SINGLE_PRECISION_REAL
	if (nq-i == 16)
	{
		hh_trafo_kernel_16_VSX_2hv_single(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 16;
	}
#endif


#ifdef DOUBLE_PRECISION_REAL
	if (nq-i == 6)
	{
		hh_trafo_kernel_6_VSX_2hv_double(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 6;
	}
#endif

#ifdef SINGLE_PRECISION_REAL
	if (nq-i == 12)
	{
		hh_trafo_kernel_12_VSX_2hv_single(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 12;
	}
#endif

#ifdef DOUBLE_PRECISION_REAL
	if (nq-i == 4)
	{
		hh_trafo_kernel_4_VSX_2hv_double(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 4;
	}
#endif

#ifdef SINGLE_PRECISION_REAL
	if (nq-i == 8)
	{
		hh_trafo_kernel_8_VSX_2hv_single(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 8;
	}
#endif

#ifdef DOUBLE_PRECISION_REAL
	if (nq-i == 2)
	{
		hh_trafo_kernel_2_VSX_2hv_double(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 2;
	}
#endif

#ifdef SINGLE_PRECISION_REAL
	if (nq-i == 4)
	{
		hh_trafo_kernel_4_VSX_2hv_single(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 4;
	}
#endif
#ifdef WITH_DEBUG
	if (worked_on != nq)
	{
		printf("Error in real VSX BLOCK2 kernel %d %d\n", worked_on, nq);
		abort();
	}
#endif
}

/**
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 12 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 24 rows of Q simultaneously, a
#endif
 * matrix Vector product with two householder
 * vectors + a rank 2 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_12_VSX_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
#endif
#ifdef SINGLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_24_VSX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [12 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
#ifdef HAVE_SSE_INTRINSICS
	// Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SSE_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif

	__SSE_DATATYPE x1 = _SSE_LOAD(0 ,&q[ldq]);
	__SSE_DATATYPE x2 = _SSE_LOAD(0, &q[ldq+offset]);
	__SSE_DATATYPE x3 = _SSE_LOAD(0, &q[ldq+2*offset]);
	__SSE_DATATYPE x4 = _SSE_LOAD(0, &q[ldq+3*offset]);
	__SSE_DATATYPE x5 = _SSE_LOAD(0, &q[ldq+4*offset]);
	__SSE_DATATYPE x6 = _SSE_LOAD(0, &q[ldq+5*offset]);

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h1 = _mm_set1_pd(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h1 = _mm_set1_ps(hh[ldh+1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
        __SSE_DATATYPE h1 = vec_splats(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SSE_DATATYPE h1 = vec_splats(hh[ldh+1]);
#endif
#endif

	__SSE_DATATYPE h2;

	__SSE_DATATYPE q1 = _SSE_LOAD(0, &q[0]);
	__SSE_DATATYPE y1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	__SSE_DATATYPE q2 = _SSE_LOAD(0, &q[offset]);
	__SSE_DATATYPE y2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
	__SSE_DATATYPE q3 = _SSE_LOAD(0, &q[2*offset]);
	__SSE_DATATYPE y3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
	__SSE_DATATYPE q4 = _SSE_LOAD(0, &q[3*offset]);
	__SSE_DATATYPE y4 = _SSE_ADD(q4, _SSE_MUL(x4, h1));
	__SSE_DATATYPE q5 = _SSE_LOAD(0, &q[4*offset]);
	__SSE_DATATYPE y5 = _SSE_ADD(q5, _SSE_MUL(x5, h1));
	__SSE_DATATYPE q6 = _SSE_LOAD(0, &q[5*offset]);
	__SSE_DATATYPE y6 = _SSE_ADD(q6, _SSE_MUL(x6, h1));
	for(i = 2; i < nb; i++)
	{
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_set1_pd(hh[i-1]);
		h2 = _mm_set1_pd(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = _mm_set1_ps(hh[i-1]);
		h2 = _mm_set1_ps(hh[ldh+i]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#endif

		q1 = _SSE_LOAD(0, &q[i*ldq]);
		x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
		y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
		q2 = _SSE_LOAD(0, &q[(i*ldq)+offset]);
		x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
		y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
		q3 = _SSE_LOAD(0, &q[(i*ldq)+2*offset]);
		x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
		y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
		q4 = _SSE_LOAD(0, &q[(i*ldq)+3*offset]);
		x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
		y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
		q5 = _SSE_LOAD(0, &q[(i*ldq)+4*offset]);
		x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
		y5 = _SSE_ADD(y5, _SSE_MUL(q5,h2));
		q6 = _SSE_LOAD(0, &q[(i*ldq)+5*offset]);
		x6 = _SSE_ADD(x6, _SSE_MUL(q6,h1));
		y6 = _SSE_ADD(y6, _SSE_MUL(q6,h2));
	}
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_set1_pd(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = _mm_set1_ps(hh[nb-1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_spalts(hh[nb-1]);
#endif
#endif

	q1 = _SSE_LOAD(0,&q[nb*ldq]);
	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
	q2 = _SSE_LOAD(0, &q[(nb*ldq)+offset]);
	x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
	q3 = _SSE_LOAD(0, &q[(nb*ldq)+2*offset]);
	x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
	q4 = _SSE_LOAD(0, &q[(nb*ldq)+3*offset]);
	x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
	q5 = _SSE_LOAD(0, &q[(nb*ldq)+4*offset]);
	x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
	q6 = _SSE_LOAD(0, &q[(nb*ldq)+5*offset]);
	x6 = _SSE_ADD(x6, _SSE_MUL(q6,h1));
	/////////////////////////////////////////////////////
	// Rank-2 update of Q [12 x nb+1]
	/////////////////////////////////////////////////////
#ifdef  HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = _mm_set1_pd(hh[0]);
	__SSE_DATATYPE tau2 = _mm_set1_pd(hh[ldh]);
	__SSE_DATATYPE vs = _mm_set1_pd(s);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = _mm_set1_ps(hh[0]);
	__SSE_DATATYPE tau2 = _mm_set1_ps(hh[ldh]);
	__SSE_DATATYPE vs = _mm_set1_ps(s);
#endif
#endif
#ifdef  HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = vec_splats(hh[0]);
	__SSE_DATATYPE tau2 = vec_splats(hh[ldh]);
	__SSE_DATATYPE vs = vec_splats(s);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = vec_splats(hh[0]);
	__SSE_DATATYPE tau2 = vec_splats(hh[ldh]);
	__SSE_DATATYPE vs = vec_splatss(s);
#endif
#endif

#ifdef HAVE_SSE_INTRINSICS
	h1 = _SSE_XOR(tau1, sign);
#endif
#ifdef HAVE_VSX_SSE
	h1 = vec_neg(tau1);
#endif
	x1 = _SSE_MUL(x1, h1);
	x2 = _SSE_MUL(x2, h1);
	x3 = _SSE_MUL(x3, h1);
	x4 = _SSE_MUL(x4, h1);
	x5 = _SSE_MUL(x5, h1);
	x6 = _SSE_MUL(x6, h1);
#ifdef HAVE_SSE_INTRINSICS
	h1 = _SSE_XOR(tau2, sign);
#endif
#ifdef HAVE_SPARC64_SSE
	h1 = vec_neg(tau2);
#endif
	h2 = _SSE_MUL(h1, vs);

	y1 = _SSE_ADD(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
	y2 = _SSE_ADD(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
	y3 = _SSE_ADD(_SSE_MUL(y3,h1), _SSE_MUL(x3,h2));
	y4 = _SSE_ADD(_SSE_MUL(y4,h1), _SSE_MUL(x4,h2));
	y5 = _SSE_ADD(_SSE_MUL(y5,h1), _SSE_MUL(x5,h2));
	y6 = _SSE_ADD(_SSE_MUL(y6,h1), _SSE_MUL(x6,h2));
	q1 = _SSE_LOAD(0, &q[0]);
	q1 = _SSE_ADD(q1, y1);
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[0]);
	q2 = _SSE_LOAD(0,&q[offset]);
	q2 = _SSE_ADD(q2, y2);
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[offset]);
	q3 = _SSE_LOAD(0, &q[2*offset]);
	q3 = _SSE_ADD(q3, y3);
	_SSE_STORE((__vector unsigned int) q3, 0, (unsigned int *) &q[2*offset]);
	q4 = _SSE_LOAD(0, &q[3*offset]);
	q4 = _SSE_ADD(q4, y4);
	_SSE_STORE((__vector unsigned int) q4, 0, (unsigned int *) &q[3*offset]);
	q5 = _SSE_LOAD(0, &q[4*offset]);
	q5 = _SSE_ADD(q5, y5);
	_SSE_STORE((__vector unsigned int) q5, 0, (unsigned int *) &q[4*offset]);
	q6 = _SSE_LOAD(0, &q[5*offset]);
	q6 = _SSE_ADD(q6, y6);
	_SSE_STORE((__vector unsigned int) q6, 0, (unsigned int *) &q[5*offset]);

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h2 = _mm_set1_pd(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = _mm_set1_ps(hh[ldh+1]);
#endif
#endif

#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+1]);
#endif
#endif

	q1 = _SSE_LOAD(0, &q[ldq]);
	q1 = _SSE_ADD(q1, _SSE_ADD(x1, _SSE_MUL(y1, h2)));
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[ldq]);
	q2 = _SSE_LOAD(0, &q[ldq+offset]);
	q2 = _SSE_ADD(q2, _SSE_ADD(x2, _SSE_MUL(y2, h2)));
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[ldq+offset]);
	q3 = _SSE_LOAD(0, &q[ldq+2*offset]);
	q3 = _SSE_ADD(q3, _SSE_ADD(x3, _SSE_MUL(y3, h2)));
	_SSE_STORE((__vector unsigned int) q3, 0, (unsigned int *) &q[ldq+2*offset]);
	q4 = _SSE_LOAD(0, &q[ldq+3*offset]);
	q4 = _SSE_ADD(q4, _SSE_ADD(x4, _SSE_MUL(y4, h2)));
	_SSE_STORE((__vector unsigned int) q4, 0, (unsigned int *) &q[ldq+3*offset]);
	q5 = _SSE_LOAD(0, &q[ldq+4*offset]);
	q5 = _SSE_ADD(q5, _SSE_ADD(x5, _SSE_MUL(y5, h2)));
	_SSE_STORE((__vector unsigned int) q5, 0, (unsigned int *) &q[ldq+4*offset]);
	q6 = _SSE_LOAD(0, &q[ldq+5*offset]);
	q6 = _SSE_ADD(q6, _SSE_ADD(x6, _SSE_MUL(y6, h2)));
	_SSE_STORE((__vector unsigned int) q6, 0, (unsigned int *) &q[ldq+5*offset]);

	for (i = 2; i < nb; i++)
	{
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_set1_pd(hh[i-1]);
		h2 = _mm_set1_pd(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = _mm_set1_ps(hh[i-1]);
		h2 = _mm_set1_ps(hh[ldh+i]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#endif

		q1 = _SSE_LOAD(0, &q[i*ldq]);
		q1 = _SSE_ADD(q1, _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2)));
		_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[i*ldq]);
		q2 = _SSE_LOAD(0, &q[(i*ldq)+offset]);
		q2 = _SSE_ADD(q2, _SSE_ADD(_SSE_MUL(x2,h1), _SSE_MUL(y2, h2)));
		_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[(i*ldq)+offset]);
		q3 = _SSE_LOAD(0, &q[(i*ldq)+2*offset]);
		q3 = _SSE_ADD(q3, _SSE_ADD(_SSE_MUL(x3,h1), _SSE_MUL(y3, h2)));
		_SSE_STORE((__vector unsigned int) q3, 0, (unsigned int *) &q[(i*ldq)+2*offset]);
		q4 = _SSE_LOAD(0, &q[(i*ldq)+3*offset]);
		q4 = _SSE_ADD(q4, _SSE_ADD(_SSE_MUL(x4,h1), _SSE_MUL(y4, h2)));
		_SSE_STORE((__vector unsigned int) q4, 0, (unsigned int *) &q[(i*ldq)+3*offset]);
		q5 = _SSE_LOAD(0, &q[(i*ldq)+4*offset]);
		q5 = _SSE_ADD(q5, _SSE_ADD(_SSE_MUL(x5,h1), _SSE_MUL(y5, h2)));
		_SSE_STORE((__vector unsigned int) q5, 0, (unsigned int *) &q[(i*ldq)+4*offset]);
		q6 = _SSE_LOAD(0, &q[(i*ldq)+5*offset]);
		q6 = _SSE_ADD(q6, _SSE_ADD(_SSE_MUL(x6,h1), _SSE_MUL(y6, h2)));
		_SSE_STORE((__vector unsigned int) q6, 0, (unsigned int *) &q[(i*ldq)+5*offset]);
	}
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_set1_pd(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = _mm_set1_ps(hh[nb-1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#endif


	q1 = _SSE_LOAD(0, &q[nb*ldq]);
	q1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[nb*ldq]);
	q2 = _SSE_LOAD(0, &q[(nb*ldq)+offset]);
	q2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[(nb*ldq)+offset]);
	q3 = _SSE_LOAD(0, &q[(nb*ldq)+2*offset]);
	q3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
	_SSE_STORE((__vector unsigned int) q3, 0, (unsigned int *) &q[(nb*ldq)+2*offset]);
	q4 = _SSE_LOAD(0, &q[(nb*ldq)+3*offset]);
	q4 = _SSE_ADD(q4, _SSE_MUL(x4, h1));
	_SSE_STORE((__vector unsigned int) q4, 0, (unsigned int *) &q[(nb*ldq)+3*offset]);
	q5 = _SSE_LOAD(0, &q[(nb*ldq)+4*offset]);
	q5 = _SSE_ADD(q5, _SSE_MUL(x5, h1));
	_SSE_STORE((__vector unsigned int) q5, 0, (unsigned int *) &q[(nb*ldq)+4*offset]);
	q6 = _SSE_LOAD(0, &q[(nb*ldq)+5*offset]);
	q6 = _SSE_ADD(q6, _SSE_MUL(x6, h1));
	_SSE_STORE((__vector unsigned int) q6, 0, (unsigned int *) &q[(nb*ldq)+5*offset]);
}



/**
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 10 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 20 rows of Q simultaneously, a
#endif
 * matrix Vector product with two householder
 * vectors + a rank 2 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_10_VSX_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
#endif
#ifdef SINGLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_20_VSX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [12 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
#ifdef HAVE_SSE_INTRINSICS
	// Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SSE_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif

	__SSE_DATATYPE x1 = _SSE_LOAD(0, &q[ldq]);
	__SSE_DATATYPE x2 = _SSE_LOAD(0, &q[ldq+offset]);
	__SSE_DATATYPE x3 = _SSE_LOAD(0, &q[ldq+2*offset]);
	__SSE_DATATYPE x4 = _SSE_LOAD(0, &q[ldq+3*offset]);
	__SSE_DATATYPE x5 = _SSE_LOAD(0, &q[ldq+4*offset]);

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h1 = _mm_set1_pd(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h1 = _mm_set1_ps(hh[ldh+1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h1 = vec_splats(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h1 = vec_splats(hh[ldh+1]);
#endif
#endif

	__SSE_DATATYPE h2;

	__SSE_DATATYPE q1 = _SSE_LOAD(0, &q[0]);
	__SSE_DATATYPE y1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	__SSE_DATATYPE q2 = _SSE_LOAD(0, &q[offset]);
	__SSE_DATATYPE y2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
	__SSE_DATATYPE q3 = _SSE_LOAD(0, &q[2*offset]);
	__SSE_DATATYPE y3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
	__SSE_DATATYPE q4 = _SSE_LOAD(0, &q[3*offset]);
	__SSE_DATATYPE y4 = _SSE_ADD(q4, _SSE_MUL(x4, h1));
	__SSE_DATATYPE q5 = _SSE_LOAD(0, &q[4*offset]);
	__SSE_DATATYPE y5 = _SSE_ADD(q5, _SSE_MUL(x5, h1));
	for(i = 2; i < nb; i++)
	{
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_set1_pd(hh[i-1]);
		h2 = _mm_set1_pd(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = _mm_set1_ps(hh[i-1]);
		h2 = _mm_set1_ps(hh[ldh+i]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#endif


		q1 = _SSE_LOAD(0, &q[i*ldq]);
		x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
		y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
		q2 = _SSE_LOAD(0, &q[(i*ldq)+offset]);
		x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
		y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
		q3 = _SSE_LOAD(0, &q[(i*ldq)+2*offset]);
		x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
		y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
		q4 = _SSE_LOAD(0, &q[(i*ldq)+3*offset]);
		x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
		y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
		q5 = _SSE_LOAD(0, &q[(i*ldq)+4*offset]);
		x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
		y5 = _SSE_ADD(y5, _SSE_MUL(q5,h2));
	}

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_set1_pd(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = _mm_set1_ps(hh[nb-1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vex_splats(hh[nb-1]);
#endif
#endif

	q1 = _SSE_LOAD(0, &q[nb*ldq]);
	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
	q2 = _SSE_LOAD(0, &q[(nb*ldq)+offset]);
	x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
	q3 = _SSE_LOAD(0, &q[(nb*ldq)+2*offset]);
	x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
	q4 = _SSE_LOAD(0, &q[(nb*ldq)+3*offset]);
	x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
	q5 = _SSE_LOAD(0, &q[(nb*ldq)+4*offset]);
	x5 = _SSE_ADD(x5, _SSE_MUL(q5,h1));
	/////////////////////////////////////////////////////
	// Rank-2 update of Q [12 x nb+1]
	/////////////////////////////////////////////////////
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = _mm_set1_pd(hh[0]);
	__SSE_DATATYPE tau2 = _mm_set1_pd(hh[ldh]);
	__SSE_DATATYPE vs = _mm_set1_pd(s);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = _mm_set1_ps(hh[0]);
	__SSE_DATATYPE tau2 = _mm_set1_ps(hh[ldh]);
	__SSE_DATATYPE vs = _mm_set1_ps(s);

#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = vec_splats(hh[0]);
	__SSE_DATATYPE tau2 = vec_splats(hh[ldh]);
	__SSE_DATATYPE vs = vec_splats(s);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = vec_splats(hh[0], hh[0]);
	__SSE_DATATYPE tau2 = vec_splats(hh[ldh]);
	__SSE_DATATYPE vs = vec_splats(s);

#endif
#endif

#ifdef HAVE_SSE_INTRINSICS
	h1 = _SSE_XOR(tau1, sign);
#endif
#ifdef HAVE_VSX_SSE
	h1 = vec_neg(tau1);
#endif
	x1 = _SSE_MUL(x1, h1);
	x2 = _SSE_MUL(x2, h1);
	x3 = _SSE_MUL(x3, h1);
	x4 = _SSE_MUL(x4, h1);
	x5 = _SSE_MUL(x5, h1);
#ifdef HAVE_SSE_INTRINSICS
	h1 = _SSE_XOR(tau2, sign);
#endif
#ifdef HAVE_VSX_SSE
	h1 = vec_neg(tau2);
#endif
	h2 = _SSE_MUL(h1, vs);

	y1 = _SSE_ADD(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
	y2 = _SSE_ADD(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
	y3 = _SSE_ADD(_SSE_MUL(y3,h1), _SSE_MUL(x3,h2));
	y4 = _SSE_ADD(_SSE_MUL(y4,h1), _SSE_MUL(x4,h2));
	y5 = _SSE_ADD(_SSE_MUL(y5,h1), _SSE_MUL(x5,h2));
	q1 = _SSE_LOAD(0, &q[0]);
	q1 = _SSE_ADD(q1, y1);
	_SSE_STORE((__vector unsigned int) q1, (unsigned int *) 0, &q[0]);
	q2 = _SSE_LOAD(0, &q[offset]);
	q2 = _SSE_ADD(q2, y2);
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[offset]);
	q3 = _SSE_LOAD(0,&q[2*offset]);
	q3 = _SSE_ADD(q3, y3);
	_SSE_STORE((__vector unsigned int) q3,0, (unsigned int *) &q[2*offset]);
	q4 = _SSE_LOAD(0, &q[3*offset]);
	q4 = _SSE_ADD(q4, y4);
	_SSE_STORE((__vector unsigned int) q4, 0, (unsigned int *) &q[3*offset]);
	q5 = _SSE_LOAD(0, &q[4*offset]);
	q5 = _SSE_ADD(q5, y5);
	_SSE_STORE((__vector unsigned int) q5, 0, (unsigned int *) &q[4*offset]);

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h2 = _mm_set1_pd(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = _mm_set1_ps(hh[ldh+1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+1]);
#endif
#endif

	q1 = _SSE_LOAD(0, &q[ldq]);
	q1 = _SSE_ADD(q1, _SSE_ADD(x1, _SSE_MUL(y1, h2)));
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[ldq]);
	q2 = _SSE_LOAD(0, &q[ldq+offset]);
	q2 = _SSE_ADD(q2, _SSE_ADD(x2, _SSE_MUL(y2, h2)));
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[ldq+offset]);
	q3 = _SSE_LOAD(0, &q[ldq+2*offset]);
	q3 = _SSE_ADD(q3, _SSE_ADD(x3, _SSE_MUL(y3, h2)));
	_SSE_STORE((__vector unsigned int) q3, 0, (unsigned int *) &q[ldq+2*offset]);
	q4 = _SSE_LOAD(0, &q[ldq+3*offset]);
	q4 = _SSE_ADD(q4, _SSE_ADD(x4, _SSE_MUL(y4, h2)));
	_SSE_STORE((__vector unsigned int) q4, 0, (unsigned int *) &q[ldq+3*offset]);
	q5 = _SSE_LOAD(0, &q[ldq+4*offset]);
	q5 = _SSE_ADD(q5, _SSE_ADD(x5, _SSE_MUL(y5, h2)));
	_SSE_STORE((__vector unsigned int) q5, 0, (unsigned int *) &q[ldq+4*offset]);

	for (i = 2; i < nb; i++)
	{
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_set1_pd(hh[i-1]);
		h2 = _mm_set1_pd(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = _mm_set1_ps(hh[i-1]);
		h2 = _mm_set1_ps(hh[ldh+i]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#endif

		q1 = _SSE_LOAD(0, &q[i*ldq]);
		q1 = _SSE_ADD(q1, _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2)));
		_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[i*ldq]);
		q2 = _SSE_LOAD(0, &q[(i*ldq)+offset]);
		q2 = _SSE_ADD(q2, _SSE_ADD(_SSE_MUL(x2,h1), _SSE_MUL(y2, h2)));
		_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[(i*ldq)+offset]);
		q3 = _SSE_LOAD(0, &q[(i*ldq)+2*offset]);
		q3 = _SSE_ADD(q3, _SSE_ADD(_SSE_MUL(x3,h1), _SSE_MUL(y3, h2)));
		_SSE_STORE((__vector unsigned int) q3, 0, (unsigned int *) &q[(i*ldq)+2*offset]);
		q4 = _SSE_LOAD(0, &q[(i*ldq)+3*offset]);
		q4 = _SSE_ADD(q4, _SSE_ADD(_SSE_MUL(x4,h1), _SSE_MUL(y4, h2)));
		_SSE_STORE((__vector unsigned int) q4, 0, (unsigned int *) &q[(i*ldq)+3*offset]);
		q5 = _SSE_LOAD(0, &q[(i*ldq)+4*offset]);
		q5 = _SSE_ADD(q5, _SSE_ADD(_SSE_MUL(x5,h1), _SSE_MUL(y5, h2)));
		_SSE_STORE((__vector unsigned int) q5, 0, (unsigned int *) &q[(i*ldq)+4*offset]);
	}
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_set1_pd(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = _mm_set1_ps(hh[nb-1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#endif

	q1 = _SSE_LOAD(0, &q[nb*ldq]);
	q1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[nb*ldq]);
	q2 = _SSE_LOAD(0, &q[(nb*ldq)+offset]);
	q2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[(nb*ldq)+offset]);
	q3 = _SSE_LOAD(0, &q[(nb*ldq)+2*offset]);
	q3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
	_SSE_STORE((__vector unsigned int) q3, 0, (unsigned int *) &q[(nb*ldq)+2*offset]);
	q4 = _SSE_LOAD(0, &q[(nb*ldq)+3*offset]);
	q4 = _SSE_ADD(q4, _SSE_MUL(x4, h1));
	_SSE_STORE((__vector unsigned int) q4, 0, (unsigned int *) &q[(nb*ldq)+3*offset]);
	q5 = _SSE_LOAD(0, &q[(nb*ldq)+4*offset]);
	q5 = _SSE_ADD(q5, _SSE_MUL(x5, h1));
	_SSE_STORE((__vector unsigned int) q5, 0, (unsigned int *) &q[(nb*ldq)+4*offset]);
}

/**
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 8 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 16 rows of Q simultaneously, a
#endif
 * matrix Vector product with two householder
 * vectors + a rank 2 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_8_VSX_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
#endif
#ifdef SINGLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_16_VSX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [12 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;

#ifdef HAVE_SSE_INTRINSICS
	// Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SSE_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif
	__SSE_DATATYPE x1 = _SSE_LOAD(0, &q[ldq]);
	__SSE_DATATYPE x2 = _SSE_LOAD(0, &q[ldq+offset]);
	__SSE_DATATYPE x3 = _SSE_LOAD(0, &q[ldq+2*offset]);
	__SSE_DATATYPE x4 = _SSE_LOAD(0, &q[ldq+3*offset]);

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h1 = _mm_set1_pd(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h1 = _mm_set1_ps(hh[ldh+1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h1 = vec_splats(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h1 = vec_splats(hh[ldh+1]);
#endif
#endif

	__SSE_DATATYPE h2;

	__SSE_DATATYPE q1 = _SSE_LOAD(0, &q[0]);
	__SSE_DATATYPE y1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	__SSE_DATATYPE q2 = _SSE_LOAD(0, &q[offset]);
	__SSE_DATATYPE y2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
	__SSE_DATATYPE q3 = _SSE_LOAD(0, &q[2*offset]);
	__SSE_DATATYPE y3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
	__SSE_DATATYPE q4 = _SSE_LOAD(0, &q[3*offset]);
	__SSE_DATATYPE y4 = _SSE_ADD(q4, _SSE_MUL(x4, h1));
	for(i = 2; i < nb; i++)
	{
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_set1_pd(hh[i-1]);
		h2 = _mm_set1_pd(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = _mm_set1_ps(hh[i-1]);
		h2 = _mm_set1_ps(hh[ldh+i]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#endif


		q1 = _SSE_LOAD(0, &q[i*ldq]);
		x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
		y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
		q2 = _SSE_LOAD(0, &q[(i*ldq)+offset]);
		x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
		y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
		q3 = _SSE_LOAD(0, &q[(i*ldq)+2*offset]);
		x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
		y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
		q4 = _SSE_LOAD(0, &q[(i*ldq)+3*offset]);
		x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
		y4 = _SSE_ADD(y4, _SSE_MUL(q4,h2));
	}
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_set1_pd(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = _mm_set1_ps(hh[nb-1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#endif
	q1 = _SSE_LOAD(0, &q[nb*ldq]);
	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
	q2 = _SSE_LOAD(0, &q[(nb*ldq)+offset]);
	x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
	q3 = _SSE_LOAD(0, &q[(nb*ldq)+2*offset]);
	x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
	q4 = _SSE_LOAD(0, &q[(nb*ldq)+3*offset]);
	x4 = _SSE_ADD(x4, _SSE_MUL(q4,h1));
	/////////////////////////////////////////////////////
	// Rank-2 update of Q [12 x nb+1]
	/////////////////////////////////////////////////////
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = _mm_set1_pd(hh[0]);
	__SSE_DATATYPE tau2 = _mm_set1_pd(hh[ldh]);
	__SSE_DATATYPE vs = _mm_set1_pd(s);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = _mm_set1_ps(hh[0]);
	__SSE_DATATYPE tau2 = _mm_set1_ps(hh[ldh]);
	__SSE_DATATYPE vs = _mm_set1_ps(s);

#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = vec_splats(hh[0]);
	__SSE_DATATYPE tau2 = vec_splats(hh[ldh]);
	__SSE_DATATYPE vs = vec_splats(s);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = vec_splats(hh[0]);
	__SSE_DATATYPE tau2 = vec_splats(hh[ldh]);
	__SSE_DATATYPE vs = vec_splats(s);

#endif
#endif

#ifdef HAVE_SSE_INTRINSICS
	h1 = _SSE_XOR(tau1, sign);
#endif
#ifdef HAVE_VSX_SSE
	h1 = vec_neg(tau1);
#endif
	x1 = _SSE_MUL(x1, h1);
	x2 = _SSE_MUL(x2, h1);
	x3 = _SSE_MUL(x3, h1);
	x4 = _SSE_MUL(x4, h1);
#ifdef HAVE_SSE_INTRINSICS
	h1 = _SSE_XOR(tau2, sign);
#endif
#ifdef HAVE_VSX_SSE
	h1 = vec_neg(tau2);
#endif
	h2 = _SSE_MUL(h1, vs);

	y1 = _SSE_ADD(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
	y2 = _SSE_ADD(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
	y3 = _SSE_ADD(_SSE_MUL(y3,h1), _SSE_MUL(x3,h2));
	y4 = _SSE_ADD(_SSE_MUL(y4,h1), _SSE_MUL(x4,h2));
	q1 = _SSE_LOAD(0, &q[0]);
	q1 = _SSE_ADD(q1, y1);
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[0]);
	q2 = _SSE_LOAD(0, &q[offset]);
	q2 = _SSE_ADD(q2, y2);
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[offset]);
	q3 = _SSE_LOAD(0, &q[2*offset]);
	q3 = _SSE_ADD(q3, y3);
	_SSE_STORE((__vector unsigned int) q3, 0, (unsigned int *) &q[2*offset]);
	q4 = _SSE_LOAD(&q[3*offset]);
	q4 = _SSE_ADD(q4, y4);
	_SSE_STORE((__vector unsigned int) q4, 0, (unsigned int *) &q[3*offset]);

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h2 = _mm_set1_pd(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = _mm_set1_ps(hh[ldh+1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+1]);
#endif
#endif

	q1 = _SSE_LOAD(0, &q[ldq]);
	q1 = _SSE_ADD(q1, _SSE_ADD(x1, _SSE_MUL(y1, h2)));
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[ldq]);
	q2 = _SSE_LOAD(0, &q[ldq+offset]);
	q2 = _SSE_ADD(q2, _SSE_ADD(x2, _SSE_MUL(y2, h2)));
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[ldq+offset]);
	q3 = _SSE_LOAD(0, &q[ldq+2*offset]);
	q3 = _SSE_ADD(q3, _SSE_ADD(x3, _SSE_MUL(y3, h2)));
	_SSE_STORE((__vector unsigned int) q3, 0, (unsigned int *) &q[ldq+2*offset]);
	q4 = _SSE_LOAD(0, &q[ldq+3*offset]);
	q4 = _SSE_ADD(q4, _SSE_ADD(x4, _SSE_MUL(y4, h2)));
	_SSE_STORE((__vector unsigned int) q4, 0, (unsigned int *) &q[ldq+3*offset]);

	for (i = 2; i < nb; i++)
	{
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_set1_pd(hh[i-1]);
		h2 = _mm_set1_pd(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = _mm_set1_ps(hh[i-1]);
		h2 = _mm_set1_ps(hh[ldh+i]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#endif


		q1 = _SSE_LOAD(0, &q[i*ldq]);
		q1 = _SSE_ADD(q1, _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2)));
		_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[i*ldq]);
		q2 = _SSE_LOAD(0, &q[(i*ldq)+offset]);
		q2 = _SSE_ADD(q2, _SSE_ADD(_SSE_MUL(x2,h1), _SSE_MUL(y2, h2)));
		_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[(i*ldq)+offset]);
		q3 = _SSE_LOAD(0, &q[(i*ldq)+2*offset]);
		q3 = _SSE_ADD(q3, _SSE_ADD(_SSE_MUL(x3,h1), _SSE_MUL(y3, h2)));
		_SSE_STORE((__vector unsigned int) q3, 0, (unsigned int *) &q[(i*ldq)+2*offset]);
		q4 = _SSE_LOAD(0, &q[(i*ldq)+3*offset]);
		q4 = _SSE_ADD(q4, _SSE_ADD(_SSE_MUL(x4,h1), _SSE_MUL(y4, h2)));
		_SSE_STORE((__vector unsigned int) q4, 0, (unsigned int *) &q[(i*ldq)+3*offset]);
	}
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_set1_pd(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = _mm_set1_ps(hh[nb-1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#endif

	q1 = _SSE_LOAD(0, &q[nb*ldq]);
	q1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[nb*ldq]);
	q2 = _SSE_LOAD(0, &q[(nb*ldq)+offset]);
	q2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[(nb*ldq)+offset]);
	q3 = _SSE_LOAD(0, &q[(nb*ldq)+2*offset]);
	q3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
	_SSE_STORE((__vector unsigned int) q3, 0, (unsigned int *) &q[(nb*ldq)+2*offset]);
	q4 = _SSE_LOAD(0, &q[(nb*ldq)+3*offset]);
	q4 = _SSE_ADD(q4, _SSE_MUL(x4, h1));
	_SSE_STORE((__vector unsigned int) q4, 0, (unsigned int *) &q[(nb*ldq)+3*offset]);
}

/**
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 6 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 12 rows of Q simultaneously, a
#endif
 * matrix Vector product with two householder
 * vectors + a rank 2 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_6_VSX_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
#endif
#ifdef SINGLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_12_VSX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [12 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
#ifdef HAVE_SSE_INTRINSICS
	// Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SSE_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif

	__SSE_DATATYPE x1 = _SSE_LOAD(0, &q[ldq]);
	__SSE_DATATYPE x2 = _SSE_LOAD(0, &q[ldq+offset]);
	__SSE_DATATYPE x3 = _SSE_LOAD(0, &q[ldq+2*offset]);

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h1 = _mm_set1_pd(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h1 = _mm_set1_ps(hh[ldh+1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h1 = vec_splats(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h1 = vec_splats(hh[ldh+1]);
#endif
#endif
	__SSE_DATATYPE h2;

	__SSE_DATATYPE q1 = _SSE_LOAD(0, &q[0]);
	__SSE_DATATYPE y1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	__SSE_DATATYPE q2 = _SSE_LOAD(0, &q[offset]);
	__SSE_DATATYPE y2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
	__SSE_DATATYPE q3 = _SSE_LOAD(0, &q[2*offset]);
	__SSE_DATATYPE y3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
	for(i = 2; i < nb; i++)
	{
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_set1_pd(hh[i-1]);
		h2 = _mm_set1_pd(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = _mm_set1_ps(hh[i-1]);
		h2 = _mm_set1_ps(hh[ldh+i]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#endif


		q1 = _SSE_LOAD(0, &q[i*ldq]);
		x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
		y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
		q2 = _SSE_LOAD(0, &q[(i*ldq)+offset]);
		x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
		y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
		q3 = _SSE_LOAD(0, &q[(i*ldq)+2*offset]);
		x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
		y3 = _SSE_ADD(y3, _SSE_MUL(q3,h2));
	}
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_set1_pd(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = _mm_set1_ps(hh[nb-1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#endif

	q1 = _SSE_LOAD(0, &q[nb*ldq]);
	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
	q2 = _SSE_LOAD(0, &q[(nb*ldq)+offset]);
	x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
	q3 = _SSE_LOAD(0, &q[(nb*ldq)+2*offset]);
	x3 = _SSE_ADD(x3, _SSE_MUL(q3,h1));
	/////////////////////////////////////////////////////
	// Rank-2 update of Q [12 x nb+1]
	/////////////////////////////////////////////////////
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = _mm_set1_pd(hh[0]);
	__SSE_DATATYPE tau2 = _mm_set1_pd(hh[ldh]);
	__SSE_DATATYPE vs = _mm_set1_pd(s);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = _mm_set1_ps(hh[0]);
	__SSE_DATATYPE tau2 = _mm_set1_ps(hh[ldh]);
	__SSE_DATATYPE vs = _mm_set1_ps(s);

#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = vec_splats(hh[0]);
	__SSE_DATATYPE tau2 = vec_splats(hh[ldh]);
	__SSE_DATATYPE vs = vec_splats(s);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = vec_splats(hh[0]);
	__SSE_DATATYPE tau2 = vec_splats(hh[ldh]);
	__SSE_DATATYPE vs = vec_splats(s);

#endif
#endif


#ifdef HAVE_SSE_INTRINSICS
	h1 = _SSE_XOR(tau1, sign);
#endif
#ifdef HAVE_VSX_SSE
	h1 = vec_nec(tau1);
#endif
	x1 = _SSE_MUL(x1, h1);
	x2 = _SSE_MUL(x2, h1);
	x3 = _SSE_MUL(x3, h1);
#ifdef HAVE_SSE_INTRINSICS
	h1 = _SSE_XOR(tau2, sign);
#endif
#ifdef HAVE_VSX_SSE
	h1 = vec_neg(tau2);
#endif
	h2 = _SSE_MUL(h1, vs);

	y1 = _SSE_ADD(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
	y2 = _SSE_ADD(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
	y3 = _SSE_ADD(_SSE_MUL(y3,h1), _SSE_MUL(x3,h2));
	q1 = _SSE_LOAD(0, &q[0]);
	q1 = _SSE_ADD(q1, y1);
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[0]);
	q2 = _SSE_LOAD(0, &q[offset]);
	q2 = _SSE_ADD(q2, y2);
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[offset]);
	q3 = _SSE_LOAD(0, &q[2*offset]);
	q3 = _SSE_ADD(q3, y3);
	_SSE_STORE((__vector unsigned int) q3, 0, (unsigned int *) &q[2*offset]);

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h2 = _mm_set1_pd(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = _mm_set1_ps(hh[ldh+1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+1]);
#endif
#endif



	q1 = _SSE_LOAD(0, &q[ldq]);
	q1 = _SSE_ADD(q1, _SSE_ADD(x1, _SSE_MUL(y1, h2)));
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[ldq]);
	q2 = _SSE_LOAD(0, &q[ldq+offset]);
	q2 = _SSE_ADD(q2, _SSE_ADD(x2, _SSE_MUL(y2, h2)));
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[ldq+offset]);
	q3 = _SSE_LOAD(0, &q[ldq+2*offset]);
	q3 = _SSE_ADD(q3, _SSE_ADD(x3, _SSE_MUL(y3, h2)));
	_SSE_STORE((__vector unsigned int) q3, 0, (unsigned int *) &q[ldq+2*offset]);

	for (i = 2; i < nb; i++)
	{
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_set1_pd(hh[i-1]);
		h2 = _mm_set1_pd(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = _mm_set1_ps(hh[i-1]);
		h2 = _mm_set1_ps(hh[ldh+i]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#endif

		q1 = _SSE_LOAD(0, &q[i*ldq]);
		q1 = _SSE_ADD(q1, _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2)));
		_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[i*ldq]);
		q2 = _SSE_LOAD(0, &q[(i*ldq)+offset]);
		q2 = _SSE_ADD(q2, _SSE_ADD(_SSE_MUL(x2,h1), _SSE_MUL(y2, h2)));
		_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[(i*ldq)+offset]);
		q3 = _SSE_LOAD(0, &q[(i*ldq)+2*offset]);
		q3 = _SSE_ADD(q3, _SSE_ADD(_SSE_MUL(x3,h1), _SSE_MUL(y3, h2)));
		_SSE_STORE((__vector unsigned int) q3, 0, (unsigned int *) &q[(i*ldq)+2*offset]);
	}
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_set1_pd(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = _mm_set1_ps(hh[nb-1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#endif
	q1 = _SSE_LOAD(0, &q[nb*ldq]);
	q1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[nb*ldq]);
	q2 = _SSE_LOAD(0, &q[(nb*ldq)+offset]);
	q2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[(nb*ldq)+offset]);
	q3 = _SSE_LOAD(0, &q[(nb*ldq)+2*offset]);
	q3 = _SSE_ADD(q3, _SSE_MUL(x3, h1));
	_SSE_STORE((__vector unsigned int) q3, 0, (unsigned int *) &q[(nb*ldq)+2*offset]);
}


/**
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 4 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 8 rows of Q simultaneously, a
#endif
 * matrix Vector product with two householder
 * vectors + a rank 2 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_4_VSX_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
#endif
#ifdef SINGLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_8_VSX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [12 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
#ifdef HAVE_SSE_INTRINSICS
	// Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SSE_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif

	__SSE_DATATYPE x1 = _SSE_LOAD(0, &q[ldq]);
	__SSE_DATATYPE x2 = _SSE_LOAD(0, &q[ldq+offset]);

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h1 = _mm_set1_pd(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h1 = _mm_set1_ps(hh[ldh+1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h1 = vec_splats(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h1 = vec_splats(hh[ldh+1]);
#endif
#endif
	__SSE_DATATYPE h2;

	__SSE_DATATYPE q1 = _SSE_LOAD(0, &q[0]);
	__SSE_DATATYPE y1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	__SSE_DATATYPE q2 = _SSE_LOAD(0, &q[offset]);
	__SSE_DATATYPE y2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
	for(i = 2; i < nb; i++)
	{
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_set1_pd(hh[i-1]);
		h2 = _mm_set1_pd(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = _mm_set1_ps(hh[i-1]);
		h2 = _mm_set1_ps(hh[ldh+i]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#endif

		q1 = _SSE_LOAD(0, &q[i*ldq]);
		x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
		y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
		q2 = _SSE_LOAD(0, &q[(i*ldq)+offset]);
		x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
		y2 = _SSE_ADD(y2, _SSE_MUL(q2,h2));
	}

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_set1_pd(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = _mm_set1_ps(hh[nb-1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#endif

	q1 = _SSE_LOAD(0, &q[nb*ldq]);
	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
	q2 = _SSE_LOAD(0, &q[(nb*ldq)+offset]);
	x2 = _SSE_ADD(x2, _SSE_MUL(q2,h1));
	/////////////////////////////////////////////////////
	// Rank-2 update of Q [12 x nb+1]
	/////////////////////////////////////////////////////
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = _mm_set1_pd(hh[0]);
	__SSE_DATATYPE tau2 = _mm_set1_pd(hh[ldh]);
	__SSE_DATATYPE vs = _mm_set1_pd(s);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = _mm_set1_ps(hh[0]);
	__SSE_DATATYPE tau2 = _mm_set1_ps(hh[ldh]);
	__SSE_DATATYPE vs = _mm_set1_ps(s);

#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = vec_splats(hh[0]);
	__SSE_DATATYPE tau2 = vec_splats(hh[ldh]);
	__SSE_DATATYPE vs = vec_splats(s);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = vec_splats(hh[0]);
	__SSE_DATATYPE tau2 = vec_splats(hh[ldh]);
	__SSE_DATATYPE vs = vec_splats(s);

#endif
#endif

#ifdef HAVE_SSE_INTRINSICS
	h1 = _SSE_XOR(tau1, sign);
#endif
#ifdef HAVE_VSX_SSE
	h1 = vec_neg(tau1);
#endif
	x1 = _SSE_MUL(x1, h1);
	x2 = _SSE_MUL(x2, h1);
#ifdef HAVE_SSE_INTRINSICS
	h1 = _SSE_XOR(tau2, sign);
#endif
#ifdef HAVE_VSX_SSE
	h1 = vec_neg(tau2);
#endif
	h2 = _SSE_MUL(h1, vs);

	y1 = _SSE_ADD(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
	y2 = _SSE_ADD(_SSE_MUL(y2,h1), _SSE_MUL(x2,h2));
	q1 = _SSE_LOAD(0, &q[0]);
	q1 = _SSE_ADD(q1, y1);
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[0]);
	q2 = _SSE_LOAD(0, &q[offset]);
	q2 = _SSE_ADD(q2, y2);
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[offset]);

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h2 = _mm_set1_pd(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = _mm_set1_ps(hh[ldh+1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+1]);
#endif
#endif

	q1 = _SSE_LOAD(0, &q[ldq]);
	q1 = _SSE_ADD(q1, _SSE_ADD(x1, _SSE_MUL(y1, h2)));
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[ldq]);
	q2 = _SSE_LOAD(&q[ldq+offset]);
	q2 = _SSE_ADD(q2, _SSE_ADD(x2, _SSE_MUL(y2, h2)));
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[ldq+offset]);

	for (i = 2; i < nb; i++)
	{
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_set1_pd(hh[i-1]);
		h2 = _mm_set1_pd(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = _mm_set1_ps(hh[i-1]);
		h2 = _mm_set1_ps(hh[ldh+i]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#endif

		q1 = _SSE_LOAD(0, &q[i*ldq]);
		q1 = _SSE_ADD(q1, _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2)));
		_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[i*ldq]);
		q2 = _SSE_LOAD(0, &q[(i*ldq)+offset]);
		q2 = _SSE_ADD(q2, _SSE_ADD(_SSE_MUL(x2,h1), _SSE_MUL(y2, h2)));
		_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[(i*ldq)+offset]);
	}
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_set1_pd(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = _mm_set1_ps(hh[nb-1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#endif
	q1 = _SSE_LOAD(0, &q[nb*ldq]);
	q1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[nb*ldq]);
	q2 = _SSE_LOAD(0, &q[(nb*ldq)+offset]);
	q2 = _SSE_ADD(q2, _SSE_MUL(x2, h1));
	_SSE_STORE((__vector unsigned int) q2, 0, (unsigned int *) &q[(nb*ldq)+offset]);
}


/**
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 2 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 4 rows of Q simultaneously, a
#endif
 * matrix Vector product with two householder
 * vectors + a rank 2 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_2_VSX_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
#endif
#ifdef SINGLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_4_VSX_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [12 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
#ifdef HAVE_SSE_INTRINSICS
	// Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set1_epi64x(0x8000000000000000LL);
#endif
#ifdef SINGLE_PRECISION_REAL
        __SSE_DATATYPE sign = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
#endif
#endif

	__SSE_DATATYPE x1 = _SSE_LOAD(&q[ldq]);

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h1 = _mm_set1_pd(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h1 = _mm_set1_ps(hh[ldh+1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE h1 = vec_splats(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE h1 = vec_splats(hh[ldh+1]);
#endif
#endif
	__SSE_DATATYPE h2;

	__SSE_DATATYPE q1 = _SSE_LOAD(0, &q[0]);
	__SSE_DATATYPE y1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	for(i = 2; i < nb; i++)
	{
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_set1_pd(hh[i-1]);
		h2 = _mm_set1_pd(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = _mm_set1_ps(hh[i-1]);
		h2 = _mm_set1_ps(hh[ldh+i]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(h[ldh+i]);
#endif
#endif

		q1 = _SSE_LOAD(0, &q[i*ldq]);
		x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
		y1 = _SSE_ADD(y1, _SSE_MUL(q1,h2));
	}

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_set1_pd(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = _mm_set1_ps(hh[nb-1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#endif


	q1 = _SSE_LOAD(0, &q[nb*ldq]);
	x1 = _SSE_ADD(x1, _SSE_MUL(q1,h1));
	/////////////////////////////////////////////////////
	// Rank-2 update of Q [12 x nb+1]
	/////////////////////////////////////////////////////
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = _mm_set1_pd(hh[0]);
	__SSE_DATATYPE tau2 = _mm_set1_pd(hh[ldh]);
	__SSE_DATATYPE vs = _mm_set1_pd(s);
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = _mm_set1_ps(hh[0]);
	__SSE_DATATYPE tau2 = _mm_set1_ps(hh[ldh]);
	__SSE_DATATYPE vs = _mm_set1_ps(s);

#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = vec_splats(hh[0]);
	__SSE_DATATYPE tau2 = vec_splats(hh[ldh]);
	__SSE_DATATYPE vs = vec_splats(s );
#endif
#ifdef SINGLE_PRECISION_REAL
	__SSE_DATATYPE tau1 = vec_splats(hh[0]);
	__SSE_DATATYPE tau2 = vec_splats(hh[ldh]);
	__SSE_DATATYPE vs = vec_splats(s);

#endif
#endif

#ifdef HAVE_SSE_INTRINSICS
	h1 = _SSE_XOR(tau1, sign);
#endif
#ifdef HAVE_VSX_SSE
	h1 = vec_neg(tau1);
#endif
	x1 = _SSE_MUL(x1, h1);
#ifdef HAVE_SSE_INTRINSICS
	h1 = _SSE_XOR(tau2, sign);
#endif
#ifdef HAVE_VSX_SSE
	h1 = vec_neg(tau2);
#endif
	h2 = _SSE_MUL(h1, vs);

	y1 = _SSE_ADD(_SSE_MUL(y1,h1), _SSE_MUL(x1,h2));
	q1 = _SSE_LOAD(0, &q[0]);
	q1 = _SSE_ADD(q1, y1);
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[0]);

#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h2 = _mm_set1_pd(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = _mm_set1_ps(hh[ldh+1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h2 = vec_splats(hh[ldh+1]);
#endif
#endif

	q1 = _SSE_LOAD(0, &q[ldq]);
	q1 = _SSE_ADD(q1, _SSE_ADD(x1, _SSE_MUL(y1, h2)));
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[ldq]);

	for (i = 2; i < nb; i++)
	{
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
		h1 = _mm_set1_pd(hh[i-1]);
		h2 = _mm_set1_pd(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = _mm_set1_ps(hh[i-1]);
		h2 = _mm_set1_ps(hh[ldh+i]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i]);
#endif
#ifdef SINGLE_PRECISION_REAL
		h1 = vec_splats(hh[i-1]);
		h2 = vec_splats(hh[ldh+i];
#endif
#endif

		q1 = _SSE_LOAD(0, &q[i*ldq]);
		q1 = _SSE_ADD(q1, _SSE_ADD(_SSE_MUL(x1,h1), _SSE_MUL(y1, h2)));
		_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[i*ldq]);
	}
#ifdef HAVE_SSE_INTRINSICS
#ifdef DOUBLE_PRECISION_REAL
	h1 = _mm_set1_pd(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = _mm_set1_ps(hh[nb-1]);
#endif
#endif
#ifdef HAVE_VSX_SSE
#ifdef DOUBLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = vec_splats(hh[nb-1]);
#endif
#endif
	q1 = _SSE_LOAD(0, &q[nb*ldq]);
	q1 = _SSE_ADD(q1, _SSE_MUL(x1, h1));
	_SSE_STORE((__vector unsigned int) q1, 0, (unsigned int *) &q[nb*ldq]);
}

#undef REALCASE
#undef DOUBLE_PRECISION

