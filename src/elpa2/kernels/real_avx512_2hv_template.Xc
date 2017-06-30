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

#ifdef DOUBLE_PRECISION_REAL
#define offset 8

#define __AVX512_DATATYPE __m512d
#define __AVX512i __m512i
#define _AVX512_LOAD  _mm512_load_pd
#define _AVX512_STORE  _mm512_store_pd
#define _AVX512_SET1 _mm512_set1_pd
#define _AVX512_ADD _mm512_add_pd
#define _AVX512_MUL _mm512_mul_pd

#ifdef HAVE_AVX512
#define __ELPA_USE_FMA__
#define _mm512_FMA_pd(a,b,c) _mm512_fmadd_pd(a,b,c)
#endif

#define _AVX512_FMA _mm512_FMA_pd
#endif /* DOUBLE_PRECISION_REAL */

#ifdef SINGLE_PRECISION_REAL
#define offset 16

#define __AVX512_DATATYPE __m512
#define __AVX512i __m512i
#define _AVX512_LOAD  _mm512_load_ps
#define _AVX512_STORE  _mm512_store_ps
#define _AVX512_SET1 _mm512_set1_ps
#define _AVX512_ADD _mm512_add_ps
#define _AVX512_MUL _mm512_mul_ps

#ifdef HAVE_AVX512
#define __ELPA_USE_FMA__
#define _mm512_FMA_ps(a,b,c) _mm512_fmadd_ps(a,b,c)
#endif

#define _AVX512_FMA _mm512_FMA_ps
#endif /* SINGLE_PRECISION_REAL */

#ifdef DOUBLE_PRECISION_REAL
//Forward declaration
__forceinline void hh_trafo_kernel_8_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_16_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_24_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_32_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);

void double_hh_trafo_real_avx512_2hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif
#ifdef SINGLE_PRECISION_REAL
//Forward declaration
__forceinline void hh_trafo_kernel_16_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_32_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_48_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);
__forceinline void hh_trafo_kernel_64_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s);

void double_hh_trafo_real_avx512_2hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh);
#endif

#ifdef DOUBLE_PRECISION_REAL
/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine double_hh_trafo_real_avx512_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_real_avx512_2hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     type(c_ptr), value      :: q
!f>     real(kind=c_double)     :: hh(pnb,6)
!f>   end subroutine
!f> end interface
!f>#endif
*/
#endif
#ifdef SINGLE_PRECISION_REAL
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
#endif

#ifdef DOUBLE_PRECISION_REAL
void double_hh_trafo_real_avx512_2hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
#ifdef SINGLE_PRECISION_REAL
void double_hh_trafo_real_avx512_2hv_single(float* q, float* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;

	// calculating scalar product to compute
	// 2 householder vectors simultaneously
#ifdef DOUBLE_PRECISION_REAL
	double s = hh[(ldh)+1]*1.0;
#endif
#ifdef SINGLE_PRECISION_REAL
	float s = hh[(ldh)+1]*1.0f;
#endif
	#pragma ivdep
	for (i = 2; i < nb; i++)
	{
		s += hh[i-1] * hh[(i+ldh)];
	}

	// Production level kernel calls with padding
#ifdef DOUBLE_PRECISION_REAL
	for (i = 0; i < nq-24; i+=32)
	{
		hh_trafo_kernel_32_AVX512_2hv_double(&q[i], hh, nb, ldq, ldh, s);
	}
#endif
#ifdef SINGLE_PRECISION_REAL
	for (i = 0; i < nq-48; i+=64)
	{
		hh_trafo_kernel_64_AVX512_2hv_single(&q[i], hh, nb, ldq, ldh, s);
	}
#endif
	if (nq == i)
	{
		return;
	}
#ifdef DOUBLE_PRECISION_REAL
	if (nq-i == 24)
	{
		hh_trafo_kernel_24_AVX512_2hv_double(&q[i], hh, nb, ldq, ldh, s);
	}
#endif
#ifdef SINGLE_PRECISION_REAL
	if (nq-i == 48)
	{
		hh_trafo_kernel_48_AVX512_2hv_single(&q[i], hh, nb, ldq, ldh, s);
	}
#endif
#ifdef DOUBLE_PRECISION_REAL
	else if (nq-i == 16)
	{
		hh_trafo_kernel_16_AVX512_2hv_double(&q[i], hh, nb, ldq, ldh, s);
	}
#endif
#ifdef SINGLE_PRECISION_REAL
	else if (nq-i == 32)
	{
		hh_trafo_kernel_32_AVX512_2hv_single(&q[i], hh, nb, ldq, ldh, s);
	}
#endif
	else
#ifdef DOUBLE_PRECISION_REAL
	{
		hh_trafo_kernel_8_AVX512_2hv_double(&q[i], hh, nb, ldq, ldh, s);
	}
#endif
#ifdef SINGLE_PRECISION_REAL
	{
		hh_trafo_kernel_16_AVX512_2hv_single(&q[i], hh, nb, ldq, ldh, s);
	}
#endif
}

/**
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 32 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 64 rows of Q simultaneously, a
#endif

 * matrix Vector product with two householder
 * vectors + a rank 2 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_32_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
#endif
#ifdef SINGLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_64_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [24 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
        __AVX512_DATATYPE sign = (__AVX512_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_REAL
        __AVX512_DATATYPE sign = (__AVX512_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif

	__AVX512_DATATYPE x1 = _AVX512_LOAD(&q[ldq]);
	__AVX512_DATATYPE x2 = _AVX512_LOAD(&q[ldq+offset]);
	__AVX512_DATATYPE x3 = _AVX512_LOAD(&q[ldq+2*offset]);
	__AVX512_DATATYPE x4 = _AVX512_LOAD(&q[ldq+3*offset]);


	__AVX512_DATATYPE h1 = _AVX512_SET1(hh[ldh+1]);
	__AVX512_DATATYPE h2;

	__AVX512_DATATYPE q1 = _AVX512_LOAD(q);
	__AVX512_DATATYPE y1 = _AVX512_FMA(x1, h1, q1);
	__AVX512_DATATYPE q2 = _AVX512_LOAD(&q[offset]);
	__AVX512_DATATYPE y2 = _AVX512_FMA(x2, h1, q2);
	__AVX512_DATATYPE q3 = _AVX512_LOAD(&q[2*offset]);
	__AVX512_DATATYPE y3 = _AVX512_FMA(x3, h1, q3);
	__AVX512_DATATYPE q4 = _AVX512_LOAD(&q[3*offset]);
	__AVX512_DATATYPE y4 = _AVX512_FMA(x4, h1, q4);

	for(i = 2; i < nb; i++)
	{
		h1 = _AVX512_SET1(hh[i-1]);
		h2 = _AVX512_SET1(hh[ldh+i]);

		q1 = _AVX512_LOAD(&q[i*ldq]);
		x1 = _AVX512_FMA(q1, h1, x1);
		y1 = _AVX512_FMA(q1, h2, y1);
		q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);
		x2 = _AVX512_FMA(q2, h1, x2);
		y2 = _AVX512_FMA(q2, h2, y2);
		q3 = _AVX512_LOAD(&q[(i*ldq)+2*offset]);
		x3 = _AVX512_FMA(q3, h1, x3);
		y3 = _AVX512_FMA(q3, h2, y3);
		q4 = _AVX512_LOAD(&q[(i*ldq)+3*offset]);
		x4 = _AVX512_FMA(q4, h1, x4);
		y4 = _AVX512_FMA(q4, h2, y4);

	}

	h1 = _AVX512_SET1(hh[nb-1]);

	q1 = _AVX512_LOAD(&q[nb*ldq]);
	x1 = _AVX512_FMA(q1, h1, x1);
	q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);
	x2 = _AVX512_FMA(q2, h1, x2);
	q3 = _AVX512_LOAD(&q[(nb*ldq)+2*offset]);
	x3 = _AVX512_FMA(q3, h1, x3);
	q4 = _AVX512_LOAD(&q[(nb*ldq)+3*offset]);
	x4 = _AVX512_FMA(q4, h1, x4);


	/////////////////////////////////////////////////////
	// Rank-2 update of Q [24 x nb+1]
	/////////////////////////////////////////////////////

	__AVX512_DATATYPE tau1 = _AVX512_SET1(hh[0]);
	__AVX512_DATATYPE tau2 = _AVX512_SET1(hh[ldh]);
	__AVX512_DATATYPE vs = _AVX512_SET1(s);

#ifdef DOUBLE_PRECISION_REAL
	h1 = (__AVX512_DATATYPE) _mm512_xor_epi64((__AVX512i) tau1, (__AVX512i) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = (__AVX512_DATATYPE) _mm512_xor_epi32((__AVX512i) tau1, (__AVX512i) sign);
#endif
	x1 = _AVX512_MUL(x1, h1);
	x2 = _AVX512_MUL(x2, h1);
	x3 = _AVX512_MUL(x3, h1);
	x4 = _AVX512_MUL(x4, h1);

        // check ofr xor_pd on skylake
#ifdef DOUBLE_PRECISION_REAL
	h1 = (__AVX512_DATATYPE) _mm512_xor_epi64((__AVX512i) tau2, (__AVX512i) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = (__AVX512_DATATYPE) _mm512_xor_epi32((__AVX512i) tau2, (__AVX512i) sign);
#endif
	h2 = _AVX512_MUL(h1, vs);
	y1 = _AVX512_FMA(y1, h1, _AVX512_MUL(x1,h2));
	y2 = _AVX512_FMA(y2, h1, _AVX512_MUL(x2,h2));
	y3 = _AVX512_FMA(y3, h1, _AVX512_MUL(x3,h2));
	y4 = _AVX512_FMA(y4, h1, _AVX512_MUL(x4,h2));

	q1 = _AVX512_LOAD(q);
	q1 = _AVX512_ADD(q1, y1);
	_AVX512_STORE(q,q1);
	q2 = _AVX512_LOAD(&q[offset]);
	q2 = _AVX512_ADD(q2, y2);
	_AVX512_STORE(&q[offset],q2);
	q3 = _AVX512_LOAD(&q[2*offset]);
	q3 = _AVX512_ADD(q3, y3);
	_AVX512_STORE(&q[2*offset],q3);
	q4 = _AVX512_LOAD(&q[3*offset]);
	q4 = _AVX512_ADD(q4, y4);
	_AVX512_STORE(&q[3*offset],q4);

	h2 = _AVX512_SET1(hh[ldh+1]);

	q1 = _AVX512_LOAD(&q[ldq]);
	q1 = _AVX512_ADD(q1, _AVX512_FMA(y1, h2, x1));
	_AVX512_STORE(&q[ldq],q1);
	q2 = _AVX512_LOAD(&q[ldq+offset]);
	q2 = _AVX512_ADD(q2, _AVX512_FMA(y2, h2, x2));
	_AVX512_STORE(&q[ldq+offset],q2);
	q3 = _AVX512_LOAD(&q[ldq+2*offset]);
	q3 = _AVX512_ADD(q3, _AVX512_FMA(y3, h2, x3));
	_AVX512_STORE(&q[ldq+2*offset],q3);
	q4 = _AVX512_LOAD(&q[ldq+3*offset]);
	q4 = _AVX512_ADD(q4, _AVX512_FMA(y4, h2, x4));
	_AVX512_STORE(&q[ldq+3*offset],q4);

	for (i = 2; i < nb; i++)
	{
		h1 = _AVX512_SET1(hh[i-1]);
		h2 = _AVX512_SET1(hh[ldh+i]);

		q1 = _AVX512_LOAD(&q[i*ldq]);
		q1 = _AVX512_FMA(x1, h1, q1);
		q1 = _AVX512_FMA(y1, h2, q1);
		_AVX512_STORE(&q[i*ldq],q1);
		q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);
		q2 = _AVX512_FMA(x2, h1, q2);
		q2 = _AVX512_FMA(y2, h2, q2);
		_AVX512_STORE(&q[(i*ldq)+offset],q2);
		q3 = _AVX512_LOAD(&q[(i*ldq)+2*offset]);
		q3 = _AVX512_FMA(x3, h1, q3);
		q3 = _AVX512_FMA(y3, h2, q3);
		_AVX512_STORE(&q[(i*ldq)+2*offset],q3);
		q4 = _AVX512_LOAD(&q[(i*ldq)+3*offset]);
		q4 = _AVX512_FMA(x4, h1, q4);
		q4 = _AVX512_FMA(y4, h2, q4);
		_AVX512_STORE(&q[(i*ldq)+3*offset],q4);

	}

	h1 = _AVX512_SET1(hh[nb-1]);

	q1 = _AVX512_LOAD(&q[nb*ldq]);
	q1 = _AVX512_FMA(x1, h1, q1);
	_AVX512_STORE(&q[nb*ldq],q1);
	q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);
	q2 = _AVX512_FMA(x2, h1, q2);
	_AVX512_STORE(&q[(nb*ldq)+offset],q2);
	q3 = _AVX512_LOAD(&q[(nb*ldq)+2*offset]);
	q3 = _AVX512_FMA(x3, h1, q3);
	_AVX512_STORE(&q[(nb*ldq)+2*offset],q3);
	q4 = _AVX512_LOAD(&q[(nb*ldq)+3*offset]);
	q4 = _AVX512_FMA(x4, h1, q4);
	_AVX512_STORE(&q[(nb*ldq)+3*offset],q4);

}

/**
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 24 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 48 rows of Q simultaneously, a
#endif

 * matrix Vector product with two householder
 * vectors + a rank 2 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_24_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
#endif
#ifdef SINGLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_48_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [24 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
        __AVX512_DATATYPE sign = (__AVX512_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_REAL
        __AVX512_DATATYPE sign = (__AVX512_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif
	__AVX512_DATATYPE x1 = _AVX512_LOAD(&q[ldq]);
	__AVX512_DATATYPE x2 = _AVX512_LOAD(&q[ldq+offset]);
	__AVX512_DATATYPE x3 = _AVX512_LOAD(&q[ldq+2*offset]);

	__AVX512_DATATYPE h1 = _AVX512_SET1(hh[ldh+1]);
	__AVX512_DATATYPE h2;

	__AVX512_DATATYPE q1 = _AVX512_LOAD(q);
	__AVX512_DATATYPE y1 = _AVX512_FMA(x1, h1, q1);
	__AVX512_DATATYPE q2 = _AVX512_LOAD(&q[offset]);
	__AVX512_DATATYPE y2 = _AVX512_FMA(x2, h1, q2);
	__AVX512_DATATYPE q3 = _AVX512_LOAD(&q[2*offset]);
	__AVX512_DATATYPE y3 = _AVX512_FMA(x3, h1, q3);

	for(i = 2; i < nb; i++)
	{
		h1 = _AVX512_SET1(hh[i-1]);
		h2 = _AVX512_SET1(hh[ldh+i]);

		q1 = _AVX512_LOAD(&q[i*ldq]);
		x1 = _AVX512_FMA(q1, h1, x1);
		y1 = _AVX512_FMA(q1, h2, y1);
		q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);
		x2 = _AVX512_FMA(q2, h1, x2);
		y2 = _AVX512_FMA(q2, h2, y2);
		q3 = _AVX512_LOAD(&q[(i*ldq)+2*offset]);
		x3 = _AVX512_FMA(q3, h1, x3);
		y3 = _AVX512_FMA(q3, h2, y3);
	}

	h1 = _AVX512_SET1(hh[nb-1]);

	q1 = _AVX512_LOAD(&q[nb*ldq]);
	x1 = _AVX512_FMA(q1, h1, x1);
	q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);
	x2 = _AVX512_FMA(q2, h1, x2);
	q3 = _AVX512_LOAD(&q[(nb*ldq)+2*offset]);
	x3 = _AVX512_FMA(q3, h1, x3);

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [24 x nb+1]
	/////////////////////////////////////////////////////

	__AVX512_DATATYPE tau1 = _AVX512_SET1(hh[0]);
	__AVX512_DATATYPE tau2 = _AVX512_SET1(hh[ldh]);
	__AVX512_DATATYPE vs = _AVX512_SET1(s);

        // check for xor_pd on skylake
#ifdef DOUBLE_PRECISION_REAL
	h1 = (__AVX512_DATATYPE) _mm512_xor_epi64((__AVX512i) tau1, (__AVX512i) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = (__AVX512_DATATYPE) _mm512_xor_epi32((__AVX512i) tau1, (__AVX512i) sign);
#endif
	x1 = _AVX512_MUL(x1, h1);
	x2 = _AVX512_MUL(x2, h1);
	x3 = _AVX512_MUL(x3, h1);

#ifdef DOUBLE_PRECISION_REAL
	h1 = (__AVX512_DATATYPE) _mm512_xor_epi64((__AVX512i) tau2, (__AVX512i) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = (__AVX512_DATATYPE) _mm512_xor_epi32((__AVX512i) tau2, (__AVX512i) sign);
#endif
	h2 = _AVX512_MUL(h1, vs);
	y1 = _AVX512_FMA(y1, h1, _AVX512_MUL(x1,h2));
	y2 = _AVX512_FMA(y2, h1, _AVX512_MUL(x2,h2));
	y3 = _AVX512_FMA(y3, h1, _AVX512_MUL(x3,h2));

	q1 = _AVX512_LOAD(q);
	q1 = _AVX512_ADD(q1, y1);
	_AVX512_STORE(q,q1);
	q2 = _AVX512_LOAD(&q[offset]);
	q2 = _AVX512_ADD(q2, y2);
	_AVX512_STORE(&q[offset],q2);
	q3 = _AVX512_LOAD(&q[2*offset]);
	q3 = _AVX512_ADD(q3, y3);
	_AVX512_STORE(&q[2*offset],q3);

	h2 = _AVX512_SET1(hh[ldh+1]);

	q1 = _AVX512_LOAD(&q[ldq]);
	q1 = _AVX512_ADD(q1, _AVX512_FMA(y1, h2, x1));
	_AVX512_STORE(&q[ldq],q1);
	q2 = _AVX512_LOAD(&q[ldq+offset]);
	q2 = _AVX512_ADD(q2, _AVX512_FMA(y2, h2, x2));
	_AVX512_STORE(&q[ldq+offset],q2);
	q3 = _AVX512_LOAD(&q[ldq+2*offset]);
	q3 = _AVX512_ADD(q3, _AVX512_FMA(y3, h2, x3));
	_AVX512_STORE(&q[ldq+2*offset],q3);

	for (i = 2; i < nb; i++)
	{
		h1 = _AVX512_SET1(hh[i-1]);
		h2 = _AVX512_SET1(hh[ldh+i]);

		q1 = _AVX512_LOAD(&q[i*ldq]);
		q1 = _AVX512_FMA(x1, h1, q1);
		q1 = _AVX512_FMA(y1, h2, q1);
		_AVX512_STORE(&q[i*ldq],q1);
		q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);
		q2 = _AVX512_FMA(x2, h1, q2);
		q2 = _AVX512_FMA(y2, h2, q2);
		_AVX512_STORE(&q[(i*ldq)+offset],q2);
		q3 = _AVX512_LOAD(&q[(i*ldq)+2*offset]);
		q3 = _AVX512_FMA(x3, h1, q3);
		q3 = _AVX512_FMA(y3, h2, q3);
		_AVX512_STORE(&q[(i*ldq)+2*offset],q3);

	}

	h1 = _AVX512_SET1(hh[nb-1]);

	q1 = _AVX512_LOAD(&q[nb*ldq]);
	q1 = _AVX512_FMA(x1, h1, q1);
	_AVX512_STORE(&q[nb*ldq],q1);
	q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);
	q2 = _AVX512_FMA(x2, h1, q2);
	_AVX512_STORE(&q[(nb*ldq)+offset],q2);
	q3 = _AVX512_LOAD(&q[(nb*ldq)+2*offset]);
	q3 = _AVX512_FMA(x3, h1, q3);
	_AVX512_STORE(&q[(nb*ldq)+2*offset],q3);

}

/**
 * Unrolled kernel that computes
#ifdef DOUBLE_PRECISION_REAL
 * 16 rows of Q simultaneously, a
#endif
#ifdef SINGLE_PRECISION_REAL
 * 32 rows of Q simultaneously, a
#endif
 * matrix Vector product with two householder
 * vectors + a rank 2 update is performed
 */
#ifdef DOUBLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_16_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
#endif
#ifdef SINGLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_32_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [16 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
        __AVX512_DATATYPE sign = (__AVX512_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_REAL
        __AVX512_DATATYPE sign = (__AVX512_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif
	__AVX512_DATATYPE x1 = _AVX512_LOAD(&q[ldq]);
	__AVX512_DATATYPE x2 = _AVX512_LOAD(&q[ldq+offset]);

	__AVX512_DATATYPE h1 = _AVX512_SET1(hh[ldh+1]);
	__AVX512_DATATYPE h2;

	__AVX512_DATATYPE q1 = _AVX512_LOAD(q);
	__AVX512_DATATYPE y1 = _AVX512_FMA(x1, h1, q1);
	__AVX512_DATATYPE q2 = _AVX512_LOAD(&q[offset]);
	__AVX512_DATATYPE y2 = _AVX512_FMA(x2, h1, q2);

	for(i = 2; i < nb; i++)
	{
		h1 = _AVX512_SET1(hh[i-1]);
		h2 = _AVX512_SET1(hh[ldh+i]);

		q1 = _AVX512_LOAD(&q[i*ldq]);
		x1 = _AVX512_FMA(q1, h1, x1);
		y1 = _AVX512_FMA(q1, h2, y1);
		q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);
		x2 = _AVX512_FMA(q2, h1, x2);
		y2 = _AVX512_FMA(q2, h2, y2);
	}

	h1 = _AVX512_SET1(hh[nb-1]);

	q1 = _AVX512_LOAD(&q[nb*ldq]);
	x1 = _AVX512_FMA(q1, h1, x1);
	q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);
	x2 = _AVX512_FMA(q2, h1, x2);

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [16 x nb+1]
	/////////////////////////////////////////////////////

	__AVX512_DATATYPE tau1 = _AVX512_SET1(hh[0]);
	__AVX512_DATATYPE tau2 = _AVX512_SET1(hh[ldh]);
	__AVX512_DATATYPE vs = _AVX512_SET1(s);
        // check for xor_pd on skylake
#ifdef DOUBLE_PRECISION_REAL
	h1 = (__AVX512_DATATYPE) _mm512_xor_epi64((__AVX512i) tau1, (__AVX512i) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = (__AVX512_DATATYPE) _mm512_xor_epi32((__AVX512i) tau1, (__AVX512i) sign);
#endif
	x1 = _AVX512_MUL(x1, h1);
	x2 = _AVX512_MUL(x2, h1);
#ifdef DOUBLE_PRECISION_REAL
	h1 = (__AVX512_DATATYPE) _mm512_xor_epi64((__AVX512i) tau2, (__AVX512i) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = (__AVX512_DATATYPE) _mm512_xor_epi32((__AVX512i) tau2, (__AVX512i) sign);
#endif
	h2 = _AVX512_MUL(h1, vs);

	y1 = _AVX512_FMA(y1, h1, _AVX512_MUL(x1,h2));
	y2 = _AVX512_FMA(y2, h1, _AVX512_MUL(x2,h2));

	q1 = _AVX512_LOAD(q);
	q1 = _AVX512_ADD(q1, y1);
	_AVX512_STORE(q,q1);
	q2 = _AVX512_LOAD(&q[offset]);
	q2 = _AVX512_ADD(q2, y2);
	_AVX512_STORE(&q[offset],q2);

	h2 = _AVX512_SET1(hh[ldh+1]);

	q1 = _AVX512_LOAD(&q[ldq]);
	q1 = _AVX512_ADD(q1, _AVX512_FMA(y1, h2, x1));
	_AVX512_STORE(&q[ldq],q1);
	q2 = _AVX512_LOAD(&q[ldq+offset]);
	q2 = _AVX512_ADD(q2, _AVX512_FMA(y2, h2, x2));
	_AVX512_STORE(&q[ldq+offset],q2);

	for (i = 2; i < nb; i++)
	{
		h1 = _AVX512_SET1(hh[i-1]);
		h2 = _AVX512_SET1(hh[ldh+i]);

		q1 = _AVX512_LOAD(&q[i*ldq]);
		q1 = _AVX512_FMA(x1, h1, q1);
		q1 = _AVX512_FMA(y1, h2, q1);
		_AVX512_STORE(&q[i*ldq],q1);
		q2 = _AVX512_LOAD(&q[(i*ldq)+offset]);
		q2 = _AVX512_FMA(x2, h1, q2);
		q2 = _AVX512_FMA(y2, h2, q2);
		_AVX512_STORE(&q[(i*ldq)+offset],q2);
	}

	h1 = _AVX512_SET1(hh[nb-1]);

	q1 = _AVX512_LOAD(&q[nb*ldq]);
	q1 = _AVX512_FMA(x1, h1, q1);
	_AVX512_STORE(&q[nb*ldq],q1);
	q2 = _AVX512_LOAD(&q[(nb*ldq)+offset]);
	q2 = _AVX512_FMA(x2, h1, q2);
	_AVX512_STORE(&q[(nb*ldq)+offset],q2);

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
 __forceinline void hh_trafo_kernel_8_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
#endif
#ifdef SINGLE_PRECISION_REAL
 __forceinline void hh_trafo_kernel_16_AVX512_2hv_single(float* q, float* hh, int nb, int ldq, int ldh, float s)
#endif
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [8 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
#ifdef DOUBLE_PRECISION_REAL
        __AVX512_DATATYPE sign = (__AVX512_DATATYPE)_mm512_set1_epi64(0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_REAL
        __AVX512_DATATYPE sign = (__AVX512_DATATYPE)_mm512_set1_epi32(0x80000000);
#endif
	__AVX512_DATATYPE x1 = _AVX512_LOAD(&q[ldq]);

	__AVX512_DATATYPE h1 = _AVX512_SET1(hh[ldh+1]);
	__AVX512_DATATYPE h2;

	__AVX512_DATATYPE q1 = _AVX512_LOAD(q);
	__AVX512_DATATYPE y1 = _AVX512_FMA(x1, h1, q1);

	for(i = 2; i < nb; i++)
	{
		h1 = _AVX512_SET1(hh[i-1]);
		h2 = _AVX512_SET1(hh[ldh+i]);

		q1 = _AVX512_LOAD(&q[i*ldq]);
		x1 = _AVX512_FMA(q1, h1, x1);
		y1 = _AVX512_FMA(q1, h2, y1);
	}

	h1 = _AVX512_SET1(hh[nb-1]);

	q1 = _AVX512_LOAD(&q[nb*ldq]);
	x1 = _AVX512_FMA(q1, h1, x1);

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [8 x nb+1]
	/////////////////////////////////////////////////////

	__AVX512_DATATYPE tau1 = _AVX512_SET1(hh[0]);
	__AVX512_DATATYPE tau2 = _AVX512_SET1(hh[ldh]);
	__AVX512_DATATYPE vs = _AVX512_SET1(s);

#ifdef DOUBLE_PRECISION_REAL
	h1 = (__AVX512_DATATYPE) _mm512_xor_epi64((__AVX512i) tau1, (__AVX512i) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = (__AVX512_DATATYPE) _mm512_xor_epi32((__AVX512i) tau1, (__AVX512i) sign);
#endif

	x1 = _AVX512_MUL(x1, h1);

#ifdef DOUBLE_PRECISION_REAL
	h1 = (__AVX512_DATATYPE) _mm512_xor_epi64((__AVX512i) tau2, (__AVX512i) sign);
#endif
#ifdef SINGLE_PRECISION_REAL
	h1 = (__AVX512_DATATYPE) _mm512_xor_epi32((__AVX512i) tau2, (__AVX512i) sign);
#endif

	h2 = _AVX512_MUL(h1, vs);

	y1 = _AVX512_FMA(y1, h1, _AVX512_MUL(x1,h2));

	q1 = _AVX512_LOAD(q);
	q1 = _AVX512_ADD(q1, y1);
	_AVX512_STORE(q,q1);

	h2 = _AVX512_SET1(hh[ldh+1]);

	q1 = _AVX512_LOAD(&q[ldq]);
	q1 = _AVX512_ADD(q1, _AVX512_FMA(y1, h2, x1));
	_AVX512_STORE(&q[ldq],q1);

	for (i = 2; i < nb; i++)
	{
		h1 = _AVX512_SET1(hh[i-1]);
		h2 = _AVX512_SET1(hh[ldh+i]);

		q1 = _AVX512_LOAD(&q[i*ldq]);
		q1 = _AVX512_FMA(x1, h1, q1);
		q1 = _AVX512_FMA(y1, h2, q1);
		_AVX512_STORE(&q[i*ldq],q1);
	}

	h1 = _AVX512_SET1(hh[nb-1]);

	q1 = _AVX512_LOAD(&q[nb*ldq]);
	q1 = _AVX512_FMA(x1, h1, q1);
	_AVX512_STORE(&q[nb*ldq],q1);

}

