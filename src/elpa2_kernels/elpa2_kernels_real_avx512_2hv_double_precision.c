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

#ifdef HAVE_AVX512
#define __ELPA_USE_FMA__
#define _mm512_FMA_pd(a,b,c) _mm512_fmadd_pd(a,b,c)
#endif


//Forward declaration
__forceinline void hh_trafo_kernel_8_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_16_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_24_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);
__forceinline void hh_trafo_kernel_32_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s);

void double_hh_trafo_real_avx512_2hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh);
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

void double_hh_trafo_real_avx512_2hv_double(double* q, double* hh, int* pnb, int* pnq, int* pldq, int* pldh)
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
	for (i = 0; i < nq-24; i+=32)
	{
		hh_trafo_kernel_32_AVX512_2hv_double(&q[i], hh, nb, ldq, ldh, s);
	}

	if (nq == i)
	{
		return;
	}

	if (nq-i == 24)
	{
		hh_trafo_kernel_24_AVX512_2hv_double(&q[i], hh, nb, ldq, ldh, s);
	}
	else if (nq-i == 16)
	{
		hh_trafo_kernel_16_AVX512_2hv_double(&q[i], hh, nb, ldq, ldh, s);
	}
	else
	{
		hh_trafo_kernel_8_AVX512_2hv_double(&q[i], hh, nb, ldq, ldh, s);
	}
}
/**
 * Unrolled kernel that computes
 * 32 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_32_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [24 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	// carefull here
        __m512d sign = (__m512d)_mm512_set1_epi64(0x8000000000000000);

	__m512d x1 = _mm512_load_pd(&q[ldq]);
	__m512d x2 = _mm512_load_pd(&q[ldq+8]);
	__m512d x3 = _mm512_load_pd(&q[ldq+16]);
	__m512d x4 = _mm512_load_pd(&q[ldq+24]);


	__m512d h1 = _mm512_set1_pd(hh[ldh+1]);
	__m512d h2;

	__m512d q1 = _mm512_load_pd(q);
	__m512d y1 = _mm512_FMA_pd(x1, h1, q1);
	__m512d q2 = _mm512_load_pd(&q[8]);
	__m512d y2 = _mm512_FMA_pd(x2, h1, q2);
	__m512d q3 = _mm512_load_pd(&q[16]);
	__m512d y3 = _mm512_FMA_pd(x3, h1, q3);
	__m512d q4 = _mm512_load_pd(&q[24]);
	__m512d y4 = _mm512_FMA_pd(x4, h1, q4);

	for(i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-1]);
		h2 = _mm512_set1_pd(hh[ldh+i]);

		q1 = _mm512_load_pd(&q[i*ldq]);
		x1 = _mm512_FMA_pd(q1, h1, x1);
		y1 = _mm512_FMA_pd(q1, h2, y1);
		q2 = _mm512_load_pd(&q[(i*ldq)+8]);
		x2 = _mm512_FMA_pd(q2, h1, x2);
		y2 = _mm512_FMA_pd(q2, h2, y2);
		q3 = _mm512_load_pd(&q[(i*ldq)+16]);
		x3 = _mm512_FMA_pd(q3, h1, x3);
		y3 = _mm512_FMA_pd(q3, h2, y3);
		q4 = _mm512_load_pd(&q[(i*ldq)+24]);
		x4 = _mm512_FMA_pd(q4, h1, x4);
		y4 = _mm512_FMA_pd(q4, h2, y4);

	}

	h1 = _mm512_set1_pd(hh[nb-1]);

	q1 = _mm512_load_pd(&q[nb*ldq]);
	x1 = _mm512_FMA_pd(q1, h1, x1);
	q2 = _mm512_load_pd(&q[(nb*ldq)+8]);
	x2 = _mm512_FMA_pd(q2, h1, x2);
	q3 = _mm512_load_pd(&q[(nb*ldq)+16]);
	x3 = _mm512_FMA_pd(q3, h1, x3);
	q4 = _mm512_load_pd(&q[(nb*ldq)+24]);
	x4 = _mm512_FMA_pd(q4, h1, x4);


	/////////////////////////////////////////////////////
	// Rank-2 update of Q [24 x nb+1]
	/////////////////////////////////////////////////////

	__m512d tau1 = _mm512_set1_pd(hh[0]);
	__m512d tau2 = _mm512_set1_pd(hh[ldh]);
	__m512d vs = _mm512_set1_pd(s);

	h1 = (__m512d) _mm512_xor_epi64((__m512i) tau1, (__m512i) sign);
	x1 = _mm512_mul_pd(x1, h1);
	x2 = _mm512_mul_pd(x2, h1);
	x3 = _mm512_mul_pd(x3, h1);
	x4 = _mm512_mul_pd(x4, h1);

	h1 = (__m512d) _mm512_xor_epi64((__m512i) tau2, (__m512i) sign);
	h2 = _mm512_mul_pd(h1, vs);
	y1 = _mm512_FMA_pd(y1, h1, _mm512_mul_pd(x1,h2));
	y2 = _mm512_FMA_pd(y2, h1, _mm512_mul_pd(x2,h2));
	y3 = _mm512_FMA_pd(y3, h1, _mm512_mul_pd(x3,h2));
	y4 = _mm512_FMA_pd(y4, h1, _mm512_mul_pd(x4,h2));

	q1 = _mm512_load_pd(q);
	q1 = _mm512_add_pd(q1, y1);
	_mm512_store_pd(q,q1);
	q2 = _mm512_load_pd(&q[8]);
	q2 = _mm512_add_pd(q2, y2);
	_mm512_store_pd(&q[8],q2);
	q3 = _mm512_load_pd(&q[16]);
	q3 = _mm512_add_pd(q3, y3);
	_mm512_store_pd(&q[16],q3);
	q4 = _mm512_load_pd(&q[24]);
	q4 = _mm512_add_pd(q4, y4);
	_mm512_store_pd(&q[24],q4);

	h2 = _mm512_set1_pd(hh[ldh+1]);

	q1 = _mm512_load_pd(&q[ldq]);
	q1 = _mm512_add_pd(q1, _mm512_FMA_pd(y1, h2, x1));
	_mm512_store_pd(&q[ldq],q1);
	q2 = _mm512_load_pd(&q[ldq+8]);
	q2 = _mm512_add_pd(q2, _mm512_FMA_pd(y2, h2, x2));
	_mm512_store_pd(&q[ldq+8],q2);
	q3 = _mm512_load_pd(&q[ldq+16]);
	q3 = _mm512_add_pd(q3, _mm512_FMA_pd(y3, h2, x3));
	_mm512_store_pd(&q[ldq+16],q3);
	q4 = _mm512_load_pd(&q[ldq+24]);
	q4 = _mm512_add_pd(q4, _mm512_FMA_pd(y4, h2, x4));
	_mm512_store_pd(&q[ldq+24],q4);

	for (i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-1]);
		h2 = _mm512_set1_pd(hh[ldh+i]);

		q1 = _mm512_load_pd(&q[i*ldq]);
		q1 = _mm512_FMA_pd(x1, h1, q1);
		q1 = _mm512_FMA_pd(y1, h2, q1);
		_mm512_store_pd(&q[i*ldq],q1);
		q2 = _mm512_load_pd(&q[(i*ldq)+8]);
		q2 = _mm512_FMA_pd(x2, h1, q2);
		q2 = _mm512_FMA_pd(y2, h2, q2);
		_mm512_store_pd(&q[(i*ldq)+8],q2);
		q3 = _mm512_load_pd(&q[(i*ldq)+16]);
		q3 = _mm512_FMA_pd(x3, h1, q3);
		q3 = _mm512_FMA_pd(y3, h2, q3);
		_mm512_store_pd(&q[(i*ldq)+16],q3);
		q4 = _mm512_load_pd(&q[(i*ldq)+24]);
		q4 = _mm512_FMA_pd(x4, h1, q4);
		q4 = _mm512_FMA_pd(y4, h2, q4);
		_mm512_store_pd(&q[(i*ldq)+24],q4);

	}

	h1 = _mm512_set1_pd(hh[nb-1]);

	q1 = _mm512_load_pd(&q[nb*ldq]);
	q1 = _mm512_FMA_pd(x1, h1, q1);
	_mm512_store_pd(&q[nb*ldq],q1);
	q2 = _mm512_load_pd(&q[(nb*ldq)+8]);
	q2 = _mm512_FMA_pd(x2, h1, q2);
	_mm512_store_pd(&q[(nb*ldq)+8],q2);
	q3 = _mm512_load_pd(&q[(nb*ldq)+16]);
	q3 = _mm512_FMA_pd(x3, h1, q3);
	_mm512_store_pd(&q[(nb*ldq)+16],q3);
	q4 = _mm512_load_pd(&q[(nb*ldq)+24]);
	q4 = _mm512_FMA_pd(x4, h1, q4);
	_mm512_store_pd(&q[(nb*ldq)+24],q4);

}



/**
 * Unrolled kernel that computes
 * 24 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_24_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [24 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	// carefull here
        __m512d sign = (__m512d)_mm512_set1_epi64(0x8000000000000000);

	__m512d x1 = _mm512_load_pd(&q[ldq]);
	__m512d x2 = _mm512_load_pd(&q[ldq+8]);
	__m512d x3 = _mm512_load_pd(&q[ldq+16]);

	 // checkthis
	__m512d h1 = _mm512_set1_pd(hh[ldh+1]);
	__m512d h2;

	__m512d q1 = _mm512_load_pd(q);
	__m512d y1 = _mm512_FMA_pd(x1, h1, q1);
	__m512d q2 = _mm512_load_pd(&q[8]);
	__m512d y2 = _mm512_FMA_pd(x2, h1, q2);
	__m512d q3 = _mm512_load_pd(&q[16]);
	__m512d y3 = _mm512_FMA_pd(x3, h1, q3);

	for(i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-1]);
		h2 = _mm512_set1_pd(hh[ldh+i]);

		q1 = _mm512_load_pd(&q[i*ldq]);
		x1 = _mm512_FMA_pd(q1, h1, x1);
		y1 = _mm512_FMA_pd(q1, h2, y1);
		q2 = _mm512_load_pd(&q[(i*ldq)+8]);
		x2 = _mm512_FMA_pd(q2, h1, x2);
		y2 = _mm512_FMA_pd(q2, h2, y2);
		q3 = _mm512_load_pd(&q[(i*ldq)+16]);
		x3 = _mm512_FMA_pd(q3, h1, x3);
		y3 = _mm512_FMA_pd(q3, h2, y3);
	}

	h1 = _mm512_set1_pd(hh[nb-1]);

	q1 = _mm512_load_pd(&q[nb*ldq]);
	x1 = _mm512_FMA_pd(q1, h1, x1);
	q2 = _mm512_load_pd(&q[(nb*ldq)+8]);
	x2 = _mm512_FMA_pd(q2, h1, x2);
	q3 = _mm512_load_pd(&q[(nb*ldq)+16]);
	x3 = _mm512_FMA_pd(q3, h1, x3);

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [24 x nb+1]
	/////////////////////////////////////////////////////

	__m512d tau1 = _mm512_set1_pd(hh[0]);
	__m512d tau2 = _mm512_set1_pd(hh[ldh]);
	__m512d vs = _mm512_set1_pd(s);

	h1 = (__m512d) _mm512_xor_epi64((__m512i) tau1, (__m512i) sign);
	x1 = _mm512_mul_pd(x1, h1);
	x2 = _mm512_mul_pd(x2, h1);
	x3 = _mm512_mul_pd(x3, h1);

	h1 = (__m512d) _mm512_xor_epi64((__m512i) tau2, (__m512i) sign);
	h2 = _mm512_mul_pd(h1, vs);
	y1 = _mm512_FMA_pd(y1, h1, _mm512_mul_pd(x1,h2));
	y2 = _mm512_FMA_pd(y2, h1, _mm512_mul_pd(x2,h2));
	y3 = _mm512_FMA_pd(y3, h1, _mm512_mul_pd(x3,h2));

	q1 = _mm512_load_pd(q);
	q1 = _mm512_add_pd(q1, y1);
	_mm512_store_pd(q,q1);
	q2 = _mm512_load_pd(&q[8]);
	q2 = _mm512_add_pd(q2, y2);
	_mm512_store_pd(&q[8],q2);
	q3 = _mm512_load_pd(&q[16]);
	q3 = _mm512_add_pd(q3, y3);
	_mm512_store_pd(&q[16],q3);

	h2 = _mm512_set1_pd(hh[ldh+1]);

	q1 = _mm512_load_pd(&q[ldq]);
	q1 = _mm512_add_pd(q1, _mm512_FMA_pd(y1, h2, x1));
	_mm512_store_pd(&q[ldq],q1);
	q2 = _mm512_load_pd(&q[ldq+8]);
	q2 = _mm512_add_pd(q2, _mm512_FMA_pd(y2, h2, x2));
	_mm512_store_pd(&q[ldq+8],q2);
	q3 = _mm512_load_pd(&q[ldq+16]);
	q3 = _mm512_add_pd(q3, _mm512_FMA_pd(y3, h2, x3));
	_mm512_store_pd(&q[ldq+16],q3);

	for (i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-1]);
		h2 = _mm512_set1_pd(hh[ldh+i]);

		q1 = _mm512_load_pd(&q[i*ldq]);
		q1 = _mm512_FMA_pd(x1, h1, q1);
		q1 = _mm512_FMA_pd(y1, h2, q1);
		_mm512_store_pd(&q[i*ldq],q1);
		q2 = _mm512_load_pd(&q[(i*ldq)+8]);
		q2 = _mm512_FMA_pd(x2, h1, q2);
		q2 = _mm512_FMA_pd(y2, h2, q2);
		_mm512_store_pd(&q[(i*ldq)+8],q2);
		q3 = _mm512_load_pd(&q[(i*ldq)+16]);
		q3 = _mm512_FMA_pd(x3, h1, q3);
		q3 = _mm512_FMA_pd(y3, h2, q3);
		_mm512_store_pd(&q[(i*ldq)+16],q3);

	}

	h1 = _mm512_set1_pd(hh[nb-1]);

	q1 = _mm512_load_pd(&q[nb*ldq]);
	q1 = _mm512_FMA_pd(x1, h1, q1);
	_mm512_store_pd(&q[nb*ldq],q1);
	q2 = _mm512_load_pd(&q[(nb*ldq)+8]);
	q2 = _mm512_FMA_pd(x2, h1, q2);
	_mm512_store_pd(&q[(nb*ldq)+8],q2);
	q3 = _mm512_load_pd(&q[(nb*ldq)+16]);
	q3 = _mm512_FMA_pd(x3, h1, q3);
	_mm512_store_pd(&q[(nb*ldq)+16],q3);

}

/**
 * Unrolled kernel that computes
 * 16 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_16_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [16 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	// carefull here
        __m512d sign = (__m512d)_mm512_set1_epi64(0x8000000000000000);
	__m512d x1 = _mm512_load_pd(&q[ldq]);
	__m512d x2 = _mm512_load_pd(&q[ldq+8]);

	__m512d h1 = _mm512_set1_pd(hh[ldh+1]);
	__m512d h2;

	__m512d q1 = _mm512_load_pd(q);
	__m512d y1 = _mm512_FMA_pd(x1, h1, q1);
	__m512d q2 = _mm512_load_pd(&q[8]);
	__m512d y2 = _mm512_FMA_pd(x2, h1, q2);

	for(i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-1]);
		h2 = _mm512_set1_pd(hh[ldh+i]);

		q1 = _mm512_load_pd(&q[i*ldq]);
		x1 = _mm512_FMA_pd(q1, h1, x1);
		y1 = _mm512_FMA_pd(q1, h2, y1);
		q2 = _mm512_load_pd(&q[(i*ldq)+8]);
		x2 = _mm512_FMA_pd(q2, h1, x2);
		y2 = _mm512_FMA_pd(q2, h2, y2);
	}

	h1 = _mm512_set1_pd(hh[nb-1]);

	q1 = _mm512_load_pd(&q[nb*ldq]);
	x1 = _mm512_FMA_pd(q1, h1, x1);
	q2 = _mm512_load_pd(&q[(nb*ldq)+8]);
	x2 = _mm512_FMA_pd(q2, h1, x2);

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [16 x nb+1]
	/////////////////////////////////////////////////////

	__m512d tau1 = _mm512_set1_pd(hh[0]);
	__m512d tau2 = _mm512_set1_pd(hh[ldh]);
	__m512d vs = _mm512_set1_pd(s);

	h1 = (__m512d) _mm512_xor_epi64((__m512i) tau1, (__m512i) sign);
	x1 = _mm512_mul_pd(x1, h1);
	x2 = _mm512_mul_pd(x2, h1);

	h1 = (__m512d) _mm512_xor_epi64((__m512i) tau2, (__m512i) sign);
	h2 = _mm512_mul_pd(h1, vs);

	y1 = _mm512_FMA_pd(y1, h1, _mm512_mul_pd(x1,h2));
	y2 = _mm512_FMA_pd(y2, h1, _mm512_mul_pd(x2,h2));

	q1 = _mm512_load_pd(q);
	q1 = _mm512_add_pd(q1, y1);
	_mm512_store_pd(q,q1);
	q2 = _mm512_load_pd(&q[8]);
	q2 = _mm512_add_pd(q2, y2);
	_mm512_store_pd(&q[8],q2);

	h2 = _mm512_set1_pd(hh[ldh+1]);

	q1 = _mm512_load_pd(&q[ldq]);
	q1 = _mm512_add_pd(q1, _mm512_FMA_pd(y1, h2, x1));
	_mm512_store_pd(&q[ldq],q1);
	q2 = _mm512_load_pd(&q[ldq+8]);
	q2 = _mm512_add_pd(q2, _mm512_FMA_pd(y2, h2, x2));
	_mm512_store_pd(&q[ldq+8],q2);

	for (i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-1]);
		h2 = _mm512_set1_pd(hh[ldh+i]);

		q1 = _mm512_load_pd(&q[i*ldq]);
		q1 = _mm512_FMA_pd(x1, h1, q1);
		q1 = _mm512_FMA_pd(y1, h2, q1);
		_mm512_store_pd(&q[i*ldq],q1);
		q2 = _mm512_load_pd(&q[(i*ldq)+8]);
		q2 = _mm512_FMA_pd(x2, h1, q2);
		q2 = _mm512_FMA_pd(y2, h2, q2);
		_mm512_store_pd(&q[(i*ldq)+8],q2);
	}

	h1 = _mm512_set1_pd(hh[nb-1]);

	q1 = _mm512_load_pd(&q[nb*ldq]);
	q1 = _mm512_FMA_pd(x1, h1, q1);
	_mm512_store_pd(&q[nb*ldq],q1);
	q2 = _mm512_load_pd(&q[(nb*ldq)+8]);
	q2 = _mm512_FMA_pd(x2, h1, q2);
	_mm512_store_pd(&q[(nb*ldq)+8],q2);

}

/**
 * Unrolled kernel that computes
 * 8 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_8_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [8 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	int i;
	// Needed bit mask for floating point sign flip
	// carefull here
        __m512d sign = (__m512d)_mm512_set1_epi64(0x8000000000000000);
	__m512d x1 = _mm512_load_pd(&q[ldq]);

	__m512d h1 = _mm512_set1_pd(hh[ldh+1]);
	__m512d h2;

	__m512d q1 = _mm512_load_pd(q);
	__m512d y1 = _mm512_FMA_pd(x1, h1, q1);

	for(i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-1]);
		h2 = _mm512_set1_pd(hh[ldh+i]);

		q1 = _mm512_load_pd(&q[i*ldq]);
		x1 = _mm512_FMA_pd(q1, h1, x1);
		y1 = _mm512_FMA_pd(q1, h2, y1);
	}

	h1 = _mm512_set1_pd(hh[nb-1]);

	q1 = _mm512_load_pd(&q[nb*ldq]);
	x1 = _mm512_FMA_pd(q1, h1, x1);

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [8 x nb+1]
	/////////////////////////////////////////////////////

	__m512d tau1 = _mm512_set1_pd(hh[0]);
	__m512d tau2 = _mm512_set1_pd(hh[ldh]);
	__m512d vs = _mm512_set1_pd(s);

	h1 = (__m512d) _mm512_xor_epi64((__m512i) tau1, (__m512i) sign);
	x1 = _mm512_mul_pd(x1, h1);

	h1 = (__m512d) _mm512_xor_epi64((__m512i) tau2, (__m512i) sign);
	h2 = _mm512_mul_pd(h1, vs);

	y1 = _mm512_FMA_pd(y1, h1, _mm512_mul_pd(x1,h2));

	q1 = _mm512_load_pd(q);
	q1 = _mm512_add_pd(q1, y1);
	_mm512_store_pd(q,q1);

	h2 = _mm512_set1_pd(hh[ldh+1]);

	q1 = _mm512_load_pd(&q[ldq]);
	q1 = _mm512_add_pd(q1, _mm512_FMA_pd(y1, h2, x1));
	_mm512_store_pd(&q[ldq],q1);

	for (i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-1]);
		h2 = _mm512_set1_pd(hh[ldh+i]);

		q1 = _mm512_load_pd(&q[i*ldq]);
		q1 = _mm512_FMA_pd(x1, h1, q1);
		q1 = _mm512_FMA_pd(y1, h2, q1);
		_mm512_store_pd(&q[i*ldq],q1);
	}

	h1 = _mm512_set1_pd(hh[nb-1]);

	q1 = _mm512_load_pd(&q[nb*ldq]);
	q1 = _mm512_FMA_pd(x1, h1, q1);
	_mm512_store_pd(&q[nb*ldq],q1);

}

#if 0
/**
 * Unrolled kernel that computes
 * 4 rows of Q simultaneously, a
 * matrix vector product with two householder
 * vectors + a rank 2 update is performed
 */
 __forceinline void hh_trafo_kernel_4_AVX512_2hv_double(double* q, double* hh, int nb, int ldq, int ldh, double s)
{
	/////////////////////////////////////////////////////
	// Matrix Vector Multiplication, Q [4 x nb+1] * hh
	// hh contains two householder vectors, with offset 1
	/////////////////////////////////////////////////////
	//
	int i;
	// Needed bit mask for floating point sign flip
        __m512d sign = (__m512d)_mm512_set1_epi64(0x8000000000000000);

	__m512d zero = {0};
	__m512d x1 = _mm512_mask_load_pd(zero, 0x0f, &q[ldq]);

	__m512d h1 = _mm512_set1_pd(hh[ldh+1]);
	__m512d h2;

	__m512d q1 = _mm512_mask_load_pd(zero, 0x0f, &q[0]);
	__m512d y1 = _mm512_FMA_pd(x1, h1, q1);

	for(i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-1]);
		h2 = _mm512_set1_pd(hh[ldh+i]);
		q1 = _mm512_mask_load_pd(zero, 0x0f, &q[i*ldq]);
		x1 = _mm512_FMA_pd(q1, h1, x1);
	}

	h1 = _mm512_set1_pd(hh[nb-1]);
	q1 = _mm512_mask_load_pd(zero, 0x0f, &q[nb*ldq]);
	x1 = _mm512_FMA_pd(q1, h1, x1);

	/////////////////////////////////////////////////////
	// Rank-2 update of Q [4 x nb+1]
	/////////////////////////////////////////////////////

	__m512d tau1 = _mm512_set1_pd(hh[0]);
	__m512d tau2 = _mm512_set1_pd(hh[ldh]);
	__m512d vs = _mm512_set1_pd(s);

	h1 = (__m512d) _mm512_xor_epi64((__m512i) tau1, (__m512i) sign);
	x1 = _mm512_mul_pd(x1, h1);
	h1 = (__m512d) _mm512_xor_epi64((__m512i) tau2, (__m512i) sign);
	h2 = _mm512_mul_pd(h1, vs);
	y1 = _mm512_FMA_pd(y1, h1, _mm512_mul_pd(x1,h2));

	q1 = _mm512_mask_load_pd(zero, 0x0f, q);
	q1 = _mm512_add_pd(q1, y1);
	_mm512_mask_store_pd(q, 0x0f ,q1);

	h2 = _mm512_set1_pd(hh[ldh+1]);
	q1 = _mm512_mask_load_pd(zero, 0x0f, &q[ldq]);
	q1 = _mm512_add_pd(q1, _mm512_FMA_pd(y1, h2, x1));
	_mm512_mask_store_pd(&q[ldq], 0x0f, q1);

	for (i = 2; i < nb; i++)
	{
		h1 = _mm512_set1_pd(hh[i-1]);
		h2 = _mm512_set1_pd(hh[ldh+i]);
		q1 = _mm512_mask_load_pd(zero, 0x0f, &q[i*ldq]);
		q1 = _mm512_FMA_pd(x1, h1, q1);
		q1 = _mm512_FMA_pd(y1, h2, q1);
		_mm512_mask_store_pd(&q[i*ldq], 0x0f, q1);
	}

	h1 = _mm512_set1_pd(hh[nb-1]);
	q1 = _mm512_mask_load_pd(zero, 0x0f, &q[nb*ldq]);
	q1 = _mm512_FMA_pd(x1, h1, q1);
	_mm512_mask_store_pd(&q[nb*ldq], 0x0f, q1);
}
#endif
