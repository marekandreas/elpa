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

#include <complex.h>
#include <x86intrin.h>

#define __forceinline __attribute__((always_inline))

#ifdef HAVE_AVX512

#define __ELPA_USE_FMA__
#define _mm512_FMADDSUB_ps(a,b,c) _mm512_fmaddsub_ps(a,b,c)
#define _mm512_FMSUBADD_ps(a,b,c) _mm512_fmsubadd_ps(a,b,c)

#endif

//Forward declaration
static  __forceinline void hh_trafo_complex_kernel_48_AVX512_1hv_single(float complex* q, float complex* hh, int nb, int ldq);
static  __forceinline void hh_trafo_complex_kernel_32_AVX512_1hv_single(float complex* q, float complex* hh, int nb, int ldq);
static  __forceinline void hh_trafo_complex_kernel_16_AVX512_1hv_single(float complex* q, float complex* hh, int nb, int ldq);

/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine single_hh_trafo_complex_avx512_1hv_single(q, hh, pnb, pnq, pldq) &
!f>                             bind(C, name="single_hh_trafo_complex_avx512_1hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq
!f>     complex(kind=c_float)     :: q(*)
!f>     complex(kind=c_float)     :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

void single_hh_trafo_complex_avx512_1hv_single(float complex* q, float complex* hh, int* pnb, int* pnq, int* pldq)
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	//int ldh = *pldh;

	for (i = 0; i < nq-32; i+=48)
	{
		hh_trafo_complex_kernel_48_AVX512_1hv_single(&q[i], hh, nb, ldq);
	}
	if (nq == i)
	{
		return;
	}
	if (nq-i == 32)
	{
		hh_trafo_complex_kernel_32_AVX512_1hv_single(&q[i], hh, nb, ldq);
	}
	else
	{
		hh_trafo_complex_kernel_16_AVX512_1hv_single(&q[i], hh, nb, ldq);
	}
}

static __forceinline void hh_trafo_complex_kernel_48_AVX512_1hv_single(float complex* q, float complex* hh, int nb, int ldq)
{
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;

	__m512 x1, x2, x3, x4, x5, x6;
	__m512 q1, q2, q3, q4, q5, q6;
	__m512 h1_real, h1_imag;
	__m512 tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
	int i=0;

	__m512 sign = (__m512)_mm512_set1_epi32(0x80000000);

	x1 = _mm512_load_ps(&q_dbl[0]);    // complex 1, 2, 3, 4, 5, 6, 7, 8
	x2 = _mm512_load_ps(&q_dbl[16]);   // complex 9, 10, 11, 12, 13, 14, 15, 16
	x3 = _mm512_load_ps(&q_dbl[32]);   // complex 17 ...24
	x4 = _mm512_load_ps(&q_dbl[48]);   // complex 25 .. 32
	x5 = _mm512_load_ps(&q_dbl[64]);   // complex 33 .. 40
	x6 = _mm512_load_ps(&q_dbl[80]);   // complex 40 .. 48

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm512_set1_ps(hh_dbl[i*2]);
		h1_imag = _mm512_set1_ps(hh_dbl[(i*2)+1]);

		q1 = _mm512_load_ps(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm512_load_ps(&q_dbl[(2*i*ldq)+16]);
		q3 = _mm512_load_ps(&q_dbl[(2*i*ldq)+32]);
		q4 = _mm512_load_ps(&q_dbl[(2*i*ldq)+48]);
		q5 = _mm512_load_ps(&q_dbl[(2*i*ldq)+64]);
		q6 = _mm512_load_ps(&q_dbl[(2*i*ldq)+80]);

		tmp1 = _mm512_mul_ps(h1_imag, q1);
		x1 = _mm512_add_ps(x1, _mm512_FMSUBADD_ps(h1_real, q1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

		tmp2 = _mm512_mul_ps(h1_imag, q2);

		x2 = _mm512_add_ps(x2, _mm512_FMSUBADD_ps(h1_real, q2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

		tmp3 = _mm512_mul_ps(h1_imag, q3);

		x3 = _mm512_add_ps(x3, _mm512_FMSUBADD_ps(h1_real, q3, _mm512_shuffle_ps(tmp3, tmp3, 0xb1)));

		tmp4 = _mm512_mul_ps(h1_imag, q4);

		x4 = _mm512_add_ps(x4, _mm512_FMSUBADD_ps(h1_real, q4, _mm512_shuffle_ps(tmp4, tmp4, 0xb1)));

		tmp5 = _mm512_mul_ps(h1_imag, q5);

		x5 = _mm512_add_ps(x5, _mm512_FMSUBADD_ps(h1_real, q5, _mm512_shuffle_ps(tmp5, tmp5, 0xb1)));

		tmp6 = _mm512_mul_ps(h1_imag, q6);

		x6 = _mm512_add_ps(x6, _mm512_FMSUBADD_ps(h1_real, q6, _mm512_shuffle_ps(tmp6, tmp6, 0xb1)));
	}

	h1_real = _mm512_set1_ps(hh_dbl[0]);
	h1_imag = _mm512_set1_ps(hh_dbl[1]);

//	h1_real = _mm512_xor_ps(h1_real, sign);
//	h1_imag = _mm512_xor_ps(h1_imag, sign);

        h1_real = (__m512) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__m512) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);

	tmp1 = _mm512_mul_ps(h1_imag, x1);

	x1 = _mm512_FMADDSUB_ps(h1_real, x1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1));

	tmp2 = _mm512_mul_ps(h1_imag, x2);

	x2 = _mm512_FMADDSUB_ps(h1_real, x2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1));

	tmp3 = _mm512_mul_ps(h1_imag, x3);

	x3 = _mm512_FMADDSUB_ps(h1_real, x3, _mm512_shuffle_ps(tmp3, tmp3, 0xb1));

	tmp4 = _mm512_mul_ps(h1_imag, x4);

	x4 = _mm512_FMADDSUB_ps(h1_real, x4, _mm512_shuffle_ps(tmp4, tmp4, 0xb1));

	tmp5 = _mm512_mul_ps(h1_imag, x5);

	x5 = _mm512_FMADDSUB_ps(h1_real, x5, _mm512_shuffle_ps(tmp5, tmp5, 0xb1));

	tmp6 = _mm512_mul_ps(h1_imag, x6);

	x6 = _mm512_FMADDSUB_ps(h1_real, x6, _mm512_shuffle_ps(tmp6, tmp6, 0xb1));

	q1 = _mm512_load_ps(&q_dbl[0]);
	q2 = _mm512_load_ps(&q_dbl[16]);
	q3 = _mm512_load_ps(&q_dbl[32]);
	q4 = _mm512_load_ps(&q_dbl[48]);
	q5 = _mm512_load_ps(&q_dbl[64]);
	q6 = _mm512_load_ps(&q_dbl[80]);

	q1 = _mm512_add_ps(q1, x1);
	q2 = _mm512_add_ps(q2, x2);
	q3 = _mm512_add_ps(q3, x3);
	q4 = _mm512_add_ps(q4, x4);
	q5 = _mm512_add_ps(q5, x5);
	q6 = _mm512_add_ps(q6, x6);

	_mm512_store_ps(&q_dbl[0], q1);
	_mm512_store_ps(&q_dbl[16], q2);
	_mm512_store_ps(&q_dbl[32], q3);
	_mm512_store_ps(&q_dbl[48], q4);
	_mm512_store_ps(&q_dbl[64], q5);
	_mm512_store_ps(&q_dbl[80], q6);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm512_set1_ps(hh_dbl[i*2]);
		h1_imag = _mm512_set1_ps(hh_dbl[(i*2)+1]);

		q1 = _mm512_load_ps(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm512_load_ps(&q_dbl[(2*i*ldq)+16]);
		q3 = _mm512_load_ps(&q_dbl[(2*i*ldq)+32]);
		q4 = _mm512_load_ps(&q_dbl[(2*i*ldq)+48]);
		q5 = _mm512_load_ps(&q_dbl[(2*i*ldq)+64]);
		q6 = _mm512_load_ps(&q_dbl[(2*i*ldq)+80]);

		tmp1 = _mm512_mul_ps(h1_imag, x1);

		q1 = _mm512_add_ps(q1, _mm512_FMADDSUB_ps(h1_real, x1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

		tmp2 = _mm512_mul_ps(h1_imag, x2);

		q2 = _mm512_add_ps(q2, _mm512_FMADDSUB_ps(h1_real, x2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

		tmp3 = _mm512_mul_ps(h1_imag, x3);

		q3 = _mm512_add_ps(q3, _mm512_FMADDSUB_ps(h1_real, x3, _mm512_shuffle_ps(tmp3, tmp3, 0xb1)));

		tmp4 = _mm512_mul_ps(h1_imag, x4);

		q4 = _mm512_add_ps(q4, _mm512_FMADDSUB_ps(h1_real, x4, _mm512_shuffle_ps(tmp4, tmp4, 0xb1)));

		tmp5 = _mm512_mul_ps(h1_imag, x5);

		q5 = _mm512_add_ps(q5, _mm512_FMADDSUB_ps(h1_real, x5, _mm512_shuffle_ps(tmp5, tmp5, 0xb1)));

		tmp6 = _mm512_mul_ps(h1_imag, x6);

		q6 = _mm512_add_ps(q6, _mm512_FMADDSUB_ps(h1_real, x6, _mm512_shuffle_ps(tmp6, tmp6, 0xb1)));

		_mm512_store_ps(&q_dbl[(2*i*ldq)+0], q1);
		_mm512_store_ps(&q_dbl[(2*i*ldq)+16], q2);
		_mm512_store_ps(&q_dbl[(2*i*ldq)+32], q3);
		_mm512_store_ps(&q_dbl[(2*i*ldq)+48], q4);
		_mm512_store_ps(&q_dbl[(2*i*ldq)+64], q5);
		_mm512_store_ps(&q_dbl[(2*i*ldq)+80], q6);
	}
}

static __forceinline void hh_trafo_complex_kernel_32_AVX512_1hv_single(float complex* q, float complex* hh, int nb, int ldq)
{
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;

	__m512 x1, x2, x3, x4;
	__m512 q1, q2, q3, q4;
	__m512 h1_real, h1_imag;
	__m512 tmp1, tmp2, tmp3, tmp4;
	int i=0;

	__m512 sign = (__m512)_mm512_set1_epi32(0x80000000);

	x1 = _mm512_load_ps(&q_dbl[0]);   // complex 1 2 3 4 5 6 7 8
	x2 = _mm512_load_ps(&q_dbl[16]);
	x3 = _mm512_load_ps(&q_dbl[32]);
	x4 = _mm512_load_ps(&q_dbl[48]);  // comlex 24 ..32

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm512_set1_ps(hh_dbl[i*2]);
		h1_imag = _mm512_set1_ps(hh_dbl[(i*2)+1]);

		q1 = _mm512_load_ps(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm512_load_ps(&q_dbl[(2*i*ldq)+16]);
		q3 = _mm512_load_ps(&q_dbl[(2*i*ldq)+32]);
		q4 = _mm512_load_ps(&q_dbl[(2*i*ldq)+48]);

		tmp1 = _mm512_mul_ps(h1_imag, q1);

		x1 = _mm512_add_ps(x1, _mm512_FMSUBADD_ps(h1_real, q1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

		tmp2 = _mm512_mul_ps(h1_imag, q2);

		x2 = _mm512_add_ps(x2, _mm512_FMSUBADD_ps(h1_real, q2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

		tmp3 = _mm512_mul_ps(h1_imag, q3);

		x3 = _mm512_add_ps(x3, _mm512_FMSUBADD_ps(h1_real, q3, _mm512_shuffle_ps(tmp3, tmp3, 0xb1)));

		tmp4 = _mm512_mul_ps(h1_imag, q4);

		x4 = _mm512_add_ps(x4, _mm512_FMSUBADD_ps(h1_real, q4, _mm512_shuffle_ps(tmp4, tmp4, 0xb1)));
	}

	h1_real = _mm512_set1_ps(hh_dbl[0]);
	h1_imag = _mm512_set1_ps(hh_dbl[1]);

//	h1_real = _mm512_xor_ps(h1_real, sign);
//	h1_imag = _mm512_xor_ps(h1_imag, sign);
        h1_real = (__m512) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__m512) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);

	tmp1 = _mm512_mul_ps(h1_imag, x1);

	x1 = _mm512_FMADDSUB_ps(h1_real, x1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1));

	tmp2 = _mm512_mul_ps(h1_imag, x2);

	x2 = _mm512_FMADDSUB_ps(h1_real, x2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1));

	tmp3 = _mm512_mul_ps(h1_imag, x3);

	x3 = _mm512_FMADDSUB_ps(h1_real, x3, _mm512_shuffle_ps(tmp3, tmp3, 0xb1));

	tmp4 = _mm512_mul_ps(h1_imag, x4);

	x4 = _mm512_FMADDSUB_ps(h1_real, x4, _mm512_shuffle_ps(tmp4, tmp4, 0xb1));

	q1 = _mm512_load_ps(&q_dbl[0]);
	q2 = _mm512_load_ps(&q_dbl[16]);
	q3 = _mm512_load_ps(&q_dbl[32]);
	q4 = _mm512_load_ps(&q_dbl[48]);

	q1 = _mm512_add_ps(q1, x1);
	q2 = _mm512_add_ps(q2, x2);
	q3 = _mm512_add_ps(q3, x3);
	q4 = _mm512_add_ps(q4, x4);

	_mm512_store_ps(&q_dbl[0], q1);
	_mm512_store_ps(&q_dbl[16], q2);
	_mm512_store_ps(&q_dbl[32], q3);
	_mm512_store_ps(&q_dbl[48], q4);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm512_set1_ps(hh_dbl[i*2]);
		h1_imag = _mm512_set1_ps(hh_dbl[(i*2)+1]);

		q1 = _mm512_load_ps(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm512_load_ps(&q_dbl[(2*i*ldq)+16]);
		q3 = _mm512_load_ps(&q_dbl[(2*i*ldq)+32]);
		q4 = _mm512_load_ps(&q_dbl[(2*i*ldq)+48]);

		tmp1 = _mm512_mul_ps(h1_imag, x1);

		q1 = _mm512_add_ps(q1, _mm512_FMADDSUB_ps(h1_real, x1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

		tmp2 = _mm512_mul_ps(h1_imag, x2);

		q2 = _mm512_add_ps(q2, _mm512_FMADDSUB_ps(h1_real, x2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

		tmp3 = _mm512_mul_ps(h1_imag, x3);

		q3 = _mm512_add_ps(q3, _mm512_FMADDSUB_ps(h1_real, x3, _mm512_shuffle_ps(tmp3, tmp3, 0xb1)));

		tmp4 = _mm512_mul_ps(h1_imag, x4);

		q4 = _mm512_add_ps(q4, _mm512_FMADDSUB_ps(h1_real, x4, _mm512_shuffle_ps(tmp4, tmp4, 0xb1)));

		_mm512_store_ps(&q_dbl[(2*i*ldq)+0], q1);
		_mm512_store_ps(&q_dbl[(2*i*ldq)+16], q2);
		_mm512_store_ps(&q_dbl[(2*i*ldq)+32], q3);
		_mm512_store_ps(&q_dbl[(2*i*ldq)+48], q4);
	}
}

static __forceinline void hh_trafo_complex_kernel_16_AVX512_1hv_single(float complex* q, float complex* hh, int nb, int ldq)
{
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;

	__m512 x1, x2;
	__m512 q1, q2;
	__m512 h1_real, h1_imag;
	__m512 tmp1, tmp2;
	int i=0;

	__m512 sign = (__m512)_mm512_set1_epi32(0x80000000);

	x1 = _mm512_load_ps(&q_dbl[0]);
	x2 = _mm512_load_ps(&q_dbl[16]);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm512_set1_ps(hh_dbl[i*2]);
		h1_imag = _mm512_set1_ps(hh_dbl[(i*2)+1]);

		q1 = _mm512_load_ps(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm512_load_ps(&q_dbl[(2*i*ldq)+16]);

		tmp1 = _mm512_mul_ps(h1_imag, q1);
		x1 = _mm512_add_ps(x1, _mm512_FMSUBADD_ps(h1_real, q1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));
		tmp2 = _mm512_mul_ps(h1_imag, q2);
		x2 = _mm512_add_ps(x2, _mm512_FMSUBADD_ps(h1_real, q2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));
	}

	h1_real = _mm512_set1_ps(hh_dbl[0]);
	h1_imag = _mm512_set1_ps(hh_dbl[1]);

//	h1_real = _mm512_xor_ps(h1_real, sign);
//	h1_imag = _mm512_xor_ps(h1_imag, sign);
	h1_real = (__m512) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
	h1_imag = (__m512) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);

	tmp1 = _mm512_mul_ps(h1_imag, x1);
	x1 = _mm512_FMADDSUB_ps(h1_real, x1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1));

	tmp2 = _mm512_mul_ps(h1_imag, x2);

	x2 = _mm512_FMADDSUB_ps(h1_real, x2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1));

	q1 = _mm512_load_ps(&q_dbl[0]);
	q2 = _mm512_load_ps(&q_dbl[16]);

	q1 = _mm512_add_ps(q1, x1);
	q2 = _mm512_add_ps(q2, x2);

	_mm512_store_ps(&q_dbl[0], q1);
	_mm512_store_ps(&q_dbl[16], q2);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm512_set1_ps(hh_dbl[i*2]);
		h1_imag = _mm512_set1_ps(hh_dbl[(i*2)+1]);

		q1 = _mm512_load_ps(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm512_load_ps(&q_dbl[(2*i*ldq)+16]);

		tmp1 = _mm512_mul_ps(h1_imag, x1);
		q1 = _mm512_add_ps(q1, _mm512_FMADDSUB_ps(h1_real, x1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

		tmp2 = _mm512_mul_ps(h1_imag, x2);

		q2 = _mm512_add_ps(q2, _mm512_FMADDSUB_ps(h1_real, x2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

		_mm512_store_ps(&q_dbl[(2*i*ldq)+0], q1);
		_mm512_store_ps(&q_dbl[(2*i*ldq)+16], q2);
	}
}
