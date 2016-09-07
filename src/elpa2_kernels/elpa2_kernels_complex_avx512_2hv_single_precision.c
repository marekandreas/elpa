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

#include <complex.h>
#include <x86intrin.h>

#define __forceinline __attribute__((always_inline))

#ifdef HAVE_AVX512

#define __ELPA_USE_FMA__
#define _mm512_FMADDSUB_ps(a,b,c) _mm512_fmaddsub_ps(a,b,c)
#define _mm512_FMSUBADD_ps(a,b,c) _mm512_fmsubadd_ps(a,b,c)
#endif


//Forward declaration

static __forceinline void hh_trafo_complex_kernel_16_AVX512_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s);
static __forceinline void hh_trafo_complex_kernel_8_AVX512_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s);
//static __forceinline void hh_trafo_complex_kernel_6_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s);
//static __forceinline void hh_trafo_complex_kernel_4_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s);
//static __forceinline void hh_trafo_complex_kernel_2_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s);

/*
!f>#if defined(HAVE_AVX512)
!f> interface
!f>   subroutine float_hh_trafo_complex_avx512_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="float_hh_trafo_complex_avx_avx2_2hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     complex(kind=c_single)     :: q(*)
!f>     complex(kind=c_single)     :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

void float_hh_trafo_complex_avx512_2hv_single(float complex* q, float complex* hh, int* pnb, int* pnq, int* pldq, int* pldh)
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;

	float complex s = conj(hh[(ldh)+1])*1.0;
	for (i = 2; i < nb; i++)
	{
		s += hh[i-1] * conj(hh[(i+ldh)]);
	}

	for (i = 0; i < nq-16; i+=32)
	{
		hh_trafo_complex_kernel_32_AVX512_2hv_single(&q[i], hh, nb, ldq, ldh, s);
	}
	if (nq == i)
	{
		return;
	}
	if (nq-i > 0)
	{
		hh_trafo_complex_kernel_16_AVX512_2hv_single(&q[i], hh, nb, ldq, ldh, s);
	}
}

static __forceinline void hh_trafo_complex_kernel_32_AVX512_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s)
{
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;
	float* s_dbl = (float*)(&s);

	__m512 x1, x2, x3, x4;
	__m512 y1, y2, y3, y4;
	__m512 q1, q2, q3, q4;
	__m512 h1_real, h1_imag, h2_real, h2_imag;
	__m512 tmp1, tmp2, tmp3, tmp4;
	int i=0;

	__m512 sign = (__m512)_mm512_set1_epi32(0x80000000);

	x1 = _mm512_load_ps(&q_dbl[(2*ldq)+0]);  // complex 1, 2, 3, 4, 5, 6, 7, 8
	x2 = _mm512_load_ps(&q_dbl[(2*ldq)+16]); // complex 9 .. 16
	x3 = _mm512_load_ps(&q_dbl[(2*ldq)+32]);
	x4 = _mm512_load_ps(&q_dbl[(2*ldq)+48]); // complex 25 .. 32

	h2_real = _mm512_set1_ps(hh_dbl[(ldh+1)*2]);
	h2_imag = _mm512_set1_ps(hh_dbl[((ldh+1)*2)+1]);

	y1 = _mm512_load_ps(&q_dbl[0]);
	y2 = _mm512_load_ps(&q_dbl[16]);
	y3 = _mm512_load_ps(&q_dbl[32]);
	y4 = _mm512_load_ps(&q_dbl[48]);

	tmp1 = _mm512_mul_ps(h2_imag, x1);

	y1 = _mm512_add_ps(y1, _mm512_FMSUBADD_ps(h2_real, x1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

	tmp2 = _mm512_mul_ps(h2_imag, x2);

	y2 = _mm512_add_ps(y2, _mm512_FMSUBADD_ps(h2_real, x2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

	tmp3 = _mm512_mul_ps(h2_imag, x3);

	y3 = _mm512_add_ps(y3, _mm512_FMSUBADD_ps(h2_real, x3, _mm512_shuffle_ps(tmp3, tmp3, 0xb1)));

	tmp4 = _mm512_mul_ps(h2_imag, x4);

	y4 = _mm512_add_ps(y4, _mm512_FMSUBADD_ps(h2_real, x4, _mm512_shuffle_ps(tmp4, tmp4, 0xb1)));

	for (i = 2; i < nb; i++)
	{
		q1 = _mm512_load_ps(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm512_load_ps(&q_dbl[(2*i*ldq)+16]);
		q3 = _mm512_load_ps(&q_dbl[(2*i*ldq)+32]);
		q4 = _mm512_load_ps(&q_dbl[(2*i*ldq)+48]);

		h1_real = _mm512_set1_ps(hh_dbl[(i-1)*2]);
		h1_imag = _mm512_set1_ps(hh_dbl[((i-1)*2)+1]);

		tmp1 = _mm512_mul_ps(h1_imag, q1);

		x1 = _mm512_add_ps(x1, _mm512_FMSUBADD_ps(h1_real, q1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

		tmp2 = _mm512_mul_ps(h1_imag, q2);

		x2 = _mm512_add_ps(x2, _mm512_FMSUBADD_ps(h1_real, q2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

		tmp3 = _mm512_mul_ps(h1_imag, q3);

		x3 = _mm512_add_ps(x3, _mm512_FMSUBADD_ps(h1_real, q3, _mm512_shuffle_ps(tmp3, tmp3, 0xb1)));

		tmp4 = _mm512_mul_ps(h1_imag, q4);

		x4 = _mm512_add_ps(x4, _mm512_FMSUBADD_ps(h1_real, q4, _mm512_shuffle_ps(tmp4, tmp4, 0xb1)));

		h2_real = _mm512_set1_ps(hh_dbl[(ldh+i)*2]);
		h2_imag = _mm512_set1_ps(hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _mm512_mul_ps(h2_imag, q1);

		y1 = _mm512_add_ps(y1, _mm512_FMSUBADD_ps(h2_real, q1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

		tmp2 = _mm512_mul_ps(h2_imag, q2);

		y2 = _mm512_add_ps(y2, _mm512_FMSUBADD_ps(h2_real, q2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

		tmp3 = _mm512_mul_ps(h2_imag, q3);

		y3 = _mm512_add_ps(y3, _mm512_FMSUBADD_ps(h2_real, q3, _mm512_shuffle_ps(tmp3, tmp3, 0xb1)));

		tmp4 = _mm512_mul_ps(h2_imag, q4);

		y4 = _mm512_add_ps(y4, _mm512_FMSUBADD_ps(h2_real, q4, _mm512_shuffle_ps(tmp4, tmp4, 0xb1)));
	}

	h1_real = _mm512_set1_ps(hh_dbl[(nb-1)*2]);
	h1_imag = _mm512_set1_ps(hh_dbl[((nb-1)*2)+1]);

	q1 = _mm512_load_ps(&q_dbl[(2*nb*ldq)+0]);
	q2 = _mm512_load_ps(&q_dbl[(2*nb*ldq)+16]);
	q3 = _mm512_load_ps(&q_dbl[(2*nb*ldq)+32]);
	q4 = _mm512_load_ps(&q_dbl[(2*nb*ldq)+48]);

	tmp1 = _mm512_mul_ps(h1_imag, q1);

	x1 = _mm512_add_ps(x1, _mm512_FMSUBADD_ps(h1_real, q1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

	tmp2 = _mm512_mul_ps(h1_imag, q2);

	x2 = _mm512_add_ps(x2, _mm512_FMSUBADD_ps(h1_real, q2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

	tmp3 = _mm512_mul_ps(h1_imag, q3);

	x3 = _mm512_add_ps(x3, _mm512_FMSUBADD_ps(h1_real, q3, _mm512_shuffle_ps(tmp3, tmp3, 0xb1)));

	tmp4 = _mm512_mul_ps(h1_imag, q4);

	x4 = _mm512_add_ps(x4, _mm512_FMSUBADD_ps(h1_real, q4, _mm512_shuffle_ps(tmp4, tmp4, 0xb1)));

	h1_real = _mm512_set1_ps(hh_dbl[0]);
	h1_imag = _mm512_set1_ps(hh_dbl[1]);

//	h1_real = _mm256_xor_ps(h1_real, sign);
//	h1_imag = _mm256_xor_ps(h1_imag, sign);
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

	h1_real = _mm512_set1_ps(hh_dbl[ldh*2]);
	h1_imag = _mm512_set1_ps(hh_dbl[(ldh*2)+1]);
	h2_real = _mm512_set1_ps(hh_dbl[ldh*2]);
	h2_imag = _mm512_set1_ps(hh_dbl[(ldh*2)+1]);

//	h1_real = _mm256_xor_ps(h1_real, sign);
//	h1_imag = _mm256_xor_ps(h1_imag, sign);
        h1_real = (__m512) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__m512) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);

//	h2_real = _mm256_xor_ps(h2_real, sign);
//	h2_imag = _mm256_xor_ps(h2_imag, sign);
        h2_real = (__m512) _mm512_xor_epi32((__m512i) h2_real, (__m512i) sign);
        h2_imag = (__m512) _mm512_xor_epi32((__m512i) h2_imag, (__m512i) sign);

	//check this
//	__m128d tmp_s_128 = _mm_loadu_ps(s_dbl);
//	tmp2 = _mm256_broadcast_ps(&tmp_s_128);

//	__m512d tmp_s = _mm512_maskz_loadu_ps (0x01 + 0x02, s_dbl);
//        tmp2 = _mm512_broadcast_f64x2(_mm512_castpd512_ps128(tmp_s));
        tmp2 = _mm512_set4_ps(s_dbl[0],s_dbl[1], s_dbl[0],s_dbl[1]);
	tmp1 = _mm512_mul_ps(h2_imag, tmp2);

	tmp2 = _mm512_FMADDSUB_ps(h2_real, tmp2, _mm512_shuffle_ps(tmp1, tmp1, 0xb1));
//check this
//	_mm_storeu_ps(s_dbl, _mm256_castpd256_ps128(tmp2));
        _mm512_mask_storeu_ps(s_dbl, 0x01 + 0x02, tmp2);

	h2_real = _mm512_set1_ps(s_dbl[0]);
	h2_imag = _mm512_set1_ps(s_dbl[1]);

	tmp1 = _mm512_mul_ps(h1_imag, y1);

	y1 = _mm512_FMADDSUB_ps(h1_real, y1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1));

	tmp2 = _mm512_mul_ps(h1_imag, y2);

	y2 = _mm512_FMADDSUB_ps(h1_real, y2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1));

	tmp3 = _mm512_mul_ps(h1_imag, y3);

	y3 = _mm512_FMADDSUB_ps(h1_real, y3, _mm512_shuffle_ps(tmp3, tmp3, 0xb1));

	tmp4 = _mm512_mul_ps(h1_imag, y4);

	y4 = _mm512_FMADDSUB_ps(h1_real, y4, _mm512_shuffle_ps(tmp4, tmp4, 0xb1));

	tmp1 = _mm512_mul_ps(h2_imag, x1);

	y1 = _mm512_add_ps(y1, _mm512_FMADDSUB_ps(h2_real, x1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

	tmp2 = _mm512_mul_ps(h2_imag, x2);

	y2 = _mm512_add_ps(y2, _mm512_FMADDSUB_ps(h2_real, x2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

	tmp3 = _mm512_mul_ps(h2_imag, x3);

	y3 = _mm512_add_ps(y3, _mm512_FMADDSUB_ps(h2_real, x3, _mm512_shuffle_ps(tmp3, tmp3, 0xb1)));

	tmp4 = _mm512_mul_ps(h2_imag, x4);

	y4 = _mm512_add_ps(y4, _mm512_FMADDSUB_ps(h2_real, x4, _mm512_shuffle_ps(tmp4, tmp4, 0xb1)));

	q1 = _mm512_load_ps(&q_dbl[0]);
	q2 = _mm512_load_ps(&q_dbl[16]);
	q3 = _mm512_load_ps(&q_dbl[32]);
	q4 = _mm512_load_ps(&q_dbl[48]);

	q1 = _mm512_add_ps(q1, y1);
	q2 = _mm512_add_ps(q2, y2);
	q3 = _mm512_add_ps(q3, y3);
	q4 = _mm512_add_ps(q4, y4);

	_mm512_store_ps(&q_dbl[0], q1);
	_mm512_store_ps(&q_dbl[16], q2);
	_mm512_store_ps(&q_dbl[32], q3);
	_mm512_store_ps(&q_dbl[48], q4);

	h2_real = _mm512_set1_ps(hh_dbl[(ldh+1)*2]);
	h2_imag = _mm512_set1_ps(hh_dbl[((ldh+1)*2)+1]);

	q1 = _mm512_load_ps(&q_dbl[(ldq*2)+0]);
	q2 = _mm512_load_ps(&q_dbl[(ldq*2)+16]);
	q3 = _mm512_load_ps(&q_dbl[(ldq*2)+32]);
	q4 = _mm512_load_ps(&q_dbl[(ldq*2)+48]);

	q1 = _mm512_add_ps(q1, x1);
	q2 = _mm512_add_ps(q2, x2);
	q3 = _mm512_add_ps(q3, x3);
	q4 = _mm512_add_ps(q4, x4);

	tmp1 = _mm512_mul_ps(h2_imag, y1);

	q1 = _mm512_add_ps(q1, _mm512_FMADDSUB_ps(h2_real, y1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

	tmp2 = _mm512_mul_ps(h2_imag, y2);

	q2 = _mm512_add_ps(q2, _mm512_FMADDSUB_ps(h2_real, y2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

	tmp3 = _mm512_mul_ps(h2_imag, y3);

	q3 = _mm512_add_ps(q3, _mm512_FMADDSUB_ps(h2_real, y3, _mm512_shuffle_ps(tmp3, tmp3, 0xb1)));

	tmp4 = _mm512_mul_ps(h2_imag, y4);

	q4 = _mm512_add_ps(q4, _mm512_FMADDSUB_ps(h2_real, y4, _mm512_shuffle_ps(tmp4, tmp4, 0xb1)));

	_mm512_store_ps(&q_dbl[(ldq*2)+0], q1);
	_mm512_store_ps(&q_dbl[(ldq*2)+16], q2);
	_mm512_store_ps(&q_dbl[(ldq*2)+32], q3);
	_mm512_store_ps(&q_dbl[(ldq*2)+48], q4);

	for (i = 2; i < nb; i++)
	{
		q1 = _mm512_load_ps(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm512_load_ps(&q_dbl[(2*i*ldq)+16]);
		q3 = _mm512_load_ps(&q_dbl[(2*i*ldq)+32]);
		q4 = _mm512_load_ps(&q_dbl[(2*i*ldq)+48]);

		h1_real = _mm512_set1_ps(hh_dbl[(i-1)*2]);
		h1_imag = _mm512_set1_ps(hh_dbl[((i-1)*2)+1]);

		tmp1 = _mm512_mul_ps(h1_imag, x1);

		q1 = _mm512_add_ps(q1, _mm512_FMADDSUB_ps(h1_real, x1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

		tmp2 = _mm512_mul_ps(h1_imag, x2);

		q2 = _mm512_add_ps(q2, _mm512_FMADDSUB_ps(h1_real, x2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

		tmp3 = _mm512_mul_ps(h1_imag, x3);

		q3 = _mm512_add_ps(q3, _mm512_FMADDSUB_ps(h1_real, x3, _mm512_shuffle_ps(tmp3, tmp3, 0xb1)));

		tmp4 = _mm512_mul_ps(h1_imag, x4);

		q4 = _mm512_add_ps(q4, _mm512_FMADDSUB_ps(h1_real, x4, _mm512_shuffle_ps(tmp4, tmp4, 0xb1)));

		h2_real = _mm512_set1_ps(hh_dbl[(ldh+i)*2]);
		h2_imag = _mm512_set1_ps(hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _mm512_mul_ps(h2_imag, y1);

		q1 = _mm512_add_ps(q1, _mm512_FMADDSUB_ps(h2_real, y1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

		tmp2 = _mm512_mul_ps(h2_imag, y2);

		q2 = _mm512_add_ps(q2, _mm512_FMADDSUB_ps(h2_real, y2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

		tmp3 = _mm512_mul_ps(h2_imag, y3);

		q3 = _mm512_add_ps(q3, _mm512_FMADDSUB_ps(h2_real, y3, _mm512_shuffle_ps(tmp3, tmp3, 0xb1)));

		tmp4 = _mm512_mul_ps(h2_imag, y4);

		q4 = _mm512_add_ps(q4, _mm512_FMADDSUB_ps(h2_real, y4, _mm512_shuffle_ps(tmp4, tmp4, 0xb1)));

		_mm512_store_ps(&q_dbl[(2*i*ldq)+0], q1);
		_mm512_store_ps(&q_dbl[(2*i*ldq)+16], q2);
		_mm512_store_ps(&q_dbl[(2*i*ldq)+32], q3);
		_mm512_store_ps(&q_dbl[(2*i*ldq)+48], q4);
	}

	h1_real = _mm512_set1_ps(hh_dbl[(nb-1)*2]);
	h1_imag = _mm512_set1_ps(hh_dbl[((nb-1)*2)+1]);

	q1 = _mm512_load_ps(&q_dbl[(2*nb*ldq)+0]);
	q2 = _mm512_load_ps(&q_dbl[(2*nb*ldq)+16]);
	q3 = _mm512_load_ps(&q_dbl[(2*nb*ldq)+32]);
	q4 = _mm512_load_ps(&q_dbl[(2*nb*ldq)+48]);

	tmp1 = _mm512_mul_ps(h1_imag, x1);

	q1 = _mm512_add_ps(q1, _mm512_FMADDSUB_ps(h1_real, x1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

	tmp2 = _mm512_mul_ps(h1_imag, x2);

	q2 = _mm512_add_ps(q2, _mm512_FMADDSUB_ps(h1_real, x2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

	tmp3 = _mm512_mul_ps(h1_imag, x3);

	q3 = _mm512_add_ps(q3, _mm512_FMADDSUB_ps(h1_real, x3, _mm512_shuffle_ps(tmp3, tmp3, 0xb1)));

	tmp4 = _mm512_mul_ps(h1_imag, x4);

	q4 = _mm512_add_ps(q4, _mm512_FMADDSUB_ps(h1_real, x4, _mm512_shuffle_ps(tmp4, tmp4, 0xb1)));

	_mm512_store_ps(&q_dbl[(2*nb*ldq)+0], q1);
	_mm512_store_ps(&q_dbl[(2*nb*ldq)+16], q2);
	_mm512_store_ps(&q_dbl[(2*nb*ldq)+32], q3);
	_mm512_store_ps(&q_dbl[(2*nb*ldq)+48], q4);
}

static __forceinline void hh_trafo_complex_kernel_16_AVX512_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s)
{
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;
	float* s_dbl = (float*)(&s);

	__m512 x1, x2;
	__m512 y1, y2;
	__m512 q1, q2;
	__m512 h1_real, h1_imag, h2_real, h2_imag;
	__m512 tmp1, tmp2;
	int i=0;

	__m512 sign = (__m512)_mm512_set1_epi32(0x80000000);

	x1 = _mm512_load_ps(&q_dbl[(2*ldq)+0]);
	x2 = _mm512_load_ps(&q_dbl[(2*ldq)+16]);

	h2_real = _mm512_set1_ps(hh_dbl[(ldh+1)*2]);
	h2_imag = _mm512_set1_ps(hh_dbl[((ldh+1)*2)+1]);

	y1 = _mm512_load_ps(&q_dbl[0]);
	y2 = _mm512_load_ps(&q_dbl[16]);

	tmp1 = _mm512_mul_ps(h2_imag, x1);

	y1 = _mm512_add_ps(y1, _mm512_FMSUBADD_ps(h2_real, x1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

	tmp2 = _mm512_mul_ps(h2_imag, x2);

	y2 = _mm512_add_ps(y2, _mm512_FMSUBADD_ps(h2_real, x2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

	for (i = 2; i < nb; i++)
	{
		q1 = _mm512_load_ps(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm512_load_ps(&q_dbl[(2*i*ldq)+16]);

		h1_real = _mm512_set1_ps(hh_dbl[(i-1)*2]);
		h1_imag = _mm512_set1_ps(hh_dbl[((i-1)*2)+1]);

		tmp1 = _mm512_mul_ps(h1_imag, q1);

		x1 = _mm512_add_ps(x1, _mm512_FMSUBADD_ps(h1_real, q1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

		tmp2 = _mm512_mul_ps(h1_imag, q2);

		x2 = _mm512_add_ps(x2, _mm512_FMSUBADD_ps(h1_real, q2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

		h2_real = _mm512_set1_ps(hh_dbl[(ldh+i)*2]);
		h2_imag = _mm512_set1_ps(hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _mm512_mul_ps(h2_imag, q1);

		y1 = _mm512_add_ps(y1, _mm512_FMSUBADD_ps(h2_real, q1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

		tmp2 = _mm512_mul_ps(h2_imag, q2);

		y2 = _mm512_add_ps(y2, _mm512_FMSUBADD_ps(h2_real, q2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));
	}

	h1_real = _mm512_set1_ps(hh_dbl[(nb-1)*2]);
	h1_imag = _mm512_set1_ps(hh_dbl[((nb-1)*2)+1]);

	q1 = _mm512_load_ps(&q_dbl[(2*nb*ldq)+0]);
	q2 = _mm512_load_ps(&q_dbl[(2*nb*ldq)+16]);

	tmp1 = _mm512_mul_ps(h1_imag, q1);

	x1 = _mm512_add_ps(x1, _mm512_FMSUBADD_ps(h1_real, q1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

	tmp2 = _mm512_mul_ps(h1_imag, q2);

	x2 = _mm512_add_ps(x2, _mm512_FMSUBADD_ps(h1_real, q2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

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

	h1_real = _mm512_set1_ps(hh_dbl[ldh*2]);
	h1_imag = _mm512_set1_ps(hh_dbl[(ldh*2)+1]);
	h2_real = _mm512_set1_ps(hh_dbl[ldh*2]);
	h2_imag = _mm512_set1_ps(hh_dbl[(ldh*2)+1]);

//	h1_real = _mm512_xor_ps(h1_real, sign);
//	h1_imag = _mm512_xor_ps(h1_imag, sign);
        h1_real = (__m512) _mm512_xor_epi32((__m512i) h1_real, (__m512i) sign);
        h1_imag = (__m512) _mm512_xor_epi32((__m512i) h1_imag, (__m512i) sign);

//	h2_real = _mm512_xor_ps(h2_real, sign);
//	h2_imag = _mm512_xor_ps(h2_imag, sign);
        h2_real = (__m512) _mm512_xor_epi32((__m512i) h2_real, (__m512i) sign);
        h2_imag = (__m512) _mm512_xor_epi32((__m512i) h2_imag, (__m512i) sign);

//check this
//	__m128d tmp_s_128 = _mm_loadu_ps(s_dbl);
//	tmp2 = _mm512_broadcast_ps(&tmp_s_128);

//	__m512d tmp_s = _mm512_maskz_loadu_ps (0x01 + 0x02, s_dbl);
//       tmp2 = _mm512_broadcast_f64x2(_mm512_castpd512_ps128(tmp_s));
        tmp2 = _mm512_set4_ps(s_dbl[0],s_dbl[1], s_dbl[0],s_dbl[1]);

	tmp1 = _mm512_mul_ps(h2_imag, tmp2);

	tmp2 = _mm512_FMADDSUB_ps(h2_real, tmp2, _mm512_shuffle_ps(tmp1, tmp1, 0xb1));
//check this
//	_mm_storeu_ps(s_dbl, _mm512_castpd512_ps128(tmp2));
        _mm512_mask_storeu_ps(s_dbl, 0x01 + 0x02, tmp2);

	h2_real = _mm512_set1_ps(s_dbl[0]);
	h2_imag = _mm512_set1_ps(s_dbl[1]);

	tmp1 = _mm512_mul_ps(h1_imag, y1);

	y1 = _mm512_FMADDSUB_ps(h1_real, y1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1));

	tmp2 = _mm512_mul_ps(h1_imag, y2);

	y2 = _mm512_FMADDSUB_ps(h1_real, y2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1));

	tmp1 = _mm512_mul_ps(h2_imag, x1);

	y1 = _mm512_add_ps(y1, _mm512_FMADDSUB_ps(h2_real, x1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

	tmp2 = _mm512_mul_ps(h2_imag, x2);

	y2 = _mm512_add_ps(y2, _mm512_FMADDSUB_ps(h2_real, x2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

	q1 = _mm512_load_ps(&q_dbl[0]);
	q2 = _mm512_load_ps(&q_dbl[16]);

	q1 = _mm512_add_ps(q1, y1);
	q2 = _mm512_add_ps(q2, y2);

	_mm512_store_ps(&q_dbl[0], q1);
	_mm512_store_ps(&q_dbl[16], q2);

	h2_real = _mm512_set1_ps(hh_dbl[(ldh+1)*2]);
	h2_imag = _mm512_set1_ps(hh_dbl[((ldh+1)*2)+1]);

	q1 = _mm512_load_ps(&q_dbl[(ldq*2)+0]);
	q2 = _mm512_load_ps(&q_dbl[(ldq*2)+16]);

	q1 = _mm512_add_ps(q1, x1);
	q2 = _mm512_add_ps(q2, x2);

	tmp1 = _mm512_mul_ps(h2_imag, y1);

	q1 = _mm512_add_ps(q1, _mm512_FMADDSUB_ps(h2_real, y1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

	tmp2 = _mm512_mul_ps(h2_imag, y2);

	q2 = _mm512_add_ps(q2, _mm512_FMADDSUB_ps(h2_real, y2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

	_mm512_store_ps(&q_dbl[(ldq*2)+0], q1);
	_mm512_store_ps(&q_dbl[(ldq*2)+16], q2);

	for (i = 2; i < nb; i++)
	{
		q1 = _mm512_load_ps(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm512_load_ps(&q_dbl[(2*i*ldq)+16]);

		h1_real = _mm512_set1_ps(hh_dbl[(i-1)*2]);
		h1_imag = _mm512_set1_ps(hh_dbl[((i-1)*2)+1]);

		tmp1 = _mm512_mul_ps(h1_imag, x1);

		q1 = _mm512_add_ps(q1, _mm512_FMADDSUB_ps(h1_real, x1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

		tmp2 = _mm512_mul_ps(h1_imag, x2);

		q2 = _mm512_add_ps(q2, _mm512_FMADDSUB_ps(h1_real, x2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

		h2_real = _mm512_set1_ps(hh_dbl[(ldh+i)*2]);
		h2_imag = _mm512_set1_ps(hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _mm512_mul_ps(h2_imag, y1);

		q1 = _mm512_add_ps(q1, _mm512_FMADDSUB_ps(h2_real, y1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

		tmp2 = _mm512_mul_ps(h2_imag, y2);

		q2 = _mm512_add_ps(q2, _mm512_FMADDSUB_ps(h2_real, y2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

		_mm512_store_ps(&q_dbl[(2*i*ldq)+0], q1);
		_mm512_store_ps(&q_dbl[(2*i*ldq)+16], q2);
	}
	h1_real = _mm512_set1_ps(hh_dbl[(nb-1)*2]);
	h1_imag = _mm512_set1_ps(hh_dbl[((nb-1)*2)+1]);

	q1 = _mm512_load_ps(&q_dbl[(2*nb*ldq)+0]);
	q2 = _mm512_load_ps(&q_dbl[(2*nb*ldq)+16]);

	tmp1 = _mm512_mul_ps(h1_imag, x1);

	q1 = _mm512_add_ps(q1, _mm512_FMADDSUB_ps(h1_real, x1, _mm512_shuffle_ps(tmp1, tmp1, 0xb1)));

	tmp2 = _mm512_mul_ps(h1_imag, x2);

	q2 = _mm512_add_ps(q2, _mm512_FMADDSUB_ps(h1_real, x2, _mm512_shuffle_ps(tmp2, tmp2, 0xb1)));

	_mm512_store_ps(&q_dbl[(2*nb*ldq)+0], q1);
	_mm512_store_ps(&q_dbl[(2*nb*ldq)+16], q2);
}

