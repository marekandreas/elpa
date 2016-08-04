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

#include <complex.h>
#include <x86intrin.h>

#define __forceinline __attribute__((always_inline))

#ifdef HAVE_AVX2

#ifdef __FMA4__
#define __ELPA_USE_FMA__
#define _mm256_FMADDSUB_ps(a,b,c) _mm256_maddsub_ps(a,b,c)
#define _mm256_FMSUBADD_ps(a,b,c) _mm256_msubadd_ps(a,b,c)
#endif

#ifdef __AVX2__
#define __ELPA_USE_FMA__
#define _mm256_FMADDSUB_ps(a,b,c) _mm256_fmaddsub_ps(a,b,c)
#define _mm256_FMSUBADD_ps(a,b,c) _mm256_fmsubadd_ps(a,b,c)
#endif

#endif

//Forward declaration
static __forceinline void hh_trafo_complex_kernel_8_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s, float complex s1);
//static __forceinline void hh_trafo_complex_kernel_6_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s, float complex s1);
static __forceinline void hh_trafo_complex_kernel_4_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s, float complex s1);
//static __forceinline void hh_trafo_complex_kernel_2_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s, float complex s1);

/*
!f>#if defined(HAVE_AVX) || defined(HAVE_AVX2)
!f> interface
!f>   subroutine double_hh_trafo_complex_avx_avx2_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>                             bind(C, name="double_hh_trafo_complex_avx_avx2_2hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq, pldh
!f>     complex(kind=c_float)   :: q(*)
!f>     complex(kind=c_float)   :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/


void double_hh_trafo_complex_avx_avx2_2hv_single(float complex* q, float complex* hh, int* pnb, int* pnq, int* pldq, int* pldh)
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;

	float complex s = conj(hh[(ldh)+1])*1.0f;
	for (i = 2; i < nb; i++)
	{
		s += hh[i-1] * conj(hh[(i+ldh)]);
	}

	for (i = 0; i < nq-4; i+=8)
	{
		hh_trafo_complex_kernel_8_AVX_2hv_single(&q[i], hh, nb, ldq, ldh, s , s);
	}
	if (nq-i > 0)
	{
		hh_trafo_complex_kernel_4_AVX_2hv_single(&q[i], hh, nb, ldq, ldh, s, s);
	}

}

static __forceinline void hh_trafo_complex_kernel_8_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s, float complex s1)
{
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;
	float* s_dbl = (float*)(&s);

	__m256 x1, x2, x3, x4;
	__m256 y1, y2, y3, y4;
	__m256 q1, q2, q3, q4;
	__m256 h1_real, h1_imag, h2_real, h2_imag;
	__m256 tmp1, tmp2, tmp3, tmp4;
	volatile int i=0;
	__m256 sign = (__m256)_mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);

	x1 = _mm256_load_ps(&q_dbl[(2*ldq)+0]);
	x2 = _mm256_load_ps(&q_dbl[(2*ldq)+8]);
//	x3 = _mm256_load_ps(&q_dbl[(2*ldq)+8]);
//	x4 = _mm256_load_ps(&q_dbl[(2*ldq)+12]);

	h2_real = _mm256_broadcast_ss(&hh_dbl[(ldh+1)*2]);
	h2_imag = _mm256_broadcast_ss(&hh_dbl[((ldh+1)*2)+1]);
#ifndef __ELPA_USE_FMA__
	// conjugate
	h2_imag = _mm256_xor_ps(h2_imag, sign);
#endif

	y1 = _mm256_load_ps(&q_dbl[0]);
	y2 = _mm256_load_ps(&q_dbl[8]);
//	y3 = _mm256_load_pd(&q_dbl[8]);
//	y4 = _mm256_load_pd(&q_dbl[12]);

	tmp1 = _mm256_mul_ps(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_add_ps(y1, _mm256_FMSUBADD_ps(h2_real, x1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
	y1 = _mm256_add_ps(y1, _mm256_addsub_ps( _mm256_mul_ps(h2_real, x1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
	tmp2 = _mm256_mul_ps(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
	y2 = _mm256_add_ps(y2, _mm256_FMSUBADD_ps(h2_real, x2, _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#else
	y2 = _mm256_add_ps(y2, _mm256_addsub_ps( _mm256_mul_ps(h2_real, x2), _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#endif
//	tmp3 = _mm256_mul_pd(h2_imag, x3);
#ifdef __ELPA_USE_FMA__
//	y3 = _mm256_add_pd(y3, _mm256_FMSUBADD_pd(h2_real, x3, _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#else
//	y3 = _mm256_add_pd(y3, _mm256_addsub_pd( _mm256_mul_pd(h2_real, x3), _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#endif
//	tmp4 = _mm256_mul_pd(h2_imag, x4);
#ifdef __ELPA_USE_FMA__
//	y4 = _mm256_add_pd(y4, _mm256_FMSUBADD_pd(h2_real, x4, _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#else
//	y4 = _mm256_add_pd(y4, _mm256_addsub_pd( _mm256_mul_pd(h2_real, x4), _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#endif

	for (i = 2; i < nb; i++)
	{
		q1 = _mm256_load_ps(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm256_load_ps(&q_dbl[(2*i*ldq)+8]);
//		q3 = _mm256_load_pd(&q_dbl[(2*i*ldq)+8]);
//		q4 = _mm256_load_pd(&q_dbl[(2*i*ldq)+12]);

		h1_real = _mm256_broadcast_ss(&hh_dbl[(i-1)*2]);
		h1_imag = _mm256_broadcast_ss(&hh_dbl[((i-1)*2)+1]);
#ifndef __ELPA_USE_FMA__
		// conjugate
		h1_imag = _mm256_xor_ps(h1_imag, sign);
#endif

		tmp1 = _mm256_mul_ps(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
		x1 = _mm256_add_ps(x1, _mm256_FMSUBADD_ps(h1_real, q1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
		x1 = _mm256_add_ps(x1, _mm256_addsub_ps( _mm256_mul_ps(h1_real, q1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
		tmp2 = _mm256_mul_ps(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
		x2 = _mm256_add_ps(x2, _mm256_FMSUBADD_ps(h1_real, q2, _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#else
		x2 = _mm256_add_ps(x2, _mm256_addsub_ps( _mm256_mul_ps(h1_real, q2), _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#endif
//		tmp3 = _mm256_mul_pd(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
//		x3 = _mm256_add_pd(x3, _mm256_FMSUBADD_pd(h1_real, q3, _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#else
//		x3 = _mm256_add_pd(x3, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q3), _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#endif
//		tmp4 = _mm256_mul_pd(h1_imag, q4);
#ifdef __ELPA_USE_FMA__
//		x4 = _mm256_add_pd(x4, _mm256_FMSUBADD_pd(h1_real, q4, _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#else
//		x4 = _mm256_add_pd(x4, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q4), _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#endif

		h2_real = _mm256_broadcast_ss(&hh_dbl[(ldh+i)*2]);
		h2_imag = _mm256_broadcast_ss(&hh_dbl[((ldh+i)*2)+1]);
#ifndef __ELPA_USE_FMA__
		// conjugate
		h2_imag = _mm256_xor_ps(h2_imag, sign);
#endif

		tmp1 = _mm256_mul_ps(h2_imag, q1);
#ifdef __ELPA_USE_FMA__
		y1 = _mm256_add_ps(y1, _mm256_FMSUBADD_ps(h2_real, q1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
		y1 = _mm256_add_ps(y1, _mm256_addsub_ps( _mm256_mul_ps(h2_real, q1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
		tmp2 = _mm256_mul_ps(h2_imag, q2);
#ifdef __ELPA_USE_FMA__
		y2 = _mm256_add_ps(y2, _mm256_FMSUBADD_ps(h2_real, q2, _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#else
		y2 = _mm256_add_ps(y2, _mm256_addsub_ps( _mm256_mul_ps(h2_real, q2), _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#endif
//		tmp3 = _mm256_mul_pd(h2_imag, q3);
#ifdef __ELPA_USE_FMA__
//		y3 = _mm256_add_pd(y3, _mm256_FMSUBADD_pd(h2_real, q3, _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#else
//		y3 = _mm256_add_pd(y3, _mm256_addsub_pd( _mm256_mul_pd(h2_real, q3), _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#endif
//		tmp4 = _mm256_mul_pd(h2_imag, q4);
#ifdef __ELPA_USE_FMA__
//		y4 = _mm256_add_pd(y4, _mm256_FMSUBADD_pd(h2_real, q4, _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#else
//		y4 = _mm256_add_pd(y4, _mm256_addsub_pd( _mm256_mul_pd(h2_real, q4), _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#endif
	}

	h1_real = _mm256_broadcast_ss(&hh_dbl[(nb-1)*2]);
	h1_imag = _mm256_broadcast_ss(&hh_dbl[((nb-1)*2)+1]);
#ifndef __ELPA_USE_FMA__
	// conjugate
	h1_imag = _mm256_xor_ps(h1_imag, sign);
#endif

	q1 = _mm256_load_ps(&q_dbl[(2*nb*ldq)+0]);
	q2 = _mm256_load_ps(&q_dbl[(2*nb*ldq)+8]);
//	q3 = _mm256_load_pd(&q_dbl[(2*nb*ldq)+8]);
//	q4 = _mm256_load_pd(&q_dbl[(2*nb*ldq)+12]);

	tmp1 = _mm256_mul_ps(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
	x1 = _mm256_add_ps(x1, _mm256_FMSUBADD_ps(h1_real, q1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
	x1 = _mm256_add_ps(x1, _mm256_addsub_ps( _mm256_mul_ps(h1_real, q1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
	tmp2 = _mm256_mul_ps(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
	x2 = _mm256_add_ps(x2, _mm256_FMSUBADD_ps(h1_real, q2, _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#else
	x2 = _mm256_add_ps(x2, _mm256_addsub_ps( _mm256_mul_ps(h1_real, q2), _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#endif
//	tmp3 = _mm256_mul_pd(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
//	x3 = _mm256_add_pd(x3, _mm256_FMSUBADD_pd(h1_real, q3, _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#else
//	x3 = _mm256_add_pd(x3, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q3), _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#endif
//	tmp4 = _mm256_mul_pd(h1_imag, q4);
#ifdef __ELPA_USE_FMA__
//	x4 = _mm256_add_pd(x4, _mm256_FMSUBADD_pd(h1_real, q4, _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#else
//	x4 = _mm256_add_pd(x4, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q4), _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#endif

	h1_real = _mm256_broadcast_ss(&hh_dbl[0]);
	h1_imag = _mm256_broadcast_ss(&hh_dbl[1]);
	h1_real = _mm256_xor_ps(h1_real, sign);
	h1_imag = _mm256_xor_ps(h1_imag, sign);

	tmp1 = _mm256_mul_ps(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
	x1 = _mm256_FMADDSUB_ps(h1_real, x1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1));
#else
	x1 = _mm256_addsub_ps( _mm256_mul_ps(h1_real, x1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1));
#endif
	tmp2 = _mm256_mul_ps(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
	x2 = _mm256_FMADDSUB_ps(h1_real, x2, _mm256_shuffle_ps(tmp2, tmp2, 0xb1));
#else
	x2 = _mm256_addsub_ps( _mm256_mul_ps(h1_real, x2), _mm256_shuffle_ps(tmp2, tmp2, 0xb1));
#endif
//	tmp3 = _mm256_mul_pd(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
//	x3 = _mm256_FMADDSUB_pd(h1_real, x3, _mm256_shuffle_pd(tmp3, tmp3, 0xb1));
#else
//	x3 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x3), _mm256_shuffle_pd(tmp3, tmp3, 0xb1));
#endif
//	tmp4 = _mm256_mul_pd(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
//	x4 = _mm256_FMADDSUB_pd(h1_real, x4, _mm256_shuffle_pd(tmp4, tmp4, 0xb1));
#else
//	x4 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x4), _mm256_shuffle_pd(tmp4, tmp4, 0xb1));
#endif

	h1_real = _mm256_broadcast_ss(&hh_dbl[ldh*2]);
	h1_imag = _mm256_broadcast_ss(&hh_dbl[(ldh*2)+1]);
	h2_real = _mm256_broadcast_ss(&hh_dbl[ldh*2]);
	h2_imag = _mm256_broadcast_ss(&hh_dbl[(ldh*2)+1]);

	h1_real = _mm256_xor_ps(h1_real, sign);
	h1_imag = _mm256_xor_ps(h1_imag, sign);
	h2_real = _mm256_xor_ps(h2_real, sign);
	h2_imag = _mm256_xor_ps(h2_imag, sign);

//carefull here
//	__m128u tmp_s_128 = _mm_loadu_pd(s_dbl);         // load 1 complex value (2 double, i.e. 16 bytes)
        __m128 tmp_s_128 =  _mm_loadu_ps(s_dbl);
//	tmp2 = _mm256_broadcast_pd(&tmp_s_128);          // broad cast the 1 complex , i.e. double it
	tmp2 = _mm256_broadcast_ps(&tmp_s_128);          // broad cast the 1 complex , i.e. double it
	tmp1 = _mm256_mul_ps(h2_imag, tmp2);             // multiply hh2_img with the complex
#ifdef __ELPA_USE_FMA__
	tmp2 = _mm256_FMADDSUB_ps(h2_real, tmp2, _mm256_shuffle_ps(tmp1, tmp1, 0xb1));
#else
	tmp2 = _mm256_addsub_ps( _mm256_mul_ps(h2_real, tmp2), _mm256_shuffle_ps(tmp1, tmp1, 0xb1));
#endif
//careful here

//	_mm_storeu_pd(s_dbl, _mm256_castpd256_pd128(tmp2));
        _mm_storeu_ps(s_dbl, _mm256_castps256_ps128(tmp2));
	h2_real = _mm256_broadcast_ss(&s_dbl[0]);
	h2_imag = _mm256_broadcast_ss(&s_dbl[1]);

	tmp1 = _mm256_mul_ps(h1_imag, y1);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMADDSUB_ps(h1_real, y1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1));
#else
	y1 = _mm256_addsub_ps( _mm256_mul_ps(h1_real, y1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1));
#endif
	tmp2 = _mm256_mul_ps(h1_imag, y2);
#ifdef __ELPA_USE_FMA__
	y2 = _mm256_FMADDSUB_ps(h1_real, y2, _mm256_shuffle_ps(tmp2, tmp2, 0xb1));
#else
	y2 = _mm256_addsub_ps( _mm256_mul_ps(h1_real, y2), _mm256_shuffle_ps(tmp2, tmp2, 0xb1));
#endif
//	tmp3 = _mm256_mul_pd(h1_imag, y3);
#ifdef __ELPA_USE_FMA__
//	y3 = _mm256_FMADDSUB_pd(h1_real, y3, _mm256_shuffle_pd(tmp3, tmp3, 0xb1));
#else
//	y3 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, y3), _mm256_shuffle_pd(tmp3, tmp3, 0xb1));
#endif
//	tmp4 = _mm256_mul_pd(h1_imag, y4);
#ifdef __ELPA_USE_FMA__
//	y4 = _mm256_FMADDSUB_pd(h1_real, y4, _mm256_shuffle_pd(tmp4, tmp4, 0xb1));
#else
//	y4 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, y4), _mm256_shuffle_pd(tmp4, tmp4, 0xb1));
#endif

	tmp1 = _mm256_mul_ps(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_add_ps(y1, _mm256_FMADDSUB_ps(h2_real, x1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
	y1 = _mm256_add_ps(y1, _mm256_addsub_ps( _mm256_mul_ps(h2_real, x1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
	tmp2 = _mm256_mul_ps(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
	y2 = _mm256_add_ps(y2, _mm256_FMADDSUB_ps(h2_real, x2, _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#else
	y2 = _mm256_add_ps(y2, _mm256_addsub_ps( _mm256_mul_ps(h2_real, x2), _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#endif
//	tmp3 = _mm256_mul_pd(h2_imag, x3);
#ifdef __ELPA_USE_FMA__
//	y3 = _mm256_add_pd(y3, _mm256_FMADDSUB_pd(h2_real, x3, _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#else
//	y3 = _mm256_add_pd(y3, _mm256_addsub_pd( _mm256_mul_pd(h2_real, x3), _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#endif
//	tmp4 = _mm256_mul_pd(h2_imag, x4);
#ifdef __ELPA_USE_FMA__
//	y4 = _mm256_add_pd(y4, _mm256_FMADDSUB_pd(h2_real, x4, _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#else
//	y4 = _mm256_add_pd(y4, _mm256_addsub_pd( _mm256_mul_pd(h2_real, x4), _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#endif

	q1 = _mm256_load_ps(&q_dbl[0]);
	q2 = _mm256_load_ps(&q_dbl[8]);
//	q3 = _mm256_load_pd(&q_dbl[8]);
//	q4 = _mm256_load_pd(&q_dbl[12]);

	q1 = _mm256_add_ps(q1, y1);
	q2 = _mm256_add_ps(q2, y2);
//	q3 = _mm256_add_pd(q3, y3);
//	q4 = _mm256_add_pd(q4, y4);

	_mm256_store_ps(&q_dbl[0], q1);
	_mm256_store_ps(&q_dbl[8], q2);
//	_mm256_store_pd(&q_dbl[8], q3);
//	_mm256_store_pd(&q_dbl[12], q4);

	h2_real = _mm256_broadcast_ss(&hh_dbl[(ldh+1)*2]);
	h2_imag = _mm256_broadcast_ss(&hh_dbl[((ldh+1)*2)+1]);

	q1 = _mm256_load_ps(&q_dbl[(ldq*2)+0]);
	q2 = _mm256_load_ps(&q_dbl[(ldq*2)+8]);
//	q3 = _mm256_load_pd(&q_dbl[(ldq*2)+8]);
//	q4 = _mm256_load_pd(&q_dbl[(ldq*2)+12]);

	q1 = _mm256_add_ps(q1, x1);
	q2 = _mm256_add_ps(q2, x2);
//	q3 = _mm256_add_pd(q3, x3);
//	q4 = _mm256_add_pd(q4, x4);

	tmp1 = _mm256_mul_ps(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_add_ps(q1, _mm256_FMADDSUB_ps(h2_real, y1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
	q1 = _mm256_add_ps(q1, _mm256_addsub_ps( _mm256_mul_ps(h2_real, y1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
	tmp2 = _mm256_mul_ps(h2_imag, y2);
#ifdef __FMA4_
	q2 = _mm256_add_ps(q2, _mm256_FMADDSUB_ps(h2_real, y2, _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#else
	q2 = _mm256_add_ps(q2, _mm256_addsub_ps( _mm256_mul_ps(h2_real, y2), _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#endif
//	tmp3 = _mm256_mul_pd(h2_imag, y3);
#ifdef __ELPA_USE_FMA__
//	q3 = _mm256_add_pd(q3, _mm256_FMADDSUB_pd(h2_real, y3, _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#else
//	q3 = _mm256_add_pd(q3, _mm256_addsub_pd( _mm256_mul_pd(h2_real, y3), _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#endif
//	tmp4 = _mm256_mul_pd(h2_imag, y4);
#ifdef __ELPA_USE_FMA__
//	q4 = _mm256_add_pd(q4, _mm256_FMADDSUB_pd(h2_real, y4, _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#else
//	q4 = _mm256_add_pd(q4, _mm256_addsub_pd( _mm256_mul_pd(h2_real, y4), _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#endif

	_mm256_store_ps(&q_dbl[(ldq*2)+0], q1);
	_mm256_store_ps(&q_dbl[(ldq*2)+8], q2);
//	_mm256_store_pd(&q_dbl[(ldq*2)+8], q3);
//	_mm256_store_pd(&q_dbl[(ldq*2)+12], q4);

	for (i = 2; i < nb; i++)
	{
		q1 = _mm256_load_ps(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm256_load_ps(&q_dbl[(2*i*ldq)+8]);
//		q3 = _mm256_load_pd(&q_dbl[(2*i*ldq)+8]);
//		q4 = _mm256_load_pd(&q_dbl[(2*i*ldq)+12]);

		h1_real = _mm256_broadcast_ss(&hh_dbl[(i-1)*2]);
		h1_imag = _mm256_broadcast_ss(&hh_dbl[((i-1)*2)+1]);

		tmp1 = _mm256_mul_ps(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_add_ps(q1, _mm256_FMADDSUB_ps(h1_real, x1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
		q1 = _mm256_add_ps(q1, _mm256_addsub_ps( _mm256_mul_ps(h1_real, x1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
							// ##################### //
#endif
		tmp2 = _mm256_mul_ps(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
		q2 = _mm256_add_ps(q2, _mm256_FMADDSUB_ps(h1_real, x2, _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#else
		q2 = _mm256_add_ps(q2, _mm256_addsub_ps( _mm256_mul_ps(h1_real, x2), _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#endif
//		tmp3 = _mm256_mul_pd(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
//		q3 = _mm256_add_pd(q3, _mm256_FMADDSUB_pd(h1_real, x3, _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#else
//		q3 = _mm256_add_pd(q3, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x3), _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#endif
//		tmp4 = _mm256_mul_pd(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
//		q4 = _mm256_add_pd(q4, _mm256_FMADDSUB_pd(h1_real, x4, _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#else
//		q4 = _mm256_add_pd(q4, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x4), _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#endif

		h2_real = _mm256_broadcast_ss(&hh_dbl[(ldh+i)*2]);
		h2_imag = _mm256_broadcast_ss(&hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _mm256_mul_ps(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_add_ps(q1, _mm256_FMADDSUB_ps(h2_real, y1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
		q1 = _mm256_add_ps(q1, _mm256_addsub_ps( _mm256_mul_ps(h2_real, y1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
		tmp2 = _mm256_mul_ps(h2_imag, y2);
#ifdef __ELPA_USE_FMA__
		q2 = _mm256_add_ps(q2, _mm256_FMADDSUB_ps(h2_real, y2, _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#else
		q2 = _mm256_add_ps(q2, _mm256_addsub_ps( _mm256_mul_ps(h2_real, y2), _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#endif
//		tmp3 = _mm256_mul_pd(h2_imag, y3);
#ifdef __ELPA_USE_FMA__
//		q3 = _mm256_add_pd(q3, _mm256_FMADDSUB_pd(h2_real, y3, _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#else
//		q3 = _mm256_add_pd(q3, _mm256_addsub_pd( _mm256_mul_pd(h2_real, y3), _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#endif
//		tmp4 = _mm256_mul_pd(h2_imag, y4);
#ifdef __ELPA_USE_FMA__
//		q4 = _mm256_add_pd(q4, _mm256_FMADDSUB_pd(h2_real, y4, _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#else
//		q4 = _mm256_add_pd(q4, _mm256_addsub_pd( _mm256_mul_pd(h2_real, y4), _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#endif

		_mm256_store_ps(&q_dbl[(2*i*ldq)+0], q1);
		_mm256_store_ps(&q_dbl[(2*i*ldq)+8], q2);
//		_mm256_store_pd(&q_dbl[(2*i*ldq)+8], q3);
//		_mm256_store_pd(&q_dbl[(2*i*ldq)+12], q4);
	}
	h1_real = _mm256_broadcast_ss(&hh_dbl[(nb-1)*2]);
	h1_imag = _mm256_broadcast_ss(&hh_dbl[((nb-1)*2)+1]);

	q1 = _mm256_load_ps(&q_dbl[(2*nb*ldq)+0]);
	q2 = _mm256_load_ps(&q_dbl[(2*nb*ldq)+8]);
//	q3 = _mm256_load_pd(&q_dbl[(2*nb*ldq)+8]);
//	q4 = _mm256_load_pd(&q_dbl[(2*nb*ldq)+12]);

	tmp1 = _mm256_mul_ps(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_add_ps(q1, _mm256_FMADDSUB_ps(h1_real, x1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
	q1 = _mm256_add_ps(q1, _mm256_addsub_ps( _mm256_mul_ps(h1_real, x1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
	tmp2 = _mm256_mul_ps(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
	q2 = _mm256_add_ps(q2, _mm256_FMADDSUB_ps(h1_real, x2, _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#else
	q2 = _mm256_add_ps(q2, _mm256_addsub_ps( _mm256_mul_ps(h1_real, x2), _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#endif
//	tmp3 = _mm256_mul_pd(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
//	q3 = _mm256_add_pd(q3, _mm256_FMADDSUB_pd(h1_real, x3, _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#else
//	q3 = _mm256_add_pd(q3, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x3), _mm256_shuffle_pd(tmp3, tmp3, 0xb1)));
#endif
//	tmp4 = _mm256_mul_pd(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
//	q4 = _mm256_add_pd(q4, _mm256_FMADDSUB_pd(h1_real, x4, _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#else
//	q4 = _mm256_add_pd(q4, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x4), _mm256_shuffle_pd(tmp4, tmp4, 0xb1)));
#endif

	_mm256_store_ps(&q_dbl[(2*nb*ldq)+0], q1);
	_mm256_store_ps(&q_dbl[(2*nb*ldq)+8], q2);
//	_mm256_store_pd(&q_dbl[(2*nb*ldq)+8], q3);
//	_mm256_store_pd(&q_dbl[(2*nb*ldq)+12], q4);
}

static __forceinline void hh_trafo_complex_kernel_4_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s, float complex s1)
{
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;
	float* s_dbl = (float*)(&s);

	__m256 x1, x2;
	__m256 y1, y2;
	__m256 q1, q2;
	__m256 h1_real, h1_imag, h2_real, h2_imag;
	__m256 tmp1, tmp2;
	int i=0;

	__m256 sign = (__m256)_mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);

	x1 = _mm256_load_ps(&q_dbl[(2*ldq)+0]);
//	x2 = _mm256_load_pd(&q_dbl[(2*ldq)+4]);

	h2_real = _mm256_broadcast_ss(&hh_dbl[(ldh+1)*2]);
	h2_imag = _mm256_broadcast_ss(&hh_dbl[((ldh+1)*2)+1]);
#ifndef __ELPA_USE_FMA__
	// conjugate
	h2_imag = _mm256_xor_ps(h2_imag, sign);
#endif

	y1 = _mm256_load_ps(&q_dbl[0]);
//	y2 = _mm256_load_pd(&q_dbl[4]);

	tmp1 = _mm256_mul_ps(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_add_ps(y1, _mm256_FMSUBADD_ps(h2_real, x1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
	y1 = _mm256_add_ps(y1, _mm256_addsub_ps( _mm256_mul_ps(h2_real, x1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
//	tmp2 = _mm256_mul_pd(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
//	y2 = _mm256_add_pd(y2, _mm256_FMSUBADD_pd(h2_real, x2, _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#else
//	y2 = _mm256_add_pd(y2, _mm256_addsub_pd( _mm256_mul_pd(h2_real, x2), _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#endif

	for (i = 2; i < nb; i++)
	{
		q1 = _mm256_load_ps(&q_dbl[(2*i*ldq)+0]);
//		q2 = _mm256_load_pd(&q_dbl[(2*i*ldq)+4]);

		h1_real = _mm256_broadcast_ss(&hh_dbl[(i-1)*2]);
		h1_imag = _mm256_broadcast_ss(&hh_dbl[((i-1)*2)+1]);
#ifndef __ELPA_USE_FMA__
		// conjugate
		h1_imag = _mm256_xor_ps(h1_imag, sign);
#endif

		tmp1 = _mm256_mul_ps(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
		x1 = _mm256_add_ps(x1, _mm256_FMSUBADD_ps(h1_real, q1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
		x1 = _mm256_add_ps(x1, _mm256_addsub_ps( _mm256_mul_ps(h1_real, q1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
//		tmp2 = _mm256_mul_pd(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
//		x2 = _mm256_add_pd(x2, _mm256_FMSUBADD_pd(h1_real, q2, _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#else
//		x2 = _mm256_add_pd(x2, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q2), _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#endif

		h2_real = _mm256_broadcast_ss(&hh_dbl[(ldh+i)*2]);
		h2_imag = _mm256_broadcast_ss(&hh_dbl[((ldh+i)*2)+1]);
#ifndef __ELPA_USE_FMA__
		// conjugate
		h2_imag = _mm256_xor_ps(h2_imag, sign);
#endif

		tmp1 = _mm256_mul_ps(h2_imag, q1);
#ifdef __ELPA_USE_FMA__
		y1 = _mm256_add_ps(y1, _mm256_FMSUBADD_ps(h2_real, q1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
		y1 = _mm256_add_ps(y1, _mm256_addsub_ps( _mm256_mul_ps(h2_real, q1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
//		tmp2 = _mm256_mul_pd(h2_imag, q2);
#ifdef __ELPA_USE_FMA__
//		y2 = _mm256_add_pd(y2, _mm256_FMSUBADD_pd(h2_real, q2, _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#else
//		y2 = _mm256_add_pd(y2, _mm256_addsub_pd( _mm256_mul_pd(h2_real, q2), _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#endif
	}

	h1_real = _mm256_broadcast_ss(&hh_dbl[(nb-1)*2]);
	h1_imag = _mm256_broadcast_ss(&hh_dbl[((nb-1)*2)+1]);
#ifndef __ELPA_USE_FMA__
	// conjugate
	h1_imag = _mm256_xor_ps(h1_imag, sign);
#endif

	q1 = _mm256_load_ps(&q_dbl[(2*nb*ldq)+0]);
//	q2 = _mm256_load_pd(&q_dbl[(2*nb*ldq)+4]);

	tmp1 = _mm256_mul_ps(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
	x1 = _mm256_add_ps(x1, _mm256_FMSUBADD_ps(h1_real, q1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
	x1 = _mm256_add_ps(x1, _mm256_addsub_ps( _mm256_mul_ps(h1_real, q1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
//	tmp2 = _mm256_mul_pd(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
//	x2 = _mm256_add_pd(x2, _mm256_FMSUBADD_pd(h1_real, q2, _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#else
//	x2 = _mm256_add_pd(x2, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q2), _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#endif

	h1_real = _mm256_broadcast_ss(&hh_dbl[0]);
	h1_imag = _mm256_broadcast_ss(&hh_dbl[1]);
	h1_real = _mm256_xor_ps(h1_real, sign);
	h1_imag = _mm256_xor_ps(h1_imag, sign);

	tmp1 = _mm256_mul_ps(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
	x1 = _mm256_FMADDSUB_ps(h1_real, x1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1));
#else
	x1 = _mm256_addsub_ps( _mm256_mul_ps(h1_real, x1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1));
#endif
	tmp2 = _mm256_mul_ps(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
	x2 = _mm256_FMADDSUB_ps(h1_real, x2, _mm256_shuffle_ps(tmp2, tmp2, 0xb1));
#else
	x2 = _mm256_addsub_ps( _mm256_mul_ps(h1_real, x2), _mm256_shuffle_ps(tmp2, tmp2, 0xb1));
#endif

	h1_real = _mm256_broadcast_ss(&hh_dbl[ldh*2]);
	h1_imag = _mm256_broadcast_ss(&hh_dbl[(ldh*2)+1]);
	h2_real = _mm256_broadcast_ss(&hh_dbl[ldh*2]);
	h2_imag = _mm256_broadcast_ss(&hh_dbl[(ldh*2)+1]);

	h1_real = _mm256_xor_ps(h1_real, sign);
	h1_imag = _mm256_xor_ps(h1_imag, sign);
	h2_real = _mm256_xor_ps(h2_real, sign);
	h2_imag = _mm256_xor_ps(h2_imag, sign);


//carefull
//	__m128d tmp_s_128 = _mm_loadu_pd(s_dbl);
	__m128 tmp_s_128 = _mm_loadu_ps(s_dbl);
	tmp2 = _mm256_broadcast_ps(&tmp_s_128);
	tmp1 = _mm256_mul_ps(h2_imag, tmp2);
#ifdef __ELPA_USE_FMA__
	tmp2 = _mm256_FMADDSUB_ps(h2_real, tmp2, _mm256_shuffle_ps(tmp1, tmp1, 0xb1));
#else
	tmp2 = _mm256_addsub_ps( _mm256_mul_ps(h2_real, tmp2), _mm256_shuffle_ps(tmp1, tmp1, 0xb1));
#endif

//	_mm_storeu_pd(s_dbl, _mm256_castpd256_pd128(tmp2));
        _mm_storeu_ps(s_dbl, _mm256_castps256_ps128(tmp2));
	h2_real = _mm256_broadcast_ss(&s_dbl[0]);
	h2_imag = _mm256_broadcast_ss(&s_dbl[1]);

	tmp1 = _mm256_mul_ps(h1_imag, y1);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_FMADDSUB_ps(h1_real, y1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1));
#else
	y1 = _mm256_addsub_ps( _mm256_mul_ps(h1_real, y1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1));
#endif
	tmp2 = _mm256_mul_ps(h1_imag, y2);
#ifdef __ELPA_USE_FMA__
//	y2 = _mm256_FMADDSUB_ps(h1_real, y2, _mm256_shuffle_ps(tmp2, tmp2, 0xb1));
#else
//	y2 = _mm256_addsub_ps( _mm256_mul_pd(h1_real, y2), _mm256_shuffle_pd(tmp2, tmp2, 0xb1));
#endif

	tmp1 = _mm256_mul_ps(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
	y1 = _mm256_add_ps(y1, _mm256_FMADDSUB_ps(h2_real, x1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
	y1 = _mm256_add_ps(y1, _mm256_addsub_ps( _mm256_mul_ps(h2_real, x1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
	tmp2 = _mm256_mul_ps(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
//	y2 = _mm256_add_pd(y2, _mm256_FMADDSUB_pd(h2_real, x2, _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#else
//	y2 = _mm256_add_pd(y2, _mm256_addsub_pd( _mm256_mul_pd(h2_real, x2), _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#endif

	q1 = _mm256_load_ps(&q_dbl[0]);
//	q2 = _mm256_load_pd(&q_dbl[4]);

	q1 = _mm256_add_ps(q1, y1);
//	q2 = _mm256_add_pd(q2, y2);

	_mm256_store_ps(&q_dbl[0], q1);
//	_mm256_store_pd(&q_dbl[4], q2);

	h2_real = _mm256_broadcast_ss(&hh_dbl[(ldh+1)*2]);
	h2_imag = _mm256_broadcast_ss(&hh_dbl[((ldh+1)*2)+1]);

	q1 = _mm256_load_ps(&q_dbl[(ldq*2)+0]);
//	q2 = _mm256_load_pd(&q_dbl[(ldq*2)+4]);

	q1 = _mm256_add_ps(q1, x1);
//	q2 = _mm256_add_pd(q2, x2);

	tmp1 = _mm256_mul_ps(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_add_ps(q1, _mm256_FMADDSUB_ps(h2_real, y1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
	q1 = _mm256_add_ps(q1, _mm256_addsub_ps( _mm256_mul_ps(h2_real, y1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
//	tmp2 = _mm256_mul_pd(h2_imag, y2);
#ifdef __FMA4_
//	q2 = _mm256_add_pd(q2, _mm256_FMADDSUB_pd(h2_real, y2, _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#else
//	q2 = _mm256_add_pd(q2, _mm256_addsub_pd( _mm256_mul_pd(h2_real, y2), _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#endif

	_mm256_store_ps(&q_dbl[(ldq*2)+0], q1);
//	_mm256_store_pd(&q_dbl[(ldq*2)+4], q2);

	for (i = 2; i < nb; i++)
	{
		q1 = _mm256_load_ps(&q_dbl[(2*i*ldq)+0]);
//		q2 = _mm256_load_pd(&q_dbl[(2*i*ldq)+4]);

		h1_real = _mm256_broadcast_ss(&hh_dbl[(i-1)*2]);
		h1_imag = _mm256_broadcast_ss(&hh_dbl[((i-1)*2)+1]);

		tmp1 = _mm256_mul_ps(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_add_ps(q1, _mm256_FMADDSUB_ps(h1_real, x1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
		q1 = _mm256_add_ps(q1, _mm256_addsub_ps( _mm256_mul_ps(h1_real, x1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
//		tmp2 = _mm256_mul_pd(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
//		q2 = _mm256_add_pd(q2, _mm256_FMADDSUB_pd(h1_real, x2, _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#else
//		q2 = _mm256_add_pd(q2, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x2), _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#endif

		h2_real = _mm256_broadcast_ss(&hh_dbl[(ldh+i)*2]);
		h2_imag = _mm256_broadcast_ss(&hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _mm256_mul_ps(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_add_ps(q1, _mm256_FMADDSUB_ps(h2_real, y1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
		q1 = _mm256_add_ps(q1, _mm256_addsub_ps( _mm256_mul_ps(h2_real, y1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
//		tmp2 = _mm256_mul_pd(h2_imag, y2);
#ifdef __ELPA_USE_FMA__
//		q2 = _mm256_add_pd(q2, _mm256_FMADDSUB_pd(h2_real, y2, _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#else
//		q2 = _mm256_add_pd(q2, _mm256_addsub_pd( _mm256_mul_pd(h2_real, y2), _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#endif

		_mm256_store_ps(&q_dbl[(2*i*ldq)+0], q1);
//		_mm256_store_pd(&q_dbl[(2*i*ldq)+4], q2);
	}
	h1_real = _mm256_broadcast_ss(&hh_dbl[(nb-1)*2]);
	h1_imag = _mm256_broadcast_ss(&hh_dbl[((nb-1)*2)+1]);

	q1 = _mm256_load_ps(&q_dbl[(2*nb*ldq)+0]);
//	q2 = _mm256_load_pd(&q_dbl[(2*nb*ldq)+4]);

	tmp1 = _mm256_mul_ps(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
	q1 = _mm256_add_ps(q1, _mm256_FMADDSUB_ps(h1_real, x1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
	q1 = _mm256_add_ps(q1, _mm256_addsub_ps( _mm256_mul_ps(h1_real, x1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
//	tmp2 = _mm256_mul_pd(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
//	q2 = _mm256_add_pd(q2, _mm256_FMADDSUB_pd(h1_real, x2, _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#else
//	q2 = _mm256_add_pd(q2, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x2), _mm256_shuffle_pd(tmp2, tmp2, 0xb1)));
#endif

	_mm256_store_ps(&q_dbl[(2*nb*ldq)+0], q1);
//	_mm256_store_pd(&q_dbl[(2*nb*ldq)+4], q2);
}
