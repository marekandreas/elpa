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
static  __forceinline void hh_trafo_complex_kernel_12_AVX_1hv_single(float complex* q, float complex* hh, int nb, int ldq);
static  __forceinline void hh_trafo_complex_kernel_8_AVX_1hv_single(float complex* q, float complex* hh, int nb, int ldq);
static  __forceinline void hh_trafo_complex_kernel_4_AVX_1hv_single(float complex* q, float complex* hh, int nb, int ldq);

/*
!f>#if defined(HAVE_AVX) || defined(HAVE_AVX2)
!f> interface
!f>   subroutine single_hh_trafo_complex_avx_avx2_1hv_single(q, hh, pnb, pnq, pldq) &
!f>                             bind(C, name="single_hh_trafo_complex_avx_avx2_1hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq
!f>     complex(kind=c_float)   :: q(*)
!f>     complex(kind=c_float)   :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/

void single_hh_trafo_complex_avx_avx2_1hv_single(float complex* q, float complex* hh, int* pnb, int* pnq, int* pldq)
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	//int ldh = *pldh;

// carefull here
	for (i = 0; i < nq-8; i+=12)
	{
		hh_trafo_complex_kernel_12_AVX_1hv_single(&q[i], hh, nb, ldq);
	}
	if (nq == i)
	{
		return;
	}
//	if (nq-i == 20)
//	{
//		hh_trafo_complex_kernel_16_AVX_1hv_single(&q[i], hh, nb, ldq);
//		hh_trafo_complex_kernel_4_AVX_1hv_single(&q[i], hh, nb, ldq);
//	}
//	if (nq-i == 16)
//	{
//		hh_trafo_complex_kernel_16_AVX_1hv_single(&q[i], hh, nb, ldq);
//	}
//	if (nq-i == 12)
//	{
//		hh_trafo_complex_kernel_8_AVX_1hv_single(&q[i], hh, nb, ldq);
//		hh_trafo_complex_kernel_4_AVX_1hv_single(&q[i], hh, nb, ldq);
//	}
	if (nq-i == 8)
	{
		hh_trafo_complex_kernel_8_AVX_1hv_single(&q[i], hh, nb, ldq);
	}
	else
	{
		hh_trafo_complex_kernel_4_AVX_1hv_single(&q[i], hh, nb, ldq);
	}
}

 static __forceinline void hh_trafo_complex_kernel_12_AVX_1hv_single(float complex* q, float complex* hh, int nb, int ldq)
{
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;

	__m256 x1, x2, x3, x4, x5, x6;
	__m256 q1, q2, q3, q4, q5, q6;
	__m256 h1_real, h1_imag;
	__m256 tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
	int i=0;

//carefull here
//	__m256d sign = (__m256d)_mm256_set_epi64x(0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000);
	__m256 sign = (__m256)_mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);

	x1 = _mm256_load_ps(&q_dbl[0]);      // load four complex q(1,1) | q(2,1) | q(3,1) q(4,1)
	x2 = _mm256_load_ps(&q_dbl[8]);      // load four complex q(5,1) | q(6,1) | q(7,1) q(8,1)
	x3 = _mm256_load_ps(&q_dbl[16]);     // load four complex q(9,1) | q(10,1) | q(11,1) q(12,1)
//	x4 = _mm256_load_pd(&q_dbl[24]);     // load four complex q(13,1) | q(14,1) | q(15,1) q(16,1)
//	x5 = _mm256_load_pd(&q_dbl[32]);     // load four complex q(17,1) | q(18,1) | q(19,1) q(20,1)
//	x6 = _mm256_load_pd(&q_dbl[40]);     // load four complex q(21,1) | q(22,1) | q(23,1) q(24,1)


	for (i = 1; i < nb; i++)
	{
		h1_real = _mm256_broadcast_ss(&hh_dbl[i*2]);
		h1_imag = _mm256_broadcast_ss(&hh_dbl[(i*2)+1]);
#ifndef __ELPA_USE_FMA__
		// conjugate
		h1_imag = _mm256_xor_ps(h1_imag, sign);       // h1_imag = - h1_imag
#endif

		q1 = _mm256_load_ps(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm256_load_ps(&q_dbl[(2*i*ldq)+8]);
		q3 = _mm256_load_ps(&q_dbl[(2*i*ldq)+16]);
//		q4 = _mm256_load_pd(&q_dbl[(2*i*ldq)+12]);
//		q5 = _mm256_load_pd(&q_dbl[(2*i*ldq)+16]);
//		q6 = _mm256_load_pd(&q_dbl[(2*i*ldq)+20]);

		tmp1 = _mm256_mul_ps(h1_imag, q1);          // tmp1 = -h1_imag * q1
		// carefull here we want x1 = x1 + q(1,i) * conjg(hh(i))
#ifdef __ELPA_USE_FMA__
		x1 = _mm256_add_ps(x1, _mm256_FMSUBADD_ps(h1_real, q1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));                    // x1 = x1 + (h1_real *q1 + 
#else
		x1 = _mm256_add_ps(x1, _mm256_addsub_ps( _mm256_mul_ps(h1_real, q1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
		tmp2 = _mm256_mul_ps(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
		x2 = _mm256_add_ps(x2, _mm256_FMSUBADD_ps(h1_real, q2, _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#else
		x2 = _mm256_add_ps(x2, _mm256_addsub_ps( _mm256_mul_ps(h1_real, q2), _mm256_shuffle_ps(tmp2, tmp2, 0xb1)));
#endif
		tmp3 = _mm256_mul_ps(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
		x3 = _mm256_add_ps(x3, _mm256_FMSUBADD_ps(h1_real, q3, _mm256_shuffle_ps(tmp3, tmp3, 0xb1)));
#else
		x3 = _mm256_add_ps(x3, _mm256_addsub_ps( _mm256_mul_ps(h1_real, q3), _mm256_shuffle_ps(tmp3, tmp3, 0xb1)));
#endif
//		tmp4 = _mm256_mul_pd(h1_imag, q4);
#ifdef __ELPA_USE_FMA__
//		x4 = _mm256_add_pd(x4, _mm256_FMSUBADD_pd(h1_real, q4, _mm256_shuffle_pd(tmp4, tmp4, 0x5)));
#else
//		x4 = _mm256_add_pd(x4, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q4), _mm256_shuffle_pd(tmp4, tmp4, 0x5)));
#endif
//		tmp5 = _mm256_mul_pd(h1_imag, q5);
#ifdef __ELPA_USE_FMA__
//		x5 = _mm256_add_pd(x5, _mm256_FMSUBADD_pd(h1_real, q5, _mm256_shuffle_pd(tmp5, tmp5, 0x5)));
#else
//		x5 = _mm256_add_pd(x5, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q5), _mm256_shuffle_pd(tmp5, tmp5, 0x5)));
#endif
//		tmp6 = _mm256_mul_pd(h1_imag, q6);
#ifdef __ELPA_USE_FMA__
//		x6 = _mm256_add_pd(x6, _mm256_FMSUBADD_pd(h1_real, q6, _mm256_shuffle_pd(tmp6, tmp6, 0x5)));
#else
//		x6 = _mm256_add_pd(x6, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q6), _mm256_shuffle_pd(tmp6, tmp6, 0x5)));
#endif
	}

	h1_real = _mm256_broadcast_ss(&hh_dbl[0]);
	h1_imag = _mm256_broadcast_ss(&hh_dbl[1]);
	h1_real = _mm256_xor_ps(h1_real, sign);
	h1_imag = _mm256_xor_ps(h1_imag, sign);

	tmp1 = _mm256_mul_ps(h1_imag, x1);
	// carefull here
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
	tmp3 = _mm256_mul_ps(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
	x3 = _mm256_FMADDSUB_ps(h1_real, x3, _mm256_shuffle_ps(tmp3, tmp3, 0xb1));
#else
	x3 = _mm256_addsub_ps( _mm256_mul_ps(h1_real, x3), _mm256_shuffle_ps(tmp3, tmp3, 0xb1));
#endif
//	tmp4 = _mm256_mul_pd(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
//	x4 = _mm256_FMADDSUB_pd(h1_real, x4, _mm256_shuffle_pd(tmp4, tmp4, 0x5));
#else
//	x4 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x4), _mm256_shuffle_pd(tmp4, tmp4, 0x5));
#endif
//	tmp5 = _mm256_mul_pd(h1_imag, x5);
#ifdef __ELPA_USE_FMA__
//	x5 = _mm256_FMADDSUB_pd(h1_real, x5, _mm256_shuffle_pd(tmp5, tmp5, 0x5));
#else
//	x5 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x5), _mm256_shuffle_pd(tmp5, tmp5, 0x5));
#endif
//	tmp6 = _mm256_mul_pd(h1_imag, x6);
#ifdef __ELPA_USE_FMA__
//	x6 = _mm256_FMADDSUB_pd(h1_real, x6, _mm256_shuffle_pd(tmp6, tmp6, 0x5));
#else
//	x6 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x6), _mm256_shuffle_pd(tmp6, tmp6, 0x5));
#endif

	q1 = _mm256_load_ps(&q_dbl[0]);
	q2 = _mm256_load_ps(&q_dbl[8]);
	q3 = _mm256_load_ps(&q_dbl[16]);
//	q4 = _mm256_load_pd(&q_dbl[12]);
//	q5 = _mm256_load_pd(&q_dbl[16]);
//	q6 = _mm256_load_pd(&q_dbl[20]);

	q1 = _mm256_add_ps(q1, x1);
	q2 = _mm256_add_ps(q2, x2);
	q3 = _mm256_add_ps(q3, x3);
//	q4 = _mm256_add_pd(q4, x4);
//	q5 = _mm256_add_pd(q5, x5);
//	q6 = _mm256_add_pd(q6, x6);

	_mm256_store_ps(&q_dbl[0], q1);
	_mm256_store_ps(&q_dbl[8], q2);
	_mm256_store_ps(&q_dbl[16], q3);
//	_mm256_store_pd(&q_dbl[12], q4);
//	_mm256_store_pd(&q_dbl[16], q5);
//	_mm256_store_pd(&q_dbl[20], q6);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm256_broadcast_ss(&hh_dbl[i*2]);
		h1_imag = _mm256_broadcast_ss(&hh_dbl[(i*2)+1]);

		q1 = _mm256_load_ps(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm256_load_ps(&q_dbl[(2*i*ldq)+8]);
		q3 = _mm256_load_ps(&q_dbl[(2*i*ldq)+16]);
//		q4 = _mm256_load_pd(&q_dbl[(2*i*ldq)+12]);
//		q5 = _mm256_load_pd(&q_dbl[(2*i*ldq)+16]);
//		q6 = _mm256_load_pd(&q_dbl[(2*i*ldq)+20]);

		tmp1 = _mm256_mul_ps(h1_imag, x1);
		// carefull
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
		tmp3 = _mm256_mul_ps(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
		q3 = _mm256_add_ps(q3, _mm256_FMADDSUB_ps(h1_real, x3, _mm256_shuffle_ps(tmp3, tmp3, 0xb1)));
#else
		q3 = _mm256_add_ps(q3, _mm256_addsub_ps( _mm256_mul_ps(h1_real, x3), _mm256_shuffle_ps(tmp3, tmp3, 0xb1)));
#endif
//		tmp4 = _mm256_mul_pd(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
//		q4 = _mm256_add_pd(q4, _mm256_FMADDSUB_pd(h1_real, x4, _mm256_shuffle_pd(tmp4, tmp4, 0x5)));
#else
//		q4 = _mm256_add_pd(q4, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x4), _mm256_shuffle_pd(tmp4, tmp4, 0x5)));
#endif
//		tmp5 = _mm256_mul_pd(h1_imag, x5);
#ifdef __ELPA_USE_FMA__
//		q5 = _mm256_add_pd(q5, _mm256_FMADDSUB_pd(h1_real, x5, _mm256_shuffle_pd(tmp5, tmp5, 0x5)));
#else
//		q5 = _mm256_add_pd(q5, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x5), _mm256_shuffle_pd(tmp5, tmp5, 0x5)));
#endif
//		tmp6 = _mm256_mul_pd(h1_imag, x6);
#ifdef __ELPA_USE_FMA__
//		q6 = _mm256_add_pd(q6, _mm256_FMADDSUB_pd(h1_real, x6, _mm256_shuffle_pd(tmp6, tmp6, 0x5)));
#else
//		q6 = _mm256_add_pd(q6, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x6), _mm256_shuffle_pd(tmp6, tmp6, 0x5)));
#endif

		_mm256_store_ps(&q_dbl[(2*i*ldq)+0], q1);
		_mm256_store_ps(&q_dbl[(2*i*ldq)+8], q2);
		_mm256_store_ps(&q_dbl[(2*i*ldq)+16], q3);
//		_mm256_store_pd(&q_dbl[(2*i*ldq)+12], q4);
//		_mm256_store_pd(&q_dbl[(2*i*ldq)+16], q5);
//		_mm256_store_pd(&q_dbl[(2*i*ldq)+20], q6);
	}
}

static __forceinline void hh_trafo_complex_kernel_8_AVX_1hv_single(float complex* q, float complex* hh, int nb, int ldq)
{
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;

	__m256 x1, x2, x3, x4;
	__m256 q1, q2, q3, q4;
	__m256 h1_real, h1_imag;
	__m256 tmp1, tmp2, tmp3, tmp4;
	int i=0;

	// carefull 
//	__m256d sign = (__m256d)_mm256_set_epi64x(0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000);
	__m256 sign = (__m256)_mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);

	x1 = _mm256_load_ps(&q_dbl[0]);   // load q(1,1) | q(2,1) | q(3,1) | q(4,1)
	x2 = _mm256_load_ps(&q_dbl[8]);   // load q(1,1) | q(2,1) | q(3,1) | q(4,1)
//	x3 = _mm256_load_pd(&q_dbl[8]);
//	x4 = _mm256_load_pd(&q_dbl[12]);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm256_broadcast_ss(&hh_dbl[i*2]);
		h1_imag = _mm256_broadcast_ss(&hh_dbl[(i*2)+1]);
#ifndef __ELPA_USE_FMA__
		// conjugate
		h1_imag = _mm256_xor_ps(h1_imag, sign);
#endif

		q1 = _mm256_load_ps(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm256_load_ps(&q_dbl[(2*i*ldq)+8]);
//		q3 = _mm256_load_pd(&q_dbl[(2*i*ldq)+8]);
//		q4 = _mm256_load_pd(&q_dbl[(2*i*ldq)+12]);

		tmp1 = _mm256_mul_ps(h1_imag, q1);
		// carefull
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
//		x3 = _mm256_add_pd(x3, _mm256_FMSUBADD_pd(h1_real, q3, _mm256_shuffle_pd(tmp3, tmp3, 0x5)));
#else
//		x3 = _mm256_add_pd(x3, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q3), _mm256_shuffle_pd(tmp3, tmp3, 0x5)));
#endif
//		tmp4 = _mm256_mul_pd(h1_imag, q4);
#ifdef __ELPA_USE_FMA__
//		x4 = _mm256_add_pd(x4, _mm256_FMSUBADD_pd(h1_real, q4, _mm256_shuffle_pd(tmp4, tmp4, 0x5)));
#else
//		x4 = _mm256_add_pd(x4, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q4), _mm256_shuffle_pd(tmp4, tmp4, 0x5)));
#endif
	}

	h1_real = _mm256_broadcast_ss(&hh_dbl[0]);
	h1_imag = _mm256_broadcast_ss(&hh_dbl[1]);
	h1_real = _mm256_xor_ps(h1_real, sign);
	h1_imag = _mm256_xor_ps(h1_imag, sign);

	tmp1 = _mm256_mul_ps(h1_imag, x1);

	// carefull
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
//	x3 = _mm256_FMADDSUB_pd(h1_real, x3, _mm256_shuffle_pd(tmp3, tmp3, 0x5));
#else
//	x3 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x3), _mm256_shuffle_pd(tmp3, tmp3, 0x5));
#endif
//	tmp4 = _mm256_mul_pd(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
//	x4 = _mm256_FMADDSUB_pd(h1_real, x4, _mm256_shuffle_pd(tmp4, tmp4, 0x5));
#else
//	x4 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x4), _mm256_shuffle_pd(tmp4, tmp4, 0x5));
#endif

	q1 = _mm256_load_ps(&q_dbl[0]);
	q2 = _mm256_load_ps(&q_dbl[8]);
//	q3 = _mm256_load_pd(&q_dbl[8]);
//	q4 = _mm256_load_pd(&q_dbl[12]);

	q1 = _mm256_add_ps(q1, x1);
	q2 = _mm256_add_ps(q2, x2);
//	q3 = _mm256_add_pd(q3, x3);
//	q4 = _mm256_add_pd(q4, x4);

	_mm256_store_ps(&q_dbl[0], q1);
	_mm256_store_ps(&q_dbl[8], q2);
//	_mm256_store_pd(&q_dbl[8], q3);
//	_mm256_store_pd(&q_dbl[12], q4);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm256_broadcast_ss(&hh_dbl[i*2]);
		h1_imag = _mm256_broadcast_ss(&hh_dbl[(i*2)+1]);

		q1 = _mm256_load_ps(&q_dbl[(2*i*ldq)+0]);
		q2 = _mm256_load_ps(&q_dbl[(2*i*ldq)+8]);
//		q3 = _mm256_load_pd(&q_dbl[(2*i*ldq)+8]);
//		q4 = _mm256_load_pd(&q_dbl[(2*i*ldq)+12]);

		tmp1 = _mm256_mul_ps(h1_imag, x1);
		// carefull
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
//		tmp3 = _mm256_mul_pd(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
//		q3 = _mm256_add_pd(q3, _mm256_FMADDSUB_pd(h1_real, x3, _mm256_shuffle_pd(tmp3, tmp3, 0x5)));
#else
//		q3 = _mm256_add_pd(q3, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x3), _mm256_shuffle_pd(tmp3, tmp3, 0x5)));
#endif
//		tmp4 = _mm256_mul_pd(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
//		q4 = _mm256_add_pd(q4, _mm256_FMADDSUB_pd(h1_real, x4, _mm256_shuffle_pd(tmp4, tmp4, 0x5)));
#else
//		q4 = _mm256_add_pd(q4, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x4), _mm256_shuffle_pd(tmp4, tmp4, 0x5)));
#endif

		_mm256_store_ps(&q_dbl[(2*i*ldq)+0], q1);
		_mm256_store_ps(&q_dbl[(2*i*ldq)+8], q2);
//		_mm256_store_pd(&q_dbl[(2*i*ldq)+8], q3);
//		_mm256_store_pd(&q_dbl[(2*i*ldq)+12], q4);
	}
}

static __forceinline void hh_trafo_complex_kernel_4_AVX_1hv_single(float complex* q, float complex* hh, int nb, int ldq)
{
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;

	__m256 x1, x2;
	__m256 q1, q2;
	__m256 h1_real, h1_imag;
	__m256 tmp1, tmp2;
	int i=0;

	// carefull
//	__m256d sign = (__m256d)_mm256_set_epi64x(0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000);
	__m256 sign = (__m256)_mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);

	x1 = _mm256_load_ps(&q_dbl[0]);    // load q(1,1) | q(2,1) | q(3,1) | q(4,1)
//	x2 = _mm256_load_pd(&q_dbl[4]);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm256_broadcast_ss(&hh_dbl[i*2]);
		h1_imag = _mm256_broadcast_ss(&hh_dbl[(i*2)+1]);
#ifndef __ELPA_USE_FMA__
		// conjugate
		h1_imag = _mm256_xor_ps(h1_imag, sign);
#endif

		q1 = _mm256_load_ps(&q_dbl[(2*i*ldq)+0]);
//		q2 = _mm256_load_pd(&q_dbl[(2*i*ldq)+4]);

		tmp1 = _mm256_mul_ps(h1_imag, q1);

		// carefull
#ifdef __ELPA_USE_FMA__
		x1 = _mm256_add_ps(x1, _mm256_FMSUBADD_ps(h1_real, q1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
		x1 = _mm256_add_ps(x1, _mm256_addsub_ps( _mm256_mul_ps(h1_real, q1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
//		tmp2 = _mm256_mul_pd(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
//		x2 = _mm256_add_pd(x2, _mm256_FMSUBADD_pd(h1_real, q2, _mm256_shuffle_pd(tmp2, tmp2, 0x5)));
#else
//		x2 = _mm256_add_pd(x2, _mm256_addsub_pd( _mm256_mul_pd(h1_real, q2), _mm256_shuffle_pd(tmp2, tmp2, 0x5)));
#endif
	}

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
//	tmp2 = _mm256_mul_pd(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
//	x2 = _mm256_FMADDSUB_pd(h1_real, x2, _mm256_shuffle_pd(tmp2, tmp2, 0x5));
#else
//	x2 = _mm256_addsub_pd( _mm256_mul_pd(h1_real, x2), _mm256_shuffle_pd(tmp2, tmp2, 0x5));
#endif

	q1 = _mm256_load_ps(&q_dbl[0]);
//	q2 = _mm256_load_pd(&q_dbl[4]);

	q1 = _mm256_add_ps(q1, x1);
//	q2 = _mm256_add_pd(q2, x2);

	_mm256_store_ps(&q_dbl[0], q1);
//	_mm256_store_pd(&q_dbl[4], q2);

	for (i = 1; i < nb; i++)
	{
		h1_real = _mm256_broadcast_ss(&hh_dbl[i*2]);
		h1_imag = _mm256_broadcast_ss(&hh_dbl[(i*2)+1]);

		q1 = _mm256_load_ps(&q_dbl[(2*i*ldq)+0]);
//		q2 = _mm256_load_pd(&q_dbl[(2*i*ldq)+4]);

		tmp1 = _mm256_mul_ps(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
		q1 = _mm256_add_ps(q1, _mm256_FMADDSUB_ps(h1_real, x1, _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
		q1 = _mm256_add_ps(q1, _mm256_addsub_ps( _mm256_mul_ps(h1_real, x1), _mm256_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
//		tmp2 = _mm256_mul_pd(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
//		q2 = _mm256_add_pd(q2, _mm256_FMADDSUB_pd(h1_real, x2, _mm256_shuffle_pd(tmp2, tmp2, 0x5)));
#else
//		q2 = _mm256_add_pd(q2, _mm256_addsub_pd( _mm256_mul_pd(h1_real, x2), _mm256_shuffle_pd(tmp2, tmp2, 0x5)));
#endif

		_mm256_store_ps(&q_dbl[(2*i*ldq)+0], q1);
//		_mm256_store_pd(&q_dbl[(2*i*ldq)+4], q2);
	}
}
