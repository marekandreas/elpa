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

#ifdef DOUBLE_PRECISION_COMPLEX
#define offset 2
#define __SSE_DATATYPE __m128d
#define _SSE_LOAD _mm_load_pd
#define _SSE_STORE _mm_store_pd
#define _SSE_MUL _mm_mul_pd
#define _SSE_ADD _mm_add_pd
#define _SSE_XOR _mm_xor_pd
#define _SSE_MADDSUB _mm_maddsub_pd
#define _SSE_ADDSUB _mm_addsub_pd
#define _SSE_SHUFFLE _mm_shuffle_pd
#endif
#ifdef SINGLE_PRECISION_COMPLEX
#define offset 4
#define __SSE_DATATYPE __m128
#define _SSE_LOAD _mm_load_ps
#define _SSE_STORE _mm_store_ps
#define _SSE_MUL _mm_mul_ps
#define _SSE_ADD _mm_add_ps
#define _SSE_XOR _mm_xor_ps
#define _SSE_MADDSUB _mm_maddsub_ps
#define _SSE_ADDSUB _mm_addsub_ps
#define _SSE_SHUFFLE _mm_shuffle_ps
#endif

#define __forceinline __attribute__((always_inline))

#ifdef HAVE_SSE_INTRINSICS
#undef __AVX__
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
//Forward declaration
static __forceinline void hh_trafo_complex_kernel_6_SSE_1hv_double(double complex* q, double complex* hh, int nb, int ldq);
static __forceinline void hh_trafo_complex_kernel_4_SSE_1hv_double(double complex* q, double complex* hh, int nb, int ldq);
static __forceinline void hh_trafo_complex_kernel_2_SSE_1hv_double(double complex* q, double complex* hh, int nb, int ldq);
#endif

#ifdef SINGLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_6_SSE_1hv_single(float complex* q, float complex* hh, int nb, int ldq);
static __forceinline void hh_trafo_complex_kernel_4_SSE_1hv_single(float complex* q, float complex* hh, int nb, int ldq);
static __forceinline void hh_trafo_complex_kernel_2_SSE_1hv_single(float complex* q, float complex* hh, int nb, int ldq);
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
/*
!f>#ifdef WITH_COMPLEX_SSE_BLOCK1_KERNEL
!f> interface
!f>   subroutine single_hh_trafo_complex_sse_1hv_double(q, hh, pnb, pnq, pldq) &
!f>                             bind(C, name="single_hh_trafo_complex_sse_1hv_double")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq
!f>     ! complex(kind=c_double_complex)     :: q(*)
!f>     type(c_ptr), value                   :: q
!f>     complex(kind=c_double_complex)     :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/
#endif

#ifdef SINGLE_PRECISION_COMPLEX
/*
!f>#ifdef HAVE_SSE_INTRINSICS
!f> interface
!f>   subroutine single_hh_trafo_complex_sse_1hv_single(q, hh, pnb, pnq, pldq) &
!f>                             bind(C, name="single_hh_trafo_complex_sse_1hv_single")
!f>     use, intrinsic :: iso_c_binding
!f>     integer(kind=c_int)     :: pnb, pnq, pldq
!f>     ! complex(kind=c_float_complex)   :: q(*)
!f>     type(c_ptr), value                :: q
!f>     complex(kind=c_float_complex)   :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
void single_hh_trafo_complex_sse_1hv_double(double complex* q, double complex* hh, int* pnb, int* pnq, int* pldq)
#endif
#ifdef SINGLE_PRECISION_COMPLEX
void single_hh_trafo_complex_sse_1hv_single(float complex* q, float complex* hh, int* pnb, int* pnq, int* pldq)
#endif
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	//int ldh = *pldh;

	for (i = 0; i < nq-4; i+=6)
	{
#ifdef DOUBLE_PRECISION_COMPLEX
		hh_trafo_complex_kernel_6_SSE_1hv_double(&q[i], hh, nb, ldq);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
		hh_trafo_complex_kernel_6_SSE_1hv_single(&q[i], hh, nb, ldq);
#endif
	}
	if (nq-i == 0) {
	  return;
	} else {

	if (nq-i > 2)
	{
#ifdef DOUBLE_PRECISION_COMPLEX
		hh_trafo_complex_kernel_4_SSE_1hv_double(&q[i], hh, nb, ldq);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
		hh_trafo_complex_kernel_4_SSE_1hv_single(&q[i], hh, nb, ldq);
#endif
	}
	else
	{
#ifdef DOUBLE_PRECISION_COMPLEX
		hh_trafo_complex_kernel_2_SSE_1hv_double(&q[i], hh, nb, ldq);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
		hh_trafo_complex_kernel_2_SSE_1hv_single(&q[i], hh, nb, ldq);
#endif
	}
    }
}

#ifdef DOUBLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_6_SSE_1hv_double(double complex* q, double complex* hh, int nb, int ldq)
#endif
#ifdef SINGLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_6_SSE_1hv_single(float complex* q, float complex* hh, int nb, int ldq)
#endif
{

#ifdef DOUBLE_PRECISION_COMPLEX
	double* q_dbl = (double*)q;
	double* hh_dbl = (double*)hh;
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;
#endif
	__SSE_DATATYPE x1, x2, x3, x4, x5, x6;
	__SSE_DATATYPE q1, q2, q3, q4, q5, q6;
	__SSE_DATATYPE h1_real, h1_imag;
	__SSE_DATATYPE tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
	int i=0;

#ifdef DOUBLE_PRECISION_COMPLEX
	__SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set_epi64x(0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	__SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif

	x1 = _SSE_LOAD(&q_dbl[0]);
	x2 = _SSE_LOAD(&q_dbl[offset]);
	x3 = _SSE_LOAD(&q_dbl[2*offset]);
#ifdef DOUBLE_PRECISION_COMPLEX
	x4 = _SSE_LOAD(&q_dbl[3*offset]);
	x5 = _SSE_LOAD(&q_dbl[4*offset]);
	x6 = _SSE_LOAD(&q_dbl[5*offset]);
#endif
	for (i = 1; i < nb; i++)
	{

#ifdef DOUBLE_PRECISION_COMPLEX
		h1_real = _mm_loaddup_pd(&hh_dbl[i*2]);
		h1_imag = _mm_loaddup_pd(&hh_dbl[(i*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
		h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[i*2]) )));
		h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i*2)+1]) )));
#endif
#ifndef __ELPA_USE_FMA__
		// conjugate
		h1_imag = _SSE_XOR(h1_imag, sign);
#endif

		q1 = _SSE_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _SSE_LOAD(&q_dbl[(2*i*ldq)+offset]);
		q3 = _SSE_LOAD(&q_dbl[(2*i*ldq)+2*offset]);
#ifdef DOUBLE_PRECISION_COMPLEX
		q4 = _SSE_LOAD(&q_dbl[(2*i*ldq)+3*offset]);
		q5 = _SSE_LOAD(&q_dbl[(2*i*ldq)+4*offset]);
		q6 = _SSE_LOAD(&q_dbl[(2*i*ldq)+5*offset]);
#endif

		tmp1 = _SSE_MUL(h1_imag, q1);

#ifdef DOUBLE_PRECISION_COMPLEX

#ifdef __ELPA_USE_FMA__
		x1 = _SSE_ADD(x1, _mm_msubadd_pd(h1_real, q1, _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#else
		x1 = _SSE_ADD(x1, _SSE_ADDSUB( _SSE_MUL(h1_real, q1), _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#endif
		tmp2 = _SSE_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
		x2 = _SSE_ADD(x2, _mm_msubadd_pd(h1_real, q2, _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#else
		x2 = _SSE_ADD(x2, _SSE_ADDSUB( _SSE_MUL(h1_real, q2), _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#endif
		tmp3 = _SSE_MUL(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
		x3 = _SSE_ADD(x3, _mm_msubadd_pd(h1_real, q3, _SSE_SHUFFLE(tmp3, tmp3, _MM_SHUFFLE2(0,1))));
#else
		x3 = _SSE_ADD(x3, _SSE_ADDSUB( _SSE_MUL(h1_real, q3), _SSE_SHUFFLE(tmp3, tmp3, _MM_SHUFFLE2(0,1))));
#endif

#endif /* DOUBLE_PRECISION_COMPEX */

#ifdef SINGLE_PRECISION_COMPLEX
#ifdef __ELPA_USE_FMA__
		x1 = _mm_add_ps(x1, _mm_msubadd_ps(h1_real, q1, _mm_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
		x1 = _mm_add_ps(x1, _mm_addsub_ps( _mm_mul_ps(h1_real, q1), _mm_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
		tmp2 = _mm_mul_ps(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
		x2 = _mm_add_ps(x2, _mm_msubadd_ps(h1_real, q2, _mm_shuffle_ps(tmp2, tmp2, 0xb1)));
#else
		x2 = _mm_add_ps(x2, _mm_addsub_ps( _mm_mul_ps(h1_real, q2), _mm_shuffle_ps(tmp2, tmp2, 0xb1)));
#endif
		tmp3 = _mm_mul_ps(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
		x3 = _mm_add_ps(x3, _mm_msubadd_ps(h1_real, q3, _mm_shuffle_ps(tmp3, tmp3, 0xb1)));
#else
		x3 = _mm_add_ps(x3, _mm_addsub_ps( _mm_mul_ps(h1_real, q3), _mm_shuffle_ps(tmp3, tmp3, 0xb1)));
#endif
#endif /* SINGLE_PRECISION_COMPLEX */


#ifdef DOUBLE_PRECISION_COMPLEX
		tmp4 = _SSE_MUL(h1_imag, q4);
#ifdef __ELPA_USE_FMA__
		x4 = _SSE_ADD(x4, _mm_msubadd_pd(h1_real, q4, _SSE_SHUFFLE(tmp4, tmp4, _MM_SHUFFLE2(0,1))));
#else
		x4 = _SSE_ADD(x4, _SSE_ADDSUB( _SSE_MUL(h1_real, q4), _SSE_SHUFFLE(tmp4, tmp4, _MM_SHUFFLE2(0,1))));
#endif
		tmp5 = _SSE_MUL(h1_imag, q5);
#ifdef __ELPA_USE_FMA__
		x5 = _SSE_ADD(x5, _mm_msubadd_pd(h1_real, q5, _SSE_SHUFFLE(tmp5, tmp5, _MM_SHUFFLE2(0,1))));
#else
		x5 = _SSE_ADD(x5, _SSE_ADDSUB( _SSE_MUL(h1_real, q5), _SSE_SHUFFLE(tmp5, tmp5, _MM_SHUFFLE2(0,1))));
#endif
		tmp6 = _SSE_MUL(h1_imag, q6);
#ifdef __ELPA_USE_FMA__
		x6 = _SSE_ADD(x6, _mm_msubadd_pd(h1_real, q6, _SSE_SHUFFLE(tmp6, tmp6, _MM_SHUFFLE2(0,1))));
#else
		x6 = _SSE_ADD(x6, _SSE_ADDSUB( _SSE_MUL(h1_real, q6), _SSE_SHUFFLE(tmp6, tmp6, _MM_SHUFFLE2(0,1))));
#endif

#endif /* DOUBLE_PRECISION_COMPLEX */
	}

#ifdef DOUBLE_PRECISION_COMPLEX
	h1_real = _mm_loaddup_pd(&hh_dbl[0]);
	h1_imag = _mm_loaddup_pd(&hh_dbl[1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[0]) )));
	h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[1]) )));
#endif
	h1_real = _SSE_XOR(h1_real, sign);
	h1_imag = _SSE_XOR(h1_imag, sign);

	tmp1 = _SSE_MUL(h1_imag, x1);

#ifdef DOUBLE_PRECISION_COMPLEX

#ifdef __ELPA_USE_FMA__
	x1 = _SSE_MADDSUB(h1_real, x1, _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1)));
#else
	x1 = _SSE_ADDSUB( _SSE_MUL(h1_real, x1), _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1)));
#endif
	tmp2 = _SSE_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
	x2 = _SSE_MADDSUB(h1_real, x2, _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1)));
#else
	x2 = _SSE_ADDSUB( _SSE_MUL(h1_real, x2), _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1)));
#endif
	tmp3 = _SSE_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
	x3 = _SSE_MADDSUB(h1_real, x3, _SSE_SHUFFLE(tmp3, tmp3, _MM_SHUFFLE2(0,1)));
#else
	x3 = _SSE_ADDSUB( _SSE_MUL(h1_real, x3), _SSE_SHUFFLE(tmp3, tmp3, _MM_SHUFFLE2(0,1)));
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */

#ifdef SINGLE_PRECISION_COMPLEX
#ifdef __ELPA_USE_FMA__
	x1 = _mm_maddsub_ps(h1_real, x1, _mm_shuffle_ps(tmp1, tmp1, 0xb1));
#else
	x1 = _mm_addsub_ps( _mm_mul_ps(h1_real, x1), _mm_shuffle_ps(tmp1, tmp1, 0xb1));
#endif
	tmp2 = _mm_mul_ps(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
	x2 = _mm_maddsub_ps(h1_real, x2, _mm_shuffle_ps(tmp2, tmp2, 0xb1));
#else
	x2 = _mm_addsub_ps( _mm_mul_ps(h1_real, x2), _mm_shuffle_ps(tmp2, tmp2, 0xb1));
#endif
	tmp3 = _mm_mul_ps(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
	x3 = _mm_maddsub_ps(h1_real, x3, _mm_shuffle_ps(tmp3, tmp3, 0xb1));
#else
	x3 = _mm_addsub_ps( _mm_mul_ps(h1_real, x3), _mm_shuffle_ps(tmp3, tmp3, 0xb1));
#endif

#endif /* SINGLE_PRECISION_COMPLEX */

#ifdef DOUBLE_PRECISION_COMPLEX
	tmp4 = _SSE_MUL(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
	x4 = _SSE_MADDSUB(h1_real, x4, _SSE_SHUFFLE(tmp4, tmp4, _MM_SHUFFLE2(0,1)));
#else
	x4 = _SSE_ADDSUB( _SSE_MUL(h1_real, x4), _SSE_SHUFFLE(tmp4, tmp4, _MM_SHUFFLE2(0,1)));
#endif
	tmp5 = _SSE_MUL(h1_imag, x5);
#ifdef __ELPA_USE_FMA__
	x5 = _SSE_MADDSUB(h1_real, x5, _SSE_SHUFFLE(tmp5, tmp5, _MM_SHUFFLE2(0,1)));
#else
	x5 = _SSE_ADDSUB( _SSE_MUL(h1_real, x5), _SSE_SHUFFLE(tmp5, tmp5, _MM_SHUFFLE2(0,1)));
#endif
	tmp6 = _SSE_MUL(h1_imag, x6);
#ifdef __ELPA_USE_FMA__
	x6 = _SSE_MADDSUB(h1_real, x6, _SSE_SHUFFLE(tmp6, tmp6, _MM_SHUFFLE2(0,1)));
#else
	x6 = _SSE_ADDSUB( _SSE_MUL(h1_real, x6), _SSE_SHUFFLE(tmp6, tmp6, _MM_SHUFFLE2(0,1)));
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */

	q1 = _SSE_LOAD(&q_dbl[0]);
	q2 = _SSE_LOAD(&q_dbl[offset]);
	q3 = _SSE_LOAD(&q_dbl[2*offset]);
#ifdef DOUBLE_PRECISION_COMPLEX 
	q4 = _SSE_LOAD(&q_dbl[3*offset]);
	q5 = _SSE_LOAD(&q_dbl[4*offset]);
	q6 = _SSE_LOAD(&q_dbl[5*offset]);
#endif

	q1 = _SSE_ADD(q1, x1);
	q2 = _SSE_ADD(q2, x2);
	q3 = _SSE_ADD(q3, x3);
#ifdef DOUBLE_PRECISION_COMPLEX 
	q4 = _SSE_ADD(q4, x4);
	q5 = _SSE_ADD(q5, x5);
	q6 = _SSE_ADD(q6, x6);
#endif

	_SSE_STORE(&q_dbl[0], q1);
	_SSE_STORE(&q_dbl[offset], q2);
	_SSE_STORE(&q_dbl[2*offset], q3);
#ifdef DOUBLE_PRECISION_COMPLEX 
	_SSE_STORE(&q_dbl[3*offset], q4);
	_SSE_STORE(&q_dbl[4*offset], q5);
	_SSE_STORE(&q_dbl[5*offset], q6);
#endif
	for (i = 1; i < nb; i++)
	{
#ifdef DOUBLE_PRECISION_COMPLEX
		h1_real = _mm_loaddup_pd(&hh_dbl[i*2]);
		h1_imag = _mm_loaddup_pd(&hh_dbl[(i*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
		h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[i*2]) )));
		h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i*2)+1]) )));
#endif

		q1 = _SSE_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _SSE_LOAD(&q_dbl[(2*i*ldq)+offset]);
		q3 = _SSE_LOAD(&q_dbl[(2*i*ldq)+2*offset]);
#ifdef DOUBLE_PRECISION_COMPLEX
		q4 = _SSE_LOAD(&q_dbl[(2*i*ldq)+3*offset]);
		q5 = _SSE_LOAD(&q_dbl[(2*i*ldq)+4*offset]);
		q6 = _SSE_LOAD(&q_dbl[(2*i*ldq)+5*offset]);
#endif
		tmp1 = _SSE_MUL(h1_imag, x1);
#ifdef DOUBLE_PRECISION_COMPLEX

#ifdef __ELPA_USE_FMA__
		q1 = _SSE_ADD(q1, _SSE_MADDSUB(h1_real, x1, _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#else
		q1 = _SSE_ADD(q1, _SSE_ADDSUB( _SSE_MUL(h1_real, x1), _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#endif
		tmp2 = _SSE_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
		q2 = _SSE_ADD(q2, _SSE_MADDSUB(h1_real, x2, _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#else
		q2 = _SSE_ADD(q2, _SSE_ADDSUB( _SSE_MUL(h1_real, x2), _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#endif
		tmp3 = _SSE_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
		q3 = _SSE_ADD(q3, _SSE_MADDSUB(h1_real, x3, _SSE_SHUFFLE(tmp3, tmp3, _MM_SHUFFLE2(0,1))));
#else
		q3 = _SSE_ADD(q3, _SSE_ADDSUB( _SSE_MUL(h1_real, x3), _SSE_SHUFFLE(tmp3, tmp3, _MM_SHUFFLE2(0,1))));
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */

#ifdef SINGLE_PRECISION_COMPLEX
#ifdef __ELPA_USE_FMA__
		q1 = _mm_add_ps(q1, _mm_maddsub_ps(h1_real, x1, _mm_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
		q1 = _mm_add_ps(q1, _mm_addsub_ps( _mm_mul_ps(h1_real, x1), _mm_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
		tmp2 = _mm_mul_ps(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
		q2 = _mm_add_ps(q2, _mm_maddsub_ps(h1_real, x2, _mm_shuffle_ps(tmp2, tmp2, 0xb1)));
#else
		q2 = _mm_add_ps(q2, _mm_addsub_ps( _mm_mul_ps(h1_real, x2), _mm_shuffle_ps(tmp2, tmp2, 0xb1)));
#endif
		tmp3 = _mm_mul_ps(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
		q3 = _mm_add_ps(q3, _mm_maddsub_ps(h1_real, x3, _mm_shuffle_ps(tmp3, tmp3, 0xb1)));
#else
		q3 = _mm_add_ps(q3, _mm_addsub_ps( _mm_mul_ps(h1_real, x3), _mm_shuffle_ps(tmp3, tmp3, 0xb1)));
#endif
#endif /* SINGLE_PRECISION_COMPLEX */

#ifdef DOUBLE_PRECISION_COMPLEX
		tmp4 = _SSE_MUL(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
		q4 = _SSE_ADD(q4, _SSE_MADDSUB(h1_real, x4, _SSE_SHUFFLE(tmp4, tmp4, _MM_SHUFFLE2(0,1))));
#else
		q4 = _SSE_ADD(q4, _SSE_ADDSUB( _SSE_MUL(h1_real, x4), _SSE_SHUFFLE(tmp4, tmp4, _MM_SHUFFLE2(0,1))));
#endif
		tmp5 = _SSE_MUL(h1_imag, x5);
#ifdef __ELPA_USE_FMA__
		q5 = _SSE_ADD(q5, _SSE_MADDSUB(h1_real, x5, _SSE_SHUFFLE(tmp5, tmp5, _MM_SHUFFLE2(0,1))));
#else
		q5 = _SSE_ADD(q5, _SSE_ADDSUB( _SSE_MUL(h1_real, x5), _SSE_SHUFFLE(tmp5, tmp5, _MM_SHUFFLE2(0,1))));
#endif
		tmp6 = _SSE_MUL(h1_imag, x6);
#ifdef __ELPA_USE_FMA__
		q6 = _SSE_ADD(q6, _SSE_MADDSUB(h1_real, x6, _SSE_SHUFFLE(tmp6, tmp6, _MM_SHUFFLE2(0,1))));
#else
		q6 = _SSE_ADD(q6, _SSE_ADDSUB( _SSE_MUL(h1_real, x6), _SSE_SHUFFLE(tmp6, tmp6, _MM_SHUFFLE2(0,1))));
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */

		_SSE_STORE(&q_dbl[(2*i*ldq)+0], q1);
		_SSE_STORE(&q_dbl[(2*i*ldq)+offset], q2);
		_SSE_STORE(&q_dbl[(2*i*ldq)+2*offset], q3);
#ifdef DOUBLE_PRECISION_COMPLEX
		_SSE_STORE(&q_dbl[(2*i*ldq)+3*offset], q4);
		_SSE_STORE(&q_dbl[(2*i*ldq)+4*offset], q5);
		_SSE_STORE(&q_dbl[(2*i*ldq)+5*offset], q6);
#endif
	}
}

#ifdef DOUBLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_4_SSE_1hv_double(double complex* q, double complex* hh, int nb, int ldq)
#endif
#ifdef SINGLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_4_SSE_1hv_single(float complex* q, float complex* hh, int nb, int ldq)
#endif
{
#ifdef DOUBLE_PRECISION_COMPLEX
	double* q_dbl = (double*)q;
	double* hh_dbl = (double*)hh;
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;
#endif
	__SSE_DATATYPE x1, x2, x3, x4;
	__SSE_DATATYPE q1, q2, q3, q4;
	__SSE_DATATYPE h1_real, h1_imag;
	__SSE_DATATYPE tmp1, tmp2, tmp3, tmp4;
	int i=0;
#ifdef DOUBLE_PRECISION_COMPLEX
	__SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set_epi64x(0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	__SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif

	x1 = _SSE_LOAD(&q_dbl[0]);
	x2 = _SSE_LOAD(&q_dbl[offset]);
#ifdef DOUBLE_PRECISION_COMPLEX
	x3 = _SSE_LOAD(&q_dbl[2*offset]);
	x4 = _SSE_LOAD(&q_dbl[3*offset]);
#endif
	for (i = 1; i < nb; i++)
	{
#ifdef DOUBLE_PRECISION_COMPLEX
		h1_real = _mm_loaddup_pd(&hh_dbl[i*2]);
		h1_imag = _mm_loaddup_pd(&hh_dbl[(i*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
		h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[i*2]) )));
		h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i*2)+1]) )));
#endif
#ifndef __ELPA_USE_FMA__
		// conjugate
		h1_imag = _SSE_XOR(h1_imag, sign);
#endif

		q1 = _SSE_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _SSE_LOAD(&q_dbl[(2*i*ldq)+offset]);
#ifdef DOUBLE_PRECISION_COMPLEX
		q3 = _SSE_LOAD(&q_dbl[(2*i*ldq)+2*offset]);
		q4 = _SSE_LOAD(&q_dbl[(2*i*ldq)+3*offset]);
#endif
		tmp1 = _SSE_MUL(h1_imag, q1);
#ifdef DOUBLE_PRECISION_COMPLEX

#ifdef __ELPA_USE_FMA__
		x1 = _SSE_ADD(x1, _mm_msubadd_pd(h1_real, q1, _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#else
		x1 = _SSE_ADD(x1, _SSE_ADDSUB( _SSE_MUL(h1_real, q1), _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#endif

		tmp2 = _SSE_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
		x2 = _SSE_ADD(x2, _mm_msubadd_pd(h1_real, q2, _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#else
		x2 = _SSE_ADD(x2, _SSE_ADDSUB( _SSE_MUL(h1_real, q2), _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */

#ifdef SINGLE_PRECISION_COMPLEX
#ifdef __ELPA_USE_FMA__
		x1 = _mm_add_ps(x1, _mm_msubadd_ps(h1_real, q1, _mm_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
		x1 = _mm_add_ps(x1, _mm_addsub_ps( _mm_mul_ps(h1_real, q1), _mm_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
		tmp2 = _mm_mul_ps(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
		x2 = _mm_add_ps(x2, _mm_msubadd_ps(h1_real, q2, _mm_shuffle_ps(tmp2, tmp2, 0xb1)));
#else
		x2 = _mm_add_ps(x2, _mm_addsub_ps( _mm_mul_ps(h1_real, q2), _mm_shuffle_ps(tmp2, tmp2, 0xb1)));
#endif
#endif /* SINGLE_PRECISION_COMPLEX */

#ifdef DOUBLE_PRECISION_COMPLEX
		tmp3 = _SSE_MUL(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
		x3 = _SSE_ADD(x3, _mm_msubadd_pd(h1_real, q3, _SSE_SHUFFLE(tmp3, tmp3, _MM_SHUFFLE2(0,1))));
#else
		x3 = _SSE_ADD(x3, _SSE_ADDSUB( _SSE_MUL(h1_real, q3), _SSE_SHUFFLE(tmp3, tmp3, _MM_SHUFFLE2(0,1))));
#endif
		tmp4 = _SSE_MUL(h1_imag, q4);
#ifdef __ELPA_USE_FMA__
		x4 = _SSE_ADD(x4, _mm_msubadd_pd(h1_real, q4, _SSE_SHUFFLE(tmp4, tmp4, _MM_SHUFFLE2(0,1))));
#else
		x4 = _SSE_ADD(x4, _SSE_ADDSUB( _SSE_MUL(h1_real, q4), _SSE_SHUFFLE(tmp4, tmp4, _MM_SHUFFLE2(0,1))));
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */
	}

#ifdef DOUBLE_PRECISION_COMPLEX
	h1_real = _mm_loaddup_pd(&hh_dbl[0]);
	h1_imag = _mm_loaddup_pd(&hh_dbl[1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[0]) )));
	h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[1]) )));
#endif
	h1_real = _SSE_XOR(h1_real, sign);
	h1_imag = _SSE_XOR(h1_imag, sign);

	tmp1 = _SSE_MUL(h1_imag, x1);
#ifdef DOUBLE_PRECISION_COMPLEX
#ifdef __ELPA_USE_FMA__
	x1 = _SSE_MADDSUB(h1_real, x1, _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1)));
#else
	x1 = _SSE_ADDSUB( _SSE_MUL(h1_real, x1), _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1)));
#endif
	tmp2 = _SSE_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
	x2 = _SSE_MADDSUB(h1_real, x2, _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1)));
#else
	x2 = _SSE_ADDSUB( _SSE_MUL(h1_real, x2), _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1)));
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */

#ifdef SINGLE_PRECISION_COMPLEX */
#ifdef __ELPA_USE_FMA__
	x1 = _mm_maddsub_ps(h1_real, x1, _mm_shuffle_ps(tmp1, tmp1, 0xb1));
#else
	x1 = _mm_addsub_ps( _mm_mul_ps(h1_real, x1), _mm_shuffle_ps(tmp1, tmp1, 0xb1));
#endif
	tmp2 = _mm_mul_ps(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
	x2 = _mm_maddsub_ps(h1_real, x2, _mm_shuffle_ps(tmp2, tmp2, 0xb1));
#else
	x2 = _mm_addsub_ps( _mm_mul_ps(h1_real, x2), _mm_shuffle_ps(tmp2, tmp2, 0xb1));
#endif
#endif /* SINGLE_PRECISION_COMPLEX */

#ifdef DOUBLE_PRECISION_COMPLEX
	tmp3 = _SSE_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
	x3 = _SSE_MADDSUB(h1_real, x3, _SSE_SHUFFLE(tmp3, tmp3, _MM_SHUFFLE2(0,1)));
#else
	x3 = _SSE_ADDSUB( _SSE_MUL(h1_real, x3), _SSE_SHUFFLE(tmp3, tmp3, _MM_SHUFFLE2(0,1)));
#endif
	tmp4 = _SSE_MUL(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
	x4 = _SSE_MADDSUB(h1_real, x4, _SSE_SHUFFLE(tmp4, tmp4, _MM_SHUFFLE2(0,1)));
#else
	x4 = _SSE_ADDSUB( _SSE_MUL(h1_real, x4), _SSE_SHUFFLE(tmp4, tmp4, _MM_SHUFFLE2(0,1)));
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */

	q1 = _SSE_LOAD(&q_dbl[0]);
	q2 = _SSE_LOAD(&q_dbl[offset]);
#ifdef DOUBLE_PRECISION_COMPLEX
	q3 = _SSE_LOAD(&q_dbl[2*offset]);
	q4 = _SSE_LOAD(&q_dbl[3*offset]);
#endif
	q1 = _SSE_ADD(q1, x1);
	q2 = _SSE_ADD(q2, x2);
#ifdef DOUBLE_PRECISION_COMPLEX
	q3 = _SSE_ADD(q3, x3);
	q4 = _SSE_ADD(q4, x4);
#endif
	_SSE_STORE(&q_dbl[0], q1);
	_SSE_STORE(&q_dbl[offset], q2);
#ifdef DOUBLE_PRECISION_COMPLEX
	_SSE_STORE(&q_dbl[2*offset], q3);
	_SSE_STORE(&q_dbl[3*offset], q4);
#endif
	for (i = 1; i < nb; i++)
	{
#ifdef DOUBLE_PRECISION_COMPLEX
		h1_real = _mm_loaddup_pd(&hh_dbl[i*2]);
		h1_imag = _mm_loaddup_pd(&hh_dbl[(i*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
		h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[i*2]) )));
		h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i*2)+1]) )));
#endif
		q1 = _SSE_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _SSE_LOAD(&q_dbl[(2*i*ldq)+offset]);
#ifdef DOUBLE_PRECISION_COMPLEX
		q3 = _SSE_LOAD(&q_dbl[(2*i*ldq)+2*offset]);
		q4 = _SSE_LOAD(&q_dbl[(2*i*ldq)+3*offset]);
#endif
		tmp1 = _SSE_MUL(h1_imag, x1);

#ifdef DOUBLE_PRECISION_COMPLEX

#ifdef __ELPA_USE_FMA__
		q1 = _SSE_ADD(q1, _SSE_MADDSUB(h1_real, x1, _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#else
		q1 = _SSE_ADD(q1, _SSE_ADDSUB( _SSE_MUL(h1_real, x1), _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#endif
		tmp2 = _SSE_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
		q2 = _SSE_ADD(q2, _SSE_MADDSUB(h1_real, x2, _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#else
		q2 = _SSE_ADD(q2, _SSE_ADDSUB( _SSE_MUL(h1_real, x2), _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */

#ifdef SINGLE_PRECISION_COMPLEX */
#ifdef __ELPA_USE_FMA__
		q1 = _mm_add_ps(q1, _mm_maddsub_ps(h1_real, x1, _mm_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
		q1 = _mm_add_ps(q1, _mm_addsub_ps( _mm_mul_ps(h1_real, x1), _mm_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
		tmp2 = _mm_mul_ps(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
		q2 = _mm_add_ps(q2, _mm_maddsub_ps(h1_real, x2, _mm_shuffle_ps(tmp2, tmp2, 0xb1)));
#else
		q2 = _mm_add_ps(q2, _mm_addsub_ps( _mm_mul_ps(h1_real, x2), _mm_shuffle_ps(tmp2, tmp2, 0xb1)));
#endif
#endif /* SINGLE_PRECISION_COMPLEX */

#ifdef DOUBLE_PRECISION_COMPLEX
		tmp3 = _SSE_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
		q3 = _SSE_ADD(q3, _SSE_MADDSUB(h1_real, x3, _SSE_SHUFFLE(tmp3, tmp3, _MM_SHUFFLE2(0,1))));
#else
		q3 = _SSE_ADD(q3, _SSE_ADDSUB( _SSE_MUL(h1_real, x3), _SSE_SHUFFLE(tmp3, tmp3, _MM_SHUFFLE2(0,1))));
#endif
		tmp4 = _SSE_MUL(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
		q4 = _SSE_ADD(q4, _SSE_MADDSUB(h1_real, x4, _SSE_SHUFFLE(tmp4, tmp4, _MM_SHUFFLE2(0,1))));
#else
		q4 = _SSE_ADD(q4, _SSE_ADDSUB( _SSE_MUL(h1_real, x4), _SSE_SHUFFLE(tmp4, tmp4, _MM_SHUFFLE2(0,1))));
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */

		_SSE_STORE(&q_dbl[(2*i*ldq)+0], q1);
		_SSE_STORE(&q_dbl[(2*i*ldq)+offset], q2);
#ifdef DOUBLE_PRECISION_COMPLEX
		_SSE_STORE(&q_dbl[(2*i*ldq)+2*offset], q3);
		_SSE_STORE(&q_dbl[(2*i*ldq)+3*offset], q4);
#endif
	}
}

#ifdef DOUBLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_2_SSE_1hv_double(double complex* q, double complex* hh, int nb, int ldq)
#endif
#ifdef SINGLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_2_SSE_1hv_single(float complex* q, float complex* hh, int nb, int ldq)
#endif
{

#ifdef DOUBLE_PRECISION_COMPLEX
	double* q_dbl = (double*)q;
	double* hh_dbl = (double*)hh;
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;
#endif
	__SSE_DATATYPE x1, x2;
	__SSE_DATATYPE q1, q2;
	__SSE_DATATYPE h1_real, h1_imag;
	__SSE_DATATYPE tmp1, tmp2;
	int i=0;

#ifdef DOUBLE_PRECISION_COMPLEX
	__SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set_epi64x(0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	__SSE_DATATYPE sign = (__SSE_DATATYPE)_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif
	x1 = _SSE_LOAD(&q_dbl[0]);
#ifdef DOUBLE_PRECISION_COMPLEX
	x2 = _SSE_LOAD(&q_dbl[offset]);
#endif
	for (i = 1; i < nb; i++)
	{
#ifdef DOUBLE_PRECISION_COMPLEX
		h1_real = _mm_loaddup_pd(&hh_dbl[i*2]);
		h1_imag = _mm_loaddup_pd(&hh_dbl[(i*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
		h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[i*2]) )));
		h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i*2)+1]) )));
#endif
#ifndef __ELPA_USE_FMA__
		// conjugate
		h1_imag = _SSE_XOR(h1_imag, sign);
#endif

		q1 = _SSE_LOAD(&q_dbl[(2*i*ldq)+0]);
#ifdef DOUBLE_PRECISION_COMPLEX
		q2 = _SSE_LOAD(&q_dbl[(2*i*ldq)+offset]);
#endif
		tmp1 = _SSE_MUL(h1_imag, q1);
#ifdef DOUBLE_PRECISION_COMPLEX
#ifdef __ELPA_USE_FMA__
		x1 = _SSE_ADD(x1, _mm_msubadd_pd(h1_real, q1, _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#else
		x1 = _SSE_ADD(x1, _SSE_ADDSUB( _SSE_MUL(h1_real, q1), _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */

#ifdef SINGLE_PRECISION_COMPLEX
#ifdef __ELPA_USE_FMA__
		x1 = _mm_add_ps(x1, _mm_msubadd_ps(h1_real, q1, _mm_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
		x1 = _mm_add_ps(x1, _mm_addsub_ps( _mm_mul_ps(h1_real, q1), _mm_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
#endif /* SINGLE_PRECISION_COMPLEX */

#ifdef DOUBLE_PRECISION_COMPLEX
		tmp2 = _SSE_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
		x2 = _SSE_ADD(x2, _mm_msubadd_pd(h1_real, q2, _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#else
		x2 = _SSE_ADD(x2, _SSE_ADDSUB( _SSE_MUL(h1_real, q2), _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */
	}

#ifdef DOUBLE_PRECISION_COMPLEX
	h1_real = _mm_loaddup_pd(&hh_dbl[0]);
	h1_imag = _mm_loaddup_pd(&hh_dbl[1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[0]) )));
	h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[1]) )));
#endif
	h1_real = _SSE_XOR(h1_real, sign);
	h1_imag = _SSE_XOR(h1_imag, sign);

	tmp1 = _SSE_MUL(h1_imag, x1);
#ifdef DOUBLE_PRECISION_COMPLEX
#ifdef __ELPA_USE_FMA__
	x1 = _SSE_MADDSUB(h1_real, x1, _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1)));
#else
	x1 = _SSE_ADDSUB( _SSE_MUL(h1_real, x1), _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1)));
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */

#ifdef SINGLE_PRECISION_COMPLEX
#ifdef __ELPA_USE_FMA__
	x1 = _mm_maddsub_ps(h1_real, x1, _mm_shuffle_ps(tmp1, tmp1, 0xb1));
#else
	x1 = _mm_addsub_ps( _mm_mul_ps(h1_real, x1), _mm_shuffle_ps(tmp1, tmp1, 0xb1));
#endif
#endif /* SINGLE_PRECISION_COMPLEX */

#ifdef DOUBLE_PRECISION_COMPLEX
	tmp2 = _SSE_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
	x2 = _SSE_MADDSUB(h1_real, x2, _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1)));
#else
	x2 = _SSE_ADDSUB( _SSE_MUL(h1_real, x2), _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1)));
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */
	q1 = _SSE_LOAD(&q_dbl[0]);
#ifdef DOUBLE_PRECISION_COMPLEX
	q2 = _SSE_LOAD(&q_dbl[offset]);
#endif
	q1 = _SSE_ADD(q1, x1);
#ifdef DOUBLE_PRECISION_COMPLEX
	q2 = _SSE_ADD(q2, x2);
#endif
	_SSE_STORE(&q_dbl[0], q1);
#ifdef DOUBLE_PRECISION_COMPLEX
	_SSE_STORE(&q_dbl[offset], q2);
#endif
	for (i = 1; i < nb; i++)
	{
#ifdef DOUBLE_PRECISION_COMPLEX
		h1_real = _mm_loaddup_pd(&hh_dbl[i*2]);
		h1_imag = _mm_loaddup_pd(&hh_dbl[(i*2)+1]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
		h1_real = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[i*2]) )));
		h1_imag = _mm_moveldup_ps(_mm_castpd_ps(_mm_loaddup_pd( (double *)(&hh_dbl[(i*2)+1]) )));
#endif

		q1 = _SSE_LOAD(&q_dbl[(2*i*ldq)+0]);
#ifdef DOUBLE_PRECISION_COMPLEX
		q2 = _SSE_LOAD(&q_dbl[(2*i*ldq)+offset]);
#endif
		tmp1 = _SSE_MUL(h1_imag, x1);

#ifdef DOUBLE_PRECISION_COMPLEX
#ifdef __ELPA_USE_FMA__
		q1 = _SSE_ADD(q1, _SSE_MADDSUB(h1_real, x1, _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#else
		q1 = _SSE_ADD(q1, _SSE_ADDSUB( _SSE_MUL(h1_real, x1), _SSE_SHUFFLE(tmp1, tmp1, _MM_SHUFFLE2(0,1))));
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */

#ifdef SINGLE_PRECISION_COMPLEX
#ifdef __ELPA_USE_FMA__
		q1 = _mm_add_ps(q1, _mm_maddsub_ps(h1_real, x1, _mm_shuffle_ps(tmp1, tmp1, 0xb1)));
#else
		q1 = _mm_add_ps(q1, _mm_addsub_ps( _mm_mul_ps(h1_real, x1), _mm_shuffle_ps(tmp1, tmp1, 0xb1)));
#endif
#endif /* SINGLE_PRECISION_COMPLEX */

#ifdef DOUBLE_PRECISION_COMPLEX
		tmp2 = _SSE_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
		q2 = _SSE_ADD(q2, _SSE_MADDSUB(h1_real, x2, _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#else
		q2 = _SSE_ADD(q2, _SSE_ADDSUB( _SSE_MUL(h1_real, x2), _SSE_SHUFFLE(tmp2, tmp2, _MM_SHUFFLE2(0,1))));
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */
		_SSE_STORE(&q_dbl[(2*i*ldq)+0], q1);
#ifdef DOUBLE_PRECISION_COMPLEX
		_SSE_STORE(&q_dbl[(2*i*ldq)+offset], q2);
#endif
	}
}
