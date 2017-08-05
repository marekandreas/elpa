//    This file is part of ELPA.
//
//    The ELPA library was originally created by the ELPA consortium,
//    consisting of the following organizations:
//
//    - Max Planck Computing and Data Facility (MPCDF), formerly known as
//	Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
//    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
//	Informatik,
//    - Technische Universität München, Lehrstuhl für Informatik mit
//	Schwerpunkt Wissenschaftliches Rechnen ,
//    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
//    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
//	Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
//	and
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
//    along with ELPA.	If not, see <http://www.gnu.org/licenses/>
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
#include <stdio.h>
#include <stdlib.h>

#define __forceinline __attribute__((always_inline))

#ifdef DOUBLE_PRECISION_COMPLEX
#define offset 4
#define __AVX_DATATYPE __m256d
#define _AVX_LOAD _mm256_load_pd
#define _AVX_STORE _mm256_store_pd
#define _AVX_ADD _mm256_add_pd
#define _AVX_MUL _mm256_mul_pd
#define _AVX_ADDSUB _mm256_addsub_pd
#define _AVX_XOR _mm256_xor_pd
#define _AVX_BROADCAST _mm256_broadcast_sd
#define _AVX_SET1 _mm256_set1_pd
#define _AVX_SHUFFLE _mm256_shuffle_pd
#define _SHUFFLE 0x5
#define _CAST _mm256_castpd256_pd128

#ifdef HAVE_AVX2

#ifdef __FMA4__
#define __ELPA_USE_FMA__
#define _mm256_FMADDSUB_pd(a,b,c) _mm256_maddsub_pd(a,b,c)
#define _mm256_FMSUBADD_pd(a,b,c) _mm256_msubadd_pd(a,b,c)
#endif

#ifdef __AVX2__
#define __ELPA_USE_FMA__
#define _mm256_FMADDSUB_pd(a,b,c) _mm256_fmaddsub_pd(a,b,c)
#define _mm256_FMSUBADD_pd(a,b,c) _mm256_fmsubadd_pd(a,b,c)
#endif

#define _AVX_FMADDSUB _mm256_FMADDSUB_pd
#define _AVX_FMSUBADD _mm256_FMSUBADD_pd
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */

#ifdef SINGLE_PRECISION_COMPLEX
#define offset 8
#define __AVX_DATATYPE __m256
#define _AVX_LOAD _mm256_load_ps
#define _AVX_STORE _mm256_store_ps
#define _AVX_ADD _mm256_add_ps
#define _AVX_MUL _mm256_mul_ps
#define _AVX_ADDSUB _mm256_addsub_ps
#define _AVX_XOR _mm256_xor_ps
#define _AVX_BROADCAST _mm256_broadcast_ss
#define _AVX_SET1 _mm256_set1_ps
#define _AVX_SHUFFLE _mm256_shuffle_ps
#define _SHUFFLE 0xb1
#define _CAST _mm256_castps256_ps128
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

#define _AVX_FMADDSUB _mm256_FMADDSUB_ps
#define _AVX_FMSUBADD _mm256_FMSUBADD_ps
#endif
#endif /* SINGLE_PRECISION_COMPLEX */

#ifdef DOUBLE_PRECISION_COMPLEX
//Forward declaration
static __forceinline void hh_trafo_complex_kernel_8_AVX_2hv_double(double complex* q, double complex* hh, int nb, int ldq, int ldh, double complex s);
static __forceinline void hh_trafo_complex_kernel_6_AVX_2hv_double(double complex* q, double complex* hh, int nb, int ldq, int ldh, double complex s);
static __forceinline void hh_trafo_complex_kernel_4_AVX_2hv_double(double complex* q, double complex* hh, int nb, int ldq, int ldh, double complex s);
static __forceinline void hh_trafo_complex_kernel_2_AVX_2hv_double(double complex* q, double complex* hh, int nb, int ldq, int ldh, double complex s);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
//Forward declaration
static __forceinline void hh_trafo_complex_kernel_16_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s, float complex s1);
static __forceinline void hh_trafo_complex_kernel_12_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s, float complex s1);
static __forceinline void hh_trafo_complex_kernel_8_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s, float complex s1);
static __forceinline void hh_trafo_complex_kernel_4_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s, float complex s1);
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
/*
!f>#if defined(HAVE_AVX) || defined(HAVE_AVX2)
!f> interface
!f>   subroutine double_hh_trafo_complex_avx_avx2_2hv_double(q, hh, pnb, pnq, pldq, pldh) &
!f>				bind(C, name="double_hh_trafo_complex_avx_avx2_2hv_double")
!f>	use, intrinsic :: iso_c_binding
!f>	integer(kind=c_int)	:: pnb, pnq, pldq, pldh
!f>	! complex(kind=c_double_complex)     :: q(*)
!f>	type(c_ptr), value		     :: q
!f>	complex(kind=c_double_complex)	   :: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/
#endif
#ifdef SINGLE_PRECISION_COMPLEX
/*
!f>#if defined(HAVE_AVX) || defined(HAVE_AVX2)
!f> interface
!f>   subroutine double_hh_trafo_complex_avx_avx2_2hv_single(q, hh, pnb, pnq, pldq, pldh) &
!f>				bind(C, name="double_hh_trafo_complex_avx_avx2_2hv_single")
!f>	use, intrinsic :: iso_c_binding
!f>	integer(kind=c_int)	:: pnb, pnq, pldq, pldh
!f>	! complex(kind=c_float_complex)   :: q(*)
!f>	type(c_ptr), value		  :: q
!f>	complex(kind=c_float_complex)	:: hh(pnb,2)
!f>   end subroutine
!f> end interface
!f>#endif
*/
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
void double_hh_trafo_complex_avx_avx2_2hv_double(double complex* q, double complex* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
#ifdef SINGLE_PRECISION_COMPLEX
void double_hh_trafo_complex_avx_avx2_2hv_single(float complex* q, float complex* hh, int* pnb, int* pnq, int* pldq, int* pldh)
#endif
{
	int i;
	int nb = *pnb;
	int nq = *pldq;
	int ldq = *pldq;
	int ldh = *pldh;
	int worked_on;

	worked_on = 0;

#ifdef DOUBLE_PRECISION_COMPLEX
	double complex s = conj(hh[(ldh)+1])*1.0;
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	float complex s = conj(hh[(ldh)+1])*1.0f;
#endif

	for (i = 2; i < nb; i++)
	{
		s += hh[i-1] * conj(hh[(i+ldh)]);
	}

#ifdef DOUBLE_PRECISION_COMPLEX
	for (i = 0; i < nq-6; i+=8)
	{
		hh_trafo_complex_kernel_8_AVX_2hv_double(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 8;
	}
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	for (i = 0; i < nq-12; i+=16)
	{
		hh_trafo_complex_kernel_16_AVX_2hv_single(&q[i], hh, nb, ldq, ldh, s , s);
		worked_on += 16;
	}
#endif
	if (nq-i == 0) {
	  return;
	}
#ifdef DOUBLE_PRECISION_COMPLEX
	if (nq-i == 6) {
		hh_trafo_complex_kernel_6_AVX_2hv_double(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 6;
	}
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	if (nq-i == 12) {
		hh_trafo_complex_kernel_12_AVX_2hv_single(&q[i], hh, nb, ldq, ldh, s, s);
		worked_on += 12;
	}
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
	if (nq-i == 4) {
		hh_trafo_complex_kernel_4_AVX_2hv_double(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 4;
	}
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	if (nq-i == 8) {
		hh_trafo_complex_kernel_8_AVX_2hv_single(&q[i], hh, nb, ldq, ldh, s, s);
		worked_on += 8;
	}
#endif
#ifdef DOUBLE_PRECISION_COMPLEX
	if (nq-i == 2) {
		hh_trafo_complex_kernel_2_AVX_2hv_double(&q[i], hh, nb, ldq, ldh, s);
		worked_on += 2;
	}
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	if (nq-i == 4) {
		hh_trafo_complex_kernel_4_AVX_2hv_single(&q[i], hh, nb, ldq, ldh, s, s);
		worked_on += 4;
	}
#endif
#ifdef WITH_DEBUG
	if (worked_on != nq) {
		printf("Error in complex avx-avx2 BLOCK 2 kernel \n");
		abort();
	}
#endif
}

#ifdef DOUBLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_8_AVX_2hv_double(double complex* q, double complex* hh, int nb, int ldq, int ldh, double complex s)
#endif
#ifdef SINGLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_16_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s, float complex s1)
#endif
{

#ifdef DOUBLE_PRECISION_COMPLEX
	double* q_dbl = (double*)q;
	double* hh_dbl = (double*)hh;
	double* s_dbl = (double*)(&s);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;
	float* s_dbl = (float*)(&s);
#endif
	__AVX_DATATYPE x1, x2, x3, x4;
	__AVX_DATATYPE y1, y2, y3, y4;
	__AVX_DATATYPE q1, q2, q3, q4;
	__AVX_DATATYPE h1_real, h1_imag, h2_real, h2_imag;
	__AVX_DATATYPE tmp1, tmp2, tmp3, tmp4;
	int i=0;

#ifdef DOUBLE_PRECISION_COMPLEX
	__AVX_DATATYPE sign = (__AVX_DATATYPE)_mm256_set_epi64x(0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	__AVX_DATATYPE sign = (__AVX_DATATYPE)_mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif

	x1 = _AVX_LOAD(&q_dbl[(2*ldq)+0]);
	x2 = _AVX_LOAD(&q_dbl[(2*ldq)+offset]);
	x3 = _AVX_LOAD(&q_dbl[(2*ldq)+2*offset]);
	x4 = _AVX_LOAD(&q_dbl[(2*ldq)+3*offset]);
	h2_real = _AVX_BROADCAST(&hh_dbl[(ldh+1)*2]);
	h2_imag = _AVX_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);
#ifndef __ELPA_USE_FMA__
	// conjugate
	h2_imag = _AVX_XOR(h2_imag, sign);
#endif

	y1 = _AVX_LOAD(&q_dbl[0]);
	y2 = _AVX_LOAD(&q_dbl[offset]);
	y3 = _AVX_LOAD(&q_dbl[2*offset]);
	y4 = _AVX_LOAD(&q_dbl[3*offset]);

	tmp1 = _AVX_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_ADD(y1, _AVX_FMSUBADD(h2_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	y1 = _AVX_ADD(y1, _AVX_ADDSUB( _AVX_MUL(h2_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
	tmp2 = _AVX_MUL(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
	y2 = _AVX_ADD(y2, _AVX_FMSUBADD(h2_real, x2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
	y2 = _AVX_ADD(y2, _AVX_ADDSUB( _AVX_MUL(h2_real, x2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

	tmp3 = _AVX_MUL(h2_imag, x3);
#ifdef __ELPA_USE_FMA__
	y3 = _AVX_ADD(y3, _AVX_FMSUBADD(h2_real, x3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
	y3 = _AVX_ADD(y3, _AVX_ADDSUB( _AVX_MUL(h2_real, x3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
	tmp4 = _AVX_MUL(h2_imag, x4);
#ifdef __ELPA_USE_FMA__
	y4 = _AVX_ADD(y4, _AVX_FMSUBADD(h2_real, x4, _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
	y4 = _AVX_ADD(y4, _AVX_ADDSUB( _AVX_MUL(h2_real, x4), _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

	for (i = 2; i < nb; i++)
	{
		q1 = _AVX_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _AVX_LOAD(&q_dbl[(2*i*ldq)+offset]);
		q3 = _AVX_LOAD(&q_dbl[(2*i*ldq)+2*offset]);
		q4 = _AVX_LOAD(&q_dbl[(2*i*ldq)+3*offset]);

		h1_real = _AVX_BROADCAST(&hh_dbl[(i-1)*2]);
		h1_imag = _AVX_BROADCAST(&hh_dbl[((i-1)*2)+1]);
#ifndef __ELPA_USE_FMA__
		// conjugate
		h1_imag = _AVX_XOR(h1_imag, sign);
#endif

		tmp1 = _AVX_MUL(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
		x1 = _AVX_ADD(x1, _AVX_FMSUBADD(h1_real, q1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
		x1 = _AVX_ADD(x1, _AVX_ADDSUB( _AVX_MUL(h1_real, q1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
		tmp2 = _AVX_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
		x2 = _AVX_ADD(x2, _AVX_FMSUBADD(h1_real, q2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
		x2 = _AVX_ADD(x2, _AVX_ADDSUB( _AVX_MUL(h1_real, q2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

		tmp3 = _AVX_MUL(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
		x3 = _AVX_ADD(x3, _AVX_FMSUBADD(h1_real, q3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
		x3 = _AVX_ADD(x3, _AVX_ADDSUB( _AVX_MUL(h1_real, q3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
		tmp4 = _AVX_MUL(h1_imag, q4);
#ifdef __ELPA_USE_FMA__
		x4 = _AVX_ADD(x4, _AVX_FMSUBADD(h1_real, q4, _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
		x4 = _AVX_ADD(x4, _AVX_ADDSUB( _AVX_MUL(h1_real, q4), _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

		h2_real = _AVX_BROADCAST(&hh_dbl[(ldh+i)*2]);
		h2_imag = _AVX_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);
#ifndef __ELPA_USE_FMA__
		// conjugate
		h2_imag = _AVX_XOR(h2_imag, sign);
#endif

		tmp1 = _AVX_MUL(h2_imag, q1);
#ifdef __ELPA_USE_FMA__
		y1 = _AVX_ADD(y1, _AVX_FMSUBADD(h2_real, q1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
		y1 = _AVX_ADD(y1, _AVX_ADDSUB( _AVX_MUL(h2_real, q1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
		tmp2 = _AVX_MUL(h2_imag, q2);
#ifdef __ELPA_USE_FMA__
		y2 = _AVX_ADD(y2, _AVX_FMSUBADD(h2_real, q2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
		y2 = _AVX_ADD(y2, _AVX_ADDSUB( _AVX_MUL(h2_real, q2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

		tmp3 = _AVX_MUL(h2_imag, q3);
#ifdef __ELPA_USE_FMA__
		y3 = _AVX_ADD(y3, _AVX_FMSUBADD(h2_real, q3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
		y3 = _AVX_ADD(y3, _AVX_ADDSUB( _AVX_MUL(h2_real, q3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
		tmp4 = _AVX_MUL(h2_imag, q4);
#ifdef __ELPA_USE_FMA__
		y4 = _AVX_ADD(y4, _AVX_FMSUBADD(h2_real, q4, _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
		y4 = _AVX_ADD(y4, _AVX_ADDSUB( _AVX_MUL(h2_real, q4), _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif
	}

	h1_real = _AVX_BROADCAST(&hh_dbl[(nb-1)*2]);
	h1_imag = _AVX_BROADCAST(&hh_dbl[((nb-1)*2)+1]);
#ifndef __ELPA_USE_FMA__
	// conjugate
	h1_imag = _AVX_XOR(h1_imag, sign);
#endif

	q1 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+0]);
	q2 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+offset]);
	q3 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+2*offset]);
	q4 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+3*offset]);

	tmp1 = _AVX_MUL(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_ADD(x1, _AVX_FMSUBADD(h1_real, q1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	x1 = _AVX_ADD(x1, _AVX_ADDSUB( _AVX_MUL(h1_real, q1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
	tmp2 = _AVX_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
	x2 = _AVX_ADD(x2, _AVX_FMSUBADD(h1_real, q2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
	x2 = _AVX_ADD(x2, _AVX_ADDSUB( _AVX_MUL(h1_real, q2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

	tmp3 = _AVX_MUL(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
	x3 = _AVX_ADD(x3, _AVX_FMSUBADD(h1_real, q3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
	x3 = _AVX_ADD(x3, _AVX_ADDSUB( _AVX_MUL(h1_real, q3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
	tmp4 = _AVX_MUL(h1_imag, q4);
#ifdef __ELPA_USE_FMA__
	x4 = _AVX_ADD(x4, _AVX_FMSUBADD(h1_real, q4, _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
	x4 = _AVX_ADD(x4, _AVX_ADDSUB( _AVX_MUL(h1_real, q4), _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

	h1_real = _AVX_BROADCAST(&hh_dbl[0]);
	h1_imag = _AVX_BROADCAST(&hh_dbl[1]);
	h1_real = _AVX_XOR(h1_real, sign);
	h1_imag = _AVX_XOR(h1_imag, sign);

	tmp1 = _AVX_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMADDSUB(h1_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
	x1 = _AVX_ADDSUB( _AVX_MUL(h1_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif
	tmp2 = _AVX_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
	x2 = _AVX_FMADDSUB(h1_real, x2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#else
	x2 = _AVX_ADDSUB( _AVX_MUL(h1_real, x2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#endif

	tmp3 = _AVX_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
	x3 = _AVX_FMADDSUB(h1_real, x3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#else
	x3 = _AVX_ADDSUB( _AVX_MUL(h1_real, x3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#endif
	tmp4 = _AVX_MUL(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
	x4 = _AVX_FMADDSUB(h1_real, x4, _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE));
#else
	x4 = _AVX_ADDSUB( _AVX_MUL(h1_real, x4), _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE));
#endif

	h1_real = _AVX_BROADCAST(&hh_dbl[ldh*2]);
	h1_imag = _AVX_BROADCAST(&hh_dbl[(ldh*2)+1]);
	h2_real = _AVX_BROADCAST(&hh_dbl[ldh*2]);
	h2_imag = _AVX_BROADCAST(&hh_dbl[(ldh*2)+1]);

	h1_real = _AVX_XOR(h1_real, sign);
	h1_imag = _AVX_XOR(h1_imag, sign);
	h2_real = _AVX_XOR(h2_real, sign);
	h2_imag = _AVX_XOR(h2_imag, sign);

#ifdef DOUBLE_PRECISION_COMPLEX
	tmp2 = _mm256_set_pd(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	tmp2 = _mm256_set_ps(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0],
			     s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif

	tmp1 = _AVX_MUL(h2_imag, tmp2);
#ifdef __ELPA_USE_FMA__
	tmp2 = _AVX_FMADDSUB(h2_real, tmp2, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
	tmp2 = _AVX_ADDSUB( _AVX_MUL(h2_real, tmp2), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

	h2_real = _AVX_SET1(tmp2[0]);
	h2_imag = _AVX_SET1(tmp2[1]);

	tmp1 = _AVX_MUL(h1_imag, y1);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMADDSUB(h1_real, y1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
	y1 = _AVX_ADDSUB( _AVX_MUL(h1_real, y1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif
	tmp2 = _AVX_MUL(h1_imag, y2);
#ifdef __ELPA_USE_FMA__
	y2 = _AVX_FMADDSUB(h1_real, y2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#else
	y2 = _AVX_ADDSUB( _AVX_MUL(h1_real, y2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#endif

	tmp3 = _AVX_MUL(h1_imag, y3);
#ifdef __ELPA_USE_FMA__
	y3 = _AVX_FMADDSUB(h1_real, y3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#else
	y3 = _AVX_ADDSUB( _AVX_MUL(h1_real, y3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#endif
	tmp4 = _AVX_MUL(h1_imag, y4);
#ifdef __ELPA_USE_FMA__
	y4 = _AVX_FMADDSUB(h1_real, y4, _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE));
#else
	y4 = _AVX_ADDSUB( _AVX_MUL(h1_real, y4), _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE));
#endif

	tmp1 = _AVX_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_ADD(y1, _AVX_FMADDSUB(h2_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	y1 = _AVX_ADD(y1, _AVX_ADDSUB( _AVX_MUL(h2_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
	tmp2 = _AVX_MUL(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
	y2 = _AVX_ADD(y2, _AVX_FMADDSUB(h2_real, x2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
	y2 = _AVX_ADD(y2, _AVX_ADDSUB( _AVX_MUL(h2_real, x2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

	tmp3 = _AVX_MUL(h2_imag, x3);
#ifdef __ELPA_USE_FMA__
	y3 = _AVX_ADD(y3, _AVX_FMADDSUB(h2_real, x3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
	y3 = _AVX_ADD(y3, _AVX_ADDSUB( _AVX_MUL(h2_real, x3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
	tmp4 = _AVX_MUL(h2_imag, x4);
#ifdef __ELPA_USE_FMA__
	y4 = _AVX_ADD(y4, _AVX_FMADDSUB(h2_real, x4, _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
	y4 = _AVX_ADD(y4, _AVX_ADDSUB( _AVX_MUL(h2_real, x4), _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

	q1 = _AVX_LOAD(&q_dbl[0]);
	q2 = _AVX_LOAD(&q_dbl[offset]);
	q3 = _AVX_LOAD(&q_dbl[2*offset]);
	q4 = _AVX_LOAD(&q_dbl[3*offset]);

	q1 = _AVX_ADD(q1, y1);
	q2 = _AVX_ADD(q2, y2);
	q3 = _AVX_ADD(q3, y3);
	q4 = _AVX_ADD(q4, y4);


	_AVX_STORE(&q_dbl[0], q1);
	_AVX_STORE(&q_dbl[offset], q2);
	_AVX_STORE(&q_dbl[2*offset], q3);
	_AVX_STORE(&q_dbl[3*offset], q4);

	h2_real = _AVX_BROADCAST(&hh_dbl[(ldh+1)*2]);
	h2_imag = _AVX_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);

	q1 = _AVX_LOAD(&q_dbl[(ldq*2)+0]);
	q2 = _AVX_LOAD(&q_dbl[(ldq*2)+offset]);
	q3 = _AVX_LOAD(&q_dbl[(ldq*2)+2*offset]);
	q4 = _AVX_LOAD(&q_dbl[(ldq*2)+3*offset]);

	q1 = _AVX_ADD(q1, x1);
	q2 = _AVX_ADD(q2, x2);
	q3 = _AVX_ADD(q3, x3);
	q4 = _AVX_ADD(q4, x4);

	tmp1 = _AVX_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_ADD(q1, _AVX_FMADDSUB(h2_real, y1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	q1 = _AVX_ADD(q1, _AVX_ADDSUB( _AVX_MUL(h2_real, y1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
	tmp2 = _AVX_MUL(h2_imag, y2);
#ifdef __ELPA_USE_FMA_
	q2 = _AVX_ADD(q2, _AVX_FMADDSUB(h2_real, y2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
	q2 = _AVX_ADD(q2, _AVX_ADDSUB( _AVX_MUL(h2_real, y2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

	tmp3 = _AVX_MUL(h2_imag, y3);
#ifdef __ELPA_USE_FMA__
	q3 = _AVX_ADD(q3, _AVX_FMADDSUB(h2_real, y3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
	q3 = _AVX_ADD(q3, _AVX_ADDSUB( _AVX_MUL(h2_real, y3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
	tmp4 = _AVX_MUL(h2_imag, y4);
#ifdef __ELPA_USE_FMA__
	q4 = _AVX_ADD(q4, _AVX_FMADDSUB(h2_real, y4, _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
	q4 = _AVX_ADD(q4, _AVX_ADDSUB( _AVX_MUL(h2_real, y4), _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

	_AVX_STORE(&q_dbl[(ldq*2)+0], q1);
	_AVX_STORE(&q_dbl[(ldq*2)+offset], q2);
	_AVX_STORE(&q_dbl[(ldq*2)+2*offset], q3);
	_AVX_STORE(&q_dbl[(ldq*2)+3*offset], q4);

	for (i = 2; i < nb; i++)
	{

		q1 = _AVX_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _AVX_LOAD(&q_dbl[(2*i*ldq)+offset]);
		q3 = _AVX_LOAD(&q_dbl[(2*i*ldq)+2*offset]);
		q4 = _AVX_LOAD(&q_dbl[(2*i*ldq)+3*offset]);

		h1_real = _AVX_BROADCAST(&hh_dbl[(i-1)*2]);
		h1_imag = _AVX_BROADCAST(&hh_dbl[((i-1)*2)+1]);

		tmp1 = _AVX_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_ADD(q1, _AVX_FMADDSUB(h1_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
		q1 = _AVX_ADD(q1, _AVX_ADDSUB( _AVX_MUL(h1_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
		tmp2 = _AVX_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
		q2 = _AVX_ADD(q2, _AVX_FMADDSUB(h1_real, x2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
		q2 = _AVX_ADD(q2, _AVX_ADDSUB( _AVX_MUL(h1_real, x2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

		tmp3 = _AVX_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
		q3 = _AVX_ADD(q3, _AVX_FMADDSUB(h1_real, x3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
		q3 = _AVX_ADD(q3, _AVX_ADDSUB( _AVX_MUL(h1_real, x3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
		tmp4 = _AVX_MUL(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
		q4 = _AVX_ADD(q4, _AVX_FMADDSUB(h1_real, x4, _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
		q4 = _AVX_ADD(q4, _AVX_ADDSUB( _AVX_MUL(h1_real, x4), _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

		h2_real = _AVX_BROADCAST(&hh_dbl[(ldh+i)*2]);
		h2_imag = _AVX_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _AVX_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_ADD(q1, _AVX_FMADDSUB(h2_real, y1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
		q1 = _AVX_ADD(q1, _AVX_ADDSUB( _AVX_MUL(h2_real, y1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
		tmp2 = _AVX_MUL(h2_imag, y2);
#ifdef __ELPA_USE_FMA__
		q2 = _AVX_ADD(q2, _AVX_FMADDSUB(h2_real, y2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
		q2 = _AVX_ADD(q2, _AVX_ADDSUB( _AVX_MUL(h2_real, y2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

		tmp3 = _AVX_MUL(h2_imag, y3);
#ifdef __ELPA_USE_FMA__
		q3 = _AVX_ADD(q3, _AVX_FMADDSUB(h2_real, y3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
		q3 = _AVX_ADD(q3, _AVX_ADDSUB( _AVX_MUL(h2_real, y3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
		tmp4 = _AVX_MUL(h2_imag, y4);
#ifdef __ELPA_USE_FMA__
		q4 = _AVX_ADD(q4, _AVX_FMADDSUB(h2_real, y4, _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
		q4 = _AVX_ADD(q4, _AVX_ADDSUB( _AVX_MUL(h2_real, y4), _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

		_AVX_STORE(&q_dbl[(2*i*ldq)+0], q1);
		_AVX_STORE(&q_dbl[(2*i*ldq)+offset], q2);
		_AVX_STORE(&q_dbl[(2*i*ldq)+2*offset], q3);
		_AVX_STORE(&q_dbl[(2*i*ldq)+3*offset], q4);
	}
	h1_real = _AVX_BROADCAST(&hh_dbl[(nb-1)*2]);
	h1_imag = _AVX_BROADCAST(&hh_dbl[((nb-1)*2)+1]);

	q1 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+0]);
	q2 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+offset]);
	q3 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+2*offset]);
	q4 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+3*offset]);

	tmp1 = _AVX_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_ADD(q1, _AVX_FMADDSUB(h1_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	q1 = _AVX_ADD(q1, _AVX_ADDSUB( _AVX_MUL(h1_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
	tmp2 = _AVX_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
	q2 = _AVX_ADD(q2, _AVX_FMADDSUB(h1_real, x2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
	q2 = _AVX_ADD(q2, _AVX_ADDSUB( _AVX_MUL(h1_real, x2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

	tmp3 = _AVX_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
	q3 = _AVX_ADD(q3, _AVX_FMADDSUB(h1_real, x3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
	q3 = _AVX_ADD(q3, _AVX_ADDSUB( _AVX_MUL(h1_real, x3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
	tmp4 = _AVX_MUL(h1_imag, x4);
#ifdef __ELPA_USE_FMA__
	q4 = _AVX_ADD(q4, _AVX_FMADDSUB(h1_real, x4, _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#else
	q4 = _AVX_ADD(q4, _AVX_ADDSUB( _AVX_MUL(h1_real, x4), _AVX_SHUFFLE(tmp4, tmp4, _SHUFFLE)));
#endif

	_AVX_STORE(&q_dbl[(2*nb*ldq)+0], q1);
	_AVX_STORE(&q_dbl[(2*nb*ldq)+offset], q2);
	_AVX_STORE(&q_dbl[(2*nb*ldq)+2*offset], q3);
	_AVX_STORE(&q_dbl[(2*nb*ldq)+3*offset], q4);
}

#ifdef DOUBLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_6_AVX_2hv_double(double complex* q, double complex* hh, int nb, int ldq, int ldh, double complex s)
#endif
#ifdef SINGLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_12_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s, float complex s1)
#endif

{
#ifdef DOUBLE_PRECISION_COMPLEX
	double* q_dbl = (double*)q;
	double* hh_dbl = (double*)hh;
	double* s_dbl = (double*)(&s);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;
	float* s_dbl = (float*)(&s);
#endif
	__AVX_DATATYPE x1, x2, x3;
	__AVX_DATATYPE y1, y2, y3;
	__AVX_DATATYPE q1, q2, q3;
	__AVX_DATATYPE h1_real, h1_imag, h2_real, h2_imag;
	__AVX_DATATYPE tmp1, tmp2, tmp3;
	int i=0;

#ifdef DOUBLE_PRECISION_COMPLEX
	__AVX_DATATYPE sign = (__AVX_DATATYPE)_mm256_set_epi64x(0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	__AVX_DATATYPE sign = (__AVX_DATATYPE)_mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif
	x1 = _AVX_LOAD(&q_dbl[(2*ldq)+0]);
	x2 = _AVX_LOAD(&q_dbl[(2*ldq)+offset]);
	x3 = _AVX_LOAD(&q_dbl[(2*ldq)+2*offset]);

	h2_real = _AVX_BROADCAST(&hh_dbl[(ldh+1)*2]);
	h2_imag = _AVX_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);
#ifndef __ELPA_USE_FMA__
	// conjugate
	h2_imag = _AVX_XOR(h2_imag, sign);
#endif

	y1 = _AVX_LOAD(&q_dbl[0]);
	y2 = _AVX_LOAD(&q_dbl[offset]);
	y3 = _AVX_LOAD(&q_dbl[2*offset]);

	tmp1 = _AVX_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_ADD(y1, _AVX_FMSUBADD(h2_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	y1 = _AVX_ADD(y1, _AVX_ADDSUB( _AVX_MUL(h2_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
	tmp2 = _AVX_MUL(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
	y2 = _AVX_ADD(y2, _AVX_FMSUBADD(h2_real, x2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
	y2 = _AVX_ADD(y2, _AVX_ADDSUB( _AVX_MUL(h2_real, x2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif
	tmp3 = _AVX_MUL(h2_imag, x3);
#ifdef __ELPA_USE_FMA__
	y3 = _AVX_ADD(y3, _AVX_FMSUBADD(h2_real, x3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
	y3 = _AVX_ADD(y3, _AVX_ADDSUB( _AVX_MUL(h2_real, x3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

	for (i = 2; i < nb; i++)
	{
		q1 = _AVX_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _AVX_LOAD(&q_dbl[(2*i*ldq)+offset]);
		q3 = _AVX_LOAD(&q_dbl[(2*i*ldq)+2*offset]);

		h1_real = _AVX_BROADCAST(&hh_dbl[(i-1)*2]);
		h1_imag = _AVX_BROADCAST(&hh_dbl[((i-1)*2)+1]);
#ifndef __ELPA_USE_FMA__
		// conjugate
		h1_imag = _AVX_XOR(h1_imag, sign);
#endif

		tmp1 = _AVX_MUL(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
		x1 = _AVX_ADD(x1, _AVX_FMSUBADD(h1_real, q1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
		x1 = _AVX_ADD(x1, _AVX_ADDSUB( _AVX_MUL(h1_real, q1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
		tmp2 = _AVX_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
		x2 = _AVX_ADD(x2, _AVX_FMSUBADD(h1_real, q2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
		x2 = _AVX_ADD(x2, _AVX_ADDSUB( _AVX_MUL(h1_real, q2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif
		tmp3 = _AVX_MUL(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
		x3 = _AVX_ADD(x3, _AVX_FMSUBADD(h1_real, q3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
		x3 = _AVX_ADD(x3, _AVX_ADDSUB( _AVX_MUL(h1_real, q3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

		h2_real = _AVX_BROADCAST(&hh_dbl[(ldh+i)*2]);
		h2_imag = _AVX_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);
#ifndef __ELPA_USE_FMA__
		// conjugate
		h2_imag = _AVX_XOR(h2_imag, sign);
#endif

		tmp1 = _AVX_MUL(h2_imag, q1);
#ifdef __ELPA_USE_FMA__
		y1 = _AVX_ADD(y1, _AVX_FMSUBADD(h2_real, q1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
		y1 = _AVX_ADD(y1, _AVX_ADDSUB( _AVX_MUL(h2_real, q1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
		tmp2 = _AVX_MUL(h2_imag, q2);
#ifdef __ELPA_USE_FMA__
		y2 = _AVX_ADD(y2, _AVX_FMSUBADD(h2_real, q2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
		y2 = _AVX_ADD(y2, _AVX_ADDSUB( _AVX_MUL(h2_real, q2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif
		tmp3 = _AVX_MUL(h2_imag, q3);
#ifdef __ELPA_USE_FMA__
		y3 = _AVX_ADD(y3, _AVX_FMSUBADD(h2_real, q3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
		y3 = _AVX_ADD(y3, _AVX_ADDSUB( _AVX_MUL(h2_real, q3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif
	}

	h1_real = _AVX_BROADCAST(&hh_dbl[(nb-1)*2]);
	h1_imag = _AVX_BROADCAST(&hh_dbl[((nb-1)*2)+1]);
#ifndef __ELPA_USE_FMA__
	// conjugate
	h1_imag = _AVX_XOR(h1_imag, sign);
#endif

	q1 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+0]);
	q2 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+offset]);
	q3 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+2*offset]);

	tmp1 = _AVX_MUL(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_ADD(x1, _AVX_FMSUBADD(h1_real, q1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	x1 = _AVX_ADD(x1, _AVX_ADDSUB( _AVX_MUL(h1_real, q1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
	tmp2 = _AVX_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
	x2 = _AVX_ADD(x2, _AVX_FMSUBADD(h1_real, q2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
	x2 = _AVX_ADD(x2, _AVX_ADDSUB( _AVX_MUL(h1_real, q2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif
	tmp3 = _AVX_MUL(h1_imag, q3);
#ifdef __ELPA_USE_FMA__
	x3 = _AVX_ADD(x3, _AVX_FMSUBADD(h1_real, q3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
	x3 = _AVX_ADD(x3, _AVX_ADDSUB( _AVX_MUL(h1_real, q3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

	h1_real = _AVX_BROADCAST(&hh_dbl[0]);
	h1_imag = _AVX_BROADCAST(&hh_dbl[1]);
	h1_real = _AVX_XOR(h1_real, sign);
	h1_imag = _AVX_XOR(h1_imag, sign);

	tmp1 = _AVX_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMADDSUB(h1_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
	x1 = _AVX_ADDSUB( _AVX_MUL(h1_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif
	tmp2 = _AVX_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
	x2 = _AVX_FMADDSUB(h1_real, x2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#else
	x2 = _AVX_ADDSUB( _AVX_MUL(h1_real, x2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#endif
	tmp3 = _AVX_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
	x3 = _AVX_FMADDSUB(h1_real, x3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#else
	x3 = _AVX_ADDSUB( _AVX_MUL(h1_real, x3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#endif

	h1_real = _AVX_BROADCAST(&hh_dbl[ldh*2]);
	h1_imag = _AVX_BROADCAST(&hh_dbl[(ldh*2)+1]);
	h2_real = _AVX_BROADCAST(&hh_dbl[ldh*2]);
	h2_imag = _AVX_BROADCAST(&hh_dbl[(ldh*2)+1]);

	h1_real = _AVX_XOR(h1_real, sign);
	h1_imag = _AVX_XOR(h1_imag, sign);
	h2_real = _AVX_XOR(h2_real, sign);
	h2_imag = _AVX_XOR(h2_imag, sign);

#ifdef DOUBLE_PRECISION_COMPLEX
	tmp2 = _mm256_set_pd(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	tmp2 = _mm256_set_ps(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0],
			     s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif

	tmp1 = _AVX_MUL(h2_imag, tmp2);
#ifdef __ELPA_USE_FMA__
	tmp2 = _AVX_FMADDSUB(h2_real, tmp2, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
	tmp2 = _AVX_ADDSUB( _AVX_MUL(h2_real, tmp2), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif
	h2_real = _AVX_SET1(tmp2[0]);
	h2_imag = _AVX_SET1(tmp2[1]);

	tmp1 = _AVX_MUL(h1_imag, y1);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMADDSUB(h1_real, y1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
	y1 = _AVX_ADDSUB( _AVX_MUL(h1_real, y1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif
	tmp2 = _AVX_MUL(h1_imag, y2);
#ifdef __ELPA_USE_FMA__
	y2 = _AVX_FMADDSUB(h1_real, y2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#else
	y2 = _AVX_ADDSUB( _AVX_MUL(h1_real, y2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#endif
	tmp3 = _AVX_MUL(h1_imag, y3);
#ifdef __ELPA_USE_FMA__
	y3 = _AVX_FMADDSUB(h1_real, y3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#else
	y3 = _AVX_ADDSUB( _AVX_MUL(h1_real, y3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE));
#endif

	tmp1 = _AVX_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_ADD(y1, _AVX_FMADDSUB(h2_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	y1 = _AVX_ADD(y1, _AVX_ADDSUB( _AVX_MUL(h2_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
	tmp2 = _AVX_MUL(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
	y2 = _AVX_ADD(y2, _AVX_FMADDSUB(h2_real, x2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
	y2 = _AVX_ADD(y2, _AVX_ADDSUB( _AVX_MUL(h2_real, x2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif
	tmp3 = _AVX_MUL(h2_imag, x3);
#ifdef __ELPA_USE_FMA__
	y3 = _AVX_ADD(y3, _AVX_FMADDSUB(h2_real, x3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
	y3 = _AVX_ADD(y3, _AVX_ADDSUB( _AVX_MUL(h2_real, x3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

	q1 = _AVX_LOAD(&q_dbl[0]);
	q2 = _AVX_LOAD(&q_dbl[offset]);
	q3 = _AVX_LOAD(&q_dbl[2*offset]);

	q1 = _AVX_ADD(q1, y1);
	q2 = _AVX_ADD(q2, y2);
	q3 = _AVX_ADD(q3, y3);

	_AVX_STORE(&q_dbl[0], q1);
	_AVX_STORE(&q_dbl[offset], q2);
	_AVX_STORE(&q_dbl[2*offset], q3);

	h2_real = _AVX_BROADCAST(&hh_dbl[(ldh+1)*2]);
	h2_imag = _AVX_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);

	q1 = _AVX_LOAD(&q_dbl[(ldq*2)+0]);
	q2 = _AVX_LOAD(&q_dbl[(ldq*2)+offset]);
	q3 = _AVX_LOAD(&q_dbl[(ldq*2)+2*offset]);

	q1 = _AVX_ADD(q1, x1);
	q2 = _AVX_ADD(q2, x2);
	q3 = _AVX_ADD(q3, x3);

	tmp1 = _AVX_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_ADD(q1, _AVX_FMADDSUB(h2_real, y1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	q1 = _AVX_ADD(q1, _AVX_ADDSUB( _AVX_MUL(h2_real, y1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
	tmp2 = _AVX_MUL(h2_imag, y2);
#ifdef __FMA4_
	q2 = _AVX_ADD(q2, _AVX_FMADDSUB(h2_real, y2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
	q2 = _AVX_ADD(q2, _AVX_ADDSUB( _AVX_MUL(h2_real, y2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif
	tmp3 = _AVX_MUL(h2_imag, y3);
#ifdef __ELPA_USE_FMA__
	q3 = _AVX_ADD(q3, _AVX_FMADDSUB(h2_real, y3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
	q3 = _AVX_ADD(q3, _AVX_ADDSUB( _AVX_MUL(h2_real, y3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

	_AVX_STORE(&q_dbl[(ldq*2)+0], q1);
	_AVX_STORE(&q_dbl[(ldq*2)+offset], q2);
	_AVX_STORE(&q_dbl[(ldq*2)+2*offset], q3);

	for (i = 2; i < nb; i++)
	{
		q1 = _AVX_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _AVX_LOAD(&q_dbl[(2*i*ldq)+offset]);
		q3 = _AVX_LOAD(&q_dbl[(2*i*ldq)+2*offset]);

		h1_real = _AVX_BROADCAST(&hh_dbl[(i-1)*2]);
		h1_imag = _AVX_BROADCAST(&hh_dbl[((i-1)*2)+1]);

		tmp1 = _AVX_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_ADD(q1, _AVX_FMADDSUB(h1_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
		q1 = _AVX_ADD(q1, _AVX_ADDSUB( _AVX_MUL(h1_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
		tmp2 = _AVX_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
		q2 = _AVX_ADD(q2, _AVX_FMADDSUB(h1_real, x2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
		q2 = _AVX_ADD(q2, _AVX_ADDSUB( _AVX_MUL(h1_real, x2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif
		tmp3 = _AVX_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
		q3 = _AVX_ADD(q3, _AVX_FMADDSUB(h1_real, x3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
		q3 = _AVX_ADD(q3, _AVX_ADDSUB( _AVX_MUL(h1_real, x3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

		h2_real = _AVX_BROADCAST(&hh_dbl[(ldh+i)*2]);
		h2_imag = _AVX_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _AVX_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_ADD(q1, _AVX_FMADDSUB(h2_real, y1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
		q1 = _AVX_ADD(q1, _AVX_ADDSUB( _AVX_MUL(h2_real, y1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
		tmp2 = _AVX_MUL(h2_imag, y2);
#ifdef __ELPA_USE_FMA__
		q2 = _AVX_ADD(q2, _AVX_FMADDSUB(h2_real, y2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
		q2 = _AVX_ADD(q2, _AVX_ADDSUB( _AVX_MUL(h2_real, y2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif
		tmp3 = _AVX_MUL(h2_imag, y3);
#ifdef __ELPA_USE_FMA__
		q3 = _AVX_ADD(q3, _AVX_FMADDSUB(h2_real, y3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
		q3 = _AVX_ADD(q3, _AVX_ADDSUB( _AVX_MUL(h2_real, y3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

		_AVX_STORE(&q_dbl[(2*i*ldq)+0], q1);
		_AVX_STORE(&q_dbl[(2*i*ldq)+offset], q2);
		_AVX_STORE(&q_dbl[(2*i*ldq)+2*offset], q3);
	}
	h1_real = _AVX_BROADCAST(&hh_dbl[(nb-1)*2]);
	h1_imag = _AVX_BROADCAST(&hh_dbl[((nb-1)*2)+1]);

	q1 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+0]);
	q2 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+offset]);
	q3 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+2*offset]);

	tmp1 = _AVX_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_ADD(q1, _AVX_FMADDSUB(h1_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	q1 = _AVX_ADD(q1, _AVX_ADDSUB( _AVX_MUL(h1_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
	tmp2 = _AVX_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
	q2 = _AVX_ADD(q2, _AVX_FMADDSUB(h1_real, x2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
	q2 = _AVX_ADD(q2, _AVX_ADDSUB( _AVX_MUL(h1_real, x2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif
	tmp3 = _AVX_MUL(h1_imag, x3);
#ifdef __ELPA_USE_FMA__
	q3 = _AVX_ADD(q3, _AVX_FMADDSUB(h1_real, x3, _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#else
	q3 = _AVX_ADD(q3, _AVX_ADDSUB( _AVX_MUL(h1_real, x3), _AVX_SHUFFLE(tmp3, tmp3, _SHUFFLE)));
#endif

	_AVX_STORE(&q_dbl[(2*nb*ldq)+0], q1);
	_AVX_STORE(&q_dbl[(2*nb*ldq)+offset], q2);
	_AVX_STORE(&q_dbl[(2*nb*ldq)+2*offset], q3);
}

#ifdef DOUBLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_4_AVX_2hv_double(double complex* q, double complex* hh, int nb, int ldq, int ldh, double complex s)
#endif
#ifdef SINGLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_8_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s, float complex s1)
#endif

{
#ifdef DOUBLE_PRECISION_COMPLEX
	double* q_dbl = (double*)q;
	double* hh_dbl = (double*)hh;
	double* s_dbl = (double*)(&s);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;
	float* s_dbl = (float*)(&s);
#endif
	__AVX_DATATYPE x1, x2;
	__AVX_DATATYPE y1, y2;
	__AVX_DATATYPE q1, q2;
	__AVX_DATATYPE h1_real, h1_imag, h2_real, h2_imag;
	__AVX_DATATYPE tmp1, tmp2;
	int i=0;

#ifdef DOUBLE_PRECISION_COMPLEX
	__AVX_DATATYPE sign = (__AVX_DATATYPE)_mm256_set_epi64x(0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	__AVX_DATATYPE sign = (__AVX_DATATYPE)_mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif

	x1 = _AVX_LOAD(&q_dbl[(2*ldq)+0]);
	x2 = _AVX_LOAD(&q_dbl[(2*ldq)+offset]);

	h2_real = _AVX_BROADCAST(&hh_dbl[(ldh+1)*2]);
	h2_imag = _AVX_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);
#ifndef __ELPA_USE_FMA__
	// conjugate
	h2_imag = _AVX_XOR(h2_imag, sign);
#endif

	y1 = _AVX_LOAD(&q_dbl[0]);
	y2 = _AVX_LOAD(&q_dbl[offset]);

	tmp1 = _AVX_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_ADD(y1, _AVX_FMSUBADD(h2_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	y1 = _AVX_ADD(y1, _AVX_ADDSUB( _AVX_MUL(h2_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

	tmp2 = _AVX_MUL(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
	y2 = _AVX_ADD(y2, _AVX_FMSUBADD(h2_real, x2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
	y2 = _AVX_ADD(y2, _AVX_ADDSUB( _AVX_MUL(h2_real, x2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

	for (i = 2; i < nb; i++)
	{
		q1 = _AVX_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _AVX_LOAD(&q_dbl[(2*i*ldq)+offset]);

		h1_real = _AVX_BROADCAST(&hh_dbl[(i-1)*2]);
		h1_imag = _AVX_BROADCAST(&hh_dbl[((i-1)*2)+1]);
#ifndef __ELPA_USE_FMA__
		// conjugate
		h1_imag = _AVX_XOR(h1_imag, sign);
#endif

		tmp1 = _AVX_MUL(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
		x1 = _AVX_ADD(x1, _AVX_FMSUBADD(h1_real, q1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
		x1 = _AVX_ADD(x1, _AVX_ADDSUB( _AVX_MUL(h1_real, q1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

		tmp2 = _AVX_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
		x2 = _AVX_ADD(x2, _AVX_FMSUBADD(h1_real, q2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
		x2 = _AVX_ADD(x2, _AVX_ADDSUB( _AVX_MUL(h1_real, q2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

		h2_real = _AVX_BROADCAST(&hh_dbl[(ldh+i)*2]);
		h2_imag = _AVX_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);
#ifndef __ELPA_USE_FMA__
		// conjugate
		h2_imag = _AVX_XOR(h2_imag, sign);
#endif

		tmp1 = _AVX_MUL(h2_imag, q1);
#ifdef __ELPA_USE_FMA__
		y1 = _AVX_ADD(y1, _AVX_FMSUBADD(h2_real, q1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
		y1 = _AVX_ADD(y1, _AVX_ADDSUB( _AVX_MUL(h2_real, q1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

		tmp2 = _AVX_MUL(h2_imag, q2);
#ifdef __ELPA_USE_FMA__
		y2 = _AVX_ADD(y2, _AVX_FMSUBADD(h2_real, q2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
		y2 = _AVX_ADD(y2, _AVX_ADDSUB( _AVX_MUL(h2_real, q2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

	}

	h1_real = _AVX_BROADCAST(&hh_dbl[(nb-1)*2]);
	h1_imag = _AVX_BROADCAST(&hh_dbl[((nb-1)*2)+1]);
#ifndef __ELPA_USE_FMA__
	// conjugate
	h1_imag = _AVX_XOR(h1_imag, sign);
#endif

	q1 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+0]);
	q2 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+offset]);

	tmp1 = _AVX_MUL(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_ADD(x1, _AVX_FMSUBADD(h1_real, q1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	x1 = _AVX_ADD(x1, _AVX_ADDSUB( _AVX_MUL(h1_real, q1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

	tmp2 = _AVX_MUL(h1_imag, q2);
#ifdef __ELPA_USE_FMA__
	x2 = _AVX_ADD(x2, _AVX_FMSUBADD(h1_real, q2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
	x2 = _AVX_ADD(x2, _AVX_ADDSUB( _AVX_MUL(h1_real, q2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

	h1_real = _AVX_BROADCAST(&hh_dbl[0]);
	h1_imag = _AVX_BROADCAST(&hh_dbl[1]);
	h1_real = _AVX_XOR(h1_real, sign);
	h1_imag = _AVX_XOR(h1_imag, sign);

	tmp1 = _AVX_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMADDSUB(h1_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
	x1 = _AVX_ADDSUB( _AVX_MUL(h1_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

	tmp2 = _AVX_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
	x2 = _AVX_FMADDSUB(h1_real, x2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#else
	x2 = _AVX_ADDSUB( _AVX_MUL(h1_real, x2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#endif

	h1_real = _AVX_BROADCAST(&hh_dbl[ldh*2]);
	h1_imag = _AVX_BROADCAST(&hh_dbl[(ldh*2)+1]);
	h2_real = _AVX_BROADCAST(&hh_dbl[ldh*2]);
	h2_imag = _AVX_BROADCAST(&hh_dbl[(ldh*2)+1]);

	h1_real = _AVX_XOR(h1_real, sign);
	h1_imag = _AVX_XOR(h1_imag, sign);
	h2_real = _AVX_XOR(h2_real, sign);
	h2_imag = _AVX_XOR(h2_imag, sign);

#ifdef DOUBLE_PRECISION_COMPLEX
	tmp2 = _mm256_set_pd(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	tmp2 = _mm256_set_ps(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0],
			     s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif

	tmp1 = _AVX_MUL(h2_imag, tmp2);
#ifdef __ELPA_USE_FMA__
	tmp2 = _AVX_FMADDSUB(h2_real, tmp2, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
	tmp2 = _AVX_ADDSUB( _AVX_MUL(h2_real, tmp2), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif
	h2_real = _AVX_SET1(tmp2[0]);
	h2_imag = _AVX_SET1(tmp2[1]);

	tmp1 = _AVX_MUL(h1_imag, y1);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMADDSUB(h1_real, y1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
	y1 = _AVX_ADDSUB( _AVX_MUL(h1_real, y1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

	tmp2 = _AVX_MUL(h1_imag, y2);
#ifdef __ELPA_USE_FMA__
	y2 = _AVX_FMADDSUB(h1_real, y2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#else
	y2 = _AVX_ADDSUB( _AVX_MUL(h1_real, y2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE));
#endif

	tmp1 = _AVX_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_ADD(y1, _AVX_FMADDSUB(h2_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	y1 = _AVX_ADD(y1, _AVX_ADDSUB( _AVX_MUL(h2_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
	tmp2 = _AVX_MUL(h2_imag, x2);
#ifdef __ELPA_USE_FMA__
	y2 = _AVX_ADD(y2, _AVX_FMADDSUB(h2_real, x2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
	y2 = _AVX_ADD(y2, _AVX_ADDSUB( _AVX_MUL(h2_real, x2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

	q1 = _AVX_LOAD(&q_dbl[0]);
	q2 = _AVX_LOAD(&q_dbl[offset]);

	q1 = _AVX_ADD(q1, y1);
	q2 = _AVX_ADD(q2, y2);

	_AVX_STORE(&q_dbl[0], q1);
	_AVX_STORE(&q_dbl[offset], q2);

	h2_real = _AVX_BROADCAST(&hh_dbl[(ldh+1)*2]);
	h2_imag = _AVX_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);

	q1 = _AVX_LOAD(&q_dbl[(ldq*2)+0]);
	q2 = _AVX_LOAD(&q_dbl[(ldq*2)+offset]);

	q1 = _AVX_ADD(q1, x1);
	q2 = _AVX_ADD(q2, x2);

	tmp1 = _AVX_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_ADD(q1, _AVX_FMADDSUB(h2_real, y1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	q1 = _AVX_ADD(q1, _AVX_ADDSUB( _AVX_MUL(h2_real, y1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

	tmp2 = _AVX_MUL(h2_imag, y2);
#ifdef __FMA4_
	q2 = _AVX_ADD(q2, _AVX_FMADDSUB(h2_real, y2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
	q2 = _AVX_ADD(q2, _AVX_ADDSUB( _AVX_MUL(h2_real, y2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

	_AVX_STORE(&q_dbl[(ldq*2)+0], q1);
	_AVX_STORE(&q_dbl[(ldq*2)+offset], q2);

	for (i = 2; i < nb; i++)
	{
		q1 = _AVX_LOAD(&q_dbl[(2*i*ldq)+0]);
		q2 = _AVX_LOAD(&q_dbl[(2*i*ldq)+offset]);

		h1_real = _AVX_BROADCAST(&hh_dbl[(i-1)*2]);
		h1_imag = _AVX_BROADCAST(&hh_dbl[((i-1)*2)+1]);

		tmp1 = _AVX_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_ADD(q1, _AVX_FMADDSUB(h1_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
		q1 = _AVX_ADD(q1, _AVX_ADDSUB( _AVX_MUL(h1_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

		tmp2 = _AVX_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
		q2 = _AVX_ADD(q2, _AVX_FMADDSUB(h1_real, x2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
		q2 = _AVX_ADD(q2, _AVX_ADDSUB( _AVX_MUL(h1_real, x2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif
		h2_real = _AVX_BROADCAST(&hh_dbl[(ldh+i)*2]);
		h2_imag = _AVX_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _AVX_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_ADD(q1, _AVX_FMADDSUB(h2_real, y1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
		q1 = _AVX_ADD(q1, _AVX_ADDSUB( _AVX_MUL(h2_real, y1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

		tmp2 = _AVX_MUL(h2_imag, y2);
#ifdef __ELPA_USE_FMA__
		q2 = _AVX_ADD(q2, _AVX_FMADDSUB(h2_real, y2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
		q2 = _AVX_ADD(q2, _AVX_ADDSUB( _AVX_MUL(h2_real, y2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

		_AVX_STORE(&q_dbl[(2*i*ldq)+0], q1);
		_AVX_STORE(&q_dbl[(2*i*ldq)+offset], q2);
	}
	h1_real = _AVX_BROADCAST(&hh_dbl[(nb-1)*2]);
	h1_imag = _AVX_BROADCAST(&hh_dbl[((nb-1)*2)+1]);

	q1 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+0]);
	q2 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+offset]);

	tmp1 = _AVX_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_ADD(q1, _AVX_FMADDSUB(h1_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	q1 = _AVX_ADD(q1, _AVX_ADDSUB( _AVX_MUL(h1_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

	tmp2 = _AVX_MUL(h1_imag, x2);
#ifdef __ELPA_USE_FMA__
	q2 = _AVX_ADD(q2, _AVX_FMADDSUB(h1_real, x2, _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#else
	q2 = _AVX_ADD(q2, _AVX_ADDSUB( _AVX_MUL(h1_real, x2), _AVX_SHUFFLE(tmp2, tmp2, _SHUFFLE)));
#endif

	_AVX_STORE(&q_dbl[(2*nb*ldq)+0], q1);
	_AVX_STORE(&q_dbl[(2*nb*ldq)+offset], q2);
}

#ifdef DOUBLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_2_AVX_2hv_double(double complex* q, double complex* hh, int nb, int ldq, int ldh, double complex s)
#endif
#ifdef SINGLE_PRECISION_COMPLEX
static __forceinline void hh_trafo_complex_kernel_4_AVX_2hv_single(float complex* q, float complex* hh, int nb, int ldq, int ldh, float complex s, float complex s1)
#endif

{
#ifdef DOUBLE_PRECISION_COMPLEX
	double* q_dbl = (double*)q;
	double* hh_dbl = (double*)hh;
	double* s_dbl = (double*)(&s);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	float* q_dbl = (float*)q;
	float* hh_dbl = (float*)hh;
	float* s_dbl = (float*)(&s);
#endif
	__AVX_DATATYPE x1;
	__AVX_DATATYPE y1;
	__AVX_DATATYPE q1;
	__AVX_DATATYPE h1_real, h1_imag, h2_real, h2_imag;
	__AVX_DATATYPE tmp1;
	int i=0;
#ifdef DOUBLE_PRECISION_COMPLEX
	__AVX_DATATYPE sign = (__AVX_DATATYPE)_mm256_set_epi64x(0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	__AVX_DATATYPE sign = (__AVX_DATATYPE)_mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
#endif
	x1 = _AVX_LOAD(&q_dbl[(2*ldq)+0]);

	h2_real = _AVX_BROADCAST(&hh_dbl[(ldh+1)*2]);
	h2_imag = _AVX_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);
#ifndef __ELPA_USE_FMA__
	// conjugate
	h2_imag = _AVX_XOR(h2_imag, sign);
#endif

	y1 = _AVX_LOAD(&q_dbl[0]);

	tmp1 = _AVX_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_ADD(y1, _AVX_FMSUBADD(h2_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	y1 = _AVX_ADD(y1, _AVX_ADDSUB( _AVX_MUL(h2_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

	for (i = 2; i < nb; i++)
	{
		q1 = _AVX_LOAD(&q_dbl[(2*i*ldq)+0]);

		h1_real = _AVX_BROADCAST(&hh_dbl[(i-1)*2]);
		h1_imag = _AVX_BROADCAST(&hh_dbl[((i-1)*2)+1]);
#ifndef __ELPA_USE_FMA__
		// conjugate
		h1_imag = _AVX_XOR(h1_imag, sign);
#endif

		tmp1 = _AVX_MUL(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
		x1 = _AVX_ADD(x1, _AVX_FMSUBADD(h1_real, q1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
		x1 = _AVX_ADD(x1, _AVX_ADDSUB( _AVX_MUL(h1_real, q1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

		h2_real = _AVX_BROADCAST(&hh_dbl[(ldh+i)*2]);
		h2_imag = _AVX_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);
#ifndef __ELPA_USE_FMA__
		// conjugate
		h2_imag = _AVX_XOR(h2_imag, sign);
#endif

		tmp1 = _AVX_MUL(h2_imag, q1);
#ifdef __ELPA_USE_FMA__
		y1 = _AVX_ADD(y1, _AVX_FMSUBADD(h2_real, q1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
		y1 = _AVX_ADD(y1, _AVX_ADDSUB( _AVX_MUL(h2_real, q1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif
	}

	h1_real = _AVX_BROADCAST(&hh_dbl[(nb-1)*2]);
	h1_imag = _AVX_BROADCAST(&hh_dbl[((nb-1)*2)+1]);
#ifndef __ELPA_USE_FMA__
	// conjugate
	h1_imag = _AVX_XOR(h1_imag, sign);
#endif

	q1 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+0]);

	tmp1 = _AVX_MUL(h1_imag, q1);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_ADD(x1, _AVX_FMSUBADD(h1_real, q1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	x1 = _AVX_ADD(x1, _AVX_ADDSUB( _AVX_MUL(h1_real, q1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

	h1_real = _AVX_BROADCAST(&hh_dbl[0]);
	h1_imag = _AVX_BROADCAST(&hh_dbl[1]);
	h1_real = _AVX_XOR(h1_real, sign);
	h1_imag = _AVX_XOR(h1_imag, sign);

	tmp1 = _AVX_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
	x1 = _AVX_FMADDSUB(h1_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
	x1 = _AVX_ADDSUB( _AVX_MUL(h1_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

	h1_real = _AVX_BROADCAST(&hh_dbl[ldh*2]);
	h1_imag = _AVX_BROADCAST(&hh_dbl[(ldh*2)+1]);
	h2_real = _AVX_BROADCAST(&hh_dbl[ldh*2]);
	h2_imag = _AVX_BROADCAST(&hh_dbl[(ldh*2)+1]);

	h1_real = _AVX_XOR(h1_real, sign);
	h1_imag = _AVX_XOR(h1_imag, sign);
	h2_real = _AVX_XOR(h2_real, sign);
	h2_imag = _AVX_XOR(h2_imag, sign);

	__AVX_DATATYPE tmp2;
#ifdef DOUBLE_PRECISION_COMPLEX
	tmp2 = _mm256_set_pd(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
#ifdef SINGLE_PRECISION_COMPLEX
	tmp2 = _mm256_set_ps(s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0],
			     s_dbl[1], s_dbl[0], s_dbl[1], s_dbl[0]);
#endif
	tmp1 = _AVX_MUL(h2_imag, tmp2);
#ifdef __ELPA_USE_FMA__
	tmp2 = _AVX_FMADDSUB(h2_real, tmp2, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
	tmp2 = _AVX_ADDSUB( _AVX_MUL(h2_real, tmp2), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif
	h2_real = _AVX_SET1(tmp2[0]);
	h2_imag = _AVX_SET1(tmp2[1]);

	tmp1 = _AVX_MUL(h1_imag, y1);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_FMADDSUB(h1_real, y1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#else
	y1 = _AVX_ADDSUB( _AVX_MUL(h1_real, y1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE));
#endif

	tmp1 = _AVX_MUL(h2_imag, x1);
#ifdef __ELPA_USE_FMA__
	y1 = _AVX_ADD(y1, _AVX_FMADDSUB(h2_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	y1 = _AVX_ADD(y1, _AVX_ADDSUB( _AVX_MUL(h2_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

	q1 = _AVX_LOAD(&q_dbl[0]);

	q1 = _AVX_ADD(q1, y1);

	_AVX_STORE(&q_dbl[0], q1);

	h2_real = _AVX_BROADCAST(&hh_dbl[(ldh+1)*2]);
	h2_imag = _AVX_BROADCAST(&hh_dbl[((ldh+1)*2)+1]);

	q1 = _AVX_LOAD(&q_dbl[(ldq*2)+0]);

	q1 = _AVX_ADD(q1, x1);

	tmp1 = _AVX_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_ADD(q1, _AVX_FMADDSUB(h2_real, y1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	q1 = _AVX_ADD(q1, _AVX_ADDSUB( _AVX_MUL(h2_real, y1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

	_AVX_STORE(&q_dbl[(ldq*2)+0], q1);

	for (i = 2; i < nb; i++)
	{
		q1 = _AVX_LOAD(&q_dbl[(2*i*ldq)+0]);

		h1_real = _AVX_BROADCAST(&hh_dbl[(i-1)*2]);
		h1_imag = _AVX_BROADCAST(&hh_dbl[((i-1)*2)+1]);

		tmp1 = _AVX_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_ADD(q1, _AVX_FMADDSUB(h1_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
		q1 = _AVX_ADD(q1, _AVX_ADDSUB( _AVX_MUL(h1_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

		h2_real = _AVX_BROADCAST(&hh_dbl[(ldh+i)*2]);
		h2_imag = _AVX_BROADCAST(&hh_dbl[((ldh+i)*2)+1]);

		tmp1 = _AVX_MUL(h2_imag, y1);
#ifdef __ELPA_USE_FMA__
		q1 = _AVX_ADD(q1, _AVX_FMADDSUB(h2_real, y1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
		q1 = _AVX_ADD(q1, _AVX_ADDSUB( _AVX_MUL(h2_real, y1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

		_AVX_STORE(&q_dbl[(2*i*ldq)+0], q1);
	}
	h1_real = _AVX_BROADCAST(&hh_dbl[(nb-1)*2]);
	h1_imag = _AVX_BROADCAST(&hh_dbl[((nb-1)*2)+1]);

	q1 = _AVX_LOAD(&q_dbl[(2*nb*ldq)+0]);

	tmp1 = _AVX_MUL(h1_imag, x1);
#ifdef __ELPA_USE_FMA__
	q1 = _AVX_ADD(q1, _AVX_FMADDSUB(h1_real, x1, _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#else
	q1 = _AVX_ADD(q1, _AVX_ADDSUB( _AVX_MUL(h1_real, x1), _AVX_SHUFFLE(tmp1, tmp1, _SHUFFLE)));
#endif

	_AVX_STORE(&q_dbl[(2*nb*ldq)+0], q1);
}
